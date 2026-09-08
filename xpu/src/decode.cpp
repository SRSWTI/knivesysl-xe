// decode.cpp — single-token forward orchestration + debug-ladder ABI + shared
// runtime utilities for libforward_qwen_xpu.so.
//
// Faithful port of the CUDA layer walk (src/forward_qwen.cu):
//   run_linear_layer_decode_from_current   (16209)
//   run_full_layer_decode_from_current     (16319)
//   run_mlp_from_resid_to_layer_out_w      (16097)
//   run_decode_layers_to_debug_x           (16754)
//   run_final_norm_device / run_lm_head_argmax_device / copy_forward_argmax_result
// The CUDA fused/multi-GEMV fast paths collapse onto the plain per-projection
// x_gemv_qmma calls (same math, one weight at a time).
//
// qwn_debug_forward_* (the stateless first-token ladder) is implemented as the
// decode path at pos == 0: decode-at-0 resets all state, and attention over the
// single position degenerates to the first-token kernel's math (softmax over
// one score == 1, out = v * sigmoid(gate)); DeltaNet starts from zero state in
// both. Values match the CUDA ladder within the float-eps band.

#include "tq_common.hpp"
#include <chrono>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <limits>
#include <new>
#include <utility>
#include <stdexcept>
#include <vector>

// ------------------------------------------------------------------ runtime
// Devices for tensor parallelism. TQ_XPU_TP=1 (default) is exactly the
// single-card engine. TQ_XPU_TP=2 builds two in-order queues on consecutive
// Level Zero devices inside ONE shared context - the shared context is what
// makes cross-card pointers and a shared host allocation legal, and it is the
// configuration `xpu/probe/tp_allreduce_probe.cpp` measured.
//
// Every allocator and transfer helper below routes through tq_q(), so
// switching the active rank re-targets allocation, upload and launch with no
// change to any kernel. That is the whole trick: TP costs the kernels nothing.
namespace {

struct TqDevices {
    sycl::context ctx;
    std::vector<sycl::queue *> qs;
    int active = 0;
};

TqDevices &tq_devs() {
    static TqDevices *d = [] {
        int want = 0;
        if (const char *e = getenv("TQ_XPU_DEV")) want = atoi(e);
        int tp = 1;
        if (const char *e = getenv("TQ_XPU_TP")) tp = atoi(e);
        std::vector<sycl::device> gpus;
        for (auto &dev : sycl::device::get_devices(sycl::info::device_type::gpu))
            if (dev.get_backend() == sycl::backend::ext_oneapi_level_zero)
                gpus.push_back(dev);
        if (gpus.empty()) {
            fprintf(stderr, "forward_qwen_xpu: no level-zero GPU found\n");
            abort();
        }
        if (want < 0 || want >= (int)gpus.size()) want = 0;
        if (tp < 1) tp = 1;
        if (tp > (int)gpus.size()) {
            fprintf(stderr, "[xpu] TQ_XPU_TP=%d but only %zu GPUs; using 1\n",
                    tp, gpus.size());
            tp = 1;
        }
        std::vector<sycl::device> use;
        for (int r = 0; r < tp; ++r)
            use.push_back(gpus[(want + r) % (int)gpus.size()]);
        auto *dd = new TqDevices{sycl::context(use), {}, 0};
        for (int r = 0; r < tp; ++r) {
            dd->qs.push_back(new sycl::queue(dd->ctx, use[r],
                                             sycl::property::queue::in_order{}));
            fprintf(stderr, "[xpu] rank %d device: %s\n", r,
                    use[r].get_info<sycl::info::device::name>().c_str());
        }
        return dd;
    }();
    return *d;
}

}  // namespace

sycl::queue &tq_q() {
    TqDevices &d = tq_devs();
    return *d.qs[d.active];
}

int tq_tp_size() { return (int)tq_devs().qs.size(); }
int tq_rank() { return tq_devs().active; }
void tq_set_rank(int r) {
    TqDevices &d = tq_devs();
    if (r < 0 || r >= (int)d.qs.size() || r >= kMaxTP) return;
    d.active = r;
    g_qwen_active = &g_qwen_ranks[r];   // queue and model move together
}
sycl::queue &tq_q_of(int r) {
    TqDevices &d = tq_devs();
    return *d.qs[r >= 0 && r < (int)d.qs.size() ? r : 0];
}

void *tq_dev_alloc(size_t bytes, const char *what) {
    void *p = sycl::malloc_device(bytes, tq_q());
    if (!p) {
        fprintf(stderr, "forward_qwen_xpu: device alloc of %zu bytes failed (%s)\n",
                bytes, what);
        abort();
    }
    return p;
}
void tq_dev_free(void *p) { if (p) sycl::free(p, tq_q()); }
void tq_h2d(void *dst, const void *src, size_t bytes) { tq_q().memcpy(dst, src, bytes).wait(); }
void tq_d2h(void *dst, const void *src, size_t bytes) { tq_q().memcpy(dst, src, bytes).wait(); }
void tq_dev_zero(void *dst, size_t bytes) { tq_q().memset(dst, 0, bytes).wait(); }

// device-to-device async copy on the shared in-order queue
static void tq_d2d_async(void *dst, const void *src, size_t bytes) {
    tq_q().memcpy(dst, src, bytes);
}

// ---- tensor-parallel all-reduce ------------------------------------------
// Sum n floats across the TP ranks in place. bufs[r] is rank r's partial
// (device memory on rank r); scratch[r] is a same-sized landing slot on
// rank r. On return every rank's buf holds the full sum.
//
// Shape and cost are exactly what tp_allreduce_probe measured: exchange the
// partials peer-to-peer, join the in-order queues (the kernel boundary IS
// the barrier - lesson 14 says a software global barrier loses to it on this
// silicon), then add locally. 37.33 us for the 20 KB hidden vector, i.e.
// 4.78 ms/token across the 128 row-parallel reductions of this model.
void tq_allreduce(float *const *bufs, float *const *scratch, int n) {
    const int tp = tq_tp_size();
    if (tp < 2 || n <= 0) return;
    const size_t bytes = (size_t)n * sizeof(float);
    // Ring exchange: rank r sends its partial to every peer's slot. For tp=2
    // that is one copy each way; the loop keeps the primitive honest for
    // wider TP without pretending we have measured it.
    for (int r = 0; r < tp; ++r)
        for (int p = 0; p < tp; ++p)
            if (p != r) tq_q_of(r).memcpy(scratch[p], bufs[r], bytes);
    for (int r = 0; r < tp; ++r) tq_q_of(r).wait();
    for (int r = 0; r < tp; ++r) {
        float *dst = bufs[r], *src = scratch[r];
        tq_q_of(r).parallel_for(
            sycl::nd_range<1>(((n + 255) / 256) * 256, 256),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const size_t i = it.get_global_linear_id();
                if (i < (size_t)n) dst[i] += src[i];
            });
    }
    for (int r = 0; r < tp; ++r) tq_q_of(r).wait();
}

enum ProfileStage {
    kProfileEmbed,
    kProfileNorm,
    kProfileQuant,
    kProfileProjection,
    kProfileConv,
    kProfileDelta,
    kProfileAttention,
    kProfileActivation,
    kProfileResidual,
    kProfileFinalNorm,
    kProfileLmHead,
    kProfileStageCount,
};

struct ProfileProjection {
    int M = 0;
    int K = 0;
    int tier = 0;
    double ms = 0.0;
    int calls = 0;
};

struct ProfileRun {
    double ms[kProfileStageCount] = {};
    int calls[kProfileStageCount] = {};
    ProfileProjection projections[16] = {};
};

ProfileRun *g_profile_run = nullptr;
std::chrono::steady_clock::time_point g_profile_mark;

static void profile_mark(ProfileStage stage) {
    if (!g_profile_run) return;
    tq_q().wait_and_throw();
    const auto now = std::chrono::steady_clock::now();
    g_profile_run->ms[stage] +=
        std::chrono::duration<double, std::milli>(now - g_profile_mark).count();
    ++g_profile_run->calls[stage];
    g_profile_mark = now;
}

static void profile_projection(const tq_qmma_weight_t *w) {
    if (!g_profile_run) return;
    tq_q().wait_and_throw();
    const auto now = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double, std::milli>(now - g_profile_mark).count();
    g_profile_run->ms[kProfileProjection] += elapsed;
    ++g_profile_run->calls[kProfileProjection];
    const int tier = w->s8_ready ? 8 : (w->s4_ready ? 4 : 0);
    for (ProfileProjection &entry : g_profile_run->projections) {
        if (entry.calls == 0 ||
            (entry.M == w->M && entry.K == w->K && entry.tier == tier)) {
            entry.M = w->M;
            entry.K = w->K;
            entry.tier = tier;
            entry.ms += elapsed;
            ++entry.calls;
            break;
        }
    }
    g_profile_mark = now;
}

// ------------------------------------------------------------- layer bodies

// MLP tail shared by both layer kinds. CUDA: run_mlp_from_resid_to_layer_out_w
// (16097): post_norm = rmsnorm(resid, post_ln); gate/up GEMVs; silu*up;
// layer_out = resid + W_down @ mlp_hidden.
static int run_mlp_from_resid(tq_layer_t *l) {
    tq_model_t *m = &g_qwen;
    x_rmsnorm_quant(m->d_post_norm, m->d_resid, l->d_post_ln, m->H, m->eps);
    profile_mark(kProfileNorm);
    int ret = 0;
    // gate and up share d_post_norm and K, so one fan-out launch replaces
    // two. Same contract as the GDN 4-way fan-out below: negative means the
    // tier is not uniformly W4-K32 and the caller falls back per weight.
    {
        const tq_qmma_weight_t *ws[2] = {&l->mlp_gate, &l->mlp_up};
        float *ys[2] = {m->d_gate, m->d_up};
        int fret = x_gemv_w4a8_fanout(ws, ys, 2);
        if (fret != 0) {
            ret = x_gemv_qmma_prepared(&l->mlp_gate, m->d_post_norm, m->d_gate);
            if (ret != 0) return -7;
            ret = x_gemv_qmma_prepared(&l->mlp_up, m->d_post_norm, m->d_up);
            if (ret != 0) return -8;
        }
        profile_projection(&l->mlp_gate);
        profile_projection(&l->mlp_up);
    }
    const bool prepared_down = (l->mlp_down.s8_ready || l->mlp_down.s4_ready) &&
                               l->mlp_down.K == m->I && m->I > 0 && (m->I % 32) == 0;
    if (prepared_down)
        x_silu_mul_quant(m->d_mlp_hidden, m->d_gate, m->d_up, m->I);
    else
        x_silu_mul(m->d_mlp_hidden, m->d_gate, m->d_up, m->I);
    profile_mark(kProfileActivation);
    ret = prepared_down
        ? x_gemv_qmma_prepared(&l->mlp_down, m->d_mlp_hidden,
                               m->d_layer_out, m->d_resid)
        : x_gemv_qmma_add(&l->mlp_down, m->d_mlp_hidden,
                          m->d_resid, m->d_layer_out);
    profile_projection(&l->mlp_down);
    if (ret != 0) return -9;
    return 0;
}

// ---- Slot-adjusted state accessors (multi-slot serving) -------------------
// Per-sequence state is allocated g_qwen.slots times; these return the
// active slot's region. Kernels stay slot-agnostic.
static inline size_t slot_kv_elems(void) {
    return (size_t)g_qwen.max_seq * g_qwen.nkv * g_qwen.hd;
}
static inline size_t slot_kvrow_elems(void) {
    return (size_t)g_qwen.max_seq * g_qwen.nkv;
}
static inline size_t slot_conv_elems(void) {
    const int kd = g_qwen.linear_num_key_heads * g_qwen.linear_key_head_dim;
    const int vd = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;
    return (size_t)(2 * kd + vd) * g_qwen.linear_conv_kernel_dim;
}
static inline size_t slot_recur_elems(void) {
    return (size_t)g_qwen.linear_num_value_heads * g_qwen.linear_key_head_dim *
           g_qwen.linear_value_head_dim;
}
static inline uint8_t *slot_kc_at(tq_layer_t *l, int s) {
    return l->d_k_cache + (size_t)s * slot_kv_elems();
}
static inline uint8_t *slot_vc_at(tq_layer_t *l, int s) {
    return l->d_v_cache + (size_t)s * slot_kv_elems();
}
static inline uint16_t *slot_ks_at(tq_layer_t *l, int s) {
    return l->d_k_scale + (size_t)s * slot_kvrow_elems();
}
static inline uint16_t *slot_vs_at(tq_layer_t *l, int s) {
    return l->d_v_scale + (size_t)s * slot_kvrow_elems();
}
static inline float *slot_conv_at(tq_layer_t *l, int s) {
    return l->d_linear_conv_state + (size_t)s * slot_conv_elems();
}
static inline float *slot_recur_at(tq_layer_t *l, int s) {
    return l->d_linear_recurrent_state + (size_t)s * slot_recur_elems();
}

// ---- Production paged KV ownership ---------------------------------------
// One physical page id names the corresponding page in every full-attention
// layer. GDN state remains per-slot and is checkpointed separately.
static int g_pg_enabled = 0, g_pg_ready = 0;
static int g_pg_page = 0, g_pg_plog = 0, g_pg_nblocks = 0;
static int g_pg_maxslots = 0, g_pg_maxblk = 0, g_pg_nfree = 0;
static int *h_block_table = nullptr;   // [slot][logical block], -1 unmapped
static int *h_free = nullptr;          // FIFO free ring
static int g_pg_fhead = 0, g_pg_ftail = 0;
static int *h_blk_ref = nullptr;
static int *h_blk_active_ref = nullptr;
static int *h_blk_ckpt_ref = nullptr;
static uint8_t *h_blk_in_free = nullptr;
static int *h_slot_nb = nullptr;
static int *h_slot_resv_total = nullptr;
static int *h_slot_resv_left = nullptr;
static int *h_slot_dirty_lo = nullptr;
static int *h_slot_dirty_hi = nullptr;
static long long g_pg_reserved = 0;
static int *d_block_table = nullptr;
static uint8_t  *g_pool_k[TQ_MAX_LAYERS] = {};
static uint8_t  *g_pool_v[TQ_MAX_LAYERS] = {};
static uint16_t *g_pool_ks[TQ_MAX_LAYERS] = {};
static uint16_t *g_pool_vs[TQ_MAX_LAYERS] = {};

// The historical content index remains available only to allocator gates. It
// is deliberately not initialized or consulted by the production APC path.
static uint64_t *h_blk_hash = nullptr;
static uint64_t *h_idx_hash = nullptr;
static int *h_idx_blk = nullptr;
static int g_pg_idx_cap = 0;

struct PagedCheckpoint {
    int id = 0;
    int committed = 0;
    std::vector<int> full_blocks;
    int tail_block = -1;
    void *gdn_state = nullptr;
    size_t gdn_bytes = 0;
};
static std::vector<PagedCheckpoint> g_pg_checkpoints;
static int g_pg_next_ckpt_id = 1;  // never reused; stale ids cannot alias

static bool tq_checked_mul(size_t a, size_t b, size_t *out) {
    if (a && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

static void *tq_try_dev_alloc(size_t bytes, const char *what) {
    if (!bytes) return nullptr;
    try {
        void *p = sycl::malloc_device(bytes, tq_q());
        if (!p)
            fprintf(stderr, "forward_qwen_xpu: device alloc of %zu bytes failed (%s)\n",
                    bytes, what);
        return p;
    } catch (const std::exception &e) {
        fprintf(stderr, "forward_qwen_xpu: device alloc failed (%s): %s\n",
                what, e.what());
    } catch (...) {
        fprintf(stderr, "forward_qwen_xpu: device alloc failed (%s)\n", what);
    }
    return nullptr;
}

static size_t tq_pool_bytes_per_block(int page) {
    if (!g_qwen.initialized || page <= 0 || g_qwen.nkv <= 0 || g_qwen.hd <= 0)
        return 0;
    size_t rows = 0, bytes = 0, per_layer = 0;
    if (!tq_checked_mul((size_t)page, (size_t)g_qwen.nkv, &rows) ||
        !tq_checked_mul(rows, (size_t)g_qwen.hd, &bytes) ||
        bytes > std::numeric_limits<size_t>::max() / 2 ||
        !tq_checked_mul(rows, 2u * sizeof(uint16_t), &per_layer) ||
        2 * bytes > std::numeric_limits<size_t>::max() - per_layer)
        return 0;
    per_layer += 2 * bytes;
    size_t nfull = 0;
    for (int i = 0; i < g_qwen.L; ++i)
        if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) ++nfull;
    size_t total = 0;
    return tq_checked_mul(per_layer, nfull, &total) ? total : 0;
}

static int tq_pool_blocks_for_budget(size_t budget_bytes, int page) {
    const size_t per = tq_pool_bytes_per_block(page);
    if (!per) return 0;
    const size_t n = budget_bytes / per;
    return n > (size_t)INT_MAX ? INT_MAX : (int)n;
}

static void tq_pool_index_free(void) {
    free(h_blk_hash); h_blk_hash = nullptr;
    free(h_idx_hash); h_idx_hash = nullptr;
    free(h_idx_blk); h_idx_blk = nullptr;
    g_pg_idx_cap = 0;
}

static int tq_pool_index_init(int num_blocks) {
    tq_pool_index_free();
    if (num_blocks <= 0 || num_blocks > INT_MAX / 2) return -1;
    int cap = 1;
    while (cap < 2 * num_blocks) {
        if (cap > INT_MAX / 2) return -1;
        cap <<= 1;
    }
    h_blk_hash = (uint64_t *)calloc((size_t)num_blocks, sizeof(uint64_t));
    h_idx_hash = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    h_idx_blk = (int *)malloc((size_t)cap * sizeof(int));
    if (!h_blk_hash || !h_idx_hash || !h_idx_blk) {
        tq_pool_index_free();
        return -1;
    }
    for (int i = 0; i < cap; ++i) h_idx_blk[i] = -1;
    g_pg_idx_cap = cap;
    return 0;
}

static int tq_pool_index_slot(uint64_t h) {
    if (!g_pg_idx_cap) return -1;
    const uint64_t mask = (uint64_t)g_pg_idx_cap - 1;
    uint64_t i = h & mask;
    for (int probe = 0; probe < g_pg_idx_cap; ++probe) {
        if (h_idx_blk[i] < 0 || h_idx_hash[i] == h) return (int)i;
        i = (i + 1) & mask;
    }
    return -1;
}

static int tq_pool_lookup(uint64_t h) {
    if (!g_pg_idx_cap || h == 0) return -1;
    const int s = tq_pool_index_slot(h);
    return (s >= 0 && h_idx_blk[s] >= 0 && h_idx_hash[s] == h)
               ? h_idx_blk[s] : -1;
}

static void tq_pool_unpublish(int b) {
    if (!g_pg_idx_cap || b < 0 || b >= g_pg_nblocks) return;
    const uint64_t h = h_blk_hash[b];
    if (!h) return;
    const int s = tq_pool_index_slot(h);
    if (s >= 0 && h_idx_blk[s] == b) {
        // Repair the probe cluster instead of turning an interior entry into
        // a false miss. This index is gate-only, but its gate remains honest.
        h_idx_blk[s] = -1;
        h_idx_hash[s] = 0;
        int cur = (s + 1) & (g_pg_idx_cap - 1);
        while (h_idx_blk[cur] >= 0) {
            const uint64_t rh = h_idx_hash[cur];
            const int rb = h_idx_blk[cur];
            h_idx_blk[cur] = -1;
            h_idx_hash[cur] = 0;
            const int dst = tq_pool_index_slot(rh);
            h_idx_hash[dst] = rh;
            h_idx_blk[dst] = rb;
            cur = (cur + 1) & (g_pg_idx_cap - 1);
        }
    }
    h_blk_hash[b] = 0;
}

static int tq_pool_publish(int b, uint64_t h) {
    if (!g_pg_idx_cap || b < 0 || b >= g_pg_nblocks) return -1;
    if (h == 0) h = 1;
    const int s = tq_pool_index_slot(h);
    if (s < 0) return -1;
    if (h_idx_blk[s] >= 0 && h_idx_blk[s] != b) return -2;
    h_idx_hash[s] = h;
    h_idx_blk[s] = b;
    h_blk_hash[b] = h;
    return 0;
}

static int tq_pool_block_unref(int b, bool checkpoint);
static void tq_paged_checkpoints_free_all(void);

static void tq_pool_storage_free(void) {
    for (int i = 0; i < TQ_MAX_LAYERS; ++i) {
        if (g_pool_k[i]) { tq_dev_free(g_pool_k[i]); g_pool_k[i] = nullptr; }
        if (g_pool_v[i]) { tq_dev_free(g_pool_v[i]); g_pool_v[i] = nullptr; }
        if (g_pool_ks[i]) { tq_dev_free(g_pool_ks[i]); g_pool_ks[i] = nullptr; }
        if (g_pool_vs[i]) { tq_dev_free(g_pool_vs[i]); g_pool_vs[i] = nullptr; }
    }
    if (d_block_table) { tq_dev_free(d_block_table); d_block_table = nullptr; }
    free(h_block_table); h_block_table = nullptr;
    free(h_free); h_free = nullptr;
    free(h_blk_ref); h_blk_ref = nullptr;
    free(h_blk_active_ref); h_blk_active_ref = nullptr;
    free(h_blk_ckpt_ref); h_blk_ckpt_ref = nullptr;
    free(h_blk_in_free); h_blk_in_free = nullptr;
    free(h_slot_nb); h_slot_nb = nullptr;
    free(h_slot_resv_total); h_slot_resv_total = nullptr;
    free(h_slot_resv_left); h_slot_resv_left = nullptr;
    free(h_slot_dirty_lo); h_slot_dirty_lo = nullptr;
    free(h_slot_dirty_hi); h_slot_dirty_hi = nullptr;
    tq_pool_index_free();
    g_pg_page = g_pg_plog = g_pg_nblocks = 0;
    g_pg_maxslots = g_pg_maxblk = g_pg_nfree = 0;
    g_pg_fhead = g_pg_ftail = 0;
    g_pg_reserved = 0;
}

static void tq_pool_free_all(void) {
    tq_pool_storage_free();
}

struct tq_pool_gate_guard {
    ~tq_pool_gate_guard() { tq_pool_free_all(); }
};

static void tq_pool_init_unwind(void) {
    try { tq_q().wait(); } catch (...) {}
    tq_pool_storage_free();
}

static int tq_pool_init_impl(int max_slots, int num_blocks, int page,
                             int max_seq, bool experimental_index) {
    if (max_slots < 1 || num_blocks < 1 || max_seq < 1 ||
        (page != 128 && page != 256 && !experimental_index) ||
        page < 1 || (page & (page - 1)))
        return -3;
    if (max_seq > INT_MAX - (page - 1)) return -3;
    tq_pool_storage_free();
    int plog = 0;
    while ((1 << plog) < page) ++plog;
    const int maxblk = (max_seq + page - 1) / page;
    if (maxblk <= 0 || (size_t)max_slots >
                           std::numeric_limits<size_t>::max() / (size_t)maxblk)
        return -3;
    const size_t btn = (size_t)max_slots * (size_t)maxblk;
    if (btn > std::numeric_limits<size_t>::max() / sizeof(int)) return -3;

    g_pg_page = page;
    g_pg_plog = plog;
    g_pg_nblocks = num_blocks;
    g_pg_maxslots = max_slots;
    g_pg_maxblk = maxblk;
    h_block_table = (int *)malloc(btn * sizeof(int));
    h_free = (int *)malloc((size_t)num_blocks * sizeof(int));
    h_blk_ref = (int *)calloc((size_t)num_blocks, sizeof(int));
    h_blk_active_ref = (int *)calloc((size_t)num_blocks, sizeof(int));
    h_blk_ckpt_ref = (int *)calloc((size_t)num_blocks, sizeof(int));
    h_blk_in_free = (uint8_t *)malloc((size_t)num_blocks);
    h_slot_nb = (int *)calloc((size_t)max_slots, sizeof(int));
    h_slot_resv_total = (int *)calloc((size_t)max_slots, sizeof(int));
    h_slot_resv_left = (int *)calloc((size_t)max_slots, sizeof(int));
    h_slot_dirty_lo = (int *)malloc((size_t)max_slots * sizeof(int));
    h_slot_dirty_hi = (int *)malloc((size_t)max_slots * sizeof(int));
    if (!h_block_table || !h_free || !h_blk_ref || !h_blk_active_ref ||
        !h_blk_ckpt_ref || !h_blk_in_free || !h_slot_nb ||
        !h_slot_resv_total || !h_slot_resv_left || !h_slot_dirty_lo ||
        !h_slot_dirty_hi) {
        tq_pool_storage_free();
        return -7;
    }
    for (size_t i = 0; i < btn; ++i) h_block_table[i] = -1;
    for (int i = 0; i < num_blocks; ++i) {
        h_free[i] = i;
        h_blk_in_free[i] = 1;
    }
    for (int s = 0; s < max_slots; ++s) {
        h_slot_dirty_lo[s] = maxblk;
        h_slot_dirty_hi[s] = -1;
    }
    if (experimental_index && tq_pool_index_init(num_blocks) != 0) {
        tq_pool_storage_free();
        return -7;
    }
    g_pg_nfree = num_blocks;
    g_pg_fhead = g_pg_ftail = 0;

    d_block_table = (int *)tq_try_dev_alloc(btn * sizeof(int),
                                            "pool.block_table");
    if (!d_block_table) {
        tq_pool_storage_free();
        return -7;
    }
    try {
        tq_q().memcpy(d_block_table, h_block_table, btn * sizeof(int));
        if (g_qwen.initialized) {
            size_t rows = 0, data_bytes = 0, scale_bytes = 0;
            if (!tq_checked_mul((size_t)num_blocks, (size_t)page, &rows) ||
                !tq_checked_mul(rows, (size_t)g_qwen.nkv, &rows) ||
                !tq_checked_mul(rows, (size_t)g_qwen.hd, &data_bytes) ||
                !tq_checked_mul(rows, sizeof(uint16_t), &scale_bytes)) {
                tq_pool_init_unwind();
                return -7;
            }
            for (int i = 0; i < g_qwen.L; ++i) {
                if (g_qwen.layer_types[i] != TQ_LAYER_FULL_ATTENTION) continue;
                g_pool_k[i] = (uint8_t *)tq_try_dev_alloc(data_bytes, "pool.k");
                g_pool_v[i] = (uint8_t *)tq_try_dev_alloc(data_bytes, "pool.v");
                g_pool_ks[i] =
                    (uint16_t *)tq_try_dev_alloc(scale_bytes, "pool.ks");
                g_pool_vs[i] =
                    (uint16_t *)tq_try_dev_alloc(scale_bytes, "pool.vs");
                if (!g_pool_k[i] || !g_pool_v[i] ||
                    !g_pool_ks[i] || !g_pool_vs[i]) {
                    tq_pool_init_unwind();
                    return -7;
                }
            }
        }
        tq_q().wait_and_throw();
    } catch (const std::exception &e) {
        fprintf(stderr, "paged KV initialization failed: %s\n", e.what());
        tq_pool_init_unwind();
        return -7;
    } catch (...) {
        tq_pool_init_unwind();
        return -7;
    }
    return 0;
}

// Gate-only pool init. Production uses tq_pool_init_impl(..., false).
static int tq_pool_init(int max_slots, int num_blocks, int page, int max_seq) {
    return tq_pool_init_impl(max_slots, num_blocks, page, max_seq, true);
}

static int tq_pool_alloc_block(bool checkpoint = false) {
    if (g_pg_nfree <= 0 || !h_free) return -1;
    const int b = h_free[g_pg_fhead];
    if (b < 0 || b >= g_pg_nblocks || !h_blk_in_free[b] || h_blk_ref[b] != 0)
        return -1;
    g_pg_fhead = (g_pg_fhead + 1) % g_pg_nblocks;
    --g_pg_nfree;
    h_blk_in_free[b] = 0;
    tq_pool_unpublish(b);
    h_blk_ref[b] = 1;
    if (checkpoint) h_blk_ckpt_ref[b] = 1;
    else h_blk_active_ref[b] = 1;
    return b;
}

static int tq_pool_block_ref(int b, bool checkpoint) {
    if (b < 0 || b >= g_pg_nblocks || !h_blk_ref ||
        h_blk_in_free[b] || h_blk_ref[b] <= 0 ||
        h_blk_ref[b] == INT_MAX)
        return -1;
    ++h_blk_ref[b];
    if (checkpoint) ++h_blk_ckpt_ref[b];
    else ++h_blk_active_ref[b];
    return 0;
}

static int tq_pool_block_unref(int b, bool checkpoint) {
    if (b < 0 || b >= g_pg_nblocks || !h_blk_ref || h_blk_in_free[b] ||
        h_blk_ref[b] <= 0)
        return -1;
    int &kind_ref = checkpoint ? h_blk_ckpt_ref[b] : h_blk_active_ref[b];
    if (kind_ref <= 0) return -1;
    if (h_blk_ref[b] != h_blk_active_ref[b] + h_blk_ckpt_ref[b])
        return -1;
    if (h_blk_ref[b] == 1 &&
        (kind_ref != 1 ||
         (checkpoint ? h_blk_active_ref[b] : h_blk_ckpt_ref[b]) != 0))
        return -1;
    --kind_ref;
    --h_blk_ref[b];
    if (h_blk_ref[b] == 0) {
        if (h_blk_active_ref[b] != 0 || h_blk_ckpt_ref[b] != 0 ||
            h_blk_in_free[b] || g_pg_nfree >= g_pg_nblocks)
            return -1;
        h_free[g_pg_ftail] = b;
        g_pg_ftail = (g_pg_ftail + 1) % g_pg_nblocks;
        h_blk_in_free[b] = 1;
        ++g_pg_nfree;
    }
    return 0;
}

static void tq_pool_mark_dirty(int slot, int lo, int hi) {
    if (slot < 0 || slot >= g_pg_maxslots || lo > hi) return;
    h_slot_dirty_lo[slot] = std::min(h_slot_dirty_lo[slot], lo);
    h_slot_dirty_hi[slot] = std::max(h_slot_dirty_hi[slot], hi);
}

static int tq_pool_sync_slot(int slot) {
    if (!d_block_table || slot < 0 || slot >= g_pg_maxslots) return -1;
    const int lo = h_slot_dirty_lo[slot], hi = h_slot_dirty_hi[slot];
    if (lo > hi) return 0;
    const size_t base = (size_t)slot * g_pg_maxblk + (size_t)lo;
    try {
        tq_q().memcpy(d_block_table + base, h_block_table + base,
                      (size_t)(hi - lo + 1) * sizeof(int));
    } catch (...) {
        return -1;
    }
    h_slot_dirty_lo[slot] = g_pg_maxblk;
    h_slot_dirty_hi[slot] = -1;
    return 0;
}

static int tq_pool_sync(void) {
    for (int s = 0; s < g_pg_maxslots; ++s)
        if (tq_pool_sync_slot(s) != 0) return -1;
    return 0;
}

static int tq_pool_prepare_many(const int *slots, const int *positions,
                                const int *counts, int n) {
    if (!g_pg_enabled) return 0;
    if (!g_pg_ready || !slots || !positions || !counts || n < 1 || n > 8)
        return -1;
    int add[8] = {}, old_nb[8] = {};
    long long unreserved_add = 0, all_add = 0;
    for (int i = 0; i < n; ++i) {
        const int s = slots[i], pos = positions[i], count = counts[i];
        if (s < 0 || s >= g_pg_maxslots || pos < 0 || count <= 0 ||
            pos > g_qwen.max_seq - count ||
            g_qwen.state_pos[s] != pos)
            return -2;
        for (int j = 0; j < i; ++j)
            if (slots[j] == s) return -2;
        const int committed_blocks =
            (pos + g_pg_page - 1) >> g_pg_plog;
        if (h_slot_nb[s] != committed_blocks) return -2;
        if ((pos & (g_pg_page - 1)) != 0) {
            const int b = h_block_table[(size_t)s * g_pg_maxblk +
                                        committed_blocks - 1];
            if (b < 0 || h_blk_active_ref[b] != 1 ||
                h_blk_ckpt_ref[b] != 0 || h_blk_ref[b] != 1)
                return -2;
        }
        const int need = (pos + count + g_pg_page - 1) >> g_pg_plog;
        if (need > g_pg_maxblk) return -2;
        old_nb[i] = h_slot_nb[s];
        add[i] = need - old_nb[i];
        if (add[i] < 0) add[i] = 0;
        if (h_slot_resv_total[s]) {
            if (need > h_slot_resv_total[s] ||
                add[i] > h_slot_resv_left[s])
                return TQ_PAGED_ERR_CAPACITY;
        } else {
            unreserved_add += add[i];
        }
        all_add += add[i];
    }
    if (all_add > g_pg_nfree ||
        unreserved_add > (long long)g_pg_nfree - g_pg_reserved)
        return TQ_PAGED_ERR_CAPACITY;

    int completed = 0;
    for (int i = 0; i < n; ++i) {
        const int s = slots[i];
        int *row = h_block_table + (size_t)s * g_pg_maxblk;
        for (int j = 0; j < add[i]; ++j) {
            const int b = tq_pool_alloc_block(false);
            if (b < 0) goto rollback;
            row[h_slot_nb[s]++] = b;
            if (h_slot_resv_total[s]) {
                --h_slot_resv_left[s];
                --g_pg_reserved;
            }
        }
        if (add[i]) tq_pool_mark_dirty(s, old_nb[i], h_slot_nb[s] - 1);
        ++completed;
    }
    return 0;

rollback:
    for (int i = 0; i <= completed && i < n; ++i) {
        const int s = slots[i];
        int *row = h_block_table + (size_t)s * g_pg_maxblk;
        while (h_slot_nb[s] > old_nb[i]) {
            const int lb = --h_slot_nb[s];
            tq_pool_block_unref(row[lb], false);
            row[lb] = -1;
            if (h_slot_resv_total[s]) {
                ++h_slot_resv_left[s];
                ++g_pg_reserved;
            }
        }
    }
    return -1;
}

// Packed reset/extend transaction. Private pages from reset slots can move to
// any segment without touching their references; checkpoint-shared pages never
// become writable. Build the complete mapping before releasing old ownership.
static int tq_pool_prepare_packed(const int *slots, const int *positions,
                                  const int *counts, int n) {
    if (!g_pg_enabled) return 0;
    if (!g_pg_ready) return -7;
    bool resetting = false;
    for (int i = 0; i < n; ++i) resetting |= positions[i] == 0;
    if (!resetting) return tq_pool_prepare_many(slots, positions, counts, n);

    std::vector<int> old[8], next[8], reusable, allocated, shared;
    int keep[8] = {}, need[8] = {};
    long long demand = 0, future_reserved = g_pg_reserved;
    try {
        for (int i = 0; i < n; ++i) {
            const int s = slots[i], pos = positions[i];
            const int *row = h_block_table + (size_t)s * g_pg_maxblk;
            old[i].assign(row, row + h_slot_nb[s]);
            need[i] = (int)(((long long)pos + counts[i] + g_pg_page - 1) >>
                            g_pg_plog);
            if (need[i] > g_pg_maxblk) return -3;
            for (int b : old[i]) {
                if (b < 0 || b >= g_pg_nblocks || h_blk_in_free[b] ||
                    h_blk_active_ref[b] < 1 ||
                    h_blk_ref[b] != h_blk_active_ref[b] + h_blk_ckpt_ref[b])
                    return -7;
                if (pos == 0 && h_blk_ref[b] == 1) reusable.push_back(b);
                if (pos == 0 && h_blk_ref[b] > 1) shared.push_back(b);
            }
            if (pos != 0) {
                keep[i] = (int)(((long long)pos + g_pg_page - 1) >> g_pg_plog);
                if (h_slot_nb[s] != keep[i]) return -3;
                if ((pos & (g_pg_page - 1)) &&
                    h_blk_ref[old[i].back()] != 1) return -3;
            }
            const int add = need[i] - keep[i];
            demand += add;
            if (h_slot_resv_total[s]) {
                if (need[i] > h_slot_resv_total[s]) return TQ_PAGED_ERR_CAPACITY;
                const int left = h_slot_resv_total[s] - need[i];
                future_reserved += left - h_slot_resv_left[s];
            }
            next[i].resize(need[i]);
            std::copy(old[i].begin(), old[i].begin() + keep[i], next[i].begin());
        }
        const long long future_free =
            (long long)g_pg_nfree + (long long)reusable.size() - demand;
        if (future_free < 0 || future_free < future_reserved)
            return TQ_PAGED_ERR_CAPACITY;
        const long long extra = std::max(0LL, demand - (long long)reusable.size());
        allocated.reserve((size_t)extra);
        for (long long j = 0; j < extra; ++j) {
            const int b = tq_pool_alloc_block(false);
            if (b < 0) {
                for (int a : allocated) tq_pool_block_unref(a, false);
                return -7;
            }
            allocated.push_back(b);
        }
    } catch (...) {
        for (int b : allocated) tq_pool_block_unref(b, false);
        return -6;
    }

    size_t used = 0, fresh = 0;
    for (int i = 0; i < n; ++i)
        for (int lb = keep[i]; lb < need[i]; ++lb)
            next[i][lb] = used < reusable.size() ? reusable[used++] : allocated[fresh++];
    // All capacity and host allocation failures are behind us. Transfer the
    // exclusive references and drop only checkpoint-shared or unused pages.
    for (int b : shared)
        if (tq_pool_block_unref(b, false) != 0) return -7;
    for (size_t j = used; j < reusable.size(); ++j)
        if (tq_pool_block_unref(reusable[j], false) != 0) return -7;
    for (int i = 0; i < n; ++i) {
        const int s = slots[i], prev = h_slot_nb[s];
        int *row = h_block_table + (size_t)s * g_pg_maxblk;
        std::copy(next[i].begin(), next[i].end(), row);
        if (need[i] < prev) std::fill(row + need[i], row + prev, -1);
        h_slot_nb[s] = need[i];
        if (h_slot_resv_total[s])
            h_slot_resv_left[s] = h_slot_resv_total[s] - need[i];
        tq_pool_mark_dirty(s, 0, std::max(prev, need[i]) - 1);
    }
    g_pg_reserved = future_reserved;
    return 0;
}

// Historical gate helper uses an inclusive token index.
static int tq_pool_ensure(int slot, int pos) {
    if (slot < 0 || slot >= g_pg_maxslots || pos < 0) return -3;
    const int need = (pos >> g_pg_plog) + 1;
    if (need > g_pg_maxblk) return -1;
    const int old_nb = h_slot_nb[slot];
    const int add = need - old_nb;
    if (add <= 0) return 0;
    if (add > g_pg_nfree) return -2;
    int *row = h_block_table + (size_t)slot * g_pg_maxblk;
    while (h_slot_nb[slot] < need) {
        const int b = tq_pool_alloc_block(false);
        if (b < 0) {
            while (h_slot_nb[slot] > old_nb) {
                const int lb = --h_slot_nb[slot];
                tq_pool_block_unref(row[lb], false);
                row[lb] = -1;
            }
            return -2;
        }
        row[h_slot_nb[slot]++] = b;
    }
    tq_pool_mark_dirty(slot, old_nb, need - 1);
    return 0;
}

static int tq_pool_release_maps(int slot, bool preserve_reservation) {
    if (slot < 0 || slot >= g_pg_maxslots) return -2;
    const int old_nb = h_slot_nb[slot];
    const int old_left = h_slot_resv_left[slot];
    if (preserve_reservation && h_slot_resv_total[slot]) {
        int will_free = 0;
        const int *row = h_block_table + (size_t)slot * g_pg_maxblk;
        for (int lb = 0; lb < old_nb; ++lb)
            if (h_blk_ref[row[lb]] == 1) ++will_free;
        const long long future = g_pg_reserved - old_left +
                                 h_slot_resv_total[slot];
        if (future > (long long)g_pg_nfree + will_free)
            return TQ_PAGED_ERR_CAPACITY;
    }
    int *row = h_block_table + (size_t)slot * g_pg_maxblk;
    for (int lb = 0; lb < old_nb; ++lb) {
        if (tq_pool_block_unref(row[lb], false) != 0) return -1;
        row[lb] = -1;
    }
    if (old_nb) tq_pool_mark_dirty(slot, 0, old_nb - 1);
    h_slot_nb[slot] = 0;
    g_pg_reserved -= old_left;
    if (preserve_reservation) {
        h_slot_resv_left[slot] = h_slot_resv_total[slot];
        g_pg_reserved += h_slot_resv_left[slot];
    } else {
        h_slot_resv_total[slot] = 0;
        h_slot_resv_left[slot] = 0;
    }
    g_qwen.state_pos[slot] = 0;
    return 0;
}

static void tq_pool_release_slot(int slot) {
    (void)tq_pool_release_maps(slot, false);
}

// Historical gate primitive retains an inclusive prefix.
static int tq_pool_retain_prefix(int slot, int pos) {
    if (slot < 0 || slot >= g_pg_maxslots || pos < 0) return -3;
    const int nb = (pos >> g_pg_plog) + 1;
    if (nb > h_slot_nb[slot]) return -1;
    const int *row = h_block_table + (size_t)slot * g_pg_maxblk;
    int retained = 0;
    for (; retained < nb; ++retained) {
        if (tq_pool_block_ref(row[retained], true) != 0) break;
    }
    if (retained != nb) {
        while (retained-- > 0) tq_pool_block_unref(row[retained], true);
        return -1;
    }
    return nb;
}

static tq_kv_layout_t tq_pool_layout(int slot) {
    if (!g_pg_enabled)
        return tq_kv_layout_t{nullptr, 0, 0};
    return tq_kv_layout_t{
        d_block_table + (size_t)slot * g_pg_maxblk,
        g_pg_plog,
        g_pg_page - 1,
    };
}

static int tq_env_u64(const char *name, unsigned long long *out) {
    const char *s = getenv(name);
    if (!s || !*s) return 0;
    if (*s == '-') return -1;
    char *end = nullptr;
    errno = 0;
    const unsigned long long v = strtoull(s, &end, 10);
    if (errno || end == s || *end || v == 0) return -1;
    *out = v;
    return 1;
}

int tq_paged_init_from_env(void) {
    const char *mode = getenv("TQ_XPU_PAGED");
    g_pg_enabled = g_pg_ready = 0;
    if (!mode || !*mode || (mode[0] == '0' && mode[1] == '\0')) return 0;
    if (!(mode[0] == '1' && mode[1] == '\0')) {
        fprintf(stderr, "TQ_XPU_PAGED must be exactly 0 or 1\n");
        return -1;
    }
    if (!g_qwen.initialized || g_qwen.max_seq <= 0 || g_qwen.slots < 1)
        return -1;
    unsigned long long requested_tp = 1;
    const int tp_set = tq_env_u64("TQ_XPU_TP", &requested_tp);
    if (tp_set < 0 || requested_tp != 1 || tq_tp_size() != 1) {
        fprintf(stderr, "paged KV requires TQ_XPU_TP=1\n");
        return -1;
    }
    unsigned long long page_u = 128;
    const int page_set = tq_env_u64("TQ_XPU_KV_PAGE", &page_u);
    if (page_set < 0 || (page_u != 128 && page_u != 256)) {
        fprintf(stderr, "TQ_XPU_KV_PAGE must be 128 or 256\n");
        return -1;
    }
    unsigned long long token_u = 0, mb_u = 0;
    const int token_set = tq_env_u64("TQ_XPU_KV_POOL_TOKENS", &token_u);
    const int mb_set = tq_env_u64("TQ_XPU_KV_POOL_MB", &mb_u);
    if (token_set < 0 || mb_set < 0 || token_set == mb_set) {
        fprintf(stderr, "paged KV requires exactly one positive pool budget "
                        "(TQ_XPU_KV_POOL_TOKENS or TQ_XPU_KV_POOL_MB)\n");
        return -1;
    }
    const int page = (int)page_u;
    int nblocks = 0;
    if (token_set) {
        const unsigned long long nb = token_u / (unsigned long long)page;
        if (!nb || nb > (unsigned long long)INT_MAX) return -1;
        nblocks = (int)nb;
    } else {
        if (mb_u > (unsigned long long)
                       std::numeric_limits<size_t>::max() / 1000000ull)
            return -1;
        const size_t budget = (size_t)mb_u * 1000000u;
        nblocks = tq_pool_blocks_for_budget(budget, page);
        if (nblocks < 1) return -1;
    }
    g_pg_enabled = 1;
    const int rc = tq_pool_init_impl(g_qwen.slots, nblocks, page,
                                     g_qwen.max_seq, false);
    if (rc != 0) {
        g_pg_enabled = 0;
        tq_pool_storage_free();
        return rc;
    }
    g_pg_ready = 1;
    fprintf(stderr, "paged KV: page=%d blocks=%d tokens=%lld bytes/block=%zu\n",
            page, nblocks, (long long)nblocks * page,
            tq_pool_bytes_per_block(page));
    return 0;
}

int tq_paged_ready(void) { return g_pg_enabled && g_pg_ready; }

void tq_paged_free(void) {
    if (g_pg_ready) {
        try { tq_q().wait_and_throw(); } catch (...) {}
    }
    tq_paged_checkpoints_free_all();
    tq_pool_storage_free();
    g_pg_ready = g_pg_enabled = 0;
}

extern "C" int qwn_paged_enabled(void) {
    return g_qwen.initialized && g_pg_enabled && g_pg_ready;
}

extern "C" int qwn_paged_reserve(int slot, int total_tokens) {
    if (!qwn_paged_enabled()) return -1;
    if (slot < 0 || slot >= g_pg_maxslots ||
        total_tokens < 0 || total_tokens > g_qwen.max_seq)
        return -2;
    const int bound = total_tokens
                          ? (total_tokens + g_pg_page - 1) >> g_pg_plog
                          : 0;
    if (bound > g_pg_nblocks || bound < h_slot_nb[slot])
        return TQ_PAGED_ERR_CAPACITY;
    const int old_left = h_slot_resv_left[slot];
    const int new_left = bound - h_slot_nb[slot];
    const long long other = g_pg_reserved - old_left;
    if (new_left > (long long)g_pg_nfree - other)
        return TQ_PAGED_ERR_CAPACITY;
    h_slot_resv_total[slot] = bound;
    h_slot_resv_left[slot] = new_left;
    g_pg_reserved = other + new_left;
    return 0;
}

extern "C" int qwn_paged_release(int slot) {
    if (!qwn_paged_enabled()) return -1;
    if (slot < 0 || slot >= g_pg_maxslots) return -2;
    try {
        tq_q().wait_and_throw();
    } catch (...) {
        return -3;
    }
    const int rc = tq_pool_release_maps(slot, false);
    return rc != 0 ? rc : tq_pool_sync_slot(slot);
}

extern "C" int qwn_paged_stats(long long *out, int count) {
    if (!qwn_paged_enabled() || !out || count < 8) return -1;
    long long active = 0, checkpoints = 0;
    for (int b = 0; b < g_pg_nblocks; ++b) {
        if (h_blk_active_ref[b] > 0) ++active;
        if (h_blk_ckpt_ref[b] > 0) ++checkpoints;
    }
    out[0] = g_pg_page;
    out[1] = g_pg_nblocks;
    out[2] = g_pg_nfree;
    out[3] = active;
    out[4] = checkpoints;
    out[5] = g_pg_reserved;
    out[6] = g_qwen.max_seq;
    out[7] = g_pg_maxslots;
    return 0;
}

int tq_paged_reset_all(void) {
    if (!g_pg_enabled) return 0;
    if (!g_pg_ready) return -1;
    try {
        tq_q().wait_and_throw();
    } catch (...) {
        return -2;
    }
    tq_paged_checkpoints_free_all();
    for (int s = 0; s < g_pg_maxslots; ++s) {
        const int rc = tq_pool_release_maps(s, false);
        if (rc != 0) return rc;
    }
    return tq_pool_sync();
}

int tq_paged_reset_slot(int slot) {
    if (!g_pg_enabled) return 0;
    if (!g_pg_ready || slot < 0 || slot >= g_pg_maxslots) return -1;
    try {
        tq_q().wait_and_throw();
    } catch (...) {
        return -2;
    }
    const int rc = tq_pool_release_maps(slot, true);
    return rc != 0 ? rc : tq_pool_sync_slot(slot);
}

static size_t tq_paged_gdn_bytes(void) {
    size_t per_layer = 0;
    if (slot_conv_elems() >
            std::numeric_limits<size_t>::max() / sizeof(float) ||
        slot_recur_elems() >
            std::numeric_limits<size_t>::max() / sizeof(float))
        return 0;
    const size_t conv = slot_conv_elems() * sizeof(float);
    const size_t recur = slot_recur_elems() * sizeof(float);
    if (conv > std::numeric_limits<size_t>::max() - recur) return 0;
    per_layer = conv + recur;
    size_t linear = 0;
    for (int i = 0; i < g_qwen.L; ++i)
        if (g_qwen.layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) ++linear;
    size_t total = 0;
    return tq_checked_mul(per_layer, linear, &total) ? total : 0;
}

static size_t tq_paged_gdn_copy(int slot, void *host, bool save) {
    uint8_t *h = static_cast<uint8_t *>(host);
    size_t off = 0;
    const size_t conv = slot_conv_elems() * sizeof(float);
    const size_t recur = slot_recur_elems() * sizeof(float);
    for (int i = 0; i < g_qwen.L; ++i) {
        if (g_qwen.layer_types[i] != TQ_LAYER_LINEAR_ATTENTION) continue;
        tq_layer_t *l = &g_qwen.layers[i];
        if (!l->d_linear_conv_state || !l->d_linear_recurrent_state) return 0;
        if (save) {
            tq_q().memcpy(h + off, slot_conv_at(l, slot), conv);
            tq_q().memcpy(h + off + conv, slot_recur_at(l, slot), recur);
        } else {
            tq_q().memcpy(slot_conv_at(l, slot), h + off, conv);
            tq_q().memcpy(slot_recur_at(l, slot), h + off + conv, recur);
        }
        off += conv + recur;
    }
    return off;
}

static void tq_paged_gdn_zero(int slot) {
    const size_t conv = slot_conv_elems() * sizeof(float);
    const size_t recur = slot_recur_elems() * sizeof(float);
    for (int i = 0; i < g_qwen.L; ++i) {
        if (g_qwen.layer_types[i] != TQ_LAYER_LINEAR_ATTENTION) continue;
        tq_layer_t *l = &g_qwen.layers[i];
        if (l->d_linear_conv_state) tq_q().memset(slot_conv_at(l, slot), 0, conv);
        if (l->d_linear_recurrent_state)
            tq_q().memset(slot_recur_at(l, slot), 0, recur);
    }
}

static void *tq_paged_host_alloc(size_t bytes) {
    if (!bytes) return nullptr;
    try {
        if (void *p = sycl::malloc_host(bytes, tq_q())) return p;
    } catch (...) {
    }
    return malloc(bytes);
}

static PagedCheckpoint *tq_paged_ckpt_find(int id) {
    if (id <= 0) return nullptr;
    for (PagedCheckpoint &c : g_pg_checkpoints)
        if (c.id == id) return &c;
    return nullptr;
}

static void tq_paged_ckpt_drop(PagedCheckpoint &c) {
    for (int b : c.full_blocks) (void)tq_pool_block_unref(b, true);
    if (c.tail_block >= 0) (void)tq_pool_block_unref(c.tail_block, true);
    if (c.gdn_state) qwn_host_free(c.gdn_state);
    c = PagedCheckpoint{};
}

static void tq_paged_checkpoints_free_all(void) {
    for (PagedCheckpoint &c : g_pg_checkpoints)
        if (c.id) tq_paged_ckpt_drop(c);
}

static bool tq_paged_copy_tail(int dst_block, int src_block, int valid_tokens) {
    if (dst_block < 0 || src_block < 0 || valid_tokens <= 0 ||
        valid_tokens >= g_pg_page)
        return false;
    const size_t rows = (size_t)valid_tokens * g_qwen.nkv;
    const size_t dst_row = (size_t)dst_block * g_pg_page * g_qwen.nkv;
    const size_t src_row = (size_t)src_block * g_pg_page * g_qwen.nkv;
    const size_t data_bytes = rows * g_qwen.hd;
    const size_t scale_bytes = rows * sizeof(uint16_t);
    for (int i = 0; i < g_qwen.L; ++i) {
        if (g_qwen.layer_types[i] != TQ_LAYER_FULL_ATTENTION) continue;
        tq_q().memcpy(g_pool_k[i] + dst_row * g_qwen.hd,
                      g_pool_k[i] + src_row * g_qwen.hd, data_bytes);
        tq_q().memcpy(g_pool_v[i] + dst_row * g_qwen.hd,
                      g_pool_v[i] + src_row * g_qwen.hd, data_bytes);
        tq_q().memcpy(g_pool_ks[i] + dst_row, g_pool_ks[i] + src_row,
                      scale_bytes);
        tq_q().memcpy(g_pool_vs[i] + dst_row, g_pool_vs[i] + src_row,
                      scale_bytes);
    }
    return true;
}

extern "C" int qwn_paged_ckpt_save(int slot, int committed) {
    if (!qwn_paged_enabled()) return -1;
    if (slot < 0 || slot >= g_pg_maxslots || committed < 0 ||
        committed > g_qwen.max_seq || g_qwen.state_pos[slot] != committed)
        return -2;
    const int full = committed >> g_pg_plog;
    const int tail_tokens = committed & (g_pg_page - 1);
    const int mapped = full + (tail_tokens ? 1 : 0);
    if (mapped != h_slot_nb[slot] || g_pg_next_ckpt_id <= 0) return -3;
    const int *row = h_block_table + (size_t)slot * g_pg_maxblk;
    for (int lb = 0; lb < mapped; ++lb)
        if (row[lb] < 0 || h_blk_active_ref[row[lb]] <= 0) return -3;
    if (tail_tokens) {
        const int b = row[full];
        if (h_blk_active_ref[b] != 1 || h_blk_ckpt_ref[b] != 0 ||
            h_blk_ref[b] != 1)
            return -3;
    }

    size_t registry_slot = g_pg_checkpoints.size();
    try {
        for (size_t i = 0; i < g_pg_checkpoints.size(); ++i)
            if (!g_pg_checkpoints[i].id) {
                registry_slot = i;
                break;
            }
        if (registry_slot == g_pg_checkpoints.size())
            g_pg_checkpoints.emplace_back();
    } catch (...) {
        return -4;
    }

    PagedCheckpoint candidate;
    candidate.committed = committed;
    try {
        candidate.full_blocks.assign(row, row + full);
    } catch (...) {
        return -4;
    }
    candidate.gdn_bytes = tq_paged_gdn_bytes();
    if (!candidate.gdn_bytes) return -4;
    candidate.gdn_state = tq_paged_host_alloc(candidate.gdn_bytes);
    if (!candidate.gdn_state) return -4;

    try {
        tq_q().wait_and_throw();
        if (tail_tokens) {
            if ((long long)g_pg_nfree - g_pg_reserved < 1) {
                qwn_host_free(candidate.gdn_state);
                return TQ_PAGED_ERR_CAPACITY;
            }
            candidate.tail_block = tq_pool_alloc_block(true);
            if (candidate.tail_block < 0 ||
                !tq_paged_copy_tail(candidate.tail_block, row[full],
                                    tail_tokens)) {
                if (candidate.tail_block >= 0)
                    tq_pool_block_unref(candidate.tail_block, true);
                qwn_host_free(candidate.gdn_state);
                return -4;
            }
        }
        if (tq_paged_gdn_copy(slot, candidate.gdn_state, true) !=
            candidate.gdn_bytes)
            throw std::runtime_error("missing GDN checkpoint storage");
        tq_q().wait_and_throw();
    } catch (const std::exception &e) {
        try { tq_q().wait(); } catch (...) {}
        fprintf(stderr, "paged checkpoint save failed: %s\n", e.what());
        if (candidate.tail_block >= 0)
            tq_pool_block_unref(candidate.tail_block, true);
        qwn_host_free(candidate.gdn_state);
        return -5;
    } catch (...) {
        try { tq_q().wait(); } catch (...) {}
        if (candidate.tail_block >= 0)
            tq_pool_block_unref(candidate.tail_block, true);
        qwn_host_free(candidate.gdn_state);
        return -5;
    }

    int retained = 0;
    for (; retained < full; ++retained)
        if (tq_pool_block_ref(candidate.full_blocks[retained], true) != 0)
            break;
    if (retained != full) {
        while (retained-- > 0)
            tq_pool_block_unref(candidate.full_blocks[retained], true);
        if (candidate.tail_block >= 0)
            tq_pool_block_unref(candidate.tail_block, true);
        qwn_host_free(candidate.gdn_state);
        return -5;
    }
    candidate.id = g_pg_next_ckpt_id;
    g_pg_next_ckpt_id =
        (g_pg_next_ckpt_id == INT_MAX) ? 0 : g_pg_next_ckpt_id + 1;
    g_pg_checkpoints[registry_slot] = std::move(candidate);
    return g_pg_checkpoints[registry_slot].id;
}

extern "C" int qwn_paged_ckpt_adopt(int slot, int id) {
    if (!qwn_paged_enabled()) return -1;
    if (slot < 0 || slot >= g_pg_maxslots || h_slot_nb[slot] != 0 ||
        g_qwen.state_pos[slot] != 0)
        return -2;
    PagedCheckpoint *c = tq_paged_ckpt_find(id);
    if (!c || c->committed < 0 || c->committed > g_qwen.max_seq) return -3;
    const int full = c->committed >> g_pg_plog;
    const int tail_tokens = c->committed & (g_pg_page - 1);
    const int mapped = full + (tail_tokens ? 1 : 0);
    if ((int)c->full_blocks.size() != full || mapped > g_pg_maxblk ||
        (h_slot_resv_total[slot] && mapped > h_slot_resv_total[slot]))
        return -3;
    for (int b : c->full_blocks)
        if (b < 0 || b >= g_pg_nblocks || h_blk_ckpt_ref[b] <= 0)
            return -3;
    if (tail_tokens &&
        (c->tail_block < 0 || c->tail_block >= g_pg_nblocks ||
         h_blk_ckpt_ref[c->tail_block] <= 0))
        return -3;
    if (tail_tokens && (long long)g_pg_nfree - g_pg_reserved < 1)
        return TQ_PAGED_ERR_CAPACITY;

    int target_tail = -1;
    try {
        if (tail_tokens) {
            target_tail = tq_pool_alloc_block(false);
            if (target_tail < 0 ||
                !tq_paged_copy_tail(target_tail, c->tail_block, tail_tokens))
                throw std::runtime_error("tail allocation failed");
        }
        if (tq_paged_gdn_copy(slot, c->gdn_state, false) != c->gdn_bytes)
            throw std::runtime_error("missing GDN restore storage");
        tq_q().wait_and_throw();
    } catch (const std::exception &e) {
        try { tq_q().wait(); } catch (...) {}
        fprintf(stderr, "paged checkpoint adopt failed: %s\n", e.what());
        if (target_tail >= 0) tq_pool_block_unref(target_tail, false);
        tq_paged_gdn_zero(slot);
        try { tq_q().wait_and_throw(); } catch (...) {}
        return -4;
    } catch (...) {
        try { tq_q().wait(); } catch (...) {}
        if (target_tail >= 0) tq_pool_block_unref(target_tail, false);
        tq_paged_gdn_zero(slot);
        try { tq_q().wait_and_throw(); } catch (...) {}
        return -4;
    }

    int retained = 0;
    for (; retained < full; ++retained)
        if (tq_pool_block_ref(c->full_blocks[retained], false) != 0) break;
    if (retained != full) {
        while (retained-- > 0)
            tq_pool_block_unref(c->full_blocks[retained], false);
        if (target_tail >= 0) tq_pool_block_unref(target_tail, false);
        tq_paged_gdn_zero(slot);
        try { tq_q().wait_and_throw(); } catch (...) {}
        return -4;
    }
    int *row = h_block_table + (size_t)slot * g_pg_maxblk;
    for (int lb = 0; lb < full; ++lb) row[lb] = c->full_blocks[lb];
    if (tail_tokens) row[full] = target_tail;
    h_slot_nb[slot] = mapped;
    if (mapped) tq_pool_mark_dirty(slot, 0, mapped - 1);
    const int old_left = h_slot_resv_left[slot];
    if (h_slot_resv_total[slot]) {
        h_slot_resv_left[slot] = h_slot_resv_total[slot] - mapped;
        g_pg_reserved += h_slot_resv_left[slot] - old_left;
    }
    if (tq_pool_sync_slot(slot) != 0) {
        if (h_slot_resv_total[slot]) {
            g_pg_reserved += old_left - h_slot_resv_left[slot];
            h_slot_resv_left[slot] = old_left;
        }
        for (int lb = 0; lb < full; ++lb) {
            tq_pool_block_unref(row[lb], false);
            row[lb] = -1;
        }
        if (tail_tokens) {
            tq_pool_block_unref(target_tail, false);
            row[full] = -1;
        }
        h_slot_nb[slot] = 0;
        if (mapped) tq_pool_mark_dirty(slot, 0, mapped - 1);
        tq_paged_gdn_zero(slot);
        try { tq_q().wait_and_throw(); } catch (...) {}
        return -4;
    }
    g_qwen.state_pos[slot] = c->committed;
    return c->committed;
}

extern "C" int qwn_paged_ckpt_free(int id) {
    if (!qwn_paged_enabled()) return -1;
    PagedCheckpoint *c = tq_paged_ckpt_find(id);
    if (!c) return -2;
    try {
        tq_q().wait_and_throw();
    } catch (...) {
        return -3;
    }
    tq_paged_ckpt_drop(*c);
    return 0;
}

extern "C" size_t qwn_paged_ckpt_host_bytes(int id) {
    PagedCheckpoint *c =
        qwn_paged_enabled() ? tq_paged_ckpt_find(id) : nullptr;
    return c ? c->gdn_bytes : 0;
}

extern "C" size_t qwn_debug_kv_bytes(int committed) {
    if (!g_qwen.initialized || committed < 0 ||
        committed > g_qwen.max_seq)
        return 0;
    size_t rows = 0, data = 0, scales = 0;
    if (!tq_checked_mul((size_t)committed, (size_t)g_qwen.nkv, &rows) ||
        !tq_checked_mul(rows, (size_t)g_qwen.hd, &data) ||
        !tq_checked_mul(rows, sizeof(uint16_t), &scales) ||
        scales > std::numeric_limits<size_t>::max() / 2 ||
        data > (std::numeric_limits<size_t>::max() - 2 * scales) / 2)
        return 0;
    return 2 * data + 2 * scales;
}

extern "C" int qwn_debug_kv_copy(int slot, int layer, int committed,
                                  void *out, size_t bytes) {
    const int S = g_qwen.slots < 1 ? 1 : g_qwen.slots;
    const size_t need = qwn_debug_kv_bytes(committed);
    if (!g_qwen.initialized || !out || !need || bytes < need ||
        slot < 0 || slot >= S || layer < 0 || layer >= g_qwen.L ||
        g_qwen.layer_types[layer] != TQ_LAYER_FULL_ATTENTION ||
        committed > g_qwen.state_pos[slot])
        return -1;
    tq_layer_t *l = &g_qwen.layers[layer];
    uint8_t *dst = static_cast<uint8_t *>(out);
    const size_t all_rows = (size_t)committed * g_qwen.nkv;
    const size_t data = all_rows * g_qwen.hd;
    const size_t scales = all_rows * sizeof(uint16_t);
    try {
        if (!g_pg_enabled) {
            tq_q().memcpy(dst, slot_kc_at(l, slot), data);
            tq_q().memcpy(dst + data, slot_vc_at(l, slot), data);
            tq_q().memcpy(dst + 2 * data, slot_ks_at(l, slot), scales);
            tq_q().memcpy(dst + 2 * data + scales, slot_vs_at(l, slot),
                          scales);
        } else {
            const int *table =
                h_block_table + (size_t)slot * g_pg_maxblk;
            size_t logical = 0;
            while (logical < (size_t)committed) {
                const int lb = (int)(logical >> g_pg_plog);
                if (lb >= h_slot_nb[slot] || table[lb] < 0) return -2;
                const size_t take = std::min(
                    (size_t)g_pg_page - (logical & (g_pg_page - 1)),
                    (size_t)committed - logical);
                const size_t src_row =
                    ((size_t)table[lb] * g_pg_page +
                     (logical & (g_pg_page - 1))) * g_qwen.nkv;
                const size_t dst_row = logical * g_qwen.nkv;
                const size_t row_count = take * g_qwen.nkv;
                tq_q().memcpy(dst + dst_row * g_qwen.hd,
                              g_pool_k[layer] + src_row * g_qwen.hd,
                              row_count * g_qwen.hd);
                tq_q().memcpy(dst + data + dst_row * g_qwen.hd,
                              g_pool_v[layer] + src_row * g_qwen.hd,
                              row_count * g_qwen.hd);
                tq_q().memcpy(dst + 2 * data +
                                  dst_row * sizeof(uint16_t),
                              g_pool_ks[layer] + src_row,
                              row_count * sizeof(uint16_t));
                tq_q().memcpy(dst + 2 * data + scales +
                                  dst_row * sizeof(uint16_t),
                              g_pool_vs[layer] + src_row,
                              row_count * sizeof(uint16_t));
                logical += take;
            }
        }
        tq_q().wait_and_throw();
    } catch (...) {
        return -3;
    }
    return 0;
}

extern "C" int qwn_debug_block_table(int slot, int *out, int count) {
    if (!qwn_paged_enabled() || slot < 0 || slot >= g_pg_maxslots ||
        !out || count < h_slot_nb[slot])
        return -1;
    const int n = h_slot_nb[slot];
    memcpy(out, h_block_table + (size_t)slot * g_pg_maxblk,
           (size_t)n * sizeof(int));
    return n;
}

static inline int tq_layer_index(const tq_layer_t *l) {
    return (int)(l - g_qwen.layers);
}
static inline uint8_t *kv_k_at(tq_layer_t *l, int slot) {
    return g_pg_enabled ? g_pool_k[tq_layer_index(l)] : slot_kc_at(l, slot);
}
static inline uint8_t *kv_v_at(tq_layer_t *l, int slot) {
    return g_pg_enabled ? g_pool_v[tq_layer_index(l)] : slot_vc_at(l, slot);
}
static inline uint16_t *kv_ks_at(tq_layer_t *l, int slot) {
    return g_pg_enabled ? g_pool_ks[tq_layer_index(l)] : slot_ks_at(l, slot);
}
static inline uint16_t *kv_vs_at(tq_layer_t *l, int slot) {
    return g_pg_enabled ? g_pool_vs[tq_layer_index(l)] : slot_vs_at(l, slot);
}
static inline uint8_t *slot_kc(tq_layer_t *l) {
    return kv_k_at(l, g_qwen.active_slot);
}
static inline uint8_t *slot_vc(tq_layer_t *l) {
    return kv_v_at(l, g_qwen.active_slot);
}
static inline uint16_t *slot_ks(tq_layer_t *l) {
    return kv_ks_at(l, g_qwen.active_slot);
}
static inline uint16_t *slot_vs(tq_layer_t *l) {
    return kv_vs_at(l, g_qwen.active_slot);
}
static inline float *slot_conv(tq_layer_t *l) { return slot_conv_at(l, g_qwen.active_slot); }
static inline float *slot_recur(tq_layer_t *l) { return slot_recur_at(l, g_qwen.active_slot); }

// CUDA: run_full_layer_decode_from_current (16319)
static int run_full_layer_decode(int layer, int pos) {
    tq_model_t *m = &g_qwen;
    tq_layer_t *l = &m->layers[layer];
    x_rmsnorm_quant(m->d_norm_out, m->d_x, l->d_input_ln, m->H, m->eps);
    profile_mark(kProfileNorm);
    int ret = 0;
    // q, k and v share d_norm_out and K: one fan-out launch replaces three.
    {
        const tq_qmma_weight_t *ws[3] = {&l->q_proj, &l->k_proj, &l->v_proj};
        float *ys[3] = {m->d_qkv, m->d_b, m->d_z};   // q+gate, k, v
        int fret = x_gemv_w4a8_fanout(ws, ys, 3);
        if (fret != 0) {
            ret = x_gemv_qmma_prepared(&l->q_proj, m->d_norm_out, m->d_qkv);
            if (ret != 0) return -5;
            ret = x_gemv_qmma_prepared(&l->k_proj, m->d_norm_out, m->d_b);
            if (ret != 0) return -6;
            ret = x_gemv_qmma_prepared(&l->v_proj, m->d_norm_out, m->d_z);
            if (ret != 0) return -7;
        }
        profile_projection(&l->q_proj);
        profile_projection(&l->k_proj);
        profile_projection(&l->v_proj);
    }
    x_full_attn_decode(m->d_core, m->d_qkv, m->d_b, m->d_z, l->d_q_norm, l->d_k_norm,
                       slot_kc(l), slot_vc(l), slot_ks(l), slot_vs(l),
                       m->d_scores, pos, m->nh, m->nkv,
                       m->hd, m->eps, m->rope_theta, m->partial_rotary_factor,
                       m->max_seq, tq_pool_layout(m->active_slot));
    profile_mark(kProfileAttention);
    ret = x_gemv_qmma_add(&l->o_proj, m->d_core, m->d_x, m->d_resid);
    profile_projection(&l->o_proj);
    if (ret != 0) return -8;
    return run_mlp_from_resid(l);
}

// CUDA: run_linear_layer_decode_from_current (16209)
static int run_linear_layer_decode(int layer) {
    tq_model_t *m = &g_qwen;
    tq_layer_t *l = &m->layers[layer];
    const int heads = m->linear_num_value_heads;
    const int key_heads = m->linear_num_key_heads;
    const int dim = m->linear_value_head_dim;
    const int key_dim = key_heads * dim;   // NOTE: key_dim uses value_head_dim (CUDA 16214)
    const int value_dim = heads * dim;
    const int conv_dim = key_dim * 2 + value_dim;

    x_rmsnorm_quant(m->d_norm_out, m->d_x, l->d_input_ln, m->H, m->eps);
    profile_mark(kProfileNorm);
    // The four input projections share d_norm_out and K, so one fan-out launch
    // replaces four. Falls back per-weight when the tier is not uniformly
    // W4-K32 (e.g. a TQ_XPU_W8 selector that covers only some of them).
    {
        const tq_qmma_weight_t *ws[4] = {&l->linear_in_qkv, &l->linear_in_z,
                                         &l->linear_in_b, &l->linear_in_a};
        float *ys[4] = {m->d_qkv, m->d_z, m->d_b, m->d_a};
        int ret = x_gemv_w4a8_fanout(ws, ys, 4);
        if (ret != 0) {
            ret = x_gemv_qmma_prepared(&l->linear_in_qkv, m->d_norm_out, m->d_qkv);
            if (ret != 0) return -6;
            ret = x_gemv_qmma_prepared(&l->linear_in_z, m->d_norm_out, m->d_z);
            if (ret != 0) return -7;
            ret = x_gemv_qmma_prepared(&l->linear_in_b, m->d_norm_out, m->d_b);
            if (ret != 0) return -8;
            ret = x_gemv_qmma_prepared(&l->linear_in_a, m->d_norm_out, m->d_a);
            if (ret != 0) return -9;
        }
        profile_projection(&l->linear_in_qkv);
    }
    // conv update runs in place on d_qkv with the layer's conv state (CUDA 16242)
    x_linear_conv_update(m->d_qkv, slot_conv(l), m->d_qkv,
                         l->d_linear_conv1d, conv_dim, m->linear_conv_kernel_dim);
    profile_mark(kProfileConv);
    x_linear_decode_core_gated(m->d_core, slot_recur(l), m->d_qkv,
                               m->d_z, m->d_b, m->d_a, l->d_linear_A_log,
                               l->d_linear_dt_bias, l->d_linear_norm,
                               key_heads, dim, heads, dim, m->eps);
    profile_mark(kProfileDelta);
    const int ret = x_gemv_qmma_add(&l->linear_out, m->d_core, m->d_x, m->d_resid);
    profile_projection(&l->linear_out);
    if (ret != 0) return -10;
    return run_mlp_from_resid(l);
}

// ---------------------------------------------------------------------------
// Chunked prefill.
//
// The per-token loop below streams all 15.05 GB of weights for EVERY prompt
// token, which is why TTFT is ctx/31.9 seconds. Here each projection runs once
// per chunk through the RC8 batched GEMM, so weight traffic is amortized over
// T tokens and the phase becomes DPAS-bound instead of bandwidth-bound.
//
// Attention, the depthwise conv and the DeltaNet recurrence stay token-serial
// inside the chunk: the recurrence is inherently sequential, and its inputs are
// all precomputed by the batched projections, so correctness is identical to
// the per-token path by construction. Making those register-resident across the
// chunk is the next step; it is what removes their remaining state traffic.
namespace {

struct prefill_scratch {
    int cap = 0;                 // tokens the buffers are sized for
    float *x = nullptr, *resid = nullptr, *normed = nullptr, *layer_out = nullptr;
    float *qkv = nullptr, *z = nullptr, *b = nullptr, *a = nullptr, *core = nullptr;
    float *gate = nullptr, *up = nullptr, *hidden = nullptr;
    float *conv = nullptr;     // chunked GDN conv output [T, conv_dim]
    int8_t *aq = nullptr;
    float *as = nullptr;
    int32_t *asum = nullptr;
    // W4A4 staging (prefill only; decode keeps s8 acts). Used when w4a4=1.
    uint8_t *aq4 = nullptr;
    float *as4 = nullptr;
    int32_t *asum4 = nullptr;
    int w4a4 = 0;              // resolved per prefill chunk (env + weights)
};
prefill_scratch g_pf;

// Spec-wave scratch (design 9.3). The wave runs the 8-row prefill path and
// must be able to REWIND GDN state to an accepted prefix: recurrent + conv
// state snapshots taken before the wave, plus per-layer caches of the wave
// inputs the re-advance needs (raw projections for the conv window, conv
// outputs and b/a for the delta-rule state math). Slots indexed by layer id
// (attention layers leave theirs untouched). KV needs no rewind: rows past
// the accepted prefix are dead by positional masking and overwritten by the
// next wave.
struct wave_scratch {
    int ready = 0;
    size_t row_w = 0;              // floats per cached row (qkv_w)
    float *qkv_cache = nullptr;    // [L][8 * row_w] raw in_qkv projections
    float *conv_cache = nullptr;   // [L][8 * row_w] post-conv outputs
    float *b_cache = nullptr;      // [L][8 * heads]
    float *a_cache = nullptr;      // [L][8 * heads]
    float *recur_snap = nullptr;   // [L][heads * dv * dv]
    float *conv_snap = nullptr;    // [L][conv_dim * ks]
    float *logits = nullptr;       // [8 * V]
    int capture = 0;               // pf_linear_layer caches inputs when set
};
wave_scratch g_wave;

void pf_free(void) {
    for (float **p : {&g_pf.x, &g_pf.resid, &g_pf.normed, &g_pf.layer_out,
                      &g_pf.qkv, &g_pf.z, &g_pf.b, &g_pf.a, &g_pf.core,
                      &g_pf.gate, &g_pf.up, &g_pf.hidden, &g_pf.conv,
                      &g_pf.as}) {
        if (*p) { tq_dev_free(*p); *p = nullptr; }
    }
    if (g_pf.aq) { tq_dev_free(g_pf.aq); g_pf.aq = nullptr; }
    if (g_pf.asum) { tq_dev_free(g_pf.asum); g_pf.asum = nullptr; }
    if (g_pf.aq4) { tq_dev_free(g_pf.aq4); g_pf.aq4 = nullptr; }
    if (g_pf.as4) { tq_dev_free(g_pf.as4); g_pf.as4 = nullptr; }
    if (g_pf.asum4) { tq_dev_free(g_pf.asum4); g_pf.asum4 = nullptr; }
    g_pf.cap = 0;
}

void wave_free(void) {
    for (float **p : {&g_wave.qkv_cache, &g_wave.conv_cache,
                      &g_wave.b_cache, &g_wave.a_cache,
                      &g_wave.recur_snap, &g_wave.conv_snap,
                      &g_wave.logits}) {
        if (*p) { tq_dev_free(*p); *p = nullptr; }
    }
    g_wave = wave_scratch{};
}

// Widest projection output and widest quantized K, so one activation-staging
// buffer serves every call site.
int pf_ensure(int T) {
    if (g_pf.cap >= T) return 0;
    pf_free();
    tq_model_t *m = &g_qwen;
    const size_t H = m->H, I = m->I, t = (size_t)T;
    const size_t qkv_w = 2u * (size_t)m->nh * m->hd;          // q+gate
    const size_t z_w = (size_t)m->linear_num_value_heads * m->linear_value_head_dim;
    const size_t core_w = qkv_w > z_w ? qkv_w : z_w;
    const size_t kmax = I > core_w ? I : core_w;
    auto fa = [&](size_t n, const char *what) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(float))
            return static_cast<float *>(nullptr);
        return static_cast<float *>(tq_try_dev_alloc(n * sizeof(float), what));
    };
    g_pf.x = fa(t * H, "pf.x");
    g_pf.resid = fa(t * H, "pf.resid");
    g_pf.normed = fa(t * H, "pf.normed");
    g_pf.layer_out = fa(t * H, "pf.layer_out");
    g_pf.qkv = fa(t * qkv_w, "pf.qkv");
    g_pf.z = fa(t * z_w, "pf.z");
    g_pf.b = fa(t * (size_t)m->nkv * m->hd, "pf.b");
    g_pf.a = fa(t * 64u, "pf.a");
    g_pf.core = fa(t * core_w, "pf.core");
    g_pf.gate = fa(t * I, "pf.gate");
    g_pf.up = fa(t * I, "pf.up");
    g_pf.hidden = fa(t * I, "pf.hidden");
    // GDN chunk conv output; conv_dim <= qkv_w by construction (the qkv
    // buffer already holds [T, conv_dim] rows for the linear layers).
    g_pf.conv = fa(t * qkv_w, "pf.conv");
    g_pf.aq = static_cast<int8_t *>(tq_try_dev_alloc(t * kmax, "pf.aq"));
    g_pf.as = fa(t * (kmax / 32u), "pf.as");
    const size_t asum_n = t * (kmax / 32u);
    g_pf.asum = asum_n <= std::numeric_limits<size_t>::max() / sizeof(int32_t)
                    ? static_cast<int32_t *>(tq_try_dev_alloc(
                          asum_n * sizeof(int32_t), "pf.asum"))
                    : nullptr;
    g_pf.aq4 = static_cast<uint8_t *>(
        tq_try_dev_alloc(t * kmax / 2u, "pf.aq4"));
    g_pf.as4 = fa(t * (kmax / 64u), "pf.as4");
    const size_t asum4_n = t * (kmax / 64u);
    g_pf.asum4 = asum4_n <= std::numeric_limits<size_t>::max() / sizeof(int32_t)
                     ? static_cast<int32_t *>(tq_try_dev_alloc(
                           asum4_n * sizeof(int32_t), "pf.asum4"))
                     : nullptr;
    if (!g_pf.x || !g_pf.resid || !g_pf.normed || !g_pf.layer_out || !g_pf.qkv ||
        !g_pf.z || !g_pf.b || !g_pf.a || !g_pf.core || !g_pf.gate || !g_pf.up ||
        !g_pf.hidden || !g_pf.conv || !g_pf.aq || !g_pf.as || !g_pf.asum ||
        !g_pf.aq4 ||
        !g_pf.as4 || !g_pf.asum4) {
        pf_free();
        return -1;
    }
    g_pf.cap = T;
    return 0;
}

// Stage only the layouts consumed by this projection fan-out. Mixed selectors
// still prepare both tiers, using the same per-weight predicate as pf_gemm.
bool pf_uses_s4(const tq_qmma_weight_t *w) {
    return g_pf.w4a4 && w->s4_k64;
}

void pf_stage(const float *src, int K, int T, const tq_qmma_weight_t *w0,
              const tq_qmma_weight_t *w1 = nullptr,
              const tq_qmma_weight_t *w2 = nullptr,
              const tq_qmma_weight_t *w3 = nullptr) {
    const tq_qmma_weight_t *weights[] = {w0, w1, w2, w3};
    bool need_s8 = false, need_s4 = false;
    for (const auto *w : weights) {
        if (!w) continue;
        if (pf_uses_s4(w)) need_s4 = true;
        else need_s8 = true;
    }
    if (need_s8)
        x_quantize_act_chunk(src, K, T, g_pf.aq, g_pf.as, g_pf.asum);
    if (need_s4)
        x_quantize_act_chunk_s4(src, K, T, g_pf.aq4, g_pf.as4, g_pf.asum4);
}

struct pf_segments {
    const int *slots;
    const int *offsets;
    const int *positions;
    int n;
};

int pf_gemm(const tq_qmma_weight_t *w, float *dst, int T,
             const pf_segments *batch = nullptr) {
    if (pf_uses_s4(w))
        return x_gemm_w4a4(w, g_pf.aq4, g_pf.as4, g_pf.asum4, dst, T);
    return x_gemm_w4a8(w, g_pf.aq, g_pf.as, g_pf.asum, dst, T,
                       batch ? batch->offsets : nullptr, batch ? batch->n : 0);
}

// Every projection a chunk touches must be W4-K32 for the batched path; a
// TQ_XPU_W8 selector covering a layer weight sends the caller back to the
// per-token loop rather than silently producing a different tier's numerics.
bool pf_layers_batchable(void) {
    tq_model_t *m = &g_qwen;
    auto ok = [](const tq_qmma_weight_t *w) {
        return w->s4_ready && !w->s4_k16 && w->d_s4 && w->d_s4_scale;
    };
    for (int i = 0; i < m->L; ++i) {
        tq_layer_t *l = &m->layers[i];
        if (!ok(&l->mlp_gate) || !ok(&l->mlp_up) || !ok(&l->mlp_down)) return false;
        if (m->layer_types[i] == TQ_LAYER_FULL_ATTENTION) {
            if (!ok(&l->q_proj) || !ok(&l->k_proj) || !ok(&l->v_proj) ||
                !ok(&l->o_proj)) return false;
        } else {
            if (!ok(&l->linear_in_qkv) || !ok(&l->linear_in_z) ||
                !ok(&l->linear_in_a) || !ok(&l->linear_in_b) ||
                !ok(&l->linear_out)) return false;
        }
    }
    return true;
}

// The W4A4 staging engages when ANY chunk projection carries K64 scales
// (TQ_XPU_K64 selector); dispatch is per-weight in pf_gemm.
bool pf_layers_any_k64(void) {
    tq_model_t *m = &g_qwen;
    for (int i = 0; i < m->L; ++i) {
        tq_layer_t *l = &m->layers[i];
        if (l->mlp_gate.s4_k64 || l->mlp_up.s4_k64 || l->mlp_down.s4_k64)
            return true;
        if (m->layer_types[i] == TQ_LAYER_FULL_ATTENTION) {
            if (l->q_proj.s4_k64 || l->k_proj.s4_k64 || l->v_proj.s4_k64 ||
                l->o_proj.s4_k64) return true;
        } else {
            if (l->linear_in_qkv.s4_k64 || l->linear_in_z.s4_k64 ||
                l->linear_in_a.s4_k64 || l->linear_in_b.s4_k64 ||
                l->linear_out.s4_k64) return true;
        }
    }
    return false;
}

int pf_mlp(tq_layer_t *l, int T, const pf_segments *batch = nullptr) {
    tq_model_t *m = &g_qwen;
    x_rmsnorm_chunk(g_pf.normed, g_pf.resid, l->d_post_ln, m->H, T, m->eps);
    pf_stage(g_pf.normed, m->H, T, &l->mlp_gate, &l->mlp_up);
    profile_mark(kProfileNorm);
    if (pf_gemm(&l->mlp_gate, g_pf.gate, T, batch) != 0) return -1;
    if (pf_gemm(&l->mlp_up, g_pf.up, T, batch) != 0) return -2;
    x_silu_mul(g_pf.hidden, g_pf.gate, g_pf.up, T * m->I);
    profile_mark(kProfileProjection);
    pf_stage(g_pf.hidden, m->I, T, &l->mlp_down);
    profile_mark(kProfileActivation);
    if (pf_gemm(&l->mlp_down, g_pf.layer_out, T, batch) != 0) return -3;
    x_add_inplace(g_pf.layer_out, g_pf.resid, T * m->H);
    profile_mark(kProfileProjection);
    return 0;
}

int pf_full_layer(int layer, int pos0, int T, const pf_segments *batch = nullptr) {
    tq_model_t *m = &g_qwen;
    tq_layer_t *l = &m->layers[layer];
    const int qkv_w = 2 * m->nh * m->hd;
    const int kv_w = m->nkv * m->hd;
    const int core_w = m->nh * m->hd;
    x_rmsnorm_chunk(g_pf.normed, g_pf.x, l->d_input_ln, m->H, T, m->eps);
    pf_stage(g_pf.normed, m->H, T, &l->q_proj, &l->k_proj, &l->v_proj);
    profile_mark(kProfileNorm);
    if (pf_gemm(&l->q_proj, g_pf.qkv, T, batch) != 0) return -1;
    if (pf_gemm(&l->k_proj, g_pf.b, T, batch) != 0) return -2;
    if (pf_gemm(&l->v_proj, g_pf.z, T, batch) != 0) return -3;
    profile_mark(kProfileProjection);
    // Queue every segment's KV write before the shared attention grid. Each
    // descriptor is an offset view into packed scratch and a slot-local KV map.
    tq_prefill_attn_request_t requests[8];
    const int segments = batch ? batch->n : 1;
    for (int i = 0; i < segments; ++i) {
        const int slot = batch ? batch->slots[i] : m->active_slot;
        const int first = batch ? batch->offsets[i] : 0;
        const int count = batch ? batch->offsets[i + 1] - first : T;
        const int pos = batch ? batch->positions[i] : pos0;
        x_prefill_kv_write(g_pf.b + (size_t)first * kv_w,
                           g_pf.z + (size_t)first * kv_w, l->d_k_norm,
                           kv_k_at(l, slot), kv_v_at(l, slot),
                           kv_ks_at(l, slot), kv_vs_at(l, slot), pos, count,
                           m->nkv, m->hd, m->eps, m->rope_theta,
                           m->partial_rotary_factor, tq_pool_layout(slot));
        requests[i] = {g_pf.core + (size_t)first * core_w,
                       g_pf.qkv + (size_t)first * qkv_w,
                       kv_k_at(l, slot), kv_v_at(l, slot),
                       kv_ks_at(l, slot), kv_vs_at(l, slot), pos, count,
                       tq_pool_layout(slot)};
    }
    if (batch) {
        x_prefill_attn_packed(requests, segments, l->d_q_norm, m->nh, m->nkv,
                              m->hd, m->eps, m->rope_theta,
                              m->partial_rotary_factor);
    } else {
        const auto &r = requests[0];
        x_prefill_attn(r.out, r.qg, l->d_q_norm, r.kc, r.vc, r.ks, r.vs,
                       r.pos0, r.T, m->nh, m->nkv, m->hd, m->eps,
                       m->rope_theta, m->partial_rotary_factor, r.layout);
    }
    profile_mark(kProfileAttention);
    pf_stage(g_pf.core, core_w, T, &l->o_proj);
    if (pf_gemm(&l->o_proj, g_pf.resid, T, batch) != 0) return -4;
    profile_mark(kProfileProjection);
    x_add_inplace(g_pf.resid, g_pf.x, T * m->H);
    profile_mark(kProfileProjection);
    return pf_mlp(l, T, batch);
}

// Projection outputs are densely packed at their actual output widths, not
// the maximum widths used to allocate scratch. Only state is slot-strided.
int pf_linear_segment(int layer, int first, int T, int slot, bool capture) {
    tq_model_t *m = &g_qwen;
    tq_layer_t *l = &m->layers[layer];
    const int heads = m->linear_num_value_heads;
    const int key_heads = m->linear_num_key_heads;
    const int dim = m->linear_value_head_dim;
    const int value_dim = heads * dim;
    const int conv_dim = key_heads * dim * 2 + value_dim;
    float *qkv = g_pf.qkv + (size_t)first * conv_dim;
    float *conv = g_pf.conv + (size_t)first * conv_dim;
    float *z = g_pf.z + (size_t)first * value_dim;
    float *b = g_pf.b + (size_t)first * heads;
    float *a = g_pf.a + (size_t)first * heads;
    float *core = g_pf.core + (size_t)first * value_dim;
    const char *dnc = getenv("TQ_XPU_DN_CHUNK");
    const bool chunked = (!dnc || dnc[0] != '0') && dim == 128 &&
                         key_heads > 0 && heads > 0 &&
                         (heads % key_heads) == 0;
    if (chunked) {
        x_linear_conv_chunk(conv, slot_conv_at(l, slot), qkv,
                            l->d_linear_conv1d, conv_dim,
                            m->linear_conv_kernel_dim, T);
        if (capture && T <= 8) {
            const size_t rowb = (size_t)conv_dim * sizeof(float);
            tq_d2d_async(g_wave.qkv_cache + (size_t)layer * 8 * g_wave.row_w,
                         qkv, (size_t)T * rowb);
            tq_d2d_async(g_wave.conv_cache + (size_t)layer * 8 * g_wave.row_w,
                         conv, (size_t)T * rowb);
            tq_d2d_async(g_wave.b_cache + (size_t)layer * 8 * heads, b,
                         (size_t)T * heads * sizeof(float));
            tq_d2d_async(g_wave.a_cache + (size_t)layer * 8 * heads, a,
                         (size_t)T * heads * sizeof(float));
        }
        if (x_deltanet_chunk(core, slot_recur_at(l, slot), conv,
                             z, b, a, l->d_linear_A_log,
                             l->d_linear_dt_bias, l->d_linear_norm, T,
                             key_heads, dim, heads, dim, m->eps) != 0)
            return -7;
    } else {
        for (int t = 0; t < T; ++t) {
            float *qkv_t = qkv + (size_t)t * conv_dim;
            x_linear_conv_update(qkv_t, slot_conv_at(l, slot), qkv_t,
                                 l->d_linear_conv1d, conv_dim,
                                 m->linear_conv_kernel_dim);
            x_linear_decode_core_gated(core + (size_t)t * value_dim,
                                       slot_recur_at(l, slot), qkv_t,
                                       z + (size_t)t * value_dim,
                                       b + (size_t)t * heads,
                                       a + (size_t)t * heads,
                                       l->d_linear_A_log, l->d_linear_dt_bias,
                                       l->d_linear_norm, key_heads, dim, heads,
                                       dim, m->eps);
        }
    }
    return 0;
}

int pf_linear_layer(int layer, int T, const pf_segments *batch = nullptr) {
    tq_model_t *m = &g_qwen;
    tq_layer_t *l = &m->layers[layer];
    const int heads = m->linear_num_value_heads;
    const int dim = m->linear_value_head_dim;
    const int value_dim = heads * dim;
    x_rmsnorm_chunk(g_pf.normed, g_pf.x, l->d_input_ln, m->H, T, m->eps);
    pf_stage(g_pf.normed, m->H, T, &l->linear_in_qkv, &l->linear_in_z,
              &l->linear_in_b, &l->linear_in_a);
    profile_mark(kProfileNorm);
    if (pf_gemm(&l->linear_in_qkv, g_pf.qkv, T, batch) != 0) return -1;
    if (pf_gemm(&l->linear_in_z, g_pf.z, T, batch) != 0) return -2;
    if (pf_gemm(&l->linear_in_b, g_pf.b, T, batch) != 0) return -3;
    if (pf_gemm(&l->linear_in_a, g_pf.a, T, batch) != 0) return -4;
    profile_mark(kProfileProjection);
    for (int i = 0; i < (batch ? batch->n : 1); ++i) {
        const int first = batch ? batch->offsets[i] : 0;
        const int count = batch ? batch->offsets[i + 1] - first : T;
        const int slot = batch ? batch->slots[i] : m->active_slot;
        if (pf_linear_segment(layer, first, count, slot,
                              !batch && g_wave.capture) != 0) return -7;
    }
    profile_mark(kProfileDelta);
    pf_stage(g_pf.core, value_dim, T, &l->linear_out);
    if (pf_gemm(&l->linear_out, g_pf.resid, T, batch) != 0) return -5;
    profile_mark(kProfileProjection);
    x_add_inplace(g_pf.resid, g_pf.x, T * m->H);
    profile_mark(kProfileProjection);
    return pf_mlp(l, T, batch);
}

// Embed + run every layer for an n-row chunk starting at pos0. Shared by
// qwn_prefill_chunk and the spec wave (design 9.3).
int pf_run_layers(const int *tokens, int n, int pos0,
                  const pf_segments *batch = nullptr) {
    tq_model_t *m = &g_qwen;
    for (int t = 0; t < n; ++t)
        x_embed_lookup(g_pf.x + (size_t)t * m->H, m->d_embed, tokens[t], m->H);
    for (int layer = 0; layer < m->L; ++layer) {
        int ret;
        if (m->layer_types[layer] == TQ_LAYER_LINEAR_ATTENTION)
            ret = pf_linear_layer(layer, n, batch);
        else if (m->layer_types[layer] == TQ_LAYER_FULL_ATTENTION)
            ret = pf_full_layer(layer, pos0, n, batch);
        else
            return -7;
        if (ret != 0) {
            fprintf(stderr, "prefill chunk: layer %d failed ret=%d\n", layer, ret);
            return -8;
        }
        float *old = g_pf.x;
        g_pf.x = g_pf.layer_out;
        g_pf.layer_out = old;
    }
    return 0;
}

}  // namespace

// One embedding/layer walk over offsets[n] rows; segments differ only at the
// stateful attention/GDN cores. No logits: callers keep their final prompt
// token for decode_batch. Validation and capacity failures mutate no slots.
extern "C" int qwn_prefill_batch(const int *slots, const int *tokens,
                                 const int *offsets, const int *positions,
                                 int n) {
    tq_model_t *m = &g_qwen;
    if (!m->initialized || !slots || !tokens || !offsets || !positions) return -1;
    const int S = m->slots < 1 ? 1 : m->slots;
    if (n < 1 || n > S || n > 8 || offsets[0] != 0) return -2;
    int counts[8];
    for (int i = 0; i < n; ++i) {
        if (slots[i] < 0 || slots[i] >= S) return -2;
        for (int j = 0; j < i; ++j)
            if (slots[i] == slots[j]) return -2;
        const long long length = (long long)offsets[i + 1] - offsets[i];
        if (offsets[i] < 0 || length <= 0 || length > INT_MAX || length % 8)
            return -2;
        counts[i] = (int)length;
        if (positions[i] < 0 || length > m->max_seq ||
            positions[i] > m->max_seq - length ||
            (positions[i] != 0 && positions[i] != m->state_pos[slots[i]]))
            return -3;
    }
    const int T = offsets[n];
    const int qkv_w = 2 * m->nh * m->hd;
    const int kv_w = m->nkv * m->hd;
    const int heads = m->linear_num_value_heads;
    const int dim = m->linear_value_head_dim;
    const int value_dim = heads * dim;
    const int conv_dim = 2 * m->linear_num_key_heads * dim + value_dim;
    const int widest = std::max({m->H, m->I, qkv_w, value_dim});
    // Elementwise kernels and GEMM grids use signed int element counts.
    if (widest <= 0 || T > INT_MAX / widest) return -3;
    for (int t = 0; t < T; ++t)
        if (tokens[t] < 0 || tokens[t] >= m->V) return -4;
    if (g_wave.capture || !pf_layers_batchable() ||
        dim != 128 || m->linear_key_head_dim != dim ||
        m->linear_num_key_heads <= 0 || heads <= 0 ||
        heads % m->linear_num_key_heads || heads > 64 || heads > kv_w ||
        conv_dim > qkv_w || kv_w > value_dim) return -5;
    for (int layer = 0; layer < m->L; ++layer)
        if (m->layer_types[layer] != TQ_LAYER_FULL_ATTENTION &&
            m->layer_types[layer] != TQ_LAYER_LINEAR_ATTENTION) return -5;

    try {
        // Complete prior work before growing shared scratch or recycling pages.
        tq_q().wait_and_throw();
        if (pf_ensure(T) != 0) return -6;
        const int prepared = tq_pool_prepare_packed(slots, positions, counts, n);
        if (prepared != 0) return prepared;
        if (g_pg_enabled) {
            for (int i = 0; i < n; ++i)
                if (tq_pool_sync_slot(slots[i]) != 0) {
                    tq_q().wait_and_throw();
                    return -7;
                }
        }
        // Do not call reset_slot here: it would discard the freshly prepared
        // maps, and reset-all would destroy independent APC checkpoints.
        for (int i = 0; i < n; ++i) {
            if (positions[i] != 0) continue;
            for (int layer = 0; layer < m->L; ++layer) {
                if (m->layer_types[layer] != TQ_LAYER_LINEAR_ATTENTION) continue;
                tq_layer_t *l = &m->layers[layer];
                tq_q().memset(slot_conv_at(l, slots[i]), 0,
                              slot_conv_elems() * sizeof(float));
                tq_q().memset(slot_recur_at(l, slots[i]), 0,
                              slot_recur_elems() * sizeof(float));
            }
        }
        const char *w4a4_env = getenv("TQ_XPU_W4A4");
        g_pf.w4a4 = ((!w4a4_env || w4a4_env[0] != '0') &&
                     pf_layers_any_k64()) ? 1 : 0;
        const pf_segments batch{slots, offsets, positions, n};
        const int ret = pf_run_layers(tokens, T, 0, &batch);
        if (ret == 0) {
            for (int i = 0; i < n; ++i)
                if (slots[i] == m->active_slot)
                    tq_d2d_async(m->d_x,
                                 g_pf.x + (size_t)(offsets[i + 1] - 1) * m->H,
                                 (size_t)m->H * sizeof(float));
        }
        tq_q().wait_and_throw();
        if (ret != 0) return -8;
        // Publish all committed positions together, after asynchronous success.
        for (int i = 0; i < n; ++i)
            m->state_pos[slots[i]] = positions[i] + counts[i];
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "packed prefill failed: %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "packed prefill failed: unknown runtime error\n");
    }
    try { tq_q().wait_and_throw(); } catch (...) {}
    return -7;
}

// Prefill n tokens at positions pos0..pos0+n-1. Leaves KV, conv and recurrent
// state exactly as the equivalent qwn_decode sequence would. n must be a
// multiple of 8 (the RC8 token group); callers hand the remainder to
// qwn_decode. Returns 0, or negative on error.
extern "C" int qwn_prefill_chunk(const int *tokens, int n, int pos0) {
    tq_model_t *m = &g_qwen;
    if (!m->initialized || !tokens) return -1;
    if (n <= 0 || (n % 8) != 0) return -2;
    if (pos0 < 0 || pos0 > m->max_seq - n) return -3;
    for (int t = 0; t < n; ++t)
        if (tokens[t] < 0 || tokens[t] >= m->V) return -4;
    if (!pf_layers_batchable()) return -5;
    if (pos0 == 0) {
        // Paged checkpoints outlive individual slots, so even a one-slot
        // candidate uses slot reset rather than destructive reset-all.
        const int r = g_pg_enabled
                          ? qwn_reset_slot(g_qwen.active_slot)
                          : ((g_qwen.slots > 1)
                                 ? qwn_reset_slot(g_qwen.active_slot)
                                 : qwn_reset_state());
        if (r != 0) return r;
    }
    if (pf_ensure(n) != 0) return -6;
    if (g_pg_enabled) {
        const int slot = m->active_slot;
        const int rc = tq_pool_prepare_many(&slot, &pos0, &n, 1);
        if (rc != 0) return rc;
        if (tq_pool_sync_slot(slot) != 0) return -7;
    }
    // W4A4 rides K64 weights by default; TQ_XPU_W4A4=0 forces the W4A8
    // path on the same weights for A/B runs.
    const char *w4a4_env = getenv("TQ_XPU_W4A4");
    g_pf.w4a4 = ((!w4a4_env || w4a4_env[0] != '0') && pf_layers_any_k64()) ? 1 : 0;
    // TQ_XPU_PROFILE_PREFILL=<pos0> profiles the chunk starting at that
    // position, so the phase split can be read at a chosen DEPTH rather than
    // inferred from arithmetic. Stage boundaries here are coarser than the
    // decode ladder's: silu lands in `projection` and the activation-quant
    // stage in `activation`.
    ProfileRun pf_profile;
    const char *pf_prof = getenv("TQ_XPU_PROFILE_PREFILL");
    if (pf_prof && pos0 == atoi(pf_prof)) {
        tq_q().wait_and_throw();
        g_profile_run = &pf_profile;
        g_profile_mark = std::chrono::steady_clock::now();
    }

    const int lret = pf_run_layers(tokens, n, pos0);
    if (lret != 0) return lret;
    if (g_profile_run == &pf_profile) {
        static const char *pf_names[kProfileStageCount] = {
            "embed", "norm", "quant", "projection", "conv", "delta",
            "attention", "activation", "residual", "final_norm", "lm_head",
        };
        double total = 0.0;
        for (int i = 0; i < kProfileStageCount; ++i) total += pf_profile.ms[i];
        fprintf(stderr, "[xpu-prefill-profile] pos0=%d T=%d total=%.3f ms "
                        "(%.3f ms/token)\n", pos0, n, total, total / n);
        for (int i = 0; i < kProfileStageCount; ++i) {
            if (pf_profile.calls[i] == 0) continue;
            fprintf(stderr, "  %-12s %8.3f ms  %5.1f%%  calls=%d\n",
                    pf_names[i], pf_profile.ms[i],
                    100.0 * pf_profile.ms[i] / (total > 0.0 ? total : 1.0),
                    pf_profile.calls[i]);
        }
        g_profile_run = nullptr;
    }
    // Hand the final token's hidden state to the single-token path so the
    // caller's next qwn_decode continues from an identical state.
    tq_d2d_async(m->d_x, g_pf.x + (size_t)(n - 1) * m->H,
                 (size_t)m->H * sizeof(float));
    m->state_pos[m->active_slot] = pos0 + n;
    tq_q().wait();
    return 0;
}

namespace {

// Allocate the wave scratch once (sized for 8 rows on this model).
int wave_ensure(void) {
    if (g_wave.ready) return 0;
    tq_model_t *m = &g_qwen;
    const int heads = m->linear_num_value_heads;
    const int dim = m->linear_value_head_dim;
    const int key_dim = m->linear_num_key_heads * dim;
    const int conv_dim = key_dim * 2 + heads * dim;
    const int ks = m->linear_conv_kernel_dim;
    g_wave.row_w = (size_t)conv_dim;
    auto fa = [&](size_t n, const char *what) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(float))
            return static_cast<float *>(nullptr);
        return static_cast<float *>(tq_try_dev_alloc(n * sizeof(float), what));
    };
    g_wave.logits = fa((size_t)8 * m->V, "wave.logits");
    if (!g_pg_enabled) {
        g_wave.qkv_cache = fa((size_t)m->L * 8 * g_wave.row_w, "wave.qkv");
        g_wave.conv_cache = fa((size_t)m->L * 8 * g_wave.row_w, "wave.conv");
        g_wave.b_cache = fa((size_t)m->L * 8 * heads, "wave.b");
        g_wave.a_cache = fa((size_t)m->L * 8 * heads, "wave.a");
        g_wave.recur_snap =
            fa((size_t)m->L * heads * dim * dim, "wave.recur");
        g_wave.conv_snap =
            fa((size_t)m->L * conv_dim * ks, "wave.convsnap");
    }
    if (!g_wave.logits ||
        (!g_pg_enabled &&
         (!g_wave.qkv_cache || !g_wave.conv_cache || !g_wave.b_cache ||
          !g_wave.a_cache || !g_wave.recur_snap || !g_wave.conv_snap))) {
        wave_free();
        return -1;
    }
    g_wave.ready = 1;
    return 0;
}

}  // namespace

int tq_decode_runtime_warm(int prefill_tokens) {
    if (!g_qwen.initialized || prefill_tokens <= 0 ||
        prefill_tokens > g_qwen.max_seq || (prefill_tokens % 8) != 0)
        return -1;
    if (pf_ensure(prefill_tokens) != 0 || wave_ensure() != 0) {
        pf_free();
        wave_free();
        return -2;
    }
    return 0;
}

void tq_decode_runtime_free(void) {
    if (g_pf.cap || g_wave.ready) {
        try { tq_q().wait_and_throw(); } catch (...) {}
    }
    pf_free();
    wave_free();
}

// Speculative wave (design 9.3, thesis 2): score tokens[0..7] - the last
// committed token plus 7 draft continuations - at positions pos0..pos0+7 in
// ONE 8-row prefill-shaped pass, then greedily verify: draft j+1 is accepted
// while it equals the argmax of row j. Writes the newly committed tokens
// (accepted drafts + the model's own next token) to out_tokens and returns
// their count (1..8), or negative on error. All state (KV, conv, recurrent,
// state_pos, d_x) is left as the equivalent accepted-only decode sequence
// would leave it: GDN state rewinds via snapshot + re-advance from cached
// wave inputs; KV rows past the accepted prefix are dead by positional
// masking. Requires the chunked GDN path (TQ_XPU_DN_CHUNK not 0).
extern "C" int qwn_spec_wave(const int *tokens, int pos0, int *out_tokens) {
    tq_model_t *m = &g_qwen;
    constexpr int W = 8;
    if (!m->initialized || !tokens || !out_tokens) return -1;
    if (g_pg_enabled) return -10;
    if (pos0 <= 0 || pos0 + W > m->max_seq) return -2;
    for (int t = 0; t < W; ++t)
        if (tokens[t] < 0 || tokens[t] >= m->V) return -3;
    if (!pf_layers_batchable()) return -4;
    const char *dnc = getenv("TQ_XPU_DN_CHUNK");
    if (dnc && dnc[0] == '0') return -5;   // wave rewind needs the chunk path
    if (m->linear_value_head_dim != 128) return -5;
    if (pf_ensure(W) != 0) return -6;
    if (wave_ensure() != 0) return -6;
    const char *w4a4_env = getenv("TQ_XPU_W4A4");
    g_pf.w4a4 = ((!w4a4_env || w4a4_env[0] != '0') && pf_layers_any_k64()) ? 1 : 0;

    const int heads = m->linear_num_value_heads;
    const int dim = m->linear_value_head_dim;
    const int key_dim = m->linear_num_key_heads * dim;
    const int conv_dim = key_dim * 2 + heads * dim;
    const int ks = m->linear_conv_kernel_dim;
    const size_t recur_f = (size_t)heads * dim * dim;
    const size_t convst_f = (size_t)conv_dim * ks;

    // Snapshot GDN state, then run the wave with input capture on.
    for (int layer = 0; layer < m->L; ++layer) {
        if (m->layer_types[layer] != TQ_LAYER_LINEAR_ATTENTION) continue;
        tq_layer_t *l = &m->layers[layer];
        tq_d2d_async(g_wave.recur_snap + (size_t)layer * recur_f,
                     slot_recur(l), recur_f * sizeof(float));
        tq_d2d_async(g_wave.conv_snap + (size_t)layer * convst_f,
                     slot_conv(l), convst_f * sizeof(float));
    }
    g_wave.capture = 1;
    const int lret = pf_run_layers(tokens, W, pos0);
    g_wave.capture = 0;
    if (lret != 0) return lret;

    // Batched lm_head over all 8 rows (W8A8 GEMM when repacked; GEMV loop
    // otherwise), then per-row argmax.
    x_rmsnorm_chunk(g_pf.normed, g_pf.x, m->d_norm, m->H, W, m->eps);
    int am[W];
    if (m->lm_head.s8_ready) {
        x_quantize_act_chunk(g_pf.normed, m->H, W, g_pf.aq, g_pf.as, g_pf.asum);
        if (x_gemm_w8a8(&m->lm_head, g_pf.aq, g_pf.as, g_wave.logits, W) != 0)
            return -7;
    } else {
        for (int j = 0; j < W; ++j)
            if (x_gemv_qmma(&m->lm_head, g_pf.normed + (size_t)j * m->H,
                            g_wave.logits + (size_t)j * m->V) != 0)
                return -7;
    }
    for (int j = 0; j < W; ++j) {
        x_argmax(g_wave.logits + (size_t)j * m->V, m->V, m->d_argmax_vals,
                 m->d_argmax_ids, &m->last_argmax_id, &m->last_argmax_logit);
        am[j] = m->last_argmax_id;
    }

    // Verify: accept drafts while they match; the model's own token follows.
    int A = 0;
    while (A < W - 1 && am[A] == tokens[A + 1]) ++A;
    for (int j = 0; j < A; ++j) out_tokens[j] = tokens[j + 1];
    out_tokens[A] = am[A];

    // Rewind GDN state to the accepted prefix (A+1 rows processed).
    if (A < W - 1) {
        for (int layer = 0; layer < m->L; ++layer) {
            if (m->layer_types[layer] != TQ_LAYER_LINEAR_ATTENTION) continue;
            tq_layer_t *l = &m->layers[layer];
            tq_d2d_async(slot_recur(l),
                         g_wave.recur_snap + (size_t)layer * recur_f,
                         recur_f * sizeof(float));
            tq_d2d_async(slot_conv(l),
                         g_wave.conv_snap + (size_t)layer * convst_f,
                         convst_f * sizeof(float));
            x_linear_conv_advance(slot_conv(l),
                                  g_wave.qkv_cache + (size_t)layer * 8 * g_wave.row_w,
                                  conv_dim, ks, A + 1);
            // Outputs are discarded (accepted rows already produced their
            // outputs in the wave); only the state math matters, and it does
            // not read the gate, so g_pf.z can hold anything.
            if (x_deltanet_chunk(g_pf.core, slot_recur(l),
                                 g_wave.conv_cache + (size_t)layer * 8 * g_wave.row_w,
                                 g_pf.z,
                                 g_wave.b_cache + (size_t)layer * 8 * heads,
                                 g_wave.a_cache + (size_t)layer * 8 * heads,
                                 l->d_linear_A_log, l->d_linear_dt_bias,
                                 l->d_linear_norm, A + 1,
                                 m->linear_num_key_heads, dim, heads, dim,
                                 m->eps) != 0)
                return -8;
        }
    }

    // Handoff: the accepted row's hidden feeds the next step; position moves
    // by the committed count.
    tq_d2d_async(m->d_x, g_pf.x + (size_t)A * m->H, (size_t)m->H * sizeof(float));
    m->state_pos[m->active_slot] = pos0 + A + 1;
    tq_q().wait();
    return A + 1;
}

// ---- Batched decode step (continuous batching) ----------------------------
// One step for n INDEPENDENT sequences: row i advances slot slots[i] from
// position positions[i] with committed token tokens[i]; the greedy next
// token lands in out_tokens[i]. Projections, MLP and lm_head ride the
// batched RC8 GEMM path - one weight stream serves every row, which is the
// whole point: decode is weight-stream-bound, so rows are nearly free.
// Attention and the GDN core run per-row against that slot's state (small
// kernels; the in-order queue serializes the shared d_scores scratch).
// Rows are independent: any subset of slots, any mix of positions. Requires
// the batchable W4 tier (same guard as prefill/wave).
// Model-free gate for the block pool. Needs no weights, so the substrate is
// provable in milliseconds BEFORE any kernel or loader work depends on it -
// the same discipline that made the TP substrate land clean (lesson 40).
// Returns 0 on pass, or the negative index of the first failing check.
extern "C" int qwn_pool_selftest(void) {
    if (g_pg_ready || g_qwen.initialized) return -32;
    tq_pool_gate_guard cleanup;
    const int slots = 4, blocks = 8, page = 128, max_seq = 1024;
    if (tq_pool_init(slots, blocks, page, max_seq) != 0) return -1;
    if (g_pg_maxblk != max_seq / page) return -2;
    if (g_pg_nfree != blocks) return -3;

    // grow slot 0 to position 300 -> ceil(301/128) = 3 blocks
    if (tq_pool_ensure(0, 300) != 0) return -4;
    if (h_slot_nb[0] != 3) return -5;
    if (g_pg_nfree != blocks - 3) return -6;
    // re-ensuring a position already covered must map nothing new
    if (tq_pool_ensure(0, 300) != 0) return -7;
    if (g_pg_nfree != blocks - 3) return -8;

    // exhaust: slot 1 wants 6 more blocks, only 5 remain -> admission failure
    if (tq_pool_ensure(1, 5 * page) != -2) return -9;

    // an exhausted ensure is allowed to have consumed what it could; the
    // scheduler's contract is that the slot is then released, not reused.
    tq_pool_release_slot(1);
    if (g_pg_nfree != blocks - 3) return -10;

    // refcount sharing: retain slot 0's first 2 blocks (positions 0..255),
    // then release the slot. The retained blocks must NOT return to the pool.
    if (tq_pool_retain_prefix(0, 255) != 2) return -11;
    tq_pool_release_slot(0);
    if (g_pg_nfree != blocks - 2) return -12;      // 2 still held by the ckpt
    if (h_slot_nb[0] != 0) return -13;

    // dropping the checkpoint's refs returns them
    for (int b = 0; b < blocks; ++b) {
        while (h_blk_active_ref[b] > 0) tq_pool_block_unref(b, false);
        while (h_blk_ckpt_ref[b] > 0) tq_pool_block_unref(b, true);
    }
    if (g_pg_nfree != blocks) return -14;

    // page must be a power of two
    if (tq_pool_init(slots, blocks, 100, max_seq) != -3) return -15;

    // addressing identity: for every position, phys*page + (t & (page-1)) must
    // land on a distinct row, and logical->physical must round-trip.
    if (tq_pool_init(slots, blocks, page, max_seq) != 0) return -16;
    if (tq_pool_ensure(2, 511) != 0) return -17;
    const int *row = h_block_table + (size_t)2 * g_pg_maxblk;
    for (int t = 0; t <= 511; ++t) {
        const int lb = t >> g_pg_plog;
        if (row[lb] < 0 || row[lb] >= blocks) return -18;
        const size_t pr = (size_t)row[lb] * page + (t & (page - 1));
        if (pr >= (size_t)blocks * page) return -19;
    }
    tq_pool_sync();
    tq_pool_release_slot(2);
    if (g_pg_nfree != blocks) return -20;

    // ---- content index (lesson 50b/50c) ----------------------------------
    // 21-23: publish / lookup / unpublish round-trip.
    const int pb = tq_pool_alloc_block();
    if (pb < 0) return -21;
    if (tq_pool_publish(pb, 0xABCDEF01u) != 0) return -22;
    if (tq_pool_lookup(0xABCDEF01u) != pb) return -23;
    if (tq_pool_lookup(0x12345678u) != -1) return -24;   // unknown hash misses
    tq_pool_unpublish(pb);
    if (tq_pool_lookup(0xABCDEF01u) != -1) return -25;

    // 26: THE correctness link. Publish, release the block back to the ring,
    // then drain the ring until that block is handed out again. The index must
    // no longer advertise it - otherwise a lookup returns a block holding a
    // different sequence's tokens, which is a silent wrong answer.
    if (tq_pool_publish(pb, 0xFEEDBEEFu) != 0) return -26;
    tq_pool_block_unref(pb, false);                      // back to the ring tail
    int drained = -1;
    for (int i = 0; i < blocks && drained != pb; ++i) drained = tq_pool_alloc_block();
    if (drained != pb) return -27;                       // ring never returned it
    if (tq_pool_lookup(0xFEEDBEEFu) != -1) return -28;   // stale hash survived!

    // 29: hash 0 is the empty sentinel and must be folded, not stored as empty.
    if (tq_pool_publish(drained, 0) != 0) return -29;
    if (tq_pool_lookup(1) != drained) return -30;

    // 31: full free list again after releasing everything we took.
    for (int b = 0; b < blocks; ++b) {
        while (h_blk_active_ref[b] > 0) tq_pool_block_unref(b, false);
        while (h_blk_ckpt_ref[b] > 0) tq_pool_block_unref(b, true);
    }
    if (g_pg_nfree != blocks) return -31;
    tq_pool_free_all();
    return 0;
}

// Device-side block-table ADDRESSING gate. This is the first half of the KV
// write port, made independently checkable: a real SYCL kernel scatters rows
// into the pool using exactly the addressing the attention kernels will use -
//     phys = block_table[slot*max_blocks + (t >> page_log)]
//     row  = phys * page + (t & (page - 1))
// - and the host then verifies each row landed where ITS OWN offset arithmetic
// says it should, reading h_block_table directly. Device math is checked
// against host math with real bytes, so a wrong shift or mask cannot pass.
//
// Rows carry a per-(token,channel) pattern rather than real KV: this gate is
// about WHERE bytes go, not what they are. Writing to the pool cannot disturb
// the flat path, which remains authoritative until the attention read is
// ported too (they are coupled - pool writes with flat reads diverge by
// construction, which is why this half is gated on its own).
extern "C" int qwn_pool_scatter_gate(int npos) {
    if (!g_qwen.initialized) return -1;
    if (g_pg_ready) return -6;
    tq_pool_gate_guard cleanup;
    if (npos < 1) return -2;
    const int page = 128;
    const int nkv = g_qwen.nkv, hd = g_qwen.hd;
    const size_t rowb = (size_t)nkv * hd;                  // bytes per token row
    const int nblocks = (npos + page - 1) / page + 2;      // + slack
    if (tq_pool_init(1, nblocks, page, g_qwen.max_seq) != 0) return -3;
    if (tq_pool_ensure(0, npos - 1) != 0) { tq_pool_free_all(); return -4; }
    tq_pool_sync();

    // pick the first full-attention layer; its slab is the scatter target
    int L = -1;
    for (int i = 0; i < g_qwen.L; ++i)
        if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) { L = i; break; }
    if (L < 0 || !g_pool_k[L]) { tq_pool_free_all(); return -5; }

    uint8_t *d_src = (uint8_t *)tq_dev_alloc((size_t)npos * rowb, "gate.src");
    uint8_t *dst = g_pool_k[L];
    const int *bt = d_block_table;
    const int plog = g_pg_plog, mask = page - 1, maxblk = g_pg_maxblk;
    // pattern then scatter, both on device
    tq_q().parallel_for(sycl::range<1>((size_t)npos * rowb), [=](sycl::id<1> id) {
        const size_t i = id[0];
        d_src[i] = (uint8_t)((i * 2654435761u) >> 13);
    });
    tq_q().parallel_for(sycl::range<1>((size_t)npos * rowb), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const int t = (int)(i / rowb);
        const size_t c = i - (size_t)t * rowb;
        const int phys = bt[(size_t)0 * maxblk + (t >> plog)];
        const size_t row = (size_t)phys * page + (size_t)(t & mask);
        dst[row * rowb + c] = d_src[i];
    });
    tq_q().wait_and_throw();

    // host-side verification with independently computed offsets
    std::vector<uint8_t> got(rowb), want(rowb);
    int bad = -1;
    for (int t = 0; t < npos && bad < 0; ++t) {
        const int phys = h_block_table[(size_t)0 * g_pg_maxblk + (t / page)];
        if (phys < 0 || phys >= nblocks) { bad = t; break; }
        const size_t row = (size_t)phys * page + (size_t)(t % page);
        tq_d2h(got.data(), dst + row * rowb, rowb);
        for (size_t c = 0; c < rowb; ++c) {
            const size_t i = (size_t)t * rowb + c;
            want[c] = (uint8_t)((i * 2654435761u) >> 13);
        }
        if (std::memcmp(got.data(), want.data(), rowb) != 0) bad = t;
    }
    tq_dev_free(d_src);
    tq_pool_free_all();
    return bad < 0 ? 0 : -(100 + bad);      // -(100+t) names the first bad token
}

// Does the block-table indirection cost anything? This is THE question that
// decides whether paging is worth porting, and it is measurable before any
// attention kernel changes: run the same scatter twice over the same bytes,
// once with contiguous addressing and once through the block table.
//
// `page` brackets the answer. At page=128 one lookup covers 128 rows (the CUDA
// TU's amortization: "a 128-aligned super-tile lies in ONE physical block, so
// one block-table lookup gives a contiguous physical-row base for the 128
// keys"). At page=1 every row pays its own lookup - the pathological floor.
// out[0] = flat ms, out[1] = paged ms.
extern "C" int qwn_pool_scatter_bench(int npos, int page, int iters,
                                      double *out) {
    if (!g_qwen.initialized || npos < 1 || iters < 1) return -1;
    if (g_pg_ready) return -6;
    tq_pool_gate_guard cleanup;
    if (page < 1 || (page & (page - 1))) return -2;
    const size_t rowb = (size_t)g_qwen.nkv * g_qwen.hd;
    const int nblocks = (npos + page - 1) / page + 2;
    if (tq_pool_init(1, nblocks, page, g_qwen.max_seq) != 0) return -3;
    if (tq_pool_ensure(0, npos - 1) != 0) { tq_pool_free_all(); return -4; }
    tq_pool_sync();
    int L = -1;
    for (int i = 0; i < g_qwen.L; ++i)
        if (g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION) { L = i; break; }
    if (L < 0 || !g_pool_k[L]) { tq_pool_free_all(); return -5; }

    const size_t n = (size_t)npos * rowb;
    uint8_t *d_src = (uint8_t *)tq_dev_alloc(n, "bench.src");
    uint8_t *dst = g_pool_k[L];
    const int *bt = d_block_table;
    const int plog = g_pg_plog, mask = page - 1, maxblk = g_pg_maxblk;
    tq_q().parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        d_src[id[0]] = (uint8_t)id[0];
    });
    tq_q().wait_and_throw();

    // FAIR baseline: the flat arm must do the SAME index decomposition, so the
    // only difference between the arms is the block-table lookup itself. A bare
    // dst[i]=src[i] baseline measures the paged arm's integer division instead
    // (that mistake reported +60% independent of page size - the tell that it
    // was not lookup cost at all).
    auto flat = [&] {
        tq_q().parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const int t = (int)(i / rowb);
            const size_t c = i - (size_t)t * rowb;
            dst[(size_t)t * rowb + c] = d_src[i];
        });
    };
    auto paged = [&] {
        tq_q().parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const int t = (int)(i / rowb);
            const size_t c = i - (size_t)t * rowb;
            const int phys = bt[(size_t)0 * maxblk + (t >> plog)];
            const size_t row = (size_t)phys * page + (size_t)(t & mask);
            dst[row * rowb + c] = d_src[i];
        });
    };
    // warm both, then time
    flat(); paged(); tq_q().wait_and_throw();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) flat();
    tq_q().wait_and_throw();
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) paged();
    tq_q().wait_and_throw();
    const auto t2 = std::chrono::steady_clock::now();
    if (out) {
        out[0] = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        out[1] = std::chrono::duration<double, std::milli>(t2 - t1).count() / iters;
    }
    tq_dev_free(d_src);
    tq_pool_free_all();
    return 0;
}

// With-model gate for the pooled SLABS. qwn_pool_selftest runs model-free, so
// it skips slab allocation entirely (g_qwen.initialized is false) - that path
// compiles but never executes there. This one needs weights loaded: it sizes a
// pool from a byte budget, checks the slabs landed on exactly the
// full-attention layers, verifies the byte math against
// tq_pool_bytes_per_block, and frees. Returns 0, or -check on failure.
// out[0..3] = bytes_per_block, num_blocks, pooled_tokens, flat_tokens.
extern "C" int qwn_pool_gate_model(int budget_mb, int page, long long *out) {
    if (!g_qwen.initialized) return -1;
    if (g_pg_ready) return -13;
    tq_pool_gate_guard cleanup;
    if (page < 1 || (page & (page - 1))) return -2;
    const size_t budget = (size_t)budget_mb * 1024u * 1024u;
    const size_t per_block = tq_pool_bytes_per_block(page);
    if (per_block == 0) return -3;
    const int nblocks = tq_pool_blocks_for_budget(budget, page);
    if (nblocks < 1) return -4;
    if ((size_t)nblocks * per_block > budget) return -5;     // must not overshoot

    // per-token cost must match the per-slot flat cost: same bytes, and the
    // whole point is that the pool does not partition them per slot.
    const size_t per_token = per_block / (size_t)page;
    if (per_token == 0) return -6;

    if (tq_pool_init(g_qwen.slots < 1 ? 1 : g_qwen.slots, nblocks, page,
                     g_qwen.max_seq) != 0) return -7;

    int nfull = 0, ngdn = 0;
    for (int i = 0; i < g_qwen.L; ++i) {
        const bool full = g_qwen.layer_types[i] == TQ_LAYER_FULL_ATTENTION;
        const bool got = g_pool_k[i] && g_pool_v[i] && g_pool_ks[i] && g_pool_vs[i];
        if (full) {
            if (!got) { tq_pool_free_all(); return -8; }   // attention layer missing slabs
            ++nfull;
        } else {
            if (got) { tq_pool_free_all(); return -9; }    // GDN layer must NOT be pooled
            ++ngdn;
        }
    }
    if (nfull == 0 || ngdn == 0) { tq_pool_free_all(); return -10; }

    // the pool must be usable end to end: map a long sequence, release it, and
    // land back on a full free list.
    const int deep = nblocks * page - 1;
    const int want = deep < g_qwen.max_seq - 1 ? deep : g_qwen.max_seq - 1;
    if (tq_pool_ensure(0, want) != 0) { tq_pool_free_all(); return -11; }
    tq_pool_release_slot(0);
    if (g_pg_nfree != nblocks) { tq_pool_free_all(); return -12; }

    if (out) {
        out[0] = (long long)per_block;
        out[1] = (long long)nblocks;
        out[2] = (long long)nblocks * page;                       // pooled tokens
        out[3] = (long long)(g_qwen.slots < 1 ? 1 : g_qwen.slots) *
                 (long long)g_qwen.max_seq;                       // flat tokens
    }
    tq_pool_free_all();
    return 0;
}

extern "C" int qwn_decode_batch(const int *slots, const int *tokens,
                                const int *positions, int n,
                                int *out_tokens) {
    tq_model_t *m = &g_qwen;
    if (!m->initialized || !slots || !tokens || !positions || !out_tokens)
        return -1;
    const int S = m->slots < 1 ? 1 : m->slots;
    if (n < 1 || n > S || n > 8) return -2;
    for (int i = 0; i < n; ++i) {
        if (slots[i] < 0 || slots[i] >= S) return -2;
        if (tokens[i] < 0 || tokens[i] >= m->V) return -3;
        if (positions[i] < 1 || positions[i] >= m->max_seq) return -3;
        for (int j = 0; j < i; ++j)
            if (slots[j] == slots[i]) return -2;
    }
    if (!pf_layers_batchable()) return -4;
    if (pf_ensure(8) != 0) return -6;
    if (wave_ensure() != 0) return -6;   // borrows the 8-row logits slab
    if (g_pg_enabled) {
        int counts[8];
        for (int i = 0; i < n; ++i) counts[i] = 1;
        const int rc = tq_pool_prepare_many(slots, positions, counts, n);
        if (rc != 0) return rc;
        for (int i = 0; i < n; ++i)
            if (tq_pool_sync_slot(slots[i]) != 0) return -10;
    }
    const char *w4a4_env = getenv("TQ_XPU_W4A4");
    g_pf.w4a4 = ((!w4a4_env || w4a4_env[0] != '0') && pf_layers_any_k64()) ? 1 : 0;

    const int heads = m->linear_num_value_heads;
    const int key_heads = m->linear_num_key_heads;
    const int dim = m->linear_value_head_dim;
    const int value_dim = heads * dim;
    const int conv_dim = key_heads * dim * 2 + value_dim;
    const int qkv_w = 2 * m->nh * m->hd;
    const int kv_w = m->nkv * m->hd;
    const int core_w = m->nh * m->hd;

    // The batched GEMMs are RC8: one 256-byte weight fragment serves EIGHT
    // activation rows, so the step's natural width is 8 and T must be a
    // multiple of 8. n < 8 real rows ride the same weight stream; the pad
    // rows cost epilogue ALU only, never weight bytes. This is why
    // concurrency up to 8 is nearly free on a weight-stream-bound decode.
    constexpr int Tp = 8;
    for (int i = 0; i < n; ++i)
        x_embed_lookup(g_pf.x + (size_t)i * m->H, m->d_embed, tokens[i], m->H);
    if (n < Tp)   // deterministic pad rows (zero in, finite through)
        tq_q().memset(g_pf.x + (size_t)n * m->H, 0,
                      (size_t)(Tp - n) * m->H * sizeof(float));
    for (int layer = 0; layer < m->L; ++layer) {
        tq_layer_t *l = &m->layers[layer];
        x_rmsnorm_chunk(g_pf.normed, g_pf.x, l->d_input_ln, m->H, Tp, m->eps);
        if (m->layer_types[layer] == TQ_LAYER_FULL_ATTENTION) {
            pf_stage(g_pf.normed, m->H, Tp, &l->q_proj, &l->k_proj, &l->v_proj);
            if (pf_gemm(&l->q_proj, g_pf.qkv, Tp) != 0) return -7;
            if (pf_gemm(&l->k_proj, g_pf.b, Tp) != 0) return -7;
            if (pf_gemm(&l->v_proj, g_pf.z, Tp) != 0) return -7;
            for (int i = 0; i < n; ++i)
                x_full_attn_decode(g_pf.core + (size_t)i * core_w,
                                   g_pf.qkv + (size_t)i * qkv_w,
                                   g_pf.b + (size_t)i * kv_w,
                                   g_pf.z + (size_t)i * kv_w,
                                   l->d_q_norm, l->d_k_norm,
                                   kv_k_at(l, slots[i]),
                                   kv_v_at(l, slots[i]),
                                   kv_ks_at(l, slots[i]),
                                   kv_vs_at(l, slots[i]),
                                   m->d_scores, positions[i], m->nh, m->nkv,
                                   m->hd, m->eps, m->rope_theta,
                                   m->partial_rotary_factor, m->max_seq,
                                   tq_pool_layout(slots[i]));
            if (n < Tp)
                tq_q().memset(g_pf.core + (size_t)n * core_w, 0,
                              (size_t)(Tp - n) * core_w * sizeof(float));
            pf_stage(g_pf.core, core_w, Tp, &l->o_proj);
            if (pf_gemm(&l->o_proj, g_pf.resid, Tp) != 0) return -7;
        } else if (m->layer_types[layer] == TQ_LAYER_LINEAR_ATTENTION) {
            pf_stage(g_pf.normed, m->H, Tp, &l->linear_in_qkv, &l->linear_in_z,
                      &l->linear_in_b, &l->linear_in_a);
            if (pf_gemm(&l->linear_in_qkv, g_pf.qkv, Tp) != 0) return -7;
            if (pf_gemm(&l->linear_in_z, g_pf.z, Tp) != 0) return -7;
            if (pf_gemm(&l->linear_in_b, g_pf.b, Tp) != 0) return -7;
            if (pf_gemm(&l->linear_in_a, g_pf.a, Tp) != 0) return -7;
            for (int i = 0; i < n; ++i) {
                float *qkv_i = g_pf.qkv + (size_t)i * conv_dim;
                x_linear_conv_update(qkv_i, slot_conv_at(l, slots[i]), qkv_i,
                                     l->d_linear_conv1d, conv_dim,
                                     m->linear_conv_kernel_dim);
                x_linear_decode_core_gated(g_pf.core + (size_t)i * value_dim,
                                           slot_recur_at(l, slots[i]), qkv_i,
                                           g_pf.z + (size_t)i * value_dim,
                                           g_pf.b + (size_t)i * heads,
                                           g_pf.a + (size_t)i * heads,
                                           l->d_linear_A_log,
                                           l->d_linear_dt_bias,
                                           l->d_linear_norm, key_heads, dim,
                                           heads, dim, m->eps);
            }
            if (n < Tp)
                tq_q().memset(g_pf.core + (size_t)n * value_dim, 0,
                              (size_t)(Tp - n) * value_dim * sizeof(float));
            pf_stage(g_pf.core, value_dim, Tp, &l->linear_out);
            if (pf_gemm(&l->linear_out, g_pf.resid, Tp) != 0) return -7;
        } else {
            return -7;
        }
        x_add_inplace(g_pf.resid, g_pf.x, Tp * m->H);
        if (pf_mlp(l, Tp) != 0) return -8;
        float *old = g_pf.x;
        g_pf.x = g_pf.layer_out;
        g_pf.layer_out = old;
    }
    x_rmsnorm_chunk(g_pf.normed, g_pf.x, m->d_norm, m->H, Tp, m->eps);
    if (m->lm_head.s8_ready) {
        x_quantize_act_chunk(g_pf.normed, m->H, Tp, g_pf.aq, g_pf.as, g_pf.asum);
        if (x_gemm_w8a8(&m->lm_head, g_pf.aq, g_pf.as, g_wave.logits, Tp) != 0)
            return -9;
    } else {
        for (int i = 0; i < n; ++i)
            if (x_gemv_qmma(&m->lm_head, g_pf.normed + (size_t)i * m->H,
                            g_wave.logits + (size_t)i * m->V) != 0)
                return -9;
    }
    for (int i = 0; i < n; ++i) {
        x_argmax(g_wave.logits + (size_t)i * m->V, m->V, m->d_argmax_vals,
                 m->d_argmax_ids, &m->last_argmax_id, &m->last_argmax_logit);
        out_tokens[i] = m->last_argmax_id;
        m->state_pos[slots[i]] = positions[i] + 1;
    }
    tq_q().wait();
    return n;
}

// ---- Prefix checkpoints (automatic prefix caching) ------------------------
// A checkpoint is the COMPLETE per-sequence state at committed position
// `pos`: the KV prefix (rows [0,pos) of every attention layer - token-major,
// so a prefix is contiguous) plus the conv and recurrent state of every GDN
// layer. Restoring one into a slot leaves that slot byte-identical to having
// prefilled those `pos` tokens, which is what makes prompt-prefix reuse
// legal here at all: 48 of 64 layers are recurrent, and recurrent state
// cannot be re-derived from a partial KV cache the way a pure-attention
// model's can. That is also why the granularity is a chunk boundary rather
// than vLLM's 16-token block - state exists only where it was snapshotted.
// Storage belongs to the caller; qwn_host_alloc keeps it on the measured
// ~6.8 GB/s host-USM path (probe: 4k-token restore 21.7 ms vs 8.9 s of
// re-prefill).
namespace {

// Walk the checkpoint layout. mode 0 = size only, 1 = save, 2 = restore.
// Copies are queued async and drained once by the caller.
size_t ckpt_walk(int slot, int pos, void *host, int mode) {
    tq_model_t *m = &g_qwen;
    const size_t kv_prefix = (size_t)pos * m->nkv * m->hd;
    const size_t sc_prefix = (size_t)pos * m->nkv * sizeof(uint16_t);
    const size_t conv_b = slot_conv_elems() * sizeof(float);
    const size_t recur_b = slot_recur_elems() * sizeof(float);
    uint8_t *h = static_cast<uint8_t *>(host);
    size_t off = 0;
    auto part = [&](void *dev, size_t bytes) {
        if (mode == 1) tq_q().memcpy(h + off, dev, bytes);
        else if (mode == 2) tq_q().memcpy(dev, h + off, bytes);
        off += bytes;
    };
    for (int i = 0; i < m->L; ++i) {
        tq_layer_t *l = &m->layers[i];
        if (m->layer_types[i] == TQ_LAYER_FULL_ATTENTION) {
            if (mode == 0) { off += 2 * kv_prefix + 2 * sc_prefix; continue; }
            part(slot_kc_at(l, slot), kv_prefix);
            part(slot_vc_at(l, slot), kv_prefix);
            part(slot_ks_at(l, slot), sc_prefix);
            part(slot_vs_at(l, slot), sc_prefix);
        } else if (m->layer_types[i] == TQ_LAYER_LINEAR_ATTENTION) {
            if (mode == 0) { off += conv_b + recur_b; continue; }
            part(slot_conv_at(l, slot), conv_b);
            part(slot_recur_at(l, slot), recur_b);
        } else {
            return 0;
        }
    }
    return off;
}

}  // namespace

// Host-USM scratch for checkpoint blobs (falls back to plain host memory,
// which still works but copies slower).
extern "C" void *qwn_host_alloc(size_t bytes) {
    if (!g_qwen.initialized || bytes == 0) return nullptr;
    void *p = sycl::malloc_host(bytes, tq_q());
    return p ? p : std::malloc(bytes);
}

extern "C" void qwn_host_free(void *p) {
    if (!p) return;
    if (sycl::get_pointer_type(p, tq_q().get_context()) ==
        sycl::usm::alloc::host)
        sycl::free(p, tq_q());
    else
        std::free(p);
}

// Bytes a checkpoint at committed position `pos` occupies.
extern "C" size_t qwn_ckpt_bytes(int pos) {
    if (g_pg_enabled || !g_qwen.initialized ||
        pos < 0 || pos > g_qwen.max_seq) return 0;
    return ckpt_walk(0, pos, nullptr, 0);
}

extern "C" int qwn_ckpt_save(int slot, int pos, void *dst) {
    tq_model_t *m = &g_qwen;
    if (!m->initialized || !dst) return -1;
    if (g_pg_enabled) return -6;
    const int S = m->slots < 1 ? 1 : m->slots;
    if (slot < 0 || slot >= S) return -2;
    if (pos < 0 || pos > m->max_seq) return -3;
    try {
        if (ckpt_walk(slot, pos, dst, 1) == 0) return -4;
        tq_q().wait_and_throw();
    } catch (const std::exception &e) {
        fprintf(stderr, "ckpt save failed: %s\n", e.what());
        return -5;
    }
    return 0;
}

// Restore a checkpoint into `slot` and set its committed position. KV rows
// at or past `pos` are left alone: they are dead by positional masking and
// get overwritten as the sequence advances.
extern "C" int qwn_ckpt_restore(int slot, int pos, const void *src) {
    tq_model_t *m = &g_qwen;
    if (!m->initialized || !src) return -1;
    if (g_pg_enabled) return -6;
    const int S = m->slots < 1 ? 1 : m->slots;
    if (slot < 0 || slot >= S) return -2;
    if (pos < 0 || pos > m->max_seq) return -3;
    // State must already be allocated; a cold engine has nothing to write to.
    for (int i = 0; i < m->L; ++i) {
        if (m->layer_types[i] == TQ_LAYER_LINEAR_ATTENTION &&
            !m->layers[i].d_linear_conv_state) {
            const int r = qwn_reset_state();
            if (r != 0) return r;
            break;
        }
        if (m->layer_types[i] == TQ_LAYER_FULL_ATTENTION &&
            !m->layers[i].d_k_cache) {
            const int r = qwn_reset_state();
            if (r != 0) return r;
            break;
        }
    }
    try {
        if (ckpt_walk(slot, pos, const_cast<void *>(src), 2) == 0) return -4;
        tq_q().wait_and_throw();
    } catch (const std::exception &e) {
        fprintf(stderr, "ckpt restore failed: %s\n", e.what());
        return -5;
    }
    m->state_pos[slot] = pos;
    return 0;
}

// CUDA: run_decode_layers_to_debug_x (16754)
static int run_decode_layers(int token_id, int pos, int n_layers) {
    tq_model_t *m = &g_qwen;
    if (!m->initialized) return -1;
    if (pos < 0 || pos >= m->max_seq) {
        fprintf(stderr, "decode layers: invalid pos=%d\n", pos);
        return -1;
    }
    if (n_layers < 0 || n_layers > m->L) return -8;
    if (token_id < 0 || token_id >= m->V) return -2;
    if (g_pg_enabled && n_layers != m->L) return -9;
    if (pos == 0) {
        const int r = g_pg_enabled
                          ? qwn_reset_slot(g_qwen.active_slot)
                          : ((g_qwen.slots > 1)
                                 ? qwn_reset_slot(g_qwen.active_slot)
                                 : qwn_reset_state());
        if (r != 0) return r;
    }
    if (g_pg_enabled) {
        const int slot = m->active_slot;
        const int count = 1;
        const int rc = tq_pool_prepare_many(&slot, &pos, &count, 1);
        if (rc != 0) return rc;
        if (tq_pool_sync_slot(slot) != 0) return -10;
    }
    x_embed_lookup(m->d_x, m->d_embed, token_id, m->H);
    profile_mark(kProfileEmbed);
    for (int layer = 0; layer < n_layers; layer++) {
        int ret;
        if (m->layer_types[layer] == TQ_LAYER_LINEAR_ATTENTION) {
            ret = run_linear_layer_decode(layer);
        } else if (m->layer_types[layer] == TQ_LAYER_FULL_ATTENTION) {
            ret = run_full_layer_decode(layer, pos);
        } else {
            return -2;
        }
        if (ret != 0) {
            fprintf(stderr, "decode layers: layer %d failed ret=%d\n", layer, ret);
            return -3;
        }
        float *old_x = m->d_x;
        m->d_x = m->d_layer_out;
        m->d_layer_out = old_x;
    }
    return 0;
}

static int run_final_norm(void) {
    x_rmsnorm(g_qwen.d_norm_out, g_qwen.d_x, g_qwen.d_norm, g_qwen.H, g_qwen.eps);
    return 0;
}

// CUDA: run_lm_head_argmax_device (16638), plain (non-fused, topk=0) branch.
static int run_lm_head_argmax(void) {
    tq_model_t *m = &g_qwen;
    int ret = x_gemv_qmma(&m->lm_head, m->d_norm_out, m->d_logits);
    if (ret != 0) return -6;
    x_argmax(m->d_logits, m->V, m->d_argmax_vals, m->d_argmax_ids,
             &m->last_argmax_id, &m->last_argmax_logit);
    return 0;
}

// ------------------------------------------------------------------- ABI

extern "C" int qwn_decode(int token_id, int pos) {
    ProfileRun profile;
    const char *profile_pos = getenv("TQ_XPU_PROFILE_POS");
    if (profile_pos && pos == atoi(profile_pos)) {
        tq_q().wait_and_throw();
        g_profile_run = &profile;
        g_profile_mark = std::chrono::steady_clock::now();
    }

    int ret = run_decode_layers(token_id, pos, g_qwen.L);
    if (ret == 0) {
        if (run_final_norm() != 0) ret = -5;
        profile_mark(kProfileFinalNorm);
    }
    if (ret == 0) {
        if (run_lm_head_argmax() != 0) ret = -6;
        profile_mark(kProfileLmHead);
    }

    if (g_profile_run == &profile) {
        static const char *names[kProfileStageCount] = {
            "embed", "norm", "quant", "projection", "conv", "delta",
            "attention", "activation", "residual", "final_norm", "lm_head",
        };
        double total = 0.0;
        for (int i = 0; i < kProfileStageCount; ++i) total += profile.ms[i];
        fprintf(stderr, "[xpu-profile] pos=%d total=%.3f ms\n", pos, total);
        for (int i = 0; i < kProfileStageCount; ++i) {
            fprintf(stderr, "  %-12s %8.3f ms  calls=%d\n",
                    names[i], profile.ms[i], profile.calls[i]);
        }
        for (const ProfileProjection &entry : profile.projections) {
            if (entry.calls == 0) break;
            fprintf(stderr, "    gemv s%d %6dx%-6d %8.3f ms  calls=%d\n",
                    entry.tier, entry.M, entry.K, entry.ms, entry.calls);
        }
        g_profile_run = nullptr;
    }

    if (ret != 0) return ret;
    g_qwen.state_pos[g_qwen.active_slot] = pos + 1;
    return g_qwen.last_argmax_id;
}

extern "C" int qwn_debug_decode_layers(int token_id, int pos, int n_layers,
                                       float *out, int max_count) {
    if (!out || max_count <= 0) return -9;
    if (g_pg_enabled) return -10;
    int ret = run_decode_layers(token_id, pos, n_layers);
    if (ret != 0) return ret;
    int count = max_count < g_qwen.H ? max_count : g_qwen.H;
    tq_q().wait();
    tq_d2h(out, g_qwen.d_x, (size_t)count * sizeof(float));
    return count;
}

extern "C" int qwn_debug_embed_input_norm(int token_id, int layer, float *out,
                                          int max_count) {
    if (!out || max_count <= 0) return -4;
    if (g_pg_enabled) return -10;
    tq_model_t *m = &g_qwen;
    if (!m->initialized) return -1;
    if (token_id < 0 || token_id >= m->V || layer < 0 || layer >= m->L) return -3;
    x_embed_lookup(m->d_x, m->d_embed, token_id, m->H);
    x_rmsnorm(m->d_norm_out, m->d_x, m->layers[layer].d_input_ln, m->H, m->eps);
    int count = max_count < m->H ? max_count : m->H;
    tq_q().wait();
    tq_d2h(out, m->d_norm_out, (size_t)count * sizeof(float));
    return count;
}

extern "C" int qwn_debug_forward_layers(int token_id, int n_layers, float *out,
                                        int max_count) {
    if (!out || max_count <= 0) return -4;
    if (g_pg_enabled) return -10;
    int ret = run_decode_layers(token_id, 0, n_layers);
    if (ret != 0) return ret;
    int count = max_count < g_qwen.H ? max_count : g_qwen.H;
    tq_q().wait();
    tq_d2h(out, g_qwen.d_x, (size_t)count * sizeof(float));
    return count;
}

extern "C" int qwn_debug_forward_final_norm(int token_id, float *out, int max_count) {
    if (!out || max_count <= 0) return -4;
    if (g_pg_enabled) return -10;
    int ret = run_decode_layers(token_id, 0, g_qwen.L);
    if (ret != 0) return ret;
    run_final_norm();
    int count = max_count < g_qwen.H ? max_count : g_qwen.H;
    tq_q().wait();
    tq_d2h(out, g_qwen.d_norm_out, (size_t)count * sizeof(float));
    return count;
}

extern "C" int qwn_debug_forward_argmax(int token_id, float *logit_out) {
    if (g_pg_enabled) return -10;
    int ret = run_decode_layers(token_id, 0, g_qwen.L);
    if (ret != 0) return ret;
    ret = run_final_norm();
    if (ret != 0) return -2;
    ret = run_lm_head_argmax();
    if (ret != 0) return -3;
    if (logit_out) *logit_out = g_qwen.last_argmax_logit;
    return g_qwen.last_argmax_id;
}

extern "C" int qwn_forward_first(int token_id) {
    return qwn_debug_forward_argmax(token_id, NULL);
}
