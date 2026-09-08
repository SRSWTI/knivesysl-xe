#include "tq_common.hpp"
#include "prefill_xmx.hpp"
#include <atomic>

// Heads per grouped-attention work-group; 3 is the shipping default (fits
// 128-GRF mode), 6 is the GRF256 experimental build (-DTQ_ATTN_G=6).
#ifndef TQ_ATTN_G
#define TQ_ATTN_G 3
#endif
#if TQ_ATTN_G >= 6
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#endif

// bf16 x bf16 RC8 DPAS for the design-9.5 attention path (same alias shapes
// as the probe-certified dpas_attn_probe.cpp).
#ifdef __SYCL_DEVICE_ONLY__
#define DPAS_BF_RC8_SEQ(dd, aa, bb)                                            \
    asm("{\n"                                                                  \
        ".decl DST     v_type=G type=F num_elts=128 alias=<%0,0>\n"            \
        ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"           \
        ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"            \
        "dpas.bf.bf.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"         \
        "}\n"                                                                  \
        : "+rw"(dd) : "rw"(aa), "rw"(bb))
#else
#define DPAS_BF_RC8_SEQ(dd, aa, bb) do { (void)(aa); (void)(bb); } while (0)
#endif

namespace {

// bf16 payload decode used by the CUDA kernels: widen the 16 payload bits into
// the high half of an IEEE-754 float (CUDA tq_bf16_to_float call sites below).
inline float tq_xpu_bf16_to_float(uint16_t bits) {
    return sycl::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

// OCP E4M3FN, the KV-cache element format.
//
// Branch-free decode. Shifting the 7 payload bits left by 20 lands E4M3's
// mantissa at fp32 bits [22:20] and its exponent at [26:23], giving a float of
// 2^(e-127) scale; one multiply by 2^120 rebases the exponent to E4M3's bias
// of 7. This is EXACT for subnormals too, which is why no fixup branch is
// needed: at e=0 the fp32 subnormal reading is m*2^-129, and m*2^-129 * 2^120
// = m*2^-9, exactly E4M3's (m/8)*2^-6.
//
// The obvious version - ldexp plus an exponent==0 branch - costs about seven
// ALU ops per element. Decoding ~268M KV elements per step at 4k context, that
// measured SLOWER than the fp32 cache it replaced (27.16 vs 30.34 tok/s); this
// form is the reason the fp8 cache is a win rather than a loss.
inline float tq_xpu_e4m3_to_float(uint8_t code) {
    const uint32_t bits = ((uint32_t)(code & 0x80u) << 24) |
                          ((uint32_t)(code & 0x7Fu) << 20);
    return sycl::bit_cast<float>(bits) * 0x1p120f;
}

// Reference decode, kept as the selftest's oracle: bit-identical to
// tq_e4m3_to_float in kernels_core.cpp (CUDA forward_qwen.cu:513-525).
inline float tq_xpu_e4m3_to_float_ref(uint8_t code) {
    const int sign = (code >> 7) & 1;
    const int exponent = (code >> 3) & 0x0f;
    const int mantissa = code & 0x07;
    if ((code & 0x7fu) == 0) return 0.0f;
    const float magnitude = exponent == 0
        ? sycl::ldexp(static_cast<float>(mantissa) * 0.125f, -6)
        : sycl::ldexp(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
    return sign ? -magnitude : magnitude;
}

inline uint8_t tq_xpu_e4m3_from_float(float v) {
    const uint8_t sign = (v < 0.0f) ? 0x80u : 0x00u;
    const float a = sycl::fabs(v);
    if (!(a > 0.0f)) return 0u;                        // zero maps to +0
    if (a >= 448.0f) return sign | 0x7Eu;              // clamp to max finite
    if (a < 0.015625f) {                               // subnormal: step 2^-9
        const int m = (int)sycl::fmin(sycl::rint(a * 512.0f), 7.0f);
        return sign | (uint8_t)m;
    }
    int e = (int)sycl::floor(sycl::log2(a));
    const float p = sycl::exp2((float)e);
    int m = (int)sycl::rint((a / p - 1.0f) * 8.0f);
    if (m >= 8) { m = 0; ++e; }
    if (e > 8) { e = 8; m = 6; }                       // 2^8 * 1.75 = 448
    return sign | (uint8_t)(((e + 7) << 3) | m);
}

template <bool Paged>
inline size_t tq_kv_physical_token(int logical_token, tq_kv_layout_t layout) {
    if constexpr (Paged) {
        const size_t physical_page = static_cast<size_t>(
            layout.block_table[logical_token >> layout.page_log]);
        return (physical_page << layout.page_log) +
               static_cast<size_t>(logical_token & layout.page_mask);
    }
    return static_cast<size_t>(logical_token);
}

template <bool Paged, typename Group>
inline size_t tq_kv_group_physical_row(
    int logical_token, int kv_head, int nkv, tq_kv_layout_t layout,
    Group group, int local_id) {
    size_t physical_token;
    if constexpr (Paged) {
        physical_token = local_id == 0
            ? tq_kv_physical_token<true>(logical_token, layout)
            : 0;
        physical_token = sycl::group_broadcast(group, physical_token, 0);
    } else {
        physical_token = static_cast<size_t>(logical_token);
    }
    return physical_token * static_cast<size_t>(nkv) +
           static_cast<size_t>(kv_head);
}

enum tq_attn_branch_t {
    TQ_ATTN_BRANCH_PREFILL = 0,
    TQ_ATTN_BRANCH_SIMD16_SHORT = 1,
    TQ_ATTN_BRANCH_SIMD16_SHARDED = 2,
    TQ_ATTN_BRANCH_SIMD16_GROUPED = 3,
    TQ_ATTN_BRANCH_GENERIC = 4,
    TQ_ATTN_BRANCH_SIMD16_DPAS = 5,
    TQ_ATTN_BRANCH_PREFILL_XMX = 6,
    TQ_ATTN_BRANCH_PREFILL_SCALAR = 7,
    TQ_ATTN_BRANCH_COUNT = 8,
};

std::atomic<unsigned long long> g_attn_branch_counts[TQ_ATTN_BRANCH_COUNT];

// Lazily grown per-(head, segment) attention partials for the grouped path:
// [head][segment][hd + 2], the trailing two slots carrying that segment's
// online-softmax max and denominator.
float *g_attn_part = nullptr;
size_t g_attn_part_cap = 0;

void ensure_attn_partials(size_t elements) {
    if (g_attn_part_cap >= elements) return;
    if (g_attn_part) tq_dev_free(g_attn_part);
    g_attn_part = static_cast<float *>(
        tq_dev_alloc(elements * sizeof(float), "attn.partial"));
    g_attn_part_cap = elements;
}

// Pre-packed bf16 Q A-fragments for the DPAS attention path (design 9.5):
// [nkv][16 k-chunks][64 dwords] - 8 rows (6 live GQA heads + 2 zero) per
// chunk in the verified apair/abyte layout.
uint32_t *g_attn_qfrag = nullptr;
size_t g_attn_qfrag_cap = 0;

void ensure_attn_qfrag(size_t dwords) {
    if (g_attn_qfrag_cap >= dwords) return;
    if (g_attn_qfrag) tq_dev_free(g_attn_qfrag);
    g_attn_qfrag = static_cast<uint32_t *>(
        tq_dev_alloc(dwords * sizeof(uint32_t), "attn.qfrag"));
    g_attn_qfrag_cap = dwords;
}

}  // namespace

extern "C" int qwn_attn_branch_counts(unsigned long long *out, int count) {
    // Preserve existing six-counter clients; extended clients request eight.
    if (!out || count < 6) return -1;
    for (int i = 0; i < TQ_ATTN_BRANCH_COUNT && i < count; ++i)
        out[i] = g_attn_branch_counts[i].load(std::memory_order_relaxed);
    return 0;
}

extern "C" void qwn_attn_branch_counts_reset(void) {
    for (int i = 0; i < TQ_ATTN_BRANCH_COUNT; ++i)
        g_attn_branch_counts[i].store(0, std::memory_order_relaxed);
}

// Exhaustive round-trip: every one of the 256 E4M3 codes must decode to a
// float that re-encodes to the same code (the NaN codes 0x7F/0xFF and negative
// zero 0x80 are excluded - no finite/signed-zero preimage here). Also bounds
// the encode error across E4M3's normal range, which IS the per-element
// quantization error an fp8 KV cache would introduce.
extern "C" int qwn_e4m3_selftest(void) {
    int *d_bad = static_cast<int *>(tq_dev_alloc(2 * sizeof(int), "e4m3.bad"));
    float *d_worst = static_cast<float *>(tq_dev_alloc(sizeof(float), "e4m3.worst"));
    tq_dev_zero(d_bad, 2 * sizeof(int));
    tq_dev_zero(d_worst, sizeof(float));
    tq_q().parallel_for(sycl::range<1>(256), [=](sycl::id<1> id) {
        const uint8_t code = (uint8_t)id[0];
        if ((code & 0x7Fu) == 0x7Fu) return;          // NaN
        // 0x80 is negative zero. The reference returns +0.0 for it (it tests
        // the magnitude bits before applying sign) while the bit-trick keeps
        // the sign bit and yields -0.0. Numerically identical, so exclude it
        // from the bit-equality check rather than pessimize the fast path.
        if (code == 0x80u) return;
        // Otherwise the fast decode must be BIT-IDENTICAL to the CUDA-faithful
        // reference for every finite code, subnormals included.
        const float fast = tq_xpu_e4m3_to_float(code);
        const float ref_v = tq_xpu_e4m3_to_float_ref(code);
        if (sycl::bit_cast<uint32_t>(fast) != sycl::bit_cast<uint32_t>(ref_v)) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                r(d_bad[0]);
            r.fetch_add(1);
        }
        if (tq_xpu_e4m3_from_float(fast) != code) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                r(d_bad[0]);
            r.fetch_add(1);
        }
    });
    tq_q().parallel_for(sycl::range<1>(4096), [=](sycl::id<1> id) {
        const float v = sycl::exp2(-6.0f + 14.0f * (float)(int)id[0] / 4095.0f);
        const float r = tq_xpu_e4m3_to_float(tq_xpu_e4m3_from_float(v));
        const float rel = sycl::fabs(r - v) / v;
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            wref(d_worst[0]);
        wref.fetch_max(rel);
        if (!(rel <= 0.07f)) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                ref(d_bad[1]);
            ref.fetch_add(1);
        }
    });
    tq_q().wait_and_throw();
    int bad[2];
    float worst;
    tq_d2h(bad, d_bad, sizeof(bad));
    tq_d2h(&worst, d_worst, sizeof(worst));
    tq_dev_free(d_bad);
    tq_dev_free(d_worst);
    std::printf("e4m3 selftest: %d/254 codes failed round-trip, %d samples over "
                "7%% rel, worst rel=%.4f  %s\n",
                bad[0], bad[1], worst,
                (bad[0] == 0 && bad[1] == 0) ? "PASS" : "FAIL");
    return (bad[0] == 0 && bad[1] == 0) ? 0 : -1;
}

void x_linear_conv_update(float *d_out, float *d_state, const float *d_x,
                          const uint16_t *d_conv_w, int conv_dim, int ks) {
    constexpr size_t local_size = 256;
    const size_t global_size =
        ((static_cast<size_t>(conv_dim) + local_size - 1) / local_size) * local_size;

    tq_q().parallel_for(sycl::nd_range<1>(global_size, local_size),
                        [=](sycl::nd_item<1> item) {
        const int channel = static_cast<int>(item.get_global_linear_id());
        if (channel >= conv_dim) return;

        float sum = 0.0f;
        float *state = d_state + static_cast<size_t>(channel) * ks;
        const uint16_t *weight = d_conv_w + static_cast<size_t>(channel) * ks;

        // CUDA lines 7977-7984: shift the causal depthwise state left first,
        // then accumulate taps in increasing kernel-index order and append x.
        for (int tap = 0; tap < ks - 1; ++tap) {
            state[tap] = state[tap + 1];
            sum += state[tap] * tq_xpu_bf16_to_float(weight[tap]);
        }
        state[ks - 1] = d_x[channel];
        sum += d_x[channel] * tq_xpu_bf16_to_float(weight[ks - 1]);

        // CUDA line 7985: the convolution output for every q/k/v channel is
        // SiLU(sum); the kernel has no unswished channel range.
        d_out[channel] = sum / (1.0f + sycl::exp(-sum));
    });
}

// Chunk-parallel causal depthwise conv + SiLU over T tokens (design 9.2).
// Token t's window is raw inputs [t-ks+1, t]; negative indices come from the
// incoming conv state (taps 1..ks-1 hold x_{-(ks-1)}..x_{-1}, tap 0 is dead -
// the serial kernel shifts it out before use). d_out MUST differ from d_x:
// later tokens read their neighbours' RAW inputs. The state advance runs as
// a second kernel so no work-item writes state while another still reads it
// (the in-order queue serializes the pair).
void x_linear_conv_chunk(float *d_out, float *d_state, const float *d_x,
                         const uint16_t *d_conv_w, int conv_dim, int ks, int T) {
    constexpr size_t local_size = 256;
    const size_t items = (size_t)conv_dim * T;
    const size_t global_size = ((items + local_size - 1) / local_size) * local_size;
    tq_q().parallel_for(sycl::nd_range<1>(global_size, local_size),
                        [=](sycl::nd_item<1> item) {
        const size_t gid = item.get_global_linear_id();
        if (gid >= items) return;
        const int t = (int)(gid / conv_dim);
        const int channel = (int)(gid - (size_t)t * conv_dim);
        const uint16_t *weight = d_conv_w + (size_t)channel * ks;
        const float *state = d_state + (size_t)channel * ks;
        float sum = 0.0f;
        for (int tap = 0; tap < ks; ++tap) {
            const int idx = t - (ks - 1) + tap;
            const float x = (idx >= 0)
                ? d_x[(size_t)idx * conv_dim + channel]
                : state[idx + ks];   // idx in [-(ks-1), -1] -> tap 1..ks-1
            sum += x * tq_xpu_bf16_to_float(weight[tap]);
        }
        d_out[(size_t)t * conv_dim + channel] = sum / (1.0f + sycl::exp(-sum));
    });
    x_linear_conv_advance(d_state, d_x, conv_dim, ks, T);
}

// Advance the conv state to "just processed token T-1": tap i holds raw
// x_{T-ks+i} (tap 0 is dead but kept exact for the serial decode resume).
// Negative indices keep the pre-chunk state's taps - also used standalone by
// the spec wave's rewind (design 9.3) after restoring the state snapshot.
void x_linear_conv_advance(float *d_state, const float *d_x, int conv_dim,
                           int ks, int T) {
    tq_q().parallel_for(sycl::range<1>((size_t)conv_dim * ks),
                        [=](sycl::id<1> id) {
        const int channel = (int)(id[0] / ks);
        const int tap = (int)(id[0] - (size_t)channel * ks);
        const int idx = T - ks + tap;
        float *state = d_state + (size_t)channel * ks;
        state[tap] = (idx >= 0) ? d_x[(size_t)idx * conv_dim + channel]
                                : state[idx + ks];
    });
}

// Chunkwise-exact gated delta rule over T prompt tokens (design 9.2; the
// CUDA twin's k_tq_deltanet_chunk, CK=8, D=128, one work-group per value
// head). Within a CK sub-chunk (incoming state S0, cumulative log-decay
// la_j, gamma_j = e^{la_j}):
//   (I + L) Delta = R,  L[j,m] = beta_j e^{la_j-la_m}(kn_j.kn_m)  (m < j)
//   R[j] = beta_j (v_j - gamma_j S0^T kn_j)
//   core_j = gamma_j S0^T qn_j + sum_{i<=j} e^{la_j-la_i}(qn_j.kn_i) Delta_i
//   S0'   = gamma_last S0 + sum_i e^{la_last-la_i} kn_i Delta_i
// Sub-chunks carry serially through S0 inside ONE launch; everything within
// a sub-chunk is dense parallel work. All decay exponents are <= 0. Same
// math as the serial core, reassociated (CUDA measured max diff 9.9e-6);
// gated by prefill_check + the parity harness.
int x_deltanet_chunk(float *d_core_out, float *d_recurrent,
                     const float *d_conv_out, const float *d_z,
                     const float *d_b, const float *d_a, const float *d_A_log,
                     const uint16_t *d_dt_bias, const float *d_norm_w, int T,
                     int nk, int dk, int nv, int dv, float eps) {
    if (dk != 128 || dv != 128 || nk <= 0 || nv <= 0 || (nv % nk) != 0)
        return -1;
    constexpr int CK = 8;
    constexpr int D = 128;
    // v2 shape (v1 postmortem: 48 WGs x 128 threads = 19% occupancy ran
    // SLOWER than the serial loop, 367 vs 464 tok/s live). This version
    // mirrors x_linear_decode_core_gated_fast's proven layout: one
    // 1024-thread work-group per head (48 x 1024 = 1.5x lane
    // oversubscription), the dk contraction split across G=8 stripes, and
    // each thread's 16 state values held in REGISTERS across all T/CK
    // sub-chunks - the recurrent state is read once and written once per
    // layer chunk instead of 2x(T/CK) times.
    constexpr int G = 8;
    constexpr int kPer = D / G;                    // 16 state rows per thread
    tq_q().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> kn_sh(sycl::range<1>(CK * D), cgh);
        sycl::local_accessor<float, 1> qn_sh(sycl::range<1>(CK * D), cgh);
        sycl::local_accessor<float, 1> psum(sycl::range<1>(G * CK * D), cgh);
        sycl::local_accessor<float, 1> Dl_sh(sycl::range<1>(CK * D), cgh);
        sycl::local_accessor<float, 1> Lmat(sycl::range<1>(CK * CK), cgh);
        sycl::local_accessor<float, 1> Amat(sycl::range<1>(CK * CK), cgh);
        sycl::local_accessor<float, 1> la_sh(sycl::range<1>(CK), cgh);
        sycl::local_accessor<float, 1> beta_sh(sycl::range<1>(CK), cgh);
        sycl::local_accessor<float, 1> qfac(sycl::range<1>(CK), cgh);
        sycl::local_accessor<float, 1> kfac(sycl::range<1>(CK), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>((size_t)nv * D * G, (size_t)D * G),
            [=](sycl::nd_item<1> item) {
                const int head = (int)item.get_group_linear_id();
                const int tid = (int)item.get_local_linear_id();
                const int c = tid & (D - 1);       // value column
                const int ch = tid >> 7;           // contraction stripe
                const int kk0 = ch * kPer;
                const int group = nv / nk;
                const int key_head = head / group;
                const int key_dim = nk * D;
                const int value_dim = nv * D;
                const int conv_dim = 2 * key_dim + value_dim;
                const float Alog = d_A_log[head];
                const float dtb = tq_xpu_bf16_to_float(d_dt_bias[head]);
                float *state = d_recurrent + (size_t)head * D * D;
                float st[kPer];
                for (int j = 0; j < kPer; ++j)
                    st[j] = state[(size_t)(kk0 + j) * D + c];
                for (int s = 0; s < T; s += CK) {
                    const int L = (T - s) < CK ? (T - s) : CK;
                    // 1a: per-token q/k inverse-L2 factors (thread j = token).
                    if (tid < L) {
                        const int t = s + tid;
                        const float *q = d_conv_out + (size_t)t * conv_dim +
                                         (size_t)key_head * D;
                        const float *k = q + key_dim;
                        float qss = 0.0f, kss = 0.0f;
                        for (int d = 0; d < D; ++d) {
                            qss += q[d] * q[d];
                            kss += k[d] * k[d];
                        }
                        qfac[tid] = sycl::rsqrt(qss + 1.0e-6f) *
                                    sycl::rsqrt((float)D);
                        kfac[tid] = sycl::rsqrt(kss + 1.0e-6f);
                    }
                    if (tid == 0) {
                        float cum = 0.0f;
                        for (int j = 0; j < L; ++j) {
                            const int t = s + j;
                            const float bb = d_b[(size_t)t * nv + head];
                            const float sp_arg =
                                d_a[(size_t)t * nv + head] + dtb;
                            const float sp =
                                sycl::log1p(sycl::exp(-sycl::fabs(sp_arg))) +
                                sycl::fmax(sp_arg, 0.0f);
                            cum += -sycl::exp(Alog) * sp;
                            beta_sh[j] = 1.0f / (1.0f + sycl::exp(-bb));
                            la_sh[j] = cum;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    // 1b: one element per thread (CK*D == D*G at CK == G).
                    if (ch < L) {
                        const int t = s + ch;
                        const float *q = d_conv_out + (size_t)t * conv_dim +
                                         (size_t)key_head * D;
                        qn_sh[ch * D + c] = q[c] * qfac[ch];
                        kn_sh[ch * D + c] = q[key_dim + c] * kfac[ch];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    // 2: decayed inner-product matrices (L*L pair-dots).
                    if (tid < L * L) {
                        const int j = tid / L, m = tid % L;
                        float dkk = 0.0f, dqk = 0.0f;
                        for (int d = 0; d < D; ++d) {
                            const float knm = kn_sh[m * D + d];
                            dkk += kn_sh[j * D + d] * knm;
                            dqk += qn_sh[j * D + d] * knm;
                        }
                        const float decay =
                            (m <= j) ? sycl::exp(la_sh[j] - la_sh[m]) : 0.0f;
                        Lmat[j * CK + m] =
                            (m < j) ? beta_sh[j] * decay * dkk : 0.0f;
                        Amat[j * CK + m] = (m <= j) ? decay * dqk : 0.0f;
                    }
                    // 3: kS0/qS0 via stripe partials from the register state,
                    // two passes through one SLM buffer.
                    for (int j = 0; j < L; ++j) {
                        float pk = 0.0f;
                        for (int k = 0; k < kPer; ++k)
                            pk += kn_sh[j * D + kk0 + k] * st[k];
                        psum[((size_t)ch * CK + j) * D + c] = pk;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    float kS0[CK], qS0[CK], Dl[CK];
                    if (ch == 0) {
                        for (int j = 0; j < L; ++j) {
                            float sum = 0.0f;
                            for (int g = 0; g < G; ++g)
                                sum += psum[((size_t)g * CK + j) * D + c];
                            kS0[j] = sum;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int j = 0; j < L; ++j) {
                        float pq = 0.0f;
                        for (int k = 0; k < kPer; ++k)
                            pq += qn_sh[j * D + kk0 + k] * st[k];
                        psum[((size_t)ch * CK + j) * D + c] = pq;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    if (ch == 0) {
                        for (int j = 0; j < L; ++j) {
                            float sum = 0.0f;
                            for (int g = 0; g < G; ++g)
                                sum += psum[((size_t)g * CK + j) * D + c];
                            qS0[j] = sum;
                        }
                        // 4+5: RHS then forward substitution (column c).
                        for (int j = 0; j < L; ++j) {
                            const int t = s + j;
                            const float vv =
                                d_conv_out[(size_t)t * conv_dim + 2 * key_dim +
                                           (size_t)head * D + c];
                            Dl[j] = beta_sh[j] *
                                    (vv - sycl::exp(la_sh[j]) * kS0[j]);
                        }
                        for (int j = 1; j < L; ++j) {
                            float acc = Dl[j];
                            for (int m = 0; m < j; ++m)
                                acc -= Lmat[j * CK + m] * Dl[m];
                            Dl[j] = acc;
                        }
                        for (int j = 0; j < L; ++j) Dl_sh[j * D + c] = Dl[j];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    // 6+7: core + gated RMSNorm (group-scope loop; only the
                    // ch==0 stripe carries values into the reduction).
                    for (int j = 0; j < L; ++j) {
                        float acc = 0.0f;
                        if (ch == 0) {
                            acc = sycl::exp(la_sh[j]) * qS0[j];
                            for (int i = 0; i <= j; ++i)
                                acc += Amat[j * CK + i] * Dl[i];
                        }
                        const float ss = sycl::reduce_over_group(
                            item.get_group(), acc * acc, sycl::plus<float>());
                        if (ch == 0) {
                            const int t = s + j;
                            const float normed =
                                acc * sycl::rsqrt(ss / (float)D + eps) *
                                d_norm_w[c];
                            const float gate = d_z[(size_t)t * value_dim +
                                                   (size_t)head * D + c];
                            d_core_out[(size_t)t * value_dim +
                                       (size_t)head * D + c] =
                                normed * (gate / (1.0f + sycl::exp(-gate)));
                        }
                    }
                    // 8: state carry in registers (all stripes).
                    const float la_last = la_sh[L - 1];
                    const float gamma_last = sycl::exp(la_last);
                    float dec[CK];
                    for (int i = 0; i < L; ++i)
                        dec[i] = sycl::exp(la_last - la_sh[i]) *
                                 Dl_sh[i * D + c];
                    for (int k = 0; k < kPer; ++k) {
                        float acc = gamma_last * st[k];
                        for (int i = 0; i < L; ++i)
                            acc += dec[i] * kn_sh[i * D + kk0 + k];
                        st[k] = acc;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                for (int j = 0; j < kPer; ++j)
                    state[(size_t)(kk0 + j) * D + c] = st[j];
            });
    });
    return 0;
}

// Shipping-shape DeltaNet decode core (dk == dv == 128).
//
// The generic kernel below launches nv=48 work-groups of dv=128 threads: 6144
// work-items, ~19% of the B70's work-item capacity, so it reached only a fifth
// of DRAM bandwidth. It also touched the recurrent state three times (two read
// passes plus the write-back).
//
// This path fixes both. The dk axis is split across 8 chunks inside a single
// 1024-thread work-group, so one work-group spans 8 XVEs and the 48 heads
// cover 384 XVE-slots - full occupancy - and each thread holds its 16 old
// state values in registers, so the state is read ONCE and written ONCE.
//
// Removing the second read uses the identity
//   core_c = sum_k S_new[k][c] * qn[k]
//          = decay * sum_k S_old[k][c] * qn[k] + delta_c * sum_k kn[k] * qn[k]
// where the last factor is a per-head scalar. Same arithmetic, reassociated:
// eps-level drift, gated by the usual TF/parity harness.
void x_linear_decode_core_gated_fast(float *d_out, float *d_recurrent,
                                     const float *d_conv_out, const float *d_z,
                                     const float *d_b, const float *d_a,
                                     const float *d_A_log,
                                     const uint16_t *d_dt_bias,
                                     const float *d_norm_w,
                                     int nk, int dk, int nv, int dv, float eps) {
    constexpr int kDim = 128;
    constexpr int kChunks = 8;
    constexpr int kPerChunk = kDim / kChunks;         // 16 state rows per thread
    constexpr size_t kLocal = (size_t)kDim * kChunks;  // 1024

    tq_q().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> qn(sycl::range<1>(kDim), cgh);
        sycl::local_accessor<float, 1> kn(sycl::range<1>(kDim), cgh);
        sycl::local_accessor<float, 1> pq(sycl::range<1>(kLocal), cgh);
        sycl::local_accessor<float, 1> pk(sycl::range<1>(kLocal), cgh);
        sycl::local_accessor<float, 1> sq(sycl::range<1>(kDim), cgh);
        sycl::local_accessor<float, 1> sk(sycl::range<1>(kDim), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>((size_t)nv * kLocal, kLocal),
            [=](sycl::nd_item<1> item) {
                const int head = (int)item.get_group_linear_id();
                const int tid = (int)item.get_local_linear_id();
                const int c = tid % kDim;             // dv column
                const int ch = tid / kDim;            // dk chunk
                const int kk0 = ch * kPerChunk;

                const int group = nv / nk;
                const int key_head = head / group;
                const size_t key_dim = (size_t)nk * dk;
                const float *q_head = d_conv_out + (size_t)key_head * dk;
                const float *k_head = d_conv_out + key_dim + (size_t)key_head * dk;
                const float *v_head = d_conv_out + 2 * key_dim + (size_t)head * dv;
                const float *z_head = d_z + (size_t)head * dv;

                // L2 norms over dk. Only the first kDim threads carry data; the
                // rest contribute zero to the work-group reduction.
                const float qv = (tid < kDim) ? q_head[tid] : 0.0f;
                const float kv = (tid < kDim) ? k_head[tid] : 0.0f;
                const float q_sum = sycl::reduce_over_group(
                    item.get_group(), qv * qv, sycl::plus<float>());
                const float k_sum = sycl::reduce_over_group(
                    item.get_group(), kv * kv, sycl::plus<float>());
                const float q_scale = sycl::rsqrt(q_sum + 1.0e-6f) *
                                      sycl::rsqrt((float)dk);
                const float k_scale = sycl::rsqrt(k_sum + 1.0e-6f);
                if (tid < kDim) {
                    qn[tid] = qv * q_scale;
                    kn[tid] = kv * k_scale;
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Per-head scalar sum_k kn[k]*qn[k].
                const float kq_term = (tid < kDim) ? qn[tid] * kn[tid] : 0.0f;
                const float KQ = sycl::reduce_over_group(
                    item.get_group(), kq_term, sycl::plus<float>());

                const float beta = 1.0f / (1.0f + sycl::exp(-d_b[head]));
                const float softplus_arg =
                    d_a[head] + tq_xpu_bf16_to_float(d_dt_bias[head]);
                const float softplus =
                    sycl::log1p(sycl::exp(-sycl::fabs(softplus_arg))) +
                    sycl::fmax(softplus_arg, 0.0f);
                const float decay = sycl::exp(-sycl::exp(d_A_log[head]) * softplus);

                // One read pass; the old column stays in registers.
                float *state = d_recurrent + (size_t)head * dk * dv;
                float st[kPerChunk];
                float acc_q = 0.0f;
                float acc_k = 0.0f;
                for (int j = 0; j < kPerChunk; ++j) {
                    st[j] = state[(size_t)(kk0 + j) * dv + c];
                    acc_q += st[j] * qn[kk0 + j];
                    acc_k += st[j] * kn[kk0 + j];
                }
                pq[tid] = acc_q;
                pk[tid] = acc_k;
                item.barrier(sycl::access::fence_space::local_space);
                if (ch == 0) {
                    float tq_ = 0.0f;
                    float tk_ = 0.0f;
                    for (int i = 0; i < kChunks; ++i) {
                        tq_ += pq[(size_t)i * kDim + c];
                        tk_ += pk[(size_t)i * kDim + c];
                    }
                    sq[c] = tq_;
                    sk[c] = tk_;
                }
                item.barrier(sycl::access::fence_space::local_space);

                const float kv_mem = decay * sk[c];
                const float delta = (v_head[c] - kv_mem) * beta;
                const float core = decay * sq[c] + delta * KQ;

                // Write-back straight from registers - no second read.
                for (int j = 0; j < kPerChunk; ++j)
                    state[(size_t)(kk0 + j) * dv + c] =
                        st[j] * decay + kn[kk0 + j] * delta;

                // RMSNorm spans the dv outputs; only chunk 0 owns a column.
                const float core_sq = (ch == 0) ? core * core : 0.0f;
                const float core_sum = sycl::reduce_over_group(
                    item.get_group(), core_sq, sycl::plus<float>());
                if (ch == 0) {
                    const float normed = core *
                        sycl::rsqrt(core_sum / (float)dv + eps) * d_norm_w[c];
                    const float gate = z_head[c];
                    d_out[(size_t)head * dv + c] =
                        normed * (gate / (1.0f + sycl::exp(-gate)));
                }
            });
    });
}

void x_linear_decode_core_gated(float *d_out, float *d_recurrent,
                                const float *d_conv_out, const float *d_z,
                                const float *d_b, const float *d_a,
                                const float *d_A_log, const uint16_t *d_dt_bias,
                                const float *d_norm_w,
                                int nk, int dk, int nv, int dv, float eps) {
    // Shipping shape takes the full-occupancy single-read-pass kernel.
    // TQ_XPU_DELTA_FAST=0 reverts to the generic reference below.
    if (dk == 128 && dv == 128 && nk > 0 && nv > 0 && (nv % nk) == 0) {
        const char *e = std::getenv("TQ_XPU_DELTA_FAST");
        if (!e || e[0] != '0') {
            x_linear_decode_core_gated_fast(d_out, d_recurrent, d_conv_out, d_z,
                                            d_b, d_a, d_A_log, d_dt_bias,
                                            d_norm_w, nk, dk, nv, dv, eps);
            return;
        }
    }
    const size_t local_size = static_cast<size_t>(dv);
    const size_t global_size = static_cast<size_t>(nv) * local_size;

    tq_q().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> qn(sycl::range<1>(static_cast<size_t>(dk)), cgh);
        sycl::local_accessor<float, 1> kn(sycl::range<1>(static_cast<size_t>(dk)), cgh);

        cgh.parallel_for(sycl::nd_range<1>(global_size, local_size),
                         [=](sycl::nd_item<1> item) {
            const int head = static_cast<int>(item.get_group_linear_id());
            const int tid = static_cast<int>(item.get_local_linear_id());

            // CUDA lines 8215-8221: value heads share q/k by GQA grouping.
            // qkv_conv slices are exactly q=[0,nk*dk), k=[nk*dk,2*nk*dk),
            // v=[2*nk*dk,2*nk*dk+nv*dv); z is [nv,dv].
            const int group = nv / nk;
            const int key_head = head / group;
            const size_t key_dim = static_cast<size_t>(nk) * dk;
            const float *q_head = d_conv_out + static_cast<size_t>(key_head) * dk;
            const float *k_head = d_conv_out + key_dim + static_cast<size_t>(key_head) * dk;
            const float *v_head = d_conv_out + 2 * key_dim + static_cast<size_t>(head) * dv;
            const float *z_head = d_z + static_cast<size_t>(head) * dv;

            float q_sum_part = 0.0f;
            float k_sum_part = 0.0f;
            for (int kk = tid; kk < dk; kk += dv) {
                const float qv = q_head[kk];
                const float kv = k_head[kk];
                q_sum_part += qv * qv;
                k_sum_part += kv * kv;
            }
            const float q_sum = sycl::reduce_over_group(
                item.get_group(), q_sum_part, sycl::plus<float>());
            const float k_sum = sycl::reduce_over_group(
                item.get_group(), k_sum_part, sycl::plus<float>());

            // CUDA lines 8222-8248: L2-normalize q and k; q alone also gets
            // rsqrt(dk), matching the decode kernel's attention scaling.
            const float q_scale = sycl::rsqrt(q_sum + 1.0e-6f) *
                                  sycl::rsqrt(static_cast<float>(dk));
            const float k_scale = sycl::rsqrt(k_sum + 1.0e-6f);
            for (int kk = tid; kk < dk; kk += dv) {
                qn[kk] = q_head[kk] * q_scale;
                kn[kk] = k_head[kk] * k_scale;
            }
            item.barrier(sycl::access::fence_space::local_space);

            // CUDA lines 8250-8254: beta=sigmoid(b), stable softplus(a+dt),
            // g=-exp(A_log)*softplus, and decay=exp(g), once per value head.
            const float beta = 1.0f / (1.0f + sycl::exp(-d_b[head]));
            const float softplus_arg =
                d_a[head] + tq_xpu_bf16_to_float(d_dt_bias[head]);
            const float softplus = sycl::log1p(sycl::exp(-sycl::fabs(softplus_arg))) +
                                   sycl::fmax(softplus_arg, 0.0f);
            const float g = -sycl::exp(d_A_log[head]) * softplus;
            const float decay = sycl::exp(g);

            // CUDA lines 8255-8267: recurrent state is [nv,dk,dv]. Each
            // work-item owns one dv column, computes the old-state retrieval,
            // then updates that column in place before contracting S_new with qn.
            float *state = d_recurrent + static_cast<size_t>(head) * dk * dv;
            float kv_mem = 0.0f;
            for (int kk = 0; kk < dk; ++kk) {
                kv_mem += state[static_cast<size_t>(kk) * dv + tid] * decay * kn[kk];
            }
            const float delta = (v_head[tid] - kv_mem) * beta;
            float core = 0.0f;
            for (int kk = 0; kk < dk; ++kk) {
                const size_t state_index = static_cast<size_t>(kk) * dv + tid;
                const float new_value = state[state_index] * decay + kn[kk] * delta;
                state[state_index] = new_value;
                core += new_value * qn[kk];
            }

            // CUDA lines 8269-8283: RMSNorm spans the dv outputs of this value
            // head, uses the supplied eps and plain fp32 norm_weight, then folds
            // the SiLU(z) gate elementwise into the normalized core.
            const float core_sum = sycl::reduce_over_group(
                item.get_group(), core * core, sycl::plus<float>());
            const float normed = core *
                sycl::rsqrt(core_sum / static_cast<float>(dv) + eps) * d_norm_w[tid];
            const float gate = z_head[tid];
            d_out[static_cast<size_t>(head) * dv + tid] =
                normed * (gate / (1.0f + sycl::exp(-gate)));
        });
    });
}

// ===== wide prefill: batched KV write + query-tiled causal flash attention =====
//
// The per-token path below (x_full_attn_decode) re-reads the ENTIRE KV history
// once per query token. That is CUDA's scalar reference kernel k_tq_wide_attn
// (forward_qwen.cu:2830), which CUDA itself abandons above ~16k. Its fast path
// is k_tq_wide_attn_mma, and forward_qwen.cu:2944 states the principle:
//
//   "PREFILL attention is COMPUTE-bound (~170 FLOP/byte: 16 query rows reuse
//    the same K), so tensor cores win here (unlike the memory-bound decode,
//    kept scalar) ... per-row STAIRCASE mask: token attends keys [0..its pos]"
//
// So: tile the QUERY dimension. One work-group owns kQRows consecutive query
// rows of one q-head, and each KV row is read ONCE for all of them - a kQRows
// reduction in attention traffic over the per-token loop.
//
// Geometry: 16 subgroups x 16 lanes. Subgroup s owns query row s, so the
// online softmax is subgroup-local and needs NO work-group barrier - the only
// barriers are for staging K/V. Lane l owns dims [16l, 16l+16), giving 16
// accumulators + 16 query values live, ~40 floats/lane inside the 128-GRF
// budget (the G=6 grouped decode variant needed ~208 and spilled).
//
// K/V stage in SLM as E4M3 BYTES: 64 keys cost 33 KB, where fp32 would cost
// 132 KB and not fit. The fp8 cache pays off twice here.
template <bool Paged>
void x_prefill_kv_write_impl(const float *d_k_proj, const float *d_v_proj,
                             const uint16_t *d_k_norm, uint8_t *d_k_cache,
                             uint8_t *d_v_cache, uint16_t *d_k_scale,
                             uint16_t *d_v_scale, int pos0, int T, int nkv,
                             int hd, float eps, float rope_theta,
                             float partial_rotary_factor,
                             tq_kv_layout_t layout) {
    constexpr int kSubgroup = 16;
    const int kLaneValues = hd / kSubgroup;
    const int rotary_dim = static_cast<int>(static_cast<float>(hd) *
                                            partial_rotary_factor);
    const int rotary_half = rotary_dim / 2;
    const int kv_w = nkv * hd;
    // Byte-identical to the per-token store inside x_full_attn_decode; only
    // the token index and position are lifted into the grid.
    tq_q().parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(nkv) * T * kSubgroup, kSubgroup),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSubgroup)]] {
            const int wg = static_cast<int>(item.get_group_linear_id());
            const int t = wg / nkv;
            const int kv_head = wg % nkv;
            const sycl::sub_group sg = item.get_sub_group();
            const int lane = static_cast<int>(sg.get_local_linear_id());
            const int pos = pos0 + t;
            const size_t base = static_cast<size_t>(t) * kv_w +
                                static_cast<size_t>(kv_head) * hd;
            float sum_sq = 0.0f;
            for (int i = 0; i < kLaneValues; ++i) {
                const float value = d_k_proj[base + lane + i * kSubgroup];
                sum_sq += value * value;
            }
            sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
            const float inv_rms =
                sycl::rsqrt(sum_sq / static_cast<float>(hd) + eps);
            float keys[16];
            float vals[16];
            float kamax = 0.0f;
            float vamax = 0.0f;
            for (int i = 0; i < kLaneValues; ++i) {
                const int d = lane + i * kSubgroup;
                float key = d_k_proj[base + d] * inv_rms *
                            (1.0f + tq_xpu_bf16_to_float(d_k_norm[d]));
                if (d < rotary_dim) {
                    const int pair_d = d < rotary_half ? d + rotary_half
                                                       : d - rotary_half;
                    const int rope_index =
                        d < rotary_half ? d : d - rotary_half;
                    const float frequency = sycl::pow(
                        rope_theta, -(2.0f * static_cast<float>(rope_index) /
                                      static_cast<float>(rotary_dim)));
                    const float angle = static_cast<float>(pos) * frequency;
                    const float pair = d_k_proj[base + pair_d] * inv_rms *
                        (1.0f + tq_xpu_bf16_to_float(d_k_norm[pair_d]));
                    key = key * sycl::cos(angle) +
                          (d < rotary_half ? -pair : pair) * sycl::sin(angle);
                }
                keys[i] = key;
                kamax = sycl::fmax(kamax, sycl::fabs(key));
                vals[i] = d_v_proj[base + d];
                vamax = sycl::fmax(vamax, sycl::fabs(vals[i]));
            }
            kamax = sycl::reduce_over_group(sg, kamax, sycl::maximum<float>());
            vamax = sycl::reduce_over_group(sg, vamax, sycl::maximum<float>());
            const float ks = (kamax > 0.0f) ? (kamax / 448.0f) : 1.0f;
            const float vs = (vamax > 0.0f) ? (vamax / 448.0f) : 1.0f;
            const size_t row = tq_kv_group_physical_row<Paged>(
                pos, kv_head, nkv, layout, sg, lane);
            if (lane == 0) {
                d_k_scale[row] = sycl::bit_cast<uint16_t>(sycl::half(ks));
                d_v_scale[row] = sycl::bit_cast<uint16_t>(sycl::half(vs));
            }
            const float ksr = (float)sycl::bit_cast<sycl::half>(
                sycl::bit_cast<uint16_t>(sycl::half(ks)));
            const float vsr = (float)sycl::bit_cast<sycl::half>(
                sycl::bit_cast<uint16_t>(sycl::half(vs)));
            for (int i = 0; i < kLaneValues; ++i) {
                const size_t ci = row * hd + lane + i * kSubgroup;
                d_k_cache[ci] = tq_xpu_e4m3_from_float(keys[i] / ksr);
                d_v_cache[ci] = tq_xpu_e4m3_from_float(vals[i] / vsr);
            }
        });
}

void x_prefill_kv_write(const float *d_k_proj, const float *d_v_proj,
                        const uint16_t *d_k_norm, uint8_t *d_k_cache,
                        uint8_t *d_v_cache, uint16_t *d_k_scale,
                        uint16_t *d_v_scale, int pos0, int T, int nkv, int hd,
                        float eps, float rope_theta,
                        float partial_rotary_factor, tq_kv_layout_t layout) {
    if (layout.block_table) {
        x_prefill_kv_write_impl<true>(
            d_k_proj, d_v_proj, d_k_norm, d_k_cache, d_v_cache, d_k_scale,
            d_v_scale, pos0, T, nkv, hd, eps, rope_theta,
            partial_rotary_factor, layout);
    } else {
        x_prefill_kv_write_impl<false>(
            d_k_proj, d_v_proj, d_k_norm, d_k_cache, d_v_cache, d_k_scale,
            d_v_scale, pos0, T, nkv, hd, eps, rope_theta,
            partial_rotary_factor, layout);
    }
}

template <bool Paged>
void x_prefill_attn_impl(float *d_out, const float *d_qg_proj,
                         const uint16_t *d_q_norm,
                         const uint8_t *d_k_cache,
                         const uint8_t *d_v_cache,
                         const uint16_t *d_k_scale,
                         const uint16_t *d_v_scale, int pos0, int T, int nh,
                         int nkv, int hd, float eps, float rope_theta,
                         float partial_rotary_factor,
                         tq_kv_layout_t layout) {
    constexpr int kQRows = 16;
    constexpr int kSubgroup = 16;
    constexpr int kLaneValues = 16;  // hd / kSubgroup, hd == 256
    constexpr int kKChunk = 64;      // 2 x 64 x 256 bytes of SLM = 33 KB
    const int rotary_dim = static_cast<int>(static_cast<float>(hd) *
                                            partial_rotary_factor);
    const int rotary_half = rotary_dim / 2;
    const int gqa = nh / nkv;
    const int tiles = (T + kQRows - 1) / kQRows;
    const int qkv_w = 2 * nh * hd;
    const int attn_w = nh * hd;
    const size_t local_size = kQRows * kSubgroup;
    tq_q().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> kbuf(kKChunk * hd, cgh);
        sycl::local_accessor<uint8_t, 1> vbuf(kKChunk * hd, cgh);
        sycl::local_accessor<float, 1> ksc(kKChunk, cgh);
        sycl::local_accessor<float, 1> vsc(kKChunk, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(nh) * tiles * local_size,
                              local_size),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(kSubgroup)]] {
                const int wg = static_cast<int>(item.get_group_linear_id());
                const int head = wg / tiles;
                const int tile = wg % tiles;
                const int kv_head = head / gqa;
                const sycl::sub_group sg = item.get_sub_group();
                const int row = static_cast<int>(sg.get_group_linear_id());
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int local = static_cast<int>(item.get_local_linear_id());
                const int t = tile * kQRows + row;
                // A tail tile can overhang T. Dead rows still execute every
                // barrier below; they simply contribute nothing.
                const bool live = (t < T);
                const int my_pos = pos0 + t;
                const size_t qg_base =
                    static_cast<size_t>(live ? t : 0) * qkv_w +
                    static_cast<size_t>(head) * 2 * hd;

                // Q prologue: RMSNorm with (1+w), then partial rotate-half
                // RoPE. pair_d is read from GLOBAL exactly as the per-token
                // kernel does, so the lane->dim map is unconstrained here.
                float q[kLaneValues];
                float sum_sq = 0.0f;
                for (int i = 0; i < kLaneValues; ++i) {
                    const float v = d_qg_proj[qg_base + lane * kLaneValues + i];
                    sum_sq += v * v;
                }
                sum_sq = sycl::reduce_over_group(sg, sum_sq,
                                                 sycl::plus<float>());
                const float inv_rms =
                    sycl::rsqrt(sum_sq / static_cast<float>(hd) + eps);
                for (int i = 0; i < kLaneValues; ++i) {
                    const int d = lane * kLaneValues + i;
                    float qv = d_qg_proj[qg_base + d] * inv_rms *
                               (1.0f + tq_xpu_bf16_to_float(d_q_norm[d]));
                    if (d < rotary_dim) {
                        const int pair_d = d < rotary_half ? d + rotary_half
                                                           : d - rotary_half;
                        const int rope_index =
                            d < rotary_half ? d : d - rotary_half;
                        const float frequency = sycl::pow(
                            rope_theta,
                            -(2.0f * static_cast<float>(rope_index) /
                              static_cast<float>(rotary_dim)));
                        const float angle =
                            static_cast<float>(my_pos) * frequency;
                        const float pair =
                            d_qg_proj[qg_base + pair_d] * inv_rms *
                            (1.0f + tq_xpu_bf16_to_float(d_q_norm[pair_d]));
                        qv = qv * sycl::cos(angle) +
                             (d < rotary_half ? -pair : pair) *
                                 sycl::sin(angle);
                    }
                    q[i] = qv;
                }

                float acc[kLaneValues];
                for (int i = 0; i < kLaneValues; ++i) acc[i] = 0.0f;
                float run_max = -3.402823466e38f;
                float run_den = 0.0f;
                const float inv_sqrt_hd =
                    sycl::rsqrt(static_cast<float>(hd));

                // Loop bound is UNIFORM across the work-group (the tile's last
                // row), so every subgroup reaches the same barriers. Per-row
                // causality is the inner break, never an outer one.
                const int tile_end = pos0 + tile * kQRows + kQRows;
                const int written_end = pos0 + T;
                const int total =
                    tile_end < written_end ? tile_end : written_end;
                for (int c0 = 0; c0 < total; c0 += kKChunk) {
                    int clen = total - c0;
                    if (clen > kKChunk) clen = kKChunk;
                    size_t stage_token0;
                    if constexpr (Paged) {
                        stage_token0 = local == 0
                            ? tq_kv_physical_token<true>(c0, layout)
                            : 0;
                        stage_token0 = sycl::group_broadcast(
                            item.get_group(), stage_token0, 0);
                    } else {
                        stage_token0 = static_cast<size_t>(c0);
                    }
                    const size_t stage_row0 =
                        stage_token0 * static_cast<size_t>(nkv) +
                        static_cast<size_t>(kv_head);
                    for (int e = local; e < clen * hd;
                         e += static_cast<int>(local_size)) {
                        const int j = e / hd;
                        const int d = e - j * hd;
                        const size_t crow =
                            (stage_row0 + static_cast<size_t>(j) * nkv) * hd;
                        kbuf[e] = d_k_cache[crow + d];
                        vbuf[e] = d_v_cache[crow + d];
                    }
                    for (int j = local; j < clen;
                         j += static_cast<int>(local_size)) {
                        const size_t sr =
                            stage_row0 + static_cast<size_t>(j) * nkv;
                        ksc[j] = (float)sycl::bit_cast<sycl::half>(
                            d_k_scale[sr]);
                        vsc[j] = (float)sycl::bit_cast<sycl::half>(
                            d_v_scale[sr]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    if (live) {
                        for (int j = 0; j < clen; ++j) {
                            if (c0 + j > my_pos) break;
                            // Row scale is linear, so it multiplies the dot
                            // once rather than each of the 16 codes.
                            float dot = 0.0f;
                            for (int i = 0; i < kLaneValues; ++i)
                                dot += q[i] * tq_xpu_e4m3_to_float(
                                    kbuf[j * hd + lane * kLaneValues + i]);
                            dot = sycl::reduce_over_group(sg, dot,
                                                          sycl::plus<float>());
                            const float score = dot * ksc[j] * inv_sqrt_hd;
                            const float next_max = sycl::fmax(run_max, score);
                            const float alpha = sycl::exp(run_max - next_max);
                            const float beta = sycl::exp(score - next_max);
                            run_den = run_den * alpha + beta;
                            run_max = next_max;
                            const float bv = beta * vsc[j];
                            for (int i = 0; i < kLaneValues; ++i)
                                acc[i] = acc[i] * alpha +
                                         bv * tq_xpu_e4m3_to_float(
                                             vbuf[j * hd +
                                                  lane * kLaneValues + i]);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (!live) return;
                for (int i = 0; i < kLaneValues; ++i) {
                    const int d = lane * kLaneValues + i;
                    const float gate = d_qg_proj[qg_base + hd + d];
                    d_out[static_cast<size_t>(t) * attn_w +
                          static_cast<size_t>(head) * hd + d] =
                        (acc[i] / run_den) *
                        (1.0f / (1.0f + sycl::exp(-gate)));
                }
            });
    });
}

void x_prefill_attn(float *d_out, const float *d_qg_proj,
                    const uint16_t *d_q_norm, const uint8_t *d_k_cache,
                    const uint8_t *d_v_cache, const uint16_t *d_k_scale,
                    const uint16_t *d_v_scale, int pos0, int T, int nh,
                    int nkv, int hd, float eps, float rope_theta,
                    float partial_rotary_factor, tq_kv_layout_t layout) {
    g_attn_branch_counts[TQ_ATTN_BRANCH_PREFILL].fetch_add(
        1, std::memory_order_relaxed);
    if (x_prefill_attn_xmx(
            d_out, d_qg_proj, d_q_norm, d_k_cache, d_v_cache, d_k_scale,
            d_v_scale, pos0, T, nh, nkv, hd, eps, rope_theta,
            partial_rotary_factor, layout)) {
        g_attn_branch_counts[TQ_ATTN_BRANCH_PREFILL_XMX].fetch_add(
            1, std::memory_order_relaxed);
        return;
    }
    g_attn_branch_counts[TQ_ATTN_BRANCH_PREFILL_SCALAR].fetch_add(
        1, std::memory_order_relaxed);
    if (layout.block_table) {
        x_prefill_attn_impl<true>(
            d_out, d_qg_proj, d_q_norm, d_k_cache, d_v_cache, d_k_scale,
            d_v_scale, pos0, T, nh, nkv, hd, eps, rope_theta,
            partial_rotary_factor, layout);
    } else {
        x_prefill_attn_impl<false>(
            d_out, d_qg_proj, d_q_norm, d_k_cache, d_v_cache, d_k_scale,
            d_v_scale, pos0, T, nh, nkv, hd, eps, rope_theta,
            partial_rotary_factor, layout);
    }
}

void x_prefill_attn_packed(const tq_prefill_attn_request_t *requests, int n,
                          const uint16_t *qnorm, int nh, int nkv, int hd,
                          float eps, float rope_theta, float partial_rotary_factor) {
    if (x_prefill_attn_xmx_packed(requests, n, qnorm, nh, nkv, hd, eps,
                                 rope_theta, partial_rotary_factor)) {
        // One device launch; counters continue to describe request segments,
        // so their meaning matches calls through the single-request entrypoint.
        g_attn_branch_counts[TQ_ATTN_BRANCH_PREFILL].fetch_add(
            n, std::memory_order_relaxed);
        g_attn_branch_counts[TQ_ATTN_BRANCH_PREFILL_XMX].fetch_add(
            n, std::memory_order_relaxed);
        return;
    }
    for (int i = 0; i < n; ++i) {
        const auto &r = requests[i];
        x_prefill_attn(r.out, r.qg, qnorm, r.kc, r.vc, r.ks, r.vs,
                       r.pos0, r.T, nh, nkv, hd, eps, rope_theta,
                       partial_rotary_factor, r.layout);
    }
}

template <bool Paged>
void x_full_attn_decode_impl(
    float *d_out, const float *d_qg_proj, const float *d_k_proj,
    const float *d_v_proj, const uint16_t *d_q_norm,
    const uint16_t *d_k_norm, uint8_t *d_k_cache, uint8_t *d_v_cache,
    uint16_t *d_k_scale, uint16_t *d_v_scale, float *d_scores, int pos,
    int nh, int nkv, int hd, float eps, float rope_theta,
    float partial_rotary_factor, int sstride, tq_kv_layout_t layout) {
    const size_t local_size = static_cast<size_t>(hd);

    // CUDA line 12145 hardcodes 64 for the shipping hd=256, prf=0.25 model.
    // Use the metadata-equivalent partial dimension required by the XPU ABI.
    const int rotary_dim = static_cast<int>(static_cast<float>(hd) * partial_rotary_factor);
    const int rotary_half = rotary_dim / 2;

    // Shipping hd=256 fast path: one SIMD16 subgroup owns a head. Each lane
    // keeps 16 Q/output elements in registers, so the online softmax needs no
    // work-group barriers or local memory. The former 256-thread path reduced
    // across 16 subgroups and rendezvoused all 256 threads once per token.
    const char *simd16 = std::getenv("TQ_XPU_ATTN_SIMD16");
    if (hd == 256 && (!simd16 || simd16[0] != '0')) {
        constexpr int kHeadDim = 256;
        constexpr int kSubgroup = 16;
        constexpr int kLaneValues = kHeadDim / kSubgroup;

        tq_q().parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(nkv) * kSubgroup, kSubgroup),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(kSubgroup)]] {
                const int kv_head = static_cast<int>(item.get_group_linear_id());
                const sycl::sub_group sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const size_t base = static_cast<size_t>(kv_head) * kHeadDim;

                float sum_sq = 0.0f;
                for (int i = 0; i < kLaneValues; ++i) {
                    const float value = d_k_proj[base + lane + i * kSubgroup];
                    sum_sq += value * value;
                }
                sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
                const float inv_rms =
                    sycl::rsqrt(sum_sq / static_cast<float>(kHeadDim) + eps);

                // E4M3 needs the row absmax before it can encode, so build the
                // RoPE'd key row in registers first, reduce across the
                // subgroup (16 lanes x 16 values = the whole hd=256 row), then
                // encode against that scale.
                float keys[kLaneValues];
                float kamax = 0.0f;
                float vals[kLaneValues];
                float vamax = 0.0f;
                for (int i = 0; i < kLaneValues; ++i) {
                    const int d = lane + i * kSubgroup;
                    float key = d_k_proj[base + d] * inv_rms *
                                (1.0f + tq_xpu_bf16_to_float(d_k_norm[d]));
                    if (d < rotary_dim) {
                        const int pair_d = d < rotary_half ? d + rotary_half
                                                           : d - rotary_half;
                        const int rope_index = d < rotary_half ? d : d - rotary_half;
                        const float frequency = sycl::pow(
                            rope_theta,
                            -(2.0f * static_cast<float>(rope_index) /
                              static_cast<float>(rotary_dim)));
                        const float angle = static_cast<float>(pos) * frequency;
                        const float pair = d_k_proj[base + pair_d] * inv_rms *
                            (1.0f + tq_xpu_bf16_to_float(d_k_norm[pair_d]));
                        key = key * sycl::cos(angle) +
                              (d < rotary_half ? -pair : pair) * sycl::sin(angle);
                    }
                    keys[i] = key;
                    kamax = sycl::fmax(kamax, sycl::fabs(key));
                    vals[i] = d_v_proj[base + d];
                    vamax = sycl::fmax(vamax, sycl::fabs(vals[i]));
                }
                kamax = sycl::reduce_over_group(sg, kamax, sycl::maximum<float>());
                vamax = sycl::reduce_over_group(sg, vamax, sycl::maximum<float>());
                const float ks = (kamax > 0.0f) ? (kamax / 448.0f) : 1.0f;
                const float vs = (vamax > 0.0f) ? (vamax / 448.0f) : 1.0f;
                const size_t row = tq_kv_group_physical_row<Paged>(
                    pos, kv_head, nkv, layout, sg, lane);
                if (lane == 0) {
                    d_k_scale[row] = sycl::bit_cast<uint16_t>(sycl::half(ks));
                    d_v_scale[row] = sycl::bit_cast<uint16_t>(sycl::half(vs));
                }
                const float ksr = (float)sycl::bit_cast<sycl::half>(
                    sycl::bit_cast<uint16_t>(sycl::half(ks)));
                const float vsr = (float)sycl::bit_cast<sycl::half>(
                    sycl::bit_cast<uint16_t>(sycl::half(vs)));
                for (int i = 0; i < kLaneValues; ++i) {
                    const size_t ci = row * kHeadDim + lane + i * kSubgroup;
                    d_k_cache[ci] = tq_xpu_e4m3_from_float(keys[i] / ksr);
                    d_v_cache[ci] = tq_xpu_e4m3_from_float(vals[i] / vsr);
                }
            });

        // DPAS-matrix path (design 9.5, queue 3b; opt-in TQ_XPU_ATTN_DPAS=1).
        // The whole 6-head GQA group packs as DPAS M-rows: per 16-token tile
        // S[8,16] = Q x K^T via 16 chained dpas.bf.bf.8.8, base-e online
        // softmax over the token axis, P staged bf16 through SLM, O += P x V
        // via 16 dpas with V staged bf16 in SLM. Each K/V row is read ONCE
        // per group (6x less than per-head, 2-3x less than G=3) at GRF128.
        // Emits the same (O, max, den) partials the grouped merge consumes,
        // with a runtime segment count. Formulation certified end to end by
        // xpu/probe/dpas_attn_probe.cpp arm C (cos 0.999989 vs fp64 ref).
        const char *dpas_env = std::getenv("TQ_XPU_ATTN_DPAS");
        if (dpas_env && dpas_env[0] == '1' && pos >= 512 &&
            nh / nkv == 6 && (nh % nkv) == 0 && kHeadDim == 256) {
            g_attn_branch_counts[TQ_ATTN_BRANCH_SIMD16_DPAS].fetch_add(
                1, std::memory_order_relaxed);
            const int gqa6 = 6;
            const int ntiles = (pos + 1 + kSubgroup - 1) / kSubgroup;
            const int nseg = ntiles < 32 ? ntiles : 32;
            const size_t part_stride = static_cast<size_t>(kHeadDim) + 2;
            ensure_attn_partials(static_cast<size_t>(nh) * nseg * part_stride);
            ensure_attn_qfrag(static_cast<size_t>(nkv) * 16 * 64);
            float *part = g_attn_part;
            uint32_t *qfrag = g_attn_qfrag;

            // Q prep: RMSNorm + q-norm + RoPE per head, packed as bf16
            // A-fragments (chunk c = dims [16c,16c+16), 8 rows x 32 B).
            tq_q().parallel_for(
                sycl::nd_range<1>(static_cast<size_t>(nh) * kSubgroup,
                                  kSubgroup),
                [=](sycl::nd_item<1> item)
                    [[sycl::reqd_sub_group_size(kSubgroup)]] {
                    const int head = static_cast<int>(item.get_group_linear_id());
                    const int lane = static_cast<int>(
                        item.get_sub_group().get_local_linear_id());
                    const sycl::sub_group sg = item.get_sub_group();
                    const size_t qg_base =
                        static_cast<size_t>(head) * 2 * kHeadDim;
                    float sum_sq = 0.0f;
                    for (int i = 0; i < kLaneValues; ++i) {
                        const float v =
                            d_qg_proj[qg_base + lane + i * kSubgroup];
                        sum_sq += v * v;
                    }
                    sum_sq = sycl::reduce_over_group(sg, sum_sq,
                                                     sycl::plus<float>());
                    const float inv_rms = sycl::rsqrt(
                        sum_sq / static_cast<float>(kHeadDim) + eps);
                    const int kv = head / gqa6;
                    const int row = head - kv * gqa6;   // 0..5 (M row)
                    for (int i = 0; i < kLaneValues; ++i) {
                        const int d = lane + i * kSubgroup;
                        float q = d_qg_proj[qg_base + d] * inv_rms *
                            (1.0f + tq_xpu_bf16_to_float(d_q_norm[d]));
                        if (d < rotary_dim) {
                            const int pair_d = d < rotary_half
                                ? d + rotary_half : d - rotary_half;
                            const int rope_index =
                                d < rotary_half ? d : d - rotary_half;
                            const float frequency = sycl::pow(
                                rope_theta,
                                -(2.0f * static_cast<float>(rope_index) /
                                  static_cast<float>(rotary_dim)));
                            const float angle =
                                static_cast<float>(pos) * frequency;
                            const float pair =
                                d_qg_proj[qg_base + pair_d] * inv_rms *
                                (1.0f + tq_xpu_bf16_to_float(d_q_norm[pair_d]));
                            q = q * sycl::cos(angle) +
                                (d < rotary_half ? -pair : pair) *
                                    sycl::sin(angle);
                        }
                        // chunk = d/16, half index = d%16; flat byte inside
                        // the 256-byte fragment = row*32 + 2*(d%16).
                        const int chunk = d >> 4;
                        const uint32_t bf =
                            sycl::bit_cast<uint32_t>(q) >> 16;
                        uint16_t *frag16 = reinterpret_cast<uint16_t *>(
                            qfrag + ((size_t)kv * 16 + chunk) * 64);
                        frag16[row * 16 + (d & 15)] = (uint16_t)bf;
                        if (row == 0 && head == kv * gqa6) {
                            // zero the two dead rows once per group
                            frag16[6 * 16 + (d & 15)] = 0;
                            frag16[7 * 16 + (d & 15)] = 0;
                        }
                    }
                });

            // Shard: one subgroup per (kv_head, segment).
            tq_q().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<float, 1> o_slm(
                    sycl::range<1>((size_t)gqa6 * kHeadDim), cgh);
                sycl::local_accessor<uint16_t, 1> v_slm(
                    sycl::range<1>((size_t)kSubgroup * kHeadDim), cgh);
                sycl::local_accessor<uint16_t, 1> p_slm(
                    sycl::range<1>(8 * kSubgroup), cgh);
                cgh.parallel_for(
                    sycl::nd_range<1>(
                        static_cast<size_t>(nkv) * nseg * kSubgroup,
                        kSubgroup),
                    [=](sycl::nd_item<1> item)
                        [[sycl::reqd_sub_group_size(kSubgroup)]] {
                        const int wg =
                            static_cast<int>(item.get_group_linear_id());
                        const int kv = wg / nseg;
                        const int seg = wg - kv * nseg;
                        const sycl::sub_group sg = item.get_sub_group();
                        const int lane = static_cast<int>(
                            sg.get_local_linear_id());
                        const int apair = lane >> 3;
                        const int abyte = (lane & 7) * 4;
                        using uintv4x = uint32_t
                            __attribute__((ext_vector_type(4)));
                        using uintv8x = uint32_t
                            __attribute__((ext_vector_type(8)));
                        using floatv8x = float
                            __attribute__((ext_vector_type(8)));
                        uintv4x afr[16];
                        for (int c = 0; c < 16; ++c) {
                            const uint8_t *ab = reinterpret_cast<const uint8_t *>(
                                qfrag + ((size_t)kv * 16 + c) * 64);
                            for (int i = 0; i < 4; ++i)
                                afr[c][i] = *reinterpret_cast<const uint32_t *>(
                                    ab + (size_t)(2 * i + apair) * 32 + abyte);
                        }
                        for (int i = lane; i < gqa6 * kHeadDim; i += kSubgroup)
                            o_slm[i] = 0.0f;
                        float run_max[6], run_den[6];
                        for (int h = 0; h < gqa6; ++h) {
                            run_max[h] = -3.402823466e38f;
                            run_den[h] = 0.0f;
                        }
                        const float inv_sqrt_hd =
                            sycl::rsqrt(static_cast<float>(kHeadDim));
                        const int ntl =
                            (pos + 1 + kSubgroup - 1) / kSubgroup;
                        for (int tile = seg; tile < ntl; tile += nseg) {
                            const int tile_token0 = tile * kSubgroup;
                            const int token = tile_token0 + lane;
                            const bool live = token <= pos;
                            const int logical_token = live ? token : pos;
                            size_t physical_token0;
                            if constexpr (Paged) {
                                physical_token0 = lane == 0
                                    ? tq_kv_physical_token<true>(
                                          tile_token0, layout)
                                    : 0;
                                physical_token0 = sycl::group_broadcast(
                                    sg, physical_token0, 0);
                            } else {
                                physical_token0 =
                                    static_cast<size_t>(tile_token0);
                            }
                            const size_t physical_token =
                                physical_token0 +
                                static_cast<size_t>(
                                    logical_token - tile_token0);
                            const size_t krow =
                                physical_token * static_cast<size_t>(nkv) +
                                static_cast<size_t>(kv);
                            const size_t cbase = krow * kHeadDim;
                            const float ksc = (float)sycl::bit_cast<sycl::half>(
                                d_k_scale[krow]);
                            floatv8x d;
                            for (int m = 0; m < 8; ++m) d[m] = 0.0f;
                            for (int c = 0; c < 16; ++c) {
                                uintv8x bfr;
                                const uint8_t *ks = d_k_cache + cbase + c * 16;
                                for (int p = 0; p < 8; ++p) {
                                    const uint32_t c0 = ks[2 * p];
                                    const uint32_t c1 = ks[2 * p + 1];
                                    const uint32_t f0 = sycl::bit_cast<uint32_t>(
                                        sycl::bit_cast<float>(
                                            ((c0 & 0x7Fu) << 20) |
                                            ((c0 & 0x80u) << 24)) * 0x1p120f);
                                    const uint32_t f1 = sycl::bit_cast<uint32_t>(
                                        sycl::bit_cast<float>(
                                            ((c1 & 0x7Fu) << 20) |
                                            ((c1 & 0x80u) << 24)) * 0x1p120f);
                                    bfr[p] = (f0 >> 16) | (f1 & 0xFFFF0000u);
                                }
                                DPAS_BF_RC8_SEQ(d, afr[c], bfr);
                            }
                            // stage this lane's V row (scaled) as bf16
                            const float vsc = (float)sycl::bit_cast<sycl::half>(
                                d_v_scale[krow]);
                            const uint8_t *vr = d_v_cache + cbase;
                            for (int dd = 0; dd < kHeadDim; ++dd) {
                                const uint32_t code = vr[dd];
                                const float fv = live
                                    ? sycl::bit_cast<float>(
                                          ((code & 0x7Fu) << 20) |
                                          ((code & 0x80u) << 24)) *
                                          0x1p120f * vsc
                                    : 0.0f;
                                v_slm[(size_t)lane * kHeadDim + dd] =
                                    (uint16_t)(sycl::bit_cast<uint32_t>(fv) >>
                                               16);
                            }
                            float resc[6];
                            for (int h = 0; h < gqa6; ++h) {
                                const float score = live
                                    ? d[h] * ksc * inv_sqrt_hd
                                    : -3.402823466e38f;
                                const float tmax = sycl::reduce_over_group(
                                    sg, score, sycl::maximum<float>());
                                const float nmax =
                                    sycl::fmax(run_max[h], tmax);
                                resc[h] = sycl::exp(run_max[h] - nmax);
                                const float p = live
                                    ? sycl::exp(score - nmax) : 0.0f;
                                run_den[h] = run_den[h] * resc[h] +
                                    sycl::reduce_over_group(
                                        sg, p, sycl::plus<float>());
                                run_max[h] = nmax;
                                p_slm[(size_t)h * kSubgroup + lane] =
                                    (uint16_t)(sycl::bit_cast<uint32_t>(p) >>
                                               16);
                            }
                            for (int h = gqa6; h < 8; ++h)
                                p_slm[(size_t)h * kSubgroup + lane] = 0;
                            item.barrier(
                                sycl::access::fence_space::local_space);
                            uintv4x pfr;
                            {
                                const uint8_t *pb =
                                    reinterpret_cast<const uint8_t *>(
                                        &p_slm[0]);
                                for (int i = 0; i < 4; ++i)
                                    pfr[i] = *reinterpret_cast<const uint32_t *>(
                                        pb + (size_t)(2 * i + apair) * 32 +
                                        abyte);
                            }
                            for (int c = 0; c < 16; ++c) {
                                uintv8x vfr;
                                for (int p = 0; p < 8; ++p) {
                                    const uint32_t lo =
                                        v_slm[(size_t)(2 * p) * kHeadDim +
                                              c * 16 + lane];
                                    const uint32_t hi =
                                        v_slm[(size_t)(2 * p + 1) * kHeadDim +
                                              c * 16 + lane];
                                    vfr[p] = lo | (hi << 16);
                                }
                                floatv8x od;
                                for (int m = 0; m < 8; ++m) od[m] = 0.0f;
                                DPAS_BF_RC8_SEQ(od, pfr, vfr);
                                for (int h = 0; h < gqa6; ++h) {
                                    const size_t oi =
                                        (size_t)h * kHeadDim + c * 16 + lane;
                                    o_slm[oi] = o_slm[oi] * resc[h] + od[h];
                                }
                            }
                            item.barrier(
                                sycl::access::fence_space::local_space);
                        }
                        for (int h = 0; h < gqa6; ++h) {
                            float *dst = part +
                                (static_cast<size_t>(kv * gqa6 + h) * nseg +
                                 seg) * part_stride;
                            for (int dd = lane; dd < kHeadDim; dd += kSubgroup)
                                dst[dd] = o_slm[(size_t)h * kHeadDim + dd];
                            if (lane == 0) {
                                dst[kHeadDim] = run_max[h];
                                dst[kHeadDim + 1] = run_den[h];
                            }
                        }
                    });
            });

            // Cross-segment merge + sigmoid gate (runtime segment count).
            tq_q().parallel_for(
                sycl::range<1>(static_cast<size_t>(nh) * kHeadDim),
                [=](sycl::id<1> id) {
                    const int idx = static_cast<int>(id[0]);
                    const int head = idx / kHeadDim;
                    const int d = idx % kHeadDim;
                    const float *base = part +
                        static_cast<size_t>(head) * nseg * part_stride;
                    float gmax = -3.402823466e38f;
                    for (int s = 0; s < nseg; ++s)
                        gmax = sycl::fmax(gmax,
                                          base[s * part_stride + kHeadDim]);
                    float num = 0.0f;
                    float den = 0.0f;
                    for (int s = 0; s < nseg; ++s) {
                        const float *seg_p = base + s * part_stride;
                        const float w = sycl::exp(seg_p[kHeadDim] - gmax);
                        num += seg_p[d] * w;
                        den += seg_p[kHeadDim + 1] * w;
                    }
                    const float gate = d_qg_proj[
                        static_cast<size_t>(head) * 2 * kHeadDim + kHeadDim +
                        d];
                    d_out[static_cast<size_t>(head) * kHeadDim + d] =
                        (num / den) / (1.0f + sycl::exp(-gate));
                });
            return;
        }

        // GQA-grouped path. One work-group owns G consecutive q-heads that
        // SHARE a kv-head, so each K/V row is fetched once and used G times
        // instead of once per head. At nh=24/nkv=4 the per-head kernel below
        // re-reads every row 6x; that redundancy is the whole cost at depth
        // (measured 2.788 ms at pos=2048, which is 96% of a roofline that is
        // itself 6x too large). G=3 keeps ~112 floats/lane live, inside the
        // 128-GRF budget; G=6 needs ~208 and spills.
        //
        // Token axis is split into kSegments so the grid stays at
        // (nh/G)*kSegments work-groups - the same count as the per-head kernel,
        // so this is a pure traffic win with unchanged occupancy. Each
        // work-group's 16 subgroups merge through SLM; a small second kernel
        // merges the segments.
        // Measured crossover: grouping saves 1.28 ms/step at pos=2048 and
        // 2.81 ms at pos=4096, but COSTS 0.09 ms at pos=256 because the
        // cross-segment merge kernel outweighs the traffic saved on a short
        // history. Break-even lands near pos 375, so gate at 512 and let
        // shallow contexts fall through to the per-head shard kernel.
        const char *grouped_env = std::getenv("TQ_XPU_ATTN_GROUPED");
        const int gqa_group = nh / nkv;
        // TQ_ATTN_G: heads sharing one work-group's K/V fetch. 3 fits the
        // 128-register mode; 6 (the full GQA group) needs GRF256 — build the
        // experimental variant with -DTQ_ATTN_G=6 (see level-up queue item 3).
        constexpr int kHeadsPerWg = TQ_ATTN_G;
        // Segments scale with G so the WG count (and the loads in flight on
        // the latency-bound KV walk) stays at 24: G=6 at kSegments=3 measured
        // 27.6/24.1 tok/s at 2k/4k vs G=3's 31.6/30.3 — traffic halved but so
        // did memory-level parallelism. Same grid, half the passes, is the
        // shape that can win.
        constexpr int kSegments = (kHeadsPerWg >= 6) ? 6 : 3;
        constexpr int kGroupedMinPos = 512;
        if (pos >= kGroupedMinPos && (!grouped_env || grouped_env[0] != '0') &&
            (gqa_group % kHeadsPerWg) == 0 && (nh % kHeadsPerWg) == 0) {
            g_attn_branch_counts[TQ_ATTN_BRANCH_SIMD16_GROUPED].fetch_add(
                1, std::memory_order_relaxed);
            constexpr int kShards = kHeadDim / kSubgroup;   // 16 subgroups
            constexpr int kWorkgroup = kHeadDim;            // 256 threads
            const int triplets = nh / kHeadsPerWg;
            const int stride_tok = kSegments * kShards;
            const size_t part_stride = static_cast<size_t>(kHeadDim) + 2;
            ensure_attn_partials(static_cast<size_t>(nh) * kSegments * part_stride);
            float *part = g_attn_part;

            tq_q().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<float, 1> smax(kShards * kHeadsPerWg, cgh);
                sycl::local_accessor<float, 1> sden(kShards * kHeadsPerWg, cgh);
                sycl::local_accessor<float, 1> sacc(kShards * kHeadDim, cgh);
                auto kern =
                    [=](sycl::nd_item<1> item)
                        [[sycl::reqd_sub_group_size(kSubgroup)]] {
                        const int wg = static_cast<int>(item.get_group_linear_id());
                        const int triplet = wg / kSegments;
                        const int seg = wg % kSegments;
                        const int local = static_cast<int>(item.get_local_linear_id());
                        const sycl::sub_group sg = item.get_sub_group();
                        const int shard = static_cast<int>(sg.get_group_linear_id());
                        const int lane = static_cast<int>(sg.get_local_linear_id());
                        const int head0 = triplet * kHeadsPerWg;
                        const int kv_head = head0 / gqa_group;

                        // Slot t maps to dimension d = 64*(t/4) + 4*lane + t%4.
                        // That is forced by wanting a COALESCED byte load: with
                        // one E4M3 byte per element, 16 lanes must each take 4
                        // consecutive codes for a load to move a full 64-byte
                        // line. The natural d = lane + 16*i mapping moved only
                        // 16 bytes per load, which made the fp8 cache SLOWER
                        // than fp32 despite reading a quarter of the bytes
                        // (28.80 vs 30.34 tok/s at 4k). The cache layout itself
                        // is untouched - only this in-kernel indexing changes.
                        auto slot_dim = [&](int t) {
                            return (t >> 2) * 64 + lane * 4 + (t & 3);
                        };
                        float query[kHeadsPerWg][kLaneValues];
                        for (int h = 0; h < kHeadsPerWg; ++h) {
                            const size_t qg_base =
                                static_cast<size_t>(head0 + h) * 2 * kHeadDim;
                            float sum_sq = 0.0f;
                            for (int i = 0; i < kLaneValues; ++i) {
                                const float v = d_qg_proj[qg_base + slot_dim(i)];
                                sum_sq += v * v;
                            }
                            sum_sq = sycl::reduce_over_group(
                                sg, sum_sq, sycl::plus<float>());
                            const float inv_rms = sycl::rsqrt(
                                sum_sq / static_cast<float>(kHeadDim) + eps);
                            for (int i = 0; i < kLaneValues; ++i) {
                                const int d = slot_dim(i);
                                float q = d_qg_proj[qg_base + d] * inv_rms *
                                    (1.0f + tq_xpu_bf16_to_float(d_q_norm[d]));
                                if (d < rotary_dim) {
                                    const int pair_d = d < rotary_half
                                        ? d + rotary_half : d - rotary_half;
                                    const int rope_index =
                                        d < rotary_half ? d : d - rotary_half;
                                    const float frequency = sycl::pow(
                                        rope_theta,
                                        -(2.0f * static_cast<float>(rope_index) /
                                          static_cast<float>(rotary_dim)));
                                    const float angle =
                                        static_cast<float>(pos) * frequency;
                                    const float pair =
                                        d_qg_proj[qg_base + pair_d] * inv_rms *
                                        (1.0f + tq_xpu_bf16_to_float(
                                            d_q_norm[pair_d]));
                                    q = q * sycl::cos(angle) +
                                        (d < rotary_half ? -pair : pair) *
                                            sycl::sin(angle);
                                }
                                query[h][i] = q;
                            }
                        }

                        float run_max[kHeadsPerWg];
                        float run_den[kHeadsPerWg];
                        float numer[kHeadsPerWg][kLaneValues];
                        for (int h = 0; h < kHeadsPerWg; ++h) {
                            run_max[h] = -3.402823466e38f;
                            run_den[h] = 0.0f;
                            for (int i = 0; i < kLaneValues; ++i)
                                numer[h][i] = 0.0f;
                        }
                        const float inv_sqrt_hd =
                            sycl::rsqrt(static_cast<float>(kHeadDim));

                        // One K/V fetch serves all G heads. Each lane pulls 4
                        // consecutive codes as one dword, so the subgroup moves
                        // a full 64-byte line per load: 4 loads per row instead
                        // of 16, and a quarter of fp32's bytes.
                        float row[kLaneValues];
                        for (int token = seg * kShards + shard; token <= pos;
                             token += stride_tok) {
                            const size_t krow =
                                tq_kv_group_physical_row<Paged>(
                                    token, kv_head, nkv, layout, sg, lane);
                            const size_t cache_base = krow * kHeadDim;
                            const float ksc = (float)sycl::bit_cast<sycl::half>(
                                d_k_scale[krow]);
                            for (int j = 0; j < 4; ++j) {
                                const uint32_t w = *reinterpret_cast<const uint32_t *>(
                                    d_k_cache + cache_base + j * 64 + lane * 4);
                                for (int s = 0; s < 4; ++s)
                                    row[j * 4 + s] = tq_xpu_e4m3_to_float(
                                        (uint8_t)((w >> (s * 8)) & 0xFFu)) * ksc;
                            }
                            float alpha[kHeadsPerWg];
                            float beta[kHeadsPerWg];
                            for (int h = 0; h < kHeadsPerWg; ++h) {
                                float lane_dot = 0.0f;
                                for (int i = 0; i < kLaneValues; ++i)
                                    lane_dot += query[h][i] * row[i];
                                const float score = sycl::reduce_over_group(
                                    sg, lane_dot, sycl::plus<float>()) *
                                    inv_sqrt_hd;
                                const float next_max =
                                    sycl::fmax(run_max[h], score);
                                alpha[h] = sycl::exp(run_max[h] - next_max);
                                beta[h] = sycl::exp(score - next_max);
                                run_den[h] = run_den[h] * alpha[h] + beta[h];
                                run_max[h] = next_max;
                            }
                            const float vsc = (float)sycl::bit_cast<sycl::half>(
                                d_v_scale[krow]);
                            for (int j = 0; j < 4; ++j) {
                                const uint32_t w = *reinterpret_cast<const uint32_t *>(
                                    d_v_cache + cache_base + j * 64 + lane * 4);
                                for (int s = 0; s < 4; ++s)
                                    row[j * 4 + s] = tq_xpu_e4m3_to_float(
                                        (uint8_t)((w >> (s * 8)) & 0xFFu)) * vsc;
                            }
                            for (int h = 0; h < kHeadsPerWg; ++h)
                                for (int i = 0; i < kLaneValues; ++i)
                                    numer[h][i] = numer[h][i] * alpha[h] +
                                                  beta[h] * row[i];
                        }

                        if (lane == 0) {
                            for (int h = 0; h < kHeadsPerWg; ++h) {
                                smax[shard * kHeadsPerWg + h] = run_max[h];
                                sden[shard * kHeadsPerWg + h] = run_den[h];
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        // Merge the 16 shards, one head at a time so SLM stays
                        // at kShards*kHeadDim floats (16 KB).
                        for (int h = 0; h < kHeadsPerWg; ++h) {
                            float gmax = -3.402823466e38f;
                            for (int s = 0; s < kShards; ++s)
                                gmax = sycl::fmax(gmax, smax[s * kHeadsPerWg + h]);
                            float gden = 0.0f;
                            for (int s = 0; s < kShards; ++s)
                                gden += sden[s * kHeadsPerWg + h] *
                                    sycl::exp(smax[s * kHeadsPerWg + h] - gmax);
                            // sacc is indexed by DIMENSION, so the slot->dim
                            // map must be applied here; the reduction below
                            // reads it as element `local`.
                            for (int i = 0; i < kLaneValues; ++i)
                                sacc[shard * kHeadDim + slot_dim(i)] = numer[h][i];
                            item.barrier(sycl::access::fence_space::local_space);
                            float sum = 0.0f;
                            for (int s = 0; s < kShards; ++s)
                                sum += sacc[s * kHeadDim + local] *
                                    sycl::exp(smax[s * kHeadsPerWg + h] - gmax);
                            float *dst = part +
                                (static_cast<size_t>(head0 + h) * kSegments + seg) *
                                part_stride;
                            dst[local] = sum;
                            if (local == 0) {
                                dst[kHeadDim] = gmax;
                                dst[kHeadDim + 1] = gden;
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                        }
                    };
                const sycl::nd_range<1> grid(
                    static_cast<size_t>(triplets) * kSegments * kWorkgroup,
                    kWorkgroup);
#if TQ_ATTN_G >= 6
                // ~215 floats/lane live at G=6: needs the 256-register mode
                // (per-kernel, so the GEMVs keep 8 threads/XVE).
                sycl::ext::oneapi::experimental::properties grf_props{
                    sycl::ext::intel::experimental::grf_size<256>};
                cgh.parallel_for(grid, grf_props, kern);
#else
                cgh.parallel_for(grid, kern);
#endif
            });

            // Cross-segment merge plus the model's elementwise sigmoid gate.
            tq_q().parallel_for(
                sycl::range<1>(static_cast<size_t>(nh) * kHeadDim),
                [=](sycl::id<1> id) {
                    const int idx = static_cast<int>(id[0]);
                    const int head = idx / kHeadDim;
                    const int d = idx % kHeadDim;
                    const float *base =
                        part + static_cast<size_t>(head) * kSegments * part_stride;
                    float gmax = -3.402823466e38f;
                    for (int s = 0; s < kSegments; ++s)
                        gmax = sycl::fmax(gmax, base[s * part_stride + kHeadDim]);
                    float num = 0.0f;
                    float den = 0.0f;
                    for (int s = 0; s < kSegments; ++s) {
                        const float *seg_p = base + s * part_stride;
                        const float w = sycl::exp(seg_p[kHeadDim] - gmax);
                        num += seg_p[d] * w;
                        den += seg_p[kHeadDim + 1] * w;
                    }
                    const float gate = d_qg_proj[
                        static_cast<size_t>(head) * 2 * kHeadDim + kHeadDim + d];
                    d_out[static_cast<size_t>(head) * kHeadDim + d] =
                        (num / den) / (1.0f + sycl::exp(-gate));
                });
            return;
        }

        // For established contexts, spread each head across eight token
        // shards. A 256-thread work-group keeps one thread per output
        // dimension; its first eight subgroups compute independent online
        // softmax shards and a single barriered reduction merges them.
        if (pos >= 64) {
            g_attn_branch_counts[TQ_ATTN_BRANCH_SIMD16_SHARDED].fetch_add(
                1, std::memory_order_relaxed);
            constexpr int kTokenShards = 16;
            constexpr int kWorkgroup = kHeadDim;
            tq_q().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<float, 1> shard_max(kTokenShards + 1, cgh);
                sycl::local_accessor<float, 1> shard_den(kTokenShards + 1, cgh);
                sycl::local_accessor<float, 1> shard_out(
                    kTokenShards * kHeadDim, cgh);
                cgh.parallel_for(
                    sycl::nd_range<1>(static_cast<size_t>(nh) * kWorkgroup,
                                      kWorkgroup),
                    [=](sycl::nd_item<1> item)
                        [[sycl::reqd_sub_group_size(kSubgroup)]] {
                        const int head = static_cast<int>(item.get_group_linear_id());
                        const int local = static_cast<int>(item.get_local_linear_id());
                        const sycl::sub_group sg = item.get_sub_group();
                        const int shard = static_cast<int>(sg.get_group_linear_id());
                        const int lane = static_cast<int>(sg.get_local_linear_id());
                        const int kv_head = head / (nh / nkv);
                        const size_t qg_base =
                            static_cast<size_t>(head) * 2 * kHeadDim;

                        float sum_sq = 0.0f;
                        for (int i = 0; i < kLaneValues; ++i) {
                            const float value =
                                d_qg_proj[qg_base + lane + i * kSubgroup];
                            sum_sq += value * value;
                        }
                        sum_sq = sycl::reduce_over_group(
                            sg, sum_sq, sycl::plus<float>());
                        const float inv_rms = sycl::rsqrt(
                            sum_sq / static_cast<float>(kHeadDim) + eps);

                        float query[kLaneValues];
                        float numerator[kLaneValues] = {};
                        for (int i = 0; i < kLaneValues; ++i) {
                            const int d = lane + i * kSubgroup;
                            float q = d_qg_proj[qg_base + d] * inv_rms *
                                (1.0f + tq_xpu_bf16_to_float(d_q_norm[d]));
                            if (d < rotary_dim) {
                                const int pair_d = d < rotary_half
                                    ? d + rotary_half : d - rotary_half;
                                const int rope_index =
                                    d < rotary_half ? d : d - rotary_half;
                                const float frequency = sycl::pow(
                                    rope_theta,
                                    -(2.0f * static_cast<float>(rope_index) /
                                      static_cast<float>(rotary_dim)));
                                const float angle =
                                    static_cast<float>(pos) * frequency;
                                const float pair =
                                    d_qg_proj[qg_base + pair_d] * inv_rms *
                                    (1.0f + tq_xpu_bf16_to_float(
                                        d_q_norm[pair_d]));
                                q = q * sycl::cos(angle) +
                                    (d < rotary_half ? -pair : pair) *
                                        sycl::sin(angle);
                            }
                            query[i] = q;
                        }

                        float running_max = -3.402823466e38f;
                        float denominator = 0.0f;
                        const float inv_sqrt_hd =
                            sycl::rsqrt(static_cast<float>(kHeadDim));
                        if (shard < kTokenShards) {
                            for (int token = shard; token <= pos;
                                 token += kTokenShards) {
                                const size_t krow =
                                    tq_kv_group_physical_row<Paged>(
                                        token, kv_head, nkv, layout, sg, lane);
                                const size_t cache_base = krow * kHeadDim;
                                const float ksc = (float)sycl::bit_cast<sycl::half>(
                                    d_k_scale[krow]);
                                float lane_dot = 0.0f;
                                for (int i = 0; i < kLaneValues; ++i)
                                    lane_dot += query[i] * tq_xpu_e4m3_to_float(
                                        d_k_cache[cache_base + lane + i * kSubgroup]);
                                const float score = sycl::reduce_over_group(
                                    sg, lane_dot, sycl::plus<float>()) * ksc *
                                    inv_sqrt_hd;
                                const float next_max =
                                    sycl::fmax(running_max, score);
                                const float alpha =
                                    sycl::exp(running_max - next_max);
                                const float beta = sycl::exp(score - next_max);
                                denominator = denominator * alpha + beta;
                                running_max = next_max;
                                const float vsc = (float)sycl::bit_cast<sycl::half>(
                                    d_v_scale[krow]);
                                for (int i = 0; i < kLaneValues; ++i)
                                    numerator[i] = numerator[i] * alpha +
                                        beta * vsc * tq_xpu_e4m3_to_float(
                                            d_v_cache[cache_base + lane + i * kSubgroup]);
                            }
                            if (lane == 0) {
                                shard_max[shard] = running_max;
                                shard_den[shard] = denominator;
                            }
                            for (int i = 0; i < kLaneValues; ++i)
                                shard_out[shard * kHeadDim + lane +
                                          i * kSubgroup] = numerator[i];
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        if (local == 0) {
                            float global_max = shard_max[0];
                            for (int s = 1; s < kTokenShards; ++s)
                                global_max = sycl::fmax(global_max, shard_max[s]);
                            float global_den = 0.0f;
                            for (int s = 0; s < kTokenShards; ++s)
                                global_den += shard_den[s] *
                                    sycl::exp(shard_max[s] - global_max);
                            shard_max[kTokenShards] = global_max;
                            shard_den[kTokenShards] = global_den;
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        float attention = 0.0f;
                        for (int s = 0; s < kTokenShards; ++s)
                            attention += shard_out[s * kHeadDim + local] *
                                sycl::exp(shard_max[s] - shard_max[kTokenShards]);
                        const float gate =
                            d_qg_proj[qg_base + kHeadDim + local];
                        d_out[static_cast<size_t>(head) * kHeadDim + local] =
                            (attention / shard_den[kTokenShards]) /
                            (1.0f + sycl::exp(-gate));
                    });
            });
            return;
        }

        g_attn_branch_counts[TQ_ATTN_BRANCH_SIMD16_SHORT].fetch_add(
            1, std::memory_order_relaxed);
        tq_q().parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(nh) * kSubgroup, kSubgroup),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(kSubgroup)]] {
                const int head = static_cast<int>(item.get_group_linear_id());
                const sycl::sub_group sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int kv_head = head / (nh / nkv);
                const size_t qg_base = static_cast<size_t>(head) * 2 * kHeadDim;

                float sum_sq = 0.0f;
                for (int i = 0; i < kLaneValues; ++i) {
                    const float value = d_qg_proj[qg_base + lane + i * kSubgroup];
                    sum_sq += value * value;
                }
                sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
                const float inv_rms =
                    sycl::rsqrt(sum_sq / static_cast<float>(kHeadDim) + eps);

                float query[kLaneValues];
                float attention[kLaneValues] = {};
                for (int i = 0; i < kLaneValues; ++i) {
                    const int d = lane + i * kSubgroup;
                    float q = d_qg_proj[qg_base + d] * inv_rms *
                              (1.0f + tq_xpu_bf16_to_float(d_q_norm[d]));
                    if (d < rotary_dim) {
                        const int pair_d = d < rotary_half ? d + rotary_half
                                                           : d - rotary_half;
                        const int rope_index = d < rotary_half ? d : d - rotary_half;
                        const float frequency = sycl::pow(
                            rope_theta,
                            -(2.0f * static_cast<float>(rope_index) /
                              static_cast<float>(rotary_dim)));
                        const float angle = static_cast<float>(pos) * frequency;
                        const float pair = d_qg_proj[qg_base + pair_d] * inv_rms *
                            (1.0f + tq_xpu_bf16_to_float(d_q_norm[pair_d]));
                        q = q * sycl::cos(angle) +
                            (d < rotary_half ? -pair : pair) * sycl::sin(angle);
                    }
                    query[i] = q;
                }

                float running_max = -3.402823466e38f;
                float denominator = 0.0f;
                const float inv_sqrt_hd = sycl::rsqrt(static_cast<float>(kHeadDim));
                for (int token = 0; token <= pos; ++token) {
                    const size_t krow =
                        tq_kv_group_physical_row<Paged>(
                            token, kv_head, nkv, layout, sg, lane);
                    const size_t cache_base = krow * kHeadDim;
                    const float ksc =
                        (float)sycl::bit_cast<sycl::half>(d_k_scale[krow]);
                    float lane_dot = 0.0f;
                    for (int i = 0; i < kLaneValues; ++i)
                        lane_dot += query[i] * tq_xpu_e4m3_to_float(
                            d_k_cache[cache_base + lane + i * kSubgroup]);
                    const float dot =
                        sycl::reduce_over_group(sg, lane_dot, sycl::plus<float>());
                    const float score = dot * ksc * inv_sqrt_hd;
                    const float next_max = sycl::fmax(running_max, score);
                    const float alpha = sycl::exp(running_max - next_max);
                    const float beta = sycl::exp(score - next_max);
                    denominator = denominator * alpha + beta;
                    running_max = next_max;
                    const float vsc =
                        (float)sycl::bit_cast<sycl::half>(d_v_scale[krow]);
                    for (int i = 0; i < kLaneValues; ++i)
                        attention[i] = attention[i] * alpha +
                            beta * vsc * tq_xpu_e4m3_to_float(
                                d_v_cache[cache_base + lane + i * kSubgroup]);
                }

                for (int i = 0; i < kLaneValues; ++i) {
                    const int d = lane + i * kSubgroup;
                    const float gate = d_qg_proj[qg_base + kHeadDim + d];
                    d_out[static_cast<size_t>(head) * kHeadDim + d] =
                        (attention[i] / denominator) /
                        (1.0f + sycl::exp(-gate));
                }
            });
        return;
    }

    g_attn_branch_counts[TQ_ATTN_BRANCH_GENERIC].fetch_add(
        1, std::memory_order_relaxed);

    // First write one K/V row per KV head. CUDA lines 12228-12231 perform the
    // same writes redundantly from every grouped q-head block; splitting this
    // phase avoids a SYCL global-memory data race while preserving the values.
    tq_q().parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(nkv) * local_size, local_size),
        [=](sycl::nd_item<1> item) {
            const int kv_head = static_cast<int>(item.get_group_linear_id());
            const int tid = static_cast<int>(item.get_local_linear_id());
            const float raw_k = d_k_proj[static_cast<size_t>(kv_head) * hd + tid];

            // CUDA lines 12147-12171: K head RMSNorm uses mean-square+eps and
            // applies the learned bf16 delta as (1 + weight), not plain weight.
            const float k_sum = sycl::reduce_over_group(
                item.get_group(), raw_k * raw_k, sycl::plus<float>());
            const float k_inv_rms = sycl::rsqrt(k_sum / static_cast<float>(hd) + eps);
            float key = raw_k * k_inv_rms *
                        (1.0f + tq_xpu_bf16_to_float(d_k_norm[tid]));

            // CUDA lines 12173-12186: partial RoPE is rotate-half, not
            // interleaved. Dimensions [0,rotary_half) pair with
            // [rotary_half,rotary_dim); the first half uses -pair*sin and the
            // second +pair*sin. freq=theta^(-2*i/rotary_dim).
            if (tid < rotary_dim) {
                const int pair_tid = tid < rotary_half ? tid + rotary_half
                                                        : tid - rotary_half;
                const int rope_index = tid < rotary_half ? tid : tid - rotary_half;
                const float frequency = sycl::pow(
                    rope_theta,
                    -(2.0f * static_cast<float>(rope_index) /
                      static_cast<float>(rotary_dim)));
                const float angle = static_cast<float>(pos) * frequency;
                const float pair =
                    d_k_proj[static_cast<size_t>(kv_head) * hd + pair_tid] * k_inv_rms *
                    (1.0f + tq_xpu_bf16_to_float(d_k_norm[pair_tid]));
                const float rotated_pair = tid < rotary_half ? -pair : pair;
                key = key * sycl::cos(angle) + rotated_pair * sycl::sin(angle);
            }

            // CUDA lines 12228-12230: cache normalized+roped K but raw V at row
            // pos. One work-group is one kv_head and one thread one element, so
            // the E4M3 row scale falls out of a work-group reduction.
            const float v_raw = d_v_proj[static_cast<size_t>(kv_head) * hd + tid];
            const float kamax = sycl::reduce_over_group(
                item.get_group(), sycl::fabs(key), sycl::maximum<float>());
            const float vamax = sycl::reduce_over_group(
                item.get_group(), sycl::fabs(v_raw), sycl::maximum<float>());
            const float ks = (kamax > 0.0f) ? (kamax / 448.0f) : 1.0f;
            const float vs = (vamax > 0.0f) ? (vamax / 448.0f) : 1.0f;
            const uint16_t ksb = sycl::bit_cast<uint16_t>(sycl::half(ks));
            const uint16_t vsb = sycl::bit_cast<uint16_t>(sycl::half(vs));
            const size_t row = tq_kv_group_physical_row<Paged>(
                pos, kv_head, nkv, layout, item.get_group(), tid);
            if (tid == 0) {
                d_k_scale[row] = ksb;
                d_v_scale[row] = vsb;
            }
            const size_t cache_index = row * hd + tid;
            d_k_cache[cache_index] = tq_xpu_e4m3_from_float(
                key / (float)sycl::bit_cast<sycl::half>(ksb));
            d_v_cache[cache_index] = tq_xpu_e4m3_from_float(
                v_raw / (float)sycl::bit_cast<sycl::half>(vsb));
        });

    tq_q().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> qn(sycl::range<1>(local_size), cgh);
        // alpha, beta, and the running softmax denominator.
        sycl::local_accessor<float, 1> softmax_shared(sycl::range<1>(3), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(nh) * local_size, local_size),
            [=](sycl::nd_item<1> item) {
                const int head = static_cast<int>(item.get_group_linear_id());
                const int tid = static_cast<int>(item.get_local_linear_id());

                // CUDA lines 12143-12147: GQA maps contiguous groups of q heads
                // to one KV head. q_proj is per-head [q(hd), gate(hd)], so q is
                // head*(2*hd)+tid and gate is head*(2*hd)+hd+tid.
                const int group = nh / nkv;
                const int kv_head = head / group;
                const size_t qg_base = static_cast<size_t>(head) * 2 * hd;
                const float raw_q = d_qg_proj[qg_base + tid];

                // CUDA lines 12146-12171: Q uses the same head RMSNorm formula
                // and learned (1 + bf16 weight) delta as K.
                const float q_sum = sycl::reduce_over_group(
                    item.get_group(), raw_q * raw_q, sycl::plus<float>());
                const float q_inv_rms = sycl::rsqrt(q_sum / static_cast<float>(hd) + eps);
                float query = raw_q * q_inv_rms *
                              (1.0f + tq_xpu_bf16_to_float(d_q_norm[tid]));

                // CUDA lines 12173-12186: identical partial rotate-half RoPE
                // convention for Q; the paired raw projection element is
                // normalized with its own q_norm weight before rotation.
                if (tid < rotary_dim) {
                    const int pair_tid = tid < rotary_half ? tid + rotary_half
                                                            : tid - rotary_half;
                    const int rope_index = tid < rotary_half ? tid : tid - rotary_half;
                    const float frequency = sycl::pow(
                        rope_theta,
                        -(2.0f * static_cast<float>(rope_index) /
                          static_cast<float>(rotary_dim)));
                    const float angle = static_cast<float>(pos) * frequency;
                    const float pair = d_qg_proj[qg_base + pair_tid] * q_inv_rms *
                        (1.0f + tq_xpu_bf16_to_float(d_q_norm[pair_tid]));
                    const float rotated_pair = tid < rotary_half ? -pair : pair;
                    query = query * sycl::cos(angle) + rotated_pair * sycl::sin(angle);
                }
                qn[tid] = query;
                item.barrier(sycl::access::fence_space::local_space);

                // Online stable softmax. Each causal K/V row is read once:
                // the old correctness kernel staged every score globally,
                // serialized softmax on lane 0, then reread the full V cache.
                // Lane 0 updates the shared scalar recurrence; the following
                // work-group reduction is also the end-of-iteration rendezvous,
                // so only one explicit barrier is needed per token.
                float running_max = -3.402823466e38f;
                float denominator = 0.0f;
                float attention = 0.0f;
                const float inv_sqrt_hd = sycl::rsqrt(static_cast<float>(hd));
                for (int token = 0; token <= pos; ++token) {
                    const size_t krow =
                        tq_kv_group_physical_row<Paged>(
                            token, kv_head, nkv, layout, item.get_group(), tid);
                    const size_t cache_index = krow * hd + tid;
                    const float ksc =
                        (float)sycl::bit_cast<sycl::half>(d_k_scale[krow]);
                    const float dot = sycl::reduce_over_group(
                        item.get_group(),
                        qn[tid] * tq_xpu_e4m3_to_float(d_k_cache[cache_index]),
                        sycl::plus<float>());
                    if (tid == 0) {
                        const float score = dot * ksc * inv_sqrt_hd;
                        const float next_max = sycl::fmax(running_max, score);
                        const float alpha = sycl::exp(running_max - next_max);
                        const float beta = sycl::exp(score - next_max);
                        denominator = denominator * alpha + beta;
                        running_max = next_max;
                        softmax_shared[0] = alpha;
                        softmax_shared[1] = beta;
                        softmax_shared[2] = denominator;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    const float vsc =
                        (float)sycl::bit_cast<sycl::half>(d_v_scale[krow]);
                    attention = attention * softmax_shared[0] +
                                softmax_shared[1] * vsc *
                                tq_xpu_e4m3_to_float(d_v_cache[cache_index]);
                }
                attention /= softmax_shared[2];

                // CUDA lines 12278-12279: the second half of each q-proj head
                // is an elementwise sigmoid output gate (not SiLU).
                const float gate = d_qg_proj[qg_base + hd + tid];
                d_out[static_cast<size_t>(head) * hd + tid] =
                    attention * (1.0f / (1.0f + sycl::exp(-gate)));
            });
    });
}

void x_full_attn_decode(
    float *d_out, const float *d_qg_proj, const float *d_k_proj,
    const float *d_v_proj, const uint16_t *d_q_norm,
    const uint16_t *d_k_norm, uint8_t *d_k_cache, uint8_t *d_v_cache,
    uint16_t *d_k_scale, uint16_t *d_v_scale, float *d_scores, int pos,
    int nh, int nkv, int hd, float eps, float rope_theta,
    float partial_rotary_factor, int sstride, tq_kv_layout_t layout) {
    if (layout.block_table) {
        x_full_attn_decode_impl<true>(
            d_out, d_qg_proj, d_k_proj, d_v_proj, d_q_norm, d_k_norm,
            d_k_cache, d_v_cache, d_k_scale, d_v_scale, d_scores, pos, nh,
            nkv, hd, eps, rope_theta, partial_rotary_factor, sstride, layout);
    } else {
        x_full_attn_decode_impl<false>(
            d_out, d_qg_proj, d_k_proj, d_v_proj, d_q_norm, d_k_norm,
            d_k_cache, d_v_cache, d_k_scale, d_v_scale, d_scores, pos, nh,
            nkv, hd, eps, rope_theta, partial_rotary_factor, sstride, layout);
    }
}
