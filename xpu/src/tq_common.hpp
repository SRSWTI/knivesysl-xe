// tq_common.hpp — shared contract for libforward_qwen_xpu.so (knivesysl-xe)
//
// Phase 1 scope: single-stream scalar-correctness engine for the Qwen3.8-27B
// TQF1 container on Intel Arc Pro B70 (Xe2). Mirrors the CUDA TU's data model
// (src/forward_qwen.cu) with the Phase-1 trims:
//   - E4M3 KV cache with flat storage by default and an explicit paged tier
//   - no MTP head upload (the loader parses and SKIPS the MTP section)
//   - no paged speculative decode, NVFP4/sparse/byte tiers
//   - weights stay in the ORIGINAL TQF QMMA-fragment dense-E2M3 payload; the
//     scalar GEMV dequantizes via the closed-form (row,k)->(lane,bit) map.
// Numerics contract: replicate the CUDA formulas exactly; float summation
// ORDER is allowed to differ (subgroup-16 vs warp-32) — gates use eps bands.
//
// ABI: existing flat `qwn_*` symbols remain available; paging adds explicit
// ownership/checkpoint exports while preserving the Python tool belt's flat
// entry points.

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define TQ_MAX_LAYERS 64
#define TQ_LAYER_LINEAR_ATTENTION 1
#define TQ_LAYER_FULL_ATTENTION 2
#define TQ_DTYPE_FP16 1
#define TQ_DTYPE_BF16 2
#define TQ_ARGMAX_BLOCKS 1024
#define TQ_PAGED_ERR_CAPACITY (-1001)

#define TQ_FLAG_FP8_QMMA_WEIGHTS (1u << 0)
#define TQ_FLAG_TIED_EMBEDDINGS (1u << 1)
#define TQ_FLAG_SOURCE_BF16_NON_QUANT (1u << 2)
#define TQ_FLAG_DROPPED_VISION (1u << 3)
#define TQ_FLAG_DROPPED_MTP (1u << 4)
#define TQ_FLAG_BLOCK_SCALED_QMMA (1u << 5)
#define TQ_FLAG_BLOCK_SCALED_ROWMAJOR (1u << 6)
#define TQ_FLAG_BLOCK_SCALED_QMMA_WORDMAJOR (1u << 7)
#define TQ_FLAG_BLOCK_SCALED_E2M3 (1u << 8)
#define TQ_FLAG_SPARSE_24_E2M3 (1u << 9)
#define TQ_FLAG_HAS_MTP (1u << 10)

// One quantized matrix in the TQF QMMA fragment layout (Phase-1 subset of the
// CUDA tq_qmma_weight_t; NVFP4/sparse/TMA members dropped, field names kept).
typedef struct {
    int M, K, Mt, Kt;          // Mt = M/16, Kt = K/32 (m16k32 tiles)
    float inv_scale;           // non-block-scaled records only (legacy)
    uint8_t *d_A;              // payload, [Mt][Kt][tile_bytes]
    float *d_block_scale_inv;  // [scale_rows][scale_cols]
    int block_scaled;
    int row_major;
    int word_major;
    int e2m3;                  // dense 6-bit codes, 384 B/tile (ship format)
    int e2m1;
    int e2m3_byte;
    int scale_rows;
    int scale_cols;
    // Phase 2 DPAS payload. Repacking selects exactly one integer tier.
    uint8_t *d_s4;             // S4 codes, Xe2 B-operand layout
    uint16_t *d_s4_scale;      // FP16 scale per output row and K32 (or K16) tile
    int s4_ready;
    int s4_k16;                // 1 = two scales per K32 tile (CUDA's NVFP4
                               // granularity); costs 0.5 bits/w and one extra
                               // masked DPAS, buys ~15% less weight error.
    int s4_k64;                // 1 = ONE scale per K64 tile (dpas.s4.s4.8.8
                               // granularity, level-up item 4): saves 0.25
                               // bits/w, costs +13.6% weight rel-L2, and
                               // unlocks the W4A4 prefill GEMM.
    uint8_t *d_s8;             // S8 codes, Xe2 B-operand layout
    uint16_t *d_s8_scale;      // FP16 scale per output row and K32 tile
    int s8_ready;
} tq_qmma_weight_t;

static inline size_t tq_w_tile_words(const tq_qmma_weight_t *w) {
    if (w->e2m1) return 64;
    if (w->e2m3 && !w->e2m3_byte) return 96;
    return 128;
}
static inline size_t tq_w_tile_bytes(const tq_qmma_weight_t *w) {
    return tq_w_tile_words(w) * 4;
}

typedef struct {
    uint16_t *d_input_ln;      // bf16 [H]
    uint16_t *d_post_ln;       // bf16 [H]

    tq_qmma_weight_t mlp_gate; // [I, H]
    tq_qmma_weight_t mlp_up;   // [I, H]
    tq_qmma_weight_t mlp_down; // [H, I]

    // full-attention layers
    uint16_t *d_q_norm;        // bf16 [hd]
    uint16_t *d_k_norm;        // bf16 [hd]
    tq_qmma_weight_t q_proj;   // [2*nh*hd, H] — q AND output gate in one matrix
    tq_qmma_weight_t k_proj;   // [nkv*hd, H]
    tq_qmma_weight_t v_proj;   // [nkv*hd, H]
    tq_qmma_weight_t o_proj;   // [H, nh*hd]

    // linear-attention (gated DeltaNet) layers
    float *d_linear_A_log;         // f32 [linear_num_value_heads]
    uint16_t *d_linear_dt_bias;    // bf16 [linear_num_value_heads]
    float *d_linear_norm;          // f32 [linear_value_head_dim]
    uint16_t *d_linear_conv1d;     // bf16 [conv_dim, kernel]
    tq_qmma_weight_t linear_in_a;  // [nv, H]
    tq_qmma_weight_t linear_in_b;  // [nv, H]
    tq_qmma_weight_t linear_in_qkv;// [2*nk*dk + nv*dv, H]
    tq_qmma_weight_t linear_in_z;  // [nv*dv, H]
    tq_qmma_weight_t linear_out;   // [H, nv*dv]

    // per-layer state (allocated by qwn_reset_state / first use)
    float *d_linear_conv_state;       // [conv_dim, kernel]
    float *d_linear_recurrent_state;  // [nv, dk, dv]
    // KV cache is E4M3 with one FP16 scale per (token, kv_head) row: 258 bytes
    // per row against 1024 for fp32. That 3.97x cut is what makes long context
    // fit (fp32 needs 34.5 GB at 256k, more than the card) and it is the entire
    // measured depth deficit against vLLM-XPU, which also caches fp8.
    uint8_t *d_k_cache;               // E4M3 [max_seq, nkv, hd]
    uint8_t *d_v_cache;               // E4M3 [max_seq, nkv, hd]
    uint16_t *d_k_scale;              // fp16  [max_seq, nkv]
    uint16_t *d_v_scale;              // fp16  [max_seq, nkv]
} tq_layer_t;

typedef struct {
    int initialized;
    uint32_t flags;
    uint32_t model_family;
    uint32_t non_quant_dtype;
    int H, I, L, V;
    int nh, nkv, hd;
    int linear_num_key_heads;
    int linear_key_head_dim;
    int linear_num_value_heads;
    int linear_value_head_dim;
    int linear_conv_kernel_dim;
    int max_position_embeddings;
    float eps;
    float rope_theta;
    float partial_rotary_factor;
    int mrope_interleaved;
    int mrope_section[3];
    int tie_word_embeddings;
    int text_only;
    int has_vision_config;
    int has_mtp;
    int has_mtp_section;
    uint8_t layer_types[TQ_MAX_LAYERS];

    uint16_t *d_embed;   // bf16 [V, H]
    uint16_t *d_norm;    // bf16 [H] final norm
    tq_qmma_weight_t lm_head;  // [V, H] (untied)
    tq_layer_t layers[TQ_MAX_LAYERS];

    int max_seq;         // TQ_CTX (default 4096 in Phase 1)

    // single-token forward scratch (fp32), mirrors the CUDA d_debug_* set.
    // Allocated by qwn_init (sizes fixed by the header dims).
    float *d_x;          // residual stream [H]                    (CUDA d_debug_x)
    float *d_norm_out;   // post input-norm [H]                    (d_debug_norm)
    float *d_qkv;        // widest projection out [max(2*nh*hd, conv_dim)] (d_debug_qkv)
    float *d_z;          // z / v projection [max(nkv*hd, nv*dv)]  (d_debug_z)
    float *d_b;          // b proj / k proj [max(heads, nkv*hd)]   (d_debug_proj / d_debug_a_proj)
    float *d_a;          // a projection [heads]                   (d_debug_a_proj)
    float *d_core;       // attn/delta core out [max(nh*hd, nv*dv)] (d_debug_core)
    float *d_resid;      // residual after attn add [H]            (d_debug_resid)
    float *d_post_norm;  // post-attention norm [H]                (d_debug_post_norm)
    float *d_gate;       // mlp gate [I]                           (d_debug_gate)
    float *d_up;         // mlp up [I]                             (d_debug_up)
    float *d_mlp_hidden; // silu(gate)*up [I]                      (d_debug_mlp_hidden)
    float *d_layer_out;  // layer output [H]                       (d_debug_layer_out)
    float *d_proj_out;   // generic projection out [max weight M]  (d_debug_qmma_out)
    float *d_logits;     // [V]
    float *d_scores;     // attention score slab [nh * max_seq]    (d_attn_scores)
    float *d_argmax_vals; int *d_argmax_ids;  // [TQ_ARGMAX_BLOCKS + 1] each
    int last_argmax_id; float last_argmax_logit;
    // Multi-slot serving state (design: concurrency campaign). Per-sequence
    // state (KV caches + scales, GDN conv + recurrent, committed position)
    // is allocated slots-times and addressed by slot-stride offsets at the
    // decode call sites; kernels are slot-agnostic. TQ_XPU_SLOTS (1..8,
    // read once at init) fixes the count.
    int slots;           // allocated sequence slots (>= 1)
    int active_slot;     // slot the single-token/prefill/wave paths use
    int state_pos[8];    // committed position per slot
} tq_model_t;

// Tensor parallelism keeps one model struct per rank. `g_qwen` resolves to
// the ACTIVE rank's struct, so every existing use site compiles unchanged;
// tq_set_rank() moves both the queue and the model together. TQ_XPU_TP=1
// leaves rank 0 selected forever, i.e. exactly the single-card engine.
constexpr int kMaxTP = 2;
extern tq_model_t g_qwen_ranks[kMaxTP];
extern tq_model_t *g_qwen_active;
#define g_qwen (*g_qwen_active)

// The one in-order SYCL queue every kernel launches on (device index from
// TQ_XPU_DEV, default 0).
sycl::queue &tq_q();
sycl::queue &tq_q_of(int rank);
// Tensor parallelism. TQ_XPU_TP=1 (default) is the single-card engine.
// Switching the active rank re-targets every allocator, transfer and kernel
// launch below, so TP costs the kernels nothing.
int tq_tp_size();
int tq_rank();
void tq_set_rank(int rank);
// Sum n floats across ranks in place: bufs[r]/scratch[r] live on rank r.
void tq_allreduce(float *const *bufs, float *const *scratch, int n);

// checked USM helpers (abort with message on failure, like the CUDA TU's macros)
void *tq_dev_alloc(size_t bytes, const char *what);
void tq_dev_free(void *p);
void tq_h2d(void *dst, const void *src, size_t bytes);
void tq_d2h(void *dst, const void *src, size_t bytes);
void tq_dev_zero(void *dst, size_t bytes);

// ---------------------------------------------------------------------------
// kernels_core.cpp — codecs, embed, norms, scalar GEMV, silu, argmax
// ---------------------------------------------------------------------------
// bf16 -> f32 embed row lookup: out[H] = embed[token_id * H .. +H) (bf16)
void x_embed_lookup(float *d_out, const uint16_t *d_embed, int token_id, int H);
// Qwen3 RMSNorm, EXACT formula from k_tq_qwen_rmsnorm (read the CUDA source for
// the weight application; do not guess (1+w) vs w): out[H] from in[H], bf16 w.
void x_rmsnorm(float *d_out, const float *d_in, const uint16_t *d_w, int H, float eps);
// RMSNorm over a head slice in place-free form (q/k head norms, width hd)
void x_rmsnorm_head(float *d_out, const float *d_in, const uint16_t *d_w, int heads,
                    int hd, float eps);
// Scalar dequant GEMV over the TQF QMMA dense-E2M3 fragment payload:
// y[M] = sum_k dequant(W[m,k]) * x[k]. Must reproduce the CUDA dequant exactly
// (code value table + block-scale application); summation order free.
int x_gemv_qmma(const tq_qmma_weight_t *w, const float *d_x, float *d_y);
// out[N] = silu(gate[N]) * up[N]
void x_silu_mul(float *d_out, const float *d_gate, const float *d_up, int N);
// residual add: x[N] += y[N]
void x_add_inplace(float *d_x, const float *d_y, int N);
// two-stage argmax over logits[V] -> (id, logit) on host
void x_argmax(const float *d_logits, int V, float *d_vals, int *d_ids,
              int *out_id, float *out_logit);

// ---------------------------------------------------------------------------
// kernels_seq.cpp — conv1d, DeltaNet core, full-attention decode (fp32 KV)
// ---------------------------------------------------------------------------
// Causal depthwise conv1d update (kernel size ks): shifts state, applies conv
// over qkv projection, SiLU where the CUDA kernel does. Mirrors
// k_tq_linear_conv_update semantics for a single token.
void x_linear_conv_update(float *d_out, float *d_state, const float *d_x,
                          const uint16_t *d_conv_w, int conv_dim, int ks);
// Chunk-parallel GDN prefill (level-up design 9.2, ported from the CUDA
// twin's k_tq_deltanet_chunk). x_linear_conv_chunk applies the causal
// depthwise conv + SiLU to T tokens in parallel (d_out != d_x; the incoming
// conv state supplies tokens before the chunk) and then advances the conv
// state to the chunk end. x_deltanet_chunk runs the chunkwise-exact delta
// rule (CK-token sub-chunks, forward-substitution solve, serial only across
// sub-chunks) producing the same gated-RMSNorm core and advanced recurrent
// state as T serial x_linear_decode_core_gated calls, up to reassociation.
void x_linear_conv_chunk(float *d_out, float *d_state, const float *d_x,
                         const uint16_t *d_conv_w, int conv_dim, int ks, int T);
int x_deltanet_chunk(float *d_core_out, float *d_recurrent,
                     const float *d_conv_out, const float *d_z,
                     const float *d_b, const float *d_a, const float *d_A_log,
                     const uint16_t *d_dt_bias, const float *d_norm_w, int T,
                     int nk, int dk, int nv, int dv, float eps);
void x_linear_conv_advance(float *d_state, const float *d_x, int conv_dim,
                           int ks, int T);
// Batched W8A8 GEMM (RC8 s8 x s8 at K=32): same task grid/staging as
// x_gemm_w4a8, plain FP16 scale (no zero point). Serves the spec wave's
// 8-row lm_head so the 1.27 GB weight streams once, not eight times.
int x_gemm_w8a8(const tq_qmma_weight_t *w, const int8_t *d_aq, const float *d_as,
                float *d_y, int T);
// Gated delta rule single-token core incl. gated RMSNorm x SiLU(z) epilogue,
// mirroring k_tq_linear_decode_core_gated (one launch, all value heads).
void x_linear_decode_core_gated(float *d_out, float *d_recurrent,
                                const float *d_conv_out, const float *d_z,
                                const float *d_b, const float *d_a,
                                const float *d_A_log, const uint16_t *d_dt_bias,
                                const float *d_norm_w,
                                int nk, int dk, int nv, int dv, float eps);
// Per-call KV addressing. In paged mode block_table points to one slot's
// logical-block row in the device table; flat mode passes a null table.
struct tq_kv_layout_t {
    const int *block_table;
    int page_log;
    int page_mask;
};

// Full-attention single-token decode with an E4M3 KV cache: writes the k/v row
// at pos (encoding it and its FP16 row scale), applies q/k head norms + partial
// RoPE, computes attention over [0..pos], applies the output gate sigmoid.
// Mirrors k_tq_full_attn_decode(+tail).
void x_full_attn_decode(float *d_out, const float *d_qg_proj, const float *d_k_proj,
                        const float *d_v_proj, const uint16_t *d_q_norm,
                        const uint16_t *d_k_norm,
                        uint8_t *d_k_cache, uint8_t *d_v_cache,
                        uint16_t *d_k_scale, uint16_t *d_v_scale,
                        float *d_scores, int pos, int nh, int nkv, int hd,
                        float eps, float rope_theta, float partial_rotary_factor,
                        int sstride, tq_kv_layout_t layout);
// Wide prefill, two phases in order on the in-order queue. Phase 1 stores every
// chunk token's normed+RoPE'd K and raw V (byte-identical to the store inside
// x_full_attn_decode), so a token can attend causally to earlier tokens of its
// OWN chunk. Phase 2 is causal tiled attention: a 16-query scalar oracle or
// a 64-query, multi-subgroup XMX tile. Packed attention combines requests in
// one grid after all their KV writes. TQ_XPU_PREFILL_XMX=auto selects profitable
// shapes; 0 selects scalar and 1 requires XMX. Three BF16 components retain
// FP32 operand precision, but DPAS/softmax reassociation is NOT bit-identical
// to the scalar algorithm; model agreement has its own measured gate.
void x_prefill_kv_write(const float *d_k_proj, const float *d_v_proj,
                        const uint16_t *d_k_norm, uint8_t *d_k_cache,
                        uint8_t *d_v_cache, uint16_t *d_k_scale,
                        uint16_t *d_v_scale, int pos0, int T, int nkv, int hd,
                        float eps, float rope_theta,
                        float partial_rotary_factor, tq_kv_layout_t layout);
void x_prefill_attn(float *d_out, const float *d_qg_proj,
                    const uint16_t *d_q_norm, const uint8_t *d_k_cache,
                    const uint8_t *d_v_cache, const uint16_t *d_k_scale,
                    const uint16_t *d_v_scale, int pos0, int T, int nh,
                    int nkv, int hd, float eps, float rope_theta,
                    float partial_rotary_factor, tq_kv_layout_t layout);
// Host-only descriptors for a single packed attention launch. The dispatcher
// captures all n (1..8) descriptors by value before returning; no device table
// or caller-owned host storage survives submission. Queue all KV writes first.
struct tq_prefill_attn_request_t {
    float *out;
    const float *qg;
    const uint8_t *kc, *vc;
    const uint16_t *ks, *vs;
    int pos0, T;
    tq_kv_layout_t layout;
};
void x_prefill_attn_packed(const tq_prefill_attn_request_t *requests, int n,
                           const uint16_t *qnorm, int nh, int nkv, int hd,
                           float eps, float rope_theta,
                           float partial_rotary_factor);

// ---------------------------------------------------------------------------
// kernels_dpas.cpp — Phase 2 native W4A8 / selective W8A8 tiers
// ---------------------------------------------------------------------------
// Repack one dense-E2M3 matrix into the requested Xe2 DPAS B layout plus FP16
// scales per output row and K32 tile. Each frees d_A and its source scales.
int x_w4_repack_weight(tq_qmma_weight_t *w);
// mode: 0 = one scale per K32; 1 = two per K32 (k16, +0.5 bits/w);
// 2 = one per K64 (k64, W4A4 DPAS granularity, -0.25 bits/w).
int x_w4_repack_weight_mode(tq_qmma_weight_t *w, int mode);
int x_w8_repack_weight(tq_qmma_weight_t *w);
// Shared fp32 -> S8 per-K32 symmetric activation quantizer. d_sum receives
// the exact integer sum for each tile (used by asymmetric W4 zero points).
void x_quantize_act_s8(const float *d_x, int K, int8_t *d_q, float *d_scale,
                       int32_t *d_sum);
int x_gemv_w4a8(const tq_qmma_weight_t *w, const float *d_x, float *d_y);
int x_gemv_w8a8(const tq_qmma_weight_t *w, const float *d_x, float *d_y);
int x_gemv_w4a8_add(const tq_qmma_weight_t *w, const float *d_x,
                    const float *d_residual, float *d_y);
int x_gemv_w8a8_add(const tq_qmma_weight_t *w, const float *d_x,
                    const float *d_residual, float *d_y);
int x_gemv_qmma_add(const tq_qmma_weight_t *w, const float *d_x,
                    const float *d_residual, float *d_y);
// Reuse one activation quantization across projections with the same input.
// The in-order queue preserves the prepared scratch until the next prepare.
void x_prepare_gemv_act_s8(const float *d_x, int K);
// Exact SiLU*up plus K32 S8 staging; retains the fp32 hidden vector for debug.
// Requires positive K divisible by 32. Consumed by the next prepared projection.
void x_silu_mul_quant(float *d_out, const float *d_gate, const float *d_up, int K);
// Fused (1+w) RMSNorm + the activation quantization its consumers need. Writes
// the normalized fp32 vector to d_out AND fills the prepared S8 scratch, so a
// caller that follows a norm with projections needs no separate prepare.
void x_rmsnorm_quant(float *d_out, const float *d_in, const uint16_t *d_w,
                     int H, float eps);
// Fan-out: several W4 projections sharing one prepared activation and one K,
// issued as a SINGLE launch. Row-groups are numbered across the set, so no
// weight concatenation is required. Returns negative if the set is not
// uniformly W4-K32 or K differs; the caller then falls back to per-weight
// calls. Requires x_prepare_gemv_act_s8 (or x_rmsnorm_quant) to have run.
int x_gemv_w4a8_fanout(const tq_qmma_weight_t *const *ws, float *const *ys,
                       int count);
// Batched (1+w) RMSNorm over T token rows, one work-group per row.
void x_rmsnorm_chunk(float *d_out, const float *d_in, const uint16_t *d_w,
                     int H, int T, float eps);
// Batched W4A8 GEMM: Y[T][M] = W[M,K] @ X[K,T] via dpas RC8 - the same
// 256-byte weight fragment against eight activation rows. T must be a multiple
// of 8. Feed it with x_quantize_act_chunk, which lays activations out as the
// probed A operand requires: Aq[kt][token][32], as[kt][token],
// asum[kt][token] — the sum arrives already reduced over the full K32 block,
// so the GEMM epilogue does not re-add two K16 halves per k-tile.
void x_quantize_act_chunk(const float *d_x, int K, int T, int8_t *d_aq,
                          float *d_as, int32_t *d_asum);
// Optional host offsets[segments+1] (<=8 positive RC8-aligned segments) keep
// each request's split-K policy in one packed launch; no device table needed.
int x_gemm_w4a8(const tq_qmma_weight_t *w, const int8_t *d_aq, const float *d_as,
                const int32_t *d_asum, float *d_y, int T,
                const int *offsets = nullptr, int segments = 0);
// W4A4 tier (requires s4_k64 weights): symmetric S4 activations at one scale
// per (token, K64) tile, layout Aq4[kt64][token][32] (64 nibbles/row, low
// nibble = even k), as4/asum4 per (kt64, token). Y via dpas.s4.s4.8.8 at
// 733 TOPS - half the DPAS count and half the epilogue of the W4A8 GEMM.
void x_quantize_act_chunk_s4(const float *d_x, int K, int T, uint8_t *d_aq4,
                             float *d_as4, int32_t *d_asum4);
int x_gemm_w4a4(const tq_qmma_weight_t *w, const uint8_t *d_aq4,
                const float *d_as4, const int32_t *d_asum4, float *d_y, int T);
int x_gemv_w4a8_prepared(const tq_qmma_weight_t *w, float *d_y,
                         const float *d_residual = nullptr);
int x_gemv_w8a8_prepared(const tq_qmma_weight_t *w, float *d_y,
                         const float *d_residual = nullptr);
int x_gemv_qmma_prepared(const tq_qmma_weight_t *w, const float *d_x, float *d_y,
                         const float *d_residual = nullptr);
// Tensor-parallel weight sharding of an already-repacked W4 weight.
// mode 0 = column-parallel (output rows, honouring packed groups such as
// q_proj's [q | gate]); mode 1 = row-parallel (the K dimension). Allocates
// on the CURRENT rank; src is untouched.
int x_w4_shard(tq_qmma_weight_t *dst, const tq_qmma_weight_t *src,
               int rank, int tp, int mode, const int *groups, int ngroups);
void x_w4_shard_free(tq_qmma_weight_t *w);

// decode.cpp-owned paging and scratch lifecycle used by tqf_loader.cpp.
int tq_paged_init_from_env(void);
int tq_paged_ready(void);
int tq_paged_reset_all(void);
int tq_paged_reset_slot(int slot);
void tq_paged_free(void);
int tq_decode_runtime_warm(int prefill_tokens);
void tq_decode_runtime_free(void);

// ---------------------------------------------------------------------------
// decode.cpp — ABI
// ---------------------------------------------------------------------------
extern "C" {
int qwn_init(const char *path);
int qwn_probe(const char *path);
void qwn_free(void);
int qwn_reset_state(void);
int qwn_reset_slot(int slot);
int qwn_num_slots(void);
int qwn_get_slot(void);
int qwn_set_slot(int slot);
int qwn_tp_selftest(int n);
int qwn_tp_shard(void);
int qwn_decode_batch(const int *slots, const int *tokens,
                     const int *positions, int n, int *out_tokens);
// Packed prompt segments: offsets[0]=0, offsets[n]=token count, each length
// positive and divisible by 8, distinct slots (n <= min(num_slots, 8)).
// positions[i]>0 must equal committed state; 0 resets only that slot while
// preserving its paged reservation and independent APC checkpoints. Returns
// 0 after all positions commit, no logits; active_slot is unchanged.
// Errors before slot mutation: -1 inputs/init, -2 slots/offsets, -3 position or
// size overflow, -4 token range, -5 unsupported tier/shape, -6 allocation,
// TQ_PAGED_ERR_CAPACITY. Runtime -7/-8 is fatal; positions stay unpublished
// but KV/GDN may be partial, so do not retry on the same engine.
int qwn_prefill_batch(const int *slots, const int *tokens, const int *offsets,
                      const int *positions, int n);
void *qwn_host_alloc(size_t bytes);
void qwn_host_free(void *p);
size_t qwn_ckpt_bytes(int pos);
int qwn_ckpt_save(int slot, int pos, void *dst);
int qwn_ckpt_restore(int slot, int pos, const void *src);
int qwn_paged_enabled(void);
int qwn_paged_reserve(int slot, int total_tokens);
int qwn_paged_release(int slot);
int qwn_paged_stats(long long *out, int count);
int qwn_paged_ckpt_save(int slot, int committed);
int qwn_paged_ckpt_adopt(int slot, int id);
int qwn_paged_ckpt_free(int id);
size_t qwn_paged_ckpt_host_bytes(int id);
size_t qwn_debug_kv_bytes(int committed);
int qwn_debug_kv_copy(int slot, int layer, int committed, void *out,
                      size_t bytes);
int qwn_debug_block_table(int slot, int *out, int count);
int qwn_attn_branch_counts(unsigned long long *out, int count);
void qwn_attn_branch_counts_reset(void);
int qwn_hidden_size(void);
int qwn_intermediate_size(void);
int qwn_vocab_size(void);
int qwn_num_layers(void);
int qwn_num_attention_heads(void);
int qwn_num_key_value_heads(void);
int qwn_head_dim(void);
int qwn_max_seq(void);
int qwn_layer_type(int layer);
int qwn_has_mtp(void);
int qwn_last_argmax_id(void);
float qwn_last_argmax_logit(void);
int qwn_decode(int token_id, int pos);
// debug ladder (Phase-1 subset used by the parity harness)
int qwn_debug_embed_input_norm(int token_id, int layer, float *out, int max_count);
int qwn_debug_forward_layers(int token_id, int n_layers, float *out, int max_count);
int qwn_debug_forward_final_norm(int token_id, float *out, int max_count);
int qwn_debug_forward_argmax(int token_id, float *logit_out);
int qwn_debug_decode_layers(int token_id, int pos, int n_layers, float *out, int max_count);
}
