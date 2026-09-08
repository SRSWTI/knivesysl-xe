#include "tq_common.hpp"

#include <cstddef>

namespace {

constexpr size_t kWorkGroupSize = 256;
constexpr size_t kGemmSubGroupSize = 16;
constexpr float kNegativeFloatMax = -3.402823466e38f;

class EmbedLookupKernel;
class RmsNormKernel;
class RmsNormHeadKernel;
class GemvQmmaKernel;
class SiluMulKernel;
class AddInplaceKernel;
class ArgmaxStage1Kernel;
class ArgmaxStage2Kernel;

// CUDA src/forward_qwen.cu:498-501: f32 -> bf16, round-to-nearest-even.
[[maybe_unused]] inline uint16_t tq_float_to_bf16(float x) {
    const uint32_t bits = sycl::bit_cast<uint32_t>(x);
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// CUDA src/forward_qwen.cu:504-510: bf16 is the high 16 bits of an f32 value.
inline float tq_bf16_to_float(uint16_t bits) {
    return sycl::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

// CUDA src/forward_qwen.cu:513-525: OCP E4M3FN decode, including subnormals.
[[maybe_unused]] inline float tq_e4m3_to_float(uint8_t code) {
    const int sign = (code >> 7) & 1;
    const int exponent = (code >> 3) & 0x0f;
    const int mantissa = code & 0x07;
    if ((code & 0x7fu) == 0) return 0.0f;

    const float magnitude = exponent == 0
        ? sycl::ldexp(static_cast<float>(mantissa) * 0.125f, -6)
        : sycl::ldexp(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
    return sign ? -magnitude : magnitude;
}

// CUDA src/forward_qwen.cu:11113-11117: E2M3 is 1 sign / 2 exponent
// (bias 1) / 3 mantissa bits; exponent zero is the m/8 subnormal range.
inline float tq_e2m3_to_float(uint32_t code) {
    const float mantissa = static_cast<float>(code & 7u);
    const uint32_t exponent = (code >> 3) & 3u;
    const float magnitude = exponent == 0
        ? mantissa * 0.125f
        : (1.0f + mantissa * 0.125f) * static_cast<float>(1u << (exponent - 1));
    return (code & 0x20u) ? -magnitude : magnitude;
}

// tools/convert.py:209-231 and CUDA src/forward_qwen.cu:11198-11212:
// recover one code from a lane's little-endian 16 x 6-bit stream.  The
// (row,col)->lane/code map is the inverse of the SM120 m16k32 A fragment.
inline uint32_t tq_e2m3_code_at(const uint8_t *payload, int Kt, int row, int col) {
    const int mt = row >> 4;
    const int kt = col >> 5;
    const int group = (row & 15) >> 1;
    const int sub = (col & 31) >> 3;
    const int lane = (group << 2) | sub;
    const int code_index = (((col >> 2) & 1) << 3) | ((row & 1) << 2) | (col & 3);
    const int bit_position = 6 * code_index;
    const int byte_index = bit_position >> 3;
    const int shift = bit_position & 7;

    const size_t tile = static_cast<size_t>(mt) * static_cast<size_t>(Kt)
                      + static_cast<size_t>(kt);
    const uint8_t *packed = payload + tile * 384u + static_cast<size_t>(lane) * 12u;
    uint32_t bits = packed[byte_index];
    if (shift > 2) bits |= static_cast<uint32_t>(packed[byte_index + 1]) << 8;
    return (bits >> shift) & 0x3fu;
}

// CUDA src/forward_qwen.cu:11990-12004 and 12018-12027 use strict greater-than
// while visiting indices in deterministic order.  Spell out the equivalent
// lowest-index tie rule so a different SYCL reduction tree cannot change it.
inline bool tq_argmax_better(float candidate, int candidate_id, float best, int best_id) {
    return candidate > best || (candidate == best && candidate_id < best_id);
}

inline size_t round_up(size_t n, size_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

}  // namespace

void x_embed_lookup(float *d_out, const uint16_t *d_embed, int token_id, int H) {
    if (!d_out || !d_embed || H <= 0) return;

    const size_t global = round_up(static_cast<size_t>(H), kWorkGroupSize);
    tq_q().parallel_for<EmbedLookupKernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kWorkGroupSize)),
        [=](sycl::nd_item<1> item) {
            const int i = static_cast<int>(item.get_global_linear_id());
            if (i >= H) return;
            // CUDA src/forward_qwen.cu:573-576: bf16 row lookup at token_id * H + i.
            const size_t offset = static_cast<size_t>(token_id) * static_cast<size_t>(H)
                                + static_cast<size_t>(i);
            d_out[i] = tq_bf16_to_float(d_embed[offset]);
        });
}

void x_rmsnorm(float *d_out, const float *d_in, const uint16_t *d_w, int H, float eps) {
    if (!d_out || !d_in || !d_w || H <= 0) return;

    tq_q().parallel_for<RmsNormKernel>(
        sycl::nd_range<1>(sycl::range<1>(kWorkGroupSize),
                          sycl::range<1>(kWorkGroupSize)),
        [=](sycl::nd_item<1> item) {
            const size_t tid = item.get_local_linear_id();
            float sum_sq = 0.0f;
            // CUDA src/forward_qwen.cu:4573-4577: FP32 input sum of squares.
            for (int i = static_cast<int>(tid); i < H; i += static_cast<int>(kWorkGroupSize)) {
                const float value = d_in[i];
                sum_sq += value * value;
            }
            sum_sq = sycl::reduce_over_group(item.get_group(), sum_sq, sycl::plus<float>());

            // CUDA src/forward_qwen.cu:4589,4593,4601-4603: eps is inside rsqrt,
            // and Qwen applies (1 + bf16_weight), not the bare stored weight.
            const float rms_inv = sycl::rsqrt(sum_sq / static_cast<float>(H) + eps);
            for (int i = static_cast<int>(tid); i < H; i += static_cast<int>(kWorkGroupSize)) {
                const float weight = 1.0f + tq_bf16_to_float(d_w[i]);
                d_out[i] = d_in[i] * rms_inv * weight;
            }
        });
}

void x_rmsnorm_head(float *d_out, const float *d_in, const uint16_t *d_w,
                    int heads, int hd, float eps) {
    if (!d_out || !d_in || !d_w || heads <= 0 || hd <= 0) return;

    const size_t global = static_cast<size_t>(heads) * kWorkGroupSize;
    tq_q().parallel_for<RmsNormHeadKernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kWorkGroupSize)),
        [=](sycl::nd_item<1> item) {
            const int head = static_cast<int>(item.get_group_linear_id());
            const int tid = static_cast<int>(item.get_local_linear_id());
            const size_t base = static_cast<size_t>(head) * static_cast<size_t>(hd);

            float sum_sq = 0.0f;
            // CUDA src/forward_qwen.cu:12146-12153: one independent FP32 sum of
            // squares over hd channels for every q/k head.
            for (int d = tid; d < hd; d += static_cast<int>(kWorkGroupSize)) {
                const float value = d_in[base + static_cast<size_t>(d)];
                sum_sq += value * value;
            }
            sum_sq = sycl::reduce_over_group(item.get_group(), sum_sq, sycl::plus<float>());

            // CUDA src/forward_qwen.cu:12170-12171: eps is inside rsqrt and the
            // shared per-channel head weight is applied as (1 + bf16_weight).
            const float rms_inv = sycl::rsqrt(sum_sq / static_cast<float>(hd) + eps);
            for (int d = tid; d < hd; d += static_cast<int>(kWorkGroupSize)) {
                const float weight = 1.0f + tq_bf16_to_float(d_w[d]);
                d_out[base + static_cast<size_t>(d)] =
                    d_in[base + static_cast<size_t>(d)] * rms_inv * weight;
            }
        });
}

int x_gemv_qmma(const tq_qmma_weight_t *w, const float *d_x, float *d_y) {
    if (!w || !d_x || !d_y) return -1;
    if (w->M <= 0 || w->K <= 0 || w->Mt <= 0 || w->Kt <= 0) return -2;
    // Repacking releases the E2M3 source, so the ready flag owns dispatch.
    if (w->s8_ready) return x_gemv_w8a8(w, d_x, d_y);
    if (w->s4_ready) return x_gemv_w4a8(w, d_x, d_y);
    if (!w->d_A || !w->d_block_scale_inv) return -3;
    if (!w->block_scaled || !w->e2m3 || w->row_major || w->word_major ||
        w->e2m1 || w->e2m3_byte) {
        return -4;
    }
    if (w->M != w->Mt * 16 || w->K != w->Kt * 32) return -5;
    const int required_scale_rows = (w->M + 127) >> 7;
    const int required_scale_cols = (w->Kt + 3) >> 2;
    if (w->scale_rows < required_scale_rows ||
        (w->scale_cols != w->Kt && w->scale_cols < required_scale_cols)) {
        return -6;
    }

    const int M = w->M;
    const int K = w->K;
    const int Mt = w->Mt;
    const int Kt = w->Kt;
    const int scale_cols = w->scale_cols;
    const uint8_t *payload = w->d_A;
    const float *scales = w->d_block_scale_inv;

    try {
        // Sixteen required size-16 Intel subgroups share each work-group; each
        // subgroup owns one row of an m16 tile.
        const size_t global = static_cast<size_t>(Mt) * kWorkGroupSize;
        tq_q().parallel_for<GemvQmmaKernel>(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kWorkGroupSize)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kGemmSubGroupSize)]] {
                const int mt = static_cast<int>(item.get_group_linear_id());
                const sycl::sub_group subgroup = item.get_sub_group();
                const int row_in_tile = static_cast<int>(subgroup.get_group_linear_id());
                const int lane = static_cast<int>(subgroup.get_local_linear_id());
                const int row = (mt << 4) + row_in_tile;
                if (row >= M) return;

                float sum = 0.0f;
                for (int col = lane; col < K; col += static_cast<int>(kGemmSubGroupSize)) {
                    const int kt = col >> 5;
                    // CUDA src/forward_qwen.cu:11113-11117 and 11200-11212:
                    // decode the original dense E2M3 fragment payload exactly.
                    const float code_value = tq_e2m3_to_float(
                        tq_e2m3_code_at(payload, Kt, row, col));

                    // CUDA src/forward_qwen.cu:6277 and 6606 plus 11264-11267:
                    // scale_row=(m/16)>>3; per-k32 containers use kt directly,
                    // otherwise one stored scale spans four k32 tiles (128 K).
                    const int scale_row = mt >> 3;
                    const int scale_col = scale_cols == Kt ? kt : (kt >> 2);
                    const float weight = code_value *
                        scales[static_cast<size_t>(scale_row) * static_cast<size_t>(scale_cols)
                               + static_cast<size_t>(scale_col)];

                    // CUDA src/forward_qwen.cu:11227 and 11267 confirm that the
                    // field named scale_inv is a dequant multiplier, not a divisor.
                    sum += weight * d_x[col];
                }
                sum = sycl::reduce_over_group(subgroup, sum, sycl::plus<float>());
                if (lane == 0) d_y[row] = sum;
            });
    } catch (const sycl::exception &) {
        return -7;
    }
    return 0;
}

int x_gemv_qmma_add(const tq_qmma_weight_t *w, const float *d_x,
                    const float *d_residual, float *d_y) {
    if (!w || !d_x || !d_residual || !d_y) return -1;
    if (w->s8_ready) return x_gemv_w8a8_add(w, d_x, d_residual, d_y);
    if (w->s4_ready) return x_gemv_w4a8_add(w, d_x, d_residual, d_y);
    const int ret = x_gemv_qmma(w, d_x, d_y);
    if (ret == 0) x_add_inplace(d_y, d_residual, w->M);
    return ret;
}

int x_gemv_qmma_prepared(const tq_qmma_weight_t *w, const float *d_x, float *d_y) {
    if (!w || !d_x || !d_y) return -1;
    if (w->s8_ready) return x_gemv_w8a8_prepared(w, d_y);
    if (w->s4_ready) return x_gemv_w4a8_prepared(w, d_y);
    return x_gemv_qmma(w, d_x, d_y);
}

void x_silu_mul(float *d_out, const float *d_gate, const float *d_up, int N) {
    if (!d_out || !d_gate || !d_up || N <= 0) return;

    const size_t global = round_up(static_cast<size_t>(N), kWorkGroupSize);
    tq_q().parallel_for<SiluMulKernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kWorkGroupSize)),
        [=](sycl::nd_item<1> item) {
            const int i = static_cast<int>(item.get_global_linear_id());
            if (i >= N) return;
            // CUDA src/forward_qwen.cu:11948-11952: SiLU(g)=g/(1+exp(-g)).
            const float gate = d_gate[i];
            d_out[i] = (gate / (1.0f + sycl::exp(-gate))) * d_up[i];
        });
}

void x_add_inplace(float *d_x, const float *d_y, int N) {
    if (!d_x || !d_y || N <= 0) return;

    const size_t global = round_up(static_cast<size_t>(N), kWorkGroupSize);
    tq_q().parallel_for<AddInplaceKernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kWorkGroupSize)),
        [=](sycl::nd_item<1> item) {
            const int i = static_cast<int>(item.get_global_linear_id());
            if (i >= N) return;
            // CUDA src/forward_qwen.cu:11934-11937, specialized to out == a.
            d_x[i] = d_x[i] + d_y[i];
        });
}

void x_argmax(const float *d_logits, int V, float *d_vals, int *d_ids,
              int *out_id, float *out_logit) {
    if (!out_id || !out_logit) return;
    if (!d_logits || !d_vals || !d_ids || V <= 0) {
        *out_id = 0;
        *out_logit = kNegativeFloatMax;
        return;
    }

    constexpr size_t stage1_local = 256;
    constexpr size_t stage1_groups = TQ_ARGMAX_BLOCKS;
    tq_q().submit([&](sycl::handler &handler) {
        sycl::local_accessor<float, 1> values(sycl::range<1>(stage1_local), handler);
        sycl::local_accessor<int, 1> ids(sycl::range<1>(stage1_local), handler);
        handler.parallel_for<ArgmaxStage1Kernel>(
            sycl::nd_range<1>(sycl::range<1>(stage1_groups * stage1_local),
                              sycl::range<1>(stage1_local)),
            [=](sycl::nd_item<1> item) {
                const int tid = static_cast<int>(item.get_local_linear_id());
                const int block = static_cast<int>(item.get_group_linear_id());
                const int first = block * static_cast<int>(stage1_local) + tid;
                const int stride = static_cast<int>(stage1_groups * stage1_local);
                float best = kNegativeFloatMax;
                int best_id = 0;

                // CUDA src/forward_qwen.cu:11987-11995: grid-stride stage-1 scan.
                for (int i = first; i < V; i += stride) {
                    const float candidate = d_logits[i];
                    if (tq_argmax_better(candidate, i, best, best_id)) {
                        best = candidate;
                        best_id = i;
                    }
                }
                values[tid] = best;
                ids[tid] = best_id;
                item.barrier(sycl::access::fence_space::local_space);

                // CUDA src/forward_qwen.cu:12000-12005: 256-way tree reduction.
                for (int offset = static_cast<int>(stage1_local >> 1); offset > 0; offset >>= 1) {
                    if (tid < offset &&
                        tq_argmax_better(values[tid + offset], ids[tid + offset],
                                         values[tid], ids[tid])) {
                        values[tid] = values[tid + offset];
                        ids[tid] = ids[tid + offset];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (tid == 0) {
                    d_vals[block] = values[0];
                    d_ids[block] = ids[0];
                }
            });
    });

    constexpr size_t stage2_local = TQ_ARGMAX_BLOCKS;
    tq_q().submit([&](sycl::handler &handler) {
        sycl::local_accessor<float, 1> values(sycl::range<1>(stage2_local), handler);
        sycl::local_accessor<int, 1> ids(sycl::range<1>(stage2_local), handler);
        handler.parallel_for<ArgmaxStage2Kernel>(
            sycl::nd_range<1>(sycl::range<1>(stage2_local), sycl::range<1>(stage2_local)),
            [=](sycl::nd_item<1> item) {
                const int tid = static_cast<int>(item.get_local_linear_id());
                // CUDA src/forward_qwen.cu:12018-12021: stage 2 consumes all
                // TQ_ARGMAX_BLOCKS partial winners.
                values[tid] = d_vals[tid];
                ids[tid] = d_ids[tid];
                item.barrier(sycl::access::fence_space::local_space);

                // CUDA src/forward_qwen.cu:12023-12028: 1024-way tree reduction.
                for (int offset = static_cast<int>(stage2_local >> 1); offset > 0; offset >>= 1) {
                    if (tid < offset &&
                        tq_argmax_better(values[tid + offset], ids[tid + offset],
                                         values[tid], ids[tid])) {
                        values[tid] = values[tid + offset];
                        ids[tid] = ids[tid + offset];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (tid == 0) {
                    // CUDA src/forward_qwen.cu:16678-16680 stores the final pair
                    // in scratch slot TQ_ARGMAX_BLOCKS.
                    d_vals[TQ_ARGMAX_BLOCKS] = values[0];
                    d_ids[TQ_ARGMAX_BLOCKS] = ids[0];
                }
            });
    });

    // CUDA src/forward_qwen.cu:16678-16680 plus its following host readback:
    // the in-order queue completes both stages before these copies, and the wait
    // is mandatory because the caller consumes the host pair immediately.
    tq_d2h(out_id, d_ids + TQ_ARGMAX_BLOCKS, sizeof(*out_id));
    tq_d2h(out_logit, d_vals + TQ_ARGMAX_BLOCKS, sizeof(*out_logit));
    tq_q().wait_and_throw();
}
