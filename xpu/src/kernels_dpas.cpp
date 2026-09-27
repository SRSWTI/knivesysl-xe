// kernels_dpas.cpp — Phase 2 W4A8 tier for knivesysl-xe.
//
// Weight path: dense-E2M3 QMMA fragment payload -> signed-INT4 codes in the
// Xe2 DPAS B-operand layout + one FP16 scale per (row, 32-K tile). Activation
// path: fp32 -> signed-INT8 with one FP32 scale per 128-K block. GEMV:
// dpas.s4.s8.8.1 (subgroup-16, D[1][16] = A[1][32] x B[32][16]); each K32
// accumulator is dequantized into the fp32 output accumulator.
//
// Operand layouts verified on-silicon by xpu/probe/dpas_layout_probe.cpp
// (one-hot decode, all PASS):
//   A (s8, rc=1): flat byte k of the 32-byte region == k; per-lane vector
//     element mapping: lane l supplies bytes [2l, 2l+1] (flat = i*16 + lane
//     dword rule specialised to a ushort per lane).
//   B (s4): flat dword (k/8)*16 + n, nibble k%8. Per (rowgroup, kt) block of
//     64 dwords, a lane's uintv4 element i sits at flat dword i*16 + lane.
//   D (s32, rc=1): one dword per lane, lane == output column n.
//
// Quantization mapping (per output row and K32 tile):
//   wf = e2m3_decode(code) * block_scale_inv[...]        (bit-exact dequant)
//   q in [-8,7], with a two-step least-squares scale refinement
//   weight ~= q * s4scale; y = sum_kt(acc32_kt * s4scale[row][kt] *
//                                     activation_scale[kt/4])

#include "tq_common.hpp"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

namespace {

using uintv4 = uint32_t __attribute__((ext_vector_type(4)));
using uintv8 = uint32_t __attribute__((ext_vector_type(8)));
using floatv8 = float   __attribute__((ext_vector_type(8)));
using intv16  = int32_t __attribute__((ext_vector_type(16)));
using intv8  = int32_t  __attribute__((ext_vector_type(8)));

constexpr int kSG = 16;
constexpr int kSubgroupsPerWorkgroup = 16;
// Max projections a single fan-out GEMV can serve.
constexpr int kFanoutMax = 4;

inline float bf16f(uint16_t b) {
    return sycl::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}

// E2M3 decode — same table as kernels_core.cpp (CUDA forward_qwen.cu:567-570).
inline float e2m3f(uint32_t c) {
    const float m = static_cast<float>(c & 7u);
    const uint32_t e = (c >> 3) & 3u;
    const float v = (e == 0) ? m * 0.125f : (1.0f + m * 0.125f) * (float)(1u << (e - 1));
    return (c & 0x20u) ? -v : v;
}

// (row, col) -> 6-bit code in the dense-E2M3 QMMA fragment payload.
// Same closed form as kernels_core.cpp (verified there 3072/3072 vs convert.py).
inline uint32_t e2m3_code_at(const uint8_t *payload, int Kt, int row, int col) {
    const int mt = row >> 4;
    const int kt = col >> 5;
    const int lane = (((row & 15) >> 1) << 2) | ((col & 31) >> 3);
    const int idx = (((col >> 2) & 1) << 3) | ((row & 1) << 2) | (col & 3);
    const int bit = 6 * idx;
    const uint8_t *p = payload + ((size_t)mt * Kt + kt) * 384u + (size_t)lane * 12u;
    uint32_t bits = p[bit >> 3];
    if ((bit & 7) > 2) bits |= (uint32_t)p[(bit >> 3) + 1] << 8;
    return (bits >> (bit & 7)) & 0x3fu;
}

#ifdef __SYCL_DEVICE_ONLY__
#define DPAS_S4S8_RC1(dd, aa, bb)                                              \
    asm("{\n"                                                                  \
        ".decl DST     v_type=G type=D num_elts=16 alias=<%0,0>\n"             \
        ".decl SRC1_UD v_type=G type=UD num_elts=64 alias=<%2,0>\n"            \
        ".decl SRC2_UD v_type=G type=UD num_elts=8 alias=<%1,0>\n"             \
        "dpas.s4.s8.8.1 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"         \
        "}\n"                                                                  \
        : "+rw"(dd) : "rw"(aa), "rw"(bb))
// Repeat-count 8: the SAME 256-byte B fragment as RC1, but eight activation
// rows per instruction, so weight bytes per output drop 8x. Operand shapes
// verified on silicon by xpu/probe/dpas_layout_probe.cpp - A is 8 rows of 32
// contiguous bytes (64 dwords), D routes (m,n) to dword m*16+n (128 ints).
#define DPAS_S4S8_RC8(dd, aa, bb)                                              \
    asm("{\n"                                                                  \
        ".decl DST     v_type=G type=D num_elts=128 alias=<%0,0>\n"            \
        ".decl SRC1_UD v_type=G type=UD num_elts=64 alias=<%2,0>\n"            \
        ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"            \
        "dpas.s4.s8.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"         \
        "}\n"                                                                  \
        : "+rw"(dd) : "rw"(aa), "rw"(bb))
// K=64 s4 x s4 (733-TOPS tier): same 8-row A footprint (64 nibbles fit the
// same 32 bytes/row), B doubles to 512 bytes (128 dwords). Probed at rate by
// dpas_probe.cpp; operand routing certified end-to-end by qwn_gemm_w4a4_check
// against the GEMV reference (the all-ones probe cannot see ordering).
#define DPAS_S4S4_RC8(dd, aa, bb)                                              \
    asm("{\n"                                                                  \
        ".decl DST     v_type=G type=D num_elts=128 alias=<%0,0>\n"            \
        ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"           \
        ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"            \
        "dpas.s4.s4.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"         \
        "}\n"                                                                  \
        : "+rw"(dd) : "rw"(aa), "rw"(bb))
#define DPAS_S8S8_RC1(dd, aa, bb)                                              \
    asm("{\n"                                                                  \
        ".decl DST     v_type=G type=D num_elts=16 alias=<%0,0>\n"             \
        ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"           \
        ".decl SRC2_UD v_type=G type=UD num_elts=8 alias=<%1,0>\n"             \
        "dpas.s8.s8.8.1 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"         \
        "}\n"                                                                  \
        : "+rw"(dd) : "rw"(aa), "rw"(bb))
// RC8 s8 x s8: same 512-byte lane-major B fragment as the W8 GEMV, eight
// activation rows per instruction. Serves the spec wave's batched lm_head.
#define DPAS_S8S8_RC8(dd, aa, bb)                                              \
    asm("{\n"                                                                  \
        ".decl DST     v_type=G type=D num_elts=128 alias=<%0,0>\n"            \
        ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"           \
        ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"            \
        "dpas.s8.s8.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"         \
        "}\n"                                                                  \
        : "+rw"(dd) : "rw"(aa), "rw"(bb))
#else
#define DPAS_S4S8_RC1(dd, aa, bb) do { (void)(aa); (void)(bb); } while (0)
#define DPAS_S8S8_RC1(dd, aa, bb) do { (void)(aa); (void)(bb); } while (0)
#define DPAS_S4S8_RC8(dd, aa, bb) do { (void)(aa); (void)(bb); } while (0)
#define DPAS_S4S4_RC8(dd, aa, bb) do { (void)(aa); (void)(bb); } while (0)
#define DPAS_S8S8_RC8(dd, aa, bb) do { (void)(aa); (void)(bb); } while (0)
#endif

// Lazily grown activation-quant scratch: S8 codes, one FP32 scale per K32
// (one DPAS worth), and one exact integer sum per K16 so both the K32 and the
// K16 weight tiers can subtract their own zero-point term.
int8_t *g_act_q = nullptr;
float *g_act_scale = nullptr;
int32_t *g_act_sum = nullptr;
int g_act_cap = 0;

void ensure_act_scratch(int K) {
    if (g_act_cap >= K) return;
    if (g_act_q) tq_dev_free(g_act_q);
    if (g_act_scale) tq_dev_free(g_act_scale);
    if (g_act_sum) tq_dev_free(g_act_sum);
    g_act_q = static_cast<int8_t *>(tq_dev_alloc((size_t)K, "act.q8"));
    g_act_scale = static_cast<float *>(tq_dev_alloc((size_t)(K / 32) * sizeof(float),
                                                    "act.scale"));
    g_act_sum = static_cast<int32_t *>(tq_dev_alloc((size_t)(K / 16) * sizeof(int32_t),
                                                    "act.sum"));
    g_act_cap = K;
}

// Lazily grown per-work-group sum-of-squares partials for the fused norm.
float *g_norm_partial = nullptr;
size_t g_norm_partial_cap = 0;

void ensure_norm_partials(size_t n) {
    if (g_norm_partial_cap >= n) return;
    if (g_norm_partial) tq_dev_free(g_norm_partial);
    g_norm_partial = static_cast<float *>(
        tq_dev_alloc(n * sizeof(float), "norm.partial"));
    g_norm_partial_cap = n;
}

float *g_split_partial = nullptr;
size_t g_split_partial_cap = 0;

// Staying at or below kSubgroupsPerWorkgroup keeps the split-K reduction
// inside one work-group, so a GEMV is always a SINGLE kernel launch. The tiny
// in_a/in_b projections (M=48, 3 row-groups) used to ask for 32 splits and
// paid a second launch for 0.2 us of arithmetic.
int choose_gemv_splits(int groups) {
    if (groups <= 4) return kSubgroupsPerWorkgroup;
    return groups >= 4096 ? 1 : 8;
}

void ensure_split_scratch(size_t elements) {
    if (g_split_partial_cap >= elements) return;
    if (g_split_partial) tq_dev_free(g_split_partial);
    g_split_partial = static_cast<float *>(
        tq_dev_alloc(elements * sizeof(float), "gemv.split_partial"));
    g_split_partial_cap = elements;
}

}  // namespace

// fp32 -> S8. One scale per K32 (exactly one Xe2 DPAS, so the K32 weight tier
// keeps its single instruction) and one exact integer sum per K16 half, which
// is the finest granularity any weight tier applies a zero point at.
template<bool Silu>
static void quantize_act_s8(const float *d_x, const float *d_up, float *d_out,
                            int K, int8_t *d_q, float *d_scale, int32_t *d_sum) {
    const int blocks = K / 32;
    tq_q().parallel_for(
        sycl::nd_range<1>((size_t)blocks * 32, 32),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
            const int b = (int)it.get_group_linear_id();
            const int t = (int)it.get_local_linear_id();
            float v = d_x[b * 32 + t];
            if constexpr (Silu) {
                v = (v / (1.0f + sycl::exp(-v))) * d_up[b * 32 + t];
                d_out[b * 32 + t] = v;
            }
            const float mx = sycl::reduce_over_group(it.get_group(), sycl::fabs(v),
                                                     sycl::maximum<float>());
            const float s = (mx > 0.0f) ? (mx / 127.0f) : 1.0f;
            const int32_t q = (int32_t)sycl::clamp(sycl::rint(v / s), -127.0f, 127.0f);
            if (t == 0) d_scale[b] = s;
            if (d_sum) {
                // Two subgroups of 16 span the tile; each owns one K16 sum.
                const auto sg = it.get_sub_group();
                const int32_t half_sum =
                    sycl::reduce_over_group(sg, q, sycl::plus<int32_t>());
                if (sg.get_local_linear_id() == 0)
                    d_sum[b * 2 + (int)sg.get_group_linear_id()] = half_sum;
            }
            d_q[b * 32 + t] = (int8_t)q;
        });
}

void x_quantize_act_s8(const float *d_x, int K, int8_t *d_q, float *d_scale,
                       int32_t *d_sum) {
    quantize_act_s8<false>(d_x, nullptr, nullptr, K, d_q, d_scale, d_sum);
}

// Shared asymmetric INT4 group fit: weight ~= scale * (q - z) with q and z
// both in [-8, 7]. Returns the FP16 scale bits actually stored (low nibble
// cleared when reserve_nibble, so z can ride there) and z itself. The repack
// and the format probe MUST agree on this math, so both call this.
static inline uint16_t fit_int4_asym(const float *wf, int n, bool reserve_nibble,
                                     int *out_z) {
    float lo = 0.0f;
    float hi = 0.0f;
    for (int j = 0; j < n; ++j) {
        lo = sycl::fmin(lo, wf[j]);
        hi = sycl::fmax(hi, wf[j]);
    }
    float s = (hi - lo) / 15.0f;
    if (!(s > 0.0f)) s = 1.0f;
    for (int iter = 0; iter < 3; ++iter) {
        const int zi = (int)sycl::clamp(sycl::rint(-lo / s) - 8.0f, -8.0f, 7.0f);
        float num = 0.0f;
        float den = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float q = sycl::clamp(sycl::rint(wf[j] / s + (float)zi),
                                        -8.0f, 7.0f);
            const float centered = q - (float)zi;
            num += wf[j] * centered;
            den += centered * centered;
        }
        if (den > 0.0f) s = num / den;
    }
    *out_z = (int)sycl::clamp(sycl::rint(-lo / s) - 8.0f, -8.0f, 7.0f);
    uint16_t sh = sycl::bit_cast<uint16_t>(sycl::half(s));
    if (reserve_nibble) sh = (uint16_t)((sh + 8u) & 0xFFF0u);
    return sh;
}

// Repack one weight: e2m3 fragment payload -> asymmetric s4 DPAS layout.
// Every group stores one FP16 scale whose low four mantissa bits carry the
// signed zero point. Clearing those bits costs at most 0.78% scale precision
// and adds no metadata traffic. mode 1 (k16) splits each K32 tile into two
// groups - CUDA's NVFP4 granularity - at +0.5 bits/w. mode 2 (k64) fits one
// scale per K32 PAIR - the dpas.s4.s4.8.8 tile - at -0.25 bits/w and +13.6%
// weight rel-L2 (grid probe scheme 10); the code payload layout is identical
// (two adjacent lane-major K32 tiles ARE the K64 B fragment).
static void w4_repack_kernel(const tq_qmma_weight_t *w, uint8_t *d_s4,
                             uint16_t *d_sc, int mode) {
    const int M = w->M, Kt = w->Kt;
    const int scale_cols = w->scale_cols;
    const uint8_t *payload = w->d_A;
    const float *bs = w->d_block_scale_inv;
    tq_q().parallel_for(
        sycl::nd_range<1>((size_t)M * Kt, 128), [=](sycl::nd_item<1> it) {
            const size_t gid = it.get_global_linear_id();
            if (gid >= (size_t)M * Kt) return;
            const int row = (int)(gid / Kt);
            const int kt = (int)(gid % Kt);
            const int mt = row >> 4;
            const int sc = (scale_cols == Kt) ? kt : (kt >> 2);
            const float b = bs[(size_t)(mt >> 3) * scale_cols + sc];
            // Including zero in the observed interval preserves exact zero and
            // makes one-sided groups use all 16 codes, not half the grid.
            float wf[32];
            for (int j = 0; j < 32; ++j)
                wf[j] = e2m3f(e2m3_code_at(payload, Kt, row, kt * 32 + j)) * b;
            const int rowgroup = row >> 4;
            const int n = row & 15;
            float gs[2];
            int gz[2];
            if (mode == 1) {
                for (int h = 0; h < 2; ++h) {
                    int zh = 0;
                    const uint16_t shh = fit_int4_asym(wf + h * 16, 16, true, &zh);
                    d_sc[((size_t)rowgroup * (Kt * 2) + kt * 2 + h) * 16u + n] =
                        (uint16_t)(shh | (zh & 0xF));
                    gs[h] = (float)sycl::bit_cast<sycl::half>(shh);
                    gz[h] = zh;
                }
            } else if (mode == 2) {
                // Fit over this tile and its K64 partner. Both items of the
                // pair compute the same fit (double decode, one-time cost);
                // only the even tile writes the shared scale.
                const int kt0 = kt & ~1;
                float wf64[64];
                for (int j = 0; j < 64; ++j)
                    wf64[j] = e2m3f(e2m3_code_at(payload, Kt, row,
                                                 kt0 * 32 + j)) * b;
                int zp = 0;
                const uint16_t sh = fit_int4_asym(wf64, 64, true, &zp);
                if ((kt & 1) == 0)
                    d_sc[((size_t)rowgroup * (Kt / 2) + (kt >> 1)) * 16u + n] =
                        (uint16_t)(sh | (zp & 0xF));
                gs[0] = gs[1] = (float)sycl::bit_cast<sycl::half>(sh);
                gz[0] = gz[1] = zp;
            } else {
                int zp = 0;
                const uint16_t sh = fit_int4_asym(wf, 32, true, &zp);
                d_sc[((size_t)rowgroup * Kt + kt) * 16u + n] =
                    (uint16_t)(sh | (zp & 0xF));
                gs[0] = gs[1] = (float)sycl::bit_cast<sycl::half>(sh);
                gz[0] = gz[1] = zp;
            }
            // Pack against the scale the GEMV actually reads after FP16
            // rounding, not the pre-rounded fit. j is even, so a nibble pair
            // never straddles the K16 boundary.
            //
            // Dword order is LANE-MAJOR: lane n owns dwords [n*4, n*4+4). The
            // DPAS register contract is unchanged (bfrag[i] still holds k-quad
            // i for column n) but the GEMV can now fetch all four with ONE
            // 16-byte vector load instead of four stride-64 dword loads, and
            // the 16 lanes still cover the tile's 256 bytes contiguously.
            const size_t base = ((size_t)rowgroup * Kt + kt) * 256u;
            for (int j = 0; j < 32; j += 2) {
                auto q4 = [&](int jj) -> uint32_t {
                    const int h = jj >> 4;
                    const float r = sycl::rint(wf[jj] / gs[h] + (float)gz[h]);
                    return (uint32_t)(((int)sycl::clamp(r, -8.0f, 7.0f)) & 0xF);
                };
                const size_t byte = base + (size_t)(n * 4 + (j >> 3)) * 4u
                                  + (size_t)((j & 7) >> 1);
                d_s4[byte] = (uint8_t)(q4(j) | (q4(j + 1) << 4));
            }
        });
}

// E2M3 -> S8 in the verified Xe2 depth-major B layout. Per-K32 FP16 scales
// make this a high-fidelity fallback for the few matrices that need more than
// four weight bits; the runtime activation and accumulator path stays shared.
static void w8_repack_kernel(const tq_qmma_weight_t *w, uint8_t *d_s8, uint16_t *d_sc) {
    const int M = w->M, Kt = w->Kt;
    const int scale_cols = w->scale_cols;
    const uint8_t *payload = w->d_A;
    const float *bs = w->d_block_scale_inv;
    tq_q().parallel_for(
        sycl::nd_range<1>((size_t)M * Kt, 128), [=](sycl::nd_item<1> it) {
            const size_t gid = it.get_global_linear_id();
            if (gid >= (size_t)M * Kt) return;
            const int row = (int)(gid / Kt);
            const int kt = (int)(gid % Kt);
            const int mt = row >> 4;
            const int sc = (scale_cols == Kt) ? kt : (kt >> 2);
            const float b = bs[(size_t)(mt >> 3) * scale_cols + sc];

            float wf[32];
            float hi = 0.0f;
            float neg = 0.0f;
            for (int j = 0; j < 32; ++j) {
                wf[j] = e2m3f(e2m3_code_at(payload, Kt, row, kt * 32 + j)) * b;
                hi = sycl::fmax(hi, wf[j]);
                neg = sycl::fmax(neg, -wf[j]);
            }
            float s = sycl::fmax(hi / 127.0f, neg / 128.0f);
            if (!(s > 0.0f)) s = 1.0f;
            for (int iter = 0; iter < 2; ++iter) {
                float num = 0.0f;
                float den = 0.0f;
                for (int j = 0; j < 32; ++j) {
                    const float q = sycl::clamp(sycl::rint(wf[j] / s), -128.0f, 127.0f);
                    num += wf[j] * q;
                    den += q * q;
                }
                if (den > 0.0f) s = num / den;
            }
            const uint16_t sh = sycl::bit_cast<uint16_t>(sycl::half(s));
            d_sc[((size_t)(row >> 4) * Kt + kt) * 16u + (row & 15)] = sh;
            const float stored_s = (float)sycl::bit_cast<sycl::half>(sh);

            const int rowgroup = row >> 4;
            const int n = row & 15;
            const size_t base = ((size_t)rowgroup * Kt + kt) * 512u;
            for (int j = 0; j < 32; ++j) {
                const float r = sycl::rint(wf[j] / stored_s);
                const int q = (int)sycl::clamp(r, -128.0f, 127.0f);
                const size_t byte = base + (size_t)(n * 8 + (j >> 2)) * 4u
                                  + (size_t)(j & 3);
                d_s8[byte] = (uint8_t)(int8_t)q;
            }
        });
}

// Host repack: an eligible matrix keeps exactly one DPAS payload. Source E2M3
// codes and block scales are released only after the conversion completes.
int x_w4_repack_weight_mode(tq_qmma_weight_t *w, int mode) {
    if (!w || !w->d_A || !w->d_block_scale_inv || !w->block_scaled || !w->e2m3) return -1;
    if (w->M % 16 || w->K % 128) return -2;
    if (mode == 2 && (w->Kt & 1)) return -3;   // K64 needs an even tile count
    const size_t code_bytes = (size_t)w->Mt * w->Kt * 256u;
    const size_t scale_groups =
        (mode == 1) ? (size_t)w->Kt * 2u : (mode == 2) ? (size_t)w->Kt / 2u
                                                       : (size_t)w->Kt;
    const size_t scale_bytes = (size_t)w->M * scale_groups * sizeof(uint16_t);
    w->d_s4 = static_cast<uint8_t *>(tq_dev_alloc(code_bytes, "w4.codes"));
    w->d_s4_scale = static_cast<uint16_t *>(tq_dev_alloc(scale_bytes, "w4.scales"));
    w4_repack_kernel(w, w->d_s4, w->d_s4_scale, mode);
    tq_q().wait_and_throw();
    tq_dev_free(w->d_A);
    tq_dev_free(w->d_block_scale_inv);
    w->d_A = nullptr;
    w->d_block_scale_inv = nullptr;
    w->s4_k16 = (mode == 1) ? 1 : 0;
    w->s4_k64 = (mode == 2) ? 1 : 0;
    w->s4_ready = 1;
    return 0;
}

int x_w4_repack_weight(tq_qmma_weight_t *w) {
    return x_w4_repack_weight_mode(w, 0);
}

// ---- tensor-parallel weight sharding ------------------------------------
// Slices an ALREADY-REPACKED W4 weight into one rank's shard. No repack is
// needed because the packed layout is row-group-major and k-tile-major
// inside each row-group:
//     codes  : row-group g at  g * Kt * 256 bytes
//     scales : row-group g at  g * (Kt >> ksh) * 16 uint16
// so a column shard (output rows) is a CONTIGUOUS range and a row shard
// (the K dimension) is Mt contiguous runs at a fixed stride. Both are pure
// memcpy.
//
// mode 0 = column-parallel: shard output rows. `groups`/`ngroups` describe a
//   PACKED output (q_proj is [q | gate], linear_in_qkv is [q | k | v]); each
//   group is sharded independently so a rank gets a slice of every group,
//   not the whole of one. Each group's row count must be a multiple of
//   16*tp. This is what vLLM's QKVParallelLinear output_sizes list does.
// mode 1 = row-parallel: shard K. Kt must be divisible by tp (and by 2*tp
//   under the K64 scale mode, which pairs k-tiles).
//
// dst's buffers are allocated on the CURRENT rank, so call this with
// tq_set_rank(r) already applied. src is left untouched.
int x_w4_shard(tq_qmma_weight_t *dst, const tq_qmma_weight_t *src,
               int rank, int tp, int mode, const int *groups, int ngroups) {
    if (!dst || !src || !src->s4_ready || !src->d_s4 || !src->d_s4_scale)
        return -1;
    if (tp < 1 || rank < 0 || rank >= tp) return -2;
    const int ksh = src->s4_k64 ? 1 : 0;          // k64 pairs k-tiles
    if (src->s4_k16) return -3;                    // k16 scale layout unhandled
    const int Kt = src->Kt;
    const int sg = Kt >> ksh;                      // scale groups per row
    *dst = *src;                                   // inherit flags/shape
    dst->d_A = nullptr;
    dst->d_block_scale_inv = nullptr;

    if (mode == 0) {
        int gs[8], ng = (groups && ngroups > 0) ? ngroups : 1;
        if (ng > 8) return -4;
        if (groups && ngroups > 0) {
            int total = 0;
            for (int i = 0; i < ng; ++i) { gs[i] = groups[i]; total += gs[i]; }
            if (total != src->M) return -5;
        } else {
            gs[0] = src->M;
        }
        for (int i = 0; i < ng; ++i)
            if (gs[i] % (16 * tp)) return -6;      // must split on row-groups
        const int Mnew = src->M / tp;
        dst->M = Mnew;
        dst->Mt = Mnew / 16;
        const size_t code_bytes = (size_t)dst->Mt * Kt * 256u;
        const size_t scale_cnt = (size_t)Mnew * sg;
        dst->d_s4 = static_cast<uint8_t *>(tq_dev_alloc(code_bytes, "shard.codes"));
        dst->d_s4_scale =
            static_cast<uint16_t *>(tq_dev_alloc(scale_cnt * 2, "shard.scales"));
        size_t src_g = 0, dst_g = 0;               // row-group cursors
        for (int i = 0; i < ng; ++i) {
            const size_t gtot = (size_t)gs[i] / 16;        // row-groups here
            const size_t gper = gtot / (size_t)tp;         // this rank's share
            const size_t from = src_g + gper * (size_t)rank;
            tq_q().memcpy(dst->d_s4 + dst_g * Kt * 256u,
                          src->d_s4 + from * Kt * 256u,
                          gper * Kt * 256u);
            tq_q().memcpy(dst->d_s4_scale + dst_g * (size_t)sg * 16u,
                          src->d_s4_scale + from * (size_t)sg * 16u,
                          gper * (size_t)sg * 16u * sizeof(uint16_t));
            src_g += gtot;
            dst_g += gper;
        }
        tq_q().wait_and_throw();
        return 0;
    }

    if (mode == 1) {
        if (Kt % tp) return -7;
        if (ksh && (Kt / tp) % 2) return -8;       // k64 needs paired tiles
        const int Ktnew = Kt / tp;
        const int sgnew = Ktnew >> ksh;
        dst->K = src->K / tp;
        dst->Kt = Ktnew;
        dst->scale_cols = sgnew;
        const size_t code_bytes = (size_t)src->Mt * Ktnew * 256u;
        const size_t scale_cnt = (size_t)src->M * sgnew;
        dst->d_s4 = static_cast<uint8_t *>(tq_dev_alloc(code_bytes, "shard.codes"));
        dst->d_s4_scale =
            static_cast<uint16_t *>(tq_dev_alloc(scale_cnt * 2, "shard.scales"));
        const size_t kt0 = (size_t)Ktnew * (size_t)rank;
        for (int g = 0; g < src->Mt; ++g) {
            tq_q().memcpy(dst->d_s4 + (size_t)g * Ktnew * 256u,
                          src->d_s4 + ((size_t)g * Kt + kt0) * 256u,
                          (size_t)Ktnew * 256u);
            tq_q().memcpy(dst->d_s4_scale + (size_t)g * sgnew * 16u,
                          src->d_s4_scale +
                              ((size_t)g * sg + (kt0 >> ksh)) * 16u,
                          (size_t)sgnew * 16u * sizeof(uint16_t));
        }
        tq_q().wait_and_throw();
        return 0;
    }
    if (mode == 2) {
        // Replicate: a full copy onto this rank. Needed for the GDN gate and
        // decay projections, which emit ONE scalar per value head (M=48).
        // 48/2 = 24 rows is not a multiple of the packed layout's 16-row
        // group, so they cannot be head-sharded at all - and at 123 KB each
        // replication costs ~12 MB per card across all 48 GDN layers, which
        // is nothing. Each rank computes all 48 head gates and reads the
        // slice belonging to its own heads.
        const size_t code_bytes = (size_t)src->Mt * Kt * 256u;
        const size_t scale_cnt = (size_t)src->M * sg;
        dst->d_s4 = static_cast<uint8_t *>(tq_dev_alloc(code_bytes, "rep.codes"));
        dst->d_s4_scale =
            static_cast<uint16_t *>(tq_dev_alloc(scale_cnt * 2, "rep.scales"));
        tq_q().memcpy(dst->d_s4, src->d_s4, code_bytes);
        tq_q().memcpy(dst->d_s4_scale, src->d_s4_scale,
                      scale_cnt * sizeof(uint16_t));
        tq_q().wait_and_throw();
        return 0;
    }
    return -9;
}

void x_w4_shard_free(tq_qmma_weight_t *w) {
    if (!w) return;
    tq_dev_free(w->d_s4);
    tq_dev_free(w->d_s4_scale);
    w->d_s4 = nullptr;
    w->d_s4_scale = nullptr;
    w->s4_ready = 0;
}

int x_w8_repack_weight(tq_qmma_weight_t *w) {
    if (!w || !w->d_A || !w->d_block_scale_inv || !w->block_scaled || !w->e2m3) return -1;
    if (w->M % 16 || w->K % 128) return -2;
    const size_t code_bytes = (size_t)w->Mt * w->Kt * 512u;
    const size_t scale_bytes = (size_t)w->M * w->Kt * sizeof(uint16_t);
    w->d_s8 = static_cast<uint8_t *>(tq_dev_alloc(code_bytes, "w8.codes"));
    w->d_s8_scale = static_cast<uint16_t *>(tq_dev_alloc(scale_bytes, "w8.scales"));
    w8_repack_kernel(w, w->d_s8, w->d_s8_scale);
    tq_q().wait_and_throw();
    tq_dev_free(w->d_A);
    tq_dev_free(w->d_block_scale_inv);
    w->d_A = nullptr;
    w->d_block_scale_inv = nullptr;
    w->s8_ready = 1;
    return 0;
}

// Split-K keeps narrow-output projections resident across the 256 XVEs.
//
// The reduction is FUSED. With splits dividing kSubgroupsPerWorkgroup, task =
// g*splits + split means one work-group holds exactly kSubgroupsPerWorkgroup/
// splits COMPLETE row-groups - every split of those groups is a subgroup of
// this work-group. So the partials never round-trip through global memory and
// no second kernel launch is needed. Summing i = 0..splits-1 out of SLM walks
// the same order the old global reduction did, so results stay bit-identical.
//
// splits > kSubgroupsPerWorkgroup cannot close locally; those shapes keep the
// global partial buffer and the caller's reduction pass.
void x_gemv_w4a8_core(const uint8_t *d_s4, const uint16_t *d_sc, const int8_t *d_q,
                      const float *d_as, const int32_t *d_sum,
                      float *d_y, const float *d_residual, float *d_partial,
                      int M, int K, int splits, int ksh) {
    const int Kt = K / 32;
    const int groups = M / 16;
    const int tasks = groups * splits;
    const size_t workgroups = (tasks + kSubgroupsPerWorkgroup - 1) /
                              kSubgroupsPerWorkgroup;
    const int fused = (splits > 1 && splits <= kSubgroupsPerWorkgroup) ? 1 : 0;
    const int gpw = fused ? (kSubgroupsPerWorkgroup / splits) : 0;
    tq_q().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm(kSubgroupsPerWorkgroup * kSG, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(workgroups * kSG * kSubgroupsPerWorkgroup,
                              kSG * kSubgroupsPerWorkgroup),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]]
                {
                const auto sg = it.get_sub_group();
                const int sgid = (int)sg.get_group_linear_id();
                const int wg = (int)it.get_group_linear_id();
                const int task = wg * kSubgroupsPerWorkgroup + sgid;
                const int lane = (int)sg.get_local_linear_id();
                float acc = 0.0f;
                if (task < tasks) {
                    const int g = task / splits;
                    const int split = task - g * splits;
                    const int kt_begin = Kt * split / splits;
                    const int kt_end = Kt * (split + 1) / splits;
                    // One k-tile per iteration left a single 288-byte load in
                    // flight and a serial FP dependency on acc. Two tiles with
                    // two accumulators double the outstanding loads and let the
                    // adds pipeline. Reassociated sum: eps-level drift.
                    const uint8_t *bbase = d_s4 + (size_t)g * Kt * 256u;
                    const uint16_t *sbase =
                        d_sc + (size_t)g * (Kt >> ksh) * 16u;
                    auto tile = [&](int kt) -> float {
                        const uint16_t afrag = *reinterpret_cast<const uint16_t *>(
                            d_q + (size_t)kt * 32 + 2 * lane);
                        const uintv4 bfrag = *reinterpret_cast<const uintv4 *>(
                            bbase + (size_t)kt * 256u + (size_t)lane * 16u);
                        int32_t dfrag = 0;
                        DPAS_S4S8_RC1(dfrag, afrag, bfrag);
                        const uint16_t packed =
                            sbase[(size_t)(kt >> ksh) * 16u + lane];
                        int zp = (int)(packed & 0xFu);
                        if (zp >= 8) zp -= 16;
                        const float ws = (float)sycl::bit_cast<sycl::half>(
                            (uint16_t)(packed & 0xFFF0u));
                        const int32_t asum = d_sum[2 * kt] + d_sum[2 * kt + 1];
                        return (float)(dfrag - zp * asum) * ws * d_as[kt];
                    };
                    // Rejected: an explicit L2 prefetch of the B tile two
                    // iterations ahead (the Triton backend's
                    // Matrix2DBlockPrefetchOp idiom) measured SLOWER - 31.2 vs
                    // 31.6 tok/s, and this shape 12.73 vs 12.25 ms. The tile
                    // walk is linear, so Xe2's L2 prefetcher already covers it
                    // and the extra issue slots cost more than they save.
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    int kt = kt_begin;
                    for (; kt + 1 < kt_end; kt += 2) {
                        acc0 += tile(kt);
                        acc1 += tile(kt + 1);
                    }
                    for (; kt < kt_end; ++kt) acc0 += tile(kt);
                    acc = acc0 + acc1;
                }
                if (!fused) {
                    if (task < tasks) {
                        const int g = task / splits;
                        const int split = task - g * splits;
                        float *dst = (splits == 1) ? d_y : d_partial;
                        dst[(size_t)split * M + g * 16 + lane] = acc;
                    }
                    return;
                }
                slm[sgid * kSG + lane] = acc;
                it.barrier(sycl::access::fence_space::local_space);
                if (sgid >= gpw) return;
                const int g = wg * gpw + sgid;
                if (g >= groups) return;
                float sum = 0.0f;
                for (int i = 0; i < splits; ++i)
                    sum += slm[(sgid * splits + i) * kSG + lane];
                const int row = g * 16 + lane;
                d_y[row] = sum + (d_residual ? d_residual[row] : 0.0f);
            });
    });
}

// K16 sibling at CUDA's NVFP4 scale granularity. The B fragment is fetched
// once and then masked into its two K16 halves: dwords 0-1 carry k=0..15 and
// dwords 2-3 carry k=16..31 (dpas_layout_probe established that B's nibble for
// k lives at dword k/8). Zeroing the other pair makes one DPAS see one half, so
// each half gets its own scale and zero point for ZERO extra weight traffic -
// the cost is one more DPAS issue, which a bandwidth-bound GEMV has room for.
void x_gemv_w4a8_core_k16(const uint8_t *d_s4, const uint16_t *d_sc,
                          const int8_t *d_q, const float *d_as,
                          const int32_t *d_sum,
                          float *d_y, const float *d_residual, float *d_partial,
                          int M, int K, int splits) {
    const int Kt = K / 32;
    const int groups = M / 16;
    const int tasks = groups * splits;
    const size_t workgroups = (tasks + kSubgroupsPerWorkgroup - 1) /
                              kSubgroupsPerWorkgroup;
    const int fused = (splits > 1 && splits <= kSubgroupsPerWorkgroup) ? 1 : 0;
    const int gpw = fused ? (kSubgroupsPerWorkgroup / splits) : 0;
    tq_q().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm(kSubgroupsPerWorkgroup * kSG, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(workgroups * kSG * kSubgroupsPerWorkgroup,
                              kSG * kSubgroupsPerWorkgroup),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]]
                {
                const auto sg = it.get_sub_group();
                const int sgid = (int)sg.get_group_linear_id();
                const int wg = (int)it.get_group_linear_id();
                const int task = wg * kSubgroupsPerWorkgroup + sgid;
                const int lane = (int)sg.get_local_linear_id();
                float acc = 0.0f;
                if (task < tasks) {
                    const int g = task / splits;
                    const int split = task - g * splits;
                    const int kt_begin = Kt * split / splits;
                    const int kt_end = Kt * (split + 1) / splits;
                    for (int kt = kt_begin; kt < kt_end; ++kt) {
                        const uint16_t afrag = *reinterpret_cast<const uint16_t *>(
                            d_q + (size_t)kt * 32 + 2 * lane);
                        const uintv4 bfull = *reinterpret_cast<const uintv4 *>(
                            d_s4 + ((size_t)g * Kt + kt) * 256u
                                 + (size_t)lane * 16u);
                        uintv4 blo;
                        uintv4 bhi;
                        blo[0] = bfull[0];
                        blo[1] = bfull[1];
                        blo[2] = 0u;
                        blo[3] = 0u;
                        bhi[0] = 0u;
                        bhi[1] = 0u;
                        bhi[2] = bfull[2];
                        bhi[3] = bfull[3];
                        int32_t dlo = 0;
                        int32_t dhi = 0;
                        DPAS_S4S8_RC1(dlo, afrag, blo);
                        DPAS_S4S8_RC1(dhi, afrag, bhi);
                        const float as = d_as[kt];
                        for (int half = 0; half < 2; ++half) {
                            const uint16_t packed =
                                d_sc[((size_t)g * (Kt * 2) + kt * 2 + half) * 16u + lane];
                            int zp = (int)(packed & 0xFu);
                            if (zp >= 8) zp -= 16;
                            const float ws = (float)sycl::bit_cast<sycl::half>(
                                (uint16_t)(packed & 0xFFF0u));
                            const int32_t d = half ? dhi : dlo;
                            acc += (float)(d - zp * d_sum[2 * kt + half]) * ws * as;
                        }
                    }
                }
                if (!fused) {
                    if (task < tasks) {
                        const int g = task / splits;
                        const int split = task - g * splits;
                        float *dst = (splits == 1) ? d_y : d_partial;
                        dst[(size_t)split * M + g * 16 + lane] = acc;
                    }
                    return;
                }
                slm[sgid * kSG + lane] = acc;
                it.barrier(sycl::access::fence_space::local_space);
                if (sgid >= gpw) return;
                const int g = wg * gpw + sgid;
                if (g >= groups) return;
                float sum = 0.0f;
                for (int i = 0; i < splits; ++i)
                    sum += slm[(sgid * splits + i) * kSG + lane];
                const int row = g * 16 + lane;
                d_y[row] = sum + (d_residual ? d_residual[row] : 0.0f);
            });
    });
}

// W8A8 sibling. The A fragment and activation scales are identical to W4A8;
// only the B fragment doubles to eight dwords per lane.
void x_gemv_w8a8_core(const uint8_t *d_s8, const uint16_t *d_sc, const int8_t *d_q,
                      const float *d_as,
                      float *d_y, const float *d_residual, float *d_partial,
                      int M, int K, int splits) {
    const int Kt = K / 32;
    const int groups = M / 16;
    const int tasks = groups * splits;
    const size_t workgroups = (tasks + kSubgroupsPerWorkgroup - 1) /
                              kSubgroupsPerWorkgroup;
    const int fused = (splits > 1 && splits <= kSubgroupsPerWorkgroup) ? 1 : 0;
    const int gpw = fused ? (kSubgroupsPerWorkgroup / splits) : 0;
    tq_q().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm(kSubgroupsPerWorkgroup * kSG, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(workgroups * kSG * kSubgroupsPerWorkgroup,
                              kSG * kSubgroupsPerWorkgroup),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]]
                {
                const auto sg = it.get_sub_group();
                const int sgid = (int)sg.get_group_linear_id();
                const int wg = (int)it.get_group_linear_id();
                const int task = wg * kSubgroupsPerWorkgroup + sgid;
                const int lane = (int)sg.get_local_linear_id();
                float acc = 0.0f;
                if (task < tasks) {
                    const int g = task / splits;
                    const int split = task - g * splits;
                    const int kt_begin = Kt * split / splits;
                    const int kt_end = Kt * (split + 1) / splits;
                    const uint8_t *bbase = d_s8 + (size_t)g * Kt * 512u;
                    const uint16_t *sbase = d_sc + (size_t)g * Kt * 16u;
                    auto tile = [&](int kt) -> float {
                        const uint16_t afrag = *reinterpret_cast<const uint16_t *>(
                            d_q + (size_t)kt * 32 + 2 * lane);
                        const uintv8 bfrag = *reinterpret_cast<const uintv8 *>(
                            bbase + (size_t)kt * 512u + (size_t)lane * 32u);
                        int32_t dfrag = 0;
                        DPAS_S8S8_RC1(dfrag, afrag, bfrag);
                        const uint16_t sh = sbase[(size_t)kt * 16u + lane];
                        const float ws = (float)sycl::bit_cast<sycl::half>(sh);
                        return (float)dfrag * ws * d_as[kt];
                    };
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    int kt = kt_begin;
                    for (; kt + 1 < kt_end; kt += 2) {
                        acc0 += tile(kt);
                        acc1 += tile(kt + 1);
                    }
                    for (; kt < kt_end; ++kt) acc0 += tile(kt);
                    acc = acc0 + acc1;
                }
                if (!fused) {
                    if (task < tasks) {
                        const int g = task / splits;
                        const int split = task - g * splits;
                        float *dst = (splits == 1) ? d_y : d_partial;
                        dst[(size_t)split * M + g * 16 + lane] = acc;
                    }
                    return;
                }
                slm[sgid * kSG + lane] = acc;
                it.barrier(sycl::access::fence_space::local_space);
                if (sgid >= gpw) return;
                const int g = wg * gpw + sgid;
                if (g >= groups) return;
                float sum = 0.0f;
                for (int i = 0; i < splits; ++i)
                    sum += slm[(sgid * splits + i) * kSG + lane];
                const int row = g * 16 + lane;
                d_y[row] = sum + (d_residual ? d_residual[row] : 0.0f);
            });
    });
}

void x_prepare_gemv_act_s8(const float *d_x, int K) {
    ensure_act_scratch(K);
    x_quantize_act_s8(d_x, K, g_act_q, g_act_scale, g_act_sum);
}

void x_silu_mul_quant(float *d_out, const float *d_gate, const float *d_up, int K) {
    if (!d_out || !d_gate || !d_up || K <= 0 || (K % 32) != 0) return;
    ensure_act_scratch(K);
    quantize_act_s8<true>(d_gate, d_up, d_out, K, g_act_q, g_act_scale, g_act_sum);
}

// Fused (1+w) RMSNorm + S8 activation quantization.
//
// The unfused pair cost 16.4 us (norm) + 8.3 us (quant) per call for 20 KB of
// work: x_rmsnorm ran ONE 256-thread work-group - 2 of 256 XVEs - and read the
// input twice, then the quantizer re-read the normalized vector from DRAM.
// Here the reduction is spread over every XVE and the quantizer rides the
// normalize pass, so the normalized vector never leaves registers. One
// subgroup is exactly one K16 sum block; two adjacent subgroups are one K32
// scale block, so both activation operands fall out of subgroup reductions.
void x_rmsnorm_quant(float *d_out, const float *d_in, const uint16_t *d_w,
                     int H, float eps) {
    if (!d_out || !d_in || !d_w || H <= 0 || (H % 32) != 0) return;
    // Rejected: collapsing this into ONE 1024-thread work-group. It halves the
    // launch count but drops from 40 XVEs to 8 and needs two barriers per
    // 1024-element pass; measured norm 1.356 -> 2.005 ms and 32.64 -> 32.43
    // tok/s. The pair is not launch-bound enough to pay for the lost width.
    constexpr size_t kWG = 256;
    const size_t wgs = ((size_t)H + kWG - 1) / kWG;
    ensure_norm_partials(wgs);
    ensure_act_scratch(H);
    float *part = g_norm_partial;
    int8_t *d_q = g_act_q;
    float *d_scale = g_act_scale;
    int32_t *d_sum = g_act_sum;
    const int nblk = (int)wgs;

    tq_q().parallel_for(
        sycl::nd_range<1>(wgs * kWG, kWG), [=](sycl::nd_item<1> it) {
            const int i = (int)it.get_global_linear_id();
            const float v = (i < H) ? d_in[i] : 0.0f;
            const float s = sycl::reduce_over_group(it.get_group(), v * v,
                                                    sycl::plus<float>());
            if (it.get_local_linear_id() == 0)
                part[it.get_group_linear_id()] = s;
        });

    tq_q().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> amax(kWG / kSG, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(wgs * kWG, kWG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]]
                {
                float sum_sq = 0.0f;
                for (int b = 0; b < nblk; ++b) sum_sq += part[b];
                const float rms_inv =
                    sycl::rsqrt(sum_sq / static_cast<float>(H) + eps);
                const int i = (int)it.get_global_linear_id();
                const auto sg = it.get_sub_group();
                const int sgid = (int)sg.get_group_linear_id();
                const int lane = (int)sg.get_local_linear_id();
                const float v = (i < H)
                    ? d_in[i] * rms_inv * (1.0f + bf16f(d_w[i])) : 0.0f;
                if (i < H) d_out[i] = v;
                const float sg_amax = sycl::reduce_over_group(
                    sg, sycl::fabs(v), sycl::maximum<float>());
                if (lane == 0) amax[sgid] = sg_amax;
                it.barrier(sycl::access::fence_space::local_space);
                const float blk = sycl::fmax(amax[sgid & ~1], amax[sgid | 1]);
                const float sc = (blk > 0.0f) ? (blk / 127.0f) : 1.0f;
                const int32_t q =
                    (int32_t)sycl::clamp(sycl::rint(v / sc), -127.0f, 127.0f);
                const int32_t half_sum =
                    sycl::reduce_over_group(sg, q, sycl::plus<int32_t>());
                if (i < H) {
                    d_q[i] = (int8_t)q;
                    if (lane == 0) {
                        d_sum[i >> 4] = half_sum;
                        if ((sgid & 1) == 0) d_scale[i >> 5] = sc;
                    }
                }
            });
    });
}

// Fan-out GEMV: several W4 projections that share one already-quantized
// activation, computed in ONE launch.
//
// The four DeltaNet input projections (in_qkv 10240, in_z 6144, in_a 48,
// in_b 48) all read the same normalized activation and all have K=5120, so
// they were four launches per recurrent layer - 192 per token - and the two
// 48-row ones are pure launch overhead (M=48 is 3 row-groups; 1.03 ms of
// wall time for 0.02 ms of arithmetic).
//
// No weight concatenation is needed: row-groups are numbered across the whole
// set and a 4-entry prefix sum maps a global row-group back to its weight, so
// each subgroup picks up the right payload, scales and output pointer. The
// fused SLM reduction is unchanged; a work-group straddling two weights just
// resolves each of its row-groups independently.
int x_gemv_w4a8_fanout(const tq_qmma_weight_t *const *ws, float *const *ys,
                       int count) {
    if (!ws || !ys || count <= 0 || count > kFanoutMax) return -1;
    int off[kFanoutMax + 1];
    const uint8_t *codes[kFanoutMax];
    const uint16_t *scales[kFanoutMax];
    off[0] = 0;
    const int K = ws[0]->K;
    const int ksh = ws[0]->s4_k64 ? 1 : 0;
    for (int i = 0; i < count; ++i) {
        const tq_qmma_weight_t *w = ws[i];
        // Uniform tier and shared K are what make one launch legal.
        if (!w || !w->s4_ready || w->s4_k16 || !w->d_s4 || !w->d_s4_scale ||
            !ys[i] || w->K != K || (w->M % 16) ||
            (w->s4_k64 ? 1 : 0) != ksh)
            return -2;
        codes[i] = w->d_s4;
        scales[i] = w->d_s4_scale;
        off[i + 1] = off[i] + w->M / 16;
    }
    if (g_act_cap < K || !g_act_q || !g_act_scale) return -3;

    const int Kt = K / 32;
    const int groups = off[count];
    const int splits = choose_gemv_splits(groups);
    if (splits > kSubgroupsPerWorkgroup) return -4;   // needs the fused path
    const int gpw = (splits > 1) ? (kSubgroupsPerWorkgroup / splits) : 1;
    const int tasks = groups * splits;
    const size_t workgroups = (tasks + kSubgroupsPerWorkgroup - 1) /
                              kSubgroupsPerWorkgroup;
    const int8_t *d_q = g_act_q;
    const float *d_as = g_act_scale;
    const int32_t *d_sum = g_act_sum;
    // Plain arrays are captured by value into the kernel.
    const int o0 = off[0], o1 = off[1];
    const int o2 = count > 1 ? off[2] : o1;
    const int o3 = count > 2 ? off[3] : o2;
    const int o4 = count > 3 ? off[4] : o3;
    const uint8_t *c0 = codes[0];
    const uint8_t *c1 = count > 1 ? codes[1] : c0;
    const uint8_t *c2 = count > 2 ? codes[2] : c0;
    const uint8_t *c3 = count > 3 ? codes[3] : c0;
    const uint16_t *s0 = scales[0];
    const uint16_t *s1 = count > 1 ? scales[1] : s0;
    const uint16_t *s2 = count > 2 ? scales[2] : s0;
    const uint16_t *s3 = count > 3 ? scales[3] : s0;
    float *y0 = ys[0];
    float *y1 = count > 1 ? ys[1] : y0;
    float *y2 = count > 2 ? ys[2] : y0;
    float *y3 = count > 3 ? ys[3] : y0;

    tq_q().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> slm(kSubgroupsPerWorkgroup * kSG, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(workgroups * kSG * kSubgroupsPerWorkgroup,
                              kSG * kSubgroupsPerWorkgroup),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]]
                {
                const auto sg = it.get_sub_group();
                const int sgid = (int)sg.get_group_linear_id();
                const int wg = (int)it.get_group_linear_id();
                const int task = wg * kSubgroupsPerWorkgroup + sgid;
                const int lane = (int)sg.get_local_linear_id();
                // Resolve a global row-group to its weight.
                auto pick = [&](int g, const uint8_t **cp, const uint16_t **sp,
                                float **yp) -> int {
                    if (g < o1) { *cp = c0; *sp = s0; *yp = y0; return g - o0; }
                    if (g < o2) { *cp = c1; *sp = s1; *yp = y1; return g - o1; }
                    if (g < o3) { *cp = c2; *sp = s2; *yp = y2; return g - o2; }
                    *cp = c3; *sp = s3; *yp = y3; return g - o3;
                };
                float acc = 0.0f;
                if (task < tasks) {
                    const int g = task / splits;
                    const int split = task - g * splits;
                    const uint8_t *cb;
                    const uint16_t *sb;
                    float *yb;
                    const int lg = pick(g, &cb, &sb, &yb);
                    const uint8_t *bbase = cb + (size_t)lg * Kt * 256u;
                    const uint16_t *sbase =
                        sb + (size_t)lg * (Kt >> ksh) * 16u;
                    const int kt_begin = Kt * split / splits;
                    const int kt_end = Kt * (split + 1) / splits;
                    int kt = kt_begin;
                    // Put two independent operand sets in flight, but keep
                    // the original serial floating-point accumulation order.
                    for (; kt + 1 < kt_end; kt += 2) {
                        const uint16_t a0 = *reinterpret_cast<const uint16_t *>(
                            d_q + (size_t)kt * 32 + 2 * lane);
                        const uint16_t a1 = *reinterpret_cast<const uint16_t *>(
                            d_q + (size_t)(kt + 1) * 32 + 2 * lane);
                        const uintv4 b0 = *reinterpret_cast<const uintv4 *>(
                            bbase + (size_t)kt * 256u + (size_t)lane * 16u);
                        const uintv4 b1 = *reinterpret_cast<const uintv4 *>(
                            bbase + (size_t)(kt + 1) * 256u + (size_t)lane * 16u);
                        const uint16_t p0 = sbase[(size_t)(kt >> ksh) * 16u + lane];
                        const uint16_t p1 = sbase[(size_t)((kt + 1) >> ksh) * 16u + lane];
                        const int32_t sum0 = d_sum[2 * kt] + d_sum[2 * kt + 1];
                        const int32_t sum1 = d_sum[2 * kt + 2] + d_sum[2 * kt + 3];
                        const float as0 = d_as[kt], as1 = d_as[kt + 1];
                        int32_t d0 = 0, d1 = 0;
                        DPAS_S4S8_RC1(d0, a0, b0);
                        DPAS_S4S8_RC1(d1, a1, b1);
                        int z0 = (int)(p0 & 0xFu), z1 = (int)(p1 & 0xFu);
                        if (z0 >= 8) z0 -= 16;
                        if (z1 >= 8) z1 -= 16;
                        const float ws0 = (float)sycl::bit_cast<sycl::half>(
                            (uint16_t)(p0 & 0xFFF0u));
                        const float ws1 = (float)sycl::bit_cast<sycl::half>(
                            (uint16_t)(p1 & 0xFFF0u));
                        acc += (float)(d0 - z0 * sum0) * ws0 * as0;
                        acc += (float)(d1 - z1 * sum1) * ws1 * as1;
                    }
                    for (; kt < kt_end; ++kt) {
                        const uint16_t afrag = *reinterpret_cast<const uint16_t *>(
                            d_q + (size_t)kt * 32 + 2 * lane);
                        const uintv4 bfrag = *reinterpret_cast<const uintv4 *>(
                            bbase + (size_t)kt * 256u + (size_t)lane * 16u);
                        int32_t dfrag = 0;
                        DPAS_S4S8_RC1(dfrag, afrag, bfrag);
                        const uint16_t packed =
                            sbase[(size_t)(kt >> ksh) * 16u + lane];
                        int zp = (int)(packed & 0xFu);
                        if (zp >= 8) zp -= 16;
                        const float ws_ = (float)sycl::bit_cast<sycl::half>(
                            (uint16_t)(packed & 0xFFF0u));
                        const int32_t asum = d_sum[2 * kt] + d_sum[2 * kt + 1];
                        acc += (float)(dfrag - zp * asum) * ws_ * d_as[kt];
                    }
                }
                if (splits == 1) {
                    if (task < tasks) {
                        const uint8_t *cb;
                        const uint16_t *sb;
                        float *yb;
                        const int lg = pick(task, &cb, &sb, &yb);
                        yb[lg * 16 + lane] = acc;
                    }
                    return;
                }
                slm[sgid * kSG + lane] = acc;
                it.barrier(sycl::access::fence_space::local_space);
                if (sgid >= gpw) return;
                const int g = wg * gpw + sgid;
                if (g >= groups) return;
                float sum = 0.0f;
                for (int i = 0; i < splits; ++i)
                    sum += slm[(sgid * splits + i) * kSG + lane];
                const uint8_t *cb;
                const uint16_t *sb;
                float *yb;
                const int lg = pick(g, &cb, &sb, &yb);
                yb[lg * 16 + lane] = sum;
            });
    });
    return 0;
}

// ---------------------------------------------------------------------------
// Batched W4A8 GEMM: Y[T][M] = W[M,K] @ X[K,T].
//
// This is the primitive the decode GEMV cannot be: at B=1 a step streams
// 15.05 GB of weights to do 0.14 ms of DPAS, a 178x imbalance. RC8 issues the
// SAME 256-byte B fragment against EIGHT activation rows, so weight bytes per
// output fall 8x and the kernel becomes compute-bound instead of
// bandwidth-bound. It is the shared foundation for chunked prefill (TTFT) and
// for batched decode (concurrency).
//
// Activation layout is dictated by the probed A operand: 8 rows of 32
// contiguous bytes per instruction, so the quantizer writes K-tile-major,
// then token-major - Aq[kt][token][32]. Lane l, vector element i then holds
// token (2i + l/8), bytes (l%8)*4..+3, which makes each element a single
// 64-byte coalesced read across the subgroup.

// Batched (1+w) RMSNorm over T token rows: out[t][*] = norm(in[t][*]).
// One work-group per token so each row's reduction stays inside a group; the
// grid is T*256 wide, which is far better occupancy than the single-token
// path can reach and needs no cross-work-group partials.
void x_rmsnorm_chunk(float *d_out, const float *d_in, const uint16_t *d_w,
                     int H, int T, float eps) {
    if (!d_out || !d_in || !d_w || H <= 0 || T <= 0) return;
    constexpr size_t kWG = 256;
    tq_q().parallel_for(
        sycl::nd_range<1>((size_t)T * kWG, kWG), [=](sycl::nd_item<1> it) {
            const int tok = (int)it.get_group_linear_id();
            const int tid = (int)it.get_local_linear_id();
            const float *row = d_in + (size_t)tok * H;
            float ss = 0.0f;
            for (int i = tid; i < H; i += (int)kWG) {
                const float v = row[i];
                ss += v * v;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float rms_inv = sycl::rsqrt(ss / static_cast<float>(H) + eps);
            float *dst = d_out + (size_t)tok * H;
            for (int i = tid; i < H; i += (int)kWG)
                dst[i] = row[i] * rms_inv * (1.0f + bf16f(d_w[i]));
        });
}

// One subgroup owns a complete quantization block. The staging layout and
// scalar scale/round/clamp expressions remain unchanged; only max and exact
// integer-sum reductions move from a work-group to a subgroup.
template <bool S4>
static void quantize_act_chunk_subgroup(const float *x, int K, int T,
                                        uint8_t *aq, float *as, int32_t *asum) {
    constexpr int width = S4 ? 64 : 32;
    constexpr int values = width / kSG;
    constexpr int subgroups = 16;
    const size_t blocks = (size_t)(K / width) * T;
    const size_t workgroups = (blocks + subgroups - 1) / subgroups;
    tq_q().parallel_for(
        sycl::nd_range<1>(workgroups * subgroups * kSG, subgroups * kSG),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const auto sg = it.get_sub_group();
            const size_t blk = it.get_group_linear_id() * subgroups +
                               sg.get_group_linear_id();
            if (blk >= blocks) return;
            const int lane = (int)sg.get_local_linear_id();
            const size_t kt = blk / T, tok = blk % T;
            float v[values];
            float mx = -INFINITY;
            for (int j = 0; j < values; ++j) {
                v[j] = x[tok * K + kt * width + j * kSG + lane];
                mx = sycl::fmax(mx, sycl::fabs(v[j]));
            }
            mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
            const float scale = (mx > 0.0f) ? (mx / (S4 ? 7.0f : 127.0f)) : 1.0f;
            int32_t sum = 0;
            for (int j = 0; j < values; ++j) {
                const int32_t q = (int32_t)sycl::clamp(
                    sycl::rint(v[j] / scale), S4 ? -7.0f : -127.0f,
                    S4 ? 7.0f : 127.0f);
                sum += q;
                if constexpr (S4) {
                    const int32_t next = sycl::shift_group_left(sg, q, 1);
                    if ((lane & 1) == 0)
                        aq[blk * 32 + j * (kSG / 2) + lane / 2] =
                            (uint8_t)((q & 0xF) | ((next & 0xF) << 4));
                } else {
                    aq[blk * 32 + j * kSG + lane] = (uint8_t)(int8_t)q;
                }
            }
            sum = sycl::reduce_over_group(sg, sum, sycl::plus<int32_t>());
            if (lane == 0) {
                as[blk] = scale;
                asum[blk] = sum;
            }
        });
}

// fp32 [T][K] -> the RC8 A layout plus a per-(token, K32) scale and a
// per-(token, K32) SUM. The sum is stored already reduced over all 32 lanes,
// not as two K16 halves: the GEMM epilogue runs once per k-tile per output
// element, so re-adding the halves there cost 8 int adds and a 16-int load per
// k-tile, against one DPAS's 8 cycles. The staging subgroup reduces the full
// K32 sum once per block before any projection consumes it.
void x_quantize_act_chunk(const float *d_x, int K, int T, int8_t *d_aq,
                          float *d_as, int32_t *d_asum) {
    quantize_act_chunk_subgroup<false>(d_x, K, T,
        reinterpret_cast<uint8_t *>(d_aq), d_as, d_asum);
}

// Split-K for the batched GEMM. The batched path derives ALL of its
// parallelism from T - tasks = rgroups * tgroups - so at the decode batch
// shape (T=8, one t-group) the small-M projections starve: mlp_down (M=5120)
// generates 320 tasks = 20 work-groups on a machine that wants >=128, i.e.
// ~16% lane occupancy. That is the measured reason the batched step sits at
// 46% of byte-roofline where the GEMV path reaches 87% (o_proj and
// linear_out are the same shape; mlp_gate's M=17408 already reaches 53%).
// Splitting K restores the work-group count.
//
// Same contract as choose_gemv_splits, plus one extra constraint: the fused
// reducer needs kSubgroupsPerWorkgroup / splits to be exact, so splits must
// DIVIDE the subgroup count - {1,2,4,8,16}, never 6 or 7. Never hand a split
// fewer than one k-tile, and never split a grid that already saturates.
// TQ_XPU_GEMM_SPLITS overrides the heuristic for same-binary A/B and tuning:
// 1 forces the unsplit kernel (the true baseline), 2/4/8/16 force that split,
// unset or 0 uses the heuristic below. Without this, any claimed gain has no
// same-binary baseline and the 128-work-group target cannot be rejected.
int choose_gemm_splits(int rgroups, int tgroups, int ktiles) {
    static const int forced = [] {
        const char *e = getenv("TQ_XPU_GEMM_SPLITS");
        if (!e || !*e) return 0;
        const int v = atoi(e);
        // Must divide the subgroup count for the fused reducer to be exact.
        return (v == 1 || v == 2 || v == 4 || v == 8 || v == 16) ? v : 0;
    }();
    const long tasks = (long)rgroups * tgroups;
    if (tasks <= 0) return 1;
    if (forced) {
        int s = forced;
        while (s > 1 && s > ktiles) s >>= 1;
        return s;
    }
    const long want = 128L * kSubgroupsPerWorkgroup;   // >= 128 work-groups
    if (tasks >= want) return 1;
    for (int s = 2; s <= kSubgroupsPerWorkgroup; s <<= 1) {
        if (s > ktiles) break;
        if (tasks * s >= want) return s;
    }
    int s = kSubgroupsPerWorkgroup;
    while (s > 1 && s > ktiles) s >>= 1;
    return s;
}

// Reuse weights across two independent RC8 token groups. Each
// output keeps the original serial K order; requests needing split-K stay on
// their existing kernel.
template <bool S4, bool Tail = false>
static int gemm_rc16(const tq_qmma_weight_t *w, const uint8_t *aq,
                      const float *as, const int32_t *asum, float *y, int T) {
    const int M = w->M, kt_count = w->K / (S4 ? 64 : 32);
    const int tgroups = Tail ? (T + 15) / 16 : T / 16;
    const int tasks = (M / 16) * tgroups;
    const size_t wgs = ((size_t)tasks + kSubgroupsPerWorkgroup - 1) /
                       kSubgroupsPerWorkgroup;
    const uint8_t *codes = w->d_s4;
    const uint16_t *scales = w->d_s4_scale;
    const int ksh = S4 ? 0 : (w->s4_k64 ? 1 : 0);
    tq_q().parallel_for(
        sycl::nd_range<1>(wgs * kSG * kSubgroupsPerWorkgroup,
                          kSG * kSubgroupsPerWorkgroup),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const auto sg = it.get_sub_group();
            const int task = (int)it.get_group_linear_id() * kSubgroupsPerWorkgroup +
                             (int)sg.get_group_linear_id();
            if (task >= tasks) return;
            const int rg = task / tgroups, tg = task - rg * tgroups;
            const int lane = (int)sg.get_local_linear_id();
            const int apair = lane >> 3, abyte = (lane & 7) * 4;
            const uint8_t *bbase = codes + (size_t)rg * kt_count * (S4 ? 512u : 256u);
            const uint16_t *sbase = scales + (size_t)rg * (kt_count >> ksh) * 16u;
            float acc[2][8];
            for (int p = 0; p < 2; ++p)
                for (int m = 0; m < 8; ++m) acc[p][m] = 0.0f;
            for (int kt = 0; kt < kt_count; ++kt) {
                const uint8_t *bb = bbase + (size_t)kt * (S4 ? 512u : 256u);
                const uintv4 blo = *reinterpret_cast<const uintv4 *>(
                    bb + (size_t)lane * 16u);
                uintv8 bfrag4;
                if constexpr (S4) {
                    const uintv4 bhi = *reinterpret_cast<const uintv4 *>(
                        bb + 256u + (size_t)lane * 16u);
                    for (int i = 0; i < 4; ++i) {
                        bfrag4[i] = blo[i];
                        bfrag4[4 + i] = bhi[i];
                    }
                }
                const uint16_t packed = sbase[(size_t)(kt >> ksh) * 16u + lane];
                int zp = (int)(packed & 0xFu);
                if (zp >= 8) zp -= 16;
                const float ws = (float)sycl::bit_cast<sycl::half>(
                    (uint16_t)(packed & 0xFFF0u));
                for (int p = 0; p < 2; ++p) {
                    if constexpr (Tail) {
                        if (tg * 16 + p * 8 >= T) continue;
                    }
                    const size_t arow = (size_t)kt * T + tg * 16 + p * 8;
                    const uint8_t *ab = aq + arow * 32;
                    uintv4 afrag;
                    for (int i = 0; i < 4; ++i)
                        afrag[i] = *reinterpret_cast<const uint32_t *>(
                            ab + (size_t)(2 * i + apair) * 32 + abyte);
                    intv8 d;
                    for (int m = 0; m < 8; ++m) d[m] = 0;
                    if constexpr (S4) {
                        DPAS_S4S4_RC8(d, afrag, bfrag4);
                    } else {
                        DPAS_S4S8_RC8(d, afrag, blo);
                    }
                    const floatv8 asv = *reinterpret_cast<const floatv8 *>(as + arow);
                    const intv8 sumv = *reinterpret_cast<const intv8 *>(asum + arow);
                    for (int m = 0; m < 8; ++m)
                        acc[p][m] += (float)(d[m] - zp * sumv[m]) * (ws * asv[m]);
                }
            }
            for (int p = 0; p < 2; ++p) {
                if constexpr (Tail) {
                    if (tg * 16 + p * 8 >= T) continue;
                }
                for (int m = 0; m < 8; ++m)
                    y[(size_t)(tg * 16 + p * 8 + m) * M + rg * 16 + lane] = acc[p][m];
            }
        });
    return 0;
}

int x_gemm_w4a8(const tq_qmma_weight_t *w, const int8_t *d_aq, const float *d_as,
                const int32_t *d_asum, float *d_y, int T,
                const int *offsets, int segments) {
    if (!w || !w->s4_ready || w->s4_k16 || !w->d_s4 || !w->d_s4_scale) return -1;
    if (!d_aq || !d_as || !d_asum || !d_y || T <= 0 || (T % 8) != 0) return -2;
    const int M = w->M, K = w->K, Kt = K / 32;
    const int rgroups = M / 16, tgroups = T / 8;
    // Capture at most eight segment boundaries and split choices by value.
    // Packed activation strides remain T; only the K partition is per request.
    struct SegmentSplits {
        int count;
        int end[8];
        int split[8];
    } partition{};
    int splits = 1;
    if (offsets) {
        if (segments < 1 || segments > 8 || offsets[0] != 0 ||
            offsets[segments] != T) return -2;
        partition.count = segments;
        for (int i = 0; i < segments; ++i) {
            if (offsets[i] < 0 || offsets[i + 1] <= offsets[i] ||
                offsets[i + 1] > T || offsets[i] % 8 || offsets[i + 1] % 8)
                return -2;
            partition.end[i] = offsets[i + 1] / 8;
            partition.split[i] = choose_gemm_splits(
                rgroups, (offsets[i + 1] - offsets[i]) / 8, Kt);
            splits = std::max(splits, partition.split[i]);
        }
    } else {
        if (segments != 0) return -2;
        partition.count = 1;
        partition.end[0] = tgroups;
        partition.split[0] = splits = choose_gemm_splits(rgroups, tgroups, Kt);
    }
    // Validated offsets partition dense rows [0,T) without gaps. Their only
    // kernel effect is the K split count, whose maximum is `splits`; when it
    // is one, pairing RC8 groups across request boundaries is row-independent.
    if (splits == 1 && T % 16 == 0)
        return gemm_rc16<false>(w, reinterpret_cast<const uint8_t *>(d_aq),
                                d_as, d_asum, d_y, T);
    // Preserve every segment's original split-K eligibility. The last RC8
    // group is real eight-row work, never a padded request or altered stride.
    if (splits == 1 && T > 16 && T % 16 == 8)
        return gemm_rc16<false, true>(w, reinterpret_cast<const uint8_t *>(d_aq),
                                     d_as, d_asum, d_y, T);
    const int rtiles = rgroups * tgroups;
    const int tasks = rtiles * splits;
    const size_t wgs = ((size_t)tasks + kSubgroupsPerWorkgroup - 1) /
                       kSubgroupsPerWorkgroup;
    const int fused = (splits > 1) ? 1 : 0;
    const int gpw = fused ? (kSubgroupsPerWorkgroup / splits) : 0;
    const uint8_t *codes = w->d_s4;
    const uint16_t *scales = w->d_s4_scale;
    const int ksh = w->s4_k64 ? 1 : 0;
    tq_q().submit([&](sycl::handler &cgh) {
        // 16 subgroups x 8 outputs x 16 lanes x 4 B = 8 KiB per work-group.
        // Lane is the FASTEST index: stride-16-dword layouts serialise 16:1
        // on the 16-bank SLM, and lane-fastest is conflict-free.
        sycl::local_accessor<float, 1> slm(
            fused ? (size_t)kSubgroupsPerWorkgroup * 8 * kSG : 1, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(wgs * kSG * kSubgroupsPerWorkgroup,
                              kSG * kSubgroupsPerWorkgroup),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]]
                {
            const auto sg = it.get_sub_group();
            const int sgid = (int)sg.get_group_linear_id();
            const int wg = (int)it.get_group_linear_id();
            const int task = wg * kSubgroupsPerWorkgroup + sgid;
            const int lane = (int)sg.get_local_linear_id();
            const int apair = lane >> 3;          // which row of each pair
            const int abyte = (lane & 7) * 4;     // byte offset inside the row
            float acc[8];
            for (int m = 0; m < 8; ++m) acc[m] = 0.0f;
            if (task < tasks) {
                const int rt = task / splits;
                const int split = task - rt * splits;
                const int rg = rt / tgroups;
                const int tg = rt - rg * tgroups;
                int segment = 0;
                while (segment + 1 < partition.count && tg >= partition.end[segment])
                    ++segment;
                const int actual_splits = partition.split[segment];
                // Unused split subgroups contribute zero, but still reach the
                // same work-group barrier as every active split subgroup.
                const int kt_begin = split < actual_splits ? Kt * split / actual_splits : Kt;
                const int kt_end = split < actual_splits ? Kt * (split + 1) / actual_splits : Kt;
                const uint8_t *bbase = codes + (size_t)rg * Kt * 256u;
                const uint16_t *sbase = scales + (size_t)rg * (Kt >> ksh) * 16u;
                for (int kt = kt_begin; kt < kt_end; ++kt) {
                    const uintv4 bfrag = *reinterpret_cast<const uintv4 *>(
                        bbase + (size_t)kt * 256u + (size_t)lane * 16u);
                    const int8_t *ab = d_aq + ((size_t)kt * T + tg * 8) * 32;
                    uintv4 afrag;
                    for (int i = 0; i < 4; ++i)
                        afrag[i] = *reinterpret_cast<const uint32_t *>(
                            ab + (size_t)(2 * i + apair) * 32 + abyte);
                    intv8 d;
                    for (int m = 0; m < 8; ++m) d[m] = 0;
                    DPAS_S4S8_RC8(d, afrag, bfrag);
                    const uint16_t packed =
                        sbase[(size_t)(kt >> ksh) * 16u + lane];
                    int zp = (int)(packed & 0xFu);
                    if (zp >= 8) zp -= 16;
                    const float ws = (float)sycl::bit_cast<sycl::half>(
                        (uint16_t)(packed & 0xFFF0u));
                    // Per-token scale and sum are lane-uniform and contiguous,
                    // so fetch each as ONE vector rather than 24 scalar loads
                    // per k-tile - that scalar traffic, not the DPAS, was
                    // capping this at 8% of the systolic rate. The sum arrives
                    // pre-reduced over K32, so the epilogue is 8 fewer int adds
                    // and 8 fewer loaded ints per k-tile.
                    const floatv8 asv = *reinterpret_cast<const floatv8 *>(
                        d_as + (size_t)kt * T + tg * 8);
                    const intv8 sumv = *reinterpret_cast<const intv8 *>(
                        d_asum + (size_t)kt * T + tg * 8);
                    for (int m = 0; m < 8; ++m)
                        acc[m] += (float)(d[m] - zp * sumv[m]) * (ws * asv[m]);
                }
            }
            if (!fused) {
                if (task < tasks) {
                    const int rg = task / tgroups;
                    const int tg = task - rg * tgroups;
                    for (int m = 0; m < 8; ++m)
                        d_y[(size_t)(tg * 8 + m) * M + rg * 16 + lane] = acc[m];
                }
                return;
            }
            for (int m = 0; m < 8; ++m)
                slm[((size_t)sgid * 8 + m) * kSG + lane] = acc[m];
            it.barrier(sycl::access::fence_space::local_space);
            if (sgid >= gpw) return;
            // max-split groups divide the work-group, so all partials of a
            // tile are local even when adjacent tiles have different splits.
            const int rt = wg * gpw + sgid;
            if (rt >= rtiles) return;
            const int rg = rt / tgroups;
            const int tg = rt - rg * tgroups;
            int segment = 0;
            while (segment + 1 < partition.count && tg >= partition.end[segment])
                ++segment;
            const int actual_splits = partition.split[segment];
            for (int m = 0; m < 8; ++m) {
                const size_t base = ((size_t)sgid * splits * 8 + m) * kSG + lane;
                // An unsplit serial projection stores acc directly. Preserve
                // that path too, rather than introducing an extra +0 round.
                float sum = actual_splits == 1 ? slm[base] : 0.0f;
                if (actual_splits > 1)
                    for (int i = 0; i < actual_splits; ++i)
                        sum += slm[base + (size_t)i * 8 * kSG];
                d_y[(size_t)(tg * 8 + m) * M + rg * 16 + lane] = sum;
            }
        });
    });
    return 0;
}

// fp32 [T][K] -> the W4A4 A layout: 64 s4 nibbles per (token, K64) row
// (32 bytes, low nibble = even k), one symmetric scale (absmax/7) and one
// integer code SUM per (token, K64) for the weight zero-point correction.
void x_quantize_act_chunk_s4(const float *d_x, int K, int T, uint8_t *d_aq4,
                             float *d_as4, int32_t *d_asum4) {
    quantize_act_chunk_subgroup<true>(d_x, K, T, d_aq4, d_as4, d_asum4);
}

// Batched W4A4 GEMM: the same task grid and epilogue shape as x_gemm_w4a8,
// but dpas.s4.s4.8.8 at K=64 - half the DPAS count and half the epilogue
// evaluations per K-span (probe-funded 1.77-1.78x, level-up item 4). The
// K64 B fragment is two adjacent lane-major K32 tiles: two 16-byte loads.
int x_gemm_w4a4(const tq_qmma_weight_t *w, const uint8_t *d_aq4,
                const float *d_as4, const int32_t *d_asum4, float *d_y, int T) {
    if (!w || !w->s4_ready || !w->s4_k64 || !w->d_s4 || !w->d_s4_scale) return -1;
    if (!d_aq4 || !d_as4 || !d_asum4 || !d_y || T <= 0 || (T % 8) != 0) return -2;
    if (T % 16 == 0)
        return gemm_rc16<true>(w, d_aq4, d_as4, d_asum4, d_y, T);
    if (T > 16 && T % 16 == 8)
        return gemm_rc16<true, true>(w, d_aq4, d_as4, d_asum4, d_y, T);
    const int M = w->M, K = w->K, Kt64 = K / 64;
    const int rgroups = M / 16, tgroups = T / 8;
    const int tasks = rgroups * tgroups;
    // Only the eight-row shape remains after the full and tailed RC16 paths.
    constexpr int nsg = 2;
    const size_t wgs = ((size_t)tasks + nsg - 1) / nsg;
    const uint8_t *codes = w->d_s4;
    const uint16_t *scales = w->d_s4_scale;
    tq_q().parallel_for(
        sycl::nd_range<1>(wgs * kSG * nsg, kSG * nsg),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]]
            {
            const auto sg = it.get_sub_group();
            const int task = (int)it.get_group_linear_id() * nsg +
                             (int)sg.get_group_linear_id();
            if (task >= tasks) return;
            const int rg = task / tgroups;
            const int tg = task - rg * tgroups;
            const int lane = (int)sg.get_local_linear_id();
            const int apair = lane >> 3;
            const int abyte = (lane & 7) * 4;
            const uint8_t *bbase = codes + (size_t)rg * Kt64 * 512u;
            const uint16_t *sbase = scales + (size_t)rg * Kt64 * 16u;
            float acc[8];
            for (int m = 0; m < 8; ++m) acc[m] = 0.0f;
            for (int kt = 0; kt < Kt64; ++kt) {
                const uintv4 blo = *reinterpret_cast<const uintv4 *>(
                    bbase + (size_t)kt * 512u + (size_t)lane * 16u);
                const uintv4 bhi = *reinterpret_cast<const uintv4 *>(
                    bbase + (size_t)kt * 512u + 256u + (size_t)lane * 16u);
                uintv8 bfrag;
                for (int i = 0; i < 4; ++i) {
                    bfrag[i] = blo[i];
                    bfrag[4 + i] = bhi[i];
                }
                const uint8_t *ab = d_aq4 + ((size_t)kt * T + tg * 8) * 32;
                uintv4 afrag;
                for (int i = 0; i < 4; ++i)
                    afrag[i] = *reinterpret_cast<const uint32_t *>(
                        ab + (size_t)(2 * i + apair) * 32 + abyte);
                intv8 d;
                for (int m = 0; m < 8; ++m) d[m] = 0;
                DPAS_S4S4_RC8(d, afrag, bfrag);
                const uint16_t packed = sbase[(size_t)kt * 16u + lane];
                int zp = (int)(packed & 0xFu);
                if (zp >= 8) zp -= 16;
                const float ws = (float)sycl::bit_cast<sycl::half>(
                    (uint16_t)(packed & 0xFFF0u));
                const floatv8 asv = *reinterpret_cast<const floatv8 *>(
                    d_as4 + (size_t)kt * T + tg * 8);
                const intv8 sumv = *reinterpret_cast<const intv8 *>(
                    d_asum4 + (size_t)kt * T + tg * 8);
                for (int m = 0; m < 8; ++m)
                    acc[m] += (float)(d[m] - zp * sumv[m]) * (ws * asv[m]);
            }
            for (int m = 0; m < 8; ++m)
                d_y[(size_t)(tg * 8 + m) * M + rg * 16 + lane] = acc[m];
        });
    return 0;
}

// Batched W8A8 GEMM (design 9.3 support): the W4A8 GEMM's task grid and
// activation staging (x_quantize_act_chunk's Aq/as reused as-is), the W8
// GEMV's 512-byte lane-major fragments, plain FP16 scales, no zero point.
// Built for the spec wave's 8-row lm_head: the 1.27 GB weight streams once
// instead of eight times.
int x_gemm_w8a8(const tq_qmma_weight_t *w, const int8_t *d_aq, const float *d_as,
                float *d_y, int T) {
    if (!w || !w->s8_ready || !w->d_s8 || !w->d_s8_scale) return -1;
    if (!d_aq || !d_as || !d_y || T <= 0 || (T % 8) != 0) return -2;
    const int M = w->M, K = w->K, Kt = K / 32;
    const int rgroups = M / 16, tgroups = T / 8;
    const int tasks = rgroups * tgroups;
    const size_t wgs = ((size_t)tasks + kSubgroupsPerWorkgroup - 1) /
                       kSubgroupsPerWorkgroup;
    const uint8_t *codes = w->d_s8;
    const uint16_t *scales = w->d_s8_scale;
    tq_q().parallel_for(
        sycl::nd_range<1>(wgs * kSG * kSubgroupsPerWorkgroup,
                          kSG * kSubgroupsPerWorkgroup),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]]
            {
            const auto sg = it.get_sub_group();
            const int task = (int)it.get_group_linear_id() * kSubgroupsPerWorkgroup +
                             (int)sg.get_group_linear_id();
            if (task >= tasks) return;
            const int rg = task / tgroups;
            const int tg = task - rg * tgroups;
            const int lane = (int)sg.get_local_linear_id();
            const int apair = lane >> 3;
            const int abyte = (lane & 7) * 4;
            const uint8_t *bbase = codes + (size_t)rg * Kt * 512u;
            const uint16_t *sbase = scales + (size_t)rg * Kt * 16u;
            float acc[8];
            for (int m = 0; m < 8; ++m) acc[m] = 0.0f;
            for (int kt = 0; kt < Kt; ++kt) {
                const uintv8 bfrag = *reinterpret_cast<const uintv8 *>(
                    bbase + (size_t)kt * 512u + (size_t)lane * 32u);
                const int8_t *ab = d_aq + ((size_t)kt * T + tg * 8) * 32;
                uintv4 afrag;
                for (int i = 0; i < 4; ++i)
                    afrag[i] = *reinterpret_cast<const uint32_t *>(
                        ab + (size_t)(2 * i + apair) * 32 + abyte);
                intv8 d;
                for (int m = 0; m < 8; ++m) d[m] = 0;
                DPAS_S8S8_RC8(d, afrag, bfrag);
                const float ws = (float)sycl::bit_cast<sycl::half>(
                    sbase[(size_t)kt * 16u + lane]);
                const floatv8 asv = *reinterpret_cast<const floatv8 *>(
                    d_as + (size_t)kt * T + tg * 8);
                for (int m = 0; m < 8; ++m)
                    acc[m] += (float)d[m] * (ws * asv[m]);
            }
            for (int m = 0; m < 8; ++m)
                d_y[(size_t)(tg * 8 + m) * M + rg * 16 + lane] = acc[m];
        });
    return 0;
}

static int x_gemv_w4a8_prepared_impl(const tq_qmma_weight_t *w, float *d_y,
                                      const float *d_residual) {
    if (!w || !w->s4_ready || !w->d_s4 || !w->d_s4_scale || !d_y) return -1;
    const int M = w->M;
    const int K = w->K;
    if (g_act_cap < K || !g_act_q || !g_act_scale) return -2;
    const int groups = M / 16;
    const int splits = choose_gemv_splits(groups);
    // Only splits that exceed one work-group still need a global partial pass.
    const bool global_reduce = splits > kSubgroupsPerWorkgroup;
    if (global_reduce) ensure_split_scratch((size_t)M * splits);
    float *partial = g_split_partial;
    if (w->s4_k16)
        x_gemv_w4a8_core_k16(w->d_s4, w->d_s4_scale, g_act_q, g_act_scale,
                             g_act_sum, d_y, d_residual, partial, M, K, splits);
    else
        x_gemv_w4a8_core(w->d_s4, w->d_s4_scale, g_act_q, g_act_scale, g_act_sum,
                         d_y, d_residual, partial, M, K, splits,
                         w->s4_k64 ? 1 : 0);
    if (global_reduce) {
        tq_q().parallel_for(sycl::range<1>((size_t)M), [=](sycl::id<1> id) {
            const int row = (int)id[0];
            float sum = 0.0f;
            for (int split = 0; split < splits; ++split)
                sum += partial[(size_t)split * M + row];
            d_y[row] = sum + (d_residual ? d_residual[row] : 0.0f);
        });
    } else if (splits == 1 && d_residual) {
        x_add_inplace(d_y, d_residual, M);
    }
    return 0;
}

int x_gemv_w4a8_prepared(const tq_qmma_weight_t *w, float *d_y,
                         const float *d_residual) {
    return x_gemv_w4a8_prepared_impl(w, d_y, d_residual);
}

int x_gemv_w4a8(const tq_qmma_weight_t *w, const float *d_x, float *d_y) {
    if (!w || !d_x) return -1;
    x_prepare_gemv_act_s8(d_x, w->K);
    return x_gemv_w4a8_prepared_impl(w, d_y, nullptr);
}

int x_gemv_w4a8_add(const tq_qmma_weight_t *w, const float *d_x,
                     const float *d_residual, float *d_y) {
    if (!w || !d_x || !d_residual) return -1;
    x_prepare_gemv_act_s8(d_x, w->K);
    return x_gemv_w4a8_prepared_impl(w, d_y, d_residual);
}

static int x_gemv_w8a8_prepared_impl(const tq_qmma_weight_t *w, float *d_y,
                                      const float *d_residual) {
    if (!w || !w->s8_ready || !w->d_s8 || !w->d_s8_scale || !d_y) return -1;
    const int M = w->M;
    const int K = w->K;
    if (g_act_cap < K || !g_act_q || !g_act_scale) return -2;
    const int groups = M / 16;
    const int splits = choose_gemv_splits(groups);
    const bool global_reduce = splits > kSubgroupsPerWorkgroup;
    if (global_reduce) ensure_split_scratch((size_t)M * splits);
    float *partial = g_split_partial;
    x_gemv_w8a8_core(w->d_s8, w->d_s8_scale, g_act_q, g_act_scale,
                     d_y, d_residual, partial, M, K, splits);
    if (global_reduce) {
        tq_q().parallel_for(sycl::range<1>((size_t)M), [=](sycl::id<1> id) {
            const int row = (int)id[0];
            float sum = 0.0f;
            for (int split = 0; split < splits; ++split)
                sum += partial[(size_t)split * M + row];
            d_y[row] = sum + (d_residual ? d_residual[row] : 0.0f);
        });
    } else if (splits == 1 && d_residual) {
        x_add_inplace(d_y, d_residual, M);
    }
    return 0;
}

int x_gemv_w8a8_prepared(const tq_qmma_weight_t *w, float *d_y,
                         const float *d_residual) {
    return x_gemv_w8a8_prepared_impl(w, d_y, d_residual);
}

int x_gemv_w8a8(const tq_qmma_weight_t *w, const float *d_x, float *d_y) {
    if (!w || !d_x) return -1;
    x_prepare_gemv_act_s8(d_x, w->K);
    return x_gemv_w8a8_prepared_impl(w, d_y, nullptr);
}

int x_gemv_w8a8_add(const tq_qmma_weight_t *w, const float *d_x,
                     const float *d_residual, float *d_y) {
    if (!w || !d_x || !d_residual) return -1;
    x_prepare_gemv_act_s8(d_x, w->K);
    return x_gemv_w8a8_prepared_impl(w, d_y, d_residual);
}

// ---------------------------------------------------------------------------
// Format probe: weight-reconstruction quality of every candidate 4-bit tier on
// one real matrix, at matched metadata budgets. This answers "is the Xe2
// uniform-INT4 grid competitive with the CUDA NVFP4 tier?" from silicon rather
// than from vendor tables, and it isolates grid choice from scale granularity.
// Pure read: the weight is not modified, so a probe run leaves the model in the
// scalar tier. Requires TQ_XPU_W4A8=0 so the E2M3 payload is still resident.
namespace {

constexpr int kProbeSchemes = 11;
// na, then dot/nb per scheme, then the scheme-9 int3-group count.
constexpr int kProbeStride = 1 + 2 * kProbeSchemes + 1;

// Round a non-negative float to the nearest OCP FP8 E4M3 value (max 448).
// NVFP4 stores its per-16 block scale in exactly this format.
static inline float round_e4m3(float v) {
    if (!(v > 0.0f)) return 0.0f;
    if (v >= 448.0f) return 448.0f;
    if (v < 0.015625f) {                                  // subnormal, step 2^-9
        return sycl::fmin(sycl::rint(v * 512.0f), 7.0f) * 0.001953125f;
    }
    int e = (int)sycl::floor(sycl::log2(v));
    float p = sycl::exp2((float)e);
    int q = (int)sycl::rint((v / p - 1.0f) * 8.0f);
    if (q >= 8) { q = 0; ++e; p *= 2.0f; }
    if (e >= 8) { e = 8; p = 256.0f; if (q > 6) q = 6; }  // 2^8 * 1.75 = 448
    return p * (1.0f + (float)q * 0.125f);
}

// Nearest E2M1 magnitude - the NVFP4 element grid. Non-uniform by construction:
// {0, .5, 1, 1.5, 2, 3, 4, 6} spends one of its four mantissa steps on values
// above 2, which is what costs it against 16 uniform levels.
static inline float round_e2m1_mag(float v) {
    const float g[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float best = 0.0f;
    float bd = sycl::fabs(v);
    for (int i = 1; i < 8; ++i) {
        const float d = sycl::fabs(v - g[i]);
        if (d < bd) { bd = d; best = g[i]; }
    }
    return best;
}

// Symmetric INT4: the pre-asymmetric ship tier, kept as the probe's baseline.
static inline float fit_int4_sym(const float *wf, int n) {
    float hi = 0.0f;
    float neg = 0.0f;
    for (int j = 0; j < n; ++j) {
        hi = sycl::fmax(hi, wf[j]);
        neg = sycl::fmax(neg, -wf[j]);
    }
    float s = sycl::fmax(hi / 7.0f, neg / 8.0f);
    if (!(s > 0.0f)) s = 1.0f;
    for (int iter = 0; iter < 2; ++iter) {
        float num = 0.0f;
        float den = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float q = sycl::clamp(sycl::rint(wf[j] / s), -8.0f, 7.0f);
            num += wf[j] * q;
            den += q * q;
        }
        if (den > 0.0f) s = num / den;
    }
    return (float)sycl::bit_cast<sycl::half>(sycl::bit_cast<uint16_t>(sycl::half(s)));
}

// Raw absmax per 128-row block. Each two-level scheme derives its own global
// from this the way its element format requires (NVFP4: /(6*448), since 6 is
// the largest E2M1 magnitude and 448 the largest E4M3 scale).
void probe_block_absmax_kernel(const tq_qmma_weight_t *w, float *d_g, int nrb) {
    const int M = w->M, Kt = w->Kt, K = w->K;
    const int scale_cols = w->scale_cols;
    const uint8_t *payload = w->d_A;
    const float *bs = w->d_block_scale_inv;
    tq_q().parallel_for(
        sycl::nd_range<1>((size_t)nrb * 256, 256), [=](sycl::nd_item<1> it) {
            const int rb = (int)it.get_group_linear_id();
            const int t = (int)it.get_local_linear_id();
            const int row0 = rb * 128;
            const int rows = sycl::min(128, M - row0);
            float mx = 0.0f;
            for (long idx = t; idx < (long)rows * K; idx += 256) {
                const int row = row0 + (int)(idx / K);
                const int k = (int)(idx % K);
                const int kt = k >> 5;
                const int sc = (scale_cols == Kt) ? kt : (kt >> 2);
                const float b = bs[(size_t)((row >> 4) >> 3) * scale_cols + sc];
                mx = sycl::fmax(mx, sycl::fabs(
                    e2m3f(e2m3_code_at(payload, Kt, row, k)) * b));
            }
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            if (t == 0) d_g[rb] = (mx > 0.0f) ? mx : 0.0f;
        });
}

// One work-item per output row; accumulates <w, w-hat>, |w|^2 and |w-hat|^2 so
// the host can report a cosine and a relative L2 error per scheme.
void probe_grid_kernel(const tq_qmma_weight_t *w, const float *d_g, float *d_acc) {
    const int M = w->M, Kt = w->Kt;
    const int scale_cols = w->scale_cols;
    const uint8_t *payload = w->d_A;
    const float *bs = w->d_block_scale_inv;
    tq_q().parallel_for(sycl::range<1>((size_t)M), [=](sycl::id<1> id) {
        const int row = (int)id[0];
        float na = 0.0f;
        float dot[kProbeSchemes] = {};
        float nb[kProbeSchemes] = {};
        float count3 = 0.0f;
        float wf64[64];   // scheme 10 buffers a k32 pair for the k64 fit
        const float global = d_g[row >> 7];
        for (int kt = 0; kt < Kt; ++kt) {
            const int sc = (scale_cols == Kt) ? kt : (kt >> 2);
            const float b = bs[(size_t)((row >> 4) >> 3) * scale_cols + sc];
            float wf[32];
            for (int j = 0; j < 32; ++j) {
                wf[j] = e2m3f(e2m3_code_at(payload, Kt, row, kt * 32 + j)) * b;
                na += wf[j] * wf[j];
            }
            for (int j = 0; j < 32; ++j) wf64[(kt & 1) * 32 + j] = wf[j];
            auto tally = [&](int scheme, float ref, float hat) {
                dot[scheme] += ref * hat;
                nb[scheme] += hat * hat;
            };
            // 0: INT4 symmetric, FP16 scale per k32 (4.50 bits/w).
            {
                const float s = fit_int4_sym(wf, 32);
                for (int j = 0; j < 32; ++j) {
                    const float q = sycl::clamp(sycl::rint(wf[j] / s), -8.0f, 7.0f);
                    tally(0, wf[j], s * q);
                }
            }
            // 1: INT4 asymmetric, FP16 scale per k32 with z in the low nibble
            //    (4.50 bits/w) - the current ship tier, same call as the repack.
            {
                int zp = 0;
                const uint16_t sh = fit_int4_asym(wf, 32, true, &zp);
                const float s = (float)sycl::bit_cast<sycl::half>(sh);
                for (int j = 0; j < 32; ++j) {
                    const float q = sycl::clamp(
                        sycl::rint(wf[j] / s + (float)zp), -8.0f, 7.0f);
                    tally(1, wf[j], s * (q - (float)zp));
                }
            }
            // 2: INT4 asymmetric at NVFP4's k16 granularity, FP16 scale
            //    (5.00 bits/w) - isolates granularity from grid.
            for (int half = 0; half < 2; ++half) {
                int zp = 0;
                const uint16_t sh = fit_int4_asym(wf + half * 16, 16, true, &zp);
                const float s = (float)sycl::bit_cast<sycl::half>(sh);
                for (int j = 0; j < 16; ++j) {
                    const float v = wf[half * 16 + j];
                    const float q = sycl::clamp(
                        sycl::rint(v / s + (float)zp), -8.0f, 7.0f);
                    tally(2, v, s * (q - (float)zp));
                }
            }
            // 3: NVFP4 exactly as CUDA ships it - E2M1 codes, E4M3 scale per
            //    k16, FP32 global per 128 rows (4.50 bits/w).
            for (int half = 0; half < 2; ++half) {
                float mx = 0.0f;
                for (int j = 0; j < 16; ++j)
                    mx = sycl::fmax(mx, sycl::fabs(wf[half * 16 + j]));
                const float g3 = (global > 0.0f) ? global / (6.0f * 448.0f) : 1.0f;
                const float e4 = round_e4m3(mx / (6.0f * g3));
                const float s = e4 * g3;
                for (int j = 0; j < 16; ++j) {
                    const float v = wf[half * 16 + j];
                    const float hat = (s > 0.0f)
                        ? sycl::copysign(round_e2m1_mag(sycl::fabs(v / s)), v) * s
                        : 0.0f;
                    tally(3, v, hat);
                }
            }
            // 4: INT8 symmetric, FP16 scale per k32 (8.50 bits/w) - the W8
            //    fallback's grid, included as the achievable ceiling.
            {
                float mx = 0.0f;
                for (int j = 0; j < 32; ++j) mx = sycl::fmax(mx, sycl::fabs(wf[j]));
                float s = (mx > 0.0f) ? (mx / 127.0f) : 1.0f;
                s = (float)sycl::bit_cast<sycl::half>(sycl::bit_cast<uint16_t>(
                    sycl::half(s)));
                for (int j = 0; j < 32; ++j) {
                    const float q = sycl::clamp(sycl::rint(wf[j] / s),
                                                -128.0f, 127.0f);
                    tally(4, wf[j], s * q);
                }
            }
            // 5: INT4 asymmetric, E4M3 scale per k16 + FP32 global per 128 rows
            //    + a separate 4-bit zero point (4.75 bits/w). NVFP4's two-level
            //    scale trick on the Intel uniform grid.
            for (int half = 0; half < 2; ++half) {
                const float g5 = (global > 0.0f) ? (2.0f * global) / (15.0f * 448.0f)
                                                 : 1.0f;
                float lo = 0.0f;
                float hi = 0.0f;
                for (int j = 0; j < 16; ++j) {
                    lo = sycl::fmin(lo, wf[half * 16 + j]);
                    hi = sycl::fmax(hi, wf[half * 16 + j]);
                }
                float s = round_e4m3(((hi - lo) / 15.0f) / g5) * g5;
                if (!(s > 0.0f)) s = g5;
                const int zp = (int)sycl::clamp(sycl::rint(-lo / s) - 8.0f,
                                               -8.0f, 7.0f);
                for (int j = 0; j < 16; ++j) {
                    const float v = wf[half * 16 + j];
                    const float q = sycl::clamp(sycl::rint(v / s + (float)zp),
                                                -8.0f, 7.0f);
                    tally(5, v, s * (q - (float)zp));
                }
            }
            // 6: INT4 symmetric, E4M3 scale per k16 + FP32 global per 128 rows
            //    (4.50 bits/w) - exactly NVFP4's metadata budget, uniform grid,
            //    no zero point. The head-to-head against scheme 1.
            for (int half = 0; half < 2; ++half) {
                const float g6 = (global > 0.0f) ? global / (8.0f * 448.0f) : 1.0f;
                float mx = 0.0f;
                for (int j = 0; j < 16; ++j)
                    mx = sycl::fmax(mx, sycl::fabs(wf[half * 16 + j]));
                float s = round_e4m3((mx / 8.0f) / g6) * g6;
                if (!(s > 0.0f)) s = g6;
                for (int j = 0; j < 16; ++j) {
                    const float v = wf[half * 16 + j];
                    const float q = sycl::clamp(sycl::rint(v / s), -8.0f, 7.0f);
                    tally(6, v, s * q);
                }
            }
            // 7: INT3 symmetric, FP16 scale per k32 (3.50 bits/w). Quality
            //    rung only: byte savings need 3-bit packing whose unpack cost
            //    a kernel probe must price separately (E4M3 ldexp lesson).
            {
                float hi = 0.0f;
                float neg = 0.0f;
                for (int j = 0; j < 32; ++j) {
                    hi = sycl::fmax(hi, wf[j]);
                    neg = sycl::fmax(neg, -wf[j]);
                }
                float s = sycl::fmax(hi / 3.0f, neg / 4.0f);
                if (!(s > 0.0f)) s = 1.0f;
                for (int iter = 0; iter < 2; ++iter) {
                    float num = 0.0f;
                    float den = 0.0f;
                    for (int j = 0; j < 32; ++j) {
                        const float q = sycl::clamp(sycl::rint(wf[j] / s),
                                                    -4.0f, 3.0f);
                        num += wf[j] * q;
                        den += q * q;
                    }
                    if (den > 0.0f) s = num / den;
                }
                s = (float)sycl::bit_cast<sycl::half>(
                    sycl::bit_cast<uint16_t>(sycl::half(s)));
                for (int j = 0; j < 32; ++j) {
                    const float q = sycl::clamp(sycl::rint(wf[j] / s),
                                                -4.0f, 3.0f);
                    tally(7, wf[j], s * q);
                }
            }
            // 8: INT3 asymmetric, FP16 scale per k32, zp in the scale nibble
            //    (3.50 bits/w) - the ship tier's fit at half the code range.
            {
                float lo = 0.0f;
                float hi = 0.0f;
                for (int j = 0; j < 32; ++j) {
                    lo = sycl::fmin(lo, wf[j]);
                    hi = sycl::fmax(hi, wf[j]);
                }
                float s = (hi - lo) / 7.0f;
                if (!(s > 0.0f)) s = 1.0f;
                int zi = 0;
                for (int iter = 0; iter < 3; ++iter) {
                    zi = (int)sycl::clamp(sycl::rint(-lo / s) - 4.0f, -4.0f,
                                          3.0f);
                    float num = 0.0f;
                    float den = 0.0f;
                    for (int j = 0; j < 32; ++j) {
                        const float q = sycl::clamp(
                            sycl::rint(wf[j] / s + (float)zi), -4.0f, 3.0f);
                        const float centered = q - (float)zi;
                        num += wf[j] * centered;
                        den += centered * centered;
                    }
                    if (den > 0.0f) s = num / den;
                }
                s = (float)sycl::bit_cast<sycl::half>((uint16_t)(
                    (sycl::bit_cast<uint16_t>(sycl::half(s)) + 8u) & 0xFFF0u));
                zi = (int)sycl::clamp(sycl::rint(-lo / s) - 4.0f, -4.0f, 3.0f);
                float err3 = 0.0f;
                for (int j = 0; j < 32; ++j) {
                    const float q = sycl::clamp(
                        sycl::rint(wf[j] / s + (float)zi), -4.0f, 3.0f);
                    const float hat = s * (q - (float)zi);
                    tally(8, wf[j], hat);
                    const float d8 = wf[j] - hat;
                    err3 += d8 * d8;
                }
                // 9: mixed 3/4-bit asym per group: keep int3 unless its error
                //    exceeds 1.8x the int4 fit's. Achieved bits are measured
                //    (count3 tallied), storage is group-aligned by design.
                int zp4 = 0;
                const uint16_t sh4 = fit_int4_asym(wf, 32, true, &zp4);
                const float s4v = (float)sycl::bit_cast<sycl::half>(sh4);
                float err4 = 0.0f;
                for (int j = 0; j < 32; ++j) {
                    const float q = sycl::clamp(
                        sycl::rint(wf[j] / s4v + (float)zp4), -8.0f, 7.0f);
                    const float hat = s4v * (q - (float)zp4);
                    const float d9 = wf[j] - hat;
                    err4 += d9 * d9;
                }
                const bool use3 = err3 <= 1.8f * err4;
                if (use3) count3 += 1.0f;
                for (int j = 0; j < 32; ++j) {
                    float hat;
                    if (use3) {
                        const float q = sycl::clamp(
                            sycl::rint(wf[j] / s + (float)zi), -4.0f, 3.0f);
                        hat = s * (q - (float)zi);
                    } else {
                        const float q = sycl::clamp(
                            sycl::rint(wf[j] / s4v + (float)zp4), -8.0f, 7.0f);
                        hat = s4v * (q - (float)zp4);
                    }
                    tally(9, wf[j], hat);
                }
            }
            // 10: INT4 asymmetric, FP16 scale per k64 with z in the low
            //     nibble (4.25 bits/w) - the W4A4 DPAS-tile granularity
            //     (dpas.s4.s4.8.8 needs one weight scale per K=64 tile).
            if (kt & 1) {
                int zp = 0;
                const uint16_t sh = fit_int4_asym(wf64, 64, true, &zp);
                const float s = (float)sycl::bit_cast<sycl::half>(sh);
                for (int j = 0; j < 64; ++j) {
                    const float q = sycl::clamp(
                        sycl::rint(wf64[j] / s + (float)zp), -8.0f, 7.0f);
                    tally(10, wf64[j], s * (q - (float)zp));
                }
            }
        }
        float *out = d_acc + (size_t)row * kProbeStride;
        out[0] = na;
        for (int s = 0; s < kProbeSchemes; ++s) {
            out[1 + 2 * s] = dot[s];
            out[2 + 2 * s] = nb[s];
        }
        out[1 + 2 * kProbeSchemes] = count3;
    });
}

}  // namespace

// Reports cosine and relative L2 error for each candidate tier on one real
// matrix. Selects the layer's mlp_gate (17408x5120) - the shape that dominates
// the decode profile. Returns 0 when the shipping tier is the best 4-bit
// option at or below its own metadata budget.
extern "C" int qwn_w4_grid_probe(int layer) {
    if (!g_qwen.initialized || layer < 0 || layer >= g_qwen.L) return -1;
    tq_qmma_weight_t *w = &g_qwen.layers[layer].mlp_gate;
    if (w->s4_ready || w->s8_ready || !w->d_A || !w->d_block_scale_inv) return -2;
    if (!w->block_scaled || !w->e2m3 || w->row_major || w->word_major) return -3;
    const int M = w->M, K = w->K;
    const int nrb = (M + 127) / 128;
    float *d_g = static_cast<float *>(tq_dev_alloc((size_t)nrb * 4, "probe.global"));
    float *d_acc = static_cast<float *>(
        tq_dev_alloc((size_t)M * kProbeStride * 4, "probe.acc"));
    probe_block_absmax_kernel(w, d_g, nrb);
    probe_grid_kernel(w, d_g, d_acc);
    tq_q().wait_and_throw();
    std::vector<float> acc((size_t)M * kProbeStride);
    tq_d2h(acc.data(), d_acc, acc.size() * sizeof(float));
    tq_dev_free(d_g);
    tq_dev_free(d_acc);

    double na = 0.0;
    double dot[kProbeSchemes] = {};
    double nb[kProbeSchemes] = {};
    double count3 = 0.0;
    for (int row = 0; row < M; ++row) {
        const float *r = acc.data() + (size_t)row * kProbeStride;
        na += r[0];
        for (int s = 0; s < kProbeSchemes; ++s) {
            dot[s] += r[1 + 2 * s];
            nb[s] += r[2 + 2 * s];
        }
        count3 += r[1 + 2 * kProbeSchemes];
    }
    static const char *names[kProbeSchemes] = {
        "int4 sym    fp16/k32", "int4 asym   fp16/k32", "int4 asym   fp16/k16",
        "NVFP4 e2m1  e4m3/k16", "int8 sym    fp16/k32", "int4 asym   e4m3/k16",
        "int4 sym    e4m3/k16", "int3 sym    fp16/k32", "int3 asym   fp16/k32",
        "mix3/4 asym fp16/k32", "int4 asym   fp16/k64",
    };
    double bits[kProbeSchemes] = {4.50, 4.50, 5.00, 4.50, 8.50,
                                  4.75, 4.50, 3.50, 3.50, 0.0, 4.25};
    const double total_groups = (double)M * (K / 32);
    // Scheme 9's bits are measured: 3 or 4 code bits per group + fp16/k32.
    bits[9] = 0.5 + (3.0 * count3 + 4.0 * (total_groups - count3)) /
                        (total_groups > 0.0 ? total_groups : 1.0);
    std::printf("grid probe layer %d mlp_gate [%dx%d]\n", layer, M, K);
    std::printf("  %-22s %7s %10s %10s\n", "scheme", "bits/w", "cos", "rel_L2");
    int best = 1;
    for (int s = 0; s < kProbeSchemes; ++s) {
        const double c = (na > 0.0 && nb[s] > 0.0)
            ? dot[s] / (std::sqrt(na) * std::sqrt(nb[s])) : 0.0;
        // |w - w_hat|^2 = |w|^2 - 2<w,w_hat> + |w_hat|^2
        const double err = std::sqrt(std::max(0.0, na - 2.0 * dot[s] + nb[s]) / na);
        std::printf("  %-22s %7.2f %10.6f %10.6f\n", names[s], bits[s], c, err);
        if (s != 1 && bits[s] <= bits[1] + 1e-9) {
            const double cb = dot[best] / (std::sqrt(na) * std::sqrt(nb[best]));
            if (c > cb) best = s;
        }
    }
    std::printf("  best tier at <=%.2f bits/w: %s\n", bits[1], names[best]);
    return best == 1 ? 0 : 1;
}

// Timed RC8 GEMM on one real weight. Returns microseconds for `iters` calls
// (excluding activation quantization, which is amortized across all
// projections of a layer), or negative on error.
extern "C" double qwn_gemm_bench(int layer, int tokens, int iters) {
    if (!g_qwen.initialized || layer < 0 || layer >= g_qwen.L) return -1.0;
    tq_qmma_weight_t *w = &g_qwen.layers[layer].mlp_gate;
    if (!w->s4_ready || w->s4_k16) return -2.0;
    const int T = (tokens <= 0) ? 8 : ((tokens + 7) / 8) * 8;
    const int M = w->M, K = w->K, Kt = K / 32;
    const int n = (iters <= 0) ? 10 : iters;

    float *d_x = static_cast<float *>(tq_dev_alloc((size_t)T * K * 4, "gb.x"));
    float *d_y = static_cast<float *>(tq_dev_alloc((size_t)T * M * 4, "gb.y"));
    int8_t *d_aq = static_cast<int8_t *>(tq_dev_alloc((size_t)Kt * T * 32, "gb.aq"));
    float *d_as = static_cast<float *>(tq_dev_alloc((size_t)Kt * T * 4, "gb.as"));
    int32_t *d_asum = static_cast<int32_t *>(
        tq_dev_alloc((size_t)Kt * T * 4, "gb.asum"));
    tq_dev_zero(d_x, (size_t)T * K * 4);
    x_quantize_act_chunk(d_x, K, T, d_aq, d_as, d_asum);
    x_gemm_w4a8(w, d_aq, d_as, d_asum, d_y, T);   // warm up
    tq_q().wait_and_throw();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) x_gemm_w4a8(w, d_aq, d_as, d_asum, d_y, T);
    tq_q().wait_and_throw();
    const auto t1 = std::chrono::steady_clock::now();

    tq_dev_free(d_x); tq_dev_free(d_y);
    tq_dev_free(d_aq); tq_dev_free(d_as); tq_dev_free(d_asum);
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / n;
}

// Timed RC1 GEMV on the same weight qwn_gemm_bench uses, activation prepared
// once outside the loop (mirrors the GEMM bench excluding quantization).
// Returns microseconds per call, or negative on error. Level-up queue item 7:
// 8 x this number vs qwn_gemm_bench(layer, 8, n) prices the spec wave.
extern "C" double qwn_gemv_bench(int layer, int iters) {
    if (!g_qwen.initialized || layer < 0 || layer >= g_qwen.L) return -1.0;
    tq_qmma_weight_t *w = &g_qwen.layers[layer].mlp_gate;
    if (!w->s4_ready) return -2.0;
    const int M = w->M, K = w->K;
    const int n = (iters <= 0) ? 50 : iters;
    float *d_x = static_cast<float *>(tq_dev_alloc((size_t)K * 4, "gvb.x"));
    float *d_y = static_cast<float *>(tq_dev_alloc((size_t)M * 4, "gvb.y"));
    tq_dev_zero(d_x, (size_t)K * 4);
    x_prepare_gemv_act_s8(d_x, K);
    if (x_gemv_w4a8_prepared(w, d_y) != 0) {   // warm up
        tq_dev_free(d_x); tq_dev_free(d_y);
        return -3.0;
    }
    tq_q().wait_and_throw();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) x_gemv_w4a8_prepared(w, d_y);
    tq_q().wait_and_throw();
    const auto t1 = std::chrono::steady_clock::now();
    tq_dev_free(d_x); tq_dev_free(d_y);
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / n;
}

// Spec-wave prefix stability: two T=8 waves share activation rows 0-3 but
// differ in rows 4-7; the shared-prefix output rows must be BYTE-identical
// across both runs. This is the hardware contract speculative decoding
// leans on: an accepted prefix must not depend on the rejected draft rows
// riding the same wave (DPAS row independence + per-token quantization).
// Returns 0 on a byte-exact prefix, negative on error, 1 on mismatch.
extern "C" int qwn_specwave_check(int layer) {
    if (!g_qwen.initialized || layer < 0 || layer >= g_qwen.L) return -1;
    tq_qmma_weight_t *w = &g_qwen.layers[layer].mlp_gate;
    if (!w->s4_ready || w->s4_k16) return -2;
    const int T = 8, kPrefix = 4;
    const int M = w->M, K = w->K, Kt = K / 32;
    float *d_x = static_cast<float *>(tq_dev_alloc((size_t)T * K * 4, "sw.x"));
    float *d_y0 = static_cast<float *>(tq_dev_alloc((size_t)T * M * 4, "sw.y0"));
    float *d_y1 = static_cast<float *>(tq_dev_alloc((size_t)T * M * 4, "sw.y1"));
    int8_t *d_aq = static_cast<int8_t *>(tq_dev_alloc((size_t)Kt * T * 32, "sw.aq"));
    float *d_as = static_cast<float *>(tq_dev_alloc((size_t)Kt * T * 4, "sw.as"));
    int32_t *d_asum = static_cast<int32_t *>(
        tq_dev_alloc((size_t)Kt * T * 4, "sw.asum"));
    float *outs[2] = {d_y0, d_y1};
    for (int run = 0; run < 2; ++run) {
        // rows < kPrefix: identical fill both runs; rows >= kPrefix: salted.
        const uint32_t salt = run ? 0x9E3779B9u : 0u;
        tq_q().parallel_for(sycl::range<1>((size_t)T * K), [=](sycl::id<1> id) {
            const int i = (int)id[0];
            const int t = i / K, k = i - t * K;
            const uint32_t s = (t >= kPrefix) ? salt : 0u;
            const float mag =
                sycl::exp2((float)(((k * 7) + t * 3 + (int)(s & 7)) % 13) - 6.0f);
            const float sgn =
                (((uint32_t)(k * 2654435761u + t) ^ s) >> 31) ? -1.0f : 1.0f;
            d_x[i] = ((k % 97) == 0) ? 0.0f : sgn * mag;
        });
        tq_q().wait_and_throw();
        x_quantize_act_chunk(d_x, K, T, d_aq, d_as, d_asum);
        if (x_gemm_w4a8(w, d_aq, d_as, d_asum, outs[run], T) != 0) {
            tq_dev_free(d_x); tq_dev_free(d_y0); tq_dev_free(d_y1);
            tq_dev_free(d_aq); tq_dev_free(d_as); tq_dev_free(d_asum);
            return -3;
        }
        tq_q().wait_and_throw();
    }
    std::vector<float> h0((size_t)kPrefix * M), h1((size_t)kPrefix * M);
    tq_d2h(h0.data(), d_y0, h0.size() * sizeof(float));
    tq_d2h(h1.data(), d_y1, h1.size() * sizeof(float));
    tq_dev_free(d_x); tq_dev_free(d_y0); tq_dev_free(d_y1);
    tq_dev_free(d_aq); tq_dev_free(d_as); tq_dev_free(d_asum);
    const int same =
        std::memcmp(h0.data(), h1.data(), h0.size() * sizeof(float)) == 0;
    std::printf("qwn_specwave_check layer %d: prefix rows 0-%d (%zu bytes) %s\n",
                layer, kPrefix - 1, h0.size() * sizeof(float),
                same ? "byte-exact PASS" : "MISMATCH FAIL");
    return same ? 0 : 1;
}

// Gate: the RC8 batched GEMM against the proven RC1 GEMV, same repacked
// weight and same activation rows. The GEMV path is already certified against
// the scalar E2M3 reference, so agreeing with it end-to-end validates the RC8
// operand layout, the K-tile-major activation staging and the epilogue at
// once. Run on a model where the layer's mlp_gate is already s4 (the shipping
// default). Prints per-row cosine and max relative error; returns 0 when every
// token row clears cos >= 0.9999.
extern "C" int qwn_gemm_check(int layer, int tokens) {
    if (!g_qwen.initialized || layer < 0 || layer >= g_qwen.L) return -1;
    tq_qmma_weight_t *w = &g_qwen.layers[layer].mlp_gate;
    if (!w->s4_ready || w->s4_k16) return -2;
    const int T = (tokens <= 0) ? 8 : ((tokens + 7) / 8) * 8;
    const int M = w->M, K = w->K, Kt = K / 32;

    float *d_x = static_cast<float *>(tq_dev_alloc((size_t)T * K * 4, "gc.x"));
    float *d_ref = static_cast<float *>(tq_dev_alloc((size_t)T * M * 4, "gc.ref"));
    float *d_got = static_cast<float *>(tq_dev_alloc((size_t)T * M * 4, "gc.got"));
    int8_t *d_aq = static_cast<int8_t *>(tq_dev_alloc((size_t)Kt * T * 32, "gc.aq"));
    float *d_as = static_cast<float *>(tq_dev_alloc((size_t)Kt * T * 4, "gc.as"));
    int32_t *d_asum = static_cast<int32_t *>(
        tq_dev_alloc((size_t)Kt * T * 4, "gc.asum"));

    // Deterministic spread with sign flips and exact zeros, per token row.
    tq_q().parallel_for(sycl::range<1>((size_t)T * K), [=](sycl::id<1> id) {
        const int i = (int)id[0];
        const int t = i / K, k = i - t * K;
        const float mag = sycl::exp2((float)(((k * 7) + t * 3) % 13) - 6.0f);
        const float sgn = (((uint32_t)(k * 2654435761u + t)) >> 31) ? -1.0f : 1.0f;
        d_x[i] = ((k % 97) == 0) ? 0.0f : sgn * mag;
    });
    tq_q().wait_and_throw();

    // Reference: the RC1 GEMV, one token row at a time.
    for (int t = 0; t < T; ++t) {
        const int rc = x_gemv_w4a8(w, d_x + (size_t)t * K, d_ref + (size_t)t * M);
        if (rc != 0) {
            tq_dev_free(d_x); tq_dev_free(d_ref); tq_dev_free(d_got);
            tq_dev_free(d_aq); tq_dev_free(d_as); tq_dev_free(d_asum);
            return -3;
        }
    }
    x_quantize_act_chunk(d_x, K, T, d_aq, d_as, d_asum);
    const int rc = x_gemm_w4a8(w, d_aq, d_as, d_asum, d_got, T);
    tq_q().wait_and_throw();
    if (rc != 0) {
        tq_dev_free(d_x); tq_dev_free(d_ref); tq_dev_free(d_got);
        tq_dev_free(d_aq); tq_dev_free(d_as); tq_dev_free(d_asum);
        return -4;
    }

    std::vector<float> ref((size_t)T * M), got((size_t)T * M);
    tq_d2h(ref.data(), d_ref, ref.size() * sizeof(float));
    tq_d2h(got.data(), d_got, got.size() * sizeof(float));
    tq_dev_free(d_x); tq_dev_free(d_ref); tq_dev_free(d_got);
    tq_dev_free(d_aq); tq_dev_free(d_as); tq_dev_free(d_asum);

    double worst_cos = 1.0, worst_rel = 0.0;
    int worst_row = -1;
    for (int t = 0; t < T; ++t) {
        double dot = 0.0, na = 0.0, nb = 0.0, rel = 0.0;
        for (int i = 0; i < M; ++i) {
            const double a = ref[(size_t)t * M + i], b = got[(size_t)t * M + i];
            dot += a * b; na += a * a; nb += b * b;
            const double den = std::fabs(a) + 1e-3;
            rel = std::max(rel, std::fabs(b - a) / den);
        }
        const double c = (na > 0.0 && nb > 0.0)
            ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
        if (c < worst_cos) { worst_cos = c; worst_row = t; }
        worst_rel = std::max(worst_rel, rel);
    }
    std::printf("qwn_gemm_check layer %d [%dx%d] T=%d: worst cos=%.8f (row %d) "
                "max_rel=%.4f %s\n",
                layer, M, K, T, worst_cos, worst_row, worst_rel,
                worst_cos >= 0.9999 ? "PASS" : "FAIL");
    return worst_cos >= 0.9999 ? 0 : -5;
}

// Gate: the W4A4 GEMM (s4 acts, dpas.s4.s4 at K=64) against the RC1 GEMV
// reference on the SAME k64-scaled weight. The GEMV under k64 scales is
// certified against the scalar tier by qwn_w4_check (TQ_XPU_K64=1), so
// agreeing with it validates the K64 B routing, the s4 A layout, the s4
// activation quantizer, and the per-K64 epilogue end-to-end. Weights must be
// loaded with TQ_XPU_K64=1. Looser band than the W4A8 gate: s4 activations
// carry real quantization error on top of the shared weight error. Measured
// on silicon: worst row cos 0.9962-0.9963 across layers 0/1/32 - exactly the
// analytic s4-act noise over K=5120 (the CUDA twin ships E4M3 activations,
// the same error class). Gate at 0.995 catches routing/sign bugs (those
// crater cos); prefill_check and the TF ladder remain the shipping arbiters.
extern "C" int qwn_gemm_w4a4_check(int layer, int tokens) {
    if (!g_qwen.initialized || layer < 0 || layer >= g_qwen.L) return -1;
    tq_qmma_weight_t *w = &g_qwen.layers[layer].mlp_gate;
    if (!w->s4_ready || !w->s4_k64) return -2;
    const int T = (tokens <= 0) ? 8 : ((tokens + 7) / 8) * 8;
    const int M = w->M, K = w->K, Kt64 = K / 64;

    float *d_x = static_cast<float *>(tq_dev_alloc((size_t)T * K * 4, "g4c.x"));
    float *d_ref = static_cast<float *>(tq_dev_alloc((size_t)T * M * 4, "g4c.ref"));
    float *d_got = static_cast<float *>(tq_dev_alloc((size_t)T * M * 4, "g4c.got"));
    uint8_t *d_aq4 = static_cast<uint8_t *>(
        tq_dev_alloc((size_t)Kt64 * T * 32, "g4c.aq4"));
    float *d_as4 = static_cast<float *>(tq_dev_alloc((size_t)Kt64 * T * 4, "g4c.as4"));
    int32_t *d_asum4 = static_cast<int32_t *>(
        tq_dev_alloc((size_t)Kt64 * T * 4, "g4c.asum4"));
    auto cleanup = [&]() {
        tq_dev_free(d_x); tq_dev_free(d_ref); tq_dev_free(d_got);
        tq_dev_free(d_aq4); tq_dev_free(d_as4); tq_dev_free(d_asum4);
    };

    tq_q().parallel_for(sycl::range<1>((size_t)T * K), [=](sycl::id<1> id) {
        const int i = (int)id[0];
        const int t = i / K, k = i - t * K;
        const float mag = sycl::exp2((float)(((k * 7) + t * 3) % 13) - 6.0f);
        const float sgn = (((uint32_t)(k * 2654435761u + t)) >> 31) ? -1.0f : 1.0f;
        d_x[i] = ((k % 97) == 0) ? 0.0f : sgn * mag;
    });
    tq_q().wait_and_throw();

    for (int t = 0; t < T; ++t) {
        const int rc = x_gemv_w4a8(w, d_x + (size_t)t * K, d_ref + (size_t)t * M);
        if (rc != 0) { cleanup(); return -3; }
    }
    x_quantize_act_chunk_s4(d_x, K, T, d_aq4, d_as4, d_asum4);
    const int rc = x_gemm_w4a4(w, d_aq4, d_as4, d_asum4, d_got, T);
    tq_q().wait_and_throw();
    if (rc != 0) { cleanup(); return -4; }

    std::vector<float> ref((size_t)T * M), got((size_t)T * M);
    tq_d2h(ref.data(), d_ref, ref.size() * sizeof(float));
    tq_d2h(got.data(), d_got, got.size() * sizeof(float));
    cleanup();

    double worst_cos = 1.0, worst_rel = 0.0;
    int worst_row = -1;
    for (int t = 0; t < T; ++t) {
        double dot = 0.0, na = 0.0, nb = 0.0, rel = 0.0;
        for (int i = 0; i < M; ++i) {
            const double a = ref[(size_t)t * M + i], b = got[(size_t)t * M + i];
            dot += a * b; na += a * a; nb += b * b;
            const double den = std::fabs(a) + 1e-3;
            rel = std::max(rel, std::fabs(b - a) / den);
        }
        const double c = (na > 0.0 && nb > 0.0)
            ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
        if (c < worst_cos) { worst_cos = c; worst_row = t; }
        worst_rel = std::max(worst_rel, rel);
    }
    std::printf("qwn_gemm_w4a4_check layer %d [%dx%d] T=%d: worst cos=%.8f "
                "(row %d) max_rel=%.4f %s\n",
                layer, M, K, T, worst_cos, worst_row, worst_rel,
                worst_cos >= 0.995 ? "PASS" : "FAIL");
    return worst_cos >= 0.995 ? 0 : -5;
}

// Timed W4A4 GEMM, same conventions as qwn_gemm_bench (activation quant
// excluded, microseconds per call, negative on error). Weights need
// TQ_XPU_K64=1.
extern "C" double qwn_gemm_bench_w4a4(int layer, int tokens, int iters) {
    if (!g_qwen.initialized || layer < 0 || layer >= g_qwen.L) return -1.0;
    tq_qmma_weight_t *w = &g_qwen.layers[layer].mlp_gate;
    if (!w->s4_ready || !w->s4_k64) return -2.0;
    const int T = (tokens <= 0) ? 8 : ((tokens + 7) / 8) * 8;
    const int K = w->K, Kt64 = K / 64;
    const int n = (iters <= 0) ? 10 : iters;

    float *d_x = static_cast<float *>(tq_dev_alloc((size_t)T * K * 4, "g4b.x"));
    float *d_y = static_cast<float *>(tq_dev_alloc((size_t)T * w->M * 4, "g4b.y"));
    uint8_t *d_aq4 = static_cast<uint8_t *>(
        tq_dev_alloc((size_t)Kt64 * T * 32, "g4b.aq4"));
    float *d_as4 = static_cast<float *>(tq_dev_alloc((size_t)Kt64 * T * 4, "g4b.as4"));
    int32_t *d_asum4 = static_cast<int32_t *>(
        tq_dev_alloc((size_t)Kt64 * T * 4, "g4b.asum4"));
    tq_dev_zero(d_x, (size_t)T * K * 4);
    x_quantize_act_chunk_s4(d_x, K, T, d_aq4, d_as4, d_asum4);
    x_gemm_w4a4(w, d_aq4, d_as4, d_asum4, d_y, T);   // warm up
    tq_q().wait_and_throw();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) x_gemm_w4a4(w, d_aq4, d_as4, d_asum4, d_y, T);
    tq_q().wait_and_throw();
    const auto t1 = std::chrono::steady_clock::now();

    tq_dev_free(d_x); tq_dev_free(d_y);
    tq_dev_free(d_aq4); tq_dev_free(d_as4); tq_dev_free(d_asum4);
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / n;
}

// Gate: scalar-E2M3 GEMV vs the W4A8 DPAS path on one real weight (layer's
// mlp_gate). Must run with TQ_XPU_W4A8=0 so the E2M3 payload is still
// resident; the check repacks the weight afterwards, so it is s4 from then on.
// Gate for the TP weight-shard layout math, on ONE card so a layout bug
// cannot hide behind TP plumbing. Uses the layer's mlp_gate (M=17408,
// K=5120):
//   column shard - the two half-M shards concatenated must be BIT-identical
//     to the full GEMV: same codes, same k-walk, same reduction order.
//   row shard    - the two half-K partials summed must match the full GEMV
//     in the eps band only: a K-split dot product sums in a different order,
//     which is exactly why TP2 cannot be bit-exact against single card.
// Prints cos + max-rel per mode; returns 0 when column is exact and row is
// within cos >= 0.9999.
// Validates x_w4_shard against EVERY projection family at its real shape and
// its intended TP mode, so the loader loop over all 496 weights cannot trip
// a divisibility or scale-layout constraint at load time. Column shards must
// reassemble bit-exactly (per-group, since packed outputs interleave shards);
// row shards must agree in the eps band. Returns the number of failures.
extern "C" int qwn_shard_check_all(int layer) {
    if (!g_qwen.initialized || layer < 0 || layer >= g_qwen.L) return -1;
    tq_layer_t *l = &g_qwen.layers[layer];
    const int lt = g_qwen.layer_types[layer];
    const int hd = g_qwen.hd;
    const int key_dim = g_qwen.linear_num_key_heads * g_qwen.linear_key_head_dim;
    const int val_dim = g_qwen.linear_num_value_heads * g_qwen.linear_value_head_dim;

    struct Item {
        const char *name;
        tq_qmma_weight_t *w;
        int mode;              // 0 column, 1 row
        int groups[3];
        int ngroups;
    };
    Item items[12];
    int n = 0;
    if (lt == TQ_LAYER_FULL_ATTENTION) {
        // q_proj is head-major [q(hd) | gate(hd)] per head (qg_base =
        // head*2*hd), so a head shard is one contiguous range: no grouping.
        items[n++] = {"q_proj", &l->q_proj, 0, {0, 0, 0}, 0};
        items[n++] = {"k_proj", &l->k_proj, 0, {0, 0, 0}, 0};
        items[n++] = {"v_proj", &l->v_proj, 0, {0, 0, 0}, 0};
        items[n++] = {"o_proj", &l->o_proj, 1, {0, 0, 0}, 0};
    } else {
        // linear_in_qkv is group-major [q(key_dim) | k(key_dim) | v(val_dim)],
        // so it needs the grouped column shard.
        items[n++] = {"lin_qkv", &l->linear_in_qkv, 0,
                      {key_dim, key_dim, val_dim}, 3};
        items[n++] = {"lin_z", &l->linear_in_z, 0, {0, 0, 0}, 0};
        // M=48 (one scalar per value head): 24 rows per rank is not a
        // multiple of the packed 16-row group, so these replicate.
        items[n++] = {"lin_b", &l->linear_in_b, 2, {0, 0, 0}, 0};
        items[n++] = {"lin_a", &l->linear_in_a, 2, {0, 0, 0}, 0};
        items[n++] = {"lin_out", &l->linear_out, 1, {0, 0, 0}, 0};
    }
    items[n++] = {"mlp_gate", &l->mlp_gate, 0, {0, 0, 0}, 0};
    items[n++] = {"mlp_up", &l->mlp_up, 0, {0, 0, 0}, 0};
    items[n++] = {"mlp_down", &l->mlp_down, 1, {0, 0, 0}, 0};
    (void)hd;

    int fails = 0;
    for (int i = 0; i < n; ++i) {
        Item &it = items[i];
        tq_qmma_weight_t *w = it.w;
        if (!w->s4_ready || w->s4_k16) {
            std::printf("  %-9s SKIP (not s4 / k16 layout)\n", it.name);
            continue;
        }
        const int M = w->M, K = w->K;
        float *d_x = static_cast<float *>(tq_dev_alloc((size_t)K * 4, "sa.x"));
        float *d_ref = static_cast<float *>(tq_dev_alloc((size_t)M * 4, "sa.ref"));
        float *d_got = static_cast<float *>(tq_dev_alloc((size_t)M * 4, "sa.got"));
        float *d_tmp = static_cast<float *>(tq_dev_alloc((size_t)M * 4, "sa.tmp"));
        std::vector<float> hx(K);
        for (int j = 0; j < K; ++j)
            hx[j] = 0.05f * (float)((j * 1103515245u + 12345u) % 97) - 2.0f;
        tq_h2d(d_x, hx.data(), (size_t)K * 4);
        x_prepare_gemv_act_s8(d_x, K);
        if (x_gemv_w4a8_prepared(w, d_ref) != 0) {
            std::printf("  %-9s FAIL reference gemv\n", it.name);
            ++fails;
            tq_dev_free(d_x); tq_dev_free(d_ref);
            tq_dev_free(d_got); tq_dev_free(d_tmp);
            continue;
        }
        tq_qmma_weight_t s0{}, s1{};
        int rc = x_w4_shard(&s0, w, 0, 2, it.mode,
                            it.ngroups ? it.groups : nullptr, it.ngroups);
        if (rc == 0)
            rc = x_w4_shard(&s1, w, 1, 2, it.mode,
                            it.ngroups ? it.groups : nullptr, it.ngroups);
        if (rc != 0) {
            std::printf("  %-9s M=%-6d K=%-6d mode=%s FAIL shard rc=%d\n",
                        it.name, M, K,
                        it.mode == 0 ? "col" : it.mode == 1 ? "row" : "rep", rc);
            ++fails;
            x_w4_shard_free(&s0); x_w4_shard_free(&s1);
            tq_dev_free(d_x); tq_dev_free(d_ref);
            tq_dev_free(d_got); tq_dev_free(d_tmp);
            continue;
        }
        if (it.mode == 0) {
            // Reassemble per group: rank r owns rows [base + r*gper, +gper)
            // of each group, so a packed output interleaves the two shards.
            x_prepare_gemv_act_s8(d_x, K);
            x_gemv_w4a8_prepared(&s0, d_tmp);              // rank 0 rows
            x_gemv_w4a8_prepared(&s1, d_tmp + M / 2);      // rank 1 rows
            tq_q().wait_and_throw();
            std::vector<float> part(M), out(M);
            tq_d2h(part.data(), d_tmp, (size_t)M * 4);
            const int ng = it.ngroups ? it.ngroups : 1;
            int gs[3] = {it.ngroups ? it.groups[0] : M,
                         it.ngroups ? it.groups[1] : 0,
                         it.ngroups ? it.groups[2] : 0};
            size_t base = 0, c0 = 0, c1 = (size_t)M / 2;
            for (int g = 0; g < ng; ++g) {
                const size_t gper = (size_t)gs[g] / 2;
                for (size_t j = 0; j < gper; ++j) out[base + j] = part[c0 + j];
                for (size_t j = 0; j < gper; ++j)
                    out[base + gper + j] = part[c1 + j];
                base += (size_t)gs[g];
                c0 += gper;
                c1 += gper;
            }
            tq_h2d(d_got, out.data(), (size_t)M * 4);
        } else if (it.mode == 2) {
            // Replicated: each rank produces the FULL output, so rank 0's
            // result alone must be bit-identical to the reference.
            x_prepare_gemv_act_s8(d_x, K);
            x_gemv_w4a8_prepared(&s0, d_got);
            tq_q().wait_and_throw();
        } else {
            x_prepare_gemv_act_s8(d_x, K / 2);
            x_gemv_w4a8_prepared(&s0, d_got);
            x_prepare_gemv_act_s8(d_x + K / 2, K / 2);
            x_gemv_w4a8_prepared(&s1, d_tmp);
            x_add_inplace(d_got, d_tmp, M);
            tq_q().wait_and_throw();
        }
        std::vector<float> a(M), b(M);
        tq_d2h(a.data(), d_ref, (size_t)M * 4);
        tq_d2h(b.data(), d_got, (size_t)M * 4);
        double dot = 0, na = 0, nb = 0, mrel = 0;
        int exact = 1;
        for (int j = 0; j < M; ++j) {
            dot += (double)a[j] * b[j];
            na += (double)a[j] * a[j];
            nb += (double)b[j] * b[j];
            if (a[j] != b[j]) exact = 0;
            const double d = std::fabs((double)a[j] - b[j]);
            const double s = std::fabs((double)a[j]) + 1e-6;
            if (d / s > mrel) mrel = d / s;
        }
        const double c =
            (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
        const bool want_exact = (it.mode != 1);   // only a K-split reassociates
        const bool ok = want_exact ? (exact != 0) : (c >= 0.9999);
        if (!ok) ++fails;
        std::printf("  %-9s M=%-6d K=%-6d mode=%s cos=%.8f max_rel=%-9.3g "
                    "exact=%-3s %s\n", it.name, M, K,
                    it.mode == 0 ? "col" : it.mode == 1 ? "row" : "rep",
                    c, mrel, exact ? "yes" : "no", ok ? "PASS" : "FAIL");
        x_w4_shard_free(&s0);
        x_w4_shard_free(&s1);
        tq_dev_free(d_x); tq_dev_free(d_ref);
        tq_dev_free(d_got); tq_dev_free(d_tmp);
    }
    return fails;
}

extern "C" int qwn_shard_check(int layer) {
    if (!g_qwen.initialized || layer < 0 || layer >= g_qwen.L) return -1;
    tq_qmma_weight_t *w = &g_qwen.layers[layer].mlp_gate;
    if (!w->s4_ready || w->s4_k16) return -2;
    const int M = w->M, K = w->K;
    float *d_x = static_cast<float *>(tq_dev_alloc((size_t)K * 4, "sh.x"));
    float *d_ref = static_cast<float *>(tq_dev_alloc((size_t)M * 4, "sh.ref"));
    float *d_got = static_cast<float *>(tq_dev_alloc((size_t)M * 4, "sh.got"));
    float *d_tmp = static_cast<float *>(tq_dev_alloc((size_t)M * 4, "sh.tmp"));
    std::vector<float> hx(K);
    for (int i = 0; i < K; ++i)
        hx[i] = 0.05f * (float)((i * 1103515245u + 12345u) % 97) - 2.0f;
    tq_h2d(d_x, hx.data(), (size_t)K * 4);

    x_prepare_gemv_act_s8(d_x, K);
    if (x_gemv_w4a8_prepared(w, d_ref) != 0) return -3;

    auto cmp = [&](const char *what, float *got, double gate) {
        std::vector<float> a(M), b(M);
        tq_d2h(a.data(), d_ref, (size_t)M * 4);
        tq_d2h(b.data(), got, (size_t)M * 4);
        double dot = 0, na = 0, nb = 0, mrel = 0;
        int exact = 1;
        for (int i = 0; i < M; ++i) {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
            if (a[i] != b[i]) exact = 0;
            const double d = std::fabs((double)a[i] - b[i]);
            const double s = std::fabs((double)a[i]) + 1e-6;
            if (d / s > mrel) mrel = d / s;
        }
        const double c = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
        std::printf("qwn_shard_check %-8s cos=%.8f max_rel=%.3g bit_exact=%s %s\n",
                    what, c, mrel, exact ? "yes" : "no",
                    c >= gate ? "PASS" : "FAIL");
        return c >= gate ? (exact ? 2 : 1) : 0;
    };

    // ---- column-parallel: concat of halves must be bit-exact -------------
    tq_qmma_weight_t sa{}, sb{};
    int rc = x_w4_shard(&sa, w, 0, 2, 0, nullptr, 0);
    if (rc == 0) rc = x_w4_shard(&sb, w, 1, 2, 0, nullptr, 0);
    int col = -1;
    if (rc == 0) {
        x_prepare_gemv_act_s8(d_x, K);
        if (x_gemv_w4a8_prepared(&sa, d_got) == 0 &&
            x_gemv_w4a8_prepared(&sb, d_got + M / 2) == 0) {
            tq_q().wait_and_throw();
            col = cmp("column", d_got, 0.99999999);
        }
    } else {
        std::printf("qwn_shard_check column shard rc=%d\n", rc);
    }
    x_w4_shard_free(&sa);
    x_w4_shard_free(&sb);

    // ---- row-parallel: summed partials, eps band -------------------------
    tq_qmma_weight_t ra{}, rb{};
    rc = x_w4_shard(&ra, w, 0, 2, 1, nullptr, 0);
    if (rc == 0) rc = x_w4_shard(&rb, w, 1, 2, 1, nullptr, 0);
    int row = -1;
    if (rc == 0) {
        x_prepare_gemv_act_s8(d_x, K / 2);
        if (x_gemv_w4a8_prepared(&ra, d_got) == 0) {
            x_prepare_gemv_act_s8(d_x + K / 2, K / 2);
            if (x_gemv_w4a8_prepared(&rb, d_tmp) == 0) {
                x_add_inplace(d_got, d_tmp, M);
                tq_q().wait_and_throw();
                row = cmp("row", d_got, 0.9999);
            }
        }
    } else {
        std::printf("qwn_shard_check row shard rc=%d\n", rc);
    }
    x_w4_shard_free(&ra);
    x_w4_shard_free(&rb);

    tq_dev_free(d_x); tq_dev_free(d_ref); tq_dev_free(d_got); tq_dev_free(d_tmp);
    return (col == 2 && row >= 1) ? 0 : -4;
}

// Prints cos + max-rel; returns 0 on cos >= 0.99.
extern "C" int qwn_w4_check(int layer) {
    if (!g_qwen.initialized || layer < 0 || layer >= g_qwen.L) return -1;
    tq_qmma_weight_t *w = &g_qwen.layers[layer].mlp_gate;
    if (w->s4_ready) return -2;   // already repacked; nothing to compare against
    const int M = w->M, K = w->K;
    float *d_x = static_cast<float *>(tq_dev_alloc((size_t)K * 4, "w4chk.x"));
    float *d_ref = static_cast<float *>(tq_dev_alloc((size_t)M * 4, "w4chk.ref"));
    float *d_got = static_cast<float *>(tq_dev_alloc((size_t)M * 4, "w4chk.got"));
    // deterministic adversarial-ish fill: sign flips, magnitude spread, zeros
    tq_q().parallel_for(sycl::range<1>((size_t)K), [=](sycl::id<1> i) {
        const int j = (int)i[0];
        const float mag = sycl::exp2((float)((j * 7) % 13) - 6.0f);
        const float sgn = ((j * 2654435761u) >> 31) ? -1.0f : 1.0f;
        d_x[j] = ((j % 97) == 0) ? 0.0f : sgn * mag;
    });
    int rc = x_gemv_qmma(w, d_x, d_ref);          // scalar E2M3 reference
    if (rc != 0) { tq_dev_free(d_x); tq_dev_free(d_ref); tq_dev_free(d_got); return -3; }
    // TQ_XPU_K64=1 certifies the k64-scaled GEMV against the same scalar
    // reference (the absolute anchor for the W4A4 chain).
    const char *k64_env = std::getenv("TQ_XPU_K64");
    rc = x_w4_repack_weight_mode(w, (k64_env && k64_env[0] == '1') ? 2 : 0);
    if (rc != 0) { tq_dev_free(d_x); tq_dev_free(d_ref); tq_dev_free(d_got); return -4; }
    rc = x_gemv_w4a8(w, d_x, d_got);
    if (rc != 0) { tq_dev_free(d_x); tq_dev_free(d_ref); tq_dev_free(d_got); return -5; }
    tq_q().wait_and_throw();
    std::vector<float> ref(M), got(M);
    tq_d2h(ref.data(), d_ref, (size_t)M * 4);
    tq_d2h(got.data(), d_got, (size_t)M * 4);
    double dot = 0, na = 0, nb = 0, worst = 0;
    for (int i = 0; i < M; i++) {
        dot += (double)ref[i] * got[i];
        na += (double)ref[i] * ref[i];
        nb += (double)got[i] * got[i];
        const double denom = sycl::fabs((double)ref[i]) + 1e-3;
        worst = std::max(worst, (double)sycl::fabs(got[i] - ref[i]) / denom);
    }
    const double c = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    std::printf("qwn_w4_check layer %d [%dx%d]: cos=%.6f max_rel=%.4f %s\n",
                layer, M, K, c, worst, c >= 0.99 ? "PASS" : "FAIL");
    tq_dev_free(d_x); tq_dev_free(d_ref); tq_dev_free(d_got);
    return c >= 0.99 ? 0 : -6;
}
