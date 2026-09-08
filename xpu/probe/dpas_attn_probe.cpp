// dpas_attn_probe.cpp — level-up design 9.5 / queue 3b, probe-first step:
// price the DPAS-matrix QK^T decode walk against the shipping scalar form
// before any engine surgery.
//
// Question: serving one GQA group (6 query heads, hd=256) over a DRAM-sized
// E4M3 K cache, does S = Q x K^T through dpas.bf.bf.8.8 (heads packed as
// M-rows, fp32 accumulators, GRF128 kept) beat the scalar shipping shape
// (3 heads per row-visit, two passes over the cache)?
//
// Layout facts this probe leans on (verified on silicon by the layout probes
// and the engine's own kernels):
//   - B fragment, bf16 K=16: flat dword (k/2)*16+n; register element i of
//     lane n = k-pair (2i, 2i+1) of column n. Column n = token n, so lane n's
//     uintv8 IS its token's contiguous 16-dim K slice as packed bf16 pairs:
//     per-lane e4m3 16-byte load + decode + pack, no transpose, no SLM.
//   - A fragment: 8 rows x 32 bytes contiguous; loaded with the engine's
//     apair/abyte idiom (row (2i+apair), byte (lane&7)*4) - reused verbatim,
//     Q is pre-packed on host as 8 rows (6 live heads + 2 zero rows).
//   - E4M3 values are exact in bf16 (3-bit mantissa, small exponent range),
//     so arm B's only extra rounding is Q fp32->bf16: score cos gate 0.9997.
//
// Build: icpx -fsycl -O2 -o dpas_attn_probe dpas_attn_probe.cpp
// Run:   ./dpas_attn_probe [device_index]

#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstring>
#include <vector>

using uintv4 = uint32_t __attribute__((ext_vector_type(4)));
using uintv8 = uint32_t __attribute__((ext_vector_type(8)));
using floatv8 = float __attribute__((ext_vector_type(8)));

static constexpr int kSG = 16;
static constexpr int kHd = 256;
static constexpr int kN = 131072;      // 32 MB e4m3 cache, past L2
static constexpr int kH = 6;           // GQA group heads

#ifdef __SYCL_DEVICE_ONLY__
#define DPAS_BF_RC8(dd, aa, bb)                                                \
    asm("{\n"                                                                  \
        ".decl DST     v_type=G type=F num_elts=128 alias=<%0,0>\n"            \
        ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"           \
        ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"            \
        "dpas.bf.bf.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"         \
        "}\n"                                                                  \
        : "+rw"(dd) : "rw"(aa), "rw"(bb))
#else
#define DPAS_BF_RC8(dd, aa, bb) do { (void)(aa); (void)(bb); } while (0)
#endif

static inline float e4m3_to_float(uint8_t c) {
    const uint32_t b =
        ((uint32_t)(c & 0x7Fu) << 20) | ((uint32_t)(c & 0x80u) << 24);
    float f;
    std::memcpy(&f, &b, 4);
    return f * 0x1p120f;
}

// Arm A: the shipping scalar shape - one subgroup strides token rows, three
// query heads' dots per row visit, q register-hoisted (int4kv lesson 26
// methodology: never accept a walk below ~70% roofline as evidence).
struct walk_scalar3_k {
    const uint8_t *codes;   // [N][hd]
    const float *rscale;    // [N]
    const float *q3;        // [3][hd]
    float *scores;          // [3][N]
    int nsg;
    [[sycl::reqd_sub_group_size(kSG)]] void
    operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int sgid = (int)(it.get_global_linear_id() / kSG);
        const int lane = (int)sg.get_local_linear_id();
        constexpr int kDpl = kHd / kSG;
        float qr[3][kDpl];
        for (int h = 0; h < 3; ++h)
            for (int d = 0; d < kDpl; ++d)
                qr[h][d] = q3[(size_t)h * kHd + lane * kDpl + d];
        for (int row = sgid; row < kN; row += nsg) {
            const uint8_t *kr = codes + (size_t)row * kHd + lane * kDpl;
            float dot[3] = {0.0f, 0.0f, 0.0f};
            for (int d = 0; d < kDpl; ++d) {
                const uint32_t c = kr[d];
                const uint32_t b = ((c & 0x7Fu) << 20) | ((c & 0x80u) << 24);
                const float v = sycl::bit_cast<float>(b) * 0x1p120f;
                for (int h = 0; h < 3; ++h) dot[h] += v * qr[h][d];
            }
            const float rs = rscale[row];
            for (int h = 0; h < 3; ++h) {
                const float s = sycl::reduce_over_group(sg, dot[h],
                                                        sycl::plus<float>());
                if (lane == 0) scores[(size_t)h * kN + row] = s * rs;
            }
        }
    }
};

// Arm B: DPAS-matrix form. One subgroup per stride of 16-token tiles; per
// tile: 16 chained dpas.bf.bf.8.8 over the 16 k-chunks; lane n rebuilds its
// token's bf16 B slices from E4M3 in registers. Q lives as pre-packed A
// fragments (16 chunks x 64 dwords, engine apair/abyte layout).
struct walk_dpas6_k {
    const uint8_t *codes;   // [N][hd]
    const float *rscale;    // [N]
    const uint32_t *qfrag;  // [16][64] packed bf16 A fragments
    float *scores;          // [kH][N]
    int nsg;
    [[sycl::reqd_sub_group_size(kSG)]] void
    operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int sgid = (int)(it.get_global_linear_id() / kSG);
        const int lane = (int)sg.get_local_linear_id();
        const int apair = lane >> 3;
        const int abyte = (lane & 7) * 4;
        // Hoist the 16 A fragments (4 dwords/lane each = 64 regs).
        uintv4 afr[16];
        for (int c = 0; c < 16; ++c) {
            const uint8_t *ab =
                reinterpret_cast<const uint8_t *>(qfrag + (size_t)c * 64);
            for (int i = 0; i < 4; ++i)
                afr[c][i] = *reinterpret_cast<const uint32_t *>(
                    ab + (size_t)(2 * i + apair) * 32 + abyte);
        }
        const int ntiles = kN / kSG;
        for (int tile = sgid; tile < ntiles; tile += nsg) {
            const int row = tile * kSG + lane;   // this lane's token
            const uint8_t *kr = codes + (size_t)row * kHd;
            floatv8 d;
            for (int m = 0; m < 8; ++m) d[m] = 0.0f;
            for (int c = 0; c < 16; ++c) {
                // lane n's 16-dim E4M3 slice -> 8 packed bf16 pairs.
                uintv8 bfr;
                const uint8_t *ks = kr + c * 16;
                for (int p = 0; p < 8; ++p) {
                    const uint32_t c0 = ks[2 * p], c1 = ks[2 * p + 1];
                    // e4m3 -> fp32 -> bf16 (exact: e4m3 subset of bf16)
                    const uint32_t f0 = sycl::bit_cast<uint32_t>(
                        sycl::bit_cast<float>(((c0 & 0x7Fu) << 20) |
                                              ((c0 & 0x80u) << 24)) *
                        0x1p120f);
                    const uint32_t f1 = sycl::bit_cast<uint32_t>(
                        sycl::bit_cast<float>(((c1 & 0x7Fu) << 20) |
                                              ((c1 & 0x80u) << 24)) *
                        0x1p120f);
                    bfr[p] = (f0 >> 16) | (f1 & 0xFFFF0000u);
                }
                DPAS_BF_RC8(d, afr[c], bfr);
            }
            const float rs = rscale[row];
            for (int h = 0; h < kH; ++h)
                scores[(size_t)h * kN + row] = d[h] * rs;
        }
    }
};

// Arm C: FULL attention tile pipeline - the engine kernel's math end to end
// at probe level. One work-group (one subgroup) per segment of token tiles:
// per 16-token tile, S = QK^T via 16 chained dpas, online softmax over the
// token axis (log2e-folded exp2, FMHA trick), P staged bf16 through SLM,
// O += P x V via 16 dpas per tile with V staged bf16 in SLM. Emits the
// engine's shard-partial interface per (head, segment): O[256], running
// max, running denominator - merged on host (same log-sum-exp combine the
// engine's merge kernel runs).
struct walk_attn_dpas_k {
    const uint8_t *kcodes;   // [N][hd] e4m3
    const float *kscale;     // [N]
    const uint8_t *vcodes;   // [N][hd]
    const float *vscale;     // [N]
    const uint32_t *qfrag;   // [16][64] bf16 A fragments
    float *part;             // [nseg][kH][hd + 2] (O, max, den)
    int nseg;
    sycl::local_accessor<float, 1> o_slm;      // [kH * hd]
    sycl::local_accessor<uint16_t, 1> v_slm;   // [16 * hd] bf16
    sycl::local_accessor<uint16_t, 1> p_slm;   // [8 * 16] bf16
    [[sycl::reqd_sub_group_size(kSG)]] void
    operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int seg = (int)it.get_group_linear_id();
        const int lane = (int)sg.get_local_linear_id();
        const int apair = lane >> 3;
        const int abyte = (lane & 7) * 4;
        constexpr float kLog2e = 1.4426950408889634f;
        uintv4 afr[16];
        for (int c = 0; c < 16; ++c) {
            const uint8_t *ab =
                reinterpret_cast<const uint8_t *>(qfrag + (size_t)c * 64);
            for (int i = 0; i < 4; ++i)
                afr[c][i] = *reinterpret_cast<const uint32_t *>(
                    ab + (size_t)(2 * i + apair) * 32 + abyte);
        }
        for (int i = lane; i < kH * kHd; i += kSG) o_slm[i] = 0.0f;
        float run_max[kH], run_den[kH];
        for (int h = 0; h < kH; ++h) {
            run_max[h] = -3.0e38f;
            run_den[h] = 0.0f;
        }
        const int ntiles = kN / kSG;
        const int t0 = ntiles * seg / nseg, t1 = ntiles * (seg + 1) / nseg;
        auto e4m3f = [](uint32_t c) {
            return sycl::bit_cast<float>(((c & 0x7Fu) << 20) |
                                         ((c & 0x80u) << 24)) * 0x1p120f;
        };
        for (int tile = t0; tile < t1; ++tile) {
            const int row = tile * kSG + lane;
            // ---- S = Q x K^T (per-lane B rebuild, no SLM) ----
            const uint8_t *kr = kcodes + (size_t)row * kHd;
            floatv8 d;
            for (int m = 0; m < 8; ++m) d[m] = 0.0f;
            for (int c = 0; c < 16; ++c) {
                uintv8 bfr;
                const uint8_t *ks = kr + c * 16;
                for (int p = 0; p < 8; ++p) {
                    const uint32_t f0 =
                        sycl::bit_cast<uint32_t>(e4m3f(ks[2 * p]));
                    const uint32_t f1 =
                        sycl::bit_cast<uint32_t>(e4m3f(ks[2 * p + 1]));
                    bfr[p] = (f0 >> 16) | (f1 & 0xFFFF0000u);
                }
                DPAS_BF_RC8(d, afr[c], bfr);
            }
            const float krs = kscale[row];
            // ---- stage this lane's V row into SLM as bf16 ----
            const uint8_t *vr = vcodes + (size_t)row * kHd;
            const float vrs = vscale[row];
            for (int dd = 0; dd < kHd; ++dd) {
                const uint32_t f =
                    sycl::bit_cast<uint32_t>(e4m3f(vr[dd]) * vrs);
                v_slm[(size_t)lane * kHd + dd] = (uint16_t)(f >> 16);
            }
            // ---- online softmax over the 16 tokens of this tile ----
            float resc[kH];
            for (int h = 0; h < kH; ++h) {
                const float s = d[h] * krs * kLog2e;   // log2-domain
                const float tmax = sycl::reduce_over_group(
                    sg, s, sycl::maximum<float>());
                const float nmax = sycl::fmax(run_max[h], tmax);
                resc[h] = sycl::exp2(run_max[h] - nmax);
                const float p = sycl::exp2(s - nmax);
                const float psum = sycl::reduce_over_group(
                    sg, p, sycl::plus<float>());
                run_den[h] = run_den[h] * resc[h] + psum;
                run_max[h] = nmax;
                const uint32_t pf = sycl::bit_cast<uint32_t>(p);
                p_slm[(size_t)h * kSG + lane] = (uint16_t)(pf >> 16);
            }
            for (int h = kH; h < 8; ++h)
                p_slm[(size_t)h * kSG + lane] = 0;
            it.barrier(sycl::access::fence_space::local_space);
            // ---- O = O * resc + P x V ----
            uintv4 pfr;
            {
                const uint8_t *pb =
                    reinterpret_cast<const uint8_t *>(&p_slm[0]);
                for (int i = 0; i < 4; ++i)
                    pfr[i] = *reinterpret_cast<const uint32_t *>(
                        pb + (size_t)(2 * i + apair) * 32 + abyte);
            }
            for (int c = 0; c < 16; ++c) {
                uintv8 vfr;
                for (int p = 0; p < 8; ++p) {
                    const uint32_t lo = v_slm[(size_t)(2 * p) * kHd + c * 16 + lane];
                    const uint32_t hi = v_slm[(size_t)(2 * p + 1) * kHd + c * 16 + lane];
                    vfr[p] = lo | (hi << 16);
                }
                floatv8 od;
                for (int m = 0; m < 8; ++m) od[m] = 0.0f;
                DPAS_BF_RC8(od, pfr, vfr);
                for (int h = 0; h < kH; ++h) {
                    const size_t oi = (size_t)h * kHd + c * 16 + lane;
                    o_slm[oi] = o_slm[oi] * resc[h] + od[h];
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        // ---- emit shard partials ----
        for (int h = 0; h < kH; ++h) {
            float *dst = part + ((size_t)seg * kH + h) * (kHd + 2);
            for (int dd = lane; dd < kHd; dd += kSG)
                dst[dd] = o_slm[(size_t)h * kHd + dd];
            if (lane == 0) {
                dst[kHd] = run_max[h];
                dst[kHd + 1] = run_den[h];
            }
        }
    }
};

template <typename K>
static double time_iters(sycl::queue &q, const K &kern, size_t global,
                         size_t local, int iters) {
    for (int i = 0; i < 3; ++i)
        q.parallel_for(sycl::nd_range<1>(global, local), kern);
    q.wait_and_throw();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        q.parallel_for(sycl::nd_range<1>(global, local), kern);
    q.wait_and_throw();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / iters;
}

static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    // round-to-nearest-even
    const uint32_t r = b + 0x7FFFu + ((b >> 16) & 1u);
    return (uint16_t)(r >> 16);
}

int main(int argc, char **argv) {
    const int devidx = argc > 1 ? std::atoi(argv[1]) : 0;
    std::vector<sycl::device> gpus;
    for (auto &d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero)
            gpus.push_back(d);
    if (devidx >= (int)gpus.size()) {
        std::fprintf(stderr, "device %d not found\n", devidx);
        return 1;
    }
    sycl::queue q(gpus[devidx], sycl::property::queue::in_order{});
    std::printf("== dpas attention probe == device %d: %s\n", devidx,
                q.get_device().get_info<sycl::info::device::name>().c_str());

    // ---- synthetic K rows -> host E4M3 cache with per-row scales ----------
    std::vector<float> hK((size_t)kN * kHd);
    for (size_t i = 0; i < hK.size(); ++i) {
        const uint32_t r = mix((uint32_t)i * 2654435761u + 12345u);
        hK[i] = ((r & 0xFFFF) + ((r >> 16) & 0xFFFF)) / 65536.0f - 1.0f;
    }
    float table[256];
    for (int c = 0; c < 256; ++c) table[c] = e4m3_to_float((uint8_t)c);
    std::vector<uint8_t> hE((size_t)kN * kHd);
    std::vector<float> hEs(kN);
    for (int r = 0; r < kN; ++r) {
        float amax = 0.0f;
        for (int d = 0; d < kHd; ++d)
            amax = std::max(amax, std::fabs(hK[(size_t)r * kHd + d]));
        const float rs = amax > 0.0f ? amax / 448.0f : 1.0f;
        hEs[r] = rs;
        for (int d = 0; d < kHd; ++d) {
            const float at = std::fabs(hK[(size_t)r * kHd + d] / rs);
            const float *hi = std::upper_bound(table, table + 127, at);
            int best = (int)(hi - table);
            if (best >= 127) best = 126;
            else if (best > 0 && at - table[best - 1] <= table[best] - at)
                --best;
            hE[(size_t)r * kHd + d] =
                (uint8_t)(best | (hK[(size_t)r * kHd + d] < 0 ? 0x80 : 0));
        }
    }

    // ---- queries: 6 heads; scalar copy + packed bf16 A fragments ----------
    std::vector<float> hq((size_t)kH * kHd);
    for (size_t i = 0; i < hq.size(); ++i) {
        const uint32_t r = mix((uint32_t)i * 40503u + 99991u);
        hq[i] = ((r & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;
    }
    // A fragments: chunk c holds rows 0..7 (rows 6,7 zero), 32 B/row of
    // bf16 halves for dims [16c, 16c+16).
    std::vector<uint32_t> hfrag((size_t)16 * 64, 0u);
    for (int c = 0; c < 16; ++c) {
        auto *bytes = reinterpret_cast<uint8_t *>(hfrag.data() + (size_t)c * 64);
        for (int m = 0; m < 8; ++m)
            for (int hh = 0; hh < 16; ++hh) {
                const uint16_t bf = (m < kH)
                    ? f32_to_bf16(hq[(size_t)m * kHd + c * 16 + hh])
                    : (uint16_t)0;
                std::memcpy(bytes + (size_t)m * 32 + 2 * hh, &bf, 2);
            }
    }

    // ---- fp32 reference over decoded E4M3 values --------------------------
    std::vector<float> href((size_t)kH * kN);
    std::vector<float> hdec((size_t)kHd);
    for (int r = 0; r < kN; ++r) {
        for (int d = 0; d < kHd; ++d)
            hdec[d] = table[hE[(size_t)r * kHd + d] & 0x7F] *
                      ((hE[(size_t)r * kHd + d] & 0x80) ? -1.0f : 1.0f);
        for (int h = 0; h < kH; ++h) {
            double s = 0.0;
            for (int d = 0; d < kHd; ++d)
                s += (double)hq[(size_t)h * kHd + d] * hdec[d];
            href[(size_t)h * kN + r] = (float)(s * hEs[r]);
        }
    }

    uint8_t *dE = sycl::malloc_device<uint8_t>((size_t)kN * kHd, q);
    float *dEs = sycl::malloc_device<float>(kN, q);
    float *dq = sycl::malloc_device<float>((size_t)kH * kHd, q);
    uint32_t *dfrag = sycl::malloc_device<uint32_t>((size_t)16 * 64, q);
    float *dsc = sycl::malloc_device<float>((size_t)kH * kN, q);
    q.memcpy(dE, hE.data(), hE.size());
    q.memcpy(dEs, hEs.data(), hEs.size() * 4);
    q.memcpy(dq, hq.data(), hq.size() * 4);
    q.memcpy(dfrag, hfrag.data(), hfrag.size() * 4);
    q.wait_and_throw();

    constexpr int kNsg = 8192;
    const size_t glob = (size_t)kNsg * kSG;
    auto check = [&](const char *name, int heads, int hoff, double sec,
                     double passes) {
        std::vector<float> got((size_t)heads * kN);
        q.memcpy(got.data(), dsc, got.size() * 4).wait();
        double dot = 0, na = 0, nb = 0;
        for (int h = 0; h < heads; ++h)
            for (int r = 0; r < kN; ++r) {
                const double a = href[(size_t)(h + hoff) * kN + r];
                const double b = got[(size_t)h * kN + r];
                dot += a * b; na += a * a; nb += b * b;
            }
        const double c = dot / (std::sqrt(na) * std::sqrt(nb));
        std::printf("  %-12s %8.3f ms/6-head-pass %8.1f GB/s  cos=%.6f\n",
                    name, sec * 1e3,
                    passes * kN * (double)kHd / sec / 1e9, c);
        return c;
    };

    // Arm A: two 3-head passes over the cache = one 6-head service.
    const walk_scalar3_k wa0{dE, dEs, dq, dsc, kNsg};
    const walk_scalar3_k wa1{dE, dEs, dq + (size_t)3 * kHd, dsc, kNsg};
    auto runA = [&]() {
        q.parallel_for(sycl::nd_range<1>(glob, 256), wa0);
        q.parallel_for(sycl::nd_range<1>(glob, 256), wa1);
    };
    runA(); q.wait_and_throw();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) runA();
    q.wait_and_throw();
    const auto t1 = std::chrono::steady_clock::now();
    const double sa = std::chrono::duration<double>(t1 - t0).count() / 20;
    // accuracy of the second pass (heads 3..5) is what's resident in dsc
    const double ca = check("scalar G3x2", 3, 3, sa, 2.0);

    // Arm B: one DPAS pass serves all six heads.
    const walk_dpas6_k wb{dE, dEs, dfrag, dsc, kNsg};
    const double sb = time_iters(q, wb, glob, 256, 20);
    const double cb = check("dpas M8", kH, 0, sb, 1.0);

    std::printf("  speedup %.2fx (bytes ratio 2.00x); cos delta %+.6f\n",
                sa / sb, cb - ca);
    std::printf("  verdict: %s\n",
                (cb >= 0.9997 && sa / sb >= 1.3) ? "FUND" : "KILL");

    // ---- Arm C: full attention pipeline, engine-shaped segment partials --
    // V cache: fresh synthetic rows, same E4M3 encode as K.
    std::vector<float> hV((size_t)kN * kHd);
    for (size_t i = 0; i < hV.size(); ++i) {
        const uint32_t r = mix((uint32_t)i * 2246822519u + 777u);
        hV[i] = ((r & 0xFFFF) + ((r >> 16) & 0xFFFF)) / 65536.0f - 1.0f;
    }
    std::vector<uint8_t> hEv((size_t)kN * kHd);
    std::vector<float> hEvs(kN);
    for (int r = 0; r < kN; ++r) {
        float amax = 0.0f;
        for (int d = 0; d < kHd; ++d)
            amax = std::max(amax, std::fabs(hV[(size_t)r * kHd + d]));
        const float rs = amax > 0.0f ? amax / 448.0f : 1.0f;
        hEvs[r] = rs;
        for (int d = 0; d < kHd; ++d) {
            const float at = std::fabs(hV[(size_t)r * kHd + d] / rs);
            const float *hi = std::upper_bound(table, table + 127, at);
            int best = (int)(hi - table);
            if (best >= 127) best = 126;
            else if (best > 0 && at - table[best - 1] <= table[best] - at)
                --best;
            hEv[(size_t)r * kHd + d] =
                (uint8_t)(best | (hV[(size_t)r * kHd + d] < 0 ? 0x80 : 0));
        }
    }
    // Host reference attention over the decoded caches (fp64 softmax).
    std::vector<float> hOref((size_t)kH * kHd);
    {
        std::vector<double> O((size_t)kH * kHd, 0.0);
        std::vector<double> den(kH, 0.0);
        std::vector<double> mx(kH, -1.0e300);
        for (int h = 0; h < kH; ++h)
            for (int r = 0; r < kN; ++r)
                mx[h] = std::max(mx[h], (double)href[(size_t)h * kN + r]);
        for (int r = 0; r < kN; ++r) {
            float vdec[kHd];
            for (int d = 0; d < kHd; ++d)
                vdec[d] = table[hEv[(size_t)r * kHd + d] & 0x7F] *
                          ((hEv[(size_t)r * kHd + d] & 0x80) ? -1.0f : 1.0f) *
                          hEvs[r];
            for (int h = 0; h < kH; ++h) {
                const double p =
                    std::exp((double)href[(size_t)h * kN + r] - mx[h]);
                den[h] += p;
                for (int d = 0; d < kHd; ++d)
                    O[(size_t)h * kHd + d] += p * vdec[d];
            }
        }
        for (int h = 0; h < kH; ++h)
            for (int d = 0; d < kHd; ++d)
                hOref[(size_t)h * kHd + d] =
                    (float)(O[(size_t)h * kHd + d] / den[h]);
    }
    uint8_t *dEv = sycl::malloc_device<uint8_t>((size_t)kN * kHd, q);
    float *dEvs = sycl::malloc_device<float>(kN, q);
    constexpr int kSeg = 128;
    float *dpart = sycl::malloc_device<float>((size_t)kSeg * kH * (kHd + 2), q);
    q.memcpy(dEv, hEv.data(), hEv.size());
    q.memcpy(dEvs, hEvs.data(), hEvs.size() * 4);
    q.wait_and_throw();
    auto launchC = [&]() {
        q.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> o_slm(
                sycl::range<1>((size_t)kH * kHd), cgh);
            sycl::local_accessor<uint16_t, 1> v_slm(
                sycl::range<1>((size_t)kSG * kHd), cgh);
            sycl::local_accessor<uint16_t, 1> p_slm(sycl::range<1>(8 * kSG),
                                                    cgh);
            cgh.parallel_for(
                sycl::nd_range<1>((size_t)kSeg * kSG, kSG),
                walk_attn_dpas_k{dE, dEs, dEv, dEvs, dfrag, dpart, kSeg,
                                 o_slm, v_slm, p_slm});
        });
    };
    launchC();
    q.wait_and_throw();
    const auto tc0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) launchC();
    q.wait_and_throw();
    const auto tc1 = std::chrono::steady_clock::now();
    const double sc =
        std::chrono::duration<double>(tc1 - tc0).count() / 20;
    // Host merge of segment partials (the engine merge kernel's math).
    std::vector<float> hpart((size_t)kSeg * kH * (kHd + 2));
    q.memcpy(hpart.data(), dpart, hpart.size() * 4).wait();
    double dotc = 0, nac = 0, nbc = 0;
    constexpr float kLog2eH = 1.4426950408889634f;
    for (int h = 0; h < kH; ++h) {
        double gmax = -1.0e300;
        for (int s = 0; s < kSeg; ++s)
            gmax = std::max(gmax, (double)hpart[((size_t)s * kH + h) *
                                                (kHd + 2) + kHd]);
        double gden = 0.0;
        std::vector<double> O(kHd, 0.0);
        for (int s = 0; s < kSeg; ++s) {
            const float *p = hpart.data() + ((size_t)s * kH + h) * (kHd + 2);
            const double resc = std::exp2((double)p[kHd] - gmax);
            gden += (double)p[kHd + 1] * resc;
            for (int d = 0; d < kHd; ++d) O[d] += (double)p[d] * resc;
        }
        for (int d = 0; d < kHd; ++d) {
            const double a = hOref[(size_t)h * kHd + d];
            const double b = O[d] / gden;
            dotc += a * b; nac += a * a; nbc += b * b;
        }
    }
    (void)kLog2eH;
    const double cc = dotc / (std::sqrt(nac) * std::sqrt(nbc));
    std::printf("  %-12s %8.3f ms/6-head-pass %8.1f GB/s  cos=%.6f "
                "(full attention, K+V)\n",
                "attn dpas", sc * 1e3,
                2.0 * kN * (double)kHd / sc / 1e9, cc);
    std::printf("  attention verdict: %s\n",
                cc >= 0.999 ? "FUND (formulation exact)" : "KILL/debug");
    sycl::free(dEv, q); sycl::free(dEvs, q); sycl::free(dpart, q);

    sycl::free(dE, q); sycl::free(dEs, q); sycl::free(dq, q);
    sycl::free(dfrag, q); sycl::free(dsc, q);
    return 0;
}
