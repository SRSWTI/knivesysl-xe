#include "prefill_xmx.hpp"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <stdexcept>
#include <string>

// SIMD16 RC8: C[8,16] += A[8,16] * B[16,16]. A is row-major
// packed BF16 pairs; B is VNNI-packed along K. These are the same operand
// aliases as the existing decode DPAS kernel and dpas_attn_probe.cpp.
#ifdef __SYCL_DEVICE_ONLY__
#define PREFILL_DPAS(dd, aa, bb)                                              \
    asm("{\n"                                                               \
        ".decl DST v_type=G type=F num_elts=128 alias=<%0,0>\n"               \
        ".decl SRC1 v_type=G type=UD num_elts=128 alias=<%2,0>\n"             \
        ".decl SRC2 v_type=G type=UD num_elts=64 alias=<%1,0>\n"              \
        "dpas.bf.bf.8.8 (M1, 16) DST.0 DST.0 SRC1.0 SRC2(0,0)\n"             \
        "}\n" : "+rw"(dd) : "rw"(aa), "rw"(bb))
#else
#define PREFILL_DPAS(dd, aa, bb) do { (void)(aa); (void)(bb); } while (0)
#endif

namespace {

// Compile-time subgroup properties belong to the kernel, not launch_config.
template <typename Body> struct PrefillKernel {
    Body body;
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::sub_group_size<16>};
    }
    void operator()(sycl::nd_item<1> item) const { body(item); }
};
template <typename Body> PrefillKernel(Body) -> PrefillKernel<Body>;
inline float bf16_float(uint16_t b) {
    return sycl::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}

inline uint16_t bf16_rne(float f) {
    const uint32_t u = sycl::bit_cast<uint32_t>(f);
    return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

inline uint16_t e4m3_bf16(uint8_t code) {
    // Every finite E4M3 value, including subnormals, is exact in BF16.
    const uint32_t bits = (static_cast<uint32_t>(code & 0x80u) << 24) |
                          (static_cast<uint32_t>(code & 0x7fu) << 20);
    return static_cast<uint16_t>(sycl::bit_cast<uint32_t>(
        sycl::bit_cast<float>(bits) * 0x1p120f) >> 16);
}

bool xe2_device() {
    const auto dev = tq_q().get_device();
    if (!dev.is_gpu() || dev.get_info<sycl::info::device::vendor_id>() != 0x8086)
        return false;
    // The inline ISA uses Xe2's SIMD16 DPAS layout; do not send it to Xe1.
    const std::string name = dev.get_info<sycl::info::device::name>();
    return name.find("B70") != std::string::npos ||
           name.find("B60") != std::string::npos ||
           name.find("B580") != std::string::npos ||
           name.find("B570") != std::string::npos ||
           name.find("BMG") != std::string::npos ||
           name.find("Battlemage") != std::string::npos;
}

template <bool Paged>
void prefill_xmx(const tq_prefill_attn_request_t *requests, int n,
                 const uint16_t *qnorm, int nh, int nkv, float eps,
                 float theta, int rotary_dim) {
    constexpr int HD = 256;
    constexpr int QR = 8;
    constexpr int KT = 16;
    constexpr int NSG = 8;
    constexpr int WG = NSG * KT;
    constexpr int QT = NSG * QR;
    // The descriptors and grid boundaries are copied into the kernel capture;
    // caller stack lifetime ends safely after submission, with no USM scratch.
    struct CapturedBatch {
        tq_prefill_attn_request_t requests[8];
        size_t group_ends[8];
    } batch{};
    size_t groups = 0;
    for (int i = 0; i < n; ++i) {
        batch.requests[i] = requests[i];
        groups += static_cast<size_t>(nh) * ((requests[i].T + QT - 1) / QT);
        batch.group_ends[i] = groups;
    }
    const int gqa = nh / nkv;
    const int rotary_half = rotary_dim / 2;
    tq_q().submit([&](sycl::handler &cgh) {
        // 47104 bytes/WG for eight query subgroups: raw FP32 Q low bits,
        // one shared V tile, and three-part probabilities per subgroup.
        // Qhi is the truncated FP32 high half retained in GRFs. Its low
        // half in SLM reconstructs Q losslessly before residual decomposition.
        sycl::local_accessor<uint16_t, 1> qlo(NSG * QR * HD, cgh);
        sycl::local_accessor<uint16_t, 1> stage(KT * HD, cgh);
        sycl::local_accessor<uint16_t, 1> prob(NSG * QR * KT, cgh);
        sycl::local_accessor<uint32_t, 1> plo(NSG * QR * KT, cgh);
        const sycl::ext::oneapi::experimental::properties grf_props{
            sycl::ext::intel::experimental::grf_size<256>};
        sycl::ext::oneapi::experimental::nd_launch(
            cgh, sycl::ext::oneapi::experimental::launch_config{
                sycl::nd_range<1>(groups * WG, WG),
                grf_props},
            PrefillKernel{[=](sycl::nd_item<1> item) {
                const size_t global_wg = item.get_group_linear_id();
                auto request = batch.requests[0];
                size_t group_begin = 0;
                // Constant indices avoid materializing an addressable private
                // descriptor array in each work-item merely for selection.
                #pragma unroll
                for (int i = 1; i < 8; ++i) {
                    if (i < n && global_wg >= batch.group_ends[i - 1]) {
                        request = batch.requests[i];
                        group_begin = batch.group_ends[i - 1];
                    }
                }
                float *out = request.out;
                const float *qg = request.qg;
                const uint8_t *kc = request.kc, *vc = request.vc;
                const uint16_t *ks = request.ks, *vs = request.vs;
                const int pos0 = request.pos0, T = request.T;
                const tq_kv_layout_t layout = request.layout;
                const int tiles = (T + QT - 1) / QT;
                const size_t wg = global_wg - group_begin;
                const int head = static_cast<int>(wg / tiles);
                const int tile_q0 = static_cast<int>(wg % tiles) * QT;
                const sycl::sub_group sg = item.get_sub_group();
                const int sgi = static_cast<int>(sg.get_group_linear_id());
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int q0 = tile_q0 + sgi * QR;
                const int kv = head / gqa;
                const int apair = lane >> 3;
                const int adim = (lane & 7) * 2;
                const int qs = sgi * QR * HD;
                const int ps = sgi * QR * KT;
                using U4 = uint32_t __attribute__((ext_vector_type(4)));
                using U8 = uint32_t __attribute__((ext_vector_type(8)));
                using F8 = float __attribute__((ext_vector_type(8)));
                U4 afr[HD / KT];

                // Recompute the once-per-query prologue to reuse the same
                // slab for Qhi then the raw FP32 tail, avoiding another Qhi
                // staging slab or another full Q matrix live in registers.
                for (int phase = 0; phase < 2; ++phase) {
                    for (int row = 0; row < QR; ++row) {
                        const int t = q0 + row;
                        const bool live = t < T;
                        const size_t base = (static_cast<size_t>(live ? t : 0) * nh + head) * (2 * HD);
                        float ss = 0.0f;
                        for (int i = 0; i < HD / KT; ++i) {
                            const float v = qg[base + lane * (HD / KT) + i];
                            ss += v * v;
                        }
                        ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                        const float inv = sycl::rsqrt(ss / static_cast<float>(HD) + eps);
                        for (int i = 0; i < HD / KT; ++i) {
                            const int d = lane * (HD / KT) + i;
                            float q = qg[base + d] * inv * (1.0f + bf16_float(qnorm[d]));
                            if (d < rotary_dim) {
                                const int pair_d = d < rotary_half ? d + rotary_half : d - rotary_half;
                                const int ri = d < rotary_half ? d : d - rotary_half;
                                const float freq = sycl::pow(theta,
                                    -(2.0f * static_cast<float>(ri) / static_cast<float>(rotary_dim)));
                                const float angle = static_cast<float>(pos0 + t) * freq;
                                const float pair = qg[base + pair_d] * inv *
                                                   (1.0f + bf16_float(qnorm[pair_d]));
                                q = q * sycl::cos(angle) +
                                    (d < rotary_half ? -pair : pair) * sycl::sin(angle);
                            }
                            const int qi = qs + (d / KT) * QR * KT + row * KT + d % KT;
                            const uint32_t bits = live ? sycl::bit_cast<uint32_t>(q) : 0;
                            qlo[qi] = static_cast<uint16_t>(phase == 0 ? bits >> 16 : bits);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    if (phase == 0) {
                        #pragma unroll
                        for (int c = 0; c < HD / KT; ++c) {
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int a = qs + c * QR * KT + (2 * i + apair) * KT + adim;
                                afr[c][i] = static_cast<uint32_t>(qlo[a]) |
                                            (static_cast<uint32_t>(qlo[a + 1]) << 16);
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                // Output columns are lane-owned; no output SLM traffic.
                F8 accum[HD / KT];
                #pragma unroll
                for (int c = 0; c < HD / KT; ++c) {
                    #pragma unroll
                    for (int r = 0; r < QR; ++r) accum[c][r] = 0.0f;
                }
                float maxima[QR], den[QR];
                #pragma unroll
                for (int r = 0; r < QR; ++r) {
                    maxima[r] = -3.402823466e38f;
                    den[r] = 0.0f;
                }
                // Work-group uniform, including fully dead subgroups in a
                // ragged final 64-query tile. All execute every V barrier.
                const int qend = tile_q0 + QT < T ? tile_q0 + QT : T;
                const int total = pos0 + qend;
                for (int k0 = 0; k0 < total; k0 += KT) {
                    const int token = k0 + lane;
                    const bool valid = token < total;
                    const int logical = valid ? token : k0;
                    size_t physical = static_cast<size_t>(logical);
                    if constexpr (Paged) {
                        if (layout.block_table)
                            physical = (static_cast<size_t>(
                                layout.block_table[logical >> layout.page_log]) <<
                                layout.page_log) + (logical & layout.page_mask);
                    }
                    const size_t kr = physical * nkv + kv;
                    const size_t kb = kr * HD;
                    const float kscale = static_cast<float>(sycl::bit_cast<sycl::half>(ks[kr]));
                    const float vscale = static_cast<float>(sycl::bit_cast<sycl::half>(vs[kr]));
                    F8 score;
                    #pragma unroll
                    for (int r = 0; r < QR; ++r) score[r] = 0.0f;
                    #pragma unroll
                    for (int c = 0; c < HD / KT; ++c) {
                        U8 b;
                        #pragma unroll
                        for (int p = 0; p < 8; ++p) {
                            b[p] = static_cast<uint32_t>(e4m3_bf16(kc[kb + c * KT + 2 * p])) |
                                   (static_cast<uint32_t>(e4m3_bf16(kc[kb + c * KT + 2 * p + 1])) << 16);
                        }
                        PREFILL_DPAS(score, afr[c], b);
                        U4 amid, alo;
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int a = qs + c * QR * KT + (2 * i + apair) * KT + adim;
                            const uint32_t hi0 = (afr[c][i] & 0xffffu) << 16;
                            const uint32_t hi1 = afr[c][i] & 0xffff0000u;
                            const float r0 = sycl::bit_cast<float>(hi0 | qlo[a]) -
                                             sycl::bit_cast<float>(hi0);
                            const float r1 = sycl::bit_cast<float>(hi1 | qlo[a + 1]) -
                                             sycl::bit_cast<float>(hi1);
                            const uint16_t m0 = bf16_rne(r0), m1 = bf16_rne(r1);
                            const uint16_t l0 = bf16_rne(r0 - bf16_float(m0));
                            const uint16_t l1 = bf16_rne(r1 - bf16_float(m1));
                            amid[i] = static_cast<uint32_t>(m0) | (static_cast<uint32_t>(m1) << 16);
                            alo[i] = static_cast<uint32_t>(l0) | (static_cast<uint32_t>(l1) << 16);
                        }
                        PREFILL_DPAS(score, amid, b);
                        PREFILL_DPAS(score, alo, b);
                    }
                    // Each subgroup contributes 1/8 of V's dimensions. One
                    // cache read/dequantization serves all64 query rows.
                    for (int d = sgi; d < HD; d += NSG)
                        stage[lane * HD + d] = valid ? e4m3_bf16(vc[kb + d]) : 0;
                    float rescale[QR];
                    #pragma unroll
                    for (int r = 0; r < QR; ++r) {
                        const bool causal = valid && q0 + r < T && token <= pos0 + q0 + r;
                        const float s = causal ? score[r] * kscale * 0.0625f : -3.402823466e38f;
                        const float m = sycl::reduce_over_group(sg, s, sycl::maximum<float>());
                        const float next = sycl::fmax(maxima[r], m);
                        rescale[r] = sycl::exp(maxima[r] - next);
                        const float p = causal ? sycl::exp(s - next) : 0.0f;
                        den[r] = den[r] * rescale[r] +
                                 sycl::reduce_over_group(sg, p, sycl::plus<float>());
                        maxima[r] = next;
                        const float pv = p * vscale;
                        const uint16_t hi = bf16_rne(pv);
                        prob[ps + r * KT + lane] = hi;
                        const float residual = pv - bf16_float(hi);
                        const uint16_t mid = bf16_rne(residual);
                        const uint16_t lo = bf16_rne(residual - bf16_float(mid));
                        plo[ps + r * KT + lane] = static_cast<uint32_t>(mid) |
                            (static_cast<uint32_t>(lo) << 16);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    U4 pfr, pmfr, plfr;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int a = ps + (2 * i + apair) * KT + adim;
                        pfr[i] = static_cast<uint32_t>(prob[a]) |
                                 (static_cast<uint32_t>(prob[a + 1]) << 16);
                        const uint32_t r0 = plo[a], r1 = plo[a + 1];
                        pmfr[i] = (r0 & 0xffffu) | (r1 << 16);
                        plfr[i] = (r0 >> 16) | (r1 & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int c = 0; c < HD / KT; ++c) {
                        U8 b;
                        #pragma unroll
                        for (int p = 0; p < 8; ++p) {
                            b[p] = static_cast<uint32_t>(stage[(2 * p) * HD + c * KT + lane]) |
                                   (static_cast<uint32_t>(stage[(2 * p + 1) * HD + c * KT + lane]) << 16);
                        }
                        F8 o;
                        #pragma unroll
                        for (int r = 0; r < QR; ++r) o[r] = 0.0f;
                        PREFILL_DPAS(o, pfr, b);
                        PREFILL_DPAS(o, pmfr, b);
                        PREFILL_DPAS(o, plfr, b);
                        #pragma unroll
                        for (int r = 0; r < QR; ++r)
                            accum[c][r] = accum[c][r] * rescale[r] + o[r];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                #pragma unroll
                for (int r = 0; r < QR; ++r) {
                    const int t = q0 + r;
                    if (t >= T) break;
                    const size_t ob = (static_cast<size_t>(t) * nh + head) * HD;
                    const size_t qb = (static_cast<size_t>(t) * nh + head) * (2 * HD);
                    #pragma unroll
                    for (int c = 0; c < HD / KT; ++c) {
                        const int d = c * KT + lane;
                        out[ob + d] = (accum[c][r] / den[r]) *
                            (1.0f / (1.0f + sycl::exp(-qg[qb + HD + d])));
                    }
                }
            }});
    });
}
} // namespace

bool x_prefill_attn_xmx(float *d_out, const float *d_qg_proj,
                        const uint16_t *d_q_norm, const uint8_t *d_k_cache,
                        const uint8_t *d_v_cache, const uint16_t *d_k_scale,
                        const uint16_t *d_v_scale, int pos0, int T, int nh,
                        int nkv, int hd, float eps, float rope_theta,
                        float partial_rotary_factor, tq_kv_layout_t layout) {
    const tq_prefill_attn_request_t request{
        d_out, d_qg_proj, d_k_cache, d_v_cache, d_k_scale, d_v_scale,
        pos0, T, layout};
    return x_prefill_attn_xmx_packed(&request, 1, d_q_norm, nh, nkv, hd,
                                    eps, rope_theta, partial_rotary_factor);
}

bool x_prefill_attn_xmx_packed(const tq_prefill_attn_request_t *requests,
                               int n, const uint16_t *qnorm,
                               int nh, int nkv, int hd, float eps,
                               float rope_theta, float partial_rotary_factor) {
    if (!requests || n < 1 || n > 8)
        throw std::invalid_argument("packed attention requires 1..8 request descriptors");
    const char *mode = std::getenv("TQ_XPU_PREFILL_XMX");
    if (mode && std::strcmp(mode, "0") == 0) return false;
    const bool required = mode && std::strcmp(mode, "1") == 0;
    if (mode && !required && std::strcmp(mode, "auto") != 0)
        throw std::invalid_argument("TQ_XPU_PREFILL_XMX must be 0, 1, or auto");
    static const bool device_supported = xe2_device();
    const int rotary_dim = static_cast<int>(static_cast<float>(hd) * partial_rotary_factor);
    bool supported = device_supported && hd == 256 && nh > 0 && nkv > 0 &&
        nh % nkv == 0 && rotary_dim >= 0 && rotary_dim <= hd && rotary_dim % 2 == 0;
    bool any_paged = false;
    bool large_enough = true;
    size_t total_rows = 0;
    for (int i = 0; i < n; ++i) {
        const auto &r = requests[i];
        supported = supported && r.T > 0 && r.pos0 >= 0 &&
            (!r.layout.block_table || ((r.layout.page_log == 7 || r.layout.page_log == 8) &&
                r.layout.page_mask == (1 << r.layout.page_log) - 1));
        any_paged = any_paged || r.layout.block_table;
        large_enough = large_enough && r.T >= (n == 1 ? 128 : 64) &&
                       static_cast<int64_t>(r.pos0) + r.T >= 256;
        total_rows += static_cast<size_t>(r.T);
    }
    if (!supported) {
        if (required)
            throw std::runtime_error("required XMX prefill needs Intel Xe2 BMG, HD256, valid GQA and flat/page128/page256 KV");
        return false;
    }
    // Tiny tails cannot populate this64-query workgroup well. Required mode
    // deliberately bypasses the measured cost guard, never silently falling back.
    if (!required && (!large_enough || total_rows < 128)) return false;
    if (any_paged)
        prefill_xmx<true>(requests, n, qnorm, nh, nkv, eps, rope_theta, rotary_dim);
    else
        prefill_xmx<false>(requests, n, qnorm, nh, nkv, eps, rope_theta, rotary_dim);
    return true;
}
