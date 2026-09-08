// w4a4_gemm_probe.cpp — level-up-xpu queue item 4: fund or kill thesis 5
// (W4A4 prefill: s4 activations x s4 weights at K=64, 733-TOPS tier) before
// touching the engine.
//
// Question: with the SAME task grid, tiling, and epilogue structure as the
// engine's x_gemm_w4a8 (kernels_dpas.cpp), how much does switching the k-tile
// from dpas.s4.s8.8.8 (K=32, 366 TOPS peak) to dpas.s4.s4.8.8 (K=64, 733
// TOPS peak) actually buy? The shipped RC8 GEMM measures 42.4 TOPS wall and
// is epilogue/ALU-bound, not DPAS-bound — W4A4 halves BOTH the DPAS count
// and the per-tile epilogue evaluations per K-span, so the model says ~2x.
// Kill criterion (level-up doc §5 item 4): < 1.5x over the W4A8 arm.
//
// Synthetic all-ones operands (dpas_probe.cpp convention): every s4/s8 code
// is 1, every scale is fp16 1.0 with zero-point nibble 0, so each output must
// equal K exactly — validates execution, K depth, layout, and epilogue in one
// number. Weight/activation buffers are full-size (44.6 MB codes) so DRAM
// streaming behavior matches the real kernel.
//
// Arm A layouts are the on-silicon contracts from dpas_layout_probe.cpp /
// dpas_w8_layout_probe.cpp. Arm B reuses them at K=64: A is still 8 rows x
// 32 bytes (64 s4 nibbles instead of 32 s8 bytes — identical footprint), B
// doubles to 512 B per fragment (nibble k at dword (k/8)*16+n, k < 64).
//
// Build: icpx -fsycl -O2 -o w4a4_gemm_probe w4a4_gemm_probe.cpp
// Run:   ./w4a4_gemm_probe [device_index]   (card 1 per device rules)

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using uintv4 = uint32_t __attribute__((ext_vector_type(4)));
using uintv8 = uint32_t __attribute__((ext_vector_type(8)));
using intv8 = int32_t __attribute__((ext_vector_type(8)));
using floatv8 = float __attribute__((ext_vector_type(8)));

static constexpr int kSG = 16;
static constexpr int kSGWG = 16;  // subgroups per work-group (engine value)
static constexpr int kM = 17408;  // mlp_gate shape, the prefill wall
static constexpr int kK = 5120;

#ifdef __SYCL_DEVICE_ONLY__
#define DPAS_S4S8_RC8(dd, aa, bb)                                              \
    asm("{\n"                                                                  \
        ".decl DST     v_type=G type=D num_elts=128 alias=<%0,0>\n"            \
        ".decl SRC1_UD v_type=G type=UD num_elts=64 alias=<%2,0>\n"            \
        ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"            \
        "dpas.s4.s8.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"         \
        "}\n"                                                                  \
        : "+rw"(dd) : "rw"(aa), "rw"(bb))
#define DPAS_S4S4_RC8(dd, aa, bb)                                              \
    asm("{\n"                                                                  \
        ".decl DST     v_type=G type=D num_elts=128 alias=<%0,0>\n"            \
        ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"           \
        ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"            \
        "dpas.s4.s4.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"         \
        "}\n"                                                                  \
        : "+rw"(dd) : "rw"(aa), "rw"(bb))
#else
#define DPAS_S4S8_RC8(dd, aa, bb) do { (void)(aa); (void)(bb); } while (0)
#define DPAS_S4S4_RC8(dd, aa, bb) do { (void)(aa); (void)(bb); } while (0)
#endif

// ---- arm A: the engine's x_gemm_w4a8 shape, synthetic buffers -------------
struct gemm_w4a8_k {
    const uint8_t *codes;    // rg-major: rg * Kt * 256
    const uint16_t *scales;  // rg * Kt * 16, fp16|zp-nibble packed
    const int8_t *aq;        // kt-major: (kt * T + t) * 32
    const float *as;         // kt * T + t
    const int32_t *asum;     // kt * T + t
    float *y;                // [T][M]
    int T, tasks, tgroups, Kt;
    [[sycl::reqd_sub_group_size(kSG)]] void
    operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int task =
            (int)it.get_group_linear_id() * kSGWG + (int)sg.get_group_linear_id();
        if (task >= tasks) return;
        const int rg = task / tgroups;
        const int tg = task - rg * tgroups;
        const int lane = (int)sg.get_local_linear_id();
        const int apair = lane >> 3;
        const int abyte = (lane & 7) * 4;
        const uint8_t *bbase = codes + (size_t)rg * Kt * 256u;
        const uint16_t *sbase = scales + (size_t)rg * Kt * 16u;
        float acc[8];
        for (int m = 0; m < 8; ++m) acc[m] = 0.0f;
        for (int kt = 0; kt < Kt; ++kt) {
            const uintv4 bfrag = *reinterpret_cast<const uintv4 *>(
                bbase + (size_t)kt * 256u + (size_t)lane * 16u);
            const int8_t *ab = aq + ((size_t)kt * T + tg * 8) * 32;
            uintv4 afrag;
            for (int i = 0; i < 4; ++i)
                afrag[i] = *reinterpret_cast<const uint32_t *>(
                    ab + (size_t)(2 * i + apair) * 32 + abyte);
            intv8 d;
            for (int m = 0; m < 8; ++m) d[m] = 0;
            DPAS_S4S8_RC8(d, afrag, bfrag);
            const uint16_t packed = sbase[(size_t)kt * 16u + lane];
            int zp = (int)(packed & 0xFu);
            if (zp >= 8) zp -= 16;
            const float ws = (float)sycl::bit_cast<sycl::half>(
                (uint16_t)(packed & 0xFFF0u));
            const floatv8 asv =
                *reinterpret_cast<const floatv8 *>(as + (size_t)kt * T + tg * 8);
            const intv8 sumv = *reinterpret_cast<const intv8 *>(
                asum + (size_t)kt * T + tg * 8);
            for (int m = 0; m < 8; ++m)
                acc[m] += (float)(d[m] - zp * sumv[m]) * (ws * asv[m]);
        }
        const int M = kM;
        for (int m = 0; m < 8; ++m)
            y[(size_t)(tg * 8 + m) * M + rg * 16 + lane] = acc[m];
    }
};

// ---- arm B: W4A4 — s4 x s4 at K=64, half the tiles, half the epilogue -----
struct gemm_w4a4_k {
    const uint8_t *codes;    // rg-major: rg * Kt64 * 512
    const uint16_t *scales;  // rg * Kt64 * 16
    const uint8_t *aq;       // kt64-major: (kt64 * T + t) * 32 (64 nibbles)
    const float *as;         // kt64 * T + t
    const int32_t *asum;     // kt64 * T + t
    float *y;
    int T, tasks, tgroups, Kt64;
    [[sycl::reqd_sub_group_size(kSG)]] void
    operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int task =
            (int)it.get_group_linear_id() * kSGWG + (int)sg.get_group_linear_id();
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
            const uintv8 bfrag = *reinterpret_cast<const uintv8 *>(
                bbase + (size_t)kt * 512u + (size_t)lane * 32u);
            const uint8_t *ab = aq + ((size_t)kt * T + tg * 8) * 32;
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
            const floatv8 asv =
                *reinterpret_cast<const floatv8 *>(as + (size_t)kt * T + tg * 8);
            const intv8 sumv = *reinterpret_cast<const intv8 *>(
                asum + (size_t)kt * T + tg * 8);
            for (int m = 0; m < 8; ++m)
                acc[m] += (float)(d[m] - zp * sumv[m]) * (ws * asv[m]);
        }
        const int M = kM;
        for (int m = 0; m < 8; ++m)
            y[(size_t)(tg * 8 + m) * M + rg * 16 + lane] = acc[m];
    }
};

template <typename K>
static double time_kernel(sycl::queue &q, const K &kern, int tasks, int iters) {
    const size_t wgs = ((size_t)tasks + kSGWG - 1) / kSGWG;
    const sycl::nd_range<1> grid(wgs * kSG * kSGWG, kSG * kSGWG);
    for (int i = 0; i < 3; ++i) q.parallel_for(grid, kern);
    q.wait_and_throw();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) q.parallel_for(grid, kern);
    q.wait_and_throw();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / iters;
}

static int check_output(sycl::queue &q, const float *d_y, int T, const char *arm) {
    std::vector<float> h(16);
    // spot-check 16 scattered outputs; all must equal K exactly
    const size_t total = (size_t)T * kM;
    int bad = 0;
    for (int i = 0; i < 16; ++i) {
        const size_t idx = (total / 17) * i + i;
        q.memcpy(h.data() + i, d_y + idx, sizeof(float)).wait();
        if (h[i] != (float)kK) ++bad;
    }
    if (bad)
        std::printf("  %s CHECK FAIL: %d/16 outputs != %d (got e.g. %.1f)\n", arm,
                    bad, kK, h[0]);
    return bad;
}

int main(int argc, char **argv) {
    const int devidx = argc > 1 ? std::atoi(argv[1]) : 0;
    const char *armsel = argc > 2 ? argv[2] : "ab";  // "a", "b", or "ab"
    std::vector<sycl::device> gpus;
    for (auto &d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero)
            gpus.push_back(d);
    if (devidx >= (int)gpus.size()) {
        std::fprintf(stderr, "device %d not found (%zu gpus)\n", devidx, gpus.size());
        return 1;
    }
    sycl::queue q(gpus[devidx], sycl::property::queue::in_order{});
    std::printf("== w4a4 gemm probe == device %d: %s\n", devidx,
                q.get_device().get_info<sycl::info::device::name>().c_str());

    const int Kt = kK / 32, Kt64 = kK / 64, rgroups = kM / 16;
    const size_t code_bytes = (size_t)kM * kK / 2;  // identical both arms
    uint8_t *codesA = sycl::malloc_device<uint8_t>(code_bytes, q);
    uint8_t *codesB = sycl::malloc_device<uint8_t>(code_bytes, q);
    uint16_t *scalesA = sycl::malloc_device<uint16_t>((size_t)rgroups * Kt * 16, q);
    uint16_t *scalesB = sycl::malloc_device<uint16_t>((size_t)rgroups * Kt64 * 16, q);
    q.memset(codesA, 0x11, code_bytes);
    q.memset(codesB, 0x11, code_bytes);
    q.fill(scalesA, (uint16_t)0x3C00, (size_t)rgroups * Kt * 16);  // 1.0 | zp=0
    q.fill(scalesB, (uint16_t)0x3C00, (size_t)rgroups * Kt64 * 16);
    q.wait_and_throw();

    std::printf("  %-6s %-6s %10s %10s %10s %8s\n", "arm", "T", "ms", "TOPS",
                "GB/s(w)", "check");
    const int Ts[] = {256, 512, 1024};
    double topsA[3] = {0}, topsB[3] = {0};
    for (int ti = 0; ti < 3; ++ti) {
        const int T = Ts[ti];
        const int tgroups = T / 8, tasks = rgroups * tgroups;
        const double flop = 2.0 * (double)kM * kK * T;
        float *y = sycl::malloc_device<float>((size_t)T * kM, q);
        // arm A activations: s8 ones + unit scales + exact sums
        int8_t *aqA = sycl::malloc_device<int8_t>((size_t)Kt * T * 32, q);
        float *asA = sycl::malloc_device<float>((size_t)Kt * T, q);
        int32_t *sumA = sycl::malloc_device<int32_t>((size_t)Kt * T, q);
        q.memset(aqA, 0x01, (size_t)Kt * T * 32);
        q.fill(asA, 1.0f, (size_t)Kt * T);
        q.fill(sumA, 32, (size_t)Kt * T);
        // arm B activations: s4 ones, half the bytes
        uint8_t *aqB = sycl::malloc_device<uint8_t>((size_t)Kt64 * T * 32, q);
        float *asB = sycl::malloc_device<float>((size_t)Kt64 * T, q);
        int32_t *sumB = sycl::malloc_device<int32_t>((size_t)Kt64 * T, q);
        q.memset(aqB, 0x11, (size_t)Kt64 * T * 32);
        q.fill(asB, 1.0f, (size_t)Kt64 * T);
        q.fill(sumB, 64, (size_t)Kt64 * T);
        q.wait_and_throw();
        double sA = 0.0;
        if (std::strchr(armsel, 'a')) {
            const gemm_w4a8_k ka{codesA, scalesA, aqA, asA, sumA, y, T, tasks,
                                 tgroups, Kt};
            sA = time_kernel(q, ka, tasks, 10);
            int bad = check_output(q, y, T, "w4a8");
            topsA[ti] = flop / sA / 1e12;
            std::printf("  %-6s %-6d %10.3f %10.1f %10.1f %8s\n", "w4a8", T,
                        sA * 1e3, topsA[ti], code_bytes / sA / 1e9,
                        bad ? "FAIL" : "PASS");
        }
        if (std::strchr(armsel, 'b')) {
            q.memset(y, 0, sizeof(float) * (size_t)T * kM).wait();
            const gemm_w4a4_k kb{codesB, scalesB, aqB, asB, sumB, y, T, tasks,
                                 tgroups, Kt64};
            const double sB = time_kernel(q, kb, tasks, 10);
            int bad = check_output(q, y, T, "w4a4");
            topsB[ti] = flop / sB / 1e12;
            std::printf("  %-6s %-6d %10.3f %10.1f %10.1f %8s\n", "w4a4", T,
                        sB * 1e3, topsB[ti], code_bytes / sB / 1e9,
                        bad ? "FAIL" : "PASS");
            if (sA > 0.0)
                std::printf("  ratio  %-6d %9.2fx  (kill < 1.50x)\n", T, sA / sB);
        }

        sycl::free(y, q);
        sycl::free(aqA, q); sycl::free(asA, q); sycl::free(sumA, q);
        sycl::free(aqB, q); sycl::free(asB, q); sycl::free(sumB, q);
    }
    sycl::free(codesA, q); sycl::free(codesB, q);
    sycl::free(scalesA, q); sycl::free(scalesB, q);
    std::printf("\ndone\n");
    return 0;
}
