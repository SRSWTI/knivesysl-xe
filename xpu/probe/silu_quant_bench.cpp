// Model-free unfused/fused SiLU staging and prepared residual GEMV comparison.
// Build against the candidate library: icpx -fsycl -O2 -Ixpu/src xpu/probe/silu_quant_bench.cpp /absolute/lib.so -Wl,-rpath,/absolute/library/directory -o /tmp/silu_quant_bench
// Run: TQ_XPU_DEV=1 /tmp/silu_quant_bench [K=18432] [M=64] [repeats=100] [warmups=10]
#include "tq_common.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Buffers {
    std::vector<void *> ptrs;
    template<class T> T *upload(const std::vector<T> &v) {
        auto *p = static_cast<T *>(tq_dev_alloc(v.size() * sizeof(T), "SiLU probe"));
        ptrs.push_back(p);
        tq_h2d(p, v.data(), v.size() * sizeof(T));
        return p;
    }
    ~Buffers() {
        tq_q().wait_and_throw();
        for (auto *p : ptrs) tq_dev_free(p);
    }
};

int main(int argc, char **argv) {
    const int K = argc > 1 ? std::atoi(argv[1]) : 18432;
    const int M = argc > 2 ? std::atoi(argv[2]) : 64;
    const int repeats = argc > 3 ? std::atoi(argv[3]) : 100;
    const int warmups = argc > 4 ? std::atoi(argv[4]) : 10;
    if (K < 64 || K > 65536 || K % 64 || M < 16 || M > 65536 || M % 16 ||
        repeats < 1 || repeats > 10000 || warmups < 0 || warmups > 10000) return 2;
    Buffers buffers;
    std::vector<float> gate(K), up(K), hidden(K), residual(M), output(M);
    uint32_t seed = 391;
    auto random = [&] {
        seed = seed * 1664525u + 1013904223u;
        return float(seed >> 8) / 8388608.0f - 1.0f;
    };
    for (int i = 0; i < K; ++i) {
        gate[i] = 8.0f * random();
        up[i] = 4.0f * random();
    }
    for (auto &v : residual) v = random();
    // Zero-scale blocks, negative signed zero, saturated-negative SiLU, and
    // very small finite blocks exercise quantizer branches without NaN casts.
    for (int i = 0; i < 32; ++i) gate[i] = 0.0f;
    gate[0] = -0.0f;
    gate[32] = -100.0f;
    gate[33] = 100.0f;
    for (int i = 64; i < std::min(K, 96); ++i) up[i] *= 1e-20f;
    auto *dg = buffers.upload(gate), *du = buffers.upload(up);
    auto *dh0 = buffers.upload(hidden), *dh1 = buffers.upload(hidden);
    auto *dr = buffers.upload(residual);
    auto *dy0 = buffers.upload(output), *dy1 = buffers.upload(output);
    auto timing = [&](auto &&launch) {
        for (int i = 0; i < warmups; ++i) launch();
        tq_q().wait_and_throw();
        std::vector<double> samples;
        for (int trial = 0; trial < 7; ++trial) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < repeats; ++i) launch();
            tq_q().wait_and_throw();
            samples.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count() / repeats);
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    };
    auto unfused = [&] {
        x_silu_mul(dh0, dg, du, K);
        x_prepare_gemv_act_s8(dh0, K);
    };
    auto fused = [&] { x_silu_mul_quant(dh1, dg, du, K); };
    unfused();
    fused();
    std::vector<float> a(K), b(K);
    tq_d2h(a.data(), dh0, K * sizeof(float));
    tq_d2h(b.data(), dh1, K * sizeof(float));
    if (std::memcmp(a.data(), b.data(), K * sizeof(float))) {
        int different = 0;
        for (int i = 0; i < K; ++i)
            different += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
        std::cerr << "FAIL hidden bits: " << different << '/' << K << '\n';
        return 1;
    }
    std::cout << std::fixed << std::setprecision(3)
              << "hidden_exact=1 K=" << K << " M=" << M
              << " unfused_stage_us=" << timing(unfused)
              << " fused_stage_us=" << timing(fused) << '\n';
    for (int tier = 0; tier < 4; ++tier) {
        const bool s8 = tier == 0;
        const int scale_k = tier == 2 ? 16 : (tier == 3 ? 64 : 32);
        tq_qmma_weight_t w{};
        w.M = M; w.K = K; w.Mt = M / 16; w.Kt = K / 32;
        std::vector<uint8_t> codes(size_t(M) * K / (s8 ? 1 : 2));
        for (auto &v : codes) { random(); v = uint8_t(seed >> 24); }
        // At K=M=64, the W8 case is an identity projection: every dequantized
        // S8 activation is observed separately, rather than only random dots.
        if (s8 && K == 64 && M == 64) {
            std::fill(codes.begin(), codes.end(), uint8_t(0));
            for (int row = 0; row < M; ++row)
                codes[((size_t(row / 16) * w.Kt + row / 32) * 16 + row % 16) * 32 + row % 32] = 1;
        }
        std::vector<uint16_t> scales(size_t(M) * (K / scale_k));
        for (size_t i = 0; i < scales.size(); ++i) {
            const uint16_t sh = sycl::bit_cast<uint16_t>(sycl::half(s8 ? 1.0f : 0.03125f));
            scales[i] = s8 ? sh : uint16_t((sh & 0xfff0u) | (i % 16));
        }
        if (s8) {
            w.s8_ready = 1; w.d_s8 = buffers.upload(codes); w.d_s8_scale = buffers.upload(scales);
        } else {
            w.s4_ready = 1; w.s4_k16 = scale_k == 16; w.s4_k64 = scale_k == 64;
            w.d_s4 = buffers.upload(codes); w.d_s4_scale = buffers.upload(scales);
        }
        auto old_path = [&] {
            x_silu_mul(dh0, dg, du, K);
            if (x_gemv_qmma_add(&w, dh0, dr, dy0)) throw std::runtime_error("unfused GEMV rejected");
        };
        auto new_path = [&] {
            fused();
            if (x_gemv_qmma_prepared(&w, dh1, dy1, dr)) throw std::runtime_error("prepared GEMV rejected");
        };
        old_path();
        new_path();
        std::vector<float> y0(M), y1(M);
        tq_d2h(y0.data(), dy0, M * sizeof(float));
        tq_d2h(y1.data(), dy1, M * sizeof(float));
        if (std::memcmp(y0.data(), y1.data(), M * sizeof(float))) {
            std::cerr << "FAIL prepared output bits: tier=" << tier << '\n';
            return 1;
        }
        std::cout << "tier=" << (s8 ? "w8" : "w4") << " scale_k=" << scale_k
                  << " output_exact=1 unfused_us=" << timing(old_path)
                  << " fused_us=" << timing(new_path) << '\n';
    }
}
