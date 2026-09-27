// Linked production activation-staging probe; no model or reference substitute.
// Build once per library: icpx -fsycl -O2 -Ixpu/src xpu/probe/chunk_quant_bench.cpp /absolute/lib.so -Wl,-rpath,/absolute/library/directory -o /tmp/chunk_quant_bench
// Run: TQ_XPU_DEV=1 /tmp/chunk_quant_bench K T repeats warmups snapshot.bin finite|exceptional
// Compare baseline/candidate snapshots with cmp; snapshots contain input plus
// complete S8/S4 code, scale and sum buffers. Timings exclude copies/readback.
#include "tq_common.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct QuantBuffers {
    std::vector<void *> allocations;
    template <class T> T *allocate(size_t count) {
        auto *p = static_cast<T *>(tq_dev_alloc(count * sizeof(T), "chunk_quant.probe"));
        if (!p) throw std::runtime_error("allocation failed");
        allocations.push_back(p);
        return p;
    }
    ~QuantBuffers() {
        tq_q().wait();
        for (void *p : allocations) tq_dev_free(p);
    }
};

template <class T>
static void snapshot(std::ofstream &out, const std::vector<T> &data) {
    out.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T));
    if (!out) throw std::runtime_error("snapshot write failed");
}

int main(int argc, char **argv) {
    try {
        if (argc != 7) {
            std::cerr << "usage: chunk_quant_bench K T repeats warmups snapshot.bin finite|exceptional\n";
            return 2;
        }
        const int K = std::stoi(argv[1]), T = std::stoi(argv[2]);
        const int repeats = std::stoi(argv[3]), warmups = std::stoi(argv[4]);
        const std::string mode = argv[6];
        if (K < 64 || K > 65536 || K % 64 || T < 1 || T > 4096 ||
            repeats < 1 || repeats > 10000 || warmups < 0 || warmups > 10000 ||
            (mode != "finite" && mode != "exceptional"))
            throw std::runtime_error("invalid shape, iterations or input mode");
        const size_t elements = (size_t)K * T;
        std::vector<float> input(elements);
        uint32_t seed = 391;
        for (float &v : input) {
            seed = seed * 1664525u + 1013904223u;
            v = (float)((int)(seed >> 8) - 8388608) / 1048576.0f;
        }
        // Different row/tile values catch transposed staging writes. Zero,
        // subnormal and half-way quantization cases exercise the scalar branches.
        for (size_t block = 0; block < elements / 64; ++block) {
            float *p = input.data() + block * 64;
            switch (block % 16) {
            case 0:
                std::fill(p, p + 64, 0.0f);
                p[0] = -0.0f;
                break;
            case 1:
                for (int j = 0; j < 64; ++j)
                    p[j] = (j & 1 ? -1.0f : 1.0f) * 1.0e-20f * (j + 1);
                break;
            case 2:
                for (int j = 0; j < 64; ++j) p[j] = (float)(j - 32) + 0.5f;
                p[0] = p[32] = 127.0f;
                break;
            case 3:
                std::fill(p, p + 64, std::numeric_limits<float>::denorm_min());
                p[0] = -std::numeric_limits<float>::denorm_min();
                break;
            case 4:
                p[0] = std::numeric_limits<float>::max();
                p[32] = -std::numeric_limits<float>::max();
                break;
            default:
                break;
            }
            if (mode == "exceptional") {
                // Walk NaN through every lane/column; include all-NaN and
                // infinities. These compare the actual device conversion and
                // native maximum behavior, not a host-cast approximation.
                p[block % 64] = std::numeric_limits<float>::quiet_NaN();
                if (block % 67 == 0)
                    std::fill(p, p + 64, std::numeric_limits<float>::quiet_NaN());
                else if (block % 67 == 1)
                    p[(block + 17) % 64] = std::numeric_limits<float>::infinity();
                else if (block % 67 == 2)
                    p[(block + 31) % 64] = -std::numeric_limits<float>::infinity();
            }
        }
        QuantBuffers buffers;
        auto *x = buffers.allocate<float>(elements);
        auto *aq = buffers.allocate<uint8_t>(elements);
        auto *as = buffers.allocate<float>(elements / 32);
        auto *asum = buffers.allocate<int32_t>(elements / 32);
        tq_h2d(x, input.data(), elements * sizeof(float));
        std::ofstream out(argv[5], std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open snapshot");
        const std::vector<uint32_t> header = {0x51535447u, 1u, (uint32_t)K,
            (uint32_t)T, mode == "exceptional" ? 1u : 0u};
        snapshot(out, header);
        snapshot(out, input);
        for (int bits : {8, 4}) {
            const size_t blocks = elements / (bits == 8 ? 32 : 64);
            const size_t code_bytes = elements / (bits == 8 ? 1 : 2);
            auto launch = [&] {
                if (bits == 8)
                    x_quantize_act_chunk(x, K, T, reinterpret_cast<int8_t *>(aq), as, asum);
                else
                    x_quantize_act_chunk_s4(x, K, T, aq, as, asum);
            };
            // Poison all output buffers so missed stores fail byte comparison.
            tq_q().memset(aq, 0xA5, elements);
            tq_q().memset(as, 0xA5, elements / 32 * sizeof(float));
            tq_q().memset(asum, 0xA5, elements / 32 * sizeof(int32_t));
            launch();
            tq_q().wait_and_throw();
            std::vector<uint8_t> codes(code_bytes);
            std::vector<float> scales(blocks);
            std::vector<int32_t> sums(blocks);
            tq_d2h(codes.data(), aq, codes.size());
            tq_d2h(scales.data(), as, scales.size() * sizeof(float));
            tq_d2h(sums.data(), asum, sums.size() * sizeof(int32_t));
            snapshot(out, codes);
            snapshot(out, scales);
            snapshot(out, sums);
            for (int i = 0; i < warmups; ++i) launch();
            tq_q().wait_and_throw();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < repeats; ++i) launch();
            tq_q().wait_and_throw();
            const double us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count() / repeats;
            std::cout << std::fixed << std::setprecision(3)
                      << "bits=" << bits << " K=" << K << " T=" << T
                      << " input=" << mode << " stage_us=" << us
                      << " minimum_bytes=" << elements * sizeof(float) + code_bytes + blocks * 8
                      << " snapshot=" << argv[5] << '\n';
        }
        out.close();
        if (!out) throw std::runtime_error("snapshot close failed");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "chunk_quant_bench: " << e.what() << '\n';
        return 1;
    }
}
