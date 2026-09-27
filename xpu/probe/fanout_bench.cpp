// Model-free benchmark of the actual production fanout kernel on deterministic
// packed weight matrices (not a host reference or a replacement implementation).
// Build per library: icpx -fsycl -O2 -Ixpu/src xpu/probe/fanout_bench.cpp /absolute/lib.so -Wl,-rpath,/absolute/libdir -o /tmp/fanout_bench
// Run: TQ_XPU_DEV=1 /tmp/fanout_bench mlp|gdn|tiny K k32|k64 repeats warmups snapshot.bin
// Compare complete baseline/candidate output snapshots using cmp. K=384 tests
// uneven/odd split spans; K=128 and tiny test empty/single-tile split spans.
#include "tq_common.hpp"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct FanoutBuffers {
    std::vector<void *> allocations;
    template <class T> T *upload(const std::vector<T> &data) {
        auto *p = static_cast<T *>(tq_dev_alloc(data.size() * sizeof(T), "fanout.probe"));
        if (!p) throw std::runtime_error("allocation failed");
        allocations.push_back(p);
        tq_h2d(p, data.data(), data.size() * sizeof(T));
        return p;
    }
    ~FanoutBuffers() {
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
            std::cerr << "usage: fanout_bench mlp|gdn|tiny K k32|k64 repeats warmups snapshot.bin\n";
            return 2;
        }
        const std::string mode = argv[1], tier = argv[3];
        const int K = std::stoi(argv[2]), repeats = std::stoi(argv[4]);
        const int warmups = std::stoi(argv[5]);
        if ((mode != "mlp" && mode != "gdn" && mode != "tiny") ||
            (tier != "k32" && tier != "k64") || K < 128 || K > 65536 || K % 128 ||
            repeats < 1 || repeats > 10000 || warmups < 0 || warmups > 10000)
            throw std::runtime_error("invalid mode, tier, shape or iteration count");
        const std::vector<int> rows = mode == "mlp" ? std::vector<int>{17408, 17408} :
            mode == "gdn" ? std::vector<int>{10240, 6144, 48, 48} : std::vector<int>{16, 48};
        const int count = (int)rows.size(), kt = K / 32;
        const int ksh = tier == "k64" ? 1 : 0;
        FanoutBuffers buffers;
        std::vector<tq_qmma_weight_t> weights(count);
        std::vector<const tq_qmma_weight_t *> ws(count);
        std::vector<float *> ys(count);
        uint32_t seed = 391;
        auto random = [&] {
            seed = seed * 1664525u + 1013904223u;
            return seed;
        };
        for (int n = 0; n < count; ++n) {
            auto &w = weights[n];
            w.M = rows[n]; w.K = K; w.Mt = w.M / 16; w.Kt = kt;
            w.s4_ready = 1; w.s4_k64 = ksh;
            std::vector<uint8_t> codes((size_t)w.M * K / 2);
            for (auto &v : codes) v = (uint8_t)(random() >> 24);
            std::vector<uint16_t> scales((size_t)w.M * (kt >> ksh));
            // Positive normal FP16 scales with reserved low nibble. Include
            // all signed zero points, including zero, plus zero-scale blocks.
            for (size_t i = 0; i < scales.size(); ++i) {
                const uint16_t magnitude = i % 37 == 0 ? 0 :
                    (uint16_t)(0x2400u + ((random() >> 22) & 0x3Fu) * 16u);
                scales[i] = (uint16_t)(magnitude | (i & 15u));
            }
            w.d_s4 = buffers.upload(codes);
            w.d_s4_scale = buffers.upload(scales);
            ys[n] = buffers.upload(std::vector<float>(w.M, -12345.0f));
            ws[n] = &w;
        }
        std::vector<float> input(K);
        for (float &v : input)
            v = (float)((int)(random() >> 8) - 8388608) / 1048576.0f;
        for (int i = 0; i < 32; ++i) input[i] = 0.0f;
        auto *x = buffers.upload(input);
        x_prepare_gemv_act_s8(x, K);
        auto launch = [&] {
            const int rc = x_gemv_w4a8_fanout(ws.data(), ys.data(), count);
            if (rc) throw std::runtime_error("fanout failed: " + std::to_string(rc));
        };
        launch();
        tq_q().wait_and_throw();
        std::ofstream out(argv[6], std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open snapshot");
        snapshot(out, std::vector<uint32_t>{0x46414E50u, 1u, (uint32_t)K,
            (uint32_t)ksh, (uint32_t)count});
        snapshot(out, rows);
        snapshot(out, input);
        size_t weight_bytes = 0;
        for (int n = 0; n < count; ++n) {
            std::vector<float> output(rows[n]);
            tq_d2h(output.data(), ys[n], output.size() * sizeof(float));
            snapshot(out, output);
            weight_bytes += (size_t)rows[n] * K / 2 + (size_t)rows[n] * (kt >> ksh) * 2;
        }
        out.close();
        if (!out) throw std::runtime_error("snapshot close failed");
        for (int i = 0; i < warmups; ++i) launch();
        tq_q().wait_and_throw();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) launch();
        tq_q().wait_and_throw();
        const double us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count() / repeats;
        std::cout << std::fixed << std::setprecision(3)
                  << "mode=" << mode << " K=" << K << " tier=" << tier
                  << " fanout_us=" << us << " minimum_weight_bytes=" << weight_bytes
                  << " snapshot=" << argv[6] << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "fanout_bench: " << e.what() << '\n';
        return 1;
    }
}
