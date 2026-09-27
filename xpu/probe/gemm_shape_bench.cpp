// Actual production W4A4/W4A8 GEMM on gate and down weights from one model layer.
// Build per library: icpx -fsycl -O2 -Ixpu/src xpu/probe/gemm_shape_bench.cpp /absolute/lib.so -Wl,-rpath,/absolute/libdir -o /tmp/gemm_shape_bench
// Run with matching quantization selectors for baseline/candidate:
// TQ_XPU_K64=gate,up,down /tmp/gemm_shape_bench model.tqf layer T repeats warmups snapshot.bin
// Optional final argument s8 selects W4A8 with caller-selected TQ_XPU_K64=''.
// Omitted or s4 preserves the original W4A4 mode and binary snapshot format.
// cmp complete snapshots from separately linked baseline/candidate probes.
// Repeated-weight timings may be cache-hot; full batch decode is the arbiter.
#include "tq_common.hpp"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct GemmModelLifetime {
    ~GemmModelLifetime() { qwn_free(); }
};

struct GemmShapeBuffers {
    std::vector<void *> allocations;
    template <class T> T *allocate(size_t count) {
        auto *p = static_cast<T *>(tq_dev_alloc(count * sizeof(T), "gemm_shape.probe"));
        if (!p) throw std::runtime_error("allocation failed");
        allocations.push_back(p);
        return p;
    }
    ~GemmShapeBuffers() {
        tq_q().wait();
        for (void *p : allocations) tq_dev_free(p);
    }
};

template <class T>
static void snapshot(std::ofstream &out, const std::vector<T> &data) {
    out.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T));
    if (!out) throw std::runtime_error("snapshot write failed");
}

static std::vector<int> parse_shapes(const std::string &text) {
    std::vector<int> shapes;
    size_t begin = 0;
    do {
        const size_t comma = text.find(',', begin);
        const size_t end = comma == std::string::npos ? text.size() : comma;
        if (begin == end) throw std::runtime_error("empty token count in shape list");
        int value = 0;
        for (size_t i = begin; i < end; ++i) {
            if (text[i] < '0' || text[i] > '9')
                throw std::runtime_error("token counts must contain decimal digits only");
            value = value * 10 + (text[i] - '0');
            if (value > 4096) throw std::runtime_error("token count exceeds 4096");
        }
        if (value < 8 || value % 8)
            throw std::runtime_error("token counts must be positive multiples of eight");
        for (int previous : shapes)
            if (previous == value) throw std::runtime_error("duplicate token count in shape list");
        shapes.push_back(value);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    } while (true);
    return shapes;
}

int main(int argc, char **argv) {
    try {
        if (argc != 7 && argc != 8) {
            std::cerr << "usage: gemm_shape_bench model.tqf layer T[,T...] repeats warmups snapshot.bin [s4|s8]\n";
            return 2;
        }
        const int layer = std::stoi(argv[2]);
        const std::string shape_text = argv[3];
        const std::vector<int> shapes = parse_shapes(shape_text);
        const bool sweep = shape_text.find(',') != std::string::npos;
        const int repeats = std::stoi(argv[4]), warmups = std::stoi(argv[5]);
        const std::string activation = argc == 8 ? argv[7] : "s4";
        if (activation != "s4" && activation != "s8")
            throw std::runtime_error("activation must be s4 or s8");
        const bool s8 = activation == "s8";
        if (layer < 0 || layer >= TQ_MAX_LAYERS ||
            repeats < 1 || repeats > 10000 || warmups < 0 || warmups > 10000)
            throw std::runtime_error("invalid layer, rows or iteration count");
        const int rc = qwn_init(argv[1]);
        if (rc) throw std::runtime_error("model initialization failed: " + std::to_string(rc));
        GemmModelLifetime model;
        if (layer >= g_qwen.L) throw std::runtime_error("layer outside loaded model");
        for (int T : shapes) {
        const std::string snapshot_path = sweep
            ? std::string(argv[6]) + ".t" + std::to_string(T) : argv[6];
        std::ofstream out(snapshot_path, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open snapshot");
        snapshot(out, std::vector<uint32_t>{s8 ? 0x474D5338u : 0x474D5350u,
                                           1u, (uint32_t)layer, (uint32_t)T});
        const tq_qmma_weight_t *weights[] = {&g_qwen.layers[layer].mlp_gate,
                                             &g_qwen.layers[layer].mlp_down};
        const char *names[] = {"gate", "down"};
        for (int index = 0; index < 2; ++index) {
            const auto *w = weights[index];
            if (!w->s4_ready || w->s4_k16 || !w->d_s4 || !w->d_s4_scale ||
                (s8 ? w->s4_k64 != 0 : w->s4_k64 == 0))
                throw std::runtime_error(s8
                    ? "S8 mode requires W4-K32 gate/down; set TQ_XPU_K64=''"
                    : "S4 mode requires W4-K64 gate/down; set TQ_XPU_K64=gate,up,down");
            const int K = w->K, M = w->M;
            const size_t elements = (size_t)K * T, blocks = elements / (s8 ? 32 : 64);
            GemmShapeBuffers buffers;
            std::vector<float> input(elements), output((size_t)M * T);
            uint32_t seed = 391;
            for (float &v : input) {
                seed = seed * 1664525u + 1013904223u;
                v = (float)((int)(seed >> 8) - 8388608) / 1048576.0f;
            }
            for (int row = 0; row < T; ++row)
                for (int col = 0; col < 64; ++col)
                    input[(size_t)row * K + col] = 0.0f;
            auto *x = buffers.allocate<float>(elements);
            auto *aq = buffers.allocate<uint8_t>(elements / (s8 ? 1 : 2));
            auto *as = buffers.allocate<float>(blocks);
            auto *asum = buffers.allocate<int32_t>(blocks);
            auto *y = buffers.allocate<float>((size_t)M * T);
            tq_h2d(x, input.data(), elements * sizeof(float));
            tq_q().memset(y, 0xA5, output.size() * sizeof(float));
            if (s8)
                x_quantize_act_chunk(x, K, T, reinterpret_cast<int8_t *>(aq), as, asum);
            else
                x_quantize_act_chunk_s4(x, K, T, aq, as, asum);
            auto launch = [&] {
                const int ret = s8
                    ? x_gemm_w4a8(w, reinterpret_cast<const int8_t *>(aq), as, asum,
                                   y, T, nullptr, 0)
                    : x_gemm_w4a4(w, aq, as, asum, y, T);
                if (ret) throw std::runtime_error("GEMM failed: " + std::to_string(ret));
            };
            launch();
            tq_q().wait_and_throw();
            tq_d2h(output.data(), y, output.size() * sizeof(float));
            snapshot(out, std::vector<uint32_t>{(uint32_t)M, (uint32_t)K});
            snapshot(out, input);
            snapshot(out, output);
            for (int i = 0; i < warmups; ++i) launch();
            tq_q().wait_and_throw();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < repeats; ++i) launch();
            tq_q().wait_and_throw();
            const double us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count() / repeats;
            std::cout << std::fixed << std::setprecision(3)
                      << "projection=" << names[index] << " M=" << M << " K=" << K
                      << " T=" << T << " activation=" << activation << " gemm_us=" << us
                      << " snapshot=" << snapshot_path << '\n';
        }
        out.close();
        if (!out) throw std::runtime_error("snapshot close failed");
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "gemm_shape_bench: " << e.what() << '\n';
        return 1;
    }
}
