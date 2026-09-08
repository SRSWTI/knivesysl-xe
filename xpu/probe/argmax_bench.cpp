// Model-free check and host-visible latency probe for the actual x_argmax API.
// Build: icpx -fsycl -O2 -Ixpu/src xpu/probe/argmax_bench.cpp /absolute/lib.so -Wl,-rpath,/absolute/library/directory -o /tmp/argmax_bench
// Run: TQ_XPU_DEV=1 /tmp/argmax_bench [vocab=151936] [repeats=100] [warmups=10]
#include "tq_common.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

struct Buffers {
    std::vector<void *> ptrs;
    template<class T> T *allocate(size_t count) {
        auto *p = static_cast<T *>(tq_dev_alloc(count * sizeof(T), "argmax probe"));
        ptrs.push_back(p);
        return p;
    }
    ~Buffers() {
        tq_q().wait_and_throw();
        for (auto *p : ptrs) tq_dev_free(p);
    }
};

static uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

int main(int argc, char **argv) {
    const int vocab = argc > 1 ? std::atoi(argv[1]) : 151936;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 100;
    const int warmups = argc > 3 ? std::atoi(argv[3]) : 10;
    if (vocab < 1 || vocab > 1048576 || repeats < 1 || repeats > 100000 ||
        warmups < 0 || warmups > 10000) return 2;
    Buffers buffers;
    const int capacity = std::max(vocab, TQ_ARGMAX_BLOCKS * 256 + 1);
    auto *logits = buffers.allocate<float>(capacity);
    auto *values = buffers.allocate<float>(TQ_ARGMAX_BLOCKS + 1);
    auto *ids = buffers.allocate<int>(TQ_ARGMAX_BLOCKS + 1);
    int checked = 0;
    auto check = [&](const char *name, const std::vector<float> &input) {
        tq_h2d(logits, input.data(), input.size() * sizeof(float));
        float expected_value = -std::numeric_limits<float>::max();
        int expected_id = 0;
        for (int i = 0; i < static_cast<int>(input.size()); ++i) {
            if (input[i] > expected_value ||
                (input[i] == expected_value && i < expected_id)) {
                expected_value = input[i];
                expected_id = i;
            }
        }
        int actual_id = -1;
        float actual_value = std::numeric_limits<float>::quiet_NaN();
        x_argmax(logits, static_cast<int>(input.size()), values, ids,
                 &actual_id, &actual_value);
        const bool exact = actual_id == expected_id && bits(actual_value) == bits(expected_value);
        if (!exact) {
            std::cerr << "FAIL " << name << " V=" << input.size()
                      << " id=" << actual_id << " expected=" << expected_id
                      << " bits=" << bits(actual_value) << " expected_bits=" << bits(expected_value) << '\n';
        }
        ++checked;
        return exact;
    };
    bool exact = true;
    const int sizes[] = {1, 255, 256, 257, 1023, 1024, 1025,
                         TQ_ARGMAX_BLOCKS * 256 + 1, vocab};
    for (int size : sizes) {
        std::vector<float> input(size, -2.0f);
        input.back() = 3.0f;
        exact &= check("last winner", input);
        input.front() = 3.0f;
        exact &= check("distant tie", input);
    }
    std::vector<float> edge(1025, std::numeric_limits<float>::quiet_NaN());
    exact &= check("all NaN", edge);
    std::fill(edge.begin(), edge.end(), -std::numeric_limits<float>::infinity());
    exact &= check("all negative infinity", edge);
    std::fill(edge.begin(), edge.end(), -std::numeric_limits<float>::max());
    exact &= check("sentinel tie", edge);
    edge[255] = std::numeric_limits<float>::infinity();
    edge[1024] = edge[255];
    exact &= check("positive infinity tie", edge);
    std::fill(edge.begin(), edge.end(), -1.0f);
    edge[255] = -0.0f;
    edge[256] = 0.0f;
    exact &= check("signed zero tie", edge);
    edge[255] = std::numeric_limits<float>::quiet_NaN();
    exact &= check("NaN beside winner", edge);
    int invalid_id = -1;
    float invalid_value = 1.0f;
    x_argmax(logits, 0, values, ids, &invalid_id, &invalid_value);
    exact &= invalid_id == 0 && bits(invalid_value) == bits(-std::numeric_limits<float>::max());
    if (!exact) return 1;

    std::vector<float> input(vocab);
    uint32_t seed = 391;
    for (float &value : input) {
        seed = seed * 1664525u + 1013904223u;
        value = static_cast<float>(seed >> 8) / 8388608.0f - 1.0f;
    }
    if (!check("deterministic random", input)) return 1;
    int id = 0;
    float value = 0.0f;
    auto launch = [&] { x_argmax(logits, vocab, values, ids, &id, &value); };
    for (int i = 0; i < warmups; ++i) launch();
    std::vector<double> samples;
    for (int trial = 0; trial < 9; ++trial) {
        tq_q().wait_and_throw();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) launch();
        samples.push_back(std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count() / repeats);
    }
    std::sort(samples.begin(), samples.end());
    std::cout << std::fixed << std::setprecision(3)
              << "exact=1 cases=" << checked + 1 << " vocab=" << vocab
              << " repeats=" << repeats << " warmups=" << warmups
              << " host_us_min=" << samples.front()
              << " host_us_median=" << samples[samples.size() / 2]
              << " host_us_max=" << samples.back()
              << " winner=" << id << " value_bits=" << bits(value) << '\n';
}
