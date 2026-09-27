// Model-free exactness and host-visible latency probe for single/batched argmax.
// Build: icpx -fsycl -O2 -Ixpu/src xpu/probe/argmax_bench.cpp /absolute/lib.so -Wl,-rpath,/absolute/library/directory -o /tmp/argmax_bench
// Run: TQ_XPU_DEV=1 /tmp/argmax_bench [vocab=151936] [repeats=100] [warmups=10] [rows=8]
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
    const int bench_rows = argc > 4 ? std::atoi(argv[4]) : TQ_ARGMAX_MAX_ROWS;
    if (vocab < 1 || vocab > 1048576 || repeats < 1 || repeats > 100000 ||
        warmups < 0 || warmups > 10000 ||
        bench_rows < 1 || bench_rows > TQ_ARGMAX_MAX_ROWS) return 2;
    Buffers buffers;
    const int capacity = std::max(vocab, TQ_ARGMAX_BLOCKS * 256 + 1);
    auto *logits = buffers.allocate<float>(static_cast<size_t>(capacity) * TQ_ARGMAX_MAX_ROWS);
    auto *values = buffers.allocate<float>(TQ_ARGMAX_BLOCKS + 1);
    auto *ids = buffers.allocate<int>(TQ_ARGMAX_BLOCKS + 1);
    constexpr size_t batch_scratch = TQ_ARGMAX_MAX_ROWS * (TQ_ARGMAX_BLOCKS + 1);
    auto *batch_values = buffers.allocate<float>(batch_scratch);
    auto *batch_ids = buffers.allocate<int>(batch_scratch);
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
    int batch_checked = 0;
    auto check_batch = [&](const char *name, const std::vector<float> &input,
                           int width, int rows) {
        tq_h2d(logits, input.data(), input.size() * sizeof(float));
        int actual_ids[TQ_ARGMAX_MAX_ROWS + 1];
        float actual_values[TQ_ARGMAX_MAX_ROWS + 1];
        std::fill_n(actual_ids, TQ_ARGMAX_MAX_ROWS + 1, -17);
        std::fill_n(actual_values, TQ_ARGMAX_MAX_ROWS + 1, 17.0f);
        x_argmax_batch(logits, width, rows, batch_values, batch_ids,
                       actual_ids, actual_values);
        bool batch_exact = actual_ids[rows] == -17 && actual_values[rows] == 17.0f;
        for (int row = 0; row < rows; ++row) {
            int single_id = -1;
            float single_value = std::numeric_limits<float>::quiet_NaN();
            x_argmax(logits + static_cast<size_t>(row) * width, width, values, ids,
                     &single_id, &single_value);
            int expected_id = 0;
            float expected_value = -std::numeric_limits<float>::max();
            for (int i = 0; i < width; ++i) {
                const float candidate = input[static_cast<size_t>(row) * width + i];
                if (candidate > expected_value ||
                    (candidate == expected_value && i < expected_id)) {
                    expected_id = i;
                    expected_value = candidate;
                }
            }
            const bool row_exact = actual_ids[row] == single_id &&
                bits(actual_values[row]) == bits(single_value) &&
                actual_ids[row] == expected_id &&
                bits(actual_values[row]) == bits(expected_value);
            if (!row_exact) {
                std::cerr << "FAIL batch " << name << " V=" << width
                          << " rows=" << rows << " row=" << row
                          << " id=" << actual_ids[row] << " single=" << single_id
                          << " expected=" << expected_id
                          << " bits=" << bits(actual_values[row])
                          << " single_bits=" << bits(single_value)
                          << " expected_bits=" << bits(expected_value) << '\n';
            }
            batch_exact &= row_exact;
            ++batch_checked;
        }
        if (!batch_exact) std::cerr << "FAIL batch output/row check " << name << '\n';
        return batch_exact;
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
    // Every row count exercises every edge pattern, with mixed neighboring rows
    // and successive widths sharing scratch to catch cross-row/stale winners.
    for (int width : sizes) {
        for (int rows = 1; rows <= TQ_ARGMAX_MAX_ROWS; ++rows) {
            for (int scenario = 0; scenario < 8; ++scenario) {
                std::vector<float> batch(static_cast<size_t>(width) * rows);
                for (int row = 0; row < rows; ++row) {
                    float *dst = batch.data() + static_cast<size_t>(row) * width;
                    std::fill_n(dst, width, -3.0f - row);
                    const int first = width > 256 ? 255 : 0;
                    const int last = width - 1;
                    switch ((scenario + row) % 8) {
                    case 0: // Negative winner at a different index per row.
                        dst[(static_cast<size_t>(row) * 509 + 17) % width] = -0.5f;
                        break;
                    case 1:
                        std::fill_n(dst, width, std::numeric_limits<float>::quiet_NaN());
                        break;
                    case 2:
                        std::fill_n(dst, width, -std::numeric_limits<float>::infinity());
                        break;
                    case 3:
                        std::fill_n(dst, width, -std::numeric_limits<float>::max());
                        break;
                    case 4:
                        dst[first] = dst[last] = std::numeric_limits<float>::infinity();
                        break;
                    case 5:
                        dst[last] = 0.0f;
                        dst[first] = -0.0f;
                        break;
                    case 6:
                        std::fill_n(dst, width, std::numeric_limits<float>::quiet_NaN());
                        dst[last] = 1.0f + row;
                        break;
                    case 7: // Cross-block and grid-stride ties.
                        dst[last] = dst[first] = 2.0f + row;
                        break;
                    }
                }
                exact &= check_batch("mixed edges", batch, width, rows);
            }
        }
    }
    int invalid_ids[TQ_ARGMAX_MAX_ROWS];
    float invalid_values[TQ_ARGMAX_MAX_ROWS];
    std::fill_n(invalid_ids, TQ_ARGMAX_MAX_ROWS, -1);
    std::fill_n(invalid_values, TQ_ARGMAX_MAX_ROWS, 1.0f);
    x_argmax_batch(logits, 0, TQ_ARGMAX_MAX_ROWS, batch_values, batch_ids,
                   invalid_ids, invalid_values);
    for (int row = 0; row < TQ_ARGMAX_MAX_ROWS; ++row)
        exact &= invalid_ids[row] == 0 &&
            bits(invalid_values[row]) == bits(-std::numeric_limits<float>::max());
    // Out-of-contract row counts must not enqueue or touch host buffers.
    x_argmax_batch(logits, vocab, 0, batch_values, batch_ids, &invalid_id, &invalid_value);
    x_argmax_batch(logits, vocab, TQ_ARGMAX_MAX_ROWS + 1, batch_values, batch_ids,
                   &invalid_id, &invalid_value);
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

    std::vector<float> batch_input(static_cast<size_t>(vocab) * bench_rows);
    for (float &v : batch_input) {
        seed = seed * 1664525u + 1013904223u;
        v = static_cast<float>(seed >> 8) / 8388608.0f - 1.0f;
    }
    if (!check_batch("deterministic random", batch_input, vocab, bench_rows)) return 1;
    int result_ids[TQ_ARGMAX_MAX_ROWS];
    float result_values[TQ_ARGMAX_MAX_ROWS];
    auto serial_launch = [&] {
        for (int row = 0; row < bench_rows; ++row)
            x_argmax(logits + static_cast<size_t>(row) * vocab, vocab, values, ids,
                     result_ids + row, result_values + row);
    };
    auto batch_launch = [&] {
        x_argmax_batch(logits, vocab, bench_rows, batch_values, batch_ids,
                       result_ids, result_values);
    };
    for (int i = 0; i < warmups; ++i) {
        serial_launch();
        batch_launch();
    }
    std::vector<double> serial_samples, batch_samples;
    auto measure = [&](auto &launch_wave, std::vector<double> &times) {
        tq_q().wait_and_throw();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) launch_wave();
        times.push_back(std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count() / repeats);
    };
    for (int trial = 0; trial < 9; ++trial) {
        if (trial & 1) {
            measure(batch_launch, batch_samples);
            measure(serial_launch, serial_samples);
        } else {
            measure(serial_launch, serial_samples);
            measure(batch_launch, batch_samples);
        }
    }
    std::sort(serial_samples.begin(), serial_samples.end());
    std::sort(batch_samples.begin(), batch_samples.end());
    std::cout << "batch_exact=1 row_cases=" << batch_checked << " vocab=" << vocab
              << " rows=" << bench_rows << " repeats=" << repeats << " warmups=" << warmups
              << " serial_wave_us_min=" << serial_samples.front()
              << " serial_wave_us_median=" << serial_samples[serial_samples.size() / 2]
              << " serial_wave_us_max=" << serial_samples.back()
              << " batch_wave_us_min=" << batch_samples.front()
              << " batch_wave_us_median=" << batch_samples[batch_samples.size() / 2]
              << " batch_wave_us_max=" << batch_samples.back() << '\n';
}
