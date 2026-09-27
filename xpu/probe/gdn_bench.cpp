// Model-free measurement of the linked library's actual GDN kernels.
// Build: icpx -fsycl -O2 -Ixpu/src xpu/probe/gdn_bench.cpp /absolute/lib.so \
//        -Wl,-rpath,/absolute/library/directory -o /tmp/gdn_bench
// Run: TQ_XPU_DEV=1 /tmp/gdn_bench [rows=128] [repeats=10] [warmups=2]
//                               [key_heads=16] [value_heads=48] [dump_prefix]
// Dumps are native-endian float32: PREFIX.MODE.output.f32 and .state.f32.
// Every invocation starts from identical nonzero recurrent state. State reset,
// output poisoning and readback are excluded from the measured wall time.
#include "tq_common.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int D = 128;
constexpr float eps = 1.0e-6f;
struct Buffers {
    std::vector<void *> pointers;
    template<class T> T *upload(const std::vector<T> &v) {
        auto *p = static_cast<T *>(tq_dev_alloc(v.size() * sizeof(T), "gdn bench"));
        pointers.push_back(p);
        tq_h2d(p, v.data(), v.size() * sizeof(T));
        return p;
    }
    ~Buffers() {
        try { tq_q().wait_and_throw(); } catch (...) {}
        for (auto *p : pointers) tq_dev_free(p);
    }
};
uint16_t bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    return uint16_t(bits >> 16);
}
// Integer PRNG and power-of-two scale avoid standard-library distribution drift.
float sample(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return float(int32_t(state & 0xffffu) - 32768) / 32768.0f;
}
template<class T> uint64_t hash(const std::vector<T> &v) {
    uint64_t h = 14695981039346656037ull;
    const auto *bytes = reinterpret_cast<const uint8_t *>(v.data());
    for (size_t i = 0; i < v.size() * sizeof(T); ++i) {
        h ^= bytes[i];
        h *= 1099511628211ull;
    }
    return h;
}
int number(const char *s, int low, int high) {
    size_t used = 0;
    const int value = std::stoi(s, &used);
    if (used != std::strlen(s) || value < low || value > high)
        throw std::invalid_argument("argument outside supported range");
    return value;
}
void dump(const std::string &path, const std::vector<float> &v) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(v.data()),
              static_cast<std::streamsize>(v.size() * sizeof(float)));
    if (!out) throw std::runtime_error("cannot write " + path);
}
struct Error {
    double max_abs = 0, relative_l2 = 0, max_reference = 0;
    size_t worst_index = 0, nonfinite = 0;
    bool pass = true;
};
Error compare(const std::vector<float> &reference, const std::vector<float> &actual) {
    Error e;
    double squared_error = 0, squared_reference = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double a = reference[i], b = actual[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++e.nonfinite;
            continue;
        }
        const double delta = std::abs(a - b);
        if (delta > e.max_abs) { e.max_abs = delta; e.worst_index = i; }
        e.max_reference = std::max(e.max_reference, std::abs(a));
        squared_error += delta * delta;
        squared_reference += a * a;
    }
    e.relative_l2 = std::sqrt(squared_error / std::max(squared_reference, 1e-30));
    // Diagnostic reassociation gate, not a substitute for whole-model parity.
    e.pass = !e.nonfinite && e.relative_l2 <= 1e-4 &&
             e.max_abs <= 1e-4 * std::max(e.max_reference, 1.0);
    return e;
}
void print_error(const char *name, const Error &e) {
    std::cout << ",\"" << name << "\":{\"max_abs\":" << e.max_abs
              << ",\"relative_l2\":" << e.relative_l2
              << ",\"max_reference\":" << e.max_reference
              << ",\"worst_index\":" << e.worst_index
              << ",\"nonfinite\":" << e.nonfinite << "}";
}
struct Result {
    std::vector<float> output, state;
    double median_ms = 0, min_ms = 0, mean_ms = 0;
};
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc > 7) throw std::invalid_argument("expected rows repeats warmups nk nv dump_prefix");
        const int rows = argc > 1 ? number(argv[1], 1, 8192) : 128;
        const int repeats = argc > 2 ? number(argv[2], 1, 10000) : 10;
        const int warmups = argc > 3 ? number(argv[3], 0, 1000) : 2;
        const int nk = argc > 4 ? number(argv[4], 1, 64) : 16;
        const int nv = argc > 5 ? number(argv[5], 1, 64) : 48;
        const std::string prefix = argc > 6 ? argv[6] : "";
        if (nv % nk) throw std::invalid_argument("value heads must be divisible by key heads");
        const int conv_dim = (2 * nk + nv) * D, value_dim = nv * D;
        uint32_t seed = 0x6947ab21u;
        std::vector<float> conv(size_t(rows) * conv_dim), z(size_t(rows) * value_dim);
        std::vector<float> b(size_t(rows) * nv), a(b.size()), alog(nv), norm(D);
        std::vector<uint16_t> bias(nv);
        std::vector<float> initial(size_t(nv) * D * D);
        for (auto &x : conv) x = sample(seed) * 0.75f;
        for (auto &x : z) x = sample(seed) * 3.0f;
        for (auto &x : b) x = sample(seed) * 5.0f;
        for (auto &x : a) x = sample(seed) * 6.0f - 2.0f;
        for (int h = 0; h < nv; ++h) {
            alog[h] = -4.0f + 6.0f * float(h) / float(std::max(nv - 1, 1));
            bias[h] = bf16(sample(seed));
        }
        for (auto &x : norm) x = 1.0f + sample(seed) * 0.25f;
        for (auto &x : initial) x = sample(seed) * 0.125f;
        // Include near-zero norm cases and both tails of stable softplus/sigmoid.
        for (int t = 0; t < rows; ++t) {
            if (t % 17 == 0) {
                std::fill_n(conv.data() + size_t(t) * conv_dim, D, 0.0f);
                std::fill_n(conv.data() + size_t(t) * conv_dim + nk * D, D, 1e-8f);
            }
            if (t % 13 == 0) {
                a[size_t(t) * nv] = (t % 2) ? 30.0f : -30.0f;
                b[size_t(t) * nv] = (t % 2) ? -30.0f : 30.0f;
            }
        }
        Buffers buffers;
        auto *dc = buffers.upload(conv), *dz = buffers.upload(z);
        auto *db = buffers.upload(b), *da = buffers.upload(a);
        auto *dalog = buffers.upload(alog), *dnorm = buffers.upload(norm);
        auto *dbias = buffers.upload(bias);
        auto *state = buffers.upload(initial);
        std::vector<float> poison(z.size(), std::numeric_limits<float>::quiet_NaN());
        auto *out = buffers.upload(poison);
        // Reused for every chunk; allocation/upload stay outside timed work.
        const size_t factor_capacity = 2 * size_t(rows) * nk;
        auto *factors = buffers.upload(std::vector<float>(
            factor_capacity, std::numeric_limits<float>::quiet_NaN()));
        std::cout << std::setprecision(12);
        std::cout << "{\"kind\":\"inputs\",\"rows\":" << rows << ",\"nk\":" << nk
                  << ",\"nv\":" << nv << ",\"seed\":\"6947ab21\",\"conv_hash\":\""
                  << std::hex << hash(conv) << "\",\"z_hash\":\"" << hash(z)
                  << "\",\"b_hash\":\"" << hash(b) << "\",\"a_hash\":\"" << hash(a)
                  << "\",\"alog_hash\":\"" << hash(alog) << "\",\"bias_hash\":\"" << hash(bias)
                  << "\",\"norm_hash\":\"" << hash(norm) << "\",\"state_hash\":\""
                  << hash(initial) << std::dec << "\",\"output_elements\":" << z.size()
                  << ",\"state_elements\":" << initial.size() << ",\"warmups\":" << warmups
                  << ",\"repeats\":" << repeats << "}" << std::endl;
        auto run = [&](int mode) {
            Result r;
            r.output.resize(z.size()); r.state.resize(initial.size());
            std::vector<double> times;
            setenv("TQ_XPU_DELTA_FAST", mode == 0 ? "0" : "1", 1);
            for (int repeat = -warmups; repeat < repeats; ++repeat) {
                tq_h2d(state, initial.data(), initial.size() * sizeof(float));
                tq_h2d(out, poison.data(), poison.size() * sizeof(float));
                tq_q().wait_and_throw();
                const auto start = std::chrono::steady_clock::now();
                if (mode < 2) {
                    for (int t = 0; t < rows; ++t)
                        x_linear_decode_core_gated(out + size_t(t) * value_dim, state,
                            dc + size_t(t) * conv_dim, dz + size_t(t) * value_dim,
                            db + size_t(t) * nv, da + size_t(t) * nv, dalog, dbias,
                            dnorm, nk, D, nv, D, eps);
                } else {
                    // A non-CK-aligned split tests carry across separate launches.
                    const int split = mode == 3 && rows > 1 ? std::min(rows - 1, 13) : rows;
                    for (int t = 0; t < rows; t += split) {
                        const int count = std::min(split, rows - t);
                        if (x_deltanet_chunk(out + size_t(t) * value_dim, state,
                                dc + size_t(t) * conv_dim, dz + size_t(t) * value_dim,
                                db + size_t(t) * nv, da + size_t(t) * nv, dalog, dbias,
                                dnorm, count, nk, D, nv, D, eps,
                                factors, factor_capacity))
                            throw std::runtime_error("x_deltanet_chunk rejected shape");
                    }
                }
                tq_q().wait_and_throw();
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                if (repeat >= 0) times.push_back(ms);
            }
            tq_d2h(r.output.data(), out, r.output.size() * sizeof(float));
            tq_d2h(r.state.data(), state, r.state.size() * sizeof(float));
            for (double ms : times) r.mean_ms += ms / times.size();
            std::sort(times.begin(), times.end());
            r.min_ms = times.front();
            r.median_ms = (times[(times.size() - 1) / 2] + times[times.size() / 2]) * 0.5;
            return r;
        };
        const auto reference = run(0);
        bool pass = true;
        const char *names[] = {"serial_generic", "serial_fast", "chunk", "chunk_split"};
        for (int mode = 0; mode < 4; ++mode) {
            const auto candidate = mode == 0 ? Result{} : run(mode);
            const auto &result = mode == 0 ? reference : candidate;
            const auto output_error = compare(reference.output, result.output);
            const auto state_error = compare(reference.state, result.state);
            pass &= output_error.pass && state_error.pass;
            if (!prefix.empty()) {
                dump(prefix + "." + names[mode] + ".output.f32", result.output);
                dump(prefix + "." + names[mode] + ".state.f32", result.state);
            }
            std::cout << "{\"kind\":\"measurement\",\"mode\":\"" << names[mode]
                      << "\",\"median_ms\":" << result.median_ms
                      << ",\"min_ms\":" << result.min_ms << ",\"mean_ms\":" << result.mean_ms
                      << ",\"us_per_token\":" << 1000 * result.median_ms / rows
                      << ",\"output_hash\":\"" << std::hex << hash(result.output)
                      << "\",\"state_hash\":\"" << hash(result.state) << std::dec << "\"";
            print_error("output", output_error); print_error("state", state_error);
            std::cout << ",\"pass\":" << (output_error.pass && state_error.pass ? "true" : "false")
                      << "}" << std::endl;
        }
        // Exact state-only oracle covers the overlapping short-tail advance.
        // Do not include this correctness check in GDN performance timings.
        bool conv_pass = true;
        for (int ks : {4, 7}) {
            constexpr int channels = 257;
            std::vector<float> before(size_t(channels) * ks);
            std::vector<float> raw(size_t(ks + 2) * channels);
            for (size_t i = 0; i < before.size(); ++i) before[i] = float(i + 1);
            for (size_t i = 0; i < raw.size(); ++i) raw[i] = -float(i + 1);
            auto *dstate = buffers.upload(before), *draw = buffers.upload(raw);
            std::vector<float> expected(before.size()), actual(before.size());
            for (int count = 1; count <= ks + 2; ++count) {
                for (int channel = 0; channel < channels; ++channel)
                    for (int tap = 0; tap < ks; ++tap) {
                        const int idx = count - ks + tap;
                        expected[size_t(channel) * ks + tap] = idx >= 0
                            ? raw[size_t(idx) * channels + channel]
                            : before[size_t(channel) * ks + tap + count];
                    }
                bool exact = true;
                for (int repeat = 0; repeat < 32; ++repeat) {
                    tq_h2d(dstate, before.data(), before.size() * sizeof(float));
                    x_linear_conv_advance(dstate, draw, channels, ks, count);
                    tq_q().wait_and_throw();
                    tq_d2h(actual.data(), dstate, actual.size() * sizeof(float));
                    exact &= std::memcmp(expected.data(), actual.data(),
                                         expected.size() * sizeof(float)) == 0;
                }
                conv_pass &= exact;
                std::cout << "{\"kind\":\"conv_advance\",\"ks\":" << ks
                          << ",\"rows\":" << count << ",\"repeats\":32,\"exact\":"
                          << (exact ? "true" : "false") << "}" << std::endl;
            }
        }
        pass &= conv_pass;
        std::cout << "{\"status\":\"" << (pass ? "PASS" : "FAIL") << "\"}" << std::endl;
        return pass ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << "gdn_bench: " << e.what() << '\n';
        return 2;
    }
}
