// Exact, model-free isolation check of the linked library's actual GDN kernels.
// Build from the repository root against the candidate library and its header:
//   icpx -fsycl -O2 -std=c++17 -Ixpu/src xpu/probe/gdn_batch_bench.cpp \
//     /absolute/libforward_qwen_xpu.so -Wl,-rpath,/absolute/library/directory \
//     -o /tmp/gdn_batch_bench
// Run: TQ_XPU_DEV=1 /tmp/gdn_batch_bench [steps=3] [key_heads=16] [value_heads=48]
// Always checks widths 1,2,8, kernel sizes 4,7 and generic/fast dispatch separately.
// Each combination runs out-of-place and with conv_in == conv_out (alias=true).
// Reference and candidate use identical dispatch modes; no cross-mode tolerance.
// JSONL diagnostics include full-array byte mismatches, hashes, nonfinite active
// values and untouched-hole failures. Exit 0 means every check passed; 1 means
// numerical/isolation failure; 2 means invalid arguments or a runtime exception.
// No timings: each step reads back all outputs and both persistent state arrays.
#include "tq_common.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int D = 128;
constexpr int slots = 19;
constexpr float eps = 1.0e-6f;
constexpr std::array<int, 8> slot_order = {13, 2, 17, 5, 11, 0, 8, 15};

struct Buffers {
    std::vector<void *> pointers;
    template<class T> T *upload(const std::vector<T> &v) {
        auto *p = static_cast<T *>(tq_dev_alloc(v.size() * sizeof(T), "gdn batch bench"));
        if (!p) throw std::runtime_error("device allocation failed");
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

// Same integer PRNG and power-of-two scaling as gdn_bench.cpp.
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

bool active_slot(int slot, int width) {
    return std::find(slot_order.begin(), slot_order.begin() + width, slot) !=
           slot_order.begin() + width;
}

// Persistent host readback and device storage: no per-step scratch allocation.
struct CheckedBuffer {
    size_t stride;
    std::vector<float> initial, reference, candidate;
    float *d_reference, *d_candidate;

    CheckedBuffer(Buffers &buffers, size_t elements_per_slot, bool output,
                  uint32_t &seed)
        : stride(elements_per_slot), initial(size_t(slots) * stride),
          reference(initial.size()), candidate(initial.size()) {
        for (auto &x : initial)
            x = output ? std::numeric_limits<float>::quiet_NaN() : sample(seed) * 0.125f;
        d_reference = buffers.upload(initial);
        d_candidate = buffers.upload(initial);
    }

    void reset() {
        tq_h2d(d_reference, initial.data(), initial.size() * sizeof(float));
        tq_h2d(d_candidate, initial.data(), initial.size() * sizeof(float));
    }

    bool check(const char *name, int width) {
        tq_d2h(reference.data(), d_reference, reference.size() * sizeof(float));
        tq_d2h(candidate.data(), d_candidate, candidate.size() * sizeof(float));
        tq_q().wait_and_throw();
        const auto *ref_bytes = reinterpret_cast<const uint8_t *>(reference.data());
        const auto *got_bytes = reinterpret_cast<const uint8_t *>(candidate.data());
        size_t mismatch_bytes = 0, first_byte = 0;
        for (size_t i = 0; i < reference.size() * sizeof(float); ++i) {
            if (ref_bytes[i] != got_bytes[i]) {
                if (!mismatch_bytes) first_byte = i;
                ++mismatch_bytes;
            }
        }
        size_t reference_hole_changes = 0, candidate_hole_changes = 0;
        size_t reference_nonfinite = 0, candidate_nonfinite = 0;
        for (int slot = 0; slot < slots; ++slot) {
            const size_t offset = size_t(slot) * stride;
            if (active_slot(slot, width)) {
                for (size_t i = offset; i < offset + stride; ++i) {
                    reference_nonfinite += !std::isfinite(reference[i]);
                    candidate_nonfinite += !std::isfinite(candidate[i]);
                }
            } else {
                for (size_t i = offset; i < offset + stride; ++i) {
                    reference_hole_changes += std::memcmp(&reference[i], &initial[i], sizeof(float)) != 0;
                    candidate_hole_changes += std::memcmp(&candidate[i], &initial[i], sizeof(float)) != 0;
                }
            }
        }
        const bool pass = !mismatch_bytes && !reference_hole_changes &&
                          !candidate_hole_changes && !reference_nonfinite &&
                          !candidate_nonfinite;
        std::cout << ",\"" << name << "\":{\"elements\":" << reference.size()
                  << ",\"reference_hash\":\"" << std::hex << hash(reference)
                  << "\",\"candidate_hash\":\"" << hash(candidate) << std::dec
                  << "\",\"mismatch_bytes\":" << mismatch_bytes
                  << ",\"reference_hole_changes\":" << reference_hole_changes
                  << ",\"candidate_hole_changes\":" << candidate_hole_changes
                  << ",\"reference_nonfinite\":" << reference_nonfinite
                  << ",\"candidate_nonfinite\":" << candidate_nonfinite;
        if (mismatch_bytes) {
            const size_t element = first_byte / sizeof(float);
            uint32_t ref_bits, got_bits;
            std::memcpy(&ref_bits, &reference[element], sizeof(ref_bits));
            std::memcpy(&got_bits, &candidate[element], sizeof(got_bits));
            std::cout << ",\"first_mismatch_byte\":" << first_byte
                      << ",\"first_mismatch_slot\":" << element / stride
                      << ",\"first_mismatch_slot_element\":" << element % stride
                      << ",\"reference_bits\":\"" << std::hex << ref_bits
                      << "\",\"candidate_bits\":\"" << got_bits << std::dec << "\"";
        }
        std::cout << ",\"pass\":" << (pass ? "true" : "false") << "}";
        return pass;
    }
};
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc > 4) throw std::invalid_argument("expected steps key_heads value_heads");
        const int steps = argc > 1 ? number(argv[1], 1, 1000) : 3;
        const int nk = argc > 2 ? number(argv[2], 1, 64) : 16;
        const int nv = argc > 3 ? number(argv[3], 1, 64) : 48;
        if (nv % nk) throw std::invalid_argument("value heads must be divisible by key heads");
        const int conv_dim = (2 * nk + nv) * D, value_dim = nv * D;
        bool pass = true;
        for (int ks : {4, 7}) {
            for (int mode : {0, 1}) {
                if (setenv("TQ_XPU_DELTA_FAST", mode ? "1" : "0", 1))
                    throw std::runtime_error("cannot select GDN dispatch mode");
                for (int scenario = 0; scenario < 6; ++scenario) {
                    constexpr int widths[] = {1, 2, 8};
                    const int width = widths[scenario % 3];
                    const bool alias = scenario >= 3;
                    // Reset the seed for every case, so modes and widths see the
                    // same slot-specific values rather than unrelated datasets.
                    uint32_t seed = 0x6947ab21u;
                    Buffers buffers;
                    CheckedBuffer conv_state(buffers, size_t(conv_dim) * ks, false, seed);
                    CheckedBuffer recurrent(buffers, size_t(nv) * D * D, false, seed);
                    CheckedBuffer conv_out(buffers, conv_dim, true, seed);
                    CheckedBuffer core_out(buffers, value_dim, true, seed);
                    std::vector<float> raw(size_t(slots) * conv_dim);
                    std::vector<float> z(size_t(slots) * value_dim);
                    std::vector<float> b(size_t(slots) * nv), a(b.size());
                    std::vector<float> alog(nv), norm(D);
                    std::vector<uint16_t> bias(nv), weights(size_t(conv_dim) * ks);
                    for (auto &x : weights) x = bf16(sample(seed) * 0.25f);
                    // Exact-zero q and tiny k channels exercise normalization
                    // epsilon. Keep nonzero q channels even with a single head.
                    const int zero_q = nk > 1 ? D : D / 2;
                    std::fill_n(weights.data(), size_t(zero_q) * ks, uint16_t(0));
                    for (int channel = nk * D; channel < nk * D + D; ++channel)
                        for (int tap = 0; tap < ks; ++tap)
                            weights[size_t(channel) * ks + tap] = bf16(sample(seed) * 1.0e-8f);
                    for (int h = 0; h < nv; ++h) {
                        alog[h] = -4.0f + 6.0f * float(h) / float(std::max(nv - 1, 1));
                        bias[h] = bf16(sample(seed));
                    }
                    for (auto &x : norm) x = 1.0f + sample(seed) * 0.25f;
                    auto *draw = buffers.upload(raw), *dz = buffers.upload(z);
                    auto *db = buffers.upload(b), *da = buffers.upload(a);
                    auto *dalog = buffers.upload(alog), *dnorm = buffers.upload(norm);
                    auto *dbias = buffers.upload(bias), *dweights = buffers.upload(weights);
                    std::cout << "{\"kind\":\"case\",\"width\":" << width
                              << ",\"mode\":\"" << (mode ? "fast" : "generic")
                              << "\",\"ks\":" << ks << ",\"steps\":" << steps
                              << ",\"nk\":" << nk << ",\"nv\":" << nv
                              << ",\"alias\":" << (alias ? "true" : "false")
                              << ",\"dk\":" << D << ",\"dv\":" << D
                              << ",\"physical_slots\":" << slots
                              << ",\"active_slots\":[";
                    for (int row = 0; row < width; ++row)
                        std::cout << (row ? "," : "") << slot_order[row];
                    std::cout << "],\"seed\":\"6947ab21\",\"weights_hash\":\""
                              << std::hex << hash(weights) << "\",\"alog_hash\":\"" << hash(alog)
                              << "\",\"bias_hash\":\"" << hash(bias)
                              << "\",\"norm_hash\":\"" << hash(norm)
                              << "\",\"initial_conv_state_hash\":\"" << hash(conv_state.initial)
                              << "\",\"initial_recurrent_hash\":\"" << hash(recurrent.initial)
                              << std::dec << "\"}" << std::endl;
                    for (int step = 0; step < steps; ++step) {
                        for (auto &x : raw) x = sample(seed) * 0.75f;
                        for (auto &x : z) x = sample(seed) * 3.0f;
                        for (auto &x : b) x = sample(seed) * 5.0f;
                        for (auto &x : a) x = sample(seed) * 6.0f - 2.0f;
                        // Both tails of stable softplus and sigmoid on every
                        // step; slot-specific inputs detect crossed row pointers.
                        for (int slot = 0; slot < slots; ++slot) {
                            const size_t index = size_t(slot) * nv;
                            a[index] = ((slot + step) % 2) ? 30.0f : -30.0f;
                            b[index] = ((slot + step) % 2) ? -30.0f : 30.0f;
                        }
                        tq_h2d(draw, raw.data(), raw.size() * sizeof(float));
                        tq_h2d(dz, z.data(), z.size() * sizeof(float));
                        tq_h2d(db, b.data(), b.size() * sizeof(float));
                        tq_h2d(da, a.data(), a.size() * sizeof(float));
                        // Carry persistent states across calls, while refreshing
                        // outputs: active conv rows hold raw inputs when aliased;
                        // all holes and the core output retain NaN poisoning.
                        conv_out.reset();
                        core_out.reset();
                        if (alias) {
                            for (int row = 0; row < width; ++row) {
                                const size_t c = size_t(slot_order[row]) * conv_dim;
                                tq_h2d(conv_out.d_reference + c, raw.data() + c,
                                       size_t(conv_dim) * sizeof(float));
                                tq_h2d(conv_out.d_candidate + c, raw.data() + c,
                                       size_t(conv_dim) * sizeof(float));
                            }
                        }
                        tq_q().wait_and_throw();
                        for (int row = 0; row < width; ++row) {
                            const int slot = slot_order[(row + step) % width];
                            const size_t c = size_t(slot) * conv_dim;
                            const size_t v = size_t(slot) * value_dim;
                            const size_t h = size_t(slot) * nv;
                            x_linear_conv_update(conv_out.d_reference + c,
                                conv_state.d_reference + size_t(slot) * conv_state.stride,
                                (alias ? conv_out.d_reference : draw) + c,
                                dweights, conv_dim, ks);
                            x_linear_decode_core_gated(core_out.d_reference + v,
                                recurrent.d_reference + size_t(slot) * recurrent.stride,
                                conv_out.d_reference + c, dz + v, db + h, da + h,
                                dalog, dbias, dnorm, nk, D, nv, D, eps);
                        }
                        std::array<tq_linear_decode_row, TQ_LINEAR_DECODE_MAX_ROWS> rows{};
                        for (int row = 0; row < width; ++row) {
                            const int slot = slot_order[(row + step) % width];
                            const size_t c = size_t(slot) * conv_dim;
                            const size_t v = size_t(slot) * value_dim;
                            const size_t h = size_t(slot) * nv;
                            rows[row] = {
                                conv_out.d_candidate + c,
                                conv_state.d_candidate + size_t(slot) * conv_state.stride,
                                (alias ? conv_out.d_candidate : draw) + c,
                                core_out.d_candidate + v,
                                recurrent.d_candidate + size_t(slot) * recurrent.stride,
                                dz + v, db + h, da + h
                            };
                        }
                        if (x_linear_conv_update_batch(rows.data(), width, dweights,
                                                       conv_dim, ks) != 0)
                            throw std::runtime_error("x_linear_conv_update_batch rejected valid rows");
                        if (x_linear_decode_core_gated_batch(rows.data(), width, dalog,
                                dbias, dnorm, nk, D, nv, D, eps) != 0)
                            throw std::runtime_error("x_linear_decode_core_gated_batch rejected valid rows");
                        tq_q().wait_and_throw();
                        std::cout << "{\"kind\":\"check\",\"width\":" << width
                                  << ",\"mode\":\"" << (mode ? "fast" : "generic")
                                  << "\",\"ks\":" << ks << ",\"step\":" << step
                                  << ",\"alias\":" << (alias ? "true" : "false")
                                  << ",\"raw_hash\":\"" << std::hex << hash(raw)
                                  << "\",\"z_hash\":\"" << hash(z)
                                  << "\",\"b_hash\":\"" << hash(b)
                                  << "\",\"a_hash\":\"" << hash(a) << std::dec << "\"";
                        bool step_pass = true;
                        step_pass &= conv_out.check("conv_output", width);
                        step_pass &= conv_state.check("conv_state", width);
                        step_pass &= core_out.check("core_output", width);
                        step_pass &= recurrent.check("recurrent_state", width);
                        std::cout << ",\"pass\":" << (step_pass ? "true" : "false")
                                  << "}" << std::endl;
                        pass &= step_pass;
                    }
                }
            }
        }
        std::cout << "{\"kind\":\"summary\",\"pass\":" << (pass ? "true" : "false")
                  << "}" << std::endl;
        return pass ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << "gdn_batch_bench: " << e.what() << '\n';
        return 2;
    }
}
