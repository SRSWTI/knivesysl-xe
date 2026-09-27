// Linked real-kernel decode parity/state gate; no model weights or mocks.
// From the repository root, with LIB naming the supplied candidate .so:
// icpx -std=c++17 -fsycl -O2 -Ixpu/src xpu/probe/decode_attn_batch_check.cpp "$LIB" -Wl,-rpath,"$(dirname "$LIB")" -o xpu/build/decode_attn_batch_check
// CLI: decode_attn_batch_check [all|grouped|sharded|generic|dpas]
//                            [all|model|small] [all|CASE_NAME]
//      decode_attn_batch_check --list
// Examples: decode_attn_batch_check grouped model all
//           decode_attn_batch_check dpas model b8-long-mixed
//           decode_attn_batch_check generic small b8-boundary-mixed
// Mode selection is explicit (inherited attention selectors are overwritten):
// grouped: SIMD16=1, DPAS=0, GROUPED unset (production default)
// sharded: SIMD16=1, DPAS=0, GROUPED=0
// generic: SIMD16=0, DPAS=0, GROUPED unset
// dpas:    SIMD16=1, DPAS=1, GROUPED unset
// Prefix all selector names with TQ_XPU_ATTN_. Branch counters must confirm
// selection; a missing/ineligible requested model branch is not a silent PASS.
// "all all" runs all four modes on 24/4/256, then generic on 6/2/64.
// Comparisons are within one mode: DPAS is compared to serial DPAS, never to
// grouped math. Every arm starts with fresh independent identically seeded
// device allocations. No timing loops or tolerance-based acceptance.
#include "tq_common.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t guard = 64;
constexpr float eps = 1e-6f, theta = 10000000.0f, rotary = 0.25f;
using Counts = std::array<unsigned long long, 8>;

std::string quoted(const std::string &text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 32) {
            const char *hex = "0123456789abcdef";
            out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15];
        } else out += char(c);
    }
    return out + '"';
}

template<class T> bool exact(const std::vector<T> &a, const std::vector<T> &b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

template<class T> size_t different_bytes(const std::vector<T> &a,
                                          const std::vector<T> &b) {
    if (a.size() != b.size()) throw std::runtime_error("comparison size mismatch");
    const auto *ap = reinterpret_cast<const unsigned char *>(a.data());
    const auto *bp = reinterpret_cast<const unsigned char *>(b.data());
    size_t count = 0;
    for (size_t i = 0; i < a.size() * sizeof(T); ++i) count += ap[i] != bp[i];
    return count;
}

template<class T> std::vector<T> guarded(size_t count, T fill) {
    return std::vector<T>(count + 2 * guard, fill);
}

// Only [begin,end) is writable; includes prefix/suffix guards, unused physical
// pages, future positions, and all previously populated history in the check.
template<class T> bool unchanged_outside(const std::vector<T> &before,
                                         const std::vector<T> &after,
                                         size_t begin, size_t end) {
    return before.size() == after.size() && end <= before.size() &&
        std::memcmp(before.data(), after.data(), begin * sizeof(T)) == 0 &&
        std::memcmp(before.data() + end, after.data() + end,
                    (before.size() - end) * sizeof(T)) == 0;
}

struct Buffers {
    std::vector<void *> pointers;
    template<class T> T *upload(const std::vector<T> &values) {
        auto *p = static_cast<T *>(tq_dev_alloc(values.size() * sizeof(T),
                                                "decode attention parity"));
        if (!p) throw std::runtime_error("device allocation failed");
        pointers.push_back(p);
        tq_h2d(p, values.data(), values.size() * sizeof(T));
        return p;
    }
    ~Buffers() {
        // A failed wait is reported by the explicit wait in the arm, not by
        // throwing a second exception during stack unwinding.
        try { tq_q().wait_and_throw(); } catch (...) {}
        for (void *p : pointers) tq_dev_free(p);
    }
};

template<class T> std::vector<T> download(T *p, size_t count) {
    std::vector<T> values(count);
    tq_d2h(values.data(), p, count * sizeof(T));
    return values;
}

uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    return uint16_t(bits >> 16);
}
uint16_t fp16(float value) { return sycl::bit_cast<uint16_t>(sycl::half(value)); }

struct Shape { const char *name; int nh, nkv, hd; };
struct Case { std::string name; std::vector<int> positions; std::string layout; };

std::vector<Case> cases() {
    std::vector<Case> result{
        {"b1-zero-flat", {0}, "flat"},
        {"b1-short-p128", {63}, "p128"},
        {"b1-sharded-p256", {64}, "p256"},
        {"b1-grouped-flat", {512}, "flat"},
        {"b1-long-p128", {4096}, "p128"},
        {"b2-short-boundary-mixed", {63, 64}, "mixed"},
        {"b2-page128-boundary-mixed", {127, 128}, "mixed"},
        {"b2-page256-boundary-p256", {255, 256}, "p256"},
        {"b2-grouped-boundary-mixed", {511, 512}, "mixed"},
        {"b2-long-p256", {4095, 4096}, "p256"}
    };
    for (const std::string layout : {"flat", "p128", "p256", "mixed"}) {
        result.push_back({"b2-samebranch-" + layout, {512, 513}, layout});
        result.push_back({"b8-short-" + layout,
                          {0, 1, 15, 16, 31, 32, 62, 63}, layout});
        result.push_back({"b8-sharded-" + layout,
                          {64, 65, 127, 128, 255, 256, 510, 511}, layout});
        result.push_back({"b8-boundary-" + layout,
                          {0, 63, 64, 127, 128, 255, 511, 512}, layout});
        result.push_back({"b8-long-" + layout,
                          {512, 513, 767, 768, 1023, 1024, 2047, 2048}, layout});
    }
    return result;
}

int page_for(const Case &test, size_t row) {
    if (test.layout == "flat") return 0;
    if (test.layout == "p128") return 128;
    if (test.layout == "p256") return 256;
    // Both page sizes occur together in each B8 paged branch bucket, along
    // with flat descriptors; row identity is independent of descriptor order.
    constexpr int pages[] = {0, 128, 256, 128, 256, 0, 128, 256};
    return pages[row % 8];
}

struct Seed {
    int pos, page;
    size_t physical_write;
    std::vector<float> qg, k, v, out, scores;
    std::vector<uint8_t> kc, vc;
    std::vector<uint16_t> ks, vs;
    std::vector<int> table;
};

Seed make_seed(const Shape &shape, int pos, int page, int row, int sstride) {
    Seed seed;
    seed.pos = pos; seed.page = page;
    const size_t kv_width = size_t(shape.nkv) * shape.hd;
    const int span = page ? page : 256;
    const int blocks = (pos + 1 + span - 1) / span;
    const int physical_blocks = page ? blocks * 2 + 2 : blocks + 1;
    const size_t capacity = size_t(physical_blocks) * span;
    seed.table = guarded<int>(size_t(blocks), -1);
    for (int b = 0; b < blocks; ++b)
        seed.table[guard + b] = 2 * (blocks - 1 - b) + 1;
    const auto physical = [&](int token) -> size_t {
        return page ? size_t(seed.table[guard + token / page]) * page + token % page
                    : size_t(token);
    };
    seed.physical_write = physical(pos);
    seed.qg = guarded<float>(size_t(shape.nh) * 2 * shape.hd, -17.25f);
    seed.k = guarded<float>(kv_width, 11.5f);
    seed.v = guarded<float>(kv_width, -13.75f);
    seed.out = guarded<float>(size_t(shape.nh) * shape.hd,
                               std::numeric_limits<float>::quiet_NaN());
    seed.scores = guarded<float>(size_t(shape.nh) * sstride, -123.25f);
    seed.kc = guarded<uint8_t>(capacity * kv_width, 0x5a);
    seed.vc = guarded<uint8_t>(capacity * kv_width, 0xa5);
    seed.ks = guarded<uint16_t>(capacity * shape.nkv, 0x3555);
    seed.vs = guarded<uint16_t>(capacity * shape.nkv, 0x3222);
    std::mt19937 rng(391u + unsigned(row) * 104729u + unsigned(pos) * 17u);
    const auto random = [&]() { return float(int(rng() % 20001) - 10000) / 10000.0f; };
    for (size_t i = guard; i < seed.qg.size() - guard; ++i) seed.qg[i] = random();
    for (size_t i = guard; i < seed.k.size() - guard; ++i) {
        seed.k[i] = random(); seed.v[i] = random() * (1.0f + row * 0.125f);
    }
    // Row-dependent inputs and nonzero histories catch descriptor aliasing.
    // Zero projections in one row also exercise the zero-amax scale fallback.
    if (row == 3) {
        std::fill(seed.k.begin() + guard, seed.k.end() - guard, 0.0f);
        std::fill(seed.v.begin() + guard, seed.v.end() - guard, 0.0f);
    }
    for (int t = 0; t < pos; ++t) {
        const size_t token = physical(t);
        for (int head = 0; head < shape.nkv; ++head) {
            const size_t scale = guard + token * shape.nkv + head;
            seed.ks[scale] = fp16((t + head) % 31 == 0 ? 0.0f : 0.013f + 0.004f * (rng() % 9));
            seed.vs[scale] = fp16((t + head) % 29 == 0 ? 0.0f : 0.021f + 0.003f * (rng() % 11));
        }
        for (size_t d = 0; d < kv_width; ++d) {
            const size_t index = guard + token * kv_width + d;
            seed.kc[index] = uint8_t((rng() % 96) | ((rng() & 1) << 7));
            seed.vc[index] = uint8_t((rng() % 96) | ((rng() & 1) << 7));
        }
    }
    return seed;
}

struct DeviceRow {
    float *qg, *k, *v, *out, *scores;
    uint8_t *kc, *vc;
    uint16_t *ks, *vs;
    int *table;
    tq_decode_attn_request_t request;
};
struct Arm {
    Buffers buffers;
    std::vector<DeviceRow> rows;
    uint16_t *qn, *kn;
    Arm(const std::vector<Seed> &seeds, const std::vector<uint16_t> &qnorm,
        const std::vector<uint16_t> &knorm) {
        qn = buffers.upload(qnorm); kn = buffers.upload(knorm);
        for (const Seed &s : seeds) {
            DeviceRow d;
            d.qg = buffers.upload(s.qg); d.k = buffers.upload(s.k); d.v = buffers.upload(s.v);
            d.out = buffers.upload(s.out); d.scores = buffers.upload(s.scores);
            d.kc = buffers.upload(s.kc); d.vc = buffers.upload(s.vc);
            d.ks = buffers.upload(s.ks); d.vs = buffers.upload(s.vs);
            d.table = buffers.upload(s.table);
            tq_kv_layout_t layout{s.page ? d.table + guard : nullptr,
                                  s.page == 128 ? 7 : (s.page == 256 ? 8 : 0),
                                  s.page ? s.page - 1 : 0};
            d.request = {d.out + guard, d.qg + guard, d.k + guard, d.v + guard,
                         d.kc + guard, d.vc + guard, d.ks + guard, d.vs + guard,
                         d.scores + guard, s.pos, layout};
            rows.push_back(d);
        }
    }
};

Counts counters() {
    Counts result{};
    if (qwn_attn_branch_counts(result.data(), int(result.size())) != 0)
        throw std::runtime_error("branch counter API failed");
    return result;
}

Counts launch(Arm &arm, const Shape &shape, int sstride, int kind) {
    const Counts before = counters();
    if (kind == 0) {
        for (const auto &d : arm.rows) {
            const auto &r = d.request;
            x_full_attn_decode(r.out, r.qg, r.k, r.v, arm.qn + guard, arm.kn + guard,
                              r.kc, r.vc, r.ks, r.vs, r.scores, r.pos,
                              shape.nh, shape.nkv, shape.hd, eps, theta, rotary,
                              sstride, r.layout);
        }
    } else {
        std::array<tq_decode_attn_request_t, 8> descriptors{};
        for (size_t i = 0; i < arm.rows.size(); ++i)
            descriptors[i] = arm.rows[kind == 2 ? arm.rows.size() - 1 - i : i].request;
        x_full_attn_decode_batch(descriptors.data(), int(arm.rows.size()),
                                 arm.qn + guard, arm.kn + guard, shape.nh, shape.nkv,
                                 shape.hd, eps, theta, rotary, sstride);
        // Volatile stores prevent dead-store elimination of this lifetime test.
        // No wait precedes overwrite; all descriptors must already be captured.
        auto *bytes = reinterpret_cast<volatile unsigned char *>(descriptors.data());
        for (size_t i = 0; i < sizeof(descriptors); ++i) bytes[i] = 0;
    }
    tq_q().wait_and_throw();
    Counts after = counters();
    for (size_t i = 0; i < after.size(); ++i) after[i] -= before[i];
    return after;
}

struct Snapshot {
    std::vector<float> out, scores;
    std::vector<uint8_t> kc, vc;
    std::vector<uint16_t> ks, vs;
    bool finite = true, untouched = true, inputs = true;
};

Snapshot snapshot(const DeviceRow &d, const Seed &s, const Shape &shape) {
    Snapshot r;
    r.out = download(d.out, s.out.size()); r.scores = download(d.scores, s.scores.size());
    r.kc = download(d.kc, s.kc.size()); r.vc = download(d.vc, s.vc.size());
    r.ks = download(d.ks, s.ks.size()); r.vs = download(d.vs, s.vs.size());
    for (size_t i = guard; i < r.out.size() - guard; ++i)
        r.finite &= std::isfinite(r.out[i]);
    const size_t cb = guard + s.physical_write * shape.nkv * shape.hd;
    const size_t ce = cb + size_t(shape.nkv) * shape.hd;
    const size_t sb = guard + s.physical_write * shape.nkv;
    const size_t se = sb + shape.nkv;
    r.untouched = unchanged_outside(s.kc, r.kc, cb, ce) &&
        unchanged_outside(s.vc, r.vc, cb, ce) &&
        unchanged_outside(s.ks, r.ks, sb, se) &&
        unchanged_outside(s.vs, r.vs, sb, se) &&
        unchanged_outside(s.out, r.out, guard, r.out.size() - guard) &&
        unchanged_outside(s.scores, r.scores, guard, r.scores.size() - guard);
    r.inputs = exact(s.qg, download(d.qg, s.qg.size())) &&
        exact(s.k, download(d.k, s.k.size())) && exact(s.v, download(d.v, s.v.size())) &&
        exact(s.table, download(d.table, s.table.size()));
    return r;
}

Counts expected_counts(const Case &test, const Shape &shape, const std::string &mode) {
    Counts counts{};
    for (int pos : test.positions) {
        int branch = 4;
        if (shape.hd == 256 && mode != "generic") {
            if (pos < 64) branch = 1;
            else if (pos < 512 || mode == "sharded") branch = 2;
            else branch = mode == "dpas" ? 5 : 3;
        }
        ++counts[branch];
    }
    return counts;
}

void print_counts(const Counts &counts) {
    std::cout << '[';
    for (size_t i = 0; i < counts.size(); ++i) std::cout << (i ? "," : "") << counts[i];
    std::cout << ']';
}

void select_mode(const std::string &mode) {
    if (setenv("TQ_XPU_ATTN_SIMD16", mode == "generic" ? "0" : "1", 1) ||
        setenv("TQ_XPU_ATTN_DPAS", mode == "dpas" ? "1" : "0", 1) ||
        (mode == "sharded" ? setenv("TQ_XPU_ATTN_GROUPED", "0", 1)
                            : unsetenv("TQ_XPU_ATTN_GROUPED")))
        throw std::runtime_error("setting branch selector failed");
}

bool run_case(const Case &test, const Shape &shape, const std::string &mode) {
    const int maximum = *std::max_element(test.positions.begin(), test.positions.end());
    const int sstride = ((maximum + 1 + 255) / 256) * 256;
    std::vector<Seed> seeds;
    for (size_t i = 0; i < test.positions.size(); ++i)
        seeds.push_back(make_seed(shape, test.positions[i], page_for(test, i), int(i), sstride));
    auto qnorm = guarded<uint16_t>(shape.hd, 0x3f00);
    auto knorm = guarded<uint16_t>(shape.hd, 0xbf00);
    for (int i = 0; i < shape.hd; ++i) {
        qnorm[guard + i] = bf16(float(i % 17 - 8) / 64.0f);
        knorm[guard + i] = bf16(float(i % 13 - 6) / 48.0f);
    }
    // All three arms coexist: buffers cannot alias through allocator reuse.
    Arm serial(seeds, qnorm, knorm), batch(seeds, qnorm, knorm), reordered(seeds, qnorm, knorm);
    const Counts expected = expected_counts(test, shape, mode);
    const Counts sc = launch(serial, shape, sstride, 0);
    const Counts bc = launch(batch, shape, sstride, 1);
    const Counts rc = launch(reordered, shape, sstride, 2);
    bool finite = true, untouched = true, inputs = true;
    std::array<size_t, 6> batch_diff{}, reorder_diff{};
    int first_bad_row = -1;
    for (size_t i = 0; i < seeds.size(); ++i) {
        const Snapshot s = snapshot(serial.rows[i], seeds[i], shape);
        const Snapshot b = snapshot(batch.rows[i], seeds[i], shape);
        const Snapshot r = snapshot(reordered.rows[i], seeds[i], shape);
        const std::array<size_t, 6> bd{
            different_bytes(s.out, b.out), different_bytes(s.kc, b.kc),
            different_bytes(s.vc, b.vc), different_bytes(s.ks, b.ks),
            different_bytes(s.vs, b.vs), different_bytes(s.scores, b.scores)};
        const std::array<size_t, 6> rd{
            different_bytes(s.out, r.out), different_bytes(s.kc, r.kc),
            different_bytes(s.vc, r.vc), different_bytes(s.ks, r.ks),
            different_bytes(s.vs, r.vs), different_bytes(s.scores, r.scores)};
        bool row_ok = s.finite && b.finite && r.finite && s.untouched && b.untouched &&
                      r.untouched && s.inputs && b.inputs && r.inputs;
        for (size_t j = 0; j < bd.size(); ++j) {
            batch_diff[j] += bd[j]; reorder_diff[j] += rd[j]; row_ok &= bd[j] == 0 && rd[j] == 0;
        }
        if (!row_ok && first_bad_row < 0) first_bad_row = int(i);
        finite &= s.finite && b.finite && r.finite;
        untouched &= s.untouched && b.untouched && r.untouched;
        inputs &= s.inputs && b.inputs && r.inputs;
    }
    for (Arm *arm : {&serial, &batch, &reordered})
        inputs &= exact(qnorm, download(arm->qn, qnorm.size())) &&
                  exact(knorm, download(arm->kn, knorm.size()));
    const bool branches = sc == expected && bc == expected && rc == expected;
    const bool pass = first_bad_row < 0 && inputs && branches;
    std::cout << "{\"type\":\"case\",\"status\":" << quoted(pass ? "PASS" : "FAIL")
              << ",\"case\":" << quoted(test.name) << ",\"mode\":" << quoted(mode)
              << ",\"shape\":" << quoted(shape.name) << ",\"nh\":" << shape.nh
              << ",\"nkv\":" << shape.nkv << ",\"hd\":" << shape.hd
              << ",\"batch\":" << seeds.size() << ",\"positions\":[";
    for (size_t i = 0; i < seeds.size(); ++i) std::cout << (i ? "," : "") << seeds[i].pos;
    std::cout << "],\"pages\":[";
    for (size_t i = 0; i < seeds.size(); ++i) std::cout << (i ? "," : "") << seeds[i].page;
    std::cout << "],\"difference_fields\":[\"out\",\"kc\",\"vc\",\"ks\",\"vs\",\"scores\"]"
              << ",\"batch_difference_bytes\":[";
    for (size_t i = 0; i < batch_diff.size(); ++i) std::cout << (i ? "," : "") << batch_diff[i];
    std::cout << "],\"reordered_difference_bytes\":[";
    for (size_t i = 0; i < reorder_diff.size(); ++i) std::cout << (i ? "," : "") << reorder_diff[i];
    std::cout << "],\"finite\":" << (finite ? "true" : "false")
              << ",\"untouched_state_and_guards\":" << (untouched ? "true" : "false")
              << ",\"inputs_unchanged\":" << (inputs ? "true" : "false")
              << ",\"descriptor_overwrite\":true,\"first_bad_row\":" << first_bad_row
              << ",\"branch_counts_exact\":" << (branches ? "true" : "false")
              << ",\"expected_counts\":";
    print_counts(expected); std::cout << ",\"serial_counts\":"; print_counts(sc);
    std::cout << ",\"batch_counts\":"; print_counts(bc);
    std::cout << ",\"reordered_counts\":"; print_counts(rc);
    std::cout << "}" << std::endl;
    return pass;
}
} // namespace

int main(int argc, char **argv) {
    const auto matrix = cases();
    if (argc == 2 && std::string(argv[1]) == "--list") {
        std::cout << "{\"type\":\"case_list\",\"cases\":[";
        for (size_t i = 0; i < matrix.size(); ++i)
            std::cout << (i ? "," : "") << quoted(matrix[i].name);
        std::cout << "]}" << std::endl;
        return 0;
    }
    const std::string mode = argc > 1 ? argv[1] : "all";
    const std::string shape = argc > 2 ? argv[2] : "all";
    const std::string filter = argc > 3 ? argv[3] : "all";
    const bool known_case = filter == "all" || std::any_of(matrix.begin(), matrix.end(),
        [&](const Case &test) { return test.name == filter; });
    if (argc > 4 || (mode != "all" && mode != "grouped" && mode != "sharded" &&
                    mode != "generic" && mode != "dpas") ||
        (shape != "all" && shape != "model" && shape != "small") || !known_case) {
        std::cout << "{\"type\":\"summary\",\"status\":\"FAIL\",\"error\":\"usage: decode_attn_batch_check [all|grouped|sharded|generic|dpas] [all|model|small] [all|CASE_NAME]; --list lists cases\"}" << std::endl;
        return 2;
    }
    size_t passed = 0, failed = 0;
    try {
        for (const Shape spec : {Shape{"model", 24, 4, 256}, Shape{"small", 6, 2, 64}}) {
            if (shape != "all" && shape != spec.name) continue;
            for (const std::string selector : {"grouped", "sharded", "generic", "dpas"}) {
                if (mode != "all" && mode != selector) continue;
                // hd64 always selects generic: avoid four identical arms when
                // asking for all modes, while explicit modes remain inspectable.
                if (mode == "all" && spec.hd != 256 && selector != "generic") continue;
                select_mode(selector);
                for (const Case &test : matrix) {
                    if (filter != "all" && filter != test.name) continue;
                    if (run_case(test, spec, selector)) ++passed; else ++failed;
                }
            }
        }
    } catch (const std::exception &error) {
        std::cout << "{\"type\":\"summary\",\"status\":\"FAIL\",\"passed\":" << passed
                  << ",\"failed\":" << failed + 1 << ",\"error\":" << quoted(error.what())
                  << "}" << std::endl;
        return 1;
    }
    std::cout << "{\"type\":\"summary\",\"status\":" << quoted(failed == 0 && passed ? "PASS" : "FAIL")
              << ",\"passed\":" << passed << ",\"failed\":" << failed
              << ",\"comparison\":\"byte_exact\",\"branch_counter_order\":[\"prefill\",\"short\",\"sharded\",\"grouped\",\"generic\",\"dpas\",\"prefill_xmx\",\"prefill_scalar\"]}"
              << std::endl;
    return failed == 0 && passed ? 0 : 1;
}
