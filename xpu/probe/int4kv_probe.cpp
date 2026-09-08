// int4kv_probe.cpp — level-up-xpu queue item 6: fund or kill int4-K KV
// (thesis 4: depth is a KV-format problem; halving K bytes flattens the
// depth-retention curve) before touching the engine.
//
// Two candidate K-cache formats, same synthetic cache (N=131072 rows x
// hd=256, 32 MB as E4M3 — deliberately past L2 so the walk is DRAM-bound,
// with outlier dims every 37th channel so rotation actually matters):
//   A: E4M3 codes + fp32 row scale (the shipping tier's shape) — 1 B/elem.
//   B: Hadamard-rotated rows quantized to symmetric s4 — 0.5 B/elem.
//      Rotation is orthonormal, so dot(q, K) == dot(Hq, HK): the query is
//      rotated once per pass, scores need no unrotation.
// Measured:
//   1. append cost: device FWHT-256 + s4 quant per row (the new kernel).
//   2. walk rate: score[i] = dot(q, K_i) over the whole cache, 4 queries,
//      subgroup-16 per token row (the engine's decode-walk shape).
//   3. score accuracy vs fp32 reference: max-rel + cosine, A vs B.
// Kill criteria (doc §5 item 6): B's score error above the E4M3 band, or
// unpack cost eating the byte savings (walk speedup << bytes ratio).
//
// Build: icpx -fsycl -O2 -o int4kv_probe int4kv_probe.cpp
// Run:   ./int4kv_probe [device_index]

#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int kSG = 16;
static constexpr int kHd = 256;             // head dim
static constexpr int kN = 131072;           // cache rows (32 MB as E4M3)
static constexpr int kQ = 4;                // queries per pass

// Branch-free E4M3 decode (the engine's bit-trick, kernels_seq.cpp).
static inline float e4m3_to_float(uint8_t c) {
    const uint32_t b =
        ((uint32_t)(c & 0x7Fu) << 20) | ((uint32_t)(c & 0x80u) << 24);
    float f;
    std::memcpy(&f, &b, 4);
    return f * 0x1p120f;
}

// Engine-shaped walk (x_full_attn_decode): ONE query per pass, q hoisted
// into registers per lane, each subgroup strides serially over token rows.
struct walk_e4m3_k {
    const uint8_t *codes;   // [N][hd]
    const float *rscale;    // [N]
    const float *q1;        // [hd]
    float *scores;          // [N]
    int nsg;                // total subgroups in the grid
    [[sycl::reqd_sub_group_size(kSG)]] void
    operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int sgid = (int)(it.get_global_linear_id() / kSG);
        const int lane = (int)sg.get_local_linear_id();
        constexpr int kDpl = kHd / kSG;
        float qr[kDpl];
        for (int d = 0; d < kDpl; ++d) qr[d] = q1[lane * kDpl + d];
        for (int row = sgid; row < kN; row += nsg) {
            const uint8_t *kr = codes + (size_t)row * kHd + lane * kDpl;
            float dot = 0.0f;
            for (int d = 0; d < kDpl; ++d) {
                const uint32_t c = kr[d];
                const uint32_t b = ((c & 0x7Fu) << 20) | ((c & 0x80u) << 24);
                dot += sycl::bit_cast<float>(b) * 0x1p120f * qr[d];
            }
            const float s = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
            if (lane == 0) scores[row] = s * rscale[row];
        }
    }
};

struct walk_s4_k {
    const uint8_t *codes;   // [N][hd/2] packed nibbles (lo = even dim)
    const float *rscale;    // [N][8] group scales (lane's group = lane/2)
    const float *q1;        // [hd] (already Hadamard-rotated)
    float *scores;          // [N]
    int nsg;
    [[sycl::reqd_sub_group_size(kSG)]] void
    operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int sgid = (int)(it.get_global_linear_id() / kSG);
        const int lane = (int)sg.get_local_linear_id();
        constexpr int kDpl = kHd / kSG;
        float qr[kDpl];
        for (int d = 0; d < kDpl; ++d) qr[d] = q1[lane * kDpl + d];
        for (int row = sgid; row < kN; row += nsg) {
            const uint8_t *kr =
                codes + (size_t)row * (kHd / 2) + lane * (kDpl / 2);
            const float rs = rscale[(size_t)row * 8 + (lane >> 1)];
            float dot = 0.0f;
            for (int b = 0; b < kDpl / 2; ++b) {
                const uint32_t byte = kr[b];
                int lo = (int)(byte & 0xFu), hi = (int)(byte >> 4);
                if (lo >= 8) lo -= 16;
                if (hi >= 8) hi -= 16;
                dot += (float)lo * qr[2 * b] + (float)hi * qr[2 * b + 1];
            }
            const float s =
                sycl::reduce_over_group(sg, dot * rs, sycl::plus<float>());
            if (lane == 0) scores[row] = s;
        }
    }
};

// FWHT-256 + symmetric s4 quant with per-32-dim group scales (8/row),
// one 256-thread work-group per row.
struct fwht_quant_k {
    const float *rows;      // [N][hd] original
    uint8_t *codes;         // [N][hd/2]
    float *rscale;          // [N][8] group scales
    [[sycl::reqd_sub_group_size(kSG)]] void
    operator()(sycl::nd_item<1> it) const {
        const int row = (int)it.get_group_linear_id();
        const int t = (int)it.get_local_linear_id();
        auto &slm = *sycl::ext::oneapi::group_local_memory_for_overwrite<
            float[kHd]>(it.get_group());
        auto &sred = *sycl::ext::oneapi::group_local_memory_for_overwrite<
            float[kHd]>(it.get_group());
        slm[t] = rows[(size_t)row * kHd + t];
        it.barrier(sycl::access::fence_space::local_space);
        for (int h = 1; h < kHd; h <<= 1) {
            if (t < kHd / 2) {
                const int i2 = (t / h) * 2 * h + (t % h);
                const float x = slm[i2], y = slm[i2 + h];
                slm[i2] = x + y;
                slm[i2 + h] = x - y;
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        // normalize by 1/sqrt(256) = 1/16 and reduce absmax per 32-dim group
        const float v = slm[t] * (1.0f / 16.0f);
        sred[t] = sycl::fabs(v);
        it.barrier(sycl::access::fence_space::local_space);
        for (int s = 16; s > 0; s >>= 1) {
            if ((t & 31) < s) sred[t] = sycl::fmax(sred[t], sred[t + s]);
            it.barrier(sycl::access::fence_space::local_space);
        }
        const float gmax = sred[(t >> 5) << 5];
        const float scale = gmax > 0.0f ? gmax / 7.0f : 1.0f;
        slm[t] = v;   // store normalized value back for packing
        it.barrier(sycl::access::fence_space::local_space);
        if (t < kHd / 2) {
            // dims 2t, 2t+1 both live in group (2t)/32 == t/16
            const float gm2 = sred[(t >> 4) << 5];
            const float sc = gm2 > 0.0f ? gm2 / 7.0f : 1.0f;
            auto enc = [&](float x) {
                int c = (int)sycl::round(x / sc);
                c = sycl::clamp(c, -7, 7);
                return (uint32_t)(c & 0xF);
            };
            const uint32_t lo = enc(slm[2 * t]), hi = enc(slm[2 * t + 1]);
            codes[(size_t)row * (kHd / 2) + t] = (uint8_t)(lo | (hi << 4));
        }
        if ((t & 31) == 0) rscale[(size_t)row * 8 + (t >> 5)] = scale;
    }
};

template <typename K>
static double time_iters(sycl::queue &q, const K &kern, size_t global,
                         size_t local, int iters) {
    for (int i = 0; i < 3; ++i)
        q.parallel_for(sycl::nd_range<1>(global, local), kern);
    q.wait_and_throw();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        q.parallel_for(sycl::nd_range<1>(global, local), kern);
    q.wait_and_throw();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / iters;
}

// host FWHT for the rotated query + reference
static void fwht_host(float *v) {
    for (int h = 1; h < kHd; h <<= 1)
        for (int i = 0; i < kHd; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float x = v[j], y = v[j + h];
                v[j] = x + y;
                v[j + h] = x - y;
            }
    for (int i = 0; i < kHd; ++i) v[i] *= 1.0f / 16.0f;
}

static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

int main(int argc, char **argv) {
    const int devidx = argc > 1 ? std::atoi(argv[1]) : 0;
    std::vector<sycl::device> gpus;
    for (auto &d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero)
            gpus.push_back(d);
    if (devidx >= (int)gpus.size()) {
        std::fprintf(stderr, "device %d not found\n", devidx);
        return 1;
    }
    sycl::queue q(gpus[devidx], sycl::property::queue::in_order{});
    std::printf("== int4-K KV probe == device %d: %s\n", devidx,
                q.get_device().get_info<sycl::info::device::name>().c_str());

    // ---- synthetic K rows: normal-ish, outlier channels every 37th dim ----
    std::vector<float> hK((size_t)kN * kHd), hq((size_t)kQ * kHd);
    for (size_t i = 0; i < hK.size(); ++i) {
        const uint32_t r = mix((uint32_t)i * 2654435761u + 12345u);
        const float u = ((r & 0xFFFF) + ((r >> 16) & 0xFFFF)) / 65536.0f - 1.0f;
        const int d = (int)(i % kHd);
        hK[i] = u * ((d % 37) == 0 ? 4.0f : 1.0f);
    }
    for (size_t i = 0; i < hq.size(); ++i) {
        const uint32_t r = mix((uint32_t)i * 40503u + 99991u);
        hq[i] = ((r & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;
    }

    // ---- host E4M3 encode (nearest via sorted-magnitude binary search) ---
    float table[256];
    for (int c = 0; c < 256; ++c) table[c] = e4m3_to_float((uint8_t)c);
    // codes 0..126 decode to monotonically increasing magnitudes (IEEE-style
    // exp/mantissa lexicographic order); 0x7F is NaN and excluded.
    std::vector<uint8_t> hE((size_t)kN * kHd);
    std::vector<float> hEs(kN);
    for (int r = 0; r < kN; ++r) {
        float amax = 0.0f;
        for (int d = 0; d < kHd; ++d)
            amax = std::max(amax, std::fabs(hK[(size_t)r * kHd + d]));
        const float rs = amax > 0.0f ? amax / 448.0f : 1.0f;
        hEs[r] = rs;
        for (int d = 0; d < kHd; ++d) {
            const float tgt = hK[(size_t)r * kHd + d] / rs;
            const uint8_t sgn = tgt < 0.0f ? 0x80 : 0x00;
            const float at = std::fabs(tgt);
            const float *hi = std::upper_bound(table, table + 127, at);
            int best = (int)(hi - table);
            if (best >= 127) best = 126;
            else if (best > 0 && at - table[best - 1] <= table[best] - at)
                --best;
            hE[(size_t)r * kHd + d] = (uint8_t)(best | sgn);
        }
    }

    // ---- fp32 reference scores + rotated query on host ------------------
    std::vector<float> href((size_t)kQ * kN);
    for (int j = 0; j < kQ; ++j)
        for (int r = 0; r < kN; ++r) {
            double s = 0.0;
            for (int d = 0; d < kHd; ++d)
                s += (double)hq[(size_t)j * kHd + d] * hK[(size_t)r * kHd + d];
            href[(size_t)j * kN + r] = (float)s;
        }
    std::vector<float> hqrot(hq);
    for (int j = 0; j < kQ; ++j) fwht_host(hqrot.data() + (size_t)j * kHd);

    // ---- device buffers --------------------------------------------------
    float *dK = sycl::malloc_device<float>((size_t)kN * kHd, q);
    uint8_t *dE = sycl::malloc_device<uint8_t>((size_t)kN * kHd, q);
    float *dEs = sycl::malloc_device<float>(kN, q);
    uint8_t *dS4 = sycl::malloc_device<uint8_t>((size_t)kN * kHd / 2, q);
    float *dS4s = sycl::malloc_device<float>((size_t)kN * 8, q);
    float *dq = sycl::malloc_device<float>((size_t)kQ * kHd, q);
    float *dqr = sycl::malloc_device<float>((size_t)kQ * kHd, q);
    float *dsc = sycl::malloc_device<float>((size_t)kQ * kN, q);
    q.memcpy(dK, hK.data(), hK.size() * 4);
    q.memcpy(dE, hE.data(), hE.size());
    q.memcpy(dEs, hEs.data(), hEs.size() * 4);
    q.memcpy(dq, hq.data(), hq.size() * 4);
    q.memcpy(dqr, hqrot.data(), hqrot.size() * 4);
    q.wait_and_throw();

    // ---- 1. append cost: FWHT+quant per row ------------------------------
    const fwht_quant_k fq{dK, dS4, dS4s};
    const double fq_s =
        time_iters(q, fq, (size_t)kN * kHd, kHd, 5);
    std::printf("  fwht+s4 quant: %.1f us per 1k rows (%.2f GB/s in)\n",
                fq_s / kN * 1e6 * 1000.0, (size_t)kN * kHd * 4 / fq_s / 1e9);

    // ---- 2+3. walks + accuracy ------------------------------------------
    // Timing: one query per pass (the engine's decode-walk reality).
    // Accuracy: 4 passes with different queries, scores collected per pass.
    constexpr int kNsg = 8192;                 // 512 WGs x 16 subgroups
    const size_t glob = (size_t)kNsg * kSG;
    auto run_arm = [&](const char *name, auto make, size_t bytes) {
        // timing with query 0
        const double sec = time_iters(q, make(0), glob, 256, 20);
        // accuracy over all queries
        std::vector<float> got((size_t)kQ * kN);
        for (int j = 0; j < kQ; ++j) {
            q.parallel_for(sycl::nd_range<1>(glob, 256), make(j));
            q.wait_and_throw();
            q.memcpy(got.data() + (size_t)j * kN, dsc, (size_t)kN * 4).wait();
        }
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double a = href[i], b = got[i];
            dot += a * b; na += a * a; nb += b * b;
        }
        const double c = dot / (std::sqrt(na) * std::sqrt(nb));
        std::printf("  %-10s %8.3f ms/pass %8.1f GB/s  cos=%.6f\n", name,
                    sec * 1e3, bytes / sec / 1e9, c);
        return std::pair<double, double>(sec, c);
    };
    auto [te, ce] = run_arm(
        "e4m3 walk",
        [&](int j) { return walk_e4m3_k{dE, dEs, dq + (size_t)j * kHd, dsc, kNsg}; },
        (size_t)kN * kHd);
    auto [ts, cs] = run_arm(
        "s4+H walk",
        [&](int j) { return walk_s4_k{dS4, dS4s, dqr + (size_t)j * kHd, dsc, kNsg}; },
        (size_t)kN * (kHd / 2 + 8 * 4));

    std::printf("  walk speedup %.2fx (bytes ratio %.2fx, g32 fp32 scales); "
                "cos delta %.6f\n",
                te / ts, (double)kHd / (kHd / 2 + 8 * 4), ce - cs);
    std::printf("  verdict: %s\n",
                (cs >= 0.999 && te / ts >= 1.3) ? "FUND" : "KILL");

    sycl::free(dK, q); sycl::free(dE, q); sycl::free(dEs, q);
    sycl::free(dS4, q); sycl::free(dS4s, q); sycl::free(dq, q);
    sycl::free(dqr, q); sycl::free(dsc, q);
    return 0;
}
