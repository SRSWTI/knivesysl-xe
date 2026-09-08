// tp_allreduce_probe.cpp - does TP=2 have a viable all-reduce on 2x Arc Pro B70?
//
// Phase 0 measured 2 GB/s STREAMING peer copy and wrote TP2 off. Wrong
// measurement. Tensor parallelism a la vLLM/SGLang shards
// ColumnParallelLinear by output rows (no comm) and RowParallelLinear by input
// columns (one all-reduce of the hidden vector). For Qwen3.8-27B that is
// 2 all-reduces x 64 layers = 128 per token, payload H=5120 floats = 20 KB.
// TP2 halves the 15.05 GB/token weight stream (~14 ms of the measured 33.1 ms
// step), so the verdict hinges on SMALL-PAYLOAD LATENCY.
//
// v2 measures the REAL shape. Partial sums live in DEVICE memory on each card
// (that is where a RowParallelLinear epilogue leaves them). An all-reduce is
// therefore: exchange the peer's partial into local device memory, then add.
// v1 wrongly ran the add in ONE work-group over host USM and measured 47 us of
// its own bad kernel.
//
//   floor    : device-memory add only, full grid (no comm at all)
//   p2p-1way : one 20 KB peer copy (latency of the interconnect)
//   p2p-2way : both directions issued before joining (PCIe is full duplex)
//   ar-p2p   : exchange + add, host joins the two in-order queues = the
//              barrier. Lesson 14: the kernel boundary beats software global
//              barriers on this silicon, so this is the shape to beat.
//   ar-host  : exchange staged through shared host USM instead of P2P, for
//              cards where P2P is crippled.
//
// Verdict on ar-*: per-token comm = 128 x per-allreduce.
//   < 25 us  -> FUND      (<= 3.2 ms/token vs ~14 ms saved)
//   > 50 us  -> KILL      (>= 6.4 ms/token eats the win)
//
// Build: icpx -fsycl -O2 -o tp_allreduce_probe tp_allreduce_probe.cpp
// Run:   ./tp_allreduce_probe [iters]
#include <sycl/sycl.hpp>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kH = 5120;                  // hidden size = all-reduce payload
constexpr size_t kBytes = (size_t)kH * sizeof(float);
constexpr int kSG = 16;
constexpr int kWG = 256;
constexpr int kAllReducesPerToken = 128;  // 64 layers x 2 row-parallel

using clk = std::chrono::steady_clock;

double us_since(clk::time_point t0) {
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

// out = a + b over kH floats, spread across the machine (not one work-group).
void add_vec(sycl::queue &q, float *out, const float *a, const float *b) {
    const size_t wgs = (kH + kWG - 1) / kWG;
    q.parallel_for(sycl::nd_range<1>(wgs * kWG, kWG),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
        const size_t i = it.get_global_linear_id();
        if (i < (size_t)kH) out[i] = a[i] + b[i];
    });
}

void report(const char *name, double us, bool is_ar) {
    if (!is_ar) {
        std::printf("  %-10s %9.2f us   %7.2f GB/s\n", name, us,
                    (double)kBytes / us / 1000.0);
        return;
    }
    const double per_token = us * kAllReducesPerToken / 1000.0;
    const char *v = us < 25.0 ? "FUND" : (us > 50.0 ? "KILL" : "MARGINAL");
    std::printf("  %-10s %9.2f us   %7.2f ms/token (x%d)   %s\n", name, us,
                per_token, kAllReducesPerToken, v);
}

}  // namespace

int main(int argc, char **argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 2000;
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    std::vector<sycl::device> gpus;
    for (auto &d : devs)
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero)
            gpus.push_back(d);
    std::printf("== TP=2 all-reduce probe (v2) ==  payload %d floats (%zu KB), "
                "iters %d\n", kH, kBytes / 1024, iters);
    for (size_t i = 0; i < gpus.size(); ++i)
        std::printf("  device %zu: %s\n", i,
                    gpus[i].get_info<sycl::info::device::name>().c_str());
    if (gpus.size() < 2) {
        std::printf("need 2 Level Zero GPUs; found %zu\n", gpus.size());
        return 1;
    }

    // ONE context spanning both devices: required for cross-device pointers
    // and for a host allocation addressable from both cards.
    sycl::context ctx({gpus[0], gpus[1]});
    sycl::queue q0(ctx, gpus[0], sycl::property::queue::in_order{});
    sycl::queue q1(ctx, gpus[1], sycl::property::queue::in_order{});

    // Per-card: own partial, a landing slot for the peer's partial, output.
    float *p0 = sycl::malloc_device<float>(kH, gpus[0], ctx);
    float *r0 = sycl::malloc_device<float>(kH, gpus[0], ctx);
    float *o0 = sycl::malloc_device<float>(kH, gpus[0], ctx);
    float *p1 = sycl::malloc_device<float>(kH, gpus[1], ctx);
    float *r1 = sycl::malloc_device<float>(kH, gpus[1], ctx);
    float *o1 = sycl::malloc_device<float>(kH, gpus[1], ctx);
    float *h0 = sycl::malloc_host<float>(kH, ctx);   // host staging
    float *h1 = sycl::malloc_host<float>(kH, ctx);
    float *chk = sycl::malloc_host<float>(kH, ctx);
    if (!p0 || !r0 || !o0 || !p1 || !r1 || !o1 || !h0 || !h1 || !chk) {
        std::printf("allocation failed\n");
        return 1;
    }

    // Deterministic partials so the reduce is checkable: p0[i]=i, p1[i]=2i.
    for (int i = 0; i < kH; ++i) { h0[i] = (float)i; h1[i] = 2.0f * (float)i; }
    q0.memcpy(p0, h0, kBytes).wait();
    q1.memcpy(p1, h1, kBytes).wait();
    std::printf("\n");

    // ---- floor: device-memory add, full grid, no communication ----------
    {
        for (int i = 0; i < 100; ++i) add_vec(q0, o0, p0, p0);
        q0.wait();
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) add_vec(q0, o0, p0, p0);
        q0.wait();
        report("floor", us_since(t0) / iters, false);
    }

    // ---- p2p one direction ----------------------------------------------
    {
        for (int i = 0; i < 50; ++i) q0.memcpy(r1, p0, kBytes);
        q0.wait();
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) q0.memcpy(r1, p0, kBytes);
        q0.wait();
        report("p2p-1way", us_since(t0) / iters, false);
    }

    // ---- p2p both directions, issued before joining ---------------------
    {
        for (int i = 0; i < 50; ++i) { q0.memcpy(r1, p0, kBytes); q1.memcpy(r0, p1, kBytes); }
        q0.wait(); q1.wait();
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) {
            q0.memcpy(r1, p0, kBytes);
            q1.memcpy(r0, p1, kBytes);
            q0.wait(); q1.wait();
        }
        report("p2p-2way", us_since(t0) / iters, false);
    }

    // ---- all-reduce via P2P exchange + device add ------------------------
    // Exactly what a RowParallelLinear epilogue needs: after this both cards
    // hold the full sum in device memory.
    {
        auto round = [&] {
            q0.memcpy(r1, p0, kBytes);        // my partial -> peer's slot
            q1.memcpy(r0, p1, kBytes);
            q0.wait(); q1.wait();             // the barrier (kernel boundary)
            add_vec(q0, o0, p0, r0);
            add_vec(q1, o1, p1, r1);
            q0.wait(); q1.wait();
        };
        for (int i = 0; i < 50; ++i) round();
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) round();
        report("ar-p2p", us_since(t0) / iters, true);
        q0.memcpy(chk, o0, kBytes).wait();
        bool ok = true;
        for (int i = 0; i < kH; ++i)
            if (chk[i] != (float)i + 2.0f * (float)i) { ok = false; break; }
        std::printf("             correctness %s (out = p0 + p1)\n",
                    ok ? "PASS" : "FAIL");
    }

    // ---- the specialised paths vLLM ships on CUDA but not on XPU ---------
    // vLLM's five CUDA all-reduce backends reduce to three ideas that apply
    // here: a custom peer kernel instead of the vendor collective (that is
    // `ar-p2p` above, already ours), a QUANTIZED payload
    // (quick_all_reduce), and FUSING the reduction into the consumer
    // (flashinfer_all_reduce, whose consumer is always an RMSNorm - exactly
    // our situation). Plus the plain question of whether the exchange can be
    // made duplex, which `p2p-2way` says it currently is not.

    // bf16 payload: halve the bytes on the wire. The reduced value is a
    // residual stream, and this engine already ships 4-bit weights and E4M3
    // KV, so bf16 on one reduction is well inside its error budget.
    uint16_t *b0 = sycl::malloc_device<uint16_t>(kH, gpus[0], ctx);
    uint16_t *b1 = sycl::malloc_device<uint16_t>(kH, gpus[1], ctx);
    uint16_t *c0 = sycl::malloc_device<uint16_t>(kH, gpus[0], ctx);
    uint16_t *c1 = sycl::malloc_device<uint16_t>(kH, gpus[1], ctx);
    if (b0 && b1 && c0 && c1) {
        auto pack = [&](sycl::queue &q, uint16_t *dst, const float *src) {
            const size_t wgs = (kH + kWG - 1) / kWG;
            q.parallel_for(sycl::nd_range<1>(wgs * kWG, kWG),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
                const size_t i = it.get_global_linear_id();
                if (i < (size_t)kH)
                    dst[i] = (uint16_t)(sycl::bit_cast<uint32_t>(src[i]) >> 16);
            });
        };
        auto unpack_add = [&](sycl::queue &q, float *out, const float *mine,
                              const uint16_t *peer) {
            const size_t wgs = (kH + kWG - 1) / kWG;
            q.parallel_for(sycl::nd_range<1>(wgs * kWG, kWG),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
                const size_t i = it.get_global_linear_id();
                if (i < (size_t)kH)
                    out[i] = mine[i] + sycl::bit_cast<float>((uint32_t)peer[i] << 16);
            });
        };
        auto round = [&] {
            pack(q0, b0, p0);
            pack(q1, b1, p1);
            q0.wait(); q1.wait();
            q0.memcpy(c1, b0, kBytes / 2);
            q1.memcpy(c0, b1, kBytes / 2);
            q0.wait(); q1.wait();
            unpack_add(q0, o0, p0, c0);
            unpack_add(q1, o1, p1, c1);
            q0.wait(); q1.wait();
        };
        for (int i = 0; i < 50; ++i) round();
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) round();
        report("ar-bf16", us_since(t0) / iters, true);
        q0.memcpy(chk, o0, kBytes).wait();
        double worst = 0.0;
        for (int i = 0; i < kH; ++i) {
            const double want = (double)i + 2.0 * (double)i;
            const double rel = want != 0.0 ? std::fabs(chk[i] - want) / want : 0.0;
            if (rel > worst) worst = rel;
        }
        std::printf("             max_rel %.3g (bf16 wire, fp32 accumulate)\n",
                    worst);
    }

    // Duplex attempt: issue BOTH directions before either join, and give the
    // copies their own queues so the runtime is free to overlap them.
    {
        sycl::queue x0(ctx, gpus[0], sycl::property::queue::in_order{});
        sycl::queue x1(ctx, gpus[1], sycl::property::queue::in_order{});
        auto round = [&] {
            auto e0 = x0.memcpy(r1, p0, kBytes);
            auto e1 = x1.memcpy(r0, p1, kBytes);
            e0.wait(); e1.wait();
        };
        for (int i = 0; i < 50; ++i) round();
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) round();
        report("p2p-duplex", us_since(t0) / iters, false);
    }

    // Fused: skip the separate add kernel entirely by having the CONSUMER
    // read the peer slab. Modelled here as one kernel that reads both and
    // writes the reduced result, i.e. what a fused allreduce+RMSNorm would
    // absorb for free. Measures the exchange plus one kernel instead of two.
    {
        auto round = [&] {
            q0.memcpy(r1, p0, kBytes);
            q1.memcpy(r0, p1, kBytes);
            q0.wait(); q1.wait();
            add_vec(q0, o0, p0, r0);          // stands in for the fused norm
            q1.wait();
        };
        for (int i = 0; i < 50; ++i) round();
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) round();
        report("ar-fused1", us_since(t0) / iters, true);
    }

    // ---- all-reduce staged through shared host USM -----------------------
    {
        auto round = [&] {
            q0.memcpy(h0, p0, kBytes);        // device -> host
            q1.memcpy(h1, p1, kBytes);
            q0.wait(); q1.wait();
            q0.memcpy(r0, h1, kBytes);        // host -> peer device
            q1.memcpy(r1, h0, kBytes);
            q0.wait(); q1.wait();
            add_vec(q0, o0, p0, r0);
            add_vec(q1, o1, p1, r1);
            q0.wait(); q1.wait();
        };
        for (int i = 0; i < 50; ++i) round();
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) round();
        report("ar-host", us_since(t0) / iters, true);
        q0.memcpy(chk, o0, kBytes).wait();
        bool ok = true;
        for (int i = 0; i < kH; ++i)
            if (chk[i] != (float)i + 2.0f * (float)i) { ok = false; break; }
        std::printf("             correctness %s (out = p0 + p1)\n",
                    ok ? "PASS" : "FAIL");
    }

    std::printf("\n  Reference: TP2 halves the 15.05 GB/token weight stream,\n"
                "  worth ~14 ms/token of the measured 33.1 ms step. Comm must\n"
                "  stay well under that to win.\n");

    for (void *p : {(void *)p0, (void *)r0, (void *)o0, (void *)p1,
                    (void *)r1, (void *)o1, (void *)h0, (void *)h1,
                    (void *)chk})
        sycl::free(p, ctx);
    std::printf("done\n");
    return 0;
}
