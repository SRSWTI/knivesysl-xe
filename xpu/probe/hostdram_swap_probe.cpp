// hostdram_swap_probe.cpp — level-up-xpu queue item 12: price the CPU-DRAM
// peripheral as a KV/checkpoint tier.
//
// Measures steady-state block swap rates device <-> imported host USM
// (`malloc_host`, the loader-proven fast path) at the sizes that matter:
//   ~4 MB   - one GDN layer state image (~3.1 MB rounded up)
//   ~33 MB  - 1k tokens of E4M3 KV across the 16 full-attention layers
//   ~134 MB - a 4k-token KV prefix (cold-KV parking unit)
//   ~512 MB - an APC checkpoint image (KV prefix + 48 GDN states)
// Both directions, idle; plus one contended cell: D2H copy racing a
// bandwidth-bound read kernel on a second in-order queue (the decode-overlap
// case), reporting both sides' degradation.
//
// Verdict math printed inline: each swap is converted to its re-prefill
// equivalent at the measured 468 tok/s prefill (a 4k prefix costs ~8.7 s to
// recompute vs ~milliseconds to restore). Kill criterion from the doc: swap
// latency > re-prefill cost — expected to fund by orders of magnitude, but
// measured > believed.
//
// Build: icpx -fsycl -O2 -o hostdram_swap_probe hostdram_swap_probe.cpp
// Run:   ./hostdram_swap_probe [device_index]

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

static double time_s(sycl::queue &q, int reps, const std::function<void()> &fn) {
    fn();
    q.wait();  // warm
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) fn();
    q.wait();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / reps;
}

struct read_k {
    const sycl::float4 *a;
    float *out;
    size_t n4;
    void operator()(sycl::nd_item<1> it) const {
        size_t i = it.get_global_linear_id();
        size_t stride = it.get_global_range(0);
        sycl::float4 acc = {0, 0, 0, 0};
        for (size_t j = i; j < n4; j += stride) acc += a[j];
        float s = acc.x() + acc.y() + acc.z() + acc.w();
        if (s == 1234567.0f) out[0] = s;
    }
};

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
    sycl::queue qc(q.get_context(), gpus[devidx],
                   sycl::property::queue::in_order{});  // copy queue
    std::printf("== host-DRAM swap probe == device %d: %s\n", devidx,
                q.get_device().get_info<sycl::info::device::name>().c_str());

    const size_t kMax = size_t(512) << 20;
    uint8_t *dbuf = sycl::malloc_device<uint8_t>(kMax, q);
    uint8_t *hbuf = static_cast<uint8_t *>(sycl::malloc_host(kMax, q));
    q.memset(dbuf, 1, kMax).wait();
    std::memset(hbuf, 2, kMax);

    // prefill equivalence: 468 tok/s measured flat chunked prefill,
    // ~33 KB/token of E4M3 KV across the 16 full-attn layers.
    const double prefill_tok_s = 468.0, kv_bytes_per_tok = 33.0 * 1024.0;

    std::printf("  %-8s %10s %10s | %14s %14s\n", "block", "D2H GB/s",
                "H2D GB/s", "restore(ms)", "reprefill(ms)");
    const size_t sizes[] = {size_t(4) << 20, size_t(33) << 20, size_t(134) << 20,
                            size_t(512) << 20};
    for (size_t sz : sizes) {
        const int reps = sz >= (size_t(134) << 20) ? 5 : 20;
        const double d2h =
            sz / time_s(q, reps, [&] { q.memcpy(hbuf, dbuf, sz); }) / 1e9;
        const double h2d =
            sz / time_s(q, reps, [&] { q.memcpy(dbuf, hbuf, sz); }) / 1e9;
        const double restore_ms = sz / (h2d * 1e9) * 1e3;
        const double tokens = sz / kv_bytes_per_tok;
        const double reprefill_ms = tokens / prefill_tok_s * 1e3;
        std::printf("  %5zuMB %10.1f %10.1f | %14.2f %14.1f\n", sz >> 20, d2h,
                    h2d, restore_ms, reprefill_ms);
    }

    // contended cell: 134 MB D2H racing a 1 GiB-working-set read kernel.
    {
        const size_t sz = size_t(134) << 20;
        const size_t n = (size_t(1) << 30) / sizeof(float);
        float *a = sycl::malloc_device<float>(n, q);
        float *sink = sycl::malloc_device<float>(1, q);
        q.fill(a, 1.0f, n).wait();
        const size_t g = 1024 * 256, l = 256;
        const read_k rk{(const sycl::float4 *)a, sink, n / 4};
        const double kern_alone =
            time_s(q, 5, [&] { q.parallel_for(sycl::nd_range<1>(g, l), rk); });
        const double copy_alone =
            time_s(qc, 5, [&] { qc.memcpy(hbuf, dbuf, sz); });
        // run both for a few reps concurrently; time each side's wall
        q.parallel_for(sycl::nd_range<1>(g, l), rk);
        qc.memcpy(hbuf, dbuf, sz);
        q.wait(); qc.wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 5; ++i) {
            q.parallel_for(sycl::nd_range<1>(g, l), rk);
            qc.memcpy(hbuf, dbuf, sz);
        }
        q.wait(); qc.wait();
        const auto t1 = std::chrono::steady_clock::now();
        const double both = std::chrono::duration<double>(t1 - t0).count() / 5;
        std::printf("  contended: read-kernel %.1f GB/s alone, copy %.1f GB/s "
                    "alone;\n             concurrent pair %.2f ms vs "
                    "max(alone)=%.2f ms (%.0f%% overhead)\n",
                    n * 4 / kern_alone / 1e9, sz / copy_alone / 1e9, both * 1e3,
                    std::max(kern_alone, copy_alone) * 1e3,
                    (both / std::max(kern_alone, copy_alone) - 1.0) * 100.0);
        sycl::free(a, q);
        sycl::free(sink, q);
    }

    sycl::free(dbuf, q);
    sycl::free(hbuf, q);
    std::printf("done\n");
    return 0;
}
