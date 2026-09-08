// persistence_probe.cpp — level-up-xpu queue item 1: fund or kill thesis 1
// (persistent grid-resident decode step) before any inference integration.
//
// Questions answered, per xpu/docs/level-up-xpu.md §5:
//   1. Is a global (cross-work-group) barrier safe at exactly-resident grid
//      sizing on Xe2, and where is the empirical residency bound?
//   2. What does one global-barrier round cost, vs the 1.4 us kernel-launch
//      alternative it would replace?
//   3. Does a persistent 3-stage pipeline beat the same pipeline expressed as
//      3 kernel launches per iteration on the in-order queue?
//
// SAFETY: Xe2 gives no cross-work-group forward-progress guarantee. Every spin
// is bounded; on timeout the work-group sets a global abort flag and exits, and
// late-scheduled work-groups observe the flag and exit immediately, so an
// over-resident launch DRAINS instead of hanging (lesson 9: a wedged device on
// this box can require a user-visible reset). Kill criterion from the doc:
// barrier round > ~2 us at exactly-resident sizing, or any hang.
//
// Build: icpx -fsycl -O2 -o persistence_probe persistence_probe.cpp
// Run:   ./persistence_probe [l0_gpu_index=1]
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr int kSG = 16;
static constexpr int kStages = 3;   // dummy pipeline depth per iteration

struct Ctl {
  int count;        // barrier arrival counter
  int gen;          // barrier generation
  int abort_flag;   // any rep timed out -> drain everything
  int err;          // cross-WG ordering violations observed
  int launched;     // WGs that actually started (residency diagnostic)
};

using aref_int = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>;

// Sense-reversing centralized barrier, one representative per work-group.
// Returns false on bounded-spin timeout.
static inline bool global_barrier(Ctl *ctl, int nwg, uint32_t max_spin) {
  aref_int a_count(ctl->count);
  aref_int a_gen(ctl->gen);
  const int my_gen = a_gen.load(sycl::memory_order::acquire);
  if (a_count.fetch_add(1, sycl::memory_order::acq_rel) == nwg - 1) {
    a_count.store(0, sycl::memory_order::relaxed);
    a_gen.fetch_add(1, sycl::memory_order::release);
    return true;
  }
  uint32_t spins = 0;
  while (a_gen.load(sycl::memory_order::acquire) == my_gen) {
    if (++spins > max_spin) return false;
  }
  return true;
}

// Two-level (tree) barrier: WGs are partitioned into groups of `branch`;
// arrivals serialize on per-group cachelines in parallel, only group-last
// arrivers touch the root. Spin traffic is distributed across group gens.
struct TreeCtl {
  int root_count;
  int root_gen;
};

static inline bool tree_barrier(TreeCtl *root, int *grp, int wg, int nwg,
                                int branch, uint32_t max_spin) {
  const int g = wg / branch;
  const int ngroups = (nwg + branch - 1) / branch;
  const int gsize = (g == ngroups - 1) ? (nwg - g * branch) : branch;
  // grp layout: group g's {count, gen} live at grp[16*g], grp[16*g+1] —
  // 16-int stride keeps each group on its own 64-byte line.
  aref_int a_gcount(grp[16 * g]);
  aref_int a_ggen(grp[16 * g + 1]);
  const int my_ggen = a_ggen.load(sycl::memory_order::acquire);
  if (a_gcount.fetch_add(1, sycl::memory_order::acq_rel) == gsize - 1) {
    a_gcount.store(0, sycl::memory_order::relaxed);
    // group-last arrives at root
    aref_int a_rcount(root->root_count);
    aref_int a_rgen(root->root_gen);
    const int my_rgen = a_rgen.load(sycl::memory_order::acquire);
    if (a_rcount.fetch_add(1, sycl::memory_order::acq_rel) == ngroups - 1) {
      a_rcount.store(0, sycl::memory_order::relaxed);
      a_rgen.fetch_add(1, sycl::memory_order::release);
    } else {
      uint32_t spins = 0;
      while (a_rgen.load(sycl::memory_order::acquire) == my_rgen)
        if (++spins > max_spin) return false;
    }
    a_ggen.fetch_add(1, sycl::memory_order::release);
  } else {
    uint32_t spins = 0;
    while (a_ggen.load(sycl::memory_order::acquire) == my_ggen)
      if (++spins > max_spin) return false;
  }
  return true;
}

struct RunResult {
  bool pass = false;
  double seconds = 0.0;
  int launched = 0;
  int err = 0;
  int aborted = 0;
};

// One persistent run: nwg work-groups of wgs threads, iters iterations of a
// kStages-stage pipeline, one global barrier per stage. Stage work: rep writes
// a parity-double-buffered slot, post-barrier verifies the ring neighbour's
// slot — proving both the barrier and cross-WG store visibility.
// branch == 0 selects the centralized barrier; branch > 0 the tree barrier.
static RunResult run_persistent(sycl::queue &q, int nwg, int wgs, int iters,
                                uint32_t max_spin, int branch = 0) {
  RunResult r;
  const int ngroups = branch > 0 ? (nwg + branch - 1) / branch : 1;
  Ctl *ctl = sycl::malloc_device<Ctl>(1, q);
  TreeCtl *tctl = sycl::malloc_device<TreeCtl>(1, q);
  int *grp = sycl::malloc_device<int>(16 * (size_t)ngroups, q);
  int *slots = sycl::malloc_device<int>(2 * (size_t)nwg, q);
  q.memset(ctl, 0, sizeof(Ctl));
  q.memset(tctl, 0, sizeof(TreeCtl));
  q.memset(grp, 0, 16 * (size_t)ngroups * sizeof(int));
  q.memset(slots, 0xFF, 2 * (size_t)nwg * sizeof(int));
  q.wait();

  const auto t0 = std::chrono::steady_clock::now();
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<int, 1> lflag(1, h);
     h.parallel_for(
         sycl::nd_range<1>((size_t)nwg * wgs, (size_t)wgs),
         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
           const int wg = (int)it.get_group_linear_id();
           const int lid = (int)it.get_local_linear_id();

           // Uniform abort check so a late-scheduled WG drains instead of
           // joining a barrier that can never complete.
           if (lid == 0) {
             aref_int a_abort(ctl->abort_flag);
             lflag[0] = a_abort.load(sycl::memory_order::relaxed);
             if (lflag[0] == 0) aref_int(ctl->launched).fetch_add(1);
           }
           it.barrier(sycl::access::fence_space::local_space);
           if (lflag[0] != 0) return;

           for (int iter = 0; iter < iters; ++iter) {
             for (int s = 0; s < kStages; ++s) {
               const int tick = iter * kStages + s;
               const int parity = tick & 1;
               if (lid == 0) slots[parity * nwg + wg] = tick;
               it.barrier(sycl::access::fence_space::local_space);
               if (lid == 0) {
                 sycl::atomic_fence(sycl::memory_order::release,
                                    sycl::memory_scope::device);
                 const bool ok =
                     branch > 0
                         ? tree_barrier(tctl, grp, wg, nwg, branch, max_spin)
                         : global_barrier(ctl, nwg, max_spin);
                 sycl::atomic_fence(sycl::memory_order::acquire,
                                    sycl::memory_scope::device);
                 if (ok) {
                   const int neigh = (wg + 1 == nwg) ? 0 : wg + 1;
                   if (slots[parity * nwg + neigh] != tick)
                     aref_int(ctl->err).fetch_add(1);
                 } else {
                   aref_int(ctl->abort_flag).store(1);
                 }
                 lflag[0] = ok ? 1 : 0;
               }
               it.barrier(sycl::access::fence_space::local_space);
               if (lflag[0] == 0) return;
             }
           }
         });
   }).wait();
  const auto t1 = std::chrono::steady_clock::now();

  Ctl host_ctl{};
  q.memcpy(&host_ctl, ctl, sizeof(Ctl)).wait();
  r.seconds = std::chrono::duration<double>(t1 - t0).count();
  r.launched = host_ctl.launched;
  r.err = host_ctl.err;
  r.aborted = host_ctl.abort_flag;
  r.pass = (host_ctl.abort_flag == 0) && (host_ctl.err == 0);
  sycl::free(ctl, q);
  sycl::free(tctl, q);
  sycl::free(grp, q);
  sycl::free(slots, q);
  return r;
}

// Baseline A: bare submission cost — empty kernels on the in-order queue.
static double run_empty_launches(sycl::queue &q, int n, int wgs) {
  int *sink = sycl::malloc_device<int>(1, q);
  for (int i = 0; i < 100; ++i)   // warm
    q.parallel_for(sycl::nd_range<1>((size_t)wgs, (size_t)wgs),
                   [=](sycl::nd_item<1> it) {
                     if (it.get_global_linear_id() == 0) *sink = i;
                   });
  q.wait();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i)
    q.parallel_for(sycl::nd_range<1>((size_t)wgs, (size_t)wgs),
                   [=](sycl::nd_item<1> it) {
                     if (it.get_global_linear_id() == 0) *sink = i;
                   });
  q.wait();
  const auto t1 = std::chrono::steady_clock::now();
  sycl::free(sink, q);
  return std::chrono::duration<double>(t1 - t0).count() / n;
}

// Baseline B: the SAME 3-stage pipeline expressed the industry way — one
// kernel launch per stage, ordering by the in-order queue. Identical stage
// work and identical verification, so persistent-vs-launched is apples to
// apples.
static double run_launched_pipeline(sycl::queue &q, int nwg, int wgs,
                                    int iters, int *err_out) {
  int *slots = sycl::malloc_device<int>(2 * (size_t)nwg, q);
  int *err = sycl::malloc_device<int>(1, q);
  q.memset(slots, 0xFF, 2 * (size_t)nwg * sizeof(int));
  q.memset(err, 0, sizeof(int));
  q.wait();

  const auto t0 = std::chrono::steady_clock::now();
  for (int iter = 0; iter < iters; ++iter) {
    for (int s = 0; s < kStages; ++s) {
      const int tick = iter * kStages + s;
      const int parity = tick & 1;
      q.parallel_for(
          sycl::nd_range<1>((size_t)nwg * wgs, (size_t)wgs),
          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const int wg = (int)it.get_group_linear_id();
            if (it.get_local_linear_id() != 0) return;
            // check the PREVIOUS tick's neighbour slot, then write ours
            if (tick > 0) {
              const int prev_parity = (tick - 1) & 1;
              const int neigh = (wg + 1 == nwg) ? 0 : wg + 1;
              if (slots[prev_parity * nwg + neigh] != tick - 1)
                aref_int(*err).fetch_add(1);
            }
            slots[parity * nwg + wg] = tick;
          });
    }
  }
  q.wait();
  const auto t1 = std::chrono::steady_clock::now();
  q.memcpy(err_out, err, sizeof(int)).wait();
  sycl::free(slots, q);
  sycl::free(err, q);
  return std::chrono::duration<double>(t1 - t0).count();
}

int main(int argc, char **argv) {
  int want = (argc > 1) ? std::atoi(argv[1]) : 1;
  std::vector<sycl::device> gpus;
  for (auto &d : sycl::device::get_devices(sycl::info::device_type::gpu))
    if (d.get_backend() == sycl::backend::ext_oneapi_level_zero)
      gpus.push_back(d);
  if (gpus.empty()) {
    std::fprintf(stderr, "no level-zero GPUs\n");
    return 1;
  }
  if (want < 0 || want >= (int)gpus.size()) want = 0;
  sycl::queue q(gpus[want], sycl::property::queue::in_order{});
  const auto name = q.get_device().get_info<sycl::info::device::name>();

  int eus = 0, thr_per_eu = 0;
  try {
    eus = q.get_device()
              .get_info<sycl::ext::intel::info::device::gpu_eu_count>();
    thr_per_eu =
        q.get_device()
            .get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
  } catch (...) {
    eus = (int)q.get_device()
              .get_info<sycl::info::device::max_compute_units>();
    thr_per_eu = 8;
  }
  const int total_threads = eus * thr_per_eu;
  std::printf("== persistence probe ==\ndevice %d: %s | XVE %d x %d threads "
              "= %d hw threads\n",
              want, name.c_str(), eus, thr_per_eu, total_threads);

  // ---- residency sweep + barrier cost per work-group size --------------
  for (int wgs : {64, 128, 256}) {
    const int thr_per_wg = wgs / kSG;
    const int expected = total_threads / thr_per_wg;
    std::printf("\n-- wg_size %d (%d hw threads/WG, expected resident %d) --\n",
                wgs, thr_per_wg, expected);
    std::printf("  %6s %8s %10s %10s %8s %6s\n", "nwg", "status",
                "us/barrier", "barriers/s", "launched", "err");

    int best_pass = 0;
    std::vector<int> sweep = {expected / 4, expected / 2, expected * 3 / 4,
                              expected - 8, expected, expected + 1,
                              expected + 8};
    for (int nwg : sweep) {
      if (nwg < 2) continue;
      // short bounded run to test residency
      RunResult s = run_persistent(q, nwg, wgs, 64, 5u * 1000u * 1000u);
      if (s.pass) {
        // long run for a stable barrier cost
        const int iters = 5000;
        RunResult l = run_persistent(q, nwg, wgs, iters, 50u * 1000u * 1000u);
        const double us =
            l.seconds * 1e6 / ((double)iters * kStages);
        std::printf("  %6d %8s %10.3f %10.0f %8d %6d\n", nwg,
                    l.pass ? "PASS" : "FAIL", us,
                    (double)iters * kStages / l.seconds, l.launched, l.err);
        if (l.pass) best_pass = nwg;
      } else {
        std::printf("  %6d %8s %10s %10s %8d %6d   (drained cleanly)\n", nwg,
                    "BAIL", "-", "-", s.launched, s.err);
      }
    }
    std::printf("  empirical residency bound (largest PASS): %d\n", best_pass);

    // ---- baselines at the residency bound ------------------------------
    if (best_pass >= 2) {
      const int iters = 2000;
      int lerr = -1;
      const double launch_us = run_empty_launches(q, 5000, wgs) * 1e6;
      const double pipe_s =
          run_launched_pipeline(q, best_pass, wgs, iters, &lerr);
      const double pipe_us = pipe_s * 1e6 / ((double)iters * kStages);
      RunResult p = run_persistent(q, best_pass, wgs, iters,
                                   50u * 1000u * 1000u);
      const double pers_us = p.seconds * 1e6 / ((double)iters * kStages);
      std::printf("  baseline: empty launch %.3f us | launched pipeline "
                  "%.3f us/stage (err=%d) | persistent %.3f us/stage -> "
                  "%.2fx\n",
                  launch_us, pipe_us, lerr, pers_us, pipe_us / pers_us);
      // tree-barrier variants at the same residency bound
      for (int branch : {8, 16, 32}) {
        RunResult t = run_persistent(q, best_pass, wgs, iters,
                                     50u * 1000u * 1000u, branch);
        const double tree_us = t.seconds * 1e6 / ((double)iters * kStages);
        std::printf("  tree(branch=%2d): %8.3f us/stage %s err=%d -> "
                    "%.2fx vs launched pipeline\n",
                    branch, tree_us, t.pass ? "PASS" : "FAIL", t.err,
                    pipe_us / tree_us);
      }
    }
  }
  std::printf("\ndone\n");
  return 0;
}
