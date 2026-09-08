// doorbell_probe.cpp — level-up-xpu queue item 2: price the device-driven
// decode loop (the surviving form of thesis 1 after persistence_probe killed
// the megakernel).
//
// Question: how fast can resident/parked device work consume a host-written
// token and publish a result back — zero per-step submissions, zero event
// syncs — vs today's submit + wait shape?
//
// Part B (pure Level Zero, RUN FIRST — it is the primitive the engine would
//   actually use): zexCommandListAppendWaitOnMemory / WriteToMemory ping-pong
//   on an async immediate list, appended in a rolling window. Parks the CCS,
//   occupies zero XVEs.
// Part A (SYCL diagnostic ladder): ONE persistent kernel loops with a bounded
//   spin on host-USM. Arms: narrow/load-spin, narrow/RMW-spin (an atomic RMW
//   forces a coherent transaction where a plain load may be served stale),
//   then wide only if narrow passed. Heartbeat counters expose device-side
//   spin progress so a failure names its mechanism instead of hanging.
// Baselines: submit + q.wait() per step (today's shape), submit + host-poll.
//
// POSTMORTEM LOG (append-only doctrine):
//   v1: pre-enqueued 2000 wait-kernels; hung 600 s. (1) spin bound calibrated
//       for register spins but host-USM polls are PCIe round trips; (2) work
//       that waits on FUTURE host input + a blocking append = deadlock.
//   v2: chain replaced by one looping kernel, but the abort path returned
//       lane 0 only -> divergent WG barrier -> hang (violated the uniform-exit
//       rule persistence_probe established).
//   v3: uniform abort + READY handshake. READY reached the host (device->host
//       writes work) but the device spin never observed the host's LATER
//       doorbell write (narrow erred at step 0), and the wide arm spent the
//       whole timeout in its 128-spinner abort path. Hence v4: zex first,
//       RMW arm, heartbeats, wide gated on narrow.
//
// SAFETY: every spin bounded (device ~2 s equivalent, host budgets explicit);
// zex waits use GREATER_THAN_EQUAL so one large host write drains everything;
// abort decisions are WG-uniform; stdout unbuffered.
//
// Build: icpx -fsycl -O2 -std=c++20 -o doorbell_probe doorbell_probe.cpp -lze_loader
// Run:   ./doorbell_probe [l0_gpu_index=1]
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---- minimal zex declarations (vendor/compute-runtime zex_cmdlist.h) ------
extern "C" {
typedef uint32_t zex_mem_action_scope_flags_t;
#define ZEX_MEM_ACTION_SCOPE_FLAG_HOST (1u << 2)
typedef uint32_t zex_wait_on_mem_action_flags_t;
#define ZEX_WAIT_ON_MEMORY_FLAG_GREATER_THAN_EQUAL (1u << 3)
typedef struct {
  zex_wait_on_mem_action_flags_t actionFlag;
  zex_mem_action_scope_flags_t waitScope;
} zex_wait_on_mem_desc_t;
typedef struct {
  zex_mem_action_scope_flags_t writeScope;
} zex_write_to_mem_desc_t;
typedef ze_result_t (*pfn_zexWaitOnMem)(ze_command_list_handle_t,
                                        zex_wait_on_mem_desc_t *, void *,
                                        uint32_t, ze_event_handle_t);
typedef ze_result_t (*pfn_zexWriteToMem)(ze_command_list_handle_t,
                                         zex_write_to_mem_desc_t *, void *,
                                         uint64_t);
}

static constexpr int kSG = 16;
static constexpr int kReady = 1000000000;

struct Stats {
  double p50_us = 0, p99_us = 0, mean_us = 0;
  int completed = 0, errors = 0;
};

static Stats summarize(std::vector<double> &lat_us, int errors) {
  Stats s;
  s.completed = (int)lat_us.size();
  s.errors = errors;
  if (lat_us.empty()) return s;
  std::sort(lat_us.begin(), lat_us.end());
  double sum = 0;
  for (double v : lat_us) sum += v;
  s.mean_us = sum / lat_us.size();
  s.p50_us = lat_us[lat_us.size() / 2];
  s.p99_us = lat_us[(size_t)((double)(lat_us.size() - 1) * 0.99)];
  return s;
}

static inline int host_load(volatile int *p) {
  return __atomic_load_n((int *)p, __ATOMIC_ACQUIRE);
}
static inline void host_store(volatile int *p, int v) {
  __atomic_store_n((int *)p, v, __ATOMIC_RELEASE);
}
static inline int host_poll_ms(volatile int *p, int want, int budget_ms) {
  const auto limit = std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(budget_ms);
  int r;
  while ((r = host_load(p)) < want && r != -1)
    if (std::chrono::steady_clock::now() > limit) break;
  return r;
}

// ---- Part B: pure Level Zero zex doorbell ping-pong ------------------------
#define ZECHK(call)                                                          \
  do {                                                                       \
    ze_result_t _r = (call);                                                 \
    if (_r != ZE_RESULT_SUCCESS) {                                           \
      std::fprintf(stderr, "  B: L0 error 0x%x at line %d\n", _r, __LINE__); \
      return;                                                                \
    }                                                                        \
  } while (0)

static void part_b_zex(int gpu_index, int steps) {
  ZECHK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t nd = 0;
  ZECHK(zeDriverGet(&nd, nullptr));
  std::vector<ze_driver_handle_t> drivers(nd);
  ZECHK(zeDriverGet(&nd, drivers.data()));
  ze_driver_handle_t drv = nullptr;
  ze_device_handle_t dev = nullptr;
  int seen = 0;
  for (auto d : drivers) {
    uint32_t ndev = 0;
    if (zeDeviceGet(d, &ndev, nullptr) != ZE_RESULT_SUCCESS) continue;
    std::vector<ze_device_handle_t> devs(ndev);
    zeDeviceGet(d, &ndev, devs.data());
    for (auto dd : devs) {
      ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      zeDeviceGetProperties(dd, &p);
      if (p.type != ZE_DEVICE_TYPE_GPU) continue;
      if (seen++ == gpu_index) { drv = d; dev = dd; }
    }
  }
  if (!dev) {
    std::fprintf(stderr, "  B: gpu index %d not found (%d gpus)\n", gpu_index,
                 seen);
    return;
  }

  pfn_zexWaitOnMem zexWait = nullptr;
  pfn_zexWriteToMem zexWrite = nullptr;
  ZECHK(zeDriverGetExtensionFunctionAddress(
      drv, "zexCommandListAppendWaitOnMemory", (void **)&zexWait));
  ZECHK(zeDriverGetExtensionFunctionAddress(
      drv, "zexCommandListAppendWriteToMemory", (void **)&zexWrite));

  ze_context_handle_t ctx = nullptr;
  ze_context_desc_t cdesc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ZECHK(zeContextCreate(drv, &cdesc, &ctx));

  ze_command_queue_desc_t qdesc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qdesc.ordinal = 0;
  qdesc.index = 0;
  qdesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  qdesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  ze_command_list_handle_t cl = nullptr;
  ZECHK(zeCommandListCreateImmediate(ctx, dev, &qdesc, &cl));

  ze_host_mem_alloc_desc_t hdesc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  void *db = nullptr, *res = nullptr;
  ZECHK(zeMemAllocHost(ctx, &hdesc, 256, 64, &db));
  ZECHK(zeMemAllocHost(ctx, &hdesc, 256, 64, &res));
  std::memset(db, 0, 256);
  std::memset(res, 0, 256);
  volatile int *vdb = (volatile int *)db;
  volatile int *vres = (volatile int *)res;

  zex_wait_on_mem_desc_t wdesc{ZEX_WAIT_ON_MEMORY_FLAG_GREATER_THAN_EQUAL,
                               ZEX_MEM_ACTION_SCOPE_FLAG_HOST};
  zex_write_to_mem_desc_t wr{ZEX_MEM_ACTION_SCOPE_FLAG_HOST};

  // Rolling window (v1 postmortem: unbounded lookahead + blocking append =
  // deadlock).
  constexpr int kWindow = 32;
  int appended = 0;
  auto top_up = [&](int upto) -> bool {
    while (appended < upto && appended < steps) {
      if (zexWait(cl, &wdesc, db, (uint32_t)(appended + 1), nullptr) !=
          ZE_RESULT_SUCCESS)
        return false;
      if (zexWrite(cl, &wr, res, (uint64_t)(appended + 1)) !=
          ZE_RESULT_SUCCESS)
        return false;
      ++appended;
    }
    return true;
  };
  if (!top_up(kWindow)) {
    std::fprintf(stderr, "  B: initial append failed\n");
    return;
  }

  std::vector<double> lat;
  lat.reserve(steps);
  int errors = 0;
  for (int k = 0; k < steps; ++k) {
    const auto t0 = std::chrono::steady_clock::now();
    host_store(vdb, k + 1);
    const int r = host_poll_ms(vres, k + 1, 2000);
    const auto t1 = std::chrono::steady_clock::now();
    if (r < k + 1) {
      ++errors;
      std::printf("  B: step %d failed (res=%d db=%d)\n", k, r,
                  host_load(vdb));
      break;
    }
    if (k >= 100)
      lat.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    if (!top_up(k + 1 + kWindow)) { ++errors; break; }
  }
  host_store(vdb, steps + 1);   // drain every parked wait
  zeCommandListHostSynchronize(cl, UINT64_MAX);
  Stats s = summarize(lat, errors);
  std::printf("  B.zex ping   : p50 %7.3f  p99 %7.3f  mean %7.3f us  "
              "(%d steps, err=%d)\n",
              s.p50_us, s.p99_us, s.mean_us, s.completed, s.errors);

  zeMemFree(ctx, db);
  zeMemFree(ctx, res);
  zeCommandListDestroy(cl);
  zeContextDestroy(ctx);
}

// ---- Part A: persistent kernel diagnostic ladder ---------------------------
// rmw: spin via fetch_or(0) (a real coherent transaction) instead of load.
// Returns true when the timed loop completed without errors.
static bool part_a_persistent(sycl::queue &q, int steps, bool wide, bool rmw) {
  int *db = sycl::malloc_host<int>(16, q);
  int *res = sycl::malloc_host<int>(16, q);   // [0]=result [8]=heartbeat
  int *work = sycl::malloc_device<int>(4096, q);
  std::memset((void *)db, 0, 64);
  std::memset((void *)res, 0, 64);

  const int nwg = wide ? 128 : 1;
  const int wgs = wide ? 256 : kSG;
  // ~2 s if polls are PCIe (~0.5 us); ~10 ms if they are stale cache hits —
  // the heartbeat count distinguishes the two.
  constexpr uint32_t kDevSpin = 4u * 1000u * 1000u;

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<int, 1> lok(1, h);
    h.parallel_for(
        sycl::nd_range<1>((size_t)nwg * wgs, (size_t)wgs),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::system,
                           sycl::access::address_space::global_space>
              a_db(db[0]);
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::system,
                           sycl::access::address_space::global_space>
              a_res(res[0]);
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::system,
                           sycl::access::address_space::global_space>
              a_hb(res[8]);
          const int lid = (int)it.get_local_linear_id();
          const int wg = (int)it.get_group_linear_id();
          const int gid = (int)it.get_global_linear_id();
          if (wg == 0 && lid == 0)
            a_res.store(kReady, sycl::memory_order::release);
          for (int k = 0; k < steps; ++k) {
            if (lid == 0) {
              uint32_t spins = 0;
              int ok = 1;
              for (;;) {
                const int seen =
                    rmw ? a_db.fetch_or(0, sycl::memory_order::acq_rel)
                        : a_db.load(sycl::memory_order::acquire);
                if (seen >= k + 1) break;
                ++spins;
                if (wg == 0 && (spins & 0x3FFFF) == 0)
                  a_hb.store((int)spins, sycl::memory_order::relaxed);
                if (spins > kDevSpin) { ok = 0; break; }
              }
              lok[0] = ok;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const int ok = lok[0];   // WG-uniform (v2 postmortem)
            if (gid < 4096) work[gid & 4095] = k + gid;
            if (wg == 0 && lid == 0)
              a_res.store(ok ? k + 1 : -1, sycl::memory_order::release);
            if (!ok) return;
            it.barrier(sycl::access::fence_space::local_space);
          }
        });
  });

  const char *tag = wide ? "wide  " : (rmw ? "rmw   " : "narrow");
  const int ready = host_poll_ms(res, kReady, 10000);
  if (ready != kReady) {
    std::printf("  A.%s: READY FAILED (saw %d) — device->host publish broken; "
                "draining\n", tag, ready);
    host_store(db, steps + 1);
    q.wait();
    sycl::free(db, q); sycl::free(res, q); sycl::free(work, q);
    return false;
  }
  host_store(res, 0);

  std::vector<double> lat;
  lat.reserve(steps);
  int errors = 0;
  for (int k = 0; k < steps; ++k) {
    const auto t0 = std::chrono::steady_clock::now();
    host_store(db, k + 1);
    const int immediate = host_load(db);   // did our own store land?
    // Hammer the store while polling: if device atomic "loads" are really
    // RMW writebacks clobbering our value, re-storing wins the race and the
    // loop completes — the confirming experiment for the v4 failure.
    const auto limit =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(3000);
    int r;
    while ((r = host_load(res)) < k + 1 && r != -1) {
      host_store(db, k + 1);
      if (std::chrono::steady_clock::now() > limit) break;
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (r == k + 1) {
      if (k >= 100)
        lat.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    } else {
      ++errors;
      std::printf("  A.%s: step %d FAILED — res=%d, db imm-readback=%d, "
                  "db now=%d, device heartbeat=%d spins\n",
                  tag, k, r, immediate, host_load(db), host_load(res + 8));
      host_store(db, steps + 1);
      break;
    }
  }
  q.wait();
  Stats s = summarize(lat, errors);
  std::printf("  A.%s: p50 %7.3f  p99 %7.3f  mean %7.3f us  (%d steps, err=%d)\n",
              tag, s.p50_us, s.p99_us, s.mean_us, s.completed, s.errors);
  sycl::free(db, q); sycl::free(res, q); sycl::free(work, q);
  return errors == 0;
}

// Baselines: today's engine shape.
static void part_a_baselines(sycl::queue &q, int steps) {
  int *res = sycl::malloc_host<int>(16, q);
  int *work = sycl::malloc_device<int>(4096, q);
  std::memset((void *)res, 0, 64);

  {  // (a) submit + q.wait() per step
    std::vector<double> lat;
    lat.reserve(steps);
    for (int k = 0; k < steps; ++k) {
      const auto t0 = std::chrono::steady_clock::now();
      q.parallel_for(sycl::nd_range<1>(kSG, kSG), [=](sycl::nd_item<1> it) {
        if (it.get_global_linear_id() == 0) {
          work[k & 4095] = k;
          res[0] = k + 1;
        }
      });
      q.wait();
      const auto t1 = std::chrono::steady_clock::now();
      if (k >= 100)
        lat.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    Stats s = summarize(lat, 0);
    std::printf("  C.submit+wait: p50 %7.3f  p99 %7.3f  mean %7.3f us\n",
                s.p50_us, s.p99_us, s.mean_us);
  }
  {  // (b) submit per step + host-poll — no event sync
    std::memset((void *)res, 0, 64);
    std::vector<double> lat;
    lat.reserve(steps);
    int errors = 0;
    for (int k = 0; k < steps; ++k) {
      const auto t0 = std::chrono::steady_clock::now();
      q.parallel_for(sycl::nd_range<1>(kSG, kSG), [=](sycl::nd_item<1> it) {
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::system,
                         sycl::access::address_space::global_space>
            a_res(res[0]);
        if (it.get_global_linear_id() == 0) {
          work[k & 4095] = k;
          a_res.store(k + 1, sycl::memory_order::release);
        }
      });
      const int r = host_poll_ms(res, k + 1, 2000);
      const auto t1 = std::chrono::steady_clock::now();
      if (r != k + 1) { ++errors; break; }
      if (k >= 100)
        lat.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    q.wait();
    Stats s = summarize(lat, errors);
    std::printf("  C.submit+poll: p50 %7.3f  p99 %7.3f  mean %7.3f us  (err=%d)\n",
                s.p50_us, s.p99_us, s.mean_us, s.errors);
  }
  sycl::free(res, q);
  sycl::free(work, q);
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
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
  std::printf("== doorbell probe ==\ndevice %d: %s\n", want,
              q.get_device().get_info<sycl::info::device::name>().c_str());

  std::printf("\npart B: pure L0 zex WaitOnMemory/WriteToMemory ping-pong "
              "(2000 steps)\n");
  part_b_zex(want, 2000);

  const int steps = 500;
  std::printf("\npart A: persistent kernel + host-USM spin (%d steps)\n",
              steps);
  const bool narrow_ok = part_a_persistent(q, steps, /*wide=*/false, /*rmw=*/false);
  const bool rmw_ok = part_a_persistent(q, steps, /*wide=*/false, /*rmw=*/true);
  if (narrow_ok || rmw_ok)
    part_a_persistent(q, steps, /*wide=*/true, /*rmw=*/rmw_ok && !narrow_ok);
  else
    std::printf("  A.wide skipped (narrow arms failed)\n");

  std::printf("\npart C baselines: per-step submission (today's shape, "
              "%d steps)\n", steps);
  part_a_baselines(q, steps);
  std::printf("\ndone\n");
  return 0;
}
