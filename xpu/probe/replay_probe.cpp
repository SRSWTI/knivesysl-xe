// replay_probe.cpp — level-up-xpu queue item 2b: the decision probe for
// thesis 1's surviving rung (doorbell + pre-recorded per-step work).
//
// A decode step today is ~350 live submissions. doorbell_probe showed every
// single-sync mechanism sits in one 5-7.5 us class, so the only way the
// device-driven loop pays is AMORTIZATION: record the step's command stream
// once, replay it per step, and reduce the host to one doorbell store + one
// poll. This probe prices exactly that, using N small memory-copy commands as
// the stand-in for a step's command stream (same command-stream mechanics,
// no SPIR-V plumbing; kernel execution time is common to all arms).
//
// Arms (all pure Level Zero, same device, same N commands per step):
//   A. live-append   : per step, append N copies to an async immediate list,
//                      then zeCommandListHostSynchronize. Today's shape.
//   B. replay-execute: record N copies in a regular list once; per step,
//                      zeCommandQueueExecuteCommandLists + queue sync.
//   C. doorbell-replay (the crown): recorded list = [zexWait(db >= 1),
//                      N copies, zexWrite(db = 0)] — SELF-REARMING: the
//                      doorbell is also the completion flag, host<->device
//                      writes alternate strictly (no lesson-17 clobber race).
//                      Executions are pre-queued in a window; steady-state
//                      host work = one store + one poll per step.
// Metrics per arm: wall us/step and HOST-CPU us/step (the decoupling number
// that decides p99-under-load), plus correctness (final copy landed).
//
// SAFETY: window pre-queueing (doorbell_probe v1 lesson), bounded host polls,
// drain via db store on abort. zexWait uses GREATER_THAN_EQUAL.
//
// Build: icpx -O2 -std=c++20 -o replay_probe replay_probe.cpp -lze_loader
// Run:   ./replay_probe [l0_gpu_index=1] [cmds_per_step=350] [steps=200]
#include <level_zero/ze_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

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

#define ZECHK(call)                                                          \
  do {                                                                       \
    ze_result_t _r = (call);                                                 \
    if (_r != ZE_RESULT_SUCCESS) {                                           \
      std::fprintf(stderr, "L0 error 0x%x at line %d\n", _r, __LINE__);      \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

static inline double thread_cpu_us() {
  timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}
static inline int host_load(volatile int *p) {
  return __atomic_load_n((int *)p, __ATOMIC_ACQUIRE);
}
static inline void host_store(volatile int *p, int v) {
  __atomic_store_n((int *)p, v, __ATOMIC_RELEASE);
}

struct Env {
  ze_driver_handle_t drv = nullptr;
  ze_device_handle_t dev = nullptr;
  ze_context_handle_t ctx = nullptr;
  pfn_zexWaitOnMem zexWait = nullptr;
  pfn_zexWriteToMem zexWrite = nullptr;
};

static Env setup(int gpu_index) {
  Env e;
  ZECHK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t nd = 0;
  ZECHK(zeDriverGet(&nd, nullptr));
  std::vector<ze_driver_handle_t> drivers(nd);
  ZECHK(zeDriverGet(&nd, drivers.data()));
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
      if (seen++ == gpu_index) { e.drv = d; e.dev = dd; }
    }
  }
  if (!e.dev) {
    std::fprintf(stderr, "gpu index %d not found (%d gpus)\n", gpu_index, seen);
    std::exit(1);
  }
  ze_context_desc_t cdesc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ZECHK(zeContextCreate(e.drv, &cdesc, &e.ctx));
  ZECHK(zeDriverGetExtensionFunctionAddress(
      e.drv, "zexCommandListAppendWaitOnMemory", (void **)&e.zexWait));
  ZECHK(zeDriverGetExtensionFunctionAddress(
      e.drv, "zexCommandListAppendWriteToMemory", (void **)&e.zexWrite));
  return e;
}

// N small copies rotating through device buffers; the last copy writes the
// step-invariant marker the host verifies once at the end.
static void append_step_commands(ze_command_list_handle_t cl, uint8_t *dsrc,
                                 uint8_t *ddst, int ncmds) {
  for (int c = 0; c < ncmds; ++c) {
    ZECHK(zeCommandListAppendMemoryCopy(cl, ddst + (c % 64) * 64,
                                        dsrc + (c % 64) * 64, 64, nullptr, 0,
                                        nullptr));
  }
}

struct ArmResult {
  double wall_us = 0, cpu_us = 0;
};

static void report(const char *tag, ArmResult r, int steps) {
  std::printf("  %-18s: wall %8.2f us/step   host-cpu %8.2f us/step\n", tag,
              r.wall_us / steps, r.cpu_us / steps);
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int gpu = (argc > 1) ? std::atoi(argv[1]) : 1;
  const int ncmds = (argc > 2) ? std::atoi(argv[2]) : 350;
  const int steps = (argc > 3) ? std::atoi(argv[3]) : 200;
  Env e = setup(gpu);
  std::printf("== replay probe ==  gpu %d, %d cmds/step, %d steps\n", gpu,
              ncmds, steps);

  ze_device_mem_alloc_desc_t ddesc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  ze_host_mem_alloc_desc_t hdesc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  void *dsrc = nullptr, *ddst = nullptr, *db = nullptr;
  ZECHK(zeMemAllocDevice(e.ctx, &ddesc, 64 * 64, 64, e.dev, &dsrc));
  ZECHK(zeMemAllocDevice(e.ctx, &ddesc, 64 * 64, 64, e.dev, &ddst));
  ZECHK(zeMemAllocHost(e.ctx, &hdesc, 256, 64, &db));
  std::memset(db, 0, 256);
  volatile int *vdb = (volatile int *)db;

  ze_command_queue_desc_t qdesc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qdesc.ordinal = 0;
  qdesc.index = 0;
  qdesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  qdesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

  // ---- arm A: live append on an immediate list -----------------------------
  {
    ze_command_list_handle_t icl = nullptr;
    ZECHK(zeCommandListCreateImmediate(e.ctx, e.dev, &qdesc, &icl));
    // warm
    for (int s = 0; s < 10; ++s) {
      append_step_commands(icl, (uint8_t *)dsrc, (uint8_t *)ddst, ncmds);
      ZECHK(zeCommandListHostSynchronize(icl, UINT64_MAX));
    }
    ArmResult r;
    const double c0 = thread_cpu_us();
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
      append_step_commands(icl, (uint8_t *)dsrc, (uint8_t *)ddst, ncmds);
      ZECHK(zeCommandListHostSynchronize(icl, UINT64_MAX));
    }
    const auto t1 = std::chrono::steady_clock::now();
    r.wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    r.cpu_us = thread_cpu_us() - c0;
    report("A.live-append", r, steps);
    zeCommandListDestroy(icl);
  }

  // ---- arm B: recorded list, execute per step ------------------------------
  {
    ze_command_queue_handle_t cq = nullptr;
    ZECHK(zeCommandQueueCreate(e.ctx, e.dev, &qdesc, &cq));
    ze_command_list_desc_t ldesc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    ldesc.commandQueueGroupOrdinal = 0;
    ze_command_list_handle_t rl = nullptr;
    ZECHK(zeCommandListCreate(e.ctx, e.dev, &ldesc, &rl));
    append_step_commands(rl, (uint8_t *)dsrc, (uint8_t *)ddst, ncmds);
    ZECHK(zeCommandListClose(rl));
    for (int s = 0; s < 10; ++s) {   // warm
      ZECHK(zeCommandQueueExecuteCommandLists(cq, 1, &rl, nullptr));
      ZECHK(zeCommandQueueSynchronize(cq, UINT64_MAX));
    }
    ArmResult r;
    const double c0 = thread_cpu_us();
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
      ZECHK(zeCommandQueueExecuteCommandLists(cq, 1, &rl, nullptr));
      ZECHK(zeCommandQueueSynchronize(cq, UINT64_MAX));
    }
    const auto t1 = std::chrono::steady_clock::now();
    r.wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    r.cpu_us = thread_cpu_us() - c0;
    report("B.replay-execute", r, steps);
    zeCommandListDestroy(rl);
    zeCommandQueueDestroy(cq);
  }

  // ---- diag: does zex wait/write work inside a REGULAR command list? -------
  bool regular_zex_ok = false;
  {
    ze_command_queue_handle_t cq = nullptr;
    ZECHK(zeCommandQueueCreate(e.ctx, e.dev, &qdesc, &cq));
    ze_command_list_desc_t ldesc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    ldesc.commandQueueGroupOrdinal = 0;
    ze_command_list_handle_t gl = nullptr;
    ZECHK(zeCommandListCreate(e.ctx, e.dev, &ldesc, &gl));
    zex_wait_on_mem_desc_t wdesc{ZEX_WAIT_ON_MEMORY_FLAG_GREATER_THAN_EQUAL,
                                 ZEX_MEM_ACTION_SCOPE_FLAG_HOST};
    zex_write_to_mem_desc_t wrdesc{ZEX_MEM_ACTION_SCOPE_FLAG_HOST};
    ZECHK(e.zexWait(gl, &wdesc, db, 1u, nullptr));
    ZECHK(e.zexWrite(gl, &wrdesc, db, 0ull));
    ZECHK(zeCommandListClose(gl));
    host_store(vdb, 0);
    ZECHK(zeCommandQueueExecuteCommandLists(cq, 1, &gl, nullptr));
    host_store(vdb, 1);
    const auto limit =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (host_load(vdb) != 0 &&
           std::chrono::steady_clock::now() < limit) {}
    regular_zex_ok = (host_load(vdb) == 0);
    std::printf("  diag: zex in regular list = %s\n",
                regular_zex_ok ? "WORKS" : "BROKEN (wait or write inert)");
    host_store(vdb, 1);   // release if parked
    zeCommandQueueSynchronize(cq, UINT64_MAX);
    host_store(vdb, 0);
    zeCommandListDestroy(gl);
    zeCommandQueueDestroy(cq);
  }

  // ---- arm C: self-rearming doorbell replay --------------------------------
  // Recorded list: wait(db >= 1) -> N copies -> write(db = 0). The queue
  // serializes queued executions, so instance k+1 parks on the wait until the
  // host's next store. Host steady state: store db=1, poll db==0.
  if (regular_zex_ok) {
    ze_command_queue_handle_t cq = nullptr;
    ZECHK(zeCommandQueueCreate(e.ctx, e.dev, &qdesc, &cq));
    ze_command_list_desc_t ldesc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    ldesc.commandQueueGroupOrdinal = 0;
    ze_command_list_handle_t rl = nullptr;
    ZECHK(zeCommandListCreate(e.ctx, e.dev, &ldesc, &rl));
    zex_wait_on_mem_desc_t wdesc{ZEX_WAIT_ON_MEMORY_FLAG_GREATER_THAN_EQUAL,
                                 ZEX_MEM_ACTION_SCOPE_FLAG_HOST};
    zex_write_to_mem_desc_t wrdesc{ZEX_MEM_ACTION_SCOPE_FLAG_HOST};
    ZECHK(e.zexWait(rl, &wdesc, db, 1u, nullptr));
    append_step_commands(rl, (uint8_t *)dsrc, (uint8_t *)ddst, ncmds);
    ZECHK(e.zexWrite(rl, &wrdesc, db, 0ull));
    ZECHK(zeCommandListClose(rl));

    constexpr int kWindow = 1;   // v3: >1 pre-queued parked executions failed — isolating
    int queued = 0;
    auto top_up = [&](int upto) {
      while (queued < upto && queued < steps) {
        ZECHK(zeCommandQueueExecuteCommandLists(cq, 1, &rl, nullptr));
        ++queued;
      }
    };
    top_up(kWindow);

    ArmResult r;
    int errors = 0;
    int failed_at = -1;
    const double c0 = thread_cpu_us();
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
      host_store(vdb, 1);
      const auto limit =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (host_load(vdb) != 0) {
        if (std::chrono::steady_clock::now() > limit) { ++errors; break; }
      }
      if (errors) { failed_at = s; break; }
      top_up(s + 1 + kWindow);
    }
    const auto t1 = std::chrono::steady_clock::now();
    r.wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    r.cpu_us = thread_cpu_us() - c0;
    if (errors) {
      std::printf("  C.doorbell-replay : FAILED at step %d (db=%d) — draining\n",
                  failed_at, host_load(vdb));
      host_store(vdb, 1);   // release anything parked, repeatedly below
      for (int i = 0; i < queued; ++i) host_store(vdb, 1);
    } else {
      report("C.doorbell-replay", r, steps);
    }
    zeCommandQueueSynchronize(cq, UINT64_MAX);
    zeCommandListDestroy(rl);
    zeCommandQueueDestroy(cq);
  } else {
    std::printf("  C.doorbell-replay : skipped (regular-list zex broken)\n");
  }
  // ---- arm D: zex on the immediate list, replay via immediate-append ------
  // Per step (windowed pre-append on ONE in-order immediate list):
  //   zexWait(db >= 1) -> ImmediateAppendCommandListsExp(recorded copies)
  //   -> zexWrite(db = 0).
  // zex stays on the immediate list (proven in doorbell_probe part B); the
  // recorded regular list carries the step's commands.
  {
    ze_command_list_handle_t icl = nullptr;
    ZECHK(zeCommandListCreateImmediate(e.ctx, e.dev, &qdesc, &icl));
    ze_command_list_desc_t ldesc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    ldesc.commandQueueGroupOrdinal = 0;
    ze_command_list_handle_t rl = nullptr;
    ZECHK(zeCommandListCreate(e.ctx, e.dev, &ldesc, &rl));
    append_step_commands(rl, (uint8_t *)dsrc, (uint8_t *)ddst, ncmds);
    ZECHK(zeCommandListClose(rl));

    zex_wait_on_mem_desc_t wdesc{ZEX_WAIT_ON_MEMORY_FLAG_GREATER_THAN_EQUAL,
                                 ZEX_MEM_ACTION_SCOPE_FLAG_HOST};
    zex_write_to_mem_desc_t wrdesc{ZEX_MEM_ACTION_SCOPE_FLAG_HOST};
    host_store(vdb, 0);

    constexpr int kWindow = 1;   // v3: >1 pre-queued parked executions failed — isolating
    int queued = 0;
    auto top_up = [&](int upto) {
      while (queued < upto && queued < steps) {
        ZECHK(e.zexWait(icl, &wdesc, db, 1u, nullptr));
        ZECHK(zeCommandListImmediateAppendCommandListsExp(icl, 1, &rl, nullptr,
                                                          0, nullptr));
        ZECHK(e.zexWrite(icl, &wrdesc, db, 0ull));
        ++queued;
      }
    };
    top_up(kWindow);

    ArmResult r;
    int errors = 0;
    int failed_at = -1;
    const double c0 = thread_cpu_us();
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
      host_store(vdb, 1);
      const auto limit =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (host_load(vdb) != 0) {
        if (std::chrono::steady_clock::now() > limit) { ++errors; break; }
      }
      if (errors) { failed_at = s; break; }
      top_up(s + 1 + kWindow);
    }
    const auto t1 = std::chrono::steady_clock::now();
    r.wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    r.cpu_us = thread_cpu_us() - c0;
    if (errors) {
      std::printf("  D.imm-append      : FAILED at step %d (db=%d) — draining\n",
                  failed_at, host_load(vdb));
      host_store(vdb, 1);
    } else {
      report("D.imm-append", r, steps);
    }
    zeCommandListHostSynchronize(icl, UINT64_MAX);
    zeCommandListDestroy(rl);
    zeCommandListDestroy(icl);
  }
  zeMemFree(e.ctx, dsrc);
  zeMemFree(e.ctx, ddst);
  zeContextDestroy(e.ctx);
  std::printf("done\n");
  return 0;
}
