#!/usr/bin/env python3
"""tp_check.py - tensor-parallel substrate gate.

Checks the TP plumbing independently of the model, so it can run before any
weights are sharded: one shared SYCL context spanning the ranks, one in-order
queue per rank, per-rank device allocation, and the peer-exchange all-reduce.
Rank r is seeded with (r+1)*i and every rank must end holding
sum_r (r+1) * i.

  TQ_XPU_DEV=0 TQ_XPU_TP=2 python3 xpu/tools/tp_check.py \
      --lib xpu/build/libforward_qwen_xpu.so

Also times the all-reduce at the model's hidden size so the per-token comm
cost is reported next to the correctness result.
"""
import argparse
import ctypes
import os
import statistics
import sys
import time

H = 5120                 # hidden size = the row-parallel all-reduce payload
ALLREDUCES_PER_TOKEN = 128   # 64 layers x 2 row-parallel reductions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="xpu/build/libforward_qwen_xpu.so")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[H, 17408, 262144])
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_tp_selftest.argtypes = [ctypes.c_int]
    lib.qwn_tp_selftest.restype = ctypes.c_int

    tp = lib.qwn_tp_selftest(64)
    if tp == 1:
        print("TQ_XPU_TP=1: single rank, nothing to reduce (set TQ_XPU_TP=2)")
        return 0
    if tp < 0:
        print(f"substrate selftest failed rc={tp}", file=sys.stderr)
        return 1
    print(f"ranks: {tp}")

    fails = 0
    for n in args.sizes:
        rc = lib.qwn_tp_selftest(n)
        ok = rc == tp
        fails += 0 if ok else 1
        print(f"  allreduce n={n:<8d} rc={rc:<3d} {'PASS' if ok else 'FAIL'}")

    # cost at the model's payload: the selftest includes alloc + H2D + D2H,
    # so time the repeat and subtract nothing - report it as an upper bound
    # alongside the probe's isolated 37.33 us.
    lat = []
    for _ in range(args.iters):
        t0 = time.perf_counter_ns()
        lib.qwn_tp_selftest(H)
        lat.append((time.perf_counter_ns() - t0) / 1e3)
    med = statistics.median(lat)
    print(f"\n  selftest round at n={H}: median {med:.1f} us "
          f"(includes alloc + H2D + D2H, so an upper bound)")
    print(f"  reference: isolated all-reduce 37.33 us -> "
          f"{37.33 * ALLREDUCES_PER_TOKEN / 1000:.2f} ms/token over "
          f"{ALLREDUCES_PER_TOKEN} reductions")
    print("\nOVERALL:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
