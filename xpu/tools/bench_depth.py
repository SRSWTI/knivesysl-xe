#!/usr/bin/env python3
"""bench_depth.py - decode latency vs depth. **OPTIMISTIC, NOT SHIPPABLE.**

REJECTED as an absolute measure - kept only to document why, and as a cheap
relative/shape probe.

The idea was that decode cost at position p is set by the BYTES READ (weights,
KV rows 0..p, DeltaNet state), not by their contents, so depth latency could be
had without paying for a real prefill, which is O(n^2) here and costs ~2 hours
at 131k.

Measured against real-prefill runs, it is optimistic, and the error grows with
depth: +0.8% at 2k, +2.8% at 8k, +5.0% at 16k, +8.1% at 32k.

The error tracks KV VOLUME. Position-advance cannot explain it - the real run's
timed steps sit only 80 rows past `depth`, 0.24% more traffic at 32k. What
differs is CONTENT: qwn_reset_state zeroes the cache, and a zeroed cache reads
faster than a populated one, i.e. the memory path is compressing uniform pages.
[INFERENCE] - concluded from the error's shape, not from a driver document.

So: any depth number past 32k must come from a real prefill. Left in the tree
because "zeros read faster than data" is a trap worth having written down, and
because the shape is still usable for A/B work at FIXED depth.

It measures LATENCY ONLY; generated tokens are meaningless. Correctness is
established by xpu_parity.py / xpu_tf_check.py, never here.
"""
import argparse
import ctypes
import os
import statistics
import time


def percentile(values, q):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower, upper = int(pos), min(int(pos) + 1, len(ordered) - 1)
    fraction = pos - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tqf", required=True)
    parser.add_argument("--depths", default="2048,8192,16384,32768,65536,98304,131072")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_init.restype = ctypes.c_int
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_decode.restype = ctypes.c_int
    lib.qwn_reset_state.argtypes = []
    lib.qwn_reset_state.restype = ctypes.c_int
    lib.qwn_free.argtypes = []
    lib.qwn_free.restype = None

    rc = lib.qwn_init(os.path.expanduser(args.tqf).encode())
    if rc != 0:
        raise RuntimeError(f"qwn_init failed: {rc}")
    try:
        for depth in [int(d) for d in args.depths.split(",")]:
            if lib.qwn_reset_state() != 0:
                raise RuntimeError("reset_state failed")
            # Step at a FIXED position so every sample reads the same
            # depth's worth of cache. Advancing would blend depths.
            for _ in range(args.warmup):
                if lib.qwn_decode(17, depth - 1) < 0:
                    raise RuntimeError(f"warm-up failed at depth {depth}")
            latencies_ms = []
            for _ in range(args.steps):
                started = time.perf_counter_ns()
                if lib.qwn_decode(17, depth - 1) < 0:
                    raise RuntimeError(f"decode failed at depth {depth}")
                latencies_ms.append((time.perf_counter_ns() - started) / 1e6)
            mean_ms = statistics.fmean(latencies_ms)
            print(f"DEPTH {depth} tok_s={1000.0 / mean_ms:.3f} "
                  f"mean_ms={mean_ms:.3f} p50={percentile(latencies_ms, 0.5):.3f} "
                  f"p99={percentile(latencies_ms, 0.99):.3f}", flush=True)
    finally:
        lib.qwn_free()


if __name__ == "__main__":
    main()
