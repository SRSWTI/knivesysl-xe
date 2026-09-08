#!/usr/bin/env python3
"""bench_batch.py - aggregate throughput of the batched decode step.

Prefills B slots to --ctx tokens each (chunked prefill), then times
qwn_decode_batch at widths 1,2,4,8. The step is RC8-shaped: one weight
stream serves 8 rows, so per-step cost should be nearly flat in B and
aggregate tok/s should scale ~linearly. Also times the single-stream
qwn_decode (GEMV) path as the baseline.

TQ_XPU_DEV=0 TQ_XPU_SLOTS=8 python3 xpu/tools/bench_batch.py \
    --lib xpu/build/libforward_qwen_xpu.so --tqf ~/models/.../*.tqf
"""
import argparse
import ctypes
import os
import statistics
import sys
import time

TOK = 9707  # arbitrary in-vocab filler token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_set_slot.argtypes = [ctypes.c_int]
    lib.qwn_prefill_chunk.argtypes = [ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_int, ctypes.c_int]
    lib.qwn_decode_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [
        ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    rc = lib.qwn_init(os.path.expanduser(args.tqf).encode())
    if rc != 0:
        print(f"qwn_init rc={rc}", file=sys.stderr)
        return 1
    nslots = lib.qwn_num_slots()
    widths = [w for w in args.widths if w <= nslots]
    maxw = max(widths)
    print(f"slots={nslots} ctx={args.ctx} steps={args.steps} widths={widths}")

    ctx = (args.ctx // 8) * 8
    IntC = ctypes.c_int * ctx
    prompt = IntC(*([TOK] * ctx))
    # each slot gets its own unique-prefix prompt so no state is shared
    for s in range(maxw):
        assert lib.qwn_set_slot(s) == 0
        p = IntC(*([TOK + s] * ctx))
        rc = lib.qwn_prefill_chunk(p, ctx, 0)
        if rc != 0:
            print(f"prefill slot {s} rc={rc}", file=sys.stderr)
            return 1

    # single-stream baseline (GEMV path, slot 0)
    assert lib.qwn_set_slot(0) == 0
    pos = ctx
    for _ in range(args.warmup):
        lib.qwn_decode(TOK, pos); pos += 1
    lat = []
    for _ in range(args.steps):
        t0 = time.perf_counter_ns()
        assert lib.qwn_decode(TOK, pos) >= 0
        lat.append((time.perf_counter_ns() - t0) / 1e6)
        pos += 1
    base_ms = statistics.fmean(lat)
    print(f"\n  {'mode':>12}  {'ms/step':>8}  {'per-stream':>11}  {'aggregate':>10}")
    print(f"  {'GEMV n=1':>12}  {base_ms:8.2f}  {1000/base_ms:11.2f}  "
          f"{1000/base_ms:10.2f}")

    Arr = lambda n, *v: (ctypes.c_int * n)(*v)
    rows = []
    for w in widths:
        slots = Arr(w, *range(w))
        poss = [ctx + args.warmup + 1] * w   # all slots past their prefill
        toks = Arr(w, *([TOK] * w))
        out = Arr(w, *([0] * w))
        pv = Arr(w, *poss)
        for _ in range(args.warmup):
            rc = lib.qwn_decode_batch(slots, toks, pv, w, out)
            assert rc == w, f"batch rc={rc}"
            for i in range(w):
                poss[i] += 1
            pv = Arr(w, *poss)
        lat = []
        for _ in range(args.steps):
            t0 = time.perf_counter_ns()
            rc = lib.qwn_decode_batch(slots, toks, pv, w, out)
            lat.append((time.perf_counter_ns() - t0) / 1e6)
            assert rc == w
            for i in range(w):
                poss[i] += 1
            pv = Arr(w, *poss)
        ms = statistics.fmean(lat)
        per, agg = 1000 / ms, w * 1000 / ms
        rows.append((w, ms, per, agg))
        print(f"  {'RC8 n=' + str(w):>12}  {ms:8.2f}  {per:11.2f}  {agg:10.2f}")
    best = max(rows, key=lambda r: r[3])
    print(f"\n  peak aggregate {best[3]:.1f} tok/s at n={best[0]} "
          f"({best[3]/(1000/base_ms):.2f}x single-stream)")
    lib.qwn_free()
    return 0


if __name__ == "__main__":
    sys.exit(main())
