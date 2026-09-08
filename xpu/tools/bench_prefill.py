#!/usr/bin/env python3
"""bench_prefill.py - TTFT: chunked prefill vs the per-token loop.

Times how long the engine takes to consume a prompt of N tokens, which is what
sets time-to-first-token. The per-token loop streams all 15.05 GB of weights per
token; the chunked path streams them once per chunk.

  python3 xpu/tools/bench_prefill.py --lib xpu/build/libforward_qwen_xpu.so \
      --tqf ~/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf
"""
import argparse
import ctypes
import os
import sys
import time

PROMPT = [151644, 872, 198, 9707, 11, 1246, 525, 498, 30, 151645, 198]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--ctx", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--loop-ref", type=int, default=512,
                    help="also time the per-token loop up to this many tokens")
    args = ap.parse_args()

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_init.restype = ctypes.c_int
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_decode.restype = ctypes.c_int
    lib.qwn_prefill_chunk.argtypes = [ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_int, ctypes.c_int]
    lib.qwn_prefill_chunk.restype = ctypes.c_int
    lib.qwn_reset_state.argtypes = []
    lib.qwn_reset_state.restype = ctypes.c_int
    lib.qwn_free.argtypes = []
    lib.qwn_free.restype = None

    if lib.qwn_init(os.path.expanduser(args.tqf).encode()) != 0:
        print("qwn_init failed", file=sys.stderr)
        return 1

    def prompt_of(n):
        return (PROMPT * ((n // len(PROMPT)) + 1))[:n]

    # Reference: per-token loop.
    ref = {}
    if args.loop_ref > 0:
        toks = prompt_of(args.loop_ref)
        lib.qwn_reset_state()
        t0 = time.perf_counter()
        for i, t in enumerate(toks):
            lib.qwn_decode(int(t), i)
        dt = time.perf_counter() - t0
        ref[args.loop_ref] = dt
        print(f"per-token loop: {args.loop_ref} tokens in {dt:.2f} s "
              f"({args.loop_ref/dt:.1f} tok/s)")

    print(f"\nchunked prefill (chunk={args.chunk})")
    print(f"{'ctx':>8} {'TTFT s':>9} {'tok/s':>9} {'vs loop':>9}")
    loop_rate = (args.loop_ref / ref[args.loop_ref]) if ref else None
    for n in args.ctx:
        n8 = (n // 8) * 8
        toks = prompt_of(n8)
        lib.qwn_reset_state()
        t0 = time.perf_counter()
        pos = 0
        rc = 0
        while pos < n8:
            take = min(args.chunk, n8 - pos)
            arr = (ctypes.c_int * take)(*toks[pos:pos + take])
            rc = lib.qwn_prefill_chunk(arr, take, pos)
            if rc != 0:
                break
            pos += take
        dt = time.perf_counter() - t0
        if rc != 0:
            print(f"{n8:>8}  prefill_chunk rc={rc}")
            continue
        rate = n8 / dt
        speed = f"{rate/loop_rate:.0f}x" if loop_rate else "-"
        print(f"{n8:>8} {dt:>9.2f} {rate:>9.0f} {speed:>9}")
    lib.qwn_free()
    return 0


if __name__ == "__main__":
    sys.exit(main())
