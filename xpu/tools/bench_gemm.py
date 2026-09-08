#!/usr/bin/env python3
"""bench_gemm.py - achieved rate of the RC8 batched W4A8 GEMM.

Reports, per token-batch T, the wall time for one 17408x5120 projection plus
the effective TOPS and the weight-bytes-per-output-element. The point of RC8 is
that the weight fragment is reused across 8 activation rows, so raising T should
move the kernel off the bandwidth roofline and onto the DPAS roofline.
"""
import argparse
import ctypes
import os
import sys

BW = 602e9          # measured B70 read bandwidth
DPAS_TOPS = 366e12  # measured s8xs4 DPAS rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--activation", choices=("s8", "s4"), default="s8")
    ap.add_argument("--check", action="store_true",
                    help="check nontrivial projection inputs before timing each shape")
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128, 256, 512, 1024])
    args = ap.parse_args()
    if args.iters <= 0 or any(t <= 0 or t % 8 for t in args.tokens):
        ap.error("iterations must be positive and token counts positive multiples of eight")

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_init.restype = ctypes.c_int
    bench = getattr(lib, "qwn_gemm_bench_w4a4" if args.activation == "s4"
                    else "qwn_gemm_bench")
    bench.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    bench.restype = ctypes.c_double
    check = getattr(lib, "qwn_gemm_w4a4_check" if args.activation == "s4"
                    else "qwn_gemm_check")
    check.argtypes = [ctypes.c_int, ctypes.c_int]
    check.restype = ctypes.c_int
    lib.qwn_free.argtypes = []
    lib.qwn_free.restype = None

    if lib.qwn_init(os.path.expanduser(args.tqf).encode()) != 0:
        print("qwn_init failed", file=sys.stderr)
        return 1

    M, K = 17408, 5120
    group = 64 if args.activation == "s4" else 32
    peak = 733e12 if args.activation == "s4" else DPAS_TOPS
    w_bytes = M * K // 2 + M * (K // group) * 2
    print(f"activation={args.activation} M={M} K={K} scale_group={group}; "
          "activation preparation excluded; byte rate counts one weight pass")
    print(f"{'T':>6} {'us':>10} {'TOPS':>8} {'%DPAS':>7} {'GB/s':>8} "
          f"{'%BW':>6} {'tok/s':>10}")
    for T in args.tokens:
        if args.check and check(args.layer, T) != 0:
            lib.qwn_free()
            raise RuntimeError(f"projection correctness check failed at T={T}")
        us = bench(args.layer, T, args.iters)
        if us < 0:
            lib.qwn_free()
            raise RuntimeError(f"projection benchmark failed at T={T}: rc={us}")
        flops = 2.0 * M * K * T
        tops = flops / (us * 1e-6) / 1e12
        gbs = w_bytes / (us * 1e-6) / 1e9
        print(f"{T:>6} {us:>10.1f} {tops:>8.1f} {tops/(peak/1e12)*100:>6.0f}% "
              f"{gbs:>8.1f} {gbs/(BW/1e9)*100:>5.0f}% {T/(us*1e-6):>10.0f}")
    lib.qwn_free()
    return 0


if __name__ == "__main__":
    sys.exit(main())
