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
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128, 256, 512, 1024])
    args = ap.parse_args()

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_init.restype = ctypes.c_int
    lib.qwn_gemm_bench.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.qwn_gemm_bench.restype = ctypes.c_double
    lib.qwn_free.argtypes = []
    lib.qwn_free.restype = None

    if lib.qwn_init(os.path.expanduser(args.tqf).encode()) != 0:
        print("qwn_init failed", file=sys.stderr)
        return 1

    M, K = 17408, 5120
    w_bytes = M * K * 4.5 / 8 + M * (K // 32) * 2   # s4 codes + fp16 scales
    print(f"{'T':>6} {'us':>10} {'TOPS':>8} {'%DPAS':>7} {'GB/s':>8} "
          f"{'%BW':>6} {'tok/s':>10}")
    for T in args.tokens:
        us = lib.qwn_gemm_bench(args.layer, T, args.iters)
        if us < 0:
            print(f"{T:>6}  error rc={us}")
            continue
        flops = 2.0 * M * K * T
        tops = flops / (us * 1e-6) / 1e12
        gbs = w_bytes / (us * 1e-6) / 1e9
        print(f"{T:>6} {us:>10.1f} {tops:>8.1f} {tops/(DPAS_TOPS/1e12)*100:>6.0f}% "
              f"{gbs:>8.1f} {gbs/(BW/1e9)*100:>5.0f}% {T/(us*1e-6):>10.0f}")
    lib.qwn_free()
    return 0


if __name__ == "__main__":
    sys.exit(main())
