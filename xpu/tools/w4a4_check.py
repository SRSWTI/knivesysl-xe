#!/usr/bin/env python3
"""w4a4_check.py - certification + bench for the W4A4 prefill tier.

Chain (level-up item 4 integration):
  1. --anchor (run with TQ_XPU_W4A8=0 TQ_XPU_K64=1): qwn_w4_check per layer -
     scalar-E2M3 reference vs the k64-scaled GEMV. Absolute quality anchor.
  2. default (run with TQ_XPU_K64=1): qwn_gemm_check (GEMV vs W4A8 GEMM under
     k64 scales - certifies the ksh epilogue) and qwn_gemm_w4a4_check (GEMV
     vs W4A4 GEMM - certifies K64 B routing, s4 A layout, s4 quantizer,
     per-K64 epilogue); then qwn_gemm_bench vs qwn_gemm_bench_w4a4.

TQ_XPU_DEV=1 TQ_XPU_K64=1 python3 xpu/tools/w4a4_check.py \
    --lib xpu/build/libforward_qwen_xpu.so --tqf ~/models/.../*.tqf
"""
import argparse
import ctypes
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--layers", default="0,1,32")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--anchor", action="store_true",
                    help="scalar-vs-GEMV anchor (needs TQ_XPU_W4A8=0)")
    args = ap.parse_args()

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_init.restype = ctypes.c_int
    rc = lib.qwn_init(os.path.expanduser(args.tqf).encode())
    if rc != 0:
        print(f"qwn_init failed rc={rc}", file=sys.stderr)
        return 1

    fails = 0
    layers = [int(t) for t in args.layers.split(",")]
    if args.anchor:
        lib.qwn_w4_check.argtypes = [ctypes.c_int]
        lib.qwn_w4_check.restype = ctypes.c_int
        for L in layers:
            if lib.qwn_w4_check(L) != 0:
                fails += 1
    else:
        lib.qwn_gemm_check.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.qwn_gemm_check.restype = ctypes.c_int
        lib.qwn_gemm_w4a4_check.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.qwn_gemm_w4a4_check.restype = ctypes.c_int
        lib.qwn_gemm_bench.argtypes = [ctypes.c_int] * 3
        lib.qwn_gemm_bench.restype = ctypes.c_double
        lib.qwn_gemm_bench_w4a4.argtypes = [ctypes.c_int] * 3
        lib.qwn_gemm_bench_w4a4.restype = ctypes.c_double
        for L in layers:
            if lib.qwn_gemm_check(L, args.tokens) != 0:
                fails += 1
            if lib.qwn_gemm_w4a4_check(L, args.tokens) != 0:
                fails += 1
        # bench on layer 0, T sweep; TOPS = 2*M*K*T / us
        M, K = 17408, 5120
        print(f"{'T':>6} {'w4a8_us':>9} {'w4a4_us':>9} {'w4a8_TOPS':>10} "
              f"{'w4a4_TOPS':>10} {'speedup':>8}")
        for T in (256, 512, 1024):
            a8 = lib.qwn_gemm_bench(0, T, 20)
            a4 = lib.qwn_gemm_bench_w4a4(0, T, 20)
            if a8 <= 0 or a4 <= 0:
                print(f"{T:>6} bench failed ({a8:.1f}/{a4:.1f})")
                fails += 1
                continue
            f = 2.0 * M * K * T
            print(f"{T:>6} {a8:>9.1f} {a4:>9.1f} {f / a8 / 1e6:>10.1f} "
                  f"{f / a4 / 1e6:>10.1f} {a8 / a4:>7.2f}x")
    lib.qwn_free()
    print("OVERALL:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
