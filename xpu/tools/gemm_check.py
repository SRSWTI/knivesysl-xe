#!/usr/bin/env python3
"""gemm_check.py - gate the RC8 batched GEMM against the certified RC1 GEMV.

The GEMV path is already validated against the scalar E2M3 reference, so
agreeing with it row-for-row validates the probed RC8 operand layout, the
K-tile-major activation staging and the epilogue in one shot.

  python3 xpu/tools/gemm_check.py --lib xpu/build/libforward_qwen_xpu.so \
      --tqf ~/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf
"""
import argparse
import ctypes
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tokens", type=int, nargs="+", default=[8, 32, 128, 512])
    args = ap.parse_args()

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_init.restype = ctypes.c_int
    lib.qwn_gemm_check.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_gemm_check.restype = ctypes.c_int
    lib.qwn_free.argtypes = []
    lib.qwn_free.restype = None

    rc = lib.qwn_init(os.path.expanduser(args.tqf).encode())
    if rc != 0:
        print(f"qwn_init failed rc={rc}", file=sys.stderr)
        return 1

    failures = 0
    for tokens in args.tokens:
        rc = lib.qwn_gemm_check(args.layer, tokens)
        failures += rc != 0
        if rc != 0:
            print(f"  T={tokens} returned rc={rc}", flush=True)
    lib.qwn_free()
    print("OVERALL:", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
