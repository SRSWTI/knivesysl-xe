#!/usr/bin/env python3
"""bench_specwave.py - level-up queue item 7: price the RC8 spec wave.

Compares one RC8 wave (T=8: 1 committed + 7 draft rows, one weight stream)
against 8 sequential RC1 GEMV steps (8 weight streams) on real layer weights,
and verifies the prefix-stability contract: output rows 0-3 of a wave are
byte-identical no matter what rides in rows 4-7.

Economics: plain decode pays gemv_us per token per matrix. A wave pays
wave_us and yields 1+A tokens (A = accepted drafts). Win iff
wave_us/(1+A) < gemv_us, i.e. A > wave_us/gemv_us - 1 (break-even drafts).

python3 xpu/tools/bench_specwave.py --lib xpu/build/libforward_qwen_xpu.so \
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
    ap.add_argument("--layers", default="0,1,32")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_init.restype = ctypes.c_int
    lib.qwn_gemv_bench.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_gemv_bench.restype = ctypes.c_double
    lib.qwn_gemm_bench.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.qwn_gemm_bench.restype = ctypes.c_double
    lib.qwn_specwave_check.argtypes = [ctypes.c_int]
    lib.qwn_specwave_check.restype = ctypes.c_int

    rc = lib.qwn_init(os.path.expanduser(args.tqf).encode())
    if rc != 0:
        print(f"qwn_init failed rc={rc}", file=sys.stderr)
        return 1

    fails = 0
    print(f"{'layer':>5} {'gemv_us':>9} {'wave_us':>9} {'wave/gemv':>10} "
          f"{'breakeven':>10} {'full-acc':>9} {'prefix':>7}")
    for tok in args.layers.split(","):
        layer = int(tok)
        gemv = lib.qwn_gemv_bench(layer, args.iters)
        wave = lib.qwn_gemm_bench(layer, 8, args.iters)
        if gemv <= 0 or wave <= 0:
            print(f"{layer:>5} bench failed (gemv={gemv:.1f} wave={wave:.1f})")
            fails += 1
            continue
        chk = lib.qwn_specwave_check(layer)
        if chk != 0:
            fails += 1
        # break-even accepted drafts per wave; full-acceptance speedup
        print(f"{layer:>5} {gemv:>9.1f} {wave:>9.1f} {wave / gemv:>9.2f}x "
              f"{max(0.0, wave / gemv - 1.0):>10.2f} "
              f"{8.0 * gemv / wave:>8.2f}x {'PASS' if chk == 0 else 'FAIL':>7}")
    lib.qwn_free()
    print("OVERALL:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
