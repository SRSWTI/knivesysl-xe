#!/usr/bin/env python3
"""grid_probe.py — run qwn_w4_grid_probe on chosen layers.

Prices weight-format candidates (reconstruction cosine / rel-L2 at matched
metadata budgets) on real weights, on silicon, before any kernel exists.
Requires the E2M3 payload resident, so TQ_XPU_W4A8=0 is forced here.

  python3 xpu/tools/grid_probe.py --lib xpu/build/libforward_qwen_xpu.so \
      --tqf ~/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf --layers 0 1 32
"""
import argparse
import ctypes
import os
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 32])
    args = ap.parse_args()

    os.environ["TQ_XPU_W4A8"] = "0"   # keep the E2M3 payload resident
    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_init.restype = ctypes.c_int
    lib.qwn_w4_grid_probe.argtypes = [ctypes.c_int]
    lib.qwn_w4_grid_probe.restype = ctypes.c_int
    lib.qwn_free.argtypes = []
    lib.qwn_free.restype = None

    rc = lib.qwn_init(os.path.expanduser(args.tqf).encode())
    if rc != 0:
        print(f"qwn_init failed rc={rc}", file=sys.stderr)
        return 1
    worst = 0
    for layer in args.layers:
        rc = lib.qwn_w4_grid_probe(layer)
        if rc < 0:
            print(f"layer {layer}: probe rc={rc}", file=sys.stderr)
            worst = 1
    lib.qwn_free()
    return worst

if __name__ == "__main__":
    sys.exit(main())
