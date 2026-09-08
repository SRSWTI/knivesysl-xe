#!/usr/bin/env python3
"""prefill_check.py - chunked prefill must be state-equivalent to the per-token loop.

Runs the same prompt two ways in two separate processes (so no state can leak
between them) and compares the greedy continuation:

  A: qwn_decode() for every prompt token, then N greedy steps
  B: qwn_prefill_chunk() over the prompt, then the same N greedy steps

The chunk path batches projections, uses tiled attention (XMX when selected),
and runs chunked convolution/DeltaNet. Reduction order and matrix operand
rounding can differ from scalar decode. Gate: >= 90% greedy continuation
agreement; prefill_batch_check.py separately gates teacher-forced numerics and
fixed-reduction-order packed-state equality.

  python3 xpu/tools/prefill_check.py --lib xpu/build/libforward_qwen_xpu.so \
      --tqf ~/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf
"""
import argparse
import ctypes
import os
import subprocess
import sys

PROMPT = [151644, 872, 198, 9707, 11, 1246, 525, 498, 30, 151645, 198, 151644,
          77091, 198, 40, 2776, 1101]


def bind(libpath, tqf):
    lib = ctypes.CDLL(os.path.abspath(libpath))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_init.restype = ctypes.c_int
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_decode.restype = ctypes.c_int
    lib.qwn_prefill_chunk.argtypes = [ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_int, ctypes.c_int]
    lib.qwn_prefill_chunk.restype = ctypes.c_int
    lib.qwn_free.argtypes = []
    lib.qwn_free.restype = None
    rc = lib.qwn_init(os.path.expanduser(tqf).encode())
    if rc != 0:
        raise RuntimeError(f"qwn_init rc={rc}")
    return lib


def run(mode, libpath, tqf, ntok, steps):
    lib = bind(libpath, tqf)
    prompt = (PROMPT * ((ntok // len(PROMPT)) + 1))[:ntok]
    if mode == "chunk":
        arr = (ctypes.c_int * ntok)(*prompt)
        rc = lib.qwn_prefill_chunk(arr, ntok, 0)
        if rc != 0:
            print(f"PREFILL_RC {rc}", flush=True)
            lib.qwn_free()
            return
    else:
        for i, t in enumerate(prompt):
            if lib.qwn_decode(int(t), i) < 0:
                raise RuntimeError(f"decode failed at {i}")
    out = []
    tok = prompt[-1]
    for s in range(steps):
        tok = lib.qwn_decode(int(tok), ntok + s)
        if tok < 0:
            raise RuntimeError(f"greedy failed at {s}: {tok}")
        out.append(tok)
    print("CHAIN " + " ".join(str(x) for x in out), flush=True)
    lib.qwn_free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--mode", choices=["loop", "chunk"])
    args = ap.parse_args()

    if args.mode:
        run(args.mode, args.lib, args.tqf, args.tokens, args.steps)
        return 0

    chains = {}
    for mode in ("loop", "chunk"):
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--lib", args.lib,
             "--tqf", args.tqf, "--tokens", str(args.tokens),
             "--steps", str(args.steps), "--mode", mode],
            text=True, capture_output=True)
        line = next((l for l in r.stdout.splitlines()
                     if l.startswith(("CHAIN", "PREFILL_RC"))), None)
        if line is None:
            print(f"{mode}: no chain\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
            return 1
        if line.startswith("PREFILL_RC"):
            print(f"chunked prefill refused: {line}")
            return 1
        chains[mode] = [int(x) for x in line.split()[1:]]

    a, b = chains["loop"], chains["chunk"]
    n = min(len(a), len(b))
    agree = sum(1 for i in range(n) if a[i] == b[i])
    first = next((i for i in range(n) if a[i] != b[i]), -1)
    rate = agree / n if n else 0.0
    print(f"prompt={args.tokens} steps={n}")
    print(f"  loop : {' '.join(str(x) for x in a[:16])} ...")
    print(f"  chunk: {' '.join(str(x) for x in b[:16])} ...")
    print(f"agreement {rate*100:.2f}% ({agree}/{n}), first divergence at {first}"
          f"  {'PASS' if rate >= 0.90 else 'FAIL'}")
    return 0 if rate >= 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
