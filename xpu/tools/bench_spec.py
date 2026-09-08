#!/usr/bin/env python3
"""bench_spec.py - design 9.3: speculative wave vs plain greedy decode.

Two arms from the same prompt:
  plain: qwn_decode greedy, one token per step.
  spec : qwn_spec_wave with prompt-lookup drafting - the longest suffix
         n-gram (3/2/1) of the sequence is searched backwards; the 7 tokens
         following the most recent earlier match ride the wave as drafts.

Gates: the spec token stream must agree with the plain stream (same
eps-band class as prefill_check; report agreement + first divergence,
PASS >= 0.90) - plus tok/s for both arms and acceptance stats.

TQ_XPU_DEV=1 python3 xpu/tools/bench_spec.py \
    --lib xpu/build/libforward_qwen_xpu.so --tqf ~/models/.../*.tqf
"""
import argparse
import ctypes
import os
import sys
import time

PROMPT = [151644, 872, 198, 9707, 11, 1246, 525, 498, 30, 151645, 198]


def draft(seq, n):
    """Prompt-lookup: continuation after the latest earlier n-gram match."""
    for g in (3, 2, 1):
        if len(seq) < g + 1:
            continue
        key = tuple(seq[-g:])
        # scan backwards, excluding the trivial suffix match
        for i in range(len(seq) - g - 1, -1, -1):
            if tuple(seq[i:i + g]) == key:
                cont = seq[i + g:i + g + n]
                if cont:
                    return (cont + [0] * n)[:n]
    return [0] * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--steps", type=int, default=256)
    args = ap.parse_args()

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_init.restype = ctypes.c_int
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_decode.restype = ctypes.c_int
    lib.qwn_reset_state.restype = ctypes.c_int
    lib.qwn_spec_wave.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int)]
    lib.qwn_spec_wave.restype = ctypes.c_int
    if lib.qwn_init(os.path.expanduser(args.tqf).encode()) != 0:
        print("init failed", file=sys.stderr)
        return 1

    def feed_prompt():
        lib.qwn_reset_state()
        nxt = -1
        for pos, tok in enumerate(PROMPT):
            nxt = lib.qwn_decode(tok, pos)
            if nxt < 0:
                raise RuntimeError(f"decode rc={nxt} at pos {pos}")
        return nxt

    # ---- plain greedy arm ----
    nxt = feed_prompt()
    seq = list(PROMPT) + [nxt]
    t0 = time.perf_counter()
    for _ in range(args.steps):
        nxt = lib.qwn_decode(nxt, len(seq) - 1)
        if nxt < 0:
            raise RuntimeError(f"decode rc={nxt}")
        seq.append(nxt)
    t_plain = time.perf_counter() - t0
    plain = seq[len(PROMPT) + 1:]

    # ---- spec arm ----
    nxt = feed_prompt()
    seq = list(PROMPT) + [nxt]
    waves = 0
    committed = 0
    out8 = (ctypes.c_int * 8)()
    t0 = time.perf_counter()
    while committed < args.steps:
        toks = (ctypes.c_int * 8)(*([seq[-1]] + draft(seq, 7)))
        rc = lib.qwn_spec_wave(toks, len(seq) - 1, out8)
        if rc <= 0:
            raise RuntimeError(f"spec_wave rc={rc}")
        seq.extend(out8[j] for j in range(rc))
        committed += rc
        waves += 1
    t_spec = time.perf_counter() - t0
    spec = seq[len(PROMPT) + 1:len(PROMPT) + 1 + args.steps]

    n = min(len(plain), len(spec), args.steps)
    agree = sum(1 for i in range(n) if plain[i] == spec[i])
    first_div = next((i for i in range(n) if plain[i] != spec[i]), -1)
    rate = agree / n
    print(f"plain: {args.steps} tokens in {t_plain:.2f}s "
          f"({args.steps / t_plain:.2f} tok/s)")
    print(f"spec : {committed} tokens in {t_spec:.2f}s "
          f"({committed / t_spec:.2f} tok/s), {waves} waves, "
          f"{committed / waves:.2f} tokens/wave")
    print(f"speedup {args.steps / t_plain and (committed / t_spec) / (args.steps / t_plain):.2f}x  "
          f"agreement {rate * 100:.2f}% over {n}, first divergence at {first_div}")
    print("OVERALL:", "PASS" if rate >= 0.90 else "FAIL")
    lib.qwn_free()
    return 0 if rate >= 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
