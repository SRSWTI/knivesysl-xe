#!/usr/bin/env python3
"""Sustained single-stream decode benchmark for libforward_qwen_xpu.so.

Uses pinned-corpus teacher forcing so every format/device receives the same token
and position sequence. Initialization, repacking, prompt feed, and warm-up are
reported separately and excluded from the timed decode window.
"""

import argparse
import ctypes
import os
import statistics
import time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS = os.path.join(HERE, "build-qwen", "tf_corpus_e3cdb42.txt")


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tqf", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--corpus", default=CORPUS,
                        help="pinned text corpus used for identical teacher-forced inputs")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=512)
    # Context setup uses qwn_prefill_chunk, which is state-equivalent to the
    # per-token loop (proven by xpu/tools/prefill_check.py) but ~7x faster.
    # At 131k the per-token loop costs over an hour of pure setup.
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    with open(os.path.expanduser(args.corpus)) as corpus_file:
        tokens = tokenizer(corpus_file.read(), add_special_tokens=False).input_ids
    required = args.prompt_tokens + args.warmup + args.steps
    if len(tokens) < required:
        raise RuntimeError(f"pinned corpus has {len(tokens)} tokens, need {required}")

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_init.restype = ctypes.c_int
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_decode.restype = ctypes.c_int
    lib.qwn_free.argtypes = []
    lib.qwn_free.restype = None
    lib.qwn_prefill_chunk.argtypes = [ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_int, ctypes.c_int]
    lib.qwn_prefill_chunk.restype = ctypes.c_int

    started = time.perf_counter()
    rc = lib.qwn_init(os.path.expanduser(args.tqf).encode())
    init_seconds = time.perf_counter() - started
    if rc != 0:
        raise RuntimeError(f"qwn_init failed: {rc}")

    try:
        prompt_started = time.perf_counter()
        chunk = max(8, (args.chunk // 8) * 8)
        done = 0
        while args.prompt_tokens - done >= chunk:
            buf = (ctypes.c_int * chunk)(*tokens[done:done + chunk])
            result = lib.qwn_prefill_chunk(buf, chunk, done)
            if result < 0:
                raise RuntimeError(f"prefill chunk failed at {done}: {result}")
            done += chunk
        for position in range(done, args.prompt_tokens):
            result = lib.qwn_decode(int(tokens[position]), position)
            if result < 0:
                raise RuntimeError(f"prompt decode failed at {position}: {result}")
        prompt_seconds = time.perf_counter() - prompt_started

        position = args.prompt_tokens
        for _ in range(args.warmup):
            result = lib.qwn_decode(int(tokens[position]), position)
            if result < 0:
                raise RuntimeError(f"warm-up decode failed at {position}: {result}")
            position += 1

        latencies_ms = []
        for _ in range(args.steps):
            step_started = time.perf_counter_ns()
            result = lib.qwn_decode(int(tokens[position]), position)
            step_ended = time.perf_counter_ns()
            if result < 0:
                raise RuntimeError(f"timed decode failed at {position}: {result}")
            latencies_ms.append((step_ended - step_started) / 1.0e6)
            position += 1

        total_seconds = sum(latencies_ms) / 1000.0
        tokens_per_second = args.steps / total_seconds
        print(f"INIT seconds={init_seconds:.3f}")
        print(f"PROMPT tokens={args.prompt_tokens} seconds={prompt_seconds:.3f} "
              f"tok_s={args.prompt_tokens / prompt_seconds:.3f}")
        print(f"DECODE steps={args.steps} warmup={args.warmup} seconds={total_seconds:.3f} "
              f"tok_s={tokens_per_second:.3f}")
        print(f"LATENCY_MS mean={statistics.fmean(latencies_ms):.3f} "
              f"p50={percentile(latencies_ms, 0.50):.3f} "
              f"p90={percentile(latencies_ms, 0.90):.3f} "
              f"p99={percentile(latencies_ms, 0.99):.3f} "
              f"min={min(latencies_ms):.3f} max={max(latencies_ms):.3f}")
    finally:
        lib.qwn_free()


if __name__ == "__main__":
    main()
