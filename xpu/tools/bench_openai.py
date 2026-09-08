#!/usr/bin/env python3
"""bench_openai.py - raw throughput/prefill bench for any OpenAI-compatible
server (vLLM-XPU vendor baseline or our own shell). No speculation flattery:
measures TTFT (streamed first token) and steady ITL, at several context
lengths and concurrency levels. Each request carries a unique random prefix
so server-side prefix caching cannot contaminate cold-prefill numbers.

python3 xpu/tools/bench_openai.py --base-url http://127.0.0.1:8000/v1 \
    --model qwen38-xpu --ctx 512,2048,8192 --conc 1,4,8 --gen 128
"""
import argparse
import concurrent.futures as cf
import json
import random
import statistics
import time
import urllib.request


def one_request(base, model, prompt_words, gen):
    # unique prefix defeats prefix caching; ~1 token per word
    salt = " ".join(str(random.randrange(10, 99)) for _ in range(8))
    prompt = salt + " " + "hello " * prompt_words
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": gen,
        "temperature": 0, "stream": True,
        "ignore_eos": True, "min_tokens": gen,
    }).encode()
    req = urllib.request.Request(base + "/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    tfirst = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        # readline() returns per SSE event; `for line in r` buffers 8 KB
        # chunks and destroys ITL timing.
        while True:
            line = r.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            if b"[DONE]" in line:
                break
            ntok += 1
            if tfirst is None:
                tfirst = time.perf_counter()
    tend = time.perf_counter()
    ttft = (tfirst - t0) if tfirst else float("nan")
    itl = (tend - tfirst) / max(1, ntok - 1) if tfirst else float("nan")
    return ttft, itl, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", default="512,2048,8192")
    ap.add_argument("--conc", default="1,4,8")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    print(f"{'ctx':>6} {'conc':>5} {'TTFT_s':>8} {'prefill_tok_s':>13} "
          f"{'ITL_ms':>8} {'dec_tok_s':>10} {'agg_tok_s':>10}")
    for ctx in (int(x) for x in args.ctx.split(",")):
        for conc in (int(x) for x in args.conc.split(",")):
            # warm one round, then measure reps rounds
            ttfts, itls = [], []
            for rep in range(args.reps + 1):
                with cf.ThreadPoolExecutor(conc) as ex:
                    futs = [ex.submit(one_request, args.base_url, args.model,
                                      ctx, args.gen) for _ in range(conc)]
                    res = [f.result() for f in futs]
                if rep == 0:
                    continue
                ttfts += [r[0] for r in res]
                itls += [r[1] for r in res]
            ttft = statistics.median(ttfts)
            itl = statistics.median(itls)
            dec = 1.0 / itl if itl > 0 else 0.0
            print(f"{ctx:>6} {conc:>5} {ttft:>8.2f} {ctx / ttft:>13.0f} "
                  f"{itl * 1e3:>8.1f} {dec:>10.2f} {dec * conc:>10.1f}")


if __name__ == "__main__":
    main()
