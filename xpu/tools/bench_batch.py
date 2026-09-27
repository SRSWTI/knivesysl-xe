#!/usr/bin/env python3
"""bench_batch.py - aggregate throughput of the batched decode step.

Prefills B slots to --ctx tokens each (chunked prefill), then times
qwn_decode_batch at widths 1,2,4,8. The step is RC8-shaped: one weight
stream serves 8 rows, so per-step cost should be nearly flat in B and
aggregate tok/s should scale ~linearly. Also times the single-stream
qwn_decode (GEMV) path as the baseline.

TQ_XPU_DEV=0 TQ_XPU_SLOTS=8 python3 xpu/tools/bench_batch.py \
    --lib xpu/build/libforward_qwen_xpu.so --tqf ~/models/.../*.tqf
Add --output report.json to retain full-resolution samples and provenance.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import time

from bench_prefill import library_identity

TOK = 9707  # arbitrary in-vocab filler token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--output", help="optional full-resolution JSON report (text output retained)")
    args = ap.parse_args()
    if args.ctx < 8 or args.ctx % 8 or args.steps < 1 or args.warmup < 0:
        ap.error("context must be a positive multiple of eight; steps positive; warmup nonnegative")

    library_path = Path(os.path.abspath(args.lib))
    model_path = Path(args.tqf).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else None
    if output is not None and output in (library_path.resolve(), model_path):
        ap.error("--output must not overwrite a library or model input")
    env_keys = {key for key in os.environ if key.startswith(("TQ_", "ZE_", "SYCL_", "ONEAPI_"))}
    env_keys.update(("TQ_CTX", "TQ_XPU_DEV", "TQ_XPU_SLOTS", "ZE_AFFINITY_MASK",
                     "ONEAPI_DEVICE_SELECTOR", "SYCL_DEVICE_FILTER", "LD_LIBRARY_PATH"))
    report = {"schema": "bench_batch/v1", "status": "running", "pid": os.getpid(),
              "argv": list(sys.argv), "library_path": str(library_path),
              "model_path": str(model_path),
              "effective_env": {key: os.environ.get(key) for key in sorted(env_keys)},
              "configuration": {"ctx": args.ctx, "steps": args.steps,
                                "warmup": args.warmup, "widths": args.widths},
              "inputs": {"prefill_token_by_slot": "9707 + slot", "decode_token": TOK,
                         "prefill_chunk_max": 512, "greedy_feedback": False},
              "timing_scope": "native call wall time; GEMV assertion included; batch status assertion excluded; reset, prefill, warmup, position updates and ctypes position-array construction excluded",
              "cases": []}
    lib = None
    free = None
    current = None
    exit_code = 1
    try:
        report["library"] = library_identity(library_path)
        lib = ctypes.CDLL(os.path.abspath(args.lib))
        lib.qwn_free.argtypes = []
        lib.qwn_free.restype = None
        free = lib.qwn_free
        lib.qwn_init.argtypes = [ctypes.c_char_p]
        lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.qwn_set_slot.argtypes = [ctypes.c_int]
        lib.qwn_prefill_chunk.argtypes = [ctypes.POINTER(ctypes.c_int),
                                          ctypes.c_int, ctypes.c_int]
        lib.qwn_decode_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [
            ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.qwn_reset_state.argtypes = []
        lib.qwn_reset_state.restype = ctypes.c_int
        rc = lib.qwn_init(os.path.expanduser(args.tqf).encode())
        if rc != 0:
            raise RuntimeError(f"qwn_init rc={rc}")
        nslots = lib.qwn_num_slots()
        report["native"] = {"slots": int(nslots), "max_seq": int(lib.qwn_max_seq())}
        widths = args.widths
        if not widths or any(w < 1 or w > nslots for w in widths):
            raise ValueError(f"requested widths must be between one and the configured {nslots} slots")
        print(f"slots={nslots} ctx={args.ctx} steps={args.steps} widths={widths}")

        ctx = args.ctx

        def prepare_slots(width):
            if lib.qwn_reset_state() != 0:
                raise RuntimeError("native reset failed")
            for slot in range(width):
                if lib.qwn_set_slot(slot) != 0:
                    raise RuntimeError(f"cannot select slot {slot}")
                pos = 0
                while pos < ctx:
                    count = min(512, ctx - pos)
                    prompt = (ctypes.c_int * count)(*([TOK + slot] * count))
                    rc = lib.qwn_prefill_chunk(prompt, count, pos)
                    if rc != 0:
                        raise RuntimeError(f"prefill slot {slot} position {pos} failed: {rc}")
                    pos += count

        def begin_case(mode, width):
            case = {"mode": mode, "width": width, "ctx": ctx,
                    "warmup": args.warmup, "steps": args.steps, "status": "running",
                    "first_timed_position": ctx + args.warmup,
                    "last_timed_position": ctx + args.warmup + args.steps - 1,
                    "raw_latencies_ms": []}
            report["cases"].append(case)
            return case

        current = begin_case("GEMV", 1)
        prepare_slots(1)
        # single-stream baseline (GEMV path, slot 0)
        assert lib.qwn_set_slot(0) == 0
        pos = ctx
        for _ in range(args.warmup):
            assert lib.qwn_decode(TOK, pos) >= 0
            pos += 1
        lat = []
        current["raw_latencies_ms"] = lat
        for _ in range(args.steps):
            t0 = time.perf_counter_ns()
            assert lib.qwn_decode(TOK, pos) >= 0
            lat.append((time.perf_counter_ns() - t0) / 1e6)
            pos += 1
        base_ms = statistics.fmean(lat)
        current.update(status="passed", timed_steps=len(lat), mean_ms=base_ms,
                       total_timed_ms=sum(lat), per_stream_tokens_per_second=1000 / base_ms,
                       aggregate_tokens_per_second=1000 / base_ms)
        print(f"\n  {'mode':>12}  {'ms/step':>8}  {'per-stream':>11}  {'aggregate':>10}")
        print(f"  {'GEMV n=1':>12}  {base_ms:8.2f}  {1000/base_ms:11.2f}  "
              f"{1000/base_ms:10.2f}")

        Arr = lambda n, *v: (ctypes.c_int * n)(*v)
        rows = []
        for w in widths:
            current = begin_case("RC8", w)
            prepare_slots(w)
            slots = Arr(w, *range(w))
            poss = [ctx] * w
            toks = Arr(w, *([TOK] * w))
            out = Arr(w, *([0] * w))
            pv = Arr(w, *poss)
            for _ in range(args.warmup):
                rc = lib.qwn_decode_batch(slots, toks, pv, w, out)
                assert rc == w, f"batch rc={rc}"
                for i in range(w):
                    poss[i] += 1
                pv = Arr(w, *poss)
            lat = []
            current["raw_latencies_ms"] = lat
            for _ in range(args.steps):
                t0 = time.perf_counter_ns()
                rc = lib.qwn_decode_batch(slots, toks, pv, w, out)
                lat.append((time.perf_counter_ns() - t0) / 1e6)
                assert rc == w
                for i in range(w):
                    poss[i] += 1
                pv = Arr(w, *poss)
            ms = statistics.fmean(lat)
            per, agg = 1000 / ms, w * 1000 / ms
            current.update(status="passed", timed_steps=len(lat), mean_ms=ms,
                           total_timed_ms=sum(lat), per_stream_tokens_per_second=per,
                           aggregate_tokens_per_second=agg)
            rows.append((w, ms, per, agg))
            print(f"  {'RC8 n=' + str(w):>12}  {ms:8.2f}  {per:11.2f}  {agg:10.2f}")
        best = max(rows, key=lambda r: r[3])
        print(f"\n  peak aggregate {best[3]:.1f} tok/s at n={best[0]} "
              f"({best[3]/(1000/base_ms):.2f}x single-stream)")
        report["peak"] = {"width": best[0], "mean_ms": best[1],
                          "aggregate_tokens_per_second": best[3],
                          "speedup_vs_single_stream": best[3] / (1000 / base_ms)}
        report["status"] = "passed"
        exit_code = 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if current is not None and current["status"] == "running":
            current.update(status="failed", error=report["error"])
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        try:
            if free is not None:
                free()
        except Exception as exc:
            report.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
            print(report["cleanup_error"], file=sys.stderr, flush=True)
            exit_code = 1
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
