#!/usr/bin/env python3
"""bench_prefill.py - native prompt ingestion: chunked prefill vs a decode loop.

Chunked timing excludes the final norm/lm_head and HTTP overhead; it is not
end-to-end time to first token. The optional per-token loop executes a complete
decode, including its output head, at every position and is a separate diagnostic.

  python3 xpu/tools/bench_prefill.py --lib xpu/build/libforward_qwen_xpu.so \
      --tqf ~/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROMPT = [151644, 872, 198, 9707, 11, 1246, 525, 498, 30, 151645, 198]


def library_identity(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != \
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"library changed while hashing: {path}")
    return {"path": str(path), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
            "device": after.st_dev, "inode": after.st_ino, "sha256": digest.hexdigest()}


def require_zero(rc, operation):
    if rc != 0:
        raise RuntimeError(f"{operation} failed rc={rc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--ctx", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--loop-ref", type=int, default=512,
                    help="also time the per-token loop up to this many tokens")
    ap.add_argument("--output", help="optional full-resolution JSON report (text output retained)")
    args = ap.parse_args()
    if args.chunk <= 0 or args.chunk % 8 or any(n <= 0 or n % 8 for n in args.ctx):
        ap.error("--chunk and every --ctx must be positive multiples of 8; no rounding is performed")
    if args.loop_ref < 0:
        ap.error("--loop-ref must be nonnegative (0 disables the reference loop)")

    library_path = Path(args.lib).expanduser().resolve()
    model_path = Path(args.tqf).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else None
    if output in (library_path, model_path):
        ap.error("--output must not overwrite a library or model input")
    report = {"schema": "bench_prefill/v1", "status": "running", "pid": os.getpid(),
              "library_path": str(library_path), "model_path": str(model_path),
              "tq_env": {key: value for key, value in sorted(os.environ.items()) if key.startswith("TQ_")},
              "configuration": {"chunk": args.chunk, "contexts": args.ctx, "loop_ref": args.loop_ref},
              "prompt_template": PROMPT, "prompt_construction": "repeat template and truncate to requested tokens",
              "timing_scope": "reset excluded; prompt construction excluded; chunk ctypes array construction included; native calls included",
              "cases": []}
    lib = None
    free = None
    current = None
    exit_code = 1
    try:
        report["library"] = library_identity(library_path)
        lib = ctypes.CDLL(str(library_path))
        lib.qwn_free.argtypes = []
        lib.qwn_free.restype = None
        free = lib.qwn_free
        lib.qwn_init.argtypes = [ctypes.c_char_p]
        lib.qwn_init.restype = ctypes.c_int
        lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.qwn_decode.restype = ctypes.c_int
        lib.qwn_prefill_chunk.argtypes = [ctypes.POINTER(ctypes.c_int),
                                          ctypes.c_int, ctypes.c_int]
        lib.qwn_prefill_chunk.restype = ctypes.c_int
        lib.qwn_reset_state.argtypes = []
        lib.qwn_reset_state.restype = ctypes.c_int
        require_zero(lib.qwn_init(str(model_path).encode()), "qwn_init")

        def prompt_of(n):
            return (PROMPT * ((n // len(PROMPT)) + 1))[:n]

        # Reference: per-token loop. Keep reset outside the original timed region.
        ref = {}
        if args.loop_ref > 0:
            toks = prompt_of(args.loop_ref)
            current = {"mode": "per_token_loop", "status": "running", "prompt_tokens": len(toks),
                       "chunk_tokens": None, "planned_calls": len(toks), "calls": 0, "consumed_tokens": 0}
            report["cases"].append(current)
            require_zero(lib.qwn_reset_state(), "qwn_reset_state(loop)")
            t0 = time.perf_counter()
            for i, t in enumerate(toks):
                rc = lib.qwn_decode(int(t), i)
                if rc < 0:
                    current.update(calls=i + 1, consumed_tokens=i,
                                   seconds=time.perf_counter() - t0, native_rc=rc)
                    raise RuntimeError(f"qwn_decode(pos={i}) failed rc={rc}")
            dt = time.perf_counter() - t0
            if dt <= 0:
                raise RuntimeError("nonpositive per-token measurement duration")
            ref[args.loop_ref] = dt
            current.update(status="passed", seconds=dt, tokens_per_second=len(toks) / dt,
                           seconds_per_token=dt / len(toks), calls=len(toks), consumed_tokens=len(toks))
            print(f"per-token loop: {args.loop_ref} tokens in {dt:.2f} s "
                  f"({args.loop_ref/dt:.1f} tok/s)")

        print(f"\nchunked prefill (chunk={args.chunk})")
        print(f"{'ctx':>8} {'prefill s':>9} {'tok/s':>9} {'vs loop':>9}")
        loop_rate = (args.loop_ref / ref[args.loop_ref]) if ref else None
        for n in args.ctx:
            toks = prompt_of(n)
            current = {"mode": "chunked_prefill", "status": "running", "prompt_tokens": len(toks),
                       "chunk_tokens": args.chunk, "planned_calls": (n + args.chunk - 1) // args.chunk,
                       "calls": 0, "consumed_tokens": 0}
            report["cases"].append(current)
            require_zero(lib.qwn_reset_state(), f"qwn_reset_state(ctx={n})")
            t0 = time.perf_counter()
            pos = 0
            rc = 0
            while pos < n:
                take = min(args.chunk, n - pos)
                arr = (ctypes.c_int * take)(*toks[pos:pos + take])
                rc = lib.qwn_prefill_chunk(arr, take, pos)
                if rc != 0:
                    current.update(calls=pos // args.chunk + 1, consumed_tokens=pos,
                                   seconds=time.perf_counter() - t0, native_rc=rc)
                    raise RuntimeError(f"qwn_prefill_chunk(ctx={n},pos={pos},count={take}) failed rc={rc}")
                pos += take
            dt = time.perf_counter() - t0
            if dt <= 0:
                raise RuntimeError("nonpositive chunked measurement duration")
            rate = n / dt
            current.update(status="passed", seconds=dt, tokens_per_second=rate,
                           seconds_per_token=dt / n, calls=current["planned_calls"], consumed_tokens=pos,
                           speedup_vs_loop=rate / loop_rate if loop_rate else None)
            speed = f"{rate/loop_rate:.0f}x" if loop_rate else "-"
            print(f"{n:>8} {dt:>9.2f} {rate:>9.0f} {speed:>9}")
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
