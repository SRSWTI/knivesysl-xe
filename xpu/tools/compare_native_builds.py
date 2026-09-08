#!/usr/bin/env python3
"""Cross-process, real-model teacher-forced and independent greedy numerical gate.

Run record to completion BEFORE check; each invocation loads exactly one library
and one model, resetting state between every prompt/mode. Example (same shell):

  export TQ_CTX=2048 TQ_XPU_SLOTS=1 TQ_XPU_GEMM_SPLITS=1
  python3 xpu/tools/compare_native_builds.py --lib BASELINE.so --tqf MODEL.tqf \
      --model-dir TOKENIZER_DIR --record --reference baseline.json --output record.json
  python3 xpu/tools/compare_native_builds.py --lib CANDIDATE.so --tqf MODEL.tqf \
      --model-dir TOKENIZER_DIR --check --reference baseline.json --output check.json

Add --exact to check unchanged-arithmetic changes. Normal acceptance is >=90%
aggregate teacher-forced top1 agreement; greedy agreement is a separate diagnostic,
not a substitute for that gate. Exact acceptance requires IDs AND float32 argmax
bits in both modes. Neither mode claims full-logit, hidden-state, or KV equality.
Use --allow-flag NAME on check only for intentionally changed kernel flags; weight
format/tier, context, device selection, and all other captured flags must match.
References are exclusive-create, never silently overwritten. No EOS early stop.
"""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback

from paged_parity import ARITHMETIC_ENV
from prefill_batch_check import BRANCHES, identity, ints, require, runtime_flags, token_sets, zero


SCHEMA = "compare_native_builds/v1"
THRESHOLD = 0.90
OPTIMIZATION_FLAGS = (
    "TQ_XPU_GEMM_SPLITS", "TQ_XPU_ATTN_SIMD16", "TQ_XPU_ATTN_GROUPED",
    "TQ_XPU_ATTN_DPAS", "TQ_XPU_DELTA_FAST", "TQ_XPU_DN_CHUNK",
    "TQ_XPU_PREFILL_XMX",
)
TIER_FLAGS = ("TQ_XPU_K64", "TQ_XPU_W4A4", "TQ_XPU_W4A8", "TQ_XPU_W8",
              "TQ_XPU_W4_K16")
NATIVE_FIELDS = ("hidden_size", "intermediate_size", "vocab_size", "num_layers",
                 "num_attention_heads", "num_key_value_heads", "head_dim",
                 "max_seq", "num_slots", "paged_enabled")


def flags():
    result = runtime_flags()
    for name in (*ARITHMETIC_ENV, *OPTIMIZATION_FLAGS, *TIER_FLAGS,
                 "TQ_XPU_PAGED", "TQ_XPU_DEV"):
        result[name] = os.environ.get(name)
    return dict(sorted(result.items()))


def save(path, report, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x" if exclusive else "w") as stream:
        stream.write(text)


def model_header(path):
    with path.open("rb") as stream:
        raw = stream.read(116)
    require(len(raw) == 116 and raw[:4] == b"TQF1", "unsupported or truncated TQF1 header")
    values = struct.unpack("<16I", raw[8:72])
    require(values[6] > 0, "invalid TQF vocabulary")
    return {"magic": "TQF1", "header_bytes": struct.unpack("<I", raw[4:8])[0],
            "fixed_header_hex": raw.hex(), "vocab_size": values[6],
            "flags": values[0], "model_family": values[1], "non_quant_dtype": values[2]}


def bind(path):
    # Bind before init so the caller owns qwn_free even when init returns failure.
    lib = ctypes.CDLL(str(path))
    ip = ctypes.POINTER(ctypes.c_int)
    bindings = [
        ("qwn_init", [ctypes.c_char_p], ctypes.c_int),
        ("qwn_free", [], None),
        ("qwn_reset_state", [], ctypes.c_int),
        ("qwn_set_slot", [ctypes.c_int], ctypes.c_int),
        ("qwn_decode", [ctypes.c_int, ctypes.c_int], ctypes.c_int),
        ("qwn_decode_batch", [ip, ip, ip, ctypes.c_int, ip], ctypes.c_int),
        ("qwn_prefill_chunk", [ip, ctypes.c_int, ctypes.c_int], ctypes.c_int),
        ("qwn_last_argmax_logit", [], ctypes.c_float),
        ("qwn_layer_type", [ctypes.c_int], ctypes.c_int),
        ("qwn_attn_branch_counts", [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int],
         ctypes.c_int),
    ]
    bindings.extend(("qwn_" + name, [], ctypes.c_int) for name in NATIVE_FIELDS)
    for name, arguments, result in bindings:
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result
    return lib


def counters(lib):
    values = (ctypes.c_ulonglong * len(BRANCHES))()
    zero(lib.qwn_attn_branch_counts(values, len(values)), "qwn_attn_branch_counts")
    return dict(zip(BRANCHES, map(int, values)))


def schedule(tokens, chunk):
    # Existing fixtures leave one token for decode and have RC8-aligned prefixes.
    return [{"position": position, "count": min(chunk, len(tokens) - 1 - position)}
            for position in range(0, len(tokens) - 1, chunk)]


def run_chain(lib, prompt, config, destination, teacher_inputs=None):
    zero(lib.qwn_reset_state(), "qwn_reset_state")
    zero(lib.qwn_set_slot(0), "qwn_set_slot(0)")
    before = counters(lib)
    destination.update(status="running", trace=[], finite=True,
                       prefill_schedule=schedule(prompt["tokens"], config["chunk"]))
    ids = prompt["tokens"]
    for segment in destination["prefill_schedule"]:
        position, count = segment["position"], segment["count"]
        zero(lib.qwn_prefill_chunk(ints(ids[position:position + count]), count, position),
             f"qwn_prefill_chunk({position},{count})")
    after_prefill = counters(lib)
    delta = {name: after_prefill[name] - before[name] for name in before}
    destination["prefill_evidence"] = {
        "calls": len(destination["prefill_schedule"]),
        "rows": sum(segment["count"] for segment in destination["prefill_schedule"]),
        "branch_delta": delta,
    }
    requested = config["runtime_flags"].get("TQ_XPU_PREFILL_XMX")
    require(delta["prefill"] > 0, "chunk prefill exercised no attention prefill branch")
    if requested == "0":
        require(delta["prefill_scalar"] > 0 and delta["prefill_xmx"] == 0,
                "explicit scalar prefill mode was not exercised")
    else:
        require(delta["prefill_xmx"] > 0,
                "chunk mode exercised no optimized XMX attention; use chunk>=128 or "
                "explicit TQ_XPU_PREFILL_XMX=1 for small chunks (0 for intentional scalar reference)")
        if requested == "1":
            require(delta["prefill_scalar"] == 0, "forced XMX prefill silently used scalar attention")
    token = ids[-1]
    for step in range(config["steps"]):
        position = len(ids) - 1 + step
        if teacher_inputs is not None:
            token = teacher_inputs[step]
        if config["decode_api"] == "single":
            output = int(lib.qwn_decode(token, position))
        else:
            values = (ctypes.c_int * 1)()
            rc = int(lib.qwn_decode_batch(ints([0]), ints([token]), ints([position]), 1, values))
            require(rc == 1, f"qwn_decode_batch expected 1, got {rc}")
            output = int(values[0])
        logit = float(lib.qwn_last_argmax_logit())
        finite = math.isfinite(logit)
        row = {"step": step, "position": position, "input_id": token,
               "argmax_id": output, "argmax_logit": logit if finite else None,
               "argmax_logit_bits": struct.pack("<f", logit).hex(), "finite": finite}
        destination["trace"].append(row)
        destination["finite"] = destination["finite"] and finite
        require(0 <= output < config["format"]["vocab_size"],
                f"{prompt['name']} step {step}: invalid native argmax {output}")
        require(finite, f"{prompt['name']} step {step}: nonfinite argmax logit")
        token = output
    after = counters(lib)
    destination.update(status="complete", branch_delta={name: after[name] - before[name]
                                                         for name in before})


def validate_records(report):
    """Reject partial/malformed reference traces before using any forced token."""
    require(report.get("schema") == SCHEMA and report.get("status") == "recorded",
            "reference must be a completed compare_native_builds/v1 record")
    config = report["config"]
    require(report.get("finite") is True, "reference not finite")
    require(len(report["records"]) == len(config["prompts"]), "reference prompt count mismatch")
    for prompt, record in zip(config["prompts"], report["records"]):
        require(record["name"] == prompt["name"], "reference prompt order mismatch")
        for mode in ("greedy", "teacher"):
            chain = record[mode]
            require(chain["status"] == "complete" and chain["finite"] is True,
                    f"incomplete reference {prompt['name']}/{mode}")
            require(chain["prefill_schedule"] == schedule(prompt["tokens"], config["chunk"]),
                    "reference chunk schedule mismatch")
            require(len(chain["trace"]) == config["steps"], "reference step count mismatch")
            for step, row in enumerate(chain["trace"]):
                expected_input = prompt["tokens"][-1] if step == 0 else \
                    record["greedy"]["trace"][step - 1]["argmax_id"]
                require(row["step"] == step and row["position"] == len(prompt["tokens"]) - 1 + step
                        and row["input_id"] == expected_input, "reference forced input/position mismatch")
                require(type(row["argmax_id"]) is int and
                        0 <= row["argmax_id"] < config["format"]["vocab_size"],
                        "reference argmax outside vocabulary")
                value = row["argmax_logit"]
                require(row["finite"] is True and isinstance(value, (int, float)) and
                        math.isfinite(value) and struct.pack("<f", value).hex() == row["argmax_logit_bits"],
                        "reference float value/bits mismatch or nonfinite")
        require(record["teacher"]["trace"] == record["greedy"]["trace"],
                "baseline reset replay was not bit-exact; reference is not deterministic")


def compare_configuration(reference, current, allowed):
    expected = reference["config"]
    for key in sorted(set(expected) | set(current)):
        if key != "runtime_flags":
            require(expected.get(key) == current.get(key), f"reference configuration mismatch: {key}")
    left, right = expected["runtime_flags"], current["runtime_flags"]
    differences = {name: {"reference": left.get(name), "candidate": right.get(name)}
                   for name in sorted(set(left) | set(right)) if left.get(name) != right.get(name)}
    forbidden = sorted(set(differences) - set(allowed))
    require(not forbidden, f"non-authorized runtime flag differences: {forbidden}")
    return differences


def compare_records(reference, candidate):
    results = {}
    for mode in ("teacher", "greedy"):
        per_prompt = []
        for left, right in zip(reference["records"], candidate["records"]):
            a, b = left[mode]["trace"], right[mode]["trace"]
            require(len(a) == len(b) == candidate["config"]["steps"], "incomplete comparison")
            require(left["name"] == right["name"], "comparison prompt mismatch")
            if mode == "teacher":
                require([r["input_id"] for r in a] == [r["input_id"] for r in b],
                        "teacher-forced input divergence")
            misses = [x["step"] for x, y in zip(a, b) if x["argmax_id"] != y["argmax_id"]]
            bit_misses = [x["step"] for x, y in zip(a, b)
                          if x["argmax_logit_bits"] != y["argmax_logit_bits"]]
            matches = len(a) - len(misses)
            per_prompt.append({"name": left["name"], "sample_size": len(a), "matches": matches,
                               "agreement": matches / len(a), "passes_90_percent": matches / len(a) >= THRESHOLD,
                               "token_mismatch_steps": misses, "logit_bit_mismatch_steps": bit_misses,
                               "exact": not misses and not bit_misses})
        total = sum(row["sample_size"] for row in per_prompt)
        matches = sum(row["matches"] for row in per_prompt)
        results[mode] = {"per_prompt": per_prompt, "sample_size": total, "matches": matches,
                         "agreement": matches / total, "passes_90_percent": matches / total >= THRESHOLD,
                         "minimum_prompt_agreement": min(row["agreement"] for row in per_prompt),
                         "exact": all(row["exact"] for row in per_prompt)}
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tqf", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--record", action="store_true")
    modes.add_argument("--check", action="store_true")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--steps", type=int, default=32, help="continuation steps per mode (1..512)")
    parser.add_argument("--chunk", type=int, default=128, help="RC8 prefill maximum (8..512, multiple of 8)")
    parser.add_argument("--decode-api", choices=("single", "batch"), default="single",
                        help="single qwn_decode or width-one qwn_decode_batch; must match reference")
    parser.add_argument("--allow-flag", action="append", choices=OPTIMIZATION_FLAGS, default=[],
                        help="explicit check-only authorization for one changed optimization flag")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 512 or not 8 <= args.chunk <= 512 or args.chunk % 8:
        parser.error("steps must be in [1,512]; chunk must be a multiple of 8 in [8,512]")
    if args.record and args.allow_flag:
        parser.error("--allow-flag is check-only")
    output = Path(args.output).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()
    sources = {Path(path).expanduser().resolve() for path in (args.lib, args.tqf, args.model_dir)}
    if output == reference_path or output in sources or reference_path in sources:
        parser.error("output and reference must be separate from each other and all inputs")
    report = {"schema": SCHEMA, "status": "running", "mode": "record" if args.record else "check",
              "pid": os.getpid(), "records": [], "finite": False,
              "policy": {"exact": args.exact, "min_agreement": THRESHOLD,
                         "gate_scope": "aggregate teacher-forced top1; exact also gates both modes' IDs and float32 bits",
                         "greedy_tolerant_role": "independent diagnostic, not the teacher-forced gate",
                         "finite_scope": "public ABI argmax logit only, not full logits or hidden state"},
              "reference_path": str(reference_path), "allowed_flag_differences": sorted(set(args.allow_flag))}
    lib = None
    start = time.perf_counter()
    try:
        require(not args.record or not reference_path.exists(), "reference already exists; choose a new path")
        reference = None
        if args.check:
            reference = json.loads(reference_path.read_text())
            validate_records(reference)
            report["reference_identity"] = identity(reference_path)
            report["reference_library"] = reference["library"]
        report["library"] = identity(args.lib)
        model = identity(args.tqf)
        header = model_header(Path(model["path"]))
        prompts = token_sets(Path(args.model_dir).expanduser().resolve(strict=True), header["vocab_size"])
        config = {"model": model, "format": header,
                  "tokenizer_directory": str(Path(args.model_dir).expanduser().resolve(strict=True)),
                  "prompts": prompts, "steps": args.steps, "chunk": args.chunk,
                  "decode_api": args.decode_api, "slot": 0, "runtime_flags": flags(),
                  "tier_flags": {name: os.environ.get(name) for name in TIER_FLAGS}}
        report["config"] = config
        if reference is not None:
            report["flag_differences"] = compare_configuration(reference, config, args.allow_flag)
            report["binary_changed"] = reference["library"]["sha256"] != report["library"]["sha256"]
        lib = bind(report["library"]["path"])
        zero(lib.qwn_init(model["path"].encode()), "qwn_init")
        native = {name: int(getattr(lib, "qwn_" + name)()) for name in NATIVE_FIELDS}
        native["layer_types"] = [int(lib.qwn_layer_type(layer)) for layer in range(native["num_layers"])]
        report["native"] = native
        require(native["vocab_size"] == header["vocab_size"], "native vocabulary/TQF mismatch")
        require(native["num_slots"] >= 1, "no native slot available")
        require(all(len(prompt["tokens"]) + args.steps - 1 <= native["max_seq"] for prompt in prompts),
                "prompt plus continuation exceeds native context; increase TQ_CTX")
        if reference is not None:
            require(reference["native"] == native, "native model/context/layout configuration mismatch")
        for index, prompt in enumerate(prompts):
            record = {"name": prompt["name"], "greedy": {}, "teacher": {}}
            report["records"].append(record)
            if args.record:
                run_chain(lib, prompt, config, record["greedy"])
                forced = [row["input_id"] for row in record["greedy"]["trace"]]
                run_chain(lib, prompt, config, record["teacher"], forced)
                require(record["greedy"]["trace"] == record["teacher"]["trace"],
                        f"{prompt['name']}: baseline reset replay is not bit-exact")
            else:
                forced = [row["input_id"] for row in reference["records"][index]["teacher"]["trace"]]
                run_chain(lib, prompt, config, record["teacher"], forced)
                run_chain(lib, prompt, config, record["greedy"])
            save(output, report)
        report["finite"] = True
        require(config["runtime_flags"] == flags(), "runtime flags changed during execution")
        if args.record:
            report["status"] = "recorded"
            validate_records(report)
            report["comparison"] = compare_records(report, report)
            report["comparison_scope"] = "baseline reset replay/self; NOT cross-build evidence"
            save(reference_path, report, exclusive=True)
        else:
            report["comparison"] = compare_records(reference, report)
            exact = all(result["exact"] for result in report["comparison"].values())
            passed = report["comparison"]["teacher"]["passes_90_percent"] and (not args.exact or exact)
            report["status"] = "passed" if passed else "failed"
            if not passed:
                report["error"] = "exact ID/logit-bit mismatch" if args.exact else "teacher-forced aggregate top1 below 90%"
        return 0 if report["status"] in ("recorded", "passed") else 1
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        print(report["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        if lib is not None:
            lib.qwn_free()
        report["duration_seconds"] = time.perf_counter() - start
        save(output, report)
        print(json.dumps({"status": report["status"], "output": str(output),
                          "reference": str(reference_path), "finite": report["finite"],
                          "duration_seconds": report["duration_seconds"]}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
