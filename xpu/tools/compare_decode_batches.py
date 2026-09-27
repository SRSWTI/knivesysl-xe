#!/usr/bin/env python3
"""Exact cross-process real-model batch-decode replay, including width eight.

Use the SAME device/runtime flags for both invocations, sequentially:
  export TQ_XPU_DEV=1 TQ_XPU_SLOTS=8 TQ_CTX=2048
  python3 xpu/tools/compare_decode_batches.py --lib BASELINE.so --tqf MODEL.tqf \
      --model-dir TOKENIZER_DIR --record --reference baseline.json --output record.json
  python3 xpu/tools/compare_decode_batches.py --lib CANDIDATE.so --tqf MODEL.tqf \
      --model-dir TOKENIZER_DIR --check --reference baseline.json --output check.json

All widths 1,2,4,8 run in one initialization, with a full reset/replay per width.
Rows have distinct real-token prompts, mixed positions, permuted/noncontiguous
slot selection, and a populated slot reset/reused while surviving slots continue.
Every batch records all output IDs, ONLY the last row's public argmax-logit bits,
and qwn_get_slot before/after the call. --kv-hashes also compares logical KV bytes
by per-slot/layer SHA-256 before and after slot reuse; physical block IDs are not
compared. There is no public state_pos getter: tracked positions are NOT reported
as observed native positions. Exact is mandatory; no tolerance or EOS early stop.
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

from compare_native_builds import (
    NATIVE_FIELDS, bind, compare_configuration, flags, model_header, save, schedule,
)
from prefill_batch_check import digest, identity, ints, require, token_sets, zero


SCHEMA = "compare_decode_batches/v1"
WIDTHS = (1, 2, 4, 8)
SLOT_ORDER = (7, 1, 6, 0, 5, 2, 4, 3)


def fixtures(model_dir, vocab):
    sources = token_sets(model_dir, vocab)
    prompts = []
    for generation in range(2):
        for slot in range(8):
            source = sources[(slot + generation) % len(sources)]
            count = (128 + 64 * slot if generation == 0 else 64 + 8 * slot) + 1
            offset = slot * 11 + generation * 23
            stream = source["tokens"] * ((offset + count) // len(source["tokens"]) + 1)
            tokens = stream[offset:offset + count]
            prompts.append({"name": f"slot{slot}-generation{generation}", "slot": slot,
                            "generation": generation, "source": source["name"],
                            "source_sha256": source["sha256"], "source_offset": offset,
                            "tokens": tokens, "sha256": digest(ints(tokens))})
    require(len({tuple(p["tokens"]) for p in prompts}) == len(prompts),
            "real-token prompt fixtures must all be distinct")
    return prompts


def row_order(width, step):
    slots = list(SLOT_ORDER[:width])
    shift = step % width
    slots = slots[shift:] + slots[:shift]
    return slots[::-1] if step % 2 else slots


def bind_extra(lib):
    for name, arguments, result in (
        ("qwn_get_slot", [], ctypes.c_int),
        ("qwn_reset_slot", [ctypes.c_int], ctypes.c_int),
        ("qwn_debug_kv_bytes", [ctypes.c_int], ctypes.c_size_t),
        ("qwn_debug_kv_copy", [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
    ):
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result


def logical_kv(lib, positions):
    result = {}
    layers = [layer for layer in range(lib.qwn_num_layers()) if lib.qwn_layer_type(layer) == 2]
    require(layers, "no full-attention layers available for logical KV evidence")
    for slot, committed in sorted(positions.items()):
        size = int(lib.qwn_debug_kv_bytes(committed))
        require(size > 0, "invalid logical KV byte size")
        buffer = (ctypes.c_ubyte * size)()
        hashes = {}
        for layer in layers:
            zero(lib.qwn_debug_kv_copy(slot, layer, committed, buffer, size),
                 f"qwn_debug_kv_copy(slot={slot},layer={layer},committed={committed})")
            hashes[str(layer)] = digest(buffer)
        result[str(slot)] = {"committed": committed, "bytes_per_layer": size, "sha256": hashes}
    return result


def establish(lib, prompt, chunk):
    slot = prompt["slot"]
    zero(lib.qwn_set_slot(slot), f"qwn_set_slot({slot})")
    for segment in schedule(prompt["tokens"], chunk):
        position, count = segment["position"], segment["count"]
        tokens = prompt["tokens"][position:position + count]
        zero(lib.qwn_prefill_chunk(ints(tokens), count, position),
             f"qwn_prefill_chunk(slot={slot},pos={position},count={count})")


def run_case(lib, config, width, destination, forced=None):
    zero(lib.qwn_reset_state(), "qwn_reset_state")
    participants = list(SLOT_ORDER[:width])
    prompts = {(p["slot"], p["generation"]): p for p in config["prompts"]}
    positions, next_tokens = {}, {}
    destination.update(width=width, status="running", reset_active_slot=int(lib.qwn_get_slot()),
                       phases=[], native_state_pos=None,
                       state_pos_scope="tracked from successful calls; no public native getter")
    for slot in participants:
        prompt = prompts[slot, 0]
        establish(lib, prompt, config["chunk"])
        positions[slot] = len(prompt["tokens"]) - 1
        next_tokens[slot] = prompt["tokens"][-1]
    global_step = 0
    for phase_index, (name, steps) in enumerate((("initial", config["steps"]),
                                                ("slot_reuse", config["reuse_steps"]))):
        phase = {"name": name, "trace": []}
        destination["phases"].append(phase)
        if phase_index:
            reused = participants[-1]
            before = int(lib.qwn_get_slot())
            zero(lib.qwn_reset_slot(reused), f"qwn_reset_slot({reused})")
            phase["reset_slot"] = {"slot": reused, "active_before": before,
                                   "active_after": int(lib.qwn_get_slot())}
            prompt = prompts[reused, 1]
            establish(lib, prompt, config["chunk"])
            positions[reused] = len(prompt["tokens"]) - 1
            next_tokens[reused] = prompt["tokens"][-1]
        phase["starting_positions"] = {str(s): positions[s] for s in sorted(positions)}
        for step in range(steps):
            slots = row_order(width, global_step)
            inputs = [next_tokens[s] for s in slots]
            current_positions = [positions[s] for s in slots]
            if forced is not None:
                expected = forced["phases"][phase_index]["trace"][step]
                require(expected["slots"] == slots and expected["positions"] == current_positions,
                        "forced replay row schedule mismatch")
                inputs = expected["input_ids"]
            # Deliberately select a slot independently of row order, including
            # inactive slots in narrower controls, so active-slot leakage is seen.
            active = (global_step + 3) % 8
            zero(lib.qwn_set_slot(active), f"qwn_set_slot({active})")
            before = int(lib.qwn_get_slot())
            outputs = (ctypes.c_int * width)()
            rc = int(lib.qwn_decode_batch(ints(slots), ints(inputs), ints(current_positions),
                                          width, outputs))
            after = int(lib.qwn_get_slot())
            require(rc == width, f"qwn_decode_batch(width={width}) returned {rc}")
            values = list(outputs)
            require(all(0 <= token < config["format"]["vocab_size"] for token in values),
                    "batch emitted an out-of-vocabulary token")
            logit = float(lib.qwn_last_argmax_logit())
            require(math.isfinite(logit), "last-row argmax logit is nonfinite")
            phase["trace"].append({"step": step, "global_step": global_step, "slots": slots,
                                   "positions": current_positions, "input_ids": inputs,
                                   "output_ids": values, "return_count": rc,
                                   "active_slot_before": before, "active_slot_after": after,
                                   "last_row_slot": slots[-1], "last_row_logit": logit,
                                   "last_row_logit_bits": struct.pack("<f", logit).hex()})
            for slot, value in zip(slots, values):
                positions[slot] += 1
                next_tokens[slot] = value
            global_step += 1
        phase["tracked_committed_positions"] = {str(s): positions[s] for s in sorted(positions)}
        phase["logical_kv"] = logical_kv(lib, positions) if config["kv_hashes"] else None
        phase["final_active_slot"] = int(lib.qwn_get_slot())
    destination["final_active_slot"] = int(lib.qwn_get_slot())
    destination["tracked_committed_positions"] = {str(s): positions[s] for s in sorted(positions)}
    destination["status"] = "complete"


def validate_reference(report):
    require(report.get("schema") == SCHEMA and report.get("status") == "recorded",
            "reference is not a completed batch replay record")
    config = report["config"]
    require(config["widths"] == list(WIDTHS), "reference must cover actual widths1,2,4,8")
    require(len(report["records"]) == len(WIDTHS), "reference has incomplete width coverage")
    prompts = {(p["slot"], p["generation"]): p for p in config["prompts"]}
    vocab = config["format"]["vocab_size"]
    for width, record in zip(WIDTHS, report["records"]):
        require(record["width"] == width and record["greedy"] == record["replay"],
                "reference reset replay is not exact")
        case = record["greedy"]
        require(case["status"] == "complete" and case["width"] == width,
                "incomplete reference case")
        participants = SLOT_ORDER[:width]
        positions = {s: len(prompts[s, 0]["tokens"]) - 1 for s in participants}
        next_tokens = {s: prompts[s, 0]["tokens"][-1] for s in participants}
        require(len(case["phases"]) == 2, "reference must include populated slot reuse")
        global_step = 0
        for phase_index, (name, count) in enumerate((("initial", config["steps"]),
                                                    ("slot_reuse", config["reuse_steps"]))):
            phase = case["phases"][phase_index]
            require(phase["name"] == name and len(phase["trace"]) == count,
                    "reference phase/step count mismatch")
            if phase_index:
                reused = participants[-1]
                require(phase["reset_slot"]["slot"] == reused, "reference reuse slot mismatch")
                positions[reused] = len(prompts[reused, 1]["tokens"]) - 1
                next_tokens[reused] = prompts[reused, 1]["tokens"][-1]
            require(phase["starting_positions"] == {str(s): positions[s] for s in positions},
                    "reference starting positions mismatch")
            for step, row in enumerate(phase["trace"]):
                slots = row_order(width, global_step)
                require(row["step"] == step and row["global_step"] == global_step and
                        row["slots"] == slots and row["positions"] == [positions[s] for s in slots] and
                        row["input_ids"] == [next_tokens[s] for s in slots],
                        "reference input/slot/position continuity mismatch")
                require(row["return_count"] == width and len(row["output_ids"]) == width and
                        all(type(t) is int and 0 <= t < vocab for t in row["output_ids"]),
                        "reference invalid batch output IDs")
                require(row["active_slot_before"] == (global_step + 3) % 8 and
                        type(row["active_slot_after"]) is int and
                        0 <= row["active_slot_after"] < report["native"]["num_slots"],
                        "reference active-slot evidence invalid")
                require(row["last_row_slot"] == slots[-1] and math.isfinite(row["last_row_logit"]) and
                        struct.pack("<f", row["last_row_logit"]).hex() == row["last_row_logit_bits"],
                        "reference last-row logit evidence invalid")
                for slot, value in zip(slots, row["output_ids"]):
                    positions[slot] += 1
                    next_tokens[slot] = value
                global_step += 1
            require(phase["tracked_committed_positions"] == {str(s): positions[s] for s in positions},
                    "reference committed-position ledger mismatch")
            require((phase["logical_kv"] is not None) == config["kv_hashes"],
                    "reference logical KV coverage mismatch")


def first_difference(left, right, path="records"):
    if type(left) is not type(right):
        return {"path": path, "reference": left, "candidate": right}
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return {"path": path, "reference_keys": sorted(left), "candidate_keys": sorted(right)}
        for key in left:
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "reference_length": len(left), "candidate_length": len(right)}
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f"{path}[{index}]")
            if difference is not None:
                return difference
    elif left != right:
        return {"path": path, "reference": left, "candidate": right}
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("lib", "tqf", "model-dir", "reference", "output"):
        parser.add_argument("--" + name, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--reuse-steps", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--kv-hashes", action="store_true")
    args = parser.parse_args(argv)
    if not (1 <= args.steps <= 512 and 1 <= args.reuse_steps <= 512 and
            8 <= args.chunk <= 512 and args.chunk % 8 == 0):
        parser.error("steps/reuse-steps must be1..512; chunk must be8..512 and divisible by8")
    output, reference_path = (Path(p).expanduser().resolve() for p in (args.output, args.reference))
    sources = {Path(p).expanduser().resolve() for p in (args.lib, args.tqf, args.model_dir)}
    if output == reference_path or output in sources or reference_path in sources:
        parser.error("output/reference must be separate from each other and inputs")
    report = {"schema": SCHEMA, "status": "running", "mode": "record" if args.record else "check",
              "pid": os.getpid(), "records": [], "reference_path": str(reference_path),
              "policy": {"exact": True, "finite_scope": "last-row public argmax logit only",
                         "state_pos_scope": "tracked positions, not native getter",
                         "kv_scope": "logical full-attention KV only; no recurrent-state hash"}}
    lib = None
    start = time.perf_counter()
    try:
        require(not args.record or not reference_path.exists(), "reference already exists")
        reference = None
        if args.check:
            reference = json.loads(reference_path.read_text())
            validate_reference(reference)
            report["reference_identity"] = identity(reference_path)
        report["library"] = identity(args.lib)
        model = identity(args.tqf)
        header = model_header(Path(model["path"]))
        model_dir = Path(args.model_dir).expanduser().resolve(strict=True)
        config = {"model": model, "format": header, "tokenizer_directory": str(model_dir),
                  "prompts": fixtures(model_dir, header["vocab_size"]), "widths": list(WIDTHS),
                  "steps": args.steps, "reuse_steps": args.reuse_steps, "chunk": args.chunk,
                  "kv_hashes": args.kv_hashes, "runtime_flags": flags()}
        report["config"] = config
        if reference is not None:
            compare_configuration(reference, config, [])
            report["reference_library"] = reference["library"]
            report["binary_changed"] = reference["library"]["sha256"] != report["library"]["sha256"]
        lib = bind(report["library"]["path"])
        bind_extra(lib)
        zero(lib.qwn_init(model["path"].encode()), "qwn_init")
        native = {name: int(getattr(lib, "qwn_" + name)()) for name in NATIVE_FIELDS}
        native["layer_types"] = [int(lib.qwn_layer_type(layer)) for layer in range(native["num_layers"])]
        report["native"] = native
        require(native["num_slots"] >= 8, "actual width8 requires TQ_XPU_SLOTS>=8")
        require(native["vocab_size"] == header["vocab_size"], "native/TQF vocabulary mismatch")
        require(all(len(p["tokens"]) + args.steps + args.reuse_steps <= native["max_seq"]
                    for p in config["prompts"]), "increase TQ_CTX for mixed prompts and continuation")
        if reference is not None:
            require(reference["native"] == native, "native layout/context mismatch")
        for index, width in enumerate(WIDTHS):
            record = {"width": width, "greedy": {}, "replay": {}}
            report["records"].append(record)
            if args.record:
                run_case(lib, config, width, record["greedy"])
                run_case(lib, config, width, record["replay"], record["greedy"])
                require(record["greedy"] == record["replay"], f"width{width}: reset replay not exact")
            else:
                run_case(lib, config, width, record["replay"], reference["records"][index]["greedy"])
                run_case(lib, config, width, record["greedy"])
            save(output, report)
        require(config["runtime_flags"] == flags(), "runtime flags changed during execution")
        if args.record:
            report["status"] = "recorded"
            validate_reference(report)
            report["comparison_scope"] = "baseline reset replay only; not cross-build evidence"
            save(reference_path, report, exclusive=True)
        else:
            mismatch = first_difference(reference["records"], report["records"])
            report["comparison"] = {"exact": mismatch is None, "first_mismatch": mismatch,
                                    "widths": list(WIDTHS), "modes": ["greedy", "replay"]}
            report["status"] = "passed" if mismatch is None else "failed"
            if mismatch is not None:
                report["error"] = "exact batch trace/state evidence mismatch"
        return 0 if report["status"] in ("recorded", "passed") else 1
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        print(report["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        try:
            if lib is not None:
                lib.qwn_free()
        finally:
            report["duration_seconds"] = time.perf_counter() - start
            save(output, report)
            print(json.dumps({"status": report["status"], "output": str(output),
                              "duration_seconds": report["duration_seconds"]}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
