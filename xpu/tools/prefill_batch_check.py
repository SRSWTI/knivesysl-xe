#!/usr/bin/env python3
"""Bounded real-model packed-prefill/state and teacher-forced XMX gate.

Run flat or paged in separate processes, with externally selected native flags.
Paged runs require a pool small enough that two <=1024-token requests exceed its
TOTAL physical capacity (1536 pool tokens is suitable for pages 128 and 256).
No engine, tokenizer, or device is initialized on import.
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback

from paged_parity import checked, load
from paged_state_check import STAT_NAMES


BRANCHES = ("prefill", "short", "sharded", "grouped", "generic", "dpas",
            "prefill_xmx", "prefill_scalar")
STATE_STEPS = 8
NUMERIC_STEPS = 32
CHUNKS = (64, 128)
PROMPTS = (
    ("coding_heap", "Implement a stable priority queue in Python with push, pop, "
     "and decrease_key. Explain the invariants, lazy deletion, equal-priority "
     "ordering, and amortized complexity. Include runnable tests for duplicate "
     "priorities, empty queues, and updates after removal. Compare a heap plus "
     "entry dictionary against a balanced tree. Show complete code and reason "
     "through a concrete sequence of operations."),
    ("reasoning_graph", "A directed dependency graph contains tasks with positive "
     "integer durations and unlimited workers. Derive an algorithm for the "
     "earliest finish time of every task, detect cycles, and recover a critical "
     "path. Explain why a visited flag alone is not sufficient for cycle "
     "detection. Work through a diamond graph with durations 3, 5, 2, 7 and "
     "then add a back edge. Prove correctness by induction on topological order."),
    ("coding_transactions", "Design a transactional in-memory key-value store "
     "with nested begin, commit, and rollback in Rust. Explain how undo records "
     "must compose when a child transaction commits but its parent later rolls "
     "back. Cover absent keys, overwrites, deletions, and repeated writes. Give "
     "an implementation, a worked trace, and a proof that rollback restores the "
     "exact prior mapping without cloning the whole map."),
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def zero(rc, operation):
    checked(int(rc), operation)
    require(int(rc) == 0, f"{operation}: expected 0, got {rc}")


def ints(values):
    return (ctypes.c_int * len(values))(*values)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(path):
    path = Path(path).expanduser().resolve(strict=True)
    before = path.stat()
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            hasher.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns),
            f"identity changed while hashing {path}")
    return {"path": str(path), "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns, "device": after.st_dev,
            "inode": after.st_ino, "sha256": hasher.hexdigest()}


def runtime_flags():
    keys = {key for key in os.environ if key.startswith(("TQ_", "ZE_", "SYCL_", "ONEAPI_"))}
    keys.update(("TQ_XPU_PREFILL_XMX", "TQ_XPU_PAGED_KV", "TQ_XPU_KV_PAGE",
                 "TQ_XPU_KV_POOL_TOKENS", "TQ_XPU_KV_POOL_MB", "TQ_XPU_SLOTS",
                 "TQ_CTX", "TQ_XPU_W4A4", "TQ_XPU_K64", "ZE_AFFINITY_MASK",
                 "ONEAPI_DEVICE_SELECTOR", "SYCL_DEVICE_FILTER", "LD_LIBRARY_PATH"))
    return {key: os.environ.get(key) for key in sorted(keys)}


def token_sets(model_dir, vocab):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir),
                                             trust_remote_code=True,
                                             local_files_only=True)
    result = []
    for name, question in PROMPTS:
        # Distinct, real coding/reasoning content; repetition extends the two
        # state probes without inventing ids or requiring a specific tokenizer.
        text = (question + "\nAlso discuss boundary cases and explain each invariant.\n") * 4
        ids = [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
        require(len(ids) >= 289, f"{name}: insufficient real tokens for state probe")
        count = min(1024, len(ids))
        # Final prompt token is fed through decode_batch in BOTH modes.
        count = ((count - 1) // 64) * 64 + 1
        ids = ids[:count]
        require(289 <= len(ids) <= 1024, f"{name}: invalid bounded prompt length")
        require(all(0 <= t < vocab for t in ids), f"{name}: token outside vocabulary")
        result.append({"name": name, "text": text, "tokens": ids,
                       "sha256": digest(ints(ids))})
    require(result[0]["tokens"] != result[1]["tokens"], "state prompts are not distinct")
    return result


class Native:
    def __init__(self, lib):
        self.lib = lib
        ip = ctypes.POINTER(ctypes.c_int)
        for name, args, restype in (
            ("qwn_prefill_batch", [ip, ip, ip, ip, ctypes.c_int], ctypes.c_int),
            ("qwn_decode_batch", [ip, ip, ip, ctypes.c_int, ip], ctypes.c_int),
            ("qwn_vocab_size", [], ctypes.c_int),
            ("qwn_num_slots", [], ctypes.c_int),
            ("qwn_get_slot", [], ctypes.c_int),
        ):
            function = getattr(lib, name)
            function.argtypes, function.restype = args, restype
        self.slots = int(lib.qwn_num_slots())
        self.maximum = int(lib.qwn_max_seq())
        self.paged = bool(lib.qwn_paged_enabled())
        self.layers = [layer for layer in range(int(lib.qwn_num_layers()))
                       if lib.qwn_layer_type(layer) == 2]
        require(self.slots >= 2, "set TQ_XPU_SLOTS>=2")
        require(self.maximum >= 1056, "set TQ_CTX>=1056 (2048 recommended)")
        require(self.layers, "no full-attention layers available for exact KV check")

    def reset(self):
        zero(self.lib.qwn_reset_state(), "qwn_reset_state")

    def counters(self):
        raw = (ctypes.c_ulonglong * 8)()
        zero(self.lib.qwn_attn_branch_counts(raw, 8), "qwn_attn_branch_counts(8)")
        return dict(zip(BRANCHES, map(int, raw)))

    def stats(self):
        if not self.paged:
            return None
        raw = (ctypes.c_longlong * 8)()
        zero(self.lib.qwn_paged_stats(raw, 8), "qwn_paged_stats")
        return dict(zip(STAT_NAMES, map(int, raw)))

    def snapshot(self, positions):
        state = {"active_slot": int(self.lib.qwn_get_slot()), "stats": self.stats(),
                 "maps": {}, "kv": {}}
        for slot in range(self.slots):
            if self.paged:
                table = (ctypes.c_int * ((self.maximum + 127) // 128))()
                count = checked(self.lib.qwn_debug_block_table(slot, table, len(table)),
                                f"qwn_debug_block_table({slot})")
                require(count <= len(table), "native block table exceeded buffer")
                state["maps"][str(slot)] = list(table[:count])
        for slot, position in positions.items():
            size = int(self.lib.qwn_debug_kv_bytes(position))
            require(size > 0, f"invalid logical KV byte count at {position}")
            layers = {}
            for layer in self.layers:
                buffer = (ctypes.c_ubyte * size)()
                zero(self.lib.qwn_debug_kv_copy(slot, layer, position, buffer, size),
                     f"qwn_debug_kv_copy({slot},{layer},{position})")
                layers[str(layer)] = bytes(buffer)
            state["kv"][str(slot)] = layers
        return state

    def serial(self, slot, tokens, position):
        zero(self.lib.qwn_set_slot(slot), f"qwn_set_slot({slot})")
        zero(self.lib.qwn_prefill_chunk(ints(tokens), len(tokens), position),
             f"qwn_prefill_chunk(slot={slot},pos={position},n={len(tokens)})")

    def packed(self, slots, segments, positions):
        offsets = [0]
        tokens = []
        for segment in segments:
            tokens.extend(segment)
            offsets.append(len(tokens))
        zero(self.lib.qwn_prefill_batch(ints(slots), ints(tokens), ints(offsets),
                                        ints(positions), len(slots)), "qwn_prefill_batch")

    def step(self, slots, tokens, positions):
        out = (ctypes.c_int * len(slots))()
        rc = checked(self.lib.qwn_decode_batch(ints(slots), ints(tokens), ints(positions),
                                               len(slots), out), "qwn_decode_batch")
        require(rc == len(slots), f"qwn_decode_batch: expected {len(slots)}, got {rc}")
        values = list(out)
        require(all(0 <= t < self.lib.qwn_vocab_size() for t in values),
                "decode emitted invalid token")
        logit = float(self.lib.qwn_last_argmax_logit())
        require(math.isfinite(logit), "nonfinite argmax logit")
        return {"tokens": values, "last_row_logit": logit,
                "last_row_logit_bits": struct.pack("<f", logit).hex()}


def evidence(snapshot):
    return {**snapshot, "kv": {
        slot: {layer: {"bytes": len(data), "sha256": digest(data)}
               for layer, data in layers.items()}
        for slot, layers in snapshot["kv"].items()}}


def exact_kv(left, right, label):
    require(left.keys() == right.keys(), f"{label}: layer set mismatch")
    for layer in left:
        if left[layer] != right[layer]:
            offset = next((i for i, (a, b) in enumerate(zip(left[layer], right[layer]))
                           if a != b), min(len(left[layer]), len(right[layer])))
            raise AssertionError(f"{label}: layer={layer} byte={offset} "
                                 f"sha256={digest(left[layer])}/{digest(right[layer])}")


class Suite:
    def __init__(self, native, prompts, report, output):
        self.n, self.prompts = native, prompts
        self.report, self.output = report, output

    def save(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_name(self.output.name + ".tmp")
        temporary.write_text(json.dumps(self.report, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.output)

    def case(self, name, function):
        start = time.perf_counter()
        before = self.n.counters()
        row = {"name": name, "status": "running", "flags": runtime_flags()}
        self.report["cases"].append(row)
        self.save()
        try:
            result = function()
            row["evidence"] = result
            row["status"] = "passed"
            return result
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            row["duration_seconds"] = time.perf_counter() - start
            after = self.n.counters()
            row["counters"] = {key: after[key] - before[key] for key in before}
            self.save()
            print(json.dumps(row, sort_keys=True), flush=True)

    def establish(self, packed, slots=(0, 1)):
        self.n.reset()
        for slot in slots:
            c = CHUNKS[slot]
            self.n.serial(slot, self.prompts[slot]["tokens"][:c], 0)
        if packed:
            self.n.packed(list(slots), [self.prompts[s]["tokens"][CHUNKS[s]:2 * CHUNKS[s]]
                                       for s in slots], [CHUNKS[s] for s in slots])
        else:
            for slot in slots:
                c = CHUNKS[slot]
                self.n.serial(slot, self.prompts[slot]["tokens"][c:2 * c], c)

    def continuation(self, slots, steps=STATE_STEPS, base=None):
        base = base or {slot: 2 * CHUNKS[slot] for slot in slots}
        return [self.n.step(list(slots),
                            [self.prompts[s]["tokens"][base[s] + step] for s in slots],
                            [base[s] + step for s in slots]) for step in range(steps)]

    def resume_checked(self):
        # Unlike flat decode_batch, packed prefill validates committed positions.
        # A successful append proves rejection did not silently alter state_pos.
        positions = [2 * CHUNKS[s] for s in (0, 1)]
        self.n.packed([0, 1], [self.prompts[s]["tokens"][positions[s]:positions[s] + 8]
                               for s in (0, 1)], positions)
        return self.continuation((0, 1), base={s: positions[s] + 8 for s in (0, 1)})

    def structural(self, mode):
        os.environ["TQ_XPU_PREFILL_XMX"] = mode
        references = {}
        for slot in (0, 1):
            def independent(slot=slot):
                self.establish(False, (slot,))
                before = self.n.snapshot({slot: 2 * CHUNKS[slot]})
                trace = self.continuation((slot,))
                after = self.n.snapshot({slot: 2 * CHUNKS[slot] + STATE_STEPS})
                references[slot] = (before, trace, after)
                return {"slot": slot, "chunk": CHUNKS[slot], "trace": trace,
                        "before": evidence(before), "after": evidence(after)}
            self.case(f"mode_{mode}_independent_{slot}", independent)

        def packed_independent():
            self.establish(True)
            before = self.n.snapshot({s: 2 * CHUNKS[s] for s in (0, 1)})
            traces = {}
            for slot in (0, 1):
                exact_kv(references[slot][0]["kv"][str(slot)], before["kv"][str(slot)],
                         f"packed initial slot {slot}")
                traces[str(slot)] = self.continuation((slot,))
                require(traces[str(slot)] == references[slot][1],
                        f"packed slot {slot}: width1 teacher token/logit pairs differ: "
                        f"actual={traces[str(slot)]} expected={references[slot][1]}")
            after = self.n.snapshot({s: 2 * CHUNKS[s] + STATE_STEPS for s in (0, 1)})
            for slot in (0, 1):
                exact_kv(references[slot][2]["kv"][str(slot)], after["kv"][str(slot)],
                         f"packed final slot {slot}")
            return {"starting_positions": list(CHUNKS), "segment_lengths": list(CHUNKS),
                    "before": evidence(before), "after": evidence(after), "traces": traces}
        self.case(f"mode_{mode}_packed_vs_independent", packed_independent)
        baseline = {}

        def serial_width2():
            self.establish(False)
            baseline["trace"] = self.continuation((0, 1))
            baseline["state"] = self.n.snapshot({s: 2 * CHUNKS[s] + STATE_STEPS
                                                 for s in (0, 1)})
            return {"trace": baseline["trace"], "state": evidence(baseline["state"])}
        self.case(f"mode_{mode}_serial_width2", serial_width2)

        def packed_width2():
            self.establish(True)
            trace = self.continuation((0, 1))
            require(trace == baseline["trace"],
                    f"width2 teacher token/logit pairs differ: {trace} vs {baseline['trace']}")
            state = self.n.snapshot({s: 2 * CHUNKS[s] + STATE_STEPS for s in (0, 1)})
            for slot in (0, 1):
                exact_kv(baseline["state"]["kv"][str(slot)], state["kv"][str(slot)],
                         f"width2 final slot {slot}")
            return {"trace": trace, "state": evidence(state)}
        self.case(f"mode_{mode}_packed_width2", packed_width2)

    def unchanged_call(self, name, call, expected, positions):
        before = self.n.snapshot(positions)
        counts = self.n.counters()
        rc = int(call())
        require(rc == expected, f"{name}: expected {expected}, got {rc}")
        after = self.n.snapshot(positions)
        require(before == after, f"{name}: state changed; before={evidence(before)} "
                f"after={evidence(after)}")
        require(self.n.counters() == counts, f"{name}: entered attention despite rejection")
        return {"return_code": rc, "unchanged": evidence(after)}

    def rejected(self):
        os.environ["TQ_XPU_PREFILL_XMX"] = "1"
        self.establish(True)
        positions = {s: 2 * CHUNKS[s] for s in (0, 1)}
        baseline = self.case("rejection_clean_continuation",
                             lambda: {"trace": self.resume_checked()})["trace"]
        self.establish(True)
        tokens = self.prompts[0]["tokens"][:8] + self.prompts[1]["tokens"][:8]
        lib = self.n.lib
        cases = (
            ("duplicate_slots", [0, 0], tokens, [0, 8, 16], [128, 128], -2),
            ("offset_origin", [0, 1], tokens, [8, 8, 16], [128, 256], -2),
            ("offset_empty", [0, 1], tokens, [0, 0, 16], [128, 256], -2),
            ("offset_descending", [0, 1], tokens, [0, 8, 0], [128, 256], -2),
            ("offset_unaligned", [0, 1], tokens, [0, 7, 16], [128, 256], -2),
            ("negative_token", [0, 1], tokens[:-1] + [-1], [0, 8, 16], [128, 256], -4),
            ("vocab_token", [0, 1], tokens[:-1] + [lib.qwn_vocab_size()],
             [0, 8, 16], [128, 256], -4),
            ("nonmatching_position", [0, 1], tokens, [0, 8, 16], [128, 264], -3),
            ("reset_then_bad_position", [0, 1], tokens, [0, 8, 16], [0, 264], -3),
            ("reset_then_bad_token", [0, 1], tokens[:-1] + [-1],
             [0, 8, 16], [0, 256], -4),
        )
        for name, slots, values, offsets, pos, expected in cases:
            self.case(name, lambda name=name, slots=slots, values=values, offsets=offsets,
                      pos=pos, expected=expected: self.unchanged_call(
                          name, lambda: lib.qwn_prefill_batch(ints(slots), ints(values),
                                                             ints(offsets), ints(pos), 2),
                          expected, positions))
        self.case("null_tokens", lambda: self.unchanged_call(
            "null_tokens", lambda: lib.qwn_prefill_batch(ints([0, 1]), None,
                                                         ints([0, 8, 16]),
                                                         ints([128, 256]), 2), -1, positions))
        def prove():
            trace = self.resume_checked()
            require(trace == baseline, f"rejected calls changed GDN/continuation: {trace} vs {baseline}")
            return {"trace": trace, "exact_clean_reference": True}
        self.case("rejection_gdn_continuation_unchanged", prove)

    def capacity(self):
        if not self.n.paged:
            self.case("capacity_applicability", lambda: {
                "applicable": False, "reason": "native attested flat mode; no physical paged pool"})
            return
        self.establish(True)
        stats = self.n.stats()
        total, page = stats["total_blocks"], stats["page_tokens"]
        require(page in (128, 256), f"unexpected native page size {page}")
        # Demand is derived from TOTAL pool size, not merely free pages. Each
        # request fits individually but the pair cannot fit even after reset.
        blocks = [(total + 1) // 2, (total + 2) // 2]
        ends = [b * page for b in blocks]
        require(total >= 4 and max(ends) <= 1024 and min(ends) > 256,
                f"bounded physical-capacity gate needs TQ_XPU_KV_POOL_TOKENS=1536; "
                f"external pool preserved, observed {stats}")
        positions = {s: 2 * CHUNKS[s] for s in (0, 1)}
        baseline = self.case("capacity_clean_continuation",
                             lambda: {"trace": self.resume_checked()})["trace"]
        self.establish(True)
        for reset in (False, True):
            name = "capacity_aggregate_reset" if reset else "capacity_aggregate_growth"
            pos = [0, 0] if reset else [positions[0], positions[1]]
            lengths = [ends[s] - pos[s] for s in (0, 1)]
            values = [self.prompts[s]["tokens"][0] for s in (0, 1)
                      for _ in range(lengths[s])]
            offsets = [0, lengths[0], sum(lengths)]
            def reject(name=name, pos=pos, values=values, offsets=offsets):
                result = self.unchanged_call(name, lambda: self.n.lib.qwn_prefill_batch(
                    ints([0, 1]), ints(values), ints(offsets), ints(pos), 2), -1001, positions)
                return {**result, "physical_total_blocks": total,
                        "requested_final_blocks": blocks, "positions": pos,
                        "offsets": offsets}
            self.case(name, reject)
        def prove():
            trace = self.resume_checked()
            require(trace == baseline, f"capacity failure changed GDN: {trace} vs {baseline}")
            return {"trace": trace, "exact_clean_reference": True}
        self.case("capacity_gdn_continuation_unchanged", prove)

    def production_packed(self):
        # Packed prefill preserves each segment's split-K geometry under AUTO.
        # Structural cases prove byte exactness first; these longer teacher
        # continuations separately report production model agreement.
        os.environ["TQ_XPU_PREFILL_XMX"] = "1"
        references = {}
        for slot in (0, 1):
            def independent(slot=slot):
                self.establish(False, (slot,))
                position = 2 * CHUNKS[slot]
                inputs, trace = [], []
                for step in range(NUMERIC_STEPS):
                    token = self.prompts[slot]["tokens"][position] if step == 0 else \
                        trace[-1]["tokens"][0]
                    inputs.append(token)
                    trace.append(self.n.step([slot], [token], [position + step]))
                result = {"slot": slot, "prompt": self.prompts[slot]["name"],
                          "prompt_tokens": position + 1, "chunk_tokens": CHUNKS[slot],
                          "teacher_inputs": inputs, "trace": trace,
                          "continuation_width": 1, "sample_size": NUMERIC_STEPS}
                references[slot] = result
                return result
            self.case(f"production_independent_{slot}", independent)

        comparisons = []
        def packed():
            self.establish(True)
            for slot in (0, 1):
                reference = references[slot]
                position = 2 * CHUNKS[slot]
                trace = [self.n.step([slot], [token], [position + step])
                         for step, token in enumerate(reference["teacher_inputs"])]
                matches = sum(a["tokens"] == b["tokens"]
                              for a, b in zip(reference["trace"], trace))
                comparisons.append({"slot": slot, "prompt": reference["prompt"],
                                    "teacher_inputs": reference["teacher_inputs"],
                                    "trace": trace, "sample_size": NUMERIC_STEPS,
                                    "matches": matches,
                                    "top1_agreement": matches / NUMERIC_STEPS,
                                    "reference_case": f"production_independent_{slot}"})
            return {"slots": comparisons, "starting_positions": list(CHUNKS),
                    "segment_lengths": list(CHUNKS), "continuation_width": 1,
                    "cache_exactness": "exercised by preceding AUTO-split structural cases"}
        self.case("production_packed_teacher_replay", packed)

        def gate():
            total = sum(row["sample_size"] for row in comparisons)
            matches = sum(row["matches"] for row in comparisons)
            rates = {str(row["slot"]): row["top1_agreement"] for row in comparisons}
            require(len(comparisons) == 2 and set(rates) == {"0", "1"} and
                    all(row["sample_size"] == NUMERIC_STEPS for row in comparisons) and
                    total == 2 * NUMERIC_STEPS, f"incomplete packed sample {total}")
            result = {"threshold": 0.90, "gate_scope": "aggregate",
                      "per_slot": rates, "minimum_slot_agreement": min(rates.values()),
                      "legacy_per_slot_pass": all(rate >= 0.90 for rate in rates.values()),
                      "matches": matches, "sample_size": total,
                      "top1_agreement": matches / total,
                      "metric": "teacher-forced, not autoregressive exactness"}
            self.report["cases"][-1]["evidence"] = result
            require(matches / total >= 0.90,
                    f"AUTO-split packed/independent aggregate agreement below 90%: {rates}; "
                    f"aggregate={matches}/{total}={matches / total:.6f}")
            for row in self.report["cases"]:
                if row["name"].startswith("production_") and "counters" in row:
                    require(row["counters"]["prefill_xmx"] > 0 and
                            row["counters"]["prefill_scalar"] == 0,
                            f"production case did not exclusively exercise XMX: {row['name']}")
            return result
        self.case("production_packed_top1_gate", gate)

    def numeric(self):
        comparisons = []
        for prompt in self.prompts:
            ids = prompt["tokens"]
            reference = {}
            for mode in ("0", "1"):
                os.environ["TQ_XPU_PREFILL_XMX"] = mode
                def arm(mode=mode):
                    self.n.reset()
                    before = self.n.counters()
                    for position in range(0, len(ids) - 1, 64):
                        self.n.serial(0, ids[position:position + 64], position)
                    inputs, trace = [], []
                    for step in range(NUMERIC_STEPS):
                        token = (ids[-1] if step == 0 else reference["trace"][step - 1]["tokens"][0]) \
                            if mode == "1" else (ids[-1] if step == 0 else trace[-1]["tokens"][0])
                        inputs.append(token)
                        trace.append(self.n.step([0], [token], [len(ids) - 1 + step]))
                    after = self.n.counters()
                    delta = {key: after[key] - before[key] for key in before}
                    active = "prefill_scalar" if mode == "0" else "prefill_xmx"
                    inactive = "prefill_xmx" if mode == "0" else "prefill_scalar"
                    require(delta[active] > 0 and delta[inactive] == 0,
                            f"mode {mode} was not exclusively exercised: {delta}")
                    result = {"prompt": prompt["name"], "prompt_tokens": len(ids),
                              "chunk_tokens": 64, "teacher_inputs": inputs,
                              "trace": trace, "branch_delta": delta}
                    if mode == "0":
                        reference.update(result)
                    else:
                        require(inputs == reference["teacher_inputs"], "teacher inputs diverged")
                        matches = sum(a["tokens"] == b["tokens"]
                                      for a, b in zip(reference["trace"], trace))
                        result.update(matches=matches, sample_size=NUMERIC_STEPS,
                                      top1_agreement=matches / NUMERIC_STEPS,
                                      scalar_reference_case=f"numeric_{prompt['name']}_mode_0")
                        comparisons.append(result)
                    return result
                self.case(f"numeric_{prompt['name']}_mode_{mode}", arm)
        def gate():
            total = sum(row["sample_size"] for row in comparisons)
            matches = sum(row["matches"] for row in comparisons)
            rates = {row["prompt"]: row["top1_agreement"] for row in comparisons}
            require(len(comparisons) == 3 and
                    set(rates) == {prompt["name"] for prompt in self.prompts} and
                    all(row["sample_size"] == NUMERIC_STEPS for row in comparisons) and
                    total == 3 * NUMERIC_STEPS, f"incomplete numeric sample {total}")
            result = {"threshold": 0.90, "gate_scope": "aggregate",
                      "per_prompt": rates, "minimum_prompt_agreement": min(rates.values()),
                      "legacy_per_prompt_pass": all(rate >= 0.90 for rate in rates.values()),
                      "matches": matches, "sample_size": total,
                      "top1_agreement": matches / total,
                      "metric": "teacher-forced, not autoregressive exactness"}
            self.report["cases"][-1]["evidence"] = result
            require(matches / total >= 0.90,
                    f"teacher-forced aggregate top1 agreement below 90%: {rates}; "
                    f"aggregate={matches}/{total}={matches / total:.6f}")
            return result
        self.case("numeric_top1_gate", gate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", required=True)
    parser.add_argument("--tqf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", help="local tokenizer directory; defaults to TQF parent")
    parser.add_argument("--production-numerics", action="store_true",
                        help="AUTO-split exact structural checks followed by packed/independent "
                             "and XMX/scalar aggregate >=90%% teacher-forced numeric gates")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    report = {"status": "running", "cases": [], "initial_flags": runtime_flags(),
              "pid": os.getpid(), "argv": sys.argv if argv is None else argv,
              "state_continuation_steps": STATE_STEPS,
              "numeric_continuation_steps": NUMERIC_STEPS,
              "logit_scope": "width1: each slot; width2: last row only (public ABI)"}
    report["mode"] = "production_numerics" if args.production_numerics else "fixed_split_exact"
    report["cache_exactness_scope"] = (
        "Exact logical KV bytes in preceding scalar and XMX AUTO-split structural cases"
        if args.production_numerics else "Exact logical KV bytes under GEMM_SPLITS=1")
    report["numeric_acceptance_policy"] = {
        "selection": "user-authorized numerically tolerant XMX",
        "gate_scope": "aggregate", "top1_threshold": 0.90,
        "per_prompt_and_slot_rates": "retained with minima and legacy per-item pass status"}
    old_mode = os.environ.get("TQ_XPU_PREFILL_XMX")
    lib, suite = None, None
    start = time.perf_counter()
    try:
        split = os.environ.get("TQ_XPU_GEMM_SPLITS")
        if args.production_numerics:
            require(split in (None, "", "0"), "--production-numerics requires AUTO GEMM splits: "
                    "unset TQ_XPU_GEMM_SPLITS or set it to 0 externally; the tool never changes it")
        else:
            require(split == "1", "exact mode requires TQ_XPU_GEMM_SPLITS=1 before qwn_init; "
                    "Set fixed splits externally or select the AUTO-split exact/numeric checks via "
                    "--production-numerics with the variable unset; no flags were changed")
        report["library"] = identity(args.lib)
        report["model"] = identity(args.tqf)
        model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else \
            Path(report["model"]["path"]).parent
        report["tokenizer_directory"] = str(model_dir)
        # Canonical loader invokes qwn_init exactly once; no arm reloads weights.
        lib = load(report["library"]["path"], report["model"]["path"])
        native = Native(lib)
        prompts = token_sets(model_dir, int(lib.qwn_vocab_size()))
        report["prompts"] = prompts
        report["native"] = {"paged": native.paged, "slots": native.slots,
                            "max_sequence_tokens": native.maximum,
                            "vocab": int(lib.qwn_vocab_size()), "pool": native.stats()}
        suite = Suite(native, prompts, report, output)
        suite.save()
        for mode in ("0", "1"):
            suite.structural(mode)
        if args.production_numerics:
            suite.production_packed()
        else:
            suite.rejected()
            suite.capacity()
        suite.numeric()
        report["status"] = "passed"
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        print(report["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        report["duration_seconds"] = time.perf_counter() - start
        if suite is not None:
            report["final_counters"] = suite.n.counters()
        if lib is not None:
            lib.qwn_free()
        if old_mode is None:
            os.environ.pop("TQ_XPU_PREFILL_XMX", None)
        else:
            os.environ["TQ_XPU_PREFILL_XMX"] = old_mode
        if suite is not None:
            suite.save()
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": report["status"], "output": str(output),
                          "duration_seconds": report["duration_seconds"]}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
