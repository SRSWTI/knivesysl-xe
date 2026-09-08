#!/usr/bin/env python3
"""Real paged-KV ownership, reservation, and native APC regression gate.

The tool intentionally uses a small, real paged pool and a loaded model.  It
never patches block tables, substitutes an allocator, or uses a fake engine.
Every expected capacity failure is induced by native reservations against real
physical capacity, and every state comparison reads logical K/V bytes back
through the public debug ABI.
"""

import argparse
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

# paged_parity is import-safe: model initialization lives under its main guard.
# Reuse its canonical CDLL loader and negative-status checker rather than grow a
# second interpretation of those basics.
from paged_parity import checked, load


CAPACITY = -1001  # TQ_PAGED_ERR_CAPACITY
STAT_NAMES = (
    "page_tokens",
    "total_blocks",
    "free_blocks",
    "active_blocks",
    "checkpoint_blocks",
    "remaining_reserved_blocks",
    "max_sequence_tokens",
    "slots",
)


class ScenarioError(RuntimeError):
    def __init__(self, scenario, message):
        super().__init__(message)
        self.scenario = scenario


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_zero(rc, operation):
    checked(int(rc), operation)
    require(int(rc) == 0, f"{operation} returned {int(rc)}, expected 0")


def expect_negative(rc, operation):
    require(int(rc) < 0, f"{operation} returned {int(rc)}, expected a negative error")
    return int(rc)


def expect_capacity(rc, operation):
    require(int(rc) == CAPACITY,
            f"{operation} returned {int(rc)}, expected TQ_PAGED_ERR_CAPACITY={CAPACITY}")


def byte_hash(data):
    return hashlib.sha256(data).hexdigest()


def token_hash(tokens):
    payload = json.dumps(list(tokens), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def first_byte_difference(left, right):
    upto = min(len(left), len(right))
    for index in range(upto):
        if left[index] != right[index]:
            return index
    return upto if len(left) != len(right) else None


class Native:
    """Explicit ctypes surface plus ownership/state inspection helpers."""

    def __init__(self, lib, tqf, expected_page, expected_blocks):
        self.lib = lib
        self.tqf = os.path.expanduser(tqf).encode()
        self.expected_page = expected_page
        self.expected_blocks = expected_blocks
        self.initialized = True  # paged_parity.load initialized it.
        self._bind_all()

    def _bind_all(self):
        int_p = ctypes.POINTER(ctypes.c_int)
        ll_p = ctypes.POINTER(ctypes.c_longlong)
        specs = (
            ("qwn_init", [ctypes.c_char_p], ctypes.c_int),
            ("qwn_free", [], None),
            ("qwn_reset_state", [], ctypes.c_int),
            ("qwn_reset_slot", [ctypes.c_int], ctypes.c_int),
            ("qwn_num_slots", [], ctypes.c_int),
            ("qwn_get_slot", [], ctypes.c_int),
            ("qwn_set_slot", [ctypes.c_int], ctypes.c_int),
            ("qwn_max_seq", [], ctypes.c_int),
            ("qwn_vocab_size", [], ctypes.c_int),
            ("qwn_num_layers", [], ctypes.c_int),
            ("qwn_layer_type", [ctypes.c_int], ctypes.c_int),
            ("qwn_decode", [ctypes.c_int, ctypes.c_int], ctypes.c_int),
            ("qwn_prefill_chunk", [int_p, ctypes.c_int, ctypes.c_int], ctypes.c_int),
            ("qwn_decode_batch", [int_p, int_p, int_p, ctypes.c_int, int_p],
             ctypes.c_int),
            ("qwn_host_alloc", [ctypes.c_size_t], ctypes.c_void_p),
            ("qwn_host_free", [ctypes.c_void_p], None),
            ("qwn_ckpt_bytes", [ctypes.c_int], ctypes.c_size_t),
            ("qwn_ckpt_save",
             [ctypes.c_int, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
            ("qwn_ckpt_restore",
             [ctypes.c_int, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
            ("qwn_paged_enabled", [], ctypes.c_int),
            ("qwn_paged_reserve", [ctypes.c_int, ctypes.c_int], ctypes.c_int),
            ("qwn_paged_release", [ctypes.c_int], ctypes.c_int),
            ("qwn_paged_stats", [ll_p, ctypes.c_int], ctypes.c_int),
            ("qwn_paged_ckpt_save", [ctypes.c_int, ctypes.c_int], ctypes.c_int),
            ("qwn_paged_ckpt_adopt", [ctypes.c_int, ctypes.c_int], ctypes.c_int),
            ("qwn_paged_ckpt_free", [ctypes.c_int], ctypes.c_int),
            ("qwn_paged_ckpt_host_bytes", [ctypes.c_int], ctypes.c_size_t),
            ("qwn_debug_kv_bytes", [ctypes.c_int], ctypes.c_size_t),
            ("qwn_debug_kv_copy",
             [ctypes.c_int, ctypes.c_int, ctypes.c_int,
              ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            ("qwn_debug_block_table", [ctypes.c_int, int_p, ctypes.c_int],
             ctypes.c_int),
        )
        for name, argtypes, restype in specs:
            fn = getattr(self.lib, name)
            fn.argtypes = argtypes
            fn.restype = restype

    def free(self):
        if self.initialized:
            self.lib.qwn_free()
            self.initialized = False

    def init(self):
        require(not self.initialized, "qwn_init lifecycle helper called while initialized")
        expect_zero(self.lib.qwn_init(self.tqf), "qwn_init(reload)")
        self.initialized = True
        self.assert_attested()

    def assert_attested(self):
        require(self.lib.qwn_paged_enabled() == 1,
                "candidate did not attest effective paged mode")
        require(self.lib.qwn_num_slots() >= 2,
                f"native model exposes {self.lib.qwn_num_slots()} slots; at least 2 required")
        stats = self.stats(check=False)
        require(stats["page_tokens"] == self.expected_page,
                f"native page={stats['page_tokens']}, configured={self.expected_page}")
        require(stats["total_blocks"] == self.expected_blocks,
                f"native blocks={stats['total_blocks']}, configured={self.expected_blocks}")
        require(stats["slots"] == self.lib.qwn_num_slots(),
                "qwn_paged_stats slot count disagrees with qwn_num_slots")
        self.assert_conservation(stats)
        return stats

    def assert_flat(self):
        require(self.lib.qwn_paged_enabled() == 0,
                "flat regression unexpectedly initialized paged ownership")
        slots = int(self.lib.qwn_num_slots())
        require(slots >= 2,
                f"native model exposes {slots} slots; at least 2 required")
        return {
            "paged_enabled": 0,
            "slots": slots,
            "max_sequence_tokens": int(self.lib.qwn_max_seq()),
            "vocab_size": int(self.lib.qwn_vocab_size()),
        }

    def reset_all(self):
        expect_zero(self.lib.qwn_reset_state(), "qwn_reset_state")
        stats = self.stats()
        require(stats["free_blocks"] == stats["total_blocks"],
                f"reset-all leaked physical blocks: {stats}")
        require(stats["active_blocks"] == 0 and stats["checkpoint_blocks"] == 0,
                f"reset-all retained native owners: {stats}")
        require(stats["remaining_reserved_blocks"] == 0,
                f"reset-all retained reservations: {stats}")
        return stats

    def stats(self, check=True):
        raw = (ctypes.c_longlong * len(STAT_NAMES))()
        expect_zero(self.lib.qwn_paged_stats(raw, len(raw)), "qwn_paged_stats")
        result = {name: int(raw[index]) for index, name in enumerate(STAT_NAMES)}
        if check:
            self.assert_conservation(result)
        return result

    def block_table(self, slot):
        capacity = max(1, self.expected_blocks)
        raw = (ctypes.c_int * capacity)()
        count = int(self.lib.qwn_debug_block_table(slot, raw, capacity))
        checked(count, f"qwn_debug_block_table(slot={slot})")
        require(count <= capacity,
                f"block table returned count={count} beyond buffer={capacity}")
        table = [int(raw[index]) for index in range(count)]
        require(len(table) == len(set(table)),
                f"slot {slot} maps the same physical block twice: {table}")
        for block in table:
            require(0 <= block < self.expected_blocks,
                    f"slot {slot} contains out-of-range block {block}")
        return table

    def assert_conservation(self, stats=None):
        stats = stats or self.stats(check=False)
        total = stats["total_blocks"]
        free = stats["free_blocks"]
        active = stats["active_blocks"]
        checkpoints = stats["checkpoint_blocks"]
        reserved = stats["remaining_reserved_blocks"]
        require(total == self.expected_blocks and total > 0,
                f"unexpected pool size in stats: {stats}")
        require(0 <= free <= total, f"invalid free count: {stats}")
        require(0 <= active <= total - free,
                f"active blocks exceed physical occupancy: {stats}")
        require(0 <= checkpoints <= total - free,
                f"checkpoint blocks exceed physical occupancy: {stats}")
        require(0 <= reserved <= free,
                f"reservation conservation violated (reserved <= free): {stats}")
        active_ids = set()
        for slot in range(int(stats["slots"])):
            active_ids.update(self.block_table(slot))
        require(len(active_ids) == active,
                f"active block statistic {active} != mapped block union {len(active_ids)}")

    def kv(self, slot, committed):
        require(committed > 0, "logical KV inspection requires a positive committed count")
        size = int(self.lib.qwn_debug_kv_bytes(committed))
        require(size > 0,
                f"qwn_debug_kv_bytes({committed}) returned an invalid size")
        result = {}
        for layer in range(int(self.lib.qwn_num_layers())):
            if int(self.lib.qwn_layer_type(layer)) != 2:
                continue
            buffer = (ctypes.c_ubyte * size)()
            expect_zero(self.lib.qwn_debug_kv_copy(
                slot, layer, committed, buffer, size),
                f"qwn_debug_kv_copy(slot={slot},layer={layer},committed={committed})")
            result[layer] = bytes(buffer)
        require(result, "model exposes no full-attention layer for logical KV inspection")
        return result

    def kv_digests(self, snapshot):
        return {str(layer): byte_hash(data) for layer, data in sorted(snapshot.items())}

    def kv_hashes(self, slot, committed):
        """Stream exact per-layer logical bytes into digests without retaining GiB."""
        require(committed > 0, "logical KV hashing requires a positive committed count")
        size = int(self.lib.qwn_debug_kv_bytes(committed))
        require(size > 0,
                f"qwn_debug_kv_bytes({committed}) returned an invalid size")
        buffer = (ctypes.c_ubyte * size)()
        result = {}
        for layer in range(int(self.lib.qwn_num_layers())):
            if int(self.lib.qwn_layer_type(layer)) != 2:
                continue
            expect_zero(self.lib.qwn_debug_kv_copy(
                slot, layer, committed, buffer, size),
                f"qwn_debug_kv_copy(slot={slot},layer={layer},committed={committed})")
            result[str(layer)] = hashlib.sha256(buffer).hexdigest()
        require(result, "model exposes no full-attention layer for logical KV hashing")
        return {"bytes_per_layer": size, "sha256": result}

    def assert_kv_equal(self, left, right, label):
        require(set(left) == set(right),
                f"{label}: full-attention layer sets differ")
        for layer in sorted(left):
            if left[layer] != right[layer]:
                offset = first_byte_difference(left[layer], right[layer])
                raise AssertionError(
                    f"{label}: exact logical K/V/scale mismatch at layer={layer}, "
                    f"byte={offset}, left_sha256={byte_hash(left[layer])}, "
                    f"right_sha256={byte_hash(right[layer])}")

    def prefill(self, slot, tokens, pos0=0, chunk=32):
        require(tokens and len(tokens) % 8 == 0,
                "prefill helper requires a nonempty multiple of eight")
        require(chunk > 0 and chunk % 8 == 0,
                "prefill helper chunk must be a positive multiple of eight")
        expect_zero(self.lib.qwn_set_slot(slot), f"qwn_set_slot({slot})")
        position = pos0
        offset = 0
        while offset < len(tokens):
            take = min(chunk, len(tokens) - offset)
            take -= take % 8
            require(take >= 8, "prefill helper produced an invalid final tile")
            array = (ctypes.c_int * take)(*tokens[offset:offset + take])
            expect_zero(self.lib.qwn_prefill_chunk(array, take, position),
                        f"qwn_prefill_chunk(slot={slot},pos={position},n={take})")
            position += take
            offset += take
        return position

    def decode(self, slot, token, position):
        expect_zero(self.lib.qwn_set_slot(slot), f"qwn_set_slot({slot})")
        return checked(int(self.lib.qwn_decode(int(token), int(position))),
                       f"qwn_decode(slot={slot},pos={position})")


class Row:
    def __init__(self, slot, position, token):
        self.slot = int(slot)
        self.position = int(position)
        self.token = int(token)
        self.outputs = []


class Suite:
    def __init__(self, native, token_sets, chunk):
        self.n = native
        self.lib = native.lib
        self.tokens = token_sets
        self.chunk = chunk
        self.results = []
        self.current = "startup"
        self.dispatch_counts = {"scalar_width_1": 0, "native_batch_width_2": 0}

    def run(self, name, method):
        self.current = name
        try:
            evidence = method()
        except Exception as exc:
            raise ScenarioError(name, str(exc)) from exc
        self.results.append({"name": name, "status": "PASS", "evidence": evidence})

    def reserve(self, slot, tokens):
        expect_zero(self.lib.qwn_paged_reserve(slot, tokens),
                    f"qwn_paged_reserve(slot={slot},tokens={tokens})")

    def release(self, slot):
        expect_zero(self.lib.qwn_paged_release(slot),
                    f"qwn_paged_release(slot={slot})")

    def continuation(self, slot, position, fixed_inputs, steps):
        require(fixed_inputs and steps > 0,
                "continuation needs at least one fixed input and one output")
        last = -1
        for token in fixed_inputs:
            last = self.n.decode(slot, token, position)
            position += 1
        outputs = [last]
        for _ in range(steps - 1):
            last = self.n.decode(slot, last, position)
            position += 1
            outputs.append(last)
        return outputs, position

    def dispatch(self, rows):
        require(rows and len(rows) <= 2, "qualification dispatch expects width one or two")
        if len(rows) == 1:
            # This is the serving invariant: do not route a lone row through the
            # padded RC8 batch path, whose arithmetic is a different reference.
            row = rows[0]
            out = self.n.decode(row.slot, row.token, row.position)
            row.position += 1
            row.token = int(out)
            row.outputs.append(int(out))
            self.dispatch_counts["scalar_width_1"] += 1
            return
        count = len(rows)
        Ints = ctypes.c_int * count
        slots = Ints(*[row.slot for row in rows])
        tokens = Ints(*[row.token for row in rows])
        positions = Ints(*[row.position for row in rows])
        outputs = Ints(*([0] * count))
        rc = int(self.lib.qwn_decode_batch(slots, tokens, positions, count, outputs))
        require(rc == count, f"qwn_decode_batch(width={count}) returned {rc}")
        for index, row in enumerate(rows):
            row.position += 1
            row.token = int(outputs[index])
            row.outputs.append(row.token)
        self.dispatch_counts["native_batch_width_2"] += 1

    def invalid_ranges_before_mutation(self):
        baseline = self.n.reset_all()
        valid = self.tokens["alpha"][0]
        max_seq = int(self.lib.qwn_max_seq())
        vocab = int(self.lib.qwn_vocab_size())
        layers = int(self.lib.qwn_num_layers())
        full_layer = next(layer for layer in range(layers)
                          if int(self.lib.qwn_layer_type(layer)) == 2)
        one_size = int(self.lib.qwn_debug_kv_bytes(1))
        require(one_size > 0, "debug size for one logical row is unavailable")
        byte_buffer = (ctypes.c_ubyte * one_size)()
        int_buffer = (ctypes.c_int * 8)(*([valid] * 8))

        errors = {}
        errors["set_negative_slot"] = expect_negative(
            self.lib.qwn_set_slot(-1), "qwn_set_slot(-1)")
        errors["reset_high_slot"] = expect_negative(
            self.lib.qwn_reset_slot(int(self.lib.qwn_num_slots())),
            "qwn_reset_slot(slot_count)")
        errors["reserve_negative_slot"] = expect_negative(
            self.lib.qwn_paged_reserve(-1, 1), "reserve negative slot")
        errors["reserve_negative_tokens"] = expect_negative(
            self.lib.qwn_paged_reserve(0, -1), "reserve negative tokens")
        errors["reserve_above_context"] = expect_negative(
            self.lib.qwn_paged_reserve(0, max_seq + 1), "reserve above context")
        errors["release_high_slot"] = expect_negative(
            self.lib.qwn_paged_release(int(self.lib.qwn_num_slots())),
            "release high slot")
        errors["stats_null"] = expect_negative(
            self.lib.qwn_paged_stats(None, 8), "stats null output")
        short_stats = (ctypes.c_longlong * 7)()
        errors["stats_short"] = expect_negative(
            self.lib.qwn_paged_stats(short_stats, 7), "stats short output")
        errors["save_bad_slot"] = expect_negative(
            self.lib.qwn_paged_ckpt_save(-1, 0), "checkpoint save bad slot")
        errors["save_uncommitted"] = expect_negative(
            self.lib.qwn_paged_ckpt_save(0, 1), "checkpoint save uncommitted")
        errors["adopt_zero_id"] = expect_negative(
            self.lib.qwn_paged_ckpt_adopt(0, 0), "checkpoint adopt id zero")
        errors["free_zero_id"] = expect_negative(
            self.lib.qwn_paged_ckpt_free(0), "checkpoint free id zero")
        require(int(self.lib.qwn_paged_ckpt_host_bytes(0)) == 0,
                "invalid checkpoint id reported host ownership")
        require(int(self.lib.qwn_debug_kv_bytes(-1)) == 0 and
                int(self.lib.qwn_debug_kv_bytes(max_seq + 1)) == 0,
                "invalid committed range reported a logical KV size")
        errors["kv_null"] = expect_negative(
            self.lib.qwn_debug_kv_copy(0, full_layer, 1, None, one_size),
            "debug KV null output")
        errors["kv_bad_slot"] = expect_negative(
            self.lib.qwn_debug_kv_copy(-1, full_layer, 1,
                                       byte_buffer, one_size),
            "debug KV bad slot")
        errors["kv_bad_layer"] = expect_negative(
            self.lib.qwn_debug_kv_copy(0, layers, 1,
                                       byte_buffer, one_size),
            "debug KV bad layer")
        errors["kv_short"] = expect_negative(
            self.lib.qwn_debug_kv_copy(0, full_layer, 1,
                                       byte_buffer, one_size - 1),
            "debug KV short output")
        errors["table_null"] = expect_negative(
            self.lib.qwn_debug_block_table(0, None, self.n.expected_blocks),
            "debug table null output")
        errors["table_bad_slot"] = expect_negative(
            self.lib.qwn_debug_block_table(-1, int_buffer, 8),
            "debug table bad slot")
        errors["decode_negative_position"] = expect_negative(
            self.lib.qwn_decode(valid, -1), "decode negative position")
        errors["decode_bad_token"] = expect_negative(
            self.lib.qwn_decode(-1, 0), "decode bad token")
        errors["decode_vocab_token"] = expect_negative(
            self.lib.qwn_decode(vocab, 0), "decode token equal to vocab size")
        errors["decode_position_gap"] = expect_negative(
            self.lib.qwn_decode(valid, 1), "decode position gap")
        errors["prefill_null"] = expect_negative(
            self.lib.qwn_prefill_chunk(None, 8, 0), "prefill null tokens")
        errors["prefill_non_tile"] = expect_negative(
            self.lib.qwn_prefill_chunk(int_buffer, 7, 0), "prefill non-tile")
        errors["prefill_bad_range"] = expect_negative(
            self.lib.qwn_prefill_chunk(int_buffer, 8, max_seq - 4),
            "prefill range past context")
        bad_tokens = (ctypes.c_int * 8)(*([valid] * 7 + [vocab]))
        errors["prefill_bad_token"] = expect_negative(
            self.lib.qwn_prefill_chunk(bad_tokens, 8, 0), "prefill bad token")
        slots = (ctypes.c_int * 2)(0, 0)
        batch_tokens = (ctypes.c_int * 2)(valid, valid)
        positions = (ctypes.c_int * 2)(1, 1)
        outputs = (ctypes.c_int * 2)()
        errors["batch_duplicate_slot"] = expect_negative(
            self.lib.qwn_decode_batch(slots, batch_tokens, positions, 2, outputs),
            "batch duplicate slot")
        errors["batch_zero_width"] = expect_negative(
            self.lib.qwn_decode_batch(slots, batch_tokens, positions, 0, outputs),
            "batch zero width")

        after = self.n.stats()
        require(after == baseline,
                f"invalid native calls mutated pool accounting: before={baseline}, after={after}")
        require(self.n.block_table(0) == [] and self.n.block_table(1) == [],
                "invalid native calls created a slot mapping")
        return {"error_codes": errors, "pool_unchanged": True,
                "invalid_checkpoint_host_bytes": 0}

    def reservations_and_reset(self):
        baseline = self.n.reset_all()
        page = baseline["page_tokens"]
        blocks = baseline["total_blocks"]
        max_seq = baseline["max_sequence_tokens"]
        pool_tokens = page * blocks
        require(pool_tokens + 1 <= max_seq,
                "configured context must exceed physical pool by at least one token")

        self.reserve(0, pool_tokens)
        exact = self.n.stats()
        require(exact["remaining_reserved_blocks"] == blocks and
                exact["free_blocks"] == blocks,
                f"exact-fit reservation accounting is wrong: {exact}")
        expect_capacity(self.lib.qwn_paged_reserve(1, page),
                        "reserve after exact fit")
        expect_capacity(self.lib.qwn_paged_reserve(1, pool_tokens + 1),
                        "impossible reservation larger than physical pool")
        expect_zero(self.lib.qwn_reset_slot(0), "reset reserved empty slot")
        require(self.n.stats() == exact,
                "reset of an empty slot did not preserve its exact-fit reservation")
        self.release(0)
        self.reserve(1, page)
        self.release(1)

        self.reserve(0, (blocks - 1) * page)
        one_left = self.n.stats()
        require(one_left["remaining_reserved_blocks"] == blocks - 1,
                f"unexpected one-block-left accounting: {one_left}")
        expect_capacity(self.lib.qwn_paged_reserve(1, 2 * page),
                        "reservation one block short")
        self.reserve(1, page)
        require(self.n.stats()["remaining_reserved_blocks"] == blocks,
                "two reservations did not exactly consume available credits")
        self.release(0)
        self.release(1)
        released = self.n.stats()
        self.release(0)
        self.release(1)
        require(self.n.stats() == released == baseline,
                "repeated release was not idempotent or did not conserve the pool")

        self.reserve(0, 2 * page)
        self.n.prefill(0, self.tokens["alpha"][:8], chunk=8)
        grown = self.n.stats()
        require(grown["free_blocks"] == blocks - 1 and
                grown["remaining_reserved_blocks"] == 1,
                f"reserved growth did not consume one block and one credit: {grown}")
        expect_zero(self.lib.qwn_reset_slot(0), "reset populated reserved slot")
        reset = self.n.stats()
        require(self.n.block_table(0) == [], "reset retained populated slot mappings")
        require(reset["free_blocks"] == blocks and
                reset["remaining_reserved_blocks"] == 2,
                f"slot reset failed to reconstitute the total reservation: {reset}")
        expect_zero(self.lib.qwn_reset_slot(0), "repeat reset reserved slot")
        require(self.n.stats() == reset,
                "repeated slot reset changed preserved reservation accounting")
        self.release(0)
        require(self.n.stats() == baseline, "reservation/reset scenario leaked ownership")
        return {
            "exact_fit_blocks": blocks,
            "one_block_short_error": CAPACITY,
            "impossible_request_error": CAPACITY,
            "reset_preserved_total_blocks": 2,
            "repeated_release_idempotent": True,
        }

    def partial_growth_rollback(self):
        self.n.reset_all()
        page = self.n.expected_page
        blocks = self.n.expected_blocks
        token = self.tokens["alpha"][8]

        # A clean one-token continuation is the GDN+KV oracle for the state that
        # must survive the later failed whole-range preflight.
        self.n.prefill(0, self.tokens["alpha"][:8], chunk=8)
        oracle_token = self.n.decode(0, token, 8)
        oracle_kv = self.n.kv(0, 9)

        self.n.reset_all()
        self.n.prefill(0, self.tokens["alpha"][:8], chunk=8)
        self.reserve(1, (blocks - 2) * page)
        before_stats = self.n.stats()
        require(before_stats["free_blocks"] -
                before_stats["remaining_reserved_blocks"] == 1,
                f"failed-growth setup did not leave one unpromised block: {before_stats}")
        before_table = self.n.block_table(0)
        before_kv = self.n.kv(0, 8)
        attempt = self.tokens["beta"][:2 * page]
        array = (ctypes.c_int * len(attempt))(*attempt)
        expect_capacity(self.lib.qwn_paged_reserve(0, 3 * page),
                        "reservation expansion requiring two unpromised blocks")
        expect_capacity(self.lib.qwn_prefill_chunk(array, len(attempt), 8),
                        "whole-range prefill partial-growth preflight")
        after_stats = self.n.stats()
        after_table = self.n.block_table(0)
        after_kv = self.n.kv(0, 8)
        require(after_stats == before_stats,
                f"failed growth changed pool accounting: before={before_stats}, after={after_stats}")
        require(after_table == before_table,
                f"failed growth changed block table: before={before_table}, after={after_table}")
        self.n.assert_kv_equal(before_kv, after_kv,
                               "failed growth committed-prefix preservation")

        self.release(1)
        replay_token = self.n.decode(0, token, 8)
        replay_kv = self.n.kv(0, 9)
        require(replay_token == oracle_token,
                f"failed growth mutated continuation state: oracle={oracle_token}, replay={replay_token}")
        self.n.assert_kv_equal(oracle_kv, replay_kv,
                               "failed growth GDN/output replay KV")
        self.release(0)
        require(self.n.stats()["free_blocks"] == blocks,
                "partial-growth scenario leaked a physical block")
        return {
            "attempted_new_blocks": 2,
            "unpromised_blocks": 1,
            "capacity_error": CAPACITY,
            "table_unchanged": True,
            "exact_prefix_bytes_unchanged": True,
            "continuation_token": replay_token,
            "continuation_kv_sha256": self.n.kv_digests(replay_kv),
        }

    def _isolated_prompt(self, slot, prompt, steps):
        self.n.reset_all()
        self.reserve(slot, len(prompt) + steps - 1)
        position = self.n.prefill(slot, prompt[:-1], chunk=self.chunk)
        outputs, position = self.continuation(slot, position, [prompt[-1]], steps)
        snapshot = self.n.kv(slot, position)
        return outputs, position, snapshot

    def independent_slots_interleaved(self):
        page = self.n.expected_page
        steps = 4
        prompt_a = self.tokens["alpha"][:page + 1]
        prompt_b = self.tokens["beta"][:page // 2 + 1]
        ref_a = self._isolated_prompt(0, prompt_a, steps)
        ref_b = self._isolated_prompt(1, prompt_b, steps)

        self.n.reset_all()
        self.reserve(0, len(prompt_a) + steps - 1)
        self.reserve(1, len(prompt_b) + steps - 1)
        prefixes = {0: prompt_a[:-1], 1: prompt_b[:-1]}
        offsets = {0: 0, 1: 0}
        # Slot 0 is already live when slot 1 joins.  Thereafter chunks alternate,
        # so both logical positions and pool growth are genuinely interleaved.
        first = min(self.chunk, len(prefixes[0]))
        self.n.prefill(0, prefixes[0][:first], 0, self.chunk)
        offsets[0] = first
        while offsets[0] < len(prefixes[0]) or offsets[1] < len(prefixes[1]):
            for slot in (1, 0):
                left = len(prefixes[slot]) - offsets[slot]
                if left <= 0:
                    continue
                take = min(self.chunk, left)
                require(take % 8 == 0, "interleaved prefill generated a partial tile")
                piece = prefixes[slot][offsets[slot]:offsets[slot] + take]
                self.n.prefill(slot, piece, offsets[slot], self.chunk)
                offsets[slot] += take

        positions = {0: len(prefixes[0]), 1: len(prefixes[1])}
        lasts = {}
        streams = {0: [], 1: []}
        final_inputs = {0: prompt_a[-1], 1: prompt_b[-1]}
        for slot in (0, 1):
            lasts[slot] = self.n.decode(slot, final_inputs[slot], positions[slot])
            positions[slot] += 1
            streams[slot].append(lasts[slot])
        for _ in range(steps - 1):
            for slot in (1, 0):
                lasts[slot] = self.n.decode(slot, lasts[slot], positions[slot])
                positions[slot] += 1
                streams[slot].append(lasts[slot])

        got_a = self.n.kv(0, positions[0])
        got_b = self.n.kv(1, positions[1])
        require(streams[0] == ref_a[0],
                f"slot 0 interleaved tokens differ from isolated scalar path: {streams[0]} vs {ref_a[0]}")
        require(streams[1] == ref_b[0],
                f"slot 1 interleaved tokens differ from isolated scalar path: {streams[1]} vs {ref_b[0]}")
        self.n.assert_kv_equal(got_a, ref_a[2],
                               "slot 0 interleaved vs isolated logical KV")
        self.n.assert_kv_equal(got_b, ref_b[2],
                               "slot 1 interleaved vs isolated logical KV")
        self.release(0)
        self.release(1)
        return {
            "new_slot_joined_mid_prefill": True,
            "interleaved_chunk_tokens": self.chunk,
            "prompt_tokens": {"slot0": len(prompt_a), "slot1": len(prompt_b)},
            "continuation_tokens": streams,
            "slot0_kv_sha256": self.n.kv_digests(got_a),
            "slot1_kv_sha256": self.n.kv_digests(got_b),
            "same_scalar_execution_shape": True,
        }

    def _seed_row(self, slot, prefix, current, total):
        self.reserve(slot, total)
        position = self.n.prefill(slot, prefix, chunk=self.chunk)
        return Row(slot, position, current)

    def _batch_reference_a(self, prefix_a, prefix_b, current_a, decoy_b):
        self.n.reset_all()
        row_a = self._seed_row(0, prefix_a, current_a, len(prefix_a) + 5)
        self.dispatch([row_a])
        self.dispatch([row_a])
        row_decoy = self._seed_row(1, prefix_b, decoy_b, len(prefix_b) + 3)
        for _ in range(3):
            self.dispatch([row_a, row_decoy])
        return list(row_a.outputs), row_a.position, self.n.kv(0, row_a.position)

    def _batch_reference_b(self, prefix_a, prefix_b, decoy_a, current_b):
        self.n.reset_all()
        row_decoy = self._seed_row(0, prefix_a, decoy_a, len(prefix_a) + 5)
        self.dispatch([row_decoy])
        self.dispatch([row_decoy])
        row_b = self._seed_row(1, prefix_b, current_b, len(prefix_b) + 3)
        for _ in range(3):
            self.dispatch([row_decoy, row_b])
        return list(row_b.outputs), row_b.position, self.n.kv(1, row_b.position)

    def batch_mixed_join(self):
        prefix_a = self.tokens["alpha"][:16]
        prefix_b = self.tokens["beta"][:32]
        current_a = self.tokens["alpha"][16]
        current_b = self.tokens["beta"][32]
        ref_a = self._batch_reference_a(
            prefix_a, prefix_b, current_a, self.tokens["gamma"][32])
        ref_b = self._batch_reference_b(
            prefix_a, prefix_b, self.tokens["gamma"][16], current_b)

        self.n.reset_all()
        row_a = self._seed_row(0, prefix_a, current_a, len(prefix_a) + 5)
        self.dispatch([row_a])
        self.dispatch([row_a])
        before_join = self.n.kv(0, row_a.position)
        row_b = self._seed_row(1, prefix_b, current_b, len(prefix_b) + 3)
        after_join = self.n.kv(0, row_a.position)
        self.n.assert_kv_equal(before_join, after_join,
                               "live row while a new slot joins")
        mixed_positions = [row_a.position, row_b.position]
        require(mixed_positions[0] != mixed_positions[1],
                "batch qualification did not create mixed positions")
        for _ in range(3):
            self.dispatch([row_a, row_b])
        got_a = self.n.kv(0, row_a.position)
        got_b = self.n.kv(1, row_b.position)
        require(row_a.outputs == ref_a[0],
                f"batch row 0 differs from row-index-matched isolated companion run: "
                f"{row_a.outputs} vs {ref_a[0]}")
        require(row_b.outputs == ref_b[0],
                f"batch row 1 differs from row-index-matched isolated companion run: "
                f"{row_b.outputs} vs {ref_b[0]}")
        self.n.assert_kv_equal(got_a, ref_a[2],
                               "batch row 0 exact KV vs same-shape reference")
        self.n.assert_kv_equal(got_b, ref_b[2],
                               "batch row 1 exact KV vs same-shape reference")
        self.release(0)
        self.release(1)

        # Exercise the multi-row preflight itself under real exhaustion.  Five
        # mapped pages in the six-page default pool leave one physical page;
        # both rows append at page boundaries and jointly need two.  Neither
        # row may advance, map, or mutate K/V when the aggregate preflight
        # fails.
        self.n.reset_all()
        page = self.n.expected_page
        blocks = self.n.expected_blocks
        long_count = (blocks - 2) * page
        alpha = self.tokens["alpha"]
        long_prefix = (alpha * ((long_count + len(alpha) - 1) // len(alpha)))[:long_count]
        short_prefix = self.tokens["beta"][:page]
        self.n.prefill(0, long_prefix, chunk=self.chunk)
        self.n.prefill(1, short_prefix, chunk=self.chunk)
        failed_batch_stats = self.n.stats()
        require(failed_batch_stats["free_blocks"] == 1,
                f"batch pressure setup did not leave one free page: {failed_batch_stats}")
        failed_tables = {0: self.n.block_table(0), 1: self.n.block_table(1)}
        failed_kv = {0: self.n.kv(0, long_count), 1: self.n.kv(1, page)}
        Pair = ctypes.c_int * 2
        slots = Pair(0, 1)
        tokens = Pair(self.tokens["gamma"][0], self.tokens["alpha"][0])
        positions = Pair(long_count, page)
        outputs = Pair()
        expect_capacity(
            self.lib.qwn_decode_batch(slots, tokens, positions, 2, outputs),
            "width-2 aggregate allocation preflight")
        require(self.n.stats() == failed_batch_stats,
                "failed width-2 batch changed pool accounting")
        require(self.n.block_table(0) == failed_tables[0] and
                self.n.block_table(1) == failed_tables[1],
                "failed width-2 batch changed a row block table")
        self.n.assert_kv_equal(failed_kv[0], self.n.kv(0, long_count),
                               "failed width-2 batch row 0 rollback")
        self.n.assert_kv_equal(failed_kv[1], self.n.kv(1, page),
                               "failed width-2 batch row 1 rollback")
        self.release(0)
        self.release(1)
        require(self.dispatch_counts["scalar_width_1"] >= 6,
                "n=1 scalar dispatch path was not exercised")
        require(self.dispatch_counts["native_batch_width_2"] >= 9,
                "width-2 native batch path was not exercised")
        return {
            "join_positions": mixed_positions,
            "dispatch_counts": dict(self.dispatch_counts),
            "n1_used_scalar_decode": True,
            "native_batch_n1_calls": 0,
            "row0_tokens": row_a.outputs,
            "row1_tokens": row_b.outputs,
            "row0_kv_sha256": self.n.kv_digests(got_a),
            "row1_kv_sha256": self.n.kv_digests(got_b),
            "failed_width2_capacity_error": CAPACITY,
            "failed_width2_tables_and_kv_unchanged": True,
            "row_index_and_width_matched_references": True,
        }

    def apc_full_pages(self):
        self.n.reset_all()
        page = self.n.expected_page
        steps = 3
        prefix = self.tokens["alpha"][:page]
        self.reserve(0, page + steps)
        self.n.prefill(0, prefix, chunk=self.chunk)
        donor_kv = self.n.kv(0, page)
        donor_table = self.n.block_table(0)
        require(len(donor_table) == 1, f"full-page donor table is not one page: {donor_table}")
        before_save = self.n.stats()
        checkpoint = int(self.lib.qwn_paged_ckpt_save(0, page))
        require(checkpoint > 0, f"full-page checkpoint save returned {checkpoint}")
        host_bytes = int(self.lib.qwn_paged_ckpt_host_bytes(checkpoint))
        require(host_bytes > 0, "full-page checkpoint has no GDN state image")
        after_save = self.n.stats()
        require(after_save["free_blocks"] == before_save["free_blocks"],
                "saving full pages allocated an unnecessary physical block")
        require(after_save["checkpoint_blocks"] == 1,
                f"full-page checkpoint ownership is wrong: {after_save}")

        self.reserve(1, page + steps)
        adopted = int(self.lib.qwn_paged_ckpt_adopt(1, checkpoint))
        require(adopted == page,
                f"full-page checkpoint adopt returned {adopted}, expected {page}")
        recipient_table = self.n.block_table(1)
        require(recipient_table == donor_table,
                f"full page was not shared: donor={donor_table}, recipient={recipient_table}")
        recipient_kv = self.n.kv(1, page)
        self.n.assert_kv_equal(donor_kv, recipient_kv,
                               "full-page checkpoint adopted logical KV")

        forced = self.tokens["gamma"][page]
        donor_stream, donor_pos = self.continuation(0, page, [forced], steps)
        donor_final = self.n.kv(0, donor_pos)
        recipient_prefix_table = self.n.block_table(1)
        recipient_prefix_kv = self.n.kv(1, page)
        self.release(0)
        require(self.n.block_table(1) == recipient_prefix_table,
                "donor release changed a live full-page recipient table")
        self.n.assert_kv_equal(recipient_prefix_kv, self.n.kv(1, page),
                               "donor release changed live recipient KV")
        expect_zero(self.lib.qwn_paged_ckpt_free(checkpoint),
                    "evict full-page checkpoint")
        require(int(self.lib.qwn_paged_ckpt_host_bytes(checkpoint)) == 0,
                "evicted full-page checkpoint retained host ownership")
        require(self.n.block_table(1) == recipient_prefix_table,
                "checkpoint eviction changed a live full-page recipient table")
        recipient_stream, recipient_pos = self.continuation(1, page, [forced], steps)
        recipient_final = self.n.kv(1, recipient_pos)
        require(recipient_stream == donor_stream,
                f"full-page APC continuation mismatch: {recipient_stream} vs {donor_stream}")
        self.n.assert_kv_equal(donor_final, recipient_final,
                               "full-page APC GDN-matched continuation KV")
        self.release(1)
        return {
            "checkpoint_id": checkpoint,
            "committed": page,
            "shared_full_block": donor_table[0],
            "private_tail": False,
            "gdn_host_bytes": host_bytes,
            "donor_released_before_recipient": True,
            "checkpoint_evicted_while_recipient_live": True,
            "continuation_tokens": recipient_stream,
            "logical_kv_sha256": self.n.kv_digests(recipient_final),
        }

    def apc_private_partial_tail(self):
        self.n.reset_all()
        page = self.n.expected_page
        committed = page + 8
        steps = 3
        prefix = self.tokens["beta"][:committed]
        self.reserve(0, committed + steps)
        self.n.prefill(0, prefix, chunk=self.chunk)
        donor_kv = self.n.kv(0, committed)
        donor_table = self.n.block_table(0)
        require(len(donor_table) == 2,
                f"partial checkpoint donor table is not two pages: {donor_table}")
        before_save = self.n.stats()
        checkpoint = int(self.lib.qwn_paged_ckpt_save(0, committed))
        require(checkpoint > 0, f"partial checkpoint save returned {checkpoint}")
        after_save = self.n.stats()
        require(after_save["free_blocks"] == before_save["free_blocks"] - 1,
                "partial checkpoint did not allocate exactly one immutable private tail")
        require(after_save["checkpoint_blocks"] == 2,
                f"partial checkpoint must own one full page and one private tail: {after_save}")
        host_bytes = int(self.lib.qwn_paged_ckpt_host_bytes(checkpoint))
        require(host_bytes > 0, "partial checkpoint has no GDN state image")

        self.reserve(1, committed + steps)
        adopted = int(self.lib.qwn_paged_ckpt_adopt(1, checkpoint))
        require(adopted == committed,
                f"partial checkpoint adopt returned {adopted}, expected {committed}")
        recipient_table = self.n.block_table(1)
        require(len(recipient_table) == 2,
                f"partial recipient table is not two pages: {recipient_table}")
        require(recipient_table[0] == donor_table[0],
                "checkpoint did not share its immutable full page")
        require(recipient_table[1] != donor_table[1],
                f"writable partial tail was aliased: donor={donor_table}, recipient={recipient_table}")
        recipient_kv = self.n.kv(1, committed)
        self.n.assert_kv_equal(donor_kv, recipient_kv,
                               "private-tail checkpoint adopted logical KV")

        forced = self.tokens["gamma"][committed]
        donor_stream, donor_pos = self.continuation(0, committed, [forced], steps)
        donor_final = self.n.kv(0, donor_pos)
        recipient_prefix = self.n.kv(1, committed)
        recipient_table_live = self.n.block_table(1)
        self.release(0)
        self.n.assert_kv_equal(recipient_prefix, self.n.kv(1, committed),
                               "partial donor release changed live recipient KV")
        require(self.n.block_table(1) == recipient_table_live,
                "partial donor release changed live recipient table")
        expect_zero(self.lib.qwn_paged_ckpt_free(checkpoint),
                    "evict partial checkpoint")
        require(int(self.lib.qwn_paged_ckpt_host_bytes(checkpoint)) == 0,
                "evicted partial checkpoint retained host ownership")
        self.n.assert_kv_equal(recipient_prefix, self.n.kv(1, committed),
                               "partial checkpoint eviction changed live recipient KV")
        recipient_stream, recipient_pos = self.continuation(
            1, committed, [forced], steps)
        recipient_final = self.n.kv(1, recipient_pos)
        require(recipient_stream == donor_stream,
                f"partial APC continuation mismatch: {recipient_stream} vs {donor_stream}")
        self.n.assert_kv_equal(donor_final, recipient_final,
                               "partial APC GDN-matched continuation KV")
        self.release(1)
        return {
            "checkpoint_id": checkpoint,
            "committed": committed,
            "shared_full_block": donor_table[0],
            "donor_private_tail": donor_table[1],
            "recipient_private_tail": recipient_table[1],
            "tails_distinct": True,
            "gdn_host_bytes": host_bytes,
            "donor_released_before_recipient": True,
            "checkpoint_evicted_while_recipient_live": True,
            "continuation_tokens": recipient_stream,
            "logical_kv_sha256": self.n.kv_digests(recipient_final),
        }

    def constrained_checkpoint_rollback(self):
        self.n.reset_all()
        page = self.n.expected_page
        blocks = self.n.expected_blocks
        committed = page + 8
        prefix = self.tokens["gamma"][:committed]
        self.n.prefill(0, prefix, chunk=self.chunk)
        prefix_kv = self.n.kv(0, committed)
        prefix_table = self.n.block_table(0)
        require(len(prefix_table) == 2, "constrained APC setup needs a partial page")

        # Promise every currently free block to another slot.  Optional APC may
        # not consume those credits even though the physical free list is nonempty.
        self.reserve(1, (blocks - 2) * page)
        before_save = self.n.stats()
        require(before_save["free_blocks"] == blocks - 2 and
                before_save["remaining_reserved_blocks"] == blocks - 2,
                f"save-pressure setup did not promise every free block: {before_save}")
        before_save_kv = self.n.kv(0, committed)
        expect_capacity(self.lib.qwn_paged_ckpt_save(0, committed),
                        "partial checkpoint save with all free blocks promised")
        after_failed_save = self.n.stats()
        require(after_failed_save == before_save,
                "failed checkpoint save changed pool/ref/reservation accounting")
        require(self.n.block_table(0) == prefix_table,
                "failed checkpoint save changed donor mappings")
        self.n.assert_kv_equal(before_save_kv, self.n.kv(0, committed),
                               "failed checkpoint save donor KV rollback")

        self.release(1)
        checkpoint = int(self.lib.qwn_paged_ckpt_save(0, committed))
        require(checkpoint > 0,
                f"checkpoint save remained blocked after credits released: {checkpoint}")
        self.release(0)
        require(self.n.block_table(0) == [], "donor release retained mappings")

        # The checkpoint now owns a shared full page plus a private checkpoint
        # tail.  Promise all remaining free pages, then prove adopt cannot steal
        # one for its required target-private tail and leaves the target empty.
        self.reserve(0, (blocks - 2) * page)
        expect_zero(self.lib.qwn_reset_slot(1), "reset adopt target")
        before_adopt = self.n.stats()
        target_table = self.n.block_table(1)
        expect_capacity(self.lib.qwn_paged_ckpt_adopt(1, checkpoint),
                        "partial checkpoint adopt with all free blocks promised")
        require(self.n.stats() == before_adopt,
                "failed checkpoint adopt changed pool/ref/reservation accounting")
        require(self.n.block_table(1) == target_table == [],
                "failed checkpoint adopt published target mappings")

        self.release(0)
        adopted = int(self.lib.qwn_paged_ckpt_adopt(1, checkpoint))
        require(adopted == committed,
                f"checkpoint adopt remained blocked after credits released: {adopted}")
        adopted_kv = self.n.kv(1, committed)
        self.n.assert_kv_equal(prefix_kv, adopted_kv,
                               "successful retry after failed adopt")
        live_table = self.n.block_table(1)
        expect_zero(self.lib.qwn_paged_ckpt_free(checkpoint),
                    "free constrained checkpoint")
        require(self.n.block_table(1) == live_table,
                "checkpoint free invalidated recipient after successful retry")
        self.n.assert_kv_equal(adopted_kv, self.n.kv(1, committed),
                               "checkpoint free changed live retry recipient")
        self.release(1)
        require(self.n.stats()["free_blocks"] == blocks,
                "constrained checkpoint rollback scenario leaked a block")
        return {
            "real_total_blocks": blocks,
            "real_free_blocks_promised": blocks - 2,
            "failed_save_error": CAPACITY,
            "failed_adopt_error": CAPACITY,
            "failed_save_transactional": True,
            "failed_adopt_transactional": True,
            "promised_credits_not_stolen": True,
            "successful_retry_after_release": True,
            "adopted_kv_sha256": self.n.kv_digests(adopted_kv),
        }

    def server_apc_exact_key_host_regression(self):
        """Load the real server Engine class without executing server startup."""
        source_path = Path(__file__).with_name("serve_openai_xpu.py")
        source = source_path.read_text()
        parsed = ast.parse(source, filename=str(source_path))
        engine_nodes = [
            node for node in parsed.body
            if isinstance(node, ast.ClassDef) and node.name == "Engine"
        ]
        require(len(engine_nodes) == 1,
                f"expected one Engine ClassDef in {source_path}, found {len(engine_nodes)}")
        engine_node = engine_nodes[0]
        module = ast.Module(body=[engine_node], type_ignores=[])
        ast.fix_missing_locations(module)
        server_args = argparse.Namespace(
            prefix_cache_min=128,
            ctx=2048,
            prefix_cache=True,
        )
        namespace = {
            "__name__": "_paged_state_real_engine_probe",
            "threading": threading,
            "args": server_args,
            "PREFILL_CHUNK_BUSY": 64,
            "PREFILL_CHUNK_IDLE": 512,
        }
        exec(compile(module, str(source_path), "exec"), namespace)
        engine_type = namespace["Engine"]
        engine = engine_type()
        require(isinstance(engine, threading.Thread),
                "AST-loaded Engine is not the real threading.Thread subclass")
        require(not engine.is_alive(),
                "host APC regression must never start the Engine thread")
        require(engine.chunk == 128 and engine.chunk_busy == 64 and
                engine.chunk_idle == 512 and engine.maxseq == 2048,
                "AST-loaded Engine did not consume the controlled real globals")

        prefix = tuple(range(1000, 1128))
        adversary = (prefix[0] + 1,) + prefix[1:]
        desired_hash = hash(prefix)

        class CollidingTuple(tuple):
            def __hash__(self):
                return desired_hash

        collision_key = CollidingTuple(adversary)
        require(collision_key != prefix and hash(collision_key) == hash(prefix),
                "adversarial tuple did not create a real unequal-key hash collision")
        collision_entry = (len(prefix), 101, 3)
        exact_entry = (len(prefix), 202, 5)
        suffix_a = (2001, 2002, 2003, 2004)
        suffix_b = (3001, 3002, 3003, 3004)
        full_prompt = prefix + suffix_a
        full_entry = (len(full_prompt), 303, 7)
        engine._register_cache(collision_key, collision_entry)
        engine._register_cache(prefix, exact_entry)
        engine._register_cache(full_prompt, full_entry)

        require(engine.cache_lengths == {
                    len(prefix): 2, len(full_prompt): 1},
                f"real Engine cache length metadata is wrong: {engine.cache_lengths}")
        require(engine.cache_bytes == 15 and len(engine.cache_lru) == 3,
                "real Engine scalar cache metadata did not track registrations")
        lookup_a = engine._lookup(full_prompt)
        lookup_b = engine._lookup(prefix + suffix_b)
        require(lookup_a is not None and lookup_a[0] == prefix and
                lookup_a[1] is exact_entry,
                "hash collision or full-prompt entry defeated exact shorter-prefix lookup")
        require(lookup_b is not None and lookup_b[0] == prefix and
                lookup_b[1] is exact_entry,
                "same prefix with a different suffix did not select the exact entry")
        require(lookup_a[1] is not collision_entry and
                lookup_b[1] is not collision_entry,
                "unequal adversarial hash-collision entry was accepted")

        server_args.prefix_cache = False
        require(engine._lookup(full_prompt) is None,
                "real Engine _lookup did not honor APC disabled")
        server_args.prefix_cache = True

        # Exercise production _store first-sighting retention on a separate
        # real Engine instance.  Every 2k-token tuple is transient; only its
        # integer fingerprint may survive.  A plain exact tuple cache entry is
        # kept alongside the sightings to guard the authority/hint distinction.
        server_args.prefix_cache_max_mb = 1
        hint_engine = engine_type()
        exact_cache_key = tuple(range(5000, 5128))
        exact_cache_entry = (len(exact_cache_key), 404, 11)
        hint_engine._register_cache(exact_cache_key, exact_cache_entry)
        cache_before = dict(hint_engine.cache)
        cache_lru_before = list(hint_engine.cache_lru)
        cache_lengths_before = dict(hint_engine.cache_lengths)
        cache_bytes_before = hint_engine.cache_bytes
        hint_count = 4096
        hint_tokens = 2048
        shared_tail = (17,) * (hint_tokens - 1)
        selected_fingerprints = set()
        candidate = 0
        last_prefix = None
        while len(selected_fingerprints) < hint_count:
            last_prefix = (400_000 + candidate,) + shared_tail
            candidate += 1
            fingerprint = hash(last_prefix)
            if fingerprint in selected_fingerprints:
                continue
            ids = last_prefix + (23,)
            hint_engine._store(ids, hint_tokens, 0)
            require(fingerprint in hint_engine.seen,
                    "real Engine _store did not retain first-sighting fingerprint")
            selected_fingerprints.add(fingerprint)
        require(hint_engine.seen == selected_fingerprints,
                "real Engine retained data other than selected first-sighting fingerprints")
        require(all(type(value) is int for value in hint_engine.seen),
                "real Engine first-sighting hints retained a non-integer payload")
        require(hint_engine.cache == cache_before and
                hint_engine.cache_lru == cache_lru_before and
                hint_engine.cache_lengths == cache_lengths_before and
                hint_engine.cache_bytes == cache_bytes_before,
                "unique first sightings mutated exact APC cache ownership metadata")
        require(all(type(key) is tuple for key in hint_engine.cache),
                "real Engine cache authority retained a non-exact-tuple key")
        retained_bytes = (
            sys.getsizeof(hint_engine.seen) +
            sum(sys.getsizeof(value) for value in hint_engine.seen)
        )
        tuple_counterfactual_bytes = (
            hint_count * sys.getsizeof(last_prefix)
        )
        require(retained_bytes * 100 < tuple_counterfactual_bytes,
                "first-sighting integer retention is not compact versus token tuples")

        # Bound the rollover without allocating 200k long tuples.  The set is
        # production Engine metadata; the one new long sighting still traverses
        # the real _store path and must leave the documented strict cap intact.
        cap_engine = engine_type()
        cap_engine.seen = set(range(200_000))
        cap_prefix = (900_000,) + shared_tail
        while hash(cap_prefix) in cap_engine.seen:
            cap_prefix = (cap_prefix[0] + 1,) + shared_tail
        cap_engine._store(cap_prefix + (29,), hint_tokens, 0)
        require(len(cap_engine.seen) <= 200_000,
                f"real Engine first-sighting cap exceeded: {len(cap_engine.seen)}")
        require(all(type(value) is int for value in cap_engine.seen),
                "real Engine rollover retained a non-integer hint")
        require(not hint_engine.is_alive() and not cap_engine.is_alive(),
                "host first-sighting regression started an Engine thread")
        require(not engine.is_alive(),
                "host APC regression unexpectedly started an Engine thread")
        return {
            "source": str(source_path),
            "source_sha256": byte_hash(source.encode()),
            "engine_class_line": engine_node.lineno,
            "actual_engine_class_ast_compiled": True,
            "engine_thread_started": False,
            "prefix_tokens": len(prefix),
            "full_prompt_tokens": len(full_prompt),
            "deliberate_unequal_hash_collision_rejected": True,
            "exact_tuple_selected_amid_collision": True,
            "same_prefix_different_suffix_matched": True,
            "full_prompt_checkpoint_skipped_for_shorter_prefix": True,
            "apc_disabled_returned_none": True,
            "first_sighting_hints": {
                "unique_long_prefixes": hint_count,
                "tokens_per_prefix": hint_tokens,
                "retained_value_type": "int",
                "retained_fingerprints": len(hint_engine.seen),
                "retained_bytes_including_set": retained_bytes,
                "full_tuple_counterfactual_bytes": tuple_counterfactual_bytes,
                "strict_post_rollover_count": len(cap_engine.seen),
                "strict_cap": 200000,
                "exact_cache_keys_remained_plain_tuples": True,
                "exact_cache_metadata_unchanged": True,
            },
            "cache_lengths": {
                str(length): count
                for length, count in sorted(engine.cache_lengths.items())
            },
            "cache_bytes": engine.cache_bytes,
        }

    def _flat_reset_all(self):
        expect_zero(self.lib.qwn_reset_state(), "flat qwn_reset_state")

    def _flat_prompt_replay(self, prompt, steps):
        self._flat_reset_all()
        position = self.n.prefill(0, prompt[:-1], chunk=self.chunk)
        outputs, position = self.continuation(
            0, position, [prompt[-1]], steps)
        snapshot = self.n.kv(0, position)
        return {
            "outputs": outputs,
            "committed": position,
            "kv": snapshot,
        }

    def _flat_batch_arm(self, row0_spec, row1_spec, steps):
        self._flat_reset_all()
        rows = []
        specs = (row0_spec, row1_spec)
        for slot, (word, length) in enumerate(specs):
            prefix = self.tokens[word][:length]
            position = self.n.prefill(slot, prefix, chunk=self.chunk)
            rows.append(Row(slot, position, self.tokens[word][length]))
        for _ in range(steps):
            self.dispatch(rows)
        return [
            {
                "outputs": list(row.outputs),
                "committed": row.position,
                "kv": self.n.kv(row.slot, row.position),
            }
            for row in rows
        ]

    def flat_regression(self):
        require(int(self.lib.qwn_paged_enabled()) == 0,
                "flat regression entered with paged ownership enabled")
        page = self.n.expected_page
        steps = 4

        # Same-build chunk+scalar continuation replay.  This is deliberately
        # reset/replayed in one process so the external unset-vs-0 comparison
        # is comparing deterministic logical results rather than warm state.
        prompt = self.tokens["alpha"][:page + 1]
        first = self._flat_prompt_replay(prompt, steps)
        second = self._flat_prompt_replay(prompt, steps)
        require(first["outputs"] == second["outputs"],
                "flat chunk/scalar continuation changed across reset replay")
        require(first["committed"] == second["committed"],
                "flat chunk/scalar replay committed a different position")
        self.n.assert_kv_equal(first["kv"], second["kv"],
                               "flat chunk/scalar reset replay")

        # Three width-2 arms keep target row index and native batch shape fixed
        # while changing only the other row.  This proves slot independence
        # without comparing RC8 batch arithmetic against scalar GEMV.
        ref_a = self._flat_batch_arm(
            ("alpha", 16), ("gamma", 32), steps)
        ref_b = self._flat_batch_arm(
            ("gamma", 16), ("beta", 32), steps)
        pair = self._flat_batch_arm(
            ("alpha", 16), ("beta", 32), steps)
        require(pair[0]["outputs"] == ref_a[0]["outputs"],
                "flat batch row 0 changed with a different companion")
        require(pair[1]["outputs"] == ref_b[1]["outputs"],
                "flat batch row 1 changed with a different companion")
        self.n.assert_kv_equal(pair[0]["kv"], ref_a[0]["kv"],
                               "flat batch row 0 same-shape logical KV")
        self.n.assert_kv_equal(pair[1]["kv"], ref_b[1]["kv"],
                               "flat batch row 1 same-shape logical KV")

        # Real flat checkpoint image: every full-attention prefix plus all GDN
        # state is copied through host storage, restored into another slot, and
        # the storage is freed before the recipient continues.
        self._flat_reset_all()
        committed = page + 8
        prefix = self.tokens["beta"][:committed]
        self.n.prefill(0, prefix, chunk=self.chunk)
        donor_prefix = self.n.kv(0, committed)
        checkpoint_bytes = int(self.lib.qwn_ckpt_bytes(committed))
        require(checkpoint_bytes > 0,
                "flat qwn_ckpt_bytes returned no real checkpoint storage")
        blob = self.lib.qwn_host_alloc(checkpoint_bytes)
        require(blob, f"flat qwn_host_alloc({checkpoint_bytes}) failed")
        try:
            expect_zero(self.lib.qwn_ckpt_save(0, committed, blob),
                        "flat qwn_ckpt_save")
            forced = self.tokens["gamma"][committed]
            donor_stream, donor_position = self.continuation(
                0, committed, [forced], steps)
            donor_final = self.n.kv(0, donor_position)
            expect_zero(self.lib.qwn_reset_slot(1),
                        "flat reset checkpoint recipient")
            expect_zero(self.lib.qwn_ckpt_restore(1, committed, blob),
                        "flat qwn_ckpt_restore")
            recipient_prefix = self.n.kv(1, committed)
            self.n.assert_kv_equal(
                donor_prefix, recipient_prefix,
                "flat checkpoint restored logical prefix")
            self.lib.qwn_host_free(blob)
            blob = None
            recipient_stream, recipient_position = self.continuation(
                1, committed, [forced], steps)
            recipient_final = self.n.kv(1, recipient_position)
        finally:
            if blob:
                self.lib.qwn_host_free(blob)
        require(recipient_stream == donor_stream,
                "flat checkpoint continuation differs after real restore")
        self.n.assert_kv_equal(
            donor_final, recipient_final,
            "flat checkpoint GDN-matched continuation logical KV")
        return {
            "effective_layout": "flat",
            "chunk_scalar_replay": {
                "prompt_tokens": len(prompt),
                "continuation_tokens": second["outputs"],
                "committed": second["committed"],
                "logical_kv_sha256": self.n.kv_digests(second["kv"]),
            },
            "independent_width2_batch": {
                "row0_tokens": pair[0]["outputs"],
                "row1_tokens": pair[1]["outputs"],
                "row0_logical_kv_sha256": self.n.kv_digests(pair[0]["kv"]),
                "row1_logical_kv_sha256": self.n.kv_digests(pair[1]["kv"]),
                "row_index_and_width_matched_references": True,
            },
            "flat_checkpoint": {
                "committed": committed,
                "host_bytes": checkpoint_bytes,
                "host_storage_freed_before_recipient_continuation": True,
                "continuation_tokens": recipient_stream,
                "logical_kv_sha256": self.n.kv_digests(recipient_final),
            },
        }

    def _long_batch_arm(self, row0_spec, row1_spec, steps):
        self.n.reset_all()
        specs = (row0_spec, row1_spec)
        totals = [length + steps for _, length in specs]
        # Both promises are installed before either sequence allocates a page:
        # this is one authoritative aggregate reservation ledger.
        self.reserve(0, totals[0])
        self.reserve(1, totals[1])
        reserved = self.n.stats()
        expected_credits = sum(
            (total + self.n.expected_page - 1) // self.n.expected_page
            for total in totals)
        require(reserved["remaining_reserved_blocks"] == expected_credits,
                f"long arm reservation ledger is not exact: {reserved}")

        rows = []
        for slot, (word, length) in enumerate(specs):
            prefix = self.tokens[word][:length]
            require(len(prefix) == length,
                    f"token calibration is shorter than long {word} prefix {length}")
            position = self.n.prefill(slot, prefix, chunk=self.chunk)
            rows.append(Row(slot, position, self.tokens[word][0]))
        prepared = self.n.stats()
        tables_before = [self.n.block_table(0), self.n.block_table(1)]
        for _ in range(steps):
            self.dispatch(rows)
        result_rows = []
        for row in rows:
            result_rows.append({
                "slot": row.slot,
                "prompt_word": specs[row.slot][0],
                "prompt_tokens": specs[row.slot][1],
                "final_committed": row.position,
                "continuation_tokens": list(row.outputs),
                "logical_kv": self.n.kv_hashes(row.slot, row.position),
                "mapped_blocks_before_decode": len(tables_before[row.slot]),
                "mapped_blocks_after_decode": len(self.n.block_table(row.slot)),
            })
        self.release(0)
        self.release(1)
        return {
            "reserved_together_before_prefill": True,
            "reservation_ledger": reserved,
            "prepared_pool": prepared,
            "rows": result_rows,
        }

    def long_pair_three_arm(self, length_a, length_b, steps):
        require(length_a > 0 and length_b > 0 and steps > 0,
                "long three-arm gate requires positive lengths and steps")
        short = 8
        reference_a = self._long_batch_arm(
            ("alpha", length_a), ("gamma", short), steps)
        reference_b = self._long_batch_arm(
            ("gamma", short), ("beta", length_b), steps)
        combined = self._long_batch_arm(
            ("alpha", length_a), ("beta", length_b), steps)
        ref_a = reference_a["rows"][0]
        ref_b = reference_b["rows"][1]
        got_a = combined["rows"][0]
        got_b = combined["rows"][1]
        require(got_a["continuation_tokens"] == ref_a["continuation_tokens"],
                "long row A continuation changed with a different width-2 companion")
        require(got_b["continuation_tokens"] == ref_b["continuation_tokens"],
                "long row B continuation changed with a different width-2 companion")
        require(got_a["logical_kv"] == ref_a["logical_kv"],
                "long row A exact logical KV digest changed with its companion")
        require(got_b["logical_kv"] == ref_b["logical_kv"],
                "long row B exact logical KV digest changed with its companion")
        return {
            "prompt_tokens": {"A": length_a, "B": length_b, "short_decoy": short},
            "decode_steps": steps,
            "prefill_chunk_tokens": self.chunk,
            "arms": {
                "A_plus_short_decoy": reference_a,
                "short_decoy_plus_B": reference_b,
                "A_plus_B": combined,
            },
            "true_native_batch_width": 2,
            "row_index_matched": True,
            "same_chunking_per_target": True,
            "one_reservation_ledger_per_arm": True,
            "exact_continuations_match": True,
            "exact_logical_kv_hashes_match": True,
        }
    def stale_ids_and_repeated_init(self):
        self.n.reset_all()
        page = self.n.expected_page
        prefix = self.tokens["alpha"][:page]
        self.n.prefill(0, prefix, chunk=self.chunk)
        stale_reset = int(self.lib.qwn_paged_ckpt_save(0, page))
        require(stale_reset > 0, f"checkpoint before reset returned {stale_reset}")
        self.n.reset_all()
        require(int(self.lib.qwn_paged_ckpt_host_bytes(stale_reset)) == 0,
                "reset-all left a stale checkpoint host allocation discoverable")
        expect_negative(self.lib.qwn_paged_ckpt_adopt(1, stale_reset),
                        "adopt id stale across reset")
        expect_negative(self.lib.qwn_paged_ckpt_free(stale_reset),
                        "free id stale across reset")
        require(self.n.block_table(1) == [], "stale reset id mutated target")

        self.n.prefill(0, prefix, chunk=self.chunk)
        stale_free = int(self.lib.qwn_paged_ckpt_save(0, page))
        require(stale_free > stale_reset,
                f"checkpoint ids were reused across reset: {stale_reset}, {stale_free}")
        self.n.free()
        require(int(self.lib.qwn_paged_enabled()) == 0,
                "qwn_free left paged mode attested")
        require(int(self.lib.qwn_paged_ckpt_host_bytes(stale_free)) == 0,
                "qwn_free left checkpoint host ownership discoverable")
        expect_negative(self.lib.qwn_paged_ckpt_adopt(1, stale_free),
                        "adopt while model freed")
        expect_negative(self.lib.qwn_paged_ckpt_free(stale_free),
                        "free checkpoint while model freed")

        self.n.init()
        expect_zero(self.lib.qwn_reset_slot(1), "reset target after first reinit")
        expect_negative(self.lib.qwn_paged_ckpt_adopt(1, stale_free),
                        "adopt id stale across free/init")
        require(self.n.block_table(1) == [], "stale free/init id mutated target")
        self.n.prefill(0, prefix, chunk=self.chunk)
        stale_second_free = int(self.lib.qwn_paged_ckpt_save(0, page))
        require(stale_second_free > stale_free,
                "checkpoint ids were reused after model reinitialization")
        self.n.free()
        self.n.init()
        expect_zero(self.lib.qwn_reset_slot(1), "reset target after second reinit")
        expect_negative(self.lib.qwn_paged_ckpt_adopt(1, stale_second_free),
                        "adopt id stale across second free/init")
        require(self.n.block_table(1) == [], "second stale id mutated target")
        final_stats = self.n.stats()
        require(final_stats["free_blocks"] == final_stats["total_blocks"] and
                final_stats["active_blocks"] == 0 and
                final_stats["checkpoint_blocks"] == 0 and
                final_stats["remaining_reserved_blocks"] == 0,
                f"repeated init/free did not return a pristine pool: {final_stats}")
        return {
            "stale_reset_id": stale_reset,
            "stale_free_id": stale_free,
            "stale_second_free_id": stale_second_free,
            "ids_never_reused": True,
            "reset_invalidated_ids": True,
            "free_init_invalidated_ids": True,
            "completed_init_free_cycles": 2,
            "one_device_context_process": True,
            "final_pool_stats": final_stats,
        }


def configure_environment(args):
    require(args.page in (128, 256), "--page must be 128 or 256")
    require(args.slots >= 2, "--slots must be at least 2")
    require(args.chunk >= 8 and args.chunk % 8 == 0,
            "--chunk must be a positive multiple of 8")
    long_enabled = args.long_a > 0 or args.long_b > 0
    require((args.long_a > 0) == (args.long_b > 0),
            "--long-a and --long-b must be supplied together")
    require(not (args.flat_regression and args.long_only),
            "--flat-regression and --long-only are mutually exclusive")
    require(not args.flat_regression or not long_enabled,
            "--flat-regression does not accept long prompt options")

    if args.flat_regression:
        flat_flag = os.environ.get("TQ_XPU_PAGED")
        require(flat_flag is None or flat_flag == "0",
                "--flat-regression requires TQ_XPU_PAGED unset or exactly 0")
        context = args.ctx or 2048
        require(context >= 2048,
                "--ctx must be at least 2048 for stable historical flat sizing")
        warm = max(2 * args.page, args.chunk)
        require(warm <= context, "prefill warm size exceeds configured context")
        explicit = {
            "TQ_XPU_TP": "1",
            "TQ_XPU_SLOTS": str(args.slots),
            "TQ_CTX": str(context),
            "TQ_XPU_PREFILL_CHUNK": str(warm),
        }
        for key, value in explicit.items():
            os.environ[key] = value
        # Deliberately do not create, remove, or rewrite TQ_XPU_PAGED here:
        # Main invokes fresh unset and explicit-0 processes and compares their
        # emitted logical results.
        return explicit

    require(args.blocks >= 6,
            "--blocks must be at least 6 for real APC pressure scenarios")
    require(not args.long_only or long_enabled,
            "--long-only requires positive --long-a and --long-b")
    require(not long_enabled or args.long_only,
            "long qualification runs separately; use --long-only")
    if long_enabled:
        require(args.long_a % 8 == 0 and args.long_b % 8 == 0,
                "long prompt lengths must be multiples of 8")
        require(args.long_steps > 0, "--long-steps must be positive")
        long_blocks = sum(
            (length + args.long_steps + args.page - 1) // args.page
            for length in (args.long_a, args.long_b))
        require(args.blocks >= long_blocks,
                f"--blocks={args.blocks} cannot hold long pair; need at least {long_blocks}")
    pool_tokens = args.page * args.blocks
    if args.long_only:
        minimum_context = max(
            1024, args.long_a + args.long_steps,
            args.long_b + args.long_steps)
    else:
        minimum_context = max(1024, pool_tokens + args.page)
    context = args.ctx or minimum_context
    if not args.long_only:
        require(context >= pool_tokens + 1,
                "--ctx must exceed --page * --blocks for the impossible-capacity case")
    require(context >= args.long_a + args.long_steps and
            context >= args.long_b + args.long_steps,
            "--ctx is shorter than a requested long continuation")
    require(context % 8 == 0, "--ctx must be a multiple of 8")
    warm = max(2 * args.page, args.chunk)
    require(warm <= context, "prefill warm size exceeds configured context")
    explicit = {
        "TQ_XPU_PAGED": "1",
        "TQ_XPU_TP": "1",
        "TQ_XPU_SLOTS": str(args.slots),
        "TQ_CTX": str(context),
        "TQ_XPU_KV_PAGE": str(args.page),
        "TQ_XPU_KV_POOL_TOKENS": str(pool_tokens),
        "TQ_XPU_PREFILL_CHUNK": str(warm),
    }
    for key, value in explicit.items():
        os.environ[key] = value
    os.environ.pop("TQ_XPU_KV_POOL_MB", None)
    return explicit


def load_token_sets(model_dir, counts, vocab_size):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(f"transformers tokenizer import failed: {exc}") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.expanduser(model_dir), trust_remote_code=True)
    token_sets = {}
    report = {}
    for word in ("alpha", "beta", "gamma"):
        count = int(counts[word])
        text = (f" {word}") * count
        ids = [int(token) for token in tokenizer.encode(
            text, add_special_tokens=False)]
        require(len(ids) == count,
                f"tokenizer calibration failed: ' {word}'*{count} produced {len(ids)} tokens")
        require(all(0 <= token < vocab_size for token in ids),
                f"tokenizer emitted an id outside native vocab for {word}")
        token_sets[word] = ids
        report[word] = {
            "text_repetitions": count,
            "token_count": len(ids),
            "token_ids_sha256": token_hash(ids),
            "first_token_id": ids[0],
        }
    return token_sets, report


def file_identity(lib_path, tqf_path):
    library = Path(lib_path).expanduser().resolve()
    model = Path(tqf_path).expanduser().resolve()
    with library.open("rb") as handle:
        library_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    with model.open("rb") as handle:
        model_header = hashlib.sha256(handle.read(65536)).hexdigest()
    model_stat = model.stat()
    return {
        "library": {"path": str(library), "sha256": library_hash},
        "model": {
            "path": str(model),
            "size": model_stat.st_size,
            "mtime_ns": model_stat.st_mtime_ns,
            "header_sha256": model_header,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lib", default="xpu/build/libforward_qwen_xpu_paged_candidate.so")
    parser.add_argument(
        "--tqf", default="~/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf")
    parser.add_argument("--model-dir", default="~/models/knivesysl")
    parser.add_argument("--page", type=int, choices=(128, 256), default=128)
    parser.add_argument("--blocks", type=int, default=6,
                        help="real physical pool blocks; minimum 6")
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--ctx", type=int, default=0,
                        help="native context; 0 chooses a safe value above pool capacity")
    parser.add_argument("--chunk", type=int, default=32,
                        help="same-path prefill chunk size, a multiple of 8")
    parser.add_argument("--long-a", type=int, default=0,
                        help="optional exact A prompt tokens for the separate long gate")
    parser.add_argument("--long-b", type=int, default=0,
                        help="optional exact B prompt tokens for the separate long gate")
    parser.add_argument("--long-steps", type=int, default=4,
                        help="true width-2 continuation steps in each long arm")
    parser.add_argument("--long-only", action="store_true",
                        help="run only the three-arm long same-shape qualification")
    parser.add_argument(
        "--flat-regression", action="store_true",
        help="preserve external unset/0 paging flag and emit flat logical results")
    args = parser.parse_args()

    native = None
    suite = None
    configured = {}
    identity = {}
    tokenization = {}
    native_config = None
    try:
        configured = configure_environment(args)
        identity = file_identity(args.lib, args.tqf)
        lib = load(args.lib, args.tqf)
        native = Native(lib, args.tqf, args.page, args.blocks)
        native_config = (
            native.assert_flat() if args.flat_regression
            else native.assert_attested())
        base_tokens = 2 * args.page + 16
        token_counts = {
            "alpha": max(base_tokens, args.long_a),
            "beta": max(base_tokens, args.long_b),
            "gamma": base_tokens,
        }
        token_sets, tokenization = load_token_sets(
            args.model_dir, token_counts, int(lib.qwn_vocab_size()))
        suite = Suite(native, token_sets, args.chunk)
        suite.run("server_apc_exact_key_host_regression",
                  suite.server_apc_exact_key_host_regression)
        if args.flat_regression:
            suite.run("flat_unset_zero_native_regression",
                      suite.flat_regression)
        elif args.long_only:
            suite.run(
                "long_pair_three_arm_true_batch",
                lambda: suite.long_pair_three_arm(
                    args.long_a, args.long_b, args.long_steps))
        else:
            suite.run("invalid_native_ranges_before_mutation",
                      suite.invalid_ranges_before_mutation)
            suite.run("reservation_exact_fit_one_short_reset_release",
                      suite.reservations_and_reset)
            suite.run("partial_growth_transactional_rollback",
                      suite.partial_growth_rollback)
            suite.run("independent_slots_interleaved_same_shape",
                      suite.independent_slots_interleaved)
            suite.run("batch_mixed_positions_join_and_n1_dispatch",
                      suite.batch_mixed_join)
            suite.run("apc_full_page_sharing_and_live_recipient",
                      suite.apc_full_pages)
            suite.run("apc_private_partial_tail_and_gdn_match",
                      suite.apc_private_partial_tail)
            suite.run("constrained_save_adopt_rollback_and_promised_credits",
                      suite.constrained_checkpoint_rollback)
            suite.run("stale_ids_reset_init_free_lifecycle",
                      suite.stale_ids_and_repeated_init)
        payload = {
            "status": "PASS",
            "tool": "paged_state_check",
            "identity": identity,
            "configured_environment": configured,
            "native_pool_config": native_config,
            "tokenizer": {
                "model_dir": str(Path(args.model_dir).expanduser().resolve()),
                "calibration": tokenization,
            },
            "proof": {
                "real_loaded_model": True,
                "real_native_c_api": True,
                "fake_allocator_or_table_mutation": False,
                "exact_logical_kv_bytes_compared": True,
                "logical_layout": "K,V,K-fp16-scales,V-fp16-scales per full-attention layer",
                "same_execution_shape_references": True,
                "at_least_two_slots": True,
                "small_physical_pool": (
                    not args.long_only and not args.flat_regression),
                "qualification_mode": (
                    "flat_regression" if args.flat_regression else
                    ("long_three_arm" if args.long_only else "lifecycle")),
            },
            "scenarios": suite.results,
        }
        print(json.dumps(payload, sort_keys=True))
        return 0
    except Exception as exc:
        scenario = exc.scenario if isinstance(exc, ScenarioError) else (
            suite.current if suite is not None else "startup")
        payload = {
            "status": "FAIL",
            "tool": "paged_state_check",
            "scenario": scenario,
            "error": f"{type(exc).__name__}: {exc}",
            "identity": identity,
            "configured_environment": configured,
            "native_pool_config": native_config,
            "tokenizer": tokenization,
            "completed_scenarios": suite.results if suite is not None else [],
        }
        print(json.dumps(payload, sort_keys=True))
        return 1
    finally:
        if native is not None:
            native.free()


if __name__ == "__main__":
    sys.exit(main())
