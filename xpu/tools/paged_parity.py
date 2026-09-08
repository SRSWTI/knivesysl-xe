#!/usr/bin/env python3
"""Attested same-build flat/paged replay and paired paged-KV matrix driver.

Direct record/check runs execute twice after whole-engine resets.  The optional
matrix launches every flat and paged arm in a fresh subprocess, sequentially, so
one process owns the model and device at a time.  Logical K/V evidence never
includes physical block ids.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import uuid

PROMPT = [151644, 872, 198] + list(range(1000, 1400)) + [151645, 198]
BRANCHES = ("prefill", "short", "sharded", "grouped", "generic", "dpas")
BOUNDARY_ROWS = (0, 1, 63, 64, 65, 127, 128, 129,
                 255, 256, 257, 511, 512, 513)
CACHE_COMPONENTS = ("K", "V", "Kfp16scales", "Vfp16scales")
ARITHMETIC_ENV = (
    "TQ_XPU_K64", "TQ_XPU_W4A4", "TQ_XPU_W4A8", "TQ_XPU_W8",
    "TQ_XPU_W4_K16", "TQ_XPU_GEMM_SPLITS", "TQ_XPU_ATTN_SIMD16",
    "TQ_XPU_ATTN_GROUPED", "TQ_XPU_ATTN_DPAS", "TQ_XPU_DELTA_FAST",
    "TQ_XPU_TP", "TQ_CTX", "TQ_XPU_SLOTS", "TQ_XPU_PREFILL_CHUNK",
    "TQ_XPU_PREFILL_XMX",
)
PAGING_ENV = ("TQ_XPU_KV_PAGE", "TQ_XPU_KV_POOL_TOKENS",
              "TQ_XPU_KV_POOL_MB")
DONOR_PREFILL_MAX = 512
MATRIX_PAGES = (128, 256)
MATRIX_CHUNKS = (8, 16, 64, 128, 512)
MATRIX_CONTEXT = 1024
MATRIX_PROMPT_TOKENS = 530
MATRIX_FINAL_CELL = ("grouped", 128, 64)
MATRIX_BRANCH_PRESETS = {
    "grouped": ({"TQ_XPU_ATTN_SIMD16": "1",
                 "TQ_XPU_ATTN_GROUPED": "1",
                 "TQ_XPU_ATTN_DPAS": "0"},
                ("prefill", "short", "sharded", "grouped")),
    "grouped-disabled": ({"TQ_XPU_ATTN_SIMD16": "1",
                          "TQ_XPU_ATTN_GROUPED": "0",
                          "TQ_XPU_ATTN_DPAS": "0"},
                         ("prefill", "short", "sharded")),
    "generic": ({"TQ_XPU_ATTN_SIMD16": "0",
                 "TQ_XPU_ATTN_GROUPED": "0",
                 "TQ_XPU_ATTN_DPAS": "0"},
                ("prefill", "generic")),
    "dpas": ({"TQ_XPU_ATTN_SIMD16": "1",
              "TQ_XPU_ATTN_GROUPED": "1",
              "TQ_XPU_ATTN_DPAS": "1"},
             ("prefill", "short", "sharded", "dpas")),
}
ATTESTATION_KEY = "_attestation"


def checked(rc, operation):
    if rc < 0:
        raise RuntimeError(f"{operation} failed rc={rc}")
    return rc


def load(libpath, tqf):
    """Bind the required native ABI and initialize it; import remains inert."""
    library_path = Path(libpath).expanduser().resolve()
    lib = ctypes.CDLL(str(library_path))
    for name, args, result in (
        ("qwn_init", [ctypes.c_char_p], ctypes.c_int),
        ("qwn_free", [], None),
        ("qwn_reset_state", [], ctypes.c_int),
        ("qwn_num_slots", [], ctypes.c_int),
        ("qwn_set_slot", [ctypes.c_int], ctypes.c_int),
        ("qwn_decode", [ctypes.c_int, ctypes.c_int], ctypes.c_int),
        ("qwn_prefill_chunk", [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                               ctypes.c_int], ctypes.c_int),
        ("qwn_paged_enabled", [], ctypes.c_int),
        ("qwn_paged_release", [ctypes.c_int], ctypes.c_int),
        ("qwn_paged_stats", [ctypes.POINTER(ctypes.c_longlong), ctypes.c_int],
                            ctypes.c_int),
        ("qwn_max_seq", [], ctypes.c_int),
        ("qwn_last_argmax_logit", [], ctypes.c_float),
        ("qwn_attn_branch_counts", [ctypes.POINTER(ctypes.c_ulonglong),
                                    ctypes.c_int], ctypes.c_int),
        ("qwn_attn_branch_counts_reset", [], None),
        ("qwn_num_layers", [], ctypes.c_int),
        ("qwn_layer_type", [ctypes.c_int], ctypes.c_int),
        ("qwn_num_key_value_heads", [], ctypes.c_int),
        ("qwn_head_dim", [], ctypes.c_int),
        ("qwn_debug_kv_bytes", [ctypes.c_int], ctypes.c_size_t),
        ("qwn_debug_kv_copy", [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
        ("qwn_debug_block_table", [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                   ctypes.c_int], ctypes.c_int),
    ):
        fn = getattr(lib, name)  # A missing attestation API is a hard failure.
        fn.argtypes, fn.restype = args, result
    checked(lib.qwn_init(str(Path(tqf).expanduser()).encode()), "qwn_init")
    return lib


def _positive_env(name, default=None):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _configured_page():
    page = _positive_env("TQ_XPU_KV_PAGE", 128)
    if page not in MATRIX_PAGES:
        raise RuntimeError(f"TQ_XPU_KV_PAGE must be 128 or 256, got {page}")
    return page


def _repeat_tokens(tokens, count):
    if not tokens:
        raise RuntimeError("cannot repeat an empty token sequence")
    return (tokens * ((count + len(tokens) - 1) // len(tokens)))[:count]


def _paged_stats(lib):
    raw = (ctypes.c_longlong * 8)()
    checked(lib.qwn_paged_stats(raw, len(raw)), "qwn_paged_stats")
    values = list(raw)
    return dict(zip(("page_tokens", "total_blocks", "free_blocks",
                     "active_blocks", "checkpoint_blocks", "reserved_blocks",
                     "native_max_seq", "slots"), values))


def _block_table(lib, slot, page):
    count = (lib.qwn_max_seq() + page - 1) // page
    table = (ctypes.c_int * count)()
    mapped = checked(lib.qwn_debug_block_table(slot, table, count),
                     f"qwn_debug_block_table slot {slot}")
    return list(table[:mapped])


def _expect_stat(stats, key, expected, stage):
    actual = stats[key]
    if actual != expected:
        raise RuntimeError(
            f"fragmentation {stage}: pool {key}={actual}, expected {expected}; "
            f"stats={stats}")


def _fill_donor(lib, tokens, donor_tokens, donor_chunk):
    donor = _repeat_tokens(tokens, donor_tokens)
    chunks = []
    pos = 0
    while pos < donor_tokens:
        count = min(donor_chunk, donor_tokens - pos)
        if count < 8 or count % 8:
            raise RuntimeError(
                f"donor remainder {count} is not a positive RC8 prefill chunk")
        buf = (ctypes.c_int * count)(*donor[pos:pos + count])
        checked(lib.qwn_prefill_chunk(buf, count, pos),
                f"donor prefill at {pos}")
        chunks.append({"position": pos, "tokens": count})
        pos += count
    return chunks


def _scalar_ranges(positions):
    if not positions:
        return []
    ranges = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            ranges.append([start, previous])
            start = position
        previous = position
    ranges.append([start, previous])
    return ranges


def _hash_parts(parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _logical_cache_evidence(lib, slot, committed):
    byte_count = lib.qwn_debug_kv_bytes(committed)
    if not byte_count:
        raise RuntimeError("native cache byte count is invalid")
    nkv = lib.qwn_num_key_value_heads()
    head_dim = lib.qwn_head_dim()
    if nkv <= 0 or head_dim <= 0:
        raise RuntimeError(f"invalid native cache shape nkv={nkv}, head_dim={head_dim}")
    data_per_token = nkv * head_dim
    scale_per_token = nkv * ctypes.sizeof(ctypes.c_uint16)
    data_bytes = committed * data_per_token
    scale_bytes = committed * scale_per_token
    expected_bytes = 2 * data_bytes + 2 * scale_bytes
    if byte_count != expected_bytes:
        raise RuntimeError(
            f"native cache bytes={byte_count}, layout arithmetic expects "
            f"{expected_bytes} for committed={committed}, nkv={nkv}, "
            f"head_dim={head_dim}")

    rows = sorted(set(row for row in BOUNDARY_ROWS if row < committed) |
                  {committed - 1})
    snapshots = {
        str(row): {"committed": row + 1,
                   "bytes_per_layer": (row + 1) *
                                      (2 * data_per_token + 2 * scale_per_token),
                   "layers": {}}
        for row in rows
    }
    cache = ctypes.create_string_buffer(byte_count)
    raw = memoryview(cache).cast("B")
    final_hashes = {}
    component_hashes = {}
    full_attention_layers = []
    for layer in range(lib.qwn_num_layers()):
        if lib.qwn_layer_type(layer) != 2:
            continue
        checked(lib.qwn_debug_kv_copy(slot, layer, committed, cache, byte_count),
                f"logical KV copy layer {layer}")
        parts = {
            "K": raw[0:data_bytes],
            "V": raw[data_bytes:2 * data_bytes],
            "Kfp16scales": raw[2 * data_bytes:2 * data_bytes + scale_bytes],
            "Vfp16scales": raw[2 * data_bytes + scale_bytes:byte_count],
        }
        layer_key = str(layer)
        full_attention_layers.append(layer)
        final_hashes[layer_key] = hashlib.sha256(raw).hexdigest()
        component_hashes[layer_key] = {
            name: hashlib.sha256(parts[name]).hexdigest()
            for name in CACHE_COMPONENTS
        }
        for row in rows:
            tokens = row + 1
            prefixes = (
                parts["K"][:tokens * data_per_token],
                parts["V"][:tokens * data_per_token],
                parts["Kfp16scales"][:tokens * scale_per_token],
                parts["Vfp16scales"][:tokens * scale_per_token],
            )
            snapshots[str(row)]["layers"][layer_key] = {
                "combined": _hash_parts(prefixes),
                **{name: hashlib.sha256(prefix).hexdigest()
                   for name, prefix in zip(CACHE_COMPONENTS, prefixes)},
            }
        if snapshots[str(committed - 1)]["layers"][layer_key]["combined"] != \
                final_hashes[layer_key]:
            raise RuntimeError(
                f"cache snapshot layout disagrees with final payload at layer {layer}")
    if not final_hashes:
        raise RuntimeError("no full-attention cache was inspected")
    return {
        "logical_cache_sha256": final_hashes,
        "logical_cache_component_sha256": component_hashes,
        "cache_bytes_per_layer": byte_count,
        "cache_component_bytes_per_token": {
            "K": data_per_token, "V": data_per_token,
            "Kfp16scales": scale_per_token,
            "Vfp16scales": scale_per_token,
        },
        "cache_snapshot_rows": rows,
        "cache_snapshots": snapshots,
        "boundary_rows_not_reached": [row for row in BOUNDARY_ROWS
                                      if row >= committed],
        "full_attention_layers": full_attention_layers,
    }


def _native_limit_evidence(lib, paged, committed, exercise_final_position):
    configured = _positive_env("TQ_CTX")
    native_max = lib.qwn_max_seq()
    if paged:
        semantics = "paged_exact_tq_ctx"
    elif configured is not None and configured < 2048 and native_max == 2048:
        semantics = "flat_historical_min_2048"
    else:
        semantics = "flat_historical_clamp"
    evidence = {
        "configured_tq_ctx": configured,
        "native_max_seq": native_max,
        "native_final_position": native_max - 1,
        "target_final_position": committed - 1,
        "target_reaches_native_final": committed == native_max,
        "semantics": semantics,
    }
    if exercise_final_position:
        expected_native = 1024 if paged else 2048
        if configured != MATRIX_CONTEXT or committed != MATRIX_CONTEXT or \
                native_max != expected_native:
            raise RuntimeError(
                "final-position qualification requires TQ_CTX=1024, exactly "
                "1024 committed target tokens, native paged max_seq=1024, and "
                f"native flat historical max_seq=2048; observed configured="
                f"{configured}, committed={committed}, native_max_seq={native_max}, "
                f"layout={'paged' if paged else 'flat'}")
        evidence.update({
            "qualification": "paired_context_1024_final_position_1023",
            "pair_final_legal_position": 1023,
            "expected_native_max_seq": expected_native,
            "final_legal_position_exercised": True,
        })
    else:
        evidence["final_legal_position_exercised"] = False
    return evidence


def _validate_fragment_table(table, total_blocks, donor_table, first_block):
    if len(table) != total_blocks:
        raise RuntimeError(
            f"fragmented target mapped {len(table)} blocks, expected all "
            f"{total_blocks}: table={table}")
    if len(set(table)) != len(table):
        raise RuntimeError(f"fragmented target table contains duplicate ids: {table}")
    if set(table) != set(range(total_blocks)):
        raise RuntimeError(
            f"fragmented target did not reuse the real pool exactly: table={table}, "
            f"pool_blocks={total_blocks}")
    if table[0] != first_block or first_block in donor_table:
        raise RuntimeError(
            f"target first block/donor ownership is inconsistent: first={first_block}, "
            f"donor={donor_table}, final={table}")
    nonidentity = any(logical != physical
                      for logical, physical in enumerate(table))
    descending = [[index - 1, table[index - 1], table[index]]
                  for index in range(1, len(table))
                  if table[index] < table[index - 1]]
    if not nonidentity or not descending:
        raise RuntimeError(
            "real donor-release history did not produce a nonidentity, "
            f"nonmonotonic target table: {table}")
    return nonidentity, descending


def chain(lib, prompt, steps, chunk, scalar_prefix, fragmented_map=False,
          require_boundaries=False, require_partial_prefill=False,
          require_page_crossing_prefill=False, exercise_final_position=False,
          donor_chunk=DONOR_PREFILL_MAX, page_size=None):
    """Run one target chain; optional fragmentation uses only native ownership APIs."""
    if steps < 1 or not prompt or chunk < 0 or chunk % 8:
        raise ValueError("chain requires prompt/steps and an RC8-compatible chunk")
    if not 0 <= scalar_prefix < len(prompt):
        raise ValueError("scalar_prefix must be within prompt")
    if donor_chunk < 8 or donor_chunk > DONOR_PREFILL_MAX or donor_chunk % 8:
        raise ValueError(
            f"donor_chunk must be an RC8 multiple in [8,{DONOR_PREFILL_MAX}]")
    checked(lib.qwn_reset_state(), "qwn_reset_state")
    paged = bool(lib.qwn_paged_enabled())
    native_max = lib.qwn_max_seq()
    committed_target = len(prompt) + steps
    if committed_target > native_max:
        raise RuntimeError(
            f"target commits {committed_target} tokens but native max_seq={native_max}")

    page = page_size if page_size is not None else _configured_page()
    effective_scalar_prefix = scalar_prefix
    fragmentation = {
        "requested": bool(fragmented_map),
        "layout": "paged" if paged else "flat",
        "page_tokens": page,
    }
    donor_release_pending = False
    if fragmented_map:
        if page not in MATRIX_PAGES:
            raise RuntimeError(f"fragmented-map page must be 128 or 256, got {page}")
        if len(prompt) <= page:
            raise RuntimeError(
                f"fragmented-map requires a prompt longer than its first {page}-token page")
        effective_scalar_prefix = max(effective_scalar_prefix, page)
        if paged:
            initial = _paged_stats(lib)
            total_blocks = initial["total_blocks"]
            target_blocks = (committed_target + page - 1) // page
            donor_blocks = total_blocks - 1
            donor_tokens = donor_blocks * page
            if initial["page_tokens"] != page:
                raise RuntimeError(
                    f"native page={initial['page_tokens']} but paired config page={page}")
            if lib.qwn_num_slots() < 2 or initial["slots"] < 2:
                raise RuntimeError("fragmented-map requires at least two native slots")
            if total_blocks < 3:
                raise RuntimeError(
                    "fragmented-map reset/replay requires at least three physical blocks")
            if target_blocks != total_blocks:
                raise RuntimeError(
                    "fragmented-map requires the target to map every pool block so "
                    "the donor-release rotation remains nonmonotonic after reset; "
                    f"target_blocks={target_blocks}, pool_blocks={total_blocks}")
            if donor_tokens > native_max:
                raise RuntimeError(
                    f"donor logical length {donor_tokens} exceeds native max_seq="
                    f"{native_max}")
            for key, expected in (("free_blocks", total_blocks),
                                  ("active_blocks", 0),
                                  ("checkpoint_blocks", 0),
                                  ("reserved_blocks", 0)):
                _expect_stat(initial, key, expected, "after whole reset")
            checked(lib.qwn_set_slot(1), "select donor slot 1")
            donor_chunks = _fill_donor(lib, prompt, donor_tokens, donor_chunk)
            donor_table = _block_table(lib, 1, page)
            after_donor = _paged_stats(lib)
            if len(donor_table) != donor_blocks or \
                    len(set(donor_table)) != donor_blocks:
                raise RuntimeError(
                    f"donor mapped invalid real block table {donor_table}; "
                    f"expected {donor_blocks} unique blocks")
            _expect_stat(after_donor, "free_blocks", 1, "after donor fill")
            _expect_stat(after_donor, "active_blocks", donor_blocks,
                         "after donor fill")
            fragmentation.update({
                "pool_tokens": total_blocks * page,
                "pool_blocks": total_blocks,
                "donor_slot": 1,
                "donor_tokens": donor_tokens,
                "donor_prefill_bound": donor_chunk,
                "donor_prefill_chunks": donor_chunks,
                "donor_block_table": donor_table,
                "pool_history": {
                    "after_whole_reset": initial,
                    "after_donor_fill": after_donor,
                },
            })
            donor_release_pending = True
        else:
            fragmentation.update({
                "flat_skipped_donor": True,
                "physical_block_table": None,
            })

    checked(lib.qwn_set_slot(0), "select target slot 0")
    # Donor work is deliberately outside the target branch/equality payload.
    lib.qwn_attn_branch_counts_reset()
    pos, nxt = 0, -1
    scalar_positions = []
    prefill_chunks = []
    while pos < len(prompt):
        # Keep a final scalar token: chunk prefill does not run the LM head.
        count = min(chunk, ((len(prompt) - pos - 1) // 8) * 8)
        if chunk and pos >= effective_scalar_prefix and count >= 8:
            buf = (ctypes.c_int * count)(*prompt[pos:pos + count])
            checked(lib.qwn_prefill_chunk(buf, count, pos), f"prefill at {pos}")
            prefill_chunks.append({"position": pos, "tokens": count})
            pos += count
        else:
            scalar_positions.append(pos)
            nxt = checked(lib.qwn_decode(prompt[pos], pos), f"decode at {pos}")
            pos += 1
        if donor_release_pending and pos == page:
            target_first_table = _block_table(lib, 0, page)
            before_release = _paged_stats(lib)
            total_blocks = fragmentation["pool_blocks"]
            _expect_stat(before_release, "free_blocks", 0,
                         "after target first scalar page")
            _expect_stat(before_release, "active_blocks", total_blocks,
                         "after target first scalar page")
            if len(target_first_table) != 1:
                raise RuntimeError(
                    f"target first page mapped {target_first_table}, expected one block")
            checked(lib.qwn_paged_release(1), "release donor slot 1")
            after_release = _paged_stats(lib)
            _expect_stat(after_release, "free_blocks", total_blocks - 1,
                         "after donor release")
            _expect_stat(after_release, "active_blocks", 1,
                         "after donor release")
            fragmentation["target_first_page_block_table"] = target_first_table
            fragmentation["pool_history"][
                "after_target_first_scalar_page"] = before_release
            fragmentation["pool_history"]["after_donor_release"] = after_release
            donor_release_pending = False
    if donor_release_pending:
        raise RuntimeError("target never reached the scalar first-page donor release point")
    if nxt < 0:
        raise RuntimeError("target prompt did not execute a scalar LM-head token")

    tokens, logits = [], []
    for _ in range(steps):
        tokens.append(nxt)
        logits.append(struct.pack("<f", lib.qwn_last_argmax_logit()).hex())
        nxt = checked(lib.qwn_decode(nxt, pos), f"decode at {pos}")
        pos += 1

    counts = (ctypes.c_ulonglong * len(BRANCHES))()
    checked(lib.qwn_attn_branch_counts(counts, len(counts)), "branch counts")
    schedule = {
        "requested_scalar_prefix": scalar_prefix,
        "effective_scalar_prefix": effective_scalar_prefix,
        "prompt_scalar_ranges": _scalar_ranges(scalar_positions),
        "prompt_scalar_calls": len(scalar_positions),
        "prefill_chunks": prefill_chunks,
        "prefill_chunk_limit": chunk,
        "final_prefill_chunk_is_partial_query_tile": bool(
            prefill_chunks and prefill_chunks[-1]["tokens"] % 64),
        "page_crossing_prefill_chunks": [
            item for item in prefill_chunks
            if item["position"] // page !=
               (item["position"] + item["tokens"] - 1) // page
        ],
        "generated_decode_calls": steps,
        "first_page_all_scalar": (not fragmented_map or
                                  all(position in scalar_positions
                                      for position in range(page))),
    }
    if fragmented_map and not schedule["first_page_all_scalar"]:
        raise RuntimeError("fragmented target first page was not entirely scalar")
    if require_partial_prefill and not \
            schedule["final_prefill_chunk_is_partial_query_tile"]:
        raise RuntimeError(
            f"final prefill chunk did not exercise a partial 64-row query tile: "
            f"{prefill_chunks}")
    if require_page_crossing_prefill and not \
            schedule["page_crossing_prefill_chunks"]:
        raise RuntimeError(
            f"no nonzero-start prefill chunk crossed a {page}-token page: "
            f"{prefill_chunks}")

    cache = _logical_cache_evidence(lib, 0, pos)
    if require_boundaries and cache["boundary_rows_not_reached"]:
        raise RuntimeError(
            f"required cache boundary rows were not reached: "
            f"{cache['boundary_rows_not_reached']}; committed={pos}")

    if fragmented_map and paged:
        final_table = _block_table(lib, 0, page)
        final_stats = _paged_stats(lib)
        total_blocks = fragmentation["pool_blocks"]
        _expect_stat(final_stats, "free_blocks", 0, "after target completion")
        _expect_stat(final_stats, "active_blocks", total_blocks,
                     "after target completion")
        _expect_stat(final_stats, "checkpoint_blocks", 0,
                     "after target completion")
        _expect_stat(final_stats, "reserved_blocks", 0,
                     "after target completion")
        nonidentity, descending = _validate_fragment_table(
            final_table, total_blocks, fragmentation["donor_block_table"],
            fragmentation["target_first_page_block_table"][0])
        fragmentation.update({
            "physical_block_table": final_table,
            "nonidentity": nonidentity,
            "nonmonotonic": bool(descending),
            "descending_edges": descending,
        })
        fragmentation["pool_history"]["after_target_completion"] = final_stats

    result = {
        "tokens": tokens,
        "argmax_logit_bits": logits,
        "branches": dict(zip(BRANCHES, counts)),
        "committed": pos,
        "target_schedule": schedule,
        **cache,
    }
    result[ATTESTATION_KEY] = {
        "native_limit": _native_limit_evidence(
            lib, paged, pos, exercise_final_position),
        "fragmentation": fragmentation,
    }
    return result


def identity(args, prompt):
    model = Path(args.tqf).expanduser().resolve()
    library_path = Path(args.lib).expanduser().resolve()
    stat = model.stat()
    # The large immutable model is identified by path/stat plus its header hash;
    # hash the much smaller candidate library completely to forbid build mixing.
    with model.open("rb") as fh:
        header = hashlib.sha256(fh.read(65536)).hexdigest()
    with library_path.open("rb") as fh:
        library = hashlib.file_digest(fh, "sha256").hexdigest()
    return {
        "model": {"path": str(model), "size": stat.st_size,
                  "mtime_ns": stat.st_mtime_ns, "header_sha256": header},
        "library_sha256": library,
        "arithmetic": {key: os.environ.get(key) for key in ARITHMETIC_ENV},
        "paging_config": {
            "page_tokens": _configured_page(),
            **{key: os.environ.get(key) for key in PAGING_ENV},
        },
        "prompt": prompt,
        "steps": args.steps,
        "chunk": args.chunk,
        "scalar_prefix": args.scalar_prefix,
        "fragmented_map": bool(getattr(args, "fragmented_map", False)),
        "donor_chunk": getattr(args, "donor_chunk", DONOR_PREFILL_MAX),
    }


def _comparison_payload(result):
    return {key: value for key, value in result.items() if key != ATTESTATION_KEY}


def _short(value):
    text = repr(value)
    return text if len(text) <= 180 else text[:177] + "..."


def _first_difference(expected, actual, path="result"):
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys, actual_keys = set(expected), set(actual)
        if expected_keys != actual_keys:
            return (f"{path} keys differ: missing={sorted(expected_keys - actual_keys)}, "
                    f"unexpected={sorted(actual_keys - expected_keys)}")
        for key in expected:
            difference = _first_difference(expected[key], actual[key],
                                           f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path} length differs: flat={len(expected)}, current={len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            if left != right:
                difference = _first_difference(left, right, f"{path}[{index}]")
                return difference or (
                    f"{path}[{index}] differs: flat={_short(left)}, "
                    f"current={_short(right)}")
        return None
    if expected != actual:
        return f"{path} differs: flat={_short(expected)}, current={_short(actual)}"
    return None


def _assert_equal(expected, actual, label):
    difference = _first_difference(expected, actual)
    if difference:
        raise RuntimeError(f"{label}: {difference}")


def _parse_child_json(completed, command):
    if completed.returncode:
        raise RuntimeError(
            f"matrix child failed rc={completed.returncode}: {' '.join(command)}\n"
            f"stdout tail:\n{completed.stdout[-4000:]}\n"
            f"stderr tail:\n{completed.stderr[-8000:]}")
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("status") == "PASS":
            return payload
    raise RuntimeError(
        f"matrix child emitted no PASS JSON: {' '.join(command)}\n"
        f"stdout tail:\n{completed.stdout[-4000:]}\n"
        f"stderr tail:\n{completed.stderr[-4000:]}")


def _run_matrix_child(command, env, timeout):
    completed = subprocess.run(command, capture_output=True, text=True,
                               env=env, timeout=timeout)
    return _parse_child_json(completed, command)


def _matrix(args):
    if not args.matrix_ref_dir:
        raise RuntimeError("--matrix requires --matrix-ref-dir")
    if args.matrix_tokens <= max(MATRIX_PAGES) + 1:
        raise RuntimeError(
            f"--matrix-tokens must exceed {max(MATRIX_PAGES) + 1} so chunked "
            "prefill follows the scalar first page")
    if args.matrix_tokens + args.steps > MATRIX_CONTEXT:
        raise RuntimeError(
            f"matrix prompt {args.matrix_tokens} + steps {args.steps} exceeds "
            f"paired context {MATRIX_CONTEXT}")
    if args.matrix_tokens + args.steps <= max(BOUNDARY_ROWS):
        raise RuntimeError("matrix base cells do not reach all required boundary rows")

    ref_dir = Path(args.matrix_ref_dir).expanduser().resolve()
    ref_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    script = str(Path(__file__).resolve())
    python = sys.executable
    model = str(Path(args.tqf).expanduser().resolve())
    library = str(Path(args.lib).expanduser().resolve())
    cells = []
    selected = {branch: False for branch in BRANCHES}

    for preset, (branch_env, required_branches) in MATRIX_BRANCH_PRESETS.items():
        for page in MATRIX_PAGES:
            for chunk in MATRIX_CHUNKS:
                final_cell = (preset, page, chunk) == MATRIX_FINAL_CELL
                steps = MATRIX_CONTEXT - args.matrix_tokens if final_cell else args.steps
                committed = args.matrix_tokens + steps
                pool_tokens = ((committed + page - 1) // page) * page
                scalar_prefix = page + 1
                ref = ref_dir / (
                    f"paged-parity-{run_id}-{preset}-p{page}-c{chunk}-"
                    f"n{committed}.json")
                common = [
                    python, script,
                    "--lib", library,
                    "--tqf", model,
                    "--ref", str(ref),
                    "--tokens", str(args.matrix_tokens),
                    "--steps", str(steps),
                    "--chunk", str(chunk),
                    "--scalar-prefix", str(scalar_prefix),
                    "--donor-chunk", str(args.donor_chunk),
                    "--fragmented-map",
                    "--require-boundaries",
                    "--require-partial-prefill",
                    "--require-page-crossing-prefill",
                ]
                for branch in required_branches:
                    common.extend(("--require-branch", branch))
                if final_cell:
                    common.append("--exercise-final-position")

                env = os.environ.copy()
                env.update(branch_env)
                env.update({
                    "TQ_XPU_SLOTS": "2",
                    "TQ_CTX": str(MATRIX_CONTEXT),
                    "TQ_XPU_TP": "1",
                    "TQ_XPU_KV_PAGE": str(page),
                    "TQ_XPU_KV_POOL_TOKENS": str(pool_tokens),
                    "TQ_XPU_PREFILL_CHUNK": str(chunk),
                })
                env.pop("TQ_XPU_KV_POOL_MB", None)

                record_command = common + ["--record"]
                record_env = env.copy()
                record_env["TQ_XPU_PAGED"] = "0"
                flat = _run_matrix_child(record_command, record_env,
                                         args.matrix_timeout)

                check_command = common + ["--check"]
                check_env = env.copy()
                check_env["TQ_XPU_PAGED"] = "1"
                paged = _run_matrix_child(check_command, check_env,
                                          args.matrix_timeout)
                for branch, count in paged["branches"].items():
                    selected[branch] = selected[branch] or count > 0
                cell = {
                    "preset": preset,
                    "page": page,
                    "chunk": chunk,
                    "scalar_prefix": scalar_prefix,
                    "steps": steps,
                    "committed": committed,
                    "pool_tokens": pool_tokens,
                    "final_legal_context_cell": final_cell,
                    "required_branches": list(required_branches),
                    "ref": str(ref),
                    "record_command": record_command,
                    "check_command": check_command,
                    "flat": flat,
                    "paged": paged,
                }
                cells.append(cell)
                print(json.dumps({
                    "status": "CELL_PASS",
                    "preset": preset,
                    "page": page,
                    "chunk": chunk,
                    "committed": committed,
                    "ref": str(ref),
                }), flush=True)

    missing = [branch for branch, entered in selected.items() if not entered]
    if missing:
        raise RuntimeError(f"paired matrix never selected branches: {missing}")
    summary = {
        "status": "PASS",
        "kind": "paged_parity_matrix",
        "run_id": run_id,
        "fresh_subprocess_per_arm": True,
        "one_process_at_a_time": True,
        "reset_replay_per_arm": True,
        "exact_argmax_logits": True,
        "exact_logical_cache_bytes": True,
        "physical_ids_excluded_from_equality": True,
        "branch_selected": selected,
        "pages": list(MATRIX_PAGES),
        "chunks": list(MATRIX_CHUNKS),
        "final_cell": {
            "preset": MATRIX_FINAL_CELL[0],
            "page": MATRIX_FINAL_CELL[1],
            "chunk": MATRIX_FINAL_CELL[2],
            "configured_context": MATRIX_CONTEXT,
            "final_position": MATRIX_CONTEXT - 1,
            "flat_native_limit": 2048,
            "paged_native_limit": 1024,
        },
        "cells": cells,
    }
    summary_path = ref_dir / f"paged-parity-matrix-{run_id}.json"
    with summary_path.open("x") as fh:
        json.dump(summary, fh, indent=2)
    summary["summary_file"] = str(summary_path)
    print(json.dumps(summary))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", default="xpu/build/libforward_qwen_xpu_paged_candidate.so")
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--ref", help="unique per-configuration flat reference file")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=len(PROMPT))
    ap.add_argument("--chunk", type=int, default=0,
                    help="0 scalar, otherwise maximum RC8 prefill chunk")
    ap.add_argument("--scalar-prefix", type=int, default=0)
    ap.add_argument("--donor-chunk", type=int, default=DONOR_PREFILL_MAX,
                    help=f"real donor prefill bound, RC8 and <= {DONOR_PREFILL_MAX}")
    ap.add_argument("--fragmented-map", action="store_true",
                    help="force real donor/release nonmonotonic paged mapping")
    ap.add_argument("--require-boundaries", action="store_true")
    ap.add_argument("--require-partial-prefill", action="store_true")
    ap.add_argument("--require-page-crossing-prefill", action="store_true")
    ap.add_argument("--exercise-final-position", action="store_true")
    ap.add_argument("--require-branch", action="append", choices=BRANCHES, default=[])
    ap.add_argument("--matrix-ref-dir",
                    help="existing references are preserved; every matrix run uses a UUID")
    ap.add_argument("--matrix-timeout", type=int, default=7200,
                    help="seconds allowed for each fresh subprocess arm")
    ap.add_argument("--matrix-tokens", type=int, default=MATRIX_PROMPT_TOKENS)
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--record", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--matrix", action="store_true")
    args = ap.parse_args()

    if args.steps < 1 or args.tokens < 1 or args.chunk < 0 or args.chunk % 8:
        ap.error("positive steps/tokens and nonnegative multiple-of-8 chunk required")
    if args.donor_chunk < 8 or args.donor_chunk > DONOR_PREFILL_MAX or \
            args.donor_chunk % 8:
        ap.error(f"donor-chunk must be a multiple of 8 in [8,{DONOR_PREFILL_MAX}]")
    if args.matrix_timeout < 1 or args.matrix_tokens < 1:
        ap.error("matrix-timeout and matrix-tokens must be positive")
    if args.matrix:
        if args.ref:
            ap.error("--matrix creates unique references; do not pass --ref")
        return _matrix(args)
    if not args.ref:
        ap.error("--record and --check require --ref")
    if not 0 <= args.scalar_prefix < args.tokens:
        ap.error("scalar-prefix must be within prompt")
    if (args.require_partial_prefill or args.require_page_crossing_prefill) and \
            not args.chunk:
        ap.error("prefill schedule requirements need a nonzero --chunk")

    prompt = _repeat_tokens(PROMPT, args.tokens)
    config = identity(args, prompt)
    reference = None
    if args.check:
        with open(args.ref) as fh:
            reference = json.load(fh)
        if reference.get("schema_version") != 2 or \
                reference.get("reference_layout") != "flat":
            raise RuntimeError(
                "reference is not a schema-2 flat paged_parity oracle; create a "
                "new unique reference rather than overwriting the existing file")
        difference = _first_difference(reference.get("config"), config, "config")
        if difference:
            raise RuntimeError(f"reference build/model/prompt configuration mismatch: {difference}")

    lib = load(args.lib, args.tqf)
    try:
        mode = lib.qwn_paged_enabled()
        if mode != int(args.check):
            raise RuntimeError(f"effective paging={mode}; expected {int(args.check)}")
        if len(prompt) + args.steps > lib.qwn_max_seq():
            raise RuntimeError(
                f"prompt + executed continuation={len(prompt) + args.steps} "
                f"exceeds native max_seq={lib.qwn_max_seq()}")
        capacity = _paged_stats(lib) if mode else None
        if capacity and capacity["page_tokens"] != config["paging_config"]["page_tokens"]:
            raise RuntimeError(
                f"native page={capacity['page_tokens']} differs from matched "
                f"configuration page={config['paging_config']['page_tokens']}")

        chain_kwargs = {
            "fragmented_map": args.fragmented_map,
            "require_boundaries": args.require_boundaries,
            "require_partial_prefill": args.require_partial_prefill,
            "require_page_crossing_prefill": args.require_page_crossing_prefill,
            "exercise_final_position": args.exercise_final_position,
            "donor_chunk": args.donor_chunk,
            "page_size": config["paging_config"]["page_tokens"],
        }
        first = chain(lib, prompt, args.steps, args.chunk, args.scalar_prefix,
                      **chain_kwargs)
        second = chain(lib, prompt, args.steps, args.chunk, args.scalar_prefix,
                       **chain_kwargs)
        first_payload = _comparison_payload(first)
        second_payload = _comparison_payload(second)
        _assert_equal(first_payload, second_payload,
                      "same-process whole-reset replay mismatch")
        for branch in args.require_branch:
            if not first_payload["branches"][branch]:
                raise RuntimeError(f"requested branch never selected: {branch}")

        if args.record:
            # Never clobber an earlier oracle by accident.
            with open(args.ref, "x") as fh:
                json.dump({
                    "schema_version": 2,
                    "reference_layout": "flat",
                    "config": config,
                    "result": first_payload,
                    "attestation": first[ATTESTATION_KEY],
                }, fh, indent=2)
        else:
            _assert_equal(reference["result"], first_payload,
                          "flat/paged exact equality mismatch")

        native_limit = first[ATTESTATION_KEY]["native_limit"]
        replay_mapping = second[ATTESTATION_KEY]["fragmentation"]
        output = {
            "status": "PASS",
            "mode": "paged" if mode else "flat",
            "tokens": args.steps,
            "committed": first_payload["committed"],
            "exact_argmax_logits": True,
            "exact_logical_cache_bytes": True,
            "cache_components": list(CACHE_COMPONENTS),
            "cache_snapshot_rows": first_payload["cache_snapshot_rows"],
            "reset_replay": True,
            "branches": first_payload["branches"],
            "target_schedule": first_payload["target_schedule"],
            "native_limit": native_limit,
            "initial_pool_stats": capacity,
            "physical_mapping": first[ATTESTATION_KEY]["fragmentation"],
            "reset_replay_physical_mapping": replay_mapping,
            "physical_ids_excluded_from_equality": True,
            "logical_cache_sha256": first_payload["logical_cache_sha256"],
            "logical_cache_component_sha256":
                first_payload["logical_cache_component_sha256"],
            "ref": args.ref,
        }
        if reference is not None:
            output["flat_reference_native_limit"] = reference["attestation"][
                "native_limit"]
        print(json.dumps(output))
        return 0
    finally:
        lib.qwn_free()


if __name__ == "__main__":
    sys.exit(main())
