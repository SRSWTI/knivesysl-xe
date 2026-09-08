#!/usr/bin/env python3
"""HTTP compatibility smoke and exact-token mixed-load qualification for XE.

The default invocation preserves the original endpoint/concurrency/APC smoke.
Supplying ``--mixed`` adds the retained, exact-token plan-8.3 measurement.  The
client only uses public HTTP APIs: native state and timings are consumed from
``/health`` and ``x_knivesysl`` response metadata.

Examples:

  xpu/tools/serve_smoke_xpu.py --base-url http://127.0.0.1:8100/v1 \
      --model knivesysl-xe-qwen3.8-27b-w4a8 --conc 8

  xpu/tools/serve_smoke_xpu.py --base-url http://127.0.0.1:8101/v1 \
      --model knivesysl-xe-qwen3.8-27b-w4a8 \
      --expect-layout paged --mixed 28000,75000 \
      --model-dir ~/models/knivesysl --gen128 --timeout 14400 \
      --repeats 2 --output result.json
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import http.client
import json
import math
import os
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


FILLER = ("You are a meticulous systems engineer. Answer briefly. "
          "Context notes follow and are irrelevant to the question. ") * 24
MARKUP = ("<think>", "</think>", "<tool_call>", "</tool_call>",
          "<function=", "<parameter=")
HEALTH_PLAN_FIELDS = (
    "status", "engine_error", "slots", "active", "queued", "kv_layout",
    "paged", "kv_page_tokens", "kv_pool_tokens", "kv_pool_bytes",
    "kv_pool_blocks_total", "kv_pool_blocks_used", "kv_pool_blocks_free",
    "kv_active_blocks", "kv_checkpoint_blocks",
    "kv_remaining_reserved_blocks", "kv_sequence_limit",
    "kv_pool_blocks_peak_used", "kv_peak_remaining_reserved_blocks",
    "capacity_waits", "capacity_waiting", "apc_enabled", "apc_entries",
    "apc_hits", "apc_misses", "cancelled_requests", "request_errors",
    "last_request_error", "attention_config", "attention_branch_counts",
    "tier", "k64",
    "decode_width_hist", "decode_tok_s_by_width", "decode_tok_s",
)
HEALTH_SAMPLE_FIELDS = (
    "active", "queued", "capacity_waiting", "kv_pool_blocks_used",
    "kv_pool_blocks_free", "kv_active_blocks", "kv_checkpoint_blocks",
    "kv_remaining_reserved_blocks", "kv_pool_blocks_peak_used",
    "kv_peak_remaining_reserved_blocks", "capacity_waits", "apc_entries",
    "apc_hits", "apc_misses", "cancelled_requests", "request_errors",
)
RESPONSE_PLAN_FIELDS = (
    "engine", "slot", "generated_tokens", "queue_ms", "admission_ms",
    "ttft_ms", "itl_ms_p50", "itl_ms_p99", "decode_width_hist",
    "batch_avg", "reused_tokens", "prefilled_tokens", "kv_layout",
    "attention_config", "attention_branch_counts", "tier", "apc_enabled",
    "k64", "greedy",
)
ATTENTION_CONFIG_KEYS = ("simd16", "grouped", "dpas", "prefill_xmx")
ATTENTION_BRANCH_KEYS = (
    "prefill", "simd16_short", "simd16_sharded", "simd16_grouped",
    "generic", "simd16_dpas", "prefill_xmx", "prefill_scalar",
)




class QualificationError(RuntimeError):
    pass
class ObservedQualificationError(QualificationError):
    def __init__(self, message, detail):
        super().__init__(message)
        self.detail = detail




def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def origin(base):
    return base.rsplit("/v1", 1)[0]


def api_url(base, path):
    if path.startswith("/v1/") or path in ("/v1", "/health"):
        return origin(base) + path
    return base + path


def get_json(base, path, timeout):
    req = urllib.request.Request(api_url(base, path),
                                 headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def post_json_status(base, path, body, timeout):
    req = urllib.request.Request(
        api_url(base, path), data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read())
            return response.status, payload, time.perf_counter() - started
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"error": {"message": raw.decode(errors="replace")}}
        return error.code, payload, time.perf_counter() - started


def post(base, path, body, timeout):
    status, payload, elapsed = post_json_status(base, path, body, timeout)
    if status < 200 or status >= 300:
        message = ((payload.get("error") or {}).get("message")
                   if isinstance(payload, dict) else None)
        raise QualificationError(
            f"POST {path} returned HTTP {status}: {message or payload}")
    return payload, elapsed


def no_markup(*values):
    visible = "\n".join(value for value in values if isinstance(value, str))
    leaked = [marker for marker in MARKUP if marker in visible]
    if leaked:
        raise QualificationError(f"raw model markup leaked: {leaked}")


def parse_sse(base, path, body, timeout):
    req = urllib.request.Request(
        api_url(base, path), data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"})
    started = time.perf_counter()
    content = []
    reasoning = []
    completion = []
    tool_calls = []
    events = 0
    finish = None
    tail = None
    done = False
    first_event = None
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" not in content_type:
            raise QualificationError(
                f"stream returned content type {content_type!r}")
        for raw in response:
            line = raw.decode("utf-8", errors="strict").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                raise QualificationError(f"invalid SSE line: {line[:120]!r}")
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            events += 1
            if first_event is None:
                first_event = time.perf_counter()
            choices = event.get("choices") or []
            if not choices:
                if event.get("usage") is not None:
                    tail = event
                continue
            choice = choices[0]
            finish = choice.get("finish_reason") or finish
            if "delta" in choice:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])
            elif choice.get("text"):
                completion.append(choice["text"])
    elapsed = time.perf_counter() - started
    if not done:
        raise QualificationError("SSE stream ended without data: [DONE]")
    if tail is None:
        raise QualificationError("SSE stream omitted usage/metadata tail")
    return {
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "text": "".join(completion),
        "tool_calls": tool_calls,
        "finish_reason": finish,
        "usage": tail.get("usage") or {},
        "x_knivesysl": tail.get("x_knivesysl") or {},
        "events": events,
        "ttfe_ms_observed": ((first_event - started) * 1000.0
                              if first_event is not None else None),
        "elapsed_seconds": elapsed,
    }


def effective_metadata_issues(metadata):
    issues = []
    config = metadata.get("attention_config")
    if not isinstance(config, dict):
        issues.append("attention_config is not an object")
    elif set(config) != set(ATTENTION_CONFIG_KEYS):
        issues.append(
            "attention_config keys are not exactly " +
            ",".join(ATTENTION_CONFIG_KEYS))
    else:
        if any(type(config[key]) is not bool for key in ("simd16", "grouped", "dpas")):
            issues.append("decode attention_config values are not booleans")
        if config["prefill_xmx"] not in ("auto", "0", "1"):
            issues.append("attention_config prefill_xmx is not auto, 0, or 1")
    counts = metadata.get("attention_branch_counts")
    if not isinstance(counts, dict):
        issues.append("attention_branch_counts is not an object")
    elif set(counts) != set(ATTENTION_BRANCH_KEYS):
        issues.append(
            "attention_branch_counts keys are not exactly " +
            ",".join(ATTENTION_BRANCH_KEYS))
    elif any(type(counts[key]) is not int or counts[key] < 0
             for key in ATTENTION_BRANCH_KEYS):
        issues.append("attention_branch_counts values are not nonnegative integers")
    if not isinstance(metadata.get("tier"), str):
        issues.append("tier is not a string")
    return issues


def response_metadata(payload):
    metadata = payload.get("x_knivesysl") or {}
    issues = [f"missing {key}" for key in RESPONSE_PLAN_FIELDS
              if key not in metadata]
    issues.extend(effective_metadata_issues(metadata))
    return metadata, issues


def response_visible_text(payload):
    choice = (payload.get("choices") or [{}])[0]
    if "message" in choice:
        message = choice.get("message") or {}
        return (message.get("content") or "",
                message.get("reasoning_content") or "")
    return (choice.get("text") or "", "")


def validate_tool_calls(calls):
    if not calls:
        raise QualificationError("model emitted no parsed tool call")
    for call in calls:
        function = call.get("function") or {}
        if call.get("type") != "function" or not function.get("name"):
            raise QualificationError(f"invalid OpenAI tool call: {call}")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise QualificationError("tool-call arguments are not a JSON string")
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise QualificationError("tool-call arguments are not a JSON object")


def health_field_dependencies():
    return {
        "health": {
            "endpoint": "/health",
            "fields": list(HEALTH_PLAN_FIELDS),
            "semantics": {
                "kv_pool_blocks_used/free": "current native physical pool occupancy",
                "kv_pool_blocks_peak_used": "server-lifetime native occupancy peak",
                "kv_active_blocks": "unique physical blocks referenced by active slots",
                "kv_checkpoint_blocks": "unique physical blocks referenced by checkpoints",
                "block_count_overlap": "active and checkpoint categories overlap and are never summed",
                "kv_remaining_reserved_blocks": "current unconsumed reservation credits",
                "kv_peak_remaining_reserved_blocks": "server-lifetime reservation peak",
                "active/queued/capacity_waiting": "current scheduler state",
                "decode_width_hist": "server-lifetime native decode-step widths",
                "decode_tok_s_by_width/decode_tok_s": "server native decode timing",
                "attention_config/tier/kv_layout/apc_enabled": "effective server configuration",
                "attention_branch_counts": "server-lifetime native attention branches",
            },
        },
        "response": {
            "location": "x_knivesysl on nonstream response or final SSE tail",
            "fields": list(RESPONSE_PLAN_FIELDS),
            "usage": ["prompt_tokens", "completion_tokens", "total_tokens"],
        },
        "client_observed": {
            "fields": [
                "HTTP wall time", "simultaneous launch offsets",
                "per-stream post-TTFT throughput",
                "aggregate post-TTFT decode throughput",
                "aggregate end-to-end throughput", "health polling peaks",
                "isolated n1 cold/warm exact APC replay",
                "completion-reservation footprint versus unshared page bound",
                "concurrent cross-path output/hash observation",
            ],
            "note": "Observed values are labelled; absent server fields remain null, never synthesized.",
        },
    }


class ExactPromptFactory:
    """Creates and verifies calibrated exact-token prompts with the real tokenizer."""

    WORDS = {"alpha": " alpha", "beta": " beta", "gamma": " gamma"}

    def __init__(self, model_dir):
        from transformers import AutoTokenizer
        self.model_dir = os.path.expanduser(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True)
        self.calibration = {}
        for word, text in self.WORDS.items():
            ids = self.tokenizer(text, add_special_tokens=False).input_ids
            if len(ids) != 1:
                raise QualificationError(
                    f"calibrated fragment {text!r} tokenized to {len(ids)}, not 1")
            self.calibration[word] = int(ids[0])

    def make(self, word, tokens, variant=0):
        if word not in self.WORDS:
            raise ValueError(word)
        if tokens < 2:
            raise QualificationError("exact prompts must contain at least two tokens")
        pieces = []
        # A small token-level variant near the beginning prevents an APC hit
        # across repeated control cells. Every resulting string is still
        # verified by the tokenizer; the construction is never treated as the
        # measurement.
        if variant:
            pieces.append(self.WORDS[word])
            alternate = "gamma" if word != "gamma" else "beta"
            bits = max(1, min(tokens - 1, variant.bit_length()))
            for bit in range(bits):
                selected = "alpha" if ((variant >> bit) & 1) else alternate
                pieces.append(self.WORDS[selected])
        remaining = tokens - len(pieces)
        pieces.append(self.WORDS[word] * remaining)
        text = "".join(pieces)
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) != tokens:
            raise QualificationError(
                f"real tokenizer measured {len(ids)} tokens, requested {tokens} "
                f"for {word} variant {variant}")
        return {
            "text": text,
            "ids": list(ids),
            "prompt_tokens": tokens,
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "tokenizer": self.model_dir,
            "calibration_word": word,
            "variant": variant,
        }


def common_prefix_tokens(left, right):
    count = 0
    for a, b in zip(left["ids"], right["ids"]):
        if a != b:
            break
        count += 1
    return count


def public_prompt(prompt, sequence):
    return {
        "sequence": sequence,
        "prompt_tokens_local": prompt["prompt_tokens"],
        "prompt_sha256": prompt["prompt_sha256"],
        "tokenizer": prompt["tokenizer"],
        "calibration_word": prompt["calibration_word"],
        "variant": prompt["variant"],
    }


class HealthMonitor:
    def __init__(self, base, timeout, interval=0.2):
        self.base = base
        self.timeout = min(timeout, 30.0)
        self.interval = interval
        self.started = time.perf_counter()
        self.samples = []
        self.transitions = []
        self.errors = []
        self.latest = None
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self._loop, name="health-monitor",
                                       daemon=True)

    @staticmethod
    def _state(sample):
        return tuple(sample.get(key) for key in (
            "active", "queued", "capacity_waiting",
            "kv_pool_blocks_used", "kv_remaining_reserved_blocks",
            "kv_checkpoint_blocks"))

    def _sample(self):
        try:
            health = get_json(self.base, "/health", self.timeout)
            entry = {"offset_seconds": round(time.perf_counter() - self.started, 3),
                     **health}
            with self.condition:
                previous = self.latest
                self.latest = entry
                self.samples.append(entry)
                if (previous is None or self._state(previous) != self._state(entry)):
                    if len(self.transitions) < 1024:
                        self.transitions.append(entry)
                self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.errors.append(str(error))
                self.condition.notify_all()

    def _loop(self):
        self._sample()
        while not self.stop_event.wait(self.interval):
            self._sample()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=self.timeout + 1.0)
        self._sample()

    def wait_for(self, predicate, timeout, description):
        deadline = time.perf_counter() + timeout
        with self.condition:
            while True:
                if self.latest is not None and predicate(self.latest):
                    return self.latest
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    latest = self.latest or {}
                    raise QualificationError(
                        f"timed out waiting for {description}; latest health "
                        f"active={latest.get('active')} queued={latest.get('queued')} "
                        f"capacity_waiting={latest.get('capacity_waiting')}")
                self.condition.wait(min(remaining, self.interval * 2))

    def summary(self):
        if not self.samples:
            return {"samples": 0, "errors": list(self.errors),
                    "start": None, "end": None, "peaks": {},
                    "mins": {}, "transitions": []}
        peaks = {}
        mins = {}
        for key in HEALTH_SAMPLE_FIELDS:
            values = [sample.get(key) for sample in self.samples
                      if isinstance(sample.get(key), (int, float))]
            if values:
                peaks[key] = max(values)
                mins[key] = min(values)
        start = self.samples[0]
        end = self.samples[-1]
        counter_deltas = {}
        for field in ("decode_width_hist", "attention_branch_counts"):
            start_counts = start.get(field) or {}
            end_counts = end.get(field) or {}
            delta_counts = {}
            for key in set(start_counts) | set(end_counts):
                delta = int(end_counts.get(key, 0)) - int(start_counts.get(key, 0))
                if delta:
                    delta_counts[str(key)] = delta
            counter_deltas[field + "_server_delta"] = dict(
                sorted(delta_counts.items()))
        return {
            "samples": len(self.samples),
            "errors": list(self.errors),
            "start": start,
            "end": end,
            "peaks": peaks,
            "mins": mins,
            **counter_deltas,
            "transitions": self.transitions,
        }


def service_columns(health, metadata=None):
    metadata = metadata or {}
    return {
        "per_sequence_limit": health.get("kv_sequence_limit"),
        "kv_page_tokens": health.get("kv_page_tokens"),
        "kv_pool_tokens": health.get("kv_pool_tokens"),
        "kv_pool_bytes": health.get("kv_pool_bytes"),
        "kv_pool_blocks_total": health.get("kv_pool_blocks_total"),
        "slots": health.get("slots"),
        "kv_layout": metadata.get("kv_layout", health.get("kv_layout")),
        "attention_config": metadata.get(
            "attention_config", health.get("attention_config")),
        "attention_branch_counts": metadata.get(
            "attention_branch_counts", health.get("attention_branch_counts")),
        "tier": metadata.get("tier", health.get("tier")),
        "k64": metadata.get("k64", health.get("k64")),
        "apc_enabled": metadata.get("apc_enabled", health.get("apc_enabled")),
    }


def histogram_add(target, source):
    for width, count in (source or {}).items():
        key = str(width)
        target[key] = target.get(key, 0) + int(count)


def exclusively_width_one(histogram, generated_tokens):
    if not isinstance(histogram, dict) or not isinstance(generated_tokens, int):
        return False
    expected_steps = generated_tokens - 1
    if expected_steps <= 0:
        return False
    try:
        normalized = {
            int(width): int(count) for width, count in histogram.items()
            if int(count) != 0
        }
    except (TypeError, ValueError):
        return False
    return normalized == {1: expected_steps}


def completion_reservation_footprint(samples, rows, requested_gen):
    result = {
        "label": "observed completion-reservation footprint reduction",
        "formula": "kv_active_blocks + kv_remaining_reserved_blocks",
        "unshared_bound_formula":
            "sum(ceil((usage.prompt_tokens + requested_output_tokens) / kv_page_tokens))",
        "eligible": False,
        "phase_isolated_at_start": False,
        "inputs_verified": False,
        "unshared_bound_blocks": None,
        "two_active_samples": 0,
        "consistent_samples": 0,
        "inconsistent_samples": 0,
        "observations": [],
        "positive_savings_observed": False,
        "all_consistent_samples_positive_savings": False,
        "maximum_shared_blocks_observed": None,
        "note": (
            "Every qualifying engine-published snapshot is computed. "
            "active and checkpoint block counts are never added."),
    }
    if not samples or len(rows) != 2:
        return result
    first = samples[0]
    result["phase_isolated_at_start"] = (
        int(first.get("active") or 0) == 0 and
        int(first.get("queued") or 0) == 0)
    page = rows[0].get("kv_page_tokens")
    actual_prompts = [row.get("usage_prompt_tokens") for row in rows]
    result["inputs_verified"] = bool(
        isinstance(page, int) and page > 0 and
        all(row.get("http_status") == 200 for row in rows) and
        all(isinstance(tokens, int) for tokens in actual_prompts) and
        all(row.get("usage_prompt_tokens") == row.get("prompt_tokens_local")
            for row in rows) and
        all(row.get("requested_output_tokens") == requested_gen for row in rows))
    if not result["inputs_verified"]:
        return result
    unshared = sum(
        (tokens + requested_gen + page - 1) // page
        for tokens in actual_prompts)
    result["unshared_bound_blocks"] = unshared
    for sample in samples:
        if sample.get("active") != 2:
            continue
        result["two_active_samples"] += 1
        values = {
            key: sample.get(key) for key in (
                "kv_active_blocks", "kv_remaining_reserved_blocks",
                "kv_pool_blocks_total", "kv_pool_blocks_used",
                "kv_pool_blocks_free")
        }
        consistent = (
            all(type(value) is int for value in values.values()) and
            sample.get("status") == "ok" and
            int(sample.get("queued") or 0) == 0 and
            int(sample.get("capacity_waiting") or 0) == 0 and
            0 <= values["kv_active_blocks"] <= values["kv_pool_blocks_used"] and
            0 <= values["kv_remaining_reserved_blocks"] <=
            values["kv_pool_blocks_free"] and
            values["kv_pool_blocks_used"] + values["kv_pool_blocks_free"] ==
            values["kv_pool_blocks_total"])
        if not consistent:
            result["inconsistent_samples"] += 1
            continue
        footprint = (
            values["kv_active_blocks"] +
            values["kv_remaining_reserved_blocks"])
        savings = unshared - footprint
        result["consistent_samples"] += 1
        result["observations"].append({
            "offset_seconds": sample.get("offset_seconds"),
            "kv_active_blocks": values["kv_active_blocks"],
            "kv_remaining_reserved_blocks":
                values["kv_remaining_reserved_blocks"],
            "observed_completion_reservation_footprint_blocks": footprint,
            "unshared_bound_blocks": unshared,
            "observed_shared_blocks": savings,
        })
    savings_values = [
        observation["observed_shared_blocks"]
        for observation in result["observations"]]
    result["eligible"] = bool(
        result["phase_isolated_at_start"] and
        result["inputs_verified"] and result["consistent_samples"] > 0)
    result["positive_savings_observed"] = bool(
        result["eligible"] and any(value > 0 for value in savings_values))
    result["all_consistent_samples_positive_savings"] = bool(
        result["eligible"] and savings_values and
        all(value > 0 for value in savings_values))
    result["maximum_shared_blocks_observed"] = (
        max(savings_values) if savings_values else None)
    return result


def completion_worker(base, model, spec, gen, timeout, barrier, result):
    try:
        barrier.wait(timeout=timeout)
        started = time.perf_counter()
        body = {"model": model, "prompt": spec["prompt"]["text"],
                "max_tokens": gen, "ignore_eos": True, "stream": False}
        status, payload, elapsed = post_json_status(
            base, "/completions", body, timeout)
        ended = time.perf_counter()
        row = {
            **public_prompt(spec["prompt"], spec["sequence"]),
            "request_start_monotonic": started,
            "request_end_monotonic": ended,
            "http_status": status,
            "wall_seconds": elapsed,
            "requested_output_tokens": gen,
            "errors": [],
        }
        if status != 200:
            message = ((payload.get("error") or {}).get("message")
                       if isinstance(payload, dict) else str(payload))
            row["errors"].append(f"HTTP {status}: {message}")
            result.update(row)
            return
        usage = payload.get("usage") or {}
        metadata, missing = response_metadata(payload)
        choice = (payload.get("choices") or [{}])[0]
        text = choice.get("text") or ""
        row.update({
            "usage_prompt_tokens": usage.get("prompt_tokens"),
            "usage_completion_tokens": usage.get("completion_tokens"),
            "usage_total_tokens": usage.get("total_tokens"),
            "generated_output_tokens": metadata.get("generated_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "output_text": text,
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "queue_ms": metadata.get("queue_ms"),
            "admission_ms": metadata.get("admission_ms"),
            "ttft_ms": metadata.get("ttft_ms"),
            "itl_ms_p50": metadata.get("itl_ms_p50"),
            "itl_ms_p99": metadata.get("itl_ms_p99"),
            "decode_width_hist": metadata.get("decode_width_hist") or {},
            "batch_avg": metadata.get("batch_avg"),
            "reused_tokens": metadata.get("reused_tokens"),
            "prefilled_tokens": metadata.get("prefilled_tokens"),
            "engine": metadata.get("engine"),
            "response_metadata_missing": missing,
            "metadata": metadata,
        })
        expected_prompt = spec["prompt"]["prompt_tokens"]
        if usage.get("prompt_tokens") != expected_prompt:
            row["errors"].append(
                f"usage.prompt_tokens={usage.get('prompt_tokens')} != real "
                f"tokenizer length {expected_prompt}")
        if usage.get("completion_tokens") != gen:
            row["errors"].append(
                f"usage.completion_tokens={usage.get('completion_tokens')} != {gen}")
        if metadata.get("generated_tokens") != gen:
            row["errors"].append(
                f"x_knivesysl.generated_tokens={metadata.get('generated_tokens')} "
                f"!= {gen}")
        if choice.get("finish_reason") != "length":
            row["errors"].append(
                f"finish_reason={choice.get('finish_reason')!r}, expected 'length'")
        if missing:
            row["errors"].append(
                "missing x_knivesysl fields: " + ", ".join(missing))
        ttft = metadata.get("ttft_ms")
        generated = usage.get("completion_tokens")
        post_first = elapsed - ttft / 1000.0 if isinstance(ttft, (int, float)) else None
        row["per_stream_decode_tok_s_observed"] = (
            max(0, generated - 1) / post_first
            if isinstance(generated, int) and post_first and post_first > 0 else None)
        row["per_stream_e2e_tok_s_observed"] = (
            generated / elapsed
            if isinstance(generated, int) and elapsed > 0 else None)
        result.update(row)
    except Exception as error:
        result.update({
            **public_prompt(spec["prompt"], spec["sequence"]),
            "requested_output_tokens": gen,
            "http_status": None,
            "errors": [str(error)],
        })


def run_completion_phase(base, model, specs, gen, timeout, label,
                         launch_order, profile, repeat):
    monitor = HealthMonitor(base, timeout)
    monitor.start()
    baseline = monitor.wait_for(
        lambda health: (
            health.get("status") == "ok" and
            int(health.get("active") or 0) == 0 and
            int(health.get("queued") or 0) == 0),
        min(timeout, 30.0), "idle initial phase health snapshot")
    with monitor.condition:
        monitor.samples = [baseline]
        monitor.transitions = [baseline]
    barrier = threading.Barrier(len(specs) + 1)
    results = [{} for _ in specs]
    threads = []
    for index in launch_order:
        thread = threading.Thread(
            target=completion_worker,
            args=(base, model, specs[index], gen, timeout, barrier,
                  results[index]),
            name=f"mixed-{label}-{index}", daemon=True)
        threads.append(thread)
        thread.start()
    phase_started = time.perf_counter()
    barrier.wait(timeout=timeout)
    deadline = time.perf_counter() + timeout
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.perf_counter()))
    alive = [thread.name for thread in threads if thread.is_alive()]
    phase_ended = time.perf_counter()
    monitor.stop()
    health = monitor.summary()
    start_health = health.get("start") or {}
    for row in results:
        row.update(service_columns(start_health, row.get("metadata")))
        if row.get("request_start_monotonic") is not None:
            row["request_start_offset_ms"] = round(
                (row.pop("request_start_monotonic") - phase_started) * 1000.0, 3)
        if row.get("request_end_monotonic") is not None:
            row["request_end_offset_ms"] = round(
                (row.pop("request_end_monotonic") - phase_started) * 1000.0, 3)
        row.pop("metadata", None)
    errors = []
    if alive:
        errors.append("request threads exceeded --timeout: " + ", ".join(alive))
    for row in results:
        errors.extend(row.get("errors") or [])
    if monitor.errors:
        errors.extend("health monitor: " + error for error in monitor.errors)
    start_snapshot = health.get("start") or {}
    end_snapshot = health.get("end") or {}
    request_errors_delta = (
        int(end_snapshot.get("request_errors") or 0) -
        int(start_snapshot.get("request_errors") or 0))
    if request_errors_delta:
        errors.append(
            f"server request_errors increased by {request_errors_delta}: "
            f"{end_snapshot.get('last_request_error')}")
    if end_snapshot.get("status") != "ok":
        errors.append(f"server health ended {end_snapshot.get('status')}: "
                      f"{end_snapshot.get('engine_error')}")
    successful = [row for row in results if row.get("http_status") == 200]
    generated = sum(row.get("usage_completion_tokens") or 0
                    for row in successful)
    first_tokens = []
    last_ends = []
    for row in successful:
        start_offset = row.get("request_start_offset_ms")
        ttft = row.get("ttft_ms")
        end_offset = row.get("request_end_offset_ms")
        if isinstance(start_offset, (int, float)) and isinstance(ttft, (int, float)):
            first_tokens.append((start_offset + ttft) / 1000.0)
        if isinstance(end_offset, (int, float)):
            last_ends.append(end_offset / 1000.0)
    decode_interval = ((max(last_ends) - min(first_tokens))
                       if first_tokens and last_ends else None)
    post_first_tokens = sum(max(0, (row.get("usage_completion_tokens") or 0) - 1)
                            for row in successful)
    request_widths = {}
    for row in successful:
        histogram_add(request_widths, row.get("decode_width_hist"))
    actual_order = [row.get("sequence") for row in sorted(
        successful, key=lambda row: row.get("request_start_offset_ms", math.inf))]
    wall = phase_ended - phase_started
    footprint = completion_reservation_footprint(
        monitor.samples, results, gen)
    return {
        "label": label,
        "profile": profile,
        "repeat": repeat,
        "launch_order": [specs[index]["sequence"] for index in launch_order],
        "observed_request_start_order": actual_order,
        "wall_seconds": wall,
        "requests": results,
        "health_observation": health,
        "observed_batch_widths_from_responses": dict(sorted(request_widths.items())),
        "completion_reservation_footprint": footprint,
        "aggregate_decode_tok_s_observed": (
            post_first_tokens / decode_interval
            if decode_interval is not None and decode_interval > 0 else None),
        "aggregate_e2e_tok_s_observed": generated / wall if wall > 0 else None,
        "server_decode_tok_s_end": ((health.get("end") or {}).get("decode_tok_s")),
        "server_decode_tok_s_by_width_end": (
            (health.get("end") or {}).get("decode_tok_s_by_width")),
        "server_request_errors_delta": request_errors_delta,
        "server_last_request_error_end": end_snapshot.get("last_request_error"),
        "capacity_waits_delta": (
            int(end_snapshot.get("capacity_waits") or 0) -
            int(start_snapshot.get("capacity_waits") or 0)),
        "apc_hits_delta": (
            int(end_snapshot.get("apc_hits") or 0) -
            int(start_snapshot.get("apc_hits") or 0)),
        "apc_misses_delta": (
            int(end_snapshot.get("apc_misses") or 0) -
            int(start_snapshot.get("apc_misses") or 0)),
        "errors": errors,
    }


def specs_for_prompts(prompts):
    return [{"sequence": f"sequence_{index}", "prompt": prompt}
            for index, prompt in enumerate(prompts)]


def summarize_repeats(phases):
    def spread(field):
        values = [phase.get(field) for phase in phases
                  if isinstance(phase.get(field), (int, float))]
        if not values:
            return None
        return {
            "min": min(values), "p50": percentile(values, 0.50),
            "max": max(values), "spread": max(values) - min(values),
        }
    ttfts = []
    itl50 = []
    itl99 = []
    for phase in phases:
        for row in phase.get("requests", []):
            if isinstance(row.get("ttft_ms"), (int, float)):
                ttfts.append(row["ttft_ms"])
            if isinstance(row.get("itl_ms_p50"), (int, float)):
                itl50.append(row["itl_ms_p50"])
            if isinstance(row.get("itl_ms_p99"), (int, float)):
                itl99.append(row["itl_ms_p99"])
    return {
        "repeats": len(phases),
        "wall_seconds": spread("wall_seconds"),
        "aggregate_decode_tok_s_observed": spread(
            "aggregate_decode_tok_s_observed"),
        "aggregate_e2e_tok_s_observed": spread(
            "aggregate_e2e_tok_s_observed"),
        "ttft_ms": ({"min": min(ttfts), "p50": percentile(ttfts, 0.50),
                     "p99": percentile(ttfts, 0.99), "max": max(ttfts)}
                    if ttfts else None),
        "itl_ms_p50_reported": ({"min": min(itl50),
                                  "p50": percentile(itl50, 0.50),
                                  "max": max(itl50)} if itl50 else None),
        "itl_ms_p99_reported": ({"min": min(itl99),
                                  "p50": percentile(itl99, 0.50),
                                  "max": max(itl99)} if itl99 else None),
    }


def phase_peak(phase, key):
    return (((phase.get("health_observation") or {}).get("peaks") or {})
            .get(key))


def prepare_warm_profile(base, model, specs, gen, timeout, profile):
    phases = []
    references = {}
    replay_phases = []
    same_path_comparisons = []
    unique = []
    seen = set()
    # Capture every exact-output reference before any checkpoint is created.
    # Priming then proceeds by ascending prompt length: in the matching
    # profile, a long checkpoint is produced by a request that adopted the
    # already-primed short checkpoint, exposing real shared pages rather than
    # two unrelated cached copies.
    ordered = sorted(specs, key=lambda spec: spec["prompt"]["prompt_tokens"])
    for spec in ordered:
        fingerprint = spec["prompt"]["prompt_sha256"]
        if fingerprint not in seen:
            unique.append(spec)
            seen.add(fingerprint)
    for item, spec in enumerate(unique):
        single = [{"sequence": spec["sequence"], "prompt": spec["prompt"]}]
        seed = run_completion_phase(
            base, model, single, gen, timeout,
            f"warm-{profile}-cold-reference-{item}", [0], profile, None)
        phases.append(seed)
        if (seed["requests"] and
                (seed["requests"][0].get("reused_tokens") or 0) != 0):
            seed["errors"].append(
                "cold reference unexpectedly reused APC state; restart the "
                "candidate service for an unambiguous warm comparison")
        if seed["requests"]:
            row = seed["requests"][0]
            references[spec["prompt"]["prompt_sha256"]] = {
                "output_text": row.get("output_text"),
                "output_sha256": row.get("output_sha256"),
                "generated_tokens": row.get("usage_completion_tokens"),
                "decode_width_hist": row.get("decode_width_hist") or {},
            }
    for item, spec in enumerate(unique):
        single = [{"sequence": spec["sequence"], "prompt": spec["prompt"]}]
        prime = run_completion_phase(
            base, model, single, gen, timeout,
            f"warm-{profile}-prime-{item}", [0], profile, None)
        phases.append(prime)
    # The APC correctness proof is deliberately isolated on both arms. The
    # cold reference and this warm replay therefore both dispatch qwn_decode
    # (GEMV), while concurrent RC8 results remain measurements rather than a
    # cross-arithmetic correctness gate.
    for item, spec in enumerate(unique):
        single = [{"sequence": spec["sequence"], "prompt": spec["prompt"]}]
        replay = run_completion_phase(
            base, model, single, gen, timeout,
            f"warm-{profile}-isolated-replay-{item}", [0], profile, None)
        replay_phases.append(replay)
        row = replay["requests"][0] if replay["requests"] else {}
        reference = references.get(spec["prompt"]["prompt_sha256"]) or {}
        cold_width_one = exclusively_width_one(
            reference.get("decode_width_hist"),
            reference.get("generated_tokens"))
        warm_width_one = exclusively_width_one(
            row.get("decode_width_hist"),
            row.get("usage_completion_tokens"))
        reused = row.get("reused_tokens") or 0
        count_equal = (
            isinstance(reference.get("generated_tokens"), int) and
            row.get("usage_completion_tokens") ==
            reference.get("generated_tokens"))
        text_equal = (
            isinstance(reference.get("output_text"), str) and
            row.get("output_text") == reference.get("output_text"))
        hash_equal = (
            isinstance(reference.get("output_sha256"), str) and
            row.get("output_sha256") == reference.get("output_sha256"))
        attested = bool(
            reused > 0 and cold_width_one and warm_width_one and
            count_equal and text_equal and hash_equal)
        comparison = {
            "sequence": spec["sequence"],
            "prompt_sha256": spec["prompt"]["prompt_sha256"],
            "proof_role": "same-path isolated n1 APC correctness gate",
            "cold_decode_width_hist":
                reference.get("decode_width_hist"),
            "warm_decode_width_hist": row.get("decode_width_hist"),
            "cold_exclusively_width1": cold_width_one,
            "warm_exclusively_width1": warm_width_one,
            "warm_reused_tokens": reused,
            "generated_output_count_equal": count_equal,
            "exact_text_equal": text_equal,
            "exact_sha256_equal": hash_equal,
            "cold_output_sha256": reference.get("output_sha256"),
            "warm_output_sha256": row.get("output_sha256"),
            "attested": attested,
        }
        same_path_comparisons.append(comparison)
        if not attested:
            replay["errors"].append(
                "isolated n1 cold/warm APC proof failed: "
                f"reuse={reused} cold_width1={cold_width_one} "
                f"warm_width1={warm_width_one} count_equal={count_equal} "
                f"text_equal={text_equal} hash_equal={hash_equal}")
    return phases, references, replay_phases, same_path_comparisons


def concurrent_output_observations(phases, references):
    observations = []
    for phase in phases:
        for row in phase.get("requests", []):
            reference = references.get(row.get("prompt_sha256")) or {}
            text_equal = (
                isinstance(reference.get("output_text"), str) and
                row.get("output_text") == reference.get("output_text"))
            hash_equal = (
                isinstance(reference.get("output_sha256"), str) and
                row.get("output_sha256") == reference.get("output_sha256"))
            count_equal = (
                isinstance(reference.get("generated_tokens"), int) and
                row.get("usage_completion_tokens") ==
                reference.get("generated_tokens"))
            observations.append({
                "phase": phase.get("label"),
                "repeat": phase.get("repeat"),
                "sequence": row.get("sequence"),
                "prompt_sha256": row.get("prompt_sha256"),
                "proof_role": (
                    "cross-path output observation only; native width/row-"
                    "matched three-arm proof remains authoritative"),
                "cold_isolated_decode_width_hist":
                    reference.get("decode_width_hist"),
                "concurrent_decode_width_hist":
                    row.get("decode_width_hist"),
                "reference_output_sha256":
                    reference.get("output_sha256"),
                "measured_output_sha256": row.get("output_sha256"),
                "generated_output_count_equal": count_equal,
                "exact_text_equal_observed": text_equal,
                "exact_sha256_equal_observed": hash_equal,
                "not_an_apc_correctness_gate": True,
            })
    return observations


def run_mixed(args, base, factory, initial_health):
    benchmark_started = time.perf_counter()
    lengths = args.mixed
    mode = args.benchmark_phase
    if mode == "auto":
        mode = "warm" if initial_health.get("apc_enabled") else "cold"
    expected_apc = mode == "warm"
    if initial_health.get("apc_enabled") is not expected_apc:
        raise QualificationError(
            f"mixed {mode} phase requires apc_enabled={expected_apc}, health "
            f"reported {initial_health.get('apc_enabled')!r}")
    if args.expect_apc is not None:
        cli_expected = args.expect_apc == "on"
        if initial_health.get("apc_enabled") is not cli_expected:
            raise QualificationError(
                f"--expect-apc {args.expect_apc}, health reported "
                f"{initial_health.get('apc_enabled')!r}")

    result = {
        "mode": "exact-token-mixed",
        "benchmark_phase": mode,
        "apc_attestation": {
            "health_apc_enabled": initial_health.get("apc_enabled"),
            "expected": expected_apc,
            "attested": initial_health.get("apc_enabled") is expected_apc,
            "initial_entries": initial_health.get("apc_entries"),
            "initial_hits": initial_health.get("apc_hits"),
        },
        "prompt_lengths": lengths,
        "requested_output_tokens": args.gen,
        "repeats": args.repeats,
        "profiles": {},
        "errors": [],
    }

    if mode == "cold":
        prompts = [factory.make("alpha", lengths[0]),
                   factory.make("beta", lengths[1])]
        specs = specs_for_prompts(prompts)
        verified_common = common_prefix_tokens(prompts[0], prompts[1])
        phases = []
        for repeat in range(args.repeats):
            order = [0, 1] if repeat % 2 == 0 else [1, 0]
            phases.append(run_completion_phase(
                base, args.model, specs, args.gen, args.timeout,
                f"cold-apc-off-repeat-{repeat + 1}", order,
                "cold-apc-off", repeat + 1))
        reused = sum((row.get("reused_tokens") or 0)
                     for phase in phases for row in phase["requests"])
        end_health = ((phases[-1].get("health_observation") or {}).get("end")
                      if phases else initial_health) or {}
        cold_attested = (initial_health.get("apc_enabled") is False and
                         reused == 0 and
                         end_health.get("apc_hits") == initial_health.get("apc_hits") and
                         end_health.get("apc_entries") == initial_health.get("apc_entries"))
        if not cold_attested:
            result["errors"].append(
                "cold APC-off phase was not attested by health and zero reuse")
        result["profiles"]["cold-apc-off"] = {
            "tokenizer_verified_common_prefix_tokens": verified_common,
            "phases": phases,
            "summary": summarize_repeats(phases),
            "attested": cold_attested,
        }
        for phase in phases:
            result["errors"].extend(phase["errors"])
        result["total_wall_seconds"] = time.perf_counter() - benchmark_started
        return result

    profiles = {
        "distinct-prefix-control": [
            factory.make("beta", lengths[0]),
            factory.make("gamma", lengths[1]),
        ],
        "matching-prefix": [
            factory.make("alpha", lengths[0]),
            factory.make("alpha", lengths[1]),
        ],
    }
    for profile, prompts in profiles.items():
        specs = specs_for_prompts(prompts)
        common = common_prefix_tokens(prompts[0], prompts[1])
        prep, references, replays, same_path = prepare_warm_profile(
            base, args.model, specs, args.gen, args.timeout, profile)
        measured = []
        for repeat in range(args.repeats):
            order = [0, 1] if repeat % 2 == 0 else [1, 0]
            measured.append(run_completion_phase(
                base, args.model, specs, args.gen, args.timeout,
                f"warm-apc-{profile}-repeat-{repeat + 1}", order,
                profile, repeat + 1))
        cross_path = concurrent_output_observations(measured, references)
        reused = sum((row.get("reused_tokens") or 0)
                     for phase in measured for row in phase["requests"])
        same_path_attested = bool(same_path) and all(
            comparison["attested"] for comparison in same_path)
        physical_samples = all(
            phase_peak(phase, "kv_pool_blocks_used") is not None and
            phase_peak(phase, "kv_checkpoint_blocks") is not None
            for phase in measured)
        footprints = [
            phase["completion_reservation_footprint"] for phase in measured]
        profile_result = {
            "tokenizer_verified_common_prefix_tokens": common,
            "preparation_phases": prep,
            "isolated_warm_replay_phases": replays,
            "same_path_apc_correctness": {
                "attested": same_path_attested,
                "comparisons": same_path,
                "proof_scope": (
                    "isolated n1 cold and warm replay both use scalar GEMV; "
                    "native width/row-matched three-arm proof remains distinct"),
            },
            "measured_phases": measured,
            "summary": summarize_repeats(measured),
            "concurrent_cross_path_output_observations": cross_path,
            "completion_reservation_footprints": footprints,
            "total_reused_tokens": reused,
            "apc_adoption_and_physical_observation": {
                "successful_adopt_reused_tokens": reused,
                "same_path_apc_correctness_attested": same_path_attested,
                "physical_pool_samples_present": physical_samples,
                "checkpoint_blocks_peaks": [
                    phase_peak(phase, "kv_checkpoint_blocks")
                    for phase in measured],
                "pool_blocks_used_peaks": [
                    phase_peak(phase, "kv_pool_blocks_used")
                    for phase in measured],
                "active_blocks_peaks": [
                    phase_peak(phase, "kv_active_blocks")
                    for phase in measured],
                "note": (
                    "Concurrent equality is a cross-path observation, never "
                    "the APC gate. Sharing requires physical ledger evidence, "
                    "successful reuse, and the isolated same-path proof."),
            },
        }
        result["profiles"][profile] = profile_result
        for phase in prep + replays + measured:
            result["errors"].extend(phase["errors"])
        if not same_path_attested:
            result["errors"].append(
                f"{profile}: isolated n1 cold/warm APC proof was not attested")
        if reused <= 0:
            result["errors"].append(
                f"{profile}: warm requests reported no successful checkpoint adoption")
        if not physical_samples:
            result["errors"].append(
                f"{profile}: native physical block counters were not exposed")

    distinct = result["profiles"]["distinct-prefix-control"]
    matching = result["profiles"]["matching-prefix"]
    comparisons = []
    for index in range(min(len(distinct["measured_phases"]),
                           len(matching["measured_phases"]))):
        dphase = distinct["measured_phases"][index]
        mphase = matching["measured_phases"][index]
        dpool = phase_peak(dphase, "kv_pool_blocks_used")
        mpool = phase_peak(mphase, "kv_pool_blocks_used")
        dactive_blocks = phase_peak(dphase, "kv_active_blocks")
        mactive_blocks = phase_peak(mphase, "kv_active_blocks")
        dactive_requests = phase_peak(dphase, "active")
        mactive_requests = phase_peak(mphase, "active")
        ddecode_overlap = any(
            int(width) >= 2 and int(count) > 0
            for width, count in
            dphase["observed_batch_widths_from_responses"].items())
        mdecode_overlap = any(
            int(width) >= 2 and int(count) > 0
            for width, count in
            mphase["observed_batch_widths_from_responses"].items())
        comparable = (
            isinstance(dactive_blocks, int) and
            isinstance(mactive_blocks, int) and
            isinstance(dactive_requests, int) and dactive_requests >= 2 and
            isinstance(mactive_requests, int) and mactive_requests >= 2 and
            ddecode_overlap and mdecode_overlap)
        active_reduction = (
            dactive_blocks - mactive_blocks if comparable else None)
        comparisons.append({
            "repeat": index + 1,
            "both_profiles_observed_with_two_active_and_decode_overlap": comparable,
            "distinct_multirow_decode_observed": ddecode_overlap,
            "matching_multirow_decode_observed": mdecode_overlap,
            "distinct_active_requests_peak": dactive_requests,
            "matching_active_requests_peak": mactive_requests,
            "distinct_unique_active_blocks_peak": dactive_blocks,
            "matching_unique_active_blocks_peak": mactive_blocks,
            "observed_unique_active_block_reduction": active_reduction,
            "distinct_pool_blocks_used_peak": dpool,
            "matching_pool_blocks_used_peak": mpool,
            "observed_pool_peak_difference_blocks": (
                dpool - mpool
                if isinstance(dpool, int) and isinstance(mpool, int) else None),
            "distinct_checkpoint_blocks_peak": phase_peak(
                dphase, "kv_checkpoint_blocks"),
            "matching_checkpoint_blocks_peak": phase_peak(
                mphase, "kv_checkpoint_blocks"),
            "physical_sharing_observed": bool(
                comparable and active_reduction is not None and
                active_reduction > 0),
        })
    physical_observed = any(
        item["physical_sharing_observed"] for item in comparisons)
    comparison_representable = any(
        item["both_profiles_observed_with_two_active_and_decode_overlap"]
        for item in comparisons)
    matching_rows = [
        row for phase in matching["measured_phases"]
        for row in phase["requests"]]
    concurrent_reuse_attested = bool(matching_rows) and all(
        (row.get("reused_tokens") or 0) > 0 for row in matching_rows)
    same_path_attested = matching["same_path_apc_correctness"]["attested"]
    footprint_phases = matching["completion_reservation_footprints"]
    footprint_samples_complete = bool(footprint_phases) and all(
        footprint["eligible"] for footprint in footprint_phases)
    footprint_positive = bool(footprint_phases) and all(
        footprint["all_consistent_samples_positive_savings"]
        for footprint in footprint_phases)
    footprint_attested = bool(
        footprint_samples_complete and footprint_positive and
        concurrent_reuse_attested and same_path_attested)
    matching["observed_completion_reservation_footprint_reduction"] = {
        "attested": footprint_attested,
        "calculated_unshared_bounds_blocks": [
            footprint["unshared_bound_blocks"] for footprint in footprint_phases],
        "maximum_shared_blocks_observed": [
            footprint["maximum_shared_blocks_observed"]
            for footprint in footprint_phases],
        "every_repeat_has_consistent_two-active_samples":
            footprint_samples_complete,
        "every_consistent_sample_has_positive_savings": footprint_positive,
        "all_concurrent_matching_requests_reused": concurrent_reuse_attested,
        "same_path_apc_correctness_attested": same_path_attested,
        "interpretation": (
            "For an engine-published two-active snapshot, unique active "
            "physical blocks plus remaining reservation credits is the active "
            "completion footprint. Its positive reduction from the unshared "
            "page bound is the duplicated mapped-page reference count."),
    }
    matching["matched_peak_overlap_evidence"] = {
        "observed": physical_observed,
        "comparison_representable": comparison_representable,
        "role": (
            "Separate matched peak/overlap evidence where both controls "
            "naturally reach multirow decode; not the constrained-pool proof."),
    }
    matching["physical_sharing_evidence"] = {
        "attested": footprint_attested,
        "evidence_label":
            "observed completion-reservation footprint reduction",
        "successful_adopt_reuse_attested": concurrent_reuse_attested,
        "same_path_apc_correctness_attested": same_path_attested,
        "positive_completion_footprint_savings_attested": footprint_positive,
        "counter_semantics": (
            "kv_active_blocks counts unique physical active pages and is added "
            "only to remaining reservation credits. active and checkpoint "
            "block counts are never added because those categories overlap."),
        "scope_limit": (
            "HTTP ledger snapshots prove observed physical sharing; exact "
            "block-table identity remains a native-gate responsibility."),
    }
    result["matching_vs_distinct_physical_comparison"] = comparisons
    if not footprint_attested:
        result["errors"].append(
            "matching-prefix APC did not attest positive observed completion-"
            "reservation footprint savings with successful concurrent reuse "
            "and the isolated same-path n1 correctness proof")
    if comparison_representable and not physical_observed:
        result["errors"].append(
            "matching-prefix APC showed no unique-active-block reduction "
            "against the concurrently representable distinct-prefix control")
    result["total_wall_seconds"] = time.perf_counter() - benchmark_started
    return result


def smoke_one(base, model, index, results, gen, timeout, barrier):
    body = {"model": model, "max_tokens": gen, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": FILLER},
                         {"role": "user",
                          "content": f"Reply with the number {index}."}]}
    try:
        barrier.wait(timeout=timeout)
        output, seconds = post(base, "/chat/completions", body, timeout)
        metadata, missing = response_metadata(output)
        usage = output.get("usage") or {}
        row = {
            "ok": not missing,
            "seconds": seconds,
            "text": ((output["choices"][0]["message"].get("content") or "")[:80]),
            "finish_reason": output["choices"][0]["finish_reason"],
            "batch_avg": metadata.get("batch_avg"),
            "reused_tokens": metadata.get("reused_tokens"),
            "ttft_ms": metadata.get("ttft_ms"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "metadata_missing": missing,
        }
        if usage.get("completion_tokens") != gen:
            row["ok"] = False
            row["error"] = (
                f"completion_tokens={usage.get('completion_tokens')} != {gen}")
        results[index] = row
    except Exception as error:
        results[index] = {"ok": False, "error": str(error)[:500]}


def smoke_wave(base, model, concurrency, gen, timeout, label):
    results = {}
    barrier = threading.Barrier(concurrency + 1)
    threads = [threading.Thread(
        target=smoke_one,
        args=(base, model, index, results, gen, timeout, barrier),
        name=f"smoke-{index}", daemon=True)
        for index in range(concurrency)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    barrier.wait(timeout=timeout)
    deadline = time.perf_counter() + timeout
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.perf_counter()))
    wall = time.perf_counter() - started
    for thread in threads:
        if thread.is_alive():
            results.setdefault(thread.name, {
                "ok": False, "error": "request exceeded --timeout"})
    ok = [row for row in results.values() if row.get("ok")]
    bad = [row for row in results.values() if not row.get("ok")]
    print(f"\n[{label}] {len(ok)}/{concurrency} ok in {wall:.2f} s")
    for row in bad:
        print(f"   FAIL {row.get('error') or row.get('metadata_missing')}")
    if ok:
        batch_values = [row["batch_avg"] for row in ok
                        if isinstance(row.get("batch_avg"), (int, float))]
        reused = sum(row.get("reused_tokens") or 0 for row in ok)
        ttfts = [row["ttft_ms"] for row in ok
                 if isinstance(row.get("ttft_ms"), (int, float))]
        if batch_values:
            print(f"   batch_avg  {statistics.fmean(batch_values):.2f}  "
                  f"(max {max(batch_values):.2f})")
        print(f"   reused     {reused} tokens over the wave")
        if ttfts:
            print(f"   ttft       mean {statistics.fmean(ttfts):.0f} ms  "
                  f"max {max(ttfts):.0f} ms")
        print(f"   throughput {len(ok) * gen / wall:.1f} tok/s aggregate")
    return {"label": label, "ok": ok, "bad": bad, "wall_seconds": wall}


def run_case(cases, failures, name, function):
    started = time.perf_counter()
    try:
        detail = function() or {}
        cases.append({"name": name, "status": "pass",
                      "seconds": time.perf_counter() - started,
                      "detail": detail})
        print(f"{name:<24} ok")
    except Exception as error:
        message = str(error)
        entry = {"name": name, "status": "fail",
                 "seconds": time.perf_counter() - started,
                 "error": message}
        if getattr(error, "detail", None) is not None:
            entry["detail"] = error.detail
        cases.append(entry)
        failures.append(f"{name}: {message}")
        print(f"{name:<24} FAIL  {message}")


def run_surface_checks(args, base, initial_health, factory, failures):
    cases = []

    def models_case():
        payload = get_json(base, "/v1/models", args.timeout)
        ids = [item["id"] for item in payload["data"]]
        if args.model not in ids:
            raise QualificationError(f"{args.model} not in {ids}")
        return {"model_ids": ids}

    def completion_case():
        payload, elapsed = post(
            base, "/completions",
            {"model": args.model, "prompt": "2 + 2 =", "max_tokens": 8},
            args.timeout)
        if payload.get("object") != "text_completion":
            raise QualificationError(f"bad object: {payload.get('object')}")
        text = (payload.get("choices") or [{}])[0].get("text") or ""
        if not text:
            raise QualificationError("empty completion")
        missing = response_metadata(payload)[1]
        if missing:
            raise QualificationError("missing metadata: " + ", ".join(missing))
        no_markup(text)
        return {"text": text, "seconds": elapsed}

    def fixed_generation_case():
        requested = 8
        payload, elapsed = post(
            base, "/completions",
            {"model": args.model, "prompt": "Count upward:",
             "max_tokens": requested, "ignore_eos": True}, args.timeout)
        usage = payload.get("usage") or {}
        if usage.get("completion_tokens") != requested:
            raise QualificationError(
                f"fixed output produced {usage.get('completion_tokens')}, "
                f"expected {requested}")
        return {"requested": requested, "usage": usage, "seconds": elapsed}

    def chat_case():
        payload, elapsed = post(
            base, "/chat/completions",
            {"model": args.model, "max_tokens": 8, "stop": ["\n"],
             "chat_template_kwargs": {"enable_thinking": False},
             "messages": [{"role": "user", "content": "Say: alpha beta"}]},
            args.timeout)
        if payload.get("object") != "chat.completion":
            raise QualificationError(f"bad object: {payload.get('object')}")
        content, reasoning = response_visible_text(payload)
        no_markup(content, reasoning)
        return {"finish_reason": payload["choices"][0]["finish_reason"],
                "content": content, "seconds": elapsed}

    def stream_completion_case():
        stream = parse_sse(
            base, "/completions",
            {"model": args.model, "prompt": "Continue: one two",
             "max_tokens": 8, "ignore_eos": True, "stream": True},
            args.timeout)
        if stream["usage"].get("completion_tokens") != 8:
            raise QualificationError(f"bad streaming usage: {stream['usage']}")
        if not stream["text"]:
            raise QualificationError("empty streaming completion")
        no_markup(stream["text"])
        missing = [key for key in RESPONSE_PLAN_FIELDS
                   if key not in stream["x_knivesysl"]]
        if missing:
            raise QualificationError("missing SSE metadata: " + ", ".join(missing))
        return stream

    def reasoning_nonstream_case():
        payload, elapsed = post(
            base, "/chat/completions",
            {"model": args.model, "max_tokens": max(128, args.gen),
             "messages": [{"role": "user",
                           "content": "Reason briefly, then answer: what is 2+2?"}],
             "chat_template_kwargs": {"enable_thinking": True}}, args.timeout)
        message = payload["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        no_markup(content, reasoning)
        if not reasoning:
            raise QualificationError("reasoning_content was not separated")
        return {"reasoning_content": reasoning, "content": content,
                "seconds": elapsed}

    def reasoning_stream_case():
        stream = parse_sse(
            base, "/chat/completions",
            {"model": args.model, "max_tokens": max(128, args.gen),
             "stream": True,
             "messages": [{"role": "user",
                           "content": "Reason briefly, then answer: what is 3+3?"}],
             "chat_template_kwargs": {"enable_thinking": True}}, args.timeout)
        no_markup(stream["content"], stream["reasoning_content"])
        if not stream["reasoning_content"]:
            raise QualificationError("stream did not separate reasoning_content")
        return stream

    tools = [{"type": "function", "function": {
        "name": "lookup_temperature",
        "description": "Look up the current temperature for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]}}}]
    tool_messages = [{"role": "user", "content":
                      "Call lookup_temperature for Oslo. Do not answer in prose."}]

    def tool_nonstream_case():
        payload, elapsed = post(
            base, "/chat/completions",
            {"model": args.model, "max_tokens": max(128, args.gen),
             "tools": tools, "messages": tool_messages,
             "chat_template_kwargs": {"enable_thinking": False}}, args.timeout)
        message = payload["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        no_markup(content, reasoning)
        calls = message.get("tool_calls") or []
        validate_tool_calls(calls)
        if payload["choices"][0].get("finish_reason") != "tool_calls":
            raise QualificationError("tool response finish_reason is not tool_calls")
        return {"content": content, "tool_calls": calls, "seconds": elapsed}

    def tool_stream_case():
        stream = parse_sse(
            base, "/chat/completions",
            {"model": args.model, "max_tokens": max(128, args.gen),
             "stream": True, "tools": tools, "messages": tool_messages,
             "chat_template_kwargs": {"enable_thinking": False}}, args.timeout)
        no_markup(stream["content"], stream["reasoning_content"])
        calls = []
        for call in stream["tool_calls"]:
            clean = dict(call)
            clean.pop("index", None)
            calls.append(clean)
        validate_tool_calls(calls)
        if stream.get("finish_reason") != "tool_calls":
            raise QualificationError("stream tool finish_reason is not tool_calls")
        stream["tool_calls"] = calls
        return stream

    def multi_token_stop_case():
        prompt = (
            "Reply with exactly this multi-word ASCII phrase and nothing else: "
            "cedar amber cobalt")
        baseline_body = {
            "model": args.model,
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}],
        }
        baseline, baseline_seconds = post(
            base, "/chat/completions", baseline_body, args.timeout)
        baseline_message = baseline["choices"][0]["message"]
        baseline_choice = baseline["choices"][0]
        captured = baseline_message.get("content") or ""
        captured_ids = (
            factory.tokenizer(captured, add_special_tokens=False).input_ids
            if captured else [])
        detail = {
            "baseline": {
                "content": captured,
                "content_sha256": (
                    hashlib.sha256(captured.encode()).hexdigest()
                    if captured else None),
                "real_tokenizer_tokens": len(captured_ids),
                "finish_reason": baseline_choice.get("finish_reason"),
                "usage": baseline.get("usage"),
                "seconds": baseline_seconds,
            },
            "variants": {},
            "attested": False,
        }
        baseline_issues = []
        try:
            no_markup(captured, baseline_message.get("reasoning_content") or "")
        except QualificationError as error:
            baseline_issues.append(str(error))
        if not captured:
            baseline_issues.append("multi-token stop baseline was empty")
        elif not captured.isascii():
            baseline_issues.append(
                f"multi-token stop baseline was not ASCII: {captured!r}")
        if len(captured_ids) <= 1:
            baseline_issues.append(
                f"captured stop tokenized to {len(captured_ids)} token(s), "
                "need more than one")
        if baseline_choice.get("finish_reason") != "stop":
            baseline_issues.append(
                "baseline did not finish at EOS: finish_reason="
                f"{baseline_choice.get('finish_reason')!r}")
        if baseline_issues:
            raise ObservedQualificationError(
                "; ".join(baseline_issues), detail)

        suffix = None
        suffix_ids = None
        word_starts = [
            index for index in range(1, len(captured))
            if captured[index].isspace() and
            index + 1 < len(captured) and
            not captured[index + 1].isspace()]
        candidate_starts = sorted(word_starts, reverse=True)
        candidate_starts.extend(
            index for index in range(len(captured) - 1, 0, -1)
            if index not in word_starts)
        for start in candidate_starts:
            candidate = captured[start:]
            prefix = captured[:start]
            candidate_ids = factory.tokenizer(
                candidate, add_special_tokens=False).input_ids
            if prefix.strip() and len(candidate_ids) > 1:
                suffix = candidate
                suffix_ids = candidate_ids
                break
        if suffix is None:
            raise ObservedQualificationError(
                "could not derive a proper multi-token suffix from baseline",
                detail)

        def exercise_variant(label, stop_text, expected_content):
            observation = {
                "stop": stop_text,
                "stop_sha256": hashlib.sha256(stop_text.encode()).hexdigest(),
                "expected_content_stripped": expected_content.strip(),
                "nonstream": None,
                "stream": None,
            }
            variant_issues = []
            nonstream_body = dict(baseline_body)
            nonstream_body["stop"] = stop_text
            try:
                nonstream, nonstream_seconds = post(
                    base, "/chat/completions", nonstream_body, args.timeout)
                choice = nonstream["choices"][0]
                content = (
                    (choice.get("message") or {}).get("content") or "")
                observation["nonstream"] = {
                    "content": content,
                    "content_stripped": content.strip(),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": nonstream.get("usage"),
                    "seconds": nonstream_seconds,
                }
                if choice.get("finish_reason") != "stop":
                    variant_issues.append(
                        f"{label} nonstream finish_reason was "
                        f"{choice.get('finish_reason')!r}, expected 'stop'")
                if content.strip() != expected_content.strip():
                    variant_issues.append(
                        f"{label} nonstream content {content!r} != expected "
                        f"{expected_content!r} after stripping")
            except Exception as error:
                observation["nonstream"] = {"error": str(error)}
                variant_issues.append(f"{label} nonstream failed: {error}")

            stream_body = dict(baseline_body)
            stream_body.update({"stop": stop_text, "stream": True})
            try:
                stream = parse_sse(
                    base, "/chat/completions", stream_body, args.timeout)
                observation["stream"] = stream
                if stream.get("finish_reason") != "stop":
                    variant_issues.append(
                        f"{label} streaming finish_reason was "
                        f"{stream.get('finish_reason')!r}, expected 'stop'")
                if stream["content"].strip() != expected_content.strip():
                    variant_issues.append(
                        f"{label} streaming content {stream['content']!r} != "
                        f"expected {expected_content!r} after stripping")
            except Exception as error:
                observation["stream"] = {"error": str(error)}
                variant_issues.append(f"{label} streaming failed: {error}")
            observation["attested"] = not variant_issues
            return observation, variant_issues

        unmatched_stop = captured + " __knivesysl_stop_suffix_not_generated__"
        variants = (
            ("full-content-match", captured, ""),
            ("unmatched-content-plus-suffix", unmatched_stop, captured),
            ("proper-multi-token-suffix",
             suffix, captured[:-len(suffix)]),
        )
        issues = []
        for label, stop_text, expected_content in variants:
            observation, variant_issues = exercise_variant(
                label, stop_text, expected_content)
            detail["variants"][label] = observation
            issues.extend(variant_issues)
        detail["suffix_calibration"] = {
            "suffix": suffix,
            "suffix_sha256": hashlib.sha256(suffix.encode()).hexdigest(),
            "real_tokenizer_tokens": len(suffix_ids),
            "legitimate_prefix": captured[:-len(suffix)],
        }
        detail["attested"] = not issues
        if issues:
            raise ObservedQualificationError("; ".join(issues), detail)
        return detail


    def reject_case(body, why):
        status, payload, elapsed = post_json_status(
            base, "/completions", body, args.timeout)
        if status != 400:
            raise QualificationError(f"{why} returned HTTP {status}, expected 400")
        return {"http_status": status, "response": payload, "seconds": elapsed}

    run_case(cases, failures, "models", models_case)
    run_case(cases, failures, "completions-nonstream", completion_case)
    run_case(cases, failures, "fixed-max-output", fixed_generation_case)
    run_case(cases, failures, "chat-nonstream-stop", chat_case)
    run_case(cases, failures, "completions-stream", stream_completion_case)
    if factory is not None:
        run_case(cases, failures, "multi-token-stop", multi_token_stop_case)
    else:
        cases.append({
            "name": "multi-token-stop",
            "status": "skipped",
            "reason": "--model-dir is required for real-tokenizer stop verification",
        })
        print(f"{'multi-token-stop':<24} SKIP  provide --model-dir")
    run_case(cases, failures, "reasoning-nonstream", reasoning_nonstream_case)
    run_case(cases, failures, "reasoning-stream", reasoning_stream_case)
    run_case(cases, failures, "tool-call-nonstream", tool_nonstream_case)
    run_case(cases, failures, "tool-call-stream", tool_stream_case)
    run_case(cases, failures, "reject-n>1", lambda: reject_case(
        {"model": args.model, "prompt": "x", "n": 2}, "n>1"))
    run_case(cases, failures, "reject-logprobs", lambda: reject_case(
        {"model": args.model, "prompt": "x", "logprobs": 3}, "logprobs"))

    limit = initial_health.get("kv_sequence_limit")
    if isinstance(limit, int) and initial_health.get("paged"):
        run_case(cases, failures, "reject-context-limit", lambda: reject_case(
            {"model": args.model, "prompt": "x", "max_tokens": limit,
             "ignore_eos": True}, "paged context overflow"))
        pool_tokens = initial_health.get("kv_pool_tokens")
        if isinstance(pool_tokens, int) and pool_tokens + 2 <= limit:
            run_case(cases, failures, "reject-empty-pool-limit", lambda: reject_case(
                {"model": args.model, "prompt": "x",
                 "max_tokens": pool_tokens, "ignore_eos": True},
                "empty pool overflow"))
    elif isinstance(limit, int) and factory is not None:
        def flat_context_case():
            prompt = factory.make("gamma", limit)
            return reject_case(
                {"model": args.model, "prompt": prompt["text"],
                 "max_tokens": 1}, "flat context overflow")
        run_case(cases, failures, "reject-context-limit", flat_context_case)
    else:
        cases.append({
            "name": "reject-context-limit", "status": "skipped",
            "reason": "flat service needs --model-dir for a tokenizer-verified exact-limit prompt",
        })
        print(f"{'reject-context-limit':<24} SKIP  provide --model-dir")
    return cases


class DisconnectingStream:
    """Streaming request whose TCP connection can be closed by the client."""

    def __init__(self, base, path, body, timeout, name):
        self.base = base
        self.path = path
        self.body = body
        self.timeout = timeout
        self.name = name
        self.opened = threading.Event()
        self.done = threading.Event()
        self.cancel_requested = threading.Event()
        self.lock = threading.Lock()
        self.connection = None
        self.response = None
        self.status = None
        self.events = 0
        self.error = None
        self.started = None
        self.cancelled_at = None
        self.thread = threading.Thread(target=self._run,
                                       name=f"disconnect-{name}", daemon=True)

    def _connection(self):
        parsed = urllib.parse.urlsplit(api_url(self.base, self.path))
        cls = (http.client.HTTPSConnection if parsed.scheme == "https"
               else http.client.HTTPConnection)
        port = parsed.port
        connection = cls(parsed.hostname, port=port, timeout=self.timeout)
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        return connection, target

    def _run(self):
        try:
            connection, target = self._connection()
            with self.lock:
                self.connection = connection
            self.started = time.perf_counter()
            data = json.dumps(self.body).encode()
            connection.request("POST", target, body=data, headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Content-Length": str(len(data)),
            })
            response = connection.getresponse()
            with self.lock:
                self.response = response
            self.status = response.status
            self.opened.set()
            if response.status != 200:
                self.error = response.read().decode(errors="replace")
                return
            while not self.cancel_requested.is_set():
                line = response.readline()
                if not line:
                    break
                if line.startswith(b"data:"):
                    self.events += 1
                    if b"[DONE]" in line:
                        break
        except Exception as error:
            if not self.cancel_requested.is_set():
                self.error = str(error)
        finally:
            self.opened.set()
            with self.lock:
                connection = self.connection
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            self.done.set()

    def start(self):
        self.thread.start()

    def cancel(self):
        self.cancelled_at = time.perf_counter()
        self.cancel_requested.set()
        with self.lock:
            connection = self.connection
            sock = connection.sock if connection is not None else None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def record(self, zero):
        return {
            "name": self.name,
            "http_status": self.status,
            "opened": self.opened.is_set(),
            "done": self.done.is_set(),
            "events_before_disconnect": self.events,
            "error": self.error,
            "start_offset_seconds": (self.started - zero
                                     if self.started is not None else None),
            "cancel_offset_seconds": (self.cancelled_at - zero
                                      if self.cancelled_at is not None else None),
        }


def parse_cancel_scenario(value):
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--cancel-scenario needs ACTIVE,PENDING,SHORT token counts") from error
    if len(parts) != 3 or any(part < 2 for part in parts):
        raise argparse.ArgumentTypeError(
            "--cancel-scenario needs three token counts >= 2")
    return parts


def run_cancel_scenario(args, base, factory, initial_health):
    active_tokens, pending_tokens, short_tokens = args.cancel_scenario
    prompts = {
        "active": factory.make("alpha", active_tokens, variant=101),
        "pending": factory.make("beta", pending_tokens, variant=102),
        "short": factory.make("gamma", short_tokens, variant=103),
    }
    limit = initial_health.get("kv_sequence_limit")
    page = initial_health.get("kv_page_tokens")
    pool_blocks = initial_health.get("kv_pool_blocks_total")
    for name, prompt in prompts.items():
        total = prompt["prompt_tokens"] + args.gen
        if isinstance(limit, int) and total > limit:
            raise QualificationError(
                f"cancel {name} needs {total} tokens, exceeds sequence limit {limit}")
        if (initial_health.get("paged") and isinstance(page, int) and
                isinstance(pool_blocks, int) and
                (total + page - 1) // page > pool_blocks):
            raise QualificationError(
                f"cancel {name} cannot fit the empty pool: {total} tokens")
    zero = time.perf_counter()
    monitor = HealthMonitor(base, args.timeout, interval=0.1)
    monitor.start()
    baseline = monitor.wait_for(lambda health: health.get("status") == "ok",
                                min(args.timeout, 30.0), "healthy baseline")
    base_active = int(baseline.get("active") or 0)
    base_queued = int(baseline.get("queued") or 0)
    base_waiting = int(baseline.get("capacity_waiting") or 0)
    base_cancelled = int(baseline.get("cancelled_requests") or 0)

    def stream_body(prompt):
        return {"model": args.model, "prompt": prompt["text"],
                "max_tokens": args.gen, "ignore_eos": True, "stream": True}

    active = DisconnectingStream(
        base, "/completions", stream_body(prompts["active"]),
        args.timeout, "active")
    pending = DisconnectingStream(
        base, "/completions", stream_body(prompts["pending"]),
        args.timeout, "pending")
    errors = []
    stages = []
    active.start()
    if not active.opened.wait(min(args.timeout, 30.0)):
        errors.append("active streaming request did not open")
    try:
        stages.append({"active_observed": monitor.wait_for(
            lambda health: int(health.get("active") or 0) > base_active,
            args.timeout, "active request")})
    except Exception as error:
        errors.append(str(error))

    pending.start()
    if not pending.opened.wait(min(args.timeout, 30.0)):
        errors.append("pending streaming request did not open")
    pending_observed = None
    try:
        pending_observed = monitor.wait_for(
            lambda health: (int(health.get("queued") or 0) > base_queued or
                            int(health.get("capacity_waiting") or 0) > base_waiting),
            args.timeout, "pending/capacity-waiting request")
        stages.append({"pending_observed": pending_observed})
    except Exception as error:
        errors.append(str(error))

    short_result = {}
    short_spec = {"sequence": "short-after-cancel", "prompt": prompts["short"]}
    short_barrier = threading.Barrier(2)
    short_thread = threading.Thread(
        target=completion_worker,
        args=(base, args.model, short_spec, args.gen, args.timeout,
              short_barrier, short_result),
        name="short-after-cancel", daemon=True)
    short_thread.start()
    short_barrier.wait(timeout=args.timeout)
    pending.cancel()
    active.cancel()
    join_deadline = time.perf_counter() + args.timeout
    for thread in (active.thread, pending.thread, short_thread):
        thread.join(timeout=max(0.0, join_deadline - time.perf_counter()))
    if active.thread.is_alive() or pending.thread.is_alive():
        errors.append("disconnected stream handler did not terminate before --timeout")
    if short_thread.is_alive():
        errors.append("short request did not complete after cancellations")
    errors.extend(short_result.get("errors") or [])

    recovery = None
    try:
        recovery = monitor.wait_for(
            lambda health: (
                int(health.get("active") or 0) == base_active and
                int(health.get("queued") or 0) == base_queued and
                int(health.get("capacity_waiting") or 0) == base_waiting and
                int(health.get("cancelled_requests") or 0) >= base_cancelled + 2),
            args.timeout, "two cancellations and scheduler recovery")
        stages.append({"recovery_observed": recovery})
    except Exception as error:
        errors.append(str(error))
    monitor.stop()
    summary = monitor.summary()
    end = summary.get("end") or {}

    conservation = {}
    for key in ("kv_active_blocks", "kv_remaining_reserved_blocks",
                "kv_pool_blocks_free"):
        before = baseline.get(key)
        after = end.get(key)
        equal = before == after if before is not None else None
        conservation[key] = {"before": before, "after": after, "returned": equal}
        if equal is False:
            errors.append(f"capacity did not return: {key} {before} -> {after}")
    if initial_health.get("paged") and pending_observed is not None:
        if not (int(pending_observed.get("capacity_waiting") or 0) > base_waiting or
                int(pending_observed.get("queued") or 0) > base_queued):
            errors.append("pending request was never observed waiting")
    cancelled_delta = ((end.get("cancelled_requests") or 0) - base_cancelled)
    if cancelled_delta < 2:
        errors.append(
            f"cancelled_requests increased by {cancelled_delta}, expected at least 2")
    return {
        "label": "cancel-active-pending-capacity-return",
        "exact_prompt_tokens": {
            name: prompt["prompt_tokens"] for name, prompt in prompts.items()},
        "requested_output_tokens": args.gen,
        "baseline": baseline,
        "stages": stages,
        "connections": [active.record(zero), pending.record(zero)],
        "short_request": short_result,
        "health_observation": summary,
        "capacity_return": conservation,
        "cancelled_requests_delta": cancelled_delta,
        "errors": errors,
    }


def parse_mixed(value):
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--mixed needs exactly two comma-separated token counts") from error
    if len(parts) != 2 or any(part < 2 for part in parts):
        raise argparse.ArgumentTypeError(
            "--mixed needs exactly two token counts >= 2")
    return parts


def write_report(path, report):
    expanded = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(expanded)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = expanded + f".tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, expanded)
    return expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--conc", type=int, default=8)
    parser.add_argument("--gen", type=int, default=32)
    parser.add_argument("--gen128", action="store_true",
                        help="set --gen 128 for retained plan-8.3 cells")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="HTTP and phase timeout in seconds")
    parser.add_argument("--expect-layout", choices=("flat", "paged"))
    parser.add_argument("--expect-apc", choices=("on", "off"))
    parser.add_argument("--mixed", type=parse_mixed, metavar="TOKENS,TOKENS",
                        help="run exact-token concurrent /completions benchmark")
    parser.add_argument("--model-dir",
                        help="real tokenizer directory (required by exact modes)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="measured mixed repeats; launch order reverses")
    parser.add_argument("--benchmark-phase", choices=("auto", "cold", "warm"),
                        default="auto",
                        help="cold requires APC off; warm requires APC on; auto attests health")
    parser.add_argument("--output",
                        help="write complete JSON qualification artifact")
    parser.add_argument("--cancel-scenario", nargs="?", const="512,512,32",
                        type=parse_cancel_scenario,
                        metavar="ACTIVE,PENDING,SHORT",
                        help="disconnect active+pending streams, then attest recovery")
    args = parser.parse_args()
    if args.gen128:
        args.gen = 128
    if args.conc <= 0 or args.gen <= 0 or args.timeout <= 0 or args.repeats <= 0:
        parser.error("--conc, --gen, --timeout and --repeats must be positive")
    if (args.mixed or args.cancel_scenario) and not args.model_dir:
        parser.error("--mixed/--cancel-scenario require --model-dir")
    if args.mixed and not args.output:
        parser.error("--mixed requires --output so the measurement is retained")
    if args.mixed and args.cancel_scenario:
        parser.error("--mixed and --cancel-scenario are separate workload modes")

    base = args.base_url.rstrip("/")
    started = time.perf_counter()
    failures = []
    report = {
        "schema_version": 3,
        "tool": "serve_smoke_xpu.py",
        "started_at": utc_now(),
        "invocation": sys.argv,
        "arguments": vars(args),
        "field_dependencies": health_field_dependencies(),
        "command_matrix": {
            "basic_flat": "--expect-layout flat --model MODEL --conc SLOTS --gen 128",
            "basic_paged": "--expect-layout paged --model MODEL --conc SLOTS --gen 128",
            "mixed_cold_apc_off": "--expect-layout paged --expect-apc off --benchmark-phase cold --mixed A,B --model-dir DIR --gen128 --timeout 14400 --repeats N --output FILE",
            "mixed_warm_apc_on": "--expect-layout paged --expect-apc on --benchmark-phase warm --mixed A,B --model-dir DIR --gen128 --timeout 14400 --repeats N --output FILE",
            "mixed_45k_45k_pool_131072": "--expect-layout paged --mixed 45000,45000 --model-dir DIR --gen128 --timeout 14400 --repeats N --output FILE",
            "mixed_28k_75k_pool_131072": "--expect-layout paged --mixed 28000,75000 --model-dir DIR --gen128 --timeout 14400 --repeats N --output FILE",
            "mixed_65k_75k_pool_131072_or_196608": "--expect-layout paged --mixed 65000,75000 --model-dir DIR --gen128 --timeout 14400 --repeats N --output FILE",
            "cancel_small": "--expect-layout paged --cancel-scenario ACTIVE,PENDING,SHORT --model-dir DIR --gen128 --timeout SEC --output FILE",
        },
    }

    try:
        initial_health = get_json(base, "/health", args.timeout)
        report["service_initial"] = initial_health
        if initial_health.get("status") != "ok":
            raise QualificationError(f"unhealthy service: {initial_health}")
        missing_health = [key for key in HEALTH_PLAN_FIELDS
                          if key not in initial_health]
        report["health_fields_missing"] = missing_health
        health_issues = [f"missing {key}" for key in missing_health]
        health_issues.extend(effective_metadata_issues(initial_health))
        report["health_metadata_issues"] = health_issues
        if health_issues:
            raise QualificationError(
                "health metadata contract failed: " + "; ".join(health_issues))
        if args.expect_layout and initial_health.get("kv_layout") != args.expect_layout:
            raise QualificationError(
                f"--expect-layout {args.expect_layout}, health attested "
                f"{initial_health.get('kv_layout')!r}")
        if args.expect_apc is not None:
            expected = args.expect_apc == "on"
            if initial_health.get("apc_enabled") is not expected:
                raise QualificationError(
                    f"--expect-apc {args.expect_apc}, health attested "
                    f"{initial_health.get('apc_enabled')!r}")
        print(f"health                   ok  layout={initial_health.get('kv_layout')} "
              f"slots={initial_health.get('slots')} "
              f"apc={'on' if initial_health.get('apc_enabled') else 'off'}")

        factory = ExactPromptFactory(args.model_dir) if args.model_dir else None
        if factory is not None:
            report["tokenizer"] = {
                "model_dir": factory.model_dir,
                "calibration_token_ids": factory.calibration,
            }
        report["surface_checks"] = run_surface_checks(
            args, base, initial_health, factory, failures)

        if not args.mixed and not args.cancel_scenario:
            before_waves = get_json(base, "/health", args.timeout)
            cold = smoke_wave(base, args.model, args.conc, args.gen, args.timeout,
                              "wave 1 (cold APC observation)")
            warm = smoke_wave(base, args.model, args.conc, args.gen, args.timeout,
                              "wave 2 (warm APC observation)")
            after_waves = get_json(base, "/health", args.timeout)
            report["smoke"] = {
                "health_before": before_waves,
                "cold_phase": cold,
                "warm_phase": warm,
                "health_after": after_waves,
            }
            failures.extend(row.get("error") or "smoke wave request failed"
                            for wave in (cold, warm) for row in wave["bad"])
            batch_values = [row.get("batch_avg") for row in cold["ok"]
                            if isinstance(row.get("batch_avg"), (int, float))]
            if cold["ok"] and (not batch_values or max(batch_values) <= 1.0):
                failures.append("concurrency: batch_avg never exceeded 1")
                print("\nconcurrency             FAIL  batch_avg never exceeded 1")
            else:
                print("\nconcurrency             ok    batched decode observed")
            warm_reused = sum(
                row.get("reused_tokens") or 0 for row in warm["ok"])
            if initial_health.get("apc_enabled"):
                if warm_reused <= 0:
                    failures.append(
                        "APC enabled but warm smoke wave reused zero tokens")
                    print("apc                     FAIL  enabled but no warm reuse")
                else:
                    print(f"apc                     ok    warm reuse={warm_reused}")
            elif warm_reused:
                failures.append("APC disabled but response reported reused tokens")
                print(f"apc                     FAIL  disabled but reuse={warm_reused}")
            else:
                print("apc                     ok    disabled and zero reuse")
        else:
            selected = "mixed" if args.mixed else "cancel-scenario"
            report["smoke"] = {
                "status": "skipped",
                "reason": (
                    f"{selected} is a selected workload mode; fixed short "
                    "concurrency/APC waves are qualified by a separate basic "
                    "smoke invocation on a suitable pool/cache configuration"),
            }
            print(f"short smoke waves        SKIP  selected {selected} workload")

        if args.mixed:
            mixed_health = get_json(base, "/health", args.timeout)
            mixed = run_mixed(args, base, factory, mixed_health)
            report["mixed"] = mixed
            failures.extend("mixed: " + error for error in mixed["errors"])
            print(f"mixed                   "
                  f"{'ok' if not mixed['errors'] else 'FAIL'}  "
                  f"phase={mixed['benchmark_phase']} lengths={args.mixed}")
        if args.cancel_scenario:
            cancellation = run_cancel_scenario(
                args, base, factory, initial_health)
            report["cancellation"] = cancellation
            failures.extend("cancellation: " + error
                            for error in cancellation["errors"])
            print(f"cancellation            "
                  f"{'ok' if not cancellation['errors'] else 'FAIL'}  "
                  f"cancelled_delta={cancellation['cancelled_requests_delta']}")
    except Exception as error:
        failures.append(str(error))
        report["fatal_error"] = str(error)
        print(f"FATAL                    {error}")

    report["finished_at"] = utc_now()
    report["total_wall_seconds"] = time.perf_counter() - started
    report["failures"] = failures
    report["status"] = "pass" if not failures else "fail"
    if args.output:
        try:
            saved = write_report(args.output, report)
            print(f"artifact                 {saved}")
        except Exception as error:
            failures.append(f"write artifact: {error}")
            report["failures"] = failures
            report["status"] = "fail"
            print(f"artifact                 FAIL  {error}")
    print("\nOVERALL:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
