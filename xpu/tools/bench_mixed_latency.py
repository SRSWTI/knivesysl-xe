#!/usr/bin/env python3
"""Matched lone-stream / incoming-prefill latency benchmark for an existing XE server.

Example (run unchanged against baseline and candidate; change only URL/output):
  python3 xpu/tools/bench_mixed_latency.py --base-url http://127.0.0.1:8000/v1 \
    --model-dir /path/to/model --decode-prompt-tokens 64 --decode-gen-tokens 64 \
    --incoming-prompt-tokens 1024 --incoming-gen-tokens 8 --trigger-event 8 \
    --seed 20260908 --repeats 3 --output mixed-baseline.json

Never starts or configures a server. APC must already be disabled. Requires the
real local transformers tokenizer. Uses raw completions, not chat templates.
Triggers and client-side gaps use nonempty SSE output-event indices, not native
token indices: the server may buffer UTF-8 or omit empty text. Native token counts
and server-reported ITL are retained separately. No device overlap is inferred.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.parse


class BenchmarkError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise BenchmarkError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def distribution(values):
    ordered = sorted(values)

    def quantile(fraction):
        if not ordered:
            return None
        rank = (len(ordered) - 1) * fraction
        low, high = math.floor(rank), math.ceil(rank)
        return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)

    return {"count": len(ordered), "no_samples": not ordered,
            "min": ordered[0] if ordered else None,
            "mean": sum(ordered) / len(ordered) if ordered else None,
            "p50": quantile(0.5), "p90": quantile(0.9),
            "p95": quantile(0.95), "p99": quantile(0.99),
            "max": ordered[-1] if ordered else None}


def health_delta(before, after):
    result = {}
    for key in before.keys() | after.keys():
        left, right = before.get(key), after.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            result[key] = health_delta(left, right)
        elif type(right) in (int, float) and type(left) in (int, float):
            result[key] = right - left
        elif key not in before and type(right) in (int, float):
            result[key] = right
        elif key not in after and type(left) in (int, float):
            result[key] = -left
    return result


class Prompts:
    WORDS = (" alpha", " beta", " gamma")

    def __init__(self, model_dir, seed):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.expanduser(model_dir), local_files_only=True, trust_remote_code=True)
        self.seed = seed
        self.calibration = {}
        for word in self.WORDS:
            ids = self.tokenizer(word, add_special_tokens=False).input_ids
            require(len(ids) == 1, f"calibration fragment {word!r} is not one token")
            self.calibration[word] = int(ids[0])

    def make(self, tokens, role, repeat):
        # A hash-defined sequence is independent of invocation order, warmup and
        # Python PRNG implementation. Verify the complete string, not fragments.
        variant = hashlib.sha256(f"xe-mixed-v1:{self.seed}:{role}:{repeat}".encode()).digest()
        prefix = "".join(self.WORDS[value % 3] for value in variant[:min(tokens, 32)])
        text = prefix + self.WORDS[variant[-1] % 3] * max(0, tokens - 32)
        ids = list(self.tokenizer(text, add_special_tokens=False).input_ids)
        require(len(ids) == tokens, f"{role}: tokenizer measured {len(ids)}, expected {tokens}")
        return {"text": text, "token_ids": ids, "tokens": tokens,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "role": role, "repeat": repeat}


class Client:
    def __init__(self, args):
        parsed = urllib.parse.urlsplit(args.base_url.rstrip("/"))
        require(parsed.scheme in ("http", "https") and parsed.hostname and
                not parsed.username and not parsed.password and not parsed.query and
                not parsed.fragment and parsed.path in ("", "/v1"),
                "--base-url must be an HTTP(S) origin, optionally ending in /v1")
        self.host, self.port = parsed.hostname, parsed.port
        self.connection_class = (http.client.HTTPSConnection if parsed.scheme == "https"
                                 else http.client.HTTPConnection)
        self.timeout = args.timeout
        self.zero = time.perf_counter()
        self.deadline = self.zero + args.deadline
        self.model = args.model
        self.health_samples = []

    def now(self):
        return time.perf_counter() - self.zero

    def remaining(self, deadline=None):
        remaining = min(self.timeout, (deadline or self.deadline) - time.perf_counter(),
                        self.deadline - time.perf_counter())
        require(remaining > 0, "client deadline exceeded")
        return remaining

    def connection(self):
        return self.connection_class(self.host, self.port, timeout=self.remaining())

    def json_request(self, path):
        connection = self.connection()
        try:
            connection.request("GET", path, headers={"Accept": "application/json"})
            response = connection.getresponse()
            payload = json.loads(response.read())
            require(response.status == 200, f"{path}: HTTP {response.status}: {payload}")
            require(isinstance(payload, dict) and "error" not in payload,
                    f"{path}: unexpected payload: {payload}")
            return payload
        finally:
            connection.close()

    def health(self):
        started = self.now()
        health = self.json_request("/health")
        self.health_samples.append({"request_submit_s": started,
                                    "response_s": self.now(), "health": health})
        require(health.get("status") == "ok" and not health.get("engine_error"),
                f"unhealthy server: {health}")
        require(health.get("apc_enabled") is False,
                "health must explicitly report apc_enabled=false; disable APC on server")
        require(not health.get("quarantined_slots"), "server has quarantined slots")
        return health

    def settle(self):
        deadline = time.perf_counter() + self.remaining()
        while True:
            health = self.health()
            require(type(health.get("active")) is int and type(health.get("queued")) is int,
                    "health lacks active/queued integer counts")
            if (health["active"] == health["queued"] == 0 and
                    health.get("capacity_waiting", 0) == 0 and
                    (not health.get("paged") or
                     (health.get("kv_active_blocks") == 0 and
                      health.get("kv_remaining_reserved_blocks") == 0))):
                return health
            # Polling is only for idle isolation, never for workload arrival.
            time.sleep(min(0.05, self.remaining(deadline)))

    def stream(self, row, prompt, count, on_output=None, on_submit=None):
        connection = None
        response = None
        timer = None
        row.update({"prompt_sha256": prompt["sha256"], "prompt_tokens": prompt["tokens"],
                    "requested_gen_tokens": count, "events": [], "output_event_times_s": [],
                    "finish_reasons": [], "usage": None, "metadata": None,
                    "done": False, "ok": False, "errors": []})
        try:
            body = {"model": self.model, "prompt": prompt["text"], "max_tokens": count,
                    "stream": True, "ignore_eos": True, "temperature": 0,
                    "stream_options": {"include_usage": True}}
            encoded = json.dumps(body).encode()
            row["request_options"] = {key: value for key, value in body.items() if key != "prompt"}
            connection = self.connection()
            deadline = time.perf_counter() + self.remaining()
            row["submit_s"] = self.now()
            if on_submit:
                on_submit(row["submit_s"])

            def abort():
                sock = connection.sock
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    sock.close()

            # Deadline watchdog bounds even a server that trickles partial SSE lines.
            timer = threading.Timer(self.remaining(deadline), abort)
            timer.daemon = True
            timer.start()
            connection.request("POST", "/v1/completions", encoded,
                               {"Content-Type": "application/json", "Accept": "text/event-stream"})
            row["request_sent_s"] = self.now()
            response = connection.getresponse()
            row["headers_s"] = self.now()
            row["http_status"] = response.status
            row["response_headers"] = dict(response.getheaders())
            if response.status != 200:
                raise BenchmarkError(
                    f"HTTP {response.status}: {response.read(2048).decode(errors='replace')}")
            require("text/event-stream" in response.getheader("Content-Type", ""),
                    "response is not text/event-stream")
            data_lines = []
            while True:
                self.remaining(deadline)
                line = response.readline(8 * 1024 * 1024 + 1)
                observed = self.now()
                require(len(line) <= 8 * 1024 * 1024, "SSE line exceeds 8 MiB")
                require(line, "SSE ended without [DONE]")
                line = line.rstrip(b"\r\n")
                if line:
                    if line.startswith(b"data:"):
                        data_lines.append(line[5:].lstrip(b" "))
                    elif not line.startswith((b":", b"event:", b"id:", b"retry:")):
                        raise BenchmarkError(f"unexpected SSE field: {line[:200]!r}")
                    continue
                if not data_lines:
                    continue
                data = b"\n".join(data_lines)
                data_lines.clear()
                record = {"event_index": len(row["events"]) + 1, "observed_s": observed}
                row["events"].append(record)
                if data == b"[DONE]":
                    record["data"] = "[DONE]"
                    row["done"] = True
                    row["done_s"] = observed
                    break
                record["raw_data"] = data.decode("utf-8")
                event = json.loads(data)
                record["data"] = event
                require(isinstance(event, dict) and "error" not in event,
                        f"stream error: {event}")
                if event.get("usage") is not None:
                    require(row["usage"] is None, "duplicate usage event")
                    row["usage"] = event["usage"]
                    row["metadata"] = event.get("x_knivesysl")
                choices = event.get("choices", [])
                require(isinstance(choices, list) and len(choices) <= 1,
                        "expected at most one raw completion choice")
                for choice in choices:
                    require(choice.get("index") == 0 and "delta" not in choice,
                            "expected raw completion choice zero, not chat deltas")
                    text = choice.get("text", "")
                    require(isinstance(text, str), "completion text is not a string")
                    finish = choice.get("finish_reason")
                    if finish is not None:
                        row["finish_reasons"].append(finish)
                    if text:
                        require(finish is None and not row["finish_reasons"],
                                "output text attached to or following a finish event")
                        row["output_event_times_s"].append(observed)
                        record["output_event_index"] = len(row["output_event_times_s"])
                        if on_output:
                            on_output(record["output_event_index"], observed)
            usage, metadata = row["usage"], row["metadata"]
            require(isinstance(usage, dict) and isinstance(metadata, dict),
                    "missing final usage or x_knivesysl metadata")
            require(usage.get("prompt_tokens") == prompt["tokens"] and
                    usage.get("completion_tokens") == count and
                    usage.get("total_tokens") == prompt["tokens"] + count,
                    f"exact token usage mismatch: {usage}")
            require(row["finish_reasons"] == ["length"],
                    f"expected one length finish: {row['finish_reasons']}")
            require(metadata.get("generated_tokens") == count and
                    metadata.get("reused_tokens") == 0 and
                    metadata.get("prefilled_tokens") == prompt["tokens"] and
                    metadata.get("apc_enabled") is False,
                    f"generation/APC/prefill metadata mismatch: {metadata}")
            require(row["output_event_times_s"], "stream emitted no observable output text")
            row["server_latency_ms"] = {name: metadata.get(name) for name in
                                        ("ttft_ms", "itl_ms_p50", "itl_ms_p99")}
            row["ok"] = True
        except Exception as error:
            row["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            row["end_s"] = self.now()
            if timer is not None:
                timer.cancel()
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()
            times = row["output_event_times_s"]
            row["observed_output_events"] = len(times)
            generated = (row.get("usage") or {}).get("completion_tokens")
            row["output_events_equal_generated_tokens"] = (
                len(times) == generated if generated is not None else None)
            row["first_output_s"] = times[0] if times else None
            row["last_output_s"] = times[-1] if times else None
            row["ttft_ms"] = ((times[0] - row["submit_s"]) * 1000
                              if times and "submit_s" in row else None)
            row["event_gaps_ms"] = [(right - left) * 1000 for left, right in zip(times, times[1:])]
            row["event_gap_distribution_ms"] = distribution(row["event_gaps_ms"])
            wall = row["end_s"] - row["submit_s"] if "submit_s" in row else None
            row["wall_s"] = wall
            row["output_tokens_per_s"] = count / wall if row["ok"] and wall else None


def phase_metrics(decode, submit, first):
    times = decode.get("output_event_times_s", [])
    buckets = {name: [] for name in ("before", "during", "after", "crossing", "unclassified")}
    intervals = []
    complete = submit is not None and first is not None
    for index, (left, right) in enumerate(zip(times, times[1:]), 2):
        if not complete:
            phase = "before" if submit is not None and right <= submit else "unclassified"
        elif right <= submit:
            phase = "before"
        elif left >= first:
            phase = "after"
        elif left >= submit and right <= first:
            phase = "during"
        else:
            phase = "crossing"
        duration = (right - left) * 1000
        buckets[phase].append(duration)
        overlaps = None
        if complete:
            overlaps = {"before": max(0.0, min(right, submit) - left) * 1000,
                        "during": max(0.0, min(right, first) - max(left, submit)) * 1000,
                        "after": max(0.0, right - max(left, first)) * 1000}
        intervals.append({"ending_output_event_index": index, "start_s": left, "end_s": right,
                          "duration_ms": duration, "phase": phase, "overlap_ms": overlaps})
    return {"valid_event_metrics": decode.get("ok", False),
            "boundaries_available": complete, "incoming_submit_s": submit,
            "incoming_first_output_s": first,
            "incoming_prefill_proxy_ms": (first - submit) * 1000 if complete else None,
            "decode_observation_span_overlaps_prefill_proxy":
                (max(times[0], submit) < min(times[-1], first)) if complete and times else None,
            "intervals": intervals,
            "phases": {name: {"interval_count": len(values), "no_intervals": not values,
                               "event_gap_distribution_ms": distribution(values)}
                       for name, values in buckets.items()}}


def check_health_delta(before, after):
    change = health_delta(before, after)
    for key in ("request_errors", "prefill_errors", "cancelled_requests", "apc_hits"):
        require(change.get(key) == 0, f"health {key} changed or missing: {change.get(key)}")
    return change


def run_trial(client, args, trial, prompt, incoming):
    trial.update({"ok": False, "errors": [], "requests": {"decode": {}},
                  "trigger": {"requested_output_event_index": args.trigger_event}})
    worker = None
    gate = threading.Event()
    decode = trial["requests"]["decode"]
    trigger = trial["trigger"]
    try:
        trial["health_before"] = client.settle()
        mixed = trial["kind"] == "mixed"
        if mixed:
            arrival = trial["requests"]["incoming"] = {}

            def incoming_worker():
                try:
                    require(gate.wait(client.remaining()), "incoming trigger wait timed out")
                    if "observed_s" not in trigger:
                        arrival.update({"ok": False, "errors": ["decode ended before trigger"],
                                        "skipped": True})
                        return

                    def submitted(timestamp):
                        trigger["incoming_submit_s"] = timestamp
                        trigger["submit_lag_ms"] = (timestamp - trigger["observed_s"]) * 1000
                        trigger["decode_output_events_at_submit"] = len(
                            decode["output_event_times_s"])

                    client.stream(arrival, incoming, args.incoming_gen_tokens,
                                  on_submit=submitted)
                except Exception as error:
                    arrival.update({"ok": False, "errors": [f"{type(error).__name__}: {error}"]})

            worker = threading.Thread(target=incoming_worker, name="incoming-stream", daemon=True)
            worker.start()

        def observed(index, timestamp):
            if index == args.trigger_event:
                trigger["observed_s"] = timestamp
                trigger["observed_output_event_index"] = index
                if mixed:
                    gate.set()

        client.stream(decode, prompt, args.decode_gen_tokens, on_output=observed)
        gate.set()
        if worker is not None:
            worker.join(timeout=client.remaining() + 1)
            require(not worker.is_alive(), "incoming worker did not finish before deadline")
        trial["health_after"] = client.settle()
        trial["health_delta"] = health_delta(trial["health_before"], trial["health_after"])
        check_health_delta(trial["health_before"], trial["health_after"])
        require("observed_s" in trigger, "decode did not reach trigger index")
        for name, row in trial["requests"].items():
            require(row.get("ok"), f"{name}: {row.get('errors')}")
        trial["ok"] = True
    except Exception as error:
        trial["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        gate.set()
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(0.0, min(args.timeout + 1,
                                           client.deadline - time.perf_counter() + 1)))
        arrival = trial["requests"].get("incoming", {})
        if trial["kind"] == "mixed":
            trial["decode_phases"] = phase_metrics(decode, arrival.get("submit_s"),
                                                    arrival.get("first_output_s"))
        rows = list(trial["requests"].values())
        starts = [row["submit_s"] for row in rows if "submit_s" in row]
        ends = [row["end_s"] for row in rows if "end_s" in row]
        wall = max(ends) - min(starts) if starts and ends else None
        trial["wall_s"] = wall
        trial["output_tokens_per_s"] = (sum(row["usage"]["completion_tokens"] for row in rows)
                                         / wall if trial["ok"] and wall else None)
        trigger["observed_event_index_verified"] = (
            trigger.get("observed_output_event_index") == args.trigger_event)
        if trial["kind"] == "mixed":
            sent, lead_done = arrival.get("request_sent_s"), decode.get("done_s")
            first, submitted = arrival.get("first_output_s"), arrival.get("submit_s")
            observed = trigger.get("observed_s")
            during = [stamp for stamp in decode.get("output_event_times_s", [])
                      if submitted is not None and first is not None and submitted <= stamp < first]
            conditions = {
                "incoming_request_sent_before_decode_done":
                    sent is not None and lead_done is not None and sent < lead_done,
                "incoming_first_output_after_decode_trigger":
                    first is not None and observed is not None and first > observed,
                "decode_output_events_observed_during_prefill_proxy": bool(during),
            }
            trial["overlap_attestation"] = {
                **conditions, "attested": all(conditions.values()),
                "decode_event_count_during_prefill_proxy": len(during),
                "decode_event_times_during_prefill_proxy_s": during,
            }
            if trial["ok"] and not all(conditions.values()):
                trial["ok"] = False
                trial["errors"].append("mixed overlap not attested; inspect overlap_attestation")
                trial["output_tokens_per_s"] = None


def summarize(pairs):
    summary = {}
    for kind in ("control", "mixed"):
        trials = [trial for pair in pairs if pair.get("ok") for trial in pair["trials"]
                  if trial["kind"] == kind]
        roles = {}
        for role in ("decode", "incoming"):
            rows = [trial["requests"][role] for trial in trials if role in trial["requests"]]
            roles[role] = {"request_count": len(rows),
                           "ttft_distribution_ms": distribution([row["ttft_ms"] for row in rows]),
                           "pooled_event_gap_distribution_ms": distribution(
                               [value for row in rows for value in row["event_gaps_ms"]]),
                           "wall_distribution_s": distribution([row["wall_s"] for row in rows]),
                           "output_tokens_per_s_distribution": distribution(
                               [row["output_tokens_per_s"] for row in rows])}
            roles[role]["server_reported_latency_distributions_ms"] = {
                metric: distribution([row["server_latency_ms"][metric] for row in rows
                                      if type(row["server_latency_ms"].get(metric)) in (int, float)])
                for metric in ("ttft_ms", "itl_ms_p50", "itl_ms_p99")}
        summary[kind] = {"matched_trial_count": len(trials), "requests": roles,
                         "wall_distribution_s": distribution([trial["wall_s"] for trial in trials]),
                         "output_tokens_per_s_distribution": distribution(
                             [trial["output_tokens_per_s"] for trial in trials])}
        if kind == "mixed":
            summary[kind]["decode_phase_pooled_event_gaps_ms"] = {
                phase: distribution([interval["duration_ms"] for trial in trials
                                     for interval in trial["decode_phases"]["intervals"]
                                     if interval["phase"] == phase])
                for phase in ("before", "during", "after", "crossing", "unclassified")}
    return summary


DEFINITIONS = {
    "clock": "All *_s event timestamps are client perf_counter seconds relative to clock_origin; UTC is correlation only.",
    "submit": "Client timestamp immediately before HTTP POST (includes connection setup, upload, queueing); not server admission.",
    "observation": "Timestamp when the SSE blank-line delimiter is read, before JSON parsing. Full SSE data events including finish, usage and DONE are retained.",
    "output_event_times_s": "Nonempty raw-completion text event observations, not native token timestamps. UTF-8 buffering and empty token text can make event count differ from generated-token usage. Empty/finish/usage/DONE events remain in events but do not enter gap distributions. Event-count equality is reported descriptively, not claimed as proof of native token timings.",
    "trigger": "Incoming worker is released at the one-based Nth nonempty decode output event, without sleeps. Actual submit timestamp, thread wakeup lag and decode-event count at submit are retained. This is an event-index trigger, never an exact native-token-index claim.",
    "ttft_ms": "1000 * (first nonempty output event observation - request submit); includes HTTP, queueing, prefill and buffering, not pure device prefill.",
    "event_gaps_ms": "1000 * (consecutive nonempty output event observation difference), excluding TTFT; exactly observed_output_events - 1 samples. Client event gaps are not native ITL. server_latency_ms reports the server's native ttft_ms, itl_ms_p50 and itl_ms_p99 separately without reconstructing unavailable token times.",
    "prefill_proxy": "Incoming submit through its first output observation, not instrumented server prefill. Includes upload, admission, queueing, first decode and network. No actual GPU overlap is inferred.",
    "phase_assignment": "Whole decode event intervals (left,right] are before if right <= incoming submit, after if left >= incoming first output, during if left >= submit and right <= first output; remaining boundary-crossing intervals go ONLY to crossing. Before/during/after distributions exclude crossings to avoid smearing a stall. Crossings retain exact per-window overlap durations, not fabricated split-event samples. Missing boundaries yield unclassified (or wholly-before when submit is known).",
    "control": "Lone decode uses identical prompt, generation length and trigger marker. It has no incoming-prefill windows; its entire event-gap distribution is the control, not fabricated during/after phases.",
    "no_intervals": "Zero phase interval count and null quantiles mean not observed, never zero latency. A crossing-only window may have real time overlap but no wholly-contained event-gap samples.",
    "distribution": "Count, min, arithmetic mean, max and linearly interpolated percentiles at rank (n-1)*q. Empty samples give null statistics. Summary pools individual intervals, not per-trial percentiles; only fully valid matched pairs contribute.",
    "wall_s": "Request: submit through stream end after DONE parsing (before connection teardown). Trial: earliest request submit to latest stream end. Excludes health probes, idle settling and warmup; includes prompt processing and both requests for mixed trials.",
    "output_tokens_per_s": "Validated completion-token count / end-to-end wall_s; trial numerator sums both streams. Not steady-state decode throughput.",
    "health_delta": "Recursive numeric after-minus-before snapshots; booleans excluded; newly added numeric histogram bins start at zero. Includes gauges/rates/lifetime peaks, whose differences are not event counts. No health polling while streams run. Before/after idle checks do not exclude unrelated external traffic during a trial.",
    "overlap_attestation": "A qualifying mixed trial requires incoming request_sent_s < decode DONE observation, incoming first output > trigger observation, and at least one decode output event in [incoming submit, incoming first output). request_sent_s is after HTTPConnection.request returns (all request bytes handed to the local socket); this attests client-observed concurrency, not server-admission or GPU execution overlap. Non-overlap is explicitly reported as a failed trial, not silently assumed or retried.",
    "paired_delta": "Within-repeat mixed minus control for decode TTFT, wall, mean event gap and output throughput. Trials alternate control/mixed and mixed/control by repeat index, deterministically.",
    "workload_identity": "SHA256 of exact prompts/token IDs, seed, generation sizes, trigger, repeat order and warmup prompts. Excludes URL, model-dir, output path and label; compare workload_sha256 across baseline/candidate. Server/model equivalence still requires inspecting health/model/tokenizer metadata.",
}


def parser():
    result = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="existing XE HTTP(S) service")
    result.add_argument("--model-dir", required=True, help="local directory containing the server tokenizer")
    result.add_argument("--model", help="served model ID; default requires exactly one /v1/models entry")
    result.add_argument("--decode-prompt-tokens", type=int, default=64)
    result.add_argument("--decode-gen-tokens", type=int, default=64)
    result.add_argument("--incoming-prompt-tokens", type=int, default=1024)
    result.add_argument("--incoming-gen-tokens", type=int, default=8)
    result.add_argument("--trigger-event", type=int, default=8,
                        help="one-based nonempty decode SSE output-event index, not native token index")
    result.add_argument("--seed", type=int, default=20260908, help="prompt recipe seed, not server sampling seed")
    result.add_argument("--repeats", type=int, default=3, help="matched pairs, alternating control-first/mixed-first")
    result.add_argument("--warmup", type=int, default=1, help="0..4 sequential 32-prompt/8-generation warmups")
    result.add_argument("--timeout", type=float, default=60, help="per-stream/settle timeout seconds")
    result.add_argument("--deadline", type=float, default=600, help="overall HTTP campaign deadline seconds")
    result.add_argument("--label", default="unlabelled", help="informational baseline/candidate label")
    result.add_argument("--output", required=True, help="JSON artifact, written on success or runtime failure")
    return result


def write_report(path, report):
    expanded = Path(path).expanduser()
    expanded.parent.mkdir(parents=True, exist_ok=True)
    temporary = expanded.with_name(expanded.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, expanded)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    args = parser().parse_args()
    report = {"schema": "xe-mixed-latency-v1", "ok": False, "args": vars(args),
              "command_argv": [sys.executable, *sys.argv], "metric_definitions": DEFINITIONS,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "errors": [], "warmup": [], "pairs": []}
    report["args"] = {key: (str(value) if isinstance(value, float) and not math.isfinite(value)
                            else value) for key, value in vars(args).items()}
    client = None
    try:
        for name in ("decode_prompt_tokens", "decode_gen_tokens", "incoming_prompt_tokens",
                     "incoming_gen_tokens", "repeats"):
            require(getattr(args, name) > 0, f"--{name.replace('_', '-')} must be positive")
        require(1 <= args.trigger_event < args.decode_gen_tokens,
                "trigger must be >= 1 and strictly less than decode generation length")
        require(0 <= args.warmup <= 4, "--warmup must be between 0 and 4")
        require(all(math.isfinite(value) and value > 0 for value in (args.timeout, args.deadline)),
                "timeouts must be finite and positive")
        client = Client(args)
        report["clock_origin"] = {"perf_counter_s": client.zero,
                                  "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        report["initial_health"] = client.settle()
        require(report["initial_health"].get("slots", 0) >= 2,
                "mixed-load benchmark requires at least two server slots")
        report["models"] = client.json_request("/v1/models")
        models = report["models"].get("data", [])
        if client.model is None:
            require(len(models) == 1 and isinstance(models[0].get("id"), str),
                    "specify --model when /v1/models does not contain exactly one model")
            client.model = models[0]["id"]
        require(any(item.get("id") == client.model for item in models),
                f"requested model {client.model!r} not advertised by /v1/models")
        report["model"] = client.model
        factory = Prompts(args.model_dir, args.seed)
        report["tokenizer"] = {"model_dir": os.path.abspath(os.path.expanduser(args.model_dir)),
                               "class": type(factory.tokenizer).__name__,
                               "calibration": factory.calibration, "local_files_only": True}
        workload = {"recipe": "sha256-alpha-beta-gamma-v1", "seed": args.seed,
                    "decode_gen_tokens": args.decode_gen_tokens,
                    "incoming_gen_tokens": args.incoming_gen_tokens, "trigger_event": args.trigger_event,
                    "warmup": [factory.make(32, "warmup", index) for index in range(args.warmup)],
                    "pairs": [{"repeat": index,
                               "order": ["control", "mixed"] if index % 2 == 0 else ["mixed", "control"],
                               "decode": factory.make(args.decode_prompt_tokens, "decode", index),
                               "incoming": factory.make(args.incoming_prompt_tokens, "incoming", index)}
                              for index in range(args.repeats)]}
        report["workload"] = workload
        report["workload_sha256"] = digest(workload)
        for prompt in workload["warmup"]:
            row = {}
            report["warmup"].append(row)
            before = client.settle()
            client.stream(row, prompt, 8)
            after = client.settle()
            row["health_before"], row["health_after"] = before, after
            row["health_delta"] = health_delta(before, after)
            check_health_delta(before, after)
            require(row["ok"], f"warmup failed: {row['errors']}")
        for spec in workload["pairs"]:
            pair = {"repeat": spec["repeat"], "order": spec["order"], "ok": False, "trials": []}
            report["pairs"].append(pair)
            for position, kind in enumerate(spec["order"]):
                trial = {"kind": kind, "order_position": position}
                pair["trials"].append(trial)
                run_trial(client, args, trial, spec["decode"], spec["incoming"])
                require(trial["ok"], f"repeat {spec['repeat']} {kind}: {trial['errors']}")
            matched = {trial["kind"]: trial["requests"]["decode"] for trial in pair["trials"]}
            pair["decode_mixed_minus_control"] = {
                key: matched["mixed"][key] - matched["control"][key]
                for key in ("ttft_ms", "wall_s", "output_tokens_per_s")}
            means = [matched[kind]["event_gap_distribution_ms"]["mean"]
                     for kind in ("control", "mixed")]
            pair["decode_mixed_minus_control"]["mean_event_gap_ms"] = (
                means[1] - means[0] if all(value is not None for value in means) else None)
            pair["ok"] = True
            write_report(args.output, report)
        report["final_health"] = client.settle()
        report["health_delta"] = health_delta(report["initial_health"], report["final_health"])
        check_health_delta(report["initial_health"], report["final_health"])
        report["ok"] = True
    except (Exception, KeyboardInterrupt) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if client is not None:
            report["health_samples"] = client.health_samples
            report["campaign_wall_s"] = client.now()
        report["summary"] = summarize(report["pairs"])
        report["completed_pairs"] = sum(pair.get("ok", False) for pair in report["pairs"])
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            write_report(args.output, report)
        except Exception as error:
            print(f"cannot write output artifact {args.output!r}: {error}", file=sys.stderr)
            print(json.dumps(report, ensure_ascii=False, allow_nan=False), file=sys.stderr)
            return 1
    print(f"{'PASS' if report['ok'] else 'FAIL'}: {args.output}; "
          f"{report['completed_pairs']}/{args.repeats} matched pairs")
    if report["errors"]:
        print("\n".join(report["errors"]), file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
