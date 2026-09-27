#!/usr/bin/env python3
"""Exact-token cold HTTP cohorts for native XE, vLLM-XPU and SGLang-XPU.

Servers must be isolated, with prefix caching and speculation disabled explicitly.
No native health schema is required. SSE gaps measure output events, NOT token ITL.
Run the same tokenizer, seed and workload flags for each endpoint; compare hashes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import http.client
import json
import math
import os
import socket
import sys
import threading
import time
import urllib.parse

from bench_mixed_latency import BenchmarkError, digest, distribution, require, write_report


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

    def make(self, tokens, identity):
        # Hash blocks define the entire nonconstant sequence, without PRNG-version
        # dependence. Re-tokenize the full text to validate boundary behavior.
        fragments = []
        for block in range((tokens + 31) // 32):
            values = hashlib.sha256(
                f"engine-compare-v1:{self.seed}:{identity}:{block}".encode()).digest()
            fragments.extend(self.WORDS[value % len(self.WORDS)] for value in values)
        text = "".join(fragments[:tokens])
        ids = list(self.tokenizer(text, add_special_tokens=False).input_ids)
        require(len(ids) == tokens, f"{identity}: measured {len(ids)}, expected {tokens}")
        return {"identity": identity, "text": text, "token_ids": ids,
                "tokens": tokens, "sha256": hashlib.sha256(text.encode()).hexdigest()}


class Client:
    def __init__(self, args):
        parsed = urllib.parse.urlsplit(args.base_url)
        require(parsed.scheme in ("http", "https") and parsed.hostname and
                not parsed.username and not parsed.password and not parsed.query and
                not parsed.fragment and parsed.path.rstrip("/") in ("", "/v1"),
                "--base-url must be an HTTP(S) origin, optionally ending in /v1")
        self.host, self.port = parsed.hostname, parsed.port
        self.connection_class = (http.client.HTTPSConnection if parsed.scheme == "https"
                                 else http.client.HTTPConnection)
        self.args = args
        self.zero = time.perf_counter()

    def now(self):
        return time.perf_counter() - self.zero

    def connection(self, timeout=None):
        return self.connection_class(self.host, self.port, timeout=timeout or self.args.timeout)

    def snapshot(self, path):
        connection = self.connection(min(5, self.args.timeout))
        result = {"path": path, "request_s": self.now()}
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            result["status"] = response.status
            result["body"] = response.read(1024 * 1024).decode(errors="replace")
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            result["response_s"] = self.now()
            connection.close()
        return result

    def stream(self, row, prompt, barrier):
        row.update({"prompt": prompt, "requested_gen_tokens": self.args.gen,
                    "ok": False, "errors": [], "events": [], "output_event_times_s": [],
                    "finish_reasons": [], "usage": None, "done": False})
        connection = response = timer = transport = None
        output = []
        try:
            body = {"model": self.args.model, "prompt": prompt["text"],
                    "temperature": 0, "max_tokens": self.args.gen,
                    "ignore_eos": True, "stream": True,
                    "stream_options": {"include_usage": True}}
            row["request_options"] = {key: value for key, value in body.items() if key != "prompt"}
            encoded = json.dumps(body).encode()
            connection = self.connection()
            row["barrier_ready_s"] = self.now()
            barrier.wait(timeout=self.args.timeout)
            row["submit_s"] = self.now()
            deadline = time.perf_counter() + self.args.timeout

            def abort():
                sock = transport if transport is not None else connection.sock
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    sock.close()

            timer = threading.Timer(self.args.timeout, abort)
            timer.daemon = True
            timer.start()
            connection.request("POST", "/v1/completions", encoded,
                               {"Content-Type": "application/json", "Accept": "text/event-stream"})
            row["request_sent_s"] = self.now()
            # getresponse() detaches connection.sock for Connection: close;
            # retain the actual transport (also the TLS wrapper) for shutdown.
            transport = connection.sock
            response = connection.getresponse()
            row["headers_s"] = self.now()
            row["http_status"] = response.status
            row["response_headers"] = dict(response.getheaders())
            if response.status != 200:
                raise BenchmarkError(
                    f"HTTP {response.status}: {response.read(8192).decode(errors='replace')}")
            require("text/event-stream" in response.getheader("Content-Type", ""),
                    "response is not text/event-stream")
            data_lines = []
            event_bytes = 0
            while True:
                require(time.perf_counter() < deadline, "stream deadline exceeded")
                line = response.readline(8 * 1024 * 1024 + 1)
                observed = self.now()
                require(len(line) <= 8 * 1024 * 1024, "SSE line exceeds 8 MiB")
                require(line, "SSE ended without [DONE]")
                line = line.rstrip(b"\r\n")
                if line:
                    if line.startswith(b"data:"):
                        data_lines.append(line[5:].lstrip(b" "))
                        event_bytes += len(line)
                        require(event_bytes <= 8 * 1024 * 1024, "SSE event exceeds 8 MiB")
                    elif not line.startswith((b":", b"event:", b"id:", b"retry:")):
                        raise BenchmarkError(f"unexpected SSE field: {line[:200]!r}")
                    continue
                if not data_lines:
                    continue
                data = b"\n".join(data_lines)
                data_lines.clear()
                event_bytes = 0
                record = {"observed_s": observed, "raw_data": data.decode("utf-8")}
                row["events"].append(record)
                if data == b"[DONE]":
                    row["done"] = True
                    row["done_s"] = observed
                    break
                event = json.loads(data)
                require(isinstance(event, dict) and "error" not in event,
                        f"stream error: {event}")
                if event.get("usage") is not None:
                    require(row["usage"] is None, "duplicate final usage event")
                    row["usage"] = event["usage"]
                choices = event.get("choices", [])
                require(isinstance(choices, list) and len(choices) <= 1,
                        "expected at most one raw completion choice")
                for choice in choices:
                    require(isinstance(choice, dict) and choice.get("index") == 0 and
                            "delta" not in choice, "expected raw completion choice zero")
                    text = choice.get("text", "")
                    require(isinstance(text, str), "completion text is not a string")
                    require(not text or not row["finish_reasons"], "text after finish event")
                    if text:
                        output.append(text)
                        row["output_event_times_s"].append(observed)
                    if choice.get("finish_reason") is not None:
                        row["finish_reasons"].append(choice["finish_reason"])
            usage = row["usage"]
            require(isinstance(usage, dict), "missing final token usage")
            require(usage.get("prompt_tokens") == prompt["tokens"] and
                    usage.get("completion_tokens") == self.args.gen and
                    usage.get("total_tokens") == prompt["tokens"] + self.args.gen,
                    f"exact token usage mismatch: {usage}")
            require(row["finish_reasons"] == ["length"],
                    f"expected one length finish: {row['finish_reasons']}")
            require(row["output_event_times_s"], "no nonempty output text events")
            row["ok"] = True
        except Exception as error:
            row["errors"].append(f"{type(error).__name__}: {error}")
            if "submit_s" not in row:
                barrier.abort()
        finally:
            row["end_s"] = self.now()
            if timer is not None:
                timer.cancel()
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()
            row["output_text"] = "".join(output)
            row["output_sha256"] = hashlib.sha256(row["output_text"].encode()).hexdigest()
            times = row["output_event_times_s"]
            row["observed_output_events"] = len(times)
            row["event_gaps_ms"] = [1000 * (right - left) for left, right in zip(times, times[1:])]
            row["event_gap_distribution_ms"] = distribution(row["event_gaps_ms"])
            submit = row.get("submit_s")
            row["ttft_ms"] = 1000 * (times[0] - submit) if times and submit is not None else None
            row["completion_wall_s"] = row["end_s"] - submit if submit is not None else None
        return row


def cohort(client, prompts, context, concurrency, repeat, phase):
    result = {"context": context, "concurrency": concurrency, "repeat": repeat,
              "phase": phase, "ok": False, "requests": [{} for _ in prompts]}
    barrier = threading.Barrier(concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(client.stream, row, prompt, barrier)
                   for row, prompt in zip(result["requests"], prompts)]
        for future in futures:
            future.result()
    rows = result["requests"]
    result["ok"] = all(row["ok"] for row in rows)
    starts = [row["submit_s"] for row in rows if "submit_s" in row]
    result["arrival_skew_ms"] = 1000 * (max(starts) - min(starts)) if starts else None
    result["wall_s"] = max(row["end_s"] for row in rows) - min(starts) if starts else None
    result["completion_tokens"] = sum(row["usage"]["completion_tokens"] for row in rows) if result["ok"] else None
    result["end_to_end_output_tokens_per_s"] = (result["completion_tokens"] / result["wall_s"]
                                               if result["ok"] and result["wall_s"] else None)
    return result


def summarize(cohorts):
    groups = {}
    for item in cohorts:
        if item["phase"] != "measured":
            continue
        key = f"{item['context']}x{item['concurrency']}"
        group = groups.setdefault(key, {"cohorts": 0, "valid_cohorts": 0,
                                       "failed_requests": 0, "ttft_ms": [], "completion_wall_s": [],
                                       "event_gaps_ms": [], "cohort_wall_s": [],
                                       "end_to_end_output_tokens_per_s": []})
        group["cohorts"] += 1
        group["failed_requests"] += sum(not row["ok"] for row in item["requests"])
        if not item["ok"]:
            continue
        group["valid_cohorts"] += 1
        for row in item["requests"]:
            group["ttft_ms"].append(row["ttft_ms"])
            group["completion_wall_s"].append(row["completion_wall_s"])
            group["event_gaps_ms"].extend(row["event_gaps_ms"])
        group["cohort_wall_s"].append(item["wall_s"])
        group["end_to_end_output_tokens_per_s"].append(item["end_to_end_output_tokens_per_s"])
    for group in groups.values():
        for key, value in tuple(group.items()):
            if isinstance(value, list):
                group[key] = distribution(value)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--contexts", default="512,2048,8192")
    parser.add_argument("--concurrency", default="1,2,8")
    parser.add_argument("--gen", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1, help="excluded full-shape cohorts per cell")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--timeout", type=float, default=900, help="per-stream absolute deadline seconds")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {"schema": "engine-compare-v1", "ok": False, "args": vars(args).copy(),
              "command_argv": [sys.executable, *sys.argv], "errors": [], "cohorts": [],
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "metric_definitions": {
                  "ttft_ms": "Client submit to first nonempty raw-completion SSE text event; includes queue/prefill/network/buffering.",
                  "event_gaps_ms": "Consecutive nonempty output SSE events, NOT native token ITL. UTF-8/output buffering can merge tokens.",
                  "completion_wall_s": "Client submit to stream completion observation, before HTTP cleanup.",
                  "end_to_end_output_tokens_per_s": "Validated cohort output token usage / earliest-submit through last-stream-end wall; includes prefill, NOT steady decode throughput.",
                  "workload_sha256": "Exact prompt text/token IDs, seed, shape, phase order and generation settings; excludes endpoint/model name/tokenizer filesystem path.",
                  "summary": "Only wholly successful measured cohorts enter latency distributions; failures and excluded warmups remain explicit.",
                  "cache_policy": "Cold prefixes and no speculation are externally configured server prerequisites; optional health is evidence, never automatic proof.",
                  "arrival_skew_ms": "Barrier-released client submission skew; not simultaneous server admission or GPU execution."}}
    if not math.isfinite(args.timeout):
        report["args"]["timeout"] = str(args.timeout)
    client = None
    try:
        contexts = [int(value) for value in args.contexts.split(",")]
        concurrencies = [int(value) for value in args.concurrency.split(",")]
        require(contexts and concurrencies and all(value > 0 for value in contexts + concurrencies),
                "contexts and concurrency must be positive integer lists")
        require(args.gen > 0 and args.repeats > 0 and args.warmup >= 0 and
                math.isfinite(args.timeout) and args.timeout > 0, "invalid generation/repeat/warmup/timeout")
        client = Client(args)
        report["health_before"] = client.snapshot("/health")
        report["models"] = client.snapshot("/v1/models")
        factory = Prompts(args.model_dir, args.seed)
        report["tokenizer"] = {"model_dir": os.path.abspath(os.path.expanduser(args.model_dir)),
                               "class": type(factory.tokenizer).__name__, "calibration": factory.calibration,
                               "local_files_only": True}
        workload = {"recipe": "engine-compare-v1", "seed": args.seed, "gen": args.gen,
                    "temperature": 0, "ignore_eos": True, "cells": []}
        for context in contexts:
            for concurrency in concurrencies:
                for phase, count in (("warmup", args.warmup), ("measured", args.repeats)):
                    for repeat in range(count):
                        identity = f"{context}:{concurrency}:{phase}:{repeat}"
                        prompts = [factory.make(context, f"{identity}:{index}") for index in range(concurrency)]
                        workload["cells"].append({"context": context, "concurrency": concurrency,
                                                  "repeat": repeat, "phase": phase, "prompts": prompts})
        report["workload_sha256"] = digest(workload)
        report["planned_cohorts"] = len(workload["cells"])
        for cell in workload["cells"]:
            result = cohort(client, **cell)
            report["cohorts"].append(result)
            report["summary"] = summarize(report["cohorts"])
            write_report(args.output, report)
            require(result["ok"], f"failed {cell['phase']} cohort {cell['context']}x{cell['concurrency']} repeat {cell['repeat']}; see request errors")
        report["ok"] = True
    except (Exception, KeyboardInterrupt) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if client is not None:
            report["health_after"] = client.snapshot("/health")
            report["campaign_wall_s"] = client.now()
        report["summary"] = summarize(report["cohorts"])
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            write_report(args.output, report)
        except Exception as error:
            print(f"cannot write artifact: {error}", file=sys.stderr)
            print(json.dumps(report, ensure_ascii=False, allow_nan=False), file=sys.stderr)
            return 1
    print(f"{'PASS' if report['ok'] else 'FAIL'}: {args.output}; {len(report['cohorts'])} cohorts")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
