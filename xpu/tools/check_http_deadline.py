#!/usr/bin/env python3
"""Exercise the real benchmark deadline against a Connection: close SSE server.

Never starts a server or substitutes responses. Uses the actual comparison Client
and local tokenizer. The existing endpoint must honor Connection: close and remain
incomplete until the deadline; increase --gen or reduce --timeout if it finishes.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import threading
import time

from bench_engine_compare import Client, Prompts
from bench_mixed_latency import require, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", required=True, help="real local tokenizer directory")
    parser.add_argument("--timeout", type=float, default=1.0, help="actual Client absolute stream deadline")
    parser.add_argument("--gen", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {"schema": "http-deadline-check-v1", "ok": False,
              "args": vars(args).copy(), "command_argv": [sys.executable, *sys.argv],
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "errors": [], "transport": {}, "request": {},
              "contract": {
                  "request": "Real 32-token locally calibrated raw prompt; actual Client.stream with ignore_eos and configured generation length.",
                  "transport": "Real HTTP(S)Connection subclass only adds Connection: close and observes socket detachment in getresponse; no response or socket replacement.",
                  "pass": "HTTP200, response Connection:close, live transport detached from connection, incomplete stream with deadline-compatible error, and bounded elapsed time near configured deadline.",
                  "elapsed_bounds": "0.9 * timeout <= elapsed <= timeout + max(0.5, 0.25 * timeout). Elapsed is submit through Client.stream return, including cleanup.",
                  "scope": "One real endpoint scenario, not proof against every network failure or timing schedule."}}
    if not math.isfinite(args.timeout):
        report["args"]["timeout"] = str(args.timeout)
    try:
        require(math.isfinite(args.timeout) and args.timeout > 0, "--timeout must be finite and positive")
        require(args.gen > 0, "--gen must be positive")
        factory = Prompts(args.model_dir, 20260908)
        prompt = factory.make(32, "http-deadline-connection-close")
        report["tokenizer"] = {"class": type(factory.tokenizer).__name__,
                               "calibration": factory.calibration, "seed": 20260908,
                               "local_files_only": True}
        client = Client(args)
        real_connection_class = client.connection_class
        transport = report["transport"]

        class CloseObservedConnection(real_connection_class):
            def putrequest(self, *positional, **keywords):
                super().putrequest(*positional, **keywords)
                self.putheader("Connection", "close")
                transport["requested_connection"] = "close"

            def getresponse(self):
                transport["connected_before_getresponse"] = self.sock is not None
                response = super().getresponse()
                transport["detached"] = self.sock is None
                transport["response_connection"] = response.getheader("Connection", "")
                transport["response_will_close"] = response.will_close
                transport["http_version"] = response.version
                return response

        client.connection_class = CloseObservedConnection
        started = time.perf_counter()
        client.stream(report["request"], prompt, threading.Barrier(1))
        report["call_elapsed_s"] = time.perf_counter() - started
        row = report["request"]
        elapsed = client.now() - row["submit_s"] if "submit_s" in row else None
        report["submit_to_return_s"] = elapsed
        report["lower_bound_s"] = 0.9 * args.timeout
        report["upper_bound_s"] = args.timeout + max(0.5, 0.25 * args.timeout)
        require(row.get("http_status") == 200,
                f"expected real HTTP200 before deadline, got {row.get('http_status')}; request errors: {row.get('errors')}")
        connection_tokens = {value.strip().lower() for value in
                             transport.get("response_connection", "").split(",")}
        require("close" in connection_tokens,
                "server did not return Connection: close; this endpoint does not exercise the required transport case")
        require(transport.get("connected_before_getresponse") is True and
                transport.get("detached") is True and transport.get("response_will_close") is True,
                "getresponse did not detach a connected transport; required ownership transition not exercised")
        require(not row.get("done") and not row.get("ok") and not row.get("finish_reasons") and
                row.get("usage") is None,
                "server completed generation before the deadline; increase --gen or reduce --timeout and rerun")
        require(row.get("errors"), "incomplete stream did not record an error")
        expected_errors = (
            "BenchmarkError: SSE ended without [DONE]",
            "BenchmarkError: stream deadline exceeded",
            "TimeoutError:",
        )
        require(all(error.startswith(expected_errors) for error in row["errors"]),
                f"stream failed for a reason other than expected deadline interruption: {row['errors']}")
        require(elapsed is not None and report["lower_bound_s"] <= elapsed <= report["upper_bound_s"],
                f"stream return {elapsed}s outside explicit deadline bounds "
                f"[{report['lower_bound_s']}, {report['upper_bound_s']}]s")
        report["ok"] = True
    except (Exception, KeyboardInterrupt) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            write_report(args.output, report)
        except Exception as error:
            print(f"cannot write artifact {args.output!r}: {error}", file=sys.stderr)
            print(json.dumps(report, ensure_ascii=False, allow_nan=False), file=sys.stderr)
            return 1
    print(f"{'PASS' if report['ok'] else 'FAIL'}: {args.output}; "
          f"submit-to-return={report.get('submit_to_return_s')}s")
    if report["errors"]:
        print("\n".join(report["errors"]), file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
