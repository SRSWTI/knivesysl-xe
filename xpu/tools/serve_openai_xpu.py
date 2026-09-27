#!/usr/bin/env python3
"""OpenAI-compatible API server for the knivesysl-xe (Intel Arc Pro B70) engine.

Endpoints: /v1/chat/completions, /v1/completions, /v1/models, /health.
Features: SSE streaming, stop strings + EOS, max_tokens, usage accounting,
Qwen tool-calling (<tool_call> JSON) round-trip, reasoning_content split at
</think>, and per-request stats in `x_knivesysl`.

Unlike the CUDA server (tools/serve_openai.py), which owns ONE sequence and
serializes requests behind a lock, this server is genuinely concurrent: a
single engine thread owns the in-order SYCL queue and runs the continuous
batching loop over `qwn_decode_batch` while HTTP handler threads submit
requests and stream tokens out of per-request queues.

  * requests are admitted into free slots MID-FLIGHT (no batch drain);
  * each engine iteration runs at most ONE bounded multi-request prefill
    wave via `qwn_prefill_batch`, plus ONE batched decode step covering
    generating slots and prompt tails (one weight stream, N rows);
  * automatic prefix caching: prompt prefixes are checkpointed at chunk
    boundaries into host USM and restored into a fresh slot on a hit.
    Granularity is the chunk rather than vLLM's 16-token block because 48
    of this model's 64 layers are recurrent - GDN conv/recurrent state
    exists only where it was snapshotted, so a partial KV cache is not a
    resumable state. Measured: 512-token prefix restore 55 ms vs 1083 ms
    to re-prefill, and TTFT mean 4.06 s -> 0.33 s on a shared-prefix load.

GREEDY ONLY. The XPU tier has no sampler: every step is argmax, which is
exactly the property its gates certify (100% greedy-identical to the
scalar reference, re-checked on every engine change). `temperature`,
`top_p` and `seed` are therefore accepted and IGNORED, with a warning in
`x_knivesysl.warnings` - clients keep working, nobody is misled. Host-side
sampling needs a logits export and is not built.
n>1, logprobs: unsupported -> 400.

Run:   TQ_XPU_DEV=0 xpu/tools/serve_openai_xpu.py --port 8000 \
           --tqf ~/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf \
           --model-dir ~/models/knivesysl --slots 8
Test:  curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":
       "user","content":"Hej!"}],"max_tokens":64,"stream":true}'
"""
from __future__ import annotations
import argparse, ctypes, json, os, queue, select, socket, sys, threading, time, uuid
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--lib", default="xpu/build/libforward_qwen_xpu.so")
ap.add_argument("--tqf", required=True)
ap.add_argument("--model-dir", required=True)
ap.add_argument("--model-name", default="knivesysl-xe-qwen3.8-27b-w4a8")
ap.add_argument("--slots", type=int, default=None,
                help="concurrent sequences (engine cap 8; env TQ_XPU_SLOTS; "
                     "default 8)")
ap.add_argument("--ctx", type=int, default=None,
                help="max prompt+gen tokens (env TQ_CTX; default 4096)")
ap.add_argument("--chunk", type=int, default=64,
                help="total prefill token budget per iteration while decoding; "
                     "smaller = lower ITL jitter, higher TTFT")
ap.add_argument("--chunk-idle", type=int, default=512,
                help="total prefill token budget when nothing is decoding")
ap.add_argument("--prefill-slice-ms", type=float, default=0.0,
                help="target busy native prefill wave milliseconds (0 disables); "
                     "adapts within the --chunk cap, minimum 8 rows")
ap.add_argument("--k64", default=None,
                help="TQ_XPU_K64 selector ('gate,up,down' or 'all'): the "
                     "opt-in W4A4 prefill tier. Raises aggregate decode "
                     "121->147 tok/s and prefill ~15-29%%, costs TF "
                     "agreement (93.4%% -> 92.2%% MLP-only, 89.9%% all). "
                     "Default off unless TQ_XPU_K64 is set.")
ap.add_argument("--prefill-xmx", choices=("auto", "0", "1"), default=None,
                help="attention prefill: auto, scalar (0), required XMX (1); "
                     "env TQ_XPU_PREFILL_XMX, default auto")
ap.set_defaults(packed_prefill=None)
ap.add_argument("--packed-prefill", dest="packed_prefill", action="store_true",
                help="pack prompt segments into one native wave (default ON)")
ap.add_argument("--no-packed-prefill", dest="packed_prefill", action="store_false",
                help="serial native prefill chunks for matched comparisons")
ap.set_defaults(paged=None)
ap.add_argument("--paged", dest="paged", action="store_true",
                help="use the shared paged KV pool (default off)")
ap.add_argument("--no-paged", dest="paged", action="store_false",
                help="force flat per-slot KV even if TQ_XPU_PAGED is set")
ap.add_argument("--kv-page", type=int, default=None,
                help="paged KV page tokens: 128 or 256 (env TQ_XPU_KV_PAGE)")
pool = ap.add_mutually_exclusive_group()
pool.add_argument("--kv-pool-tokens", type=int, default=None,
                  help="paged physical token capacity "
                       "(env TQ_XPU_KV_POOL_TOKENS)")
pool.add_argument("--kv-pool-mb", type=int, default=None,
                  help="paged decimal-MB KV-only budget "
                       "(env TQ_XPU_KV_POOL_MB)")
ap.add_argument("--response-idle-timeout", type=float, default=300.0,
                help="seconds without an engine token before cancelling a "
                     "response (default 300)")
ap.add_argument("--prefix-cache", dest="prefix_cache", action="store_true",
                default=True, help="automatic prefix caching (default ON)")
ap.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false")
ap.add_argument("--prefix-cache-min", type=int, default=128,
                help="min cached prefix tokens (also the chunk granularity)")
ap.add_argument("--prefix-cache-max-mb", type=int, default=8192,
                help="host-USM budget for checkpoints; LRU-evicted")
args = ap.parse_args()


def _env_int(name, default=None):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        ap.error(f"{name} must be an integer, got {raw!r}")


# Resolve once before qwn_init. An explicit CLI value always overwrites the
# corresponding native environment value; omitted options retain intentional
# environment configuration. The canonical values below are what native sees.
if args.paged is None:
    paged_env = os.environ.get("TQ_XPU_PAGED", "0")
    if paged_env not in ("0", "1"):
        ap.error(f"TQ_XPU_PAGED must be 0 or 1, got {paged_env!r}")
    args.paged = paged_env == "1"
os.environ["TQ_XPU_PAGED"] = "1" if args.paged else "0"

if args.prefill_xmx is None:
    args.prefill_xmx = os.environ.get("TQ_XPU_PREFILL_XMX", "auto")
if args.prefill_xmx not in ("auto", "0", "1"):
    ap.error("TQ_XPU_PREFILL_XMX must be auto, 0 or 1")
os.environ["TQ_XPU_PREFILL_XMX"] = args.prefill_xmx
if args.packed_prefill is None:
    packed_env = os.environ.get("TQ_XPU_PACKED_PREFILL", "1")
    if packed_env not in ("0", "1"):
        ap.error("TQ_XPU_PACKED_PREFILL must be 0 or 1")
    args.packed_prefill = packed_env == "1"
os.environ["TQ_XPU_PACKED_PREFILL"] = "1" if args.packed_prefill else "0"

if args.slots is None:
    args.slots = _env_int("TQ_XPU_SLOTS", 8)
if args.paged and not 1 <= args.slots <= 8:
    ap.error("--slots must be between 1 and 8 in paged mode")
args.slots = max(1, min(8, args.slots))  # preserve the flat server's clamp
os.environ["TQ_XPU_SLOTS"] = str(args.slots)

if args.ctx is None:
    args.ctx = _env_int("TQ_CTX", 4096)
if args.ctx <= 0 or args.ctx > 2_147_483_647:
    ap.error("--ctx must be between 1 and 2147483647")
os.environ["TQ_CTX"] = str(args.ctx)

if args.k64 is None:
    args.k64 = os.environ.get("TQ_XPU_K64", "")
elif args.k64:
    os.environ["TQ_XPU_K64"] = args.k64
else:
    os.environ.pop("TQ_XPU_K64", None)

if args.kv_page is not None:
    if args.kv_page not in (128, 256):
        ap.error("--kv-page must be 128 or 256")
    os.environ["TQ_XPU_KV_PAGE"] = str(args.kv_page)
elif args.paged:
    args.kv_page = _env_int("TQ_XPU_KV_PAGE", 128)
    if args.kv_page not in (128, 256):
        ap.error("TQ_XPU_KV_PAGE must be 128 or 256")
    os.environ["TQ_XPU_KV_PAGE"] = str(args.kv_page)

cli_pool_tokens = args.kv_pool_tokens is not None
cli_pool_mb = args.kv_pool_mb is not None
if cli_pool_tokens:
    os.environ["TQ_XPU_KV_POOL_TOKENS"] = str(args.kv_pool_tokens)
    os.environ.pop("TQ_XPU_KV_POOL_MB", None)
elif cli_pool_mb:
    os.environ["TQ_XPU_KV_POOL_MB"] = str(args.kv_pool_mb)
    os.environ.pop("TQ_XPU_KV_POOL_TOKENS", None)
elif args.paged:
    args.kv_pool_tokens = _env_int("TQ_XPU_KV_POOL_TOKENS")
    args.kv_pool_mb = _env_int("TQ_XPU_KV_POOL_MB")
    if args.kv_pool_tokens is not None and args.kv_pool_mb is not None:
        ap.error("TQ_XPU_KV_POOL_TOKENS and TQ_XPU_KV_POOL_MB are mutually "
                 "exclusive")

if args.kv_pool_tokens is not None and args.kv_pool_tokens <= 0:
    ap.error("--kv-pool-tokens/TQ_XPU_KV_POOL_TOKENS must be positive")
if args.kv_pool_mb is not None and args.kv_pool_mb <= 0:
    ap.error("--kv-pool-mb/TQ_XPU_KV_POOL_MB must be positive")
if args.paged and args.kv_pool_tokens is None and args.kv_pool_mb is None:
    ap.error("paged mode requires --kv-pool-tokens or --kv-pool-mb "
             "(or the corresponding native environment variable)")
if args.paged:
    tp = _env_int("TQ_XPU_TP", 1)
    if tp != 1:
        ap.error("paged mode currently requires TQ_XPU_TP=1")
    if args.ctx < 8:
        ap.error("paged mode requires --ctx >= 8")

if args.response_idle_timeout <= 0:
    ap.error("--response-idle-timeout must be positive")
if not math.isfinite(args.prefill_slice_ms) or args.prefill_slice_ms < 0:
    ap.error("--prefill-slice-ms must be finite and nonnegative")
if args.prefix_cache_max_mb < 0:
    ap.error("--prefix-cache-max-mb must be nonnegative")

PREFILL_CHUNK_BUSY = max(8, (args.chunk // 8) * 8)
PREFILL_CHUNK_IDLE = max(PREFILL_CHUNK_BUSY,
                         (args.chunk_idle // 8) * 8)
if args.paged:
    warm_chunk = min(PREFILL_CHUNK_IDLE, (args.ctx // 8) * 8)
    os.environ["TQ_XPU_PREFILL_CHUNK"] = str(warm_chunk)

from transformers import AutoTokenizer  # noqa: E402

print(f"[serve-xe] loading {args.tqf} ...", flush=True)
TOK = AutoTokenizer.from_pretrained(os.path.expanduser(args.model_dir),
                                    trust_remote_code=True)
LIB = ctypes.CDLL(os.path.abspath(os.path.expanduser(args.lib)))
LIB.qwn_init.argtypes = [ctypes.c_char_p]
LIB.qwn_init.restype = ctypes.c_int
LIB.qwn_num_slots.argtypes = []
LIB.qwn_num_slots.restype = ctypes.c_int
LIB.qwn_max_seq.argtypes = []
LIB.qwn_max_seq.restype = ctypes.c_int
LIB.qwn_num_layers.argtypes = []
LIB.qwn_num_layers.restype = ctypes.c_int
LIB.qwn_num_key_value_heads.argtypes = []
LIB.qwn_num_key_value_heads.restype = ctypes.c_int
LIB.qwn_head_dim.argtypes = []
LIB.qwn_head_dim.restype = ctypes.c_int
LIB.qwn_layer_type.argtypes = [ctypes.c_int]
LIB.qwn_layer_type.restype = ctypes.c_int
LIB.qwn_attn_branch_counts.argtypes = [
    ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int]
LIB.qwn_attn_branch_counts.restype = ctypes.c_int
LIB.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
LIB.qwn_decode.restype = ctypes.c_int
LIB.qwn_set_slot.argtypes = [ctypes.c_int]
LIB.qwn_set_slot.restype = ctypes.c_int
LIB.qwn_reset_slot.argtypes = [ctypes.c_int]
LIB.qwn_reset_slot.restype = ctypes.c_int
LIB.qwn_prefill_chunk.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                  ctypes.c_int]
LIB.qwn_prefill_chunk.restype = ctypes.c_int
try:
    LIB.qwn_prefill_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4 + [
        ctypes.c_int]
    LIB.qwn_prefill_batch.restype = ctypes.c_int
except AttributeError:
    sys.exit("XPU library is missing required qwn_prefill_batch ABI; "
             "rebuild with xpu/build.sh (no serial compatibility fallback)")
LIB.qwn_decode_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [
    ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
LIB.qwn_decode_batch.restype = ctypes.c_int
LIB.qwn_host_alloc.argtypes = [ctypes.c_size_t]
LIB.qwn_host_alloc.restype = ctypes.c_void_p
LIB.qwn_host_free.argtypes = [ctypes.c_void_p]
LIB.qwn_host_free.restype = None
LIB.qwn_ckpt_bytes.argtypes = [ctypes.c_int]
LIB.qwn_ckpt_bytes.restype = ctypes.c_size_t
LIB.qwn_ckpt_save.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
LIB.qwn_ckpt_save.restype = ctypes.c_int
LIB.qwn_ckpt_restore.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
LIB.qwn_ckpt_restore.restype = ctypes.c_int
LIB.qwn_paged_enabled.argtypes = []
LIB.qwn_paged_enabled.restype = ctypes.c_int
LIB.qwn_paged_reserve.argtypes = [ctypes.c_int, ctypes.c_int]
LIB.qwn_paged_reserve.restype = ctypes.c_int
LIB.qwn_paged_release.argtypes = [ctypes.c_int]
LIB.qwn_paged_release.restype = ctypes.c_int
LIB.qwn_paged_stats.argtypes = [ctypes.POINTER(ctypes.c_longlong),
                                ctypes.c_int]
LIB.qwn_paged_stats.restype = ctypes.c_int
LIB.qwn_paged_ckpt_save.argtypes = [ctypes.c_int, ctypes.c_int]
LIB.qwn_paged_ckpt_save.restype = ctypes.c_int
LIB.qwn_paged_ckpt_adopt.argtypes = [ctypes.c_int, ctypes.c_int]
LIB.qwn_paged_ckpt_adopt.restype = ctypes.c_int
LIB.qwn_paged_ckpt_free.argtypes = [ctypes.c_int]
LIB.qwn_paged_ckpt_free.restype = ctypes.c_int
LIB.qwn_paged_ckpt_host_bytes.argtypes = [ctypes.c_int]
LIB.qwn_paged_ckpt_host_bytes.restype = ctypes.c_size_t
PAGED_CAPACITY_ERROR = -1001  # TQ_PAGED_ERR_CAPACITY in tq_common.hpp
EOS_IDS = set(int(t) for t in [TOK.eos_token_id] if t is not None)
for name in ("<|im_end|>", "<|endoftext|>"):
    try:
        i = TOK.convert_tokens_to_ids(name)
        if isinstance(i, int) and i >= 0:
            EOS_IDS.add(i)
    except Exception:
        pass
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"


# --------------------------------- engine ---------------------------------
WAIT, PREFILL, DECODE, DONE = 0, 1, 2, 3


def _percentile_ms(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    value = ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)
    return round(value * 1e3, 1)


class Gen:
    """One in-flight generation. The engine thread writes; a handler reads."""

    def __init__(self, ids, max_new, ignore_eos):
        self.ids = list(ids)
        self.max_new = max_new
        self.total_tokens = len(self.ids) + max_new
        self.ignore_eos = ignore_eos
        self.q = queue.Queue()      # token id, or ("end", reason)
        self.cancel = threading.Event()
        self.state, self.slot, self.assigned_slot, self.pos = WAIT, -1, -1, 0
        self.owned = False
        self.capacity_waiting = False
        self.wait_epoch = -1
        self.n_out = 0
        self.reused = 0
        self.t_sub = time.perf_counter()
        self.t_admit = None
        self.t_first = None
        self.admission_work = 0.0
        self.token_times = []
        self.batch_widths = []
        self.prefill_waves = self.prefill_rows = 0
        self.prefill_wave_rows = 0
        self.prefill_width_hist = {}
        self.last = -1

    def metrics(self):
        itls = [self.token_times[i] - self.token_times[i - 1]
                for i in range(1, len(self.token_times))]
        widths = {}
        for width in self.batch_widths:
            widths[width] = widths.get(width, 0) + 1
        return {
            "slot": self.assigned_slot,
            "generated_tokens": self.n_out,
            "queue_ms": round((self.t_admit - self.t_sub) * 1e3, 1)
            if self.t_admit is not None else None,
            "admission_ms": round(self.admission_work * 1e3, 1),
            "ttft_ms": round((self.t_first - self.t_sub) * 1e3, 1)
            if self.t_first is not None else None,
            "itl_ms_p50": _percentile_ms(itls, 0.50),
            "itl_ms_p99": _percentile_ms(itls, 0.99),
            "decode_width_hist": dict(sorted(widths.items())),
            "packed_prefill": args.packed_prefill,
            "prefill_xmx": args.prefill_xmx,
            "prefill_waves": self.prefill_waves,
            "prefill_rows": self.prefill_rows,
            "prefill_wave_rows": self.prefill_wave_rows,
            "prefill_width_hist": dict(sorted(self.prefill_width_hist.items())),
            "batch_avg": round(sum(self.batch_widths) /
                               max(1, len(self.batch_widths)), 2),
        }


class Engine(threading.Thread):
    daemon = True

    def __init__(self):
        super().__init__(name="xe-engine")
        self.chunk = max(8, (args.prefix_cache_min // 8) * 8)
        self.chunk_busy = PREFILL_CHUNK_BUSY
        self.chunk_idle = PREFILL_CHUNK_IDLE
        self.prefill_budget_current = self.chunk_busy
        self.nslots = 0
        self.maxseq = args.ctx
        self.free = []
        self.pending, self.active = [], []
        self.inbox, self.lock = [], threading.Lock()
        self.cache, self.cache_lru = {}, []
        self.cache_lengths = {}
        self.seen = set()
        self.cache_bytes = 0
        self.hits = self.misses = 0
        self.capacity_waits = 0
        self.resource_epoch = 0
        self.quarantined = set()
        self.request_errors = 0
        self.last_request_error = None
        self.cancelled_requests = 0
        self.kv_bytes_per_token = None
        self.dec_rows, self.dec_time = 0, 0.0
        self.width_hist = {}
        self.width_time = {}
        self.prefill_cursor = 0
        self.prefill_tails = []
        self.prefill_waves = self.prefill_rows = self.prefill_errors = 0
        self.prefill_width_hist = {}
        self.prefill_last_width = self.prefill_last_rows = 0
        self.prefill_max_width = self.prefill_max_rows = 0
        self.paged = False
        self.initialized = False
        self.attention_config = {
            "simd16": None, "grouped": None, "dpas": None,
            "prefill_xmx": args.prefill_xmx}
        self.attention_branch_counts = {
            "prefill": 0,
            "simd16_short": 0,
            "simd16_sharded": 0,
            "simd16_grouped": 0,
            "generic": 0,
            "simd16_dpas": 0,
            "prefill_xmx": 0,
            "prefill_scalar": 0,
        }
        self.terminal_error = None
        self.peak_used_blocks = 0
        self.peak_remaining_reserved = 0
        self.pool_stats = {
            "kv_page_tokens": None,
            "kv_pool_tokens": None,
            "kv_pool_bytes": None,
            "kv_pool_blocks_total": None,
            "kv_pool_blocks_used": None,
            "kv_pool_blocks_free": None,
            "kv_active_blocks": None,
            "kv_checkpoint_blocks": None,
            "kv_remaining_reserved_blocks": None,
            "kv_sequence_limit": args.ctx,
        }
        self.ready = threading.Event()
        self._snapshot = {
            "status": "starting",
            "engine_error": None,
            "kv_layout": "starting",
            "paged": None,
        }

    # ---- public snapshots ------------------------------------------------
    def submit(self, g):
        with self.lock:
            if self.terminal_error is not None:
                return self.terminal_error
            if self.ready.is_set() and not self.is_alive():
                return "engine thread stopped unexpectedly"
            self.inbox.append(g)
        return None

    def stats(self):
        with self.lock:
            result = dict(self._snapshot)
            for name in ("decode_width_hist", "decode_tok_s_by_width",
                         "prefill_width_hist", "attention_config",
                         "attention_branch_counts"):
                if name in result:
                    result[name] = dict(result[name])
            if (self.ready.is_set() and not self.is_alive() and
                    result.get("status") != "error"):
                result["status"] = "error"
                result["engine_error"] = "engine thread stopped unexpectedly"
            return result

    def request_error(self, prompt_tokens, output_tokens):
        snapshot = self.stats()
        if snapshot.get("status") == "error":
            return 503, f"XPU engine unavailable: {snapshot['engine_error']}"
        if not snapshot.get("paged"):
            return None
        total = prompt_tokens + output_tokens
        limit = snapshot["kv_sequence_limit"]
        if total > limit:
            return 400, (
                f"request requires {total} prompt+output tokens, exceeds "
                f"paged context limit {limit}; max_tokens is not clamped "
                "in paged mode"
            )
        page = snapshot["kv_page_tokens"]
        blocks = (total + page - 1) // page
        total_blocks = snapshot["kv_pool_blocks_total"]
        if blocks > total_blocks:
            return 400, (
                f"request requires {blocks} KV pages for {total} tokens, "
                f"exceeds empty paged pool capacity {total_blocks} pages "
                f"({snapshot['kv_pool_tokens']} tokens)"
            )
        return None

    # ---- health ----------------------------------------------------------
    def _capture_pool_stats(self):
        if not self.paged:
            return
        out = (ctypes.c_longlong * 8)()
        rc = LIB.qwn_paged_stats(out, 8)
        if rc != 0:
            raise RuntimeError(f"qwn_paged_stats failed rc={rc}")
        page, total, free, active, checkpoints, reserved, limit, slots = (
            int(out[i]) for i in range(8)
        )
        if (page <= 0 or total <= 0 or free < 0 or free > total or
                active < 0 or checkpoints < 0 or reserved < 0 or
                limit <= 0 or slots <= 0):
            raise RuntimeError(
                "qwn_paged_stats returned invalid values "
                f"{[int(out[i]) for i in range(8)]}"
            )
        if reserved > free:
            raise RuntimeError(
                f"paged reservation conservation violated: "
                f"reserved={reserved}, free={free}")
        used = total - free
        self.peak_used_blocks = max(self.peak_used_blocks, used)
        self.peak_remaining_reserved = max(
            self.peak_remaining_reserved, reserved)
        self.pool_stats = {
            "kv_page_tokens": page,
            "kv_pool_tokens": page * total,
            "kv_pool_bytes": (page * total * self.kv_bytes_per_token
                              if self.kv_bytes_per_token is not None else None),
            "kv_pool_blocks_total": total,
            "kv_pool_blocks_used": used,
            "kv_pool_blocks_free": free,
            "kv_active_blocks": active,
            "kv_checkpoint_blocks": checkpoints,
            "kv_remaining_reserved_blocks": reserved,
            "kv_sequence_limit": limit,
        }
        if slots != self.nslots:
            raise RuntimeError(
                f"paged stats slots={slots}, qwn_num_slots={self.nslots}")

    def _capture_attention_stats(self):
        out = (ctypes.c_ulonglong * 8)()
        rc = LIB.qwn_attn_branch_counts(out, 8)
        if rc != 0:
            raise RuntimeError(f"qwn_attn_branch_counts failed rc={rc}")
        names = (
            "prefill",
            "simd16_short",
            "simd16_sharded",
            "simd16_grouped",
            "generic",
            "simd16_dpas",
            "prefill_xmx",
            "prefill_scalar",
        )
        self.attention_branch_counts = {
            name: int(out[index]) for index, name in enumerate(names)}

    def _publish_stats(self, capture=True):
        if capture and self.initialized:
            self._capture_pool_stats()
            self._capture_attention_stats()
        with self.lock:
            self._snapshot = {
                "status": "error" if self.terminal_error else "ok",
                "engine_error": self.terminal_error,
                "slots": self.nslots,
                "active": len(self.active),
                "queued": len(self.pending) + len(self.inbox),
                "quarantined_slots": sorted(self.quarantined),
                "kv_layout": "paged" if self.paged else "flat",
                "attention_config": dict(self.attention_config),
                "attention_branch_counts":
                    dict(self.attention_branch_counts),
                "paged": self.paged,
                **self.pool_stats,
                "kv_pool_blocks_peak_used": self.peak_used_blocks
                if self.paged else None,
                "kv_peak_remaining_reserved_blocks":
                    self.peak_remaining_reserved if self.paged else None,
                "capacity_waits": self.capacity_waits,
                "capacity_waiting": sum(
                    1 for request in self.pending
                    if request.capacity_waiting),
                "apc_enabled": args.prefix_cache,
                "apc_entries": len(self.cache),
                "apc_hits": self.hits,
                "apc_misses": self.misses,
                "apc_mb": round(self.cache_bytes / 1e6, 1),
                "apc_host_mb": round(self.cache_bytes / 1e6, 1),
                "request_errors": self.request_errors,
                "last_request_error": self.last_request_error,
                "cancelled_requests": self.cancelled_requests,
                "tier": (f"k64:{args.k64}" if args.k64 else "k32"),
                "k64": args.k64 or "off",
                "packed_prefill": args.packed_prefill,
                "prefill_budget_busy": self.chunk_busy,
                "prefill_budget_idle": self.chunk_idle,
                "prefill_slice_ms": args.prefill_slice_ms,
                "prefill_budget_busy_current": self.prefill_budget_current,
                "prefill_waves": self.prefill_waves,
                "prefill_rows": self.prefill_rows,
                "prefill_errors": self.prefill_errors,
                "prefill_width_hist": dict(sorted(self.prefill_width_hist.items())),
                "prefill_last_width": self.prefill_last_width,
                "prefill_last_rows": self.prefill_last_rows,
                "prefill_max_width": self.prefill_max_width,
                "prefill_max_rows": self.prefill_max_rows,
                "decode_width_hist": dict(sorted(self.width_hist.items())),
                "decode_tok_s_by_width": {
                    width: round(width * self.width_hist[width] /
                                 self.width_time[width], 1)
                    for width in sorted(self.width_time)
                    if self.width_time[width] > 0
                },
                "decode_tok_s": round(self.dec_rows / self.dec_time, 1)
                if self.dec_time > 0 else None,
            }

    # ---- APC -------------------------------------------------------------
    def _lookup(self, ids):
        if not args.prefix_cache:
            return None
        for length in sorted(self.cache_lengths, reverse=True):
            if length >= len(ids):
                continue
            key = tuple(ids[:length])
            entry = self.cache.get(key)
            if entry is not None:
                return key, entry
        return None

    def _touch(self, key):
        if key in self.cache_lru:
            self.cache_lru.remove(key)
        self.cache_lru.append(key)

    def _register_cache(self, key, entry):
        self.cache[key] = entry
        self.cache_lru.append(key)
        length = entry[0]
        self.cache_lengths[length] = self.cache_lengths.get(length, 0) + 1
        self.cache_bytes += entry[2]

    def _evict(self, key):
        entry = self.cache[key]
        if self.paged:
            rc = LIB.qwn_paged_ckpt_free(entry[1])
            if rc != 0:
                raise RuntimeError(
                    f"qwn_paged_ckpt_free id={entry[1]} failed rc={rc}")
            self.resource_epoch += 1
        else:
            LIB.qwn_host_free(entry[1])
        del self.cache[key]
        if key in self.cache_lru:
            self.cache_lru.remove(key)
        length = entry[0]
        left = self.cache_lengths[length] - 1
        if left:
            self.cache_lengths[length] = left
        else:
            del self.cache_lengths[length]
        self.cache_bytes -= entry[2]

    def _store(self, ids, pos, slot):
        if (not args.prefix_cache or pos < self.chunk or pos >= len(ids) or
                args.prefix_cache_max_mb == 0):
            return
        if self.paged and any(g.capacity_waiting for g in self.pending):
            return  # optional APC must not prolong an existing capacity wait
        key = tuple(ids[:pos])
        if key in self.cache:
            return
        # Second-sighting admission is only a compact, lossy hint: a hash
        # collision may checkpoint a prefix early, but cannot select or restore
        # it because self.cache remains keyed by the exact token tuple above.
        fingerprint = hash(key)
        if fingerprint not in self.seen:
            if len(self.seen) >= 200_000:
                self.seen.clear()
            self.seen.add(fingerprint)
            return
        budget = args.prefix_cache_max_mb * 1_000_000
        if self.paged:
            checkpoint = int(LIB.qwn_paged_ckpt_save(slot, pos))
            if checkpoint < 0:
                return
            charge = int(LIB.qwn_paged_ckpt_host_bytes(checkpoint))
            self._register_cache(key, (pos, checkpoint, charge))
            while self.cache_bytes > budget and self.cache_lru:
                self._evict(self.cache_lru[0])
            return

        charge = int(LIB.qwn_ckpt_bytes(pos))
        if charge <= 0:
            return
        while self.cache_bytes + charge > budget and self.cache_lru:
            self._evict(self.cache_lru[0])
        if self.cache_bytes + charge > budget:
            return
        blob = LIB.qwn_host_alloc(charge)
        if not blob:
            return
        if LIB.qwn_ckpt_save(slot, pos, blob) != 0:
            LIB.qwn_host_free(blob)
            return
        self._register_cache(key, (pos, blob, charge))

    # ---- ownership/admission --------------------------------------------
    def _quarantine(self, slot):
        self.quarantined.add(slot)
        self.free = [candidate for candidate in self.free
                     if candidate != slot]

    def _reset_empty_slot(self, slot):
        rc = LIB.qwn_reset_slot(slot)
        if rc != 0:
            self._quarantine(slot)
            raise RuntimeError(f"qwn_reset_slot({slot}) failed rc={rc}")

    def _release_native(self, slot, resource_change):
        rc = LIB.qwn_paged_release(slot)
        if rc != 0:
            self._quarantine(slot)
            raise RuntimeError(f"qwn_paged_release({slot}) failed rc={rc}")
        if resource_change:
            self.resource_epoch += 1

    def _admit_paged(self, g, slot):
        while True:
            candidate = self._lookup(g.ids)
            hit_key = None
            hit_pos = 0
            adoption_failed = False
            self._reset_empty_slot(slot)
            if candidate is not None:
                key, entry = candidate
                adopted = int(LIB.qwn_paged_ckpt_adopt(slot, entry[1]))
                if adopted == entry[0]:
                    hit_key, hit_pos = key, adopted
                else:
                    # Adopt is transactional, and release is idempotent. Still
                    # close the ownership transaction explicitly before cold
                    # fallback, including allocation-pressure failures.
                    self._release_native(slot, False)
                    if adopted >= 0:
                        raise RuntimeError(
                            "qwn_paged_ckpt_adopt returned mismatched "
                            f"position {adopted}, expected {entry[0]}")
                    adoption_failed = True
                    self._reset_empty_slot(slot)

            rc = LIB.qwn_paged_reserve(slot, g.total_tokens)
            if rc == 0:
                g.owned = True
                g.pos = hit_pos
                g.reused = hit_pos
                if hit_key is not None:
                    self._touch(hit_key)
                return True, hit_key is not None

            eviction_can_help = True
            if rc == PAGED_CAPACITY_ERROR and not adoption_failed:
                # Prove whether dropping every optional checkpoint could make
                # this reservation fit while the adopted target still appears
                # in active-block ownership. Preserve useful APC entries when
                # active completion credits make admission impossible anyway.
                self._capture_pool_stats()
                page = self.pool_stats["kv_page_tokens"]
                target_maps = (hit_pos + page - 1) // page
                needed = ((g.total_tokens + page - 1) // page) - target_maps
                max_free_after_eviction = (
                    self.pool_stats["kv_pool_blocks_total"] -
                    self.pool_stats["kv_active_blocks"])
                existing_credits = self.pool_stats[
                    "kv_remaining_reserved_blocks"]
                eviction_can_help = (
                    existing_credits + needed <= max_free_after_eviction)

            self._release_native(slot, False)
            if rc != PAGED_CAPACITY_ERROR:
                raise RuntimeError(
                    f"qwn_paged_reserve({slot}, {g.total_tokens}) "
                    f"failed rc={rc}")
            if self.cache_lru and eviction_can_help:
                # Optional checkpoints are the only evictable ownership. A
                # retry always performs a fresh lookup because this eviction
                # may invalidate the candidate just adopted and rolled back.
                self._evict(self.cache_lru[0])
                continue
            if not self.active:
                raise RuntimeError(
                    "native capacity rejected a feasible request with no "
                    "active requests or checkpoints")
            if not g.capacity_waiting:
                g.capacity_waiting = True
                self.capacity_waits += 1
            g.wait_epoch = self.resource_epoch
            return False, False

    def _admit_flat(self, g, slot):
        candidate = self._lookup(g.ids)
        self._reset_empty_slot(slot)
        if candidate is not None:
            key, entry = candidate
            if LIB.qwn_ckpt_restore(slot, entry[0], entry[1]) == 0:
                g.pos = g.reused = entry[0]
                self._touch(key)
                return True
            self._evict(key)
            self._reset_empty_slot(slot)
        g.pos = g.reused = 0
        return False

    def _admit(self):
        while self.free and self.pending:
            g = self.pending[0]
            if g.cancel.is_set():
                self.pending.pop(0)
                self._finish(g, "cancelled")
                continue
            if (self.paged and g.capacity_waiting and
                    g.wait_epoch == self.resource_epoch):
                return
            slot = self.free[0]
            started = time.perf_counter()
            if self.paged:
                admitted, hit = self._admit_paged(g, slot)
            else:
                admitted, hit = True, self._admit_flat(g, slot)
            g.admission_work += time.perf_counter() - started
            if not admitted:
                return
            self.pending.pop(0)
            self.free.pop(0)
            g.slot = g.assigned_slot = slot
            g.capacity_waiting = False
            g.t_admit = time.perf_counter()
            g.state = PREFILL
            if hit:
                self.hits += 1
            else:
                self.misses += 1
            self.active.append(g)

    # ---- model loop ------------------------------------------------------
    def _request_error(self, message):
        self.request_errors += 1
        self.last_request_error = message

    def _select_slot(self, g):
        rc = LIB.qwn_set_slot(g.slot)
        if rc == 0:
            return True
        self._request_error(f"qwn_set_slot({g.slot}) failed rc={rc}")
        self._finish(g, "engine_error")
        return False

    def _prefill_wave(self):
        self.prefill_tails = []
        for g in self.active:
            if g.state == PREFILL and g.cancel.is_set():
                self._finish(g, "cancelled")
        candidates = sorted(
            (g for g in self.active if g.state == PREFILL),
            key=lambda g: (g.slot - self.prefill_cursor) % self.nslots)
        if not candidates:
            return
        segments = []
        for g in candidates:
            # Leave the final prompt token for the logits-producing decode
            # batch. Stop at the deepest APC boundary before passing it.
            available = ((len(g.ids) - g.pos - 1) // 8) * 8
            deepest = ((len(g.ids) - 1) // self.chunk) * self.chunk
            if g.pos < deepest:
                available = min(available, deepest - g.pos)
            if available >= 8:
                segments.append([g, 0, available, deepest])
            else:
                self.prefill_tails.append(g)
        decoding = any(g.state == DECODE for g in self.active)
        busy = decoding or bool(self.prefill_tails)
        tune_busy = args.prefill_slice_ms > 0 and decoding
        budget = (self.prefill_budget_current if tune_busy else
                  self.chunk_busy if busy else self.chunk_idle)
        # Each tail consumes one prompt row in this iteration's decode batch.
        # Reserve those rows before distributing the ONE shared wave budget.
        units = (budget - len(self.prefill_tails)) // 8
        if not args.packed_prefill:
            segments = segments[:1]
        remaining = list(segments)
        while units > 0 and remaining:
            share = max(1, units // len(remaining))
            for segment in remaining:
                take = min(share, units, (segment[2] - segment[1]) // 8)
                segment[1] += take * 8
                units -= take
                if take:
                    self.prefill_cursor = (segment[0].slot + 1) % self.nslots
            remaining = [s for s in remaining if s[1] < s[2]]
        segments = [s for s in segments if s[1]]
        if not segments:
            return
        width = len(segments)
        offsets = [0]
        tokens = []
        for g, take, _, _ in segments:
            tokens.extend(g.ids[g.pos:g.pos + take])
            offsets.append(len(tokens))
        rows = len(tokens)
        Arr = ctypes.c_int * width
        toks = (ctypes.c_int * rows)(*tokens)
        if args.packed_prefill:
            slots = Arr(*(s[0].slot for s in segments))
            bounds = (ctypes.c_int * (width + 1))(*offsets)
            positions = Arr(*(s[0].pos for s in segments))
            started = time.perf_counter() if tune_busy else None
            rc = LIB.qwn_prefill_batch(slots, toks, bounds, positions, width)
            elapsed = time.perf_counter() - started if tune_busy else None
            operation = "qwn_prefill_batch"
        else:
            g = segments[0][0]
            if not self._select_slot(g):
                return
            started = time.perf_counter() if tune_busy else None
            rc = LIB.qwn_prefill_chunk(toks, rows, g.pos)
            elapsed = time.perf_counter() - started if tune_busy else None
            operation = "qwn_prefill_chunk"
        if rc != 0:
            self.prefill_errors += 1
            message = f"{operation} width={width} rows={rows} failed rc={rc}"
            self._request_error(message)
            # No host positions or APC snapshots may be published on failure.
            # Fatal native runtime errors require engine-wide cleanup, rather
            # than returning potentially poisoned device state to admission.
            if args.packed_prefill and rc in (-7, -8):
                raise RuntimeError(message)
            for g, _, _, _ in segments:
                self._finish(g, "engine_error")
            return
        if tune_busy and elapsed > 0:
            # Learn only from completed busy waves and their actual native
            # rows, not idle waves or unused/tail-reserved budget. This is
            # a next-wave estimate, not a hard wall-time guarantee.
            target_rows = min(self.chunk_busy,
                              rows * (args.prefill_slice_ms / (elapsed * 1e3)))
            self.prefill_budget_current = max(8, (int(target_rows) // 8) * 8)
        self.prefill_waves += 1
        self.prefill_rows += rows
        self.prefill_width_hist[width] = self.prefill_width_hist.get(width, 0) + 1
        self.prefill_last_width, self.prefill_last_rows = width, rows
        self.prefill_max_width = max(self.prefill_max_width, width)
        self.prefill_max_rows = max(self.prefill_max_rows, rows)
        for g, take, _, deepest in segments:
            g.pos += take
            g.prefill_waves += 1
            g.prefill_rows += take
            g.prefill_wave_rows += rows
            g.prefill_width_hist[width] = g.prefill_width_hist.get(width, 0) + 1
            if g.cancel.is_set():
                self._finish(g, "cancelled")
            elif g.pos == deepest:
                self._store(g.ids, g.pos, g.slot)
            # Use only this wave's remaining prompt-row budget; the existing
            # decode step can produce the first token without another iteration.
            if (g.state == PREFILL and len(g.ids) - g.pos == 1 and
                    rows + len(self.prefill_tails) < budget):
                self.prefill_tails.append(g)

    def _complete_decode_row(self, g, token, width):
        if token < 0:
            self._request_error(
                f"negative decode output slot={g.slot} pos={g.pos} rc={token}")
            self._finish(g, "engine_error")
            return
        g.pos += 1
        if g.cancel.is_set():
            self._finish(g, "cancelled")
            return
        if g.state == PREFILL:
            if g.pos < len(g.ids):
                return  # intermediate remainder logits are not output tokens
            g.state = DECODE
        g.batch_widths.append(width)
        self._emit(g, token)

    def _decode_one(self, g):
        if g.cancel.is_set():
            self._finish(g, "cancelled")
            return
        if not self._select_slot(g):
            return
        t0 = time.perf_counter()
        token = g.ids[g.pos] if g.state == PREFILL else g.last
        nxt = LIB.qwn_decode(token, g.pos)
        dt = time.perf_counter() - t0
        self.dec_time += dt
        if nxt < 0:
            self._request_error(
                f"qwn_decode slot={g.slot} pos={g.pos} failed rc={nxt}")
            self._finish(g, "engine_error")
            return
        self.dec_rows += 1
        self.width_hist[1] = self.width_hist.get(1, 0) + 1
        self.width_time[1] = self.width_time.get(1, 0.0) + dt
        self._complete_decode_row(g, nxt, 1)

    def _decode_step(self):
        for g in self.active:
            if g.state in (PREFILL, DECODE) and g.cancel.is_set():
                self._finish(g, "cancelled")
        rows = ([g for g in self.active if g.state == DECODE] +
                [g for g in self.prefill_tails if g.state == PREFILL])
        # The batched decode ABI accepts continuations only (position >= 1).
        # Bootstrap short prompts individually, consuming their reserved tail
        # row exactly once; exclude them before any call advances positions.
        starting = [g for g in rows if g.pos == 0]
        rows = [g for g in rows if g.pos > 0]
        for g in starting:
            self._decode_one(g)
        if not rows:
            return
        n = len(rows)
        if n == 1:
            # Single decoding row keeps the measured faster GEMV path.
            self._decode_one(rows[0])
            return
        Arr = ctypes.c_int * n
        slots = Arr(*[g.slot for g in rows])
        toks = Arr(*(g.ids[g.pos] if g.state == PREFILL else g.last
                     for g in rows))
        poss = Arr(*[g.pos for g in rows])
        out = Arr(*([0] * n))
        t0 = time.perf_counter()
        rc = LIB.qwn_decode_batch(slots, toks, poss, n, out)
        dt = time.perf_counter() - t0
        self.dec_time += dt
        if rc != n:
            self._request_error(
                f"qwn_decode_batch width={n} failed rc={rc}")
            for g in rows:
                self._finish(g, "engine_error")
            return
        self.dec_rows += n
        self.width_hist[n] = self.width_hist.get(n, 0) + 1
        self.width_time[n] = self.width_time.get(n, 0.0) + dt
        for i, g in enumerate(rows):
            self._complete_decode_row(g, out[i], n)

    def _emit(self, g, tok):
        if int(tok) < 0:
            self._request_error(
                f"negative output token slot={g.slot} rc={int(tok)}")
            self._finish(g, "engine_error")
            return
        now = time.perf_counter()
        if g.t_first is None:
            g.t_first = now
        g.token_times.append(now)
        g.last = int(tok)
        g.n_out += 1
        # EOS terminates the stream and is NOT surfaced: its literal text
        # (e.g. "<|im_end|>") must never reach the client.
        if not g.ignore_eos and int(tok) in EOS_IDS:
            self._finish(g, "stop")
            return
        g.q.put(int(tok))
        if g.n_out >= g.max_new:
            self._finish(g, "length")
        elif g.pos >= args.ctx:
            self._finish(g, "length")
        elif g.cancel.is_set():
            self._finish(g, "cancelled")

    def _finish(self, g, reason):
        if g.state == DONE:
            return
        if reason == "cancelled":
            self.cancelled_requests += 1
        g.state = DONE
        g.q.put(("end", reason))

    def _retire(self):
        still = []
        for g in self.active:
            if g.state != DONE:
                still.append(g)
                continue
            if g.slot >= 0:
                if self.paged and g.owned:
                    self._release_native(g.slot, True)
                    g.owned = False
                self.free.append(g.slot)
                g.slot = -1
        self.active = still

    # ---- startup/failure -------------------------------------------------
    def _initialize_native(self):
        rc = LIB.qwn_init(os.path.expanduser(args.tqf).encode())
        if rc != 0:
            raise RuntimeError(f"qwn_init failed rc={rc}")
        self.initialized = True
        self.nslots = int(LIB.qwn_num_slots())
        self.maxseq = int(LIB.qwn_max_seq())
        if self.nslots <= 0 or self.maxseq <= 0:
            raise RuntimeError(
                f"invalid native limits slots={self.nslots} ctx={self.maxseq}")
        if self.nslots != args.slots:
            raise RuntimeError(
                f"requested slots={args.slots}, native reported {self.nslots}")
        self.paged = bool(LIB.qwn_paged_enabled())
        if self.paged != args.paged:
            raise RuntimeError(
                f"requested {'paged' if args.paged else 'flat'} KV but native "
                f"reported {'paged' if self.paged else 'flat'}")
        if self.paged and args.ctx != self.maxseq:
            raise RuntimeError(
                f"requested ctx={args.ctx}, native reported {self.maxseq}")
        if not self.paged and args.ctx > self.maxseq:
            args.ctx = self.maxseq  # preserve the flat server's old clamp
        self.free = list(range(self.nslots))
        if self.paged:
            layers = int(LIB.qwn_num_layers())
            kv_heads = int(LIB.qwn_num_key_value_heads())
            head_dim = int(LIB.qwn_head_dim())
            full_layers = sum(
                1 for layer in range(layers)
                if LIB.qwn_layer_type(layer) == 2)
            if (layers <= 0 or kv_heads <= 0 or head_dim <= 0 or
                    full_layers <= 0):
                raise RuntimeError(
                    "invalid native shape for paged KV byte accounting: "
                    f"layers={layers}, full={full_layers}, "
                    f"kv_heads={kv_heads}, head_dim={head_dim}")
            # E4M3 K and V bytes plus two FP16 row scales, for every
            # full-attention layer.
            self.kv_bytes_per_token = (
                full_layers * kv_heads * (2 * head_dim + 4))
            self._capture_pool_stats()
            if self.pool_stats["kv_page_tokens"] != args.kv_page:
                raise RuntimeError(
                    f"requested page={args.kv_page}, native reported "
                    f"{self.pool_stats['kv_page_tokens']}")
            if self.pool_stats["kv_sequence_limit"] != self.maxseq:
                raise RuntimeError(
                    "qwn_paged_stats sequence limit disagrees with qwn_max_seq")
            if args.kv_pool_tokens is not None:
                expected = ((args.kv_pool_tokens //
                             self.pool_stats["kv_page_tokens"]) *
                            self.pool_stats["kv_page_tokens"])
                if self.pool_stats["kv_pool_tokens"] != expected:
                    raise RuntimeError(
                        f"requested pool_tokens={args.kv_pool_tokens} "
                        f"(rounded {expected}), native reported "
                        f"{self.pool_stats['kv_pool_tokens']}")
        else:
            self.pool_stats["kv_sequence_limit"] = self.maxseq
        simd16 = os.environ.get("TQ_XPU_ATTN_SIMD16")
        grouped = os.environ.get("TQ_XPU_ATTN_GROUPED")
        dpas = os.environ.get("TQ_XPU_ATTN_DPAS")
        self.attention_config = {
            "simd16": simd16 is None or not simd16.startswith("0"),
            "grouped": grouped is None or not grouped.startswith("0"),
            "dpas": dpas is not None and dpas.startswith("1"),
            "prefill_xmx": args.prefill_xmx,
        }
        self._capture_attention_stats()

    def _terminalize(self, error):
        cleanup_errors = []
        with self.lock:
            waiting = self.pending + self.inbox
            self.pending = []
            self.inbox = []
        for g in self.active:
            if self.paged and g.owned and g.slot not in self.quarantined:
                rc = LIB.qwn_paged_release(g.slot)
                if rc != 0:
                    self._quarantine(g.slot)
                    cleanup_errors.append(
                        f"release slot {g.slot} failed rc={rc}")
                else:
                    g.owned = False
            self._finish(g, "engine_error")
        for g in waiting:
            self._finish(g, "engine_error")
        self.active = []
        for key in list(self.cache_lru):
            entry = self.cache.get(key)
            if entry is None:
                continue
            if self.paged:
                rc = LIB.qwn_paged_ckpt_free(entry[1])
                if rc != 0:
                    cleanup_errors.append(
                        f"free checkpoint {entry[1]} failed rc={rc}")
            else:
                LIB.qwn_host_free(entry[1])
        self.cache.clear()
        self.cache_lru.clear()
        self.cache_lengths.clear()
        self.cache_bytes = 0
        detail = str(error)
        if cleanup_errors:
            detail += "; cleanup: " + ", ".join(cleanup_errors)
        self.terminal_error = detail
        print(f"[serve-xe] TERMINAL ENGINE ERROR: {detail}",
              file=sys.stderr, flush=True)
        self._publish_stats(capture=False)
        self.ready.set()

    def run(self):
        try:
            self._initialize_native()
            self._publish_stats(capture=False)
            self.ready.set()
            while True:
                with self.lock:
                    if self.inbox:
                        self.pending.extend(self.inbox)
                        self.inbox.clear()
                self._admit()
                if self.paged:
                    # Capture reservations before prefill consumes credits so
                    # health retains a true peak for mixed-length runs.
                    self._capture_pool_stats()
                if not self.active:
                    self._publish_stats()
                    time.sleep(0.001)
                    continue
                self._prefill_wave()
                self._decode_step()
                self._retire()
                self._publish_stats()
        except BaseException as exc:
            self._terminalize(exc)


ENG = Engine()
ENG.start()
ENG.ready.wait()
START_STATS = ENG.stats()
if START_STATS.get("status") != "ok":
    sys.exit(f"XPU engine startup failed: {START_STATS.get('engine_error')}")
pool_desc = (
    f"page={START_STATS['kv_page_tokens']} "
    f"pool_tokens={START_STATS['kv_pool_tokens']}"
    if START_STATS["paged"] else "flat"
)
print(f"[serve-xe] ready on :{args.port}  model={args.model_name} "
      f"slots={START_STATS['slots']} ctx={args.ctx} chunk={ENG.chunk} "
      f"apc={'on' if args.prefix_cache else 'off'} "
      f"kv={pool_desc} k64={args.k64 or 'off'} "
      f"eos={sorted(EOS_IDS)}", flush=True)


# ------------------------------ text helpers ------------------------------
def split_think(full, primed):
    """Qwen reasoning split: the chat template opens `<think>\\n` in the
    PROMPT, so output carries reasoning up to </think>."""
    if not primed:
        return None, full
    i = full.find("</think>")
    if i < 0:
        return full, ""
    return full[:i], full[i + len("</think>"):]


def _coerce_arg(val, typ):
    """Coerce an XML-ish parameter STRING to its JSON-schema type. The XML form
    carries no types, so without this every argument arrives as a string and
    clients that declared `integer`/`boolean` params get the wrong thing."""
    s = val.strip() if isinstance(val, str) else val
    if typ in ("integer", "number"):
        try:
            return int(s)
        except (ValueError, TypeError):
            try:
                return float(s)
            except (ValueError, TypeError):
                return val
    if typ == "boolean":
        if isinstance(s, str) and s.lower() in ("true", "false"):
            return s.lower() == "true"
        return val
    if typ in ("array", "object"):
        try:
            return json.loads(s)
        except Exception:
            return val
    if typ == "string":
        return val
    try:
        j = json.loads(s)
        return j if isinstance(j, (int, float, bool, list, dict)) or j is None else val
    except Exception:
        return val


def parse_tool_calls(text, tools=None):
    """Qwen tool-call formats, BOTH supported (mirrors tools/serve_openai.py):
    (a) JSON:    <tool_call>{"name":..., "arguments": {...}}</tool_call>
    (b) XML-ish: <tool_call><function=NAME>
                 <parameter=KEY>VALUE</parameter>...</function></tool_call>
    The chat template emits (b) for this model, and parsing only (a) leaked the
    raw block into `content` - clients then saw no tool_calls and stopped.
    XML values are type-coerced against the request's declared schema.
    Returns (clean_text, tool_calls_list_or_None)."""
    import re as _re
    types = {}
    for t in (tools or []):
        fn = t.get("function", t) if isinstance(t, dict) else {}
        props = (((fn.get("parameters") or {}).get("properties")) or {})
        types[fn.get("name", "")] = {k: (v or {}).get("type") for k, v in props.items()}
    calls = []

    def _emit(name, args):
        calls.append({"id": "call_" + uuid.uuid4().hex[:20],
                      "type": "function",
                      "function": {"name": name,
                                   "arguments": json.dumps(args, ensure_ascii=False)}})

    def _take(m):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
            _emit(obj.get("name", ""), obj.get("arguments", {}))
            return ""
        except Exception:
            pass
        fm = _re.search(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", raw, _re.S)
        if fm:
            name = fm.group(1)
            ptypes = types.get(name, {})
            args = {}
            for pm in _re.finditer(
                    r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", fm.group(2), _re.S):
                args[pm.group(1)] = _coerce_arg(pm.group(2), ptypes.get(pm.group(1)))
            _emit(name, args)
            return ""
        return m.group(0)   # unparseable: leave verbatim in content

    clean = _re.sub(_re.escape(TOOL_OPEN) + r"(.*?)" + _re.escape(TOOL_CLOSE),
                    _take, text, flags=_re.S)
    return clean.strip(), (calls or None)


def first_stop(text, stops):
    hit = -1
    for s in stops:
        if not s:
            continue
        j = text.find(s)
        if j >= 0 and (hit < 0 or j < hit):
            hit = j
    return hit


# --------------------------------- HTTP -----------------------------------
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _peer_closed(self):
        """Observe a remote close without consuming pipelined request bytes."""
        try:
            readable, _, _ = select.select((self.connection,), (), (), 0)
            if not readable:
                return False
            flags = socket.MSG_PEEK | socket.MSG_DONTWAIT
            return self.connection.recv(1, flags) == b""
        except (BlockingIOError, InterruptedError):
            return False
        except (ConnectionResetError, OSError, ValueError):
            return True

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.rstrip("/")
        if p == "/health":
            snapshot = ENG.stats()
            return self._json(200 if snapshot["status"] == "ok" else 503,
                              snapshot)
        if p in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": args.model_name, "object": "model",
                 "owned_by": "knivesysl-xe"}]})
        return self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": {"message": f"bad json: {e}"}})
        p = self.path.rstrip("/")
        chat = p.endswith("/chat/completions")
        comp = p.endswith("/completions") and not chat
        if not (chat or comp):
            return self._json(404, {"error": {"message": "not found"}})
        if req.get("n", 1) != 1 or req.get("logprobs"):
            return self._json(400,
                              {"error": {"message": "n>1/logprobs unsupported"}})

        warns = []
        if float(req.get("temperature", 0.0)) not in (0.0,):
            warns.append("temperature ignored: XPU tier is greedy-only (argmax)")
        if float(req.get("top_p", 1.0)) != 1.0:
            warns.append("top_p ignored: greedy-only")
        if req.get("seed") is not None:
            warns.append("seed ignored: greedy-only")

        max_new = int(req.get("max_tokens") or
                      req.get("max_completion_tokens") or 256)
        stops = req.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        ignore_eos = bool(req.get("ignore_eos", False))
        tools = req.get("tools") or None

        primed = False
        if chat:
            msgs = list(req.get("messages", []))
            for m in msgs:
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"])
                        except Exception:
                            pass
            ckw = dict(req.get("chat_template_kwargs") or {})
            # vLLM accepts `enable_thinking` as a TOP-LEVEL field (that is what
            # openai-python's extra_body= produces, e.g. tools/axe_vllm.py's
            # --no-thinking). Fold it in so the flag is not silently ignored;
            # an explicit chat_template_kwargs entry still wins.
            if "enable_thinking" in req and "enable_thinking" not in ckw:
                ckw["enable_thinking"] = bool(req["enable_thinking"])
            try:
                text = TOK.apply_chat_template(
                    msgs, tools=tools, add_generation_prompt=True,
                    tokenize=False, **ckw)
                ids = TOK(text, add_special_tokens=False).input_ids
            except Exception as e:
                return self._json(400,
                                  {"error": {"message": f"chat template: {e}"}})
            primed = text.rstrip().endswith("<think>")
        else:
            ids = TOK(req.get("prompt", ""), add_special_tokens=False).input_ids
        if not ids:
            ids = TOK("\n", add_special_tokens=False).input_ids[-2:]
        if ENG.stats().get("paged"):
            max_new = max(1, max_new)
            problem = ENG.request_error(len(ids), max_new)
            if problem is not None:
                code, message = problem
                return self._json(code, {"error": {"message": message}})
        else:
            if len(ids) >= args.ctx:
                return self._json(400, {"error": {"message":
                    f"prompt {len(ids)} tokens exceeds ctx {args.ctx}"}})
            # Preserve the shipping flat path's output clamp exactly.
            max_new = max(1, min(max_new, args.ctx - len(ids)))

        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        stream = bool(req.get("stream", False))
        g = Gen(ids, max_new, ignore_eos)
        submit_error = ENG.submit(g)
        if submit_error is not None:
            return self._json(
                503, {"error": {"message":
                      f"XPU engine unavailable: {submit_error}"}})

        # ---- consume ----------------------------------------------------
        def pump():
            """Yield (delta_text, finish_reason|None). Owns stop-string cuts."""
            buf = []
            active_stops = [stop for stop in stops if stop]
            max_hold = (max(len(stop) for stop in active_stops) - 1
                        if active_stops else 0)
            held = ""
            idle_deadline = (time.monotonic() +
                             args.response_idle_timeout)
            while True:
                remaining = idle_deadline - time.monotonic()
                if remaining <= 0:
                    g.cancel.set()
                    yield "", "timeout"
                    return
                try:
                    item = g.q.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    if self._peer_closed():
                        g.cancel.set()
                        yield "", "cancelled"
                        return
                    continue
                idle_deadline = (time.monotonic() +
                                 args.response_idle_timeout)
                if isinstance(item, tuple):
                    tail = TOK.decode(buf) if buf else ""
                    if active_stops:
                        held += tail
                        match = first_stop(held, active_stops)
                        if match >= 0:
                            g.cancel.set()
                            if match:
                                yield held[:match], None
                            yield "", "stop"
                            return
                        if held:
                            yield held, None
                    elif tail:
                        yield tail, None
                    yield "", item[1]
                    return
                buf.append(item)
                txt = TOK.decode(buf)
                if txt.endswith("\ufffd"):
                    continue          # incomplete utf-8, hold the tail
                buf = []
                if not active_stops:
                    yield txt, None   # no-stop fast path: no string history
                    continue
                held += txt
                match = first_stop(held, active_stops)
                if match >= 0:
                    g.cancel.set()
                    if match:
                        yield held[:match], None
                    yield "", "stop"
                    return
                flush = len(held) - max_hold
                if flush > 0:
                    yield held[:flush], None
                    held = held[flush:]

        if not stream:
            parts, finish = [], "stop"
            for delta, fin in pump():
                parts.append(delta)
                if fin:
                    finish = fin
                    break
            full = "".join(parts)
            reasoning, content = split_think(full, primed)
            calls = None
            if tools:
                content, calls = parse_tool_calls(content, tools)
            usage = {"prompt_tokens": len(ids), "completion_tokens": g.n_out,
                     "total_tokens": len(ids) + g.n_out}
            effective = ENG.stats()
            xk = {"engine": "knivesysl-xe",
                  **g.metrics(),
                  "reused_tokens": g.reused,
                  "prefilled_tokens": len(ids) - g.reused,
                  "kv_layout": effective["kv_layout"],
                  "attention_config": effective["attention_config"],
                  "attention_branch_counts":
                      effective["attention_branch_counts"],
                  "tier": effective["tier"],
                  "apc_enabled": effective["apc_enabled"],
                  "k64": effective["k64"],
                  "greedy": True}
            if warns:
                xk["warnings"] = warns
            if chat:
                msg = {"role": "assistant", "content": content or None}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                if calls:
                    msg["tool_calls"] = calls
                out = {"id": rid, "object": "chat.completion", "created": created,
                       "model": args.model_name, "choices": [
                           {"index": 0, "message": msg,
                            "finish_reason": "tool_calls" if calls else finish}],
                       "usage": usage, "x_knivesysl": xk}
            else:
                out = {"id": rid, "object": "text_completion", "created": created,
                       "model": args.model_name, "choices": [
                           {"index": 0, "text": full, "finish_reason": finish,
                            "logprobs": None}],
                       "usage": usage, "x_knivesysl": xk}
            return self._json(200, out)

        # ---- streaming (SSE over chunked transfer) ----------------------
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def sse(obj):
            data = f"data: {json.dumps(obj)}\n\n".encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def chunk(delta, finish=None):
            if chat:
                return {"id": rid, "object": "chat.completion.chunk",
                        "created": created, "model": args.model_name,
                        "choices": [{"index": 0, "delta": delta,
                                     "finish_reason": finish}]}
            return {"id": rid, "object": "text_completion",
                    "created": created, "model": args.model_name,
                    "choices": [{"index": 0, "text": delta.get("content", ""),
                                 "finish_reason": finish, "logprobs": None}]}

        try:
            if chat:
                sse(chunk({"role": "assistant"}))
            finish, acc, raw = "stop", [], ""
            # TWO streaming state machines over the raw token text.
            #
            # (1) REASONING. The chat template opens `<think>` in the PROMPT, so
            #     generation STARTS inside reasoning with no opening tag and
            #     ends at `</think>`. Non-streaming calls split_think() on the
            #     finished text; streaming cannot, so without this the reasoning
            #     goes out as `content` and the literal `</think>` leaks to the
            #     client (observed in axe_vllm: reasoning printed under
            #     `assistant>` with a bare `</think>` in the middle).
            # (2) TOOL BLOCK. The raw <tool_call> markup must not reach the
            #     client as content, or a client that also gets the parsed
            #     tool_calls puts BOTH into history and the model reads its own
            #     markup back next turn.
            #
            # Both hold back len(marker)-1 chars so a delta ending mid-marker
            # can never be emitted, and both flush text UP TO the marker before
            # going quiet so no real characters are lost. `acc` collects
            # everything; only what we SEND is gated.
            THINK_CLOSE = "</think>"
            hold_t = len(THINK_CLOSE) - 1
            hold_c = len(TOOL_OPEN) - 1 if tools else 0
            in_think = bool(primed)
            cut = 0 if not primed else -1     # raw index where content begins
            sent_r, sent_c, in_tool = 0, 0, False
            for delta, fin in pump():
                if delta:
                    acc.append(delta)
                    raw += delta
                    if in_think:
                        # vLLM's Qwen3ReasoningParser: "Qwen3.5 models may emit
                        # <tool_call> inside the thinking block without closing
                        # </think> first. <tool_call> is treated as an implicit
                        # end of reasoning." Without this, a tool call emitted
                        # inside thinking is swallowed as reasoning and NEVER
                        # parsed. Whichever marker lands first ends reasoning;
                        # </think> is consumed, <tool_call> is NOT (it belongs
                        # to the content phase for parse_tool_calls).
                        from_i = max(0, sent_r - max(hold_t, hold_c))
                        j = raw.find(THINK_CLOSE, from_i)
                        k = raw.find(TOOL_OPEN, from_i) if tools else -1
                        if k >= 0 and (j < 0 or k < j):
                            if k > sent_r:
                                sse(chunk({"reasoning_content": raw[sent_r:k]}))
                            sent_r = k
                            cut = k                      # marker kept
                            sent_c = k
                            in_think = False
                        elif j >= 0:
                            if j > sent_r:
                                sse(chunk({"reasoning_content": raw[sent_r:j]}))
                            sent_r = j
                            cut = j + len(THINK_CLOSE)   # tag itself is dropped
                            sent_c = cut
                            in_think = False
                        else:
                            # hold back enough for EITHER marker to complete
                            upto = len(raw) - max(hold_t, hold_c)
                            if upto > sent_r:
                                sse(chunk({"reasoning_content": raw[sent_r:upto]}))
                                sent_r = upto
                    if not in_think and not in_tool:
                        p = (raw.find(TOOL_OPEN, max(cut, sent_c - hold_c))
                             if tools else -1)
                        if p >= 0:
                            if p > sent_c:
                                sse(chunk({"content": raw[sent_c:p]}))
                                sent_c = p
                            in_tool = True
                        else:
                            upto = len(raw) - hold_c
                            if upto > sent_c:
                                sse(chunk({"content": raw[sent_c:upto]}))
                                sent_c = upto
                if fin:
                    finish = fin
                    break
            full = "".join(acc)
            _reasoning, _content = split_think(full, primed)
            calls = None
            if tools and TOOL_OPEN in _content:
                _, calls = parse_tool_calls(_content, tools)
            # Flush whichever phase we ended in.
            if in_think:
                if len(full) > sent_r:
                    sse(chunk({"reasoning_content": full[sent_r:]}))
            elif not in_tool and len(full) > sent_c:
                sse(chunk({"content": full[sent_c:]}))
            if calls:
                sse(chunk({"tool_calls": [
                    {"index": i, **c} for i, c in enumerate(calls)]}))
                finish = "tool_calls"
            sse(chunk({}, finish))
            usage = {"prompt_tokens": len(ids), "completion_tokens": g.n_out,
                     "total_tokens": len(ids) + g.n_out}
            effective = ENG.stats()
            tail = {"id": rid, "object": "chat.completion.chunk" if chat
                    else "text_completion", "created": created,
                    "model": args.model_name, "choices": [], "usage": usage,
                    "x_knivesysl": {"engine": "knivesysl-xe",
                                    **g.metrics(),
                                    "reused_tokens": g.reused,
                                    "prefilled_tokens":
                                        len(ids) - g.reused,
                                    "kv_layout": effective["kv_layout"],
                                    "attention_config":
                                        effective["attention_config"],
                                    "attention_branch_counts":
                                        effective["attention_branch_counts"],
                                    "tier": effective["tier"],
                                    "apc_enabled": effective["apc_enabled"],
                                    "k64": effective["k64"],
                                    "greedy": True,
                                    **({"warnings": warns} if warns else {})}}
            sse(tail)
            data = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            g.cancel.set()


if __name__ == "__main__":
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()
