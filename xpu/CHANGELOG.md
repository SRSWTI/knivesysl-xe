# knivesysl XPU changelog

All Intel GPU work is isolated under `xpu/`. The CUDA engine in
`src/forward_qwen.cu`, its build, and its production server remain unchanged.

## Unreleased — Intel Arc Pro B70 bring-up

### Measured native optimization campaign (2026-09-08)

Accepted campaign binary: `build/optimization-20260908-142035/libfinal.so`,
SHA-256 `508c9cbc8ed72e16d7dafff550d49b3a874de637e2557867fbe7aeab07eb7685`.
Frozen baseline: `libbaseline.so`, SHA-256
`46343dbdb682e935bdd512b98aa1e2182978b2f74778ace13a7dabfb24919175`.
All evidence names in this section are relative to that local campaign directory;
generated libraries and raw build receipts are not distributed with the source.
The tracked [portable campaign evidence summary](results/optimization-20260908.json)
retains samples, library/source identities, correctness gates, HTTP/SSE metrics,
the rejected RC32 decision, and limitations without requiring that local directory.

Retained changes:

- RC16 projection tiles reuse quantized weights across more activation rows;
  consumer-specific staging avoids producing activation layouts that the next
  projection does not consume. Existing quantization tiers remain available.
- GDN loop invariants are hoisted without reassociating the recurrence arithmetic.
  Convolution-tail advancement fixes overlapping read/write ownership for short
  tails. The faster reassociated GDN proposals below were not retained.
- SiLU/multiply and down-projection quantization are fused, preserving the required
  hidden output and consumer quantization. `silu-fusion-identity.log` records
  hidden/output exactness for the exercised W8 and W4 K16/K32/K64 synthetic cases;
  it is not proof of every shape or input.
- Argmax can remain queued until its consumer needs the result; batched execution
  and reset avoid unnecessary queue synchronizations. The existing attention
  implementation remains unchanged after the measured alternatives below.
- Optional `--prefill-slice-ms` / launcher `PREFILL_SLICE_MS` adapts busy-prefill
  budgets from measured wave cost. **Default zero** preserves static scheduling;
  the original 8-row budget rounding is retained. It is an estimate, not a deadline
  or GPU preemption mechanism.

RC32 was **rejected and removed**, despite passing the sampled exactness gate
and improving the W4A4 T512 projection from 918.7 to 783.4 microseconds in the
same-card control. Whole-model prefill showed no consistent improvement:
`rc32-prefill.json` measured about 14.417 s at 8192 (two-run mean) and 37.619 s
at 16384 (single run), versus accepted RC16 14.349 s and 37.431 s.
The headline results and accepted binary remain RC16; a microbenchmark win was
not treated as a whole-model win.

#### Native timing boundary and results

`quiet-baseline-prefill.json` and `quiet-final-prefill.json` are the precise,
uninstrumented `bench_prefill.py` receipts, with matching invocation JSON/logs.
Both use the same TQF, K64 gate/up/down, automatic XMX, page128, two state slots,
16384-token context/pool, and 512-token chunks on an isolated B70. The synthetic
prompt repeats a fixed token template. The timer includes native prefill calls
and Python/ctypes chunk-array construction, but excludes model initialization,
reset and prompt construction. It includes no HTTP, tokenization, admission,
or output streaming. This is a whole-model call-loop timer, not GPU kernel time.

| Prompt tokens | Samples/build | Baseline seconds | Accepted seconds | Baseline → accepted tokens/s | Latency reduction |
|---:|---:|---:|---:|---:|---:|
| 512 | 2 | 0.835939 | 0.649505 | 612.5 → 788.3 | 22.3% |
| 2048 | 2 | 3.542637 | 2.779666 | 578.1 → 736.8 | 21.5% |
| 8192 | 2 | 17.256871 | 14.349054 | 474.7 → 570.9 | 16.9% |
| 16384 | **1** | 43.115557 | 37.430914 | 380.0 → 437.7 | 13.2% |

Smaller rows are means of two timings per build; tokens/s is prompt length divided
by mean seconds. The 16k row is **one measurement per build**, not repeated paired
evidence or a confidence interval. Earlier instrumented/profile runs are diagnostic
and are not silently substituted for these quiet timings.

#### Paired HTTP and interference measurements

`baseline-http-pair-current-metadata.json` versus `final-http-pair.json` uses
the same cold 2048/4096-token pair, 32 output tokens per request, APC off, four
slots, 16384 context, 32768-token page128 pool, K64 MLP, automatic XMX, packed
prefill, busy/idle budgets 64/512, and time target zero. Two repetitions reverse
the request launch order. Means:

| Metric | Baseline | Accepted |
|---|---:|---:|
| 2048 request TTFT | 8.376 s | 6.864 s |
| 4096 request TTFT | 14.198 s | 12.049 s |
| Complete pair wall time | 15.206 s | 13.055 s |
| 2048 request reported ITL p50 / p99 | 215.4 / 236.1 ms | 191.0 / 208.8 ms |

TTFT/ITL are server request metadata, not native prefill timings. TTFT measures
submission-to-first-token within the server lifecycle; pair wall time is client
elapsed time. The ITL entries average per-request percentiles, not pooled samples.
The corresponding TTFT reductions are 18.1% and 15.1%, with 14.1% lower pair wall
time. Later uncontended 4096-request ITL p50 remains about 32.4 ms in both arms.

`safe-static-mixed.json` and `final-target100-mixed.json` separately measure a
64-prompt/96-output stream receiving a 2048-prompt/16-output competitor after its
eighth nonempty SSE event. Three matched control/mixed repeats alternate order.
Pooled client decode text-event gaps during incoming prefill changed from
**140.3/165.2 ms p50/p99** at target zero to **120.0/136.8 ms** at target 100 ms,
while incoming mean client TTFT rose from **5.012 to 6.360 s** (about 27%).
These SSE events are not native token timestamps: buffering and empty token text
affect event counts. The target does not guarantee 100 ms event latency.
The separate tile-rounded experiment (`final-target100-tile-mixed.json`) reduced
p99 to 129.1 ms but raised incoming mean TTFT to 7.560 s; that rounding change
was not retained. This is why the controller remains opt-in, not a default win.

#### Correctness gates and rejected experiments

- `final-cross-build-exact.json`: finite values and **96/96 greedy plus 96/96
  teacher-forced tokens**, no sampled logit-bit mismatch, against the frozen
  baseline on heap/graph/transactions continuations. This proves only the sampled
  cross-build configuration, not all inputs, all tiers, or scalar/XMX equivalence.
- `final-native-page256.json`: 21 production-numerics cases passed and scalar/XMX
  aggregate agreement 90/96. `final-native-flat-fixed.json`: 31 fixed-split flat
  cases passed and aggregate 91/96. Graph remains 28/32 in both, below a per-prompt
  90% gate; the retained policy is explicitly aggregate ≥90%. Logical KV exactness
  is scoped to each report's structural mode; width-two logits expose only the
  last row through the public ABI. The earlier failed flat invocation is not a
  passing gate; the named fixed-split report is the qualifying run.
- `final-http-regression.json`: seven scenarios passed, including short/ragged
  prompts, packed three-wave APC reuse, exact streaming counts, split stops,
  structured tools, and disconnect during admission/decode with recovery.
- Parallel/reassociated GDN was fast but failed the existing full-model numerical
  gate: `integrated-native-production.json` scored **83/96 (86.46%)** and
  `gdn-dots-native-production.json` **82/96 (85.42%)**. The synthetic 128-row
  chunk median fell from 1.033 ms (`gdn-baseline-128.log`) to 0.543 ms
  (`gdn-parallel-128.log`) or 0.362 ms (`gdn-dots-norm-128.log`), but output/state
  hashes changed. Passing a small relative-error kernel probe was insufficient.
  Exact-order vectorization retained those baseline hashes but slowed the chunk
  to 1.298 ms (`gdn-vector-128.log`), so it was not retained either.
- Attention experiments passed their direct probes but did not justify promotion.
  At position 8192/query count 512, clean control XMX was 32.481 ms; four subgroups
  took 39.888 ms, broadcast 40.476 ms. Cached K took 34.011 ms and increased the
  packed pair from 59.922 to 69.319 ms despite a small paged-single improvement.
  Sources: `attention-clean-control-8192-512.log`,
  `attention-clean-nsg4-8192-512.log`, `attention-clean-broadcast-8192-512.log`,
  `attention-kcache-8192-512.log`; Q-cache and six-subgroup receipts remain in
  `attention-cacheq-8192-512.log` and `attention-nsg6-8192-512.log`.

Reproducible native and HTTP commands using shipped tools are in
[`readme.md`](../readme.md#reproduce-the-campaign-workloads). A fresh baseline
reference can be created with `compare_native_builds.py --record`; neither CUDA
nor an archived development-workspace reference is required. Run model-owning
tools sequentially on a free explicitly selected card. No new vendor head-to-head,
hardware-ceiling determination, universal-exactness claim, or 75k full-model
performance claim follows from this campaign.

### XMX prefill and packed prompt execution (2026-09-07)

- Added native Xe2 XMX attention in `src/kernels_prefill.cpp`: DPAS for both
  QK and PV, FP32 softmax/accumulators, and three BF16 components for Q and
  scaled probabilities. Raw FP32 Q bits are retained across the shared-memory
  staging split. Existing E4M3 KV bytes and per-token/head FP16 scales are
  unchanged; no PyTorch dependency or full-context cache conversion was added.
- `qwn_prefill_batch` executes one layer walk over packed prompt rows. Projection
  GEMMs and MLPs combine requests; GDN state remains slot-local. W4A8 GEMMs retain
  each segment's original split-K choice and reduction order rather than letting
  the larger packed shape silently change its arithmetic.
- `x_prefill_attn_packed` combines requests in one attention grid after all their
  KV writes. Its bounded request descriptors are captured by value; temporary
  host descriptor storage is not retained by the device.
- `PREFILL_XMX=auto` and `PACKED_PREFILL=1` are launcher defaults. Auto keeps
  scalar attention for unprofitable small work; `PREFILL_XMX=0` is the scalar
  reference and `1` requires XMX. `PACKED_PREFILL=0` selects serial prompt waves.
  `CHUNK` and `CHUNK_IDLE` remain total wave budgets, not per-request budgets.
- Prompt tails use the busy budget, are advanced without emitting intermediate
  logits, and bootstrap position zero through scalar decode before positive
  positions enter decode batching. Nonempty one-token prompts are no longer
  padded. APC, cancellation, slot isolation, and fixed output accounting remain.

Numerics are deliberately **not bit-identical to scalar attention**. The user
approved the existing >=90% aggregate reference-token agreement policy with
per-prompt results retained. Production AUTO-split qualification measured
90/96 (93.75%): heap 31/32, graph 28/32, transactions 31/32. Packed versus
independent execution matched 64/64. Fixed-split flat/page256 runs measured
91/96. These are sampled token-agreement measurements, not task-accuracy scores.
The stricter per-prompt failures were not erased: even a one-ULP perturbation
of the original scalar attention made graph/transactions score 28/32 in the
positive-ULP control. Actual-input diagnostics found S8 quantization threshold
crossings after attention differences around 1e-7.

Qualification: 11 direct-kernel cases cover ragged queries, reversed page maps,
128/256-token pages, the final 131072-token boundary, automatic dispatch, and
packed descriptor lifetime. Relative L2 and peak-normalized error gates are
1e-4; flat/paged and packed/single XMX comparisons are bit-exact. Native suites
passed 21 production, 31 flat, and 34 page256 cases. HTTP regression covers
1/8/9/16/17-token prompts, packed waves, three-wave APC reuse, streaming counts,
split stops, structured tools, and disconnect/recovery. A real `axe_vllm.py`
bash-tool round trip also passed.
The final eight-slot candidate and promoted endpoint each passed all seven HTTP
scenarios. Eight simultaneous 256-token requests each returned exactly 16 output
tokens, with four width-8 packed prefill waves and 23 width-8 decode steps observed.

Paired HTTP comparison: one B70, same library and K64-MLP weights, two concurrent
4096/8192-token prompts, 32 generated tokens each, no prefix cache, two repetitions
with reversed request launch order. Means:

| metric | scalar + serial prompt waves | auto XMX + packed waves |
|---|---:|---:|
| 4096-token request TTFT | 19.905 s | 16.802 s |
| 8192-token request TTFT | 35.161 s | 29.908 s |
| complete pair wall time | 36.263 s | 31.009 s |

This is about 15% lower TTFT and 14.5% lower pair completion time, not a claim
that whole-model prefill accelerates by the isolated kernel's 2.5-2.7x. Decode
ITL while competing with prefill remains about 273 ms in this pair; the later
uncontended stream measured 35.3 ms. No new 75k full-model run or vendor
head-to-head was performed.

Evidence, including rejected candidates and the numerical-policy decision,
lives in `build/prefill-results-20260907-014701/`: `summary.json`,
`final-comparison.csv`, `final-http-benchmark.json`, `kernel-summary.json`,
and the `qualified-*.json` native reports. The scalar/reference modes remain
available for reproducible comparisons; legacy parity identities now include
`TQ_XPU_PREFILL_XMX` so incompatible arithmetic cannot share a reference.

Deployment verified on `127.0.0.1:8101`: the qualified binary is installed as
`build/libforward_qwen_xpu.so`, with eight slots, a 131072-token per-sequence limit,
a shared 131072-token pool, page size 128, K64-MLP weights, APC enabled, automatic
XMX selection, and packed prefill enabled. The promoted endpoint exercised both
scalar and XMX prefill, reused APC prefixes, and returned to idle with no request
or engine errors. Pool capacity is shared, not 131072 tokens reserved per slot.
Receipts: `deployment.json`, `deployed-http-regression.json`,
`final-http-isolated.json`, and `final-eight-way.json` in the evidence directory.

The first eight-slot HTTP run is retained but excluded from qualification: an
accidental duplicate candidate launch loaded another model on the same card and
then failed to bind the occupied port. Isolated reruns passed the unchanged tests.
Final cutover checked process ownership of the listening socket, not merely a
successful health response from an existing server.

### Serving fixes found by driving a real client (2026-09-05)

Pointed `tools/axe_vllm.py` (the agentic client, unmodified) at
`xpu/tools/serve_openai_xpu.py` and fixed what broke. Every one of these was
invisible to `serve_smoke_xpu.py`, which drives our own request shapes.

- **XML tool-calls were dropped.** The chat template for this model emits
  `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`,
  but `parse_tool_calls` only accepted the JSON form, so the block leaked into
  `content`, the response carried no `tool_calls`, and the client treated a
  tool request as a final answer and stopped. Ported the dual-format parser
  plus `_coerce_arg` from `tools/serve_openai.py` (the CUDA server already had
  it) so XML values are type-coerced against the request's declared schema
  rather than arriving as strings. Verified end to end: model emitted a `bash`
  call, client executed `wc -l < xpu/build.sh`, model consumed `25` and
  answered correctly.
- **`enable_thinking` was silently ignored.** The server read it only from
  `chat_template_kwargs`, while openai-python's `extra_body=` (what
  `axe_vllm.py --no-thinking` uses, and what vLLM accepts) puts it at the TOP
  level. Now folded in, with an explicit `chat_template_kwargs` entry still
  winning.
- **Bound to loopback.** `--host` defaulted to `0.0.0.0` on an endpoint with no
  authentication; `serve_xe.sh` now defaults `HOST=127.0.0.1` and every client
  we drive it with is local. Set `HOST=0.0.0.0` deliberately if ever needed.
- **Card pinned at the driver level.** `serve_xe.sh` now exports
  `ZE_AFFINITY_MASK=$CARD` (with `TQ_XPU_DEV=0` selecting rank 0 within the
  masked list), matching the rule `serve_vllm_xpu.sh` already carried. This
  makes the other B70 invisible to Level Zero so the process cannot touch it.
  It does NOT suppress the kernel's `"ACS redirect is set between the client
  and provider"` lines - those come from the `xe` driver evaluating peer
  capability and appear while both cards are bound, regardless of the mask.
- **Launcher watchdog false negative (self-inflicted).** Adding `--host` ahead
  of `--port` broke `pgrep -f "serve_openai_xpu.py --port $PORT"`, so the
  launcher reported "died during load" while the server was loading fine and
  went on to serve. Now matches the script name only.

**Open:** the raw `<tool_call>` block is still streamed into `content` before
the parsed `tool_calls` arrive in the final chunk, so conversation history
carries both. The CUDA server buffers the block to the end of the stream
(`dec_state["tool"]`); the XPU streaming path does not yet.

### Host stability — a named kernel WARN, first hard evidence (2026-09-05)

A third hard cut at 17:54:39 produced the session's only kernel stack trace:

```
WARNING: drivers/gpu/drm/xe/xe_force_wake.c:212 at xe_force_wake_get+0x2e0/0x320 [xe]
CPU: 4  Comm: nvtop
  xe_force_wake_get <- xe_guc_pc_get_cur_freq <- cur_freq_show [xe]
  <- sysfs_kf_seq_show <- vfs_read <- __x64_sys_read
```

`nvtop` reading the `cur_freq` sysfs attribute on an xe device; force-wake of
the GT failed and the driver warned. Per-boot counts: 0, 0, **4**, 0 - present
only in the boot that died, four times, seconds before it ended.

**This does not close the question.** Two of the three hard cuts carried ZERO
`xe_force_wake` warnings, so nvtop cannot explain them, and even here
force-wake failing is equally consistent with nvtop wedging a power-gated card
OR with the card already being sick and nvtop's read being the first thing to
notice. Cause remains unestablished. Cheap mitigation adopted: do not run
nvtop or any xe-sysfs-polling monitor while the B70s are working.

Two corrections to earlier entries: **kdump IS armed**
(`kexec_crash_loaded: 1`, `kdump-tools` active, and the oops header itself says
`Kdump: loaded`) - but it fires on panic, and these are WARN-then-hang, so no
vmcore is written and `/var/crash` holds only files from Sep 2-4. And the
`fput_close_sync`/`__x64_sys_close` frames first seen in that trace are
unreliable `?` entries: the real syscall is `read` on sysfs, not fd teardown
from a `pkill`.

### Strategy — campaign doc and recon

- Added `xpu/docs/level-up-xpu.md`, the living strategy document: machine budgets,
  the SLO contract (including the concurrency-at-large-context mandate), six
  theses, the build-vs-take split, the priced microbench queue with kill criteria,
  and the append-only lessons log. Update protocol is in the doc; nothing tried is
  ever deleted from it.
- Archived the Intel GPU Optimization Guide (2025.2) per chapter under
  `xpu/docs/gpu-optim-intel/archive/pages/` and read it end to end against our
  measurements. Known defects recorded in the doc: 13 chapters truncated at 50 KB
  by the scraper, `overview` page absent, "Immediate Command Lists" chapter only
  upstream.
- Cloned `vendor/PerfSpect` and completed a 20-repo recon of the Intel vendor
  tree (triton backend, sycl-tla, vllm-xpu-kernels, sgl-kernel-xpu, llm-scaler,
  b70_ai_things, oneDNN, oneCCL, compute-runtime, level-zero, IGC, intel/llvm,
  cookbook, gpu-ai-skills, Xe-Forge/Xe-Fuse, oneAPI-samples, and more); the
  claim-by-claim reference index lives in `level-up-xpu.md` §8. Headline verdicts:
  TP2 stays dead (oneCCL source), vendor graph capture structurally cannot span
  GDN layers, joint_matrix on BMG has no INT4 (ESIMD s4 DPAS is the sanctioned
  route), and SYCL-Graph host-synchronizes before every replay.

### Split-K for the batched RC8 GEMM (2026-09-05) — W4A8 only

The batched path derived ALL parallelism from T (`tasks = rgroups * tgroups`),
so at the decode batch shape (T=8, one t-group) the small-M projections
starved: `mlp_down` (M=5120) generated 320 tasks = 20 work-groups against a
machine that wants >=128, ~16% lane occupancy. Measured consequence: the
batched step ran at 46% of byte-roofline where the GEMV path reaches 87%.

- Added `choose_gemm_splits(rgroups, tgroups, ktiles)` and split-K to
  `x_gemm_w4a8` in `kernels_dpas.cpp`. Mirrors the proven GEMV scheme: tasks
  ordered `(rg*tgroups+tg)*splits + split`, each subgroup's 8x16 fp32
  accumulator staged in SLM lane-fastest (16 sg x 8 out x 16 lanes x 4 B =
  8 KiB/WG, conflict-free on the 16-bank SLM), one barrier, then the first
  `16/splits` subgroups reduce. **No global partial buffer and no second
  launch.** `splits` is constrained to divisors of the subgroup count
  ({1,2,4,8,16}) so `gpw` is exact, capped by k-tile count, and disabled when
  the unsplit grid already saturates.
- `TQ_XPU_GEMM_SPLITS` overrides the heuristic (1 = forced unsplit baseline,
  2/4/8/16 = forced, unset = heuristic) so every claim below has a
  **same-binary** baseline and the 128-work-group target can be rejected.
- **Numerics: this is NOT bit-identical to the unsplit kernel.** Partitioning
  K reassociates the fp32 accumulation. Gated on tolerance, not byte
  equality: `gemm_check` PASS at cos 1.00000000 for forced splits 1/2/4/8/16
  (max_rel 0.0065/0.0055/0.0040/0.0041/0.0037 — tolerance compliance across
  all branches; the reference is the GEMV path, not a high-precision oracle,
  so no numerical-quality claim is made from the spread).
- `batch_check` (batched vs serial single-slot) PASS 48/48 on both slots with
  split-K active AND with splits forced off — the eps-level reassociation
  shifts no argmax.

Paired A/B, one binary, alternating arms in REVERSED order, 3 trials,
`bench_batch --ctx 512`, card 1, `TQ_XPU_SLOTS=8`:

| n | baseline ms/step | split-K ms/step | baseline agg tok/s | split-K agg tok/s |
|--:|--:|--:|--:|--:|
| 1 | 58.15 | 40.98 | 17.20 | 24.40 |
| 2 | 60.05 | 42.95 | 33.31 | 46.56 |
| 4 | 63.88 | 47.20 | 62.62 | 84.75 |
| 8 | **71.13** | **55.01** | **112.47** | **145.42** |

n=8 medians over 3 alternating trials; spread 0.04% (split-K 55.00/55.02/55.01,
baseline 71.13/71.13/71.13), so **1.293x / +29.3% aggregate** is not warm-cache
or clock drift. Single-stream GEMV unchanged at 30.47 ms / 32.82 tok/s, and
decode re-gated after the change: `bench_decode` 33.281 tok/s (p99 30.2 ms),
`gemm_check` OVERALL PASS.

**Not done, explicitly:** split-K is in `x_gemm_w4a8` only. It does NOT stack
with W4A4 — the dispatcher selects one kernel, so `x_gemm_w4a4` (K64, half the
DPAS count) needs its own port and its own measurement, where the bottleneck
may differ. `x_gemm_w8a8` (lm_head) likewise untouched. Spec-wave parity is a
separate gate and was not run. RC8 per-stream at n=8 is 18.17 tok/s against
GEMV's 32.82, so headroom remains.

### Host stability — cause not established (2026-09-05)

Three hard cuts of the host coincided with GPU work. Four theories were tested
against the journal and left UNSUPPORTED (per-boot counts in
`docs/level-up-xpu.md` lesson 44): display-poll-touches-wedged-card (the boot
that cut had ZERO poll-hog events; the 13 h boot that survived had six),
cross-card P2P storming the root port (the PME warnings came from `00:02.2`,
which is not upstream of either B70 — both hang off `00:02.1` — and sits in
IOMMU group 5, not 20), PME storm (3649 lines on the surviving 13 h boot vs 88
on the boot that died), and a `force-gaming-display.service` restart loop
(320/197 restarts on the two boots that died, but 113 on one that stayed up —
confounded with GPU load, not separable). Two bugs ARE confirmed and were ours:
a spin bounded by 200M *iterations* run against PCIe-speed host USM, and a
functor missing `[[sycl::reqd_sub_group_size(16)]]` that compiled SIMD32 and
took a card down with `DEVICE_LOST`. `tq_allreduce` was left UNMODIFIED rather
than "fixed" on a collapsed hypothesis.

- Added `xpu/tools/gpu_health.sh`: PCI presence, driver binding, config-space
  read, narrow GPU-fault journal patterns, and a trivial kernel per card (the
  only load-bearing signal). Gates on unrecognised `hogged CPU` workers;
  `output_poll_execute` and `pci_pme_list_scan` are named background noise and
  do not gate. First strict run immediately caught a real one:
  **`pf_worker_restart_func [xe]`** — the Xe recoverable-page-fault worker —
  hogging >10 ms x4 during the benchmark session, while both cards stayed live.
  The faulting allocation is NOT identified: the journal line names the Xe
  page-fault worker, not the VM or address that faulted, and those same
  processes both staged 22.6 GB through host USM AND ran many split-K kernels,
  so timestamp coincidence cannot distinguish loader upload from kernel
  execution. Attribute only after an upload-only A/B (host-USM staging vs the
  `malloc` fallback) or a driver trace shows the faulting address. Cards
  passing liveness afterwards establishes RECOVERY, not that the stall was
  benign. Left flagged, gate left strict.
- `force-gaming-display.service` stopped and disabled (unit file untouched).
  It targeted a hardcoded `HDMI-11`, which belongs to the AMD iGPU; the monitor
  is `HDMI-9` on the 5090, and connector numbering shifts with enumeration
  order, so a durable fix selects by EDID rather than by name. Its
  `StartLimitBurst=10`/`StartLimitIntervalSec=60` never tripped because each
  cycle costs ~8-11 s (`sleep 5` + gdctl + `RestartSec=3`), i.e. ~5-7 starts
  per minute — under its own cap.
- No crash-capture is armed (`/sys/fs/pstore` empty, no `ramoops=`, netconsole
  not loaded) while `crashkernel=` IS reserved. A hard lockup loses its final
  buffered journal writes, so the next one is equally undiagnosable until kdump
  is armed. VFIO is not an option: both B70s share IOMMU group 20 with Wi-Fi,
  5GbE, USB XHCI and SATA (24 members, no ACS on the AMD 600-series switch).
  Both cards do expose FLR.
- `/models` was a symlink into `$HOME` that did not survive the reboot; the
  22.6 GB TQF is intact at `/home/shooting-brake007/models/knivesysl/`. Root fs
  is at 96% (82 G free).

### Rejected — Unified Memory Framework (2026-09-05)

Cloned `vendor/unified-memory-framework` and recon'd it (report:
`history://UmfScout`). **Skip.** Its one plausible fit is the Fixed Memory
Provider under the Disjoint Pool, which can sub-allocate an address range we
already own while keeping allocator metadata in host memory. But it has no
copy or batching primitive, so it cannot touch the gap we hoped for — the
176 MB APC checkpoint moving at 3.2 GB/s against a 6.5-6.8 GB/s host-DRAM
floor, because we write it as 256 separate ~3.3 MB per-layer copies. That is a
layout-and-submission fix we must make ourselves. The L0 provider also cannot
adopt an existing allocation or import an ordinary `malloc` buffer with
`prepare_for_device_copy` semantics, so it cannot express the one allocation
win we did measure (26.9 -> 45.4 GB/s). L0 IPC duplicates the producer FD
(ptrace permission required) and opens it against the consumer's chosen
device, delegating cross-card success to L0/P2P — it exposes allocations across
processes, it does not guarantee peer transport. hwloc is effectively
mandatory. Against an engine that allocates ~10 slabs at init and none per
token, it buys nothing on the run path; our churn is one object size, so a
purpose-built fixed-slab bump/free-list is the right shape.


### Same-weights head-to-head vs vLLM-XPU (2026-09-04, live)

Both engines on this box, `Qwen3.8-27B-GPTQ-Int4` weights for vLLM-XPU
(0.27.1 + vllm-xpu-kernels 0.1.12.3, torch 2.13+xpu, Triton attention,
PIECEWISE XPU-graph, fp8 KV, card 1) vs our engine (hybrid k64 + DN chunk +
DPAS attention, card 0). Client: `tools/bench_openai.py`, streamed
completions, unique prompt prefixes (prefix cache defeated), gen=128,
`ignore_eos` (early-EOS burned one round of numbers first - client fixed).
Venv resurrection: uv rebuild; +xpu wheel force-reinstalls (torchvision
0.28.0+xpu, torchaudio 2.11.0+xpu - vLLM's dep pull grabs CPU wheels);
torchcodec removed; platform resolver patched to prefer xpu on dual-vendor
activation; NVIDIA-only minimax warmup import guarded; triton_shim extended
to neuter CudaDriver.is_active (vLLM counts ACTIVE drivers and disables
Triton wholesale at 2). Box doc CORRECTED: the display runs on the NVIDIA
5090 (card1 HDMI connected/enabled); BOTH B70s have no connected outputs -
either is safe for compute (serve_vllm_xpu.sh comment fixed).

| ctx | conc | vLLM TTFT | vLLM prefill | vLLM dec/stream | vLLM agg | ours (c1) |
|---|---|---|---|---|---|---|
| 512 | 1 | 0.56 s | 910 | 27.7 | 27.7 | dec 34.0, prefill 683 (TTFT 0.75 s) |
| 512 | 4 | 2.06 s | 249 | 25.3 | 101 | - |
| 512 | 8 | 3.92 s | 130 | 19.7 | 158 | - |
| 2048 | 1 | 4.32 s | 474 | 19.9 | 19.9 | dec 31.4, prefill 593 (TTFT 3.45 s) |
| 2048 | 4 | 16.8 s | 122 | 19.1 | 76.5 | - |
| 2048 | 8 | 26.9 s | 76 | 7.8 | 62.2 | - |
| 8192 | 1 | 55.3 s | 148 | 10.5 | 10.5 | dec 30.1, prefill 375 (TTFT 21.8 s) |

Reads: their decode collapses with depth (27.7 -> 10.5, -62%) while ours
holds (34.0 -> 30.1, -11%): at 8k we are 2.4x faster to first token and
2.9x faster per decoded token on identical weights. Their aggregate goes
BACKWARDS from c4 to c8 at 2k (76.5 -> 62.2; per-stream 7.8 tok/s). They
keep two columns: shallow prefill (910 vs 683 at 512 - graph-fused
constants; our replay lever) and single-card batching (158 agg at 512c8 -
our multi-slot batch, next build). Everywhere deeper than ~1k, at c1, we
are the faster engine on their own quantization's weights.

Final vendor cell (8k x c4, live): TTFT 190.5 s, prefill 43 tok/s, ITL
332 ms, 3.01 tok/s per stream, 12.0 aggregate. Their batching advantage
inverts at depth: 158 aggregate at 512ctx collapses to 12 at 8k. Vendor
matrix complete.

### Concurrency campaign — continuous batching + prefix cache (2026-09-05)

- Slot-strided per-sequence state (`TQ_XPU_SLOTS`, 1..8, fixed at init):
  KV caches + scales, GDN conv + recurrent state, and committed positions
  allocate slots-times and are addressed by slot offsets at the decode
  call sites. Kernels unchanged. New ABI: `qwn_set_slot/qwn_get_slot/
  qwn_num_slots/qwn_reset_slot`. pos-0 resets route per-slot when
  slots > 1 so a fresh sequence cannot wipe its neighbours.
- Added `qwn_decode_batch(slots, tokens, positions, n, out)`: one step for
  n independent sequences. Projections, MLP and lm_head ride the batched
  RC8 GEMMs (one weight stream, 8 rows); attention and the GDN core run
  per row against that slot's state. The step always runs the 8-wide RC8
  shape (the GEMMs enforce `T % 8 == 0`) with zeroed pad rows.
- Added prefix checkpoints (`qwn_ckpt_bytes/save/restore`,
  `qwn_host_alloc/free`): whole per-sequence state at a chunk boundary,
  staged through host USM. Required rather than block reuse because 48 of
  64 layers are recurrent — GDN state exists only where snapshotted.
- Gates, all byte-exact: `tools/slot_check.py` interleaved-slot decode
  48/48 both slots; `tools/batch_check.py` batched-vs-serial 48/48 both
  slots; `tools/apc_check.py` restored-continuation 32/32. Default
  single-slot `prefill_check` still 100% (32/32).
- Aggregate decode (`tools/bench_batch.py`, K64=all): 512ctx 24.7 (n=1) ->
  85.5 (n=4) -> 147.1 (n=8) tok/s; 8192ctx 21.9 -> 59.8 -> 84.4 tok/s.
  vLLM-XPU best cells on the same weights: 158 at 512xc8, 12.0 at 8kxc4 —
  so 0.93x shallow, 7.0x at depth.
  **Comparison correction (standalone repository publication):** the 7.0x figure
  above is not a valid server-to-server speedup: it mixes a standalone native
  n=8 result with a vLLM HTTP n=4 result. The retained stage-1 HTTP matrix measured
  14.2 aggregate tok/s at 8192ctx/c4 versus vLLM's 12.0, or 1.18x. See the root
  readme and `benchmarks/xe-stage1-20260905.csv`; this historical entry is retained
  for provenance, not a current performance claim.
- Continuous batching scheduler (`tools/serve_batch.py`): mid-flight
  admission, bounded prefill chunk per iteration, batched decode over all
  generating slots, immediate slot release, APC lookup at chunk
  granularity. 16 requests / shared 512-token prefix / 1.5 arrivals/s:
  wall 18.63 -> 11.39 s, TTFT mean 4.06 -> 0.33 s (12.3x), TTFT p99
  6.79 -> 1.07 s, ITL mean 175.8 -> 51.9 ms, 7168 of 8208 prefill tokens
  skipped off a single 176 MB checkpoint (14 hits / 2 misses).
- Server-vs-engine settled with a per-width metric on `/health`: at the
  SAME format tier the server delivers 99.8% of the standalone engine at
  width 1 (33.9 vs 33.96) and 98.1% at width 8 (144.3 vs 147.06). The
  apparent gap was a tier mismatch (k64-off server compared against
  k64-all engine numbers) plus prefill/ramp inside the end-to-end
  average, not overhead. No server optimisation was warranted.
- Fused the remaining fan-out opportunities in the decode path:
  `mlp_gate`+`mlp_up` (shared post-norm activation) and `q`+`k`+`v`
  (shared input norm) now ride `x_gemv_w4a8_fanout`, which the GDN 4-way
  input had used since Phase 2. Launches per token 352 -> 256; projection
  27.98 -> 26.92 ms, step 34.17 -> 33.11 ms (-3.1%), decode 32.5 -> 33.26
  tok/s. Gates: gemm_check cos 1.00000000, prefill_check 32/32, parity
  10/10 probes + argmax 64/64.
- Profiled both paths on silicon (see `docs/level-up-xpu.md` §10):
  - decode 33.11 ms = 81.3% projection at **82% of byte-roofline** and
    lm_head at 92%; the 8.1 ms above the 25.0 ms bandwidth floor is
    ~3.9 ms of small-kernel launch/ramp (304 launches moving 0.27 ms
    worth of traffic) plus ~4.9 ms of projection ramp.
  - prefill T=512 976 ms = 68.2% projection at **42.5 TOPS = 11.6% of the
    366 TOPS DPAS peak** (epilogue-bound, exactly the measured RC8 wall)
    and 23.3% GDN chunk scan.
  - Lesson 37: decode is memory-bound, prefill is compute-bound; they need
    opposite fixes and a single roofline percentage for "the engine" is
    meaningless.
- Added `xpu/tools/bench_matrix_xpu.sh`, the XPU analogue of
  `tools/bench_core_matrix.sh`. Truth rules baked in: **cold prefill (APC
  off), no speculative decode**, ignore_eos, fixed gen, one server per
  arm; warm-prefix/APC runs only under a labelled EXTRAS section, never
  mixed into the truth cells. Arms: ship / k64mlp / k64all / dpas.
  First full run, aggregate tok/s (cold prefill, no spec, gen 128):

  | arm | 512c1 | 512c4 | 512c8 | 2048c1 | 2048c4 | 2048c8 | prefill c1 | TF |
  |---|--:|--:|--:|--:|--:|--:|--:|--:|
  | ship k32 | 33.4 | 53.8 | 77.3 | 32.2 | 32.3 | 39.7 | 396/425 | 93.4% |
  | k64mlp | 34.1 | 59.0 | 85.2 | 33.2 | 36.0 | 44.7 | 459/496 | 92.2% |
  | k64all | 34.8 | 67.5 | **95.3** | 33.6 | 40.5 | **49.5** | **501/545** | 89.9% |
  | dpas | 33.4 | 53.7 | 77.2 | 32.8 | 32.6 | 40.0 | 393/424 | 92.6% |

  Per-width engine rate at width 8, same runs: ship 120.9-121.1, k64mlp
  130.7, k64all **147.1** — the last matching standalone `bench_batch`
  (147.06) to three digits, which is the cleanest proof yet that the
  server delivers the engine's rate and that the earlier "server is
  slower" reading was a tier mismatch.

  Reads: k64all is the throughput winner everywhere (+23% c8 agg at 512,
  +25% at 2048, +27% prefill) and k64mlp is the balanced pick (+10%/+13%/
  +16% while clearing the 0.90 TF gate at 92.2%). DPAS attention is a wash
  at <= 2048 (40.0 vs ship's 39.7 at 2048c8) — it is a depth lever
  (+7.5% measured at 8k), so it needs the 8192/32768 cells to earn its
  place and does not belong in a shallow matrix.
- **TP=2 re-verdicted FUNDED, correcting the Phase-0 kill.** Phase 0
  killed TP2 on 2 GB/s streaming peer bandwidth, but tensor parallelism
  moves no bulk data: following vLLM's layer structure it needs 2
  all-reduces x 64 layers = 128 per token of a 20 KB hidden vector while
  **halving** the 15.05 GB/token weight stream. New probe
  `xpu/probe/tp_allreduce_probe.cpp` (one SYCL context spanning both
  cards): device-add floor 4.74 us, 20 KB p2p 1-way 10.76 us (1.90 GB/s,
  confirming Phase 0), 2-way 27.69 us (the directions do NOT overlap),
  full all-reduce via p2p exchange + device add **37.33 us**
  (correctness PASS), via host USM 42.98 us. 37.33 x 128 = 4.78 ms/token
  of comm against ~14 ms of weight stream saved: projected ~22.0 ms step
  = 45.5 tok/s, **1.37x single-stream**. Lesson 39: a bandwidth number
  cannot kill a latency-bound design.
- **TP=2 substrate built and gated** (the sharded forward is NOT built; see
  `docs/level-up-xpu.md` §10.4 for the executable plan). `TQ_XPU_TP`
  (default 1 = today's engine) builds one in-order queue per rank on
  consecutive Level Zero devices in ONE shared SYCL context. The cheap
  part: every allocator and transfer already routes through `tq_q()`, so
  `tq_set_rank(r)` re-targets allocation, upload and launch with no change
  to any kernel. Added `tq_q_of/tq_tp_size/tq_rank/tq_set_rank`,
  `tq_allreduce` (peer exchange -> queue join as barrier -> local add) and
  the model-free export `qwn_tp_selftest(n)`.
  Gate `tools/tp_check.py`: 2 ranks, all-reduce correct at n =
  5120/17408/262144. **TP=1 verified regression-clean after the `tq_q()`
  rewrite**: gemm cos 1.00000000, prefill 32/32, parity 10/10 + argmax
  64/64, decode 33.21 tok/s vs 33.26 before (noise).
  Remaining for TP2, in dependency order: rank dimension on the layer
  structs, `read_qmma_sharded` (column-parallel is a contiguous payload
  slice; row-parallel needs a strided host gather), per-rank forward with
  all-reduce after o_proj/mlp_down/linear_out, vocab-parallel lm_head with
  a 16-byte argmax-pair reduction instead of a 993 KB logits all-gather,
  and an eps-band parity gate — TP2 is **not** bit-exact against single
  card by construction, because a K-sharded dot product sums in a
  different order.
  Lesson 40: build the substrate before the feature and gate it
  model-free; `qwn_tp_selftest` proved the plumbing in 0.5 s, before any
  loader surgery could make a plumbing bug look like a numerics bug.
- **TP=2 rank indirection + weight-shard primitive landed and measured.**
  `g_qwen` is now a macro over `*g_qwen_active` with one `tq_model_t` per
  rank, so `tq_set_rank()` moves queue and model together and every
  existing use site compiled unchanged (TP=1 keeps rank 0 forever).
  `x_w4_shard(dst, src, rank, tp, mode, groups, ngroups)` slices an
  already-repacked W4 weight with **no repack**: the packed layout is
  row-group-major (`codes: g*Kt*256`, `scales: g*(Kt>>ksh)*16`), so a
  column shard is one contiguous range and a row shard is `Mt` contiguous
  runs at fixed stride — both pure memcpy. Column mode takes a `groups`
  list so packed outputs (`q_proj` = [q|gate], `linear_in_qkv` = [q|k|v])
  shard per group, the job vLLM's `QKVParallelLinear.output_sizes` does.
  Gate `qwn_shard_check(layer)`, deliberately single-card so a layout bug
  cannot hide behind TP plumbing, on real mlp_gate weights (17408x5120),
  layers 0/1/32: **column concat bit-exact** (cos 1.00000000, max_rel 0),
  **row sum cos 1.00000000 but max_rel 0.0009-0.0019, not bit-exact**.
  Lesson 41: that asymmetry is now measured, not assumed — a column shard
  walks the same k-tiles in the same order so it is byte-identical; a row
  shard reassociates the K sum. It localises TP2's inexactness precisely
  to the three row-parallel weights (o_proj/mlp_down/linear_out) and
  settles that TP2 ships as an eps-band tier, not a byte-exact one.
  TP=1 re-verified after both changes: gemm cos 1.00000000, prefill
  32/32, parity 10/10 + argmax 64/64.
- **TP shard mode table validated across every weight family** via
  `qwn_shard_check_all(layer)`, which runs `x_w4_shard` at each
  projection's real shape and intended mode so the loader's loop over all
  496 weights cannot trip a constraint at load time. Both layer types, 15
  families, **0 failures**: GDN lin_qkv col-grouped
  [2048|2048|6144] / lin_z col / lin_a+lin_b **replicated** / lin_out row;
  attention q_proj col / k_proj col / v_proj col / o_proj row; mlp_gate +
  mlp_up col / mlp_down row. Column and replicate bit-exact (max_rel 0),
  row cos 1.00000000 at max_rel 1.3e-4..1.2e-2.
  Two findings the gate produced that the plan had wrong:
  - Lesson 42: `q_proj` is **head-major** (`qg_base = head*2*hd`, i.e.
    `[q|gate]` per head) so a head shard is contiguous and needs NO
    grouping, while `linear_in_qkv` is group-major and does. Same
    conceptual weight, opposite packing — only the consumer kernel says.
  - Lesson 43: `lin_a`/`lin_b` emit one scalar per value head (M=48), so
    24 rows per rank is not a multiple of the packed 16-row group and
    `x_w4_shard` rejected both with rc=-6 on the first run. They
    replicate (123 KB each, ~12 MB per card over 48 layers) and each rank
    reads its own heads' slice. A TP plan derived from head counts alone
    is wrong wherever the packed layout is coarser than a head.
- Lessons 31-33 recorded. Next lever: split-K for the batched GEMM at
  small T — the batch step runs at 46% of byte-roofline versus GEMV's 87%
  because `tgroups=1` leaves the small-M projections at 16% occupancy;
  projected 271 tok/s aggregate if closed.
- Added `xpu/tools/serve_openai_xpu.py`, an OpenAI/vLLM-compatible server
  mirroring the CUDA server's surface (`tools/serve_openai.py`): stdlib
  ThreadingHTTPServer, chunked SSE streaming, `/v1/chat/completions`,
  `/v1/completions`, `/v1/models`, `/health`, Qwen chat template with
  tools and `<tool_call>` round-trip, `reasoning_content` split at
  `</think>`, stop strings, usage accounting, `x_knivesysl` per-request
  stats, `n>1`/`logprobs` -> 400. Unlike the CUDA server (one sequence
  behind a lock) this one is genuinely concurrent: one engine thread runs
  the continuous-batching loop over `qwn_decode_batch` while handler
  threads submit and drain per-request queues.
- Server measured live with the same client that measured vLLM-XPU
  (`tools/bench_openai.py`, 512ctx, gen 256): c1 32.8 tok/s / TTFT
  1.31 s, c4 59.9 aggregate, c8 **93.8 aggregate** / ITL 85.3 ms. c1
  matches the standalone engine (33.6 vs 34.0) so server overhead is ~0.
  Gate `tools/serve_smoke_xpu.py`: surface checks, 8/8 concurrent both
  waves, batch_avg 6.66, 3072 prefix tokens reused on the warm wave.
- Implemented lesson 31's rule that had only been written down: a lone
  decoding row routes through the GEMV path, not the RC8 batch step.
  Server c1 decode 19.56 -> 33.62 tok/s.
- Lessons 34-36 recorded from gate failures: a prompt length that is a
  multiple of 8 leaves no tail token for the first-logits step (emitted
  -1); EOS must terminate before being surfaced (leaked `<|im_end|>` into
  content); and a prefix cache with no admission policy is a tax — on
  unique-prompt load it stored 104 checkpoints (8 GB, ~55 ms each) for a
  zero hit rate, so caching now requires a prefix's SECOND sighting,
  which cut the cache to 171 MB and improved c8 aggregate 91.8 -> 93.8
  and TTFT 8.61 -> 8.30 s.

### Phase 4 (opened) — orchestration microbenches

- Added `xpu/probe/persistence_probe.cpp` (level-up queue item 1): bounded-spin
  global barriers with abort-drain, residency sweep, and launched-pipeline
  baselines. Measured on card 1:
  - resident work-group bound matches `total_threads / (wg_size/16)` exactly
    (512/256/128 for wg 64/128/256); over-residency by +1 drains cleanly with
    zero cross-WG ordering errors.
  - centralized global barrier costs max(~1.2 us, ~36 ns per participant):
    4.2 us at 128 WGs, 19.0 us at 512. A two-level tree barrier is WORSE
    (7.8-35.9 us) at every branching factor.
  - a kernel-boundary pipeline runs 1.48 us/stage with full-grid work
    (empty launch 1.43 us, re-confirming Phase 0).
- Consequence recorded in `docs/level-up-xpu.md` (§5 item 1, lesson 14): the
  megakernel form of thesis 1 is killed — the in-order queue's kernel boundary
  is the efficient all-to-all sync on this silicon. The thesis survives as the
  doorbell-driven device loop, continued fusion, and flag-based
  producer-consumer sync, all of which the probe's residency/drain results
  prove safe to build.
- Ran level-up queue item 8 (restrict + AOT A/B) on card 1, paired same-session
  arms; both null results, recorded as lessons 15-16:
  - `[[intel::kernel_args_restrict]]` on all 8 hot DPAS kernels: GEMV decode
    loop 33.0 vs 33.1 tok/s, prefill 468/423 vs 467/422 tok/s, RC8 GEMM
    identical at T>=256 (42.4 TOPS both) and 5-8% SLOWER at T=32-128.
    Reverted; the inline-vISA DPAS block plus distinct-base-pointer access
    patterns leave IGC nothing to hoist.
  - AOT for bmg-g31: steady state identical to JIT and init identical with a
    warm JIT cache (10.18 s both, disk + repack bound); zero spill warnings
    across the TU. Kept as opt-in: `TQ_XPU_AOT=1 ./build.sh` (plus
    `TQ_XPU_AOT_OPTS` for per-kernel IGC options — the plumbing the GRF256
    experiment needs).
  - Bonus measurement: idle-host uplift vs the recorded contended session is
    +17% on RC8 GEMM (42.4 vs 36.2 TOPS at T=1024) and +15% on chunked
    prefill (468 vs 406 tok/s at 512), confirming the earlier "conservative
    floor" caveat and hardening the paired-arms benchmark rule.
- Rebuilt `build/libforward_qwen_xpu.so` from the reverted sources;
  `gemm_check` gate: worst cos 1.00000000 at T = 8/32/128/512, PASS.
- Added `xpu/probe/doorbell_probe.cpp` (level-up queue item 2), four versions
  deep with the postmortems kept in the header. Results on card 1:
  - zex WaitOnMemory/WriteToMemory ping-pong on an async immediate list:
    p50 7.34 us, p99 10.0 us, 1900/1900 steps clean — the driver-level
    doorbell works and is priced.
  - single-ping latency class: persistent RMW-spin 5.17 us, submit+wait
    5.62 us, load-spin 6.00 us, submit+poll 6.50 us, zex 7.34 us. Nothing
    beats plain submission on raw latency; the doorbell pays only as
    per-step amortization paired with command-list replay (new queue item
    2b).
  - sharp finding: device-side atomic reads of host-USM clobber concurrent
    host stores (db regressed 2 -> 1 with a load-only spin; fixed by
    hammer-stores: 400/400 clean). One-shot host doorbell writes to
    host-USM are unsafe on this stack; zex or hammer only. 128 spinners on
    one host line collapse to 258 ms/step. Lessons 17-19.
- Added `xpu/probe/replay_probe.cpp` (level-up queue item 2b) — the decision
  probe for thesis 1's surviving rung, and it FUNDED it: with 350
  copy-commands per step as the command-stream stand-in, live per-command
  appends cost 454-529 us of host CPU per step, while a recorded command
  list replayed per step costs 28.1 us (self-rearming doorbell variant
  29.9 us, immediate-append 31.1 us) — a 15-19x host-CPU reduction, the
  structural fix for the engine's measured host-contention sensitivity.
  Sharp edge found: more than one pre-queued execution parked on a zex
  semaphore wedges; window must be 1. Design consequence recorded in
  `docs/level-up-xpu.md` lesson 20: keep recorded args static by moving
  per-step variability into device-resident state so replay needs no
  mutable command lists.
- Extended `qwn_w4_grid_probe` with sub-4-bit rungs (schemes 7-9: int3 sym,
  int3 asym with nibble zero point, error-gated 3/4 mix with measured
  achieved bits) and added `xpu/tools/grid_probe.py` as its runner. Ladder
  on layers 0/1/32 mlp_gate: int3 asym at 3.50 bits/w = rel-L2 0.168-0.170
  (2.14x the ship tier's 0.0783); the 3/4 mix selected ~0% int3 groups
  (achieved 4.50 bits). The naive sub-4-bit decode-ceiling lever is dead;
  4.5 b/w int4-asym stands as the byte-optimal uniform-grid point
  (level-up doc lesson 21). Rebuilt the ship .so with the extended probe
  (probe is cold-path only; `TQ_XPU_W4A8=0` required to run it).
- Ran level-up queue item 3 (GRF256 + G=6 grouped attention): built the
  `-DTQ_ATTN_G=6` variant with a per-kernel `grf_size<256>` property
  (AOT-verified spill-free) and depth-scaled segments. KILLED the scalar
  form: 29.7/27.5/24.0 tok/s at 2k/4k/8k vs G=3's 31.7/30.3/28.1, gap
  widening with depth — GRF256's 4-threads/XVE halves latency hiding on the
  dependent-load KV walk and costs more than the halved traffic (lesson 22).
  The G=6 win moves to the DPAS-matrix formulation (queue 3b). Kept: the
  `TQ_ATTN_G` build knob, segment scaling, and the per-kernel GRF property
  pattern. Ship binary rebuilt (G=3 semantics unchanged);
  `prefill_check.py` gate: 100.00% agreement (32/32), PASS.
- Switched the TQF upload staging buffer to host USM (`sycl::malloc_host`,
  malloc fallback kept) per level-up queue item 10: model load 10.2 ->
  9.35 s average (best 8.99 s). `gemm_check` gate PASS after the loader
  change (worst cos 1.00000000).
- Ran level-up queue item 4 (W4A4 GEMM probe): added
  `xpu/probe/w4a4_gemm_probe.cpp` - standalone A/B with the engine's exact
  RC8 tiling/epilogue on full-size synthetic buffers, all-ones numeric
  verification. Baseline arm reproduces the shipped wall (42.6/42.8 TOPS at
  T=512/1024); the `dpas.s4.s4.8.8` K=64 arm reaches 75.5/76.1 TOPS =
  **1.77-1.78x, thesis 5 FUNDED** (kill line 1.5x). Activation-quant error
  half deferred to engine integration behind the TF-ladder gate. Sharp edge:
  the probe initially compiled its functor kernels without
  `reqd_sub_group_size(16)`, went SIMD32, and took the device down
  (DEVICE_LOST) - lesson 23.
- Ran level-up queue item 7 (RC8 spec-wave probe): added `qwn_gemv_bench`
  and `qwn_specwave_check` exports plus `tools/bench_specwave.py`. On real
  layer weights (0/1/32): one RC8 wave (T=8) costs 1.11-1.17x a single RC1
  GEMV (96-102 us vs 86-87 us) — break-even 0.11-0.17 accepted drafts per
  wave, 6.8-7.2x projection-stream multiplier at full acceptance; prefix
  rows 0-3 byte-exact under differing draft rows (all PASS). **Thesis 2
  FUNDED** (lesson 24). `gemm_check` gate PASS after the TU rebuild.
- Ran level-up queue item 12 (host-DRAM tier probe): added
  `xpu/probe/hostdram_swap_probe.cpp`. B70 PCIe swap floor 6.5-6.8 GB/s
  both directions (4-512 MB blocks); 4k-token KV restore 21.7 ms vs 8.9 s
  re-prefill (~400x, FUNDED); bulk D2H racing a 590 GB/s read kernel shows
  0% mutual overhead — swaps are free under decode (lesson 25).
- Wrote §9 "Integration designs" in `docs/level-up-xpu.md`: 9.1 replay with
  device-resident step state (per-shape recorded lists, static args, ~1 us
  advance kernel, window=1), 9.2 chunk-64 WY-UT DeltaNet prefill scan (bf16
  DPAS tier), 9.3 n-gram archive + spec wave (staircase attention, serial
  GDN rows, exact-prefix verify), 9.4 CPU-peripheral pass (KV parking, APC
  spillover, drafting on freed cores). Build order 9.2 -> 9.3 -> 9.1 -> 9.4,
  each with its own kill criteria. Reran the full parity gate after the
  day's three TU changes: 10/10 probes cos >= 0.999979, argmax chain 100%
  (64/64), OVERALL PASS.
- Ran level-up queue item 6 (int4-K KV micro): added
  `xpu/probe/int4kv_probe.cpp` (device FWHT-256+s4 quant, engine-shaped
  attention walk on a 32 MB synthetic cache). KILLED on accuracy: s4+H
  score cos 0.9929 row-scale / 0.9957 g32 vs E4M3's 0.9997 (14-24x the
  band). Byte thesis confirmed the other way: walk 1.6-2.1x at 1.6x bytes,
  both arms near roofline, unpack cost unmeasurable (lesson 26). K stays
  E4M3; depth bytes move to the 3b attention formulation.
- Integrated the W4A4 prefill tier (level-up item 4 completion): `TQ_XPU_K64`
  selector repacks W4 weights to per-K64 FP16 scales (mode 2; codes layout
  unchanged), new `x_quantize_act_chunk_s4` + `x_gemm_w4a4`
  (dpas.s4.s4.8.8) with hybrid per-weight prefill dispatch behind
  `TQ_XPU_W4A4`, k64-aware scale indexing (ksh) in the GEMV cores, fanout,
  and RC8 GEMM, plus `qwn_gemm_w4a4_check`/`qwn_gemm_bench_w4a4` exports
  and `tools/w4a4_check.py`. Gates: scalar anchor cos 0.9957+, gemm_check
  1.00000000, w4a4 check 0.9963 (analytic s4-act band), prefill_check 100%
  pure and hybrid, parity chain 100%. Live on card 1: prefill 465->545
  tok/s @512ctx hybrid (`gate,up,down`, TTFT -15%), ->600 all-k64; decode
  32.85->34.07 tok/s hybrid. TF prices the tiers: 92.22% hybrid PASS,
  89.88% all-k64 FAIL, k32 default (97-98%) unchanged - k64 ships as the
  documented opt-in TTFT tier (lesson 27). Session baselines re-measured:
  decode 32.85 tok/s (p50 30.4 ms), prefill 466/421/298 tok/s at
  512/2k/8k, init 9.05 s.
- Built level-up design 9.2 (chunk-parallel GDN prefill, ported from the
  CUDA twin's k_tq_deltanet_chunk): `x_linear_conv_chunk` (batched causal
  conv + separate state-advance kernel) and `x_deltanet_chunk` (CK=8
  sub-chunks, forward-substitution solve, serial only across sub-chunks),
  wired behind `TQ_XPU_DN_CHUNK` (default on) in pf_linear_layer. v1
  (CUDA launch shape) was 21% SLOWER at 19% occupancy; v2 (1024-thread
  WGs, G=8 contraction stripes, register-resident state across the whole
  layer chunk) ships: prefill 464->510/421->457/298->315 tok/s at
  512/2k/8k, prefill_check 100% (32/32) alone and combined with the W4A4
  hybrid (lesson 28). Stacked live totals vs the session baseline:
  hybrid k64 + chunk = 615/539/352 tok/s (+33/+28/+18%, TTFT 1.10->0.83 s
  at 512); all-k64 + chunk = 683/593/375 (+47/+41/+26%, TF-gated tier).
- Built design 9.3 (speculative wave): `qwn_spec_wave` runs 1 committed
  token + 7 drafts as one 8-row prefill-shaped pass (staircase attention
  and GDN wave semantics come free from the pf path + the 9.2 chunk kernel
  at T=8), batched 8-row lm_head via new `x_gemm_w8a8` (dpas.s8.s8.8.8,
  1.27 GB streamed once), greedy verify on host, and exact GDN rewind
  (state snapshots + re-advance from per-layer cached wave inputs via the
  split-out `x_linear_conv_advance`; KV rows past the accepted prefix are
  dead by positional masking). `tools/bench_spec.py` with prompt-lookup
  drafting: decode 33.07 -> 48.24 tok/s (1.46x) at 256 steps and 33.91 ->
  51.08 tok/s (1.51x) at 512 steps under the hybrid k64 tier, 2.99
  tokens/wave, **100.00% greedy agreement in every run** (768 verified
  tokens total). prefill_check 100% and parity chain 100% re-run after the
  TU change (lesson 29). Thesis 2 is now shipped mechanism, not projection.
- Hygiene sweep on the final TU: prefill_check 100%, parity chain 100%,
  TF 93.39% PASS (default k32 tier vs scalar refs) - which recalibrates
  the band: W4A8-vs-scalar is the ~93-94.5% class, so the k64 tiers cost
  -1.2 (MLP hybrid) and -3.5 (all) points, milder than first framed
  (lesson 30, superseding lesson 27's band comparison). Head-to-head
  vLLM-XPU re-measure is blocked: the tmpfs venv did not survive the /tmp
  wipe; rebuild recipe lives in `serve_vllm_xpu.sh`.
- Ran queue 3b's probe-first step (design 9.5): pre-read sycl-tla FMHA
  (f32->bf16x2 vISA idiom, log2e/exp2 fold, FragO V-tiling) and added
  `xpu/probe/dpas_attn_probe.cpp` with three arms. QK^T: dpas.bf.bf.8.8 M8
  serves a 6-head GQA group in 0.468 ms vs the scalar G3x2 shipping
  shape's 0.686 ms (**1.47x, FUNDED**) at score cos 0.999999 - E4M3
  decodes exactly into bf16, and B-fragment lane n is its token's own
  contiguous K slice (no transpose, no SLM). Full pipeline arm: online
  softmax in the log2 domain, bf16 P staging, PV dpas with SLM V tiles,
  engine-shaped segment partials + host log-sum-exp merge = **cos 0.999989
  vs fp64 reference attention** - the whole formulation is silicon-proven.
  Remaining for the engine kernel: occupancy (probe arm C = 6% machine)
  and vectorized e4m3 decode (all arms ALU-bound at 36-98 GB/s).

### Phase 0 — Xe2 hardware characterization

- Added `xpu/probe/dpas_probe.cpp`, a native inline-vISA DPAS throughput probe.
- Verified both Arc Pro B70 devices through Level Zero using oneAPI 2026.1.1.
- Verified native Xe2 DPAS execution and numerical checks for:
  - signed/unsigned INT4 at K=64: 733 TOPS;
  - signed/unsigned INT8 at K=32: 366 TOPS;
  - mixed S8×S4 at K=32: 365 TOPS;
  - BF16 at K=16: 183 TOPS;
  - TF32 at K=8: 91.6 TOPS.
- Measured 602 GB/s pure-read bandwidth, approximately 99% of the B70's
  advertised 608 GB/s bandwidth.
- Measured 1.42–1.44 µs asynchronous in-order launch latency.
- Measured only 2.0 GB/s peer copies between the two B70s. Phase-0 decision:
  dual-card deployment uses independent replicas rather than TP2.
- Added `xpu/probe/dpas_layout_probe.cpp` and decoded the mixed S8×S4 operand
  layout with one-hot tests on silicon:
  - A/S8 is a contiguous logical K row;
  - B/S4 uses depth-major VNNI packing;
  - D/S32 maps one output column to each SIMD16 lane;
  - S4 is two's-complement.
- Recorded reproducible measurements and decisions in `xpu/PROBE_RESULTS.md`.

### Phase 1 — scalar correctness engine

- Added an independent SYCL shared library build:
  `xpu/build/libforward_qwen_xpu.so`.
- Added a TQF1 loader for the shipping 27B Qwen3.8 model. The loader parses all
  64 decoder layers and skips the native MTP section intentionally; the XPU
  Phase-1/2 ABI reports no MTP support.
- Ported the BF16, E4M3, and E2M3 numeric codecs and the inverse lookup for the
  SM120 QMMA E2M3 fragment payload.
- Ported scalar-correctness implementations of:
  - embedding lookup and `(1+w)` RMSNorm;
  - E2M3 projection GEMV;
  - causal depthwise convolution;
  - gated DeltaNet recurrence and recurrent-state updates;
  - full-attention decode with Q/K normalization, partial RoPE, FP32 KV, and
    output gating;
  - MLP activation, residual adds, final norm, LM head, and argmax.
- Preserved the existing `qwn_*` ctypes boundary for the implemented
  single-stream API.
- Added `xpu/tools/xpu_parity.py` and captured a CUDA reference answer key once.
- Observed scalar XPU parity against the CUDA FP6 engine:
  - layer-64 hidden-state cosine: 0.998724;
  - all recorded layer probes passed;
  - 64/64 greedy tokens matched;
  - teacher-forced agreement: 97.28% over 257 positions.
- Observed full-model load time of 15.6 seconds and scalar decode latency of
  approximately 0.29 seconds after the first step. The scalar tier is retained
  as a correctness oracle, not a shipping performance path.

### Phase 2 — native integer DPAS decode

- Added `xpu/src/kernels_dpas.cpp`.
- Added load-time dense-E2M3 → signed-INT4 repacking into the verified Xe2 DPAS
  B-operand layout.
- Added per-row/per-K32 FP16 weight scales.
- Added per-K128 signed-INT8 activation quantization with FP32 scales.
- Added native `dpas.s4.s8.8.1` GEMV and wired repacked weights through the
  projection dispatcher.
- Repack discipline matches the CUDA NVFP4 tier: an eligible projection owns
  either its original E2M3 payload or its DPAS payload, not both. The original
  payload is freed after a successful conversion.
- Repacked 497 model matrices with zero conversion failures in the all-W4
  configuration.
- Observed representative 17408×5120 projection cosine of 0.9953 between the
  W4A8 DPAS result and the scalar E2M3 result.
- Observed an initial short-run decode rate of 28.6 tok/s on one B70.
- Quality measurements against the saved CUDA FP6 teacher-forced stream:
  - all eligible projections W4A8, initial per-K32 symmetric scales: 86.77%;
  - keeping `in_proj_a/b` in scalar E2M3: 85.60% (rejected; worse);
  - refined signed-range + least-squares K32 scales: 89.49%.
- Switched the quality gate to the scalar-XPU teacher-forced stream
  (`/tmp/knivesysl_xpu_scalar_tf.txt`) once the saved CUDA stream aged out of
  `/tmp`; the scalar engine is CUDA-certified at 97.28% so it transfers the
  same reference without re-occupying the 5090.
- Made W4 asymmetric: each (row, K32) group fits `weight ~= scale * (q - z)`
  with a signed 4-bit zero point packed into the low nibble of the FP16 scale
  (16-ULP scale rounding, no extra metadata bytes or reads). The activation
  quantizer now emits exact per-K32 integer sums so the GEMV subtracts
  `z * sum(a)` inside the s32 domain. Weight-only cosines rose to
  0.9970-0.9971 (from 0.9934-0.9953) and teacher-forced agreement to 92.22%
  all-W4.
- Retuned activation scales from per-K128 to per-K32 to match the DPAS scale
  granularity.
- Rejected: raising mid-size GEMV split counts from 8 to 16 (22.35 ->
  22.16 tok/s).
- Sharded long-context attention decode: past pos 64 each head runs eight
  online-softmax token shards in one 256-thread work-group (one barriered
  merge, no per-token barriers). Attention at pos 256 dropped 4.86 ms ->
  0.62 ms per step; end-to-end decode 22.3 -> 25.2 tok/s.
- Selective W8A8 sensitivity sweep (teacher-forced agreement, all else W4):
  - `q,k,v,o` (full attention): 93.00%;
  - `gate,up,down` (MLP): 91.44%;
  - `q,k,v,o,out`: 94.16%;
  - `q,k,v,o,gate`: 94.16%;
  - `q,k,v,o,out,lm_head`: 94.16% (earliest divergence pushed to pos 11);
  - `q,k,v,o,out,in_qkv,in_z,in_a,in_b`: 94.16% (first divergence pos 32).
- Shipped default: integer DPAS decode is now on by default with the
  `q,k,v,o,out,lm_head` W8 fallback (W4=384, W8=113). `TQ_XPU_W4A8=0` selects
  the scalar oracle; `TQ_XPU_W8` still overrides the selector exactly.
- `xpu/tools/xpu_parity.py` now forces `TQ_XPU_W4A8=0` (assignment, not
  `setdefault` — an inherited `TQ_XPU_W4A8=1` was observed swapping the engine
  under the gate) so the Phase-1 gate keeps certifying the scalar engine after
  the default flip.
- Persisted both references outside `/tmp`, which was cleaned mid-session:
  `~/knivesysl_xpu_refs/scalar_ref.npz` (probe ladder + 64-token chain, cut
  from the pinned scalar tier) and `~/knivesysl_xpu_refs/scalar_tf.txt`
  (257-position teacher-forced stream). Scalar-vs-scalar regression gate:
  10/10 probes at cos 1.000000, argmax chain 100%.
- Measured shipping config on both B70s: 25.2 tok/s sustained dense decode
  (512 steps, warm-up excluded), latency p50 39.7 ms, p99 40.3 ms, identical
  on device 0 and device 1.

#### Format ladder vs the CUDA tiers (measured on silicon)

- Added `qwn_w4_grid_probe` (`xpu/src/kernels_dpas.cpp`): weight-reconstruction
  cosine and relative L2 for every candidate tier on one real 17408x5120
  `mlp_gate`, at matched metadata budgets. Pure read; it shares the exact
  asymmetric fit with the repack via `fit_int4_asym`, so the measured "ship
  tier" row cannot drift from what actually loads.
- Measured, stable to +/-0.0004 cosine across layers 0, 1 and 32:

  | tier | bits/w | cos | rel L2 |
  |---|---|---|---|
  | int4 sym, fp16 per k32 | 4.50 | 0.996062 | 0.0887 |
  | **int4 asym, fp16 per k32 (ship)** | **4.50** | **0.996926** | **0.0783** |
  | int4 sym, e4m3 per k16 + fp32 global | 4.50 | 0.995940 | 0.0912 |
  | **NVFP4 e2m1, e4m3 per k16 + fp32 global (CUDA's tier)** | **4.50** | **0.995541** | **0.0943** |
  | int4 asym, e4m3 per k16 + fp32 global + 4-bit zp | 4.75 | 0.997435 | 0.0716 |
  | int4 asym, fp16 per k16 | 5.00 | 0.997803 | 0.0663 |
  | int8 sym, fp16 per k32 (W8 fallback) | 8.50 | 0.999988 | 0.0049 |

- Conclusion 1: porting CUDA's NVFP4 tier would be a **downgrade**. At an
  identical 4.50 bits/w it carries 20% more reconstruction error than the
  shipping tier (0.0943 vs 0.0783), because E2M1 spends one of its four
  mantissa steps above magnitude 2 while INT4 spreads 16 uniform levels. This
  reproduces Intel's own INT4-beats-FP4 claim on this actual model, so the
  NVFP4 tier is deliberately not ported.
- Conclusion 2: NVFP4's two-level scale trick (cheap E4M3 group scale plus one
  FP32 global per 128 rows) does not buy k16 granularity for free here. At
  4.50 bits/w the E4M3 scale's 3-bit mantissa plus the lost zero point costs
  more than the finer grouping returns (0.0912 vs 0.0783). The ladder is
  monotone in bits - 4.50 -> 0.0783, 4.75 -> 0.0716, 5.00 -> 0.0663 - so the
  current 4.50-bit point is optimal at its budget and finer scales genuinely
  cost bytes.
- Added the K16 rung anyway, since it is the one axis where CUDA's format was
  ahead: `TQ_XPU_W4_K16` is a selector (same labels as `TQ_XPU_W8`, or `all`).
  Each K32 tile carries two scale groups; the B fragment is read once and
  masked into its two K16 halves (dwords 0-1 vs 2-3, per the layout probe), so
  the extra granularity costs one more DPAS issue and **zero** extra weight
  traffic - only the scale array doubles.
- The activation quantizer now emits one exact integer sum per K16 while
  keeping one scale per K32, so both weight tiers subtract their own
  zero-point term exactly and the K32 path keeps its single DPAS. Verified
  non-regressing: `qwn_w4_check` is bit-identical before and after
  (cos 0.997050 / 0.997016 on layers 0/1).
- Teacher-forced agreement vs the CUDA-certified scalar stream, and cost:

  | config | TF agreement | tok/s |
  |---|---|---|
  | k32 everywhere, no W8 | 92.22% | - |
  | k16 everywhere, no W8 | 93.39% | - |
  | **k32 + W8 (default)** | **94.16%** | **25.2** |
  | k16 on `gate,up` + W8 | 95.33% | - |
  | k16 on `down,in_qkv,in_z,in_a,in_b` + W8 | 95.33% | - |
  | k16 everywhere + W8 | 96.50% | 23.8 |

- Kept the default at k32 + W8: 94.16% clears the preferred 94% quality gate
  *and* the 25 tok/s throughput gate. All-k16 reaches 96.50% - 0.78 points off
  the scalar-XPU ceiling of 97.28% - but 23.8 tok/s fails the throughput gate,
  so it ships as an opt-in quality tier rather than the default.
- Cross-engine caveat, stated because it is easy to get wrong: the CUDA quality
  table in `README.md` (fp8 95.94, fp6 91.30, e2m1 86.46, nvfp4 85.78) is
  top-1 agreement against **bf16**, while every XPU number above is agreement
  against the **CUDA FP6 engine**. The two axes are not interchangeable.
  Composing them through the triangle inequality on disagreement rates puts
  all-k16 at 87.8-94.8% against bf16 (point estimate near 88-89%, since the two
  quantizers' errors are largely independent) versus 85.78-86.46% for CUDA's
  own 4-bit tiers. A direct bf16 measurement remains unavailable on either
  engine - the CUDA figures are themselves inherited from the upstream fork and
  flagged pending re-measurement (`README.md:44`).

#### Decode throughput pass — 25.2 -> 31.6 tok/s

Profiled the step at pos=256 against a 602 GB/s roofline first. The gap was
never the format: projection ran at 74% of its byte-roofline while `norm` cost
16.4 us for 20 KB of work and the step issued ~1443 kernel launches. Five
structural fixes, each measured on one B70, default config:

1. **Fused the split-K reduction into the GEMV** (25.2 -> 26.1). With splits
   dividing `kSubgroupsPerWorkgroup`, `task = g*splits + split` means one
   work-group already holds every split of a complete row-group, so the
   partials reduce through SLM instead of round-tripping ~230 MB/token through
   global memory and launching ~400 extra reduction kernels. Splits above one
   work-group keep the global path.
2. **Fused (1+w) RMSNorm with the activation quantizer** (26.1 -> 26.7).
   `x_rmsnorm` had been launching a SINGLE 256-thread work-group - 2 of 256
   XVEs - and reading its input twice, then the quantizer re-read the result
   from DRAM. `x_rmsnorm_quant` spreads the reduction over every XVE and emits
   the S8 codes, per-K32 scales and per-K16 sums from the normalize pass: one
   subgroup is exactly one K16 sum block and two adjacent subgroups are one
   K32 scale block. norm+quant 3.12 ms -> 1.37 ms.
3. **Rewrote the DeltaNet decode core** (26.7 -> 28.1). The generic kernel
   launched nv=48 work-groups of dv=128 (~19% of work-item capacity) and
   touched the recurrent state three times. The shipping-shape kernel splits
   the dk axis across 8 chunks of a 1024-thread work-group, so 48 heads cover
   384 XVE-slots, and each thread keeps its 16 old state values in registers.
   The second read pass is removed outright using
   `core_c = decay * sum_k S_old[k][c] q_k + delta_c * sum_k k_k q_k`,
   whose last factor is a per-head scalar. delta 2.90 ms -> 1.13 ms.
   `TQ_XPU_DELTA_FAST=0` reverts to the generic reference.
4. **Capped `choose_gemv_splits` at one work-group** (28.1 -> 28.2). The M=48
   `in_a`/`in_b` projections asked for 32 splits and paid a second launch for
   0.2 us of arithmetic. 1.36 ms -> 1.17 ms.
5. **Two-tile unroll with two accumulators, plus a lane-major B payload**
   (28.2 -> 28.9). One k-tile per iteration left a single 288-byte load in
   flight behind a serial FP dependency. Storing lane n's four k-quads
   contiguously also turns four stride-64 dword loads into one 16-byte vector
   load; the DPAS register contract is unchanged and `qwn_w4_check` stays
   bit-identical (cos 0.997050 / 0.997016). The layout change alone was
   roughly neutral - kept for the cleaner single load, not for the number.

Rejected: mid-size GEMV splits 8 -> 16 (22.35 -> 22.16 tok/s, measured before
this pass).

Not bit-identical, contrary to the initial expectation for change 1: FMA
contraction differs after the rewrite and the reassociated sums move
near-ties. Every step was re-gated rather than assumed.

The big GEMVs now run at 87% (`s4 17408x5120`, 523 of 602 GB/s) and 83%
(`s4 5120x17408`) of their byte-roofline, so the projection tier is close to
its practical ceiling for this access pattern. What remains off-roofline is
small-kernel overhead: `norm` 1.37 ms, `attention` 0.88 ms and `conv` +
`activation` 0.80 ms against ~0.15 ms of actual bytes, all launch/ramp bound.

#### Measured format ladder and the shipped default

All numbers: one B70, 512 decode steps with 16 warm-up excluded, and
teacher-forced agreement over 257 positions against the CUDA-certified scalar
stream. **Resolution caveat: 257 positions is 0.39% per position, so gaps
under ~1.5 points are not distinguishable from near-tie noise.**

| `TQ_XPU_W8` | `TQ_XPU_W4_K16` | tok/s | TF agreement | note |
|---|---|---|---|---|
| `""` (all W4) | - | **32.7** | 93.77% | max throughput |
| **`lm_head`** | **-** | **31.6** | **94.55%** | **shipped default** |
| `lm_head` | `gate,up,down,in_qkv,in_z,out` | 29.0 | 95.33% | quality tier |
| `q,k,v,o,out` | - | 29.8 | 93.77% | dominated |
| `q,k,v,o,out,lm_head` | - | 28.9 | 94.16% | dominated (old default) |
| `lm_head` | `all` | 28.6 | 94.55% | dominated |
| `q,k,v,o,out,lm_head` | `all` | 26.6 | 96.11% | max quality |

- **Changed the default from `q,k,v,o,out,lm_head` to `lm_head`.** lm_head is
  the only tensor that earns 8 bits: it emits the logits, so its argmax margins
  are exactly what teacher-forced agreement scores. Keeping q/k/v/o/out in W8
  as well cost 9.6% throughput for a one-position quality difference. The new
  default strictly dominates the old one on both axes.
- Counter-intuitively, `lm_head` at k16 (5.0 bits) is worse than at s4 (4.5
  bits) - 92.61% vs 93.77% - and slower. With 248320 rows its top-1 margins are
  thin enough that rounding luck outweighs scale granularity. Recorded because
  it contradicts the monotone-in-bits result that holds for every other tensor.
- k16 pays only on the wide projections. `K16=all` (which adds the tiny 48-row
  `in_a`/`in_b` and q/k/v/o) is no better than `K16=<none>` at 94.55% while
  costing 3 tok/s.
- Net for the pass: **25.2 -> 31.6 tok/s (+25.5%) with quality slightly up**
  (94.16% -> 94.55%). Verified identical on both B70s (31.601 / 31.628 tok/s,
  p50 31.65 / 31.62 ms) and the scalar regression gate still returns 10/10
  probes at cos 1.000000 with a 100% argmax chain.
- Roofline honesty: for the shipped mix the byte floor is 25.83 ms/step -
  15.05 GB of weights (lm_head at 8.5 bits, the other 24.35G params at 4.5)
  plus 302 MB of DeltaNet state and 202 MB of KV at pos 256, all at the
  measured 602 GB/s. At 31.65 ms the engine runs at **82% of its byte-roofline
  (38.7 tok/s)**. The remaining 5.8 ms is the small-kernel overhead itemized
  above, not weight traffic. Past ~38 tok/s needs fewer weight bytes, which is
  a format decision, not a kernel one.

#### What the vendored stacks actually gave us

Read `vendor/zml` (oneAPI PJRT path) and
`vendor/intel-xpu/intel-xpu-backend-for-triton` for anything reusable.

**Used, and it paid:**

- ZML's `select3dConfig` uses eight Intel warps for decode attention
  specifically to spread work and dodge register spill
  (`vendor/zml/zml/attention/triton_attention.zig:216-256`). Auditing our
  equivalent found `x_full_attn_decode` running `kWorkgroup = 256` (16
  subgroups) with `kTokenShards = 8` - **half the threads idled through the
  hot loop**. Raising to 16 shards took attention 0.884 -> 0.612 ms and the
  step to 31.99 tok/s at unchanged 94.55% agreement.
- ZML sets `.kv_cache_modifier = .none` because `.cg` "becomes fully uncached
  on this backend and defeats L2 prefetch" (`triton_attention.zig:660-662`).
  That predicted our own result below.

**Tried and rejected, measured:**

- The Triton backend's `Matrix2DBlockPrefetchOp` idiom - an L2-targeted
  prefetch, up to 256 B/row when the driver reports `support_256b_prefetch`,
  and our W4 tile is exactly 256 B. Prefetching the B tile two k-tiles ahead
  measured **slower**: 31.2 vs 31.6 tok/s, `s4 17408x5120` 12.73 vs 12.25 ms.
  The tile walk is linear, so Xe2's L2 prefetcher already covers it - exactly
  what ZML's `.cg` comment implies. Removed; comment left at the call site.

**Confirmed, no action needed:**

- `DPAS.cpp` `DPASEngineTypeXe2::S32_S32_S8_S8` packs A as 2xi8 into i16 and B
  as i32 per `opsPerChannel` - identical to the packing our one-hot silicon
  probes derived, so `dpas_layout_probe.cpp` and the Triton lowering agree.
- ZML's GatedDeltaNet is generic StableHLO (`stablehlo.convolution` plus a
  scalar-step `stablehlo.while`), not a hand-written Intel kernel, and its
  recurrence math matches ours exactly. Its Intel backend is an opaque
  prebuilt `libpjrt_oneapi.so`, so there is no kernel to copy - only the
  decomposition.

**Found, quantified, NOT yet fixed - the largest known deficit:**

ZML processes all six Q heads of a GQA group in one program so each K/V tile
is loaded once (`unified_attention_oneapi.zig:301-435`). Ours assigns one
work-group per Q head, so the 6 heads sharing a KV head each re-read the same
K/V rows from DRAM. Measured: attention costs 0.612 ms at pos=256 but
**2.788 ms at pos=2048, which is 96% of its byte-roofline** - the kernel is
efficient, the roofline is simply 6x too big. Co-location cannot fix it: the
six work-groups are already adjacent in the grid, and at pos=2048 the
four-KV-head working set is ~16.8 MB, past L2.

| pos | current KV read | GQA-grouped | recoverable |
|---|---|---|---|
| 256 | 0.34 ms | 0.06 ms | 0.28 ms |
| 2048 | 2.68 ms | 0.45 ms | 2.23 ms |
| 4096 | 5.35 ms | 0.89 ms | 4.46 ms |
| 16384 | 21.40 ms | 3.57 ms | 17.84 ms |
| 262144 | 342.46 ms | 57.08 ms | 285.38 ms |

Decode already measures 30.0 tok/s at pos=2048 versus 32.0 at pos=256, and
that gap is almost entirely this term. **The engine does not hold its shape
with depth**, which is the property the 256k-context claim rests on.

The fix is not a tweak. Sharing a K row across all six subgroups needs either
SLM staging with a barrier per token, or ~208 floats/thread of register
blocking that spills. **Implemented below at G=3, which fits.** The full
G=6 formulation is the matrix one ZML uses - an 8x256 Q tile times
K-transpose through DPAS - and remains open Phase-3 work.

#### Depth pass — GQA-grouped attention and GEMV fan-out

- **Grouped attention** (`TQ_XPU_ATTN_GROUPED=0` reverts). One work-group now
  owns G=3 consecutive q-heads sharing a kv-head, so each K/V row is fetched
  once and used three times instead of once per head. G=3 keeps ~112
  floats/lane live, inside the 128-GRF budget; G=6 needs ~208 and spills. The
  token axis splits into 3 segments so the grid stays at 24 work-groups - the
  same count as the per-head kernel - making this a pure traffic win at
  unchanged occupancy. A work-group's 16 subgroups merge through 16 KB of SLM
  one head at a time; a small second kernel merges segments and applies the
  sigmoid gate.
- Also found by auditing against ZML's eight-warp choice: the per-head kernel
  ran `kWorkgroup=256` (16 subgroups) with `kTokenShards=8`, leaving **half the
  threads idle** through the hot loop. Raised to 16.
- Gated at `pos >= 512`. Measured crossover: grouping saves 1.28 ms/step at
  pos=2048 and 2.81 ms at pos=4096 but COSTS 0.09 ms at pos=256, where the
  segment-merge kernel outweighs the traffic saved on a short history.

| pos | grouped | per-head | delta |
|---|---|---|---|
| 256 | 32.07 | 32.17 | -0.3% (gated off) |
| 2048 | 31.23 | 30.03 | **+4.0%** |
| 4096 | 30.31 | 27.94 | **+8.5%** |
| 8192 | 28.73 | 24.52 | **+17.2%** |

- **Fan-out GEMV** (`x_gemv_w4a8_fanout`). The four DeltaNet input projections
  (in_qkv 10240, in_z 6144, in_a 48, in_b 48) share one already-quantized
  activation and one K=5120, so they now issue as a SINGLE launch. No weight
  concatenation: row-groups are numbered across the set and a 4-entry prefix
  sum maps a global row-group back to its payload, scales and output pointer.
  Falls back per-weight when the tier is not uniformly W4-K32. Launches/token
  496 -> 352; those four projections 5.88 -> 4.44 ms; step p50 30.92 -> 30.675.
  The wall-clock gain (+0.8%) is smaller than the profiled kernel saving
  because the removed launches partly overlapped other work - this matters more
  once batched decode makes host submission the bottleneck.
- Rejected with reasoning: fusing the depthwise conv into the DeltaNet core.
  The conv is launch-bound (0.40 ms for 0.033 ms of bytes), but the conv-state
  shift must stay race-free and any fused variant still needs a second kernel
  for the state update, so the launch count does not drop. Eliminating it
  properly needs a circular conv state plus folding the state append into the
  in_qkv GEMV epilogue - that touches the shared, heavily tuned GEMV for ~1.2%,
  so it is deferred rather than attempted.

#### Context-length sweep (shipped default, one B70)

512 decode steps at 128 ctx, 64 steps elsewhere; warm-up excluded; TQ_CTX
raised to 16384. p99 tracks p50 within 0.15 ms at every depth.

| context | tok/s | p50 ms | retention |
|---|---|---|---|
| 128 | **32.64** | 30.68 | 100% |
| 512 | 31.92 | 31.33 | 98.7% |
| 1024 | 31.68 | 31.56 | 98.0% |
| 2048 | 31.23 | 32.02 | 96.6% |
| 4096 | 30.34 | 32.96 | 93.8% |
| 8192 | **29.18** | 34.27 | 89.4% |

Before grouping, 8192 retained only 76.2%. Attention traffic is still 2x its
GQA-ideal, since G=3 rather than 6.

#### Concurrency: what is actually achievable today

- The XPU ABI is single-stream. `nm` on the shipped `.so` shows `qwn_decode`,
  `qwn_reset_state` and the debug/metadata entry points only - no
  `qwn_batched_decode_step`, no `qwn_paged_*`, no `qwn_prefill_chunk`.
  Per-engine concurrency is **not implemented**; the achievable shape is
  independent replicas.
- Measured two replicas, one per card, 2048 ctx simultaneously: 31.214 +
  31.238 = **62.45 tok/s aggregate**, p50 32.04 / 32.01 ms, per-stream
  retention **99.95%**. Perfect 2.0x scaling with no interference, confirming
  the Phase-0 replicas-over-TP2 decision under real load.
- Per-sequence state at fp32 KV: 0.70 GB at 4k ctx, 2.31 GB at 16k, 8.75 GB at
  64k, and **34.52 GB at 256k - larger than the card**. fp32 KV, not weights,
  is what blocks the 256k story; the CUDA int4-K/E4M3-V tier is 8x smaller and
  is the real prerequisite. With 19.31 GB free after weights, 4k-context slots
  would be 27 at fp32 KV or 65 with a Q4 KV tier.
- **Batched decode is the largest remaining lever by an order of magnitude.**
  At B=1 the step reads 15.05 GB of weights (25.0 ms) to do 0.14 ms of DPAS - a
  178x imbalance. Batching reads those weights ONCE for B tokens and stays
  bandwidth-bound past B=32, so aggregate throughput scales near-linearly:
  B=8 projects to ~320 tok/s on one card versus 32.6 today. `[INFERENCE]` from
  the roofline, not measured - it needs the batched GEMM (DPAS repeat-count 8
  instead of 1) plus per-slot KV and DeltaNet state.

#### Head-to-head vs vLLM-XPU on the same card and model

Sourced from `vendor/intel-xpu/intel-arc-pro-b70-inference-cookbook`. This is
the first genuinely comparable external baseline: **same dense Qwen3.8-27B,
same single Arc Pro B70 32 GiB**, vLLM `0.27.2rc1.dev77` +
`vllm-xpu-kernels 0.1.12.3`, `VLLM_TARGET_DEVICE=xpu`, no speculative config
(`docs/qwen38-27/QWEN38-VLLM-XPU.md:30-47,62-74`;
`submissions/vllm-qwen38-mtp4-gptq-int4.json:13-22`).

Their config differs from ours in three ways that matter: GPTQ 4-bit symmetric
G128 weights, **FP8 KV cache**, FP16 compute, and XPU graph capture. Their
figure is client-side post-first - `(completion_tokens-1)/(end - first token)`,
so TTFT and the first token are excluded - median of 5 C1 requests after a
discarded warm-up (`docs/BENCHMARK-FORMAT.md:23-39,64-75,111-125`).

Matched at g512 (their 512-output cells, closest to our 512 decode steps):

| context | vLLM-XPU | knivesysl-xe | delta |
|---|---|---|---|
| p512 | **32.7** | 31.92 | -2.4% |
| p8192 | **31.5** | 29.18 | **-7.4%** |

**We are behind, and the depth gap is entirely our FP32 KV cache.** Our KV read
per decode step at G=3 grouping is 0.134 GB at p512 but 2.148 GB at p8192
(3.57 ms). At their FP8 KV that is 0.537 GB / 0.89 ms - a 2.68 ms saving, which
projects our p8192 to 31.65 tok/s, i.e. level with their 31.5. At p512 the same
change projects 32.09 vs their 32.7, leaving ~1.8% that is most plausibly their
XPU graph capture against our ~350 per-token launches. `[INFERENCE]` on both
projections: roofline arithmetic, not measured.

Caveats recorded honestly: their numbers are self-reported/provisional with
independent reproduction pending (`QWEN38-VLLM-XPU.md:158-165`), the cookbook
never enumerates the 16-attention/48-DeltaNet split so architectural
byte-equivalence is unverified, and their `mtp.*` tensors stay BF16. Also note
their 133 tok/s headline is **Qwen3.6-35B-A3B** (3B active, MoE) with GPTQ-INT4
+ MTP - not comparable to a dense 27B and not to be quoted as such.

Consequence for the roadmap. FP8/Q4 KV moves from "Phase-3 nice-to-have" to
**the single highest-priority item**: it closes the entire measured depth
deficit, it is the prerequisite for 256k context (fp32 KV needs 34.52 GB at
256k, more than the card), and it quadruples concurrent slots. Order is now:
(1) FP8/Q4 KV cache, (2) command-list replay to cut submission overhead,
(3) batched decode for aggregate throughput, (4) G=6 DPAS attention.

### Phase 3 (in progress) — batched RC8 GEMM

#### The real deficit is prefill, not decode

Our decode sits within 7% of vLLM-XPU. Prefill is a different story: there is
no batched prefill at all. `xpu/tools/bench_decode.py:65-69` shows the "prompt"
phase is a per-token `qwn_decode` loop, so every prompt token streams the full
15.05 GB of weights. TTFT is therefore ctx/31.9 seconds: **16 s at 512 tokens,
257 s at 8192, 68 minutes at 131072.** That is the gap to close first.

#### RC8 operand layout, verified on silicon

Re-ran `xpu/probe/dpas_layout_probe.cpp` on a B70 to confirm the repeat-count-8
contract before building anything on it. All four probes pass:

- **A** is 8 rows of 32 contiguous bytes (64 dwords); byte j maps to row j/32,
  k = j%32.
- **B** is unchanged from RC1 - VNNI dword (k/8)*16+n, nibble k%8 - so the
  existing s4 payload is reusable as-is with no repack.
- **D** routes (m, n) to dword m*16+n, i.e. 8 int32 per lane, lane = output
  column.
- s4 is two's complement.

Because B is identical, one `dpas.s4.s8.8.8` consumes the SAME 256-byte weight
fragment as the GEMV but against eight activation rows: weight bytes per output
drop 8x. This is the shared primitive for chunked prefill and batched decode.

#### Kernel and gate

- Added `x_quantize_act_chunk` and `x_gemm_w4a8` (`kernels_dpas.cpp`). The
  activation staging is dictated by the probed A operand: `Aq[kt][token][32]`,
  K-tile-major then token-major, so lane l element i holds token (2i + l/8),
  bytes (l%8)*4..+3 - which makes each vector element one 64-byte coalesced
  read across the subgroup.
- Added `qwn_gemm_check` (`xpu/tools/gemm_check.py`): the RC8 GEMM against the
  already-certified RC1 GEMV on the same repacked weight and same activation
  rows. **Worst-row cosine 1.00000000 at T = 8, 32, 128 and 512**, first try -
  validating the operand layout, the staging and the epilogue together. (Per
  element max-rel 0.0065 is the different summation order showing up on
  near-zero outputs; the cosine is the meaningful signal.)

#### Measured rate, and what actually limits it

`xpu/tools/bench_gemm.py`, one 17408x5120 projection on a B70:

| T | us | TOPS | % DPAS peak | GB/s | % BW |
|---|---|---|---|---|---|
| 8 | 100.3 | 14.2 | 4% | 555 | **92%** |
| 64 | 384.1 | 29.7 | 8% | 145 | 24% |
| 512 | 2581.1 | 35.4 | 10% | 21.6 | 4% |
| 1024 | 5036.5 | **36.2** | 10% | 11.1 | 2% |

RC8 does exactly what it was supposed to at the bandwidth end - T=8 already
runs at 92% of the read roofline, and by T=1024 weight traffic is down to 2% of
bandwidth. But it plateaus at **10% of the 366 TOPS systolic rate**, so the
kernel is now neither bandwidth- nor compute-bound.

The wall is the epilogue. Per k-tile a lane issues ONE DPAS (8192 MACs, ~8
cycles) against ~40 scalar ALU ops for the eight per-token scale/zero-point
corrections. Fetching the per-token scale and sum as one vector each instead of
24 scalar loads per k-tile bought **+17% (30.2 -> 36.2 TOPS)** at unchanged
cosine, which confirms the diagnosis; the remaining ~40 ALU ops per DPAS are
inherent to per-(row, K32) weight scales combined with per-(token, K32)
activation scales.

Consequence for prefill, from the measured 36.2 TOPS rather than the peak:
706 tok/s, i.e. **22x** faster than today's per-token loop - TTFT 8192 goes
257 s -> 11.6 s. Reaching the systolic peak would be 7143 tok/s.

The identified route past the epilogue is the Phase-0 opt-in tier: `s4 x s4`
runs at K=64 (733 TOPS, double systolic depth), so quantizing prefill
activations to int4 halves both the DPAS count and the epilogue count for the
same work - roughly 2x, and it is exactly the "W4A4 for prefill-heavy work"
slot recorded in `xpu/PROBE_RESULTS.md`.

#### Chunked prefill — `qwn_prefill_chunk`

- Added `x_rmsnorm_chunk` (batched (1+w) RMSNorm, one work-group per token row)
  and `qwn_prefill_chunk(tokens, n, pos0)` (`decode.cpp`). Every projection now
  runs ONCE per chunk through the RC8 GEMM instead of once per token.
- Attention, the depthwise conv and the DeltaNet recurrence stay token-serial
  inside the chunk. The recurrence is inherently sequential, but its inputs are
  all precomputed by the batched projections, so the chunk path is equivalent
  to the per-token path by construction rather than by approximation.
- Refuses (returns -5) unless every layer weight is W4-K32, so a `TQ_XPU_W8`
  selector covering a layer weight sends the caller back to the per-token loop
  rather than silently mixing tiers.
- Gate `xpu/tools/prefill_check.py`: same prompt through both paths in two
  separate processes, then 32 greedy steps. **100.00% agreement (32/32), no
  divergence.** State-equivalent.

Measured TTFT on one B70 (`xpu/tools/bench_prefill.py`, chunk 512), against a
per-token reference of 32.7 tok/s:

| ctx | TTFT | prefill tok/s | vs per-token loop |
|---|---|---|---|
| 512 | **1.26 s** | 406 | **12x** |
| 2048 | **5.62 s** | 364 | **11x** |
| 8192 | **33.87 s** | 242 | **7x** |

So TTFT at 8192 goes 257 s -> 33.9 s. Short of the 706 tok/s the GEMM alone
would allow, and the rate DECAYS with depth (406 -> 242) because what remains
is now the token-serial work: each token's attention re-reads its whole history
and each recurrence step reads and writes 6.3 MB of DeltaNet state. Those, not
the projections, are the next prefill bottleneck.

Still to build, in value order: (1) register-resident chunked DeltaNet
recurrence - the state is already register-resident per token in the rewritten
delta kernel, so carrying it across a chunk turns per-token state traffic into
one read and one write per chunk; (2) batched chunk attention so a chunk's
queries share one KV pass; (3) W4A4 prefill activations for the K=64 DPAS.

Measurement caveat: the TTFT figures above were taken while a vLLM-XPU server
was starting on the other card. Our engine issues ~350 host submissions per
token, so it is unusually sensitive to host CPU contention - a decode benchmark
that normally completes in 40 s later exceeded 1200 s while that server plus a
client benchmark were saturating cores. The prefill numbers are therefore a
conservative floor and are pending a re-measurement on an idle host. This
launch-bound sensitivity is itself a finding: it is the same submission
overhead that command-list replay would address.

#### Operational note — do not co-locate two model loads on one card

Two benchmark processes were launched in the same batch and therefore ran
CONCURRENTLY on card 0, each loading the 22.6 GB TQF into a 32 GiB card. The
result was `UR_RESULT_ERROR_DEVICE_LOST`, and the device stayed wedged
afterwards: `sycl-ls` still enumerates both B70s and `xpu-smi discovery` still
reports `Device State: normal`, but every new Level-Zero context on
`ZE_AFFINITY_MASK=0` fails immediately, while `ZE_AFFINITY_MASK=1` runs the
layout probe to a clean PASS. So neither `xpu-smi` state nor device enumeration
detects this condition - only an actual context creation does.

Rules going forward:

- One model-loading process per card, ever. Serialize benchmark invocations;
  never issue two in a batch that could execute in parallel.
- Cross-card work is safe and measured (two replicas at 99.95% per-stream
  retention), but that is one load per card, not two on one.
- B70 #0 is display-attached on this box: Xwayland holds `/dev/dri/card2` and
  several desktop apps hold its render node `renderD129`. A device reset or
  driver reload to recover it is therefore user-visible and must not be done
  unattended.
- Backups and large artifacts belong OUTSIDE `/tmp`, because **`/tmp` on this
  box is a 30 GB tmpfs - RAM, not disk.** Two consequences bit us in one
  session: a pre-change tarball placed there was destroyed by a reboot, so the
  fp32-KV state could not be restored (current state is archived at
  `~/xpu-fp8kv-state.tgz`); and the same-model Qwen3.8 vLLM comparison failed
  with "Disk quota exceeded" after 11 GiB because an 11 GiB venv plus an 11 GiB
  model download exhausted the RAM disk. `/` has 217 GB free. Anything large
  goes on `/` or `$HOME`; `/tmp` is only for small scratch.

#### E4M3 KV cache

- KV cache converted from fp32 to E4M3 with one FP16 scale per (token,
  kv_head) row: 258 bytes per row against 1024, a 3.97x cut. This is what makes
  long context reachable at all - fp32 KV needs 34.36 GB at 256k, more than the
  card - and it matches vLLM-XPU, which also caches fp8, so it is the correct
  configuration for a fair head-to-head.
- The decoder is a branch-free bit-trick:
  `bitcast((c & 0x7F) << 20) * 2^120`. Shifting 20 places lands E4M3's mantissa
  at fp32 [22:20] and its exponent at [26:23]; the multiply rebases the
  exponent from fp32's bias of 127 to E4M3's 7. It is EXACT for subnormals with
  no fixup branch: at e=0 the fp32 subnormal reading is m*2^-129, and
  m*2^-129 * 2^120 = m*2^-9, precisely E4M3's (m/8)*2^-6.
- `qwn_e4m3_selftest` proves it: **all 254 finite codes decode BIT-IDENTICALLY
  to the CUDA-faithful reference** (`kernels_core.cpp`, CUDA
  forward_qwen.cu:513-525), every code round-trips through the encoder, and the
  worst encode error over E4M3's normal range is 0.0586, just inside the
  2^-4 = 6.25% bound the format implies. Code 0x80 (negative zero) is excluded
  from bit-equality: the reference returns +0.0 there because it tests the
  magnitude bits before applying sign. Numerically identical.
- Quality is unaffected: parity 10/10 probes at cos >= 0.999979 with a 100%
  argmax chain, and teacher-forced agreement unchanged at 94.16%.
- Two performance traps, both measured rather than guessed:
  1. The obvious `ldexp`-plus-branch decode (~7 ALU ops/element) made the fp8
     cache SLOWER than the fp32 one it replaced: 27.16 vs 30.34 tok/s at 4k.
     The bit-trick lifted that to 28.80.
  2. The remaining gap was the LOAD WIDTH. One E4M3 byte per lane moves only
     16 bytes per instruction - a quarter cache line - so 4x fewer bytes came
     with 4x more load instructions. Having each lane take four consecutive
     codes as one dword restores full 64-byte coalescing, which forces the
     in-kernel slot map `d = 64*(t/4) + 4*lane + t%4` (the cache layout itself
     is unchanged). That lifted 4k from 28.80 to **29.95 tok/s**.
- Net decode against our own fp32 baseline: -1.3% at 4096 (29.95 vs 30.34) and
  -5.2% at 8192 (27.67 vs 29.18). The unpack still costs slightly more than the
  bytes it saves, so this is not yet a decode win - it is the price of 4x the
  context and 4x the concurrency slots, and the same fp8 format the vendor uses.
  Remaining ideas: decode straight to the accumulator without the intermediate
  float, or fold the 2^120 rebase into the row scale to remove one multiply per
  element (needs an overflow guard, since scale * 2^120 must stay finite).

#### First measured vLLM-XPU comparison

Our own reproduction, not the cookbook's self-reported table. vLLM 0.27.1 +
vllm-xpu-kernels 0.1.12.3, torch 2.13.0+xpu, one B70 on card 1, fp8 KV,
prefix caching off, 5 measured requests after a discarded warm-up.

**Reference-only caveat: this run is Qwen3.6-27B-GPTQ-Int4 while our figures
are Qwen3.8-27B. Same architecture and size, different weights.** A same-model
Qwen3.8 run is downloading and will be the headline.

| ctx | vLLM-XPU TTFT | knivesysl-xe TTFT | |
|---|---|---|---|
| 512 | **0.50 s** | 1.26 s | we lose 2.5x |
| 2048 | **4.17 s** | 5.62 s | we lose 1.35x |
| 8192 | 53.86 s | **33.87 s** | **we win 1.6x** |

The shape is the interesting part. Their prefill rate collapses with length -
1024 tok/s at 512 down to 152 tok/s at 8192 - while ours is far flatter, 406
down to 242. So we already win TTFT at long context and lose it at short, where
their fused prefill has the better constant factor and ours still runs
attention, the depthwise conv and the DeltaNet recurrence token-serially inside
each chunk. Fixing exactly those is what would win short-context TTFT too.

Their decode was 26.99 tok/s at 512 ctx on Qwen3.6, against our 31.92 (fp32) /
32.43 at 128 (fp8) on Qwen3.8 - suggestive but NOT a valid comparison until the
same-model run lands.

### Current Phase-2 acceptance gates

- Per-projection DPAS cosine: at least 0.99; preferred at least 0.995.
- Teacher-forced top-1 agreement against CUDA FP6: at least 90%; preferred 94%
  or higher. The scalar-XPU ceiling under the different batching order is
  97.28%.
- Sustained dense decode: at least 25 tok/s, measured after warm-up over at
  least 256 tokens with initialization/repacking excluded. Current default
  measures 31.6 tok/s, so the gate is raised to 30 tok/s going forward.
- Deterministic reset and repeated-run behavior on both B70 devices.

#### Wide prefill — query-tiled attention, and what the profile then said

TTFT was `O(n^2)` with a large per-token constant: 1605 s at 64k, 26.7 minutes.
The cause was that `pf_full_layer` batched only the GEMMs and left attention,
the conv and the DeltaNet recurrence running ONE TOKEN AT A TIME, so every
token re-read the whole KV history in its own kernel launch. That is CUDA's
scalar reference kernel `k_tq_wide_attn` (forward_qwen.cu:2830), which CUDA
itself abandons above ~16k. Its fast path states the principle at :2944 —
"16 query rows reuse the same K", which flips the kernel from memory-bound to
compute-bound at ~170 FLOP/byte.

Ported that shape: `x_prefill_kv_write` (all T tokens' K/V, byte-identical to
the per-token store) then `x_prefill_attn`, causal flash attention over 16-query
tiles with a per-row staircase mask. 16 subgroups x 16 lanes, subgroup s owning
query row s, so the online softmax is subgroup-local and needs NO work-group
barrier — the only barriers stage K/V. Lane l owns dims [16l, 16l+16): ~40
floats live, inside the 128-GRF budget the G=6 decode variant overran. K/V
stage as E4M3 BYTES, so 64 keys cost 33 KB of SLM where fp32 would need 132 KB
and not fit; the fp8 cache pays off a second time here.

Gate: `prefill_check.py` 100.00% greedy agreement, zero divergence, at 64 and
512 tokens, in separate processes so no state can leak.

Measured: TTFT 6.358 -> 5.168 s at 2k (+23%), 39.131 -> 28.723 s at 8k (+36%).
Decode is untouched by design. The gain grows with depth because attention is a
larger share deeper in.

Then `TQ_XPU_PROFILE_PREFILL` was added and the profile corrected the roadmap.
At pos0=0, T=512, 2.264 ms/token:

| phase | ms | share |
|---|---|---|
| projection | 734.9 | **63.4%** |
| delta | 344.9 | 29.7% |
| attention | 48.3 | **4.2%** |
| activation | 16.5 | 1.4% |
| norm | 14.9 | 1.3% |

Attention is 4.2%, against the ~80% claimed from arithmetic before measuring.
Two of my estimates in a row were wrong in this area; the instrumentation went
in precisely because of that, and it should be read before any further prefill
work.

**The wall is the batched GEMM.** Those projections move 15.05 GB of weights,
whose bandwidth floor at 602 GB/s is 25 ms. They take 735 ms — 29x the floor.
That is consistent with `bench_gemm`: RC8 reaches 35.4 of 366 TOPS, 10% of DPAS
peak, because the epilogue evaluates PER K-TILE (~40 scalar ALU ops per DPAS's
8192 MACs, and K=5120 means 160 k-tiles per output element). Vectorizing the
per-token scale/sum fetches already bought 30.2 -> 36.2 TOPS; the structural
route past it is the tier Phase 0 identified — `s4 x s4` at K=64 and 733 TOPS,
which halves BOTH the DPAS count and the epilogue count — or coarser weight
scale granularity for the prefill path only, at a quality cost that
`qwn_w4_grid_probe` can price before any kernel is written.

#### Vendor baseline — what vLLM-XPU can and cannot do on this silicon

Two corrections to earlier claims in this file, both from measurement.

**1. XPU graph capture does nothing for this model class.** `b70_ai_things`
FINDINGS.md reports 50.37 tok/s PIECEWISE vs 61.55 FULL+Triton (+22.2%) and an
85.87 tok/s accepted arm, against the 19.98 tok/s measured here — so the first
conclusion was that our eager serve config was unfair and the comparison had to
be retracted. It was then tested rather than assumed: graph capture enabled via
`VLLM_XPU_ENABLE_XPU_GRAPH=1` plus
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY",...}'`, verified in
the log as `Capturing CUDA graphs (decode, FULL)`.

Result: **no change.** Decode 19.979 vs 19.98 tok/s at 2k, 10.297 vs ~10.5 at
8k, TTFT 4.18 vs 4.11 s. The reason is visible in the dumped compilation
config: `splitting_ops` contains `vllm::linear_attention`,
`vllm::qwen_gdn_attention_core`, `vllm::gdn_attention_core_xpu` and
`vllm::olmo_hybrid_gdn_full_forward`. Splitting ops are where the graph is CUT,
and this architecture has 48 GDN layers, so FULL capture cannot span the work
that matters — it captured 4 sizes in 2 s using 0.12 GiB, i.e. small non-GDN
fragments. Graph capture is structurally unavailable to vLLM-XPU on hybrid GDN
models. Their 50-86 tok/s arms therefore rest on MTP spec-decode (`MTPTOK=5`,
worth 2-3x), their private fused GDN kernels, and/or MoE models — not graphs.

So the comparison stands, scoped precisely: **no spec-decode on either side,
stock kernels, identical architecture.**

| ctx | vLLM-XPU | ours | raw | roofline-normalized |
|---|---|---|---|---|
| 2048 | 19.98 | 31.20 | 1.56x | **1.24x** (63% -> 78% of ceiling) |
| 8192 | 10.30 | 27.63 | 2.68x | **2.1x** (33% -> 69%) |

The roofline column is the honest one: their weights are 19 GB and ours 15.05
GB, so part of the raw gap is format, not kernel. Their TTFT is client-side over
HTTP+SSE including queue delay while ours is in-process, so our TTFT figure is
the optimistic side and its 2k loss should be read as a ceiling.

**2. The same-model Qwen3.8 comparison is blocked at the VENDOR, not here.**
`~/models/qwen38-27b-nvfp4-radixark` is an exact architectural match to our
ship model (5120 hidden, 64 layers = 48 linear + 16 full, 24/4 heads, hd 256,
48 linear value heads), mixed FP8/NVFP4 compressed-tensors. Serving it on
Intel's own newest image, `intel/llm-scaler-vllm:0.26.0-b1` published the same
day, fails at engine init:

    ValueError: Failed to find a kernel that can implement the NVFP4 linear
    layer. Reasons:

There is no NVFP4 kernel for Xe2 in the vendor stack. That is precisely what
`xpu/PROBE_RESULTS.md` measured on silicon in Phase 0 — native `s4 x s4` DPAS
at 733 TOPS, no FP4 MAC — and it is why `b70_ai_things` carries private fused
kernels (`nvfp4_f8scale_kernel_gdn/_xpu_C.abi3.so`,
`libgdn_attn_kernels_xe_2.so`) that are not in the vendored tree. Their Qwen3.8
W8A8 artifact is also 35 GB and needs TP=2 to fit, against our 15.05 GB on one
card.

Independent confirmation of the Phase-0 format decision: integer int4 was the
correct native tier, and the FP4 path Intel's own tables advertise does not
execute on this hardware.
