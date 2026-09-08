# level-up-xpu — the knivesysl-xe campaign doc

This is the living strategy document for knivesysl-xe: the from-scratch SYCL/Level-Zero
inference engine for Qwen3.8-27B on Intel Arc Pro B70. It records what we decided,
why, what we measured, and what we killed. **Append-only discipline: nothing tried is
ever deleted — a rejected experiment with its number is a lesson, and lessons are
load-bearing.** Update this file as work lands; every claim carries a measurement or
names the microbench that will produce one.

We are optimizers, but we are also building the thing others will stand on. The bar
is not "competitive with the vendor stack" — it is to beat every lab **everywhere on
the SLO surface, including concurrency at large context, by a wide margin**, on
hardware nobody else is taking seriously. Single model first (Qwen3.8-27B), by
design: every constant burned into kernels is an optimization class the generic
engines structurally cannot use. Generalization later means templating a working
engine, not designing an abstract one.

**Current implementation (2026-09-07):** shared paged KV, hybrid APC, continuous
decode batching, and packed multi-request prefill are implemented. Prefill now
uses native XMX QK/PV on profitable shapes, with a scalar reference path for
small work. Projection GEMMs preserve each request's split-K arithmetic when
packed; recurrent GDN state remains per slot. Numerical policy, exact test
coverage, and paired latency measurements are recorded in the
[current changelog](../CHANGELOG.md#xmx-prefill-and-packed-prompt-execution-2026-09-07).
Older measurements below are retained history, not claims about today's build.

Companion documents:

- `xpu/PROBE_RESULTS.md` — Phase-0 silicon facts. The final arbiter for any claim
  about what this hardware does.
- `xpu/CHANGELOG.md` — the full measured history. This doc states strategy; the
  changelog holds the receipts.
- `xpu/docs/paged-kv-implementation-plan.md` — retained 2026-09-06 rollout
  specification: 14 work items, kernel sites, capacity/admission/APC contracts,
  and original validation criteria. Its pre-port source observations are a
  historical snapshot; paging and serving integration are now implemented.
- `xpu/docs/gpu-optim-intel/archive/pages/` — the Intel GPU Optimization Guide
  (2025.2), scraped per chapter. Reference for occupancy math, GRF modes, SLM bank
  rules, LSC prefetch builtins, immediate command lists. NOTE: 13 pages were
  truncated at 50 KB by the scraper (footer `[Showing lines ...]` baked into the
  file) and `overview-*.md` is absent; the `Source:` URL in each header serves the
  full text. The "Immediate Command Lists" chapter exists only upstream.
- Vendor evidence lives in `vendor/` (see `vendor/README.md` for why each clone
  exists and the upstream PRs/issues we filed).

---

## 1. The machine, reduced to budgets

Probe-verified on both B70s (BMG-G31, Xe2, 32 Xe-cores x 8 XVE = 256 XVE, 32 GiB),
oneAPI 2026.1.1. Source: `xpu/probe/dpas_probe.cpp`, `xpu/PROBE_RESULTS.md`.

| budget | value | consequence |
|---|---|---|
| DRAM read | 602 GB/s achievable (99% of spec) | decode at B=1 is a weight-streaming problem: 15.05 GB/step -> 25 ms floor -> ~40 tok/s ceiling at the current format |
| DPAS | 733 TOPS s4xs4 (K=64) / 366 s8xs4 (K=32) / 183 bf16 / 91.6 tf32 | at B=1 the step does 0.14 ms of math per 25 ms of reading — a 178:1 idle systolic array. Compute is free during decode |
| fp16 DPAS | anomalously slow (63-69 TOPS) | quarantined; bf16 everywhere. Revisit only after a driver update |
| launch | 1.42 us async in-order | ~350 submissions/token; measured 5.8 ms/step of small-kernel overhead = the entire 82%->100% roofline gap |
| fast memory | 16 MB aggregate GRF + 4 MB SLM + L3 | the register file is the largest managed fast pool on the chip — bigger than SLM. Industry code treats GRF as scratch; we treat it as a residency tier |
| P2P | 2.0 GB/s between cards | TP2 dead on arrival (oneCCL source confirms no rescue: its fast path rides the same peer link, alternatives stage through host). Dual-card = two replicas + routing |
| Xe-core occupancy | 64 HW threads/core; WG threads = wg_size/sg_size; SLM co-residency gates | a 256-thread SG16 work-group = 16 threads = 25% of a core; >32 KB SLM/WG caps co-residency (opt-guide thread-mapping chapter, verified against our shapes) |
| GRF modes | 128 regs/8 threads or 256 regs/4 threads per XVE | the G=6 attention formulation (~208 floats/lane) spills at 128 and fits at 256 |

Every idea in this document is one of four moves:
**fewer bytes per token, more tokens per byte, read bytes fewer times, or don't read
them at all.**

## 2. The SLO contract

Historical planning snapshot (2026-09-04, one B70 unless noted); current status
and measurements are linked above:

| SLO | owned by | current | ceiling / target |
|---|---|---|---|
| decode tok/s, n=1 | weight bytes + orchestration | 31.6 (E4M3 KV: 29.95 @4k) | 38.7 at current format; format rungs move the ceiling itself; spec wave multiplies by acceptance |
| TTFT | prefill GEMM epilogue + serial DeltaNet | 406 tok/s @512 -> 242 @8k | 706 at measured RC8 rate; W4A4 targets ~2x that; must beat vendor's short-context constant (they win <=2k, we win >=8k today) |
| ITL p50 | decode tok/s | 31-33 ms class | tracks decode |
| ITL p99 | host scheduling jitter | untested on XPU; CUDA batched arm bleeds 400-1500 ms cells | device-resident loop is THE p99 project — structural fix, not tuning |
| throughput n>1 | RC8 batched decode + scheduling | not implemented | B=8 projects ~320 tok/s aggregate from the roofline [INFERENCE] |
| context / slots | KV bytes | E4M3+fp16 scale = 258 B/row; 256k fits | int4-K tier halves again; slots quadruple vs fp32 baseline |
| depth retention | attention traffic | 89.4% at 8k (G=3) | G=6 halves remaining attention traffic; int4-K halves KV bytes |
| quality gate | format ladder | 94.55% TF vs CUDA-certified scalar stream (ship default) | >=94% for any default; opt-in tiers may trade below with the probe+TF ladder as gate |

**The out-of-the-box mandate: concurrency at large context.** Everyone degrades at
n>1 x 32k-256k — vendors lose to KV bytes, scheduler jitter, and prefill collapse.
Our stack attacks all three at once: int4-K KV (8x smaller than fp32, 2x smaller
than the vendor's fp8), G=6 attention (KV read once per 6 heads), device-resident
scheduling (no host jitter in p99), flat prefill (per-chunk RC8 + chunk-parallel
DeltaNet), and APC-style hybrid checkpoints ported from the CUDA engine (deltanet
state is not rewindable, so checkpoint reuse — not dense KV reuse — is the only
correct prefix cache for this architecture, and we already own that design). No lab
has all five; most have none, because their frameworks cannot graph-capture, cannot
change KV formats freely, and cannot touch GDN.

## 3. The six theses

1. **Kill the host — the decode loop belongs on the device.** The op-per-kernel
   decomposition is a CUDA-cultural artifact. Vendor graph capture is structurally
   dead on GDN models (measured: `VLLM_XPU_ENABLE_XPU_GRAPH=1` FULL capture = no
   change, splitting_ops cut the graph at all 48 GDN layers; see CHANGELOG "vendor
   baseline"), and SYCL-Graph host-synchronizes before every replay
   (`vendor/llvm/unified-runtime/.../command_list_manager.cpp`). So we leapfrog:
   persistent grid-resident step, global-atomic stage barriers, doorbell input
   (`zexCommandListAppendWaitOnMemory`), on-device argmax/sampling, host as
   detokenizing spectator. Risk: no forward-progress guarantee across work-groups on
   Xe2 — sizing must be exactly-resident. Fallback ladder: more fusion -> L0
   command-list replay with mutable args -> full persistence.
   **STATUS 2026-09-04: megakernel form killed by probe 1** — a software global
   barrier costs max(~1.2 us, ~36 ns x participants) vs 1.48 us for a kernel
   boundary; the in-order queue's boundary IS the efficient global barrier on
   this silicon. The thesis survives as: doorbell-driven device loop (probe 2),
   continued fusion, and flag-based producer-consumer sync between few WGs
   (safe: residency is exactly predictable and over-residency drains).
2. **Speculation is the default shape of a step, not a bolt-on.** RC8 serves 8
   activation rows from the same 256-byte weight fragment: the marginal weight cost
   of 8 candidates is zero. Every decode step becomes a wave: 1 committed token + 7
   tree-speculated candidates (n-gram archive first — CUDA side measured up to 2.03x
   — later an on-device drafter / native MTP head), verified by staircase mask
   inside the wave, no host round-trip. Where drafts don't verify, the wave carries
   real batch rows instead. One primitive, three products: single-stream speed, spec
   decode, batched decode.
3. **Format is the roofline; we own a format lab.** `qwn_w4_grid_probe` prices any
   quantization grid on real weights in minutes, on silicon, before a kernel exists.
   It already killed the NVFP4 port (0.0943 vs 0.0783 rel-L2 at equal 4.50 bits/w)
   and the vendor stack has no NVFP4 Xe2 kernel anyway. Next rungs: selective k16
   (5.0 b/w quality tier, shipped as opt-in), then sub-4-bit for the fat MLPs
   (int3 + outlier handling — the probe prices it in an afternoon). Every 0.5 b/w
   shaved is ~+11% decode ceiling.
4. **Depth is a KV-format problem.** E4M3 KV bought vendor parity; the CUDA tier
   (rotated asymmetric int4-K + Hadamard, E4M3-V) is 2x smaller again and flips the
   depth-retention curve against vLLM-XPU. Pair with G=6 GQA attention unlocked by
   GRF256. Attention at depth is bandwidth-bound, so halving threads/XVE there is
   nearly free.
5. **Prefill gets the 733-TOPS tier.** Prefill is our only compute-bound phase.
   W4A4 activations halve both the DPAS count and the epilogue count (the measured
   wall: ~40 scalar ALU ops per DPAS caps RC8 at 36.2/366 TOPS). Chunk-parallel
   DeltaNet (chunk-64 WY/UT; math in our CUDA TU and cross-checkable against
   `vendor/intel-xpu/vllm-xpu/vllm-xpu-kernels/csrc/xpu/gdn_attn/xe_2/`) kills the
   token-serial 30% of prefill. Vendor prefill collapses with length (1024 -> 152
   tok/s); ours is flat. Win the constant and we win TTFT everywhere.
6. **Own the schedule, not the codegen.** Our hand-written GEMV runs at 87% of its
   byte-roofline — IGC codegen is not the bottleneck, orchestration is. The
   "compiler" we build is thin and ours: AOT for BMG (`-fsycl-targets=spir64_gen
   -Xs "-device bmg"`), per-kernel GRF mode, `[[intel::kernel_args_restrict]]`,
   inline vISA for DPAS, IGC shader dumps + unitrace spill reports as the feedback
   loop. joint_matrix has no INT4 on BMG (dead for our formats); oneDNN stays off
   the hot path.

## 4. Build vs take

Rule derived from the model itself: 48 of 64 layers are gated DeltaNet — the least
commoditized kernel in the industry (vendors need private kernels for it; graph
capture cannot span it). **Build everything the hybrid makes unusual. Take only
contracts and math from the commoditized parts.**

| atomic step | owns | verdict | note |
|---|---|---|---|
| W4A8/W4A4 GEMV+GEMM | decode tok/s | **built — keep ours** | 87% roofline, format-fused; vendor kernels dequantize to 16-bit DPAS and cannot match |
| weight format | decode ceiling | **build** (lab exists) | probe-first; no vendor iterates formats without framework drag |
| DeltaNet decode core | ITL p50 | **built — keep ours** | full-occupancy single-read-pass, 2.90 -> 1.13 ms |
| DeltaNet chunk scan | TTFT | **built; keep the chunkwise math** | slot-local convolution/recurrence is retained in packed prefill |
| attention prefill | TTFT | **built — XMX QK/PV plus scalar reference** | automatic dispatch, paged KV, and one packed attention grid; no external runtime dependency |
| attention decode | ITL at depth | **build (G=6 next); steal PackGQA idea** | GRF256 unlocks it; split-K combine layout from FA/vendor worth reading |
| KV format | context, slots, retention | **build** | int4-K + Hadamard is our own tier; vendors stop at fp8 |
| 2D block IO / DPAS contracts | all kernels | **take contracts only** | sycl-tla `copy_xe_2d.hpp`, IGC LSC legality tables, Triton-XPU gating conditions; our one-hot probes stay final arbiter |
| norm/quant/silu/argmax | ITL p50 | **built — keep fusing** | the win is fusion count; no library provides that |
| spec decode wave | n=1 tok/s | **build** | fused into RC8; frameworks pay a scheduler round-trip per round |
| launch/step orchestration | ITL p99 + roofline gap | **build (the ladder)** | uncontested ground; vendor graphs dead on GDN |
| batched scheduling / paged KV / APC | throughput, TTFT under load | **built** | shared pool, native hybrid checkpoints, packed prompt waves, and decode batching |
| sampling beyond argmax | quality, tail | **build (tiny)** | keeps the device loop closed |
| tokenizer / HTTP shell | none | **take** | Python + HF tokenizer, as today; zero alpha |
| collectives / TP2 | — | **neither** | 2 GB/s P2P; replicas + routing. Closed |
| profiling | all | **take tools, build gates** | unitrace, IGC dumps, xpu-smi; our TF/parity/probe gates are the truth |

We take exactly three things: hardware contracts, published attention math, and the
serving shell. Every row with alpha is ours — and none starts from zero.

## 5. The microbench queue

Doctrine: thesis -> microbench -> smoke -> integration. Every item is a standalone
probe in the `dpas_probe.cpp` mold with an explicit kill criterion, run **before**
any inference integration. Status column is updated in place; killed items move to
the lessons log with their numbers.

| # | probe | funds | kill criterion | status |
|---|---|---|---|---|
| 1 | **persistence probe** — one kernel, exactly-resident grid, global-atomic barrier, 10k iterations of a dummy 3-stage pipeline | thesis 1 | barrier round > ~2 us, or any hang at exact-resident sizing | **KILLED 2026-09-04** (megakernel form). `xpu/probe/persistence_probe.cpp`, card 1: barrier = max(~1.2 us, ~36 ns x nwg) -> 4.2 us @128 WGs, 19.0 us @512; tree barrier WORSE (7.8-35.9 us); kernel boundary = 1.48 us/stage with full-grid work. Survivors: doorbell rung (probe 2), fusion, flag-based producer-consumer sync. Keeper facts: residency formula exact (2048/(wg/16)); over-residency +1 drains cleanly, 0 ordering errors — cooperative designs are safe. Lesson 14. |
| 2 | **doorbell probe** — zex WaitOnMemory chain, host writes token id, device wave consumes, N steps zero host syncs | thesis 1 fallback rung | steps/sec no better than current submission path; #982/#983 sharp edges (unregistered host ptr snapshots, no timeout) | **DONE 2026-09-04** — primitive priced, `xpu/probe/doorbell_probe.cpp` (v1-v4 postmortems in header). Single-ping p50: persistent RMW-spin 5.17 us < submit+wait 5.62 < load-spin 6.00 < submit+poll 6.50 < zex ping 7.34 (p99 10.0, 1900/1900 clean). Verdict: NO mechanism beats plain submission on raw latency — doorbell value is per-step amortization + host-CPU decoupling, so this rung only pays PAIRED WITH REPLAY (item 2b). Sharp findings: lessons 17-19 (host-USM RMW-writeback clobbering, wide-spin pathology at 258 ms/step, the 5-7.5 us equivalence class). |
| 2b | **replay probe** — record a ~350-kernel step in a regular L0 command list once; per step: zex wait -> zeCommandQueueExecuteCommandLists (or immediate-append replay); measure step overhead + host CPU time vs 350 live SYCL submissions, and p99 under artificial host load | thesis 1 (the surviving rung) | replayed step no cheaper in host CPU AND no better p99 under load than live submission; or mutable-args update cost eats the win (b70 XPUGraph leak doc names the sharp edge) | **DONE 2026-09-04 — FUNDED, 15-19x.** `xpu/probe/replay_probe.cpp` (350 cmds/step, 200 steps): live-append 454-529 us host-CPU/step; replay-execute 28.1; doorbell-replay 29.9; imm-append 31.1. Sharp edge: >1 pre-queued executions parked on a zex semaphore wedge (window=1 required); single wait+write in a regular list verified working. Design note: per-step variability moved into device-resident state (tiny end-of-step advance kernel) makes recorded args fully static — no mutable command lists needed. Lesson 20. Next: engine integration = record-per-shape + replay. |
| 3 | **GRF256 A/B** — grouped attention rebuilt `-ze-opt-large-register-file`, `bench_depth` fixed-depth cells, G=3@128 vs G=6@256 | thesis 4 | G=6@256 slower at pos 8192 | **KILLED 2026-09-04** (scalar G=6 form). Built `-DTQ_ATTN_G=6` + per-kernel `grf_size<256>` (no spills, property verified via AOT). Fixed-depth cells, paired session: G6 27.6/24.1 tok/s at 2k/4k vs G3 31.6/30.3; segment scaling (kSegments 3->6, same 24-WG grid) recovered to 29.7/27.5/24.0 at 2k/4k/8k vs G3 31.7/30.3/28.1 — still loses, gap WIDENS with depth. Mechanism: GRF256 = 4 threads/XVE halves latency-hiding on a dependent-load-bound KV walk; the occupancy tax beats the halved traffic. The G=6 win requires the DPAS-matrix formulation -> item 3b. Plumbing kept: `-DTQ_ATTN_G`, scaled segments, per-kernel GRF property pattern. Lesson 22. |
| 3b | **DPAS-matrix decode attention** — Q-tile(6 heads packed as M) x K^T via dpas + 2D block loads; GRF128 kept | thesis 4 (the real depth lever) | slower than the G=3 scalar kernel at pos 8192, or spills at GRF128 | **SHIPPED opt-in 2026-09-04.** Probe (`dpas_attn_probe.cpp`): QK^T 1.47x vs scalar G3x2 at cos 0.999999; full pipeline (softmax, P/V staging, segment partials) cos 0.999989 vs fp64 ref. Engine: `TQ_XPU_ATTN_DPAS=1` branch in kernels_seq — Q pre-packed to bf16 A-fragments by a RoPE prep kernel, per-(kv_head, segment) subgroups (runtime nseg <= 32), per-lane K/V e4m3->bf16 rebuild, base-e online softmax matching the merge contract, runtime-nseg merge. **Live depth cells: 31.59->32.19 (+1.9%) @2k, 30.28->31.44 (+3.9%) @4k, 27.98->30.08 (+7.5%) @8k — the gap GROWS with depth, inverse of the killed G6-scalar curve.** TF 92.61% PASS (-0.78 vs the 93.39 k32 band; bf16-QK reassociation class). Kept opt-in pending band review; occupancy headroom documented (single-subgroup WGs; V staging not yet shared). |
| 4 | **W4A4 GEMM probe** — s4-activation RC8 path on one matrix in `bench_gemm`; TOPS vs the 36.2 wall + activation-quant error | thesis 5 | <1.5x over W4A8 RC8, or activation error that the TF ladder prices above the k16 tier | **FUNDED 2026-09-04 (probe 1.77-1.78x) -> INTEGRATED same day as the opt-in TTFT tier.** Probe: `xpu/probe/w4a4_gemm_probe.cpp` (lesson 23). Engine: `TQ_XPU_K64` selector repacks W4 weights to one FP16 scale per K64 tile (grid probe scheme 10: rel-L2 0.0890 vs k32's 0.0783, +13.6%; codes layout unchanged — two lane-major K32 tiles ARE the K64 B fragment), `x_quantize_act_chunk_s4` + `x_gemm_w4a4` (dpas.s4.s4.8.8), hybrid per-weight prefill dispatch, `tools/w4a4_check.py` gates. Certification: scalar-vs-GEMV-k64 anchor cos 0.9957-0.9962 PASS; gemm_check under ksh cos 1.00000000; w4a4 check cos 0.9963 (= analytic s4-act noise; band 0.995); prefill_check 100% (32/32) pure and hybrid; parity chain 100%. Real-weight kernel: 42 -> 71 TOPS (1.62-1.69x). **Live: prefill 465->545 tok/s @512 (hybrid `gate,up,down`, TTFT 1.10->0.94 s, +17%), ->600 all-k64 (+29%); decode 32.85->34.07 tok/s under hybrid (+3.7%, halved MLP scale traffic). TF price: hybrid 92.22% PASS, all-k64 89.88% FAIL (k32 band 97-98%) — ship default stays k32; `TQ_XPU_K64=gate,up,down` is the documented TTFT tier.** Lesson 27. |
| 5 | **sub-4-bit grid rungs** — new schemes in `qwn_w4_grid_probe` (int3+outliers, mixed 3/4 by row norm) | thesis 3 | rel-L2 ladder says the bytes don't pay | **KILLED 2026-09-04** (simple rungs). Ladder on layers 0/1/32 mlp_gate: int3 asym @3.50 b/w = rel-L2 0.168-0.170 (2.14x ship tier; NVFP4's rejected 0.094 for scale); int3 sym 0.193; mix3/4 at err-ratio 1.8 selected ~0% int3 groups (achieved bits 4.50 = pure int4). The bits ladder cliffs below 4. Surviving avenues logged, not pursued: outlier-split storage, codebooks, per-tensor TF-guided selective 3.5 (lm_head lesson warns of surprises). Ceiling lever shifts to KV/attention/spec. Lesson 21. |
| 6 | **int4-K KV micro** — Hadamard+quant kernel + attention-read microbench on synthetic cache vs E4M3 path | thesis 4 | score error above the E4M3-tier band, or unpack cost > bytes saved (the E4M3 ldexp lesson) | **KILLED 2026-09-04 (accuracy), with the byte thesis CONFIRMED.** `xpu/probe/int4kv_probe.cpp`: 32 MB synthetic cache (past L2), outlier channels, engine-shaped walk (1 query/subgroup, q register-hoisted). Walks near roofline: E4M3 ~430 GB/s, s4+H 440-570 GB/s — **1.6-2.1x at 1.6x bytes ratio** (unpack cost is NOT the blocker; anticipated kill arm refuted). But score error: row-scale s4+H cos 0.9929 (24x the E4M3 (1-cos) band), g32 scales 0.9957 (14x band) — far above the tier band both rungs. FWHT+quant append cost negligible (13-37 us/1k rows). Depth-byte savings must come from the attention formulation (3b) or hybrid/selective schemes, not raw 4-bit K. Survivors logged, not pursued: g16, int5/6-K, error feedback, calibrated rotations. Lesson 26. |
| 7 | **RC8 spec-wave verify probe** — 1+7 staircase verify vs 8 sequential RC1 steps on one layer; byte-exact accepted prefix | thesis 2 | verify wave costs more than acceptance can repay at measured n-gram rates | **FUNDED 2026-09-04 (probe: wave = 1.11-1.17x one GEMV, break-even 0.11-0.17 drafts) -> SHIPPED same day as `qwn_spec_wave` (design 9.3, lesson 29).** The wave IS the 8-row prefill path (staircase + GDN semantics free); new pieces: `x_gemm_w8a8` batched lm_head, GDN snapshot/rewind via cached wave inputs, host verify, `tools/bench_spec.py` prompt-lookup drafter. Live: decode 33.07 -> 48.24 tok/s (1.46x, 256 steps) and 33.91 -> 51.08 (1.51x, 512 steps, hybrid k64), 2.99 tokens/wave, **100.00% greedy agreement in every run** (768 verified tokens). Next rungs: real n-gram archive (host, design 9.4), tree drafts, wave under replay. |
| 8 | **restrict + AOT A/B** — `[[intel::kernel_args_restrict]]` on hot kernels; `-fsycl-targets=spir64_gen -Xs "-device bmg"`; before/after `bench_gemm` + `bench_decode` | thesis 6 | none (free knowledge either way) | **DONE 2026-09-04**, both null results. restrict on all 8 hot DPAS kernels: GEMV loop 33.0 vs 33.1 tok/s, RC8 wash at T>=256 (42.4=42.4 TOPS), 5-8% SLOWER at T=32-128 -> reverted (lesson 15). AOT bmg-g31: steady-state identical, init identical at warm JIT cache (10.18 s both), zero spill warnings across the TU; kept as `TQ_XPU_AOT=1` in `build.sh` (cold-cache determinism + required plumbing for probe 3's IGC options). Bonus: idle-host uplift quantified (lesson 16). |
| 10 | **host-USM TQF staging** — `malloc_host` (or prepare_for_device_copy) staging buffer in `upload_payload`; load-time A/B | ops QoL | none; opt guide measured 26.9 -> 45.4 GB/s H2D with import | **DONE 2026-09-04, SHIPPED.** `tqf_loader.cpp` staging buffer switched to `sycl::malloc_host` (malloc fallback kept): init 10.2 -> 9.35 s avg (best 8.99), ~-8%. `gemm_check` gate PASS (cos 1.00000000). |
| 12 | **host-DRAM tier probe** — steady-state block swap rates host<->device with imported host USM (prefill-sized KV blocks, APC checkpoint images); prices the CPU-DRAM peripheral as: checkpoint spillover (CUDA Track-D analog), cold-KV parking for 32k-256k sessions, draft-model home | concurrency x large context (slots beyond 32 GiB) | swap latency > re-prefill cost at the measured 468+ tok/s prefill | **FUNDED 2026-09-04, ~400x.** `xpu/probe/hostdram_swap_probe.cpp`: 6.5-6.8 GB/s both directions at 4-512 MB blocks (PCIe floor on this B70 — far below the opt guide's Max-series 26.9-45.4; plan around ~6.8). Restore of a 4k-token KV prefix (134 MB): 21.7 ms vs 8.9 s re-prefill. Contended cell: 134 MB D2H racing a 1 GiB read kernel (590.8 GB/s) = **0% mutual overhead** — swaps are free under decode. Cold-KV parking, APC spillover, and draft-model home are all viable; design lands in the CPU-peripheral pass. Lesson 25. |
| 13 | **CPU-peripheral design pass** — n-gram archive maintenance, sampling post-processing, and scheduling on host cores (they are idle once replay lands); no device probe needed, lands with items 7/11 | decode multiplier + p99 | — | pending |
| 14 | **learning cadence** — recurring: re-read the relevant opt-guide chapter + vendor kernel (sycl-tla FMHA, vllm-xpu-kernels GDN) BEFORE each kernel build; log deltas here | all | — | recurring |
| 9 | **L1 LSC prefetch on attention** — `__builtin_IB_lsc_prefetch_global_*` with `LSC_LDCC_L1C_L3C` on the strided KV walk (linear-walk prefetch already measured a loss) | thesis 4 | slower at pos>=2048, like the L3 variant was on the GEMV | pending |

After the queue, the roadmap is priced, not believed: "persistence buys X ms p99,
W4A4 buys Y s TTFT at 8k."

## 6. Shipped ground (what is already under us)

Condensed from `xpu/CHANGELOG.md`; the receipts live there.

Current additions: XMX prefill attention and packed prompt execution are
qualified under the explicitly recorded numerical policy. The phase figures
below are older receipts, not a refreshed vendor comparison.


- Phase 0: DPAS rates, operand layouts (one-hot probes: A row-major bytes, B
  depth-major VNNI nibbles, D lane==column, s4 two's-complement, RC8 A = 8x32
  contiguous rows), 602 GB/s, 1.4 us launch, TP2 dead.
- Phase 1: scalar-correctness engine, TQF1 loader, CUDA parity (cos 0.9987 at layer
  64, 64/64 greedy, 97.28% TF).
- Phase 2: W4A8 asymmetric DPAS tier (zp in scale low nibble, exact integer-sum
  zero-point subtraction), fused split-K, fused norm+quant, rewritten delta core,
  fan-out GEMV, GQA-grouped attention G=3, 16-shard softmax. 31.6 tok/s = 82% of
  byte-roofline; format ladder measured, NVFP4 port rejected on evidence.
- Phase 3 (in progress): RC8 batched GEMM (cos 1.00000000 vs RC1 first try),
  chunked prefill 7-12x TTFT, E4M3 KV (branch-free decoder, bit-identical selftest),
  wide prefill query-tiled attention, prefill profiler. Head-to-heads: we beat
  eager vLLM-XPU 1.24-2.1x roofline-normalized; their graph capture does nothing on
  GDN; their stack cannot run NVFP4 at all.

## 7. Lessons log (append-only — nothing leaves)

Every entry: what we tried, the number that killed or shaped it, the rule extracted.

1. **L3-targeted prefetch on a linear walk loses.** Triton-XPU's
   Matrix2DBlockPrefetch idiom, 2 tiles ahead: 31.2 vs 31.6 tok/s. Xe2's L2
   prefetcher already covers linear walks; extra issue slots cost more than they
   save. (ZML's `.cg`-modifier comment predicted it.) Rule: prefetch only what the
   HW prefetcher cannot see — strided/indirect walks — and prefer L1 targets.
2. **The obvious E4M3 decoder made fp8 KV slower than fp32.** ldexp+branch decode:
   27.16 vs 30.34 tok/s. The branch-free bit-trick (`bitcast((c&0x7F)<<20) *
   2^120`, exact for subnormals) plus 4-codes-per-lane dword loads recovered it.
   Rule: at KV-decode volumes, per-element ALU count and load width dominate the
   bytes saved; price the decoder before the format.
3. **Byte savings without coalescing is a loss.** One E4M3 byte per lane moved 16
   bytes per instruction — 4x fewer bytes, 4x more loads, net slower. The in-kernel
   slot remap (`d = 64*(t/4) + 4*lane + t%4`) restored 64-byte lines. Rule:
   sub-group-contiguous access is the unit of bandwidth, not element count.
4. **More split-K is not more speed.** GEMV splits 8 -> 16 on mid-size shapes:
   22.35 -> 22.16 tok/s. And the M=48 projections at 32 splits paid a second launch
   for 0.2 us of math. Rule: cap splits at one work-group so the reduction fuses;
   launches are the expensive resource.
5. **Fewer launches is not automatically faster.** Collapsing norm+quant into one
   1024-thread work-group halved launches but dropped 40 XVEs to 8: norm 1.356 ->
   2.005 ms. Rule: width first, launch count second — unless the phase is
   submission-bound (then see thesis 1).
6. **lm_head contradicts the monotone-bits rule.** k16 (5.0 b/w) on lm_head was
   *worse* than s4 (4.5 b/w): 92.61% vs 93.77% TF, and slower. 248320 rows of thin
   top-1 margins make rounding luck beat scale granularity. Rule: quality tiers are
   per-tensor empirical, never assumed from bits.
7. **W8 was over-applied.** Old default W8 on q,k,v,o,out,lm_head cost 9.6%
   throughput for a one-position TF difference vs lm_head-only. Rule: only the
   logit-emitting tensor has earned 8 bits so far.
8. **Depth benchmarks on a zeroed cache lie.** `bench_depth` optimism grows with
   depth (+0.8% @2k -> +8.1% @32k): a zeroed KV reads faster than data
   [INFERENCE: memory-path compression]. Rule: any depth number past 32k requires a
   real prefill; the tool survives only as a fixed-depth A/B probe.
9. **Two model loads on one card wedge the device.** Concurrent 22.6 GB loads ->
   `UR_RESULT_ERROR_DEVICE_LOST`, context creation fails while xpu-smi still says
   "normal"; only a context create detects it. Card 0 is display-attached — a reset
   is user-visible. Rule: one loading process per card, serialize benchmark
   batches, prefer card 1 for experiments.
10. **/tmp is RAM on this box (30 GB tmpfs).** It destroyed a pre-change backup at
    reboot and killed a vLLM comparison with "Disk quota exceeded". Rule: artifacts
    live on `/` or `$HOME`; references persisted to `~/knivesysl_xpu_refs/`.
11. **Conv-into-delta fusion deferred with reasoning.** The depthwise conv is
    launch-bound (0.40 ms for 0.033 ms of bytes) but a race-free fused variant still
    needs a second kernel for the state shift; eliminating it properly needs a
    circular conv state folded into the in_qkv GEMV epilogue — touching the tuned
    GEMV for ~1.2%. Rule: log the real fix, don't ship the half-fix.
12. **Estimates in the attention/prefill area were wrong twice.** Pre-measurement
    claims (attention ~80% of prefill; grouped decode crossover) both missed; the
    profiler (`TQ_XPU_PROFILE_PREFILL`) showed attention at 4.2% and projections at
    63%. Rule: this domain gets instrumentation before roadmap — that is why the
    profile hooks exist and why this queue is microbench-first.
13. **The scraped opt-guide archive is imperfect.** 13 chapters truncated at 50 KB
    with the pager footer baked in; `overview` page lost. Rule: for the truncated
    chapters, the `Source:` URL is authoritative; re-scrape before trusting a tail.
14. **A software global barrier loses to the kernel boundary on Xe2 — and a
    tree barrier loses to the centralized one.** `persistence_probe` on card 1:
    centralized barrier max(~1.2 us, ~36 ns x nwg) = 4.2/10.1/19.0 us at
    128/256/512 resident WGs; two-level tree at branch 8/16/32 measured 7.8 to
    35.9 us — hierarchy adds sequential atomic hops and an extra release
    propagation that cost more than one hot cacheline's serialization. A
    kernel-boundary pipeline runs 1.48 us/stage with full-grid work, barely
    above the 1.43 us empty launch. Rules: NEO's direct-submission ring is the
    cheapest all-to-all sync — use kernel boundaries for global sync; reserve
    in-kernel sync for producer-consumer flags among few WGs; CUDA-derived
    cooperative-kernel intuition does not transfer. Keeper facts: resident WG
    bound = total_threads/(wg_size/16) held exactly for wg 64/128/256, and
    over-residency by +1 drained cleanly with zero ordering errors across ~4.6M
    barrier rounds — bounded-spin + abort-drain is the safe pattern for any
    future cooperative kernel.
15. **`kernel_args_restrict` is a no-op on our hot DPAS kernels — and mildly
    harmful mid-T.** All 8 hot kernels annotated: GEMV decode loop 33.0 vs
    33.1 tok/s, prefill 468/423 vs 467/422 tok/s, RC8 GEMM identical at
    T>=256 but 5-8% slower at T=32-128 (30.6 -> 28.1 TOPS at T=32). Reverted.
    Probable cause: the inline-vISA DPAS asm already limits reordering, and
    the kernels' distinct-base-pointer access patterns leave IGC nothing to
    hoist. AOT (bmg-g31) likewise changed nothing at steady state and nothing
    at init with a warm JIT cache (10.18 s both arms, dominated by disk +
    repack); kept as opt-in `TQ_XPU_AOT=1` for cold-cache determinism and as
    the plumbing for per-kernel IGC options. Rule: opt-guide checklist items
    are hypotheses, not free money — this codebase's hand-tuned kernels have
    already consumed most generic headroom; measure before adopting.
16. **Idle-host uplift is large and now quantified.** Same binary, same card,
    idle host vs the recorded contended-session numbers: RC8 GEMM 42.4 vs
    36.2 TOPS at T=1024 (+17%), chunked prefill 468 vs 406 tok/s at 512
    (+15%), per-token loop 33.0 vs 32.7. Confirms the changelog's caveat that
    the earlier prefill numbers were a conservative floor taken while a vLLM
    server was starting on the other card. Rule: every recorded number carries
    host state; A/B arms must be paired within one session, and cross-session
    comparisons of absolute numbers are invalid without an idle-host note.
17. **Device-side atomic reads of host-USM clobber concurrent host stores.**
    doorbell_probe v3/v4: host stored db=2 (immediate readback confirmed 2);
    seconds later the host read db=1 — the value the spinning kernel had been
    reading. A LOAD-only device spin regressed the location. Poll rate
    (~1.5M/3s ~ 2 us each) matches PCIe atomic round trips. [INFERENCE on
    mechanism: IGC lowers system-scope atomic loads on host-USM as RMW
    (writeback of the read value); the observed clobbering itself is certain.]
    Confirmed by fix: host hammer-stores the doorbell while polling -> 400/400
    steps clean at p50 6.0 us (load-spin) / 5.17 us (fetch_or spin). Rules:
    a one-shot host write to a host-USM location being spin-read by a kernel
    is UNSAFE on this stack; either hammer the store, or use zex
    WaitOnMemory (the CCS poll does not corrupt — 1900/1900 clean). Any
    future flag-based producer-consumer design keeps its flags in DEVICE
    USM, with only device-side writers.
18. **Never point many spinners at one host-USM line.** A.wide (128 WGs
    spinning + host hammering one line): p50 258 ms/step, p99 1.6 s — four
    orders of magnitude off. One agent watches a doorbell: the CCS (zex) or a
    single WG.
19. **All host<->device round-trip mechanisms live in one 5-7.5 us class.**
    p50: persistent RMW-spin 5.17, submit+wait 5.62, load-spin 6.00,
    submit+poll 6.50, zex ping 7.34 us. Plain in-order submission is already
    near-optimal for a single sync — NEO earns its keep. Consequence: the
    device-driven loop pays only as amortization (one sync per STEP + host
    CPU freed) which requires pre-recorded per-step work: the replay probe
    (queue 2b) is the decision point for thesis 1's surviving rung.
20. **Command-list replay is the orchestration win: 15-19x host CPU.**
    replay_probe (350 copy-commands/step as the command-stream stand-in):
    live per-command appends cost 454-529 us host-CPU/step (~1.4 us/command —
    this is the engine's host-contention sensitivity in one number); a
    recorded regular list replayed per step costs 28.1 us; the self-rearming
    doorbell variant ([zexWait(db>=1), commands, zexWrite(db=0)], db doubling
    as the completion flag) 29.9 us. Sharp edge: MORE THAN ONE pre-queued
    execution parked on a zex semaphore wedges — keep the execute window at 1
    (single wait+write in a regular list verified working; minimal upstream
    repro worth filing). Design rule for engine integration: keep recorded
    args STATIC by moving per-step variability (token id, pos, KV offsets)
    into device-resident state advanced by a tiny end-of-step kernel — no
    mutable command lists required.

21. **The bits ladder cliffs below 4.** int3 asym at 3.50 b/w carries 2.14x
    the ship tier's reconstruction error (rel-L2 0.168 vs 0.0783) on the
    shapes that dominate decode, and an error-ratio-gated 3/4 mix finds ~0%
    of K32 groups where int3 is acceptable — this weight distribution has no
    group-granular slack to harvest. 4.5 b/w (int4 asym, fp16/K32, zp in the
    scale nibble) stands as the byte-optimal uniform-grid point on this
    model. Rule: the decode-ceiling lever is NOT naive weight bits below 4;
    it is KV bytes, attention traffic, and speculation. Sub-4-bit only
    returns via structurally different storage (outliers/codebooks), each of
    which must first repay its own unpack cost (E4M3 ldexp lesson).

22. **GRF256's occupancy tax beats halved traffic on a latency-bound walk.**
    Scalar G=6 grouped attention (register-blocked, per-kernel grf_size<256>,
    verified spill-free): 29.7/27.5/24.0 tok/s at 2k/4k/8k vs G=3's
    31.7/30.3/28.1 — and the gap widens with depth, the opposite of the
    traffic model. 256-register mode runs 4 threads/XVE instead of 8; on a
    dependent-load KV walk that halves latency hiding, which costs more than
    reading each KV row 4x instead of 8x. Segment scaling (same 24-WG grid)
    recovered 2 tok/s but not the mode tax. Rules: (a) never buy registers
    with occupancy on a latency-bound kernel; (b) the G=6 traffic win must
    come from the DPAS-matrix formulation at GRF128 (queue 3b); (c) diagnose
    GRF-mode experiments with AOT spill warnings before trusting run numbers.

23. **A DPAS functor without `reqd_sub_group_size(16)` can compile SIMD32 and
    kill the device.** The w4a4 probe's struct kernels (engine-identical math,
    attribute forgotten) died with `UR_RESULT_ERROR_DEVICE_LOST` in 3 s: at
    SIMD32 the lane id runs 0-31, fragment loads walk off their 256/512-byte
    regions, and the `(M1, 16)` inline-vISA dpas executes under a mismatched
    execution size. `dpas_probe.cpp`'s functors survive by compiler luck, not
    by contract. Rule: every kernel touching subgroup-shaped layouts or inline
    vISA carries the attribute explicitly - on functors it prefixes the
    `operator()` declaration (`[[sycl::reqd_sub_group_size(16)]] void
    operator()(...) const`), never trailing after `const` (parses as a type
    attribute and is rejected).

24. **The spec wave is nearly free on the projection stream.** An RC8 T=8
    GEMM on a real layer weight costs 1.11-1.17x one RC1 GEMV (96-102 us vs
    86-87 us, layers 0/1/32) while carrying 8 token rows — the same weight
    fragment serves all rows, so 7 draft candidates ride for 11-17% overhead.
    Break-even is 0.11-0.17 accepted drafts per wave; any n-gram scheme
    clearing ~15% single-draft acceptance pays. And the accepted-prefix
    contract is real on silicon: wave output rows 0-3 are byte-identical
    regardless of what rides in rows 4-7 (DPAS row independence + per-token
    quantization, `qwn_specwave_check`). Thesis 2 stands measured: default
    execution should be the wave, not the single-token step.

25. **The PCIe tier is slow, free, and still wins by 400x.** Measured B70
    host<->device swap floor: 6.5-6.8 GB/s each way (NOT the opt guide's
    26.9-45.4 GB/s Max-series numbers — B-series PCIe reality; corroborates
    llm-scaler's documented PCIe limitations). But a 134 MB KV-prefix restore
    is 21.7 ms against 8.9 s of re-prefill, and a bulk copy racing a
    590 GB/s read kernel adds 0% overhead to either side (DMA and DRAM reads
    coexist; the copy taxes ~1% of bandwidth). Rule: park cold KV and APC
    checkpoints in host DRAM aggressively, overlap swaps with decode, and
    never size a design around burst PCIe numbers from other silicon.

26. **On the KV walk the bytes pay; the bits don't.** int4-K with Hadamard
    rotation walks 1.6-2.1x faster at a 1.6x byte ratio, near roofline both
    arms, and the s4 unpack (shift/mask/convert x2 per byte) costs nothing
    measurable — the E4M3-ldexp fear does not transfer to nibbles. What
    kills it is precision: score cosine 0.9929 (row scale) / 0.9957 (g32)
    vs E4M3's 0.9997 — 14-24x the tier's (1-cos) band, even with rotation
    spreading the outlier channels. Rule: the depth lever keeps E4M3-class
    K precision; go get the depth bytes from the attention formulation
    (queue 3b) or hybrid schemes, and re-test any sub-8-bit K against the
    score band before believing walk numbers. Also: an engine-shaped
    microbench (q hoisted, serial row stride) reads 3x faster than a naive
    one - never accept a walk measurement below ~70% of roofline as format
    evidence.

27. **The K64 scale trade is a tier, not a free lunch — and the codes come
    along for free.** dpas.s4.s4.8.8 demands ONE weight scale per K64 tile;
    two adjacent lane-major K32 tiles already form the exact K64 B fragment,
    so the repack only touches scale granularity. The measured chain: weight
    rel-L2 +13.6% (scheme 10) -> GEMV anchor cos 0.9962 -> decode TF 89.9%
    all-k64 (FAIL) / 92.2% MLP-only (PASS) vs the k32 tier's 97-98%. The
    speed side is real and live: prefill +17% hybrid / +29% all, decode
    +3.7% (halved MLP scale traffic on the GEMV walk), W4A4 GEMM
    1.62-1.69x on real weights. Rules: (a) an int-DPAS K-depth change IS a
    weight-format change - price it on the TF ladder, not just rel-L2;
    (b) confine format experiments with selectors so quality costs land
    only where the speed lands; (c) activation s4 noise (cos 0.9963/GEMM)
    is the same error class as CUDA's shipped E4M3 acts - the gate that
    matters is end-to-end continuation, where it holds 100%.

28. **Port the math, then port the occupancy.** The CUDA chunk kernel's
    algebra transplanted verbatim and passed prefill_check 100% on the first
    run - but its launch shape (one 128-thread WG per head, 48 WGs = 19%
    lane occupancy) ran 21% SLOWER than the serial loop it replaced, exactly
    the under-occupancy its own CUDA comments warned about. The fix was the
    same trick our decode-fast kernel already shipped: 1024-thread WGs with
    the dk contraction split G=8 ways, and each thread's 16 state values
    held in registers across ALL T/CK sub-chunks (state read/written once
    per layer chunk, not 2x(T/CK) times). v2: +10%/+9%/+6% live prefill at
    512/2k/8k, still 100% exact vs the loop. Rule: a correctness-first port
    is the right first step, but never A/B it against a tuned baseline
    until the launch geometry matches the machine - and steal occupancy
    patterns from your own shipped kernels before inventing new ones.

29. **The wave was already built - it just didn't have a name.** Design
    9.3's "staircase attention" and "GDN wave semantics" turned out to be
    the 8-row prefill path verbatim: pf attention stores-then-attends
    causally (the staircase), and the 9.2 chunk kernel at T=8 is exactly one
    sub-chunk. The only new engine pieces were per-row lm_head (x_gemm_w8a8
    streams the 1.27 GB head once for 8 rows), GDN state REWIND (snapshot +
    re-advance from cached wave inputs; KV needs none - dead rows are
    position-masked and overwritten), and host verify. First live run:
    1.46x at 100.00% greedy agreement over 256 tokens; 512-token runs
    1.43x/1.51x (k32/hybrid), 2.99 tokens/wave from bare prompt-lookup
    drafting. Rules: (a) before building a "new" execution mode, check
    whether an existing path IS that mode under different indices; (b) spec
    exactness on this stack is real - rewind via snapshot + chunk re-advance
    reproduced plain greedy byte-for-byte across 768 verified tokens.

30. **Know which band you are comparing against.** Lesson 27 priced k64
    against "the k32 tier's 97-98%" - that figure is the SAME-tier
    two-config eps band. The correct baseline is the W4A8-vs-scalar TF
    band, measured today on the current TU at 93.39% (and recorded at
    ~94.2-94.6% in the W8 experiments). Corrected k64 prices: MLP-only
    92.22% = -1.2 points, all-k64 89.88% = -3.5 points. The hybrid tier is
    cheaper than lesson 27 implied (superseded there, pointer here); the
    all-k64 tier still fails the 0.90 gate only barely and remains opt-in.
    Rule: every agreement number carries its reference tier; a "band" is a
    (config A, config B, corpus) triple, never a bare percentage.

31. **The batched decode step's natural width is 8, and rows below that are
    not free — they're pre-paid.** The batched GEMMs enforce `T % 8 == 0`
    because RC8 means one 256-byte weight fragment serves eight activation
    rows. So `qwn_decode_batch` always runs the 8-wide shape and `n < 8`
    real rows ride it with zeroed pad rows. Measured at 512ctx (K64=all):
    n=1 40.6 ms, n=8 54.4 ms — per-step cost rises 34% while rows rise 8x,
    which is the whole batching thesis on a weight-stream-bound decode.
    Aggregate 24.7 -> 147.1 tok/s (4.33x). Corollary: serving at
    concurrency 1 through the batch path is a 1.4x *loss* versus the GEMV
    path (40.6 vs 29.5 ms) — the scheduler must keep the single-stream
    path for n=1.

32. **The batch step is epilogue-bound, not stream-bound — and the fix is
    already named.** At n=8 the step moves 15.05 GB in 54.4 ms = 277 GB/s,
    46% of the 602 GB/s roofline, while the n=1 GEMV path runs at 87%.
    Cause: at T=8 the RC8 GEMM has `tgroups = 1`, so `tasks = M/16` and the
    small-M projections (`mlp_down`, `o_proj`, `linear_out`, M=5120) launch
    only 20 work-groups — 16% lane occupancy. Padding to T=16 would double
    weight traffic, so the lever is **split-K for the batched GEMM at small
    T**, exactly the fused split-K the GEMV path already uses. Projected
    ceiling if it reaches GEMV efficiency: 8 x 1000/29.45 = 271 tok/s
    aggregate. This is the highest-value unbuilt kernel on the board.

33. **On this model a prefix cache must be a state checkpoint, not a block
    table.** 48 of 64 layers are recurrent: GDN conv + recurrent state
    cannot be re-derived from a partial KV cache, so vLLM's 16-token block
    reuse has no analogue. What works is a whole-state snapshot at a chunk
    boundary. Measured at 512 tokens: 175.8 MB, save 54.0 ms, restore
    55.2 ms (3.2 GB/s), versus 1082.9 ms to re-prefill — 20x. The KV part
    is only 16.9 MB of that; the 159 MB GDN part is **position-independent**,
    so the payoff grows with prefix length (break-even ~26 tokens, ~95x
    projected at 4k) and the cache costs a flat 176 MB per distinct prompt.
    Restore leaves the slot byte-identical: continuation matched the cold
    path 32/32.

## 9b. Serving: continuous batching + prefix cache (built 2026-09-05)

Both landed and live-measured on card 0, same binary, same session.

**ABI added.** `TQ_XPU_SLOTS` (1..8, fixed at init) slot-strides every
per-sequence buffer; `qwn_set_slot/get_slot/num_slots`, `qwn_reset_slot`,
`qwn_decode_batch(slots, tokens, positions, n, out)`, and the checkpoint
trio `qwn_ckpt_bytes/save/restore` plus `qwn_host_alloc/free`. Kernels are
untouched — slotting is pointer arithmetic at the call sites, and the batch
step reuses `pf_gemm`/`pf_mlp` wholesale.

**Gates (all byte-exact, not eps-band).**
- `tools/slot_check.py` — two prompts decoded interleaved per token on
  slots 0/1 vs fresh single-slot subprocesses: 48/48 both slots.
- `tools/batch_check.py` — same two prompts advanced through
  `qwn_decode_batch` vs serial single-slot: 48/48 both slots.
- `tools/apc_check.py` — checkpoint/restore continuation vs the cold path:
  32/32.

**Aggregate throughput** (`tools/bench_batch.py`, K64=all):

| ctx | n=1 GEMV | n=1 RC8 | n=4 | n=8 | vendor best |
|---|--:|--:|--:|--:|--:|
| 512 | 34.0 | 24.7 | 85.5 | **147.1** | 158 (c8) |
| 8192 | 29.1 | 21.9 | 59.8 | **84.4** | 12.0 (c4) |

Shallow we are at 0.93x of their best cell; at 8k we are **7.0x** theirs,
because their aggregate collapses with depth (158 -> 12) and ours does not
(147 -> 84). Split-K (lesson 32) is the lever for the shallow cell.

**Scheduler** (`tools/serve_batch.py`): admission into free slots
mid-flight, one bounded prefill chunk per iteration so admission cannot
starve decode, one batched decode step over every generating slot,
immediate slot release on completion, APC lookup at chunk granularity.
16 requests sharing a 512-token prefix, 1.5 arrivals/s, gen 32, 8 slots:

| metric | APC off | APC on |
|---|--:|--:|
| wall | 18.63 s | **11.39 s** |
| TTFT mean | 4.06 s | **0.33 s** (12.3x) |
| TTFT p99 | 6.79 s | **1.07 s** (6.3x) |
| ITL mean | 175.8 ms | **51.9 ms** (3.4x) |
| prefill tokens | 8208 | 1040 (7168 skipped) |
| cache | — | 176 MB, 14 hits / 2 misses |

### 9b.1 OpenAI/vLLM-compatible server (built 2026-09-05)

`xpu/tools/serve_openai_xpu.py` mirrors the CUDA server's surface
(`tools/serve_openai.py`): stdlib `ThreadingHTTPServer`, HTTP/1.1, manual
chunked SSE, `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/health`; Qwen chat template with tools, `<tool_call>` round-trip,
`reasoning_content` split at `</think>`, stop strings, usage accounting,
per-request `x_knivesysl` stats; `n>1`/`logprobs` -> 400.

It differs from the CUDA server in the one way that matters: that one owns
ONE sequence and serializes requests behind a lock, while this one runs a
single engine thread over `qwn_decode_batch` and serves genuinely
concurrently. Handler threads submit and drain per-request queues; the
engine admits into free slots mid-flight, runs one bounded prefill chunk
per iteration, then one batched decode step.

Live, same client (`tools/bench_openai.py`) that measured vLLM-XPU:

| ctx | conc | TTFT s | ITL ms | per-stream | aggregate |
|---|--:|--:|--:|--:|--:|
| 512 | 1 | 1.31 | 30.5 | 32.8 | 32.8 |
| 512 | 4 | 4.15 | 66.8 | 15.0 | 59.9 |
| 512 | 8 | 8.30 | 85.3 | 11.7 | **93.8** |

`gen=256`; c1 matches the standalone engine (33.6 vs 34.0), so server
overhead is ~0. Surface + concurrency + APC gate: `serve_smoke_xpu.py`
(batch_avg 6.66, 3072 prefix tokens reused on the second wave).

Three defects the gate caught, all worth the lesson:

34. **A prompt whose length is a multiple of 8 has no tail token.** The
    chunked prefill consumed the whole prompt, so the single-token step
    that produces the FIRST logits never ran and `-1` was emitted as a
    token id. Prefill must stop at `((left-1)//8)*8` and leave >= 1 token:
    re-running the last position instead would double-advance the GDN
    recurrent state.
35. **EOS text must never reach the client.** Emitting the token and then
    stopping leaked the literal `<|im_end|>` into `content`. Terminate on
    EOS *before* surfacing it.
36. **A prefix cache with no admission policy is a tax.** With unique
    prompts (the benchmark shape) the cache stored 104 checkpoints -
    8 GB of host USM and ~55 ms of engine-thread copy each - for a hit
    rate of exactly zero. Caching only on a prefix's SECOND sighting cut
    the cache to 171 MB and *improved* c8 aggregate 91.8 -> 93.8 and TTFT
    8.61 -> 8.30 s. Corollary: measure a cache's cost on the workload that
    never hits it, not just the one that does.

Also implemented from lesson 31: the scheduler routes a lone decoding row
through the GEMV path instead of the batch step (`n == 1` fast path).
Measured on the server: 19.56 -> 33.62 tok/s at c1. Writing the lesson
down was not enough - the code has to obey it.

**Honestly open.** Greedy only: the XPU tier has no sampler, so
`temperature`/`top_p`/`seed` are accepted and ignored with a warning in
`x_knivesysl.warnings` (host-side sampling needs a logits export).
Split-K RC8 (lesson 32) is unbuilt, so c8 aggregate is 93.8 rather than
the ~170 the roofline allows. No `/v1/responses`, no tool-call streaming
mid-generation (calls are emitted in the final chunk).


## 10. Op-level breakdown on silicon (2026-09-05)

Both paths profiled with the in-engine stage timers
(`TQ_XPU_PROFILE_POS`, `TQ_XPU_PROFILE_PREFILL`), ship tier, card 1.

### 10.1 Decode step — 33.1 ms, and it is a memory problem

| stage | ms | % | launches | bytes | effective |
|---|--:|--:|--:|--:|--:|
| projection | 26.92 | 81.3 | 256 | ~13.8 GB | 493 GB/s (**82% of roofline**) |
| lm_head | 2.29 | 6.9 | 1 | 1.27 GB | 554 GB/s (**92%**) |
| norm | 1.30 | 3.9 | 128 | 2.6 MB | launch-bound |
| delta | 1.04 | 3.1 | 48 | ~150 MB | 4x off roofline |
| attention | 0.70 | 2.1 | 16 | 6.5 MB | **9.4 GB/s** |
| activation | 0.39 | 1.2 | 64 | 4.5 MB | launch-bound |
| conv | 0.38 | 1.1 | 48 | small | launch-bound |

Pure-bandwidth floor is 15.05 GB / 602 GB/s = **25.0 ms**, so the step runs
at 73% of what the memory system allows. The 8.1 ms gap splits into:

- **~3.9 ms of small-kernel launch + ramp.** norm/conv/delta/activation/
  attention together move 164 MB — 0.27 ms of actual traffic — in 3.9 ms.
  At 1.43 us launch and ~4.7 us measured floor for a full-grid trivial
  kernel (`tp_allreduce_probe` floor arm), 304 launches is 1.4-3.5 ms of
  pure overhead. This is the replay lever (design 9.1), quantified.
- **~4.9 ms of projection roofline gap** across 256 launches, i.e. ~19 us
  each of ramp-up before the weight stream saturates.

Consequence: **at decode there is nothing algorithmic left in the
projections.** 82-92% of roofline means the only levers are fewer weight
bytes (format), more rows per byte (batch/spec), or fewer launches.

### 10.2 Prefill chunk — 976 ms at T=512, and it is a compute problem

| stage | ms | % | launches |
|---|--:|--:|--:|
| projection | 665.5 | 68.2 | 320 |
| **delta** (GDN chunk scan) | **227.0** | **23.3** | 48 |
| attention | 47.3 | 4.8 | 16 |
| activation | 20.6 | 2.1 | 64 |
| norm | 15.5 | 1.6 | 128 |

The projection number pins the ceiling exactly: 27.6e9 weights x 512 tokens
x 2 = 2.83e13 FLOP in 0.665 s = **42.5 TOPS**, matching the RC8 GEMM's
measured 42.4 TOPS wall — which is **11.6% of the 366 TOPS s8xs4 DPAS
peak**. Prefill is not bandwidth-bound at all; it is epilogue-bound, and
W4A4/K64 lifts the same kernel to 75.5 TOPS (20.6%). The second target is
the GDN chunk scan at 23.3% — 4.73 ms per layer per 512-token chunk.

37. **Decode and prefill are bound by opposite resources, so they need
    opposite fixes.** Decode: 82-92% of *memory* roofline, dominated by
    weight streaming — fix with format/rows/launches. Prefill: 11.6% of
    *compute* peak, dominated by GEMM epilogue ALU — fix with K64 and
    epilogue reduction. A single "make it faster" instinct applied to both
    wastes effort on the wrong half. Quoting one roofline percentage for
    "the engine" is meaningless.

38. **Fan-out is a launch-count lever and it pays.** `mlp_gate`+`mlp_up`
    share the post-norm activation; `q`+`k`+`v` share the input norm. Both
    were issuing separate GEMVs while the GDN 4-way input had used
    `x_gemv_w4a8_fanout` since Phase 2. Fusing them cut 352 launches to
    256 and measured projection 27.98 -> 26.92 ms, step 34.17 -> 33.11 ms
    (**-3.1%**), decode 32.5 -> 33.26 tok/s. Gates: gemm_check cos
    1.00000000, prefill_check 32/32, parity 10/10 + argmax 64/64.

### 10.3 TP=2 is FUNDED — the Phase-0 verdict was measured wrong

Phase 0 recorded "TP2 is dead on arrival - 2 GB/s cross-device". That
number is real but it answers the wrong question: tensor parallelism moves
**no bulk data**. Following vLLM's structure exactly
(`ColumnParallelLinear` shards output rows and needs no comm;
`RowParallelLinear` shards input columns and all-reduces its output;
`QKVParallelLinear` shards by head), Qwen3.8-27B needs **2 all-reduces x
64 layers = 128 per token of a 20 KB hidden vector** — 2.56 MB/token
against a 15.05 GB/token weight stream that TP2 **halves**.

`xpu/probe/tp_allreduce_probe.cpp`, both cards, one shared SYCL context:

| arm | us | note |
|---|--:|---|
| floor (device add, full grid) | 4.74 | small-kernel floor |
| p2p 1-way, 20 KB | 10.76 | 1.90 GB/s — confirms Phase 0 |
| p2p 2-way | 27.69 | **does not overlap**; duplex would be ~11 |
| **ar-p2p** = exchange + add, queue join as barrier | **37.33** | correctness PASS |
| ar-host (staged through shared host USM) | 42.98 | correctness PASS |

37.33 us x 128 = **4.78 ms/token of comm**. Applied to the measured 33.11 ms
step (projections 26.92 -> 13.46, lm_head 2.29 -> 1.15 vocab-parallel with
an argmax-pair reduce, delta/attn/conv/act halved by head sharding, norms
replicated at 1.30): **~17.2 ms compute + 4.78 ms comm = ~22.0 ms ->
45.5 tok/s, a 1.37x single-stream win** [INFERENCE: projection from
measured stage times; not yet an end-to-end TP2 run].

39. **A "dead" verdict inherits the workload it was measured on.** Phase 0
    killed TP2 on streaming bandwidth. TP2's actual traffic is 128 x 20 KB
    of latency-bound exchange, where the same link measures 10.76 us/hop
    and the full all-reduce 37.33 us. The kill was not wrong about the
    hardware, it was wrong about the algorithm. Rule: a bandwidth number
    can never kill a latency-bound design, and vice versa.

Open lever inside TP2: `p2p-2way` at 27.69 us against `p2p-1way` at 10.76
shows the two directions serialize. If they can be made to overlap the
all-reduce drops toward ~20 us (2.56 ms/token, ~1.5x). Reduce-scatter +
all-gather (what vLLM calls sequence parallelism) would halve the payload
again. Also note the choice is not free: 2 independent replicas give a
clean **2.0x aggregate** but no latency win, while TP2 gives 1.37x on
*every* stream plus half the per-card weight/KV footprint.


### 10.4 TP=2 substrate: built and gated (2026-09-05)

**Landed.** `TQ_XPU_TP` (default 1 = today's single-card engine) builds one
in-order queue per rank on consecutive Level Zero devices inside ONE shared
SYCL context — the shared context is what makes cross-card pointers legal
and is the configuration the probe measured. The design point that makes
this cheap: every allocator and transfer helper already routes through
`tq_q()`, so `tq_set_rank(r)` re-targets allocation, upload and launch with
**no change to any kernel**. TP costs the kernels nothing.

New: `tq_q_of(rank)`, `tq_tp_size()`, `tq_rank()`, `tq_set_rank()`,
`tq_allreduce(bufs, scratch, n)` (peer exchange -> queue join as the barrier
-> local add, the shape the probe timed at 37.33 us), and the model-free
export `qwn_tp_selftest(n)`.

Gate `tools/tp_check.py`: 2 ranks, per-rank allocation, all-reduce correct at
n = 5120 / 17408 / 262144 (rank r seeded (r+1)*i, every rank ends holding
sum_r (r+1)*i). **TP=1 verified regression-clean after the `tq_q()` rewrite**:
gemm_check cos 1.00000000, prefill_check 32/32, parity 10/10 + argmax 64/64,
decode 33.21 tok/s (33.26 before, inside noise).

**Not built: the sharded forward.** Honest statement of what remains and why
it is a multi-session build rather than a patch.

The blocker is a data-structure fact, not a kernel one: `g_qwen.layers[i]`
holds ONE `tq_qmma_weight_t` per projection. Sharded weights need a rank
dimension, which ripples through the loader, `decode.cpp`, every call site in
`kernels_dpas.cpp`, and every check. Executable plan, in dependency order:

1. **Rank dimension.** `tq_layer_t layers[L]` -> per-rank layer arrays
   (`g_qwen.rank[r].layers[L]`), with the TP=1 path aliasing rank 0 so the
   certified single-card code path is literally unchanged.
2. **Loader sharding.** `read_qmma` currently sets M/K/Mt/Kt/scale_cols and
   streams `d_A` + `d_block_scale_inv` straight from the file. Add
   `read_qmma_sharded(f, w[TP], M, K, mode)`:
   - *column-parallel* (shard M: q/k/v, gate/up, linear_in_*): the E2M3
     payload is row-major, so rank r's slice is a **contiguous** byte range —
     stream it directly, set M/2, and each rank repacks its own shard with
     the existing `w4_repack_kernel`. M/2 must stay a multiple of 16 (the
     row-group granularity); it does for every weight in this model.
   - *row-parallel* (shard K: o_proj, mlp_down, linear_out): rank r takes
     k in [rK/2, (r+1)K/2) for every row — **strided**, so stage the payload
     to host once (67 MB worst case, the loader already stages 64 MB chunks)
     and gather per rank. Set K/2, so `Kt` halves and the packed layout is
     exactly that of a K/2 weight. Scales take the matching half of
     `scale_cols`.
   - *replicated*: norms, embed, and the GDN per-head vectors.
3. **Forward.** Per layer: `tq_set_rank(r)` and launch each rank's half
   (queues are independent, so both cards run concurrently), join, then
   `tq_allreduce` after `o_proj` / `mlp_down` / `linear_out`. Attention and
   GDN shard by head, so their state and KV halve per card with no comm.
   Norms are replicated and run after the all-reduce, when every rank holds
   the full hidden.
4. **lm_head.** Column-parallel by vocab (248320/2 per card), then instead of
   all-gathering 993 KB of logits, each rank argmaxes its own shard and the
   host reduces the two (value, index) pairs — **16 bytes of comm** for what
   would otherwise be the single largest exchange in the step.
5. **Gate.** TP=2 vs TP=1 must agree on the argmax chain. Note the contract
   honestly: a K-sharded dot product sums in a different order, so TP=2 is
   **not** bit-exact against single-card by construction — it belongs in the
   eps-band class with a cosine + argmax-chain gate, like the format tiers,
   not in the byte-exact class.
6. **Matrix arm.** `bench_matrix_xpu.sh` gains a `tp2` arm; the interesting
   comparison is TP2 against **two independent replicas**, which give a clean
   2.0x aggregate but no latency win, versus TP2's projected 1.37x on every
   stream plus half the per-card weight/KV footprint.

40. **Build the substrate before the feature, and gate it model-free.**
    `qwn_tp_selftest` needs no weights, so the shared context, per-rank
    queues and all-reduce were proven correct in 0.5 s — before any of the
    loader surgery that would otherwise have made a plumbing bug look like a
    numerics bug. The same discipline that put `dpas_probe` before the
    kernels.


### 10.5 TP=2 weight sharding: primitive built and measured (2026-09-05)

**Landed.** `x_w4_shard(dst, src, rank, tp, mode, groups, ngroups)` slices an
already-repacked W4 weight into one rank's shard with **no repack**, because
the packed layout is row-group-major and k-tile-major inside each row-group:

    codes  : row-group g at  g * Kt * 256 bytes
    scales : row-group g at  g * (Kt >> ksh) * 16 uint16    (ksh=1 under K64)

so a column shard is a CONTIGUOUS range and a row shard is `Mt` contiguous
runs at a fixed stride. Both are pure memcpy. `mode 0` (column) takes a
`groups` list so a packed output shards per group rather than handing one
rank the whole of one group — `q_proj` is `[q | gate]` and `linear_in_qkv`
is `[q | k | v]`, exactly what vLLM's `QKVParallelLinear.output_sizes`
exists for. Each group's row count must be a multiple of `16*tp`.

Also landed: the rank indirection everything else needs. `g_qwen` is now a
macro over `*g_qwen_active`, one `tq_model_t` per rank, and `tq_set_rank()`
moves **queue and model together**. Every existing use site compiled
unchanged; TP=1 keeps rank 0 selected forever.

**Gate `qwn_shard_check(layer)` — deliberately single-card**, so a layout bug
cannot hide behind TP plumbing. On real `mlp_gate` weights (M=17408, K=5120),
layers 0 / 1 / 32:

| mode | cos | max_rel | bit-exact |
|---|--:|--:|---|
| column (concat of the two half-M shards) | 1.00000000 | 0 | **yes** |
| row (sum of the two half-K partials) | 1.00000000 | 0.0009-0.0019 | **no** |

41. **The column/row asymmetry in TP is a numerics fact, now measured, not
    an assumption.** A column shard splits output rows: each shard walks the
    same k-tiles in the same order over the same codes, so concatenating the
    halves is **byte-identical** to the unsharded GEMV. A row shard splits K:
    the two partials are summed at the end instead of accumulating in one
    chain, so it reassociates — cos 1.00000000 but max_rel ~1e-3. This is
    the proof that **TP2 cannot be bit-exact against single-card**, and it
    localises the inexactness precisely to `o_proj` / `mlp_down` /
    `linear_out` (the three row-parallel weights). TP2 therefore ships as an
    eps-band tier with a cosine + argmax-chain gate, like the format tiers —
    not in the byte-exact class the rest of this engine holds.

TP=1 re-verified after both changes: gemm cos 1.00000000, prefill 32/32,
parity 10/10 + argmax 64/64.

**Still open** (the sharded forward): a per-weight mode table + per-rank load
orchestration in the loader (the primitive is done; what remains is calling
it for all 496 weights and freeing the full copies to actually halve the
11.3 GB per card), the per-rank forward split at the all-reduce points, the
vocab-parallel lm_head with the 16-byte argmax-pair reduction, and the
eps-band parity gate.

### 10.6 The validated TP shard mode table (2026-09-05)

`qwn_shard_check_all(layer)` runs `x_w4_shard` against **every projection
family at its real shape and intended mode**, so the loader's loop over all
496 weights cannot trip a divisibility or scale-layout constraint at load
time. Column and replicate must reassemble bit-exactly; row is eps-band.
Both layer types, 15 families, **0 failures**:

| layer | weight | M | K | mode | cos | max_rel | exact |
|---|---|--:|--:|---|--:|--:|---|
| GDN | lin_qkv | 10240 | 5120 | col (grouped) | 1.00000000 | 0 | yes |
| GDN | lin_z | 6144 | 5120 | col | 1.00000000 | 0 | yes |
| GDN | lin_b | 48 | 5120 | **rep** | 1.00000000 | 0 | yes |
| GDN | lin_a | 48 | 5120 | **rep** | 1.00000000 | 0 | yes |
| GDN | lin_out | 5120 | 6144 | row | 1.00000000 | 3.7e-4 | no |
| attn | q_proj | 12288 | 5120 | col | 1.00000000 | 0 | yes |
| attn | k_proj | 1024 | 5120 | col | 1.00000000 | 0 | yes |
| attn | v_proj | 1024 | 5120 | col | 1.00000000 | 0 | yes |
| attn | o_proj | 5120 | 6144 | row | 1.00000000 | 1.9e-4 | no |
| both | mlp_gate | 17408 | 5120 | col | 1.00000000 | 0 | yes |
| both | mlp_up | 17408 | 5120 | col | 1.00000000 | 0 | yes |
| both | mlp_down | 5120 | 17408 | row | 1.00000000 | 1.3e-4..1.2e-2 | no |

Two findings the gate produced that a design document would have missed:

42. **`q_proj` needs no grouping; `linear_in_qkv` does.** The attention
    q-projection is **head-major** — `qg_base = head * 2 * hd`, i.e.
    `[q(hd) | gate(hd)]` per head — so a head shard is one contiguous row
    range. `linear_in_qkv` is **group-major**
    `[q(2048) | k(2048) | v(6144)]`, so a head shard there is three
    disjoint ranges and needs the grouped path. Same conceptual weight,
    opposite packing; only reading the consumer kernel settles it.

43. **Two GDN projections cannot be sharded at all, and the gate found it,
    not the plan.** `lin_a` and `lin_b` emit ONE scalar per value head, so
    M=48. At TP=2 that is 24 rows per rank, and 24 is not a multiple of the
    packed layout's 16-row group — `x_w4_shard` returned -6 for both on the
    first run. They are 123 KB each, so they **replicate**: ~12 MB per card
    across all 48 GDN layers, and each rank reads the slice for its own
    heads. Rule: a TP plan derived from head counts alone is wrong whenever
    the packed layout has a coarser row granularity than a head.

44. **Host stability postmortem, 2026-09-05: cause not established, four
    theories left unsupported.** Three hard cuts coincided with GPU work.
    Journal evidence weakened every mechanism proposed: (a) "wedged card ->
    Xe display poll -> IOMMU timeout" — the boot that hard-cut had ZERO
    `output_poll_execute` hog events, the 13 h boot that survived idle had
    six; (b) "cross-card P2P storms the root port" — the PME warnings came
    from `00:02.2`, which is not upstream of either B70 (both hang off
    `00:02.1`) and sits in IOMMU group 5, not 20; (c) "PME interrupt storm" —
    3649 lines on the 13 h boot that survived, 88 on the boot that died, 96
    in 35 min of a healthy boot; (d) `force-gaming-display.service` looping
    on a hardcoded connector — 320/197 restarts on the two boots that died,
    but 113 on one that stayed up, confounded with GPU load and not
    separable. Two bugs ARE confirmed and were mine: a spin bounded by 200M
    *iterations* (calibrated for register speed, run against PCIe-speed host
    USM at ~0.5 us/iter = 100 s/kernel), and a functor missing
    `[[sycl::reqd_sub_group_size(16)]]` that compiled SIMD32 and took a card
    down with `DEVICE_LOST` in 3 s. Rule: **bound spins by wall clock, never
    iteration count**, and correlation across four boots weakens a theory —
    it does not disprove a mechanism.

45. **A health gate that flags background noise is worse than none.** The
    first `gpu_health.sh` treated any `hogged CPU` line as a fault and
    reported UNHEALTHY on a machine whose cards both passed liveness. The
    counts show why that was wrong (lesson 44c). But blanket-demoting the
    pattern trades a false positive for a broad false negative — an xe/GuC
    worker stuck before a risky run is real. It now excludes the two task
    names measured as routine here (`output_poll_execute`,
    `pci_pme_list_scan`) and **gates on any other**. Also noted: this host
    has no crash-capture armed (`/sys/fs/pstore` empty, no `ramoops=`, no
    netconsole) while `crashkernel=` IS reserved — so a hard lockup loses its
    final journal writes and the next one is equally undiagnosable. kdump is
    the gap worth closing before chasing this further.

46. **Find out where a kernel's parallelism comes from before assuming it
    scales.** The batched RC8 GEMM took ALL of its work-group count from T
    (`tasks = rgroups * tgroups`). That is invisible at prefill, where T=512
    gives 64 t-groups and the grid saturates — and fatal at the decode batch
    shape, where T=8 gives ONE t-group and `mlp_down` (M=5120) produces 320
    tasks = 20 work-groups on a machine that wants >=128. Splitting K, using
    the GEMV's existing fused-SLM scheme unchanged, moved n=8 from 71.13 to
    55.01 ms/step, +29.3% aggregate (112.47 -> 145.42 tok/s). The lever was
    not the DPAS, the format, or the memory system: it was 16% lane occupancy.
    Corollary: `splits` must DIVIDE the subgroup count for the fused reducer's
    `gpw` to be exact — the GEMV heuristic only ever returns 16/8/1, and a
    plausible-looking `ceil(want/tasks)` producing 6 or 7 silently corrupts
    which partials each reader sums.

47. **A speedup claim needs a same-binary baseline and alternating arms.** An
    env override (`TQ_XPU_GEMM_SPLITS`) that forces the unsplit kernel costs
    nothing and converts "29% faster than the code I deleted" into a paired
    measurement. Run the arms ALTERNATING and in REVERSED order across trials:
    ours came back 55.00/55.02/55.01 vs 71.13/71.13/71.13, a 0.04% spread that
    rules out warm-cache and clock drift — which a single before/after pair
    never can. The same override is what makes the heuristic's 128-work-group
    target rejectable later instead of load-bearing folklore.

48. **Split-K is not bit-identical, so it needs a different gate.** Partitioning
    K reassociates the fp32 accumulation, so byte equality is the wrong
    instrument — `gemm_check`'s cosine/max-rel is the right one (PASS at cos
    1.00000000 for forced splits 1/2/4/8/16), and `batch_check` confirmed the
    eps-level drift shifts no argmax (48/48 both slots, split-K on and off).
    Note what that does NOT license: the max_rel spread across split counts
    (0.0065/0.0055/0.0040/0.0041/0.0037) is non-monotonic and measured against
    the GEMV path rather than a high-precision oracle, so it establishes
    tolerance compliance only — not that more splits are numerically better.

49. **The fused residual add is a TP trap, and §10.4 does not mention it.**
    Found by reading `run_mlp_from_resid` while scoping the forward split, not
    by running anything. The MLP tail ends in
    `x_gemv_qmma_add(&l->mlp_down, d_mlp_hidden, d_resid, d_layer_out)` — the
    residual add is **fused into the GEMV epilogue**. `mlp_down` is the
    row-parallel weight, so under TP each rank produces a PARTIAL `[H]` vector,
    and a fused add would apply the residual **once per rank** — i.e. twice at
    TP=2, silently, with no shape error to catch it. The same trap sits on
    `o_proj` in the attention layer, which is the other row-parallel weight and
    the other all-reduce point. Rule for the forward split: at every
    all-reduce point call the NON-add GEMV variant
    (`x_gemv_qmma_prepared`), all-reduce the partials, and add the residual
    exactly once afterwards. Corollary for the parity gate: this bug would show
    up as roughly-double-magnitude activations, so a cosine check would still
    look healthy — cosine is scale-invariant. The eps-band gate must assert
    max_rel too, not just cos.

    Shape of the TP MLP tail, for whoever writes it:
    norm replicated -> `mlp_gate`/`mlp_up` column-sharded to `I/2` (the
    existing `x_gemv_w4a8_fanout` works unchanged on the shard) ->
    `x_silu_mul` elementwise over `I/2`, no comm -> `mlp_down` row-sharded to
    `K=I/2` producing a partial `[H]` -> `tq_allreduce` -> add residual once.

50. **Reading vLLM's and SGLang's pool found three defects in the block pool I
    had already written and gated.** The substrate in `decode.cpp`
    (`tq_pool_init/alloc/unref/ensure/release/retain_prefix`, gated model-free
    by `qwn_pool_selftest`, 20/20 in 0.94 s) is correct as an allocator and
    WRONG as a prefix cache. Sources: `vendor/benchmarks-vllm/vllm/vllm/v1/
    core/block_pool.py`, `vendor/sglang/python/sglang/srt/mem_cache/`.
    a. **The free list must be eviction-ordered, not LIFO.** vLLM: *"the
       free_block_queue stores the free blocks in eviction order to enable
       allocation, free, and cache EVICTION."* My `h_free` is a plain stack, so
       a block held by a cached prefix can never be reclaimed under pressure -
       that is a leak wearing a cache's clothes. A cached-but-unreferenced
       block has to stay allocatable, reclaimed oldest-first.
    b. **A content-hash -> block map is the missing half of sharing.**
       `cached_block_hash_to_block` is how a NEW request discovers that another
       sequence already holds its prefix. `tq_pool_retain_prefix` pins blocks
       but nothing can look them up, so today it enables no sharing at all -
       only self-retention. Refcounts are necessary and not sufficient.
    c. **Only FULL blocks are cacheable** (`cache_full_blocks`): a partial
       trailing block's contents are not final, so publishing its hash would
       let a later request match a prefix whose tail is still being written.
    Rule extracted: a paged KV pool is TWO data structures, an eviction-ordered
    allocator and a content-addressed index. Building the allocator first and
    calling it "the pool" understates the remaining work - which is why the
    14-item plan splits PoolStorage / Addressing / Scheduler / Gates rather
    than treating paging as one task.

51. **The block-table indirection is cache-resident, and my first two attempts
    to measure it were both measuring my own benchmark.** `qwn_pool_scatter_bench`
    runs the same scatter twice over the same bytes, once contiguous and once
    through the block table, on a real pooled slab (card 1, page 128/256):

    | npos | bytes | flat | paged | delta |
    |--:|--:|--:|--:|--:|
    | 4096 | 4.2 MB | 0.029 ms | 0.046 ms | +60% |
    | 16384 | 16.8 MB | 0.303 ms | 0.296 ms | -2% (noise) |

    Two methodology traps, both caught by the SHAPE of the result rather than
    by inspection:
    a. First version's flat arm was a bare `dst[i]=src[i]` while the paged arm
       computed `t = i / rowb` - a runtime 64-bit division. That compared
       memcpy against memcpy-plus-index-math. Tell: the overhead was IDENTICAL
       at page=1 and page=256, which is impossible if it were lookup cost.
       Fixing the baseline to do the same decomposition barely moved it.
    b. The real artifact: because the free ring hands out blocks in order,
       `phys == logical`, so `phys*page + (t & mask) == t` and BOTH arms write
       identical addresses. The only remaining difference is one load from the
       block table **per work-item, i.e. per BYTE** - a shape no real kernel
       has. The CUDA TU hoists it once per 128-key tile into registers/SLM
       (`pr_chunk[j]`). So +60% is a pathological upper bound, not a forecast.
    What DOES generalise: at 16.8 MB the indirection vanishes even at one
    lookup per byte, because the table is a few hundred ints and stays hot in
    cache. The cost of paging is therefore LATENCY EXPOSURE PER LOOKUP, not
    lookup throughput - which is exactly why `page >= 128` plus per-tile
    hoisting is the right shape, and why my earlier claim that paging must cost
    us our depth advantage was wrong for a second, now-measured reason.
    Rule: when a measured overhead is invariant to the knob that should control
    it, the benchmark is measuring itself.

TP=1 re-verified after every step of this work: gemm cos 1.00000000,
prefill 32/32, parity 10/10 + argmax 64/64.



## 8. Reference index (external evidence, by claim)

- DPAS lowering + 2D-block gating rules: `vendor/intel-xpu-backend-for-triton/`
  (`third_party/intel/lib/TritonIntelGPUTransforms/{MaterializeBlockPointer,BlockIOUtils}.cpp`,
  `DotOpToLLVM/DPAS.cpp`).
- Xe2 copy/MMA atom contracts: `vendor/intel-xpu/sycl-tla/include/cute/arch/{copy_xe_2d,mma_xe}.hpp`;
  FMHA w/ paged KV under `applications/flash_attention_v2/`.
- LSC legality + GRF selection + spill thresholds: `vendor/intel-graphics-compiler/`
  (`visa/`, `IGC/Compiler/Optimizer/OpenCLPasses/Subgroup2DBlockIoResolution/`).
- SYCL runtime truths: BMG defaults to L0 v2 adapter; in-order no-event queue is
  the fast path; SYCL-Graph host-syncs per replay; joint_matrix on BMG has no INT4;
  ESIMD exposes s4 DPAS + transformed 2D loads: `vendor/llvm/` (sycl/, unified-runtime/).
- NEO driver: immediate list -> CCS direct-submission ring; zex inventory; device
  USM = host-inaccessible KMD GEM (issue #981 mechanics), WaitOnMemory snapshot
  fallback (#982): `vendor/compute-runtime/`.
- Vendor serving stacks + B70 field data: `vendor/llm-scaler/` (patches carry the
  real kernels), `vendor/intel-xpu/vllm-xpu/{vllm-xpu-kernels,sgl-kernel-xpu,b70_ai_things}/`,
  `vendor/intel-xpu/intel-arc-pro-b70-inference-cookbook/`.
- Distilled Xe2 tuning folklore (verify before trusting): `vendor/intel-xpu/vllm-xpu/Xe-Forge/knowledge_base/`.
- Attention algorithm math (portable parts only): `vendor/flash-attention/`
  (softmax.h guards, split-KV combine, PackGQA, paged pointer sharing).
- Profiling: unitrace / xpu-smi workflows in `vendor/intel-xpu/gpu-ai-skills/`;
  PerfSpect is not usable on this AMD host (Intel-CPU-focused) — noted so nobody
  retries it.

## 9. Integration designs (funded, awaiting build)

Written 2026-09-04 after the probe campaign priced every mechanism. Each
design cites the probes that funded it and carries its own build kill
criteria. Build order: 9.2 -> 9.3 -> 9.1 -> 9.4 (wave changes the step shape,
so record-replay lands after the wave; peripherals land last on freed cores).

### 9.1 Replay integration with device-resident step state

Funded by: item 2b (replay 15-19x host-CPU cut), item 2 (doorbell pricing),
lessons 17-20.

- **Record per shape class, not per step.** One regular L0 command list per
  step shape (dense decode; wave decode; shallow vs deep attention gate at
  `pos >= 512` = two lists, host picks). Recording runs the live step once
  through an interop path that appends the same ~350 kernels into the list.
- **Device-resident step state block** (device USM, one cacheline-ish
  struct): `{ token_id, pos, kv_ring_offsets[], rope_base, argmax_slot }`.
  Every hot kernel reads its varying values from this block instead of
  kernel args - recorded args become fully static (lesson 20). A ~1 us
  end-of-step advance kernel commits the argmax token id, increments `pos`,
  and advances KV offsets.
- **Host loop per step**: optional zex doorbell wait + one
  `zeCommandQueueExecuteCommandLists` + argmax readback (5.4-6.5 us class,
  measured). Window = 1 pre-queued execution, always (parked-execution wedge,
  probe 2b sharp edge). One-shot host doorbell writes to host-USM are
  PROHIBITED on this stack (RMW-writeback, lesson 17): zex or hammer only.
- **Expected**: 454 -> ~28 us host CPU/step (measured on the stand-in
  stream); wall recovery ~0.5-1.5 ms of the 5.8 ms/step gap (+2-5%
  single-stream); the real prize is batched p99 (CUDA twin's 400-1500 ms
  tails are scheduling jitter) and ~2 freed host cores for §9.4.
- **Build kills**: replayed full step >= live step wall; or the state-block
  indirection costs > 2% on the GEMV loop; or any hot kernel cannot express
  its variability through the state block without a grid change.

### 9.2 Chunk-parallel DeltaNet scan (WY-UT port from the CUDA twin)

Funded by: prefill is the one compute-bound phase; token-serial GDN is ~30%
of prefill; the math already exists in our CUDA TU (chunk-64 WY/UT) and
vllm-xpu-kernels' Xe2 GDN kernel is the tiling reference (recon queue 14 -
read both BEFORE building, learning cadence).

- **Chunking**: T split into 64-token chunks. Intra-chunk: build the WY
  representation (triangular solve / UT transform) with bf16 DPAS
  (183 TOPS tier) and fp32 accumulation - the chunk math is dense GEMM-
  shaped, exactly what the systolic array wants. Inter-chunk: serial state
  carry, 48 layers x T/64 carries, each small (~3.1 MB state/layer).
- **Wave interaction**: within a spec wave (9.3) GDN rows stay serial - an
  8-iteration in-kernel loop over wave rows; chunk parallelism applies to
  prefill only.
- **Expected**: kills most of the 30% token-serial prefill share; flat 468
  tok/s prefill toward ~600; TTFT at 8k from ~17 s toward ~13.5 s, before
  W4A4 multiplies the projection share (funded 1.78x, item 4).
- **Build kills**: chunk kernel slower than the token-serial path at T=512;
  or chunk-vs-loop agreement below the prefill_check 0.90 gate.

### 9.3 n-gram archive + spec wave (port from the CUDA twin)

Funded by: item 7 (wave = 1.11-1.17x one GEMV for 8 rows, break-even
0.11-0.17 accepted drafts, prefix byte-exactness proven on silicon, lesson
24); CUDA twin measured 2.03x at 8k x 1 with the same drafting scheme.

- **Drafting (host, §9.4 cores)**: n-gram archive over prompt + generated
  stream (n = 2..4, hash-map to next-token candidates), maintained
  asynchronously off the critical path. Per step the host proposes up to 7
  chain drafts; misses cost one wave row riding free.
- **Wave execution**: rows = [committed, d1..d7] at sequential positions.
  Projections via the existing RC8 GEMM (measured). Full-attn layers:
  decode attention extended to 8 query rows with per-row position
  (staircase: row i attends <= pos+i). GDN: in-kernel serial loop over the
  8 rows (state math is tiny per token).
- **Verify**: wave logits row i argmax must equal draft token i+1;
  first mismatch truncates - the accepted prefix is byte-exact by the
  qwn_specwave_check contract, so acceptance is exact, not eps-band.
- **Expected**: acceptance A per wave multiplies decode by (1+A) at
  11-17% wave overhead; CUDA's repetitive-load figure suggests 1.5-2x on
  chat/code sessions. Zero risk to plain decode: wave of 1 = today's step.
- **Build kills**: measured acceptance on a real chat corpus below 0.2
  drafts/wave sustained; or staircase attention costs push wave overhead
  past ~1.5x GEMV; or any TF-gate regression on the committed stream.

### 9.4 CPU-peripheral pass

Funded by: item 12 (swaps 6.8 GB/s, 0% contention, ~400x vs re-prefill,
lesson 25) + host cores freed by 9.1 (~450 us/step recovered).

- Host cores own: n-gram archive maintenance (9.3), sampling
  post-processing beyond argmax (top-k/temperature tail), KV
  parking/checkpoint spillover scheduling (134 MB prefix = 21.7 ms restore,
  overlapped under decode at zero cost), and the serving shell.
- Policy: park KV beyond the hot window to host DRAM for 32k-256k
  sessions; APC checkpoints spill to host by default; restores are
  prefetched one turn ahead (doorbell/replay loop never blocks on PCIe).
- No device probe needed; lands with 9.1 + 9.3. Never plan around burst
  PCIe numbers from other silicon (lesson 25).

### 9.5 DPAS-matrix decode attention at GRF128 (queue 3b, implementation notes)

Pre-read done 2026-09-04 (sycl-tla FMHA mainloop + our on-silicon DPAS
contracts). Funded by lesson 22's kill of the scalar G=6 form: the traffic
halving must come from matrix formulation, not register blocking.

- **Shape**: one work-group per (kv_head, token-range); the 6-head GQA group
  packs as DPAS M-rows (M=8: 6 live + 2 dead rows, PackGQA-as-M). Per
  16-token tile: S[8,16] = Q[8,256] x K_tile[256,16] as 16 chained
  `dpas.bf.bf.8.8` (K=16 each, fp32 accumulator); online softmax over the
  token axis; O += P[8,16] x V_tile[16,256] as bf16 DPAS with V split into
  VTiles to cap register pressure (FMHA's FragO trick). All fragments are
  8-row tiles, NOT per-lane blocks - GRF128 occupancy is preserved (the
  whole point vs lesson 22).
- **Operand builds**: Q fp32 -> bf16 A-fragments once per step (row-major
  8x16 halves = 32B/row, lane pair mapping as the s8 A contract). K/V are
  E4M3 + row scales -> decode to bf16 tiles staged in SLM, then per-lane
  vector loads assemble B fragments (bf16 VNNI: dword (k/2)*16+n holds the
  (k, k+1) pair). Steal the FMHA f32->bf16x2 vISA idiom (add 0x8000 bias +
  strided mov) for P and the staging paths; fold log2(e) into the softmax
  scale and use exp2 (FMHA line 297 trick).
- **Sizing** (hd=256, nkv=4, nh=24): per 16-token tile the group reads K+V
  once (16x256x2 e4m3 bytes) and serves 6 heads - vs the shard kernel's
  per-head-triple reads. Expected: the G=6 traffic win (measured 4x fewer
  row passes) at 8 threads/XVE.
- **Probe first**: extend the standalone probe family with a bf16 QK^T
  microkernel vs the scalar walk on a synthetic 32 MB cache (int4kv probe
  harness reusable; score cos gate >= 0.9997 - the E4M3 band, since math is
  exact bf16 over the same decoded values).
- **Build kills**: slower than the G=3 scalar kernel at pos 8192; spills at
  GRF128 (AOT-check per lesson 22c); or score band regression vs the
  E4M3-tier reference.
- **Order**: probe -> engine kernel behind TQ_XPU_ATTN_DPAS -> bench_depth
  cells at 2k/4k/8k/16k -> parity + TF gates -> flip default if it wins
  everywhere.



---

*Update protocol: when a queue item completes, move its numbers into §5's status
column and, if killed, write the lesson in §7. When a thesis ships, add the measured
deltas to §2 and record the date. Never delete; strike through only when a statement
is superseded, with a pointer to what superseded it.*
