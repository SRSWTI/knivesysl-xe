# Shared paged KV: implementation and rollout plan

Date: 2026-09-06. **Historical rollout plan; the paging implementation is now integrated.**

Current status (2026-09-07): native pool ownership, paged attention, reservation
admission, hybrid APC, and serving integration are implemented. XMX attention
and packed prompt execution are documented in the
[current changelog](../CHANGELOG.md#xmx-prefill-and-packed-prompt-execution-2026-09-07).
The numbered rollout steps and source observations below retain the original
pre-implementation snapshot; they are not a current outstanding-work list.
Previously stopped/unrun long-context warm measurements remain unmeasured;
the new prefill work does not claim to have completed those benchmark cells.

This document replaces the missing actionable plan referenced in
[level-up-xpu.md](level-up-xpu.md), lessons 50–51. Those entries remain historical
records, not descriptions of the current implementation. Source locations below
were inspected for this plan; relocate by symbol when line numbers move.

## 1. Deliverable and capacity contract

Finish shared-pool continuous batching for the XPU engine: independent sequences
use blocks from one full-attention KV pool instead of each reserving `CTX` rows.
Keep per-slot DeltaNet recurrent/convolution state. Preserve automatic prefix
reuse, chunked prefill, greedy generation, tool calls, reasoning separation, and
the OpenAI HTTP surface already implemented.

Three independent limits must be visible and enforced:

- `CTX`: maximum prompt plus requested generation for **one** sequence.
- `SLOTS`: simultaneous sequence-state slots; current engine limit remains 8.
- Pool capacity: aggregate **physical** KV blocks, shared by active sequences and
  cached checkpoints. It is not `SLOTS * CTX` and does not multiply automatically.

At page size 128, using decimal prompt lengths and ignoring generation for this
illustration:

| Requests | Rounded prompt footprint | Pool = 131,072 tokens | Required behavior |
|---|---:|---|---|
| 45,000 + 45,000 | 90,112 tokens / 704 blocks | fits | Both can be admitted, subject to output reservations and free slots. |
| 28,000 + 75,000 | 103,040 tokens / 805 blocks | fits | Same shared pool; no uniform per-slot 32k ceiling. |
| 65,000 + 75,000, unrelated | 140,032 tokens / 1,094 blocks | does not fit | Both requests can complete, but queue one; do not promise simultaneous residency. |
| 65,000 + 75,000, unrelated, pool = 196,608 | 140,032 tokens / 1,094 blocks | larger pool | Concurrent admission if generation reservations also fit and allocation succeeds. |

Full shared-prefix blocks can reduce physical occupancy. Reuse is allowed only
with a matching hybrid checkpoint, not merely equal KV block hashes. Round each
request's **prompt-plus-output** bound to pages before admission. A request that
cannot fit an otherwise empty pool must fail before generation, rather than wait
forever. Prompt and output limits must not be silently truncated to make a case fit.

Concurrent admission is not proof of overlapping batched decode: a short prompt
may finish generation before a much longer prompt finishes prefill. Measure the
actual overlapping decode steps separately; never hold completed output merely
to manufacture a batch-width result.

Non-goals: TP2, a new weight/KV format, speculative-serving rollout, host offload
of pooled KV, a new attention algorithm, or a vLLM/SGlang performance claim. No
CUDA implementation changes. Do not couple these to completion of this port.

## 2. Protect what already works

The user explicitly asked whether this could break existing work. The answer is
not an unconditional guarantee: implementation bugs and shared-host driver faults
remain possible. These are mandatory containment rules:

1. **Writing this plan changes documentation only.** Do not build, run GPU probes,
   restart a service, change launch defaults, or execute the commands below while
   producing the document.
2. Add a real `TQ_XPU_PAGED` flag, default **0**. It does not exist today. A flag
   cannot make broken code compile and cannot substitute for flat-path regression
   checks. Unset/0 must select the existing flat implementation and allocations.
3. Build a distinct candidate library, e.g.
   `xpu/build/libforward_qwen_xpu_paged_candidate.so`. **Never overwrite the `.so`
   mapped by a live server.** The current `xpu/build.sh` writes the shipping path;
   do not use it against a serving process during this work.
4. Leave the working service's PID, port, card, library, and launch configuration
   alone. Before any eventual deployment, record its actual argv and only the
   relevant configuration variables; historical port/card claims are not live
   status checks. Do not use broad `pkill -f serve_openai_xpu` patterns.
5. Candidate tests need a known-free card, `ZE_AFFINITY_MASK` set before process
   startup, `TQ_XPU_DEV=0` within that mask, and a separate loopback port. Never
   load a second model onto an occupied card. Do not assume the matrix launcher
   already supplies these controls: its current startup lacks the mask.
6. A second GPU still shares the host. Given the recorded host hangs, run GPU
   tests only in an approved test window; a separate port/card does **not** prove
   machine-level isolation. `gpu_health.sh` itself launches GPU liveness kernels;
   it is not a harmless host-only inspection. Do not invoke it while GPU work is
   paused, and do not reset devices or alter display/driver settings on a guess.
7. Exercise the candidate in **both** flat and paged modes before cutover. Start
   the production candidate only in a deliberate maintenance window after gates
   pass. Retain the old library and exact old launch configuration. Rollback means
   stopping only the candidate and restarting the known-good flat configuration;
   it does not preserve in-flight requests or APC state across processes.
8. No default-on paging change is implied by landing code. Never claim the live
   server uses paging until its own effective-layout report and end-to-end tests
   demonstrate it. Source edits are not a deployment.

## 3. Current source: what exists and what does not

| File / symbol | Inspected locations | Current behavior / required connection |
|---|---|---|
| [decode.cpp](../src/decode.cpp), slot accessors | 248–283, 574–579 | Flat `slot * max_seq` KV offsets; GDN offsets are separately per-slot. |
| `tq_pool_bytes_per_block`, `tq_pool_init`, `tq_pool_free_all` | 307–423 | Per-full-attention-layer K/V/scale pool slabs and host/device block tables. No production activation. |
| `tq_pool_alloc_block`, `tq_pool_ensure`, `tq_pool_release_slot`, `tq_pool_retain_prefix`, `tq_pool_sync` | 428–500 | FIFO free ring/refcounts; growth can partially succeed before `-2`; retain uses inclusive `[0,pos]`; sync copies the whole table. |
| `tq_pool_lookup/publish/unpublish` | 502–573 | Experimental hash index; no production discovery/adoption caller. |
| `run_full_layer_decode` | 582–615 | Single-token attention still receives flat active-slot bases. |
| `pf_full_layer`, `qwn_prefill_chunk` | 852–889, 994–1053 | Chunk write/read use flat bases; reset-at-zero precedes layer execution. |
| `qwn_decode_batch` | 1501–1620 | RC8 projection wave; attention invoked separately for each real slot/position. |
| `ckpt_walk`, `qwn_ckpt_*` | 1623–1746 | Full prefix KV plus GDN copied to/from host buffers. |
| `run_decode_layers`, `qwn_decode` | 1748–1843 | Reset at position zero; ordinary decode does not commit `state_pos` today. |
| [tqf_loader.cpp](../src/tqf_loader.cpp), `qwn_free/init/reset_state` | 660–726, 730–921, 924–1024 | Model scratch at init; state lazily allocated as `slots * max_seq` at reset; pool globals not in model teardown. |
| `qwn_reset_slot` | 1324–1371 | Cold detection relies on flat K pointers; otherwise zeros one entire flat KV region and GDN state. |
| [tq_common.hpp](../src/tq_common.hpp) | 85–185, 268–302, 378–421 | Existing structs and attention/public ABI; no paged descriptor or pool exports declared. |
| [serve_openai_xpu.py](../tools/serve_openai_xpu.py), `Engine` | 45–109, 184–388 | Free-slot-only admission; Python hash APC lookup; host checkpoint allocation; retirement does not release native pool blocks. |
| [serve_xe.sh](../tools/serve_xe.sh) | 14–40, 62–65 | Defaults currently four slots, `CTX=32768`, MLP K64; loopback and device mask. No paging options. |
| [paged_parity.py](../tools/paged_parity.py) | 33–56, 72–126 | Same-process double reset/run for recording; `--check` does not activate or attest paging. |

All existing `tq_pool_*` production connections are missing: its callers are the
four exports `qwn_pool_selftest`, `qwn_pool_scatter_gate`,
`qwn_pool_scatter_bench`, and `qwn_pool_gate_model`, plus internal helpers. Their
historical successes do not certify attention, serving, or APC integration.

Corrections to earlier claims:

- This is not “one function, four small edits.” `x_full_attn_decode` spans
  approximately 1,000 lines and several distinct read paths; prefill has separate
  write/read functions. The eight address groups are enumerated in section 6.
- The pool uses E4M3 **K and V**, with FP16 row scales. Do not copy CUDA's Q4-K
  element strides into XPU. Weight K64 selection does not change KV encoding.
- Paging does not inherently give 2.18x capacity for equal bytes. Prior comparisons
  gave a 9-GiB pool more bytes than four flat 32k slots. Its benefit here is flexible
  allocation and prefix sharing, minus page-tail waste, not compression.
- The existing 405-token prompt plus 64 decode steps never reaches position 512.
  It cannot gate grouped or DPAS attention. Also, a paged `--check` can currently
  report success while running flat, because it never checks the effective mode.
- FIFO is not universally required for correctness. CUDA's explicit checkpoint
  ownership uses a stack. Cache lifetime and eviction policy must match the
  chosen design; refcounts alone do not discover prefixes or make tails immutable.

## 4. Implementation decisions and shared contracts

### 4.1 Storage and configuration

Retain the existing E4M3 representation. Let `P` be page tokens, `B` physical
blocks, `Hkv` KV heads, `D` head dimension, and `F` full-attention layers:

```text
per_layer_K_bytes = B * P * Hkv * D
per_layer_V_bytes = B * P * Hkv * D
per_layer_K_scale_bytes = B * P * Hkv * sizeof(uint16_t)
per_layer_V_scale_bytes = B * P * Hkv * sizeof(uint16_t)
bytes_per_block = F * P * Hkv * (2*D + 2*sizeof(uint16_t))
```

For the inspected target `F=16,Hkv=4,D=256`: 33,024 bytes/token;
4,227,072 bytes/page-128 (4.03125 MiB); a 131,072-token pool is 4.03125 GiB of
KV alone. A 196,608-token pool is 6.046875 GiB. Add actual resident weights,
GDN state for every slot, projection/attention scratch, table metadata, and a
measured safety margin. Do not infer available VRAM from the TQF file size.

Implement these **new**, explicitly proposed controls:

| Native environment | Server option | Meaning |
|---|---|---|
| `TQ_XPU_PAGED=0/1` | `--paged` | Default off; effective mode must be queryable. |
| `TQ_XPU_KV_PAGE=128` | `--kv-page` | Production supports 128 and 256 initially; reject other values explicitly. |
| `TQ_XPU_KV_POOL_TOKENS` | `--kv-pool-tokens` | Physical capacity target, rounded down to full pages. |
| `TQ_XPU_KV_POOL_MB` | `--kv-pool-mb` | Alternative decimal-MB **KV-only** byte budget; mutually exclusive with token target. |

Require an explicit pool budget when paging is enabled. Keep existing `TQ_CTX`
and `TQ_XPU_SLOTS` meanings, but `CTX` only sizes the logical table/scratch, not a
per-slot KV reservation. Parse/check arithmetic before allocating. Reject TP>1
with paged mode until rank-scoped ownership exists. Unsupported configuration or
allocation failure must fail startup cleanly, never silently fall back to flat.

Read configuration once in init, after model shape is available and before state
allocation. Keep pool ownership in `decode.cpp`; add narrow lifecycle declarations
in the header for the loader rather than creating another allocator subsystem.
Initialize pool and GDN state before reporting candidate readiness; warm required
scratch on the candidate so the first admitted request cannot trigger an unbudgeted
surprise allocation. Only one KV representation is allocated per process.

### 4.2 Kernel addressing contract

Add an internal trivially-copyable descriptor in `tq_common.hpp`:

```cpp
struct tq_kv_layout_t {
    const int *block_table;  // device pointer to ONE slot's logical-block row
    int page_log;
    int page_mask;
};
```

Append it by value to `x_full_attn_decode`, `x_prefill_kv_write`, and
`x_prefill_attn`, updating every caller. Existing K/V/scales pointer parameters
remain: in flat mode they point to that slot's flat bases; in paged mode they
point to the layer's global pool bases. The descriptor's table is already offset
by `slot * max_blocks`; kernels must not apply the slot offset again.

Use internal `template<bool Paged>` implementations with host-side dispatch so
flat instantiations eliminate address translation. Do not clone whole kernels or
change reduction/quantization order. The descriptor is internal C++; no ctypes
struct matching is needed for it.

```text
physical_token(t) = t                                      [flat]
physical_token(t) = (size_t(table[t >> page_log]) << page_log)
                    + (t & page_mask)                      [paged]
scale_row(t,kv) = physical_token(t) * Hkv + kv
byte_offset(t,kv,d) = scale_row(t,kv) * D + d
```

Translate only cache addresses. Query positions, RoPE positions, causal masks,
segment selection, token iteration order, and scale arithmetic remain logical.
Use one translated row for **both** K/V bytes and their scales. Preserve alignment
of the current dword loads. Validate mappings before kernels; never map invalid
entries to block zero as a fake fallback.

### 4.3 Native ownership, position, and reservation contract

Public positions are **committed token counts** for checkpoint/reservation APIs:
`committed=n` means rows `[0,n)` are valid and next append is at position `n`.
Kernel `pos` remains a zero-based token index. Audit the existing inclusive
`tq_pool_retain_prefix(slot,pos)` carefully; do not expose that ambiguity as a new
public contract.

Proposed C ABI (all new declarations must be bound with explicit ctypes argtypes
and restype; checked status values must not collide with existing generic errors):

```cpp
int qwn_paged_enabled(void);                         // 0 or 1 after init
int qwn_paged_reserve(int slot, int total_tokens);    // 0 success; named capacity error is retryable
int qwn_paged_release(int slot);                     // release slot maps/reservation; idempotent
int qwn_paged_stats(long long *out, int count);      // 0 success; require count >= 8
int qwn_paged_ckpt_save(int slot, int committed);    // >=0 opaque id; <0 no valid checkpoint
int qwn_paged_ckpt_adopt(int slot, int id);           // committed count or <0, target initially empty
int qwn_paged_ckpt_free(int id);                      // 0 success; stale id rejected
size_t qwn_paged_ckpt_host_bytes(int id);             // actual state-image charge, not KV capacity
```

`qwn_paged_stats` output order: page tokens, total blocks, free blocks, blocks with
active-slot references, blocks with checkpoint references, remaining reserved
blocks, configured per-sequence limit, slots. Active/checkpoint counts can overlap;
do not add them to compute physical occupancy. `free` means refcount zero;
`remaining_reserved` means future blocks promised but not yet mapped.

Keep reservations native and authoritative. For slot `s`, retain a total block
bound `ceil(total_tokens/P)`; after adoption, shared full blocks already mapped to
`s` satisfy part of that bound. Conservation rule:

```text
sum(remaining_reserved_blocks[s]) <= free_physical_blocks
```

Growing a reserved slot consumes one free block and one reserved credit together.
APC tail copies/new allocations may consume only **unreserved** free blocks. They
must not steal a block promised to an active request. Unrelated direct C callers
without a reservation may grow only from unreserved capacity and get a named
capacity error before any layer state mutates.

Preflight **all** rows of a batch and a whole prefill range before submitting any
embedding/GDN/attention work. Ensure failures are transactional: roll back just
that operation's new maps/refs and leave committed prefixes and credits intact.
Upload only dirty logical-table ranges on the same in-order queue before their
consumer kernels; do not add a per-layer full-table copy or host wait.

Reset-at-zero must precede allocation, but must not erase a scheduler reservation.
Separate slot state initialization from end-of-request release. On successful
full forward commit, update `state_pos`; fix ordinary `qwn_decode` as well as
prefill/batch. Debug partial-layer runs must not claim full-model commitment.
Never zero pooled bytes belonging to another slot/checkpoint during reset.
Synchronize outstanding consumers before destroying storage or reusing ownership
where queue ordering cannot provide the guarantee. A partial model-execution
failure is not retryable capacity pressure: retire/quarantine the affected request;
do not replay a mutated GDN state.

### 4.4 APC decision: automatic discovery, explicit native ownership

Reuse the existing server's **automatic** prefix discovery/second-sighting
admission and LRU. Back each entry with a native checkpoint id, following CUDA's
checkpoint lifetime model. Clients need not name checkpoints. This preserves
implicit user-visible sharing without introducing a second request-matching layer.

Change Python cache keys from a bare `hash(tuple(tokens))` to exact token tuples
(or a bucket plus exact token comparison); Python's dictionary resolves tuple hash
collisions. Require a matching prefix identity, model/configuration epoch, and
GDN state at precisely that token boundary. A matching KV block alone cannot
restore a hybrid sequence. Existing matching has no collision verification.

Checkpoint contents:

- References to `floor(committed/P)` immutable full blocks across all attention
  layers (one physical block id names the same token page in each layer pool).
- If `committed % P != 0`, an independently allocated **private tail** with only
  those valid rows copied, including K/V scales. Never share a writable tail.
- A complete GDN conv/recurrent snapshot. Retain the existing host-USM mechanism
  for this state image, bounded by the existing host APC budget; do not claim
  that GDN restoration is zero-copy.
- Committed count, valid epoch/generation, ownership of state/tail allocations.

Save waits for committed device work, completes all allocations/copies, then
publishes the entry atomically. Adopt requires an empty target, obtains full-block
refs, allocates/copies a **new** target tail, restores GDN, and commits the table
and position only on success. The checkpoint's own tail remains immutable. Free
releases every reference and host allocation once; stale ids must not adopt a
reused registry entry. Pool teardown invalidates all ids.

Never restore a full-prompt checkpoint without saved next logits: `_prefill_one`
needs at least one unprocessed token. Select a checkpoint strictly shorter than
the prompt and prefill the remaining suffix. Do not replay the last token on an
already-advanced GDN state. Continue to degrade an unavailable optional cache
entry to full prefill, provided a cold reservation can actually be obtained.

The existing native hash index is **not** the production APC authority in this
plan. Do not use lookup→manual refcount increments for adoption: a zero-ref block
may still be in the free ring. Its current deletion marks a linear-probe entry
empty without repairing the cluster, and publication does not enforce fullness.
Before any independent block-index reuse is exposed, it would need collision-chain
repair, exact identity verification, atomic free-ring acquisition, and full-block
publication enforcement. That extra cache architecture is not required here;
retain it as clearly experimental gate-only code, not a claimed serving feature.

Reference: [CUDA implementation](../../src/forward_qwen.cu),
`qwn_paged_init` at 26378, `qwn_paged_ckpt_save/adopt/free` at 26650–26723;
[hybrid APC contract](../../README.md#apc--checkpointed-prefix-reuse). Port ownership
and ordering principles, **not** CUDA Q4 strides or all host-spill machinery.

## 5. Execution order and completion criteria

Do not stop after allocator gates and label the feature delivered. The original
14 items are preserved below. Previously tested substrate is a starting point,
not grounds to skip requalification after lifecycle changes. Gates here are
correctness prerequisites, not a request for unrelated test cleanup in advance.

| # | Original work item | Concrete completion condition |
|---|---|---|
| 1 | Read vLLM and SGLang pool structure | Prior research exists; this plan selects automatic Python matching plus CUDA-style native checkpoint ownership. No new broad recon before coding. |
| 2 | Allocate per-layer pooled KV slabs | Production paged mode allocates full-attention pools instead of flat KV, with GDN per-slot and teardown wired. |
| 3 | Size pool from memory budget | Checked token/byte budget, exact effective capacity reported, no flat+pool double allocation or invented VRAM headroom. |
| 4 | Gate pool alloc and free | Repeated init/reset/free, partial allocation failure, multi-slot ownership, budget/credit conservation pass. |
| 5 | Port KV write to block table | Chunk write and both decode writers place K, V, and both scales in the mapped rows. |
| 6 | Port scalar decode attention | Grouped, sharded, short, and generic paths read mapped rows with unchanged arithmetic. |
| 7 | Port DPAS decode attention | Existing opt-in DPAS reads mapped K/V/scales, preserving lane masks and DPAS arithmetic. |
| 8 | Port chunked prefill attention | Paged staging and causal/partial-tile bounds work for arbitrary supported chunk start and size. |
| 9 | Wire ensure release into engine loop | Single decode, prefill, batch, reset, retirement and cancellation use the same ownership lifecycle. |
| 10 | Admission on pool exhaustion | Whole prompt/output reserved; temporary shortage queues safely; impossible jobs reject; active jobs retain progress guarantees. |
| 11 | Refcount prefix sharing replaces copy APC | Exact automatic match + native full-block sharing, private tails, matching GDN state and pressure eviction verified. |
| 12 | Byte equality paged vs flat single sequence | Effective layout attested; matched arithmetic branches compare tokens and cache bytes, including fragmented physical maps. |
| 13 | Mixed context 28k plus 75k concurrent | Actual exact-token HTTP scenario completes; occupancy and overlap reported, no per-slot fixed-size limitation. |
| 14 | Measure aggregate at mixed lengths | Same-session matched settings; cold and APC results separate; capacity/queueing and TTFT/ITL/throughput reported. |

### Step A — first functioning native slice (items 2–6, part of 9/12)

1. Before changing exported symbols, use LSP references where available. Update
   header declarations and every callsite together; use scoped source lookup only
   if no server is available. Do not leave compatibility aliases for the old
   internal attention signatures.
2. Add config, effective-mode query, layout descriptor, and loader/pool lifecycle.
   `qwn_reset_state` and `qwn_reset_slot` must not treat a null flat K pointer in
   paged mode as evidence that the whole engine is uninitialized. Track readiness
   explicitly. `qwn_free` releases checkpoints, maps, then pool allocations while
   its owning queue/context still exists; partial init errors unwind too.
3. Port both decode writers and **all scalar decode readers together**, plus
   single-token ensure/sync/commit/reset. First milestone is real paged token
   generation on a separate candidate, with a same-branch flat reference.
4. During this slice, paged entry points not yet connected (chunk prefill, batch,
   optional DPAS, flat host checkpoint APIs, speculative/debug paths) must reject
   unsupported use **before** launching kernels. Never route pooled pointers into
   a remaining flat reader or silently disable a requested branch.
5. Run the native scalar scenario in section 8. This proves only this slice; keep
   the public service on its known-good flat configuration and continue to B–E.

### Step B — finish every serving attention path (items 5, 7–9, 12)

Complete all sites in section 6. Wire `pf_full_layer` and per-row
`qwn_decode_batch` to the correct layer pool and **that row's slot**, not the
ambient active slot or batch-row index. Prepare maps for all real rows before the
wave; padded RC8 rows do not own cache pages. Preserve one-row GEMV dispatch.

Guard `qwn_spec_wave` and unported debug APIs explicitly in paged mode; their
speculative state/rollback semantics are outside this serving rollout. Normal
flat use remains available. No unsupported path may dereference null flat KV.

### Step C — scheduler admission and lifetime (items 9–10)

In `Engine._admit`, choose an empty slot and verified checkpoint candidate; reset
state, acquire/adopt candidate if available, then reserve the full prompt/output
bound. Failed reservations undo adoption and ownership and leave the request
pending. Evict optional cached checkpoints oldest-first if their blocks can make
room; never evict active-slot refs or consume reserved credits. After an eviction
invalidates the candidate, re-evaluate from a fresh candidate or cold start.

Use FIFO capacity waiting initially: insufficient capacity for the oldest feasible
pending request stops admission, but must **not** stop active prefill/decode.
Retire completed/cancelled/error requests, release maps/reservations after device
completion, then re-admit on the next iteration. A request larger than empty-pool
capacity is rejected synchronously; it must not block everything behind it.

Check every reset/adopt/release status. Do not return a failed-release slot to
`self.free`. Surface a terminal engine failure rather than replying `status: ok`
while the only engine thread has died. Keep native calls on the existing engine
thread; HTTP handlers must not mutate pool metadata concurrently.

`Handler.do_POST` currently rejects prompts at/above `ctx` but clamps requested
output to the remaining context. For paged requests validate the normalized full
prompt/output bound explicitly and return a clear capacity/context error instead
of silently shrinking it. Expose mode, pool capacity, used/free blocks, remaining
reservations, and capacity-wait count in `/health`; append these to existing fields.

Long prompts must survive their actual queue+prefill time: `pump()` currently
uses a hardcoded 300-second token wait. Add an explicit `--response-idle-timeout`
option (default preserves 300), use it consistently, and configure long-context
runs with a sufficiently large value and matching client timeout. Do not report
long-context support while requests time out before their first token.

### Step D — refcount APC (item 11)

Implement the section 4.4 checkpoint contract and update `_lookup`, `_store`,
`_admit`, and LRU eviction as one change. Flat mode retains its existing blob
representation; paged mode uses checkpoint ids and never calls flat `ckpt_walk`
with pooled storage. Keep host-state bytes and physical pool occupancy separately
accounted. Cache creation must be optional under pressure, and may not prevent a
reserved active request from finishing. After the first end-to-end smoke works,
extend the existing gates with APC/cancellation/pressure regressions before rollout.

### Step E — prove the user scenarios and deploy deliberately (items 12–14)

Complete section 8, then run exact-token mixed-length requests through the real
server, not only the native library. Extend the existing smoke/benchmark tools
rather than shipping a second scheduler or mock engine. Keep all results labelled
as native versus HTTP, measured versus calculated, cold versus APC. Only after
that schedule deployment and verify the effective mode on the endpoint the user
will actually use. Keep the rollback artifact and old launch configuration.

## 6. Exact kernel edit inventory

All attention-cache sites are in
[kernels_seq.cpp](../src/kernels_seq.cpp). The DPAS **attention** branch is also
here; [kernels_dpas.cpp](../src/kernels_dpas.cpp) contains projection primitives,
not KV cache accesses, and needs no paging rewrite.

| Site | Current lines / dispatch | Required address change |
|---|---|---|
| Chunk writer `x_prefill_kv_write` | 737–816; row at 801, scales 803–804, bytes 811–813 | Translate `pos0+t`, use the same row for all four stores. Projection input row stays chunk-local. |
| Chunk reader `x_prefill_attn` | 818–972; staged bytes 919–922, scales 926–931 | Translate `c0+j` for both staged bytes and scales; leave SLM attention math unchanged. |
| Common SIMD16 decode writer | 999–1066; row 1052, stores 1054–1064 | Translate `pos` once per head/subgroup before all decode branch selection. |
| DPAS decode | 1068–1344; row/base 1203–1206, K 1208–1213, V 1231–1232 | Translate the **clamped logical token** `live ? token : pos`; preserve zero-probability padded lanes. |
| Grouped scalar decode | 1346–1597; row/base 1475–1477, K 1478–1482, V 1503–1507 | Translate the subgroup's strided logical `token`; preserve `seg*16+shard`, stride and reduction order. |
| Sharded scalar per-head decode | 1599–1732; row/base 1671–1673 | Same translation for K/V/scales; preserve the `pos>=64` branch and shard order. |
| Short scalar per-head decode | 1734–1814; row/base 1780–1782 | Translate each visited logical row; do not alter small-position behavior. |
| Generic decode writer and reader | 1816–1973; write 1866–1875, read 1935–1961 | Translate both stores and reads; exercise using `TQ_XPU_ATTN_SIMD16=0`, even on hd=256. |

Important boundary constraints:

- Prefill's K staging tile is **64**, not CUDA's 128. `c0` starts at zero and is
  64-aligned. Pages 128/256 contain each such tile, so one block base plus tile
  offset suffices for a fully valid stage. Hoist/broadcast the lookup rather than
  paying it per channel byte; retain uniform barriers.
- `x_prefill_attn` currently computes staging `total=pos0+tile*16+16`. For a final
  partial query tile this can exceed `pos0+T`. Cap staged logical rows at the
  actual written end **before** table access, without changing per-live-query
  causality or work-group barrier participation. Masked query rows do not justify
  an out-of-range block-table read.
- DPAS tiles are 16-aligned and fit supported pages. Clamp invalid tail lanes
  before translation. Grouped scalar subgroups instead stride by 48 (or 96 with
  `TQ_ATTN_G=6`); do not assume their whole walk belongs to one physical page.
  A cached page base is valid only while that subgroup's logical page is unchanged.
- No new SLM barriers in lane-divergent early-return paths. Preserve the existing
  subgroup-size attributes and workgroup-uniform staging loops.
- A page remap must preserve the dword-aligned vector load bases. Translation is
  outside the dimension loops; K and V reuse it. Performance requires measurement,
  not the historical per-byte scatter benchmark.

## 7. Pool and lifecycle correctness details

The storage substrate is small enough to strengthen in place. Before exposing it:

- Check positive `max_seq`, supported page sizes, `ceil`/multiplication overflow,
  and block-table capacity before allocation; validate every native range.
- Make repeated slot release idempotent without double-pushing blocks into the
  ring. Reject/refuse refcount underflow; verify each zero-ref block appears in
  the free ring exactly once and no referenced block appears there.
- Make ensure/reserve failure atomic and preserve other slots. Test capacity
  exhaustion after some candidate allocations, not only when the pool starts full.
- Separate pool allocated/ready state from `g_qwen.initialized` and flat pointers.
  Whole-state reset drops/invalidate checkpoints and releases slot maps consistently;
  slot reset affects only its GDN state and references, not other clients.
- Enforce exclusive ownership for every written partial page. Full shared blocks
  cannot be appended into or rewritten; private tail copies are allocated before
  writes. No uninitialized/padded row may become a published checkpoint.
- Existing selftests create/free the process-global pool. Do not run destructive
  selftests inside a serving process or against a live production pool. Give their
  early-error paths cleanup so a failed gate does not contaminate later cases.

## 8. Verification commands and acceptance matrix

**Future implementation instructions, not commands executed for this plan.** The
proposed APIs/options must exist first. Run one candidate GPU process at a time
on a confirmed available device in the approved test window. All paths below are
from the repo root. Choose `TEST_CARD` from current occupancy, not this example.

### 8.1 Candidate build and immediate native smoke

This reproduces the current default JIT build flags while changing only its output
path. Keep the shipping library untouched. An AOT comparison must use matching
AOT flags for both arms; do not mix JIT/AOT configurations and call it paging cost.

```bash
source /opt/intel/oneapi/setvars.sh --force
mkdir -p xpu/build
icpx -fsycl -O2 -fPIC -std=c++17 -Ixpu/src -shared -o xpu/build/libforward_qwen_xpu_paged_candidate.so xpu/src/decode.cpp xpu/src/tqf_loader.cpp xpu/src/kernels_core.cpp xpu/src/kernels_seq.cpp xpu/src/kernels_dpas.cpp
```

First extend `paged_parity.py` so `--record` requires
`qwn_paged_enabled()==0` and `--check` requires `==1`. No export/mode mismatch may
pass. Store the actual prompt tokens, model/config identity, arithmetic branch
settings, page size for checks, steps, and reference layout. Preserve the existing
reference file; use per-configuration reference files outside the source tree.
Check deterministic reset/replay and also fresh-process replay. Use the same model,
weight tier, branch flags, build and logical tokens in each flat/paged pair.

Then run this **paired scalar gate**, after Step A:

```bash
export TEST_CARD=1
export ZE_AFFINITY_MASK="$TEST_CARD"
export TQ_XPU_DEV=0 TQ_XPU_TP=1 TQ_XPU_SLOTS=1 TQ_CTX=1024
export TQ_XPU_K64=gate,up,down TQ_XPU_W4A4=1
export TQ_XPU_ATTN_SIMD16=1 TQ_XPU_ATTN_GROUPED=1 TQ_XPU_ATTN_DPAS=0
export TQ_XPU_KV_PAGE=128 TQ_XPU_KV_POOL_TOKENS=1024
export MODEL="$HOME/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf"
TQ_XPU_PAGED=0 .venv/bin/python xpu/tools/paged_parity.py --record --steps 256 --lib xpu/build/libforward_qwen_xpu_paged_candidate.so --tqf "$MODEL" --ref /tmp/xe-paged-scalar-ref.json
TQ_XPU_PAGED=1 .venv/bin/python xpu/tools/paged_parity.py --check --steps 256 --lib xpu/build/libforward_qwen_xpu_paged_candidate.so --tqf "$MODEL" --ref /tmp/xe-paged-scalar-ref.json
```

The existing prompt length is 405. At 256 steps the run crosses position 512;
64 steps do not. Branch counters/debug reporting must confirm actual selection.
Check the source spelling of each environment variable when implementing; do not
let a misspelled selector silently test the default branch.

### 8.2 Required correctness scenarios

After the native smoke succeeds, extend existing gates with these behavioral
cases. Use real kernels and real engine/server calls, never mocks. Run only the
targeted tools/tests changed for this port; do not launch a project-wide suite.

| Scenario | Required evidence |
|---|---|
| Flat flag unset vs `0` | Candidate preserves reference tokens, chunk prefill, multi-slot outputs, APC restore and normal client streaming. |
| Each attention branch | Separate flat/paged pairs: SIMD16 short/sharded/grouped; grouped disabled; SIMD16 disabled generic; DPAS enabled on both arms. Confirm branch entered. |
| Position/page boundaries | Rows 0, 1, 63/64/65, 127/128/129, 255/256/257, 511/512/513, final legal context position; page 128 and 256. |
| Fragmented mappings | Force nonidentity, nonmonotonic physical block order via real allocation/release histories; independently verify table and exact K/V/scale bytes. Identity maps alone cannot certify indirection. |
| Numerical equivalence | Exact cache bytes and same-path output/logit checks where available, plus greedy token equality. Never compare DPAS to scalar and relax tolerance to hide an addressing bug. |
| Chunked prefill | `T=8,16,64,128,512`, nonzero starts crossing pages, final partial query tile, then decode; compare each flat/paged pair with identical chunking. Token-by-token prefill and chunked prefill are not interchangeable numerical references. |
| Independent slots | Different prompts/positions, interleaved growth, new slot joining while another decodes, cancellation and immediate slot reuse; outputs match isolated same-path references. |
| Reservations/exhaustion | Exact fit, one block short, partial growth rollback, impossible request, release unblocks queued work, APC cannot consume promised credits; conservation holds throughout. |
| APC correctness | Same full prefix/different suffix, deliberate hash collision, matching full prompt, shared full pages, private tails, GDN state isolation, donor release, cache eviction while a recipient lives, invalid/stale ids, failed save/adopt rollback. |
| Error/lifecycle | Invalid page/range, allocation failure via a real constrained budget, repeated init/free, reset before first token, decode failure and disconnect; no leaked ownership or silent flat fallback. |
| HTTP compatibility | Health/models, streaming/nonstreaming generation, reasoning separation, tool-call parsing with no markup leakage, stop/EOS/max-output behavior; run the actual `axe_vllm.py` round trip as well. |

Greedy-token equality alone is not byte-level KV proof, and a kernel gate is not
proof of scheduler/APC correctness. Preserve those distinctions in results.

### 8.3 Candidate service and mixed-length scenarios

After all serving paths and admission/APC gates pass, extend `serve_openai_xpu.py`
with the proposed CLI and start a candidate explicitly (no launcher defaults
changed). Example foreground command; no background job is created by this plan:

```bash
ZE_AFFINITY_MASK="$TEST_CARD" TQ_XPU_DEV=0 TQ_XPU_TP=1 .venv/bin/python xpu/tools/serve_openai_xpu.py --host 127.0.0.1 --port 8101 --lib xpu/build/libforward_qwen_xpu_paged_candidate.so --tqf "$MODEL" --model-dir "$HOME/models/knivesysl" --slots 4 --ctx 131072 --k64 gate,up,down --paged --kv-page 128 --kv-pool-tokens 131072 --response-idle-timeout 14400
```

Implement CLI/environment precedence deliberately: explicit CLI wins, native
queries are authoritative, and conflicts/unsupported choices fail visibly. The
current `setdefault` pattern must not let stale environment settings silently
change the requested mode or capacity. Default flat invocations remain unchanged.

Basic existing HTTP smoke command after its paging assertions are added:

```bash
.venv/bin/python xpu/tools/serve_smoke_xpu.py --base-url http://127.0.0.1:8101/v1 --model knivesysl-xe-qwen3.8-27b-w4a8 --conc 4 --gen 128
```

Extend that tool with an exact-token mixed-prompt scenario using the real tokenizer
and the server's measured `usage.prompt_tokens`; do not equate characters to tokens.
Use a matching long HTTP timeout, `ignore_eos` and fixed generation for comparison.
Run and retain results for:

1. 45k + 45k, pool 131,072, output 128 each: both admitted, both complete.
2. 28k + 75k, same pool/output: both admitted; report actual overlap rather than
   assuming batched decode persists through dissimilar prefills.
3. 65k + 75k, pool 131,072, output 128 each, APC off: one capacity-waits, then
   completes after release; no deadlock, corruption, silent truncation or OOM.
4. Same pair with pool 196,608: both admitted if the **whole** model/state/scratch
   allocation fits; otherwise report that configuration's allocation failure,
   not that paging supplies impossible memory.
5. Warm matching-prefix versions and distinct-prefix controls; show physical block
   reduction and exact continuation equivalence, not only a cache-hit counter.
6. Queue a short request during the long run; cancel an active and a pending
   request; verify capacity returned and later requests still complete.

For deterministic native proof of two long sequences sharing a decode wave,
prepare both prefixes under one valid reservation ledger, then invoke real batch
steps and compare against their isolated references. For HTTP report naturally
observed overlap. Do not impose an artificial server-side prefill barrier.

Benchmark using the same client/rules as `bench_openai.py`: cold APC-off cells,
fixed generation, no speculative decoding; warm APC is a separately labelled set.
Existing `bench_matrix_xpu.sh` iterates homogeneous contexts and is not sufficient
for the mixed-length cases. Use equal usable capacity for flat/paged memory
comparisons where both configurations are representable; label flat configurations
that cannot fit rather than changing their allocation silently.

Required result columns: exact prompt lengths, requested/generated output tokens,
per-request limit, pool page/tokens/bytes, slots, effective mode/attention/tier,
APC state/reused tokens, admission/queue time, TTFT, ITL p50/p99, per-stream and
aggregate decode tok/s, end-to-end tok/s, observed batch widths, peak physical
blocks and reservations, errors, and total wall time. Run paired alternating-order
repeats on the same card/configuration; report spread. A scatter microbenchmark
or old vendor result is not an attention/server performance comparison.

## 9. Completion, cleanup, and handoff

After the first functioning native smoke, finish the remaining implementation and
targeted behavioral regression coverage above; no “foundation complete” stop is a
feature-delivery boundary. After end-to-end success, update
[CHANGELOG.md](../CHANGELOG.md), the campaign doc and launcher/help text with the
actual mode, capacity semantics and measured results. Preserve historical lessons,
append corrections rather than rewriting history, and remove temporary diagnostic
hooks. Keep the candidate/known-good library separation until rollback is no
longer needed. Do not remove deliberately supported flat mode as “cleanup.”

The feature is complete only when all 14 items have evidence, existing flat
behavior still passes, pooled serving handles capacity pressure and APC safely,
and the deliberately selected endpoint has been verified in its effective mode.
A built library, an allocator selftest, a deterministic short chain, or a written
plan satisfies none of those deployment claims by itself.

**Next implementation action:** Step A, the coupled native storage + decode
write/read slice in a separate candidate build, then the attested scalar gate.
Do not restart the working server or begin another vendor-research campaign first.
