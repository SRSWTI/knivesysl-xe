# knivesysl-xe Phase 0 — probe results & format decision

Measured 2026-09-01 on 2x Intel Arc Pro B70 (BMG-G31, Xe2, 256 XVE, 32 GiB, 2800 MHz),
oneAPI 2026.1.1, level-zero V2 1.15.39122, same host as the 5090 dev box.
Harness: `xpu/probe/dpas_probe.cpp` (inline vISA `dpas`, subgroup-16, M=8, 8
independent accumulator chains, all-ones operands so the accumulator value
proves both execution and K depth: `d == iters * K`).

## DPAS rates (device 0 / device 1, TOPS)

| config | K | dev0 | dev1 | check |
|---|--:|--:|--:|---|
| s4 x s4 | 64 | **733.3** | 733.5 | PASS |
| u4 x u4 | 64 | 733.3 | 733.4 | PASS |
| s8 x s8 | 32 | 366.4 | 366.7 | PASS |
| u8 x u8 | 32 | 366.6 | 366.7 | PASS |
| a:s8 x b:s4 | 32 | 364.8 | 364.9 | PASS |
| a:s4 x b:s8 | 32 | 351.7 | 351.5 | PASS |
| bf16 | 16 | 183.4 | 183.3 | PASS |
| tf32 | 8 | 91.6 | 91.7 | PASS |
| fp16 | 16 | 68.8 | 63.2 | PASS (anomalously slow — do not use; bf16 instead) |

Every precision doubles cleanly down the ladder (91 -> 183 -> 366 -> 733):
the systolic array is issue-limited, K depth is free. Mixed s8xs4 runs at
K=32 (s8 rate) per `K = 256/max(bits)` — the 733 tier requires s4 on BOTH
operands.

## Memory / latency / cross-device

| probe | dev0 | dev1 |
|---|--:|--:|
| pure read | 602.5 GB/s | 601.9 GB/s (99% of 608 spec) |
| triad 2r+1w | 517.3 GB/s | 518.9 GB/s |
| launch latency, async in-order chain | 1.42 us | 1.44 us |
| launch latency, sync each | 6.19 us | 5.37 us |
| P2P copy (peer access reports "yes") | 2.0 GB/s both directions | |

## Decisions

1. **Ship weight tier: W4A8** — s4 weights (with our block scales applied to the
   s32 accumulator), s8 activations, mixed DPAS at 366 TOPS. ~15 GiB weight
   read/decode-step -> dense decode ceiling ~40 tok/s; x accept 2.7-3.3 spec
   decode -> ~90-130 tok/s single-stream target.
   *Phase-2 outcome:* confirmed as the ship tier, and the roofline estimate held
   - measured 31.6 tok/s dense decode (79% of the ~40 tok/s ceiling predicted
   here) with signed-INT4 made asymmetric and only `lm_head` kept at s8. The
   spec-decode multiplier is still unclaimed: MTP is not ported. See
   `xpu/CHANGELOG.md`.
2. **Opt-in tier: W4A4** — s4 x s4 at 733 TOPS for prefill-heavy work; the
   NVFP4-role analog (same quality-vs-speed slot as `TQ_W_NVFP4` on SM120).
3. **DeltaNet chunk scan keeps `TQ_DN_MM=3` semantics** — tf32 DPAS is native
   (91.7 TOPS); no precision downgrade to bf16 needed.
4. **fp16 is quarantined** — 3x slower than bf16 on this silicon/driver; all
   fp MMA paths use bf16.
5. **TP2 is dead on arrival** — 2 GB/s cross-device despite peer access
   reporting supported. Dual-card = two replicas + APC-aware routing.
   (Corroborates vllm#41663 dual-B70 TP2 instability.)
6. Launch latency is CUDA-class (1.4 us async): the spec-round launch count is
   not an architectural blocker; keep in-order queues, defer command-list
   replay to Phase 4.
7. Roofline honesty: 602 GB/s read is achievable in plain SYCL — decode GEMV
   KPI is % of 602, not of the spec sheet.

## Open items carried forward
- fp16 slowness: revisit after a driver update; harmless (bf16 everywhere).
- P2P 2 GB/s: check large-BAR / IAF config someday; not load-bearing.
