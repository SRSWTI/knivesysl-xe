# knivesysl-xe

native Intel Xe2 inference for the Qwen3.8-27b text tower, built with SYCL, Level Zero, and hand-written DPAS/XMX kernels on Intel Arc Pro B70.

this is the Intel engine extracted from [srswti/knivesysl](https://github.com/SRSWTI/knivesysl), whose primary engine targets NVIDIA CUDA. the existing `xpu/` directory layout is retained so build scripts, launchers, and qualification commands continue to resolve their paths. this repository does not contain the CUDA engine, model weights, compiled binaries, or vendor checkouts.

Python handles tokenization, scheduling, and the OpenAI-compatible HTTP interface. model execution lives in `libforward_qwen_xpu.so`, accessed through a C ABI. the native inference path does not use PyTorch, vLLM, or Triton; vendor comparison scripts are optional research tools, not engine dependencies.

## what we built

- **native quantized projections.** probe-verified Xe2 DPAS layouts, W4A8 GEMV/GEMM, batched projections, and an optional K64 quantization tier. the serving launcher uses K64 for the gate/up/down MLP projections by default.
- **shared paged KV.** requests draw full-attention KV blocks from one reference-counted pool instead of reserving a maximum-context slab for every slot. reservation-based admission prevents requests from collectively promising more blocks than the pool contains.
- **hybrid prefix caching.** immutable full KV pages can be shared; gated DeltaNet recurrent/convolution state remains slot-local and is captured with the prefix checkpoint. releasing a request preserves pages still owned by checkpoints.
- **continuous decode batching.** requests can join and leave without draining the entire batch. cancellation returns active page ownership and reservation credits.
- **XMX prefill attention.** DPAS computes both QK and probability-times-v. softmax and accumulation remain FP32; three BF16 components represent q and scaled probabilities. the existing E4M3 KV bytes and per-token/head FP16 scales are retained, without a full-cache conversion.
- **packed multi-request prefill.** prompt segments share projection GEMMs and one layer walk; attention can combine segments in one grid after their KV writes. each segment retains its original split-k reduction order rather than silently changing arithmetic with packed batch size.
- **serving integration.** total prefill-wave budgets, scalar bootstrap for position-zero tails, exact requested output counts, streaming stops, structured tool calls, APC, and disconnect recovery.

the target is a hybrid model: 64 layers, including 16 full-attention layers and 48 gated DeltaNet layers. paging applies to full-attention KV; it does not turn recurrent state into token-addressable KV.

## requirements

- Linux with an Intel Arc Pro B70 / BMG-G31 Xe2 GPU and a working Intel compute-runtime / Level Zero installation.
- Intel oneAPI DPC++ compiler (`icpx`); development and qualification used oneAPI 2026.1.
- Python 3.12, `transformers`, and `numpy` for the Python tools.
- a compatible Qwen3.8-27b TQF model and its matching local tokenizer/chat-template files.

weights are not bundled. use an existing compatible TQF, or the parent project's [`tools/convert_qwen_tqf.py`](https://github.com/SRSWTI/knivesysl/blob/main/tools/convert_qwen_tqf.py) and its conversion instructions. an arbitrary Hugging Face weights directory is not a substitute for the native TQF file.

## build

run commands from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

source /opt/intel/oneapi/setvars.sh --force
bash xpu/build.sh
# output: xpu/build/libforward_qwen_xpu.so
```

the build defaults to SPIR-V/JIT. to include BMG-G31 ahead-of-time code:

```bash
TQ_XPU_AOT=1 bash xpu/build.sh
```

the build and launcher source `/opt/intel/oneapi/setvars.sh`; adapt your environment if oneAPI is installed elsewhere. the Intel runtime and compiler libraries must remain discoverable when loading the shared library.

## serve

an explicit paged configuration matching the qualified capacity settings:

```bash
MODEL=/absolute/path/to/qwen3_8-27b-e2m3-mtp.tqf \
MODEL_DIR=/absolute/path/to/tokenizer-directory \
CARD=0 PORT=8101 SLOTS=8 CTX=131072 \
PAGED=1 KV_PAGE=128 KV_POOL_TOKENS=131072 \
PREFILL_XMX=auto PACKED_PREFILL=1 \
CHUNK=64 CHUNK_IDLE=512 \
RESPONSE_IDLE_TIMEOUT=14400 \
bash xpu/tools/serve_xe.sh
```

the launcher detaches the server and writes `/tmp/xe_serve_8101.log` unless `LOG` is supplied. prefix caching is enabled by the server default. the example changes capacity explicitly: the launcher's unconfigured defaults are **four slots, 32768 context, and flat KV**, not this eight-slot paged configuration.

**do not load two model instances on the same card.** check current GPU ownership and listening ports before starting another process. `CARD` selects the Level Zero device through `ZE_AFFINITY_MASK`; it is not a Linux DRM node number. the server binds to loopback by default and has no authentication. do not expose it to an untrusted network.

```bash
curl http://127.0.0.1:8101/health
curl http://127.0.0.1:8101/v1/models
curl http://127.0.0.1:8101/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain prefix caching briefly."}],"max_tokens":128}'
```

key controls:

| control | meaning |
|---|---|
| `CTX` | per-sequence prompt-plus-output limit; not the aggregate pool capacity. |
| `SLOTS` | maximum simultaneous sequence-state slots. |
| `KV_POOL_TOKENS` | shared physical KV capacity across active requests and retained checkpoints. |
| `KV_POOL_MB` | alternative decimal-MB KV budget; mutually exclusive with `KV_POOL_TOKENS`. |
| `KV_PAGE` | supported page sizes: 128 or 256 tokens. |
| `PREFILL_XMX=auto` | select XMX for profitable supported shapes and scalar attention for small work. |
| `PREFILL_XMX=0` / `1` | select scalar reference / require XMX. |
| `PACKED_PREFILL=0` / `1` | serial / packed prompt waves. |
| `CHUNK`, `CHUNK_IDLE` | total prefill-wave token budgets, not per-request budgets. |

eight slots with a 131072-token pool **does not mean eight independent 131072-token reservations**. requests that cannot currently obtain their reservation wait for capacity; impossible requests are rejected. GDN state, weights, and scratch also consume device memory.

## where we beat vLLM-XPU

**historical HTTP-server comparisons, september 4–5, 2026; not a fresh benchmark of the current XMX/paged build.** each engine used one Arc Pro B70 on the same host. requests used cold prefixes, no speculative decoding, and 128 generated tokens with `ignore_eos`. the retained knivesysl-xe stage-1 matrix used one repetition per cell: these are point measurements, not confidence intervals.

the knivesysl-xe arm was the then-current W4-K32 ship tier with scalar-G3 attention. the recorded vLLM-XPU baseline was vLLM 0.27.1, XPU kernels 0.1.12.3, torch 2.13+XPU, Triton attention, piecewise XPU graphs, FP8 KV, and GPTQ-Int4 weights for the same base model. **weight representations and KV formats differ; this is not identical quantized tensors or demonstrated equal task quality.**

| prompt tokens | concurrency | metric | knivesysl-xe | vLLM-XPU | advantage |
|---:|---:|---|---:|---:|---|
| 8192 | 1 | TTFT, seconds ↓ | **28.14** | 55.30 | **1.97× lower latency** |
| 8192 | 1 | reported prefill, tokens/s ↑ | **291** | 148 | **1.97× throughput** |
| 8192 | 1 | decode per stream, tokens/s ↑ | **28.54** | 10.50 | **2.72× throughput** |
| 8192 | 4 | mean TTFT, seconds ↓ | **84.38** | 190.50 | **2.26× lower latency** |
| 8192 | 4 | reported prefill per request, tokens/s ↑ | **97** | 43 | **2.26× throughput** |
| 8192 | 4 | aggregate decode, tokens/s ↑ | **14.2** | 12.0 | **1.18× throughput** |

prefill here is the client's prompt-length/TTFT metric, not an isolated kernel measurement. the four-request aggregate win is **1.18×, not 7×**: the old 7× claim mixed a standalone native-engine result with a server result and different concurrency. it is not a valid server-to-server comparison.

### where vLLM-XPU still won

| prompt tokens | concurrency | metric | knivesysl-xe | vLLM-XPU |
|---:|---:|---|---:|---:|
| 512 | 1 | reported prefill, tokens/s ↑ | 363 | **910** |
| 512 | 8 | aggregate decode, tokens/s ↑ | 81.3 | **158** |
| 2048 | 4 | aggregate decode, tokens/s ↑ | 32.6 | **76.5** |

the historical advantage was at depth, not across the whole workload surface. the stage-1 server also dropped from 28.5 aggregate tokens/s at 8192×1 to 12.6 at 8192×2; that concurrency limitation must not be hidden behind the winning cells. the later XMX/packed-prefill improvement below is a separate same-engine comparison; do not multiply its speedup into these older vendor numbers.

sources: [retained stage-1 server matrix](benchmarks/xe-stage1-20260905.csv), [transcribed vendor baseline](benchmarks/vllm-xpu-20260904.csv), and the [historical baseline/configuration record](xpu/CHANGELOG.md#same-weights-head-to-head-vs-vllm-xpu-2026-09-04-live). the vendor CSV is transcribed from that record, not presented as recovered raw request traces.

### SGLang-XPU: no measured win claimed

we researched SGLang's Intel attention kernels and packed scheduling, but have not recovered a comparable SGLang-XPU/B70 benchmark for this engine. the available parent-workspace `sglang_core.log` belongs to a CUDA/NVFP4 campaign with CUDA graph capture; using those numbers as an Intel-XPU comparison would be wrong. **there is no substantiated SGLang-XPU speedup number to publish yet.** source support and kernel reconnaissance do not establish a performance win.

## measured results

paired HTTP measurement on one B70, the same library and K64-MLP weights, concurrent 4096- and 8192-token prompts, 32 generated tokens per request, APC disabled, and two repetitions with reversed launch order:

| metric | scalar attention + serial prefill | automatic XMX + packed prefill |
|---|---:|---:|
| 4096-token request TTFT | 19.905 s | **16.802 s** |
| 8192-token request TTFT | 35.161 s | **29.908 s** |
| complete pair wall time | 36.263 s | **31.009 s** |

approximately 15% lower TTFT and 14.5% lower pair completion time in this workload. this is not a several-fold whole-model speedup or a claim of superiority over vendor engines. decode latency while competing with prefill remained around 273 ms in this pair; the later uncontended stream measured 35.3 ms.

### numerical policy

XMX attention is **not bit-identical to scalar attention**. the explicitly approved qualification policy is at least 90% aggregate reference-token agreement, while retaining per-prompt results:

- production automatic-split result: **90/96 (93.75%)**.
- heap: 31/32; graph: **28/32**; transactions: 31/32.
- packed versus independent execution: **64/64**.
- fixed-split flat and page256 runs: 91/96.

the graph prompt did not pass a stricter per-prompt 90% threshold. this is documented, not hidden. these are small sampled token-agreement measurements, not task-accuracy scores or a general quality guarantee. quantized activation thresholds can amplify small floating-point differences.

## qualification

recorded before extraction into this repository:

- **86 native cases:** 21 production, 31 flat, and 34 page256 cases passed.
- **11 direct-kernel cases:** ragged queries, reversed page maps, both page sizes, the final 131072-token boundary, automatic dispatch, and packed descriptor lifetime. relative L2 and peak-normalized error gates were `1e-4`; the compared flat/paged and packed/single XMX outputs were bit-exact.
- **seven HTTP scenarios:** short prompts, packed three-wave APC reuse, raw streaming counts, split stop strings, structured tools, and disconnect/recovery passed on the promoted endpoint.
- **actual width eight:** eight simultaneous 256-token requests returned 16 tokens each, with four width-8 prefill waves and 23 width-8 decode steps observed.

the retained runnable qualification tools include:

```bash
.venv/bin/python xpu/tools/prefill_http_check.py \
  --base-url http://127.0.0.1:8101/v1 \
  --model-dir /absolute/path/to/tokenizer-directory \
  --output /tmp/prefill-http-check.json

.venv/bin/python xpu/tools/prefill_batch_check.py --help
.venv/bin/python xpu/tools/paged_parity.py --help
.venv/bin/python xpu/tools/paged_state_check.py --help
```

native qualification initializes a model: run it only on a free, explicitly selected card. do not run it alongside a serving model on that same card. HTTP qualification instead drives an existing endpoint.

no new 75k full-model benchmark or vendor head-to-head was performed for the XMX/packed-prefill release. some older long-context warm measurements were stopped or never run. paging improves capacity sharing; it does not remove the cost of long cold prefill.

## source map and provenance

| path | purpose |
|---|---|
| `xpu/src/decode.cpp` | native execution, packed waves, pool ownership, reservations, checkpoints. |
| `xpu/src/tqf_loader.cpp` | model loading and state allocation. |
| `xpu/src/kernels_dpas.cpp` | quantized DPAS projection kernels. |
| `xpu/src/kernels_seq.cpp` | attention reference paths, KV writes, and GDN/convolution kernels. |
| `xpu/src/kernels_prefill.cpp` | native XMX prefill attention and packed attention dispatch. |
| `xpu/src/kernels_core.cpp` | core embedding, normalization, activation, and argmax operations. |
| `xpu/tools/serve_openai_xpu.py` | scheduler, prefix caching, and OpenAI-compatible HTTP service. |
| `xpu/probe/` | source for hardware/layout and kernel probes. |
| `xpu/CHANGELOG.md` | detailed measured history, including failed experiments and limitations. |
| `xpu/docs/level-up-xpu.md` | design rationale and optimization history. |
| `xpu/docs/paged-kv-implementation-plan.md` | historical paging rollout specification. |

historical documents refer to parent-repository CUDA files, vendor clones, and local `xpu/build/*results*/` receipts. those references describe the original development workspace; they are not promises that those files are bundled here. the publication omits generated binaries, build results, model/tokenizer files, local Python environments, and scraped third-party documentation archives. scraping utilities and original project research notes are retained.

optional vendor comparison scripts require their own vLLM environment and models. CUDA-oracle tools require the parent CUDA engine or previously generated reference artifacts. none of those are needed to build or serve the native Intel engine.
