#!/usr/bin/env bash
# Serve the vendor baseline (vLLM-XPU) on ONE Intel Arc Pro B70, as an
# OpenAI-compatible endpoint that tools/axe_vllm.py can drive unmodified.
#
# Layout on this box:
#   :8000  vLLM-XPU, Qwen3.6-27B-GPTQ-Int4   <- this script, axe_vllm's DEFAULT
#   :8010  knivesysl CUDA (serve_batched.py, model "ksl") on the 5090
# so `axe_vllm.py` hits the vendor and `axe_vllm.py --base-url
# http://127.0.0.1:8010/v1` hits ours.
#
# HARD RULES learned the painful way, do not relax these:
#  - ZE_AFFINITY_MASK pins ONE card. Two 22 GiB model loads on one card wedged
#    it with Level-Zero DEVICE_LOST and needed a reboot to clear.
#  - CORRECTED 2026-09-04: the display runs on the NVIDIA 5090 (card1,
#    HDMI connected/enabled); BOTH B70s (card2/card3) have no connected
#    outputs. Desktop apps merely hold enumeration fds on the Intel render
#    nodes. Either B70 is safe for compute; card 1 stays the serving
#    default only by convention.
#  - /tmp is a 30 GB tmpfs, i.e. RAM. HF_HOME must point at real disk or an
#    11 GiB download dies with "Disk quota exceeded".
#  - The repo's own .venv is a CUDA torch build and MUST NOT be touched. This
#    uses the isolated /tmp/vllmxpu-venv (torch 2.13.0+xpu, vLLM 0.27.1,
#    vllm-xpu-kernels 0.1.12.3). Because that venv lives on tmpfs it does NOT
#    survive a reboot and has to be rebuilt; see xpu/CHANGELOG.md.
#
# Differences from the BENCHMARK config (xpu/CHANGELOG.md records the numbers):
# prefix caching is ON here. Benchmarks disabled it to measure cold prefill,
# but for real agentic traffic through axe_vllm it is both vLLM's default and a
# large win on shared prompt prefixes, so leaving it off would hobble the
# vendor unfairly.
set -euo pipefail

CARD="${CARD:-1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-$HOME/models/Qwen3.6-27B-MTP-Preserved-GPTQ-Int4}"
NAME="${NAME:-qwen36-xpu}"
MAXLEN="${MAXLEN:-131072}"
VENV="${VENV:-/tmp/vllmxpu-venv}"

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "no vLLM at $VENV (tmpfs wipes it on reboot) -- rebuild it first" >&2
  exit 1
fi
if [ ! -d "$MODEL" ]; then
  echo "no model at $MODEL" >&2
  exit 1
fi

export ZE_AFFINITY_MASK="$CARD"
export VLLM_TARGET_DEVICE=xpu
export HF_HOME=/var/tmp/vllmxpu/hf-home
export HF_HUB_CACHE=/var/tmp/vllmxpu/hf-home/hub
mkdir -p "$HF_HUB_CACHE"

# Graph capture is OPT-IN on vLLM-XPU and its absence is worth 2-4x. From
# b70_ai_things (vllm/nvfp4/serve_nvfp4_27b.sh): the default arm is
# --enforce-eager, and GRAPH=1 means env VLLM_XPU_ENABLE_XPU_GRAPH=1 plus a
# --compilation-config carrying cudagraph_mode. Their measured profile
# (docs/20260826_qwen36_graph_runtime_profile.md, quoted in FINDINGS.md):
# PIECEWISE = 41 graph pieces, 41 fence resets, 41 host event waits, 82
# command-list submissions PER TOKEN; FULL decode + Triton attention = one
# fence, two waits, two submissions, and 61.55 vs 50.37 tok/s, +22.2%.
# An earlier eager run here measured 19.98 tok/s at 2k, so eager is NOT a fair
GRAPH="${GRAPH:-1}"
CGMODE="${CGMODE:-PIECEWISE}"
CAPSIZES="${CAPSIZES:-1,2,4,8,16,32}"
# Triton is DEAD in this venv without the shim: two GPU vendors in one box make
# Triton's driver auto-select raise "2 active drivers ([XPUDriver, CudaDriver])".
# vLLM spawns its engine core as a separate process, so the fix has to be on
# every interpreter start -> sitecustomize.py on PYTHONPATH. Without it
# TRITON_ATTN silently cannot engage, MTP has no rejection sampler, and FULL
# capture is blocked. b70_ai_things calls this TRITONSHIM.
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/triton_shim${PYTHONPATH:+:$PYTHONPATH}"
GRAPH_ARGS=( --enforce-eager )
if [ "$GRAPH" = 1 ]; then
  export VLLM_XPU_ENABLE_XPU_GRAPH=1
  export VLLM_USE_AOT_COMPILE=0
  _CAPS="[$(echo "$CAPSIZES" | tr -d ' ')]"
  GRAPH_ARGS=( --compilation-config \
    "{\"cudagraph_mode\":\"$CGMODE\",\"cudagraph_capture_sizes\":$_CAPS}" )
fi

exec "$VENV/bin/vllm" serve "$MODEL" \
  --quantization gptq \
  --dtype float16 \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --port "$PORT" \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --served-model-name "$NAME" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  "${GRAPH_ARGS[@]}" \
  --language-model-only
