#!/usr/bin/env bash
# build_vllm_venv.sh - rebuild the vLLM-XPU venv used by serve_vllm_xpu.sh.
#
# WHY THIS EXISTS. The venv lived at /tmp/vllmxpu-venv, and /tmp on this box is
# a 30 GB tmpfs - so every reboot destroyed it and the recipe survived only as
# three lines of prose in xpu/CHANGELOG.md. It has been reconstructed by hand
# more than once. This makes the automatable part one command, and defaults to
# REAL DISK so a reboot stops undoing it.
#
#   xpu/tools/build_vllm_venv.sh                  # -> /var/tmp/vllmxpu-venv
#   VENV=/somewhere/else xpu/tools/build_vllm_venv.sh
#
# PROVENANCE: the pins and the first three workarounds are transcribed from the
# 2026-09-04 head-to-head entry in xpu/CHANGELOG.md ("Venv resurrection").
# They are what that session converged on after fighting the dep resolver; they
# are NOT independently re-derived. The smoke check at the end is the only
# evidence this venv is usable - a clean import proves nothing about numerics.
set -euo pipefail
cd "$(dirname "$0")/../.."

VENV="${VENV:-/var/tmp/vllmxpu-venv}"
TORCH_IDX="${TORCH_IDX:-https://download.pytorch.org/whl/xpu}"

command -v uv >/dev/null || { echo "uv not on PATH; install it first" >&2; exit 1; }
case "$VENV" in
    /tmp/*) echo "WARNING: $VENV is on tmpfs - a reboot will wipe it again." >&2 ;;
esac

echo "[venv] creating $VENV"
uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python"

# torch FIRST, from the xpu index: vllm must not be allowed to resolve a CPU
# torch over the top of it.
echo "[venv] torch 2.13+xpu stack"
uv pip install --python "$PY" --index-url "$TORCH_IDX" \
    "torch==2.13.*" "torchvision==0.28.*" "torchaudio==2.11.*"

echo "[venv] vllm 0.27.1 + vllm-xpu-kernels 0.1.12.3"
uv pip install --python "$PY" "vllm==0.27.1" "vllm-xpu-kernels==0.1.12.3"

# vllm's dependency pull grabs CPU wheels for torchvision/torchaudio and
# overwrites the +xpu ones above. Force them back.
echo "[venv] restoring +xpu wheels that vllm's dep pull clobbered"
uv pip install --python "$PY" --index-url "$TORCH_IDX" --reinstall \
    "torchvision==0.28.*" "torchaudio==2.11.*"

# torchcodec has no +xpu wheel and its import aborts model load.
echo "[venv] removing torchcodec (no +xpu wheel; breaks load)"
uv pip uninstall --python "$PY" torchcodec 2>/dev/null || true

SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "[venv] site-packages: $SITE"

cat <<NOTE

TWO PATCHES ARE MANUAL, and this script does NOT fake them:

  (a) platform resolver must prefer xpu over cuda. This host drives its display
      from an RTX 5090 while computing on two Arc Pro B70s, so upstream's probe
      order resolves CUDA and the XPU backend never initialises.
        -> $SITE/vllm/platforms/__init__.py
  (b) the NVIDIA-only minimax warmup import needs guarding.

The changelog records that both were needed but not the exact diffs, and
generating a guess at upstream internals is how you get a venv that imports
cleanly and computes wrong. Apply them if the smoke check below fails, then
paste the REAL diffs into this script so the next rebuild is one command.

The triton shim needs no patching: serve_vllm_xpu.sh already puts
xpu/tools/triton_shim on PYTHONPATH, which neuters CudaDriver.is_active so
Triton does not disable itself when it sees two GPU vendors.

NOTE

echo "[venv] smoke check"
ZE_AFFINITY_MASK=0 "$PY" - <<'SMOKE' || echo "  SMOKE FAILED - apply the manual patches above"
import torch
print("  torch", torch.__version__)
print("  xpu available:", torch.xpu.is_available())
if torch.xpu.is_available():
    print("  devices:", torch.xpu.device_count(), "name:", torch.xpu.get_device_name(0))
import vllm
print("  vllm", vllm.__version__)
SMOKE

cat <<EOF

[venv] done -> $VENV

serve the SAME weights we benchmark against:
  VENV=$VENV CARD=1 PORT=8000 \\
    MODEL=\$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \\
    NAME=qwen38-vllm-xpu MAXLEN=8448 \\
    xpu/tools/serve_vllm_xpu.sh
EOF
