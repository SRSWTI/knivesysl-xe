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
# PROVENANCE: 2026-09-08 real vLLM 0.27.1/XPU startup and exact HTTP smoke.
# Pin this known stack; source-shape checks below reject unknown patch targets.
# The MiniMax guard follows Intel's retained scripts/vllm/vllm-fix.patch;
# explicit-XPU platform selection is the locally qualified dual-vendor fix.
# An import smoke is not numerical qualification or a serving benchmark.
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
echo "[venv] torch 2.13.0+xpu stack"
uv pip install --python "$PY" --index-url "$TORCH_IDX" \
    "torch==2.13.0+xpu" "torchvision==0.28.0+xpu" "torchaudio==2.11.0+xpu"

echo "[venv] vllm 0.27.1 + vllm-xpu-kernels 0.1.12.3"
uv pip install --python "$PY" "vllm==0.27.1" "vllm-xpu-kernels==0.1.12.3"

# vllm's dependency pull grabs CPU wheels for torchvision/torchaudio and
# overwrites the +xpu ones above. Force them back.
echo "[venv] restoring +xpu wheels that vllm's dep pull clobbered"
uv pip install --python "$PY" --index-url "$TORCH_IDX" --reinstall \
    "torch==2.13.0+xpu" "torchvision==0.28.0+xpu" "torchaudio==2.11.0+xpu"

# torchcodec has no +xpu wheel and its import aborts model load.
echo "[venv] removing torchcodec (no +xpu wheel; breaks load)"
uv pip uninstall --python "$PY" torchcodec 2>/dev/null || true

# CUDA Triton and triton-xpu overlap the same Python package files. Uninstall
# CUDA first, then restore the exact Intel wheel, even if already installed.
echo "[venv] restoring Intel-only Triton 3.7.2 and NumPy 2.3.5"
uv pip uninstall --python "$PY" triton
uv pip install --python "$PY" --index-url "$TORCH_IDX" --reinstall --no-deps \
    "triton-xpu==3.7.2"
uv pip install --python "$PY" "numpy==2.3.5"

SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "[venv] site-packages: $SITE"

"$PY" - <<'PATCH'
from importlib.metadata import version
from pathlib import Path
import sysconfig

expected = {
    "vllm": "0.27.1", "vllm-xpu-kernels": "0.1.12.3",
    "torch": "2.13.0+xpu", "torchvision": "0.28.0+xpu",
    "torchaudio": "2.11.0+xpu", "triton-xpu": "3.7.2", "numpy": "2.3.5",
}
for package, wanted in expected.items():
    actual = version(package)
    if actual != wanted:
        raise SystemExit(f"Refusing unqualified {package}: {actual}, expected {wanted}")

site = Path(sysconfig.get_paths()["purelib"])
resolver = site / "vllm/platforms/__init__.py"
warmup = site / "vllm/model_executor/warmup/minimax_m3_msa_warmup.py"
anchor = "    activated_oot_plugins = list(set(activated_plugins) & set(platform_plugins.keys()))\n"
selection = anchor + '''
    # Explicit XPU serving must not select the NVIDIA display adapter.
    if os.environ.get("VLLM_TARGET_DEVICE") == "xpu" and "xpu" in activated_builtin_plugins:
        activated_builtin_plugins = ["xpu"]
'''
import_line = "from vllm.models.minimax_m3.nvidia.model import MiniMaxM3SparseAttention\n"
function_line = 'def minimax_m3_msa_warmup(worker: "Worker") -> None:\n'
guard = '''    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return
'''

# Validate every target in memory before writing either file. Already-applied
# exact patches are accepted; unfamiliar source is a hard failure, not a guess.
original_resolver = resolver.read_text()
original_warmup = warmup.read_text()
if original_resolver.count(selection) == 1:
    patched_resolver = original_resolver
elif original_resolver.count(anchor) == 1 and "# Explicit XPU serving" not in original_resolver:
    patched_resolver = original_resolver.replace(anchor, selection, 1)
else:
    raise SystemExit(f"Unexpected vLLM 0.27.1 resolver source: {resolver}")
patched_function = function_line + guard + "\n    " + import_line + "\n"
if original_warmup.count(patched_function) == 1 and original_warmup.count(guard) == 1:
    patched_warmup = original_warmup
elif (original_warmup.count("\n" + import_line) == 1 and
      original_warmup.count(function_line) == 1 and original_warmup.count(guard) == 1):
    patched_warmup = original_warmup.replace("\n" + import_line, "\n", 1)
    patched_warmup = patched_warmup.replace(guard, "", 1)
    patched_warmup = patched_warmup.replace(function_line, patched_function, 1)
else:
    raise SystemExit(f"Unexpected vLLM 0.27.1 MiniMax source: {warmup}")
for path, original, patched in ((resolver, original_resolver, patched_resolver),
                                (warmup, original_warmup, patched_warmup)):
    if patched != original:
        path.write_text(patched)
    print(f"  qualified compatibility patch: {path}")
PATCH

echo "[venv] smoke check"
env -u LD_LIBRARY_PATH ZE_AFFINITY_MASK="${CARD:-0}" CUDA_VISIBLE_DEVICES="" \
    VLLM_TARGET_DEVICE=xpu PYTHONPATH="$PWD/xpu/tools/triton_shim${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" - <<'SMOKE'
import torch
print("  torch", torch.__version__)
print("  xpu available:", torch.xpu.is_available())
assert torch.xpu.is_available(), "XPU is unavailable"
print("  devices:", torch.xpu.device_count(), "name:", torch.xpu.get_device_name(0))
import vllm
print("  vllm", vllm.__version__)
from vllm.platforms import current_platform
assert current_platform.is_xpu(), "vLLM did not select XPU"
SMOKE

cat <<EOF

[venv] done -> $VENV

serve the same BASE MODEL using a different quantized representation:
  env -u LD_LIBRARY_PATH VENV=$VENV CARD=1 PORT=8000 \\
    MODEL=\$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \\
    NAME=qwen38-vllm-xpu MAXLEN=8448 \\
    xpu/tools/serve_vllm_xpu.sh

The clean LD_LIBRARY_PATH is required here: inherited oneAPI libraries caused
an observed XCCL initialization segfault; the isolated wheel runtime started.
This launcher enables prefix caching. For cold comparisons invoke vllm serve
with --no-enable-prefix-caching and no speculative configuration, using the
same model/graph/Triton settings and the shared bench_engine_compare.py client.
EOF
