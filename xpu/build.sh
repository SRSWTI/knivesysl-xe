#!/usr/bin/env bash
# build libforward_qwen_xpu.so
#
# TQ_XPU_AOT=1 compiles ahead-of-time for BMG-G31 (adds native ISA next to
# SPIR-V). Measured 2026-09-04: steady-state identical to JIT and init time
# identical with a warm JIT cache; AOT buys cold-cache determinism and is the
# required path for per-kernel IGC options (e.g. -ze-opt-large-register-file
# for the GRF256 experiments — see docs/level-up-xpu.md queue item 3).
set -euo pipefail
cd "$(dirname "$0")"
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
mkdir -p build
ICPX_FLAGS="-fsycl -O2 -fPIC -std=c++17 -Isrc"
if [ "${TQ_XPU_AOT:-0}" = "1" ]; then
  ICPX_FLAGS="$ICPX_FLAGS -fsycl-targets=spir64_gen"
  AOT_BACKEND=(-Xsycl-target-backend "-device bmg-g31 ${TQ_XPU_AOT_OPTS:-}")
  echo "[build] AOT for bmg-g31${TQ_XPU_AOT_OPTS:+ (opts: $TQ_XPU_AOT_OPTS)}"
else
  AOT_BACKEND=()
fi
echo "[build] compiling TU..."
icpx $ICPX_FLAGS "${AOT_BACKEND[@]}" -shared -o build/libforward_qwen_xpu.so \
    src/decode.cpp src/tqf_loader.cpp src/kernels_core.cpp src/kernels_seq.cpp \
    src/kernels_dpas.cpp src/kernels_prefill.cpp
echo "[build] -> xpu/build/libforward_qwen_xpu.so"
