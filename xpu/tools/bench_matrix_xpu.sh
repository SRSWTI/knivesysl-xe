#!/usr/bin/env bash
# bench_matrix_xpu.sh - the XPU decision matrix, mirroring the CUDA side's
# tools/bench_core_matrix.sh.
#
# TRUTH RULES (non-negotiable, this is the headline harness):
#   * COLD PREFILL. Prefix caching is OFF (--no-prefix-cache). Every request
#     pays its full prompt. APC is a real feature and it is measured, but in
#     its own labelled run below - never mixed into the truth cells.
#   * NO SPECULATIVE DECODE. Every token is one committed forward step.
#   * ignore_eos + fixed gen, so every cell does identical work.
#   * one server per arm: the format/attention selectors are read in qwn_init,
#     so an arm change means a restart, not a flag flip.
#
# Cells: arm x context x concurrency. Each arm restarts the server.
#
#   xpu/tools/bench_matrix_xpu.sh                 # core truth matrix
#   ARMS="ship k64all" CTXS="512,8192" xpu/tools/bench_matrix_xpu.sh
#   EXTRAS=1 xpu/tools/bench_matrix_xpu.sh        # + labelled APC/warm runs
set -euo pipefail
cd "$(dirname "$0")/../.."
# the engine .so links libsvml from the icpx runtime
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="${OUT:-results/xpu_matrix_$STAMP.log}"
mkdir -p "$(dirname "$OUT")"

CARD="${CARD:-0}"
PORT="${PORT:-8100}"
SLOTS="${SLOTS:-8}"
GEN="${GEN:-128}"
REPS="${REPS:-2}"
CTXS="${CTXS:-512,2048,8192}"
CONCS="${CONCS:-1,2,4,8}"
ARMS="${ARMS:-ship k64mlp k64all dpas}"
EXTRAS="${EXTRAS:-0}"
MODEL_NAME="knivesysl-xe-qwen3.8-27b-w4a8"
# ctx must cover prompt + gen for the deepest cell
MAXCTX=$(( $(echo "$CTXS" | tr ',' '\n' | sort -n | tail -1) + GEN + 64 ))

log() { echo "$@" | tee -a "$OUT"; }

# Port-scoped: a blanket pkill would take down an unrelated serve_xe.sh
# endpoint on the other card (that is exactly what happened once - the matrix
# and a live server can share the box, one card each).
stop() { pkill -f "[s]erve_openai_xpu.py --port $PORT" 2>/dev/null || true; sleep 4; }

start() {  # start <k64-selector> <attn-dpas> <apc-flag>
    local k64="$1" dpas="$2" apc="$3"
    stop
    local extra=""
    [ "$apc" = "off" ] && extra="--no-prefix-cache"
    TQ_XPU_ATTN_DPAS="$dpas" TQ_XPU_DEV="$CARD" \
        setsid nohup .venv/bin/python xpu/tools/serve_openai_xpu.py \
        --port "$PORT" --lib xpu/build/libforward_qwen_xpu.so \
        --tqf "$HOME/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf" \
        --model-dir "$HOME/models/knivesysl" \
        --slots "$SLOTS" --ctx "$MAXCTX" ${k64:+--k64 "$k64"} $extra \
        > "/tmp/xe_matrix_$PORT.log" 2>&1 < /dev/null &
    disown || true
    for _ in $(seq 1 90); do
        sleep 5
        curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        pgrep -f "serve_openai_xpu.py --port $PORT" >/dev/null || {
            log "  !! server died; tail:"; tail -8 "/tmp/xe_matrix_$PORT.log" | tee -a "$OUT"
            return 1; }
    done
    log "  !! server never became ready"; return 1
}

cell() {  # cell <label>
    local label="$1"
    log ""
    log "--- $label  (ctx=$CTXS conc=$CONCS gen=$GEN reps=$REPS) ---"
    python3 xpu/tools/bench_openai.py \
        --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL_NAME" \
        --ctx "$CTXS" --conc "$CONCS" --gen "$GEN" --reps "$REPS" \
        2>&1 | tee -a "$OUT"
    log "  engine widths: $(curl -s -m 3 "http://127.0.0.1:$PORT/health")"
}

log "XPU CORE MATRIX  $STAMP"
log "rules: COLD prefill (no APC), NO spec decode, ignore_eos, fixed gen"
log "host: $(hostname)  card: $CARD  slots: $SLOTS  ctx cap: $MAXCTX"
log "engine: $(cd xpu && git log -1 --format=%h 2>/dev/null || echo untracked)"

for arm in $ARMS; do
    case "$arm" in
      ship)   K64="";             DPAS=0; DESC="ship tier: W4-K32, scalar-G3 attention" ;;
      k64mlp) K64="gate,up,down"; DPAS=0; DESC="W4A4 K64 on MLP only (TF 92.2%)" ;;
      k64all) K64="all";          DPAS=0; DESC="W4A4 K64 everywhere (TF 89.9%)" ;;
      dpas)   K64="";             DPAS=1; DESC="ship tier + DPAS-matrix attention" ;;
      *) log "unknown arm $arm"; continue ;;
    esac
    log ""
    log "==================================================================="
    log "ARM $arm  -  $DESC"
    log "==================================================================="
    start "$K64" "$DPAS" off || continue
    cell "arm=$arm cold no-spec"
done

if [ "$EXTRAS" = "1" ]; then
    log ""
    log "==================================================================="
    log "LABELLED EXTRAS - NOT part of the truth matrix"
    log "==================================================================="
    start "" 0 on || true
    log ""
    log "--- APC warm-prefix run (shared prefix, second wave hits cache) ---"
    python3 xpu/tools/serve_smoke_xpu.py \
        --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL_NAME" \
        --conc "$SLOTS" --gen "$GEN" 2>&1 | tee -a "$OUT"
fi

stop
log ""
log "matrix complete -> $OUT"
