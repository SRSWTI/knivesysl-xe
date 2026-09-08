#!/usr/bin/env bash
# Serve knivesysl-xe over an OpenAI-compatible endpoint on ONE Arc Pro B70.
# Mirrors serve_vllm_xpu.sh (which serves the VENDOR stack) so the two can be
# driven by the same client for head-to-head runs.
#
#   xpu/tools/serve_xe.sh                    # card 0, :8100, 8 slots
#   CARD=1 PORT=8101 xpu/tools/serve_xe.sh   # the other B70
#   K64=all xpu/tools/serve_xe.sh            # opt-in W4A4 prefill tier
#   PAGED=1 KV_POOL_TOKENS=131072 CTX=131072 xpu/tools/serve_xe.sh
#   PAGED=1 KV_POOL_MB=4032 KV_PAGE=128 xpu/tools/serve_xe.sh
#
# Detaches via setsid so it outlives the launching shell. Log: $LOG.
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root

CARD="${CARD:-0}"
PORT="${PORT:-8100}"
# Loopback by default: the endpoint has no authentication, and every client we
# drive it with (axe_vllm.py, bench_openai.py) is local. Set HOST=0.0.0.0
# deliberately if you ever want it on the LAN.
HOST="${HOST:-127.0.0.1}"
# Slot/context budget. Weights are ~22.6 GB resident on a 32 GB card, leaving
# ~9.4 GB. Each slot costs ctx x 33 KB of KV plus a FIXED 159 MB of GDN
# recurrent state, so slots and context trade directly:
#   8 x 32768 = 9.9 GB  (over budget)
#   8 x 16384 = 5.6 GB
#   4 x 32768 = 5.0 GB  <- default: an interactive client uses one slot at a
#   2 x 65536 = 4.6 GB     time, so context is worth more than idle slots
#   1 x 131072 = 4.5 GB
SLOTS="${SLOTS:-4}"
CTX="${CTX:-32768}"
# Paging stays explicitly off unless PAGED=1. A paged launch must choose one
# physical KV-only budget; it never derives capacity from SLOTS*CTX.
PAGED="${PAGED:-0}"
KV_PAGE="${KV_PAGE:-128}"
KV_POOL_TOKENS="${KV_POOL_TOKENS:-}"
KV_POOL_MB="${KV_POOL_MB:-}"
RESPONSE_IDLE_TIMEOUT="${RESPONSE_IDLE_TIMEOUT:-300}"
# Auto selects native prefill dispatch; 1 forces XMX, 0 the scalar reference.
PREFILL_XMX="${PREFILL_XMX:-${TQ_XPU_PREFILL_XMX:-auto}}"
PACKED_PREFILL="${PACKED_PREFILL:-${TQ_XPU_PACKED_PREFILL:-1}}"
CHUNK="${CHUNK:-64}"
CHUNK_IDLE="${CHUNK_IDLE:-512}"
PREFILL_SLICE_MS="${PREFILL_SLICE_MS:-0}"
# k64mlp is the DEFAULT tier as of 2026-09-05: one FP16 scale per K64 tile on
# the MLP trio only. Measured +30% prefill (460 -> 597 tok/s at T=512,
# in-process, paired same-session) while clearing the 0.90 teacher-forced
# agreement gate at 92.2%. K64=all is +52% but fails that gate at 89.9%;
# K64="" restores the certified W4-K32 ship tier.
K64="${K64:-gate,up,down}"
MODEL="${MODEL:-$HOME/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/knivesysl}"
LIB="${LIB:-xpu/build/libforward_qwen_xpu.so}"
LOG="${LOG:-/tmp/xe_serve_$PORT.log}"
PY="${PY:-.venv/bin/python}"        # needs transformers for the chat template

case "$PREFILL_XMX" in
    auto|0|1) ;;
    *) echo "PREFILL_XMX must be auto, 0 or 1" >&2; exit 1 ;;
esac
case "$PACKED_PREFILL" in
    0) PREFILL_ARGS=(--no-packed-prefill) ;;
    1) PREFILL_ARGS=(--packed-prefill) ;;
    *) echo "PACKED_PREFILL must be 0 or 1" >&2; exit 1 ;;
esac

case "$PAGED" in
    0) PAGING_ARGS=(--no-paged) ;;
    1)
        case "$KV_PAGE" in
            128|256) ;;
            *) echo "KV_PAGE must be 128 or 256" >&2; exit 1 ;;
        esac
        if [[ -n "$KV_POOL_TOKENS" && -n "$KV_POOL_MB" ]] ||
           [[ -z "$KV_POOL_TOKENS" && -z "$KV_POOL_MB" ]]; then
            echo "PAGED=1 requires exactly one of KV_POOL_TOKENS or KV_POOL_MB" >&2
            exit 1
        fi
        PAGING_ARGS=(--paged --kv-page "$KV_PAGE")
        if [[ -n "$KV_POOL_TOKENS" ]]; then
            PAGING_ARGS+=(--kv-pool-tokens "$KV_POOL_TOKENS")
        else
            PAGING_ARGS+=(--kv-pool-mb "$KV_POOL_MB")
        fi
        ;;
    *) echo "PAGED must be 0 or 1" >&2; exit 1 ;;
esac

[ -f "$MODEL" ]  || { echo "no model at $MODEL" >&2; exit 1; }
[ -f "$LIB" ]    || { echo "no engine at $LIB (run xpu/build.sh)" >&2; exit 1; }
[ -x "$PY" ]     || { echo "no python at $PY" >&2; exit 1; }

# Both B70s are safe compute targets: the display runs on the NVIDIA 5090
# (card1 HDMI connected); card2/card3 are the B70s with no connected outputs.
# ZE_AFFINITY_MASK/TQ_XPU_DEV index Level Zero devices, not DRM nodes.
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u

if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "already serving on :$PORT"; exit 0
fi

echo "[serve_xe] card=$CARD port=$PORT slots=$SLOTS ctx=$CTX k64=${K64:-off} paged=$PAGED"
echo "[serve_xe] prefill_xmx=$PREFILL_XMX packed_prefill=$PACKED_PREFILL chunk=$CHUNK chunk_idle=$CHUNK_IDLE prefill_slice_ms=$PREFILL_SLICE_MS"
echo "[serve_xe] log -> $LOG"
# ZE_AFFINITY_MASK makes the OTHER B70 invisible to Level Zero, so this process
# physically cannot touch it and the driver skips peer/P2P evaluation between
# the two cards entirely (that path logs "ACS redirect is set between the
# client and provider" even on a single-card load). serve_vllm_xpu.sh carries
# the same rule. TQ_XPU_DEV then selects rank 0 within the masked-down list.
ZE_AFFINITY_MASK="$CARD" TQ_XPU_DEV=0 setsid nohup "$PY" xpu/tools/serve_openai_xpu.py \
    --host "$HOST" --port "$PORT" --lib "$LIB" --tqf "$MODEL" --model-dir "$MODEL_DIR" \
    --slots "$SLOTS" --ctx "$CTX" --k64 "$K64" \
    --prefill-xmx "$PREFILL_XMX" "${PREFILL_ARGS[@]}" \
    --chunk "$CHUNK" --chunk-idle "$CHUNK_IDLE" \
    --prefill-slice-ms "$PREFILL_SLICE_MS" \
    --response-idle-timeout "$RESPONSE_IDLE_TIMEOUT" "${PAGING_ARGS[@]}" \
    > "$LOG" 2>&1 < /dev/null &
disown || true

for _ in $(seq 1 60); do
    sleep 5
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[serve_xe] READY on :$PORT"
        curl -s -m 3 "http://127.0.0.1:$PORT/health"; echo
        exit 0
    fi
    # Match on the script name only: the arg order changed once and a stricter
    # pattern silently reported "died" while the server was loading fine.
    if ! pgrep -f "serve_openai_xpu.py" >/dev/null; then
        echo "[serve_xe] died during load; tail of $LOG:" >&2
        tail -15 "$LOG" >&2; exit 1
    fi
done
echo "[serve_xe] timed out waiting for readiness; see $LOG" >&2
exit 1
