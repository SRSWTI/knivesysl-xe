#!/usr/bin/env bash
# gpu_health.sh - preflight/postflight for the B70s.
#
# WHAT THIS CHECKS, AND WHY EACH CHECK EARNS ITS PLACE.
#
#   PCI presence, driver binding, config-space read - catches a card that fell
#     off the bus or lost its driver.
#   Narrow journal patterns - only messages that appear when the device or its
#     submission path is genuinely broken: CT write failure, GPU HANG, GT
#     reset, DEVICE_LOST, wedged, Hardware Error.
#   Workqueue hog - GATING, except two task names measured as routine
#     background on this host (output_poll_execute, pci_pme_list_scan). An
#     unrecognised worker stuck before a risky run is worth stopping for.
#   Trivial kernel per card - the only load-bearing signal here. Config space
#     can read fine while GuC submission is already dead.
#
# Host facts that constrain recovery (verified on this box):
#   * Both B70s share IOMMU group 20 with Wi-Fi, 5GbE, USB XHCI and SATA - no
#     ACS on the AMD 600-series chipset switch - so VFIO passthrough would take
#     keyboard, network and disk with it. 24 members enumerated.
#   * Both expose FLR (/sys/bus/pci/devices/*/reset), so a wedged card may be
#     recoverable without a reboot. NEVER a bus-level reset: see group 20.
#   * No crash-capture armed: /sys/fs/pstore holds no records, no ramoops= on
#     the cmdline, netconsole not loaded. A hard lockup loses its final
#     buffered journal writes, so a hard cut is not diagnosable after the fact.
#     crashkernel= IS reserved, so kdump is the gap worth closing.
#
# Incident history, and the four theories tested against it and left
# unsupported, live in xpu/docs/level-up-xpu.md (host stability postmortem,
# 2026-09-05) - not here; this script is meant to stay reusable.
#
#   xpu/tools/gpu_health.sh            # check
#   xpu/tools/gpu_health.sh --since 1h # only scan recent journal
set -uo pipefail

# Scan window. A single recovered warning must NOT mark every later run
# UNHEALTHY for the rest of the boot, so postflight gates on NEW events only:
#   gpu_health.sh --mark    stamp "everything before now is already triaged"
#   gpu_health.sh           gate on events after the last --mark (else whole boot)
#   gpu_health.sh --all     force the whole boot (use once at session start)
#   gpu_health.sh --since 1h  explicit window
# Per-user runtime dir: cleared on reboot, not shared between users. A mark in
# /tmp would survive a reboot and leak another session's triage state.
RTDIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
[ -d "$RTDIR" ] || RTDIR="${TMPDIR:-/tmp}"
MARK="$RTDIR/gpu_health.mark"
BOOTID=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)
SINCE="${SINCE:-boot}"
CURSOR=""
case "${1:-}" in
    --mark)
        cur=$(journalctl -k -n0 --show-cursor --no-pager 2>/dev/null |
              sed -n 's/^-- cursor: //p')
        if [ -z "$cur" ]; then echo "could not read a journal cursor"; exit 2; fi
        printf '%s\n%s\n' "$BOOTID" "$cur" > "$MARK"
        echo "marked $(date -Is) - later runs gate on events after this cursor"
        exit 0 ;;
    --all)   : ;;
    --since) SINCE="${2:-boot}" ;;
    "")
        # A real cursor, not an mtime: mtime->--since can reach back into a
        # PREVIOUS boot. Guarded by boot id so a stale mark is ignored.
        if [ -f "$MARK" ]; then
            mb=$(sed -n 1p "$MARK" 2>/dev/null)
            mc=$(sed -n 2p "$MARK" 2>/dev/null)
            [ -n "$mc" ] && [ "$mb" = "$BOOTID" ] && CURSOR="$mc"
        fi ;;
esac
FAILS=0
CARDS=(0000:11:00.0 0000:15:00.0)
# Devices the findings actually NAME. Only these may be offered a reset - a
# workqueue-stall warning names no PCI function, and resetting a card on that
# basis is a guess that can hit the wrong one.
IMPLICATED=""

say() { printf "  %-22s %s\n" "$1" "$2"; }

echo "== B70 health =="

# 1. Are both devices present and bound to xe?
for d in "${CARDS[@]}"; do
    if [ ! -e "/sys/bus/pci/devices/$d" ]; then
        say "$d" "MISSING from PCI - host reboot needed"
        FAILS=$((FAILS+1)); IMPLICATED="$IMPLICATED $d"; continue
    fi
    drv=$(basename "$(readlink -f "/sys/bus/pci/devices/$d/driver" 2>/dev/null)" 2>/dev/null)
    if [ "$drv" != "xe" ]; then
        say "$d" "driver='${drv:-none}' (expected xe)"
        FAILS=$((FAILS+1)); IMPLICATED="$IMPLICATED $d"; continue
    fi
    # PCI config readable? A wedged device often returns all-ones.
    vid=$(cat "/sys/bus/pci/devices/$d/vendor" 2>/dev/null)
    if [ "$vid" != "0x8086" ]; then
        say "$d" "config space reads '$vid' - device is gone"
        FAILS=$((FAILS+1)); IMPLICATED="$IMPLICATED $d"; continue
    fi
    say "$d" "present, bound to xe"
done

# 1b. Is anything ALREADY using a card? This is the check whose absence cost a
#     DEVICE_LOST on 2026-09-05: the matrix harness held card 1 with a 22.6 GB
#     model and a second 22.6 GB load was launched onto the same card. 45 GB on
#     a 32 GB card. The rule was already written at the top of
#     serve_vllm_xpu.sh ("two 22 GiB model loads on one card wedged it with
#     Level-Zero DEVICE_LOST") and the tooling did not enforce it.
#
#     Only heavy, known consumers gate. Desktop processes (terminals,
#     compositors) legitimately hold render nodes just by enumerating them and
#     must not block work - see the header on false positives.
HEAVY='serve_openai_xpu|bench_prefill|bench_decode|bench_gemm|bench_batch|bench_openai|vllm|dpas_probe|tp_allreduce_probe|doorbell_probe|persistence_probe'
occupants=$(pgrep -f "$HEAVY" 2>/dev/null | grep -v "^$$\$" || true)
busy=""
if [ -n "$occupants" ]; then
    for pid in $occupants; do
        # Which card? Read the pin out of the process environment - the same
        # place serve_xe.sh and the harnesses set it.
        envf="/proc/$pid/environ"
        c=$(tr '\0' '\n' < "$envf" 2>/dev/null |
            sed -n 's/^ZE_AFFINITY_MASK=\([0-9]*\).*/\1/p' | head -1)
        [ -z "$c" ] && c=$(tr '\0' '\n' < "$envf" 2>/dev/null |
            sed -n 's/^TQ_XPU_DEV=\([0-9]*\)$/\1/p' | head -1)
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-72)
        say "in use" "card ${c:-?}  pid $pid  $cmd"
        [ -n "$c" ] && busy="$busy $c"
    done
    # Report free/busy per card rather than gating globally: blocking work on a
    # FREE card because the other one is serving is the same false-positive
    # class the journal heuristic had. Only an unknown pin gates, because then
    # we cannot prove the target card is free.
    for i in 0 1; do
        case " $busy " in
            *" $i "*) say "card $i" "BUSY - do not load a second model here" ;;
            *)        say "card $i" "free" ;;
        esac
    done
    if printf '%s\n' "$occupants" | while read -r p; do
           tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null |
               grep -qE '^(ZE_AFFINITY_MASK|TQ_XPU_DEV)=' || exit 1
       done; then :; else
        say "in use" "a process has NO card pin - target card unprovable  <-- UNHEALTHY"
        FAILS=$((FAILS+1))
    fi
else
    say "in use" "no heavy GPU process running; both cards free"
fi

# 2. Journal evidence of a REAL GPU fault. Kept deliberately narrow: patterns
#    that only ever appear when the device or its submission path is broken.
#    Routine workqueue-hog and PME chatter is NOT here - measured counts show
#    it anti-correlates with actual failures (see header).
if command -v journalctl >/dev/null 2>&1 || [ -n "${GPU_HEALTH_LOG_FILE:-}" ]; then
    # -b is ALWAYS the outer bound, so no window can reach into a previous
    # boot; --since narrows within this boot, --after-cursor narrows to events
    # newer than a triage mark taken during THIS boot.
    JARGS=(-b)
    [ "$SINCE" != "boot" ] && JARGS=(-b --since "-$SINCE")
    [ -n "$CURSOR" ] && JARGS=(-b --after-cursor "$CURSOR")
    # Test hook: feed synthetic records to exercise the matcher and the gating
    # verdict without manufacturing a real kernel warning. Never set in normal
    # use; the window flags above are ignored when it is.
    if [ -n "${GPU_HEALTH_LOG_FILE:-}" ]; then
        log=$(cat "$GPU_HEALTH_LOG_FILE" 2>/dev/null || true)
        say "journal" "TEST HOOK: reading $GPU_HEALTH_LOG_FILE (not the journal)"
    else
        log=$(journalctl "${JARGS[@]}" -k --no-pager 2>/dev/null || true)
    fi
    for pat in "CT write: non-zero status" \
               "Completion-Wait loop timed out" \
               "GPU HANG" \
               "GT[0-9]*: reset" \
               "DEVICE_LOST" \
               "wedged" \
               "Hardware Error"; do
        n=$(printf '%s\n' "$log" | grep -ciE "$pat" || true)
        if [ "${n:-0}" -gt 0 ]; then
            say "journal" "$n x '$pat'  <-- UNHEALTHY"
            FAILS=$((FAILS+1))
        fi
    done
    # Workqueue hog. Two task names are treated as known background noise on
    # this host; their counts anti-correlate with real failures (postmortem
    # doc has the per-boot table). Everything else GATES - a stuck worker we
    # do not recognise is exactly what we want to catch before piling on work.
    hogs=$(printf '%s\n' "$log" | grep -iE "hogged CPU" || true)
    kn=0
    if [ -n "$hogs" ]; then
        kn=$(printf '%s\n' "$hogs" | grep -cE "output_poll_execute|pci_pme_list_scan" || true)
        other=$(printf '%s\n' "$hogs" | grep -vE "output_poll_execute|pci_pme_list_scan" || true)
        if [ -n "$other" ]; then
            n=$(printf '%s\n' "$other" | grep -c . || true)
            nm=$(printf '%s\n' "$other" | grep -oE "workqueue: [a-zA-Z0-9_]+" \
                 | sed 's/workqueue: //' | sort -u | tr '\n' ' ')
            say "journal" "$n x 'hogged CPU' on ${nm:-unknown}  <-- UNHEALTHY"
            FAILS=$((FAILS+1))
        fi
    fi
    [ "$FAILS" -eq 0 ] && say "journal" "clean (no CT/hang/reset/wedge/HW-error/unknown-hog)"
    [ "${kn:-0}" -gt 0 ] && say "noise" "$kn x 'hogged CPU' on known background workers"
    n=$(printf '%s\n' "$log" | grep -ciE "Spurious native interrupt" || true)
    [ "${n:-0}" -gt 0 ] && say "noise" "$n x 'Spurious native interrupt' (background)"
else
    say "journal" "journalctl unavailable - skipped"
fi

# 3. Does a trivial kernel still run on each card? This is the real liveness
#    test; config space can look fine while GuC submission is dead.
if command -v icpx >/dev/null 2>&1; then
    tmp=$(mktemp -d)
    cat > "$tmp/live.cpp" <<'EOF'
#include <sycl/sycl.hpp>
#include <cstdio>
int main(int argc, char **argv) {
    const int want = argc > 1 ? std::atoi(argv[1]) : 0;
    std::vector<sycl::device> g;
    for (auto &d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) g.push_back(d);
    if (want >= (int)g.size()) { std::printf("no such device\n"); return 2; }
    sycl::queue q(g[want], sycl::property::queue::in_order{});
    int *p = sycl::malloc_device<int>(64, q);
    if (!p) { std::printf("alloc failed\n"); return 3; }
    q.parallel_for(sycl::nd_range<1>(64, 64), [=](sycl::nd_item<1> it)
        [[sycl::reqd_sub_group_size(16)]] {
        p[it.get_global_linear_id()] = (int)it.get_global_linear_id();
    }).wait();
    int h[64];
    q.memcpy(h, p, sizeof h).wait();
    sycl::free(p, q);
    return h[63] == 63 ? 0 : 4;
}
EOF
    if icpx -fsycl -O1 -o "$tmp/live" "$tmp/live.cpp" >/dev/null 2>&1; then
        for i in 0 1; do
            if timeout 30 "$tmp/live" "$i" >/dev/null 2>&1; then
                say "device $i liveness" "PASS (trivial kernel completed)"
            else
                say "device $i liveness" "FAIL / TIMEOUT  <-- do not launch work"
                FAILS=$((FAILS+1))
            fi
        done
    else
        say "liveness" "could not build probe - skipped"
    fi
    rm -rf "$tmp"
else
    say "liveness" "icpx not on PATH (source setvars.sh) - skipped"
fi

echo
if [ "$FAILS" -eq 0 ]; then
    echo "HEALTHY - safe to run"
    exit 0
fi
echo "UNHEALTHY ($FAILS finding(s)) - stop and investigate before more work"
echo
echo "  1. INVESTIGATE first. The findings above are the whole evidence."
echo "     A workqueue-stall warning names no PCI function, and cards passing"
echo "     liveness means RECOVERED - not that the event was benign. Identify"
echo "     the faulting path before touching hardware."
n=1
for d in $IMPLICATED; do
    rmth=$(cat "/sys/bus/pci/devices/$d/reset_method" 2>/dev/null)
    n=$((n+1))
    case "$rmth" in
        *flr*)
            echo "  $n. $d IS named in the findings and reset_method='$rmth':"
            echo "       sudo sh -c 'echo 1 > /sys/bus/pci/devices/$d/reset'"
            echo "     Function-level only. NEVER a bus reset - IOMMU group 20"
            echo "     holds the USB, SATA and network controllers." ;;
        *)
            echo "  $n. $d is named in the findings, but reset_method="
            echo "     '${rmth:-unknown}' does not advertise flr - do not reset it." ;;
    esac
done
echo "  $((n+1)). sudo modprobe -r xe && sudo modprobe xe   (releases both cards)"
echo "  $((n+2)). reboot"
echo
echo "Once a finding is triaged: gpu_health.sh --mark, then re-run to gate on new events."
exit 1
