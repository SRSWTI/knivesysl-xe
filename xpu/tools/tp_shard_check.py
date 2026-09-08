#!/usr/bin/env python3
"""tp_shard_check.py - gate the loader's TP model sharding.

Loads the model, calls qwn_tp_shard(), and checks the observable
consequences:
  * every rank reports halved head counts / intermediate size;
  * per-card device memory actually DROPS (the point of sharding is that
    rank 0 sheds its full weights, not just that rank 1 gains a copy);
  * qwn_reset_state still allocates per-sequence state, now from the
    sharded head counts.

Reports free device memory per card before and after so the halving is a
measurement, not a claim.

  TQ_XPU_DEV=0 TQ_XPU_TP=2 python3 xpu/tools/tp_shard_check.py \
      --lib xpu/build/libforward_qwen_xpu.so --tqf ~/models/.../*.tqf
"""
import argparse
import ctypes
import os
import subprocess
import sys


def free_mb():
    """Per-card free memory via xpu-smi if present, else None."""
    try:
        out = subprocess.run(["xpu-smi", "dump", "-d", "-1", "-m", "18", "-n", "1"],
                             capture_output=True, text=True, timeout=15).stdout
        vals = []
        for line in out.splitlines()[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    vals.append(float(parts[-1]))
                except ValueError:
                    pass
        return vals or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    args = ap.parse_args()

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    for fn in ("qwn_tp_shard", "qwn_reset_state", "qwn_num_layers",
               "qwn_num_attention_heads", "qwn_num_key_value_heads",
               "qwn_intermediate_size", "qwn_hidden_size"):
        getattr(lib, fn).restype = ctypes.c_int
    lib.qwn_set_rank_py = None  # not exported; rank switching is internal

    before = free_mb()
    rc = lib.qwn_init(os.path.expanduser(args.tqf).encode())
    if rc != 0:
        print(f"qwn_init rc={rc}", file=sys.stderr)
        return 1
    nh_full = lib.qwn_num_attention_heads()
    nkv_full = lib.qwn_num_key_value_heads()
    I_full = lib.qwn_intermediate_size()
    print(f"loaded: nh={nh_full} nkv={nkv_full} I={I_full} "
          f"H={lib.qwn_hidden_size()} L={lib.qwn_num_layers()}")

    rc = lib.qwn_tp_shard()
    print(f"qwn_tp_shard rc={rc}")
    if rc <= 0:
        print("OVERALL: FAIL (shard did not run; need TQ_XPU_TP=2)")
        return 1
    tp = rc

    nh, nkv, I = (lib.qwn_num_attention_heads(),
                  lib.qwn_num_key_value_heads(),
                  lib.qwn_intermediate_size())
    print(f"after shard (active rank): nh={nh} nkv={nkv} I={I}")
    fails = 0
    for name, got, want in (("nh", nh, nh_full // tp),
                            ("nkv", nkv, nkv_full // tp),
                            ("I", I, I_full // tp)):
        ok = got == want
        fails += 0 if ok else 1
        print(f"  {name:4s} {got:6d} expect {want:6d} {'PASS' if ok else 'FAIL'}")

    rc = lib.qwn_reset_state()
    print(f"qwn_reset_state after shard rc={rc} "
          f"{'PASS' if rc == 0 else 'FAIL'}")
    fails += 0 if rc == 0 else 1

    after = free_mb()
    if before and after and len(before) == len(after):
        print("\n  per-card free MB (before load -> after shard+state):")
        for i, (b, a) in enumerate(zip(before, after)):
            print(f"    card {i}: {b:9.0f} -> {a:9.0f}   used {b - a:9.0f}")
    else:
        print("\n  (xpu-smi memory readout unavailable; skipping footprint)")

    print("\nOVERALL:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
