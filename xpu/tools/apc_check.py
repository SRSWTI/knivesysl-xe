#!/usr/bin/env python3
"""apc_check.py - prefix-checkpoint (APC) correctness + value gate.

Arm A (cold): prefill P tokens on slot 0, then decode the tail and generate.
Arm B (APC):  checkpoint slot 0 at position P, restore into slot 1, then run
              the SAME tail + generation there.
The two streams must be byte-identical: a restored checkpoint has to leave a
slot indistinguishable from having prefilled those tokens (KV prefix + the
48 GDN layers' conv/recurrent state). Also reports save/restore wall time
against the re-prefill it replaces - the actual APC payoff.

TQ_XPU_DEV=0 TQ_XPU_SLOTS=2 python3 xpu/tools/apc_check.py \
    --lib xpu/build/libforward_qwen_xpu.so --tqf ~/models/.../*.tqf
"""
import argparse
import ctypes
import os
import sys
import time

TOK = 9707
TAIL = 24     # prompt tokens past the checkpoint boundary
GEN = 32      # greedy tokens to compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--prefix", type=int, default=512)
    args = ap.parse_args()

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_set_slot.argtypes = [ctypes.c_int]
    lib.qwn_prefill_chunk.argtypes = [ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_int, ctypes.c_int]
    lib.qwn_host_alloc.argtypes = [ctypes.c_size_t]
    lib.qwn_host_alloc.restype = ctypes.c_void_p
    lib.qwn_host_free.argtypes = [ctypes.c_void_p]
    lib.qwn_ckpt_bytes.argtypes = [ctypes.c_int]
    lib.qwn_ckpt_bytes.restype = ctypes.c_size_t
    lib.qwn_ckpt_save.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.qwn_ckpt_restore.argtypes = [ctypes.c_int, ctypes.c_int,
                                     ctypes.c_void_p]
    assert lib.qwn_init(os.path.expanduser(args.tqf).encode()) == 0
    if lib.qwn_num_slots() < 2:
        print("need TQ_XPU_SLOTS>=2", file=sys.stderr)
        return 1

    P = (args.prefix // 8) * 8
    prompt = [TOK + (i % 5) for i in range(P)]
    tail = [TOK + 1 + (i % 3) for i in range(TAIL)]
    IntP = ctypes.c_int * P

    # ---- arm A: cold prefill on slot 0
    assert lib.qwn_set_slot(0) == 0
    t0 = time.perf_counter()
    rc = lib.qwn_prefill_chunk(IntP(*prompt), P, 0)
    prefill_s = time.perf_counter() - t0
    assert rc == 0, f"prefill rc={rc}"

    # checkpoint here, before arm A mutates slot 0 past P
    nbytes = lib.qwn_ckpt_bytes(P)
    blob = lib.qwn_host_alloc(nbytes)
    assert blob, "host alloc failed"
    t0 = time.perf_counter()
    rc = lib.qwn_ckpt_save(0, P, blob)
    save_s = time.perf_counter() - t0
    assert rc == 0, f"ckpt_save rc={rc}"

    def run_tail(slot, pos):
        assert lib.qwn_set_slot(slot) == 0
        nxt = -1
        for t in tail:
            nxt = lib.qwn_decode(t, pos)
            assert nxt >= 0, f"tail rc={nxt}"
            pos += 1
        out = [nxt]
        for _ in range(GEN - 1):
            nxt = lib.qwn_decode(nxt, pos)
            assert nxt >= 0
            out.append(nxt)
            pos += 1
        return out

    stream_a = run_tail(0, P)

    # ---- arm B: restore into slot 1, same tail
    t0 = time.perf_counter()
    rc = lib.qwn_ckpt_restore(1, P, blob)
    restore_s = time.perf_counter() - t0
    assert rc == 0, f"ckpt_restore rc={rc}"
    stream_b = run_tail(1, P)

    lib.qwn_host_free(blob)
    lib.qwn_free()

    match = stream_a == stream_b
    agree = sum(1 for x, y in zip(stream_a, stream_b) if x == y)
    mb = nbytes / 1e6
    print(f"checkpoint      {mb:.1f} MB at pos {P}")
    print(f"save            {save_s*1e3:.1f} ms  ({mb/save_s/1e3:.2f} GB/s)")
    print(f"restore         {restore_s*1e3:.1f} ms  ({mb/restore_s/1e3:.2f} GB/s)")
    print(f"re-prefill      {prefill_s*1e3:.1f} ms  "
          f"-> APC is {prefill_s/restore_s:.0f}x cheaper")
    print(f"continuation    {'MATCH' if match else 'MISMATCH'} ({agree}/{GEN})")
    print("OVERALL:", "PASS" if match else "FAIL")
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
