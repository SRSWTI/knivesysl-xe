#!/usr/bin/env python3
"""batch_check.py - batched decode step vs serial single-slot equality.

Prefills two prompts on slots 0/1, then generates via qwn_decode_batch
(both rows in ONE step). Streams must byte-match fresh serial single-slot
runs - proves the batched step (batched RC8 GEMMs + per-row state kernels)
is numerically the same engine.

TQ_XPU_DEV=0 TQ_XPU_SLOTS=2 python3 xpu/tools/batch_check.py \
    --lib xpu/build/libforward_qwen_xpu.so --tqf ~/models/.../*.tqf
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys

PROMPT_A = [151644, 872, 198, 9707, 11, 1246, 525, 498, 30, 151645, 198]
PROMPT_B = [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198]
STEPS = 48


def run_single(lib_path, tqf, prompt):
    code = (
        "import ctypes, json\n"
        f"lib = ctypes.CDLL({lib_path!r})\n"
        "lib.qwn_init.argtypes = [ctypes.c_char_p]\n"
        "lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]\n"
        f"assert lib.qwn_init({tqf!r}.encode()) == 0\n"
        f"prompt = {prompt!r}\n"
        "nxt = -1\n"
        "for pos, tok in enumerate(prompt):\n"
        "    nxt = lib.qwn_decode(tok, pos); assert nxt >= 0\n"
        "out = [nxt]\n"
        f"for i in range({STEPS} - 1):\n"
        "    nxt = lib.qwn_decode(nxt, len(prompt) + i); assert nxt >= 0\n"
        "    out.append(nxt)\n"
        "print(json.dumps(out))\n"
    )
    env = dict(os.environ)
    env["TQ_XPU_SLOTS"] = "1"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    args = ap.parse_args()
    lib_path = os.path.abspath(args.lib)
    tqf = os.path.expanduser(args.tqf)

    lib = ctypes.CDLL(lib_path)
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_set_slot.argtypes = [ctypes.c_int]
    lib.qwn_decode_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [
        ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    assert lib.qwn_init(tqf.encode()) == 0
    assert lib.qwn_num_slots() >= 2, "need TQ_XPU_SLOTS>=2"

    prompts = {0: PROMPT_A, 1: PROMPT_B}
    last, pos = {}, {}
    for s in (0, 1):  # serial per-slot prompt feed
        assert lib.qwn_set_slot(s) == 0
        nxt = -1
        for p, tok in enumerate(prompts[s]):
            nxt = lib.qwn_decode(tok, p)
            assert nxt >= 0, f"feed slot {s} pos {p} rc={nxt}"
        last[s], pos[s] = nxt, len(prompts[s])

    streams = {0: [last[0]], 1: [last[1]]}
    IntArr = ctypes.c_int * 2
    for _ in range(STEPS - 1):
        slots = IntArr(0, 1)
        toks = IntArr(last[0], last[1])
        poss = IntArr(pos[0], pos[1])
        out = IntArr(0, 0)
        rc = lib.qwn_decode_batch(slots, toks, poss, 2, out)
        assert rc == 2, f"batch rc={rc}"
        for s in (0, 1):
            last[s] = out[s]
            streams[s].append(out[s])
            pos[s] += 1
    lib.qwn_free()

    ref = {0: run_single(lib_path, tqf, PROMPT_A),
           1: run_single(lib_path, tqf, PROMPT_B)}
    ok = True
    for s in (0, 1):
        match = streams[s] == ref[s]
        agree = sum(1 for x, y in zip(streams[s], ref[s]) if x == y)
        print(f"slot{s} batch vs serial: "
              f"{'MATCH' if match else 'MISMATCH'} ({agree}/{STEPS})")
        ok = ok and match
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
