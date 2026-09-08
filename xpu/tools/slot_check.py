#!/usr/bin/env python3
"""slot_check.py - multi-slot state isolation gate.

Runs two different prompts INTERLEAVED per token on slots 0 and 1
(TQ_XPU_SLOTS=2), then replays each prompt on a fresh single-slot engine in
a subprocess. Greedy streams must match token-for-token: any cross-slot
state bleed (KV, conv, recurrent, state_pos) breaks equality immediately.

TQ_XPU_DEV=0 TQ_XPU_SLOTS=2 python3 xpu/tools/slot_check.py \
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
    """Fresh process, one slot, plain greedy - the reference stream."""
    code = (
        "import ctypes, json, sys\n"
        f"lib = ctypes.CDLL({lib_path!r})\n"
        "lib.qwn_init.argtypes = [ctypes.c_char_p]\n"
        "lib.qwn_init.restype = ctypes.c_int\n"
        "lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]\n"
        "lib.qwn_decode.restype = ctypes.c_int\n"
        f"assert lib.qwn_init({tqf!r}.encode()) == 0\n"
        f"prompt = {prompt!r}\n"
        "nxt = -1\n"
        "for pos, tok in enumerate(prompt):\n"
        "    nxt = lib.qwn_decode(tok, pos)\n"
        "    assert nxt >= 0\n"
        "out = [nxt]\n"
        f"for i in range({STEPS} - 1):\n"
        "    nxt = lib.qwn_decode(nxt, len(prompt) + i)\n"
        "    assert nxt >= 0\n"
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
    lib.qwn_init.restype = ctypes.c_int
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_decode.restype = ctypes.c_int
    lib.qwn_set_slot.argtypes = [ctypes.c_int]
    lib.qwn_set_slot.restype = ctypes.c_int
    lib.qwn_num_slots.restype = ctypes.c_int
    assert lib.qwn_init(tqf.encode()) == 0
    ns = lib.qwn_num_slots()
    if ns < 2:
        print(f"need TQ_XPU_SLOTS>=2 (got {ns})", file=sys.stderr)
        return 1

    # interleaved: one token on slot 0, one on slot 1, alternating.
    streams = {0: [], 1: []}
    prompts = {0: list(PROMPT_A), 1: list(PROMPT_B)}
    nxt = {0: None, 1: None}
    pos = {0: 0, 1: 0}
    # feed prompts interleaved
    while any(pos[s] < len(prompts[s]) for s in (0, 1)):
        for s in (0, 1):
            if pos[s] < len(prompts[s]):
                assert lib.qwn_set_slot(s) == 0
                nxt[s] = lib.qwn_decode(prompts[s][pos[s]], pos[s])
                if nxt[s] < 0:
                    print(f"prompt feed rc={nxt[s]} slot={s} pos={pos[s]}")
                    return 1
                pos[s] += 1
    for s in (0, 1):
        streams[s].append(nxt[s])
    for _ in range(STEPS - 1):
        for s in (0, 1):
            assert lib.qwn_set_slot(s) == 0
            nxt[s] = lib.qwn_decode(nxt[s], pos[s])
            if nxt[s] < 0:
                print(f"gen rc={nxt[s]} slot={s} pos={pos[s]}")
                return 1
            streams[s].append(nxt[s])
            pos[s] += 1
    lib.qwn_free()

    ref_a = run_single(lib_path, tqf, PROMPT_A)
    ref_b = run_single(lib_path, tqf, PROMPT_B)
    ok_a = streams[0] == ref_a
    ok_b = streams[1] == ref_b
    print(f"slot0 vs fresh: {'MATCH' if ok_a else 'MISMATCH'} "
          f"({sum(1 for x, y in zip(streams[0], ref_a) if x == y)}/{STEPS})")
    print(f"slot1 vs fresh: {'MATCH' if ok_b else 'MISMATCH'} "
          f"({sum(1 for x, y in zip(streams[1], ref_b) if x == y)}/{STEPS})")
    print("OVERALL:", "PASS" if (ok_a and ok_b) else "FAIL")
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    sys.exit(main())
