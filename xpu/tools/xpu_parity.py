#!/usr/bin/env python3
"""xpu_parity.py — Phase 1 gate: XPU engine vs CUDA engine, same TQF.

Two-step protocol (separate processes so CUDA and SYCL never share one):
  1. dump reference with the CUDA lib on the 5090:
       python3 xpu/tools/xpu_parity.py dump --lib build-qwen/libforward_qwen.so \
           --tqf ~/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf --out /tmp/ref.npz
  2. compare the XPU lib against it:
       python3 xpu/tools/xpu_parity.py check --lib xpu/build/libforward_qwen_xpu.so \
           --tqf ~/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf --ref /tmp/ref.npz

What is compared (deterministic token ids, no tokenizer needed):
  - debug_embed_input_norm(tok, layer 0)                  -> cos vs ref
  - debug_decode_layers(tok, pos=0, n) for n in LADDER    -> cos vs ref
  - final_norm(tok)                                       -> cos
  - greedy decode chain from a fixed prefix               -> argmax agreement
Expected difference source: the CUDA GEMV quantizes activations to E4M3
(pow2 128-blocks) inside the SF path; the XPU Phase-1 GEMV runs fp32
activations. Gates are therefore cosine/agreement bands, not bit equality:
  PASS: cos >= 0.99 per ladder rung, argmax agreement >= 90% over the chain.
Run the CUDA dump with TQ_ATTN_MMA=0 (scalar attention tails) to minimize
band width.
"""
import argparse, ctypes, os, sys
import numpy as np

LADDER = [1, 2, 4, 8, 16, 32, 48, 64]
PROBE_TOK = 17                                    # fixed probe token id
CHAIN_PREFIX = [151644, 872, 198, 9707, 11, 1246, 525, 498, 30, 151645]
GEN = 64


def load(libpath, tqf):
    # This gate certifies the scalar Phase-1 engine against CUDA. The integer
    # DPAS tier has its own gate (xpu_tf_check.py); never let the loader
    # default silently swap the engine under this comparison.
    os.environ["TQ_XPU_W4A8"] = "0"
    lib = ctypes.CDLL(os.path.abspath(libpath), mode=ctypes.RTLD_LOCAL)
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_debug_decode_layers.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.qwn_debug_embed_input_norm.argtypes = [ctypes.c_int, ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.qwn_debug_forward_final_norm.argtypes = [ctypes.c_int,
                                                 ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.qwn_last_argmax_logit.restype = ctypes.c_float
    rc = lib.qwn_init(tqf.encode())
    if rc != 0:
        print(f"qwn_init failed rc={rc}", file=sys.stderr)
        sys.exit(1)
    return lib


def fvec(fn, *args, n=8192):
    buf = (ctypes.c_float * n)()
    rc = fn(*args, buf, n)
    if rc < 0:
        raise RuntimeError(f"debug call rc={rc}")
    return np.frombuffer(buf, dtype=np.float32, count=rc).copy()


def collect(lib):
    H = lib.qwn_hidden_size()
    L = lib.qwn_num_layers()
    out = {}
    out["embed_norm"] = fvec(lib.qwn_debug_embed_input_norm, PROBE_TOK, 0, n=H)
    for n in LADDER:
        if n > L:
            break
        print(f"  [collect] layers_{n}", file=sys.stderr)
        out[f"layers_{n:02d}"] = fvec(lib.qwn_debug_decode_layers, PROBE_TOK, 0, n, n=H)
    out["final_norm"] = fvec(lib.qwn_debug_forward_final_norm, PROBE_TOK, n=H)
    chain = []
    pos = 0
    nxt = None
    for t in CHAIN_PREFIX:
        nxt = lib.qwn_decode(t, pos)
        if nxt < 0:
            raise RuntimeError(f"qwn_decode rc={nxt} at pos {pos}")
        pos += 1
    for _ in range(GEN):
        chain.append(nxt)
        nxt = lib.qwn_decode(nxt, pos)
        if nxt < 0:
            raise RuntimeError(f"qwn_decode rc={nxt} at pos {pos}")
        pos += 1
    out["chain"] = np.array(chain, dtype=np.int64)
    return out


def cos(a, b):
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def load_final_norm_weight(tqf):
    """Read the final-norm bf16 weights straight from the TQF container.

    Layout (tools/convert_qwen_tqf.py): "TQF1" | u32 header_bytes |
    <16I 3f 8I> | u8 layer_types[L] | embed bf16[V*H] | norm bf16[H] | ...
    """
    with open(tqf, "rb") as f:
        assert f.read(4) == b"TQF1"
        hdr_bytes = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        u = np.frombuffer(f.read(64), dtype=np.uint32)
        f.read(12)  # 3 floats
        f.read(32)  # 8 tail uints
        H, L, V = int(u[3]), int(u[5]), int(u[6])
        f.read(L)   # layer_types
        assert 8 + 64 + 12 + 32 + L == hdr_bytes
        f.seek(2 * V * H, 1)  # skip embed
        w = np.frombuffer(f.read(2 * H), dtype=np.uint16)
    return (w.astype(np.uint32) << 16).view(np.float32), H


def qwen_rmsnorm(x, w_bf16_as_f32, eps=1e-6):
    """(1+w) RMSNorm, matching k_tq_qwen_rmsnorm (forward_qwen.cu:4567)."""
    x = x.astype(np.float64)
    inv = 1.0 / np.sqrt((x * x).mean() + eps)
    return (x * inv * (1.0 + w_bf16_as_f32.astype(np.float64))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "check"])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--out")
    ap.add_argument("--ref")
    ap.add_argument("--gen", type=int, default=GEN)
    args = ap.parse_args()

    lib = load(args.lib, args.tqf)
    got = collect(lib)

    if args.mode == "dump":
        np.savez(args.out, **got)
        print(f"[dump] wrote {args.out}: " + ", ".join(sorted(got)))
        return

    ref = np.load(args.ref)
    # CUDA's qwn_debug_forward_final_norm runs the FIRST-TOKEN forward path
    # (k_tq_linear_first_core_gated / k_tq_full_attn_first_token), a different
    # kernel family from decode-at-0 — comparing it against the XPU decode-path
    # value is apples-to-oranges. Recompute the reference analytically:
    # final_norm := (1+w)-rmsnorm(ref[layers_64], w = TQF final-norm weights).
    ref = {k: np.asarray(ref[k]) for k in ref.files}
    deepest = max(k for k in ref if k.startswith("layers_"))
    norm_w, _H = load_final_norm_weight(args.tqf)
    ref["final_norm"] = qwen_rmsnorm(ref[deepest], norm_w)
    fails = 0
    print(f"{'probe':<16} {'cos':>10}  gate")
    for k in sorted(got):
        if k == "chain":
            continue
        if k not in ref:
            print(f"{k:<16} {'':>10}  MISSING in ref")
            continue
        c = cos(got[k], ref[k])
        ok = c >= 0.99
        fails += not ok
        note = "(analytic ref)" if k == "final_norm" else ""
        print(f"{k:<16} {c:>10.6f}  {'PASS' if ok else 'FAIL'}  {note}")
    a, b = got["chain"], np.asarray(ref["chain"])
    n = min(len(a), len(b))
    agree = (a[:n] == b[:n])
    rate = float(agree.mean())
    first_div = int(np.argmin(agree)) if not agree.all() else -1
    ok = rate >= 0.90
    fails += not ok
    print(f"{'argmax chain':<16} {rate*100:>9.1f}%  {'PASS' if ok else 'FAIL'}"
          f"  (n={n}, first divergence at {first_div})")
    print("OVERALL:", "PASS" if fails == 0 else f"FAIL ({fails})")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
