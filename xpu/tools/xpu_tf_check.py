#!/usr/bin/env python3
"""xpu_tf_check.py — teacher-forced argmax agreement: XPU engine vs a CUDA
`TF <ids...>` line produced by tools/tf_agreement.py on the pinned corpus.

The CUDA side prefilled with its chunked path; the XPU Phase-1 engine has no
prefill API, so the prompt is fed with per-token qwn_decode (teacher forcing
from position 0 — same math, different batching). Gate mirrors the repo's
convention: report the agreement rate and first divergence; the engine's own
two-config eps band on this corpus is ~97-98%, so PASS >= 0.90 stays honest.

Usage:
  .venv/bin/python xpu/tools/xpu_tf_check.py \
      --lib xpu/build/libforward_qwen_xpu.so \
      --tqf ~/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf \
      --model-dir ~/models/knivesysl \
      --cuda-tf /tmp/knivesysl_tf_cuda.txt \
      --prompt-tokens 512 --steps 256
"""
import argparse, ctypes, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS = os.path.join(HERE, "build-qwen", "tf_corpus_e3cdb42.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--cuda-tf", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--steps", type=int, default=256)
    args = ap.parse_args()

    cuda_tf = None
    with open(args.cuda_tf) as f:
        for line in f:
            if line.startswith("TF "):
                cuda_tf = [int(x) for x in line.split()[1:]]
    if not cuda_tf:
        print(f"no TF line in {args.cuda_tf}", file=sys.stderr)
        sys.exit(2)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    ids = tok(open(CORPUS).read(), add_special_tokens=False).input_ids
    P = args.prompt_tokens
    assert len(ids) >= P + args.steps + 8

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    rc = lib.qwn_init(os.path.expanduser(args.tqf).encode())
    if rc != 0:
        print(f"qwn_init rc={rc}", file=sys.stderr)
        sys.exit(1)

    # per-token prompt feed; the argmax after consuming ids[0..P-1] is the seed
    t0 = time.time()
    a = -1
    for p in range(P):
        a = lib.qwn_decode(ids[p], p)
        if a < 0:
            print(f"qwn_decode rc={a} at pos {p}", file=sys.stderr)
            sys.exit(1)
    print(f"PREFILL tokens={P} per-token secs={time.time()-t0:.1f}", flush=True)

    out = [int(a)]
    pos = P - 1
    for i in range(args.steps):
        t = ids[P + i]                     # the TRUE next token
        a = lib.qwn_decode(int(t), pos + 1)
        out.append(int(a))
        pos += 1
    print("TF " + " ".join(str(x) for x in out), flush=True)

    n = min(len(out), len(cuda_tf))
    agree = [out[i] == cuda_tf[i] for i in range(n)]
    rate = sum(agree) / n
    first_div = agree.index(False) if not all(agree) else -1
    ok = rate >= 0.90
    print(f"TF agreement: {rate*100:.2f}% over {n} positions, "
          f"first divergence at {first_div}  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
