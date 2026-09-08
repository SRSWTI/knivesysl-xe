#!/usr/bin/env python3
"""serve_batch.py - continuous batching + automatic prefix caching.

A real continuous-batching engine loop over the multi-slot ABI:

  * requests arrive on a schedule and are admitted into free slots
    mid-flight - no waiting for a batch to drain;
  * each iteration runs at most ONE chunked-prefill chunk (so admission
    cannot starve decode) plus ONE batched decode step covering every
    slot currently generating (qwn_decode_batch: one weight stream, N rows);
  * finished requests free their slot immediately and the next waiter is
    admitted on the following iteration;
  * APC: prompt prefixes are checkpointed at chunk boundaries into host USM
    and restored into a fresh slot on a hit, skipping that much prefill.
    Granularity is the chunk, not vLLM's 16-token block, because 48 of 64
    layers are recurrent - GDN state only exists where it was snapshotted.

Reports per-request TTFT, mean ITL, and engine aggregate throughput.

TQ_XPU_DEV=0 TQ_XPU_SLOTS=8 TQ_XPU_K64=all python3 xpu/tools/serve_batch.py \
    --lib xpu/build/libforward_qwen_xpu.so --tqf ~/models/.../*.tqf \
    --reqs 12 --shared 8 --prompt 512 --gen 32 --rate 4 --apc 1
"""
import argparse
import ctypes
import os
import statistics
import sys
import time

TOK = 9707
WAIT, PREFILL, DECODE, DONE = 0, 1, 2, 3


class Req:
    __slots__ = ("rid", "prompt", "gen", "arrival", "slot", "pos", "state",
                 "out", "t_first", "t_last", "itl", "apc_hit", "admitted",
                 "ttft")

    def __init__(self, rid, prompt, gen, arrival):
        self.rid, self.prompt, self.gen, self.arrival = rid, prompt, gen, arrival
        self.slot, self.pos, self.state = -1, 0, WAIT
        self.out, self.itl = [], []
        self.t_first = self.t_last = None
        self.apc_hit = 0
        self.admitted = None


class Engine:
    def __init__(self, lib, slots, chunk, apc):
        self.lib, self.nslots, self.chunk, self.apc = lib, slots, chunk, apc
        self.free = list(range(slots))
        self.active = []
        self.cache = {}          # prefix-tuple-hash -> (pos, blob)
        self.cache_bytes = 0
        self.hits = self.misses = 0
        self.dec_rows = 0        # rows retired by batched decode steps
        self.dec_time = 0.0      # wall time spent inside those steps

    # ---- APC ------------------------------------------------------------
    def lookup(self, prompt):
        """Longest cached prefix at chunk granularity."""
        if not self.apc:
            return 0, None
        n = (len(prompt) // self.chunk) * self.chunk
        while n >= self.chunk:
            ent = self.cache.get(hash(tuple(prompt[:n])))
            if ent:
                return ent[0], ent[1]
            n -= self.chunk
        return 0, None

    def store(self, prompt, pos, slot):
        if not self.apc or pos < self.chunk:
            return
        key = hash(tuple(prompt[:pos]))
        if key in self.cache:
            return
        nb = self.lib.qwn_ckpt_bytes(pos)
        blob = self.lib.qwn_host_alloc(nb)
        if not blob:
            return
        if self.lib.qwn_ckpt_save(slot, pos, blob) != 0:
            self.lib.qwn_host_free(blob)
            return
        self.cache[key] = (pos, blob)
        self.cache_bytes += nb

    def release(self):
        for pos, blob in self.cache.values():
            self.lib.qwn_host_free(blob)
        self.cache.clear()

    # ---- scheduling -----------------------------------------------------
    def admit(self, pending, now):
        while self.free and pending and pending[0].arrival <= now:
            r = pending.pop(0)
            r.slot = self.free.pop(0)
            r.admitted = now
            hit_pos, blob = self.lookup(r.prompt)
            if blob is not None and self.lib.qwn_ckpt_restore(
                    r.slot, hit_pos, blob) == 0:
                r.pos, r.apc_hit = hit_pos, hit_pos
                self.hits += 1
            else:
                self.lib.qwn_reset_slot(r.slot)
                r.pos = 0
                self.misses += 1
            r.state = PREFILL
            self.active.append(r)

    def prefill_step(self):
        """One chunk for one request - bounded so decode is not starved."""
        for r in self.active:
            if r.state != PREFILL:
                continue
            lib = self.lib
            assert lib.qwn_set_slot(r.slot) == 0
            left = len(r.prompt) - r.pos
            take = min(self.chunk, (left // 8) * 8)
            if take >= 8:
                arr = (ctypes.c_int * take)(*r.prompt[r.pos:r.pos + take])
                rc = lib.qwn_prefill_chunk(arr, take, r.pos)
                if rc != 0:
                    raise RuntimeError(f"prefill rc={rc}")
                r.pos += take
                # One checkpoint per distinct prompt: the deepest chunk
                # boundary only. Caching every boundary costs a full state
                # copy (~176 MB at 512 tokens) for no extra hit rate.
                if r.pos == (len(r.prompt) // self.chunk) * self.chunk:
                    self.store(r.prompt, r.pos, r.slot)
                return True
            # tail (< 8 tokens): per-token, then this request starts decoding
            nxt = -1
            for t in r.prompt[r.pos:]:
                nxt = lib.qwn_decode(t, r.pos)
                if nxt < 0:
                    raise RuntimeError(f"tail rc={nxt}")
                r.pos += 1
            r.state = DECODE
            r.t_first = time.perf_counter()
            r.t_last = r.t_first
            r.out.append(nxt)
            return True
        return False

    def decode_step(self):
        rows = [r for r in self.active if r.state == DECODE
                and len(r.out) < r.gen]
        if not rows:
            return 0
        rows = rows[:min(self.nslots, 8)]
        n = len(rows)
        Arr = ctypes.c_int * n
        slots = Arr(*[r.slot for r in rows])
        toks = Arr(*[r.out[-1] for r in rows])
        poss = Arr(*[r.pos for r in rows])
        out = Arr(*([0] * n))
        t0 = time.perf_counter()
        rc = self.lib.qwn_decode_batch(slots, toks, poss, n, out)
        now = time.perf_counter()
        if rc != n:
            raise RuntimeError(f"decode_batch rc={rc}")
        self.dec_rows += n
        self.dec_time += now - t0
        for i, r in enumerate(rows):
            r.itl.append((now - r.t_last) * 1e3)
            r.t_last = now
            r.out.append(out[i])
            r.pos += 1
        return n

    def retire(self):
        still = []
        for r in self.active:
            if r.state == DECODE and len(r.out) >= r.gen:
                r.state = DONE
                self.free.append(r.slot)
            else:
                still.append(r)
        self.active = still


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--reqs", type=int, default=12)
    ap.add_argument("--shared", type=int, default=8,
                    help="requests sharing one system prefix (APC target)")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--rate", type=float, default=4.0, help="arrivals/s")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--apc", type=int, default=1)
    args = ap.parse_args()

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_set_slot.argtypes = [ctypes.c_int]
    lib.qwn_reset_slot.argtypes = [ctypes.c_int]
    lib.qwn_prefill_chunk.argtypes = [ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_int, ctypes.c_int]
    lib.qwn_decode_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [
        ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.qwn_host_alloc.argtypes = [ctypes.c_size_t]
    lib.qwn_host_alloc.restype = ctypes.c_void_p
    lib.qwn_host_free.argtypes = [ctypes.c_void_p]
    lib.qwn_ckpt_bytes.argtypes = [ctypes.c_int]
    lib.qwn_ckpt_bytes.restype = ctypes.c_size_t
    lib.qwn_ckpt_save.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.qwn_ckpt_restore.argtypes = [ctypes.c_int, ctypes.c_int,
                                     ctypes.c_void_p]
    rc = lib.qwn_init(os.path.expanduser(args.tqf).encode())
    if rc != 0:
        print(f"qwn_init rc={rc}", file=sys.stderr)
        return 1
    nslots = lib.qwn_num_slots()

    P = (args.prompt // 8) * 8
    shared_prefix = [TOK + (i % 7) for i in range(P)]
    reqs = []
    for i in range(args.reqs):
        if i < args.shared:                      # same system prefix
            pr = list(shared_prefix) + [TOK + 20 + i]
        else:                                    # unique prompt
            pr = [TOK + 30 + i + (j % 5) for j in range(P)] + [TOK + 20 + i]
        reqs.append(Req(i, pr, args.gen, i / args.rate))

    eng = Engine(lib, nslots, args.chunk, args.apc)
    pending = list(reqs)
    print(f"slots={nslots} reqs={args.reqs} shared={args.shared} "
          f"prompt={P} gen={args.gen} rate={args.rate}/s "
          f"chunk={args.chunk} apc={args.apc}")

    t0 = time.perf_counter()
    iters = 0
    while pending or eng.active:
        now = time.perf_counter() - t0
        eng.admit(pending, now)
        if not eng.active:
            time.sleep(0.002)
            continue
        eng.prefill_step()
        eng.decode_step()
        eng.retire()
        iters += 1
    wall = time.perf_counter() - t0

    for r in reqs:
        r.ttft = (r.t_first - t0) - r.arrival if r.t_first else float("nan")
    gen_tokens = sum(len(r.out) for r in reqs)
    prompt_tokens = sum(len(r.prompt) - r.apc_hit for r in reqs)
    ttfts = [r.ttft for r in reqs]
    itls = [x for r in reqs for x in r.itl]
    print(f"\n  {'rid':>4} {'apc_hit':>8} {'ttft_s':>8} {'itl_ms':>8} {'toks':>5}")
    for r in reqs:
        m = statistics.fmean(r.itl) if r.itl else float("nan")
        print(f"  {r.rid:>4} {r.apc_hit:>8} {r.ttft:>8.2f} {m:>8.1f} "
              f"{len(r.out):>5}")
    print(f"\n  wall              {wall:.2f} s   ({iters} engine iterations)")
    print(f"  apc               {eng.hits} hits / {eng.misses} misses, "
          f"{eng.cache_bytes/1e6:.0f} MB cached")
    print(f"  prefill tokens    {prompt_tokens} "
          f"({sum(r.apc_hit for r in reqs)} skipped by APC)")
    print(f"  decode tokens     {gen_tokens}")
    print(f"  ttft mean/p99     {statistics.fmean(ttfts):.2f} / "
          f"{sorted(ttfts)[max(0, int(0.99*len(ttfts))-1)]:.2f} s")
    print(f"  itl mean/p99      {statistics.fmean(itls):.1f} / "
          f"{sorted(itls)[max(0, int(0.99*len(itls))-1)]:.1f} ms")
    print(f"  decode in-step    {eng.dec_rows/eng.dec_time:.1f} tok/s "
          f"(batched steps only, {eng.dec_rows} rows in "
          f"{eng.dec_time:.2f} s)")
    print(f"  decode wall       {gen_tokens/wall:.1f} tok/s "
          f"(incl. prefill interference)")
    print(f"  total aggregate   {(gen_tokens+prompt_tokens)/wall:.1f} tok/s")
    eng.release()
    lib.qwn_free()
    return 0


if __name__ == "__main__":
    sys.exit(main())
