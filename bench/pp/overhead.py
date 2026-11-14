"""Where a pass's time goes besides the network: every stage's spans (SGLang's and ours, p2p.pp.label) and GPU marks.

    bench/pp/netns.sh 0 python bench/pp/overhead.py --model M --pp 4 --comm fused [bench.py's options] [--tokens 24]
        --out runs/overhead/x
    python -I bench/pp/overhead.py --parse runs/overhead/x

For one request: ms per token and per pass as bench.py; then --tokens more with every span of every stage on the
machine's clock (Scheduler.p2p_spans); then --tokens more under SGLang's profiler, a Chrome trace per stage in
out/traces (only its kernel times are used). A stage's window runs from the end of its last wait to its send; a pass's
chain links the windows through the gaps between one stage's send and the next one's input. Once the input is received
ahead the CPU no longer waits for it, so the same is read off the GPUs (p2p.pp.mark): per stage its window, the gap
before it, and how long before its input its CPU had queued the pass. Prints the tables, writes out/summary.json.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import random
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


def _files(path: str, pattern: str, n: int) -> list:
    for _ in range(1200):
        if len(found := sorted(glob.glob(os.path.join(path, pattern)))) >= n:
            sizes = None
            while sizes != (sizes := [os.path.getsize(f) for f in found]):     # traces: until all are written
                time.sleep(3)
            return found
        time.sleep(0.5)
    raise TimeoutError(f"{path}: {len(glob.glob(os.path.join(path, pattern)))} of {n} {pattern}")


def run(a) -> None:
    os.environ["P2P_PP_SPANS"] = "1"
    from bench import engine
    m = engine(a)
    rng = random.Random(0)
    prompt = [rng.randrange(1000, 20000) for _ in range(32)]

    def generate(n):
        t = time.perf_counter()
        o = m.engine.generate(input_ids=[prompt], sampling_params={"temperature": 0, "max_new_tokens": n,
                                                                  "ignore_eos": True})[0]
        return time.perf_counter() - t, o["meta_info"].get("spec_verify_ct", 0)

    spans, traces = (os.path.abspath(os.path.join(a.out, d)) for d in ("spans", "traces"))
    for d in (spans, traces):
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*")):
            os.remove(f)
    try:
        generate(8)                                             # warm up
        timed = {n: [generate(n) for _ in range(2)] for n in (32, 96)}
        s = {n: statistics.median(t for t, _ in r) for n, r in timed.items()}
        passes = timed[96][0][1] - timed[32][0][1]
        m.rpc("p2p_spans", on=True)
        spanned = generate(a.tokens)
        m.rpc("p2p_spans", on=False, path=spans)
        _files(spans, "spans_pp*.json", a.pp)
        m.engine.start_profile(output_dir=traces, activities=["CPU", "GPU"], with_stack=False, record_shapes=False)
        traced = generate(a.tokens)
        m.engine.stop_profile()
        _files(traces, "*.trace.json.gz", a.pp)
    finally:
        m.close()
    meta = {"args": vars(a), "ms_per_token": 1000 * (s[96] - s[32]) / 64,
            "ms_per_pass": 1000 * (s[96] - s[32]) / passes if passes else 1000 * (s[96] - s[32]) / 64,
            "tokens_per_pass": 64 / passes if passes else 1.0, "spans_s": spanned[0], "trace_s": traced[0],
            "delay_ms": float(os.environ.get("P2P_NET_DELAY_MS", "nan"))}           # bench/pp/netns.sh's
    json.dump(meta, open(os.path.join(a.out, "run.json"), "w"), indent=1)


# Reading them.

def _rank(path: str) -> int:
    m = re.search(r"(?:PP-|spans_pp)(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _self_times(spans: list, start: int, end: int) -> dict:
    """{name: its own time inside [start, end]}, without the time of the spans nested in it; "(none)": no span's."""
    inside = sorted(((max(s, start), min(e, end), n) for n, s, e in spans if s < end and e > start),
                    key=lambda x: (x[0], -x[1]))
    own, stack = collections.Counter(), []                     # stack: [end, name, own time so far]
    top = 0

    def close(upto):
        while stack and stack[-1][0] <= upto:
            e, n, t = stack.pop()
            own[n] += t
    for s, e, n in inside:
        close(s)
        if stack:
            stack[-1][2] -= e - s
        else:
            top += e - s
        stack.append([e, n, e - s])
    close(float("inf"))
    own["(none)"] = (end - start) - top
    return own


def _out(names: set, rank: int, stages: int) -> str:
    """The span that starts a stage's output on its way: what ends its part of a pass."""
    if "p2p:embed_send" in names:
        return "p2p:embed_send"                                 # the head, under p2p.route: the next pass's input
    if stages == 1:
        return "scheduler.run_batch"
    if rank == stages - 1:
        return "send_res_dict_to_next_stage"                    # SGLang's last stage: outputs to stage 0
    return "send_proxy_dict_to_next_stage"


def _windows(spans: list, rank: int, stages: int) -> list:
    """This stage's part of each pass: {"start", "out", "src", "parts"}, in ns."""
    out_name = _out({n for n, _, _ in spans}, rank, stages)
    outs = sorted(s for n, s, _ in spans if n == out_name)
    recvs = [(s, e, n) for n, s, e in spans if n.startswith(("p2p:recv_dict<-", "p2p:recv_reqs<-"))]
    waits = []
    for n, s, e in spans:
        if n == "p2p:wait":
            around = [x for x in recvs if x[0] <= s and e <= x[1]]
            src = int(min(around, key=lambda x: x[1] - x[0])[2].split("<-")[1]) if around else None
            waits.append((e, src))
    waits.sort()
    out, prev = [], None
    for t in outs:
        before = [w for w in waits if w[0] <= t and (prev is None or w[0] > prev)]
        start, src = before[-1] if before else (prev, None)
        if start is not None:
            out.append({"start": start, "out": t, "src": src, "parts": _self_times(spans, start, t)})
        prev = t
    return out


def _chains(windows: dict, ring: int, stages: int) -> list:
    """Each pass's chain, from the stage the ring starts at back round to it: [(stage, window, gap before it)], ns."""
    chains = []
    for prev, last in zip(windows[ring], windows[ring][1:]):
        chain, cur, r = [], last, ring
        for _ in range(stages + 2):
            src = cur["src"]
            if src is None:
                break
            sent = [w for w in windows.get(src, []) if w["out"] <= cur["start"]]
            if not sent:
                break
            before = sent[-1]
            chain.append((r, cur["out"] - cur["start"], cur["start"] - before["out"]))
            if src == ring and before["out"] <= prev["out"]:
                break
            cur, r = before, src
        else:
            continue
        if chain and src == ring:
            chains.append({"pass": last["out"] - prev["out"], "chain": chain[::-1]})
    return chains


def _marked(spans: dict) -> dict:
    """From the GPU marks (p2p.route: gpu:in when a stage's input has come and its layers can run, gpu:out when its output
    can go): per stage, each pass's {"start": in, "out", "src": the stage whose output came last before it, "queued": when
    the CPU had queued it all (its send's span ends; None on the head, whose output is the next pass's embedding), "done":
    when its forward had run (gpu:done; on the head, the logits too)}."""
    outs = {r: sorted(s for n, s, _ in sp if n == "gpu:out") for r, sp in spans.items()}
    out = {}
    for r, sp in spans.items():
        dones = sorted(s for n, s, _ in sp if n == "gpu:done")
        sends = sorted((s, e) for n, s, e in sp if n == "send_proxy_dict_to_next_stage")
        rows = []
        for t in sorted(s for n, s, _ in sp if n == "gpu:in"):
            after = [o for o in outs[r] if o > t]
            if not after:
                continue
            last = {k: max((o for o in os_ if o <= t), default=None) for k, os_ in outs.items() if k != r}
            src = max((k for k, o in last.items() if o is not None), key=lambda k: last[k], default=None)
            queued = [e for s, e in sends if s <= after[0]]           # its send starts before its output can go
            done = [x for x in dones if t < x <= after[0]]
            rows.append({"start": t, "out": after[0], "src": src, "gap": t - last[src] if src is not None else None,
                         "queued": queued[-1] if queued else None, "done": done[0] if done else None})
        out[r] = rows
    return out


def _kind(kernel: str) -> str:
    if kernel.startswith("nccl"):
        return "nccl"
    if "spmm" in kernel or "_unpack" in kernel:
        return "compact"                                        # p2p.compact: replicas' pruned projections (CUDA, Triton)
    if any(k in kernel.lower() for k in ("gemm", "gemv", "cutlass", "xmma")):
        return "dense"                                          # dense matmuls (cuBLAS, CUTLASS)
    return "other"


def _gpu(trace: str, stages: int) -> dict:
    """This stage's kernel time per pass, between the passes its spans mark (as _windows): all but NCCL ("compute"), of
    which compact replicas' and dense matmuls'; and NCCL's, which includes receive kernels waiting for their data."""
    try:
        ev = json.load(gzip.open(trace))["traceEvents"]
    except (EOFError, OSError, ValueError):                    # cut short: its stage was stopped while writing it
        print(f"unreadable: {trace}")
        return {}
    rank, ann = _rank(trace), [e for e in ev if e.get("cat") == "user_annotation"]
    out_name = _out({e["name"] for e in ann}, rank, stages)
    passes = sorted(e["ts"] for e in ann if e["name"] == out_name)
    if len(passes) < 3:
        return {}
    lo, hi, n = passes[1], passes[-1], len(passes) - 2
    by = collections.Counter()
    for e in ev:
        if e.get("cat") == "kernel" and lo <= e["ts"] < hi:
            by[_kind(e["name"])] += e["dur"] / n / 1000
    return {"rank": rank, "compute_ms": sum(v for k, v in by.items() if k != "nccl"), "compact_ms": by["compact"],
            "dense_ms": by["dense"], "nccl_ms": by["nccl"]}


def parse(out: str) -> None:
    meta = json.load(open(os.path.join(out, "run.json"))) if os.path.exists(os.path.join(out, "run.json")) else {}
    spans = {_rank(f): json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "spans", "spans_pp*.json")), key=_rank)}
    stages = len(spans)
    windows = {r: _windows(s, r, stages) for r, s in spans.items()}
    med = lambda xs: statistics.median(xs) / 1e6 if xs else float("nan")
    rows = {}
    for r, ws in windows.items():
        ws = ws[1:-1] or ws                                     # the first and last may be cut short
        parts = {n: [w["parts"].get(n, 0) for w in ws] for n in set().union(*(w["parts"] for w in ws))}
        rows[r] = {"passes": len(ws), "window_ms": med([w["out"] - w["start"] for w in ws]),
                   "pass_ms": med([b["out"] - a["out"] for a, b in zip(ws, ws[1:])]),
                   "waits_on": collections.Counter(w["src"] for w in ws).most_common(1)[0][0] if ws else None,
                   "parts_ms": dict(sorted(((n, med(v)) for n, v in parts.items()), key=lambda x: -x[1]))}
    ring = stages - 1 if any(n == "p2p:embed_send" for n, _, _ in spans.get(stages - 1, [])) else 0
    chains = _chains(windows, ring, stages)
    chains = chains[1:-1] or chains
    chain = []
    if chains:
        common = collections.Counter(tuple(r for r, _, _ in c["chain"]) for c in chains).most_common(1)[0][0]
        same = [c for c in chains if tuple(r for r, _, _ in c["chain"]) == common]
        chain = [{"stage": r, "window_ms": med([c["chain"][i][1] for c in same]),
                  "gap_ms": med([c["chain"][i][2] for c in same])} for i, r in enumerate(common)]
        chain_pass = med([c["pass"] for c in same])
    gpu = {g["rank"]: g for g in (_gpu(f, stages) for f in sorted(glob.glob(os.path.join(out, "traces", "*.trace.json.gz"))))
           if g}
    marked = _marked(spans) if any(n == "gpu:in" for sp in spans.values() for n, _, _ in sp) else {}
    marks, gchain, gchain_pass = {}, [], None
    for r, ws in marked.items():
        ws = ws[1:-1] or ws
        marks[r] = {"window_ms": med([w["out"] - w["start"] for w in ws]),
                    "gap_ms": med([w["gap"] for w in ws if w["gap"] is not None]),
                    "src": collections.Counter(w["src"] for w in ws).most_common(1)[0][0] if ws else None,
                    "ahead_ms": med([w["start"] - w["queued"] for w in ws if w["queued"] is not None]),
                    "layers_ms": med([w["done"] - w["start"] for w in ws if w["done"] is not None]),
                    "own_ms": med([w["out"] - w["done"] for w in ws if w["done"] is not None])}
    if marked:
        gchains = _chains(marked, ring, stages)
        gchains = gchains[1:-1] or gchains
        if gchains:
            common = collections.Counter(tuple(r for r, _, _ in c["chain"]) for c in gchains).most_common(1)[0][0]
            same = [c for c in gchains if tuple(r for r, _, _ in c["chain"]) == common]
            gchain = [{"stage": r, "window_ms": med([c["chain"][i][1] for c in same]),
                       "gap_ms": med([c["chain"][i][2] for c in same])} for i, r in enumerate(common)]
            gchain_pass = med([c["pass"] for c in same])
    summary = {"meta": meta, "stages": rows, "chain": chain, "chain_pass_ms": chain_pass if chain else None, "gpu": gpu,
               "marks": marks, "gpu_chain": gchain, "gpu_chain_pass_ms": gchain_pass}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)

    a = meta.get("args", {})
    print(f"# {out}: {a.get('model', '?')} pp {stages}" + (f", drafting {a.get('draft')} {a.get('draft_k')}"
                                                           if a.get("draft") else ""))
    if meta:
        print(f"timed: {meta['ms_per_token']:.2f} ms per token, {meta['ms_per_pass']:.2f} ms per pass "
              f"({meta['tokens_per_pass']:.2f} tokens per pass)")
    for r, row in rows.items():
        g = gpu.get(r, {})
        print(f"stage {r}: pass {row['pass_ms']:.2f} ms, window {row['window_ms']:.2f} ms after its wait on "
              f"{row['waits_on']}; GPU kernels per pass {g.get('compute_ms', float('nan')):.2f} ms")
        for n, v in row["parts_ms"].items():
            if v >= 0.02:
                print(f"    {v:7.3f}  {n}")
    if chain:
        print(f"chain of a pass ({chain_pass:.2f} ms): " + " -> ".join(
            f"[gap {c['gap_ms']:.2f}] stage {c['stage']} {c['window_ms']:.2f}" for c in chain))
        print(f"    windows {sum(c['window_ms'] for c in chain):.2f} ms + gaps {sum(c['gap_ms'] for c in chain):.2f} ms")
    for r, m in marks.items():
        print(f"stage {r} on its GPU: input in to output out {m['window_ms']:.2f} ms, after a gap of {m['gap_ms']:.2f} ms "
              f"from stage {m['src']}'s output" + (f"; its CPU had queued the pass {m['ahead_ms']:.2f} ms before the input "
                                                  f"came" if m["ahead_ms"] == m["ahead_ms"] else ""))
    if gchain:
        print(f"chain of a pass on the GPUs ({gchain_pass:.2f} ms): " + " -> ".join(
            f"[gap {c['gap_ms']:.2f}] stage {c['stage']} {c['window_ms']:.2f}" for c in gchain))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parse", default="", help="only read what a run wrote to this directory")
    a, rest = p.parse_known_args()
    if a.parse:
        return parse(a.parse)
    from bench import arguments
    p = argparse.ArgumentParser()
    arguments(p)
    p.add_argument("--tokens", type=int, default=0, help="new tokens recorded, and again traced (default: about 24 "
                   "passes, as many tokens as a drafting pass verifies)")
    p.add_argument("--out", required=True)
    a = p.parse_args(rest)
    a.tokens = a.tokens or 24 * ((a.draft_tokens or a.draft_k + 1) if a.draft else 1)
    os.makedirs(a.out, exist_ok=True)
    run(a)
    parse(a.out)


if __name__ == "__main__":
    main()
