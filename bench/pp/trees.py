"""What the drafting tree looks like and where a pass's drafts are accepted (p2p.plugin's P2P_TREES).

    python bench/pp/trees.py --model M --out runs/trees/x [--n 32] [--draft-k 16 --draft-topk 4 --draft-tokens 32]
    python -I bench/pp/trees.py --parse runs/trees/x
    python -I bench/pp/trees.py --pool runs/pool/x [--depths 8,12,16,20,24] [--phases 6:24]

Decodes the first --n GSM8K problems greedily on one GPU, drafting with the MTP layer, and records every pass's tree and
every candidate the beam search drafted. --parse prints the tree's shape, how deep passes are accepted, and what
smaller trees would yield, cut as SGLang cuts them. --pool chooses trees from all the candidates offline, per size and
depth: sglang (the best by the drafter's score), calibrated (by how often such a node is accepted, from the other half
of the problems), threshold (a size per pass), chain, ceiling (perfect choice), and with --phases a tree whose first
rows are fixed after some draft steps (p2p.stream.grown); and the rows of each tree the head needs to settle a pass.
"""
import argparse
import bisect
import collections
import glob
import json
import os
import statistics
import sys


def run(a) -> None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    trees = os.path.abspath(os.path.join(a.out, "trees"))
    os.makedirs(trees, exist_ok=True)
    for f in glob.glob(os.path.join(trees, "*")):
        os.remove(f)
    os.environ["P2P_TREES"] = trees
    from p2p import data
    from p2p.engine import Model, speculative
    m = Model(a.model, tp=1, batch=1, mem_fraction_static=0.9, max_total_tokens=8192,
              **speculative(a.model, "mtp", a.draft_k, a.draft_topk, a.draft_tokens))
    try:
        with open(os.path.join(a.out, "outputs.jsonl"), "w") as f:
            for r in data.load("gsm8k")[:a.n]:
                prompt = data.prompt_ids(m.tok, r["question"])
                o = m.generate([prompt], a.max_new)[0]
                f.write(json.dumps({"id": r["id"], "prompt": len(prompt), "output": o["output_ids"],
                                    "correct": data.correct(o["text"], r["gold"])}) + "\n")
    finally:
        m.engine.shutdown()
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)


def _passes(out: str) -> list:
    """Each pass with the request it belongs to: requests run one at a time, so a request's passes come in order, and a
    new request's first is the one whose length drops."""
    outputs = [json.loads(line) for line in open(os.path.join(out, "outputs.jsonl"))]
    rows = [json.loads(line) for f in sorted(glob.glob(os.path.join(out, "trees", "*.jsonl"))) for line in open(f)]
    passes, k, last = [], -1, None
    for t in rows:
        if last is None or t["length"] <= last:
            k += 1
        last = t["length"]
        if k < len(outputs):
            passes.append((t, outputs[k]))
    return passes


def _walk(t: dict, path: list, keep=None) -> tuple:
    """(drafts accepted, the rank by score among its siblings of each node taken) following path from the root,
    through nodes in keep only (all if None)."""
    cur, ranks = 0, []
    for token in path:
        kids, c = [], t["next"][cur]
        while c != -1:
            kids.append(c)
            c = t["sibling"][c]
        kids = [c for c in kids if keep is None or c in keep]
        match = [c for c in kids if t["tokens"][c] == token]
        if not match:
            break
        by_score = sorted(kids, key=lambda c: -t["scores"][c - 1])
        ranks.append(by_score.index(match[0]))
        cur = match[0]
    return len(ranks), ranks


def _depths(t: dict) -> list:
    depth, todo = [0] * len(t["tokens"]), [0]
    while todo:
        cur = todo.pop()
        c = t["next"][cur]
        while c != -1:
            depth[c] = depth[cur] + 1
            todo.append(c)
            c = t["sibling"][c]
    return depth


def parse(out: str) -> None:
    passes = _passes(out)
    n = len(passes[0][0]["tokens"])
    accepted, ranks, off_first = [], collections.Counter(), 0
    nodes, leaves, deepest = collections.Counter(), collections.Counter(), []
    cut = {k: [] for k in range(2, n + 1)}
    steps = {d: [] for d in range(1, 17)}
    chain = []
    used = 0
    for t, o in passes:
        at = t["length"] - o["prompt"]                # where the root, the last token taken, is in the output
        if at < 0 or at >= len(o["output"]) or o["output"][at] != t["tokens"][0]:
            continue
        path = o["output"][at + 1:]
        if len(path) < 16:                            # the output ends within this pass's reach
            continue
        used += 1
        got, r = _walk(t, path)
        accepted.append(got)
        ranks.update(r)
        off_first += any(x > 0 for x in r)
        depth = _depths(t)
        deepest.append(max(depth))
        for i, d in enumerate(depth):
            nodes[d] += 1
            leaves[d] += t["next"][i] == -1
        order = sorted(range(1, n), key=lambda c: -t["scores"][c - 1])
        for k in cut:
            cut[k].append(_walk(t, path, set(order[:k - 1]))[0])
        for d in steps:
            steps[d].append(min(got, d))
        best = set()
        cur = 0
        while True:                                   # the first choice at every step
            kids, c = [], t["next"][cur]
            while c != -1:
                kids.append(c)
                c = t["sibling"][c]
            if not kids:
                break
            cur = max(kids, key=lambda c: t["scores"][c - 1])
            best.add(cur)
        chain.append(_walk(t, path, best)[0])
    mean = lambda xs: sum(xs) / len(xs)
    a = json.load(open(os.path.join(out, "args.json"))) if os.path.exists(os.path.join(out, "args.json")) else {}
    print(f"# {out}: {used} passes (of {len(passes)}), {n} tokens a pass verifies"
          + (f", {a.get('draft_k')} deep, {a.get('draft_topk')} a step" if a else ""))
    print(f"tokens a pass yields: {1 + mean(accepted):.2f} (drafts accepted {mean(accepted):.2f}, the bonus token 1)")
    print("shape, per pass: depth: nodes, of which branches' ends")
    print("   " + ", ".join(f"{d}: {nodes[d] / used:.1f}/{leaves[d] / used:.1f}" for d in sorted(nodes)))
    print(f"   branches (leaves) per tree {sum(leaves.values()) / used:.1f}; deepest node: median "
          f"{statistics.median(deepest)}, range {min(deepest)}-{max(deepest)}")
    hist = collections.Counter(accepted)
    print("drafts accepted per pass: " + ", ".join(f"{k}: {hist[k] / used:.0%}" for k in sorted(hist)))
    print(f"accepted branch left the drafter's first choice at some step in {off_first / used:.0%} of passes; steps taken "
          "by the drafter's rank among siblings: " + ", ".join(f"{k}: {v}" for k, v in sorted(ranks.items())))
    print("smaller trees, tokens a pass yields:")
    print("   best N by score: " + ", ".join(f"{k}: {1 + mean(v):.2f}" for k, v in cut.items() if k in (2, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64)))
    print("   at most D steps deep (at least): " + ", ".join(f"{d}: {1 + mean(v):.2f}" for d, v in steps.items() if d in (4, 6, 8, 10, 12, 14, 16)))
    print(f"   the first choice only, a chain (at least): {1 + mean(chain):.2f}")
    json.dump({"passes": used, "tokens_per_pass": 1 + mean(accepted), "cut": {k: 1 + mean(v) for k, v in cut.items()},
               "steps": {d: 1 + mean(v) for d, v in steps.items()}, "chain": 1 + mean(chain),
               "nodes": {d: nodes[d] / used for d in sorted(nodes)}, "leaves": {d: leaves[d] / used for d in sorted(leaves)},
               "accepted": {k: hist[k] / used for k in sorted(hist)}, "off_first": off_first / used,
               "ranks": dict(ranks)}, open(os.path.join(out, "summary.json"), "w"), indent=1)


_QBINS = [0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99, 1.01]
_DBINS = [1, 2, 3, 5, 9, 17, 10 ** 6]


def _bin(edges: list, x: float) -> int:
    return next(i for i in range(len(edges) - 1) if x < edges[i + 1])


def _pool(t: dict, path: list, topk: int) -> dict:
    """A pass's pool of candidates as a tree under the root: each one's depth, the drafter's probability for it given its
    parent, its rank among its siblings by that, and whether it is on the model's own path."""
    p = t["pool"]
    score, n = p["scores"], len(p["scores"])
    parent = [p["parents"][j // topk] for j in range(n)]
    depth, q, on = [0] * n, [0.0] * n, [False] * n
    for j in range(n):                                # parents come before their children
        a = parent[j]
        depth[j] = 1 if a < 0 else depth[a] + 1
        q[j] = score[j] / (1.0 if a < 0 else max(score[a], 1e-30))
        on[j] = (a < 0 or on[a]) and depth[j] <= len(path) and p["tokens"][j] == path[depth[j] - 1]
    kids = collections.defaultdict(list)
    for j in range(n):
        kids[parent[j]].append(j)
    rank = [0] * n
    for js in kids.values():
        for r, j in enumerate(sorted(js, key=lambda j: -q[j])):
            rank[j] = r
    return {"n": n, "parent": parent, "depth": depth, "q": q, "rank": rank, "on": on, "score": score}


def _key(P: dict, j: int) -> tuple:
    return min(P["depth"][j], 24), _bin(_QBINS, P["q"][j]), min(P["rank"][j], 3)


def _fit(pools: list) -> dict:
    """How often a node is accepted when its parent is, by _key, shrunk to the drafter's own probability where seen
    rarely."""
    hits, seen, prior = collections.Counter(), collections.Counter(), collections.defaultdict(float)
    for P in pools:
        for j in range(P["n"]):
            a = P["parent"][j]
            if a < 0 or P["on"][a]:
                k = _key(P, j)
                seen[k] += 1
                hits[k] += P["on"][j]
                prior[k] += P["q"][j]
    return {k: (hits[k] + 4 * prior[k] / seen[k]) / (seen[k] + 4) for k in seen}


def _value(P: dict, table: dict) -> list:
    v = [0.0] * P["n"]
    for j in range(P["n"]):
        a = P["parent"][j]
        v[j] = (1.0 if a < 0 else v[a]) * table.get(_key(P, j), P["q"][j])
    return v


def _yields(P: dict, order: list, sizes: list) -> list:
    """Tokens a pass yields verifying the root and the first N - 1 nodes of order (closed under parents), each N."""
    best, out, at = 0, [], 0
    for N in sizes:
        while at < min(N - 1, len(order)):
            j = order[at]
            if P["on"][j]:
                best = max(best, P["depth"][j])
            at += 1
        out.append(1 + best)
    return out


def _needed(P: dict, order: list, sizes: list) -> list:
    """The rows after the root that the head needs of the tree of the root and the first N - 1 nodes of order, each N:
    the place of the deepest accepted node among them, laid out as SGLang does (in the order drafted); 0: the root's own."""
    laid, deep, out, at = [], None, [], 0
    for N in sizes:
        while at < min(N - 1, len(order)):
            j = order[at]
            bisect.insort(laid, j)
            if P["on"][j] and (deep is None or P["depth"][j] > P["depth"][deep]):
                deep = j
            at += 1
        out.append(0 if deep is None else bisect.bisect_left(laid, deep) + 1)
    return out


def _phased(P: dict, topk: int, N: int, depth: int, cuts: list) -> list:
    """The candidates of a tree of N rows drafted `depth` steps deep whose first rows are fixed in phases (cuts:
    [(steps, rows)]), in SGLang's layout: p2p.stream.grown, phase by phase, as p2p.route's head does."""
    from p2p.stream import grown
    parents = [P["parent"][g * topk] for g in range(P["n"] // topk)]
    known = lambda steps: topk + (steps - 1) * topk * topk          # candidates drafted in that many steps
    chosen, first = [], 0
    for steps, rows in cuts:
        beam = sorted({P["parent"][j] for j in range(known(steps), known(steps + 1))})     # what the next step expands
        chosen += grown(P["score"][:known(steps)], parents, topk, set(chosen), first, steps, beam, rows - 1 - len(chosen))
        first = steps
    return chosen + grown(P["score"][:known(depth)], parents, topk, set(chosen), first, depth, [], N - 1 - len(chosen))


def pool(out: str, depths: list, phases: list = ()) -> None:
    a = json.load(open(os.path.join(out, "args.json")))
    topk, k = a["draft_topk"], a["draft_k"]
    outputs = [json.loads(line) for line in open(os.path.join(out, "outputs.jsonl"))]
    passes = _passes(out)
    pools, owner = [], []
    for t, o in passes:
        at = t["length"] - o["prompt"]
        if at < 0 or at >= len(o["output"]) or o["output"][at] != t["tokens"][0]:
            continue
        path = o["output"][at + 1:]
        if len(path) < k:                            # the output ends within this pass's reach
            continue
        P = _pool(t, path, topk)
        verified = sorted(range(P["n"]), key=lambda j: -P["score"][j])[:len(t["tokens"]) - 1]
        assert sorted(P["score"][j] for j in verified) == sorted(t["scores"]), "the pool doesn't hold the verified tree"
        pools.append(P)
        owner.append(outputs.index(o) % 2)
    sizes = list(range(2, 129))
    kept = [N for N in (13, 16, 20, 24, 32, 48, 64, 96, 128)]
    needed = collections.defaultdict(lambda: {N: collections.Counter() for N in kept})
    phased = collections.defaultdict(lambda: {N: collections.Counter() for N in kept})
    phased_yield = collections.defaultdict(collections.Counter)
    halves = [[P for P, h in zip(pools, owner) if h == x] for x in (0, 1)]
    tables = [_fit(halves[1]), _fit(halves[0])]       # each half valued by what the other half found
    curves = collections.defaultdict(lambda: [0.0] * len(sizes))
    thresholds = [10 ** (-x / 8) for x in range(0, 33)]
    by_t = [[0.0, 0.0] for _ in thresholds]
    for P, h in zip(pools, owner):
        v = _value(P, tables[h])
        for D in depths:
            within = [j for j in range(P["n"]) if P["depth"][j] <= D]
            for name, key in (("sglang", lambda j: -P["score"][j]), ("calibrated", lambda j: -v[j])):
                for i, y in enumerate(_yields(P, sorted(within, key=key), sizes)):
                    curves[(name, D)][i] += y
            for N, rows in zip(sizes, _needed(P, sorted(within, key=lambda j: (-P["score"][j], j)), sizes)):
                if N in kept:
                    needed[D][N][rows] += 1
            for N in kept if phases and phases[-1][0] < D else ():
                if N <= phases[-1][1]:
                    continue
                order = _phased(P, topk, N, D, phases)
                on = [j for j in order if P["on"][j]]
                deep = max(on, key=lambda j: P["depth"][j]) if on else None
                phased[D][N][0 if deep is None else order.index(deep) + 1] += 1
                phased_yield[D][N] += 1 + (P["depth"][deep] if on else 0)
        chain, cur = [], -1
        kids = collections.defaultdict(list)
        for j in range(P["n"]):
            kids[P["parent"][j]].append(j)
        while kids[cur]:
            cur = max(kids[cur], key=lambda j: P["q"][j])
            chain.append(cur)
        for i, y in enumerate(_yields(P, chain, sizes)):
            curves[("chain", k)][i] += y
        deepest = max([P["depth"][j] for j in range(P["n"]) if P["on"][j]], default=0)
        for i, N in enumerate(sizes):
            curves[("ceiling", k)][i] += 1 + min(deepest, N - 1)
        order = sorted(range(P["n"]), key=lambda j: -v[j])
        for i, tau in enumerate(thresholds):
            take = [j for j in order if v[j] >= tau]
            by_t[i][0] += 1 + len(take)
            by_t[i][1] += _yields(P, take, [len(take) + 1])[0]
    n = len(pools)
    print(f"# {out}: {n} passes, pools of {pools[0]['n']} candidates ({k} deep, {topk} a step); tokens a pass yields "
          "verifying the root and N - 1 drafts chosen from the pool")
    show = [N for N in (4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128) if N in sizes]
    print("| Rule | Depth | " + " | ".join(f"N={N}" for N in show) + " |")
    print("|---|---|" + "---|" * len(show))
    summary = {"passes": n, "pool": pools[0]["n"], "sizes": sizes, "curves": {}, "threshold": [],
               "needed": {f"sglang/{D}": {N: [c[r] / n for r in range(N)] for N, c in by.items()} for D, by in needed.items()}}
    for (name, D), c in sorted(curves.items(), key=lambda x: (["sglang", "calibrated", "chain", "ceiling"].index(x[0][0]), x[0][1])):
        c = [x / n for x in c]
        summary["curves"][f"{name}/{D}"] = dict(zip(sizes, c))
        print(f"| {name} | {D} | " + " | ".join(f"{c[sizes.index(N)]:.2f}" for N in show) + " |")
    for D, by in sorted(needed.items()):
        print(f"rows the head needs after the root (sglang, {D} deep): " + ", ".join(
            f"N={N}: {sum(r * c for r, c in by[N].items()) / n:.1f}" for N in kept))
    summary["phases"] = [list(x) for x in phases]
    for D, by in sorted(phased.items()):
        sized = [N for N in kept if by[N]]
        summary["curves"][f"phased/{D}"] = {N: phased_yield[D][N] / n for N in sized}
        summary["needed"][f"phased/{D}"] = {N: [by[N][r] / n for r in range(N)] for N in sized}
        print(f"first rows fixed after {phases} ({D} deep): tokens a pass " + ", ".join(
            f"N={N}: {phased_yield[D][N] / n:.2f}" for N in sized) + "; rows the head needs " + ", ".join(
            f"N={N}: {sum(r * c for r, c in by[N].items()) / n:.1f}" for N in sized))
    print("size per pass (calibrated value above a threshold): mean size -> tokens a pass yields")
    for tau, (m, y) in zip(thresholds, by_t):
        summary["threshold"].append({"threshold": tau, "size": m / n, "yields": y / n})
    print("   " + ", ".join(f"{x['size']:.1f} -> {x['yields']:.2f}" for x in summary["threshold"] if 4 <= x["size"] <= 64))
    table = _fit(pools)
    summary["calibration"] = {"qbins": _QBINS, "dbins": _DBINS, "table": [[*k, v] for k, v in sorted(table.items())]}
    json.dump(summary, open(os.path.join(out, "pool.json"), "w"), indent=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parse", default="")
    p.add_argument("--pool", default="", help="choose trees offline from every candidate recorded (see above)")
    p.add_argument("--depths", default="8,12,16,20,24")
    p.add_argument("--phases", default="", help="--pool: also trees whose first rows are fixed in phases, as P2P_PP_PHASES")
    p.add_argument("--model", default="")
    p.add_argument("--out", default="")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--draft-k", type=int, default=16)
    p.add_argument("--draft-topk", type=int, default=4)
    p.add_argument("--draft-tokens", type=int, default=32)
    a = p.parse_args()
    if a.parse:
        return parse(a.parse)
    if a.pool:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        return pool(a.pool, [int(x) for x in a.depths.split(",")],
                    [tuple(int(v) for v in x.split(":")) for x in a.phases.split(",") if x])
    os.makedirs(a.out, exist_ok=True)
    run(a)
    parse(a.out)


if __name__ == "__main__":
    main()
