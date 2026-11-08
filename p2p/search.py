"""Which layers to run in parallel and what each replica keeps, from measured costs. No torch, no SGLang.

A `shape` maps a layer's projections to [rows, cols] and a `keep` maps them to the percentage of weights a replica keeps.
A segment (start, n) is n consecutive layers run in one step; its last layer's node holds a replica of layer start + n - 2.
"""
from __future__ import annotations

import json
import math
import statistics

MLP = ("gate", "up", "down")
ROUTED = ("egate", "eup", "edown")   # a MoE's routed experts
STATE_KEEP = 7                 # k and v write only the current token's state, so they cost little to tighten


def kinds(layers: list) -> list:
    """A label per layer, shared by layers of the same class and projection shapes. The class alone isn't enough: Gemma 4's
    full-attention layers have wider heads than its sliding ones. layers: [{"type", "proj"}]."""
    seen = {}
    return [seen.setdefault(json.dumps(x, sort_keys=True), f"{x['type']}/{len(seen)}") for x in layers]


def size(shape: dict, keep: dict) -> float:
    """A replica's size as a fraction of the bf16 layer: 16 bits per kept weight, plus a 1-bit mask or a column index
    per kept weight, whichever is smaller."""
    bits = total = 0
    for name, (rows, cols) in shape.items():
        total += rows * cols
        k = max(1, round(keep[name] / 100 * cols))
        bits += 16 * rows * cols if keep[name] >= 100 else \
            16 * k * rows + min(rows * cols, rows * k * math.ceil(math.log2(cols)))
    return bits / (16 * total)


def _filled(shape: dict, mixer: dict, cap: float, rest: tuple = MLP) -> "dict | None":
    """`mixer` plus the largest whole percentage for the `rest` projections that fits `cap`."""
    fits = [y for y in range(1, 100) if size(shape, {**mixer, **dict.fromkeys(rest, y)}) <= cap]
    return {**mixer, **dict.fromkeys(rest, fits[-1])} if fits else None


def splits(shape: dict, cap: float) -> dict:
    """Ways to split a replica's budget, by name, each at most `cap`: uniform, and k, v tightened in favour of the output
    projection, with q and z moved around that and the MLP taking the rest."""
    x = math.floor(100 * (cap - 1 / 16))
    if ROUTED[0] in shape:                 # routed experts are nearly the whole layer, so all the rest is cheap to keep
        others = [n for n in shape if n not in ROUTED]
        recipes = {"uniform": dict.fromkeys(shape, x)}
        recipes |= {f"rest{r}": _filled(shape, dict.fromkeys(others, r), cap, ROUTED) for r in (50, 70, 90)}
        return {k: v for k, v in recipes.items() if v}
    out = "out" if "out" in shape else "o"
    linear = "z" in shape                  # a linear-attention mixer has its own z block and a smaller q
    base = {"k": STATE_KEEP, "v": STATE_KEEP, "q": x // 2 if linear else max(x - 5, 1), out: x + 22, **({"z": x} if linear else {})}
    moves = {"shift": {}, "out+12": {out: x + 12}, "out+32": {out: x + 32}, "q=x": {"q": x}}
    moves |= {"z-8": {"z": max(x - 8, 1)}, "z+7": {"z": x + 7}} if linear else {"q-13": {"q": max(x - 13, 1)}}
    recipes = {"uniform": dict.fromkeys(shape, x)}
    for name, move in moves.items():
        mixer = {n: v for n, v in {**base, **move}.items() if n in shape}
        recipes[name] = _filled(shape, mixer, cap)
    return {k: v for k, v in recipes.items() if v}


def representatives(types: list) -> dict:
    """{layer type: (bands of pair segments, triple segments)} to compare splits on: six pairs spread over three depth
    bands and two triples, by the type of the layer their replica copies."""
    def spread(xs: list, n: int) -> list:
        return [xs[(2 * j + 1) * len(xs) // (2 * n)] for j in range(n)] if len(xs) >= n else xs

    out = {}
    for t in dict.fromkeys(types):
        p = spread([(i, 2) for i in range(len(types) - 1) if types[i] == t], 6)
        bands = [b for b in (p[:2], p[2:4], p[4:]) if b]
        out[t] = (bands, spread([(a, 3) for a in range(len(types) - 2) if types[a + 1] == t], 2))
    return out


def pick(cost, recipes: dict, bands: list, triples: list, n: int = 3) -> dict:
    """{"pair": names, "triple": names} worth sweeping: uniform, the best split against it per band, then the best overall,
    up to `n`; for triples, uniform and the best split. cost(segment, name): KL above the floor."""
    names = [k for k in recipes if k != "uniform"]

    def ratio(name: str, segments: list) -> float:
        return statistics.median(cost(s, name) / max(cost(s, "uniform"), 1e-9) for s in segments)

    pairs = [s for b in bands for s in b]
    chosen = list(dict.fromkeys(min(names, key=lambda k: ratio(k, b)) for b in bands))
    chosen += [k for k in sorted(names, key=lambda k: ratio(k, pairs)) if k not in chosen] if pairs else []
    return {"pair": ["uniform", *chosen[:n]],
            "triple": ["uniform", min(names, key=lambda k: ratio(k, triples))] if triples else ["uniform"]}


def tilings(L: int, options: dict, k: int = 1) -> dict:
    """{steps: [(summed cost, summed size, [(start, n, option)])]}: for each step count, the k splits of layers 0..L-1
    into single layers and measured segments with the least summed cost (ties: least size), best first, each segment
    taking its cheapest option. options: {(start, n): {option: (size, cost)}}."""
    best = [{} for _ in range(L + 1)]      # best[p][s] = the k best (cost, size, how layer p was reached in s steps)
    best[0][0] = [(0.0, 0.0, None)]
    for p in range(L + 1):
        for s in best[p]:                  # every way into p is in: keep the k best
            best[p][s] = sorted(best[p][s], key=lambda w: w[:2])[:k]
        if p == L:
            break
        for s, ways in best[p].items():
            for n in (1, 2, 3):
                opts = {None: (0.0, 0.0)} if n == 1 else options.get((p, n))
                if not opts or p + n > L:
                    continue
                option, (dz, dc) = min(opts.items(), key=lambda kv: (kv[1][1], kv[1][0]))
                for i, (c, z, _) in enumerate(ways):
                    best[p + n].setdefault(s + 1, []).append((c + dc, z + dz, (p, s, i, n, option)))
    out = {}
    for steps, ways in best[L].items():
        out[steps] = []
        for c, z, back in ways:
            segments = []
            while back is not None:
                p, s, i, n, option = back
                segments.append((p, n, option))
                back = best[p][s][i][2]
            out[steps].append((c, z, segments[::-1]))
    return out


def refine(cost, shape: dict, cap: float, keep: dict, rounds: int, step: int = 5) -> tuple:
    """(keep, its cost) after a local search from `keep`, every split within `cap`. Each round tries each projection
    `step` points lower, then moves budget from the projection whose cut cost least per size to one whose cut cost most,
    and takes the cheapest of these if it beats the current split. cost(keep): its KL; it is called in a fixed order, so
    rerunning on the saved measurements repeats the search."""
    now = cost(keep)
    for _ in range(rounds):
        cuts = {p: {**keep, p: keep[p] - step} for p in keep if keep[p] > step}
        tries = [(cost(c), c) for c in cuts.values()]
        per_size = {p: (kl - now) / max(size(shape, keep) - size(shape, c), 1e-12) for p, (kl, c) in zip(cuts, tries)}
        moved = _moved(shape, cap, keep, per_size, step) if per_size else None
        tries += [(cost(moved), moved)] if moved else []
        if not tries or min(t[0] for t in tries) >= now:
            break
        now, keep = min(tries, key=lambda t: t[0])
    return keep, now


def _moved(shape: dict, cap: float, keep: dict, per_size: dict, step: int) -> "dict | None":
    """keep with the projection whose cut cost least per size `step` points lower, and the size it frees given to the
    measured projection with the highest cost per size that can take more; None if none can."""
    src = min(per_size, key=per_size.get)
    cut = {**keep, src: keep[src] - step}
    for dst in sorted((p for p in per_size if p != src), key=lambda p: -per_size[p]):
        fits = [y for y in range(keep[dst] + 1, 101) if size(shape, {**cut, dst: y}) <= cap]
        if fits:
            return {**cut, dst: fits[-1]}
    return None


def plan(model: str, segments: list, keep_of) -> dict:
    """The plan of [(start, n, option)]: groups of parallel layers and what each replica keeps. keep_of(option): its keep."""
    grouped = [(a, n, o) for a, n, o in segments if n > 1]
    return {"model": model, "groups": [list(range(a, a + n)) for a, n, _ in grouped],
            "keep": {str(a + n - 2): keep_of(o) for a, n, o in grouped}}
