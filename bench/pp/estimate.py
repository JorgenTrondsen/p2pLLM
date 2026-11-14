"""Decode speed with one layer per node over a network, estimated from parts measured on the GPUs.

    python -I bench/pp/estimate.py --plan plan.json --trees runs/pool/k24_t4_all/pool.json [--hop-ms 20] [--mbps 150]
        [--head-own-ms 54.8] [--schedule 1,2,4] [--show 32,64]

A streamed pass's first piece pays every hop's delay, one piece over each link and each node's GPU on it; the later
pieces follow at the pace of the busiest link or node; then the head runs its layer and its own part. Inputs: the
tokens a pass yields and the rows the head needs, per tree size (--trees: trees.py --pool), and from --runs
(overhead.py's GPU marks) how late the first step starts and the head's times. Prints tokens/s by tree size with each
change added in turn (the head stops when settled, the embedding streamed, the slowest node served first, pieces of
--schedule, the first rows sent while drafting), and what the delays alone would give.
"""
import argparse
import json
import os
import sys


def _load(runs: str, name: str) -> dict:
    return json.load(open(os.path.join(runs, name, "summary.json")))


def _per_token(members: list, hidden: int, mbps: float, per_value: list) -> list:
    """ms the busiest link of each hop takes for each token of a pass: the head's embedding to the first step, then
    from each step to the next (members[i] ranks in step i, the last the head's); per_value: bytes a value as the first
    member, the other members, a whole layer's rank and the head send it."""
    if not mbps:
        return [0.0] * len(members)
    first, other, single, embed = per_value
    ms = lambda nbytes: nbytes * 8 / (mbps * 1e3)
    out = [ms(members[0] * hidden * embed)]
    for i in range(1, len(members)):
        sent = [hidden * (single if members[i - 1] == 1 else first if k == 0 else other) for k in range(members[i - 1])]
        out.append(ms(max(members[i] * max(sent), sum(sent))))
    return out


FIXES = ("streamed as before", "+ the head stops when settled", "+ the embedding streamed",
         "+ the slowest node served first", "+ pieces of {schedule}, one tensor from a single layer",
         "+ the first rows sent while drafting ({phases})")


def _bounds(n: int, sizes: list, cuts=()) -> list:
    """Where each piece of a pass of n tokens ends: a piece of each size, the last size repeated, and at each of cuts
    (p2p.stream.pieces)."""
    out, at = [], 0
    for stop in [c for c in cuts if c < n] + [n]:
        while at < stop:
            at = min(stop, at + sizes[min(len(out), len(sizes) - 1)])
            out.append(at)
    return out


def _fixed(n: int, hist: list, members: list, hidden: int, a, h: dict, fixes: int, phases=()) -> float:
    """ms a pass of n tokens takes streamed with the first `fixes` of FIXES after the first (see above); hist[r]: the
    share of passes in which the head needs r rows after the root; phases: [(draft steps, rows)] of a phased tree."""
    early, embed, slowest, paced, phased = (fixes >= k for k in range(1, 6))
    phases = list(phases) if phased else []
    per_value = [a.bytes[0], a.bytes[1], a.bytes[0] if paced else a.bytes[2], a.bytes[3]]
    every = _per_token(members, hidden, a.mbps, per_value)          # a token to every rank of the next step
    ms = lambda nbytes: nbytes * 8 / (a.mbps * 1e3) if a.mbps else 0.0
    one = [ms(hidden * per_value[3])] + [ms(hidden * max(per_value[2] if members[i - 1] == 1 else per_value[0],
                                                           per_value[1])) for i in range(1, len(members))]
    bounds = _bounds(n, [int(x) for x in a.schedule.split(",")] if paced else [1], [rows for _, rows in phases])
    size = bounds[0]
    first = a.hop_ms * len(members)
    for j in range(len(members) - 1):                   # into step j, and its GPU on the piece
        link = size * every[j] if j or embed else n * every[0]
        if slowest:
            slow = a.piece_gpu_ms if members[j] > 1 else a.piece_fast_ms
            first += max(size * one[j] + slow, link + a.piece_fast_ms) if j or embed else link + slow
        else:
            first += link + a.piece_gpu_ms
    first += size * every[-1]                            # into the head
    busiest = max(every if embed else every[1:])
    # A piece on a node: its CPU sets it up and launches it, and its GPU verifies every row so far again.
    node = lambda k: max(a.piece_cpu_ms, a.piece_gpu_ms + a.piece_row_ms * bounds[k])
    # When the head has each phase's rows to send, after it has settled the pass before: the last (or only) one after
    # all its own time, an earlier one after the draft steps that fix it (the candidates of s steps take s - 1 forwards).
    leaves = [a.draft_pre_ms + (steps - 1) * a.draft_step_ms + a.phase_ms for steps, _ in phases] + [h["head_own"]]
    phase_of = lambda b: sum(b > rows for _, rows in phases)
    # Right behind a streamed pass's first rows, the head sends the next steps' ranks the pass's output (p2p.pp.NEAR).
    near = ms(sum(members[1:1 + a.near]) * a.output_kb * 1024) if embed else 0.0
    at, arrives = near, [0.0]                            # when each piece reaches the head, after the first
    for k in range(1, len(bounds)):
        at = max(at, leaves[phase_of(bounds[k])] - leaves[0]) + max((bounds[k] - bounds[k - 1]) * busiest, node(k))
        arrives.append(at)
    piece_of = lambda row: next(k for k, b in enumerate(bounds) if b > row)
    drain = sum(share * arrives[piece_of(r)] for r, share in enumerate(hist)) if early else arrives[-1]
    late = max(0.0, -h["first_ahead"])
    return first + drain + late + h["head_layer"] + leaves[0]


def _fixes(a, width: int, members: list, hidden: int, h: dict) -> None:
    got = json.load(open(a.trees))
    depth = f"sglang/{a.depth}"
    c, needed = {int(n): y for n, y in got["curves"][depth].items()}, got["needed"][depth]
    scale = a.tokens / got["curves"]["sglang/16"][str(width)]
    sizes = sorted(int(n) for n in needed)
    step = a.draft_step_ms * (a.depth - 16)
    print(f"\nStreamed in pieces, each change added in turn (SGLang's trees, {a.depth} deep; a piece "
          f"{a.piece_gpu_ms:g} ms on a replica rank's GPU, {a.piece_fast_ms:g} ms on a one-layer rank's, {a.piece_cpu_ms:g} ms "
          f"of a node's CPU); tokens/s, and ms a pass at {a.show}")
    print("| Fixes | " + " | ".join(f"N={n}" for n in sizes) + " | Best | " + " | ".join(f"ms at {n}" for n in a.show) + " |")
    print("|---|" + "---|" * (len(sizes) + 1 + len(a.show)))
    phases, phased = [tuple(x) for x in got.get("phases", [])], f"phased/{a.depth}"
    at = next(k for k, name in enumerate(FIXES) if "first rows" in name)
    for k, name in enumerate(FIXES):
        use = k >= at and phased in got["curves"]       # from here on, the trees and rows of a phased tree
        if k == at and not use:
            continue                                    # the pool wasn't read with --phases
        cs, hs = ({int(n): y for n, y in got["curves"][phased].items()}, got["needed"][phased]) if use else (c, needed)
        t = {n: _fixed(n, hs[str(n)], members, hidden, a, h, k, phases if use else ()) + step for n in sizes if str(n) in hs}
        vals = {n: 1000 * cs[n] * scale / t[n] for n in t}
        best = max((v, n) for n, v in vals.items())
        print(f"| {name.format(schedule=a.schedule, phases=','.join(f'{s}:{r}' for s, r in phases))} | "
              + " | ".join(f"{vals[n]:.2f}" if n in vals else "" for n in sizes)
              + f" | {best[0]:.2f} at {best[1]} | " + " | ".join(f"{t[n]:.0f}" if n in t else "" for n in a.show) + " |")
    print("delays alone: " + ", ".join(f"N={n}: {1000 * c[n] * scale / (a.hop_ms * len(members)):.2f}" for n in sizes))


def main() -> None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from p2p.route import layout, steps
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--trees", required=True, help="bench/pp/trees.py --pool's pool.json")
    p.add_argument("--runs", default="runs/ahead")
    p.add_argument("--hop-ms", type=float, default=20.0)
    p.add_argument("--mbps", type=float, default=150.0, help="each node's link, Mbit/s each way (0: the delay only)")
    p.add_argument("--bytes", default="2,2,4,2", type=lambda x: [float(v) for v in x.split(",")],
                   help="bytes a value is sent as: by a step's first member, its other members, a whole layer's rank, "
                        "the head (bfloat16, bfloat16, two bfloat16, bfloat16)")
    p.add_argument("--piece-gpu-ms", type=float, default=1.42, help="a replica rank's GPU for a pass's first piece "
                   "(runs/zero/q38x4_spans' marks: its replica 0.73 ms, its own layer 0.68)")
    p.add_argument("--piece-fast-ms", type=float, default=0.74, help="a one-layer rank's GPU for a pass's first piece")
    p.add_argument("--piece-row-ms", type=float, default=0.055, help="a replica rank's GPU for each row a later piece "
                   "verifies again (runs/zero/q38x4_spans_ph64: 3.2 ms a piece over 64 rows)")
    p.add_argument("--piece-cpu-ms", type=float, default=3.0, help="a node's CPU for a piece, setting it up and launching "
                   "it (runs/stream's spans: 2.4-3.9 ms)")
    p.add_argument("--schedule", default="1,2,4", help="tokens in each piece, the last repeated (P2P_PP_STREAM)")
    p.add_argument("--draft-pre-ms", type=float, default=5.7, help="from the head settling a pass to its draft starting "
                   "(verify 1.2 ms, extending the draft 4.5: runs/zero/q38x4_spans)")
    p.add_argument("--draft-step-ms", type=float, default=2.6, help="the head's drafting, a step")
    p.add_argument("--phase-ms", type=float, default=3.0, help="fixing a phase's rows and sending them")
    p.add_argument("--head-own-ms", type=float, default=0.0, help="the head's own time, from settling a pass to the next "
                   "one's rows going out whole (default: --runs' marks, a 32-token tree; a head of one layer with a "
                   "64-token tree: 54.8 ms, runs/zero/q38cut_spans)")
    p.add_argument("--near", type=int, default=4, help="the steps after the first whose ranks take a pass's output right "
                   "behind its first rows (P2P_PP_NEAR)")
    p.add_argument("--output-kb", type=float, default=6.2, help="a pass's output to one rank")
    p.add_argument("--depth", type=int, default=16, help="the trees' depth (draft steps)")
    p.add_argument("--show", default="32,64", type=lambda x: [int(v) for v in x.split(",")])
    p.add_argument("--tokens", type=float, default=8.35, help="tokens a pass yields at the runs' tree, on the plan")
    p.add_argument("--logits-ms", type=float, default=1.6, help="the head's logits, part of its forward")
    a = p.parse_args()
    plan = json.load(open(a.plan))
    config = json.load(open(os.path.join(plan["model"], "config.json")))
    text = config.get("text_config", config)
    layers, hidden = text["num_hidden_layers"], text["hidden_size"]
    members = [len(st) for st in steps(plan["groups"], layers)[:-1]] + [1]
    first, mid = _load(a.runs, "q38_ahead1"), _load(a.runs, "q38mid_ahead1")
    f, m, args = first["marks"], mid["marks"], mid["meta"]["args"]
    slow = max((r for r in f if f[r]["src"] == len(f) - 1), key=lambda r: f[r]["gap_ms"] + f[r]["window_ms"])
    head = str(len(m) - 1)
    r = layout(json.load(open(args["plan"]))["groups"], args["spread"], layers, args["split"])[-1]
    measured = {"first_ahead": f[slow]["ahead_ms"], "head_own": a.head_own_ms or m[head]["own_ms"],
                "head_layer": (m[head]["layers_ms"] - a.logits_ms) / (r.end - r.start) + a.logits_ms}
    print(f"{a.plan}: {layers} layers, {len(members)} steps, every step but the head's spread; hop {a.hop_ms:g} ms"
          + (f" and {a.mbps:g} Mbit/s, bytes a value {a.bytes}" if a.mbps else "") + f"; {a.tokens:g} tokens a pass at "
          f"the runs' {first['meta']['args']['draft_tokens']}-token tree")
    _fixes(a, first["meta"]["args"]["draft_tokens"], members, hidden, measured)


if __name__ == "__main__":
    main()
