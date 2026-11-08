"""p2p eval: full generation under a plan, with accuracy. p2p report: accuracy of generation directories, paired."""
from __future__ import annotations

import json
import os

from scipy.stats import binomtest

from p2p import data
from p2p.engine import open_model, parallel, solution


def _generate(a, rows: list, k: int) -> None:
    with open_model(a) as m:
        if a.plan:
            m.rpc("p2p_load_calibration", path=a.calib)
            m.rpc("p2p_plan", plan=json.load(open(a.plan)))
        for b in range(0, len(rows), 4 * a.batch):
            chunk = rows[b: b + 4 * a.batch]
            outs = m.generate([data.prompt_ids(m.tok, r["question"]) for r in chunk], a.max_new)
            with open(f"{a.out}/shard_{k}.jsonl", "a") as f:
                for r, o in zip(chunk, outs):
                    f.write(json.dumps({"id": r["id"], "correct": None if r["gold"] is None else data.correct(o["text"], r["gold"]),
                                        "n_tok": len(solution(o)), "truncated": o["meta_info"]["finish_reason"]["type"] == "length",
                                        "passes": o["meta_info"].get("spec_verify_ct"),    # speculative decoding only
                                        "completion_tokens": o["meta_info"]["completion_tokens"],
                                        "ids": solution(o), "text": o["text"]}) + "\n")
            print(f"[{k}] {b + len(chunk)}/{len(rows)}", flush=True)


def evaluate(a) -> int:
    """Full generation under a plan (no --plan: the exact model); resumes per chunk."""
    os.makedirs(a.out, exist_ok=True)
    done = {r["id"] for r in data.lines(f"{a.out}/shard_*.jsonl")}
    rows = [r for r in data.load(a.data) if r["id"] not in done][:a.n or None]
    workers = min(len(a.gpus) // a.tp, max(len(rows), 1))
    if rows:
        parallel(a, _generate, [(a, rows[k::workers], k) for k in range(workers)])
    a.dirs = [a.out]
    return report(a)


def mcnemar_p(lost: int, won: int) -> float:
    """Exact two-sided binomial test on the problems only one side got right."""
    return binomtest(lost, lost + won, 0.5).pvalue if lost + won else 1.0


def report(a) -> int:
    """Accuracy of each generation directory, paired against --against."""
    load = lambda d: {r["id"]: r for r in data.lines(f"{d}/shard_*.jsonl")}
    ref = load(a.against) if a.against else None
    for d in a.dirs:
        rows = load(d)
        scored = [r for r in rows.values() if r["correct"] is not None]
        line = f"{d}: n {len(rows)}  acc {sum(r['correct'] for r in scored) / max(len(scored), 1):.3f}"
        line += f"  truncated {sum(r['truncated'] for r in rows.values()) / max(len(rows), 1):.3f}"
        spec = [r for r in rows.values() if r.get("passes")]      # tokens per verify pass, as SGLang counts them
        if spec:
            line += f"  tokens per pass {sum(r['completion_tokens'] for r in spec) / sum(r['passes'] for r in spec):.2f}"
        if ref:
            both = [i for i in rows if i in ref and rows[i]["correct"] is not None]
            lost = sum(ref[i]["correct"] and not rows[i]["correct"] for i in both)
            won = sum(rows[i]["correct"] and not ref[i]["correct"] for i in both)
            same = sum(rows[i]["ids"] == ref[i]["ids"] for i in both) / max(len(both), 1)
            line += f"  vs {a.against}: {100 * (won - lost) / max(len(both), 1):+.1f} points, lost {lost} won {won}, p {mcnemar_p(lost, won):.3f}, identical {same:.3f}"
        print(line)
    return 0
