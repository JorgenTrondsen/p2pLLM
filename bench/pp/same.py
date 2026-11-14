"""A plan on one GPU, then on pipeline stages with some of its groups spread (p2p.route): the same tokens?

    python bench/pp/same.py --model M --plan plan.json --calib calib.pt --spread 0 --split 34 --pp 4 [--draft mtp] --out same.json

GSM8K questions one at a time, so both run the same batches, greedy. One GPU first (the first visible), then --pp stages.
Writes, per question, its answer's length and the first token where the two differ (None if none).
"""
import argparse
import json
import os


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--spread", required=True)
    p.add_argument("--split", default="")
    p.add_argument("--pp", type=int, default=4)
    p.add_argument("--draft", default="")
    p.add_argument("--draft-k", type=int, default=16)
    p.add_argument("--draft-topk", type=int, default=1, help="candidates kept per draft step: above 1, a tree")
    p.add_argument("--draft-tokens", type=int, default=0, help="tokens a tree's pass verifies (default k + 1)")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--max-total-tokens", type=int, default=4096)
    p.add_argument("--mem-fraction", type=float, default=0.88)
    p.add_argument("--replicas", default="", help="replicas prepared ahead of time (p2p replicas): needed if the plan prunes")
    p.add_argument("--replica-form", default="compact")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from p2p import data
    from p2p.engine import Model, speculative
    plan = json.load(open(a.plan))
    questions = data.load("gsm8k")[:a.n]
    ids = {}
    for where in ("one", "spread"):
        extra = {}
        if where == "spread":
            os.environ.update(P2P_PP_COMM="fused", P2P_PP_SPREAD=a.spread, P2P_PP_SPLIT=a.split, P2P_PP_PLAN=a.plan,
                              P2P_PP_CALIB=a.calib)
            if a.replicas:
                os.environ.update(P2P_PP_REPLICAS=a.replicas, P2P_PP_REPLICA_FORM=a.replica_form)
            if a.draft:
                os.environ["SGLANG_ENABLE_PP_SPEC"] = "1"
            extra = {"pp_size": a.pp}
        m = Model(a.model, tp=1, batch=1, max_total_tokens=a.max_total_tokens, mem_fraction_static=a.mem_fraction,
                  skip_server_warmup=True, **speculative(a.model, a.draft, a.draft_k if a.draft else 0, a.draft_topk, a.draft_tokens), **extra)
        try:
            m.rpc("p2p_load_calibration", path=a.calib)
            m.rpc("p2p_plan", plan=plan)
            ids[where] = [m.generate([data.prompt_ids(m.tok, q["question"])], a.max_new)[0]["output_ids"]
                          for q in questions]
        finally:
            m.close()
    rows = []
    for q, x, y in zip(questions, ids["one"], ids["spread"]):
        first = next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), None)
        if first is None and len(x) != len(y):
            first = min(len(x), len(y))
        rows.append({"question": q["id"], "tokens": len(x), "first_difference": first})
    out = {"model": os.path.basename(a.model.rstrip("/")), "draft": a.draft, "draft_topk": a.draft_topk,
           "draft_tokens": a.draft_tokens, "spread": a.spread, "split": a.split,
           "replicas": a.replica_form if a.replicas else "",
           "same": all(r["first_difference"] is None for r in rows), "rows": rows}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("draft", "draft_topk", "draft_tokens", "same")}), [r["first_difference"] for r in rows])


if __name__ == "__main__":
    main()
