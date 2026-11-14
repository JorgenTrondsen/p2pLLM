"""p2p scan | kl | eval | report | replicas: find a plan for given steps and replica size, check it, and prepare its
replicas for deployment."""
from __future__ import annotations

import argparse

from p2p.evaluate import evaluate, report
from p2p.prepare import replicas
from p2p.scan import kl, scan


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--model", required=True)
        s.add_argument("--data", default="gsm8k", help="gsm8k, or a .jsonl or .parquet file with question and answer columns")
        s.add_argument("--out", required=True)
        s.add_argument("--gpus", default="0,1,2,3", type=lambda x: [int(g) for g in x.split(",")])
        s.add_argument("--tp", type=int, default=1, help="GPUs per worker: the model must fit")
        s.add_argument("--batch", type=int, default=32)
        s.add_argument("--mem-fraction", type=float, default=0.85)
        s.add_argument("--max-total-tokens", type=int, default=0, help="KV pool size; leaves room for replicas")
        s.add_argument("--calib", default="", help="Wanda's calibration sums; scan and kl default to <out>/calib.pt")

    def scored(s):
        common(s)
        s.add_argument("--n", type=int, default=64, help="problems KL is measured on")
        s.add_argument("--max-sol", type=int, default=256, help="solution tokens scored per problem")
        s.add_argument("--min-sol", type=int, default=0, help="only problems with this many solution tokens")
        s.add_argument("--solutions", default="", help="existing solutions, a glob of JSONL with id and ids")
        s.add_argument("--calib-text", default="", help="a .jsonl or .parquet with a text column")
        s.add_argument("--calib-n", type=int, default=128)
        s.add_argument("--calib-len", type=int, default=2048)
        s.add_argument("--draft-k", type=int, default=16, help="score in passes of k drafted tokens plus one, as speculative "
                       "decoding runs them, with replicas on their own state inside a pass (0: one token a step, for plans "
                       "that run without drafting)")
        s.set_defaults(draft="forced")

    s = sub.add_parser("scan", help="find the least damaging plan for --steps and --replica-size")
    scored(s)
    s.add_argument("--steps", type=int, required=True)
    s.add_argument("--replica-size", type=float, required=True, help="largest replica, as a fraction of a layer")
    s.add_argument("--candidates", type=int, default=16, help="plans measured whole, best by summed KL first; the lowest "
                   "is kept (1: the best by sum)")
    s.add_argument("--refine-rounds", type=int, default=3, help="rounds of searching each group's split freely, from its "
                   "best menu split (0: the menu only)")
    s.set_defaults(fn=scan)
    s = sub.add_parser("kl", help="KL of one plan against the exact model")
    scored(s)
    s.add_argument("--plan", required=True)
    s.set_defaults(fn=kl)
    s = sub.add_parser("eval", help="full generation under a plan, with accuracy")
    common(s)
    s.add_argument("--plan", default="")
    s.add_argument("--max-new", type=int, default=1024)
    s.add_argument("--n", type=int, default=0, help="only the first n problems")
    s.add_argument("--against", default="", help="another generation directory to pair with")
    s.add_argument("--draft", default="", help="speculative decoding: mtp, mtp:PATH (a native drafter), or a small model's path")
    s.add_argument("--draft-k", type=int, default=16, help="tokens drafted per pass (a tree's depth)")
    s.add_argument("--draft-topk", type=int, default=1, help="candidates kept per draft step: above 1, a tree")
    s.add_argument("--draft-tokens", type=int, default=0, help="tokens a tree's pass verifies, root included "
                   "(default: a chain's k + 1)")
    s.set_defaults(fn=evaluate)
    s = sub.add_parser("replicas", help="prune a plan's replicas ahead of time, as a peer node does before deployment")
    s.add_argument("--model", required=True)
    s.add_argument("--plan", required=True)
    s.add_argument("--calib", required=True, help="the scan's calibration sums")
    s.add_argument("--out", required=True)
    s.add_argument("--layers", default="", type=lambda x: [int(i) for i in x.split(",")] if x else [],
                   help="only these copied layers (default: every replica of the plan); SGLang loads every layer from "
                        "the first to the last of them, on one GPU")
    s.add_argument("--dtype", default="auto", help="as the deployment loads the model")
    s.add_argument("--mem-fraction", type=float, default=0.85)
    s.set_defaults(fn=replicas)
    s = sub.add_parser("report", help="accuracy of generation directories")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--against", default="")
    s.set_defaults(fn=report)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
