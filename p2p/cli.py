"""p2p scan | kl | eval | report: find a plan for given steps and replica size, and check it."""
from __future__ import annotations

import argparse

from p2p.evaluate import evaluate, report
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

    s = sub.add_parser("scan", help="find the least damaging plan for --steps and --replica-size")
    scored(s)
    s.add_argument("--steps", type=int, required=True)
    s.add_argument("--replica-size", type=float, required=True, help="largest replica, as a fraction of a layer")
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
    s.set_defaults(fn=evaluate)
    s = sub.add_parser("report", help="accuracy of generation directories")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--against", default="")
    s.set_defaults(fn=report)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
