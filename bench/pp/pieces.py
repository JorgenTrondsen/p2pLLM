"""A streamed run's pieces on each rank's GPU and the head's own time by part, from overhead.py's GPU marks.

    python -I bench/pp/pieces.py runs/zero/q38x4_spans

Medians over the run's whole passes: per rank before the head, its GPU on a pass's first piece and on a later one, and
how long it sat idle before a later piece; for the head, a pass from its rows going out to the next pass's, by part.
"""
import glob
import json
import statistics as st
import sys


def main() -> None:
    out = sys.argv[1]
    files = sorted(glob.glob(f"{out}/spans/spans_pp*.json"), key=lambda f: int(f.rsplit("pp", 1)[1].split(".")[0]))
    sp = [sorted((s, n) for n, s, _ in json.load(open(f)) if n.startswith("gpu:")) for f in files]
    med = lambda xs: st.median(xs) if xs else float("nan")
    ms = lambda ns: ns / 1e6

    def passes(marks: list, start: str) -> list:
        rows, cur = [], None
        for t, n in marks:
            if n == start:
                cur = []
                rows.append(cur)
            if cur is not None:
                cur.append((t, n))
        return rows[1:-1]                   # whole passes only

    for r, marks in enumerate(sp[:-1]):
        first, rest, copy, own, wait, total, count = [], [], [], [], [], [], []
        for p in passes(marks, "gpu:in"):
            ins, outs = ([t for t, n in p if n == name] for name in ("gpu:piece_in", "gpu:piece"))
            cops = [t for t, n in p if n == "gpu:copied"]
            if len(ins) != len(outs) or not ins:
                continue                    # a pass this rank ran whole
            count.append(len(ins))
            for k, (a, b) in enumerate(zip(ins, outs)):
                (rest if k else first).append(ms(b - a))
                if k:
                    wait.append(ms(a - outs[k - 1]))
            if len(cops) == len(ins):
                copy.append(ms(cops[0] - ins[0]))
                own.append(ms(outs[0] - cops[0]))
            total.append(ms(outs[-1] - ins[0]))
        if not total:
            print(f"rank {r}: ran its passes whole")
            continue
        print(f"rank {r}: {len(total)} passes of {med(count):.0f} pieces; GPU on the first piece {med(first):.2f} ms"
              + (f" (the replica {med(copy):.2f}, its own layer {med(own):.2f})" if copy else "")
              + f", on a later piece {med(rest):.2f}; idle before a later piece {med(wait):.2f}; "
              f"first piece in to last out {med(total):.1f}")
    head = sp[-1]
    parts, runs = {}, []
    rows = passes(head, "gpu:out")          # from a pass's last rows going out to the next pass's
    for p in rows:
        at = lambda name: next((t for t, n in p if n == name), None)
        ins, outs = ([t for t, n in p if n == name] for name in ("gpu:piece_in", "gpu:piece"))
        if not outs or at("gpu:draft-") is None:
            continue
        nxt = head[head.index(p[-1]) + 1][0]
        settled = outs[-1]
        for name, value in (("rows out to the first piece in", ins[0] - p[0][0]), ("first piece in to settled", settled - ins[0]),
                            ("settled to verified", at("gpu:verify-") - settled),
                            ("extending the draft", at("gpu:extend-") - at("gpu:extend+")),
                            ("draft", at("gpu:draft-") - at("gpu:draft+")),
                            ("draft to the last rows out", nxt - at("gpu:draft-")),
                            ("settled to the last rows out", nxt - settled), ("pass", nxt - p[0][0])):
            parts.setdefault(name, []).append(ms(value))
        if at("gpu:phase") is not None:
            parts.setdefault("settled to the first rows out", []).append(ms(at("gpu:phase") - settled))
        parts.setdefault("runs", []).append(len(outs))
        last = None
        for o in outs:
            runs.append(ms(o - max(max(x for x in ins if x < o), last or 0)))
            last = o
    print(f"head: {len(parts.get('pass', []))} passes; " + "; ".join(f"{k} {med(v):.2f}" for k, v in parts.items())
          + f"; a run on its GPU {med(runs):.2f} ms")


if __name__ == "__main__":
    main()
