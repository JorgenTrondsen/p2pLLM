"""p2p scan: the least damaging plan for given steps and replica size. p2p kl: the damage of a given plan."""
from __future__ import annotations

import glob
import json
import os

from p2p import data, search
from p2p.engine import open_model, parallel, solution


def _items(a, tok) -> list:
    """The first a.n problems with at least a.min_sol solution tokens: {"id", "prompt", "sol"}, shortest first."""
    questions = {r["id"]: r["question"] for r in data.load(a.data)}
    rows = sorted((r for r in data.lines(a.solutions) if len(r["ids"]) >= a.min_sol), key=lambda r: r["id"])[:a.n]
    items = [{"id": r["id"], "prompt": data.prompt_ids(tok, questions[r["id"]]), "sol": r["ids"][:a.max_sol]} for r in rows]
    return sorted(items, key=lambda r: len(r["prompt"]) + len(r["sol"]))


def _prepare(a) -> None:
    """Shapes, exact solutions, calibration sums and the reference, each only if missing."""
    with open_model(a) as m:
        m.rpc("p2p_shapes", path=f"{a.out}/shapes.json")
        if not glob.glob(a.solutions):
            rows = data.load(a.data)[:a.n]
            outs = m.generate([data.prompt_ids(m.tok, r["question"]) for r in rows], a.max_sol)
            with open(a.solutions, "w") as f:
                for r, o in zip(rows, outs):
                    f.write(json.dumps({"id": r["id"], "ids": solution(o)}) + "\n")
        if a.calib_text and not os.path.exists(a.calib):
            m.calibrate(data.windows(a.calib_text, m.tok, a.calib_n, a.calib_len), a.calib)
        if not os.path.exists(a.reference):
            m.record_reference(_items(a, m.tok), a.reference)


def _setup(a) -> None:
    a.calib = a.calib or f"{a.out}/calib.pt"
    a.solutions = a.solutions or f"{a.out}/solutions.jsonl"
    a.reference = f"{a.out}/reference_{a.n}_{a.max_sol}_{a.min_sol}.pt"
    os.makedirs(a.out, exist_ok=True)
    needed = [f"{a.out}/shapes.json", a.reference] + ([a.calib] if a.calib_text else [])
    if not glob.glob(a.solutions) or not all(map(os.path.exists, needed)):
        parallel(a, _prepare, [(a,)])


def _measure(a, arms: list, k: int) -> None:
    with open_model(a) as m:
        if os.path.exists(a.calib):        # plans that prune nothing don't need it
            m.rpc("p2p_load_calibration", path=a.calib)
        m.rpc("p2p_load_reference", path=a.reference)
        items = _items(a, m.tok)
        for arm in arms:
            row = {**m.measure(items, arm.get("plan")), **arm}
            with open(f"{a.out}/results_{k}.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{arm['name']}] KL {row['kl']:.5f}  top-1 {row['top1']:.4f}", flush=True)


def measure(a, arms: list) -> dict:
    """Measure the arms not yet in the results; {arm name: result}."""
    results = lambda: {r["name"]: r for r in data.lines(f"{a.out}/results_*.jsonl")}
    todo = [x for x in arms if x["name"] not in results()]
    workers = min(len(a.gpus) // a.tp, len(todo))
    if todo:
        parallel(a, _measure, [(a, todo[k::workers], k) for k in range(workers)])
    return results()


def scan(a) -> int:
    _setup(a)
    shapes = {int(i): s for i, s in json.load(open(f"{a.out}/shapes.json")).items()}
    types = search.kinds([shapes[i] for i in sorted(shapes)])
    L, shape_of = len(types), {t: shapes[types.index(t)]["proj"] for t in types}
    recipes = {t: search.splits(shape_of[t], a.replica_size) for t in shape_of}
    reps = search.representatives(types)

    def arm(start: int, n: int, name: str) -> dict:
        t = types[start + n - 2]
        return {"name": f"{start}x{n}:{name}", "start": start, "n": n, "recipe": name, "size": search.size(shape_of[t], recipes[t][name]),
                "plan": search.plan(a.model, [(start, n, name)], recipes[t].get)}

    stage1 = [{"name": "exact"}] + [arm(*s, name) for t, (bands, triples) in reps.items()
                                    for s in [x for b in bands for x in b] + triples for name in recipes[t]]
    got = measure(a, stage1)
    floor = got["exact"]["kl"]
    cost = lambda s, name: max(0.0, got[f"{s[0]}x{s[1]}:{name}"]["kl"] - floor)
    menu = {t: search.pick(cost, recipes[t], *reps[t]) for t in reps if reps[t][0] or reps[t][1]}
    print("menu", json.dumps(menu), flush=True)
    sweep = [arm(s, n, name) for n, kind in ((2, "pair"), (3, "triple")) for s in range(L - n + 1)
             if types[s + n - 2] in menu for name in menu[types[s + n - 2]][kind]]
    got = measure(a, stage1 + sweep)
    options = {}
    for r in got.values():
        if "start" in r and r["size"] <= a.replica_size + 1e-9:
            options.setdefault((r["start"], r["n"]), {})[r["name"]] = (r["size"], max(0.0, r["kl"] - floor))
    best = search.tilings(L, options)
    if a.steps not in best:
        raise SystemExit(f"--steps {a.steps} is out of reach: {min(best)} to {max(best)}")
    predicted, _, segments = best[a.steps]
    plan = search.plan(a.model, segments, lambda name: next(iter(got[name]["plan"]["keep"].values())))
    json.dump(plan, open(f"{a.out}/plan.json", "w"), indent=0)
    print(f"plan -> {a.out}/plan.json: {len(plan['groups'])} groups, {a.steps} steps, predicted KL {floor + predicted:.5f}", flush=True)
    whole = {"name": f"plan_S{a.steps}", "plan": plan}
    print(f"measured KL {measure(a, [whole])[whole['name']]['kl']:.5f}, floor {floor:.5f}", flush=True)
    return 0


def kl(a) -> int:
    """KL of one plan (and the exact floor) against the exact model."""
    _setup(a)
    plan = json.load(open(a.plan))
    got = measure(a, [{"name": "exact"}, {"name": os.path.basename(a.plan), "plan": plan}])
    floor, r = got["exact"]["kl"], got[os.path.basename(a.plan)]
    print(f"{r['name']}: KL {r['kl']:.5f}  above floor {r['kl'] - floor:.5f}  top-1 {r['top1']:.4f}  floor {floor:.5f}  {r['tokens']} tokens")
    return 0
