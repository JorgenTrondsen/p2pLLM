"""p2p scan: the least damaging plan for given steps and replica size. p2p kl: the damage of a given plan."""
from __future__ import annotations

import contextlib
import glob
import hashlib
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
    a.results = f"{a.out}/passes{a.draft_k}" if a.draft_k else f"{a.out}/results"     # the two never mix
    os.makedirs(a.out, exist_ok=True)
    needed = [f"{a.out}/shapes.json", a.reference] + ([a.calib] if a.calib_text else [])
    if not glob.glob(a.solutions) or not all(map(os.path.exists, needed)):
        parallel(a, _prepare, [(a,)])


def _results(a) -> dict:
    return {r["name"]: r for r in data.lines(f"{a.results}_*.jsonl")}


@contextlib.contextmanager
def _measurer(a, k: int):
    """run(arm) -> its result: saved before, or measured on this worker's engine and saved to its results file."""
    saved = _results(a)
    with open_model(a) as m:
        if os.path.exists(a.calib):        # plans that prune nothing don't need it
            m.rpc("p2p_load_calibration", path=a.calib)
        m.rpc("p2p_load_reference", path=a.reference)
        items = _items(a, m.tok)

        def run(arm: dict) -> dict:
            if arm["name"] not in saved:
                saved[arm["name"]] = row = {**m.measure(items, arm.get("plan")), **arm}
                with open(f"{a.results}_{k}.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{arm['name']}] KL {row['kl']:.5f}  top-1 {row['top1']:.4f}", flush=True)
            return saved[arm["name"]]
        yield run


def _measure(a, arms: list, k: int) -> None:
    with _measurer(a, k) as run:
        for arm in arms:
            run(arm)


def measure(a, arms: list) -> dict:
    """Measure the arms not yet in the results; {arm name: result}."""
    todo = [x for x in arms if x["name"] not in _results(a)]
    workers = min(len(a.gpus) // a.tp, len(todo))
    if todo:
        parallel(a, _measure, [(a, todo[k::workers], k) for k in range(workers)])
    return _results(a)


def _arm(model: str, start: int, n: int, shape: dict, keep: dict, name: str = "") -> dict:
    """Layers start..start+n-1 in one step, the rest exact, with the replica keeping `keep` (named by its percentages)."""
    name = name or "".join(f"{p}{v}" for p, v in sorted(keep.items()))
    return {"name": f"{start}x{n}:{name}", "start": start, "n": n, "recipe": name, "size": search.size(shape, keep),
            "plan": search.plan(model, [(start, n, name)], lambda _: keep)}


def _refined(a, run, job: tuple) -> tuple:
    start, n, shape, keep = job
    return search.refine(lambda k: run(_arm(a.model, start, n, shape, k))["kl"], shape, a.replica_size, keep, a.refine_rounds)


def _refine(a, jobs: list, k: int) -> None:
    with _measurer(a, k) as run:
        for job in jobs:
            _refined(a, run, job)


def refine(a, jobs: list) -> list:
    """(keep, KL) per job (start, n, shape, keep) after search.refine, its measurements spread over the workers. The
    search is replayed on the saved results, so a finished one costs nothing and a stopped one resumes."""
    def replay():
        got = _results(a)
        return [_refined(a, lambda arm: got[arm["name"]], job) for job in jobs]
    try:
        return replay()
    except KeyError:
        workers = min(len(a.gpus) // a.tp, len(jobs))
        parallel(a, _refine, [(a, jobs[k::workers], k) for k in range(workers)])
        return replay()


def _whole(steps: int, plan: dict) -> dict:
    return {"name": f"plan_S{steps}_{hashlib.sha1(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:8]}", "plan": plan}


def scan(a) -> int:
    _setup(a)
    shapes = {int(i): s for i, s in json.load(open(f"{a.out}/shapes.json")).items()}
    types = search.kinds([shapes[i] for i in sorted(shapes)])
    L, shape_of = len(types), {t: shapes[types.index(t)]["proj"] for t in types}
    recipes = {t: search.splits(shape_of[t], a.replica_size) for t in shape_of}
    reps = search.representatives(types)

    def arm(start: int, n: int, name: str) -> dict:
        t = types[start + n - 2]
        return _arm(a.model, start, n, shape_of[t], recipes[t][name], name)

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
    for r in (got[x["name"]] for x in stage1[1:] + sweep):     # the menu's arms only, so later stages never move the plan
        if r["size"] <= a.replica_size + 1e-9:
            options.setdefault((r["start"], r["n"]), {})[r["name"]] = (r["size"], max(0.0, r["kl"] - floor))
    best = search.tilings(L, options, a.candidates).get(a.steps)
    if not best:
        raise SystemExit(f"--steps {a.steps} is out of reach")
    keep_of = lambda name: next(iter(got[name]["plan"]["keep"].values()))
    whole = [_whole(a.steps, search.plan(a.model, segments, keep_of)) for _, _, segments in best]
    got = measure(a, whole)
    for (predicted, _, _), w in zip(best, whole):
        print(f"{w['name']}: predicted KL {floor + predicted:.5f}, measured {got[w['name']]['kl']:.5f}", flush=True)
    i = min(range(len(best)), key=lambda j: got[whole[j]["name"]]["kl"])
    predicted, plan, kl = floor + best[i][0], whole[i]["plan"], got[whole[i]["name"]]["kl"]
    print(f"kept {whole[i]['name']}, {i + 1} of {len(best)} by summed KL", flush=True)
    if a.refine_rounds:
        groups = [(s, n, name) for s, n, name in best[i][2] if n > 1]
        found = refine(a, [(s, n, shape_of[types[s + n - 2]], keep_of(name)) for s, n, name in groups])
        w = _whole(a.steps, search.plan(a.model, [(s, n, j) for j, (s, n, _) in enumerate(groups)], lambda j: found[j][0]))
        measured = measure(a, [w])[w["name"]]["kl"]
        print(f"refined splits: measured KL {measured:.5f} against {kl:.5f}, {'kept' if measured < kl else 'dropped'}", flush=True)
        if measured < kl:
            predicted, plan, kl = floor + sum(max(0.0, c - floor) for _, c in found), w["plan"], measured
    json.dump(plan, open(f"{a.out}/plan.json", "w"), indent=0)
    print(f"plan -> {a.out}/plan.json: {len(plan['groups'])} groups, {a.steps} steps, predicted KL {predicted:.5f}", flush=True)
    print(f"measured KL {kl:.5f}, floor {floor:.5f}", flush=True)
    return 0


def kl(a) -> int:
    """KL of one plan (and the exact floor) against the exact model."""
    _setup(a)
    plan = json.load(open(a.plan))
    got = measure(a, [{"name": "exact"}, {"name": os.path.basename(a.plan), "plan": plan}])
    floor, r = got["exact"]["kl"], got[os.path.basename(a.plan)]
    print(f"{r['name']}: KL {r['kl']:.5f}  above floor {r['kl'] - floor:.5f}  top-1 {r['top1']:.4f}  floor {floor:.5f}  {r['tokens']} tokens")
    return 0
