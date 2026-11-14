"""A plan's steps across pipeline ranks (p2p.route). The layout on CPU; on four GPUs, one spread step against the same
plan on one GPU, for each tiny family and kind of copy (attention, linear attention, sliding window, routed experts),
and for the first pair, which the head embeds for. Requests go one at a time, so both run the same batches. The exact
model, full copies and prepared replicas expanded to dense must give one GPU's numbers exactly; compact replicas round
differently. Needs P2P_TEST_TOKENIZER, as test_sglang does:

    CUDA_VISIBLE_DEVICES=0,1,2,3 pytest tests/test_route.py -s
"""
import json
import os
import random
import time
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from p2p.route import Rank, layout, partners, sources, targets

GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_a_spread_pair_gets_a_rank_per_member():
    assert layout(GROUPS, "2", 8) == [Rank(0, 2, 0, 0, None, 0), Rank(2, 3, 1, 1, 0, 2), Rank(2, 4, 1, 1, 1, 3),
                                      Rank(4, 8, 2, 3, None, 4)]


def test_members_run_at_once_on_passes_and_in_order_in_prefill():
    ranks = layout(GROUPS, "2", 8)
    assert [sources(ranks, i, True) for i in range(4)] == [[3], [0], [0], [1, 2]]     # rank 0's input: the head's
    assert [targets(ranks, i, True) for i in range(4)] == [[2, 1], [3], [3], [0]]     # embedding of the next tokens;
    # the replica's rank, which runs two layers, is served first
    assert [sources(ranks, i, False) for i in range(4)] == [[3], [0], [1], [2]]    # prefill: each layer on its rank
    assert [targets(ranks, i, False) for i in range(4)] == [[1], [2], [3], [0]]
    assert [partners(ranks, i) for i in range(4)] == [(None, None), (2, None), (None, 1), (None, None)]


def test_the_head_embeds_for_a_spread_first_pair():
    ranks = layout(GROUPS, "0", 8, "4")                     # and the rest cut in two at layer 4
    assert ranks == [Rank(0, 1, 0, 0, 0, 0), Rank(0, 2, 0, 0, 1, 1), Rank(2, 4, 1, 1, None, 2), Rank(4, 8, 2, 3, None, 4)]
    assert [sources(ranks, i, True) for i in range(4)] == [[3], [3], [0, 1], [2]]
    assert [targets(ranks, i, True) for i in range(4)] == [[2], [2], [3], [1, 0]]
    assert [sources(ranks, i, False) for i in range(4)] == [[3], [0], [1], [2]]
    assert [targets(ranks, i, False) for i in range(4)] == [[1], [2], [3], [0]]
    assert [partners(ranks, i) for i in range(4)] == [(1, None), (None, 0), (None, None), (None, None)]


def test_a_spread_triple_and_single_layers():
    ranks = layout([[0, 1], [2, 3, 4], [5, 6]], "all", 8)
    assert [(r.start, r.end, r.member) for r in ranks] == [(0, 1, 0), (0, 2, 1), (2, 3, 0), (3, 4, 1), (3, 5, 2),
                                                           (5, 6, 0), (5, 7, 1), (7, 8, None)]
    assert sources(ranks, 5, True) == [2, 3, 4]
    assert targets(ranks, 0, True) == [4, 2, 3]             # the triple's replica rank first, the others in order
    assert [targets(ranks, i, False) for i in range(2, 5)] == [[3], [4], [5]]
    assert [partners(ranks, i) for i in range(2, 5)] == [(None, None), (4, None), (None, 3)]   # layer 3 feeds its copy
    singles = layout([[2, 3]], "all", 6)                    # layers 1 and 4 each on a rank of their own
    assert [(r.start, r.end) for r in singles] == [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6)]


def test_the_last_step_cannot_be_spread_and_runs_are_cut_at_steps():
    with pytest.raises(ValueError):
        layout(GROUPS, "6", 8)
    with pytest.raises(ValueError):
        layout(GROUPS, "0", 8, "5")                         # inside the pair [4, 5]
    assert [(r.start, r.end) for r in layout(GROUPS, "2", 8, "6")] == [(0, 2), (2, 3), (2, 4), (4, 6), (6, 8)]


# On four GPUs.

def _wait(path: str) -> str:
    for _ in range(600):
        if os.path.exists(path):
            return path
        time.sleep(0.1)
    raise TimeoutError(path)


def _hidden(m, items: list, plan, tmp: str, tag: str) -> dict:
    """The final hidden states per solution token, teacher forced under a plan, one request at a time."""
    path = f"{tmp}/{tag}.pt"
    m.rpc("p2p_plan", plan=plan)
    m.rpc("p2p_record", on=True)
    for r in items:
        m.force([{**r, "id": f"{tag}/{r['id']}"}])
    m.rpc("p2p_record", on=False)
    m.rpc("p2p_save_reference", path=path)
    return {k.split("/")[1]: v.float() for k, v in torch.load(_wait(path)).items() if k.startswith(tag + "/")}


CASES = {"qwen3": ("qwen3", GROUPS, "2", ""), "qwen3_5": ("qwen3_5", GROUPS, "2", ""),   # copies layer 2
         "qwen3_5 attention copy": ("qwen3_5", [[0, 1], [3, 4], [6, 7]], "3", ""),        # copies layer 3
         "qwen3 first pair": ("qwen3", GROUPS, "0", "4"),       # copies layer 0: the head embeds for both members
         "qwen3_5 first pair": ("qwen3_5", GROUPS, "0", "4"),
         "gemma4 first pair": ("gemma4", GROUPS, "0", "4"),      # a sliding-window copy; Gemma scales its embedding
         "gemma4 full attention copy": ("gemma4", [[0, 1], [3, 4], [6, 7]], "3", ""),
         "qwen3_5_moe": ("qwen3_5_moe", GROUPS, "2", ""),         # routed experts in the copy, every one used per token
         "qwen3_5_moe first pair": ("qwen3_5_moe", GROUPS, "0", "4")}


@pytest.fixture(scope="module", params=list(CASES))
def runs(request, tmp_path_factory):
    """"one": {"exact", "full", "pruned"} and "routed": {"exact", "full"}: hidden states per request; "prepared":
    {"dense", "compact"}: under the pruned plan, with its replica prepared ahead of time (p2p.prepare) and expanded or
    kept compact; "held": {form: {rank: {"layers": {layer: {"zero", "bytes"}}, "embedding": bytes}}} under those;
    "copied": the copied layer; "exact", "replica": the ranks holding it exact and as the replica."""
    if torch.cuda.device_count() < 4:
        pytest.skip("needs four GPUs")
    from test_sglang import _checkpoint, _plan
    from p2p import prepare
    from p2p.engine import Model
    (kind, groups, spread, split), tmp = CASES[request.param], str(tmp_path_factory.mktemp("route"))
    path = _checkpoint(kind)
    rng = random.Random(0)
    token = lambda: rng.randrange(1000, 100000)
    items = [{"id": i, "prompt": [token() for _ in range(8 + 5 * i)], "sol": [token() for _ in range(16 + 3 * i)]}
             for i in range(4)]
    plans = {"exact": None, "full": _plan(groups), "pruned": _plan(groups, keep=33)}
    json.dump(plans["full"], open(f"{tmp}/plan.json", "w"))
    json.dump(plans["pruned"], open(f"{tmp}/pruned.json", "w"))
    args = dict(tp=1, batch=4, max_total_tokens=8192, skip_server_warmup=True, dtype="float16")
    out = {"prepared": {}, "held": {}}
    m = Model(path, **args)
    try:
        m.calibrate([[token() for _ in range(64)] for _ in range(8)], f"{tmp}/calib.pt")
        out["one"] = {name: _hidden(m, items, p, tmp, f"one_{name}") for name, p in plans.items()}
    finally:
        m.close()
    prepare.replicas(SimpleNamespace(model=path, plan=f"{tmp}/pruned.json", calib=f"{tmp}/calib.pt", out=f"{tmp}/replicas",
                                     layers=[int(spread)], dtype="float16", mem_fraction=0.5))
    env = {"P2P_PP_COMM": "fused", "P2P_PP_SPREAD": spread, "P2P_PP_SPLIT": split, "P2P_PP_PLAN": f"{tmp}/plan.json"}
    for form in ("", "dense", "compact"):
        replicas = {"P2P_PP_REPLICAS": f"{tmp}/replicas", "P2P_PP_REPLICA_FORM": form, "P2P_PP_PLAN": f"{tmp}/pruned.json",
                    "P2P_PP_CALIB": f"{tmp}/calib.pt"} if form else {}     # a deployment's plan is its replicas' plan
        with mock.patch.dict(os.environ, env | replicas):
            m = Model(path, pp_size=4, **args)
            try:
                m.rpc("p2p_load_calibration", path=f"{tmp}/calib.pt")       # for the groups stages run whole
                if not form:
                    out["routed"] = {name: _hidden(m, items, plans[name], tmp, f"routed_{name}") for name in ("exact", "full")}
                    continue
                out["prepared"][form] = _hidden(m, items, plans["pruned"], tmp, f"prepared_{form}")
                os.makedirs(f"{tmp}/held_{form}")
                m.rpc("p2p_held", path=f"{tmp}/held_{form}")
                out["held"][form] = {i: json.load(open(_wait(f"{tmp}/held_{form}/pp{i}.json"))) for i in range(4)}
            finally:
                m.close()
    ranks = layout(groups, spread, 8, split)
    exact, replica = (next(i for i in range(4) if partners(ranks, i)[j] is not None) for j in (0, 1))
    return SimpleNamespace(**out, copied=int(spread), exact=exact, replica=replica)


def _diff(a: dict, b: dict, at=slice(None)) -> float:
    assert a.keys() == b.keys()
    return max((a[k][at] - b[k][at]).abs().max().item() for k in a)


def test_the_exact_model_matches_one_gpu(runs):
    assert _diff(runs.routed["exact"], runs.one["exact"]) == 0


def test_full_copies_match_one_gpu(runs):
    assert _diff(runs.routed["full"], runs.one["full"]) == 0


def test_prepared_replicas_expanded_match_one_gpu(runs):
    """Pruned ahead of time from the checkpoint, they are the replicas pruned on one GPU, weight for weight."""
    assert _diff(runs.prepared["dense"], runs.one["pruned"]) == 0
    assert _diff(runs.prepared["dense"], runs.one["full"]) > 0      # pruning changed something


def test_the_replicas_rank_holds_the_copied_layer_only_pruned(runs):
    held = {(rank, int(layer)): x["zero"] for rank, h in runs.held["dense"].items() for layer, x in h["layers"].items()}
    assert held[runs.replica, runs.copied] > 0.6                # the last member's rank: keep 33
    assert all(zero < 0.01 for (rank, layer), zero in held.items() if (rank, layer) != (runs.replica, runs.copied))


def test_only_the_head_holds_the_embedding(runs):
    assert [runs.held["compact"][i]["embedding"] > 0 for i in range(4)] == [False, False, False, True]


def _err(a: dict, b: dict) -> float:
    return max(((a[k] - b[k]).norm() / b[k].norm()).item() for k in b)


def test_compact_replicas_match_one_gpu_within_rounding(runs):
    """The compact kernel rounds differently: far less than pruning itself changes, on one GPU."""
    err, pruning = _err(runs.prepared["compact"], runs.one["pruned"]), _err(runs.one["pruned"], runs.one["full"])
    print(f"\ncompact against one GPU: {err:.4f}; pruning's own change: {pruning:.4f}")
    assert 0 < err < 0.1 * pruning


def test_a_compact_replica_costs_about_its_size(runs):
    exact = runs.held["compact"][runs.exact]["layers"][str(runs.copied)]["bytes"]
    replica = runs.held["compact"][runs.replica]["layers"][str(runs.copied)]["bytes"]     # compact, keep 33
    assert replica < 0.6 * exact


@pytest.mark.parametrize("tree", [False, True], ids=["chain", "tree"])
def test_drafting_gives_one_gpus_tokens(tmp_path, tree):
    """Tiny Qwen3.5 drafting 3 with its MTP layer, under a full copy of layer 0 and a prepared pruned one: verify
    passes go step to step too, and the copy keeps its own K/V and state only within a pass. A chain, and a tree (2
    candidates a step, 6 tokens a pass, in bfloat16 as SGLang's tree kernel needs)."""
    if torch.cuda.device_count() < 4:
        pytest.skip("needs four GPUs")
    from test_sglang import _plan
    from test_spec import _with_mtp
    from p2p import prepare
    from p2p.engine import Model, speculative
    path, plans = _with_mtp(), {"full": _plan(GROUPS), "pruned": _plan(GROUPS, keep=33)}
    json.dump(plans["full"], open(f"{tmp_path}/plan.json", "w"))
    json.dump(plans["pruned"], open(f"{tmp_path}/pruned.json", "w"))
    rng = random.Random(0)
    windows = [[rng.randrange(1000, 100000) for _ in range(64)] for _ in range(8)]
    prompts = [[rng.randrange(1000, 20000) for _ in range(8 + 5 * i)] for i in range(4)]
    params = [{"temperature": 0, "max_new_tokens": 12 + 4 * i, "ignore_eos": True} for i in range(4)]
    dtype = "bfloat16" if tree else "float16"
    args = dict(tp=1, batch=4, max_total_tokens=8192, skip_server_warmup=True, dtype=dtype,
                **speculative(path, "mtp", 3, topk=2 if tree else 1, tokens=6 if tree else 0))
    env = {"P2P_PP_COMM": "fused", "P2P_PP_SPREAD": "0", "P2P_PP_SPLIT": "4", "P2P_PP_PLAN": f"{tmp_path}/plan.json",
           "SGLANG_ENABLE_PP_SPEC": "1"}
    ready = lambda form: env | {"P2P_PP_REPLICAS": f"{tmp_path}/replicas", "P2P_PP_REPLICA_FORM": form,
                                "P2P_PP_PLAN": f"{tmp_path}/pruned.json", "P2P_PP_CALIB": f"{tmp_path}/calib.pt"}
    ids = {}
    for name, extra, patch, kinds in [("one", {}, {}, ("full", "pruned")), ("routed", {"pp_size": 4}, env, ("full",)),
                                      ("dense", {"pp_size": 4}, ready("dense"), ("pruned",)),
                                      ("compact", {"pp_size": 4}, ready("compact"), ("pruned",))]:
        with mock.patch.dict(os.environ, patch):
            m = Model(path, **args, **extra)
            try:
                if name == "one":
                    m.calibrate(windows, f"{tmp_path}/calib.pt")
                m.rpc("p2p_load_calibration", path=f"{tmp_path}/calib.pt")
                for kind in kinds:
                    m.rpc("p2p_plan", plan=plans[kind])
                    ids[name, kind] = [m.engine.generate(input_ids=[x], sampling_params=y)[0]["output_ids"]
                                       for x, y in zip(prompts, params)]
            finally:
                m.close()
        if name == "one":
            prepare.replicas(SimpleNamespace(model=path, plan=f"{tmp_path}/pruned.json", calib=f"{tmp_path}/calib.pt",
                                             out=f"{tmp_path}/replicas", layers=[0], dtype=dtype, mem_fraction=0.5))
    assert ids["routed", "full"] == ids["one", "full"]
    assert ids["dense", "pruned"] == ids["one", "pruned"]
    assert ids["one", "pruned"] != ids["one", "full"]
    # A compact replica rounds differently, so greedy tokens may part ways; they must all be there.
    assert [len(x) for x in ids["compact", "pruned"]] == [len(x) for x in ids["one", "pruned"]]
    agree = [next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), len(x))
             for x, y in zip(ids["compact", "pruned"], ids["one", "pruned"])]
    print(f"\ndrafting with compact replicas: tokens agreeing with one GPU before they part: {agree} of "
          f"{[len(x) for x in ids['one', 'pruned']]}")


@pytest.mark.parametrize("groups,spread,split", [(GROUPS, "0", "4"), (GROUPS, "2", ""), ([[0, 1]], "0", "3"),
                                                 ([[0, 1]], "0", "4")],
                         ids=["first pair", "pair 2", "plain layer", "several plain steps"])
def test_streamed_passes_give_one_gpus_tokens(tmp_path, groups, spread, split):
    """Verify passes streamed in pieces (P2P_PP_STREAM): tiny Qwen3.5 drafting a tree with its MTP layer, a full copy
    spread, on four layouts (a pair's members, a group whole, a plain layer, a rank of several steps). In pieces of 2,
    with the head's early stop and without, one GPU's tokens exactly. With the root in a piece of its own it takes
    cuBLAS's one-row matmuls, which round otherwise, so its tokens may part and only their number is checked. With the
    tree sent in phases (its first 3 rows after one draft step, 5 after two), the same tokens, including each
    request's last pass, whose tree is never used."""
    if torch.cuda.device_count() < 4:
        pytest.skip("needs four GPUs")
    from test_sglang import _plan
    from test_spec import _with_mtp
    from p2p.engine import Model, speculative
    path, plan = _with_mtp(), _plan(groups)
    json.dump(plan, open(f"{tmp_path}/plan.json", "w"))
    rng = random.Random(0)
    windows = [[rng.randrange(1000, 100000) for _ in range(64)] for _ in range(8)]
    prompts = [[rng.randrange(1000, 20000) for _ in range(8 + 5 * i)] for i in range(4)]
    params = [{"temperature": 0, "max_new_tokens": 12 + 4 * i, "ignore_eos": True} for i in range(4)]
    args = dict(tp=1, batch=4, max_total_tokens=8192, skip_server_warmup=True, dtype="bfloat16",
                **speculative(path, "mtp", 3, topk=2, tokens=6))
    env = {"P2P_PP_COMM": "fused", "P2P_PP_SPREAD": spread, "P2P_PP_SPLIT": split, "P2P_PP_PLAN": f"{tmp_path}/plan.json",
           "SGLANG_ENABLE_PP_SPEC": "1"}
    ids, pieces = {}, {}
    for name, extra, patch in [("one", {}, {}), ("2", {"pp_size": 4}, env | {"P2P_PP_STREAM": "2"}),
                               ("late", {"pp_size": 4}, env | {"P2P_PP_STREAM": "2", "P2P_PP_EARLY": "0"}),
                               ("phased", {"pp_size": 4}, env | {"P2P_PP_STREAM": "2", "P2P_PP_PHASES": "1:3,2:5"}),
                               ("1", {"pp_size": 4}, env | {"P2P_PP_STREAM": "1,2"})] + (
            [("check", {"pp_size": 4}, env | {"P2P_PP_STREAM": "2", "P2P_PP_EARLY": "check"})] if spread == "2" else []):
        with mock.patch.dict(os.environ, patch):
            m = Model(path, **args, **extra)
            try:
                if name == "one":
                    m.calibrate(windows, f"{tmp_path}/calib.pt")
                m.rpc("p2p_load_calibration", path=f"{tmp_path}/calib.pt")
                m.rpc("p2p_plan", plan=plan)
                ids[name] = [m.engine.generate(input_ids=[x], sampling_params=y)[0]["output_ids"]
                             for x, y in zip(prompts, params)]
                if name != "one":
                    os.makedirs(f"{tmp_path}/held_{name}")
                    m.rpc("p2p_held", path=f"{tmp_path}/held_{name}")
                    held = [json.load(open(_wait(f"{tmp_path}/held_{name}/pp{i}.json"))) for i in range(4)]
                    pieces[name], checked = [h["pieces"] for h in held], held[3]["check"]
                    if name == "check":     # each pass the head stopped early, against the same pass run whole
                        assert checked and all(c["ends"] == c["whole_ends"] and c["logits"] == 0 for c in checked)
                        assert ids[name] == ids["one"]
            finally:
                m.close()
    print(f"\npieces sent and received per rank: {pieces}")
    assert all(p["sent"] for p in pieces["2"]) and all(p["received"] for p in pieces["2"])   # the head's embedding too
    head, late = pieces["2"][3], pieces["late"][3]
    assert head["passes"] and head["rows"] < head["of"] and not late["passes"]       # the head stopped early
    assert ids["2"] == ids["one"] and ids["late"] == ids["one"] and ids["phased"] == ids["one"]
    assert [len(x) for x in ids["1"]] == [len(x) for x in ids["one"]]
    agree = [next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), len(x)) for x, y in zip(ids["1"], ids["one"])]
    print(f"the root alone, then pieces of 2: tokens agreeing with one GPU before they part: {agree} of "
          f"{[len(x) for x in ids['one']]}")
