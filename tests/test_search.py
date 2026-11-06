"""Replica sizes, budget splits and the DP, against what the draft found for Qwen3.8: W40."""
import itertools
import json
import math
import os
import random

import pytest

from p2p import search

HERE = os.path.dirname(__file__)
MLP = {"gate": [17408, 5120], "up": [17408, 5120], "down": [5120, 17408]}
LINEAR = {"q": [2048, 5120], "k": [2048, 5120], "v": [6144, 5120], "z": [6144, 5120], "out": [5120, 6144], **MLP}
ATTENTION = {"q": [12288, 5120], "k": [1024, 5120], "v": [1024, 5120], "o": [5120, 6144], **MLP}
TYPES = ["A" if i % 4 == 3 else "L" for i in range(64)]
W40 = json.load(open(os.path.join(HERE, "..", "configs", "qwen3.8_w40.json")))
COSTS = os.path.join(HERE, "data", "q38_s32_costs.json")    # measured by the scan at 32 steps, replica size .40


def test_w40_replica_sizes_match_the_draft():
    sizes = {round(search.size(ATTENTION if TYPES[int(i)] == "A" else LINEAR, k), 3) for i, k in W40["keep"].items()}
    assert sizes == {0.398, 0.397, 0.396, 0.395, 0.393}


def test_w40_has_32_steps_with_a_replica_per_group():
    assert 64 - sum(len(g) - 1 for g in W40["groups"]) == 32
    assert {str(g[-2]) for g in W40["groups"]} == set(W40["keep"]) and len(W40["keep"]) == 28


def test_the_search_finds_the_scans_plan():
    data = json.load(open(COSTS))
    options = {}
    for start, n, name, size, cost in data["options"]:
        options.setdefault((start, n), {})[name] = (size, cost)
    cost, _, segments = search.tilings(64, options)[32]
    assert data["floor"] + cost == pytest.approx(0.009465, abs=1e-6)
    assert [list(s) for s in segments] == data["segments"]


def test_layers_of_one_class_with_other_shapes_are_other_kinds():
    sliding = {"q": [8192, 5376], "k": [4096, 5376], "v": [4096, 5376], "o": [5376, 8192]} | dict.fromkeys(("gate", "up"), [21504, 5376])
    sliding["down"] = [5376, 21504]
    full = {**sliding, "q": [16384, 5376], "k": [2048, 5376], "v": [2048, 5376], "o": [5376, 16384]}
    layers = [{"type": "Gemma4DecoderLayer", "proj": p} for p in (sliding, sliding, full, sliding, full)]
    types = search.kinds(layers)
    assert types[0] == types[1] == types[3] != types[2] == types[4]
    for shape in (sliding, full):
        assert all(search.size(shape, keep) <= 0.40 for keep in search.splits(shape, 0.40).values())


def test_routed_experts_take_the_cut():
    shape = {"q": [2048, 2048], "k": [2048, 2048], "v": [4096, 2048], "z": [4096, 2048], "out": [2048, 4096], "gate": [512, 2048],
             "up": [512, 2048], "down": [2048, 512], "egate": [131072, 2048], "eup": [131072, 2048], "edown": [524288, 512]}
    recipes = search.splits(shape, 0.40)
    assert set(recipes) == {"uniform", "rest50", "rest70", "rest90"}
    assert all(search.size(shape, keep) <= 0.40 for keep in recipes.values())
    assert recipes["rest90"]["q"] == 90 and recipes["rest90"]["eup"] < recipes["rest50"]["eup"]


def _brute_force(L: int, options: dict) -> dict:
    """{steps: least summed cost} over every split of L layers into singles, pairs and triples."""
    best = {}
    for n in range(1, L + 1):
        for sizes in itertools.product((1, 2, 3), repeat=n):
            if sum(sizes) != L:
                continue
            total, p = 0.0, 0
            for size in sizes:
                if size > 1:
                    if (p, size) not in options:
                        total = math.inf
                        break
                    total += min(c for _, c in options[p, size].values())
                p += size
            best[n] = min(best.get(n, math.inf), total)
    return {k: v for k, v in best.items() if v < math.inf}


@pytest.mark.parametrize("seed", range(5))
def test_the_dp_matches_brute_force(seed):
    rng = random.Random(seed)
    L = 12
    options = {(p, n): {f"o{j}": (rng.random(), rng.random()) for j in range(2)}
               for n in (2, 3) for p in range(L - n + 1) if rng.random() < 0.7}
    got = search.tilings(L, options)
    want = _brute_force(L, options)
    assert set(got) == set(want)
    for steps, (cost, size, segments) in got.items():
        assert cost == pytest.approx(want[steps])
        assert len(segments) == steps and sum(n for _, n, _ in segments) == L
        assert [p for p, _, _ in segments] == list(itertools.accumulate((n for _, n, _ in segments[:-1]), initial=0))


def test_the_dp_prefers_less_size_on_equal_cost():
    options = {(0, 2): {"big": (0.5, 1.0), "small": (0.2, 1.0)}}
    assert search.tilings(2, options)[1][2] == [(0, 2, "small")]


@pytest.mark.parametrize("cap,x", [(0.30, 23), (0.35, 28), (0.40, 33), (0.50, 43)])
def test_uniform_keep_matches_the_draft(cap, x):
    for shape in (LINEAR, ATTENTION):
        assert set(search.splits(shape, cap)["uniform"].values()) == {x}


def test_splits_reproduce_the_w40_recipes():
    keep = lambda q, z, out, mlp: {"q": q, "k": 7, "v": 7, "z": z, "out": out, "gate": mlp, "up": mlp, "down": mlp}
    want = {"shift": keep(16, 33, 55, 36), "z+7": keep(16, 40, 55, 35), "out+12": keep(16, 33, 45, 37), "q=x": keep(33, 33, 55, 35)}
    got = search.splits(LINEAR, 0.40)
    for name, recipe in want.items():
        assert got[name] == recipe, name
        assert recipe in W40["keep"].values(), name


def test_every_split_fits_its_cap():
    for cap in (0.15, 0.3, 0.4, 0.5):
        for shape in (LINEAR, ATTENTION):
            recipes = search.splits(shape, cap)
            assert "uniform" in recipes
            assert all(search.size(shape, k) <= cap + 1e-9 and set(k) == set(shape) for k in recipes.values())


def test_representatives_cover_each_type():
    types = TYPES
    reps = search.representatives(types)
    assert set(reps) == {"L", "A"}
    for t, (bands, triples) in reps.items():
        assert sum(map(len, bands)) == 6 and len(triples) == 2
        assert all(types[a] == t for b in bands for a, _ in b) and all(types[a + 1] == t for a, _ in triples)
        assert sorted(a for b in bands for a, _ in b) == [a for b in bands for a, _ in b]


def test_representatives_handle_a_short_model():
    reps = search.representatives(["L", "L", "A"])
    assert reps["L"] == ([[(0, 2), (1, 2)]], [(0, 3)])
    assert reps["A"] == ([], [])


def test_pick_takes_the_best_split_per_band_and_keeps_uniform():
    recipes = dict.fromkeys(["uniform", "a", "b", "c", "d"], {})
    table = {"uniform": 1.0, "a": 0.9, "b": 0.8, "c": 0.7, "d": 1.2}
    shallow, deep = [(0, 2)], [(10, 2)]
    cost = lambda s, name: 0.5 if s == (0, 2) and name == "a" else table[name]    # a wins only the shallow band
    got = search.pick(cost, recipes, [shallow, deep], [(5, 3)], n=3)
    assert got["pair"] == ["uniform", "a", "c", "b"]
    assert got["triple"] == ["uniform", "c"]


def test_pick_without_triples():
    assert search.pick(lambda s, n: 1.0, {"uniform": {}, "a": {}}, [[(0, 2)]], [])["triple"] == ["uniform"]


def test_plan_names_the_copied_layer():
    p = search.plan("m", [(0, 2, "x"), (2, 1, None), (3, 3, "y")], {"x": {"q": 1}, "y": {"q": 2}}.get)
    assert p == {"model": "m", "groups": [[0, 1], [3, 4, 5]], "keep": {"0": {"q": 1}, "4": {"q": 2}}}


def test_a_tiling_with_no_measured_segments_is_all_single_layers():
    t = search.tilings(4, {})
    assert list(t) == [4] and t[4][2] == [(i, 1, None) for i in range(4)] and not math.isnan(t[4][0])
