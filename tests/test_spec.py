"""Plans under SGLang's speculative decoding: tiny Qwen3 and Gemma 4 draft for themselves (STANDALONE), and tiny Qwen3.5
(linear and full attention) drafts with an MTP layer of random weights, as the real Qwen3.5 family does. Random tiny
models keep few drafts, but every verify pass still runs K + 1 tokens through the plan. Needs GPUs and
P2P_TEST_TOKENIZER, as test_sglang does."""
import os
import random
import shutil
from types import SimpleNamespace

import pytest
import torch
from test_sglang import PAIRS, ROOT, _checkpoint, _plan

from p2p.engine import Model, speculative

K = 3


def _with_mtp() -> str:
    """The tiny Qwen3.5 checkpoint plus an MTP layer: a copy of its full-attention layer 3, which has the same names."""
    path = f"{ROOT}/qwen3_5_mtp"
    if not os.path.exists(f"{path}/mtp.safetensors"):
        from safetensors.torch import load_file, save_file
        shutil.copytree(_checkpoint("qwen3_5"), path, dirs_exist_ok=True)
        w = load_file(f"{path}/model.safetensors")
        layer, norm = "model.language_model.layers.3.", w["model.language_model.norm.weight"]
        mtp = {f"mtp.layers.0.{k[len(layer):]}": v for k, v in w.items() if k.startswith(layer)}
        mtp |= {f"mtp.{n}.weight": norm.clone() for n in ("norm", "pre_fc_norm_hidden", "pre_fc_norm_embedding")}
        mtp["mtp.fc.weight"] = 0.1 * torch.randn(norm.numel(), 2 * norm.numel(), generator=torch.Generator().manual_seed(0)).to(norm)
        save_file(mtp, f"{path}/mtp.safetensors")
    return path


@pytest.fixture(scope="module", params=[(kind, tp) for kind in ("qwen3", "gemma4", "qwen3_5") for tp in (1, 2)], ids=str)
def e(request, tmp_path_factory):
    kind, tp = request.param
    path, draft = (_with_mtp(), "mtp") if kind == "qwen3_5" else (_checkpoint(kind), "self")
    rng = random.Random(0)
    token = lambda: rng.randrange(1000, 100000)
    m = Model(path, tp=tp, batch=4, max_total_tokens=8192, skip_server_warmup=True, dtype="float16",
              **speculative(path, path if draft == "self" else "mtp", K))
    calib = f"{tmp_path_factory.mktemp('spec')}/calib.pt"
    m.calibrate([[token() for _ in range(64)] for _ in range(8)], calib)
    m.rpc("p2p_load_calibration", path=calib)
    env = SimpleNamespace(m=m, prompts=[[token() for _ in range(8 + 5 * i)] for i in range(4)])
    env.exact = _run(env, None)
    yield env
    m.close()


def _run(e, plan) -> SimpleNamespace:
    """Greedy tokens under a plan, and tokens per verify pass (spec_verify_ct is missing without speculative decoding)."""
    e.m.rpc("p2p_plan", plan=plan)
    outs = e.m.generate(e.prompts, 16)
    per_pass = sum(o["meta_info"]["completion_tokens"] for o in outs) / sum(o["meta_info"]["spec_verify_ct"] for o in outs)
    return SimpleNamespace(ids=[o["output_ids"] for o in outs], per_pass=per_pass)


def test_drafter_arguments():
    """SGLang turns EAGLE with a Gemma 4 assistant as the drafter into FROZEN_KV_MTP by itself."""
    assert speculative("m", "", 3) == {}
    for draft, algorithm, path in (("mtp", "EAGLE", "m"), ("mtp:a", "EAGLE", "a"), ("small", "STANDALONE", "small")):
        args = speculative("m", draft, 3)
        assert (args["speculative_algorithm"], args["speculative_draft_model_path"]) == (algorithm, path)
        assert args["speculative_num_draft_tokens"] == 4


def test_decoding_is_speculative(e):
    """SGLang counts verify passes only when it drafts."""
    assert e.exact.per_pass >= 1


@pytest.mark.parametrize("groups", PAIRS, ids=str)
def test_full_copy_matches_exact(e, groups):
    """A pass of several tokens needs the copy's K/V overwrite and state restore right, as a decode step does."""
    assert _run(e, _plan(groups)).ids == e.exact.ids


def test_pruned_copy_runs_in_verify_passes(e):
    """Without this, the test above could pass because verify passes ran the exact model."""
    assert _run(e, _plan(PAIRS[3], keep=5)).ids != e.exact.ids


@pytest.fixture(scope="module", params=[(kind, tp) for kind in ("gemma4", "qwen3_5") for tp in (1, 2)], ids=str)
def f(request, tmp_path_factory):
    """An engine that scores forced requests in verify passes of K + 1 tokens (p2p scan/kl --draft-k K)."""
    kind, tp = request.param
    path, rng, tmp = _checkpoint(kind), random.Random(1), str(tmp_path_factory.mktemp("passes"))
    token = lambda: rng.randrange(1000, 100000)
    items = [{"id": i, "prompt": [token() for _ in range(8 + 5 * i)], "sol": [token() for _ in range(16 + 3 * i)]} for i in range(4)]
    m = Model(path, tp=tp, batch=4, max_total_tokens=8192, skip_server_warmup=True, dtype="float16",
              **speculative(path, "forced", K))
    m.calibrate([[token() for _ in range(64)] for _ in range(8)], f"{tmp}/calib.pt")
    m.rpc("p2p_load_calibration", path=f"{tmp}/calib.pt")
    m.record_reference(items, f"{tmp}/reference.pt")
    m.rpc("p2p_load_reference", path=f"{tmp}/reference.pt")
    yield SimpleNamespace(m=m, items=items, tmp=tmp)
    m.close()


def test_forced_passes_keep_the_solution(f):
    outs = f.m.force(f.items)
    f.m.rpc("p2p_result", path=f"{f.tmp}/discard.json")
    assert [o["output_ids"] for o in outs] == [r["sol"] for r in f.items]
    assert [o["meta_info"]["spec_verify_ct"] for o in outs] == [-(-(len(r["sol"]) - 1) // (K + 1)) for r in f.items]


def test_scoring_in_passes(f):
    """As test_sglang.test_kl_scoring, with every token scored inside a pass."""
    floor = f.m.measure(f.items, None)
    assert floor["tokens"] == sum(len(r["sol"]) - 1 for r in f.items)
    assert floor["kl"] < 5e-4 and floor["top1"] > 0.98
    full = f.m.measure(f.items, _plan(PAIRS[3]))
    pruned = f.m.measure(f.items, _plan(PAIRS[3], keep=5))
    assert full["kl"] <= floor["kl"] + 1e-3
    assert pruned["kl"] > max(10 * full["kl"], 1e-3)
