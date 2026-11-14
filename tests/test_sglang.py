"""Replica wiring on a real SGLang engine, on tiny checkpoints with large random weights (dummy weights are so small that
a layer hardly changes its input, and any wiring would pass). On two GPUs, with a tokenizer in P2P_TEST_TOKENIZER:

    CUDA_VISIBLE_DEVICES=0,1 pytest tests/test_sglang.py
"""
import itertools
import os
import random
import shutil
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from p2p import replica
from p2p.engine import Model

ROOT = ".cache/tiny"
TOKENIZER = os.environ.get("P2P_TEST_TOKENIZER", "Qwen/Qwen3.5-2B")
ROUTED = ("egate", "eup", "edown")
NAMES = ("q", "k", "v", "z", "o", "out", "gate", "up", "down") + ROUTED
TOL = 0.005                  # rounding in float16, 8x finer than bf16's (FlashInfer has no float32): a wiring fault shows
PAIRS = [[[0, 1]], [[2, 3]], [[3, 4]], [[0, 1], [2, 3], [4, 5], [6, 7]]]      # linear and attention copies, then together
SEEN = {}                    # hidden states per (kind, tp), to compare TP sizes
tags = itertools.count()


def _checkpoint(kind: str, layers: int = 8) -> str:
    """A tiny checkpoint of a family, saved once. Heads of 64: FlashInfer has no kernel for 32. layers: a multiple of 4."""
    path = f"{ROOT}/{kind}" + (f"_{layers}layers" if layers != 8 else "")
    if os.path.exists(f"{path}/config.json"):
        return path
    from transformers import (AutoTokenizer, Gemma4Config, Gemma4ForConditionalGeneration, Qwen3Config, Qwen3ForCausalLM,
                              Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration)
    text = dict(vocab_size=248320, hidden_size=256, intermediate_size=512, num_hidden_layers=layers, num_attention_heads=4,
                num_key_value_heads=2, head_dim=64, initializer_range=0.1, max_position_embeddings=4096)
    if kind == "qwen3":
        model = Qwen3ForCausalLM(Qwen3Config(**text, rope_parameters={"rope_type": "default", "rope_theta": 10000.0}))
    elif kind == "gemma4":       # the real model's head sizes, wider on full attention, and its k = v
        layer_types = (["sliding_attention"] * 3 + ["full_attention"]) * (layers // 4)
        model = Gemma4ForConditionalGeneration(Gemma4Config(text_config={
            **text, "layer_types": layer_types, "head_dim": 256, "global_head_dim": 512, "num_global_key_value_heads": 2,
            "sliding_window": 1024, "attention_k_eq_v": True, "final_logit_softcapping": 30.0, "hidden_size_per_layer_input": 0},
            vision_config={"num_hidden_layers": 1}))
    else:
        layer_types = (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4)
        rope = {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                "mrope_section": [3, 3, 2]}
        vision = dict(depth=12, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=256)
        # Every expert runs for every token: routing is all or nothing, so rounding would flip experts and hide wiring faults.
        moe = dict(num_experts=4, num_experts_per_tok=4, moe_intermediate_size=128, shared_expert_intermediate_size=128)
        config, cls = (Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration) if kind == "qwen3_5_moe" else \
            (Qwen3_5Config, Qwen3_5ForConditionalGeneration)
        model = cls(config(
            text_config={**text, **(moe if kind == "qwen3_5_moe" else {}), "layer_types": layer_types, "linear_conv_kernel_dim": 4,
                         "linear_key_head_dim": 32, "linear_value_head_dim": 32, "linear_num_key_heads": 2,
                         "linear_num_value_heads": 2, "full_attention_interval": 4, "rope_parameters": rope, "attn_output_gate": True},
            vision_config=vision))
    model.to(torch.bfloat16).save_pretrained(path)
    AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(path)
    return path


def _plan(groups: list, keep: int = 100) -> dict:
    return {"groups": groups, "keep": {str(g[-2]): {n: keep for n in NAMES} for g in groups}}


def _hidden(e, plan: "dict | None") -> dict:
    """The final hidden states per token under a plan."""
    tag, path = f"t{next(tags)}", f"{e.tmp}/hidden.pt"
    e.m.rpc("p2p_plan", plan=plan)
    e.m.rpc("p2p_record", on=True)
    e.m.force([{**r, "id": f"{tag}/{r['id']}"} for r in e.items])
    e.m.rpc("p2p_record", on=False)
    e.m.rpc("p2p_save_reference", path=path)
    return {k.split("/")[1]: v.float() for k, v in torch.load(path).items() if k.startswith(tag + "/")}


def _err(a: dict, b: dict) -> float:
    """The largest relative difference over the requests."""
    assert a.keys() == b.keys()
    return max(((a[k] - b[k]).norm() / b[k].norm()).item() for k in a)


@pytest.fixture(scope="module", params=[("qwen3_5", 1), ("qwen3_5", 2), ("qwen3", 1), ("qwen3", 2), ("gemma4", 1), ("gemma4", 2), ("qwen3_5_moe", 1), ("qwen3_5_moe", 2)], ids=str)
def e(request, tmp_path_factory):
    kind, tp = request.param
    rng = random.Random(0)
    token = lambda: rng.randrange(1000, 100000)
    items = [{"id": i, "prompt": [token() for _ in range(8 + 5 * i)], "sol": [token() for _ in range(16 + 3 * i)]} for i in range(4)]
    m = Model(_checkpoint(kind), tp=tp, batch=4, max_total_tokens=8192, skip_server_warmup=True, dtype="float16")
    tmp = str(tmp_path_factory.mktemp(f"{kind}_{tp}"))
    m.calibrate([[token() for _ in range(64)] for _ in range(8)], f"{tmp}/calib.pt")
    m.rpc("p2p_load_calibration", path=f"{tmp}/calib.pt")
    # A group's sum is not the layers in sequence to the last bit: each part is rounded to the model's dtype, as a node
    # sends it (replica.py). These random models amplify that (runs/review/tiny_err.py: up to 0.3% on Qwen3 and Qwen3.5,
    # 0.5% on the MoE, which sums more terms, 1.2% on Gemma's first pair), while a fault is another order: a copy pruned
    # to 5% is 100% off. On real weights it costs nothing (Gemma 4 31B's plan: KL .01880 against .01881 unrounded).
    env = SimpleNamespace(m=m, kind=kind, tp=tp, items=items, tmp=tmp, tol={"qwen3_5_moe": 2 * TOL, "gemma4": 4 * TOL}.get(kind, TOL))
    env.exact = _hidden(env, None)
    yield env
    m.close()


def test_forced_tokens_are_kept(e):
    e.m.rpc("p2p_record", on=True)            # recording: nothing to score against yet
    outs = e.m.force([{**r, "id": f"forced/{r['id']}"} for r in e.items])
    e.m.rpc("p2p_record", on=False)
    assert [o["output_ids"] for o in outs] == [r["sol"] for r in e.items]


def test_exact_run_is_repeatable(e):
    assert _err(_hidden(e, None), e.exact) < TOL


def test_no_groups_matches_stock(e):
    prompts = [r["prompt"] for r in e.items]
    e.m.rpc("p2p_plan")
    stock = e.m.generate(prompts, 12)
    e.m.rpc("p2p_plan", plan={"groups": [], "keep": {}})
    assert [o["output_ids"] for o in e.m.generate(prompts, 12)] == [o["output_ids"] for o in stock]


@pytest.mark.parametrize("groups", PAIRS, ids=str)
def test_full_copy_matches_sequential(e, groups):
    """h' = l0(h) + l1(copy(h)) - copy(h) = l1(l0(h)) up to rounding (e.tol), which needs the plain runs, state restore
    and K/V overwrite all right."""
    assert _err(_hidden(e, _plan(groups)), e.exact) < e.tol


def test_pruned_copy_differs(e):
    """Without this, the test above could be passing because these models don't react to anything."""
    assert _err(_hidden(e, _plan(PAIRS[3], keep=5)), e.exact) > 10 * TOL


def test_pruned_copy_is_the_same_at_any_tp(e):
    """Wanda's per-row choice must not depend on how weights are split across ranks. They share one calibration: rounding
    makes each TP size's sums differ by about 0.1%, which flips borderline weights and has nothing to do with the split."""
    if e.kind == "gemma4":
        pytest.skip("its exact model already differs 8% between TP sizes here: large random weights amplify rounding. Real weights agree")
    shared = f"{os.path.dirname(e.tmp)}/{e.kind}_calib.pt"
    if not os.path.exists(shared):
        shutil.copy(f"{e.tmp}/calib.pt", shared)
    e.m.rpc("p2p_load_calibration", path=shared)
    SEEN[e.kind, e.tp] = _hidden(e, _plan(PAIRS[3], keep=33))
    other = SEEN.get((e.kind, 3 - e.tp))
    if other is None:
        pytest.skip("the other TP size hasn't run yet")
    assert _err(SEEN[e.kind, e.tp], other) < 2 * e.tol


def test_pruned_experts_differ(e):
    """The copy's kernel must read the copy's experts, not the real layer's."""
    if e.kind != "qwen3_5_moe":
        pytest.skip("no routed experts")
    keep = {**dict.fromkeys(NAMES, 100), **dict.fromkeys(ROUTED, 30)}
    assert _err(_hidden(e, {"groups": PAIRS[0], "keep": {"0": keep}}), e.exact) > 10 * TOL


def test_expert_formula_matches_the_kernel():
    """Calibration works out by itself what each expert's down reads, so it has to agree with SGLang's kernel."""
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    torch.manual_seed(0)
    experts, hidden, rows, n = 8, 64, 32, 16
    w13 = (torch.randn(experts, 2 * rows, hidden, device="cuda", dtype=torch.bfloat16) * 0.3)
    w2 = (torch.randn(experts, hidden, rows, device="cuda", dtype=torch.bfloat16) * 0.3)
    x = torch.randn(n, hidden, device="cuda", dtype=torch.bfloat16)
    weights = torch.softmax(torch.randn(n, 2, device="cuda"), -1).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(experts, device="cuda")[:2] for _ in range(n)])
    config = MoeRunnerConfig(num_experts=experts, num_local_experts=experts, hidden_size=hidden,
                             intermediate_size_per_partition=rows, top_k=2)
    with mock.patch(f"{fused_moe.__name__}_triton_config.get_exec") as a, mock.patch(f"{fused_moe.__name__}.get_exec") as b:
        a.return_value.deterministic.enable_deterministic_inference = False
        b.return_value.moe.enable_fused_moe_sum_all_reduce = False
        out = fused_moe.fused_experts(x.clone(), w13, w2, StandardTopKOutput(weights, ids, None), config).float()
    mine = torch.zeros_like(out)
    for t, k in itertools.product(range(n), range(2)):
        gate, up = (x[t].float() @ w13[ids[t, k]].float().T).chunk(2, -1)
        mine[t] += weights[t, k].float() * ((nn.functional.silu(gate) * up) @ w2[ids[t, k]].float().T)
    assert (out - mine).norm() / out.norm() < 0.01


def test_triple_runs_and_differs_from_sequential(e):
    err = _err(_hidden(e, _plan([[0, 1, 2], [4, 5, 6]])), e.exact)
    assert 0 < err < float("inf")


def test_kl_scoring(e):
    path = f"{e.tmp}/reference.pt"
    e.m.record_reference(e.items, path)
    e.m.rpc("p2p_load_reference", path=path)
    floor = e.m.measure(e.items, None)
    assert floor["tokens"] == sum(len(r["sol"]) - 1 for r in e.items)
    assert floor["kl"] < 5e-4 and floor["top1"] > 1 - 2 * e.tol
    full = e.m.measure(e.items, _plan(PAIRS[3]))
    pruned = e.m.measure(e.items, _plan(PAIRS[3], keep=5))
    assert full["kl"] <= floor["kl"] + 1e-3
    assert pruned["kl"] > max(10 * full["kl"], 1e-3)


class _Layer(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states, forward_batch):
        return self.scale * hidden_states + hidden_states ** 2


@pytest.fixture
def plain_stream(monkeypatch):
    stream = SimpleNamespace(export=lambda out: (out, None))
    monkeypatch.setattr(replica, "residual_batch", SimpleNamespace(set_written=lambda x, fb: x, stream_of=lambda fb: stream))


def _batch(mode: str):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: mode == "decode", is_target_verify=lambda: mode == "verify"))


@pytest.mark.parametrize("mode", ["decode", "verify"])
def test_triple_matches_the_formula(plain_stream, mode):
    """On decode steps, and on speculative decoding's verify passes."""
    a, b, c, copy = _Layer(0.5), _Layer(0.7), _Layer(0.9), _Layer(0.8)
    h = torch.tensor([0.1, -0.4, 0.3], dtype=torch.float64)
    group, fb = [a, b, c], _batch(mode)
    expected = a(h, fb) + (b(h, fb) - h) + (c(copy(h, fb), fb) - copy(h, fb))
    got = replica._member(group, copy, 0, h, fb)
    assert torch.allclose(got, expected.to(got.dtype))
    assert replica._member(group, copy, 1, got, fb) is got
    assert replica._member(group, copy, 2, got, fb) is got


def test_prefill_runs_layers_alone(plain_stream):
    a, b = _Layer(0.5), _Layer(0.7)
    h = torch.tensor([0.1, -0.4])
    assert torch.equal(replica._member([a, b], _Layer(0.8), 1, h, _batch("extend")), b(h, None))
