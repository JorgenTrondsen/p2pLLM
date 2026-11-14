"""How pipeline stages talk (p2p.pp): fused must give stock SGLang's tokens, and send no gloo message at all. Tiny Qwen3, and tiny Qwen3.5 drafting with its MTP layer, at two stages on two GPUs, with
four requests of different lengths at once, so the tensors' shapes change as they finish. Needs P2P_TEST_TOKENIZER, as
test_sglang does."""
import os
import random
import tempfile
from unittest import mock

import pytest
from test_sglang import _checkpoint
from test_spec import _with_mtp

from p2p import pp
from p2p.engine import Model, speculative


@pytest.fixture(scope="module", params=["qwen3", "qwen3_5 drafting"])
def runs(request) -> dict:
    """{variant: (greedy tokens per request, message counts per stage)}."""
    path, draft = (_with_mtp(), "mtp") if request.param.endswith("drafting") else (_checkpoint("qwen3"), "")
    rng = random.Random(0)
    prompts = [[rng.randrange(1000, 20000) for _ in range(8 + 5 * i)] for i in range(4)]
    out = {}
    for comm in ("stock", "fused"):
        env = {"P2P_PP_COMM": comm, "P2P_PP_TRACE": "1"} | ({"SGLANG_ENABLE_PP_SPEC": "1"} if draft else {})
        with mock.patch.dict(os.environ, env), tempfile.TemporaryDirectory() as d:
            m = Model(path, tp=1, batch=4, pp_size=2, max_total_tokens=8192, skip_server_warmup=True, dtype="float16",
                      **speculative(path, draft, 3 if draft else 0))
            try:
                m.rpc("p2p_pp_counts")
                outs = m.engine.generate(input_ids=prompts, sampling_params=[
                    {"temperature": 0, "max_new_tokens": 12 + 4 * i, "ignore_eos": True} for i in range(4)])
                m.rpc("p2p_pp_counts", path=d)
                out[comm] = [o["output_ids"] for o in outs], pp.counts(d, 2)
            finally:
                m.close()
    return out


def _gloo(counts: dict) -> int:
    return sum(n for c in counts.values() for k, n in c.items() if k.startswith("gloo") and not k.endswith("bytes"))


def test_the_same_tokens_as_stock(runs):
    assert runs["fused"][0] == runs["stock"][0]


def test_stock_sends_metadata_over_gloo(runs):
    assert _gloo(runs["stock"][1]) > 0


def test_fused_sends_nothing_over_gloo(runs):
    assert _gloo(runs["fused"][1]) == 0
