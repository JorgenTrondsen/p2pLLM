"""Compact replicas (p2p.compact): packing round-trips, and compact.cu's tiles multiply as a dense matmul does, within
rounding. One GPU."""
import pytest
import torch

from p2p import compact, wanda

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _pruned(rows: int, cols: int, keep: float, dtype=torch.float16, seed: int = 0) -> tuple:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(rows, cols, generator=g, device="cuda").to(dtype)
    norms = torch.rand(cols, generator=g, device="cuda") + 0.1
    pruned, mask = wanda.prune(w, keep, norms, with_mask=True)
    return pruned, mask


@pytest.mark.parametrize("rows,cols,keep", [(64, 64, 50), (100, 77, 7), (256, 5120, 40), (130, 1000, 90), (8, 33, 5)])
def test_pack_round_trips(rows, cols, keep):
    w, mask = _pruned(rows, cols, keep)
    values, words = compact.pack(w, mask)
    assert values.shape == (rows, max(1, round(keep / 100 * cols))) and words.dtype == torch.int32
    assert torch.equal(compact.unpack(values, words, cols), w)


def test_compact_method_puts_blocks_in_row_order():
    _cuda()
    a, ma = _pruned(48, 96, 30)
    b, _ = _pruned(32, 96, 100, seed=1)
    method = compact.Compact([("sparse", *compact.pack(a, ma), 96), ("dense", b)])
    x = torch.randn(2, 5, 96, device="cuda").half()
    want = x.float() @ torch.cat([a, b]).float().T
    got = method.apply(None, x).float()
    assert got.shape == (2, 5, 80) and (got - want).abs().max() <= 1e-2 * want.abs().max()


def _cuda():
    if compact._ext() is None:
        pytest.skip("compact.cu doesn't build here")
    return compact._ext()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("t", [1, 3, 17, 32, 64, 100])
@pytest.mark.parametrize("rows,cols,keep", [(512, 5120, 40), (300, 1000, 7), (96, 80, 90), (1024, 2048, 25), (17, 8, 50)])
def test_tiles_match_a_dense_matmul(dtype, t, rows, cols, keep):
    """compact.cu's tiles, any shape (rows and columns not whole tiles and chunks) and up to 64 tokens a launch."""
    _cuda()
    w, mask = _pruned(rows, cols, keep, dtype)
    tiles = compact.Tiles([("sparse", *compact.pack(w, mask), cols)])
    x = torch.randn(t, cols, device="cuda").to(dtype)
    want = x.float() @ w.float().T
    got = tiles(x).float()
    assert got.shape == (t, rows) and (got - want).abs().max() <= 1e-2 * want.abs().max()


@pytest.mark.parametrize("splits", [1, 3, 7])
def test_tiles_split_along_the_columns_add_up(splits):
    ext = _cuda()
    w, mask = _pruned(256, 5120, 35, torch.bfloat16)
    t = compact.Tiles([("sparse", *compact.pack(w, mask), 5120)])
    x = torch.randn(32, 5120, device="cuda").bfloat16()
    want = x.float() @ w.float().T
    got = ext.spmm(x, t.vals, t.woff, t.masks, t.rows, t.cols, t.vcap, splits).float()
    assert (got - want).abs().max() <= 1e-2 * want.abs().max()


def test_tiles_hold_a_module_at_about_its_size():
    """A module's blocks as one matrix of tiles: a dense block keeps every weight, the masks cost a bit each."""
    _cuda()
    a, ma = _pruned(300, 1024, 40, torch.bfloat16)
    b, _ = _pruned(100, 1024, 100, torch.bfloat16, seed=1)
    values, words = compact.pack(a, ma)
    tiles = compact.Tiles([("sparse", values, words, 1024), ("dense", b)])
    stored = (values.numel() + b.numel()) * 2 + 400 * 1024 / 8
    assert tiles.rows == 400 and stored <= tiles.nbytes() <= 1.05 * stored


def test_compact_moe_expands_the_routed_experts_and_renumbers_them():
    """CompactMoE expands only the experts the tokens are routed to, renumbers the tokens' expert ids into them, and
    hands both to the layer's own method: here a reference MoE that reads the layer's weights as SGLang's does."""
    from typing import NamedTuple
    import torch.nn as nn
    import torch.nn.functional as F

    class TopK(NamedTuple):
        topk_weights: torch.Tensor
        topk_ids: torch.Tensor

    class Dispatch(NamedTuple):
        hidden_states: torch.Tensor
        topk_output: TopK

    class Method:
        def apply(self, layer, d):
            x, t, out = d.hidden_states.float(), d.topk_output, 0
            for j in range(t.topk_ids.shape[1]):
                gate, up = torch.einsum("trh,th->tr", layer.w13_weight[t.topk_ids[:, j]].float(), x).chunk(2, -1)
                out = out + t.topk_weights[:, j:j + 1] * torch.einsum("thr,tr->th", layer.w2_weight[t.topk_ids[:, j]].float(),
                                                                      F.silu(gate) * up)
            return out

    E, R, H, T = 16, 32, 64, 5
    g = torch.Generator(device="cuda").manual_seed(0)
    w13 = torch.randn(E, 2 * R, H, generator=g, device="cuda").half()
    w2 = torch.randn(E, H, R, generator=g, device="cuda").half()
    blocks, pruned = {"w13_weight": [], "w2_weight": []}, {}
    for name, w, rows in (("w13_weight", w13, (slice(0, R), slice(R, 2 * R))), ("w2_weight", w2, (slice(None),))):
        parts = []
        for r in rows:
            norms = torch.rand(E, 1, w.shape[-1], generator=g, device="cuda") + 0.1
            p, mask = wanda.prune(w[:, r], 40, norms, with_mask=True)
            values, words = compact.pack(p.reshape(-1, w.shape[-1]), mask.reshape(-1, w.shape[-1]))
            blocks[name].append(("sparse", values.view(E, p.shape[1], -1), words.view(E, p.shape[1], -1), w.shape[-1]))
            parts.append(p)
        pruned[name] = torch.cat(parts, dim=1)
    dense, empty = nn.Module(), nn.Module()
    for name in pruned:
        dense.register_parameter(name, nn.Parameter(pruned[name], requires_grad=False))
        empty.register_parameter(name, nn.Parameter(pruned[name].new_empty(0), requires_grad=False))
    ids = torch.tensor([[3, 11], [11, 0], [7, 3], [15, 7], [0, 11]], device="cuda")     # 5 of 16 experts used
    d = Dispatch(torch.randn(T, H, generator=g, device="cuda").half(), TopK(torch.rand(T, 2, device="cuda"), ids))
    got = compact.CompactMoE(Method(), blocks).apply(empty, d)
    assert torch.equal(got, Method().apply(dense, d))
    assert all(empty._parameters[name].numel() == 0 for name in pruned)         # the layer's own weights are back
