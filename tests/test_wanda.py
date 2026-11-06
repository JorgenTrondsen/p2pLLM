"""Wanda's choice of weights, with fake TP ranks."""
import types

import pytest
import torch

from p2p import data, family, wanda


class FakeGroup:
    """all_gather across ranks that all live in this process."""

    def __init__(self, parts, rank):
        self.parts, self.rank_in_group = parts, rank

    def all_gather(self, x, dim=0):
        return torch.cat([x if r == self.rank_in_group else p for r, p in enumerate(self.parts)], dim)


def test_keeps_the_top_scores_in_each_row():
    w = torch.tensor([[1.0, -5.0, 2.0, 3.0], [4.0, 1.5, -1.0, 0.5]])
    got = wanda.prune(w, 50, torch.ones(4))
    assert got.tolist() == [[0.0, -5.0, 0.0, 3.0], [4.0, 1.5, 0.0, 0.0]]


def test_ties_keep_exactly_k_weights():
    w = torch.ones(3, 10)
    assert (wanda.prune(w, 30, torch.ones(10)) != 0).sum(1).tolist() == [3, 3, 3]


def test_norms_change_what_is_kept():
    w = torch.tensor([[1.0, 2.0]])
    assert wanda.prune(w, 50, torch.tensor([10.0, 1.0])).tolist() == [[1.0, 0.0]]


def test_matches_the_drafts_formula():
    torch.manual_seed(0)
    w, norms = torch.randn(50, 200), torch.rand(200)
    score = w.abs() * norms[None]
    k = round(0.33 * 200)
    draft = w * torch.zeros_like(score, dtype=torch.bool).scatter_(1, score.topk(k, dim=1).indices, True)
    assert torch.equal(wanda.prune(w, 33, norms), draft)


def test_row_parallel_ranks_choose_as_one_matrix():
    torch.manual_seed(1)
    w, norms = torch.randn(20, 100), torch.rand(100)
    whole = wanda.prune(w, 33, norms)
    cols = [w[:, :50], w[:, 50:]]
    parts = [c.abs() * n[None] for c, n in zip(cols, (norms[:50], norms[50:]))]
    got = [wanda.prune(c, 33, n, FakeGroup(parts, r)) for r, (c, n) in enumerate(zip(cols, (norms[:50], norms[50:])))]
    assert torch.equal(torch.cat(got, 1), whole)


def test_row_parallel_norms_are_sliced_per_rank():
    group = types.SimpleNamespace(rank_in_group=1)
    assert wanda.channel_norms(torch.tensor([1.0, 4.0, 9.0, 16.0]), torch.zeros(2, 2), group).tolist() == [3.0, 4.0]


def test_each_expert_is_pruned_on_its_own_inputs():
    w = torch.ones(2, 3, 4)                   # [expert, rows, columns]: all weights equal, so the norms alone decide
    sums = torch.tensor([[9.0, 1, 1, 1], [1, 1, 1, 9.0]])
    got = wanda.prune(w, 25, wanda.channel_norms(sums, w))
    assert got.sum().item() == 6 and got[0, :, 0].sum().item() == 3 and got[1, :, 3].sum().item() == 3


def test_expert_inputs_count_only_routed_tokens():
    torch.manual_seed(0)
    m = types.SimpleNamespace(w13_weight=torch.randn(3, 4, 5), w2_weight=torch.randn(3, 5, 2))
    x, ids = torch.randn(4, 5), torch.tensor([[0, 2], [0, 2], [2, 0], [2, 0]])
    into, gated = wanda.expert_inputs(m, x, ids)
    gate, up = (x @ m.w13_weight[2].T).chunk(2, -1)
    assert torch.allclose(into[2].float(), x.pow(2).sum(0)) and into[1].sum() == 0
    assert torch.allclose(gated[2].float(), (torch.nn.functional.silu(gate) * up).pow(2).sum(0)) and gated[1].sum() == 0


def test_a_layer_mostly_outside_the_table_is_refused():
    class Linear(torch.nn.Linear):
        def __init__(self, rows, cols, *names):
            super().__init__(cols, rows, bias=False)
            self.input_size, self.output_sizes = cols, [rows // len(names)] * len(names)
            self.output_partition_sizes = self.output_sizes

    class Layer(torch.nn.Module):
        def __init__(self, unknown):
            super().__init__()
            self.qkv_proj, self.gate_up_proj = Linear(30, 8, "q", "k", "v"), Linear(20, 8, "gate", "up")
            self.down_proj, self.experts = Linear(8, 10, "down"), torch.nn.Linear(unknown, unknown, bias=False)

    assert set(family.projections(Layer(2))) == {"q", "k", "v", "gate", "up", "down"}
    with pytest.raises(ValueError, match="family.TABLE"):
        family.projections(Layer(100))


def test_extract_and_correct_match_the_draft():
    assert data.extract("so \\boxed{1,234}") == "1234"
    assert data.extract("about 3 then 4.5") == "4.5"
    assert data.extract("nothing") == ""
    assert data.correct("\\boxed{18}", "18") and data.correct("\\boxed{18.0}", "18") and not data.correct("\\boxed{19}", "18")
    assert not data.correct("no number", "18")
    assert data.correct("the answer is Paris", "Paris") and not data.correct("London", "Paris")


def test_an_expert_no_token_reached_keeps_its_largest_weights():
    w = torch.tensor([[[1.0, 3.0, 2.0]], [[1.0, 3.0, 2.0]]])
    sums = torch.tensor([[100.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    out = wanda.prune(w, 34, wanda.channel_norms(sums, w))
    assert out.tolist() == [[[1.0, 0.0, 0.0]], [[0.0, 3.0, 0.0]]]
