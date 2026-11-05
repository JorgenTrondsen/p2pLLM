"""Wanda's choice of weights, with fake TP ranks."""
import types

import torch

from p2p import data, wanda


class FakeGroup:
    """all_gather across ranks that all live in this process."""

    def __init__(self, parts, rank):
        self.parts, self.rank_in_group = parts, rank

    def all_gather(self, x, dim=0):
        return torch.cat([x if r == self.rank_in_group else p for r, p in enumerate(self.parts)], dim)


def module(w, input_size=None, group=None):
    return types.SimpleNamespace(weight=w, input_size=input_size or w.shape[1], tp_group=group)


def test_keeps_the_top_scores_in_each_row():
    w = torch.tensor([[1.0, -5.0, 2.0, 3.0], [4.0, 1.5, -1.0, 0.5]])
    got = wanda.prune(w, 50, torch.ones(4), module(w))
    assert got.tolist() == [[0.0, -5.0, 0.0, 3.0], [4.0, 1.5, 0.0, 0.0]]


def test_ties_keep_exactly_k_weights():
    w = torch.ones(3, 10)
    assert (wanda.prune(w, 30, torch.ones(10), module(w)) != 0).sum(1).tolist() == [3, 3, 3]


def test_norms_change_what_is_kept():
    w = torch.tensor([[1.0, 2.0]])
    assert wanda.prune(w, 50, torch.tensor([10.0, 1.0]), module(w)).tolist() == [[1.0, 0.0]]


def test_matches_the_drafts_formula():
    torch.manual_seed(0)
    w, norms = torch.randn(50, 200), torch.rand(200)
    score = w.abs() * norms[None]
    k = round(0.33 * 200)
    draft = w * torch.zeros_like(score, dtype=torch.bool).scatter_(1, score.topk(k, dim=1).indices, True)
    assert torch.equal(wanda.prune(w, 33, norms, module(w)), draft)


def test_row_parallel_ranks_choose_as_one_matrix():
    torch.manual_seed(1)
    w, norms = torch.randn(20, 100), torch.rand(100)
    whole = wanda.prune(w, 33, norms, module(w))
    cols = [w[:, :50], w[:, 50:]]
    parts = [c.abs() * n[None] for c, n in zip(cols, (norms[:50], norms[50:]))]
    got = [wanda.prune(c, 33, n, module(c, 100, FakeGroup(parts, r))) for r, (c, n) in enumerate(zip(cols, (norms[:50], norms[50:])))]
    assert torch.equal(torch.cat(got, 1), whole)


def test_row_parallel_norms_are_sliced_per_rank():
    w = torch.zeros(2, 2)
    m = module(w, 4, types.SimpleNamespace(rank_in_group=1))
    assert wanda.channel_norms(torch.tensor([1.0, 4.0, 9.0, 16.0]), m).tolist() == [3.0, 4.0]


def test_extract_and_correct_match_the_draft():
    assert data.extract("so \\boxed{1,234}") == "1234"
    assert data.extract("about 3 then 4.5") == "4.5"
    assert data.extract("nothing") == ""
    assert data.correct("\\boxed{18}", "18") and data.correct("\\boxed{18.0}", "18") and not data.correct("\\boxed{19}", "18")
    assert not data.correct("no number", "18")
    assert data.correct("the answer is Paris", "Paris") and not data.correct("London", "Paris")
