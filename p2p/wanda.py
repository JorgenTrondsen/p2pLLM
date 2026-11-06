"""Wanda pruning: keep the weights that matter most given what the layer reads, score |w_ij| * ||x_j|| in each row."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from p2p import family


def expert_inputs(m, x: torch.Tensor, ids: torch.Tensor) -> tuple:
    """Sums of squares of what a FusedMoE's experts read, per channel and over the tokens routed to each: for gate and up
    [expert, hidden], for down [expert, rows]. ids: each token's experts."""
    x = x.float()
    into = x.new_zeros(m.w13_weight.shape[0], m.w13_weight.shape[-1], dtype=torch.double)
    gated = x.new_zeros(m.w2_weight.shape[0], m.w2_weight.shape[-1], dtype=torch.double)
    for e in ids.unique().tolist():
        routed = x[(ids == e).any(-1)]
        gate, up = (routed @ m.w13_weight[e].float().T).chunk(2, -1)
        into[e], gated[e] = routed.pow(2).sum(0).double(), (F.silu(gate) * up).pow(2).sum(0).double()
    return into, gated


class Calibration:
    """Per-channel sums of squares of every projection's input, collected on one rank through forward pre-hooks."""

    def __init__(self, layers):
        self.sums, self.handles = {}, []
        for i, layer in enumerate(layers):
            modules = dict(layer.named_modules())
            for path in {p.path for p in family.projections(layer).values()}:
                hook = self._experts if hasattr(modules[path], "w13_weight") else self._linear
                self.handles.append(modules[path].register_forward_pre_hook(hook(i, path)))

    def _add(self, key, x) -> None:
        self.sums[key] = self.sums.get(key, 0) + x

    def _linear(self, i, path):
        def hook(module, args):
            self._add((i, path), args[0].reshape(-1, args[0].shape[-1]).float().pow(2).sum(0).double())
        return hook

    def _experts(self, i, path):
        def hook(module, args):
            into, gated = expert_inputs(module, args[0], args[1].topk_ids)
            self._add((i, f"{path}.w13_weight"), into)
            self._add((i, f"{path}.w2_weight"), gated)
        return hook

    def finish(self, layers) -> dict:
        """Stop collecting. {(layer, projection key): sums over all ranks}."""
        for h in self.handles:
            h.remove()
        out = {}
        for i, layer in enumerate(layers):
            modules = dict(layer.named_modules())
            for key, p in {p.key: p for p in family.projections(layer).values()}.items():
                if (i, key) in self.sums:
                    x, group = self.sums[i, key], family.split_group(modules[p.path], p)
                    out[i, key] = (group.all_gather(x, dim=x.dim() - 1) if group else x).cpu()
        return out


def channel_norms(sums: torch.Tensor, w: torch.Tensor, group=None) -> torch.Tensor:
    """||x_j|| for the columns of w this rank holds, ready to multiply w: [columns], or [expert, 1, columns] for experts.
    group: as family.split_group."""
    sums = sums.to(w.device)
    if group:
        cols = w.shape[-1]
        sums = sums[..., group.rank_in_group * cols: (group.rank_in_group + 1) * cols]
    norms = sums.sqrt().float()
    if norms.dim() == 2:                   # an expert no token reached has nothing to go on but its weights' size
        return norms.where(norms.sum(-1, keepdim=True) > 0, torch.ones_like(norms))[:, None]
    return norms


def prune(w: torch.Tensor, keep: float, norms: torch.Tensor, group=None) -> torch.Tensor:
    """w with zeros everywhere but the `keep` percent of each row's weights that score highest. A projection split across
    ranks compares scores across them, so the choice doesn't depend on the TP size."""
    score = w.float().abs() * norms
    full = group.all_gather(score, dim=score.dim() - 1) if group else score
    k = max(1, round(keep / 100 * full.shape[-1]))
    mask = torch.zeros_like(full, dtype=torch.bool).scatter_(-1, full.topk(k, dim=-1).indices, True)
    start = group.rank_in_group * w.shape[-1] if group else 0
    return (w.float() * mask[..., start: start + w.shape[-1]]).to(w.dtype)
