"""Wanda pruning: keep the weights that matter most given what the layer reads, score |w_ij| * ||x_j|| in each row."""
from __future__ import annotations

import torch

from p2p import family


class Calibration:
    """Per-channel sums of squares of every projection's input, collected on one rank through forward pre-hooks."""

    def __init__(self, layers):
        self.sums, self.handles = {}, []
        for i, layer in enumerate(layers):
            for path in {p.path for p in family.projections(layer).values()}:
                module = dict(layer.named_modules())[path]
                self.handles.append(module.register_forward_pre_hook(self._hook((i, path))))

    def _hook(self, key):
        def hook(module, args):
            x = args[0].reshape(-1, args[0].shape[-1]).float().pow(2).sum(0).double()
            self.sums[key] = self.sums.get(key, 0) + x
        return hook

    def finish(self, layers) -> dict:
        """Stop collecting. {(layer, module path): sums over all ranks}."""
        for h in self.handles:
            h.remove()
        out = {}
        for (i, path), x in self.sums.items():
            module = dict(layers[i].named_modules())[path]
            out[i, path] = (module.tp_group.all_gather(x, dim=0) if family.is_row_parallel(module) else x).cpu()
        return out


def channel_norms(sums: torch.Tensor, module) -> torch.Tensor:
    """||x_j|| for the columns this rank holds."""
    sums = sums.to(module.weight.device)
    if family.is_row_parallel(module):
        rank = module.tp_group.rank_in_group
        cols = module.weight.shape[1]
        sums = sums[rank * cols: (rank + 1) * cols]
    return sums.sqrt().float()


def prune(w: torch.Tensor, keep: float, norms: torch.Tensor, module) -> torch.Tensor:
    """w with zeros everywhere but the `keep` percent of each row's weights that score highest. A row-parallel projection
    compares scores across ranks, so the choice doesn't depend on the TP size."""
    score = w.float().abs() * norms[None, :]
    parallel = family.is_row_parallel(module)
    full = module.tp_group.all_gather(score, dim=1) if parallel else score
    k = max(1, round(keep / 100 * full.shape[1]))
    mask = torch.zeros_like(full, dtype=torch.bool).scatter_(1, full.topk(k, dim=1).indices, True)
    start = module.tp_group.rank_in_group * w.shape[1] if parallel else 0
    return (w.float() * mask[:, start: start + w.shape[1]]).to(w.dtype)
