"""What we need to know about a model family: its layers, projections and linear-attention state.

To support a family with other projection names, add them to TABLE. Row blocks come from each module's output_sizes.
"""
from __future__ import annotations

import contextlib
from typing import NamedTuple

import torch.nn as nn

TABLE = {"qkv_proj": ("q", "k", "v"), "in_proj_qkvz": ("q", "k", "v", "z"), "o_proj": ("o",), "out_proj": ("out",),
         "gate_up_proj": ("gate", "up"), "down_proj": ("down",)}


class Projection(NamedTuple):
    path: str                # the module inside its layer
    rows: slice              # its rows of that module's weight, on this rank
    shape: list              # [rows, cols] over all ranks


def layers(model: nn.Module) -> nn.ModuleList:
    """The decoder layers: the longest module list of layers with the MLP in TABLE (a vision tower can be longer)."""
    found = [m for m in model.modules() if isinstance(m, nn.ModuleList) and len(m)
             and {"gate_up_proj", "down_proj"} <= {path.rsplit(".", 1)[-1] for path, _ in m[0].named_modules()}]
    if not found:
        raise ValueError(f"{type(model).__name__}: add its attention and MLP projections to p2p.family.TABLE")
    return max(found, key=len)


def projections(layer: nn.Module) -> dict:
    """{projection name: Projection} of a decoder layer."""
    out = {}
    for path, m in layer.named_modules():
        names = TABLE.get(path.rsplit(".", 1)[-1])
        if names:
            local = getattr(m, "output_partition_sizes", None) or [m.weight.shape[0]]
            start = 0
            for name, total, rows in zip(names, getattr(m, "output_sizes", None) or [m.output_size], local, strict=True):
                out[name] = Projection(path, slice(start, start + rows), [total, m.input_size])
                start += rows
    if not {"gate", "up", "down"} <= set(out) or not {"q", "k", "v"} <= set(out):
        raise ValueError(f"{type(layer).__name__}: add its attention and MLP projections to p2p.family.TABLE")
    return out


def is_row_parallel(module: nn.Module) -> bool:
    """Whether each rank holds only some columns of this projection's rows."""
    return module.weight.shape[1] != module.input_size


@contextlib.contextmanager
def restored_state(layer: nn.Module):
    """Undo what running a copy does to a linear-attention layer's conv and recurrent state at this batch's slots.
    Attention layers need nothing: the real layer overwrites the copy's K/V for the current token."""
    if not hasattr(layer, "linear_attn"):
        yield
        return
    from sglang.srt.model_executor.forward_context import get_attn_backend
    backend = get_attn_backend()
    backend = getattr(backend, "linear_attn_backend", backend)
    slots = backend.forward_metadata.mamba_cache_indices.long()
    state = backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    saved = [(t, t[slots].clone()) for t in (*state.conv, state.temporal)]
    yield
    for t, values in saved:
        t[slots] = values
