"""What we need to know about a model family: its layers, projections and linear-attention state.

To support a family with other projection names, add them to TABLE. Row blocks come from each module's output_sizes.
Routed MoE experts live in one module with one tensor per kind of projection, so they are projections too (_routed_experts).
"""
from __future__ import annotations

import contextlib
from typing import NamedTuple

import torch.nn as nn

TABLE = {"qkv_proj": ("q", "k", "v"), "in_proj_qkvz": ("q", "k", "v", "z"), "o_proj": ("o",), "out_proj": ("out",),
         "gate_up_proj": ("gate", "up"), "down_proj": ("down",)}


class Projection(NamedTuple):
    path: str                # the module inside its layer
    rows: "slice | tuple"    # where it is in that module's tensor, on this rank
    shape: list              # [rows, cols] over all ranks
    param: str = "weight"    # that module's tensor

    @property
    def key(self) -> str:
        """What its calibration sums are stored under."""
        return self.path if self.param == "weight" else f"{self.path}.{self.param}"


def layers(model: nn.Module) -> nn.ModuleList:
    """The decoder layers: the longest module list of layers with the MLP in TABLE (a vision tower can be longer). Under
    pipeline parallelism the layers a rank doesn't hold are placeholders."""
    held = lambda m: next((x for x in m if type(x).__name__ != "PPMissingLayer"), None)
    found = [m for m in model.modules() if isinstance(m, nn.ModuleList) and held(m) is not None
             and {"gate_up_proj", "down_proj"} <= {path.rsplit(".", 1)[-1] for path, _ in held(m).named_modules()}]
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
        elif hasattr(m, "w13_weight"):
            out |= _routed_experts(path, m)
    if not {"gate", "up", "down"} <= set(out) or not {"q", "k", "v"} <= set(out):
        raise ValueError(f"{type(layer).__name__}: add its attention and MLP projections to p2p.family.TABLE")
    _check_covered(layer, out)
    return out


def _check_covered(layer: nn.Module, projected: dict) -> None:
    """Stop if a large part of the layer isn't among the projections: replica sizes would look far smaller than they are."""
    modules = dict(layer.named_modules())
    known = {id(w) for p in projected.values() for w in modules[p.path].parameters()}
    counted = sum(rows * cols for rows, cols in (p.shape for p in projected.values()))
    other = sum(w.numel() for w in layer.parameters() if id(w) not in known)
    if counted < 0.9 * (counted + other):
        raise ValueError(f"{type(layer).__name__}: only {100 * counted / (counted + other):.0f}% of its weights are in "
                         "p2p.family.TABLE, so replica sizes would be wrong")


def _routed_experts(path: str, m: nn.Module) -> dict:
    """A FusedMoE's experts: w13_weight [expert, 2 * rows, hidden] holds gate then up, w2_weight [expert, hidden, rows] holds down."""
    config = m.moe_runner_config
    if m.use_triton_kernels or not config.is_gated or config.activation != "silu" or m.num_local_experts != m.num_experts:
        raise ValueError(f"{path}: p2p handles gated SiLU experts in the default layout, all on one rank group")
    rows, total = m.intermediate_size_per_partition, m.intermediate_size_per_partition * m.moe_tp_size
    gate_up = [m.num_experts * total, m.hidden_size]
    return {"egate": Projection(path, (slice(None), slice(0, rows)), gate_up, "w13_weight"),
            "eup": Projection(path, (slice(None), slice(rows, 2 * rows)), gate_up, "w13_weight"),
            "edown": Projection(path, (slice(None), slice(None)), [m.num_experts * m.hidden_size, total], "w2_weight")}


def split_group(module: nn.Module, p: Projection):
    """The TP group whose ranks each hold only some of this projection's columns, or None if every rank holds them all."""
    if getattr(module, p.param).shape[-1] == p.shape[1]:
        return None
    if hasattr(module, "tp_group"):
        return module.tp_group
    from sglang.srt.runtime_context import get_parallel
    return get_parallel().tp_group


@contextlib.contextmanager
def restored_state(*layers: nn.Module):
    """Undo what running a copy, or a verify pass more than once (p2p.stream: SGLang's tree convolution moves the conv
    state on), does to linear-attention layers' conv and recurrent state at this batch's slots. Attention layers need
    nothing: the real layer overwrites the copy's K/V for the current token."""
    linear = [layer for layer in layers if hasattr(layer, "linear_attn")]
    if not linear:
        yield
        return
    from sglang.srt.model_executor.forward_context import get_attn_backend
    backend = get_attn_backend()
    backend = getattr(backend, "linear_attn_backend", backend)
    slots = backend.forward_metadata.mamba_cache_indices.long()
    states = [backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id) for layer in linear]
    saved = [(t, t[slots].clone()) for state in states for t in (*state.conv, state.temporal)]
    yield
    for t, values in saved:
        t[slots] = values
