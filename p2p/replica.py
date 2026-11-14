"""Run groups of layers in parallel inside SGLang's own forward, on decode steps and speculative decoding's verify passes.

Every member of a group reads the group's input h. The last member's node also holds a shrunk copy of the member before
it, which gives that member a better input: h~ = copy(h). The next step reads

    h' = l_0(h) + sum_{k>0} (l_k(x_k) - x_k),    x_k = h~ for the last member, else h

added in float32, each difference rounded to the model's dtype first, as a node sends it to the next step's (p2p.route).
Prefill stays exact. The copy restarts from the copied layer's state every step, as if that layer's node sends its update
for each finished token: it runs just before that layer, on the same K/V and state slots, and leaves no trace. A verify
pass carries several tokens, and the copy keeps its own K/V and state for them, since the copied layer's updates for
the pass arrive too late.
"""
from __future__ import annotations

import copy
import functools
import inspect

import torch.nn as nn
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

from p2p import family, wanda


def _shallow(m: nn.Module) -> nn.Module:
    """A copy of a module tree that shares every tensor until one is replaced."""
    c = copy.copy(m)
    c._parameters, c._buffers = dict(m._parameters), dict(m._buffers)
    c._modules = {k: _shallow(v) for k, v in m._modules.items()}
    return c


def shrink(layer: nn.Module, keep: dict, sums: dict, i: int) -> nn.Module:
    """A copy of layer i that keeps `keep[projection]` percent of each projection's weights. sums: Calibration.finish()."""
    c = _shallow(layer)
    c.__dict__.pop("forward", None)
    modules = dict(c.named_modules())
    weights = {}
    for name, p in family.projections(layer).items():
        if keep[name] < 100:
            module = modules[p.path]
            if (p.path, p.param) not in weights:
                weights[p.path, p.param] = getattr(module, p.param).data.clone()
            w, group = weights[p.path, p.param], family.split_group(module, p)
            w[p.rows] = wanda.prune(w[p.rows], keep[name], wanda.channel_norms(sums[i, p.key], w, group), group)
    for (path, param), w in weights.items():
        setattr(modules[path], param, nn.Parameter(w, requires_grad=False))
    return c


def _by_name(layer: nn.Module, args: tuple, kw: dict) -> dict:
    """A layer call's arguments by name, since families pass them differently."""
    named = dict(inspect.signature(type(layer).forward).bind(layer, *args, **kw).arguments)
    named.pop("self")
    return {**named, **named.pop("kwargs", {})}


def _run(module: nn.Module, x, named: dict):
    """module as a plain function of x: starts mid-stack and completes its output's TP sum and residual."""
    x = x.clone()                          # the stream may update its residual in place
    fb = named["forward_batch"]
    residual_batch.set_written(x, fb)
    out = type(module).forward(module, **{**named, "hidden_states": x})
    out, residual = residual_batch.stream_of(fb).export(out)
    return out if residual is None else out + residual


def _member(group: list, replica: nn.Module, k: int, *args, **kw):
    layer = group[k]
    named = _by_name(layer, args, kw)
    fb, h = named["forward_batch"], named["hidden_states"]
    if not (fb.forward_mode.is_decode() or fb.forward_mode.is_target_verify()):
        return type(layer).forward(layer, **named)
    if k:                                  # the first member did the work
        return h
    out, residual = residual_batch.stream_of(fb).export(h)
    h = out if residual is None else out + residual
    with family.restored_state(group[-2]):
        h_copy = _run(replica, h, named)
    total = _run(group[0], h, named).float()
    for layer in group[1:-1]:
        total += (_run(layer, h, named).float() - h.float()).to(h.dtype)
    total += (_run(group[-1], h_copy, named).float() - h_copy.float()).to(h.dtype)
    return residual_batch.set_written(total.to(h.dtype), fb)


def install(layers: nn.ModuleList, plan: dict, sums: dict, groups: "list | None" = None) -> None:
    """Apply a plan (see search.plan), replacing any plan installed before; only `groups` of it, if given."""
    remove(layers)
    for group in plan["groups"] if groups is None else groups:
        copied = group[-2]
        replica = shrink(layers[copied], plan["keep"][str(copied)], sums, copied)
        members = [layers[i] for i in group]
        for k, layer in enumerate(members):
            layer.forward = functools.partial(_member, members, replica, k)


def remove(layers: nn.ModuleList) -> None:
    for layer in layers:
        layer.__dict__.pop("forward", None)
