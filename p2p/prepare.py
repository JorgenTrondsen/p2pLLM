"""Replicas pruned ahead of time from a plan, as a node prepares its own before deployment (p2p replicas).

SGLang loads the copied layers on one GPU and nothing else of the model; each projection is pruned as a plan prunes it
on one GPU and written compact (p2p.compact) to dir/layer<i>.safetensors. A stage with P2P_PP_REPLICAS=dir replaces the
copied layer's projections with these (install_replica).
"""
from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from p2p import compact, family, wanda

SAVES = 0.9            # a pruned block is stored compact only if that takes at most this share of its dense bytes


def install() -> None:
    """In an engine started by replicas(): hold only P2P_PREPARE_LAYERS ("start,end"), as a stage in the middle would."""
    if not os.environ.get("P2P_PREPARE_LAYERS"):
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from p2p import route
    start, end = (int(x) for x in os.environ["P2P_PREPARE_LAYERS"].split(","))
    route.patch_indices(lambda original, num_layers, pp_rank, pp_size: (start, end))
    route.patch_stage()
    HookRegistry.register("sglang.srt.model_loader.loader.DefaultModelLoader.load_model", _alone, HookType.AROUND)


def _alone(original, loader, **kwargs):
    from p2p import route
    route.S.pretend = {"is_first_rank": False, "is_last_rank": False}
    try:
        return original(loader, **kwargs)
    finally:
        route.S.pretend = {}


def write(layers, plan: dict, sums: dict, which: list, out: str) -> None:
    """dir/layer<i>.safetensors for each layer in which: per projection tensor its row blocks, each dense (kept whole,
    or where compact would save too little) or pruned and packed."""
    for i in which:
        layer, keep = layers[i], plan["keep"][str(i)]
        modules, projections, tensors, meta = dict(layer.named_modules()), family.projections(layer), {}, {}
        for name, p in projections.items():
            module = modules[p.path]
            w = getattr(module, p.param).data
            block, cols = w[p.rows], w.shape[-1]
            blocks = meta.setdefault(p.key, {"cols": cols, "blocks": []})["blocks"]
            if keep[name] >= 100:
                tensors[f"{p.key}:{name}:dense"] = block
                blocks.append([name, block.shape[-2], 0])
                continue
            group = family.split_group(module, p)
            pruned, mask = wanda.prune(block, keep[name], wanda.channel_norms(sums[i, p.key], w, group), group,
                                       with_mask=True)
            values, words = compact.pack(pruned.reshape(-1, cols), mask.reshape(-1, cols))
            if values.nbytes + words.nbytes > SAVES * block.nbytes:    # kept nearly whole: compact saves too little
                tensors[f"{p.key}:{name}:dense"] = pruned
                blocks.append([name, block.shape[-2], 0])
                continue
            lead = block.shape[:-1]
            tensors[f"{p.key}:{name}:values"], tensors[f"{p.key}:{name}:mask"] = values.view(*lead, -1), words.view(*lead, -1)
            blocks.append([name, block.shape[-2], values.shape[-1]])
        path = os.path.join(out, f"layer{i}.safetensors")
        save(tensors, {"layer": i, "keep": keep, "modules": meta}, path)
        held = {id(t): t for t in (getattr(modules[p.path], p.param) for p in projections.values())}
        dense = sum(t.numel() * t.element_size() for t in held.values())
        print(f"layer {i}: {os.path.getsize(path) / 1e6:.0f} MB, {os.path.getsize(path) / dense:.0%} of its dense projections",
              flush=True)


def save(tensors: dict, meta: dict, path: str) -> None:
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, path, metadata={"p2p": json.dumps(meta)})


def load(path: str) -> tuple:
    """(tensors, metadata) that save wrote."""
    with safe_open(path, "pt") as h:
        return {k: h.get_tensor(k) for k in h.keys()}, json.loads(h.metadata()["p2p"])


def install_replica(model, file: str, form: str = "compact") -> dict:
    """Replace a copied layer's projections with the replica in file and return its metadata: compact as their
    modules' quant_method, or expanded (form "dense", a check)."""
    import torch.nn as nn
    tensors, meta = load(file)
    layer = family.layers(model)[meta["layer"]]
    modules, projections = dict(layer.named_modules()), family.projections(layer)
    where, experts = {p.key: (p.path, p.param) for p in projections.values()}, {}
    first = lambda p: ((p.rows[-1] if isinstance(p.rows, tuple) else p.rows).start or 0)   # its first row in the tensor
    for key, info in meta["modules"].items():
        path, param = where[key]
        module = modules[path]
        w = getattr(module, param)
        parts = []
        for p, rows, k in info["blocks"]:
            block = ("dense", tensors[f"{key}:{p}:dense"]) if k == 0 else \
                ("sparse", tensors[f"{key}:{p}:values"], tensors[f"{key}:{p}:mask"], info["cols"])
            parts.append((first(projections[p]), tuple(t.to(w.device, w.dtype if t.is_floating_point() else t.dtype)
                                                       if torch.is_tensor(t) else t for t in block)))
        parts = [b for _, b in sorted(parts, key=lambda x: x[0])]
        if form == "dense" or all(b[0] == "dense" for b in parts):     # nothing compact: SGLang's own method runs it
            w.data.copy_(torch.cat([compact.expand(b) for b in parts], dim=-2))
        elif param == "weight":
            module.quant_method = compact.Compact(parts)
            module.weight = nn.Parameter(w.new_empty(w.shape[0], 0), requires_grad=False)
        else:
            experts.setdefault(path, {})[param] = parts
    for path, weights in experts.items():
        module = modules[path]
        module.quant_method = compact.CompactMoE(module.quant_method, weights)
        for param in weights:
            setattr(module, param, nn.Parameter(getattr(module, param).new_empty(0), requires_grad=False))
    torch.cuda.empty_cache()
    return meta


def replicas(a) -> int:
    from p2p.engine import Model
    plan = json.load(open(a.plan))
    layers = a.layers or [g[-2] for g in plan["groups"]]
    os.makedirs(a.out, exist_ok=True)
    os.environ["P2P_PREPARE_LAYERS"] = f"{min(layers)},{max(layers) + 1}"
    try:
        m = Model(a.model, tp=1, batch=1, max_total_tokens=1024, skip_server_warmup=True, dtype=a.dtype,
                  mem_fraction_static=a.mem_fraction)
    finally:
        del os.environ["P2P_PREPARE_LAYERS"]
    try:
        m.rpc("p2p_load_calibration", path=a.calib)
        m.rpc("p2p_prepare", plan=plan, layers=layers, out=os.path.abspath(a.out))
    finally:
        m.close()
    missing = [i for i in layers if not os.path.exists(os.path.join(a.out, f"layer{i}.safetensors"))]
    if missing:
        raise RuntimeError(f"no replicas written for layers {missing}: see the engine's log")
    return 0
