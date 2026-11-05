"""SGLang hooks and RPC methods, loaded in every scheduler process through the `sglang.srt.plugins` entry point.

Requests with custom_params {"force": [token ids]} are teacher forced. While a reference is being recorded their final
hidden states are kept; otherwise each decode step scores KL(reference || model) against them.
"""
from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F


class _State:
    scheduler = None
    logits_call = None         # the last LogitsProcessor call: (original, self, input_ids, hidden states, lm head, batch)
    calibration = None
    sums = {}
    reference = {}             # {rid: [final hidden state per decoded token]}
    recording = False
    kl = agree = tokens = 0


S = _State()


def _logits(original, proc, input_ids, hidden_states, lm_head, metadata, *args, **kwargs):
    S.logits_call = original, proc, input_ids, hidden_states, lm_head, metadata
    return original(proc, input_ids, hidden_states, lm_head, metadata, *args, **kwargs)


def _sample(original, runner, logits_output, fb):
    reqs = S.scheduler.cur_batch_for_debug.reqs if S.scheduler else ()
    forced = [(r.sampling_params.custom_params or {}).get("force") for r in reqs]
    if not any(f is not None for f in forced):
        return original(runner, logits_output, fb)
    if fb.forward_mode.is_decode():
        _score(reqs, logits_output.next_token_logits)
    ids = original(runner, logits_output, fb)
    ids[:] = torch.as_tensor([f[len(r.output_ids)] for f, r in zip(forced, reqs)], device=ids.device, dtype=ids.dtype)
    return ids


def _score(reqs, logits) -> None:
    logits_fn, proc, input_ids, hidden, lm_head, metadata = S.logits_call
    if S.recording:
        for r, h in zip(reqs, hidden):
            S.reference.setdefault(r.rid, []).append(h.cpu())
        return
    lq = F.log_softmax(logits.float(), -1)         # the reference call below may reuse the logits buffer
    ref = torch.stack([S.reference[r.rid][len(r.output_ids) - 1] for r in reqs]).to(hidden)
    lp = F.log_softmax(logits_fn(proc, input_ids, ref, lm_head, metadata).next_token_logits.float(), -1)
    S.kl += (lp.exp() * (lp - lq)).sum().item()
    S.agree += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
    S.tokens += len(reqs)


def _layers(scheduler):
    from p2p import family
    return family.layers(scheduler.tp_worker.model_runner.model)


def _on_rank0(scheduler) -> bool:
    return scheduler.tp_group.rank_in_group == 0


def p2p_start(self) -> None:
    S.scheduler = self


def p2p_shapes(self, path: str) -> None:
    """{layer: {"type", "proj": {name: [rows, cols]}}} -> path."""
    from p2p import family
    out = {str(i): {"type": type(layer).__name__, "proj": {n: p.shape for n, p in family.projections(layer).items()}}
           for i, layer in enumerate(_layers(self))}
    if _on_rank0(self):
        json.dump(out, open(path, "w"))


def p2p_calibrate(self, path: str = "") -> None:
    """Start collecting calibration sums, or stop and save them to path (and use them)."""
    from p2p import wanda
    if not path:
        S.calibration = wanda.Calibration(_layers(self))
        return
    S.sums = S.calibration.finish(_layers(self))
    S.calibration = None
    if _on_rank0(self):
        torch.save(S.sums, path)


def p2p_load_calibration(self, path: str) -> None:
    S.sums = torch.load(path)


def p2p_plan(self, plan: "dict | None" = None) -> None:
    """Install a plan, or the exact model without one."""
    from p2p import replica
    replica.remove(_layers(self))
    torch.cuda.empty_cache()
    if plan:
        replica.install(_layers(self), plan, S.sums)


def p2p_record(self, on: bool) -> None:
    """Start or stop recording the reference from the next forced requests."""
    S.recording = on


def p2p_save_reference(self, path: str) -> None:
    if _on_rank0(self):
        torch.save({k: torch.stack(v) for k, v in S.reference.items()}, path)


def p2p_load_reference(self, path: str) -> None:
    S.reference = {k: list(v) for k, v in torch.load(path).items()}


def p2p_result(self, path: str) -> None:
    """Write {"kl", "top1", "tokens"} since the last call to path."""
    if _on_rank0(self):
        json.dump({"kl": S.kl / max(S.tokens, 1), "top1": S.agree / max(S.tokens, 1), "tokens": S.tokens}, open(path, "w"))
    S.kl = S.agree = S.tokens = 0


def install() -> None:
    if "sglang.srt.managers.scheduler" not in sys.modules:         # this process doesn't run the model
        return
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.model_executor.model_runner.ModelRunner.sample", _sample, HookType.AROUND)
    HookRegistry.register("sglang.srt.layers.logits_processor.LogitsProcessor.forward", _logits, HookType.AROUND)
    for fn in (p2p_start, p2p_shapes, p2p_calibrate, p2p_load_calibration, p2p_plan, p2p_record, p2p_save_reference,
               p2p_load_reference, p2p_result):
        setattr(Scheduler, fn.__name__, fn)
