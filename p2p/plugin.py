"""SGLang hooks and RPC methods, loaded in every scheduler process through the `sglang.srt.plugins` entry point.

Requests with custom_params {"force": [token ids]} are teacher forced. While a reference is being recorded their final
hidden states are kept; otherwise each decode step scores KL(reference || model) against them. Under SGLang's NGRAM
speculative decoding (p2p scan/kl --draft-k), forced requests draft their own solution instead, and every verify pass
is scored and kept whole, so replicas run on their own state inside each pass, as they would with a real drafter.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


class _State:
    scheduler = None
    logits_call = None         # the last LogitsProcessor call: (original, self, input_ids, hidden states, lm head, batch)
    calibration = None
    sums = {}
    reference = {}             # {rid: [final hidden state per solution token after the first]}
    recording = False
    kl = agree = tokens = 0
    tree_scores = None        # P2P_TREES: the scores of the last tree the drafter organized
    tree_pool = None          # P2P_TREES: every candidate the drafter considered for it


S = _State()


def _forced(reqs) -> list:
    return [(r.sampling_params.custom_params or {}).get("force") for r in reqs]


def _logits(original, proc, input_ids, hidden_states, lm_head, metadata, *args, **kwargs):
    S.logits_call = original, proc, input_ids, hidden_states, lm_head, metadata
    out = original(proc, input_ids, hidden_states, lm_head, metadata, *args, **kwargs)
    if metadata.forward_mode.is_target_verify() and getattr(S.scheduler, "cur_batch_for_debug", None) is not None:
        _verify(out.next_token_logits, hidden_states)       # not SGLang's CUDA graph warm-ups and captures: no batch
    return out


def _sample(original, runner, logits_output, fb):
    reqs = S.scheduler.cur_batch_for_debug.reqs if S.scheduler else ()
    forced = _forced(reqs)
    if not any(f is not None for f in forced):
        return original(runner, logits_output, fb)
    if fb.forward_mode.is_decode():
        _score([(i, r.rid, len(r.output_ids) - 1) for i, r in enumerate(reqs)], logits_output.next_token_logits)
    ids = original(runner, logits_output, fb)
    ids[:] = torch.as_tensor([f[len(r.output_ids)] for f, r in zip(forced, reqs)], device=ids.device, dtype=ids.dtype)
    return ids


def _drafts(original, worker, batch):
    """Forced requests draft the next tokens of their solution, as a chain."""
    drafts, mask = original(worker, batch)
    forced, n = _forced(batch.reqs), worker.draft_token_num
    if not any(f is not None for f in forced):
        return drafts, mask
    drafts = [(list(r.output_ids[-1:]) + f[len(r.output_ids):] + [0] * n)[:n] for r, f in zip(batch.reqs, forced)]
    return np.array(drafts, dtype=np.int64).ravel(), np.tile(np.tril(np.ones((n, n), dtype=bool)).ravel(), len(drafts))


def _verify(logits, hidden) -> None:
    """A verify pass of forced requests: score each position whose next token is in the solution, then make that token
    the model's choice, so the pass keeps all its drafts."""
    reqs = S.scheduler.cur_batch_for_debug.reqs
    forced = _forced(reqs)
    if not any(f is not None for f in forced):
        return
    width = len(logits) // len(reqs)
    rows = [(i * width + j, r.rid, len(r.output_ids) - 1 + j, f[len(r.output_ids) + j])
            for i, (r, f) in enumerate(zip(reqs, forced)) for j in range(min(width, len(f) - len(r.output_ids)))]
    if not rows:
        return
    _score([x[:3] for x in rows], logits)
    at = [x[0] for x in rows]
    logits[at] = 0
    logits[at, [x[3] for x in rows]] = 1


def _score(rows: list, logits) -> None:
    """rows: (row of the batch, request id, index into its reference) per scored token. While recording, keep the final
    hidden states; otherwise add up KL(reference || model)."""
    logits_fn, proc, input_ids, hidden, lm_head, metadata = S.logits_call
    at = [row for row, _, _ in rows]
    if S.recording:
        for row, rid, _ in rows:
            S.reference.setdefault(rid, []).append(hidden[row].cpu())
        return
    lq = F.log_softmax(logits[at].float(), -1)     # a copy: the reference call below may reuse the logits buffer
    ref = hidden.clone()
    ref[at] = torch.stack([S.reference[rid][i] for _, rid, i in rows]).to(hidden)
    lp = F.log_softmax(logits_fn(proc, input_ids, ref, lm_head, metadata).next_token_logits[at].float(), -1)
    S.kl += (lp.exp() * (lp - lq)).sum().item()
    S.agree += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
    S.tokens += len(rows)


def _scores(original, score_list, token_list, parents_list, num_draft_token):
    """P2P_TREES: the drafter's score (the product of its probabilities along the path) of each node a pass verifies, and
    every candidate its beam search considered: their tokens, scores, and SGLang's parent list (candidate j's parent is
    candidate parents[j // topk], -1 the root)."""
    out = original(score_list, token_list, parents_list, num_draft_token)
    scores = torch.cat(score_list, dim=1).flatten(1)
    S.tree_scores = scores.gather(1, out[1])
    S.tree_pool = scores, torch.cat(token_list, dim=1), out[0]
    return out


def _tree(original, bonus_tokens, parent_list, top_scores_index, draft_tokens, seq_lens, *args, **kwargs):
    """P2P_TREES: each pass's tree, as SGLang lays it out to verify, a line per request to <dir>/trees_<pid>.jsonl: its
    tokens (the root first), each node's first child and next sibling (-1: none), the scores of all but the root, and the
    request's length without them (bench/pp/trees.py); and the pool of candidates it was chosen from (_scores)."""
    out = original(bonus_tokens, parent_list, top_scores_index, draft_tokens, seq_lens, *args, **kwargs)
    tokens, nxt, sib = out[5], out[3], out[4]
    n = tokens.numel() // seq_lens.numel()
    with open(os.path.join(os.environ["P2P_TREES"], f"trees_{os.getpid()}.jsonl"), "a") as f:
        for i, length in enumerate(seq_lens.tolist()):
            f.write(json.dumps({"length": length, "tokens": tokens[i * n:(i + 1) * n].tolist(),
                                "next": nxt.view(-1, n)[i].tolist(), "sibling": sib.view(-1, n)[i].tolist(),
                                "scores": S.tree_scores[i].float().tolist(),
                                "pool": {"scores": S.tree_pool[0][i].float().tolist(), "tokens": S.tree_pool[1][i].tolist(),
                                         "parents": S.tree_pool[2][i].tolist()}}) + "\n")
    return out


def _without_pp(original, worker, batch, *args, pp_proxy_tensors=None, **kwargs):
    """SGLang 0.5.23 passes pp_proxy_tensors (None without pipeline parallelism) to every drafting worker, but the one for
    Gemma 4's assistant (FROZEN_KV_MTP) takes no such argument."""
    assert pp_proxy_tensors is None
    return original(worker, batch, *args, **kwargs)


def _layers(scheduler):
    from p2p import family
    return family.layers(scheduler.tp_worker.model_runner.model)


def _on_rank0(scheduler) -> bool:
    """The rank that writes results: the first of the TP group that holds the logits (the last pipeline stage)."""
    return scheduler.tp_group.rank_in_group == 0 and scheduler.pp_group.is_last_rank


def _save(obj, path: str) -> None:
    """Write whole, then rename: under pipeline parallelism an RPC returns before the last stage has run it."""
    torch.save(obj, path + ".part")
    os.replace(path + ".part", path)


def p2p_start(self) -> None:
    from p2p import route
    S.scheduler = self
    if route.S.ranks is not None:            # spread ranks hold layers they must not run exact, with or without a plan
        route.install_plan(self, _layers(self), None, {})


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
    assert self.pp_group.world_size == 1, "calibrate on one pipeline stage: each holds only some layers"
    if not path:
        S.calibration = wanda.Calibration(_layers(self))
        return
    S.sums = S.calibration.finish(_layers(self))
    S.calibration = None
    if _on_rank0(self):
        _save(S.sums, path)


def p2p_load_calibration(self, path: str) -> None:
    S.sums = torch.load(path)


def p2p_prepare(self, plan: dict, layers: list, out: str) -> None:
    """Write these layers' replicas under plan, pruned from the weights this engine loaded (p2p.prepare)."""
    from p2p import prepare
    prepare.write(_layers(self), plan, S.sums, layers, out)


def p2p_plan(self, plan: "dict | None" = None) -> None:
    """Install a plan, or the exact model without one."""
    from p2p import replica, route
    replica.remove(_layers(self))
    torch.cuda.empty_cache()
    if route.S.ranks is not None:
        route.install_plan(self, _layers(self), plan, S.sums)
    elif plan:
        replica.install(_layers(self), plan, S.sums)


def p2p_record(self, on: bool) -> None:
    """Start or stop recording the reference from the next forced requests."""
    S.recording = on


def p2p_save_reference(self, path: str) -> None:
    if _on_rank0(self):
        _save({k: torch.stack(v) for k, v in S.reference.items()}, path)


def p2p_load_reference(self, path: str) -> None:
    S.reference = {k: list(v) for k, v in torch.load(path).items()}


def p2p_result(self, path: str) -> None:
    """Write {"kl", "top1", "tokens"} since the last call to path."""
    if _on_rank0(self):
        json.dump({"kl": S.kl / max(S.tokens, 1), "top1": S.agree / max(S.tokens, 1), "tokens": S.tokens}, open(path, "w"))
    S.kl = S.agree = S.tokens = 0


def install() -> None:
    from p2p import pp, route
    pp.install_launcher()                   # in the process that launches the schedulers: their GPUs, their sockets
    route.install_launcher()
    if "sglang.srt.managers.scheduler" not in sys.modules:         # this process doesn't run the model
        return
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.model_executor.model_runner.ModelRunner.sample", _sample, HookType.AROUND)
    HookRegistry.register("sglang.srt.layers.logits_processor.LogitsProcessor.forward", _logits, HookType.AROUND)
    HookRegistry.register("sglang.srt.speculative.frozen_kv_mtp_worker_v2.FrozenKVMTPWorkerV2.forward_batch_generation",
                          _without_pp, HookType.AROUND)
    HookRegistry.register("sglang.srt.speculative.ngram_worker.NGRAMWorker._prepare_draft_tokens", _drafts, HookType.AROUND)
    if os.environ.get("P2P_TREES"):
        HookRegistry.register("sglang.srt.speculative.eagle_utils.organize_draft_results", _scores, HookType.AROUND)
        HookRegistry.register("sglang.srt.speculative.eagle_utils.build_tree_kernel_efficient", _tree, HookType.AROUND)
    for fn in (p2p_start, p2p_shapes, p2p_calibrate, p2p_load_calibration, p2p_prepare, p2p_plan, p2p_record,
               p2p_save_reference, p2p_load_reference, p2p_result):
        setattr(Scheduler, fn.__name__, fn)
    from p2p import pp, prepare, route, stream
    Scheduler.p2p_stream_check = stream.p2p_stream_check
    pp.install()
    route.install()
    prepare.install()
