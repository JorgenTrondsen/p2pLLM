"""The tree arithmetic of streaming a verify pass in pieces (p2p.route, P2P_PP_STREAM).

SGLang lays a verify tree out parents first, so a pass's first b tokens are a tree of their own: piece k is SGLang's
verify of the first b_k tokens (subtree), whose rows past b_(k-1) are new. settled tells when the rows so far decide a
greedy pass; grown and regrown let a tree leave the head in phases (P2P_PP_PHASES).
p2p_stream_check (one GPU, for tests) compares every piece's rows with the whole pass's.
"""
from __future__ import annotations

import copy
import functools
import inspect
import json

import torch

from p2p import family


def pieces(n: int, sizes, cuts=()) -> list:
    """Where each piece of a pass of n tokens ends: pieces of `sizes` tokens (one size, or one per piece with the last
    repeated), and an end at each of cuts (the rows a phased tree is fixed up to)."""
    sizes = [sizes] if isinstance(sizes, int) else list(sizes)
    out, at = [], 0
    for stop in [c for c in cuts if c < n] + [n]:
        while at < stop:
            at = min(stop, at + sizes[min(len(out), len(sizes) - 1)])
            out.append(at)
    return out


def grown(scores: list, parents: list, topk: int, chosen: set, first: int, last: int, beam: list, room: int) -> list:
    """The candidates a phase adds to a tree, by SGLang's numbering (candidate j's parent is parents[j // topk], -1
    the root), from draft steps first..last-1 only, so they come after the earlier phases' rows in SGLang's layout:
    the beam and its ancestors first, then the best by score whose parent is in, until `room` are added."""
    step = lambda j: 0 if j < topk else 1 + (j - topk) // (topk * topk)
    parent = lambda j: parents[j // topk]
    new = set()
    for b in sorted(beam, key=lambda j: (-scores[j], j)):
        chain = []
        while b >= 0 and b not in chosen and b not in new and step(b) >= first:
            chain.append(b)
            b = parent(b)
        if (b < 0 or b in chosen or b in new) and len(new) + len(chain) <= room:
            new.update(chain)
    for j in sorted((j for j in range(len(scores)) if first <= step(j) < last), key=lambda j: (-scores[j], j)):
        if len(new) >= room:
            break
        if j not in new and (parent(j) < 0 or parent(j) in chosen or parent(j) in new):
            new.add(j)
    return sorted(new)


def regrown(runner, fb, chain, parents, top_scores):
    """fb with another tree in its rows: the tokens `chain` under the topology the head relays, laid out by SGLang's
    tree kernel as every stage lays out the relay. Its K/V writes stay fb's."""
    from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient, default_tree_mask_mode
    info, device = fb.spec_info, fb.input_ids.device
    n = info.draft_token_num
    rows = chain.to(device=device, dtype=torch.int64).reshape(-1, n)
    verify_mask = getattr(runner.attn_backend, "verify_mask", None)
    mode = default_tree_mask_mode() if verify_mask is None else verify_mask.mode
    mask, positions, index, first_child, next_sibling, tokens = build_tree_kernel_efficient(
        rows[:, 0].contiguous(), parents.to(device=device, dtype=torch.long), top_scores.to(device=device, dtype=torch.long),
        rows[:, 1:].contiguous(), fb.seq_lens, int(fb.seq_lens_cpu.sum()), info.topk, info.spec_steps, n, mode, None,
        fill_prefix_mask=True)
    new = copy.copy(fb)
    s = new.spec_info = copy.copy(info)
    s.draft_token, s.custom_mask, s.positions = tokens, mask, positions
    s.retrieve_index, s.retrieve_next_token, s.retrieve_next_sibling = index, first_child, next_sibling
    new.input_ids = tokens.to(fb.input_ids.dtype)
    if fb.mrope_positions is not None:      # text tokens: the same shift on every axis
        new.mrope_positions = fb.mrope_positions + (positions - fb.positions).unsqueeze(0)
    new.positions = positions
    return new


def settled(tokens: list, first_child: list, next_sibling: list, choice: list) -> tuple:
    """(the row a greedy pass's accepted branch ends at, or None; the row needed next), from the model's choice at the
    first len(choice) rows: the branch follows the child whose token is the choice, and ends where none is."""
    row = 0
    while row < len(choice):
        child = first_child[row]
        while child != -1 and tokens[child] != choice[row]:
            child = next_sibling[child]
        if child == -1:
            return row, row
        row = child
    return None, row


def subtree(runner, fb, b: int):
    """The verify ForwardBatch of fb's first b tokens (one request): tokens, positions, K/V writes (SGLang's partial
    write window), the tree mask's first rows, and the links past them dropped."""
    from sglang.srt.mem_cache.kv_loc_plan import TokenSpan
    info = fb.spec_info
    n = info.draft_token_num
    if b == n:
        return fb
    assert fb.batch_size == 1 and 0 < b < n and fb.kv_loc_cols is None, (fb.batch_size, b, n)
    sub = copy.copy(fb)
    sub.input_ids, sub.positions = fb.input_ids[:b], fb.positions[:b]
    if fb.mrope_positions is not None:
        sub.mrope_positions = fb.mrope_positions[:, :b]
    fb.kv_loc_plan.bind(sub, runner.kv_index_translator, cols=TokenSpan(0, b))
    seq = int(fb.seq_lens_cpu[0])
    cut = lambda links: torch.where(links[:, :b] >= b, -1, links[:, :b])
    s = sub.spec_info = copy.copy(info)
    s.draft_token, s.positions = info.draft_token[:b], info.positions[:b]
    s.custom_mask = info.custom_mask[: n * (seq + n)].view(n, seq + n)[:b, : seq + b].reshape(-1)
    s.retrieve_index = info.retrieve_index[:, :b]
    s.retrieve_next_token, s.retrieve_next_sibling = cut(info.retrieve_next_token), cut(info.retrieve_next_sibling)
    s.draft_token_num = s.num_tokens_per_req = s.num_tokens_for_logprob_per_req = b
    if info.prepared_out_cache_loc is not None:
        s.prepared_out_cache_loc = info.prepared_out_cache_loc[:b]
    return sub


def _rows(out):
    """The per-token tensors of a target forward's output: logits (and hidden states, if kept) on the last stage, the
    proxy tensors elsewhere."""
    if hasattr(out, "next_token_logits"):
        return {k: v for k, v in (("logits", out.next_token_logits), ("hidden", getattr(out, "hidden_states", None)))
                if v is not None}
    return dict(out.tensors)


class _Check:
    size = 0
    found = []          # per verify pass: per piece, {name: max |piece - whole| / max |whole|} over its new rows


def p2p_stream_check(self, size: int = 0, path: str = "") -> None:
    """Scheduler RPC: from now on run every verify pass of one request in pieces of `size` tokens before running it
    whole (0: stop); path: write what was found."""
    if path:
        json.dump(_Check.found, open(path, "w"))
        return
    runner = self.tp_worker.model_runner
    model = runner.model
    if not hasattr(model, "_p2p_whole_forward"):
        model._p2p_whole_forward = original = model.forward

        @functools.wraps(original)
        def forward(*args, **kwargs):
            named = inspect.signature(original).bind(*args, **kwargs).arguments
            fb = named["forward_batch"]
            if not (_Check.size and fb.forward_mode.is_target_verify() and fb.batch_size == 1):
                return original(*args, **kwargs)
            got, start = {}, 0
            for end in pieces(fb.spec_info.draft_token_num, _Check.size):
                sub = subtree(runner, fb, end)
                runner.attn_backend.init_forward_metadata(sub)
                with family.restored_state(*family.layers(model)):
                    rows = _rows(original(**{**named, "input_ids": sub.input_ids, "positions": sub.positions,
                                             "forward_batch": sub}))
                got[start, end] = {k: v[start:end].float().clone() for k, v in rows.items()}
                start = end
            runner.attn_backend.init_forward_metadata(fb)
            out = original(*args, **kwargs)
            whole = _rows(out)
            _Check.found.append([{k: ((v - whole[k][a:b].float()).abs().max() /
                                      whole[k][a:b].float().abs().max().clamp_min(1e-30)).item() for k, v in g.items()}
                                 | {"rows": [a, b]} for (a, b), g in got.items()])
            return out
        model.forward = forward
    _Check.size, _Check.found = size, []
