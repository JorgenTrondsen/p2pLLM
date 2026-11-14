"""A plan's steps across pipeline ranks, every member of a spread step on a rank of its own, all running at once.

A spread step costs one hop: each rank sends its part to every rank of the next step, which sums them (replica.py's
sum). The last rank is the head: it embeds each pass's tokens for the first step, takes requests in, and sends them and
each pass's outputs to every rank, so a pass starts and ends there. A last member's rank holds the layer it copies only
as its replica (p2p.prepare), restarted every token from the updates the copied layer's rank sends after its part.
A decode or verify pass's input is received ahead: laid out as the last one from that rank, so its receive is only
queued, and the GPU runs the rank's part and sends it on without the CPU.

    P2P_PP_SPREAD   the groups to spread, by first layer ("0", "0,10"), or "all" but the last step (P2P_PP_COMM=fused)
    P2P_PP_PLAN     the plan; P2P_PP_SPLIT: layers where a run of whole steps is cut; P2P_PP_REPLICAS: its replicas
    P2P_PP_STREAM   a verify pass of one request goes in pieces of these sizes ("1,2,4", the last repeated; p2p.stream)
    P2P_PP_EARLY    the head stops waiting for a greedy pass's pieces once it is settled (0: not; check: compares both)
    P2P_PP_PHASES   "6:24": a tree's first 24 rows leave the head after 6 draft steps, the rest once drafted
    P2P_PP_NEAR     the steps after the first whose ranks get a pass's output right behind its first rows (4)
"""
from __future__ import annotations

import collections
import contextlib
import copy as _copy
import functools
import inspect
import json
import os
import sys
from typing import NamedTuple

import torch
import torch.distributed as dist

from p2p import family, pp, stream


class Rank(NamedTuple):
    start: int               # the layers it holds: [start, end)
    end: int
    first: int               # the steps it runs: first..last
    last: int
    member: "int | None"     # its member of a spread step, or None if it runs its steps whole
    prefill: int             # the layer its prefill starts from: a last member doesn't run the layer it copies


def steps(groups: list, num_layers: int) -> list:
    """The plan's steps in order: its groups, and every other layer alone."""
    starts = {g[0]: g for g in groups}
    out, i = [], 0
    while i < num_layers:
        out.append(list(starts.get(i, [i])))
        i += len(out[-1])
    return out


def layout(groups: list, spread: str, num_layers: int, split: str = "") -> list:
    """The ranks, in pipeline order. spread: "all", or the first layers of the groups to spread, comma separated. split:
    the layers where a run of whole steps is cut onto another rank, comma separated."""
    s = steps(groups, num_layers)
    firsts = [step[0] for step in s]
    wanted = set(range(len(s) - 1)) if spread == "all" else {firsts.index(int(x)) for x in spread.split(",")}
    cuts = {int(x) for x in split.split(",") if x}
    if len(s) - 1 in wanted:
        raise ValueError("the last step's rank holds the head, so it can't be spread")
    if not cuts <= set(firsts):
        raise ValueError(f"a run can only be cut where a step starts: {sorted(cuts - set(firsts))}")
    ranks, run = [], []
    def flush():
        if run:
            ranks.append(Rank(s[run[0]][0], s[run[-1]][-1] + 1, run[0], run[-1], None, s[run[0]][0]))
            run.clear()
    for j, step in enumerate(s):
        if step[0] in cuts:
            flush()
        if j not in wanted:
            run.append(j)
            continue
        flush()
        if len(step) == 1:                  # a single layer spread is a rank of its own
            run.append(j)
            flush()
            continue
        for k, layer in enumerate(step):
            ranks.append(Rank(step[-2] if k == len(step) - 1 else layer, layer + 1, j, j, k, layer))
    flush()
    return ranks


def sources(ranks: list, i: int, stepwise: bool) -> list:
    """The ranks that rank i receives its input from, in member order; stepwise on decode and verify passes. The first
    step's input is the embedding, from the head."""
    r = ranks[i]
    if stepwise:
        return _of_step(ranks, r.first - 1) if r.first else [len(ranks) - 1]
    return [k for k, x in enumerate(ranks) if x.end == r.prefill] if r.prefill else [len(ranks) - 1]


def targets(ranks: list, i: int, stepwise: bool) -> list:
    """The ranks that rank i sends its output to. On decode and verify passes the next step's, the replica's rank
    first: it runs two layers, so the step waits for one copy on each link, not for all of them."""
    r = ranks[i]
    slowest_first = lambda ks: sorted(ks, key=lambda k: not _copies(ranks[k]))
    if i == len(ranks) - 1:                 # the head embeds the next pass's tokens for the first step
        return slowest_first(_of_step(ranks, 0)) if stepwise else [k for k, x in enumerate(ranks) if x.prefill == 0]
    if stepwise:
        return slowest_first(_of_step(ranks, r.last + 1))
    return [k for k, x in enumerate(ranks) if x.prefill == r.end]


def _copies(r: Rank) -> bool:
    """Whether a rank holds a replica beside its own layer: the last member of a spread step."""
    return r.member is not None and r.end - r.start > 1


def partners(ranks: list, i: int) -> tuple:
    """(the rank that rank i sends its layer's updates to, the rank it receives its replica's updates from), each None
    where there is none: the copied layer's rank and the last member's rank of a spread group send and receive."""
    r = ranks[i]
    if r.member is None:
        return None, None
    peers = [k for k, x in enumerate(ranks) if x.member is not None and x.first == r.first]
    return (peers[-1] if i == peers[-2] else None), (peers[-2] if i == peers[-1] else None)


def _of_step(ranks: list, j: int) -> list:
    return [k for k, x in enumerate(ranks) if x.first <= j <= x.last]


# In each scheduler process.

class _State:
    groups = spread = None
    split = ""
    ranks = None              # layout() for this model
    target = None             # this process's target model, once loaded (a drafter loads after it)
    pretend = {}              # while a model is built: what the pipeline group's is_first_rank, is_last_rank answer
    planned = False           # a plan is installed, so decode and verify passes go step to step
    outputs = collections.deque()   # the head's own outputs, for its own bookkeeping
    scheduler = None
    feeds = fed = None        # partners(): the rank this one sends updates to, the one it receives them from
    copied = layer = None     # the copied layer (id and module) on both of those ranks
    taken = {}                # the copied layer's rank: what this pass's update holds so far ("kv": set_kv_buffer calls)
    pending = 0               # the replica's rank: updates sent but not applied yet, one per pass
    saved = collections.deque()     # the replica's rank: its own linear-attention calls, to replay with the exact inputs
    prepared = None           # the replica's rank: the metadata of the replica it loaded ahead of time (P2P_PP_REPLICAS)
    replaying = False
    inbox = collections.defaultdict(list)   # per source rank: messages received ahead of one of another kind
    received = None           # rank 0: the head's embedding of this pass's tokens
    layouts = {}              # (source rank, kind of pass): the layout of its last input from there (p2p.pp.receive)
    arrives = None            # the event of this pass's input, which the target model's forward waits for (_after_input)
    stream = ()               # P2P_PP_STREAM: tokens in each piece of a streamed pass, the last repeated (none: whole)
    early = True              # P2P_PP_EARLY: the head stops waiting for a greedy pass's pieces once it is settled
    check = None              # P2P_PP_EARLY=check: per such pass, how its early stop compares with the whole pass
    trailing = []             # the head: passes whose last pieces it didn't wait for, kept until they have come
    phases = ()               # P2P_PP_PHASES: (draft steps, rows) per phase but the last: a tree's first rows are fixed
                              # after that many steps and sent on while the rest is drafted (the head: _cut)
    result = None             # the head: this pass's batch and verify result, which a phased tree's output is made from
    drafting = None           # the head: the tree it is drafting in phases (_candidates)
    sent_rows = 0             # the head: the next pass's rows that went out while it was drafted
    tree = None               # the head: that pass's whole tree (SGLang's parent list, the candidates chosen)
    early_out = False         # the head: that tree's first phase took this pass's output along, so SGLang's send is skipped
    trees = 0                 # the head: phased trees so far; each one's messages carry its number
    phased = 0                # another rank: the number of the phased tree its next pass verifies (0: a tree sent whole)
    upto = {}                 # per source rank: its last output announced a phased tree, whose first rows come next
    rows = {}                 # per source rank: the rows of the phased pass coming from it, for its later phases
    voided = None             # the rows and trees of a phased pass that never came, kept while their receives are queued
    shapes = {}               # per source rank: {key: (dtype, shape)} of its first verify pass of one request
    primed = False            # this rank has sent a verify pass of one request whole: the next ones go in pieces
    incoming = None           # this pass's input coming in pieces (_recv), for the forward to take piece by piece
    streamed = ()             # the ranks this pass's output went to in pieces, which SGLang's own send then skips
    side = None               # the stream pieces are set up on (_set_up)
    planners = {}             # (attention backend, piece): FlashInfer wrappers of the piece's own
    alive = []                # this pass's pieces' set-ups, kept until the next pass
    pieces = {"sent": 0, "received": 0, "passes": 0, "runs": 0, "rows": 0, "of": 0}     # the last four: the head's


S = _State()


def install_launcher() -> None:
    """Hooks for the process that launches the schedulers: across machines, the tokenizer's sockets reach the head."""
    if os.environ.get("P2P_PP_SPREAD"):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
        HookRegistry.register("sglang.srt.server_args.PortArgs.init_new", _ports, HookType.AROUND)


def install() -> None:
    if not os.environ.get("P2P_PP_SPREAD"):
        return
    assert os.environ.get("P2P_PP_COMM") == "fused", "P2P_PP_SPREAD needs P2P_PP_COMM=fused"
    S.spread, S.groups = os.environ["P2P_PP_SPREAD"], json.load(open(os.environ["P2P_PP_PLAN"]))["groups"]
    S.split = os.environ.get("P2P_PP_SPLIT", "")
    S.stream = tuple(int(x) for x in os.environ.get("P2P_PP_STREAM", "").split(",") if x.strip() and int(x))
    S.early = os.environ.get("P2P_PP_EARLY", "1") != "0"
    S.check = [] if os.environ.get("P2P_PP_EARLY") == "check" else None
    S.phases = tuple(tuple(int(v) for v in x.split(":")) for x in os.environ.get("P2P_PP_PHASES", "").split(",") if x.strip())
    os.environ.setdefault("SGLANG_PP_COMM_OVERLAP", "1")    # inputs on a stream of their own, so the GPU waits for one alone
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    def indices(original, num_layers, pp_rank, pp_size):
        if pp_size == 1:                    # a drafter's own pipeline group holds all its layers
            return original(num_layers, pp_rank, pp_size)
        S.ranks = layout(S.groups, S.spread, num_layers, S.split)      # SGLang asks for other ranks' layers too
        assert len(S.ranks) == pp_size, \
            f"the plan spread as {S.spread}, split at {S.split or 'nothing'}, needs {len(S.ranks)} pipeline ranks, not {pp_size}"
        head = pp_size - 1                  # it sends each pass's outputs and requests to every rank: at once to those
        pp.URGENT = set(targets(S.ranks, head, True)) | set(targets(S.ranks, head, False))   # it sends inputs to,
        pp.ORDER = sorted(range(head), key=lambda r: (r not in pp.URGENT, S.ranks[r].first, not _copies(S.ranks[r]), r))
        # The ranks of the next steps need a pass's output (and its requests) before the rest of its rows has left the
        # head: theirs go right behind the first rows (P2P_PP_NEAR steps; the later ranks' once all rows are out).
        pp.NEAR = frozenset(r for r in range(head) if S.ranks[r].first <= int(os.environ.get("P2P_PP_NEAR", "4")))
        return S.ranks[pp_rank].start, S.ranks[pp_rank].end
    patch_indices(indices)
    patch_stage()
    pp.ENTRY = -1                           # the head: a pass's tokens start and end there
    pp.HANDOFF = _handoff
    from sglang.srt.environ import envs
    assert envs.SGLANG_REQ_WAITING_TIMEOUT.get() <= 0 and envs.SGLANG_REQ_RUNNING_TIMEOUT.get() <= 0, \
        "SGLang polls request timeouts on stage 0, which takes no requests in here"
    Scheduler.p2p_held = p2p_held
    on = lambda path, fn: HookRegistry.register("sglang.srt." + path, fn, HookType.AROUND)
    on("managers.scheduler_pp_mixin.SchedulerPPMixin._pp_send_dict_to_next_stage", _send)
    on("managers.scheduler_pp_mixin.SchedulerPPMixin._pp_recv_typed_dict", _recv)
    on("managers.scheduler_pp_mixin.SchedulerPPMixin._pp_recv_proxy_tensors", _proxies)
    on("managers.scheduler_pp_mixin.SchedulerPPMixin.init_pp_loop_state", _loop)
    on("managers.scheduler_pp_mixin.SchedulerPPMixin._pp_wait_forward_dependencies", _defer)
    on("model_executor.model_runner.ModelRunner.forward", _forward)
    on("layers.radix_linear_attention.RadixLinearAttention.forward", _linear)
    on("speculative.spec_utils.commit_mamba_states_after_verify", _commit)
    on("model_loader.loader.DefaultModelLoader.load_model", _construct)
    on("model_loader.loader.DefaultModelLoader.load_weights_and_postprocess", _load)
    if S.phases:                            # a tree's first rows go on while the rest is drafted
        on("speculative.eagle_worker_v2.EAGLEWorkerV2.verify", _verified)
        on("speculative.spec_utils.select_top_k_tokens", _candidates)
        on("speculative.eagle_utils.organize_draft_results", _organized)
    if os.environ.get("P2P_PP_SPANS"):      # where the head's own time goes, on its GPU's timeline (bench/pp/overhead.py)
        for name, path in (("verify", "EAGLEWorkerV2.verify"), ("extend", "EagleDraftWorker._draft_extend_for_decode"),
                           ("draft", "EagleDraftWorker.draft")):
            on("speculative.eagle_worker_v2." + path, functools.partial(_timed, name))


def _timed(name: str, original, *args, **kwargs):
    """Marks where the GPU starts and ends a part of SGLang's drafting worker (p2p.pp.mark)."""
    pp.mark(f"gpu:{name}+")
    try:
        return original(*args, **kwargs)
    finally:
        pp.mark(f"gpu:{name}-")


def patch_indices(fn) -> None:
    """SGLang's get_pp_indices becomes fn(original, num_layers, pp_rank, pp_size): the layers each stage holds."""
    import sglang.srt.distributed.utils as utils
    original = utils.get_pp_indices
    patched = functools.partial(fn, original)
    for module in list(sys.modules.values()):    # modules import it by name; later imports find it patched here
        if getattr(module, "get_pp_indices", None) is original:
            module.get_pp_indices = patched


def patch_stage() -> None:
    """The pipeline group's is_first_rank and is_last_rank answer S.pretend while a model is built (_construct,
    p2p.prepare), so SGLang builds and loads what such a stage holds: its embedding, its head."""
    from sglang.srt.distributed.parallel_state import GroupCoordinator, get_pp_group
    if getattr(GroupCoordinator, "_p2p_stage", False):
        return
    for name in ("is_first_rank", "is_last_rank"):
        def answer(group, _name=name, _original=getattr(GroupCoordinator, name)):
            if _name in S.pretend and group is get_pp_group():
                return S.pretend[_name]
            return _original.fget(group)
        setattr(GroupCoordinator, name, property(answer))
    GroupCoordinator._p2p_stage = True


def _loop(original, scheduler):
    """At the start of SGLang's pipeline loop: one stream needs two microbatch slots, not one per stage
    (P2P_PP_MICROBATCHES), and the plan in P2P_PP_PLAN is installed (P2P_PP_CALIB: the sums for pruned groups a rank
    runs whole)."""
    from sglang.srt.runtime_context import get_parallel
    original(scheduler)
    n = int(os.environ.get("P2P_PP_MICROBATCHES", "2")) + get_parallel().pp_async_batch_depth
    if n < scheduler.pp_loop_size:
        scheduler.pp_loop_size = n
        for name in ("mbs", "last_mbs", "running_mbs", "mb_metadata"):
            setattr(scheduler, name, getattr(scheduler, name)[:n])
    if S.scheduler is None:
        from p2p import plugin
        plan = json.load(open(os.environ["P2P_PP_PLAN"]))
        sums = torch.load(os.environ["P2P_PP_CALIB"]) if os.environ.get("P2P_PP_CALIB") else {}
        install_plan(scheduler, plugin._layers(scheduler), plan if plan.get("keep") else None, sums)


def _ports(original, server_args, *args, **kwargs):
    """Across machines the head, which holds the tokenizer's sockets (p2p.pp ENTRY), is reached over the TCP addresses
    SGLang derives from --dist-init-addr, not over its local socket files."""
    out = original(server_args, *args, **kwargs)
    if server_args.nnodes > 1 and out.scheduler_input_ipc_name.startswith("ipc://"):
        from sglang.srt.utils.network import NetworkAddress
        at = NetworkAddress.parse(server_args.dist_init_addr)
        tcp = lambda k: NetworkAddress(at.host, at.port + 1 + k).to_tcp()      # SGLang's port_base and the four after it
        out.tokenizer_ipc_name, out.detokenizer_ipc_name, out.rpc_ipc_name, out.metrics_ipc_name = (tcp(k) for k in range(4))
        out.scheduler_input_ipc_name = tcp(4)
    return out


def _stepwise(scheduler) -> bool:
    mode = scheduler.cur_batch_for_debug.forward_mode    # some drafting workers mark the batch itself as verifying
    return S.planned and (mode.is_decode() or mode.is_target_verify())


def _send(original, scheduler, tensor_dict, async_send=True, msg_type="default", ready_event=None):
    me, group = scheduler.pp_group.rank_in_group, scheduler.pp_group
    head = group.world_size - 1
    if msg_type == "output":
        if me != head:
            return []
        dsts = range(head)                  # at once to the first step's ranks, to the rest once the next input is out
    else:
        assert msg_type == "proxy", msg_type
        dsts = targets(S.ranks, me, _stepwise(scheduler))
    tensor_dict["__msg_type__"] = msg_type
    works = []
    if msg_type == "proxy":
        if _one(scheduler):
            S.primed = True                 # the receivers know its layout now
        dsts = [d for d in dsts if d not in S.streamed]     # those went in pieces
        S.streamed = ()
    with scheduler.pp_comm_stream_ctx, pp.label(f"p2p:send_{msg_type}->{list(dsts)}"):
        if ready_event is not None:
            scheduler.device_module.current_stream().wait_event(ready_event)
        if msg_type == "proxy":
            pp.mark("gpu:out")
            tensor_dict = _one_tensor(tensor_dict, me)      # here: once the forward has made them
        if msg_type == "output":
            S.outputs.append({k: v.clone() if hasattr(v, "clone") else v for k, v in tensor_dict.items()})
            if S.early_out:                 # the other ranks have it: it went with the next tree's first phase (_cut)
                S.early_out = False
                return works
            packed = pp.pack(group, tensor_dict)                # once, for every rank
            return pp.stagger(list(dsts), lambda ds: [w for d in ds for w in pp.send_packed(group, packed, d)])
        for dst in dsts:
            works += group.send_tensor_dict(tensor_dict, dst=dst, all_gather_group=scheduler.attn_tp_group,
                                            async_send=async_send)
        if msg_type == "proxy" and S.planned:           # after the proxies: the next step's input goes first
            if S.feeds is not None:
                # Not among the works the scheduler waits on: over a direct GPU link a large update completes only once
                # the replica's rank drains it, and that rank may be waiting on this one (a deadlock on Qwen3.8).
                with pp.label("p2p:send_update"):
                    pp.keep(group.send_tensor_dict(_update(scheduler), dst=S.feeds,
                                                   all_gather_group=scheduler.attn_tp_group, async_send=True))
            if S.fed is not None:
                S.pending += 1
    return works


def _one_tensor(tensors: dict, me: int) -> dict:
    """What a rank of whole layers sends a spread step on a decode or verify pass: its hidden states and residual as one
    tensor, added in the model's dtype as each member adds them on receipt (_member), at half the bytes."""
    if not (S.planned and "residual" in tensors and _stepwise(S.scheduler) and me != len(S.ranks) - 1
            and all(S.ranks[d].member is not None for d in targets(S.ranks, me, True))):
        return tensors
    return {"hidden_states": tensors["hidden_states"] + tensors["residual"],
            **{k: v for k, v in tensors.items() if k not in ("hidden_states", "residual")}}


def _recv(original, scheduler, expected_kind="default", all_gather_group=None):
    me, group = scheduler.pp_group.rank_in_group, scheduler.pp_group
    head = group.world_size - 1
    if expected_kind == "output" and me == head:
        return S.outputs.popleft(), scheduler._pp_record_comm_event()
    if expected_kind == "proxy":
        _drain(scheduler)                   # updates sent before this pass's input come first on that link
        if S.phased and me != head and not _one(scheduler):
            _void(scheduler)                # the phased tree's pass never came: its request ended
    srcs = [head] if expected_kind == "output" else sources(S.ranks, me, _stepwise(scheduler))
    one = expected_kind == "proxy" and _one(scheduler) and srcs != [head]
    if one and all(src in S.shapes for src in srcs):        # a rank of several steps runs once it all has come
        whole = not _early(scheduler) if me == head else S.ranks[me].last > S.ranks[me].first
        return _pieces_in(scheduler, srcs, whole)
    like = _like(scheduler) if expected_kind == "proxy" else None
    with scheduler.pp_comm_stream_ctx, pp.label(f"p2p:recv_{expected_kind}<-{srcs}"):
        got = [_take(group, src, expected_kind, like) for src in srcs]
        if one and S.stream:
            for src, d in zip(srcs, got):
                S.shapes[src] = {k: (v.dtype, tuple(v.shape)) for k, v in d.items() if hasattr(v, "dtype")}
        out = got[0] if len(got) == 1 else _summed(got, scheduler.tp_worker.model_runner.dtype)
        if expected_kind == "output":       # whether the next pass's tree comes in phases (_cut), and which it is
            S.phased = out.pop("__phases__", 0)
        if "__incoming__" in out:           # the head's embedding, the rest of it still coming (_rest_in)
            S.incoming = out.pop("__incoming__")
            if S.ranks[me].last > S.ranks[me].first:        # a rank of several steps runs once it all has come
                if S.phased:
                    S.incoming["whole"] = True              # with each later phase's rows, queued behind its tree
                else:
                    _wait(S.incoming["works"])
                    S.incoming = None
        event = scheduler._pp_record_comm_event()
    if like is not None:                    # the CPU may let go of them before the forward that reads them has run
        for t in out.values():
            if hasattr(t, "record_stream") and t.is_cuda:
                t.record_stream(scheduler.forward_stream)
    return out, event


def _like(scheduler):
    """What a decode or verify pass's input is received ahead under (p2p.pp.receive): its kind and batch size, which fix
    its layout. Not with another batch in flight, whose outputs could come first on the same link."""
    batch = scheduler.cur_batch_for_debug
    if not (_stepwise(scheduler) and sum(b is not None for b in scheduler.mbs) == 1):
        return None
    return str(batch.forward_mode), batch.batch_size()


def _take(group, src: int, kind: str, like=None) -> dict:
    """The next message of this kind from src; one of another kind waits in S.inbox (the head sends the first step's
    ranks both outputs and inputs). like: a pass's input, received ahead once one like it has come."""
    box = S.inbox[src]
    for j, d in enumerate(box):
        if d["__msg_type__"] == kind:
            return box.pop(j)
    known = S.layouts.get((src, like)) if like is not None and not box else None
    while True:
        d, layout = pp.receive(group, src, known)
        if d["__msg_type__"] == "output":
            S.upto[src] = "__phases__" in d
        if "__rows__" in d:
            _rest_in(group, src, d)
        if "__more__" in d:
            _more_in(group, src, d)
        if d["__msg_type__"] == kind:
            if like is not None and layout is not None:
                S.layouts[src, like] = layout
            return d
        box.append(d)


def _rest_in(group, src: int, d: dict) -> None:
    """A streamed pass's embedding from the head: its first piece tells the pass's rows, and the others follow it raw,
    so their receives are queued as soon as it is read."""
    first, n = d["hidden_states"], d.pop("__rows__")
    bounds = _bounds(n)
    assert bounds[0] == first.shape[0], (bounds, first.shape)
    upto = S.phases[0][1] if S.upto.get(src) else n         # a phased tree: its first phase's rows only, for now
    S.upto[src] = False
    rows = S.rows[src] = torch.empty(n, *first.shape[1:], dtype=first.dtype, device=first.device)
    rows[:bounds[0]] = first
    works = [[]] + [[dist.irecv(rows[a:b], group.ranks[src], group=group.device_group)] if b <= upto else None
                    for a, b in zip(bounds, bounds[1:])]
    S.pieces["received"] += sum(w is not None for w in works)
    d["hidden_states"] = rows
    d["__incoming__"] = {"bounds": bounds, "works": works, "parts": [{"hidden_states": rows}],
                         "out": {"hidden_states": rows}, "multi": False}


def _more_in(group, src: int, d: dict) -> None:
    """A later phase of a tree, on a rank of the first step: the head sends that phase's rows raw right behind it, so
    their receives are queued as soon as it is read, as _rest_in does behind a pass's first piece."""
    a, b = d.pop("__more__")
    rows = S.rows[src]
    bounds = _bounds(rows.shape[0])
    d["__works__"] = {k: [dist.irecv(rows[x:y], group.ranks[src], group=group.device_group)]
                      for k, (x, y) in enumerate(zip([0] + bounds, bounds)) if a < y <= b}
    S.pieces["received"] += len(d["__works__"])


def _tree(group, src: int, number: int) -> dict:
    """The next phase of phased tree `number` from the head. One of a tree whose pass never came is dropped."""
    S.inbox[src] = [d for d in S.inbox[src] if d["__msg_type__"] != "tree" or d["__tree__"] == number]
    while True:
        d = _take(group, src, "tree")
        if d["__tree__"] == number:
            return d


def _void(scheduler) -> None:
    """A phased tree whose pass never came, its request having ended: a rank of the first step takes what the head
    sent of it off its link; another rank drops its trees where they turn up (_tree)."""
    group = scheduler.pp_group
    head = group.world_size - 1
    if sources(S.ranks, group.rank_in_group, True) == [head]:
        with scheduler.pp_comm_stream_ctx, pp.label("p2p:void"):
            S.voided = [_take(group, head, "proxy")] + [_tree(group, head, S.phased) for _ in S.phases[1:]]
    S.phased = 0


def _defer(original, scheduler):
    """The target model's forward waits for its input (_after_input), not SGLang's set-up before it: FlashInfer's
    planning there would hold the CPU until the input came."""
    S.arrives, scheduler.pp_proxy_recv_event = scheduler.pp_proxy_recv_event, None
    return original(scheduler)


def _after_input(model) -> None:
    """The target model's forward runs once its input has come (SGLang reads the signature, which it keeps)."""
    original = model.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        if S.arrives is not None:
            torch.cuda.current_stream().wait_event(S.arrives)
            S.arrives = None
        pp.mark("gpu:in")
        incoming, S.incoming = S.incoming, None
        phased, S.phased = S.phased, 0
        named = None
        if incoming is not None or phased or (S.stream and S.primed and S.planned):
            named = inspect.signature(original).bind(*args, **kwargs).arguments
            fb = named["forward_batch"]
            one = _passes(fb) and fb.batch_size == 1
            phased = phased if one else 0   # else its pass never came (a rank that _void doesn't serve)
            if not (incoming is not None or phased or (one and not S.scheduler.pp_group.is_last_rank)):
                named = None
        out = original(*args, **kwargs) if named is None else _in_pieces(model, original, named, incoming, phased)
        pp.mark("gpu:done")                 # its layers (and on the head, the logits) have run
        return out
    model.forward = forward


def _bounds(n: int) -> list:
    """Where each piece of a streamed pass of n tokens ends, on every rank alike (p2p.stream.pieces)."""
    return stream.pieces(n, S.stream, [rows for _, rows in S.phases])


def _one(scheduler) -> bool:
    """A decode or verify pass of one request: what streams (P2P_PP_STREAM), the same pass on every rank."""
    return bool(S.stream) and _stepwise(scheduler) and scheduler.cur_batch_for_debug.batch_size() == 1


def _early(scheduler) -> bool:
    """Whether the head can stop waiting for this pass's pieces once it is settled (p2p.stream.settled): a verify pass
    that SGLang accepts greedily, by the model's first choice at each node."""
    batch = scheduler.cur_batch_for_debug
    info = getattr(batch, "sampling_info", None)
    return S.early and batch.forward_mode.is_target_verify() and bool(getattr(info, "is_all_greedy", False))


def _pieces_in(scheduler, srcs: list, whole: bool):
    """This pass's input from srcs in pieces, each piece's receives queued into its rows. whole: wait for them all and
    run once (a rank of several steps; the head on a pass it can't stop early)."""
    group, device = scheduler.pp_group, scheduler.pp_group.device
    n = next(iter(S.shapes[srcs[0]].values()))[1][0]
    bounds = _bounds(n)
    parts = [{k: torch.empty(shape, dtype=dtype, device=device) for k, (dtype, shape) in S.shapes[src].items()}
             for src in srcs]
    works = [[] for _ in bounds]
    with scheduler.pp_comm_stream_ctx, pp.label(f"p2p:recv_pieces<-{srcs}"):
        for src, bufs in zip(srcs, parts):
            assert not S.inbox[src], f"messages from rank {src} waiting while its input comes in pieces"
            start = 0
            for k, end in enumerate(bounds):
                works[k] += [dist.irecv(t[start:end], group.ranks[src], group=group.device_group) for t in bufs.values()]
                start = end
            S.pieces["received"] += len(bounds)
        if whole:
            _wait(works)
            out = parts[0] if len(parts) == 1 else _summed(parts, scheduler.tp_worker.model_runner.dtype)
        else:
            dtype = scheduler.tp_worker.model_runner.dtype
            out = parts[0] if len(parts) == 1 else {"hidden_states": torch.empty(n, *parts[0]["hidden_states"].shape[1:],
                                                                                  dtype=dtype, device=device)}
            S.incoming = {"bounds": bounds, "works": works, "parts": parts, "out": out, "multi": len(parts) > 1}
        event = scheduler._pp_record_comm_event()
    for t in (*(t for bufs in parts for t in bufs.values()), *out.values()):
        if hasattr(t, "record_stream"):
            t.record_stream(scheduler.forward_stream)
    return {**out, "__msg_type__": "proxy"}, event


def _in_pieces(model, original, named: dict, incoming, phased: int = 0) -> object:
    """The target model's forward on a streamed pass: each piece is SGLang's verify of the pass's first tokens, run
    once its input has come, its new rows sent on at once. Every piece but the last runs with linear attention's state
    put back and on a set-up of its own (_set_up); only the last one's calls feed the replica's updates. phased: the
    number of the pass's phased tree, whose next phase is taken before the first piece that reaches into it
    (_next_tree)."""
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    fb = named["forward_batch"]
    if incoming is None or incoming.get("whole"):
        if phased:
            named = _whole_tree(named, incoming, phased)
        if not (S.stream and S.primed and S.planned):
            return original(**named)
        return _whole_in_pieces_out(original, named, named["forward_batch"])
    scheduler = S.scheduler
    runner, group = scheduler.tp_worker.model_runner, scheduler.pp_group
    if group.is_last_rank:
        return _head_in_pieces(model, original, named, incoming)
    n, bounds = _tokens(fb), incoming["bounds"]
    assert bounds[-1] == n, (bounds, n)
    me = group.rank_in_group
    dsts = targets(S.ranks, me, True) if S.primed else []
    backends, whole_metadata = _side(runner)
    layers, kept, outs, start = family.layers(model), len(S.saved), [], 0
    cuts, phase = [rows for _, rows in S.phases] if phased else [], 0
    for k, end in enumerate(bounds):
        while phase < len(cuts) and end > cuts[phase]:
            fb, phase = _next_tree(fb, incoming, phased), phase + 1
        last = end == n
        sub, metadata = (fb, whole_metadata) if last and not phase else _set_up(runner, backends, fb, end, k)
        for b, m in zip(backends, metadata):
            b.forward_metadata = m
        for w in incoming["works"][k]:
            w.wait()                        # this stream waits for this piece alone
        pp.mark("gpu:piece_in")
        _sum_piece(incoming, start, end, runner.dtype)
        # a copy: a layer may update its input's residual in place, which the next piece reads again and this one sends
        given = {key: t[:end].clone() for key, t in incoming["out"].items()}
        if me == 0:                         # rank 0 has no input but its stub embedding's (_received)
            S.received, given = given["hidden_states"], {}
        else:
            given = {"pp_proxy_tensors": PPProxyTensors(given)}
        if last:                            # the replica's updates: the whole tree's calls only
            S.taken = {}
            while len(S.saved) > kept:
                S.saved.pop()
        with pp.label("p2p:piece_forward"), contextlib.nullcontext() if last else family.restored_state(*layers):
            out = original(**{**named, "input_ids": sub.input_ids, "positions": sub.positions, "forward_batch": sub,
                              **given})
        pp.mark("gpu:piece")
        new = {key: t[start:end] for key, t in _one_tensor(out.tensors, me).items()}
        if dsts:
            _send_piece(group, new, dsts)
            S.pieces["sent"] += len(dsts)
        outs.append(new)
        start = end
    S.streamed = tuple(dsts)
    return PPProxyTensors({key: torch.cat([o[key] for o in outs]) for key in outs[0]})


def _next_tree(fb, incoming, number: int):
    """The pass with the next phase of its tree from the head, laid out by SGLang's kernel (p2p.stream.regrown). The
    CPU waits here for the head to have drafted that far."""
    scheduler = S.scheduler
    group = scheduler.pp_group
    with scheduler.pp_comm_stream_ctx, pp.label("p2p:recv_tree"):
        d = _tree(group, group.world_size - 1, number)
        event = scheduler._pp_record_comm_event()
    torch.cuda.current_stream().wait_event(event)
    if incoming is not None:
        for k, works in d.get("__works__", {}).items():
            incoming["works"][k] = works
    return stream.regrown(scheduler.tp_worker.model_runner, fb, d["spec_next_chain"], d["spec_next_parents"],
                          d["spec_next_top_scores"])


def _whole_tree(named: dict, incoming, number: int) -> dict:
    """A pass whose tree comes in phases, on a rank that runs it whole: every later phase of the tree first (and, from
    the head, its rows), then SGLang's set-up for the whole tree."""
    fb = named["forward_batch"]
    for _ in S.phases:
        fb = _next_tree(fb, incoming, number)
    if incoming is not None:
        _wait(incoming["works"])
    S.scheduler.tp_worker.model_runner.attn_backend.init_forward_metadata(fb)
    return {**named, "input_ids": fb.input_ids, "positions": fb.positions, "forward_batch": fb}


def _head_in_pieces(model, original, named: dict, incoming) -> object:
    """The head's forward on a greedy verify pass in pieces: it runs SGLang's verify on the rows that have come and
    follows the model's choices (p2p.stream.settled). Where the accepted branch ends, that run's output is the pass's,
    its other rows zero and the links to them cut. A run that doesn't settle the pass has linear attention's state put
    back; the pieces not waited for stay queued (S.trailing)."""
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    fb = named["forward_batch"]
    runner = S.scheduler.tp_worker.model_runner
    info, n, bounds, works = fb.spec_info, _tokens(fb), incoming["bounds"], incoming["works"]
    assert bounds[-1] == n, (bounds, n)
    tree = [t.reshape(-1).tolist() for t in (info.draft_token, info.retrieve_next_token, info.retrieve_next_sibling)]
    backends, whole_metadata = _side(runner)
    S.trailing = [x for x in S.trailing if not all(w.is_completed() for ws in x["works"] for w in ws)]
    layers, waited, want = family.layers(model), 0, 0
    whole = _whole_first(original, named, incoming, layers) if S.check is not None else None
    while True:
        k = next(i for i, b in enumerate(bounds) if b > want)
        while k + 1 < len(bounds) and whole is None and all(w.is_completed() for w in works[k + 1]):
            k += 1                          # and every later piece that has come already
        end = bounds[k]
        sub, metadata = (fb, whole_metadata) if end == n else _set_up(runner, backends, fb, end, k)
        for b, m in zip(backends, metadata):
            b.forward_metadata = m
        for i in range(waited, k + 1):
            for w in works[i]:
                w.wait()
            pp.mark("gpu:piece_in")
            _sum_piece(incoming, bounds[i - 1] if i else 0, bounds[i], runner.dtype)
        waited = k + 1
        given = PPProxyTensors({key: t[:end].clone() for key, t in incoming["out"].items()})
        with contextlib.ExitStack() as undo, pp.label("p2p:piece_forward"):
            undo.enter_context(family.restored_state(*layers))
            out = original(**{**named, "input_ids": sub.input_ids, "positions": sub.positions, "forward_batch": sub,
                              "pp_proxy_tensors": given})
            pp.mark("gpu:piece")
            S.pieces["runs"] += 1
            at, want = stream.settled(*tree, out.next_token_logits[:end].argmax(-1).tolist())   # the CPU waits here
            if at is not None:
                undo.pop_all()              # the settling run's state stays
                break
    for b, m in zip(backends, whole_metadata):
        b.forward_metadata = m
    S.pieces["passes"] += 1
    S.pieces["rows"] += end
    S.pieces["of"] += n
    if whole is not None:               # P2P_PP_EARLY=check: this pass against itself run whole
        ref, got = whole[:end].float(), out.next_token_logits[:end].float()
        top = ref.topk(2, dim=-1).values
        S.check.append({"rows": end, "of": n, "ends": at, "whole_ends": stream.settled(*tree, whole.argmax(-1).tolist())[0],
                        "logits": ((got - ref).abs().max() / ref.abs().max()).item(),
                        "closest": (top[:, 0] - top[:, 1]).min().item()})
    if end < n:
        for links in (info.retrieve_next_token, info.retrieve_next_sibling):
            links[links >= end] = -1
        out = _copy.copy(out)
        for name, v in vars(out).items():
            if torch.is_tensor(v) and v.shape[0] == end:
                setattr(out, name, torch.cat([v, v.new_zeros(n - end, *v.shape[1:])]))
        S.trailing.append(incoming)
    return out


def _whole_first(original, named: dict, incoming, layers):
    """P2P_PP_EARLY=check: the whole pass's logits, run before the head's early stop on the same pass (once every piece
    has come, its state put back), to compare the two (p2p_held writes S.check)."""
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    _wait(incoming["works"])
    rows = incoming["out"]
    if incoming["multi"]:
        rows = {k: v for k, v in _summed(incoming["parts"], S.scheduler.tp_worker.model_runner.dtype).items()
                if torch.is_tensor(v)}
    with family.restored_state(*layers):
        out = original(**{**named, "pp_proxy_tensors": PPProxyTensors({k: t.clone() for k, t in rows.items()})})
    return out.next_token_logits.clone()


def _wait(works: list) -> None:
    """Every receive of an input coming in pieces: [[a work per tensor] per piece]."""
    for w in (w for ws in works for w in ws):
        w.wait()


def _sum_piece(incoming: dict, a: int, b: int, dtype) -> None:
    """Rows a..b of a step input coming in pieces from several ranks: their parts summed as _summed sums whole ones."""
    if incoming["multi"]:
        rows = [{"hidden_states": p["hidden_states"][a:b]} for p in incoming["parts"]]
        incoming["out"]["hidden_states"][a:b] = _summed(rows, dtype)["hidden_states"]


def _side(runner) -> tuple:
    """(the attention backends, SGLang's set-up of each for the whole pass). The side stream, on which the other
    pieces are set up, waits for what that set-up reads; last pass's set-ups are let go."""
    backends = list(getattr(runner.attn_backend, "attn_backend_list", None) or [runner.attn_backend])
    S.side = S.side or torch.cuda.Stream()
    S.side.wait_event(torch.cuda.current_stream().record_event())
    S.alive = []
    return backends, [b.forward_metadata for b in backends]


def _set_up(runner, backends: list, fb, end: int, k: int) -> tuple:
    """(piece k's forward batch, each attention backend's set-up for it), made on the side stream. FlashInfer plans
    into wrappers of the piece's own, so no plan is overwritten before it has run."""
    main = torch.cuda.current_stream()
    with pp.label("p2p:piece_setup"), torch.cuda.stream(S.side):
        sub = stream.subtree(runner, fb, end)
        own = [b for b in backends if hasattr(b, "prefill_wrappers_verify")]
        kept = [b.prefill_wrappers_verify for b in own]
        for b in own:
            if (id(b), k) not in S.planners:
                S.planners[id(b), k] = [type(w)(b.workspace_buffer, "NHD", backend=b.prefill_backend)
                                        for w in b.prefill_wrappers_verify]
            b.prefill_wrappers_verify = S.planners[id(b), k]
        try:
            runner.attn_backend.init_forward_metadata(sub)
        finally:
            for b, w in zip(own, kept):
                b.prefill_wrappers_verify = w
        metadata = [b.forward_metadata for b in backends]
        ready = S.side.record_event()
    main.wait_event(ready)
    S.alive.append((sub, metadata))
    return sub, metadata


def _tokens(fb) -> int:
    """The tokens of a pass: its verify tree's, or a decode's."""
    return fb.spec_info.draft_token_num if getattr(fb.spec_info, "draft_token_num", None) else fb.input_ids.shape[0]


def _whole_in_pieces_out(original, named: dict, fb) -> object:
    """A pass this rank runs whole (its input came whole, or it runs several steps): run once, then send the output in
    pieces."""
    group = S.scheduler.pp_group
    out = original(**named)
    dsts, start = targets(S.ranks, group.rank_in_group, True), 0
    tensors = _one_tensor(out.tensors, group.rank_in_group)
    for end in _bounds(_tokens(fb)):
        _send_piece(group, {key: t[start:end] for key, t in tensors.items()}, dsts)
        S.pieces["sent"] += len(dsts)
        start = end
    S.streamed = tuple(dsts)
    return out


def _send_piece(group, rows: dict, dsts: list) -> None:
    """A piece's new rows to each rank in dsts, a message per tensor, once the current stream has made them."""
    from sglang.srt.distributed.parallel_state import P2PWork
    works = []
    with pp.link():
        for dst in dsts:
            for t in rows.values():
                t = t.contiguous()
                pp.occupy(t.numel() * t.element_size())
                works.append(P2PWork(dist.isend(t, group.ranks[dst], group=group.device_group), t))
    pp.keep(works)


def _handoff(boundary, forward_batch):
    """The boundary a stage's input is restored at (p2p.pp HANDOFF): where layers run in sequence, the replica's rank
    starts at its own layer, not at the layer it copies."""
    if S.fed is not None and not (S.planned and _passes(forward_batch)):
        from sglang.srt.runtime_context import get_parallel
        layers, r = family.layers(S.target), S.ranks[get_parallel().pp_rank]
        first = layers[r.start]
        name = next(k for k, v in vars(first).items() if v is boundary)
        boundary = getattr(layers[r.prefill], name)
    return boundary


def _summed(parts: list, dtype) -> dict:
    """One step input from its members' parts, added in float32 in member order as replica.py adds them."""
    total = parts[0]["hidden_states"].to(torch.float32, copy=True)
    for p in parts[1:]:
        total += p["hidden_states"]
    return {"hidden_states": total.to(dtype), "__msg_type__": "proxy"}


# The embedding, on the head: the ring's input.

class _Later:
    """The head's input, received in its forward once it has sent the embedding of that pass's tokens."""

    def __init__(self, scheduler):
        self.scheduler = scheduler


def _proxies(original, scheduler):
    """Rank 0 receives its input from the head like the other ranks, and its stub embedding hands it to the model."""
    group = scheduler.pp_group
    if group.is_last_rank:
        return _Later(scheduler)
    if not group.is_first_rank:
        return original(scheduler)
    d, scheduler.pp_proxy_recv_event = scheduler._pp_recv_typed_dict(expected_kind="proxy",
                                                                     all_gather_group=scheduler.attn_tp_group)
    S.received = d["hidden_states"]
    return None


def _forward(original, runner, forward_batch, *args, **kwargs):
    """The head's forward: embed the pass's tokens, as SGLang made them for it, and send them to the first step's ranks;
    then receive this rank's own input and run."""
    later = kwargs.get("pp_proxy_tensors")
    if not isinstance(later, _Later):
        with pp.label("p2p:forward_draft" if runner.is_draft_worker else "p2p:forward"):
            return original(runner, forward_batch, *args, **kwargs)
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    scheduler, group = later.scheduler, later.scheduler.pp_group
    with pp.label("p2p:embed_send"):
        x = runner.model.get_input_embeddings()(forward_batch.input_ids)
        made = scheduler.device_module.current_stream()
        with scheduler.pp_comm_stream_ctx:
            scheduler.device_module.current_stream().wait_stream(made)
            pp.mark("gpu:out")
            streamed = _one(scheduler) and forward_batch.forward_mode.is_target_verify()
            sent, S.sent_rows = S.sent_rows if streamed else 0, 0
            if sent:                        # a tree that went in phases (_cut): its last one, behind the whole tree
                _to_all(scheduler, _tree_message(forward_batch.input_ids, *S.tree), more=(sent, x.shape[0]))
            _send_rows(scheduler, x[sent:], sent, x.shape[0], streamed)
            pp.flush()                      # the other ranks' outputs and requests, while this pass is under way
    d, S.arrives = scheduler._pp_recv_typed_dict(expected_kind="proxy", all_gather_group=scheduler.attn_tp_group)
    kwargs["pp_proxy_tensors"] = PPProxyTensors(d)
    with pp.label("p2p:forward"):
        return original(runner, forward_batch, *args, **kwargs)


def _send_rows(scheduler, x, start: int, n: int, streamed: bool) -> None:
    """The head's embedding of a pass's rows from `start` on (x), to the first step's ranks. A streamed pass's goes in
    pieces: the first a message that tells the pass's rows (n), the rest raw behind it."""
    group = scheduler.pp_group
    dsts = targets(S.ranks, group.rank_in_group, _stepwise(scheduler))
    at, end = start, start + x.shape[0]
    for b in [b for b in _bounds(n) if start < b <= end] if streamed else [end]:
        rows = {"hidden_states": x[at - start:b - start]}
        if at == 0:                         # packed here, not by SGLang's send: this may run inside its draft
            packed = pp.pack(group, {**rows, "__msg_type__": "proxy", **({"__rows__": n} if streamed and n > b else {})})
            for dst in dsts:
                pp.keep(pp.send_packed(group, packed, dst))
        else:
            _send_piece(group, rows, dsts)
            S.pieces["sent"] += len(dsts)
        if at == start:                     # behind the first rows: what the next steps' ranks wait for (pp.NEAR)
            pp.flush(near=True)
        at = b


def _to_all(scheduler, tensor_dict: dict, more=None) -> None:
    """A message from the head to every other rank, staggered as a pass's output is (p2p.pp.stagger). more: the rows
    that follow it raw to the first step's ranks (_more_in)."""
    group = scheduler.pp_group
    head = group.world_size - 1
    first = targets(S.ranks, head, True)
    packed = pp.pack(group, tensor_dict)
    with_rows = pp.pack(group, {**tensor_dict, "__more__": more}) if more else packed
    pp.keep(pp.stagger(list(range(head)), lambda ds: [w for d in ds for w in
                                                      pp.send_packed(group, with_rows if d in first else packed, d)]))


def _tree_message(chain, parents, index) -> dict:
    """A phase of the phased tree the head is drafting, as every rank lays it out: the tokens of its rows (the root
    first), SGLang's parent list and the candidates chosen (p2p.stream.regrown)."""
    return {"spec_next_chain": chain, "spec_next_parents": parents, "spec_next_top_scores": index,
            "__msg_type__": "tree", "__tree__": S.trees}


# A tree whose first rows go on while the rest is drafted (P2P_PP_PHASES), on the head.

def _verified(original, worker, batch, *args, **kwargs):
    """SGLang's verify on the head: its result is what a phased tree's first phase sends on as the pass's output."""
    out = original(worker, batch, *args, **kwargs)
    if S.scheduler is not None and S.scheduler.pp_group.is_last_rank:
        S.result = batch, out
    return out


def _candidates(original, i, topk_p, *args, **kwargs):
    """SGLang's choice of the candidates a draft step expands. On the head's draft of a streamed tree they are kept,
    and after the step that ends a phase the tree's next rows are fixed and sent (_cut)."""
    out = original(i, topk_p, *args, **kwargs)
    scheduler = S.scheduler
    if i == 0:
        topk = topk_p.shape[-1]
        on_head = scheduler is not None and scheduler.pp_group.is_last_rank
        S.drafting = {"scores": [], "tokens": [], "parents": [], "chosen": [], "cut": 0, "topk": topk} \
            if on_head and S.result is not None and topk > 1 and topk_p.shape[0] == 1 and _one(scheduler) else None
    d = S.drafting
    if d is not None:
        for name, t in zip(("scores", "tokens", "parents"), out[3]):
            d[name].append(t)
        if d["cut"] < len(S.phases) and i + 1 == S.phases[d["cut"]][0]:
            _cut(d, i)
    return out


def _cut(d: dict, i: int) -> None:
    """After the draft step that ends a phase: fix the tree's next rows (p2p.stream.grown) and send them while SGLang
    drafts the rest. The first phase takes the pass's output along, the rows not yet fixed as placeholders under the
    root, and the next rounds' requests go ahead (p2p.pp.ahead); a later phase is that tree again, further on."""
    from sglang.srt.runtime_context import get_spec
    scheduler = S.scheduler
    runner, spec = scheduler.tp_worker.model_runner, get_spec()
    c, topk, n, depth = d["cut"], d["topk"], spec.speculative_num_draft_tokens, spec.speculative_num_steps
    steps, rows = S.phases[c]
    assert steps < depth and rows < n, f"P2P_PP_PHASES={S.phases}: a tree of {n} rows is drafted in {depth} steps"
    scores = torch.cat(d["scores"], dim=1).flatten(1)[0].float().tolist()      # the CPU waits for the draft so far
    parents = torch.cat(d["parents"], dim=1)
    beam = d["parents"][-1][0].tolist() if i else list(range(topk))            # what the next step expands
    new = stream.grown(scores, parents[0].tolist(), topk, set(d["chosen"]), S.phases[c - 1][0] if c else 0, steps, beam,
                       rows - 1 - len(d["chosen"]))
    assert len(new) == rows - 1 - len(d["chosen"]), f"P2P_PP_PHASES={S.phases}: {steps} steps draft too few for {rows} rows"
    d["chosen"] += new
    d["cut"] += 1
    batch, result = S.result
    index = torch.tensor([d["chosen"] + [0] * (n - rows)], dtype=torch.long, device=parents.device)
    drafts = torch.gather(torch.cat(d["tokens"], dim=1), 1, index)
    chain = torch.cat([result.next_draft_input.bonus_tokens.view(1, 1).to(drafts.dtype), drafts], dim=1).flatten()
    parents = torch.nn.functional.pad(parents, (0, topk + 1 + topk * (depth - 2) - parents.shape[1]))
    x = runner.model.get_input_embeddings()(chain[S.sent_rows:rows])
    made = scheduler.device_module.current_stream()
    with scheduler.pp_comm_stream_ctx, pp.label("p2p:phase_send"):
        scheduler.device_module.current_stream().wait_stream(made)
        pp.mark("gpu:phase")
        if c == 0:
            S.trees += 1
            result.next_verify_chain, result.next_verify_parent_list, result.next_verify_top_scores_index = chain, parents, index
            _to_all(scheduler, {**scheduler._pp_prepare_tensor_dict(result, batch), "__msg_type__": "output",
                                "__phases__": S.trees})
            pp.ahead(scheduler.pp_loop_size)
            S.early_out = True
        else:
            _to_all(scheduler, _tree_message(chain, parents, index), more=(S.sent_rows, rows))
        _send_rows(scheduler, x, S.sent_rows, n, True)
    S.sent_rows = rows


def _organized(original, score_list, token_list, parents_list, num_draft_token):
    """SGLang's choice of a tree from every candidate drafted. A tree whose first phases are fixed keeps them and adds
    the best of the candidates drafted since (p2p.stream.grown), which come after them in SGLang's layout."""
    out = original(score_list, token_list, parents_list, num_draft_token)
    d, S.drafting, S.result = S.drafting, None, None
    if d is None or not d["cut"]:
        return out
    parent_list, index, _ = out
    scores = torch.cat(score_list, dim=1).flatten(1)[0].float().tolist()
    new = stream.grown(scores, parent_list[0].tolist(), d["topk"], set(d["chosen"]), S.phases[d["cut"] - 1][0],
                       len(score_list), [], num_draft_token - 1 - len(d["chosen"]))
    assert len(new) == num_draft_token - 1 - len(d["chosen"]), f"P2P_PP_PHASES={S.phases}: too few candidates drafted"
    index = torch.tensor([d["chosen"] + new], dtype=index.dtype, device=index.device)
    S.tree = parent_list, index
    return parent_list, index, torch.gather(torch.cat(token_list, dim=1), 1, index)


def _received():

    class Received(torch.nn.Module):
        """Rank 0's embedding: what the head embedded and sent for this pass (its first tokens, for a piece of it)."""

        def forward(self, input_ids):
            x, n = S.received, input_ids.shape[0]
            assert x is not None and x.shape[0] >= n, "rank 0 runs only on the head's embedding"
            if x.shape[0] == n:
                S.received = None
            return x[:n]
    return Received()


def _embedding(model) -> tuple:
    """(parent module, attribute) of the model's input embedding: SGLang's get_input_embeddings()."""
    assert hasattr(model, "get_input_embeddings"), f"{type(model).__name__} has no get_input_embeddings()"
    embed = model.get_input_embeddings()
    name = next(n for n, m in model.named_modules() if m is embed)
    parent, _, attr = name.rpartition(".")
    return model.get_submodule(parent), attr


def _construct(original, loader, **kwargs):
    """The target model, built and loaded by SGLang as on a stage that embeds (the head) or doesn't (rank 0): the head
    holds the model's own input embedding module and its weights, rank 0 none."""
    from sglang.srt.distributed.parallel_state import get_pp_group
    g = get_pp_group()
    if S.target is None and g.world_size > 1 and g.rank_in_group in (0, g.world_size - 1):
        S.pretend = {"is_first_rank": g.rank_in_group == g.world_size - 1}
    try:
        return original(loader, **kwargs)
    finally:
        S.pretend = {}


def _tie(model) -> None:
    """A model that ties its embedding to its head: under pipelining SGLang loads the table into the head only, so the
    embedding module takes the head's."""
    config = model.config
    if any(getattr(c, "tie_word_embeddings", False) for c in (config, getattr(config, "text_config", None)) if c is not None):
        model.get_input_embeddings().weight = _head(model).weight
        torch.cuda.empty_cache()


def _head(model):
    """The model's output head: SGLang's ParallelLMHead."""
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
    heads = [m for m in model.modules() if isinstance(m, ParallelLMHead)]
    assert len(heads) == 1, f"{type(model).__name__}: {len(heads)} output heads"
    return heads[0]


# The copied layer's updates for its replica, from its rank to the replica's.

def _update(scheduler) -> dict:
    """What the copied layer's pass leaves that the replica needs: its K/V writes; or linear-attention state after prefill,
    and the inputs of its state update on decode and verify passes."""
    taken, S.taken = S.taken, {}
    runner = scheduler.tp_worker.model_runner
    if "call" in taken:
        return {"__msg_type__": "update", "kind": "call", **taken["call"]}
    if "slots" in taken:
        slots, cache = taken["slots"], runner.req_to_token_pool.mamba2_layer_cache(S.copied)
        return {"__msg_type__": "update", "kind": "state", "slots": slots, "temporal": cache.temporal[slots],
                **{f"conv{i}": c[slots] for i, c in enumerate(cache.conv)}}
    assert taken.get("kv"), f"layer {S.copied} wrote no K/V through its pool's set_kv_buffer (a fused writer?)"
    out = {"__msg_type__": "update", "kind": "kv", "calls": len(taken["kv"])}
    for j, call in enumerate(taken["kv"]):
        out |= {f"{j}:{k}": v for k, v in call.items()}
    return out


def _kv_written(original, layer, loc, cache_k, cache_v, *args, **kwargs):
    """The copied layer's rank: its pool's set_kv_buffer, keeping what the copied layer writes for the replica's rank."""
    if S.planned and layer.layer_id == S.copied:
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc, unwrap_write_loc, write_loc_is_physical
        at, swa, full = unwrap_write_loc(loc)
        call = {"layer": layer.layer_id, "loc": at, "swa_loc": swa, "full_loc": full, "physical": write_loc_is_physical(loc),
                "wrapped": isinstance(loc, KVWriteLoc), "k": cache_k.clone(),
                "v": cache_v.clone() if cache_v is not None else None, "args": len(args), "keys": sorted(kwargs)}
        call |= {f"arg{i}": a for i, a in enumerate(args)} | {f"kw:{k}": v for k, v in kwargs.items()}
        S.taken.setdefault("kv", []).append(call)
    return original(layer, loc, cache_k, cache_v, *args, **kwargs)


def _drain(scheduler) -> None:
    """Apply every update sent so far, in order: before this rank receives a pass's input, and before it commits a
    verify pass's linear-attention state."""
    if not (S.planned and S.fed is not None and S.pending):
        return
    current = scheduler.device_module.current_stream()
    current.wait_stream(scheduler.forward_stream)   # the replica's own pass is done with its K/V and state
    with pp.label("p2p:apply_updates"):
        while S.pending:
            _apply(scheduler, _take(scheduler.pp_group, S.fed, "update"))
            S.pending -= 1
    scheduler.forward_stream.wait_stream(current)   # and the next pass sees them


def _apply(scheduler, d: dict) -> None:
    runner = scheduler.tp_worker.model_runner
    if d["kind"] == "kv":
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc
        for j in range(d["calls"]):
            c = {k.split(":", 1)[1]: v for k, v in d.items() if k.startswith(f"{j}:")}
            loc = KVWriteLoc(c["loc"], c["swa_loc"], c["full_loc"], physical=c["physical"]) if c["wrapped"] else c["loc"]
            layer = next(m for m in S.layer.modules() if getattr(m, "layer_id", None) == c["layer"] and
                         hasattr(m, "tp_k_head_num"))      # the copy's RadixAttention: the same layer, pruned
            runner.token_to_kv_pool.set_kv_buffer(layer, loc, c["k"], c["v"], *(c[f"arg{i}"] for i in range(c["args"])),
                                                  **{k: c[f"kw:{k}"] for k in c["keys"]})
    elif d["kind"] == "state":
        cache = runner.req_to_token_pool.mamba2_layer_cache(S.copied)
        for i, c in enumerate(cache.conv):
            c[d["slots"]] = d[f"conv{i}"]
        cache.temporal[d["slots"]] = d["temporal"]
    else:
        _replay(d, *S.saved.popleft())


def _replay(d: dict, fb, metadata, ctx) -> None:
    """Re-run the replica's own linear-attention call of that pass, on the copied layer's inputs: the conv and recurrent
    state advance as that layer's did, through the same kernels."""
    from sglang.srt.model_executor.forward_context import forward_context
    attn = next(m for m in S.layer.modules() if type(m).__name__ == "RadixLinearAttention")
    backend = getattr(ctx.attn_backend, "linear_attn_backend", ctx.attn_backend)
    qkv = d["qkv0"] if d["qkv_n"] == 0 else tuple(d[f"qkv{i}"] for i in range(d["qkv_n"]))
    kept, S.replaying = backend.forward_metadata, True
    try:
        with forward_context(ctx):
            backend.forward_metadata = metadata
            attn(fb, mixed_qkv=qkv, a=d["a"], b=d["b"])
    finally:
        backend.forward_metadata, S.replaying = kept, False


def _linear(original, attn, forward_batch, *args, **kwargs):
    """The copied layer's linear-attention call: its rank keeps its inputs (passes) or the slots of its state (prefill);
    the replica's rank keeps what it needs to replay its own call of the same pass."""
    if S.planned and not S.replaying and attn.layer_id == S.copied and S.copied is not None:
        from sglang.srt.model_executor.forward_context import get_forward_context
        ctx = get_forward_context()
        backend = getattr(ctx.attn_backend, "linear_attn_backend", ctx.attn_backend)
        if S.feeds is not None:
            if _passes(forward_batch):
                named = dict(zip(("mixed_qkv", "a", "b"), args)) | kwargs
                qkv = named["mixed_qkv"]
                parts = qkv if isinstance(qkv, tuple) else (qkv,)
                S.taken["call"] = {"qkv_n": len(parts) if isinstance(qkv, tuple) else 0, "a": named["a"].clone(),
                                   "b": named["b"].clone(), **{f"qkv{i}": t.clone() for i, t in enumerate(parts)}}
            else:
                S.taken["slots"] = backend.forward_metadata.mamba_cache_indices.clone()
        elif S.fed is not None and _passes(forward_batch):
            S.saved.append((forward_batch, _cloned(backend.forward_metadata), ctx))
    return original(attn, forward_batch, *args, **kwargs)


def _cloned(metadata):
    out = _copy.copy(metadata)
    for k, v in vars(metadata).items():
        if hasattr(v, "clone"):
            setattr(out, k, v.clone())
    return out


def _commit(original, *args, **kwargs):
    """Before a verify pass's linear-attention state is committed, its intermediate states must be the copied layer's."""
    if S.scheduler is not None:
        _drain(S.scheduler)
    return original(*args, **kwargs)


# Replicas prepared ahead of time (p2p.prepare), loaded instead of the copied layer's projections.

def _load(original, model, weights, target_device):
    """The target model's weights. On a replica's rank the prepared replica replaces the copied layer's projections
    (P2P_PP_REPLICAS); the head lends its embedding to a drafter; rank 0's embedding takes what the head sends."""
    if S.ranks is None or S.target is not None:            # not routed, or a drafter after its target
        return original(model, weights, target_device)
    from p2p import prepare
    from sglang.srt.runtime_context import get_parallel
    S.target, i = model, get_parallel().pp_rank
    out = original(model, weights, target_device)
    _after_input(model)
    folder = os.environ.get("P2P_PP_REPLICAS")
    if folder and partners(S.ranks, i)[1] is not None:
        S.prepared = prepare.install_replica(model, os.path.join(folder, f"layer{S.ranks[i].start}.safetensors"),
                                             os.environ.get("P2P_PP_REPLICA_FORM", "compact"))
        assert S.prepared["layer"] == S.ranks[i].start, S.prepared["layer"]
    parent, attr = _embedding(model)
    if i == len(S.ranks) - 1:
        _tie(model)
        model.get_embed_and_head = lambda: (model.get_input_embeddings().weight, _head(model).weight)
    elif i == 0:
        setattr(parent, attr, _received())
    return out


# A spread member's share of replica.py's sum, on decode and verify passes; prefill runs its layers alone.

def install_plan(scheduler, layers, plan: "dict | None", sums: dict) -> None:
    """Apply a plan on this rank: its whole groups as on one GPU, or its member of a spread step. With no plan the
    spread ranks run one layer each, in sequence."""
    from p2p import replica
    from sglang.srt.runtime_context import get_parallel
    if S.scheduler is not None:
        _drain(S.scheduler)                 # the last pass's update under the plan before
    replica.remove(layers)
    i = get_parallel().pp_rank
    r = S.ranks[i]
    S.scheduler, S.planned = scheduler, bool(plan and plan["groups"])
    S.feeds, S.fed = partners(S.ranks, i)
    S.taken, S.pending, S.saved, S.layouts = {}, 0, collections.deque(), {}
    group = steps(S.groups, len(layers))[r.first]
    me = layers[r.start]
    if S.feeds is not None or S.fed is not None:
        S.copied, S.layer = r.start, me
    pool = scheduler.tp_worker.model_runner.token_to_kv_pool
    if S.feeds is not None and not isinstance(pool.set_kv_buffer, functools.partial):
        pool.set_kv_buffer = functools.partial(_kv_written, pool.set_kv_buffer)
    if S.fed is not None:                   # the replica's rank never runs the copied layer exact
        me.forward = functools.partial(_skip, me)
    if not S.planned:
        return
    assert plan["groups"] == S.groups, "install the plan in P2P_PP_PLAN: the ranks' layers were laid out for it"
    if r.member is None:
        replica.install(layers, plan, sums, [g for g in plan["groups"] if r.start <= g[0] and g[-1] < r.end])
        return
    if S.fed is None:
        me.forward = functools.partial(_member, me, r.member, None)
        return
    keep = plan["keep"][str(S.copied)]
    if S.prepared is not None:
        assert S.prepared["keep"] == keep, f"the replica in P2P_PP_REPLICAS keeps {S.prepared['keep']}, the plan {keep}"
    else:
        assert min(keep.values()) >= 100, \
            f"layer {S.copied}'s replica is pruned: prepare it ahead of time (p2p replicas) and set P2P_PP_REPLICAS"
    own = layers[group[-1]]
    me.forward = functools.partial(_member, me, r.member, own)
    own.forward = functools.partial(_through, own)


def _passes(fb) -> bool:
    return fb.forward_mode.is_decode() or fb.forward_mode.is_target_verify()


def _member(me, k: int, own, *args, **kw):
    """Member k's part of the step, in the model's dtype: l(h) for the first, l(h) - h for the middle ones, own(x) - x
    for the last, where x = copy(h) runs on the replica. In prefill the last member passes its input on to its own
    layer."""
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from p2p import replica
    named = replica._by_name(me, args, kw)
    fb, h = named["forward_batch"], named["hidden_states"]
    if not _passes(fb):
        return h if own is not None else type(me).forward(me, **named)
    out, residual = residual_batch.stream_of(fb).export(h)
    h = out if residual is None else out + residual
    if own is None:
        y = replica._run(me, h, named)
        part = y if k == 0 else (y.float() - h.float()).to(y.dtype)
    else:
        with family.restored_state(me):        # the copy's guess for this token leaves no trace in the layer's state
            x = replica._run(me, h, named)
        pp.mark("gpu:copied")
        part = (replica._run(own, x, named).float() - x.float()).to(x.dtype)
    return residual_batch.set_written(part, fb)


def _through(layer, *args, **kw):
    """The last member's own layer: its part was made in the copied layer's call, so it passes it on."""
    from p2p import replica
    named = replica._by_name(layer, args, kw)
    if _passes(named["forward_batch"]):
        return named["hidden_states"]
    return type(layer).forward(layer, **named)


def _skip(layer, *args, **kw):
    from p2p import replica
    return replica._by_name(layer, args, kw)["hidden_states"]


def p2p_held(self, path: str) -> None:
    """Write path/pp<rank>.json: per layer this rank holds, the share of zero weights and its bytes; its embedding's
    bytes; the pieces it sent and received; and P2P_PP_EARLY=check's comparisons."""
    from sglang.srt.runtime_context import get_parallel
    from p2p.plugin import _layers
    model = self.tp_worker.model_runner.model
    embedding = sum(p.numel() * p.element_size() for p in model.get_input_embeddings().parameters())
    out = {}
    for i, layer in enumerate(_layers(self)):
        if type(layer).__name__ == "PPMissingLayer":
            continue
        modules = dict(layer.named_modules())
        ws = [getattr(modules[p.path], p.param) for p in {p.path: p for p in family.projections(layer).values()}.values()]
        n = sum(w.numel() for w in ws)
        compact = sum(m.quant_method.nbytes() for m in layer.modules() if hasattr(getattr(m, "quant_method", None), "nbytes"))
        out[i] = {"zero": sum((w == 0).sum().item() for w in ws) / n if n else None,
                  "bytes": sum(w.numel() * w.element_size() for w in layer.parameters()) + compact}
    name = os.path.join(path, f"pp{get_parallel().pp_rank}.json")
    with open(name + ".part", "w") as f:
        json.dump({"layers": out, "embedding": embedding, "pieces": S.pieces, "check": S.check}, f)
    os.replace(name + ".part", name)
