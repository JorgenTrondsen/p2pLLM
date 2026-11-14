"""How pipeline-parallel stages talk, for deployment over a network (P2P_PP_COMM).

    stock   SGLang's own: requests over gloo, tensors as gloo metadata then an NCCL message each: about three delays a hop
    fused   requests over NCCL, each tensor dict as two NCCL messages (its sizes, then one buffer with the metadata and
            every tensor): one delay a hop. Needs one GPU per stage. A receiver that knows a message's layout can queue
            its receive without waiting for it (receive)

The entry stage (p2p.route's head) sends requests and outputs to every other stage, some at once and some held (stagger,
hold, flush), and can send the next rounds' requests ahead (ahead).
Measuring: label and mark name spans and points on the GPU (P2P_PP_SPANS, bench/pp/overhead.py); P2P_PP_TRACE counts
messages. Simulating: P2P_PP_MBPS a stage's uplink rate (link, occupy); P2P_PP_SHARE_GPUS several stages to a GPU.
P2P_PP_EAGER_NCCL=1 makes one NCCL communicator per group, not one per pair of stages.
"""
from __future__ import annotations

import collections
import json
import math
import os
import pickle
import time

import torch
import torch.distributed as dist

ENTRY = None               # set by p2p.route: the stage that takes requests in and sends them to every other stage
SENT = []                  # sends under way, alive until done (keep)
ORDER = URGENT = None      # set by p2p.route: the other stages in the order the entry stage sends to them, and those it
LATER = collections.deque()    # sends to at once; the rest it holds (LATER: sends, in order) until flush(), once the
AGE = 0                    # next pass's input is out, or the loop has gone round 3 times (AGE) without one
NEAR = frozenset()         # set by p2p.route: of the stages it holds for, those whose sends go right behind the next
                           # pass's first rows (flush(near=True)): they are needed before the rest of the rows is out
AHEAD = 0                  # rounds whose requests the entry stage has sent already, empty (ahead)
WARM = False               # whether its pipeline-group links exist
HANDOFF = None             # set by p2p.route: fn(boundary, forward_batch), the boundary a stage restores its input at
ALIGN = 256                # bytes, as CUDA allocations are: every tensor in a fused buffer is a view that kernels accept
COUNTS = collections.Counter()
SPANS = None               # this stage's spans while p2p_spans records: (name, start ns, end ns)
MARKS = []                 # and its marks (mark): (name, CUDA event), timed against REFERENCE: (event, its ns)
REFERENCE = None


def label(name: str):
    """A named span of a stage's pass, as SGLang's profile_range makes its own."""
    from sglang.srt.utils import nvtx_utils
    return nvtx_utils.profile_range(name)


class _Span:
    """A span of SGLang's (torch.profiler.record_function in its PP loop, profile_range elsewhere) or ours, timed on the
    machine's monotonic clock while p2p_spans records; under SGLang's profiler, its span as before."""
    __slots__ = ("name", "start", "inner")
    record = None                   # torch.profiler.record_function

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.inner = _Span.record(name, *args, **kwargs) if torch.autograd._profiler_enabled() else None

    def __enter__(self):
        if self.inner is not None:
            return self.inner.__enter__()
        self.start = time.monotonic_ns()
        return self

    def __exit__(self, *exc):
        if self.inner is not None:
            return self.inner.__exit__(*exc)
        if SPANS is not None:
            SPANS.append((self.name, self.start, time.monotonic_ns()))
        return False


def _spans() -> None:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.utils import nvtx_utils
    _Span.record = torch.profiler.record_function
    torch.profiler.record_function = _Span
    nvtx_utils.profile_range = lambda debug_name, **kwargs: _Span(debug_name)
    Scheduler.p2p_spans = p2p_spans


def mark(name: str) -> None:
    """A point on the GPU's timeline, where the current stream has got to: a span of no length when the GPU gets there,
    while p2p_spans records."""
    if SPANS is not None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        MARKS.append((name, event))


def p2p_spans(self, on: bool, path: str = "") -> None:
    """Record this stage's spans afresh (on), or stop and write them to path/spans_pp<rank>.json: marks timed on the GPU
    against an event whose time on the machine's clock is known."""
    global SPANS, REFERENCE
    from sglang.srt.runtime_context import get_parallel
    if on:
        torch.cuda.synchronize()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        event.synchronize()
        SPANS, REFERENCE = [], (event, time.monotonic_ns())
        MARKS.clear()
        return
    spans, SPANS = SPANS or [], None
    if MARKS:
        torch.cuda.synchronize()
        event, ns = REFERENCE
        for name, e in MARKS:
            t = ns + round(event.elapsed_time(e) * 1e6)
            spans.append((name, t, t))
        MARKS.clear()
    if path:
        out = os.path.join(path, f"spans_pp{get_parallel().pp_rank}.json")
        with open(out + ".tmp", "w") as f:
            json.dump(spans, f)
        os.replace(out + ".tmp", out)


def install_launcher() -> None:
    """Hooks for the process that launches the schedulers."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    if os.environ.get("P2P_PP_SHARE_GPUS"):
        os.environ["NCCL_MULTI_RANK_GPU_ENABLE"] = "1"      # NCCL refuses several ranks of a group on one GPU otherwise
        HookRegistry.register("sglang.srt.entrypoints.engine.maybe_reindex_device_id", _shared_gpu, HookType.AROUND)


def install() -> None:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    Scheduler.p2p_pp_counts = p2p_pp_counts
    HookRegistry.register("sglang.srt.layers.layer_boundary.stage.StageBoundary.from_pp", _written, HookType.AROUND)
    if os.environ.get("P2P_PP_TRACE"):
        _trace()
    if os.environ.get("P2P_PP_SPANS"):
        _spans()
    if os.environ.get("P2P_PP_EAGER_NCCL"):
        _eager()
    comm = os.environ.get("P2P_PP_COMM", "stock")
    assert comm in ("stock", "fused"), f"P2P_PP_COMM={comm}"
    if comm == "stock":
        return
    on = lambda path, fn: HookRegistry.register("sglang.srt." + path, fn, HookType.AROUND)
    on("managers.scheduler_pp_mixin.SchedulerPPMixin._pp_send_pyobj_to_next_stage", _send_reqs)
    on("managers.scheduler_components.request_receiver.SchedulerRequestReceiver._pull_raw_reqs", _pull_reqs)
    on("managers.scheduler_components.ipc_channels.SchedulerIpcChannels.create", _channels)
    on("managers.scheduler.Scheduler.init_ipc_channels", _ipc)
    on("distributed.parallel_state.GroupCoordinator.send_tensor_dict", _send_fused)
    on("distributed.parallel_state.GroupCoordinator.recv_tensor_dict", _recv_fused)


def _eager() -> None:
    original = dist.init_process_group

    def init_process_group(*args, **kwargs):
        if "nccl" in str(kwargs.get("backend", args[0] if args else "")) and torch.cuda.is_available():
            kwargs.setdefault("device_id", torch.device("cuda", torch.cuda.current_device()))
        return original(*args, **kwargs)
    dist.init_process_group = init_process_group


def _shared_gpu(original, gpu_id):
    return original(gpu_id % int(os.environ["P2P_PP_SHARE_GPUS"]))


def _written(original, boundary, tensors, forward_batch, **kwargs):
    """A stage's input without a separate residual is a stream already summed (a group's sum, a spread step's parts).
    SGLang takes that only from layers that declare it."""
    if "residual" not in tensors.tensors:
        kwargs["allow_missing_residual"] = True
    if HANDOFF is not None:
        boundary = HANDOFF(boundary, forward_batch)
    return original(boundary, tensors, forward_batch, **kwargs)


def p2p_pp_counts(self, path: str = "") -> None:
    """Write this scheduler's message counts to path/pp<rank>_tp<rank>.json (if given), then start counting afresh."""
    from sglang.srt.runtime_context import get_parallel
    if path:
        p = get_parallel()
        with open(os.path.join(path, f"pp{p.pp_rank}_tp{p.tp_rank}.json"), "w") as f:
            json.dump(dict(COUNTS), f)
    COUNTS.clear()


def counts(path: str, stages: int) -> dict:
    """{"pp<rank>_tp<rank>": counts} that p2p_pp_counts wrote, once every stage has: an RPC returns when the stage that
    takes requests in has run it, and it reaches the others afterwards."""
    for _ in range(600):
        if len(names := sorted(os.listdir(path))) == stages:
            time.sleep(0.2)                     # the last writer may still be writing
            return {f.removesuffix(".json"): json.load(open(os.path.join(path, f))) for f in names}
        time.sleep(0.1)
    raise TimeoutError(f"{path}: {os.listdir(path)} of {stages} stages")


def _trace() -> None:
    for name in ("send", "isend", "recv", "irecv"):
        def counted(tensor, *args, _fn=getattr(dist, name), _name=name, **kwargs):
            backend = dist.get_backend(kwargs.get("group"))
            COUNTS[f"{backend} {_name}"] += 1
            COUNTS[f"{backend} {_name} bytes"] += tensor.numel() * tensor.element_size()
            return _fn(tensor, *args, **kwargs)
        setattr(dist, name, counted)


# Requests, relayed from stage to stage as point_to_point_pyobj does, but on the world group's NCCL communicator,
# or (ENTRY) sent from one stage to every other.

def _peers():
    """World ranks (this stage, the one it receives requests from, the ones it sends them to) of the rank that relays
    requests, else None. Under ENTRY that stage sends them to every other."""
    from sglang.srt.runtime_context import get_parallel
    p = get_parallel()
    if p.attn_tp_rank or p.attn_cp_rank:
        return None
    at = _world
    if ENTRY is not None:
        e = ENTRY % p.pp_size
        return at(p.pp_rank), at(e), [at(r) for r in range(p.pp_size) if r != e] if p.pp_rank == e else []
    return at(p.pp_rank), at(p.pp_rank - 1), [at(p.pp_rank + 1)]


def _obj_tensors(data, device) -> list:
    payload = pickle.dumps(data) if data else b""
    tensors = [_on_gpu(torch.tensor([len(payload)], dtype=torch.long), device)]
    if payload:
        tensors.append(_on_gpu(torch.frombuffer(bytearray(payload), dtype=torch.uint8), device))
    return tensors


def _isend(tensors: list, dsts: list, group, async_send: bool = True) -> list:
    from sglang.srt.distributed.parallel_state import P2PWork
    fn = dist.isend if async_send else dist.send
    works = [P2PWork(fn(t, dst, group=group), t) for dst in dsts for t in tensors]
    return works if async_send else []


def _world(pp: int) -> int:
    """The world rank of a pipeline rank's request relay (its attention-TP rank 0)."""
    from sglang.srt.runtime_context import get_parallel
    p = get_parallel()
    return (pp % p.pp_size) * p.tp_size + p.attn_dp_rank * p.attn_cp_size * p.attn_tp_size


def _send_obj(data, dsts: list, async_send: bool) -> list:
    from sglang.srt.distributed.parallel_state import get_world_group
    g = get_world_group()
    with label(f"p2p:send_reqs->{len(dsts)}"):
        return _isend(_obj_tensors(data, g.device), dsts, g.device_group, async_send)


def keep(works) -> None:
    """Sends under way, kept alive (their tensors with them) until they are done."""
    SENT[:] = [w for w in SENT if not w.work.is_completed()] + list(works)


def stagger(dsts: list, send, rank=lambda r: r) -> list:
    """send(stages) issues the entry stage's sends and returns their works. At once to URGENT (returned); the others
    held: NEAR until the next pass's first rows are out, the rest until all of them are. rank maps a pipeline rank to
    dsts' numbering."""
    if URGENT is None:
        return send(list(dsts))
    order = [rank(r) for r in ORDER]
    urgent, near = {rank(r) for r in URGENT}, {rank(r) for r in NEAR} - {rank(r) for r in URGENT}
    assert set(order) == set(dsts), (order, dsts)
    works = send([d for d in order if d in urgent])
    for later, soon in (([d for d in order if d in near], True), ([d for d in order if d not in urgent | near], False)):
        if later:
            hold(lambda later=later: send(later), near=soon)
    return works


def hold(send, near: bool = False) -> None:
    """Send later (flush), after what the current stream has queued so far: send() issues sends, returns their works.
    near: with the sends that go right behind a pass's first rows."""
    ready = torch.cuda.Event()
    ready.record()
    LATER.append((ready, send, near))


def flush(near: bool = False) -> None:
    """The held sends, in order; near: only those held for right behind a pass's first rows."""
    global AGE
    if not near:
        AGE = 0
    if not any(n or not near for _, _, n in LATER):
        return
    with label(f"p2p:flush{'_near' if near else ''}->{len(LATER)}"):
        kept = []
        while LATER:
            ready, send, n = LATER.popleft()
            if near and not n:
                kept.append((ready, send, n))
                continue
            torch.cuda.current_stream().wait_event(ready)
            keep(send())
        LATER.extend(kept)


def ahead(rounds: int) -> None:
    """The entry stage sends the next rounds' requests now, empty, so the stages can start on what the head sends
    before its round ends. For those rounds it takes none in (_pull_reqs), so every stage still sees the same requests
    in the same round."""
    global AHEAD
    from sglang.srt.distributed.parallel_state import get_world_group
    peers, g, extra = _peers(), get_world_group(), rounds - AHEAD
    if not peers or extra <= 0:
        return
    AHEAD = rounds
    tensors = _obj_tensors([], g.device)
    with label("p2p:send_reqs_ahead"):
        keep(stagger(peers[2], lambda dsts: [w for _ in range(extra) for w in _isend(tensors, dsts, g.device_group)],
                     rank=_world))


def _recv_obj(src: int):
    from sglang.srt.distributed.parallel_state import get_world_group
    g = get_world_group()
    with label(f"p2p:recv_reqs<-{src}"):
        size = torch.zeros(1, dtype=torch.long, device=g.device)
        with label("p2p:wait"):
            dist.irecv(size, src, group=g.device_group).wait()
            n = int(size.item())
        if not n:
            return []
        payload = torch.empty(n, dtype=torch.uint8, device=g.device)
        dist.irecv(payload, src, group=g.device_group).wait()
        return pickle.loads(payload.cpu().numpy().tobytes())


def _send_reqs(original, scheduler, data, async_send: bool = False):
    peers = _peers()
    return _send_obj(data, peers[2], async_send) if peers else []


def _pull_reqs(original, receiver):
    from sglang.srt.runtime_context import get_parallel
    p = get_parallel()
    entry = 0 if ENTRY is None else ENTRY % p.pp_size
    if p.pp_rank == entry == 0:
        return original(receiver)
    peers = _peers()
    if not peers:
        return None
    if ENTRY is not None:
        _warm(entry)
    if p.pp_rank != entry:
        return _recv_obj(peers[1])
    global AGE, AHEAD
    if AHEAD:                                            # this round's went ahead, empty (ahead)
        AHEAD -= 1
        return []
    AGE += bool(LATER)
    if AGE >= 3:                                         # no pass under way to send them after
        flush()
    reqs = _read(receiver)
    receiver.unwrap_pickle_wrapper(reqs)                 # SGLang does this on stage 0 only
    from sglang.srt.distributed.parallel_state import get_world_group
    g = get_world_group()
    with label("p2p:send_reqs"):
        tensors = _obj_tensors(reqs, g.device)
        keep(stagger(peers[2], lambda dsts: _isend(tensors, dsts, g.device_group), rank=_world))
    return reqs


def _warm(entry: int) -> None:
    """NCCL makes a pair's communicator at its first message and both stages must join, so the entry stage opens its
    links to every stage at the first round."""
    global WARM
    from sglang.srt.distributed.parallel_state import get_pp_group
    if WARM:
        return
    g, WARM = get_pp_group(), True
    t = torch.zeros(1, device=g.device)
    if g.rank_in_group != entry:
        return dist.recv(t, g.ranks[entry], group=g.device_group)
    for r in range(g.world_size):
        if r != entry:
            dist.send(t, g.ranks[r], group=g.device_group)


def _read(receiver) -> list:
    """What SGLang's stage 0 takes in each round: requests from the tokenizer, then RPCs, without waiting for any."""
    import zmq
    from sglang.srt.managers.io_struct import sock_recv
    reqs = []
    for sock in (receiver.recv_from_tokenizer, receiver.recv_from_rpc):
        while not receiver.recv_limit_reached(len(reqs)):
            try:
                reqs.append(sock_recv(sock, zmq.NOBLOCK))
            except zmq.ZMQError:
                break
    return reqs


def _channels(original, cls, **kwargs):
    """Under ENTRY the entry stage, not stage 0, holds the sockets to the tokenizer: requests and RPCs in, outputs out."""
    from sglang.srt.runtime_context import get_parallel
    if ENTRY is not None:
        p = get_parallel()
        kwargs["is_rank_zero"] = p.pp_rank == ENTRY % p.pp_size and not p.attn_tp_rank and not p.attn_cp_rank
    return original(cls, **kwargs)


def _ipc(original, scheduler, port_args):
    original(scheduler, port_args)
    if ENTRY is not None:
        scheduler.recv_from_tokenizer = scheduler.ipc_channels.recv_from_tokenizer


def _ours(coord, all_gather_group) -> bool:
    """Whether a tensor dict goes between pipeline stages, where fused takes it."""
    from sglang.srt.distributed.parallel_state import get_pp_group
    if coord is not get_pp_group() or coord.world_size == 1:
        return False
    assert all_gather_group is None or all_gather_group.world_size == 1, "P2P_PP_COMM=fused needs one GPU per stage"
    return True


def _padded(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def _send_fused(original, coord, tensor_dict, dst=None, all_gather_group=None, async_send=False):
    """Two NCCL messages (pack): [metadata bytes, buffer bytes], then the buffer."""
    if not _ours(coord, all_gather_group):
        return original(coord, tensor_dict, dst, all_gather_group, async_send)
    dst = (coord.rank_in_group + 1) % coord.world_size if dst is None else dst
    with label(f"p2p:send_dict->{dst}"):
        return send_packed(coord, pack(coord, tensor_dict), dst, async_send)


def pack(coord, tensor_dict: dict) -> tuple:
    """A tensor dict as fused sends it: [metadata bytes, buffer bytes], and the buffer (pickled metadata, then each
    tensor, aligned). Pack once to send the same dict to several stages."""
    from sglang.srt.distributed.parallel_state import _split_tensor_dict
    metadata, tensors = _split_tensor_dict(tensor_dict)
    head = pickle.dumps(metadata)
    parts = [torch.frombuffer(bytearray(head), dtype=torch.uint8).pin_memory()]
    parts += [t.detach().contiguous().reshape(-1).view(torch.uint8) for t in tensors]
    buffer = torch.zeros(sum(_padded(p.numel()) for p in parts), dtype=torch.uint8, device=coord.device)
    at = 0
    for p in parts:
        buffer[at:at + p.numel()].copy_(p, non_blocking=True)
        at += _padded(p.numel())
    return _on_gpu(torch.tensor([len(head), buffer.numel()], dtype=torch.long), coord.device), buffer


def _on_gpu(t: torch.Tensor, device) -> torch.Tensor:
    """A CPU tensor copied to the GPU in stream order, without the CPU waiting for the stream to get there."""
    return t.pin_memory().to(device, non_blocking=True)


def send_packed(coord, packed: tuple, dst: int, async_send: bool = True) -> list:
    """pack()'s two messages to dst, a rank in coord's group."""
    with link():
        occupy(sum(t.numel() * t.element_size() for t in packed))
        return _isend(list(packed), [coord.ranks[dst]], coord.device_group, async_send)


MBPS = float(os.environ.get("P2P_PP_MBPS", "0") or 0)
_LINK = None                # the stream a stage sends on, under P2P_PP_MBPS
_CYCLES_PER_MS = None


def link():
    """Where a stage's sends between stages go, after what the current stream has queued: under P2P_PP_MBPS a stream of
    their own, one after another as on the stage's uplink; else the current stream."""
    import contextlib
    global _LINK
    if not MBPS:
        return contextlib.nullcontext()
    if _LINK is None:
        _LINK = torch.cuda.Stream()
    _LINK.wait_stream(torch.cuda.current_stream())
    return torch.cuda.stream(_LINK)


def occupy(nbytes: int) -> None:
    """Under P2P_PP_MBPS, hold the current stream as long as an uplink of that rate takes to send nbytes."""
    global _CYCLES_PER_MS
    if not MBPS:
        return
    if _CYCLES_PER_MS is None:              # the GPU's clock, once: a spin of known cycles, timed
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(torch.cuda.Stream()):
            start.record()
            torch.cuda._sleep(10 ** 7)
            end.record()
        end.synchronize()
        _CYCLES_PER_MS = 10 ** 7 / start.elapsed_time(end)
    torch.cuda._sleep(int(nbytes * 8 / (MBPS * 1e3) * _CYCLES_PER_MS))


def _recv_fused(original, coord, src=None, all_gather_group=None):
    if not _ours(coord, all_gather_group):
        return original(coord, src, all_gather_group)
    src = (coord.rank_in_group - 1) % coord.world_size if src is None else src
    return receive(coord, src)[0]


def receive(coord, src: int, layout=None) -> tuple:
    """(the tensor dict src sent, its layout or None with CPU tensors). Given the layout of an earlier message like
    it, the receive is only queued; a message laid out otherwise stops the GPU (a device-side assert)."""
    from sglang.srt.distributed.parallel_state import TensorMetadata
    peer, group, device = coord.ranks[src], coord.device_group, coord.device
    with label(f"p2p:recv_dict<-{src}"):
        if layout is not None:
            expected, n, items = layout
            sizes = torch.empty(2, dtype=torch.long, device=device)
            buffer = torch.empty(n, dtype=torch.uint8, device=device)
            dist.irecv(sizes, peer, group=group).wait()
            dist.irecv(buffer, peer, group=group).wait()
            torch._assert_async((sizes == expected).all())
            return _unpacked(buffer, items), layout
        sizes = torch.zeros(2, dtype=torch.long, device=device)
        with label("p2p:wait"):
            dist.irecv(sizes, peer, group=group).wait()
            m, n = sizes.tolist()
        buffer = torch.empty(n, dtype=torch.uint8, device=device)
        dist.irecv(buffer, peer, group=group).wait()
        items, at = [], _padded(m)
        for key, value in pickle.loads(buffer[:m].cpu().numpy().tobytes()):
            if isinstance(value, TensorMetadata):
                items.append((key, None, (value.dtype, value.size, at, value.device == "cpu")))
                at += _padded(value.dtype.itemsize * math.prod(value.size))
            else:
                items.append((key, value, None))
        on_cpu = any(place is not None and place[3] for _, _, place in items)
        return _unpacked(buffer, items), None if on_cpu else (sizes, n, items)


def _unpacked(buffer: torch.Tensor, items: list) -> dict:
    out = {}
    for key, value, place in items:
        if place is None:
            out[key] = value
            continue
        dtype, size, at, on_cpu = place
        t = buffer[at:at + dtype.itemsize * math.prod(size)].view(dtype).reshape(size)
        out[key] = t.cpu() if on_cpu else t
    return out
