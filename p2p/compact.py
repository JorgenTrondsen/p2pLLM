"""Pruned weights stored and multiplied compactly: each row's kept weights in column order, plus a bitmask.

Wanda keeps the same number of weights in every row, so no index is stored (pack): 16 bits per kept weight plus 1 bit
per weight. On the GPU a module's blocks are one matrix of 8 x 8 pieces (Tiles), which compact.cu, built at first use,
multiplies without expanding. Routed experts are expanded per pass, only those in use (CompactMoE).
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

logger = logging.getLogger(__name__)
_EXT = None                 # compact.cu, once built: None until tried, False if it can't be


def _ext():
    """compact.cu as a torch extension, built at first use (cached by torch), or None if it can't be built here."""
    global _EXT
    if _EXT is None:
        try:
            from torch.utils.cpp_extension import load
            _EXT = load("p2p_compact", [os.path.join(os.path.dirname(__file__), "compact.cu")],
                        extra_cuda_cflags=["-O3"], verbose=False)
        except Exception as e:                     # no compiler, or a GPU without bfloat16 tensor cores
            logger.warning("p2p.compact: compact.cu not built (%s)", e)
            _EXT = False
    return _EXT or None


def pack(w: torch.Tensor, mask: torch.Tensor) -> tuple:
    """(values [rows, k], words [rows, ceil(cols / 32)] int32) of w where mask, which keeps the same k in every row."""
    rows, cols = w.shape
    k = int(mask[0].sum())
    assert bool((mask.sum(-1) == k).all()), "every row must keep the same number of weights"
    values = w[mask].view(rows, k)
    n = (cols + 31) // 32
    bits = F.pad(mask, (0, 32 * n - cols)).view(rows, n, 32).to(torch.int64)
    words = (bits << torch.arange(32, device=w.device)).sum(-1)
    return values, torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)


def unpack(values: torch.Tensor, words: torch.Tensor, cols: int) -> torch.Tensor:
    """The dense matrix that pack stored."""
    rows, k = values.shape
    if values.is_cuda:
        out = torch.empty(rows, cols, dtype=values.dtype, device=values.device)
        if rows:
            _unpack[lambda meta: (triton.cdiv(rows, meta["BR"]),)](values.contiguous(), words.contiguous(), out, rows, cols,
                                                                  words.shape[1], k)
        return out
    bits = (words.to(torch.int64)[..., None] >> torch.arange(32, device=words.device)) & 1
    mask = bits.view(rows, -1)[:, :cols].bool()
    out = torch.zeros(rows, cols, dtype=values.dtype, device=values.device)
    out[mask] = values.reshape(-1)
    return out


@triton.autotune(configs=[triton.Config({"BR": br, "BW": bw}, num_warps=nw) for br, bw, nw in
                          [(16, 4, 4), (32, 4, 4), (16, 8, 4), (8, 8, 2)]], key=["C"])
@triton.jit
def _unpack(v, m, y, R, C, W, K, BR: tl.constexpr, BW: tl.constexpr):
    """Each row's kept weights back in place, zeros elsewhere: a kept weight's place among its row's values is the count
    of mask bits before it."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    in_rows = rows < R
    seen = tl.zeros((BR,), dtype=tl.int32)
    bit = tl.arange(0, 32)
    below = (1 << bit) - 1
    for w0 in range(0, W, BW):
        words = w0 + tl.arange(0, BW)
        mw = tl.load(m + rows[:, None] * W + words[None, :], mask=in_rows[:, None] & (words < W)[None, :], other=0)
        count = libdevice.popc(mw)
        first = seen[:, None] + tl.cumsum(count, axis=1) - count
        b = tl.reshape((mw[:, :, None] >> bit[None, None, :]) & 1, (BR, BW * 32))
        at = tl.reshape(first[:, :, None] + libdevice.popc(mw[:, :, None] & below[None, None, :]), (BR, BW * 32))
        cols = w0 * 32 + tl.arange(0, BW * 32)
        vals = tl.load(v + rows[:, None] * K + at, mask=(b != 0) & in_rows[:, None], other=0.0)
        tl.store(y + rows[:, None] * C + cols[None, :], vals, mask=in_rows[:, None] & (cols < C)[None, :])
        seen += tl.sum(count, axis=1)


def expand(block: tuple, experts=None) -> torch.Tensor:
    """A stored row block as a dense matrix: ("dense", weight) or ("sparse", values, words, cols), each with a leading
    expert dimension for routed experts; of those, only `experts` (ids), if given."""
    if block[0] == "dense":
        return block[1] if experts is None else block[1][experts]
    _, values, words, cols = block
    if experts is not None:
        values, words = values[experts], words[experts]
    lead = values.shape[:-1]
    return unpack(values.reshape(-1, values.shape[-1]), words.reshape(-1, words.shape[-1]), cols).view(*lead, cols)


def _nbytes(blocks) -> int:
    return sum(t.numel() * t.element_size() for b in blocks for t in b[1:] if torch.is_tensor(t))


def _popc(words: torch.Tensor) -> torch.Tensor:
    """The bits set in each int32 word, as int64."""
    v = words.to(torch.int64) & 0xFFFFFFFF
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return ((v * 0x01010101) & 0xFFFFFFFF) >> 24


class Tiles:
    """A module's row blocks as one matrix for compact.cu: 8 x 8 pieces, each a 64-bit mask and its kept values, in
    the order the kernel reads them; woff: where each warp's values of each chunk of columns start."""

    @staticmethod
    def usable(blocks: list) -> bool:
        t = blocks[0][1]
        cols = blocks[0][3] if blocks[0][0] == "sparse" else t.shape[-1]
        return (t.is_cuda and t.dtype in (torch.bfloat16, torch.float16) and cols % 8 == 0 and t.dim() == 2
                and _ext() is not None)

    def __init__(self, blocks: list):
        ext = _ext()
        rt, kc, warps = ext.RT, ext.KCOLS, ext.WARPS
        parts = [(b[1], b[2]) if b[0] == "sparse" else pack(b[1], torch.ones_like(b[1], dtype=torch.bool))
                 for b in blocks]
        self.cols = blocks[0][3] if blocks[0][0] == "sparse" else blocks[0][1].shape[1]
        self.rows = sum(v.shape[0] for v, _ in parts)
        device, dtype = parts[0][0].device, parts[0][0].dtype
        tiles, chunks = -(-self.rows // rt), -(-self.cols // kc)
        width = chunks * kc
        starts = [0]
        for v, _ in parts:
            starts.append(starts[-1] + v.shape[0])
        step = max(1, (1 << 24) // (rt * width))               # tiles rearranged at once
        vals, masks, counts = [], [], []
        bit = torch.arange(32, device=device, dtype=torch.int32)
        for t0 in range(0, tiles, step):
            nt = min(step, tiles - t0)
            r0, r1 = t0 * rt, min(self.rows, (t0 + nt) * rt)
            dense = torch.zeros(nt * rt, width, dtype=dtype, device=device)
            kept = torch.zeros(nt * rt, width, dtype=torch.bool, device=device)
            for (v, w), a, b in zip(parts, starts, starts[1:]):
                lo, hi = max(a, r0), min(b, r1)
                if lo >= hi:
                    continue
                vv, ww = v[lo - a:hi - a].contiguous(), w[lo - a:hi - a].contiguous()
                dense[lo - r0:hi - r0, :self.cols] = unpack(vv, ww, self.cols)
                kept[lo - r0:hi - r0, :self.cols] = ((ww.unsqueeze(-1) >> bit) & 1).bool().view(hi - lo, -1)[:, :self.cols]
            order = lambda m: m.view(nt, warps, 2, 2, 8, chunks, kc // 16, 2, 8).permute(0, 5, 1, 6, 2, 7, 3, 4, 8)
            kept = order(kept).contiguous()
            vals.append(order(dense).contiguous()[kept])
            pieces = kept.view(-1, 64).long()
            masks.append((pieces << torch.arange(64, device=device)).sum(-1))
            counts.append(kept.view(nt, chunks, warps, -1).sum(-1).flatten())
        self.vals = torch.cat(vals + [torch.zeros(8, dtype=dtype, device=device)])   # cp.async reads 16 bytes on
        self.masks = torch.cat(masks)
        counts = torch.cat(counts)
        self.woff = torch.zeros(counts.numel() + 1, dtype=torch.int64, device=device)
        self.woff[1:] = counts.cumsum(0)
        per_chunk = int(counts.view(-1, warps).sum(-1).max()) if counts.numel() else 0
        self.vcap = -(-(per_chunk + 7) // 8) * 8
        self.tiles, self.chunks = tiles, chunks
        self._splits = self._tune()

    def _tune(self) -> dict:
        """{tokens a launch takes (8, 16, 32, 64): parts to split the columns into}, the fastest of a few, timed once
        at load. Splits are added in a fixed order, so a choice gives the same sums every pass."""
        ext, best = _ext(), {}
        candidates = [c for c in (1, 2, 3, 4, 6, 8, 12, 16) if c == 1 or c * 2 <= self.chunks]
        for n in (8, 16, 32, 64):
            x = torch.randn(n, self.cols, device=self.vals.device, dtype=self.vals.dtype)
            times = {}
            for c in candidates:
                run = lambda: ext.spmm(x, self.vals, self.woff, self.masks, self.rows, self.cols, self.vcap, c)
                run()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):
                    run()
                end.record()
                end.synchronize()
                times[c] = start.elapsed_time(end)
            best[n] = min(times, key=times.get)
        return best

    def splits(self, tokens: int) -> int:
        return self._splits[8 if tokens <= 8 else 16 if tokens <= 16 else 32 if tokens <= 32 else 64]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x [T, cols] @ W.T: [T, rows]."""
        x = x.contiguous()
        if x.data_ptr() % 16:
            x = x.clone()
        out = [_ext().spmm(x[i:i + 64], self.vals, self.woff, self.masks, self.rows, self.cols, self.vcap,
                           self.splits(x.shape[0] - i)) for i in range(0, max(x.shape[0], 1), 64)]
        return out[0] if len(out) == 1 else torch.cat(out)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.vals, self.masks, self.woff))


class Compact:
    """A linear module's quant_method for a replica: its row blocks as Tiles. SGLang's linear layers call
    quant_method.apply(layer, x, bias)."""

    def __init__(self, blocks: list):
        assert Tiles.usable(blocks), ("a compact replica needs compact.cu (a CUDA compiler, a GPU with tensor cores) "
                                      "and bfloat16 or float16 weights whose columns are a multiple of 8")
        self.tiles = Tiles(blocks)

    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        out = self.tiles(x.reshape(-1, x.shape[-1]))
        if bias is not None:
            out = out + bias
        return out.reshape(*x.shape[:-1], out.shape[-1])

    def process_weights_after_loading(self, layer) -> None:
        pass

    def nbytes(self) -> int:
        return self.tiles.nbytes()


class CompactMoE(torch.nn.Module):
    """A FusedMoE's quant_method for compact routed experts ({weight name: row blocks with a leading expert
    dimension}): each pass expands only the experts in use, and the layer's own method runs them."""

    def __init__(self, method, weights: dict):
        super().__init__()
        self.method, self.weights = method, weights

    def __getattr__(self, name):                # anything else SGLang asks of the layer's method
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("method"), name)

    def apply(self, layer, dispatch_output):
        import torch.nn as nn
        topk = dispatch_output.topk_output
        ids = topk.topk_ids
        used = torch.unique(ids[ids >= 0])
        at = torch.full((int(ids.max()) + 1 if ids.numel() else 1,), -1, dtype=ids.dtype, device=ids.device)
        at[used] = torch.arange(len(used), dtype=ids.dtype, device=ids.device)
        dense = {name: torch.cat([expand(b, used) for b in blocks], dim=-2) for name, blocks in self.weights.items()}
        kept = {name: layer._parameters[name] for name in dense}
        try:
            for name, w in dense.items():
                layer._parameters[name] = nn.Parameter(w, requires_grad=False)
            renumbered = topk._replace(topk_ids=torch.where(ids >= 0, at[ids.clamp(min=0)], ids))
            return self.method.apply(layer, dispatch_output._replace(topk_output=renumbered))
        finally:
            layer._parameters.update(kept)

    def process_weights_after_loading(self, layer) -> None:
        pass

    def nbytes(self) -> int:
        return sum(_nbytes(blocks) for blocks in self.weights.values())
