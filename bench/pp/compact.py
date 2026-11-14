"""A prepared replica's projections multiplied compactly against the same weights dense, on one GPU: by compact.cu's
tiles (p2p.compact.Tiles, as replicas run).

    python bench/pp/compact.py runs/pp/route5/replicas/layer32.safetensors [--tokens 1,17,32,64] [--out compact.json]

Per projection and for the whole layer, at 1 token (decode) and 17 (a pass drafting 16) by default, or --tokens (a tree's
pass verifies 32 or 64): microseconds, bytes read, and the bandwidth they come to. For routed experts, the time to expand
8 of them (a decoded token's) and all of them.
"""
import argparse
import json

import torch
import torch.nn.functional as F
import triton

from p2p import compact, prepare


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("replica")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--tokens", default="1,17", help="tokens a pass multiplies, comma-separated")
    p.add_argument("--out", default="")
    a = p.parse_args()
    dtype = getattr(torch, a.dtype)
    tensors, meta = prepare.load(a.replica)
    rows, total = [], {}
    for name, info in meta["modules"].items():
        for proj, n, k in info["blocks"]:
            if k == 0:
                continue
            values = tensors[f"{name}:{proj}:values"].cuda().to(dtype)
            words = tensors[f"{name}:{proj}:mask"].cuda()
            if values.dim() == 3:               # routed experts: CompactMoE expands the experts a pass routes to
                for e in (8, values.shape[0]):
                    used = torch.arange(e, device="cuda")
                    us = 1000 * triton.testing.do_bench(lambda: compact.expand(("sparse", values, words, info["cols"]), used))
                    print(f"{proj:5s} {values.shape[0]} experts x {n} x {info['cols']} kept {k / info['cols']:.0%}: expanding "
                          f"{e} takes {us:.0f} us", flush=True)
                    rows.append({"projection": proj, "experts": e, "rows": n, "cols": info["cols"], "us_expand": us})
                continue
            dense = compact.unpack(values, words, info["cols"])
            tiles = compact.Tiles([("sparse", values, words, info["cols"])])
            for t in (int(n) for n in a.tokens.split(",")):
                x = torch.randn(t, info["cols"], device="cuda", dtype=dtype)
                us_dense = 1000 * triton.testing.do_bench(lambda: F.linear(x, dense))
                us_cuda = 1000 * triton.testing.do_bench(lambda: tiles(x))
                b_dense, b_compact = dense.numel() * dense.element_size(), values.numel() * values.element_size() + words.numel() * 4
                row = {"projection": proj, "rows": n, "cols": info["cols"], "kept": k / info["cols"], "tokens": t,
                       "us_dense": us_dense, "us_cuda": us_cuda, "bytes_dense": b_dense,
                       "bytes_compact": b_compact, "bytes_tiles": tiles.nbytes()}
                rows.append(row)
                s = total.setdefault(t, {"us_dense": 0, "us_cuda": 0, "bytes_dense": 0, "bytes_compact": 0})
                for key in s:
                    s[key] += row[key]
                print(f"{proj:5s} {n:6d} x {info['cols']:6d} kept {k / info['cols']:.0%} T={t:2d}: dense {us_dense:6.0f} us "
                      f"({b_dense / us_dense / 1e3:5.0f} GB/s), cuda {us_cuda:6.0f} us ({b_compact / us_cuda / 1e3:5.0f} GB/s, "
                      f"{us_cuda / us_dense:.2f}x); {b_compact / b_dense:.2f}x the bytes", flush=True)
    for t, s in total.items():
        print(f"layer {meta['layer']}, T={t}: dense {s['us_dense']:.0f} us, cuda {s['us_cuda']:.0f} us "
              f"({s['us_cuda'] / s['us_dense']:.2f}x), bytes {s['bytes_compact'] / s['bytes_dense']:.2f}x")
    if a.out:
        json.dump({"layer": meta["layer"], "rows": rows, "total": total}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
