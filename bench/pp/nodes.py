"""Simulated nodes: one SGLang launch per pipeline stage, as on separate machines, with SGLang's own multi-node flags.

    bench/pp/netns.sh D python bench/pp/nodes.py --model M --nodes 4 [--gpus 4] [--tokens 32] [--out runs/nodes/x]
        [--draft mtp [--draft-k 16 --draft-topk 4 --draft-tokens 64]]

Node r runs launch_server (text only) with --pp-size N --nnodes N --node-rank r on GPU r mod --gpus; P2P_* variables
pass through to every node. Node 0 serves HTTP: this sends one greedy request, prints its tokens and timing, and stops
the nodes. --draft drafts a tree 16 deep, 4 candidates a step, 64 tokens a pass (SGLang's tree kernels need bfloat16 on
models with linear attention). With P2P_PP_SPREAD it also streams passes (P2P_PP_STREAM=1,2,4) and sends a tree of 48
tokens or more in phases (P2P_PP_PHASES=6:24), unless those are set.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import requests

SETTINGS = ["--disable-cuda-graph", "--disable-overlap-schedule", "--disable-radix-cache", "--schedule-policy", "fcfs",
            "--skip-server-warmup"]
# SGLang's own entry point (python -m sglang.launch_server), text only: the CLI can turn multimodal on but not off.
NODE = """import os, sys
from sglang.launch_server import run_server
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
load_plugins()
args = prepare_server_args(sys.argv[1:])
args.enable_multimodal = False
try:
    run_server(args)
finally:
    kill_process_tree(os.getpid(), include_parent=False)
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--port", type=int, default=31000, help="node 0's HTTP port")
    p.add_argument("--dist-port", type=int, default=31500)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-total-tokens", type=int, default=2048)
    p.add_argument("--mem-fraction", type=float, default=0.9)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--draft", default="", help="speculative decoding, as p2p eval's --draft (mtp: the model's MTP layer)")
    p.add_argument("--draft-k", type=int, default=16, help="draft depth")
    p.add_argument("--draft-topk", type=int, default=4, help="candidates kept a step (1: a chain)")
    p.add_argument("--draft-tokens", type=int, default=64, help="tokens a pass verifies (0 with --draft-topk 1: k + 1)")
    p.add_argument("--timeout", type=int, default=1800, help="seconds to wait for node 0 to serve")
    p.add_argument("--out", default="runs/nodes/last")
    a, extra = p.parse_known_args()
    os.makedirs(a.out, exist_ok=True)
    if os.environ.get("P2P_PP_SPREAD") and a.draft:
        os.environ.setdefault("P2P_PP_STREAM", "1,2,4")
        if os.environ["P2P_PP_STREAM"] not in ("", "0") and a.draft_topk > 1 and a.draft_tokens >= 48 and a.draft_k >= 8:
            os.environ.setdefault("P2P_PP_PHASES", "6:24")
    from p2p.engine import speculative
    draft = [f"--{k.replace('_', '-')}={v}" for k, v in speculative(a.model, a.draft, a.draft_k if a.draft else 0,
                                                                  a.draft_topk, a.draft_tokens).items()]
    procs = []
    try:
        for r in range(a.nodes):
            cmd = [sys.executable, "-c", NODE, "--model-path", a.model, "--pp-size", str(a.nodes),
                   "--nnodes", str(a.nodes), "--node-rank", str(r), "--dist-init-addr", f"127.0.0.1:{a.dist_port}",
                   "--port", str(a.port + 20 * r), "--dtype", a.dtype, "--max-running-requests", "1",   # ports of its own
                   "--max-total-tokens", str(a.max_total_tokens), "--mem-fraction-static", str(a.mem_fraction),
                   *SETTINGS, *draft, *extra]
            env = os.environ | {"CUDA_VISIBLE_DEVICES": str(r % a.gpus), "NCCL_MULTI_RANK_GPU_ENABLE": "1",
                                "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false"} | (
                {"SGLANG_ENABLE_PP_SPEC": "1"} if a.draft else {})
            log = open(os.path.join(a.out, f"node{r}.log"), "w")
            procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        t = time.time()
        url = f"http://127.0.0.1:{a.port}"
        while True:
            dead = [r for r, q in enumerate(procs) if q.poll() is not None]
            if dead:
                raise SystemExit(f"node(s) {dead} exited: see {a.out}/node{dead[0]}.log")
            try:
                if requests.get(f"{url}/health", timeout=5).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.time() - t > a.timeout:
                raise SystemExit(f"node 0 did not serve within {a.timeout} s")
            time.sleep(5)
        started = time.time() - t
        prompt = [1000 + i for i in range(32)]
        runs = []
        for n in (8, a.tokens):
            t = time.time()
            o = requests.post(f"{url}/generate", json={"input_ids": prompt, "sampling_params": {
                "temperature": 0, "max_new_tokens": n, "ignore_eos": True}}, timeout=a.timeout).json()
            runs.append({"tokens": n, "seconds": time.time() - t, "ids": o.get("output_ids", o),
                         "passes": o.get("meta_info", {}).get("spec_verify_ct") if isinstance(o, dict) else None})
        row = {"nodes": a.nodes, "started_s": started, "runs": runs, "draft": draft,
               "env": {k: v for k, v in os.environ.items() if k.startswith("P2P_")}}
        json.dump(row, open(os.path.join(a.out, "result.json"), "w"), indent=1)
        print(json.dumps({"nodes": a.nodes, "started_s": round(started), **{f"{r['tokens']} tokens s": round(r["seconds"], 2)
                                                                           for r in runs}, "ids": str(runs[-1]["ids"])[:80]}))
    finally:
        for q in procs:
            if q.poll() is None:
                try:
                    os.killpg(q.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(5)
        for q in procs:
            if q.poll() is None:
                try:
                    os.killpg(q.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    main()
