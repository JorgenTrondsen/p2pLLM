"""One request at a time through SGLang's pipeline parallelism: ms per token and per pass, messages per stage.

    bench/pp/netns.sh D python bench/pp/bench.py --model M --pp 4 --comm fused --out results.jsonl [--draft mtp]
        [--plan plan.json [--calib calib.pt] [--spread 0 [--split 4] [--replicas dir]]] [--partition 1,1,2,4] [--gsm8k n]

--plan installs a plan (without --calib it must keep every weight); --spread and --split place it on the ranks
(p2p.route), --partition sets the layers per rank otherwise. ms per token = (time for 96 new tokens - time for 32) / 64,
median of --repeats; with drafting, ms per pass likewise from SGLang's verify-pass counts. The prompt is 32 random
tokens, or the first n GSM8K questions, whose drafts are accepted as real text's are. Appends one JSON row to --out.
"""
import argparse
import json
import os
import random
import statistics
import tempfile
import time


def arguments(p: argparse.ArgumentParser) -> None:
    """The deployment's options (bench/pp/overhead.py takes them too)."""
    p.add_argument("--model", required=True)
    p.add_argument("--name", default="", help="the model's name in the results (default: the path's last part)")
    p.add_argument("--pp", type=int, default=4)
    p.add_argument("--comm", default="stock")
    p.add_argument("--draft", default="")
    p.add_argument("--draft-k", type=int, default=16)
    p.add_argument("--draft-topk", type=int, default=1, help="candidates kept per draft step: above 1, a tree")
    p.add_argument("--draft-tokens", type=int, default=0, help="tokens a tree's pass verifies (default k + 1)")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-total-tokens", type=int, default=8192)
    p.add_argument("--mem-fraction", type=float, default=0.85)
    p.add_argument("--plan", default="")
    p.add_argument("--calib", default="")
    p.add_argument("--spread", default="")
    p.add_argument("--split", default="")
    p.add_argument("--partition", default="")
    p.add_argument("--replicas", default="", help="replicas prepared ahead of time (p2p replicas), for --spread")
    p.add_argument("--replica-form", default="compact")
    p.add_argument("--batch", type=int, default=4, help="SGLang's max running requests (drafting on one GPU: 1, as its "
                   "linear-attention state is kept per request and drafted token)")
    p.add_argument("--cuda-graph", action="store_true", help="SGLang's CUDA graphs (no plan: our routing runs eager)")


def engine(a):
    """The deployment that a's options describe, started, with its plan installed."""
    os.environ["P2P_PP_COMM"], os.environ["P2P_PP_TRACE"] = a.comm, "1"
    if a.spread:
        os.environ["P2P_PP_SPREAD"], os.environ["P2P_PP_PLAN"], os.environ["P2P_PP_SPLIT"] = a.spread, a.plan, a.split
        if a.calib:
            os.environ["P2P_PP_CALIB"] = a.calib            # a rank that runs pruned groups whole installs them at start
    if a.replicas:
        os.environ["P2P_PP_REPLICAS"], os.environ["P2P_PP_REPLICA_FORM"] = a.replicas, a.replica_form
    if a.partition:
        os.environ["SGLANG_PP_LAYER_PARTITION"] = a.partition
    if a.draft:
        os.environ["SGLANG_ENABLE_PP_SPEC"] = "1"
    from p2p.engine import Model, speculative
    m = Model(a.model, tp=1, batch=a.batch, pp_size=a.pp, max_total_tokens=a.max_total_tokens, mem_fraction_static=a.mem_fraction,
              skip_server_warmup=True, dtype=a.dtype, disable_cuda_graph=not a.cuda_graph, **speculative(a.model, a.draft, a.draft_k if a.draft else 0, a.draft_topk, a.draft_tokens))
    if a.calib:
        m.rpc("p2p_load_calibration", path=a.calib)
    if a.plan:
        m.rpc("p2p_plan", plan=json.load(open(a.plan)))
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    arguments(p)
    p.add_argument("--gsm8k", type=int, default=0, help="time on the first n GSM8K questions, not a random prompt")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    m = engine(a)
    from p2p.pp import counts as _counts
    if a.gsm8k:
        from p2p import data
        prompts = [data.prompt_ids(m.tok, r["question"]) for r in data.load("gsm8k")[:a.gsm8k]]
    else:
        rng = random.Random(0)
        prompts = [[rng.randrange(1000, 20000) for _ in range(32)]]
    runs = {(i, n): [] for i in range(len(prompts)) for n in (32, 96)}
    m.engine.generate(input_ids=[prompts[0]], sampling_params={"temperature": 0, "max_new_tokens": 8})    # warm up
    with tempfile.TemporaryDirectory() as d:
        for _ in range(a.repeats):
            for (i, n), rs in runs.items():
                m.rpc("p2p_pp_counts")
                t = time.perf_counter()
                o = m.engine.generate(input_ids=[prompts[i]], sampling_params={"temperature": 0, "max_new_tokens": n,
                                                                              "ignore_eos": True})[0]
                seconds = time.perf_counter() - t
                at = f"{d}/{i}_{n}_{len(rs)}"
                os.makedirs(at)
                m.rpc("p2p_pp_counts", path=at)
                counts = _counts(at, a.pp)
                rs.append({"s": seconds, "passes": o["meta_info"].get("spec_verify_ct", 0), "ids": o["output_ids"],
                           "counts": counts})
        head = check = None
        if a.spread and os.environ.get("P2P_PP_STREAM"):    # the pieces the head took, and the rows it waited for
            os.makedirs(f"{d}/held")
            m.rpc("p2p_held", path=f"{d}/held")
            for _ in range(300):            # every rank writes its own
                if all(os.path.exists(f"{d}/held/pp{i}.json") for i in range(a.pp)):
                    break
                time.sleep(0.1)
            held = json.load(open(f"{d}/held/pp{a.pp - 1}.json"))
            head, check = held["pieces"], held.get("check")
    m.close()
    med = {k: statistics.median(r["s"] for r in rs) for k, rs in runs.items()}
    last = {k: rs[-1] for k, rs in runs.items()}
    tokens = (96 - 32) * len(prompts)
    per_token = {rank: {k: (last[0, 96]["counts"][rank].get(k, 0) - c.get(k, 0)) / (96 - 32)
                        for k in set(c) | set(last[0, 96]["counts"][rank])}
                 for rank, c in last[0, 32]["counts"].items() if rank in last[0, 96]["counts"]}
    passes = sum(last[i, 96]["passes"] - last[i, 32]["passes"] for i in range(len(prompts)))
    med = {n: sum(med[i, n] for i in range(len(prompts))) for n in (32, 96)}
    last = {n: last[len(prompts) - 1, n] for n in (32, 96)}
    row = {"model": a.name or os.path.basename(a.model.rstrip("/")), "pp": a.pp, "comm": a.comm, "draft": a.draft,
           "plan": a.plan, "spread": a.spread, "split": a.split, "partition": a.partition,
           "replicas": a.replica_form if a.replicas else "", "cuda_graph": a.cuda_graph,
           "draft_k": a.draft_k if a.draft else 0, "draft_topk": a.draft_topk, "draft_tokens": a.draft_tokens, "delay_ms": float(os.environ.get("P2P_DELAY", "nan")),
           "gsm8k": a.gsm8k, "stream": os.environ.get("P2P_PP_STREAM", ""), "early": os.environ.get("P2P_PP_EARLY", "1"),
           "phases": os.environ.get("P2P_PP_PHASES", ""),
           "head": head, "check": check,
           "mbps": float(os.environ.get("P2P_PP_MBPS", "0") or 0),
           "ms_per_token": 1000 * (med[96] - med[32]) / tokens,
           "ms_per_pass": 1000 * (med[96] - med[32]) / passes if passes else None,
           "tokens_per_pass": tokens / passes if passes else None,
           "seconds": {f"{i}:{n}": [r["s"] for r in rs] for (i, n), rs in runs.items()},
           "per_token": per_token, "ids": last[96]["ids"]}
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in ("comm", "draft", "delay_ms", "ms_per_token", "ms_per_pass", "tokens_per_pass")}))


if __name__ == "__main__":
    main()
