"""A running SGLang engine with our plugin (p2p.plugin), what we ask of it, and one engine per worker."""
from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import tempfile

SETTINGS = dict(disable_cuda_graph=True,       # our hooks need eager execution
                disable_overlap_schedule=True, disable_radix_cache=True, schedule_policy="fcfs", log_level="warning",
                enable_multimodal=False)       # text only: skips the vision tower, about 1 GiB


class Model:
    def __init__(self, path: str, tp: int = 1, batch: int = 32, **engine_args):
        os.environ.setdefault("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", "false")  # GPUs may be shared; pools fit the fullest
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # whole plans nearly fill a GPU
        if engine_args.get("speculative_num_draft_tokens"):   # sliding windows reserve 128 eviction steps per request,
            os.environ.setdefault("SGLANG_SWA_EVICTION_INTERVAL",   # times the drafted tokens: keep it as without drafting
                                  str(max(1, 128 // engine_args["speculative_num_draft_tokens"])))
        import sglang as sgl
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.engine = sgl.Engine(model_path=path, tp_size=tp, max_running_requests=batch, **{**SETTINGS, **engine_args})
        self.rpc("p2p_start")

    def rpc(self, method: str, **kwargs) -> None:
        self.engine.collective_rpc(method, **kwargs)

    def generate(self, prompts: list, max_new: int) -> list:
        """Greedy generation under the installed plan: Engine outputs ("output_ids", "text", "meta_info"), in order."""
        return self.engine.generate(input_ids=prompts, sampling_params={"temperature": 0, "max_new_tokens": max_new})

    def force(self, items: list) -> list:
        """Run {"id", "prompt", "sol"} items teacher forced on their solutions, under the installed plan."""
        params = [{"temperature": 0, "max_new_tokens": len(r["sol"]), "ignore_eos": True, "custom_params": {"force": r["sol"]}}
                  for r in items]
        return self.engine.generate(input_ids=[r["prompt"] for r in items], sampling_params=params,
                                    rid=[str(r["id"]) for r in items])

    def calibrate(self, windows: list, path: str) -> None:
        """Collect Wanda's calibration sums over token windows -> path."""
        self.rpc("p2p_calibrate")
        self.generate(windows, 1)
        self.rpc("p2p_calibrate", path=path)

    def record_reference(self, items: list, path: str) -> None:
        """The exact model's final hidden states on the items' solutions -> path."""
        self.rpc("p2p_plan")
        self.rpc("p2p_record", on=True)
        self.force(items)
        self.rpc("p2p_record", on=False)
        self.rpc("p2p_save_reference", path=path)

    def measure(self, items: list, plan: "dict | None") -> dict:
        """{"kl", "top1", "tokens"} of a plan against the reference. The exact model (None) runs in other batches than the
        reference did: its KL is the floor, the numerical noise that every plan's KL holds too (rerun in the same batches
        it would be exactly 0)."""
        self.rpc("p2p_plan", plan=plan)
        for part in [items] if plan else [items[::2], items[1::2]]:
            self.force(part)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "result.json")
            self.rpc("p2p_result", path=path)
            return json.load(open(path))

    def close(self) -> None:
        self.engine.shutdown()


def solution(o: dict) -> list:
    """An Engine output's generated tokens, without the stop token that SGLang includes."""
    return o["output_ids"][:-1] if o["meta_info"]["finish_reason"]["type"] == "stop" else o["output_ids"]


def speculative(model: str, draft: str, k: int, topk: int = 1, tokens: int = 0) -> dict:
    """SGLang's speculative decoding, drafting k tokens deep a pass: with the checkpoint's own MTP layer ("mtp"), a
    native drafter shipped as its own checkpoint ("mtp:PATH", such as Gemma 4's assistant), or a small model of the same
    vocabulary (its path). A chain by default; with topk > 1 a tree, each step keeping topk candidates, of which the
    pass verifies the tokens likeliest (tokens, root included). "forced": forced requests draft their own solution (plugin._drafts), for scoring in passes."""
    if not draft or not k:
        return {}
    if draft == "forced":
        return dict(speculative_algorithm="NGRAM", speculative_num_draft_tokens=k + 1, speculative_ngram_max_bfs_breadth=1)
    native = draft == "mtp" or draft.startswith("mtp:")
    return dict(speculative_algorithm="EAGLE" if native else "STANDALONE",
                speculative_draft_model_path=model if draft == "mtp" else draft.removeprefix("mtp:"),
                speculative_num_steps=k, speculative_eagle_topk=topk, speculative_num_draft_tokens=tokens or k + 1)


def open_model(a):
    """A model that shuts down on any error: its scheduler processes would otherwise keep the run alive, doing nothing."""
    extra = {"max_total_tokens": a.max_total_tokens} if a.max_total_tokens else {}
    extra |= speculative(a.model, getattr(a, "draft", ""), getattr(a, "draft_k", 0), getattr(a, "draft_topk", 1),
                         getattr(a, "draft_tokens", 0))
    return contextlib.closing(Model(a.model, tp=a.tp, batch=a.batch, mem_fraction_static=a.mem_fraction, **extra))


def _on_gpus(gpus: list, fn, job: tuple) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    fn(*job)


def parallel(a, fn, jobs: list) -> None:
    """Run fn(*job) in one process per a.tp GPUs of a.gpus: jobs[k] owns the k-th set."""
    assert len(jobs) * a.tp <= len(a.gpus), f"{len(jobs)} workers of {a.tp} GPUs on {a.gpus}"
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_on_gpus, args=(a.gpus[k * a.tp: (k + 1) * a.tp], fn, job)) for k, job in enumerate(jobs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode for p in procs):
        raise SystemExit(f"{fn.__name__}: a worker failed")
