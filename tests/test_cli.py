"""scan, then eval, end to end through the command line: tiny models and a small JSONL dataset, on the first GPU of
CUDA_VISIBLE_DEVICES. Needs a tokenizer in P2P_TEST_TOKENIZER, as test_sglang does."""
import json
import os

import pytest
from test_sglang import _checkpoint

from p2p import data, search
from p2p.cli import main


@pytest.mark.parametrize("kind", ["qwen3", "gemma4", "qwen3_5_moe"])
def test_scan_then_eval(tmp_path, kind):
    questions, text = tmp_path / "questions.jsonl", tmp_path / "text.jsonl"
    questions.write_text("".join(json.dumps({"question": f"What is {i} + 1?", "answer": str(i + 1)}) + "\n" for i in range(8)))
    text.write_text("".join(json.dumps({"text": "the quick brown fox jumps over the lazy dog " * 40}) + "\n" for _ in range(8)))
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    common = ["--model", _checkpoint(kind), "--data", str(questions), "--gpus", gpu, "--batch", "4",
              "--max-total-tokens", "4096", "--mem-fraction", "0.5"]
    scan, exact, planned = (str(tmp_path / d) for d in ("scan", "exact", "planned"))

    assert main(["scan", *common, "--out", scan, "--steps", "6", "--replica-size", "0.4", "--n", "4", "--max-sol", "16",
                 "--calib-text", str(text), "--calib-n", "4", "--calib-len", "64", "--candidates", "4", "--refine-rounds", "1"]) == 0
    plan = json.load(open(f"{scan}/plan.json"))
    assert 8 - sum(len(g) - 1 for g in plan["groups"]) == 6
    shapes = json.load(open(f"{scan}/shapes.json"))
    assert all(search.size(shapes[i]["proj"], keep) <= 0.4 for i, keep in plan["keep"].items())
    assert kind != "qwen3_5_moe" or all(keep["egate"] < 100 for keep in plan["keep"].values())

    assert main(["eval", *common, "--out", exact, "--max-new", "8"]) == 0
    assert main(["eval", *common, "--out", planned, "--max-new", "8", "--plan", f"{scan}/plan.json", "--calib", f"{scan}/calib.pt",
                 "--against", exact]) == 0
    assert len(data.lines(f"{planned}/shard_*.jsonl")) == 8


def test_eval_with_drafts(tmp_path, capsys):
    """The tiny model drafts for itself."""
    questions, model, out = tmp_path / "questions.jsonl", _checkpoint("qwen3"), str(tmp_path / "spec")
    questions.write_text("".join(json.dumps({"question": f"What is {i} + 1?", "answer": str(i + 1)}) + "\n" for i in range(8)))
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    assert main(["eval", "--model", model, "--data", str(questions), "--gpus", gpu, "--batch", "4", "--max-total-tokens", "4096",
                 "--mem-fraction", "0.5", "--out", out, "--max-new", "8", "--draft", model, "--draft-k", "3"]) == 0
    assert any(r["passes"] for r in data.lines(f"{out}/shard_*.jsonl"))
    assert "tokens per pass" in capsys.readouterr().out
