"""Datasets: `gsm8k`, or a .jsonl or .parquet file with a `question` column and optionally `answer`."""
from __future__ import annotations

import glob
import json
import math
import random
import re

GSM8K_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def lines(pattern: str) -> list:
    """The rows of every JSONL file matching a glob."""
    return [json.loads(line) for p in sorted(glob.glob(pattern)) for line in open(p)]


def _rows(path: str) -> list:
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in open(path)]
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def load(spec: str) -> list:
    """[{"id", "question", "gold"}]; gold is None without an answer."""
    if spec == "gsm8k":
        from datasets import load_dataset
        rows = [{**r, "question": r["question"] + GSM8K_SUFFIX} for r in load_dataset("openai/gsm8k", "main", split="test")]
    else:
        rows = _rows(spec)
    gold = lambda a: None if a is None else str(a).split("####")[-1].strip().replace(",", "")
    return [{"id": i, "question": r["question"], "gold": gold(r.get("answer"))} for i, r in enumerate(rows)]


def prompt_ids(tok, question: str) -> list:
    text = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def extract(text: str) -> str:
    """The number in the last \\boxed{...}, else the last number in the text."""
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    nums = re.findall(r"-?\d+(?:\.\d+)?", (boxed[-1] if boxed else text).replace(",", ""))
    return "" if not nums else nums[0] if boxed else nums[-1]


def correct(text: str, gold: str) -> bool:
    """Numeric golds match the extracted number; any other gold must appear in the text."""
    try:
        return math.isclose(float(extract(text)), float(gold), abs_tol=1e-6)
    except ValueError:
        return gold in text if not re.fullmatch(r"-?\d+(\.\d+)?", gold) else False


def windows(path: str, tok, n: int, length: int, seed: int = 0) -> list:
    """`n` windows of `length` tokens from shuffled documents (a `text` column) joined by EOS, for calibration."""
    docs = [r["text"] for r in _rows(path)]
    random.Random(seed).shuffle(docs)
    ids, out = [], []
    for d in docs:
        ids += tok(d, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        while len(ids) >= length and len(out) < n:
            out.append(ids[:length])
            ids = ids[length:]
        if len(out) == n:
            break
    return out
