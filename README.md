# p2pLLM

Given a number of **steps** and a **replica size**, find which layers of a model to run in parallel, and how to shrink each replica, so the model loses as little accuracy as possible.

When each layer sits on a different machine, every layer costs a network hop. Running a group of neighbouring layers at once costs one hop for the whole group, so fewer steps means less latency. Every member of a group reads the group's input `h`, and the last member's machine also holds a shrunk copy (a *replica*) of the member before it, which gives that member a better input:

```
h' = l_0(h) + Σ_{k>0} (l_k(x_k) − x_k)      x_k = replica(h) for the last member, h for the others
```

Replicas are shrunk with Wanda pruning, and budget shifting decides what percentage of weights each projection keeps. The project is a thin plugin on the [SGLang](https://github.com/sgl-project/sglang) wheel, which provides the engine, batching, caches and kernels.

## Install

Needs Linux with glibc 2.34+, Python 3.11+, CUDA and `nvcc`. SGLang compiles its kernels on first use.

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install --prerelease=allow -e . pytest
source env.sh    # activates .venv and sets CUDA_HOME, PATH and HF_HOME; edit it if CUDA isn't in /usr/local/cuda
```

## Find a plan

```bash
p2p scan --model MODEL --data gsm8k --steps 32 --replica-size 0.40 \
    --calib-text text.parquet --calib-n 128 --calib-len 2048 \
    --batch 24 --max-total-tokens 30000 --out runs/x
```

`--calib-text` is a `.jsonl` or `.parquet` file with a `text` column, used for Wanda's calibration. The scan:

1. Generates exact solutions on the first `--n` problems and records the exact model's hidden states.
2. Collects calibration sums on the generic text.
3. Measures every budget split on sample layers at shallow, mid and deep positions, and keeps the best splits for each layer type.
4. Measures every pair and triple of neighbouring layers with those splits, alone, as KL above the noise floor.
5. Picks the split into exactly `--steps` steps with the least summed KL, with every replica within `--replica-size`.
6. Measures the whole plan's real KL and writes `runs/x/plan.json`.

Results are saved per measurement, so rerunning the same command resumes where it stopped.

## Evaluate a plan

```bash
p2p eval --model MODEL --out runs/exact          # the exact model, as the reference
p2p eval --model MODEL --plan runs/x/plan.json --calib runs/x/calib.pt --out runs/x/eval --against runs/exact
p2p report runs/x/eval --against runs/exact      # reprint the comparison
```

`--against` pairs the two runs problem by problem and reports the points gained or lost, with a McNemar p-value. To measure only KL on a plan, use `p2p kl --plan plan.json --calib calib.pt --out runs/y`.

`--data` is `gsm8k`, or a `.jsonl` or `.parquet` file with a `question` column and an optional `answer` column.

A plan is plain JSON. `groups` lists the layers run together, and `keep` gives the percentage of weights kept per projection, for the replica of each group's second-to-last layer:

```json
{"model": "...", "groups": [[0, 1], [10, 11, 12]],
 "keep": {"0": {"q": 16, "k": 7, "v": 7, "z": 33, "out": 55, "gate": 36, "up": 36, "down": 36},
          "11": {"q": 33, "k": 33, "v": 33, "o": 33, "gate": 33, "up": 33, "down": 33}}}
```

## GPU memory

By default there is one worker per GPU (`--gpus 0,1,2,3`). `--tp 2` spreads each worker over two GPUs. Replicas stay dense in GPU memory, and their reported size is what storing them sparsely would take, so a whole plan needs more memory than the model alone. Measured on Qwen3.8-27B on an 80 GB A100:

- **The scan** measures one group at a time, but its last step installs the whole plan.
- **`--tp 1`:** a whole 32-step plan fits at `--batch 24 --max-total-tokens 30000`, with very little to spare. It needs the GPU to itself.
- **`--tp 2`:** this is the safer choice on shared GPUs, and the scan can be rerun with it to finish that last step.

## Results on Qwen3.8-27B

All at 32 steps (64 layers) with every replica at or below 40% of a layer, on the full GSM8K test set (1,319 problems):

| Plan | Accuracy | Against exact | KL above floor |
|---|---|---|---|
| Exact | .953 | | |
| `configs/qwen3.8_w40.json` | .957 | +0.4 points, p .47 | .01028 |
| Fresh scan | .952 | −0.1 points, p 1.0 | .01010 |

KL is on the first 32 problems, up to 256 tokens each. The two plans share 26 of their 28 groups, and their accuracy gap isn't significant (p .31).

## Layout

| File | Job |
|---|---|
| `p2p/search.py` | Replica sizes, budget splits and the search for the best plan (no torch, no SGLang) |
| `p2p/family.py` | Finds a model's layers and projections |
| `p2p/wanda.py` | Calibration sums and pruning |
| `p2p/replica.py` | Builds replicas and runs a group inside SGLang's forward |
| `p2p/plugin.py` | SGLang hooks and the calls into each scheduler process |
| `p2p/engine.py` | Starts engines and runs generation, forced runs and calibration |
| `p2p/scan.py`, `p2p/evaluate.py`, `p2p/cli.py` | The commands |

**Model families.** Tested on Qwen3 and Qwen3.5/3.8 text models. To support another family, add its projection names to `TABLE` in `p2p/family.py`; it also has to use SGLang's layer boundary API. A model whose MLP names aren't in the table stops with an error.

## Tests

```bash
pytest tests/test_search.py tests/test_wanda.py    # CPU only

# Tiny models on a real engine, at TP 1 and 2. Takes about 5 minutes.
CUDA_VISIBLE_DEVICES=0,1 P2P_TEST_TOKENIZER=<tokenizer path> pytest tests -s
```
