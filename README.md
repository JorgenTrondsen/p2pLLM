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

`--calib-text` is a `.jsonl` or `.parquet` file with a `text` column, used for Wanda's calibration. On a sparse MoE model, use text like your dataset, but not the data you test on, because different topics route to different experts (see [Results](#results)). For GSM8K we used its train split, with each question followed by its worked answer. The scan:

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

`--against` pairs the two runs problem by problem and reports the points gained or lost, with a McNemar p-value. `--max-new` caps each answer, at 1,024 tokens by default; Qwen3.6 needs 2,048, or long answers count as wrong. To measure only KL on a plan, use `p2p kl --plan plan.json --calib calib.pt --out runs/y`.

`--data` is `gsm8k`, or a `.jsonl` or `.parquet` file with a `question` column and an optional `answer` column.

A plan is plain JSON. `groups` lists the layers run together, and `keep` gives the percentage of weights kept per projection, for the replica of each group's second-to-last layer:

```json
{"model": "...", "groups": [[0, 1], [10, 11, 12]],
 "keep": {"0": {"q": 16, "k": 7, "v": 7, "z": 33, "out": 55, "gate": 36, "up": 36, "down": 36},
          "11": {"q": 33, "k": 33, "v": 33, "o": 33, "gate": 33, "up": 33, "down": 33}}}
```

On a sparse MoE model, `keep` also has `egate`, `eup` and `edown` for the routed experts. Each expert is pruned on its own, using only the tokens routed to it.

## GPU memory

By default there is one worker per GPU (`--gpus 0,1,2,3`). `--tp 2` spreads each worker over two GPUs. Replicas stay dense in GPU memory, and their reported size is what storing them sparsely would take, so a whole plan needs more memory than the model alone.

The scan measures one group at a time, but its last step installs the whole plan. If that step runs out of memory, rerun the same command with `--tp 2`: it resumes and only has that step left.

Settings that worked on an 80 GB A100:

| Model | One GPU per worker, for the scan | Two GPUs per worker, for a whole plan |
|---|---|---|
| Qwen3.8-27B | `--batch 24 --max-total-tokens 30000` | `--tp 2 --batch 24 --max-total-tokens 30000` |
| Gemma 4 31B-it | `--batch 8 --mem-fraction 0.95 --max-total-tokens 10000` | `--tp 2 --batch 16 --max-total-tokens 20000` |
| Qwen3.6-35B-A3B, `--max-new 2048` | `--batch 16 --max-total-tokens 40000` | `--tp 2 --batch 64 --max-total-tokens 160000` |

The one-GPU Qwen3.8 settings also fit its whole 32-step plan, with very little to spare, if the GPU is otherwise empty. The one-GPU Gemma settings leave little room, so use an otherwise empty GPU. Qwen3.6 on one GPU also needs the GPU empty to start.

A measurement's time is set by rounds of `--batch` problems, not by the number of problems. Scanning Qwen3.6 on the default 64 problems took 42 minutes with `--tp 2 --batch 64 --max-total-tokens 160000`. With one GPU per worker at `--batch 16`, we estimate about 75 minutes.

## Results

Each plan has half as many steps as the model has layers, with every replica at or below 40% of a layer unless the table says otherwise. Accuracy is on the full GSM8K test set (1,319 problems), with answers up to 1,024 tokens unless the table says otherwise. KL is on the first 32 problems, up to 256 tokens each.

| Model | Plan | Accuracy | Against exact | KL above floor |
|---|---|---|---|---|
| Qwen3.8-27B | Exact | .953 | | |
| | `configs/qwen3.8_w40.json` | .957 | +0.4 points, p .47 | .01028 |
| | Fresh scan, 32 steps | .952 | −0.1 points, p 1.0 | .01010 |
| Gemma 4 31B-it | Exact | .971 | | |
| | Fresh scan, 30 steps | .968 | −0.3 points, p .42 | .01886 |
| Qwen3.6-35B-A3B | Exact | .959 | | |
| | Fresh scan, 20 steps | .939 | −2.0 points, p .001 | .01172 |
| | Same plan, recalibrated on GSM8K's train split | .949 | −1.0 points, p .047 | .01083 |
| | Same groups and calibration, replicas at .50 | .953 | −0.6 points, p .23 | .00698 |
| | Same groups and calibration, experts keep 50% (replicas up to .58) | .957 | −0.2 points, p .71 | .00492 |
| | Exact, answers up to 2,048 tokens | .966 | | |
| | Recalibrated .40 plan, answers up to 2,048 tokens | .959 | −0.7 points, p .16 | |
| | Replicas at .50, answers up to 2,048 tokens | .962 | −0.4 points, p .44 | |
| | Fresh scan with maths calibration, 20 steps, answers up to 2,048 tokens | .962 | −0.4 points, p .33 | .01233 |
| | Exact, on 1,319 held-out GSM8K train problems | .951 | | |
| | Fresh scan, on the same held-out problems | .951 | +0.1 points, p 1.0 | |
| | Scan on 64 problems, two GPUs per worker | .963 | −0.3 points, p .59 | .01106 |
| | Same plan, on the held-out problems | .951 | +0.1 points, p 1.0 | |
| | Plan chosen by a confidence score instead of KL, from that scan | .960 | −0.6 points, p .17 | .01264 |
| | Same plan, on the held-out problems | .953 | +0.2 points, p .69 | |
| | Scan on 128 problems, two GPUs per worker | .960 | −0.6 points, p .17 | .01106 |
| | Same plan, on the held-out problems | .949 | −0.2 points, p .82 | |

We count a plan as passing if it loses no more than 1 point and the loss isn't significant (p above .05). The Qwen3.8 and Gemma plans pass, and the two Qwen3.8 plans share 26 of their 28 groups.

**Qwen3.6 needs calibration text like the task, and room for long answers.** Routed experts are 96% of each of its layers, so budget shifting has almost nothing else to take from. Which experts get used also depends on the topic: across its 40 layers, the correlation between expert traffic on generic web text and on maths text has a median of .10 (range −.10 to .53). Recalibrating on maths text recovered about half of the loss (+1.1 points, p .049).

Its answers are also long. With a 1,024-token limit, more than half of the problems the recalibrated plans lost ran out of room without repeating themselves. At 2,048 tokens, exact rises to .966, and the plans at .40 and .50 both pass. The 2,048-token plan runs also include a fix for experts that calibration text never reaches.

A fresh scan at .40 with maths calibration then picked a different plan: 20 pairs, sharing 2 groups with the first plan, with 12.5 GiB of replicas. It passes on the test set, and on 1,319 GSM8K train problems that the calibration text never read, where it matches exact.

**Compare KL within one model, not across models.** Gemma lost 0.3 points at a higher KL than Qwen3.6 lost 2.0 at. Within Qwen3.6, KL on 128 problems ranks the six plans that share groups in the same order as their accuracy (rank correlation −.89), except where their accuracy gap isn't significant. Moving budget from rarely used experts to busy ones cut KL by 28–36%, but no accuracy gain showed up (−0.4 points against no shift at .50, p .51), so it isn't in the code.

**On 32 problems KL is noisier, so the scan uses 64 by default (`--n`).** The same ranking holds in 45% of resamples of the problems, against 99% with 128 problems. On 256 held-out problems that no scan saw, the plan from a 64-problem scan has KL .01033 above the floor, against .01051 and .01204 for the plans from two 32-problem scans. A 128-problem scan took 82 minutes, twice as long, and picked the same groups as the 64-problem scan, with the same split in 18 of 19. Its KL on the held-out problems is .01030, and on GSM8K the two plans tie (−0.3 points on the test set, p .50). So 64 problems is enough. We also tried a confidence score: the probability a plan takes from the exact model's top token, wherever that model is more than 90% sure. It held in 96% of resamples on 32 problems, but in the 64-problem scan it picked a different plan, sharing 2 of 19 groups, with the same accuracy (2,523 against 2,525 correct of 2,638 test and held-out problems, p .90) and higher KL on 64 test problems that no scan saw (.01226 against .01070). So the scan uses KL alone. Whole-plan KL also came out two to two and a half times the sum of its groups on 32 problems, and 1.3 times on 64, so check any MoE plan with `p2p eval`.

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

**Model families.** Tested on Qwen3, Qwen3.5/3.8, Gemma 4 and Qwen3.6 sparse MoE text models. To support another family, add its projection names to `TABLE` in `p2p/family.py`; it also has to use SGLang's layer boundary API.

A model stops with an error if under 90% of a layer's weights are in the table, because its replica sizes would be wrong. Routed experts must be gated SiLU in SGLang's default `triton` layout, all on one rank group; anything else also stops with an error.

## Tests

```bash
pytest tests/test_search.py tests/test_wanda.py    # CPU only

# Tiny models on a real engine, at TP 1 and 2. Takes about 15 minutes.
CUDA_VISIBLE_DEVICES=0,1 P2P_TEST_TOKENIZER=<tokenizer path> pytest tests -s
```
