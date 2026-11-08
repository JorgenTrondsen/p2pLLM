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
    --tp 2 --batch 16 --mem-fraction 0.75 --max-total-tokens 70000 --out runs/x
```

`--calib-text` is a `.jsonl` or `.parquet` file with a `text` column, used for Wanda's calibration. On a sparse MoE model, use text like your dataset, but not the data you test on, because different topics route to different experts (see [Results](#results)). For GSM8K we used its train split, with each question followed by its worked answer. The scan:

1. Generates exact solutions on the first `--n` problems (64 by default) and records the exact model's hidden states.
2. Collects calibration sums on the generic text.
3. Measures every budget split on sample layers at shallow, mid and deep positions, and keeps the best splits for each layer type.
4. Measures every pair and triple of neighbouring layers with those splits, alone, as KL above the noise floor.
5. Finds the 16 splits into exactly `--steps` steps with the least summed KL (`--candidates`), with every replica within `--replica-size`.
6. Measures each of them whole, and keeps the one with the least real KL. Groups interact, so a plan's real KL can be far from its sum: on Gemma 4 the best sum was .01665, measured .02459.
7. Searches each group's split freely, starting from its best menu split (`--refine-rounds`, 3 by default). Each round tries every projection 5 points lower, gate, up and down each on its own. It then moves budget from the projection whose cut cost least KL for its size to the one whose cut cost most. It takes the best of these if it beats the current split.
8. Measures the refined plan whole, keeps it if its KL is lower, and writes `runs/x/plan.json`.

Results are saved per measurement, so rerunning the same command resumes where it stopped.

Plans are meant to run with speculative decoding (see [below](#speculative-decoding)), so by default every measurement is scored in passes of 16 drafted tokens plus one (`--draft-k 16`), with replicas on their own state inside each pass, as they are when drafting. Each solution drafts itself, through SGLang's n-gram drafting, so any model works without a drafter, and every pass keeps all its tokens. For a plan that will run without drafting, add `--draft-k 0` to score one token at a time. The two kinds of results are saved apart, in `passes<K>_*.jsonl` and `results_*.jsonl`. With drafting, plans scanned in passes are as accurate as plans scanned one token at a time, and the scan is 2.5 to 5 times faster (see [Scanning in passes](#scanning-in-passes)).

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

## Speculative decoding

```bash
p2p eval --model MODEL --plan runs/x/plan.json --calib runs/x/calib.pt --out runs/x/mtp16 --against runs/exact \
    --draft mtp
```

SGLang drafts `--draft-k` tokens in a chain (16 by default), then the model checks them all in one pass. It keeps the drafts up to the first one it disagrees with, plus one token of its own, so answers come out the same as without drafting. `p2p report` adds the tokens kept per pass. A pass goes through every step once, so with one network hop per step:

```
time per token ≈ (steps × hop time + draft-k × time per drafted token) / tokens per pass
```

Inside a pass, a replica can't get the copied layer's updates for the pass's tokens in time. So it starts each pass from the copied layer's exact state, and keeps its own state for the rest of the pass.

`--draft mtp` drafts with the checkpoint's own MTP layer. `--draft mtp:PATH` drafts with a native drafter that ships as its own checkpoint, such as Google's `gemma-4-31B-it-assistant` for Gemma 4 31B-it. The assistant has 470M parameters and keeps no K/V cache of its own. It reads the main model's last hidden state and the K/V cache of its last two layers, so in a deployment it would sit with those layers. `--draft PATH` drafts with a small model of the same vocabulary, but SGLang 0.5.23 only supports drafters with full attention only. That rules out Qwen3.5-family models, such as Qwen3.5-2B, because they have linear-attention layers. With a full-attention drafter it works: Qwen3-32B drafting 4 tokens for itself kept 4.7 to 4.9 tokens per pass. Decoding is greedy. Because our hooks turn off CUDA graphs, drafting runs eagerly, so wall-clock times aren't realistic.

On a model with linear attention, SGLang keeps a state per drafted token for each linear-attention layer. On Qwen3.8 at `--tp 2`, that is 72 MiB per request per drafted token, per GPU: 20 GiB at `--batch 32` drafting 8. W40 fit with `--tp 2 --mem-fraction 0.75 --max-total-tokens 70000`, at `--batch 32`, or `--batch 16` when drafting 16 (65 GB per GPU).

Each model drafted with its own MTP layer, on the full GSM8K test set. Qwen3.6's MTP layer is itself a sparse MoE layer. Each run is paired against the exact model without drafting: .953 for Qwen3.8, and .966 for Qwen3.6 with answers up to 2,048 tokens. The last column gives the tokens per pass from the earlier Transformers simulation:

| Model | Plan | Drafted | Accuracy | Against exact | Tokens per pass | Simulation |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | Exact | 8 | .949 | −0.4 points, p .30 | 6.04 | |
| | `configs/qwen3.8_w40.json` | 4 | .952 | −0.1 points, p 1.0 | 4.29 | 4.28 |
| | | 8 | .950 | −0.3 points, p .64 | 6.01 | 6.01 |
| | | 16 | .952 | −0.1 points, p 1.0 | 7.09 | 7.08 |
| | Fresh scan, 32 steps | 16 | .944 | −0.9 points, p .12 | 6.95 | |
| Qwen3.6-35B-A3B | Exact | 16 | .964 | −0.2 points, p .61 | 7.05 | |
| | Scan on 64 problems, 20 steps | 16 | .962 | −0.4 points, p .42 | 6.89 | |
| Gemma 4 31B-it, with the assistant | Exact | 16 | .970 | −0.1 points, p 1.0 | 7.58 | |
| | Fresh scan, 30 steps | 16 | .962 | −0.9 points, p .012 | 7.43 | |

All pass except the Gemma plan. Drafting costs each plan a little against itself without drafting, because inside a pass the replicas run on their own state:

| Plan | Drafted | Against the same plan without drafting |
|---|---|---|
| Qwen3.8 W40 | 4 / 8 / 16 | −0.5 / −0.7 / −0.5 points, p .29 / .064 / .21 |
| Qwen3.8, fresh scan | 16 | −0.8 points, p .061 |
| Qwen3.6, scan on 64 problems | 16 | −0.1 points, p 1.0 |
| Gemma 4, fresh scan | 16 | −0.6 points, p .039 |

The Gemma plan already lost 0.3 points without drafting, so the extra 0.6 takes it past the pass line. Wiring faults are ruled out: tiny Gemma 4 passes the full-copy tests in verify passes, and so do Qwen3 and Qwen3.5. Its settings were `--tp 2 --batch 32 --mem-fraction 0.72 --max-total-tokens 80000 --max-new 2048`, at 74 GB per GPU. With drafting, the exact model's answers match its own answers without drafting token for token on 59% of problems, because verify passes round differently from one-token steps.

These plans came from scans that scored one token at a time.

### Scanning in passes

Each model was scanned twice on the same problems, calibration and reference: one token at a time (`--draft-k 0`) and in passes of 16 drafted tokens plus one (`--draft-k 16`). Both plans were then scored on GSM8K train problems that neither scan saw, in both modes. KL above the floor:

| Model | Held-out problems | Plan from | One token at a time | In passes of 17 |
|---|---|---|---|---|
| Gemma 4 31B-it, 30 steps | 64 | One-token scan | **.01820** | .02650 |
| | | Scan in passes | .02074 | **.02274** |
| Qwen3.8-27B, 32 steps | 128 | One-token scan | **.01167** | **.01285** |
| | | Scan in passes | .01205 | .01319 |
| Qwen3.6-35B-A3B, 20 steps | 256 | One-token scan | **.01000** | .01014 |
| | | Scan in passes | .01008 | **.01007** |

GSM8K, drafting with each model's own drafter, against the exact model without drafting:

| Model | Plan from | Without drafting | Drafting 16 |
|---|---|---|---|
| Gemma 4 31B-it, exact .971 | One-token scan | .968, −0.3 points, p .42 | .962, −0.9 points, p .012 (fails) |
| | Scan in passes | .967, −0.5 points, p .18 | .966, −0.5 points, p .12 |
| Qwen3.8-27B, exact .953 | One-token scan | .952, −0.1 points, p 1.0 | .944, −0.9 points, p .12 |
| | Scan in passes | .948, −0.5 points, p .46 | .944, −0.9 points, p .13 |
| Qwen3.6-35B-A3B, exact .966 | One-token scan | .963, −0.3 points, p .59 | .962, −0.4 points, p .42 |
| | Scan in passes | .962, −0.4 points, p .36 | .959, −0.7 points, p .14 |

On accuracy with drafting, the two scans tie: over the three models, the plans scanned in passes got 1 more problem right out of 3,957. On KL, scanning in passes helps where drafting hurts. On Gemma, passes raise the one-token plan's KL by ×1.46. The plan scanned in passes has 14% lower KL in passes, and it passes with drafting, where the other plan fails. Head to head, with drafting, it gains 0.4 points (p .33). On Qwen3.8 and Qwen3.6, passes cost only ×1.10 and ×1.01, so the two scans pick nearly the same plan:

- **Qwen3.8:** the plans share 20 of 28 groups.
- **Qwen3.6:** the plans have the same 19 groups, and 4 of them split their budget the other way between the dense weights and the routed experts.

On both Qwen models the two plans tie: within 3% on KL, and with drafting 0.0 and −0.3 points head to head. Without drafting, the plans scanned in passes are slightly behind on all three models: 3% to 14% higher KL scored one token at a time, and 0.1 to 0.4 points lower on GSM8K. That is within noise, but it goes the same way each time, so scan with `--draft-k 0` for a plan that will run without drafting.

The scan in passes is also faster, because one step scores 17 tokens of each problem. It took 17 minutes on Gemma, against an hour and a half with the same settings, and 17 minutes on Qwen3.6, against 42 at twice the batch. Qwen3.8 took 18.5 minutes on two workers of two GPUs, against 2 hours 40 one token at a time on four single GPUs. Every pass keeps all 17 of its tokens, while drafting keeps about 7. So the scan gives late positions in a pass more weight than drafting does.

### Whole plans and the free search

The plans above came from the menu only: steps 1 to 5, with only the plan with the least summed KL measured whole. Steps 6 to 8 were then run on the same menu measurements, and each plan was scored in passes of 17 on the same held-out problems as above. Gemma and Qwen3.8 were scanned again from scratch on 64 problems. Qwen3.6 had already been scanned on 64. KL above the floor:

| Plan | Gemma 4 31B-it, 30 steps | Qwen3.8-27B, 32 steps | Qwen3.6-35B-A3B, 20 steps |
|---|---|---|---|
| Menu only, scanned on 32 problems (64 for Qwen3.6) | .02274 | .01319 | **.01007** |
| Best of 16 measured whole | .02152 | .01287 | **.01007** |
| After the free search | .01975 | **.01266** | .01008 |
| Best of 16 measured whole, scanned on 64 problems | .02190 | .01327 | |
| After the free search, scanned on 64 problems (the default) | **.01881** | .01282 | |

Measuring 16 plans whole adds little on problems the scan didn't see. The gain comes from the free search: 17% lower KL than the menu plan on Gemma, 3% to 4% on Qwen3.8, and none on Qwen3.6. On 32 problems the free search fits noise: its gain on the scan's own problems was about twice its gain on unseen ones. On 64 the two match. On Gemma, KL fell 16% on the scan's own problems and 14% on unseen ones.

GSM8K, drafting 16 with each model's own drafter, against the exact model without drafting. Gemma is scored on the test set and the 1,319 held-out problems together, where the exact model gets .964:

| Model | Plan | Accuracy | Against exact | Against the menu plan |
|---|---|---|---|---|
| Gemma 4 31B-it | Menu only | .959 | −0.49 points, p .029 (fails) | |
| | After the free search, 32 problems | .956 | −0.80 points, p .001 (fails) | −0.30 points, p .20 |
| | After the free search, 64 problems | .961 | −0.34 points, p .093 | +0.15 points, p .59 |
| Qwen3.8-27B | Menu only | .944 | −0.9 points, p .13 | |
| | After the free search, 32 problems | .948 | −0.5 points, p .37 | +0.4 points, p .57 |
| | After the free search, 64 problems | .949 | −0.4 points, p .52 | +0.5 points, p .39 |
| Qwen3.6-35B-A3B | Menu only | .959 | −0.7 points, p .14 | |
| | After the free search | .961 | −0.5 points, p .27 | +0.2 points, p .79 |

All the plans tie with their menu plan. The 64-problem Gemma plan is the only Gemma plan with drafting that passes over 2,638 problems: .967 on the test set (−0.4 points, p .27) and .955 on the held-out set (−0.3 points, p .34). It also beats the 32-problem free search by 0.45 points (p .043). Tokens per pass stay within 0.05 of the menu plan's on every model. So the scan now uses 64 problems, 16 whole plans and 3 rounds of the free search by default. Scanning takes about twice as long as on 32 problems (see [GPU memory](#gpu-memory)).

## GPU memory

By default there is one worker per GPU (`--gpus 0,1,2,3`). `--tp 2` spreads each worker over two GPUs. Replicas stay dense in GPU memory, and their reported size is what storing them sparsely would take, so a whole plan needs more memory than the model alone.

The scan measures one group at a time, but steps 6 and 8 install whole plans. If they run out of memory, rerun the same command with `--tp 2`: it resumes from there, and the free search between them runs at `--tp 2` too.

A scan in passes holds 17 tokens of each problem per step. On a model with linear attention, SGLang keeps a state per token for each linear-attention layer (see [Speculative decoding](#speculative-decoding)). On one GPU, Qwen3.8 would need 2.4 GiB per request for these states, so its scans in passes need `--tp 2`.

Settings that worked on an 80 GB A100:

| Model | Scan in passes (the default) | Scan one token at a time (`--draft-k 0`) | Whole plan without drafting |
|---|---|---|---|
| Qwen3.8-27B | `--tp 2 --batch 16 --mem-fraction 0.75 --max-total-tokens 70000` | `--batch 24 --max-total-tokens 30000` | `--tp 2 --batch 24 --max-total-tokens 30000` |
| Gemma 4 31B-it | `--tp 2 --batch 16 --max-total-tokens 20000` | `--batch 8 --mem-fraction 0.95 --max-total-tokens 10000` | `--tp 2 --batch 16 --max-total-tokens 20000` |
| Qwen3.6-35B-A3B, `--max-new 2048` | `--tp 2 --batch 32 --mem-fraction 0.72 --max-total-tokens 80000` | `--batch 16 --max-total-tokens 40000` | `--tp 2 --batch 64 --max-total-tokens 160000` |

The two-GPU scan settings also fit each model's whole plans. The one-GPU Qwen3.8 settings also fit its whole 32-step plan, with very little to spare, if the GPU is otherwise empty. The one-GPU Gemma settings leave little room, so use an otherwise empty GPU, and rerun with the whole-plan settings when the scan reaches the whole plans. Qwen3.6 on one GPU also needs the GPU empty to start. For settings with drafting, see [Speculative decoding](#speculative-decoding).

A measurement's time is set by how many rounds of `--batch` problems it takes, and by the steps in each round. The default scan, on 64 problems in passes with whole plans and the free search, took 99 minutes from scratch on Qwen3.8 and 111 on Gemma, with the two-GPU settings above on two workers. That was about 990 measurements each, a little over half of them in the free search. Before the whole plans and the free search were added, scanning Qwen3.6 on 64 problems took 17 minutes in passes, at `--tp 2 --batch 32`, and 42 minutes one token at a time, at `--tp 2 --batch 64`. The two new stages added 27 minutes in passes.

## Results

Each plan has half as many steps as the model has layers, with every replica at or below 40% of a layer unless the table says otherwise. Accuracy is on the full GSM8K test set (1,319 problems), with answers up to 1,024 tokens unless the table says otherwise. KL is on the first 32 problems, up to 256 tokens each. These scans scored one token at a time, and so does the KL.

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
