# p2pLLM

Given a number of **steps** and a **replica size**, find which layers of a model to run in parallel, and how to shrink each replica, so the model loses as little accuracy as possible. Then serve the model with one layer per machine, where a token costs one network hop per step instead of one per layer.

When each layer sits on a different machine, every layer costs a network hop. Running a group of neighbouring layers at once costs one hop for the whole group, so fewer steps means less latency. Every member of a group reads the group's input `h`, and the last member's machine also holds a shrunk copy (a *replica*) of the member before it, which gives that member a better input:

```
h' = l_0(h) + Σ_{k>0} (l_k(x_k) − x_k)      x_k = replica(h) for the last member, h for the others
```

Replicas are shrunk with Wanda pruning, and budget shifting decides what percentage of weights each projection keeps. The project is a thin plugin on the [SGLang](https://github.com/sgl-project/sglang) wheel (0.5.23). SGLang provides the engine, batching, caches, kernels, speculative decoding and pipeline parallelism; the plugin adds the plan search, the replicas, and the routing that puts a plan's layers on separate machines.

The work has three parts:

1. **Find a plan** (`p2p scan`): which layers to group, and how to prune each replica.
2. **Check it** (`p2p eval`): accuracy and KL against the exact model, with or without speculative decoding.
3. **Deploy it** (SGLang's pipeline parallelism plus `P2P_PP_*` settings): one layer per node, each step one hop.

## Install

Needs Linux with glibc 2.34+, Python 3.11+, CUDA and `nvcc`. SGLang compiles its kernels on first use, and so does the replicas' kernel.

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

`--calib-text` is a `.jsonl` or `.parquet` file with a `text` column, used for Wanda's calibration. On a sparse MoE model, use text like your dataset, but not the data you test on, because different topics route to different experts. The scan:

1. Generates exact solutions on the first `--n` problems (64 by default) and records the exact model's hidden states.
2. Collects calibration sums on the generic text.
3. Measures every budget split on sample layers at shallow, mid and deep positions, and keeps the best splits for each layer type.
4. Measures every pair and triple of neighbouring layers with those splits, alone, as KL above the noise floor.
5. Finds the 16 splits into exactly `--steps` steps with the least summed KL (`--candidates`), with every replica within `--replica-size`.
6. Measures each of them whole, and keeps the one with the least real KL. Groups interact, so a plan's real KL can be far from its sum.
7. Searches each group's split freely, starting from its best menu split (`--refine-rounds`, 3 by default). Each round tries every projection 5 points lower, and moves budget from the projection whose cut cost least KL for its size to the one whose cut cost most.
8. Measures the refined plan whole, keeps it if its KL is lower, and writes `runs/x/plan.json`.

Results are saved per measurement, so rerunning the same command resumes where it stopped.

- **Scored in passes.** Plans are meant to run with speculative decoding, so by default every measurement is scored in passes of 16 drafted tokens plus one (`--draft-k 16`), with replicas on their own state inside each pass, as they are when drafting. Each solution drafts itself, so any model works without a drafter. For a plan that will run without drafting, add `--draft-k 0` to score one token at a time.
- **GPU memory.** There is one worker per GPU by default (`--gpus 0,1,2,3`); `--tp 2` spreads each worker over two GPUs. Replicas stay dense in GPU memory during a scan, so a whole plan needs more memory than the model alone. If steps 6 and 8 run out of memory, rerun the same command with `--tp 2`: it resumes from there.

## Evaluate a plan

```bash
p2p eval --model MODEL --out runs/exact          # the exact model, as the reference
p2p eval --model MODEL --plan runs/x/plan.json --calib runs/x/calib.pt --out runs/x/eval --against runs/exact
p2p report runs/x/eval --against runs/exact      # reprint the comparison
```

`--against` pairs the two runs problem by problem and reports the points gained or lost, with a McNemar p-value. We count a plan as passing if it loses no more than 1 point and the loss isn't significant (p above .05). `--max-new` caps each answer, at 1,024 tokens by default. To measure only KL on a plan, use `p2p kl --plan plan.json --calib calib.pt --out runs/y`. Compare KL within one model, not across models.

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
    --draft mtp [--draft-k 16] [--draft-topk 4 --draft-tokens 32]
```

SGLang drafts `--draft-k` tokens in a chain (16 by default), then the model checks them all in one pass. It keeps the drafts up to the first one it disagrees with, plus one token of its own, so answers come out the same as without drafting. A pass goes through every step once, so with one network hop per step:

```
time per token ≈ (steps × hop time + the rest of a pass) / tokens kept per pass
```

- **A tree instead of a chain.** With `--draft-topk` above 1, each draft step keeps that many candidates, and a pass verifies the likeliest `--draft-tokens` of them, root included. The network is most of a pass, so tokens kept per pass are the main lever.
- **Replicas inside a pass.** A replica can't get the copied layer's updates for the pass's tokens in time. So it starts each pass from the copied layer's exact state, and keeps its own state for the rest of the pass.
- **Drafters.** `--draft mtp` drafts with the checkpoint's own MTP layer. `--draft mtp:PATH` drafts with a native drafter that ships as its own checkpoint, such as Google's assistant for Gemma 4. `--draft PATH` drafts with a small model of the same vocabulary, which SGLang 0.5.23 supports only for drafters with full attention.
- **Decoding is greedy,** and our hooks turn off CUDA graphs, so wall-clock times on one machine aren't realistic.

## Deployment

In deployment each step runs on its own machines. This builds on SGLang's pipeline parallelism (PP), where each pipeline stage is a machine. The plugin changes four things, all through SGLang's hook registry and `P2P_PP_*` environment variables, with the same settings on every node.

### How stages talk

Per round, SGLang sends each stage boundary its new requests as pickled objects over gloo, and its tensors as gloo metadata followed by one NCCL message per tensor. A gloo message costs an extra round trip when its receive is posted late, so a hop costs about three one-way delays where one would do.

`P2P_PP_COMM=fused` (`p2p/pp.py`) sends requests over NCCL, and each set of tensors as two NCCL messages: their sizes, then one buffer holding the metadata and every tensor. A hop then costs exactly one one-way delay. It needs one GPU per stage. `stock`, the default, is SGLang's own.

### Steps across stages

`P2P_PP_SPREAD` (`p2p/route.py`) puts each member of a step on a stage of its own. It names the groups to spread by their first layer (`0`, or `0,10`), or `all`, which spreads every step but the last. `P2P_PP_PLAN` names the plan, since the stages' layers depend on its groups.

- **One exact layer per stage.** The last member's stage also holds the layer it copies, but only as its replica, and never runs that layer exact.
- **Steps that aren't spread** sit whole on one stage, as on one GPU. `P2P_PP_SPLIT` cuts such runs onto further stages at the layers it names.
- **The last stage is the head.** It holds the last step, the output head and the input embedding. `pp_size` must equal the number of stages that follow from the plan.

A step's members all read the step's input, so a spread step costs one hop however many members it has. For that, all three of SGLang's streams leave its chain:

- **Hidden states.** On decode and verify passes, each stage sends its part to every stage of the next step, in the model's dtype. A stage whose input comes from several stages sums their parts, as `p2p/replica.py` does on one GPU. Prefill, and the exact model, run layer after layer.
- **Requests and outputs.** The head takes requests in and sends them, and each pass's outputs, to every stage.
- **The ring.** The head embeds each pass's tokens and sends them to the first step's stages, so the first step can be spread too. A pass starts and ends on the head, and no stage waits on another's requests.

**The replica restarts from the copied layer's state every token, as on one GPU.** After each pass the copied layer's stage sends the replica's stage what it left: the K/V it wrote, which the replica's stage writes through SGLang's own KV cache call; or, for linear attention, the inputs of its state update, which the replica's stage replays through the same kernels. These updates go after the hidden states, off the critical path.

With a full copy or with replicas expanded to dense, routed runs give one GPU's numbers bit for bit (see [Tests](#tests)).

### Replicas prepared ahead of time

Each node that holds a replica prepares it once, before deployment, from the scanned plan and its calibration sums:

```bash
p2p replicas --model <checkpoint> --plan plan.json --calib calib.pt --out replicas [--layers 32] [--dtype auto]
# writes replicas/layer<i>.safetensors for each copied layer (or only those named)
```

- **How.** SGLang loads the copied layers on one GPU, with no embedding, head or other layer, and each projection is pruned with Wanda exactly as a plan prunes it on one GPU (`p2p/prepare.py`).
- **The format.** Each row's kept weights in column order, plus a 1-bit mask. Wanda keeps the same number in every row, so no indices are stored: a 40% replica is about 40% of its layer on disk and on the GPU.
- **At deployment,** `P2P_PP_REPLICAS=replicas` makes the replica's stage replace the copied layer's projections with the file's. A CUDA kernel (`p2p/compact.cu`, built at first use, installed as the projection's `quant_method`) multiplies by the compact weights without expanding them. Routed experts of an MoE layer are expanded per pass, only those the pass's tokens are routed to, and SGLang's own MoE method runs them.
- **A check.** `P2P_PP_REPLICA_FORM=dense` expands the replicas instead, which must give one GPU's tokens exactly.

### What depends on the model comes from SGLang

Nothing in the deployment is written per model family:

- **The embedding.** While the head's model is built, SGLang treats the head as a first stage, so it builds and loads the model's own embedding module (scaling and tied weights included). The first stage is built as a middle one and holds no table.
- **K/V updates** repeat SGLang's own cache write call, whatever the cache's layout (sliding window, compressed).
- **Replicas** are mapped from the checkpoint by SGLang's loader.
- **Trees** are laid out by SGLang's own tree kernel on every node.

To support another family, add its projection names to `TABLE` in `p2p/family.py`; it also has to use SGLang's layer boundary API. A model stops with an error if under 90% of a layer's weights are in the table.

### A pass without waiting

Three things keep a node from waiting longer than the network makes it (`p2p/route.py`, `p2p/stream.py`):

- **The input is received ahead.** A pass's input is laid out as the last one from the same rank, so the receive is only queued. The CPU sets the pass up and queues the layers and the send behind it; when the input lands, the GPU computes and sends without the CPU.
- **A verify pass is streamed in pieces** (`P2P_PP_STREAM=1,2,4`: a piece of one token, one of two, then fours). A node runs a pass's first tokens and sends them on while the rest are still coming, so each hop waits for one piece, not for the whole pass. This is exact: SGLang lays a tree out parents first, so a pass's first tokens are a tree of their own. A piece goes to the next step's slowest node first.
- **The head stops waiting once a pass is settled** (`P2P_PP_EARLY`, on by default). It follows the model's choices through the rows that have come; where the accepted branch ends, no later row can matter. On greedy passes only.

And one keeps the head's drafting off the path:

- **A tree leaves the head in phases** (`P2P_PP_PHASES=6:24`). After 6 draft steps the tree's first 24 rows are fixed and sent, with the pass's output, while the head drafts the rest. Nodes start on those rows and take the rest of the tree when it comes. The nodes of the next few steps (`P2P_PP_NEAR`, 4) get a pass's output right behind its first rows; the rest get it once all rows are out.

### Launching nodes

Each node runs SGLang's own launcher, as on any multi-machine pipeline, with the same `P2P_*` settings everywhere:

```bash
P2P_PP_COMM=fused P2P_PP_SPREAD=all P2P_PP_PLAN=plan.json P2P_PP_REPLICAS=replicas P2P_PP_EAGER_NCCL=1 \
P2P_PP_STREAM=1,2,4 P2P_PP_PHASES=6:24 SGLANG_ENABLE_PP_SPEC=1 \
    python -m sglang.launch_server --model-path <checkpoint> --pp-size 64 --nnodes 64 --node-rank <r> \
    --dist-init-addr <node 0's address>:<port> \
    --speculative-algorithm EAGLE --speculative-draft-model-path <checkpoint> \
    --speculative-num-steps 16 --speculative-eagle-topk 4 --speculative-num-draft-tokens 64 ...
```

`bench/pp/nodes.py` launches N such nodes on one machine, each with its own sockets and ports, and serves a request through node 0's HTTP server. Each node installs the plan in `P2P_PP_PLAN` when SGLang's pipeline loop starts. Across machines the tokenizer reaches the head over the TCP addresses SGLang derives from `--dist-init-addr`.

### Settings

| Variable | What it does |
|---|---|
| `P2P_PP_COMM` | `fused`: one delay per hop. `stock` (default): SGLang's own |
| `P2P_PP_SPREAD`, `P2P_PP_PLAN` | The groups to spread (`all`, or their first layers), and the plan |
| `P2P_PP_SPLIT` | Layers where a run of whole steps is cut onto another stage |
| `P2P_PP_CALIB` | Calibration sums, for a stage that runs pruned groups whole |
| `P2P_PP_REPLICAS`, `P2P_PP_REPLICA_FORM` | The folder of prepared replicas; `dense` expands them, as a check |
| `P2P_PP_STREAM` | Tokens in each piece of a streamed pass, the last size repeated (unset: whole passes) |
| `P2P_PP_EARLY` | `0`: the head waits for every piece. `check`: also compares each early stop with the whole pass |
| `P2P_PP_PHASES` | `steps:rows`, comma separated: a tree's first rows go out after that many draft steps |
| `P2P_PP_NEAR` | The steps after the first whose nodes get a pass's output behind its first rows (4) |
| `P2P_PP_MICROBATCHES` | Microbatch slots in SGLang's pipeline loop (2: one stream) |
| `P2P_PP_EAGER_NCCL` | `1`: one NCCL communicator per group, not one per pair of stages |
| `P2P_PP_TRACE`, `P2P_PP_SPANS` | Measuring: message counts, and spans and GPU marks of a pass |
| `P2P_PP_MBPS`, `P2P_PP_SHARE_GPUS` | Simulating: each stage's uplink rate, and several stages to a GPU |
| `P2P_TREES` | A folder to record every pass's drafting tree in (`bench/pp/trees.py`) |

### Limits

- **One GPU per stage,** and SGLang's own scheduler loop (not disaggregated prefill and decode).
- **No request timeouts.** SGLang checks them on the first stage, so `SGLANG_REQ_*_TIMEOUT` is refused.
- **Text only.** Images and audio are embedded on the first stage in SGLang, and aren't routed through the head.
- **Streaming, the early stop and phases apply to one request at a time.** Batches of several requests run whole.
- **MoE replicas on long prefills:** a pass that reaches every expert expands them all for that pass.
- **Startup memory:** SGLang still builds and loads the copied layer dense, so a replica's stage briefly needs that layer's dense size.
- **Per-layer inputs:** Gemma 4's small models compute per-layer inputs on the first stage, and those don't travel with the embedding yet.
- **Trees need bfloat16** on models with linear attention: SGLang's tree kernel fails in float16. Gemma's assistant drafter is untested as a tree.
- **Fused K/V writes:** a kernel that writes K/V without SGLang's write call isn't seen; the copied layer's stage stops with an error then.
- **Not run on real machines yet.** Nodes are simulated on one machine, with delays and link rates emulated.

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
| `p2p/pp.py` | How pipeline stages talk (`fused`), what the head sends when, and the measuring and simulating switches |
| `p2p/route.py` | A plan's steps across pipeline stages: who holds which layer, who sends what to whom, the head's ring, the replica's updates, passes received ahead and streamed |
| `p2p/stream.py` | The tree arithmetic of streaming: a pass's first tokens as a verify of their own, when rows settle a pass, a tree grown in phases |
| `p2p/prepare.py` | Replicas pruned ahead of time (`p2p replicas`) and installed on their stage |
| `p2p/compact.py`, `p2p/compact.cu` | Pruned weights stored compactly, and the CUDA kernel that multiplies by them |
| `bench/pp/` | Tools for checking and measuring a deployment (below) |

## Tests

```bash
pytest tests/test_search.py tests/test_wanda.py    # CPU only

# Tiny models on a real engine, at TP 1 and 2, and two and four pipeline stages (the four-stage tests skip on fewer
# GPUs). Takes about 40 minutes.
CUDA_VISIBLE_DEVICES=0,1,2,3 P2P_TEST_TOKENIZER=<tokenizer path> pytest tests -s
```

| Tests | What they check |
|---|---|
| `test_search.py`, `test_wanda.py`, `test_cli.py` | The plan search, pruning and the command line |
| `test_sglang.py`, `test_spec.py` | Plans inside SGLang at TP 1 and 2 on tiny Qwen3, Qwen3.5, Gemma 4 and Qwen3.5 MoE; drafting |
| `test_pp.py` | `fused` gives `stock`'s tokens and sends no gloo message |
| `test_route.py` | Spread steps against the same plan on one GPU: bit for bit with full copies and dense replicas, close with compact ones; drafting as a chain and a tree; streamed passes, the early stop and phased trees give one GPU's tokens |
| `test_stream.py` | Pieces, settled rows and phased trees on hand-made trees; a piece's rows equal the whole pass's |
| `test_compact.py` | Packing round-trips; the CUDA kernel and the MoE expansion match dense arithmetic |

To check a deployment on a real model (`bench/pp/`, on one machine with several GPUs):

```bash
# The same tokens as one GPU? A plan on one GPU, then on 4 stages with its first pair spread.
python bench/pp/same.py --model M --plan plan.json --calib calib.pt --spread 0 --split 34 --pp 4 \
    --replicas replicas --replica-form dense --out same.json

# Timing over an emulated network: every packet delayed 20 ms one way (no root needed).
bench/pp/netns.sh 20 python bench/pp/bench.py --model M --pp 4 --comm fused --plan plan.json --calib calib.pt \
    --spread 0 --split 34 --replicas replicas --draft mtp --out results.jsonl

# Nodes launched as on separate machines, several to a GPU.
P2P_PP_COMM=fused P2P_PP_SPREAD=all P2P_PP_PLAN=plan.json P2P_PP_REPLICAS=replicas P2P_PP_EAGER_NCCL=1 \
    bench/pp/netns.sh 20 python bench/pp/nodes.py --model M --nodes 64 --draft mtp
```

| Tool | Job |
|---|---|
| `same.py` | A plan on one GPU against the same plan routed: the first token where they differ |
| `bench.py`, `netns.sh` | ms per token and per pass, and messages per stage, over an emulated delay |
| `nodes.py` | N nodes launched with SGLang's multi-node flags on one machine |
| `overhead.py`, `pieces.py` | Where a pass's time goes: every stage's spans and GPU marks; a streamed run's pieces |
| `estimate.py` | Decode speed for a plan at a given delay and link rate, from the measured parts |
| `trees.py` | Records drafting trees, and what smaller, larger or phased trees would yield |
| `compact.py` | Times the replicas' kernel against dense matmuls |

Runs write under `runs/`, which git ignores; the drivers of past experiments are kept there with their results.
