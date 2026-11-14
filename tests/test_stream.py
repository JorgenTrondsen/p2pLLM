"""Streaming a verify pass in pieces (p2p.stream): where a pass is cut, when its first rows settle it, and (one GPU) each
piece as SGLang's verify of the pass's first tokens, whose new rows are the whole pass's, within rounding."""
import json
import random

import pytest
import torch

def test_pieces_are_where_a_pass_is_cut():
    from p2p.stream import pieces
    assert pieces(6, 1) == [1, 2, 3, 4, 5, 6] and pieces(6, 4) == [4, 6] and pieces(6, 6) == [6] and pieces(3, 8) == [3]
    assert pieces(12, (1, 2, 4)) == [1, 3, 7, 11, 12] and pieces(1, (1, 2, 4)) == [1] and pieces(6, (2,)) == [2, 4, 6]
    assert pieces(12, (1, 2, 4), (5,)) == [1, 3, 5, 9, 12] and pieces(6, 2, (3, 5)) == [2, 3, 5, 6]    # and at each cut


def test_a_tree_grows_in_phases_behind_the_rows_already_fixed():
    """2 candidates a step, 3 steps: candidates 0-1, then 2-5 (0's children 2-3, 1's 4-5), then 6-9, the children of
    the two that the second step expanded (2 and 4)."""
    from p2p.stream import grown
    scores, parents = [.6, .4, .5, .1, .3, .1, .4, .1, .2, .1], [-1, 0, 1, 2, 4]
    assert grown(scores, parents, 2, set(), 0, 1, [0, 1], 2) == [0, 1]
    assert grown(scores, parents, 2, {0, 1}, 1, 3, [], 3) == [2, 4, 6]          # the best whose parents are in
    assert grown(scores, parents, 2, {0, 1}, 1, 2, [2, 4], 2) == [2, 4]         # what the next step expands is kept
    assert grown(scores, parents, 2, {0, 1, 2, 4}, 2, 3, [], 1) == [6]
    # after two steps at once, room for 3: candidate 2 with its parent, then the best left; 4 doesn't fit with its own
    assert grown(scores, parents, 2, set(), 0, 2, [2, 4], 3) == [0, 1, 2]
    assert grown(scores, parents, 2, {0}, 1, 3, [], 4) == [2, 3, 6, 7]          # only what hangs on rows in the tree


def test_a_pass_is_settled_where_its_accepted_branch_ends():
    """The root (token 5) with children 7 and 8 (rows 1, 2); row 1 has the child 9 (row 3), row 3 the child 4 (row 4)."""
    from p2p.stream import settled
    tree = [5, 7, 8, 9, 4], [1, 3, -1, 4, -1], [-1, 2, -1, -1, -1]
    assert settled(*tree, [6]) == (0, 0)                    # the model's choice at the root isn't drafted
    assert settled(*tree, [8]) == (None, 2)                 # it is row 2's token: that row is needed
    assert settled(*tree, [8, 0, 1]) == (2, 2)              # and row 2 has no children
    assert settled(*tree, [7, 9]) == (None, 3)              # the branch goes on beyond the rows that have come
    assert settled(*tree, [7, 9, 0, 4]) == (None, 4)
    assert settled(*tree, [7, 9, 0, 3]) == (3, 3)
    assert settled(*tree, [7, 9, 0, 4, 1]) == (4, 4)        # the whole tree always settles


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("size", [1, 2, 4])
def test_each_piece_gives_the_whole_pass_rows(tmp_path, size):
    """Tiny Qwen3.5 (linear and full attention) drafting a tree with its MTP layer: 2 candidates a step, 3 steps, 6
    tokens a pass, in bfloat16 as SGLang's tree convolution kernel needs."""
    from test_spec import _with_mtp
    from p2p.engine import Model, speculative
    path = _with_mtp()
    rng = random.Random(0)
    prompts = [[rng.randrange(1000, 20000) for _ in range(8 + 5 * i)] for i in range(3)]
    m = Model(path, tp=1, batch=1, max_total_tokens=4096, skip_server_warmup=True, dtype="bfloat16",
              **speculative(path, "mtp", 3, topk=2, tokens=6))
    try:
        m.rpc("p2p_stream_check", size=size)
        for p in prompts:
            m.engine.generate(input_ids=[p], sampling_params={"temperature": 0, "max_new_tokens": 16, "ignore_eos": True})
        m.rpc("p2p_stream_check", path=f"{tmp_path}/found.json")
    finally:
        m.close()
    found = json.load(open(f"{tmp_path}/found.json"))
    assert len(found) >= 3 and all(len(f) == len(range(0, 6, size)) for f in found)
    one = [p for f in found for p in f if p["rows"][1] == 1]     # a piece of the root alone
    rest = [p for f in found for p in f if p["rows"][1] > 1]
    worst = lambda ps: {k: max(p[k] for p in ps) for k in ("logits", "hidden")} if ps else {}
    print(f"\npieces of {size}: {len(found)} passes, largest difference from the whole pass: the root alone "
          f"{worst(one)}, the rest {worst(rest)}")
    # Exact, but for a piece of one token, whose matmuls take cuBLAS's one-row path and round otherwise.
    assert all(v == 0 for v in worst(rest).values())
    assert all(v < 5e-2 for v in worst(one).values())
