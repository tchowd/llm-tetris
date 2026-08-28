import random

from tetris.board import HEIGHT, WIDTH
from tetris.engine import Game
from tetris.placement import legal_placements_on
from tetris import teacher


def test_pick_always_returns_a_legal_placement():
    for seed in range(15):
        g = Game(seed=seed)
        agent_rng = random.Random(seed + 500)
        for _ in range(60):
            if g.game_over:
                break
            snap = g.snapshot()
            legal = snap["legal"]
            rot, x = teacher.pick(snap, legal)
            assert (rot, x) in {(p["rot"], p["x"]) for p in legal}
            # advance with a random legal move so later turns see varied,
            # not just teacher-curated, boards
            p = agent_rng.choice(legal)
            g.step(p["rot"], p["x"])


def test_tie_break_prefers_lower_rot_then_lower_x():
    # Flat empty board + symmetric piece (O) + symmetric next (O): locking
    # at x=0 and x=8 are exact mirror images of each other, so they must
    # score identically under every mirror-invariant feature we compute.
    g = Game(seed=0)
    g.current = "O"
    g.next = "O"
    snap = g.snapshot()
    legal = snap["legal"]
    board = [list(row) for row in snap["board"]]

    values = {p["x"]: teacher.value_of_placement(board, p, snap["next"]) for p in legal}
    max_value = max(values.values())
    tied_xs = sorted(x for x, v in values.items() if v == max_value)
    assert len(tied_xs) >= 2, "expected this board to produce a genuine tie among placements"

    rot, x = teacher.pick(snap, legal)
    assert (rot, x) == (0, tied_xs[0])


def test_teacher_avoids_a_fatal_placement_in_favor_of_a_surviving_one():
    # cols0-2, 7, 8 are solid walls (no headroom). cols3-6 are a 4-wide
    # pocket with exactly 2 rows of headroom at the very top. col9 is
    # permanently empty so no row is ever pre-completed (unrealistic
    # false line-clears would otherwise contaminate the scenario).
    # Locking O at the pocket's two middle columns (x=4) stone-walls both
    # remaining pocket columns into isolated, too-narrow-for-O leftovers:
    # fatal. Locking at either edge (x=3 or x=5) leaves an adjacent pair
    # open: survives.
    board = [["X"] * WIDTH for _ in range(HEIGHT)]
    for row in range(HEIGHT):
        board[row][9] = "."
    for row in (0, 1):
        for c in (3, 4, 5, 6):
            board[row][c] = "."

    legal = legal_placements_on(board, "O")
    legal = [{"rot": p["rot"], "x": p["x"], "cells": [list(c) for c in p["cells"]]} for p in legal]
    snapshot = {"board": ["".join(row) for row in board], "next": "O"}

    # Confirm the scenario is what we think it is before trusting pick().
    fatal_x, safe_xs = 4, {3, 5}
    for p in legal:
        board1, _, _ = teacher.overlay_and_clear(board, [tuple(c) for c in p["cells"]])
        next_legal = legal_placements_on(board1, "O")
        if p["x"] == fatal_x:
            assert next_legal == [], "x=4 was expected to be fatal"
        elif p["x"] in safe_xs:
            assert next_legal != [], f"x={p['x']} was expected to survive"

    rot, x = teacher.pick(snapshot, legal)
    assert x != fatal_x
    assert x in safe_xs


def _brute_force_pick(snapshot, legal):
    """Re-does teacher.pick's search independently (own loop, own max),
    to catch a loop/tie-break bug rather than a weights/features bug."""
    board = [list(row) for row in snapshot["board"]]
    next_piece = snapshot["next"]
    best_key = None
    best_action = None
    for p in legal:
        value = teacher.value_of_placement(board, p, next_piece)
        key = (value, -p["rot"], -p["x"])
        if best_key is None or key > best_key:
            best_key = key
            best_action = (p["rot"], p["x"])
    return best_action


def test_best_score_audit_matches_independent_recompute():
    for seed in range(10):
        g = Game(seed=seed)
        agent_rng = random.Random(seed + 777)
        for _ in range(30):
            if g.game_over:
                break
            snap = g.snapshot()
            legal = snap["legal"]
            assert teacher.pick(snap, legal) == _brute_force_pick(snap, legal)
            p = agent_rng.choice(legal)
            g.step(p["rot"], p["x"])
