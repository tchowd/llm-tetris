"""Stage 3: turn the Stage 2 teacher into `games.jsonl` + `rows.jsonl`.

This module holds only pure, reusable logic (play one game, build one row,
decide a game's split). It writes nothing to disk — see
`scripts/generate_dataset.py` for the CLI that drives this at scale, and
`tetris/dataset_validate.py` for the validator that proves the output is
not a lie. See plan/stage-3-dataset.md.
"""
from __future__ import annotations

import hashlib
import random

from . import teacher as teacher_mod
from .engine import Game
from .serialize import serialize_action

# Part of the on-disk format: changing this changes which turns are noisy
# for a given seed, so games generated before/after a change are no longer
# reproducible from seed alone.
NOISE_SALT = 0x7E7215
DEFAULT_MAX_PIECES = 400
NOISY_EVERY_NTH_PIECE = 8
EVAL_SPLIT_PERCENT = 5


def split_for_game_id(game_id: str) -> str:
    """Deterministic train/eval split, independent of file order or how
    many games exist — see stage-3-dataset.md's "Split" row."""
    digest = hashlib.sha1(game_id.encode("utf-8")).hexdigest()
    return "eval" if int(digest, 16) % 100 < EVAL_SPLIT_PERCENT else "train"


def row_from_snapshot(snap: dict, label: tuple[int, int], explored: bool) -> dict:
    """Build one `rows.jsonl` row from a pre-move Stage 1 snapshot and the
    teacher's label for it. Pure function of its arguments — the same
    snapshot + label always produces the same row, which is what makes the
    dataset regenerable from `games.jsonl` alone."""
    rot, x = label
    return {
        "game_id": snap["game_id"],
        "seed": snap["seed"],
        "turn": snap["turn"],
        "prompt": snap["prompt"],
        "completion": serialize_action(rot, x),
        "rot": rot,
        "x": x,
        "piece": snap["piece"],
        "next": snap["next"],
        "heights": snap["heights"],
        "holes": snap["holes"],
        "wells": snap["wells"],
        "bumpiness": snap["bumpiness"],
        "aggregate_height": snap["aggregate_height"],
        "holes_total": snap["holes_total"],
        "max_height": snap["max_height"],
        "board": snap["board"],
        "lines": snap["lines"],
        "score": snap["score"],
        "explored": explored,
        "split": split_for_game_id(snap["game_id"]),
    }


def is_noisy_game(seed: int) -> bool:
    """~10% of games are noisy. Tied directly to `seed` (not a separate
    hash) so it's obvious and exact for a sequential seed range."""
    return seed % 10 == 0


def generate_game(seed: int, max_pieces: int = DEFAULT_MAX_PIECES, noisy: bool | None = None) -> tuple[list[dict], dict]:
    """Play one game with the teacher, optionally injecting exploration
    noise. Returns (rows, game_record) — game_record is one `games.jsonl`
    line, rows are that game's `rows.jsonl` lines.
    """
    if noisy is None:
        noisy = is_noisy_game(seed)

    g = Game(seed=seed)
    noise_rng = random.Random(seed ^ NOISE_SALT)
    rows: list[dict] = []
    actions: list[list[int]] = []
    labels: list[list[int]] = []
    explored_turns: list[int] = []

    while not g.game_over and g.turn < max_pieces:
        snap = g.snapshot()
        legal = snap["legal"]
        rot, x = teacher_mod.pick(snap, legal)
        # Stage 2 already guarantees this; assert so a future weight change
        # can never quietly poison the file (stage-3-dataset.md).
        assert (rot, x) in {(p["rot"], p["x"]) for p in legal}, (
            f"teacher produced an illegal label rot={rot} x={x} at game_id={g.game_id} turn={g.turn}"
        )

        explore = noisy and g.turn % NOISY_EVERY_NTH_PIECE == NOISY_EVERY_NTH_PIECE - 1
        if explore:
            chosen = noise_rng.choice(legal)
            act_rot, act_x = chosen["rot"], chosen["x"]
            explored_turns.append(g.turn)
        else:
            act_rot, act_x = rot, x

        rows.append(row_from_snapshot(snap, label=(rot, x), explored=explore))
        actions.append([act_rot, act_x])
        labels.append([rot, x])
        g.step(act_rot, act_x)

    game_record = {
        "game_id": g.game_id,
        "seed": seed,
        "actions": actions,
        "labels": labels,
        "explored_turns": explored_turns,
        "pieces": g.turn,
        "lines": g.lines,
        "score": g.score,
        "died": g.game_over,
    }
    return rows, game_record


def rebuild_rows_from_game(game_record: dict, max_pieces: int = DEFAULT_MAX_PIECES) -> list[dict]:
    """Replay a game purely from its `games.jsonl` record (seed + executed
    actions) and recompute every row from scratch, including re-asking the
    teacher for the label at each turn. Used by the validator to prove
    `rows.jsonl` matches what `games.jsonl` actually describes — no read of
    `rows.jsonl` happens here.
    """
    g = Game(seed=game_record["seed"])
    explored_turns = set(game_record["explored_turns"])
    rows: list[dict] = []
    for turn, (act_rot, act_x) in enumerate(game_record["actions"]):
        if g.game_over or g.turn >= max_pieces:
            break
        snap = g.snapshot()
        rot, x = teacher_mod.pick(snap, snap["legal"])
        rows.append(row_from_snapshot(snap, label=(rot, x), explored=turn in explored_turns))
        g.step(act_rot, act_x)
    return rows
