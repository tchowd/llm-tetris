"""Recovery experiment contracts; never alters the frozen engine or prompt."""
from __future__ import annotations

import json
from pathlib import Path

from tetris.engine import Game
from tetris.pieces import CANONICAL_ROTS, NORMALIZED_SHAPES, shape_width
from tetris.placement import drop_offset
from tetris.rl import restore_game, state_hash


def placement_failure(game: Game, action: tuple[int, int] | None) -> str:
    if action is None:
        return "parse_failure"
    rot, x = action
    if rot not in CANONICAL_ROTS[game.current]:
        return "noncanonical_rotation"
    shape = NORMALIZED_SHAPES[game.current][rot]
    if not 0 <= x <= 10 - shape_width(shape):
        return "outside_board"
    if drop_offset(game.board, shape, x) is None:
        return "blocked_at_top"
    return "legal"


def load_start_bank(path: Path, allowed_seeds: list[int]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("recovery start bank is empty")
    ids, hashes = set(), set()
    for row in rows:
        if row.get("split") != "train" or row["seed"] not in allowed_seeds:
            raise ValueError("recovery start is not from the registered training partition")
        if row["state_id"] in ids or row["state_hash"] in hashes:
            raise ValueError("duplicate recovery start")
        game = restore_game(row["seed"], row["action_prefix"], expected=row)
        if game.game_over or not game.legal_placements() or state_hash(game.snapshot()) != row["state_hash"]:
            raise ValueError("invalid recovery starting state")
        ids.add(row["state_id"])
        hashes.add(row["state_hash"])
    return rows


def make_group_starts(seed: int, group_size: int, start: dict | None = None) -> list[Game]:
    if group_size < 2:
        raise ValueError("a trajectory group requires at least two games")
    if start is not None and start["seed"] != seed:
        raise ValueError("recovery seed disagrees with recorded start")
    games = []
    for index in range(group_size):
        game = restore_game(seed, start["action_prefix"], expected=start) if start else Game(seed=seed)
        if game.game_over:
            raise ValueError("cannot sample from a terminal start")
        game.game_id = f"episode-{seed}-{index}"
        games.append(game)
    return games
