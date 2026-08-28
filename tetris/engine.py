"""The Tetris engine: source of truth for rules, features, and the
train/eval serializer. Human UI, teacher, and LLM all drive this through the
same `Game` API (see plan/stage-1-game.md)."""
from __future__ import annotations

import copy
import random

from .board import HEIGHT, WIDTH
from .features import bumpiness, column_heights, column_holes, wells
from .pieces import PIECE_LETTERS
from .placement import legal_placements_on
from .serialize import serialize_action, serialize_prompt

# Guideline line-clear points, indexed by number of lines cleared (0..4).
LINE_POINTS = (0, 100, 300, 500, 800)
HARD_DROP_POINTS_PER_CELL = 2
LINES_PER_LEVEL = 10


class Game:
    def __init__(self, seed: int, game_id: str | None = None):
        self.seed = seed
        self.game_id = game_id if game_id is not None else str(seed)
        self._rng = random.Random(seed)
        self._bag: list[str] = []

        self.board: list[list[str]] = [["."] * WIDTH for _ in range(HEIGHT)]
        self.score = 0
        self.lines = 0
        self.level = 1
        self.turn = 0
        self.game_over = False

        self.current = self._draw()
        self.next = self._draw()
        if not self.legal_placements():
            self.game_over = True

    # -- randomizer ---------------------------------------------------

    def _draw(self) -> str:
        if not self._bag:
            self._bag = list(PIECE_LETTERS)
            self._rng.shuffle(self._bag)
        return self._bag.pop()

    # -- placement ------------------------------------------------------

    def legal_placements(self) -> list[dict]:
        placements = []
        for p in legal_placements_on(self.board, self.current):
            r_top = min(r for r, _ in p["cells"])
            r_bottom = max(r for r, _ in p["cells"])
            landing_height = HEIGHT - (r_top + r_bottom) / 2
            placements.append(
                {
                    "rot": p["rot"],
                    "x": p["x"],
                    "cells": [[r, c] for r, c in p["cells"]],
                    "landing_height": landing_height,
                }
            )
        return placements

    def step(self, rot: int, x: int) -> dict:
        if self.game_over:
            raise ValueError("game is over")
        match = next(
            (p for p in self.legal_placements() if p["rot"] == rot and p["x"] == x),
            None,
        )
        if match is None:
            raise ValueError(f"illegal placement: rot={rot} x={x}")

        r_top = min(r for r, _ in match["cells"])
        drop_distance = r_top  # shape's normalized top row is 0 -> offset == r_top

        for r, c in match["cells"]:
            self.board[r][c] = self.current

        lines_cleared = self._clear_lines()
        self.score += LINE_POINTS[lines_cleared]
        self.score += HARD_DROP_POINTS_PER_CELL * drop_distance
        self.level = 1 + self.lines // LINES_PER_LEVEL

        self.turn += 1
        self.current = self.next
        self.next = self._draw()
        if not self.legal_placements():
            self.game_over = True

        return self.snapshot()

    def _clear_lines(self) -> int:
        remaining = [row for row in self.board if any(c == "." for c in row)]
        cleared = HEIGHT - len(remaining)
        if cleared:
            self.board = [["."] * WIDTH for _ in range(cleared)] + remaining
            self.lines += cleared
        return cleared

    # -- features ---------------------------------------------------------

    def features(self) -> dict:
        heights = column_heights(self.board)
        holes = column_holes(self.board)
        well_depths = wells(heights)
        bump = bumpiness(heights)
        return {
            "piece": self.current,
            "next": self.next,
            "heights": heights,
            "holes": holes,
            "wells": well_depths,
            "bumpiness": bump,
            "aggregate_height": sum(heights),
            "holes_total": sum(holes),
            "max_height": max(heights),
        }

    # -- snapshot / serialization -----------------------------------------

    def snapshot(self) -> dict:
        feats = self.features()
        prompt = serialize_prompt(feats)
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "turn": self.turn,
            "piece": feats["piece"],
            "next": feats["next"],
            "heights": feats["heights"],
            "holes": feats["holes"],
            "wells": feats["wells"],
            "bumpiness": feats["bumpiness"],
            "aggregate_height": feats["aggregate_height"],
            "holes_total": feats["holes_total"],
            "max_height": feats["max_height"],
            "board": ["".join(row) for row in self.board],
            "legal": self.legal_placements(),
            "prompt": prompt,
            "score": self.score,
            "lines": self.lines,
            "game_over": self.game_over,
        }

    @staticmethod
    def serialize_prompt(state: dict) -> str:
        return serialize_prompt(state)

    @staticmethod
    def serialize_action(rot: int, x: int) -> str:
        return serialize_action(rot, x)

    def clone(self) -> "Game":
        return copy.deepcopy(self)
