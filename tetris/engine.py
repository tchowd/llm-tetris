"""The Tetris engine: source of truth for rules, features, and the
train/eval serializer. Human UI, teacher, and LLM all drive this through the
same `Game` API (see plan/stage-1-game.md)."""
from __future__ import annotations

import copy
import random

from .pieces import CANONICAL_ROTS, NORMALIZED_SHAPES, PIECE_LETTERS, shape_width
from .serialize import serialize_action, serialize_prompt

WIDTH = 10
HEIGHT = 20

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

    def _drop_offset(self, shape: tuple[tuple[int, int], ...], x: int) -> int | None:
        """Largest row offset (from the shape sitting at the very top of the
        board) that does not collide, or None if even offset 0 collides."""
        last_valid = None
        offset = 0
        while True:
            collides = False
            for r, c in shape:
                row = r + offset
                if row >= HEIGHT or self.board[row][c + x] != ".":
                    collides = True
                    break
            if collides:
                break
            last_valid = offset
            offset += 1
        return last_valid

    def legal_placements(self) -> list[dict]:
        piece = self.current
        placements = []
        for rot in CANONICAL_ROTS[piece]:
            shape = NORMALIZED_SHAPES[piece][rot]
            width = shape_width(shape)
            for x in range(WIDTH - width + 1):
                offset = self._drop_offset(shape, x)
                if offset is None:
                    continue
                cells = [(r + offset, c + x) for r, c in shape]
                r_top = min(r for r, _ in cells)
                r_bottom = max(r for r, _ in cells)
                landing_height = HEIGHT - (r_top + r_bottom) / 2
                placements.append(
                    {
                        "rot": rot,
                        "x": x,
                        "cells": [[r, c] for r, c in cells],
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

    def _column_heights(self) -> list[int]:
        heights = [0] * WIDTH
        for col in range(WIDTH):
            for row in range(HEIGHT):
                if self.board[row][col] != ".":
                    heights[col] = HEIGHT - row
                    break
        return heights

    def _column_holes(self) -> list[int]:
        holes = [0] * WIDTH
        for col in range(WIDTH):
            seen_filled = False
            for row in range(HEIGHT):
                if self.board[row][col] != ".":
                    seen_filled = True
                elif seen_filled:
                    holes[col] += 1
        return holes

    def _wells(self, heights: list[int]) -> list[int]:
        wells = [0] * WIDTH
        for col in range(WIDTH):
            left = heights[col - 1] if col > 0 else HEIGHT
            right = heights[col + 1] if col < WIDTH - 1 else HEIGHT
            if heights[col] < left and heights[col] < right:
                wells[col] = min(left, right) - heights[col]
        return wells

    @staticmethod
    def _bumpiness(heights: list[int]) -> int:
        return sum(abs(heights[i] - heights[i + 1]) for i in range(WIDTH - 1))

    def features(self) -> dict:
        heights = self._column_heights()
        holes = self._column_holes()
        wells = self._wells(heights)
        bumpiness = self._bumpiness(heights)
        return {
            "piece": self.current,
            "next": self.next,
            "heights": heights,
            "holes": holes,
            "wells": wells,
            "bumpiness": bumpiness,
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
