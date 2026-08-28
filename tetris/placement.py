"""Pure board-array placement mechanics: hard-drop offsets and the legal-move
enumeration. No `Game` instance, no mutation — shared by Stage 1's
`Game.legal_placements()` and Stage 2's hypothetical-placement scoring so the
two never compute "where does this piece land" two different ways."""
from __future__ import annotations

from .board import HEIGHT, WIDTH
from .pieces import CANONICAL_ROTS, NORMALIZED_SHAPES, shape_width


def drop_offset(board: list[list[str]], shape: tuple[tuple[int, int], ...], x: int) -> int | None:
    """Largest row offset (shape sitting at the very top of the board = 0)
    that does not collide, or None if even offset 0 collides."""
    last_valid = None
    offset = 0
    while True:
        collides = False
        for r, c in shape:
            row = r + offset
            if row >= HEIGHT or board[row][c + x] != ".":
                collides = True
                break
        if collides:
            break
        last_valid = offset
        offset += 1
    return last_valid


def legal_placements_on(board: list[list[str]], piece: str) -> list[dict]:
    """Every drop-from-top (rot, x) that fits `piece` on `board`.

    Returns dicts with `rot`, `x`, and `cells` (a tuple of (row, col) pairs).
    Only canonical rotations are surfaced (see tetris.pieces).
    """
    placements = []
    for rot in CANONICAL_ROTS[piece]:
        shape = NORMALIZED_SHAPES[piece][rot]
        width = shape_width(shape)
        for x in range(WIDTH - width + 1):
            offset = drop_offset(board, shape, x)
            if offset is None:
                continue
            cells = tuple((r + offset, c + x) for r, c in shape)
            placements.append({"rot": rot, "x": x, "cells": cells})
    return placements
