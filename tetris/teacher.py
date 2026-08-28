"""2-ply Dellacherie / El-Tetris teacher.

`pick(snapshot, legal) -> (rot, x)` scores every legal placement of the
current piece plus the best the known `next` piece can do afterward, using
lightweight board-overlay simulation (no `Game` instance, no mutation of the
live engine). See plan/stage-2-teacher.md.
"""
from __future__ import annotations

from .board import HEIGHT, WIDTH
from .features import column_heights, column_holes, wells
from .placement import legal_placements_on

Cells = tuple[tuple[int, int], ...]

# A large constant, not a tuned weight: guarantees "dies in one more move"
# always loses to any candidate that keeps the game alive, regardless of
# how good score1 looks in isolation.
DEATH_PENALTY = 1_000_000.0

# One internally-consistent starting ranking (stage-2-teacher.md's "Weights"
# section). Absolute scale never matters, only the ratios.
WEIGHTS = {
    "landing_height": -1.0,
    "eroded_cells": 1.0,
    "row_transitions": -1.0,
    "column_transitions": -1.0,
    "holes": -4.0,
    "well_sums": -1.0,
}


def _cells(raw_cells) -> Cells:
    return tuple((r, c) for r, c in raw_cells)


def overlay_and_clear(board: list[list[str]], cells: Cells) -> tuple[list[list[str]], int, int]:
    """Lock `cells` onto `board` and clear full rows.

    Returns (new_board, lines_cleared, piece_cells_in_cleared_rows) — the
    third value is how many of `cells` themselves sat in a row that cleared,
    the raw count the eroded-cells feature is built from.
    """
    overlaid = [row[:] for row in board]
    for r, c in cells:
        overlaid[r][c] = "#"

    cleared_rows = {r for r in range(HEIGHT) if all(ch != "." for ch in overlaid[r])}
    lines_cleared = len(cleared_rows)
    piece_cells_in_cleared_rows = sum(1 for r, _ in cells if r in cleared_rows)

    remaining = [row for i, row in enumerate(overlaid) if i not in cleared_rows]
    new_board = [["."] * WIDTH for _ in range(lines_cleared)] + remaining
    return new_board, lines_cleared, piece_cells_in_cleared_rows


def landing_height(cells: Cells) -> float:
    r_top = min(r for r, _ in cells)
    r_bottom = max(r for r, _ in cells)
    return HEIGHT - (r_top + r_bottom) / 2


def row_transitions(board: list[list[str]]) -> int:
    total = 0
    for row in board:
        padded = ["#", *row, "#"]  # side walls: real, solid boundaries
        for a, b in zip(padded, padded[1:]):
            if (a != ".") != (b != "."):
                total += 1
    return total


def column_transitions(board: list[list[str]]) -> int:
    total = 0
    for col in range(WIDTH):
        column = [board[row][col] for row in range(HEIGHT)]
        column.append("#")  # floor: real, solid boundary. No top pad.
        for a, b in zip(column, column[1:]):
            if (a != ".") != (b != "."):
                total += 1
    return total


def well_sums(board: list[list[str]]) -> int:
    depths = wells(column_heights(board))
    return sum(d * (d + 1) // 2 for d in depths)


def dellacherie_features(
    board: list[list[str]],
    cells: Cells,
    lines_cleared: int,
    piece_cells_in_cleared_rows: int,
) -> dict:
    return {
        "landing_height": landing_height(cells),
        "eroded_cells": lines_cleared * piece_cells_in_cleared_rows,
        "row_transitions": row_transitions(board),
        "column_transitions": column_transitions(board),
        "holes": sum(column_holes(board)),
        "well_sums": well_sums(board),
    }


def dellacherie_score(
    board: list[list[str]],
    cells: Cells,
    lines_cleared: int,
    piece_cells_in_cleared_rows: int,
    weights: dict = WEIGHTS,
) -> float:
    feats = dellacherie_features(board, cells, lines_cleared, piece_cells_in_cleared_rows)
    return sum(weights[name] * value for name, value in feats.items())


def score_candidate(board: list[list[str]], cells: Cells, weights: dict = WEIGHTS) -> float:
    """Overlay+clear one hypothetical placement and score the result."""
    new_board, lines_cleared, eroded = overlay_and_clear(board, cells)
    return dellacherie_score(new_board, cells, lines_cleared, eroded, weights)


def value_of_placement(board: list[list[str]], placement: dict, next_piece: str, weights: dict = WEIGHTS) -> float:
    """2-ply value of locking `placement` (current piece) on `board`, given
    the already-known `next_piece`."""
    cells = _cells(placement["cells"])
    board1, lines1, eroded1 = overlay_and_clear(board, cells)
    score1 = dellacherie_score(board1, cells, lines1, eroded1, weights)

    next_legal = legal_placements_on(board1, next_piece)
    if not next_legal:
        return score1 - DEATH_PENALTY

    best_next = max(score_candidate(board1, p["cells"], weights) for p in next_legal)
    return score1 + best_next


def pick(snapshot: dict, legal: list[dict], weights: dict = WEIGHTS) -> tuple[int, int]:
    """`teacher.pick(snapshot, legal) -> (rot, x)`.

    Reads only `snapshot["board"]` / `snapshot["next"]` and the caller-
    supplied `legal` list — Stage 1's public API, nothing more.
    """
    board = [list(row) for row in snapshot["board"]]
    next_piece = snapshot["next"]

    best = max(
        legal,
        key=lambda p: (value_of_placement(board, p, next_piece, weights), -p["rot"], -p["x"]),
    )
    return (best["rot"], best["x"])
