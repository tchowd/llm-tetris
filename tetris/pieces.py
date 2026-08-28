"""Piece shape tables.

Cell offsets are (row, col) pairs authored against the standard SRS spawn
diagrams (row 0 = top of the piece's bounding box, col 0 = left). We do not
use SRS kicks — only the four rotation shapes themselves, hard-dropped from
the top. Shapes are normalized so the minimum row and minimum column are both
0; the caller then places the normalized shape at a given `x` (leftmost
occupied column) and drop offset.
"""
from __future__ import annotations

PIECE_LETTERS = "IJLOSTZ"

# Un-normalized (row, col) cell offsets per piece, per rot (0..3), matching
# the standard Tetris Guideline / SRS spawn-orientation diagrams.
_RAW_SHAPES: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {
    "I": (
        ((1, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 2), (1, 2), (2, 2), (3, 2)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
        ((0, 1), (1, 1), (2, 1), (3, 1)),
    ),
    "O": (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    ),
    "T": (
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (1, 2), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (2, 1)),
    ),
    "S": (
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 1), (1, 1), (1, 2), (2, 2)),
        ((1, 1), (1, 2), (2, 0), (2, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
    ),
    "Z": (
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 2), (1, 1), (1, 2), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (2, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
    ),
    "J": (
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
    ),
    "L": (
        ((0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (1, 2), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
    ),
}


def _normalize(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


# piece -> [normalized shape per rot 0..3]
NORMALIZED_SHAPES: dict[str, list[tuple[tuple[int, int], ...]]] = {
    piece: [_normalize(cells) for cells in rots] for piece, rots in _RAW_SHAPES.items()
}

# piece -> sorted list of canonical rot indices (lowest rot per unique shape)
CANONICAL_ROTS: dict[str, list[int]] = {}
for _piece, _shapes in NORMALIZED_SHAPES.items():
    _seen: set[tuple[tuple[int, int], ...]] = set()
    _canon: list[int] = []
    for _rot, _shape in enumerate(_shapes):
        if _shape not in _seen:
            _seen.add(_shape)
            _canon.append(_rot)
    CANONICAL_ROTS[_piece] = _canon


def shape_width(shape: tuple[tuple[int, int], ...]) -> int:
    return max(c for _, c in shape) + 1


def shape_height(shape: tuple[tuple[int, int], ...]) -> int:
    return max(r for r, _ in shape) + 1
