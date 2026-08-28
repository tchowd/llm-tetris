"""Shared board constants. A single source so Stage 1's engine and Stage 2's
teacher never drift on field size or the empty-cell marker."""

WIDTH = 10
HEIGHT = 20
EMPTY = "."


def board_to_lists(board_rows: list[str]) -> list[list[str]]:
    """The board representation snapshots/JSONL store (20 strings) ->
    the mutable list-of-lists placement/teacher code operates on."""
    return [list(row) for row in board_rows]
