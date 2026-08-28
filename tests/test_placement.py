import pytest

from tetris.engine import Game, WIDTH
from tetris.pieces import CANONICAL_ROTS, NORMALIZED_SHAPES, PIECE_LETTERS, shape_width


def occupied_cols(board):
    return {c for row in board for c, ch in enumerate(row) if ch != "."}


def test_horizontal_I_at_x0_occupies_0_to_3():
    g = Game(seed=1)
    g.current = "I"
    g.step(0, 0)
    assert occupied_cols(g.board) == {0, 1, 2, 3}


def test_horizontal_I_at_x6_occupies_6_to_9():
    g = Game(seed=1)
    g.current = "I"
    g.step(0, 6)
    assert occupied_cols(g.board) == {6, 7, 8, 9}


def test_vertical_I_x9_legal_x10_illegal():
    g = Game(seed=1)
    g.current = "I"
    legal_pairs = {(p["rot"], p["x"]) for p in g.legal_placements()}
    assert (1, 9) in legal_pairs
    assert (1, 10) not in legal_pairs

    g.step(1, 9)
    assert occupied_cols(g.board) == {9}


def test_step_rejects_out_of_bounds_x():
    g = Game(seed=1)
    g.current = "I"
    with pytest.raises(ValueError):
        g.step(1, 10)


@pytest.mark.parametrize("piece", list(PIECE_LETTERS))
def test_min_occupied_column_equals_x_for_every_legal_placement(piece):
    for rot in CANONICAL_ROTS[piece]:
        width = shape_width(NORMALIZED_SHAPES[piece][rot])
        for x in range(WIDTH - width + 1):
            g = Game(seed=0)
            g.current = piece
            g.step(rot, x)
            cols = occupied_cols(g.board)
            assert min(cols) == x
            assert max(cols) == x + width - 1


# Empty-board legal-move counts from stage-1-game.md ("~9 (O) to ~34 (T/J/L)").
EXPECTED_LEGAL_COUNTS = {
    "O": 9,
    "I": 17,
    "S": 17,
    "Z": 17,
    "T": 34,
    "J": 34,
    "L": 34,
}


@pytest.mark.parametrize("piece", list(PIECE_LETTERS))
def test_legal_set_counts_and_no_duplicate_shapes_on_empty_board(piece):
    g = Game(seed=0)
    g.current = piece
    legal = g.legal_placements()
    assert len(legal) == EXPECTED_LEGAL_COUNTS[piece]

    # No two entries should produce identical final cells.
    seen = set()
    for placement in legal:
        key = tuple(sorted((r, c) for r, c in placement["cells"]))
        assert key not in seen
        seen.add(key)

    # Every listed placement must actually lock without error.
    for placement in legal:
        clone = g.clone()
        clone.step(placement["rot"], placement["x"])  # should not raise


def test_step_rejects_anything_not_in_legal_placements():
    g = Game(seed=0)
    g.current = "T"
    legal_pairs = {(p["rot"], p["x"]) for p in g.legal_placements()}
    for rot in range(4):
        for x in range(-1, WIDTH + 2):
            if (rot, x) not in legal_pairs:
                clone = g.clone()
                with pytest.raises(ValueError):
                    clone.step(rot, x)
