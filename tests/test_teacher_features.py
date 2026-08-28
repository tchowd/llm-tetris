from tetris.board import HEIGHT, WIDTH
from tetris.engine import Game
from tetris.teacher import (
    dellacherie_features,
    landing_height,
    overlay_and_clear,
    row_transitions,
    column_transitions,
    well_sums,
)


def build_board(filled_row_ranges: dict[int, set[int]]) -> list[list[str]]:
    """row -> set of columns to leave EMPTY in that row (all other rows fully empty)."""
    board = [["."] * WIDTH for _ in range(HEIGHT)]
    for row, empty_cols in filled_row_ranges.items():
        for c in range(WIDTH):
            if c not in empty_cols:
                board[row][c] = "X"
    return board


def test_row_transitions_hand_built_row():
    # One modified row (walls padded on both sides), 19 fully-empty rows.
    board = [["."] * WIDTH for _ in range(HEIGHT)]
    board[10] = list("XX...XXX..")
    # padded: # X X . . . X X X . . #
    # diffs:    -  -  d  -  -  d  -  -  d  -  d   = 4
    assert row_transitions(board) == 19 * 2 + 4


def test_row_transitions_empty_board_is_two_per_row():
    board = [["."] * WIDTH for _ in range(HEIGHT)]
    assert row_transitions(board) == HEIGHT * 2


def test_column_transitions_hand_built_column():
    board = [["."] * WIDTH for _ in range(HEIGHT)]
    for row in (15, 16, 18, 19):
        board[row][3] = "X"
    # col3 sequence (+floor): 15 dots, X, X, ., X, X, #
    # diffs: dot->X(1), X->X(0), X->.(1), .->X(1), X->X(0), X->#(0) = 3
    # other 9 columns are empty: each contributes 1 (just the floor pad)
    assert column_transitions(board) == 9 * 1 + 3


def test_column_transitions_empty_board_is_one_per_column():
    board = [["."] * WIDTH for _ in range(HEIGHT)]
    assert column_transitions(board) == WIDTH * 1


def test_well_sums_interior_and_wall_wells():
    # Same board as Stage 1's feature test: wall-wells at both edges (depth
    # 5 and 4) plus a flat interior column that is NOT a well (col2).
    board = [["."] * WIDTH for _ in range(HEIGHT)]
    for row in (15, 16, 17, 18, 19):
        board[row][1] = "X"
    for row in (18, 19):
        board[row][2] = "X"
    for row in (17, 19):
        board[row][5] = "X"
    for row in (16, 17, 18, 19):
        board[row][8] = "X"
    # heights: [0,5,2,0,0,3,0,0,4,0] -> wells [5,0,0,0,0,0,0,0,0,4]
    assert well_sums(board) == 5 * 6 // 2 + 4 * 5 // 2


def test_eroded_cells_single_row_clear():
    board = build_board({19: {3, 4, 5}})
    cells = ((18, 3), (19, 3), (19, 4), (19, 5))
    new_board, lines_cleared, piece_cells_in_cleared_rows = overlay_and_clear(board, cells)
    assert lines_cleared == 1
    assert piece_cells_in_cleared_rows == 3
    feats = dellacherie_features(new_board, cells, lines_cleared, piece_cells_in_cleared_rows)
    assert feats["eroded_cells"] == 1 * 3


def test_eroded_cells_multi_row_clear():
    board = build_board({18: {7}, 19: {7}})
    cells = ((18, 7), (19, 7))
    new_board, lines_cleared, piece_cells_in_cleared_rows = overlay_and_clear(board, cells)
    assert lines_cleared == 2
    assert piece_cells_in_cleared_rows == 2
    feats = dellacherie_features(new_board, cells, lines_cleared, piece_cells_in_cleared_rows)
    assert feats["eroded_cells"] == 2 * 2


def test_eroded_cells_zero_when_nothing_clears():
    board = [["."] * WIDTH for _ in range(HEIGHT)]
    cells = ((18, 0), (18, 1), (19, 0), (19, 1))
    new_board, lines_cleared, piece_cells_in_cleared_rows = overlay_and_clear(board, cells)
    assert lines_cleared == 0
    assert piece_cells_in_cleared_rows == 0
    feats = dellacherie_features(new_board, cells, lines_cleared, piece_cells_in_cleared_rows)
    assert feats["eroded_cells"] == 0


def test_landing_height_hand_computed():
    # Horizontal I resting on the floor: r_top == r_bottom == 19.
    assert landing_height(((19, 0), (19, 1), (19, 2), (19, 3))) == 1
    # Vertical I resting on the floor: r_top=16, r_bottom=19.
    assert landing_height(((16, 0), (17, 0), (18, 0), (19, 0))) == 2.5


def test_landing_height_matches_stage1_legal_placements():
    for seed in range(10):
        g = Game(seed=seed)
        for p in g.legal_placements():
            cells = tuple((r, c) for r, c in p["cells"])
            assert landing_height(cells) == p["landing_height"]
