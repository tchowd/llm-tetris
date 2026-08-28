from tetris.engine import Game, HEIGHT, WIDTH


def build_board(filled: dict[int, set[int]]) -> list[list[str]]:
    """filled: column -> set of row indices to mark filled ('X')."""
    board = [["."] * WIDTH for _ in range(HEIGHT)]
    for col, rows in filled.items():
        for row in rows:
            board[row][col] = "X"
    return board


def test_heights_holes_wells_bumpiness_on_hand_built_board():
    g = Game(seed=0)
    # col0: empty                      -> height 0, wall-well on the left edge
    # col1: solid rows 15-19           -> height 5
    # col2: solid rows 18-19           -> height 2
    # col5: rows 17 and 19, hole at 18 -> height 3, 1 covered hole
    # col8: solid rows 16-19           -> height 4
    # col9: empty                      -> height 0, wall-well on the right edge
    g.board = build_board(
        {
            1: {15, 16, 17, 18, 19},
            2: {18, 19},
            5: {17, 19},
            8: {16, 17, 18, 19},
        }
    )

    feats = g.features()

    assert feats["heights"] == [0, 5, 2, 0, 0, 3, 0, 0, 4, 0]
    assert feats["holes"] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert feats["wells"] == [5, 0, 0, 0, 0, 0, 0, 0, 0, 4]
    assert feats["bumpiness"] == 24
    assert feats["aggregate_height"] == 14
    assert feats["holes_total"] == 1
    assert feats["max_height"] == 5


def test_empty_board_features_are_all_zero():
    g = Game(seed=0)
    feats = g.features()
    assert feats["heights"] == [0] * WIDTH
    assert feats["holes"] == [0] * WIDTH
    assert feats["wells"] == [0] * WIDTH
    assert feats["bumpiness"] == 0
    assert feats["aggregate_height"] == 0
    assert feats["holes_total"] == 0
    assert feats["max_height"] == 0
