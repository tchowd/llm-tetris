from tetris.engine import Game, HEIGHT, WIDTH


def fill_row_except(board, row, except_cols):
    for c in range(WIDTH):
        if c not in except_cols:
            board[row][c] = "X"


def test_single_line_clear_score_and_compaction():
    g = Game(seed=0)
    board = [["."] * WIDTH for _ in range(HEIGHT)]
    fill_row_except(board, 19, {4, 5})
    g.board = board
    g.current = "O"

    g.step(0, 4)

    assert g.lines == 1
    assert g.level == 1
    # drop distance for O from the top of an otherwise-empty board is 18
    # rows; line points: 100 * level(1); hard-drop bonus: 2 * 18.
    assert g.score == 100 * 1 + 2 * 18
    assert all(cell == "." for cell in g.board[0])
    assert all(cell == "." for row in g.board[:19] for cell in row)


def test_tetris_clears_four_lines_and_scores_800():
    g = Game(seed=0)
    board = [["."] * WIDTH for _ in range(HEIGHT)]
    for row in (16, 17, 18, 19):
        fill_row_except(board, row, {0})
    g.board = board
    g.current = "I"

    g.step(1, 0)  # vertical I, canonical rot=1, dropped straight into col 0

    assert g.lines == 4
    assert g.level == 1
    assert g.score == 800 * 1 + 2 * 16
    assert all(cell == "." for row in g.board for cell in row)


def test_level_increments_every_ten_lines():
    g = Game(seed=0)
    for _ in range(9):
        board = [["."] * WIDTH for _ in range(HEIGHT)]
        fill_row_except(board, 19, {4, 5})
        g.board = board
        g.current = "O"
        g.step(0, 4)
    assert g.lines == 9
    assert g.level == 1

    board = [["."] * WIDTH for _ in range(HEIGHT)]
    fill_row_except(board, 19, {4, 5})
    g.board = board
    g.current = "O"
    g.step(0, 4)
    assert g.lines == 10
    assert g.level == 2
