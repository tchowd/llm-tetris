import pytest

from tetris.engine import Game


def test_game_over_stops_play_and_turn_counts_successful_locks():
    g = Game(seed=1)
    steps = 0
    while not g.game_over:
        legal = g.legal_placements()
        assert legal, "engine reports not-game-over but has no legal placements"
        p = legal[0]
        g.step(p["rot"], p["x"])
        steps += 1
        assert steps < 2000, "did not top out within a reasonable number of pieces"

    assert g.turn == steps
    assert g.legal_placements() == []

    with pytest.raises(ValueError):
        g.step(0, 0)


def test_manually_topped_out_board_has_no_legal_placements():
    g = Game(seed=0)
    g.board = [["X"] * 10 for _ in range(20)]
    g.current = "O"
    assert g.legal_placements() == []
