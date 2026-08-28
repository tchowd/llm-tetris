from tetris.engine import Game
from tetris.pieces import PIECE_LETTERS


def _pick(game):
    """A simple survival-biased policy: land as flat as possible, so the
    game reliably survives long enough to observe several bags."""
    legal = game.legal_placements()
    return min(legal, key=lambda p: (p["landing_height"], p["rot"], p["x"]))


def reveal_sequence(seed, num_steps):
    g = Game(seed=seed)
    seq = [g.current, g.next]
    for _ in range(num_steps):
        if g.game_over:
            break
        p = _pick(g)
        g.step(p["rot"], p["x"])
        seq.append(g.next)
    return seq


def test_same_seed_gives_same_piece_sequence():
    a = reveal_sequence(seed=42, num_steps=30)
    b = reveal_sequence(seed=42, num_steps=30)
    assert a == b
    assert len(a) >= 21  # at least three full bags revealed


def test_different_seeds_can_diverge():
    a = reveal_sequence(seed=1, num_steps=30)
    b = reveal_sequence(seed=2, num_steps=30)
    assert a != b


def test_each_bag_is_a_permutation_of_all_seven_pieces():
    seq = reveal_sequence(seed=7, num_steps=30)
    full_bags = len(seq) // 7
    assert full_bags >= 3
    for i in range(full_bags):
        chunk = seq[i * 7 : (i + 1) * 7]
        assert sorted(chunk) == sorted(PIECE_LETTERS)
