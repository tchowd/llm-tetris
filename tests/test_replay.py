import random

from tetris.engine import Game


def play_random_legal_game(seed, agent_seed, max_steps=300):
    g = Game(seed=seed)
    agent_rng = random.Random(agent_seed)
    actions = []
    trace = []
    while not g.game_over and len(actions) < max_steps:
        legal = g.legal_placements()
        p = agent_rng.choice(legal)
        actions.append((p["rot"], p["x"]))
        g.step(p["rot"], p["x"])
        trace.append(
            {
                "board": [row[:] for row in g.board],
                "features": g.features(),
                "lines": g.lines,
                "score": g.score,
                "turn": g.turn,
                "game_over": g.game_over,
            }
        )
    return actions, trace


def replay(seed, actions):
    g = Game(seed=seed)
    trace = []
    for rot, x in actions:
        g.step(rot, x)
        trace.append(
            {
                "board": [row[:] for row in g.board],
                "features": g.features(),
                "lines": g.lines,
                "score": g.score,
                "turn": g.turn,
                "game_over": g.game_over,
            }
        )
    return g, trace


def test_replay_reproduces_many_random_legal_games_exactly():
    for seed in range(20):
        actions, original_trace = play_random_legal_game(seed=seed, agent_seed=seed * 1000 + 1)
        assert len(actions) > 0

        replayed_game, replayed_trace = replay(seed, actions)

        assert replayed_trace == original_trace
        assert replayed_game.turn == len(actions)


def test_replay_reproduces_a_full_game_to_death():
    seed = 5
    actions, original_trace = play_random_legal_game(seed=seed, agent_seed=99, max_steps=2000)
    assert original_trace[-1]["game_over"] is True

    replayed_game, replayed_trace = replay(seed, actions)

    assert replayed_trace == original_trace
    assert replayed_game.game_over is True
    assert replayed_game.turn == len(actions)
