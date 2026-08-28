from tetris.engine import Game
from tetris.serialize import serialize_action, serialize_prompt


def test_serialize_prompt_matches_contract():
    feats = {
        "piece": "T",
        "next": "I",
        "heights": [0, 0, 2, 3, 3, 1, 0, 0, 0, 0],
        "holes": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "wells": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "bumpiness": 6,
    }
    expected = (
        "Piece: T\n"
        "Next: I\n"
        "Heights: 0 0 2 3 3 1 0 0 0 0\n"
        "Holes:   0 0 0 1 0 0 0 0 0 0\n"
        "Wells:   0 0 0 0 0 0 0 0 0 0\n"
        "Bumpiness: 6"
    )
    assert serialize_prompt(feats) == expected


def test_serialize_action():
    assert serialize_action(1, 4) == "Action: rot=1 x=4"
    assert serialize_action(0, 0) == "Action: rot=0 x=0"


def test_snapshot_prompt_is_produced_by_the_one_serializer():
    g = Game(seed=0)
    snap = g.snapshot()
    assert snap["prompt"] == serialize_prompt(g.features())

    g.current = "T"
    g.step(0, 3)
    snap2 = g.snapshot()
    assert snap2["prompt"] == serialize_prompt(g.features())
    assert snap2["prompt"] != snap["prompt"]
