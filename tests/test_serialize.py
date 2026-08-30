import pytest

from tetris.engine import Game
from tetris.serialize import parse_action, serialize_action, serialize_prompt


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


def test_parse_action_round_trips_every_combination():
    for rot in range(4):
        for x in range(10):
            assert parse_action(serialize_action(rot, x)) == (rot, x)


def test_parse_action_tolerates_surrounding_whitespace():
    assert parse_action("  Action: rot=1 x=4  \n") == (1, 4)


@pytest.mark.parametrize(
    "text",
    [
        "Action: rot=4 x=0",  # rot out of range
        "Action: rot=0 x=10",  # x out of range
        "Action: rot=1x=4",
        "rot=1 x=4",
        "Action: rot=1 x=4\nextra line",
        "",
        "garbage",
    ],
)
def test_parse_action_rejects_malformed_output(text):
    with pytest.raises(ValueError):
        parse_action(text)


def test_snapshot_prompt_is_produced_by_the_one_serializer():
    g = Game(seed=0)
    snap = g.snapshot()
    assert snap["prompt"] == serialize_prompt(g.features())

    g.current = "T"
    g.step(0, 3)
    snap2 = g.snapshot()
    assert snap2["prompt"] == serialize_prompt(g.features())
    assert snap2["prompt"] != snap["prompt"]
