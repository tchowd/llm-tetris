from __future__ import annotations

import json

import pytest

from scripts.audit_recovery_failures import audit_record
from scripts.check_recovery_pilot import recovery_checks
from tetris.engine import Game
from tetris.recovery import load_start_bank, make_group_starts, placement_failure
from tetris.rl import record_state
from tetris.serialize import serialize_action


def starting_state(seed=20_000_000):
    game = Game(seed=seed)
    actions = []
    for _ in range(4):
        p = game.legal_placements()[0]
        action = [p["rot"], p["x"]]
        actions.append(action)
        game.step(*action)
    state = record_state(game, actions, state_id="train-example")
    state["split"] = "train"
    return state


def test_recovery_group_has_identical_independent_replayed_games():
    state = starting_state()
    games = make_group_starts(state["seed"], 4, state)
    assert all(g.snapshot()["board"] == state["board"] and g.turn == state["turn"] for g in games)
    p = games[0].legal_placements()[0]
    games[0].step(p["rot"], p["x"])
    assert games[1].turn == state["turn"]
    with pytest.raises(ValueError, match="seed"):
        make_group_starts(state["seed"] + 1, 4, state)


def test_training_start_bank_rejects_eval_seeds_and_tampering(tmp_path):
    state = starting_state()
    path = tmp_path / "starts.jsonl"
    path.write_text(json.dumps(state) + "\n")
    assert load_start_bank(path, [state["seed"]]) == [state]
    with pytest.raises(ValueError, match="training partition"):
        load_start_bank(path, [state["seed"] + 1])
    state["split"] = "eval"
    path.write_text(json.dumps(state) + "\n")
    with pytest.raises(ValueError, match="training partition"):
        load_start_bank(path, [state["seed"]])
    state["split"] = "train"
    state["state_hash"] = "wrong"
    path.write_text(json.dumps(state) + "\n")
    with pytest.raises(ValueError, match="invalid recovery"):
        load_start_bank(path, [state["seed"]])


def test_failure_classification_checks_coordinates_not_membership_in_dicts():
    game = Game(seed=0)
    game.current = "S"
    assert placement_failure(game, (0, 8)) == "outside_board"
    assert placement_failure(game, (0, 0)) == "legal"
    assert placement_failure(game, None) == "parse_failure"
    game.board[0][1] = "X"
    assert placement_failure(game, (0, 0)) == "blocked_at_top"
    assert game.legal_placements()  # illegal choice is not unavoidable game over


def test_audit_rejects_a_legal_action_mislabeled_as_illegal():
    game = Game(seed=0)
    p = game.legal_placements()[0]
    row = {"seed": 0, "actions": [], "raw_model_output": [serialize_action(p["rot"], p["x"])]}
    with pytest.raises(ValueError, match="actually legal"):
        audit_record(row)


def test_recovery_research_gate_is_distinct_from_production_promotion():
    candidate = {"fixed_parse_rate": 1, "fixed_legal_rate": 1, "all_metrics": {
        "long_horizon": {"deaths": 0, "parse_failure_rate": {"mean": 0}, "illegal_rate": {"mean": 0}, "score_per_100_pieces": {"mean": 100}},
        "recovery_rollouts": {"deaths": 7, "illegal_action_deaths": 3, "parse_failure_rate": {"mean": 0}}}}
    fresh = {"deaths": 3, "illegal_deaths": 1, "parse_failure_rate": 0, "fixed_parse_rate": 1, "fixed_legal_rate": 1}
    baseline = {**fresh, "illegal_deaths": 2}
    rules = {"min_long_score_ratio_to_sft": .99, "max_original_recovery_illegal_deaths": 3, "max_original_recovery_deaths": 8}
    assert all(recovery_checks(candidate, candidate, fresh, baseline, {"passed": True}, rules).values())
    fresh["illegal_deaths"] = 2
    assert not recovery_checks(candidate, candidate, fresh, baseline, {"passed": True}, rules)["fresh_illegal_deaths_improved"]


@pytest.mark.parametrize("status", ["passed", "not_passed"])
def test_recovery_checker_cli_preserves_gate_result(tmp_path, monkeypatch, status):
    import sys
    import scripts.check_recovery_pilot as checker
    out = tmp_path / "gate.json"
    monkeypatch.setattr(checker, "assess", lambda _: {"status": status, "checks": {"example": status == "passed"}})
    monkeypatch.setattr(sys, "argv", ["check_recovery_pilot", "--registration", "unused.json", "--out", str(out)])
    if status == "passed":
        checker.main()
    else:
        with pytest.raises(SystemExit) as error:
            checker.main()
        assert error.value.code == 2
    assert json.loads(out.read_text())["status"] == status


def test_sampler_uses_recovery_start_and_reports_incremental_score():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from scripts.train_episode_rl import sample_group
    from tetris.rl import EpisodeRewardWeights
    from tetris.teacher import pick
    prefix_game = Game(seed=20_000_000)
    prefix = []
    for _ in range(32):
        move = pick(prefix_game.snapshot(), prefix_game.legal_placements())
        prefix.append(list(move))
        prefix_game.step(*move)
        if prefix_game.lines:
            break
    assert prefix_game.lines > 0  # catches accidentally reporting cumulative prefix lines
    state = record_state(prefix_game, prefix, state_id="train-with-lines")
    state["split"] = "train"
    game = make_group_starts(state["seed"], 2, state)[0]
    p = game.legal_placements()[0]
    action = (p["rot"], p["x"])

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 7

        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]["content"]

        def __call__(self, value, **kwargs):
            if isinstance(value, str):
                return {"input_ids": [1]}
            return {"input_ids": torch.ones((len(value), 1), dtype=torch.long),
                    "attention_mask": torch.ones((len(value), 1), dtype=torch.long)}

        def batch_decode(self, ids, **kwargs):
            return [serialize_action(*action)] * len(ids)

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(8))

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, 8))

        def generate(self, input_ids, **kwargs):
            return torch.cat([input_ids, torch.tensor([[2, 7]] * len(input_ids))], dim=1)

    trajectories = sample_group(Policy(), Policy(), Tokenizer(), seed=state["seed"], group_size=2,
        horizon=1, temperature=1, reward_weights=EpisodeRewardWeights(), start=state)
    game.step(*action)
    for row in trajectories:
        assert row["start_actions"] == state["action_prefix"]
        assert row["steps"][0]["turn"] == state["turn"]
        assert row["score"] == game.score - state["score"]
        assert row["lines"] == game.lines - state["lines"]
        assert row["replay_validation"]["ok"]
