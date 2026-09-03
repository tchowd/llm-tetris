from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tetris.engine import Game
from tetris.rl import (
    DenseRewardWeights,
    EpisodeRewardWeights,
    dense_transition,
    discounted_reward_to_go,
    episode_transition,
    grouped_policy_loss,
    paired_comparison,
    restore_game,
    runtime_budget,
    state_hash,
    trajectory_advantages,
    validate_seed_manifest,
    validate_entry_gate,
    validate_trajectory,
)

ROOT = Path(__file__).resolve().parent.parent


def test_runtime_budget_stops_on_projected_overrun_before_actual_limit():
    result = runtime_budget(elapsed_seconds=100, hourly_usd=1.05, max_hours=1,
                            dollar_limit=20, remaining_updates=100, seconds_per_update=40)
    assert result["stop"] and result["projected_seconds"] == 4100
    assert not runtime_budget(elapsed_seconds=100, hourly_usd=1.05, max_hours=1,
                              dollar_limit=20, remaining_updates=10, seconds_per_update=40)["stop"]
    assert runtime_budget(elapsed_seconds=100, hourly_usd=3600, max_hours=1,
                          dollar_limit=101, remaining_updates=1, seconds_per_update=2)["stop"]


def test_grpo_observation_checkpoint_roundtrip(tmp_path):
    from scripts.train_rl import load_observations, save_observations
    observed = {"completions": 16, "tokens": 48, "unique_actions": {(0, 1), (1, 2)},
                "components": {"illegal": [0.0, 1.0]}, "rewards": [1.0, -10.0]}
    path = tmp_path / "observations.json"
    save_observations(path, observed)
    assert load_observations(path) == observed


def line_clear_game() -> Game:
    game = Game(7)
    game.board = [["."] * 10 for _ in range(20)]
    game.board[-1] = ["."] + ["X"] * 9
    game.current = "I"
    game.next = "O"
    game.game_over = False
    return game


def test_dense_reward_components_on_known_line_clear_and_does_not_mutate():
    game = line_clear_game()
    before = copy.deepcopy(game.snapshot())
    weights = DenseRewardWeights(lines=2.0, holes=0.0, aggregate_height=0.0, bumpiness=0.0, illegal=9.0)
    result = dense_transition(game, (1, 0), weights)

    assert result.components["lines_cleared"] == 1
    assert result.reward == 2.0
    assert result.after["lines"] == 1
    assert game.snapshot() == before


def test_illegal_action_is_terminal_and_matches_registered_penalty():
    game = Game(9)
    dense = dense_transition(game, None, DenseRewardWeights(illegal=13.0))
    episode = episode_transition(game, (3, 99), EpisodeRewardWeights(illegal_penalty=17.0))

    assert dense.terminal and dense.terminal_reason == "illegal_action" and dense.reward == -13.0
    assert episode.terminal and episode.terminal_reason == "illegal_action" and episode.reward == -17.0
    assert game.turn == 0


def test_episode_step_reward_is_normalized_score_delta():
    game = line_clear_game()
    result = episode_transition(game, (1, 0), EpisodeRewardWeights(score_scale=100.0, death_penalty=2.0))
    assert result.components["normalized_score_delta"] > 0
    assert result.reward == result.components["normalized_score_delta"]


def test_reward_to_go_and_same_turn_group_normalization():
    assert discounted_reward_to_go([1.0, 2.0, 3.0], gamma=0.5) == [2.75, 3.5, 3.0]
    advantages = trajectory_advantages([[1.0, 0.0], [0.0, 1.0]], gamma=1.0)
    assert advantages[0][0] == pytest.approx(0.0)
    assert advantages[1][0] == pytest.approx(0.0)
    assert advantages[0][1] < 0 < advantages[1][1]


def test_positive_advantage_increases_contributing_action_probability():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=0.2)
    before = torch.softmax(logits.detach(), dim=0)[1].item()
    chosen_logp = torch.log_softmax(logits, dim=0)[1].reshape(1)
    loss = grouped_policy_loss(chosen_logp, chosen_logp.detach(), torch.tensor([1.0]), beta=0.0)
    loss.backward()
    optimizer.step()
    after = torch.softmax(logits.detach(), dim=0)[1].item()
    assert after > before


def test_frozen_stress_manifest_has_disjoint_seeds_and_replayable_states():
    manifest = json.loads((ROOT / "benchmarks/stress-v1/manifest.json").read_text())
    validation = validate_seed_manifest(manifest, stage3_ranges=[range(0, 3140)])
    assert validation["ok"]
    assert validation["counts"]["development_seeds"] == 20
    assert validation["counts"]["test_seeds"] == 100

    with (ROOT / "benchmarks/stress-v1/states.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 48
    development_sources = {row["seed"] for row in rows if row["split"] == "development"}
    test_sources = {row["seed"] for row in rows if row["split"] == "test"}
    assert development_sources.isdisjoint(test_sources)
    for row in rows:
        game = restore_game(row["seed"], row["action_prefix"], expected=row)
        assert state_hash(game.snapshot()) == row["state_hash"]


def test_seed_overlap_is_rejected():
    manifest = {
        "training_seeds": [1],
        "development_seeds": [2],
        "test_seeds": [3],
        "stage5_seeds": [1],
    }
    with pytest.raises(ValueError, match="seed overlap"):
        validate_seed_manifest(manifest)


def test_trajectory_replay_detects_drift():
    game = Game(123)
    start = game.snapshot()
    steps = []
    for _ in range(3):
        before = game.snapshot()
        action = (before["legal"][0]["rot"], before["legal"][0]["x"])
        after = game.step(*action)
        steps.append(
            {
                "turn": before["turn"],
                "action": list(action),
                "before_state_hash": state_hash(before),
                "after_state_hash": state_hash(after),
                "terminal": game.game_over,
                "terminal_reason": "topped_out" if game.game_over else None,
            }
        )
    trajectory = {"seed": 123, "start_actions": [], "start_state": start, "steps": steps}
    assert validate_trajectory(trajectory)["steps_checked"] == 3
    trajectory["steps"][1]["after_state_hash"] = "bad"
    with pytest.raises(ValueError, match="trajectory drift"):
        validate_trajectory(trajectory)


def test_paired_comparison_requires_identical_seeds_and_reports_ci():
    report = paired_comparison({1: 10.0, 2: 10.0, 3: 10.0}, {1: 11.0, 2: 12.0, 3: 13.0}, bootstrap_samples=500)
    assert report["mean_difference"] == 2.0
    assert report["relative_improvement"] == pytest.approx(0.2)
    assert report["bootstrap_95_ci"][0] > 0
    with pytest.raises(ValueError, match="paired seeds differ"):
        paired_comparison({1: 1.0}, {2: 1.0})


def test_entry_gate_rejects_frozen_policy_with_illegal_death(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"seeds": [1, 2], "cap": 500}))
    metrics = {"model": {"strict": {"deaths": 1, "parse_failure_rate": {"mean": 0.0}, "illegal_rate": {"mean": 0.0}, "illegal_action_deaths": 1, "lines": {"mean": 197.77}}}}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="entry gate failed"):
        validate_entry_gate(tmp_path / "manifest.json", [1, 2])


def test_weakened_sft_seeds_all_rngs_before_loading_data_or_initializing_lora(tmp_path, monkeypatch):
    import random
    import sys

    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from scripts import train_sft

    observed = []

    class BeforeModelInitialization(Exception):
        pass

    def capture_rngs(*args, **kwargs):
        observed.append((random.random(), float(numpy.random.random()), torch.rand(4).tolist()))
        raise BeforeModelInitialization

    monkeypatch.setattr(train_sft, "load_rows", capture_rngs)
    for index, seed in enumerate([12, 12, 13]):
        out_dir = tmp_path / f"weak-{index}"
        if index != 1:
            out_dir = out_dir / "rl"
        monkeypatch.setattr(sys, "argv", ["train_sft.py", "--data-dirs", str(tmp_path),
            "--out-dir", str(out_dir), "--seed", str(seed), "--device", "cpu"])
        with pytest.raises(BeforeModelInitialization):
            train_sft.main()
        first_event = json.loads((out_dir / "events.jsonl").read_text().splitlines()[0])
        assert first_event["run_id"] == f"weak-{index}"
    assert observed[0] == observed[1]
    assert observed[0] != observed[2]
