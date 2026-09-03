from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from scripts.check_episode_proof import audit_batch, independent_token_logprobs, projection
from tetris.engine import Game
from tetris.rl import (EpisodeRewardWeights, discounted_reward_to_go, episode_transition,
    state_hash, trajectory_advantages, validate_trajectory)
from tetris.serialize import serialize_action


def proof_batch():
    trajectories = []
    for index in range(4):
        game = Game(seed=20_000_000)
        trajectory = {"episode_id": str(index), "seed": 20_000_000,
            "start_actions": [], "start_state": game.snapshot(), "steps": []}
        for _ in range(3):
            before = game.snapshot()
            placement = game.legal_placements()[index]
            action = (placement["rot"], placement["x"])
            transition = episode_transition(game, action, EpisodeRewardWeights())
            after = game.step(*action)
            trajectory["steps"].append({"turn": before["turn"], "before_state_hash": state_hash(before),
                "serialized_prompt": before["prompt"], "after_state_hash": state_hash(after),
                "action": list(action), "raw_completion": serialize_action(*action), "parsed": True, "legal": True,
                "terminal": transition.terminal, "terminal_reason": transition.terminal_reason,
                "immediate_reward": transition.reward, "reward_components": transition.components})
        trajectory.update(score=game.score, lines=game.lines, pieces=3,
            episode_return=sum(s["immediate_reward"] for s in trajectory["steps"]))
        trajectory["replay_validation"] = validate_trajectory(trajectory)
        trajectories.append(trajectory)
    rewards = [[s["immediate_reward"] for s in t["steps"]] for t in trajectories]
    for trajectory, values, row_rewards in zip(trajectories, trajectory_advantages(rewards), rewards):
        for step, advantage, reward_to_go in zip(trajectory["steps"], values, discounted_reward_to_go(row_rewards)):
            step.update(advantage=advantage, reward_to_go=reward_to_go)
    return {"update": 1, "environment_seed": 20_000_000, "trajectories": trajectories}


def test_episode_proof_replays_reward_credit_and_rejects_corruption():
    recipe = {"group_size": 4, "horizon": 20, "gamma": .99}
    original = proof_batch()
    assert audit_batch(original, recipe, [20_000_000], []) == 12
    for key, value in (("advantage", 999), ("reward_to_go", 999), ("immediate_reward", 999), ("legal", False)):
        broken = copy.deepcopy(original)
        broken["trajectories"][0]["steps"][0][key] = value
        with pytest.raises(ValueError):
            audit_batch(broken, recipe, [20_000_000], [])
    original["update"] = 2
    with pytest.raises(ValueError, match="recovery bank"):
        audit_batch(original, recipe, [20_000_000], [])


def test_independent_logprob_audit_agrees_for_unequal_lengths():
    torch = pytest.importorskip("torch")
    from scripts.train_episode_rl import sequence_logprobs

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.table[:input_ids.shape[1]].unsqueeze(0).expand(input_ids.shape[0], -1, -1))

    model = Model()
    rows = [{"prompt_ids": [1, 2], "completion_ids": [3, 0]}, {"prompt_ids": [1], "completion_ids": [2]}]
    expected = sequence_logprobs(model, rows, 0, .8, return_tokens=True)
    actual = independent_token_logprobs(model, rows, 0, .8)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=1e-7, rtol=0)
    sum(x.mean() for x in actual).backward()
    assert model.table.grad is not None


def test_pilot_projection_assumes_full_length_and_worst_observed_rate():
    r = {"recipe": {"updates": 4}, "pilot_projection_startup_seconds": 120,
        "pilot_projection_safety_multiplier": 1.5, "hourly_usd": 1.05}
    p = {"episode_pilot": {"updates": 32, "group_size": 4, "horizon": 128, "max_training_hours": 1.5}}
    result = projection(r, p, [{"seconds": 10, "turns": 80}] * 4)
    assert result["full_length_pilot_turns"] == 16384
    assert result["projected_seconds"] == 3192
    assert result["within_training_limit"]
    result = projection(r, p, [{"seconds": 30, "turns": 80}] * 4)
    assert not result["within_training_limit"]
    with pytest.raises(ValueError):
        projection(r, p, [{"seconds": float("nan"), "turns": 80}] * 4)


def test_episode_pilot_requires_r2_but_preserves_negative_r1(tmp_path, monkeypatch):
    import json
    import scripts.check_episode_pilot as pilot
    from tetris.rl import file_sha256
    monkeypatch.chdir(tmp_path)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"protocol": "stage6-recovery-v1", "final_test_access": False,
        "sft": {"run_id": "r1"}, "episode_pilot": {"run_id": "r3", "updates": 32},
        "questions": ["data", "proof", "episode"], "episode_promotion": {"min_relative_score_gain": .03}}))
    gate_path = tmp_path / "runs/r1/rl/r1-gate.json"
    gate_path.parent.mkdir(parents=True)
    r1 = {"status": "not_passed", "checks": {"recovery": False}, "generated_at": "earlier"}
    gate_path.write_text(json.dumps(r1))
    proof_registration, proof_gate = tmp_path / "proof-registration.json", tmp_path / "proof-gate.json"
    proof_registration.write_text("{}")
    proof_gate.write_text("{}")
    monkeypatch.setattr(pilot, "assess_r1", lambda _: {**r1, "generated_at": "now"})
    monkeypatch.setattr(pilot, "validate_proof_report", lambda *args: {"protocol_sha256": file_sha256(protocol), "pilot_projection": {"within_training_limit": True}})
    result = pilot.register(protocol, proof_registration, proof_gate, 12)
    assert result["r1_status"] == "not_passed"
    assert result["recipe"]["updates"] == 32
    assert result["final_test_access"] is False
    with pytest.raises(ValueError, match="budget"):
        pilot.register(protocol, proof_registration, proof_gate, 90)
    monkeypatch.setattr(pilot, "validate_proof_report", lambda *args: (_ for _ in ()).throw(ValueError("R2 not passed")))
    with pytest.raises(ValueError, match="R2 not passed"):
        pilot.register(protocol, proof_registration, proof_gate, 12)


def test_proof_retry_preserves_recipe_tolerances_and_failed_evidence(tmp_path, monkeypatch):
    import json
    import scripts.check_episode_proof as proof
    from tetris.rl import file_sha256
    monkeypatch.chdir(tmp_path)
    protocol = tmp_path / "protocol.json"
    recipe = {"run_id": "original", "updates": 4, "horizon": 20}
    protocol.write_text(json.dumps({"protocol": "stage6-recovery-v1", "final_test_access": False,
        "data": {"data_dir": "data"}, "trajectory_proof": recipe}))
    (tmp_path / "data").mkdir(); (tmp_path / "data/manifest.json").write_text("{}")
    for name in ("scripts/train_episode_rl.py", "scripts/check_episode_proof.py", "tetris/rl.py", "tetris/recovery.py"):
        path = tmp_path / name; path.parent.mkdir(exist_ok=True); path.write_text(name)
    monkeypatch.setattr(proof, "validate_inputs", lambda *args: {"registration_sha256": file_sha256(protocol)})
    original = proof.register(protocol, 12)
    root = tmp_path / "runs/original/rl"; root.mkdir(parents=True)
    previous = root / "registration.json"; previous.write_text(json.dumps(original))
    (root / "block-state.json").write_text(json.dumps({"status": "failed"}))
    diagnosis = tmp_path / "runs/stage6-recovery-v1/rl/r2-gradient-diagnosis.json"
    diagnosis.parent.mkdir(parents=True)
    diagnosis.write_text(json.dumps({"default_repeat": {"equal": False}, "deterministic_repeat": {"equal": True}, "adapter_unchanged": True, "optimizer_updates": 0}))
    retry = proof.register(protocol, 13, previous)
    assert retry["recipe"] == original["recipe"] == recipe
    assert retry["run_id"] != original["run_id"]
    for name in ("logprob_absolute_tolerance", "resume_weights_absolute_tolerance", "pilot_projection_safety_multiplier", "pilot_projection_startup_seconds"):
        assert retry[name] == original[name]
    assert retry["retry_of_registration_sha256"] == file_sha256(previous)
    (root / "block-state.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="retained failed"):
        proof.register(protocol, 13, previous)
