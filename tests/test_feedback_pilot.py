import importlib.util
import json
import random
from pathlib import Path

import pytest

from scripts.stage6_feedback import paired_interval, schedule, training_command
from tetris.rl import trajectory_advantages


def test_baseline_is_bit_exact_with_preserved_original_for_ragged_groups():
    path = Path("experiments/stage6-feedback-v1/prechange/rl.py")
    spec = importlib.util.spec_from_file_location("tetris._original_rl", path)
    import sys
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    rng = random.Random(918)
    for _ in range(50):
        groups = [[rng.choice([0, -10, 1, 2, .25]) for _ in range(rng.randrange(1, 128))] for _ in range(4)]
        assert trajectory_advantages(groups) == original.trajectory_advantages(groups)


def test_paired_schedule_does_not_depend_on_generation_random_draws():
    bank = [{"seed": 12, "state_hash": "a", "state_id": "a"}, {"seed": 13, "state_hash": "b", "state_id": "b"}]
    a = schedule(6201, [12, 13], bank)
    for _ in range(1000):
        random.random()
    assert schedule(6201, [12, 13], bank) == a
    assert len(a) == 32
    assert all(bool(row["state_hash"]) == (row["update"] % 2 == 0) for row in a)


def test_crossed_interval_keeps_pairs_and_detects_known_effects():
    pytest.importorskip("numpy")
    ci = paired_interval([[1.] * 128] * 3, repetitions=100)
    assert ci["mean"] == 1
    assert ci["ci95"] == [1, 1]
    assert paired_interval([[0.] * 128] * 3, repetitions=100)["ci95"] == [0, 0]
    with pytest.raises(ValueError):
        paired_interval([[float("nan")]])


def test_historical_approval_is_rejected(tmp_path):
    from scripts.eval_feedback import approved_session
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"experiment": "SCALE10X", "status": "user_approved", "hard_limit_usd": 250}))
    with pytest.raises(ValueError, match="new explicit"):
        approved_session(path)


def test_commands_start_from_original_sft_and_keep_scheduler_horizon():
    r = {"question": "test", "initial_adapter": "runs/sft-v1/adapter", "base_model": "pinned",
         "base_model_revision": "abc", "benchmark_manifest": "benchmark", "training_seed_file": "train",
         "recovery_start_file": "bank", "recipe": {"updates": 32, "horizon": 128}}
    run = {"run_id": "pilot", "method": "fixed_zero", "seed": 6201}
    cmd = training_command(r, run)
    assert cmd[cmd.index("--adapter-dir")+1] == "runs/sft-v1/adapter"
    assert cmd[cmd.index("--frozen-sft-adapter-dir")+1] == "runs/sft-v1/adapter"
    assert cmd[cmd.index("--updates")+1] == "32"
    assert "--resume" not in cmd


def test_recovery_generator_uses_distinct_live_source_state():
    from scripts.stage6_feedback import recovery_state
    from tetris.rl import restore_game
    row = recovery_state(81_000_000)
    game = restore_game(row["seed"], row["action_prefix"], expected=row)
    assert not game.game_over and game.turn >= 8 and game.snapshot()["max_height"] >= 16
    assert row["split"] == "development"


def test_decision_needs_effect_uncertainty_replication_and_regression_guards():
    from scripts.analyze_feedback import decision
    analysis = {"minimum_useful_reduction": .1, "require_lower_bound_above": 0, "min_improving_seeds": 2}
    kwargs = dict(ordinary_ok=True, recovery_ok=True, signal_ok=True, analysis=analysis)
    ci = {"mean": .15, "ci95": [.01, .25], "per_training_seed": [.1, .2, .15]}
    assert decision(ci, **kwargs) == ("helps", True)
    for flag in ("ordinary_ok", "recovery_ok", "signal_ok"):
        assert decision(ci, **{**kwargs, flag: False}) == ("inconclusive", False)
    assert decision({**ci, "ci95": [0., .25]}, **kwargs) == ("inconclusive", False)
    assert decision({**ci, "per_training_seed": [.5, -.01, -.04]}, **kwargs) == ("inconclusive", False)
    assert decision({**ci, "mean": .09}, **kwargs) == ("inconclusive", False)
    assert decision({**ci, "mean": -.2, "ci95": [-.3, -.1]}, **kwargs) == ("hurts", False)
