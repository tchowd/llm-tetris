from __future__ import annotations

import json

import pytest

from scripts.analyze_stage6 import stage5_gate, validate_evaluation
from scripts.check_e2_learning import baseline_checks, evaluation, learning_checks
from pathlib import Path


def test_promotion_rejects_smoke_cap(tmp_path):
    benchmark = {"long_horizon": {"development_cap": 2000}, "development_seeds": [1, 2]}
    manifest = {
        "suite": "development",
        "benchmark_manifest_sha256": "abc",
        "registered_cap": 2000,
        "cap": 10,
        "seeds": [1, 2],
        "mode": "strict",
        "greedy": True,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="smoke/changed cap"):
        validate_evaluation(tmp_path, suite="development", benchmark=benchmark, benchmark_hash="abc")


def test_stage5_gate_requires_frozen_seeds_and_thresholds(tmp_path):
    row = {
        "lines": {"mean": 196.8},
        "deaths": 0,
        "parse_failure_rate": {"mean": 0.0},
        "illegal_rate": {"mean": 0.0},
    }
    (tmp_path / "metrics.json").write_text(json.dumps({"candidate": {"strict": row}}))
    (tmp_path / "manifest.json").write_text(json.dumps({"cap": 500, "seeds": list(range(10_000_000, 10_000_100))}))
    rules = {"stage5_min_mean_lines": 196.77, "stage5_max_deaths": 0, "stage5_max_parse_failures": 0, "stage5_max_illegal_actions": 0}
    assert stage5_gate(tmp_path / "metrics.json", "candidate", rules)["passed"]
    row["deaths"] = 1
    (tmp_path / "metrics.json").write_text(json.dumps({"candidate": {"strict": row}}))
    assert not stage5_gate(tmp_path / "metrics.json", "candidate", rules)["passed"]


def e2_rules():
    return {
        "minimum_mean_reward_gain": 0.05, "minimum_strong_minus_weak_baseline_reward": 0.05,
        "baseline_fixed_parse_floor": 0.95, "baseline_fixed_legal_floor": 0.90,
        "baseline_long_parse_floor": 0.99, "baseline_long_legal_floor": 0.95,
        "candidate_fixed_parse_floor": 0.95, "candidate_fixed_legal_floor": 0.95,
        "require_no_validity_regression": True, "bootstrap_samples": 100, "bootstrap_seed": 6002,
    }


def e2_summary(reward, adapter="weak"):
    return {"mean_reward": reward, "adapter_sha256": adapter, "rewards": {"a": reward, "b": reward},
            "fixed_parse_rate": 1.0, "fixed_legal_rate": 1.0, "long_parse_rate": 1.0, "long_legal_rate": 1.0}


def test_e2_requires_weaker_but_valid_initialization():
    weak, strong = e2_summary(-1), e2_summary(0, "strong")
    assert all(baseline_checks(weak, strong, e2_rules()).values())
    assert not baseline_checks(strong, strong, e2_rules())["different_adapter"]
    assert not baseline_checks(e2_summary(0), strong, e2_rules())["measurably_weaker"]
    weak["fixed_parse_rate"] = 0.5
    assert not baseline_checks(weak, strong, e2_rules())["fixed_parse_floor"]


def test_e2_target_gain_does_not_override_format_regression():
    weak, candidate = e2_summary(-1), e2_summary(-0.8)
    checks, comparison = learning_checks(weak, candidate, e2_rules())
    assert all(checks.values())
    assert comparison["mean_difference"] == pytest.approx(0.2)
    assert len(comparison["differences"]) == 2
    candidate["long_legal_rate"] = 0.999
    checks, _ = learning_checks(weak, candidate, e2_rules())
    assert checks["target_improved"] and not checks["long_legal_non_regression"]
    checks, _ = learning_checks(weak, weak, e2_rules())
    assert not checks["target_improved"]
    candidate["rewards"]["c"] = 1
    with pytest.raises(ValueError, match="paired states differ"):
        learning_checks(weak, candidate, e2_rules())


def test_e2_rejects_incomplete_and_duplicate_evaluation(tmp_path):
    from tetris.rl import file_sha256
    benchmark = {"long_horizon": {"development_cap": 2000}, "development_seeds": [1]}
    bp, sp = tmp_path / "benchmark.json", tmp_path / "states.jsonl"
    bp.write_text(json.dumps(benchmark))
    sp.write_text(json.dumps({"state_id": "a", "split": "development", "state_hash": "h", "kind": "recovery"}) + "\n")
    reg = {"benchmark_manifest": str(bp), "benchmark_manifest_sha256": file_sha256(bp), "states_path": str(sp), "base_model_revision": "revision"}
    out = tmp_path / "evaluation"
    out.mkdir()
    manifest = {"suite": "development", "benchmark_manifest_sha256": file_sha256(bp), "cap": 2000, "registered_cap": 2000,
                "recovery_cap": 200, "seeds": [1], "mode": "strict", "greedy": True, "states_sha256": file_sha256(sp),
                "status": "running", "policy_metadata": {"weak": {"base_model_revision": "revision"}}}
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        evaluation(out, "weak", reg)
    manifest["status"] = "passed"
    (out / "manifest.json").write_text(json.dumps(manifest))
    row = {"policy": "weak", "state_id": "a", "state_hash": "h", "dense_reward": 0}
    (out / "states.jsonl").write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate fixed-state"):
        evaluation(out, "weak", reg)


def test_e2_entrypoint_runs_directly():
    import subprocess
    import sys
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, str(root / "scripts/check_e2_learning.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--baseline-gate" in result.stdout


def test_e3_selects_smallest_valid_kl_not_largest_score():
    from scripts.select_e3_kl import select_smallest
    rows = [{"kl_beta": 0.01, "checks": {"valid": True}, "score": 1},
            {"kl_beta": 0.05, "checks": {"valid": True}, "score": 100},
            {"kl_beta": 0.1, "checks": {"valid": True}, "score": 200}]
    assert select_smallest(rows, [0.01, 0.05, 0.1])["kl_beta"] == 0.01
    rows[0]["checks"]["valid"] = False
    assert select_smallest(rows, [0.01, 0.05, 0.1])["kl_beta"] == 0.05
    for row in rows:
        row["checks"]["valid"] = False
    assert select_smallest(rows, [0.01, 0.05, 0.1]) is None
    with pytest.raises(ValueError, match="exactly once"):
        select_smallest(rows[:2], [0.01, 0.05, 0.1])
    with pytest.raises(ValueError, match="exactly once"):
        select_smallest([rows[0]] * 3, [0.01, 0.05, 0.1])


def test_e3_guardrails_reject_recovery_illegal_actions_and_long_deaths():
    from scripts.select_e3_kl import validity_checks
    summary = e2_summary(0)
    summary["all_metrics"] = {"long_horizon": {"deaths": 0, "illegal_action_deaths": 0},
        "recovery_rollouts": {"parse_failure_rate": {"mean": 0}, "illegal_rate": {"mean": 0}, "illegal_action_deaths": 0}}
    assert all(validity_checks(summary).values())
    summary["all_metrics"]["recovery_rollouts"]["illegal_rate"]["mean"] = 0.01
    assert not validity_checks(summary)["recovery_illegal_actions_zero"]
    summary["all_metrics"]["long_horizon"]["deaths"] = 1
    assert not validity_checks(summary)["long_deaths_zero"]


def test_e3_rejects_changed_training_recipe_or_reference():
    from scripts.select_e3_kl import validate_training
    r = {"frozen_sft_adapter_sha256": "sft", "base_model_revision": "rev", "benchmark_manifest_sha256": "bench",
         "training": {"completed_updates": 256, "training_seed": 0}, "reward_weights": {"lines": 1},
         "expected_completions_per_candidate": 4096}
    m = {"experiment": "E3", "status": "completed", "initialization_kind": "sft", "adapter_sha256": "sft",
         "frozen_sft_adapter_sha256": "sft", "base_model_revision": "rev", "benchmark_manifest_sha256": "bench",
         "kl_beta": 0.01, "completed_updates": 256, "training_seed": 0, "reward": {"weights": {"lines": 1}},
         "reference_frozen": True, "reference_adapter_sha256_before": "ref", "reference_adapter_sha256_after": "ref",
         "rollout_statistics": {"completions": 4096}}
    validate_training(m, r, 0.01)
    m["adapter_sha256"] = "weak"
    with pytest.raises(ValueError, match="adapter_sha256"):
        validate_training(m, r, 0.01)
    m["adapter_sha256"] = "sft"
    m["reference_adapter_sha256_after"] = "changed"
    with pytest.raises(ValueError, match="reference weights changed"):
        validate_training(m, r, 0.01)


def test_e3_entrypoint_runs_directly():
    import subprocess
    import sys
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, str(root / "scripts/select_e3_kl.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def e4_entry_fixture(tmp_path):
    import copy
    from tetris.rl import file_sha256
    from scripts.select_e3_kl import validity_checks
    root = Path(__file__).resolve().parent.parent
    r = {
        "experiment": "E3", "base_model": "model", "base_model_revision": "revision", "frozen_sft_adapter_sha256": "sft",
        "benchmark_manifest": str(root / "benchmarks/stress-v1/manifest.json"),
        "states_path": str(root / "benchmarks/stress-v1/states.jsonl"), "suite": "development", "baseline_path": "baseline",
        "training": {"training_seed": 0, "max_updates": 256, "completed_updates": 256,
                     "num_states": 256, "group_size": 4, "batch_size": 4, "gradient_accumulation": 4,
                     "learning_rate": 1e-6, "temperature": 1, "max_completion_length": 16,
                     "sampling": {"temperature": 1, "top_p": 1, "top_k": 0}},
        "reward_weights": {"lines": 1, "holes": 1, "aggregate_height": .05, "bumpiness": .02, "illegal": 10}, "final_test_access": False,
        "candidates": [{"run_id": f"candidate{i}", "label": f"c{i}", "kl_beta": beta} for i, beta in enumerate((.01, .05, .1))],
    }
    for key, hash_key in (("benchmark_manifest", "benchmark_manifest_sha256"), ("states_path", "states_sha256")):
        r[key] = str((root / r[key]).resolve())
        r[hash_key] = file_sha256(Path(r[key]))
    summary = e2_summary(0, "sft")
    summary["all_metrics"] = {"long_horizon": {"deaths": 0, "illegal_action_deaths": 0},
        "recovery_rollouts": {"parse_failure_rate": {"mean": 0}, "illegal_rate": {"mean": 0}, "illegal_action_deaths": 0}}
    ep, sp = tmp_path / "e3-registration.json", tmp_path / "selection.json"
    ep.write_text(json.dumps(r))
    selection = {"status": "passed", "registration_sha256": file_sha256(ep), "selected_kl_beta": .01,
                 "selected_run_id": r["candidates"][0]["run_id"],
                 "candidates": [{**item, "checks": validity_checks(summary), "evaluation": copy.deepcopy(summary)} for item in r["candidates"]]}
    sp.write_text(json.dumps(selection))
    return ep, sp


def test_e4_registration_requires_complete_e3_and_preserves_recipe(tmp_path):
    from scripts.check_e4_pilot import register, validate_registration
    ep, sp = e4_entry_fixture(tmp_path)
    r = register(ep, sp, 10)
    validate_registration(r)
    assert r["kl_beta"] == .01 and r["training"]["completed_updates"] == 512
    assert r["expected_completions_per_candidate"] == 8192 and not r["final_test_access"]
    assert r["budgets"]["development_timeout_minutes"] == 75
    assert r["budgets"]["block_max_hours_including_sync"] == 3
    r["reward_weights"]["lines"] = 2
    with pytest.raises(ValueError, match="recipe differs"):
        validate_registration(r)
    with pytest.raises(ValueError, match="budget"):
        register(ep, sp, 99)
    with pytest.raises(ValueError, match="budget"):
        register(ep, sp, float("nan"))
    selection = json.loads(sp.read_text())
    selection["candidates"].pop()
    sp.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="identities"):
        register(ep, sp, 10)


def test_e4_rejects_wrong_or_falsified_kl_selection(tmp_path):
    from scripts.check_e4_pilot import register
    ep, sp = e4_entry_fixture(tmp_path)
    selection = json.loads(sp.read_text())
    selection["selected_kl_beta"] = .1
    sp.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="smallest"):
        register(ep, sp, 10)
    selection["selected_kl_beta"] = .01
    selection["candidates"][0]["checks"]["long_deaths_zero"] = False
    sp.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="disagree"):
        register(ep, sp, 10)


def test_e4_gain_never_overrides_validity_or_frozen_gate(tmp_path):
    from scripts.check_e4_pilot import promotion_checks
    _, sp = e4_entry_fixture(tmp_path)
    summary = json.loads(sp.read_text())["candidates"][0]["evaluation"]
    assert all(promotion_checks({"relative_improvement": .04}, summary, {"passed": True}, .03).values())
    assert not promotion_checks({"relative_improvement": .02}, summary, {"passed": True}, .03)["development_gain"]
    assert not promotion_checks({"relative_improvement": .5}, summary, {"passed": False}, .03)["stage5_non_inferiority"]
    summary["all_metrics"]["long_horizon"]["deaths"] = 1
    assert not promotion_checks({"relative_improvement": .5}, summary, {"passed": True}, .03)["long_deaths_zero"]


def test_e4_stage5_requires_adapter_identity_full_cohort_and_saved_validity(tmp_path):
    from scripts.check_e4_pilot import frozen_stage5
    rules = {"stage5_min_mean_lines": 196.77, "stage5_max_deaths": 0, "stage5_max_parse_failures": 0, "stage5_max_illegal_actions": 0}
    manifest = {"status": "passed", "modes": ["strict"], "greedy": True, "adapter_sha256": "adapter",
                "policy_metadata": {"e4": {"base_model_revision": "rev"}}, "cap": 500, "seeds": list(range(10_000_000, 10_000_100))}
    metrics = {"lines": {"mean": 197}, "deaths": 0, "n_games": 100, "parse_failure_rate": {"mean": 0}, "illegal_rate": {"mean": 0}}
    rows = [{"policy": "e4", "mode": "strict", "seed": seed, "lines": 197, "pieces": 500, "died": False,
             "incidents": [], "raw_actions": [[0, 0]] * 500, "actions": [[0, 0]] * 500} for seed in manifest["seeds"]]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "metrics.json").write_text(json.dumps({"e4": {"strict": metrics}}))
    games_path = tmp_path / "games.jsonl"
    games_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert frozen_stage5(tmp_path, "e4", "adapter", "rev", rules)["passed"]
    with pytest.raises(ValueError, match="different model"):
        frozen_stage5(tmp_path, "e4", "wrong", "rev", rules)
    rows[0]["incidents"] = [0]
    games_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert not frozen_stage5(tmp_path, "e4", "adapter", "rev", rules)["passed"]
    games_path.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        frozen_stage5(tmp_path, "e4", "adapter", "rev", rules)


def test_e4_entrypoint_runs_directly():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "scripts/check_e4_pilot.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mutate_adapter", [False, True])
def test_stage5_metadata_keeps_model_identity_and_refuses_overwrites(tmp_path, monkeypatch, mutate_adapter):
    import sys
    from scripts import eval_closed_loop
    from tetris.rl import directory_sha256
    from tetris.rollout import random_legal_policy
    adapter, output = tmp_path / "adapter", tmp_path / "evaluation"
    adapter.mkdir()
    weight = adapter / "adapter_model.safetensors"
    weight.write_bytes(b"test-only-placeholder")
    expected_hash = directory_sha256(adapter)
    policy = random_legal_policy()

    def load(*args):
        def choose(snapshots, teacher_infos):
            if mutate_adapter:
                weight.write_bytes(b"changed-test-placeholder")
            return policy(snapshots, teacher_infos)
        choose.metadata = {"base_model_revision": "test-revision"}
        return choose

    monkeypatch.setattr(eval_closed_loop, "build_model_policy", load)
    monkeypatch.setattr(sys, "argv", ["eval_closed_loop.py", "--policies", "model", "--modes", "strict",
        "--model-label", "e4", "--adapter-dir", str(adapter), "--out-dir", str(output),
        "--num-seeds", "2", "--cap", "4", "--device", "cpu"])
    if mutate_adapter:
        with pytest.raises(RuntimeError, match="adapter changed"):
            eval_closed_loop.main()
        assert not (output / "manifest.json").exists()
    else:
        eval_closed_loop.main()
        m = json.loads((output / "manifest.json").read_text())
        assert m["adapter_sha256"] == expected_hash
        assert m["policy_metadata"]["e4"]["base_model_revision"] == "test-revision"
        assert m["greedy"] and m["status"] == "passed"
    before = (output / "games.jsonl").read_bytes()
    with pytest.raises(SystemExit, match="already exist"):
        eval_closed_loop.main()
    assert (output / "games.jsonl").read_bytes() == before
