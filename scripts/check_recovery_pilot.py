#!/usr/bin/env python3
"""Judge R1 against its new registration; never reinterpret the old E3 gate."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analyze_stage6 import game_metric
from scripts.check_e2_learning import evaluation, read_rows
from scripts.check_e4_pilot import frozen_stage5
from scripts.train_recovery_sft import validate_inputs
from tetris.rl import atomic_write_json, file_sha256, paired_comparison


def fresh_summary(path: Path, label: str, expected_adapter: str, registration: dict, registration_hash: str):
    manifest = json.loads((path / "manifest.json").read_text())
    states_path = Path(registration["data"]["data_dir"]) / "validation-starts.jsonl"
    for key, expected in {"status": "completed", "suite": "fresh_validation", "label": label,
                          "mode": "strict", "greedy": True, "cap": 200, "final_test_access": False,
                          "adapter_sha256": expected_adapter, "registration_sha256": registration_hash,
                          "states_sha256": file_sha256(states_path)}.items():
        if manifest.get(key) != expected:
            raise ValueError(f"fresh validation mismatch: {key}")
    if manifest["policy_metadata"].get("base_model_revision") != registration["base_model_revision"]:
        raise ValueError("fresh validation base revision changed")
    expected = {s["state_id"]: s for s in read_rows(states_path)}
    games, fixed = read_rows(path / "games.jsonl"), read_rows(path / "states.jsonl")
    if len(games) != len(expected) or {g["game_id"] for g in games} != set(expected):
        raise ValueError("incomplete fresh recovery games")
    if len(fixed) != len(expected) or {s["state_id"] for s in fixed} != set(expected):
        raise ValueError("incomplete fresh fixed states")
    for game in games:
        if game["starting_state"] != expected[game["game_id"]] or game["policy"] != label:
            raise ValueError("fresh starting-state identity changed")
    for row in fixed:
        if row["state_hash"] != expected[row["state_id"]]["state_hash"] or row["policy"] != label:
            raise ValueError("fresh fixed-state identity changed")
    result = json.loads((path / "metrics.json").read_text())
    recovery = result["recovery_rollouts"]
    if recovery["n_games"] != len(games) or recovery["illegal_action_deaths"] != sum(g["death_reason"] == "illegal_action" for g in games) or recovery["deaths"] != sum(g["died"] for g in games):
        raise ValueError("fresh summary disagrees with saved outcomes")
    return {"deaths": recovery["deaths"], "illegal_deaths": recovery["illegal_action_deaths"],
            "parse_failure_rate": recovery["parse_failure_rate"]["mean"],
            "fixed_parse_rate": sum(s["parsed"] for s in fixed) / len(fixed),
            "fixed_legal_rate": sum(s["legal"] for s in fixed) / len(fixed),
            "manifest_sha256": file_sha256(path / "manifest.json")}


def recovery_checks(candidate: dict, baseline: dict, fresh: dict, fresh_baseline: dict, stage5: dict, rules: dict):
    long, recovery = candidate["all_metrics"]["long_horizon"], candidate["all_metrics"]["recovery_rollouts"]
    original_score = baseline["all_metrics"]["long_horizon"]["score_per_100_pieces"]["mean"]
    score = long["score_per_100_pieces"]["mean"]
    return {"fixed_parse_ceiling": candidate["fixed_parse_rate"] == 1, "fixed_legal_ceiling": candidate["fixed_legal_rate"] == 1,
            "long_deaths_zero": long["deaths"] == 0, "long_parse_ceiling": long["parse_failure_rate"]["mean"] == 0,
            "long_legality_ceiling": long["illegal_rate"]["mean"] == 0,
            "long_score_noninferiority": math.isfinite(score) and score >= original_score * rules["min_long_score_ratio_to_sft"],
            "recovery_illegal_deaths_halved": recovery["illegal_action_deaths"] <= rules["max_original_recovery_illegal_deaths"],
            "recovery_total_deaths_nonregression": recovery["deaths"] <= rules["max_original_recovery_deaths"],
            "recovery_parse_ceiling": recovery["parse_failure_rate"]["mean"] == 0,
            "fresh_illegal_deaths_improved": fresh["illegal_deaths"] < fresh_baseline["illegal_deaths"] if fresh_baseline["illegal_deaths"] else fresh["illegal_deaths"] == 0,
            "fresh_total_deaths_nonregression": fresh["deaths"] <= fresh_baseline["deaths"],
            "fresh_parse_ceiling": fresh["parse_failure_rate"] == 0 and fresh["fixed_parse_rate"] == 1,
            "fresh_fixed_legality_ceiling": fresh["fixed_legal_rate"] == 1, "stage5_noninferiority": stage5["passed"]}


def assess(registration_path: Path):
    r = json.loads(registration_path.read_text())
    data_dir = Path(r["data"]["data_dir"])
    data = validate_inputs(r, data_dir)
    root = Path("runs") / r["sft"]["run_id"] / "rl"
    training = json.loads((root / "manifest.json").read_text())
    registration_hash = file_sha256(registration_path)
    if data["registration_sha256"] != registration_hash or training["dataset_manifest_sha256"] != file_sha256(data_dir / "manifest.json") or not training.get("frozen_sft_unchanged"):
        raise ValueError("training data or frozen control identity changed")
    if training["status"] != "completed" or training["registration_sha256"] != registration_hash or training["recipe"] != r["sft"] or training["completed_updates"] != r["sft"]["updates"] or training["adapter_sha256"] != r["frozen_sft_adapter_sha256"]:
        raise ValueError("training incomplete or registration changed")
    baseline_path = Path("runs/stage6-e0/rl/stress-development")
    baseline, candidate = evaluation(baseline_path, "sft", r), evaluation(root / "stress-development", "r1", r)
    if candidate["adapter_sha256"] != training["output_adapter_sha256"] or baseline["adapter_sha256"] != r["frozen_sft_adapter_sha256"]:
        raise ValueError("evaluated adapter mismatch")
    fresh_baseline = fresh_summary(Path("runs/stage6-recovery-v1/rl/fresh-sft"), "sft", r["frozen_sft_adapter_sha256"], r, registration_hash)
    fresh = fresh_summary(root / "fresh-recovery", "r1", training["output_adapter_sha256"], r, registration_hash)
    benchmark = json.loads(Path(r["benchmark_manifest"]).read_text())
    stage5 = frozen_stage5(root / "stage5", "r1", training["output_adapter_sha256"], r["base_model_revision"], benchmark["decision_rules"])
    checks = recovery_checks(candidate, baseline, fresh, fresh_baseline, stage5, r["sft_learning_gate"])
    comparison = paired_comparison(game_metric(baseline_path, "sft")[0], game_metric(root / "stress-development", "r1")[0], bootstrap_samples=10000, seed=6104)
    report = {"experiment": "R1", "status": "passed" if all(checks.values()) else "not_passed", "checks": checks,
              "registration_sha256": registration_hash, "training_manifest_sha256": file_sha256(root / "manifest.json"),
              "baseline": baseline, "candidate": candidate, "fresh_baseline": fresh_baseline, "fresh_candidate": fresh,
              "stage5": stage5, "paired_score_comparison": comparison, "research_complete": False, "deployment_authorized": False,
              "next": "Retain R1 as a data-coverage result, then run the independently authorized R2 multi-turn proof from original SFT.",
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite a completed gate")
    report = assess(args.registration)
    atomic_write_json(args.out, report)
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
