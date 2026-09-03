#!/usr/bin/env python3
"""Validate the pre-registered E2 learning smoke; never consumes final test data."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

# Direct script execution otherwise exposes only scripts/, not its parent.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analyze_stage6 import validate_evaluation
from tetris.rl import atomic_write_json, bootstrap_mean_ci, file_sha256


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluation(path: Path, label: str, registration: dict) -> dict:
    benchmark_path = Path(registration["benchmark_manifest"])
    benchmark = json.loads(benchmark_path.read_text())
    if file_sha256(benchmark_path) != registration["benchmark_manifest_sha256"]:
        raise ValueError("registered benchmark hash changed")
    manifest = validate_evaluation(path, suite="development", benchmark=benchmark,
                                   benchmark_hash=registration["benchmark_manifest_sha256"])
    states_path = Path(registration["states_path"])
    if manifest.get("status") != "passed" or manifest.get("states_sha256") != file_sha256(states_path):
        raise ValueError("evaluation incomplete or fixed states changed")
    if manifest["policy_metadata"][label].get("base_model_revision") != registration["base_model_revision"]:
        raise ValueError("base model revision changed")
    expected = {r["state_id"]: r for r in read_rows(states_path) if r["split"] == "development"}
    rows = [r for r in read_rows(path / "states.jsonl") if r["policy"] == label]
    if len(rows) != len(expected) or {r["state_id"] for r in rows} != set(expected):
        raise ValueError("incomplete or duplicate fixed-state cohort")
    for row in rows:
        if row["state_hash"] != expected[row["state_id"]]["state_hash"]:
            raise ValueError("fixed-state hash mismatch")
        if not math.isfinite(row["dense_reward"]):
            raise ValueError("non-finite dense reward")
    games = [r for r in read_rows(path / "games.jsonl") if r["policy"] == label]
    if len(games) != len(benchmark["development_seeds"]) or sorted(r["seed"] for r in games) != sorted(benchmark["development_seeds"]):
        raise ValueError("incomplete or duplicate development game cohort")
    recovery = [r for r in read_rows(path / "recovery_games.jsonl") if r["policy"] == label]
    recovery_ids = {k for k, r in expected.items() if r["kind"] == "recovery"}
    if len(recovery) != len(recovery_ids) or {r["game_id"] for r in recovery} != recovery_ids:
        raise ValueError("incomplete or duplicate recovery cohort")
    metrics = json.loads((path / "metrics.json").read_text())[label]
    long = metrics["long_horizon"]
    return {
        "adapter_sha256": manifest["adapter_sha256"],
        "manifest_sha256": file_sha256(path / "manifest.json"),
        "rewards": {r["state_id"]: r["dense_reward"] for r in rows},
        "mean_reward": statistics.mean(r["dense_reward"] for r in rows),
        "fixed_parse_rate": statistics.mean(bool(r["parsed"]) for r in rows),
        "fixed_legal_rate": statistics.mean(bool(r["legal"]) for r in rows),
        "long_parse_rate": 1 - long["parse_failure_rate"]["mean"],
        "long_legal_rate": 1 - long["illegal_rate"]["mean"],
        "all_metrics": metrics,
    }


def baseline_checks(weak: dict, strong: dict, rules: dict) -> dict:
    return {
        "measurably_weaker": strong["mean_reward"] - weak["mean_reward"] >= rules["minimum_strong_minus_weak_baseline_reward"],
        "different_adapter": weak["adapter_sha256"] != strong["adapter_sha256"],
        **{f"{kind}_floor": weak[f"{kind}_rate"] >= rules[f"baseline_{kind}_floor"]
           for kind in ("fixed_parse", "fixed_legal", "long_parse", "long_legal")},
    }


def learning_checks(weak: dict, candidate: dict, rules: dict) -> tuple[dict, dict]:
    if set(weak["rewards"]) != set(candidate["rewards"]):
        raise ValueError("paired states differ")
    ids = sorted(weak["rewards"])
    differences = [candidate["rewards"][key] - weak["rewards"][key] for key in ids]
    mean = statistics.mean(differences)
    checks = {
        "target_improved": mean >= rules["minimum_mean_reward_gain"],
        "fixed_parse_floor": candidate["fixed_parse_rate"] >= rules["candidate_fixed_parse_floor"],
        "fixed_legal_floor": candidate["fixed_legal_rate"] >= rules["candidate_fixed_legal_floor"],
    }
    if rules["require_no_validity_regression"]:
        checks.update({f"{kind}_non_regression": candidate[f"{kind}_rate"] >= weak[f"{kind}_rate"]
                       for kind in ("fixed_parse", "fixed_legal", "long_parse", "long_legal")})
    comparison = {"state_ids": ids, "differences": differences, "mean_difference": mean,
                  "median_difference": statistics.median(differences),
                  "bootstrap_95_ci": list(bootstrap_mean_ci(differences, samples=rules["bootstrap_samples"], seed=rules["bootstrap_seed"]))}
    return checks, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--weak", type=Path, required=True)
    parser.add_argument("--strong", type=Path, default=Path("runs/stage6-e0/rl/stress-development"))
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-gate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    if registration["experiment"] != "E2" or registration["suite"] != "development" or registration["final_test_access"]:
        raise ValueError("E2 requires a development-only registration")
    weak = evaluation(args.weak, "weak", registration)
    strong = evaluation(args.strong, "sft", registration)
    if strong["adapter_sha256"] != registration["frozen_sft_adapter_sha256"]:
        raise ValueError("strong SFT hash changed")
    checks = baseline_checks(weak, strong, registration["rules"])
    report = {"experiment": "E2", "kind": "learning_gate" if args.candidate else "baseline_gate",
              "registration_sha256": file_sha256(args.registration), "checks": checks,
              "weak": weak, "strong": strong, "research_complete": False}
    if args.candidate:
        if not args.baseline_gate:
            raise ValueError("candidate analysis requires the pre-RL baseline gate")
        prior = json.loads(args.baseline_gate.read_text())
        if prior["status"] != "passed" or prior["registration_sha256"] != report["registration_sha256"] or prior["weak"] != weak:
            raise ValueError("pre-RL baseline gate missing, failed or changed")
        training = json.loads((args.candidate.parent / "manifest.json").read_text())
        candidate = evaluation(args.candidate, "e2", registration)
        if training["experiment"] != "E2" or training["status"] != "completed" or training["adapter_sha256"] != weak["adapter_sha256"]:
            raise ValueError("candidate is not completed E2 training from the measured weak adapter")
        if training["output_adapter_sha256"] != candidate["adapter_sha256"] or not training["reference_frozen"] or training["reference_adapter_sha256_before"] != training["reference_adapter_sha256_after"]:
            raise ValueError("candidate adapter or frozen reference mismatch")
        extra, comparison = learning_checks(weak, candidate, registration["rules"])
        report.update(candidate=candidate, comparison=comparison)
        checks.update(extra)
    report["status"] = "passed" if all(checks.values()) else "not_passed"
    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(args.out, report)
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
