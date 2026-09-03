#!/usr/bin/env python3
"""Select the smallest registered KL preserving strict development validity."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analyze_stage6 import game_metric
from scripts.check_e2_learning import evaluation
from tetris.rl import atomic_write_json, file_sha256, paired_comparison


def validity_checks(summary: dict) -> dict:
    long = summary["all_metrics"]["long_horizon"]
    recovery = summary["all_metrics"]["recovery_rollouts"]
    return {
        "fixed_parse_ceiling": summary["fixed_parse_rate"] == 1,
        "fixed_legality_ceiling": summary["fixed_legal_rate"] == 1,
        "long_parse_ceiling": summary["long_parse_rate"] == 1,
        "long_legality_ceiling": summary["long_legal_rate"] == 1,
        "long_deaths_zero": long["deaths"] == 0,
        "long_illegal_deaths_zero": long["illegal_action_deaths"] == 0,
        "recovery_parse_failures_zero": recovery["parse_failure_rate"]["mean"] == 0,
        "recovery_illegal_actions_zero": recovery["illegal_rate"]["mean"] == 0,
        "recovery_illegal_deaths_zero": recovery["illegal_action_deaths"] == 0,
    }


def select_smallest(rows: list[dict], expected_betas: list[float]) -> dict | None:
    if len(rows) != len(expected_betas) or sorted(row["kl_beta"] for row in rows) != sorted(expected_betas):
        raise ValueError("all registered KL candidates must be evaluated exactly once")
    eligible = [row for row in rows if all(row["checks"].values())]
    return min(eligible, key=lambda row: row["kl_beta"]) if eligible else None


def validate_training(training: dict, registration: dict, beta: float, *, experiment: str = "E3") -> None:
    expected = {
        "experiment": experiment, "status": "completed", "initialization_kind": "sft",
        "adapter_sha256": registration["frozen_sft_adapter_sha256"],
        "frozen_sft_adapter_sha256": registration["frozen_sft_adapter_sha256"],
        "base_model_revision": registration["base_model_revision"],
        "benchmark_manifest_sha256": registration["benchmark_manifest_sha256"],
        "kl_beta": beta, **registration["training"],
    }
    for name, value in expected.items():
        if training.get(name) != value:
            raise ValueError(f"training registration mismatch: {name}")
    if training["reward"]["weights"] != registration["reward_weights"]:
        raise ValueError("reward weights changed during KL comparison")
    if not training.get("reference_frozen") or not training.get("reference_adapter_sha256_before") or training["reference_adapter_sha256_before"] != training["reference_adapter_sha256_after"]:
        raise ValueError("reference weights changed")
    if training["rollout_statistics"]["completions"] != registration["expected_completions_per_candidate"]:
        raise ValueError("incomplete rollout sample accounting")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    r = json.loads(args.registration.read_text())
    if r["experiment"] != "E3" or r["suite"] != "development" or r["final_test_access"]:
        raise ValueError("E3 is a development-only comparison")
    if file_sha256(Path(r["states_path"])) != r["states_sha256"]:
        raise ValueError("registered fixed states changed")
    if file_sha256(Path(r["entry_gate"])) != r["entry_gate_sha256"]:
        raise ValueError("E2 entry-gate evidence changed")
    baseline_path = Path(r["baseline_path"])
    baseline = evaluation(baseline_path, "sft", r)
    if baseline["adapter_sha256"] != r["frozen_sft_adapter_sha256"]:
        raise ValueError("frozen SFT baseline changed")
    baseline_values, _ = game_metric(baseline_path, "sft")
    rows, banks = [], set()
    for item in r["candidates"]:
        root = Path("runs") / item["run_id"] / "rl"
        training = json.loads((root / "manifest.json").read_text())
        validate_training(training, r, item["kl_beta"])
        banks.add(training["state_bank_sha256"])
        path = root / "stress-development"
        summary = evaluation(path, item["label"], r)
        if summary["adapter_sha256"] != training["output_adapter_sha256"]:
            raise ValueError("evaluated adapter differs from the trained output")
        values, _ = game_metric(path, item["label"])
        comparison = paired_comparison(baseline_values, values, bootstrap_samples=10000, seed=6003)
        rows.append({**item, "checks": validity_checks(summary), "evaluation": summary,
                     "paired_score_comparison": comparison,
                     "training_manifest_sha256": file_sha256(root / "manifest.json")})
    if len(banks) != 1:
        raise ValueError("KL candidates did not use the identical training state bank")
    selected = select_smallest(rows, [item["kl_beta"] for item in r["candidates"]])
    report = {
        "experiment": "E3", "status": "passed" if selected else "not_passed",
        "registration_sha256": file_sha256(args.registration), "candidates": rows,
        "selected_run_id": selected["run_id"] if selected else None,
        "selected_kl_beta": selected["kl_beta"] if selected else None,
        "selection_rule": r["selection_rule"], "state_bank_sha256": next(iter(banks)),
        "research_complete": False, "deployment_authorized": False,
        "next": "Register E4 with selected KL; begin from frozen SFT, not a selected intermediate adapter." if selected else "No KL preserves the registered validity guardrails; do not launch E4.",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(args.out, report)
    print(json.dumps({key: report[key] for key in ("status", "selected_run_id", "selected_kl_beta")}, indent=2))
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
