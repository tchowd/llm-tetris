#!/usr/bin/env python3
"""Register and assess E4 only after the complete, fixed E3 KL comparison."""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analyze_stage6 import game_metric, stage5_gate
from scripts.check_e2_learning import evaluation, read_rows
from scripts.select_e3_kl import select_smallest, validate_training, validity_checks
from tetris.rl import atomic_write_json, file_sha256, paired_comparison


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def selected_kl(selection: dict, registration: dict, registration_hash: str) -> float:
    if selection.get("status") != "passed" or selection.get("registration_sha256") != registration_hash:
        raise ValueError("complete passed E3 selection with matching registration is required")
    rows = selection["candidates"]
    expected = registration["candidates"]
    if sorted((r["run_id"], r["label"], r["kl_beta"]) for r in rows) != sorted(
        (r["run_id"], r["label"], r["kl_beta"]) for r in expected
    ):
        raise ValueError("E3 candidate identities differ from registration")
    for row in rows:
        if row["checks"] != validity_checks(row["evaluation"]):
            raise ValueError("E3 checks disagree with evaluated validity")
    chosen = select_smallest(rows, [r["kl_beta"] for r in expected])
    if not chosen or selection.get("selected_kl_beta") != chosen["kl_beta"] or selection.get("selected_run_id") != chosen["run_id"]:
        raise ValueError("E3 did not select the smallest eligible KL")
    return chosen["kl_beta"]


def register(e3_path: Path, selection_path: Path, prior: float) -> dict:
    # The trainer reserves its full pilot ceiling in addition to conservative
    # prior spend (which includes the entire enclosing block allowance).
    if not math.isfinite(prior) or prior < 0 or prior + 3 * 1.05 + 20 > 100:
        raise ValueError("E4 block exceeds existing Stage 6 budget")
    e3 = json.loads(e3_path.read_text())
    selection = json.loads(selection_path.read_text())
    beta = selected_kl(selection, e3, file_sha256(e3_path))
    r = {name: copy.deepcopy(e3[name]) for name in (
        "base_model", "base_model_revision", "frozen_sft_adapter_sha256", "benchmark_manifest",
        "benchmark_manifest_sha256", "states_path", "states_sha256", "suite", "baseline_path",
        "training", "reward_weights", "final_test_access",
    )}
    if e3["experiment"] != "E3" or r["suite"] != "development" or r["final_test_access"]:
        raise ValueError("E4 requires the development-only E3 registration")
    fixed_training = {"training_seed": 0, "max_updates": 256, "completed_updates": 256,
                      "num_states": 256, "group_size": 4, "batch_size": 4, "gradient_accumulation": 4,
                      "learning_rate": 1e-6, "temperature": 1, "max_completion_length": 16,
                      "sampling": {"temperature": 1, "top_p": 1, "top_k": 0}}
    if r["training"] != fixed_training or r["reward_weights"] != {
        "lines": 1, "holes": 1, "aggregate_height": .05, "bumpiness": .02, "illegal": 10
    }:
        raise ValueError("E3 recipe is not compatible with the fixed E4 workflow")
    r["training"].update(max_updates=512, completed_updates=512)
    r.update(
        registration_version=1, experiment="E4", run_id="rl-e4-seed0", label="e4", registered_at=now(),
        question="Does dense one-step learning transfer to the development stress suite without regressing frozen Stage 5?",
        e3_registration=str(e3_path), e3_registration_sha256=file_sha256(e3_path),
        entry_gate=str(selection_path), entry_gate_sha256=file_sha256(selection_path),
        kl_beta=beta, expected_completions_per_candidate=8192, research_complete=False,
        budgets={"prior_stage_spend_usd": prior, "pilot_usd": 20, "stage_usd": 100,
                 "hourly_usd": 1.05, "training_max_hours": 1, "block_max_hours_including_sync": 3,
                 "training_external_timeout_minutes": 65, "development_timeout_minutes": 75,
                 "stage5_timeout_minutes": 60,
                 "workflow_timeout_minutes": 170, "sync_reserve_minutes": 10},
        selection_rule="At least 3% paired development score improvement, all E3 validity guardrails, and frozen Stage 5 non-inferiority. No reward tuning or final-test access.",
        no_improvement="Retain the dense null/negative result. Do not scale it blindly; E5 trajectory correctness is a separate research question, not dense promotion.",
    )
    return r


def validate_registration(r: dict) -> None:
    if r["experiment"] != "E4" or r["suite"] != "development" or r["final_test_access"]:
        raise ValueError("E4 is development-only")
    for path_key, hash_key in (("e3_registration", "e3_registration_sha256"), ("entry_gate", "entry_gate_sha256"),
                               ("benchmark_manifest", "benchmark_manifest_sha256"), ("states_path", "states_sha256")):
        if file_sha256(Path(r[path_key])) != r[hash_key]:
            raise ValueError(f"registered evidence changed: {path_key}")
    expected = register(Path(r["e3_registration"]), Path(r["entry_gate"]), r["budgets"]["prior_stage_spend_usd"])
    for name in expected:
        if name != "registered_at" and r.get(name) != expected[name]:
            raise ValueError(f"E4 recipe differs from fixed registration: {name}")


def frozen_stage5(path: Path, label: str, adapter_hash: str, revision: str, rules: dict) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("status") != "passed" or manifest.get("modes") != ["strict"] or manifest.get("greedy") is not True:
        raise ValueError("Stage 5 evaluation must be completed strict greedy")
    if manifest.get("adapter_sha256") != adapter_hash or manifest["policy_metadata"][label].get("base_model_revision") != revision:
        raise ValueError("Stage 5 evaluated a different model or adapter")
    rows = [r for r in read_rows(path / "games.jsonl") if r["policy"] == label and r["mode"] == "strict"]
    if len(rows) != 100 or sorted(r["seed"] for r in rows) != list(range(10_000_000, 10_000_100)):
        raise ValueError("Stage 5 cohort is incomplete or duplicated")
    gate = stage5_gate(path / "metrics.json", label, rules)
    metrics = gate["metrics"]
    if metrics["n_games"] != 100 or metrics["deaths"] != sum(r["died"] for r in rows) or not math.isclose(
        metrics["lines"]["mean"], statistics.mean(r["lines"] for r in rows), abs_tol=1e-10
    ):
        raise ValueError("Stage 5 summary differs from saved games")
    gate["checks"].update(
        saved_games_no_incidents=all(not row["incidents"] for row in rows),
        saved_games_parse_ceiling=all(action is not None for row in rows for action in row["raw_actions"]),
        saved_games_full_survival=all(not row["died"] and row["pieces"] == 500 and len(row["actions"]) == len(row["raw_actions"]) == 500 for row in rows),
    )
    gate["passed"] = all(gate["checks"].values())
    gate["manifest_sha256"] = file_sha256(path / "manifest.json")
    return gate


def promotion_checks(comparison: dict, summary: dict, stage5: dict, threshold: float) -> dict:
    gain = comparison["relative_improvement"]
    return {"development_gain": gain is not None and math.isfinite(gain) and gain >= threshold,
            **validity_checks(summary), "stage5_non_inferiority": stage5["passed"]}


def assess(r: dict, registration_path: Path) -> dict:
    validate_registration(r)
    root = Path("runs") / r["run_id"] / "rl"
    training = json.loads((root / "manifest.json").read_text())
    validate_training(training, r, r["kl_beta"], experiment="E4")
    if training["state_bank_sha256"] != json.loads(Path(r["entry_gate"]).read_text())["state_bank_sha256"]:
        raise ValueError("E4 training state bank differs from the selected E3 control")
    baseline_path, candidate_path = Path(r["baseline_path"]), root / "stress-development"
    baseline, candidate = evaluation(baseline_path, "sft", r), evaluation(candidate_path, r["label"], r)
    if baseline["adapter_sha256"] != r["frozen_sft_adapter_sha256"] or candidate["adapter_sha256"] != training["output_adapter_sha256"]:
        raise ValueError("baseline or candidate adapter hash mismatch")
    baseline_values, _ = game_metric(baseline_path, "sft")
    candidate_values, _ = game_metric(candidate_path, r["label"])
    comparison = paired_comparison(baseline_values, candidate_values, bootstrap_samples=10000, seed=6004)
    rules = json.loads(Path(r["benchmark_manifest"]).read_text())["decision_rules"]
    stage5 = frozen_stage5(root / "stage5", r["label"], training["output_adapter_sha256"], r["base_model_revision"], rules)
    checks = promotion_checks(comparison, candidate, stage5, rules["development_relative_improvement"])
    return {"experiment": "E4", "status": "passed" if all(checks.values()) else "not_passed", "checks": checks,
            "registration_sha256": file_sha256(registration_path), "training_manifest_sha256": file_sha256(root / "manifest.json"),
            "baseline": baseline, "candidate": candidate, "paired_score_comparison": comparison,
            "stage5_non_inferiority": stage5, "research_complete": False, "deployment_authorized": False,
            "generated_at": now(), "next": "Record the dense result, then register the independent E5 trajectory proof. E7 remains gated on pilot promotion."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--register-from-e3", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--prior-stage-spend-usd", type=float)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.register_from_e3:
        if args.registration.exists() or not args.selection or args.prior_stage_spend_usd is None:
            parser.error("registration must be new; --selection and --prior-stage-spend-usd are required")
        result = register(args.register_from_e3, args.selection, args.prior_stage_spend_usd)
        validate_registration(result)
        atomic_write_json(args.registration, result)
        print(f"registered E4 with KL {result['kl_beta']}")
        return
    r = json.loads(args.registration.read_text())
    validate_registration(r)
    if args.preflight:
        print("E4 registration and E3 entry evidence verified")
        return
    if not args.out or args.out.exists():
        parser.error("--out must be a new gate path")
    report = assess(r, args.registration)
    atomic_write_json(args.out, report)
    print(json.dumps({key: report[key] for key in ("status", "checks")}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
