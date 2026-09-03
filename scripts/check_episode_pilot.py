#!/usr/bin/env python3
"""Register and assess independent R3 after R2; old E3 remains not passed."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analyze_stage6 import game_metric
from scripts.check_e2_learning import evaluation
from scripts.check_e4_pilot import frozen_stage5, promotion_checks
from scripts.check_episode_proof import audit_batch, validate_proof_report
from scripts.check_episode_runtime import validate_amendment
from scripts.check_recovery_pilot import assess as assess_r1, fresh_summary
from tetris.rl import atomic_write_json, directory_sha256, file_sha256, paired_comparison


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def register(protocol_path: Path, proof_registration: Path, proof_gate: Path, prior: float, runtime_amendment: Path | None = None):
    p = json.loads(protocol_path.read_text())
    if p["protocol"] != "stage6-recovery-v1" or p["final_test_access"]:
        raise ValueError("R3 requires the authorized development-only protocol")
    amended = validate_amendment(runtime_amendment, proof_registration, proof_gate) if runtime_amendment else None
    proof = amended["entry"] if amended else validate_proof_report(proof_registration, proof_gate)
    recipe = dict(p["episode_pilot"])
    block_hours, workflow_minutes, training_timeout_minutes = 5, 290, 95
    if amended:
        recipe["max_training_hours"] = amended["limits"]["max_training_hours"]
        block_hours, workflow_minutes, training_timeout_minutes = (amended["limits"][k] for k in
            ("block_hours", "workflow_minutes", "training_timeout_minutes"))
    if proof["protocol_sha256"] != file_sha256(protocol_path):
        raise ValueError("R2 belongs to a different protocol")
    r1_path = Path("runs") / p["sft"]["run_id"] / "rl/r1-gate.json"
    r1 = json.loads(r1_path.read_text())
    expected = assess_r1(protocol_path)
    if any(r1.get(k) != value for k, value in expected.items() if k != "generated_at"):
        raise ValueError("R1 comparison evidence cannot be reproduced")
    if not math.isfinite(prior) or prior < 0 or block_hours * 1.05 > 20 or prior + block_hours * 1.05 + 20 > 100:
        raise ValueError("R3 exceeds the existing Stage 6 budget")
    result = {"experiment": "R3", "status": "registered", "registered_at": now(),
        "run_id": p["episode_pilot"]["run_id"], "label": "r3", "recipe": recipe,
        "question": p["questions"][2], "protocol_path": str(protocol_path), "protocol_sha256": file_sha256(protocol_path),
        "proof_registration": str(proof_registration), "proof_registration_sha256": file_sha256(proof_registration),
        "entry_gate": str(proof_gate), "entry_gate_sha256": file_sha256(proof_gate),
        "r1_gate": str(r1_path), "r1_gate_sha256": file_sha256(r1_path), "r1_status": r1["status"],
        "prior_stage_spend_usd": prior, "pilot_usd": 20, "stage_usd": 100, "hourly_usd": 1.05,
        "block_hours": block_hours, "workflow_minutes": workflow_minutes,
        "training_timeout_minutes": training_timeout_minutes, "pilot_projection": proof["pilot_projection"],
        "promotion": p["episode_promotion"], "final_test_access": False, "research_complete": False}
    if amended:
        result.update(runtime_amendment=str(runtime_amendment), runtime_amendment_sha256=file_sha256(runtime_amendment),
            entry_status=proof["status"], historical_proof_status=proof["historical_proof_status"])
    return result


def validate_registration(path: Path):
    r = json.loads(path.read_text())
    expected = register(Path(r["protocol_path"]), Path(r["proof_registration"]), Path(r["entry_gate"]), r["prior_stage_spend_usd"],
        Path(r["runtime_amendment"]) if r.get("runtime_amendment") else None)
    if any(r.get(k) != value for k, value in expected.items() if k != "registered_at"):
        raise ValueError("R3 registration or prerequisites changed")
    return r, json.loads(Path(r["protocol_path"]).read_text())


def validate_training(root: Path, r: dict, p: dict, registration_path: Path):
    m = json.loads((root / "manifest.json").read_text())
    recipe = r["recipe"]
    data = Path(p["data"]["data_dir"])
    seeds = json.loads((data / "training-seeds.json").read_text())
    expected = {k: recipe[k] for k in ("updates", "group_size", "horizon", "training_seed", "learning_rate", "kl_beta", "gamma", "train_batch_size")}
    expected.update(status="completed", experiment="E6", completed_updates=recipe["updates"],
        external_registration_sha256=file_sha256(registration_path), research_question=r["question"],
        adapter_sha256=p["frozen_sft_adapter_sha256"], reference_adapter_sha256=p["frozen_sft_adapter_sha256"],
        frozen_sft_adapter_sha256=p["frozen_sft_adapter_sha256"], base_model_revision=p["base_model_revision"],
        requested_base_model_revision=p["base_model_revision"], benchmark_manifest_sha256=p["benchmark_manifest_sha256"],
        environment_training_seeds=seeds, recovery_starts_sha256=file_sha256(data / "train-starts.jsonl"),
        temperature=1, reference_frozen=True, budgets_usd={"pilot": 20, "stage": 100},
        prior_stage_spend_usd=r["prior_stage_spend_usd"], instance_hourly_usd=1.05,
        max_wall_clock_hours=recipe["max_training_hours"],
        reward={"formula": "normalized_score - death - illegal", "weights": {k: recipe[k] for k in ("score_scale", "death_penalty", "illegal_penalty")}})
    if any(m.get(k) != v for k, v in expected.items()) or m["registered_at"] < r["registered_at"]:
        raise ValueError("R3 training incomplete or differs from registered recipe")
    if m["reference_weights_sha256_before"] != m["reference_weights_sha256_after"] or directory_sha256(Path("runs/sft-v1/adapter")) != p["frozen_sft_adapter_sha256"]:
        raise ValueError("frozen SFT reference changed")
    if directory_sha256(root / "adapter") != m["output_adapter_sha256"]:
        raise ValueError("R3 output adapter identity changed")
    paths = sorted((root / "trajectory_batches").glob("*.json"))
    if len(paths) != recipe["updates"]:
        raise ValueError("R3 lost or duplicated trajectory batches")
    bank = [json.loads(s) for s in (data / "train-starts.jsonl").read_text().splitlines()]
    counts = []
    for update, path in enumerate(paths, 1):
        batch = json.loads(path.read_text())
        if batch["update"] != update:
            raise ValueError("R3 committed update order differs")
        counts.append(audit_batch(batch, recipe, seeds, bank))
    if counts != [x["turns"] for x in m["update_metrics"]] or sum(counts) != m["sample_count"]:
        raise ValueError("R3 sample accounting differs from replay")
    return m


def assess(registration_path: Path):
    r, p = validate_registration(registration_path)
    root = Path("runs") / r["run_id"] / "rl"
    m = validate_training(root, r, p, registration_path)
    baseline_path = Path("runs/stage6-e0/rl/stress-development")
    candidate_path = root / "stress-development"
    baseline, candidate = evaluation(baseline_path, "sft", p), evaluation(candidate_path, "r3", p)
    if baseline["adapter_sha256"] != p["frozen_sft_adapter_sha256"] or candidate["adapter_sha256"] != m["output_adapter_sha256"]:
        raise ValueError("R3 or baseline evaluated a different adapter")
    rules = json.loads(Path(p["benchmark_manifest"]).read_text())["decision_rules"]
    stage5 = frozen_stage5(root / "stage5", "r3", m["output_adapter_sha256"], p["base_model_revision"], rules)
    comparison = paired_comparison(game_metric(baseline_path, "sft")[0], game_metric(candidate_path, "r3")[0], bootstrap_samples=10000, seed=6105)
    r1_path = Path("runs") / p["sft"]["run_id"] / "rl/stress-development"
    versus_r1 = paired_comparison(game_metric(r1_path, "r1")[0], game_metric(candidate_path, "r3")[0], bootstrap_samples=10000, seed=6106)
    fresh = fresh_summary(root / "fresh-recovery", "r3", m["output_adapter_sha256"], p, file_sha256(Path(r["protocol_path"])))
    checks = promotion_checks(comparison, candidate, stage5, p["episode_promotion"]["min_relative_score_gain"])
    return {"experiment": "R3", "status": "passed" if all(checks.values()) else "not_passed", "checks": checks,
        "registration_sha256": file_sha256(registration_path), "training_manifest_sha256": file_sha256(root / "manifest.json"),
        "baseline": baseline, "candidate": candidate, "paired_score_comparison": comparison,
        "paired_comparison_vs_recovery_sft": versus_r1, "fresh_candidate": fresh,
        "fresh_original_sft": json.loads(Path(r["r1_gate"]).read_text())["fresh_baseline"],
        "fresh_recovery_sft": json.loads(Path(r["r1_gate"]).read_text())["fresh_candidate"],
        "stage5": stage5, "final_test_access": False, "research_complete": False,
        "deployment_authorized": False, "generated_at": now(),
        "next": "Replicate only a passed pilot under the original confirmation rule and budget. Otherwise retain the negative result, complete reporting and verified Stage6-only cleanup."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--register-from-protocol", type=Path)
    parser.add_argument("--proof-registration", type=Path)
    parser.add_argument("--proof-gate", type=Path)
    parser.add_argument("--runtime-amendment", type=Path)
    parser.add_argument("--prior-stage-spend-usd", type=float)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.register_from_protocol:
        if args.registration.exists() or not args.proof_registration or not args.proof_gate or args.prior_stage_spend_usd is None:
            parser.error("new registration, R2 registration/gate and prior spend are required")
        atomic_write_json(args.registration, register(args.register_from_protocol, args.proof_registration, args.proof_gate, args.prior_stage_spend_usd, args.runtime_amendment))
        return
    validate_registration(args.registration)
    if args.preflight:
        print("R3 prerequisites and registration verified")
        return
    if not args.out or args.out.exists():
        parser.error("--out must be a new gate path")
    report = assess(args.registration)
    atomic_write_json(args.out, report)
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
