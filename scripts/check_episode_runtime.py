#!/usr/bin/env python3
"""Validate the explicit R3 runtime amendment without rewriting the failed R2 gate."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.check_episode_proof import evidence, projection, validate_registration
from tetris.rl import atomic_write_json, file_sha256


CORRECTNESS_CHECKS = {
    "all_trajectories_replayed", "exact_gpu_resume_trajectories",
    "exact_gpu_resume_adapter_tensors", "exact_token_alignment",
    "positive_delayed_reward_direction", "probe_weights_restored",
    "reference_unchanged", "gpu_headroom", "frozen_sft_unchanged",
}
LIMITS = {"max_training_hours": 12, "block_hours": 12, "workflow_minutes": 710,
    "training_timeout_minutes": 540, "pilot_usd": 20, "stage_usd": 100, "hourly_usd": 1.05}
SOURCES = ("scripts/check_episode_runtime.py", "scripts/check_episode_pilot.py",
    "infra/rl-r3-block.sh", "scripts/report_recovery_outcome.py")
APPROVAL = {"received": True, "received_on": "2026-09-02",
    "user_quote": "yes proceed - make it 12 hours",
    "scope": "R3 runtime ceilings only; 12 hours training and 12 hours overall. Reserve evaluation and sync inside the overall ceiling; budgets and scientific recipe unchanged."}


def correctness(proof_registration: Path, proof_gate: Path):
    """Reproduce replay/identity and bind retained GPU-only evidence, strictly."""
    r, p = validate_registration(proof_registration)
    root, control, manifests, _ = evidence(r, p, proof_registration)
    g = json.loads(proof_gate.read_text())
    if (g.get("experiment") != "R2" or g.get("status") != "not_passed"
            or g.get("registration_sha256") != file_sha256(proof_registration)
            or g.get("protocol_sha256") != r["protocol_sha256"]
            or g.get("final_test_access") is not False):
        raise ValueError("amendment requires the retained hash-bound R2 time-gate failure")
    expected_checks = {name: True for name in CORRECTNESS_CHECKS}
    expected_checks["pilot_projection_fits"] = False
    if g.get("checks") != expected_checks:
        raise ValueError("runtime amendment cannot waive any correctness check")
    paths = {str(f) for base in (root, control) for f in
        [base / "manifest.json", *sorted((base / "trajectory_batches").glob("*.json"))]}
    paths.add(str(root / "paused-manifest.json"))
    if set(g["evidence_sha256"]) != paths or any(file_sha256(Path(f)) != sha for f, sha in g["evidence_sha256"].items()):
        raise ValueError("retained GPU proof evidence changed")
    # Byte-identical safetensors files imply identical keys, dtypes and tensors.
    if file_sha256(root / "adapter/adapter_model.safetensors") != file_sha256(control / "adapter/adapter_model.safetensors"):
        raise ValueError("resumed and control adapter tensors differ")
    measured = projection(r, p, manifests[0]["update_metrics"])
    if g["pilot_projection"] != measured or measured["within_training_limit"] is not False or g["sample_count"] != manifests[0]["sample_count"]:
        raise ValueError("retained projection or sample accounting differs")
    d, gpu = g["positive_direction"], g["gpu"]
    if not (0 <= g["max_absolute_logprob_error"] <= r["logprob_absolute_tolerance"]
            and g["tokens_checked"] > 0 and math.isfinite(d["gradient_norm"]) and d["gradient_norm"] > 0
            and math.isfinite(d["mean_logprob_before"]) and math.isfinite(d["mean_logprob_after"])
            and d["mean_logprob_after"] - d["mean_logprob_before"] > r["positive_direction_min_logprob_gain"]
            and d["restored"] is True):
        raise ValueError("GPU probability or credit-direction evidence did not pass")
    if not (0 < gpu["peak_allocated_bytes"] <= gpu["total_bytes"]
            and math.isclose(gpu["headroom_fraction"], 1 - gpu["peak_allocated_bytes"] / gpu["total_bytes"], abs_tol=1e-12)
            and gpu["headroom_fraction"] >= r["minimum_allocated_gpu_headroom_fraction"]):
        raise ValueError("GPU headroom evidence did not pass")
    return r, p, g


def amendment(proof_registration: Path, proof_gate: Path):
    r, p, g = correctness(proof_registration, proof_gate)
    seconds = g["pilot_projection"]["projected_seconds"]
    if (seconds > LIMITS["training_timeout_minutes"] * 60
            or LIMITS["block_hours"] * LIMITS["hourly_usd"] > LIMITS["pilot_usd"]):
        raise ValueError("measured pilot does not fit the authorized operational reserve/budget")
    return {"amendment": "stage6-r3-runtime-v1", "status": "user_approved",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "approval": APPROVAL, "protocol_path": r["protocol_path"], "protocol_sha256": r["protocol_sha256"],
        "proof_registration": str(proof_registration), "proof_registration_sha256": file_sha256(proof_registration),
        "proof_gate": str(proof_gate), "proof_gate_sha256": file_sha256(proof_gate),
        "historical_proof_status": "not_passed", "only_amended_check": "pilot_projection_fits",
        "previous_limits": {"max_training_hours": p["episode_pilot"]["max_training_hours"], "block_hours": 5},
        "limits": LIMITS, "scientific_recipe_unchanged": True, "promotion_rules_unchanged": True,
        "operational_reserve": "At most 9 hours in the training subprocess; the remaining overall window reserves development, Stage5, fresh recovery, analysis and encrypted backup. The 12-hour trainer limit is a ceiling, not a duration target.",
        "source_sha256": {name: file_sha256(Path(name)) for name in SOURCES},
        "entry": {"status": "correctness_passed_runtime_amended", "historical_proof_status": "not_passed",
            "protocol_sha256": r["protocol_sha256"], "correctness_checks": {k: g["checks"][k] for k in sorted(CORRECTNESS_CHECKS)},
            "pilot_projection": {**g["pilot_projection"], "within_training_limit": True,
                "within_reserved_training_window": True, "authorized_training_hours": LIMITS["max_training_hours"]}},
        "final_test_access": False, "research_complete": False}


def validate_amendment(path: Path, proof_registration: Path, proof_gate: Path):
    saved = json.loads(path.read_text())
    expected = amendment(proof_registration, proof_gate)
    if set(saved) != set(expected) or any(saved.get(k) != v for k, v in expected.items() if k != "recorded_at"):
        raise ValueError("runtime amendment approval, limits, sources or linked evidence changed")
    gate = json.loads(proof_gate.read_text())
    if not gate["generated_at"] <= saved["recorded_at"] <= expected["recorded_at"]:
        raise ValueError("runtime amendment must follow the retained proof")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-registration", type=Path, required=True)
    parser.add_argument("--proof-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--record-user-approval", action="store_true")
    args = parser.parse_args()
    if not args.record_user_approval or args.out.exists():
        parser.error("explicit received user approval and a new output path are required")
    result = amendment(args.proof_registration, args.proof_gate)
    atomic_write_json(args.out, result)
    print(json.dumps(result["entry"], indent=2))


if __name__ == "__main__":
    main()
