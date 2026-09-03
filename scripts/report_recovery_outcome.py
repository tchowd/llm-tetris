#!/usr/bin/env python3
"""Close the authorized negative-result branch only after evidence and cleanup."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.check_recovery_pilot import assess as assess_r1
from scripts.check_episode_pilot import assess as assess_r3
from scripts.check_episode_proof import validate_proof_report
from scripts.check_episode_runtime import CORRECTNESS_CHECKS, validate_amendment
from tetris.rl import atomic_write_json, file_sha256


def same_gate(actual: dict, expected: dict):
    if any(actual.get(k) != v for k, v in expected.items() if k != "generated_at"):
        raise ValueError("saved gate cannot be independently reproduced")


def closure_decision(r1: dict, r2: dict, r3: dict, artifacts: dict, cleanup: dict):
    amended = (r2.get("status") == "correctness_passed_runtime_amended"
        and r2.get("historical_proof_status") == "not_passed"
        and r2.get("correctness_checks") == {k: True for k in CORRECTNESS_CHECKS}
        and r2.get("pilot_projection", {}).get("within_reserved_training_window") is True)
    if r1.get("status") not in ("passed", "not_passed") or not (r2.get("status") == "passed" or amended) or r3.get("status") != "not_passed":
        raise ValueError("negative closure requires completed R1, passed R2 and a completed non-promoted R3")
    if artifacts.get("status") != "passed" or cleanup.get("status") != "passed" or not cleanup.get("checks") or not all(cleanup["checks"].values()) or cleanup.get("instances") or cleanup.get("volumes"):
        raise ValueError("verified encrypted backups and complete Stage6 cleanup are required")
    return {"research_complete": True, "operations_complete": True, "decision": "retain_frozen_sft",
        "replication": "not_applicable_no_qualifying_episode_pilot",
        "final_test": "not_accessed_no_qualifying_episode_pilot", "historical_e3_status": "not_passed",
        "new_model_promoted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("runs/stage6-recovery-v1/rl/registration.json"))
    parser.add_argument("--artifact-audit", type=Path, required=True)
    parser.add_argument("--cleanup-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.out_dir / "report.json").exists() or (args.out_dir / "report.md").exists():
        parser.error("report paths must be new")
    p = json.loads(args.protocol.read_text())
    r1_root = Path("runs") / p["sft"]["run_id"] / "rl"
    r3_root = Path("runs") / p["episode_pilot"]["run_id"] / "rl"
    r3_registration = json.loads((r3_root / "registration.json").read_text())
    r2_registration = Path(r3_registration["proof_registration"])
    r2_root = r2_registration.parent
    r1 = json.loads((r1_root / "r1-gate.json").read_text())
    same_gate(r1, assess_r1(args.protocol))
    runtime_amendment = None
    if r3_registration.get("runtime_amendment"):
        runtime_amendment = validate_amendment(Path(r3_registration["runtime_amendment"]), r2_registration, Path(r3_registration["entry_gate"]))
    r2 = runtime_amendment["entry"] if runtime_amendment else validate_proof_report(r2_registration, r2_root / "r2-gate.json")
    r3 = json.loads((r3_root / "r3-gate.json").read_text())
    same_gate(r3, assess_r3(r3_root / "registration.json"))
    artifacts = json.loads(args.artifact_audit.read_text())
    cleanup = json.loads(args.cleanup_report.read_text())
    decision = closure_decision(r1, r2, r3, artifacts, cleanup)
    proof_registration = json.loads(r2_registration.read_text())
    expected_runs = {p["sft"]["run_id"], proof_registration["run_id"], p["episode_pilot"]["run_id"],
        proof_registration["control_run_id"]}
    failed_proofs = []
    if proof_registration.get("retry_of_registration"):
        previous_path = Path(proof_registration["retry_of_registration"])
        if file_sha256(previous_path) != proof_registration["retry_of_registration_sha256"]:
            raise ValueError("failed proof registration changed")
        previous = json.loads(previous_path.read_text())
        block_path = previous_path.parent / "block-state.json"
        block = json.loads(block_path.read_text())
        if block["status"] != "failed" or file_sha256(block_path) != proof_registration["failed_block_sha256"]:
            raise ValueError("failed proof outcome changed")
        expected_runs.update((previous["run_id"], previous["control_run_id"]))
        failed_proofs.append({"run_id": previous["run_id"], "status": "failed", "registration_sha256": file_sha256(previous_path), "block": block})
    if {r["run_id"] for r in artifacts["runs"]} != expected_runs or len(artifacts["runs"]) != len(expected_runs):
        raise ValueError("artifact audit does not cover all recovery experiment adapters")
    for row in artifacts["runs"]:
        manifest_path = Path("runs") / row["run_id"] / "rl/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if row["status"] != "passed" or row["encryption"] != "AES256" or row["optimizer_checkpoints_present"] or row["adapter_sha256"] != manifest["output_adapter_sha256"] or row["training_manifest_sha256"] != file_sha256(manifest_path):
            raise ValueError("adapter backup identity differs from completed experiment")
        for key, digest in row["object_sha256"].items():
            if file_sha256(Path(key)) != digest:
                raise ValueError("local evidence changed after the S3 audit")
    if file_sha256(Path("runs/stage6-e3/rl/selection.json")) != p["e3_selection_sha256"] or file_sha256(args.protocol.parent / "failure-audit.json") != p["failure_audit_sha256"]:
        raise ValueError("historical E3 or R0 evidence changed")
    for path in Path("runs").glob("*/rl/**/manifest.json"):
        if json.loads(path.read_text()).get("suite") == "test":
            raise ValueError("final-test access found; negative development-only closure is not applicable")
    ledger = json.loads(Path("runs/stage6-aws/rl/manifest.json").read_text())
    if ledger.get("last_observed_instance_state") != "terminated" or not ledger.get("stage6_root_volume_deleted_verified") or ledger["estimated_accrued_usd"] > 100:
        raise ValueError("exact worker termination, root deletion and final budget accounting must be recorded")
    report = {"stage": 6, "protocol": p["protocol"], "status": "completed_negative_result", **decision,
        "scope": "User-authorized R0/R1/R2/R3 follow-up; historical unrun E4/E7 are not counted as passed.",
        "protocol_sha256": file_sha256(args.protocol), "r1": r1, "r2": r2, "r3": r3, "retained_failed_proof_attempts": failed_proofs,
        "runtime_amendment": runtime_amendment,
        "artifact_audit_sha256": file_sha256(args.artifact_audit), "cleanup_report_sha256": file_sha256(args.cleanup_report),
        "estimated_total_stage6_usd": ledger["estimated_accrued_usd"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_write_json(args.out_dir / "report.json", report)
    r1_failures = ", ".join(k for k, v in r1["checks"].items() if not v) or "none"
    r3_failures = ", ".join(k for k, v in r3["checks"].items() if not v)
    lines = ["# Stage 6 recovery follow-up — completed negative result", "",
        "The original frozen SFT remains the accepted model. No episode candidate qualified for replication or final testing.", "",
        f"- R0: replayed E0/E3 illegal endings; original SFT already had difficult-board placement failures.",
        f"- R1 recovery-data SFT: {r1['status']}. Unmet research checks: {r1_failures}.",
        "- R2: GPU trajectory replay, exact token probabilities, controlled delayed-credit direction and pause/resume proof passed.",
        "- R2 runtime: the original time-gate failure is retained; R3 used the separately authorized 12-hour runtime amendment." if runtime_amendment else "- R2 runtime: original registered projection passed.",
        f"- Earlier failed R2 attempts retained and backed up: {len(failed_proofs)}; they are not reclassified as passed.",
        f"- R3 episode-return pilot: not passed. Unmet promotion checks: {r3_failures}.",
        "- Replication and final tests: not applicable, not passed. Test data remained untouched.",
        "- Encrypted final adapters and metadata were independently read back from S3; optimizer checkpoints were excluded.",
        "- Exact Stage 6 worker terminated and its root volume deleted. Stage 4 resources were outside the cleanup scope.", "",
        f"Conservative Stage 6 compute/storage estimate: ${ledger['estimated_accrued_usd']:.2f}; AWS billing is authoritative.", "",
        "This answers the registered bounded experiments. It does not prove that longer training or a different design cannot improve the model, and it does not turn historical E3 into a pass.", "",
        "Full paired metrics, checks and evidence hashes are in report.json.", ""]
    (args.out_dir / "report.md").write_text("\n".join(lines))
    print(f"completed negative-result report: {args.out_dir}")


if __name__ == "__main__":
    main()
