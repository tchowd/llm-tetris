#!/usr/bin/env python3
"""Close a negative 10x experiment only after verified artifacts and exact cleanup."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.stage6_scale10x import ROOT, REGISTRATION, PILOT, RUNS, assess, read, write_new, now
from tetris.rl import file_sha256


def closure(gate, audit, cleanup, ledger):
    if gate["status"] != "not_passed":
        raise ValueError("a passing pilot requires conditional replication; no negative closure")
    if audit.get("status") != "passed" or {r["run_id"] for r in audit["runs"]} != set(RUNS) or any(r["status"] != "passed" or not r["adapter_directory_hash_recomputed_from_s3"] or r["optimizer_checkpoints_present"] for r in audit["runs"]):
        raise ValueError("all three adapters require encrypted read-back")
    if cleanup.get("status") != "passed" or not all(cleanup["checks"].values()) or cleanup["instances"] or cleanup["volumes"]:
        raise ValueError("Stage 6 resource cleanup incomplete")
    if ledger.get("instance_state") != "terminated" or ledger.get("root_volume_deleted") is not True or not ledger["stage4_untouched"] or any(s["end_epoch"] is None for s in ledger["compute_sessions"]):
        raise ValueError("exact worker cleanup and closed spending ledger required")
    if ledger["estimated_experiment_usd"] > 50 or ledger["prior_stage6_estimate_usd"] + ledger["estimated_experiment_usd"] > 250:
        raise ValueError("spending exceeded approved limits")
    return {"research_complete": True, "operations_complete": True, "accepted_model": "original_sft",
        "replication": "not_applicable_no_qualifying_pilot", "final_test_access": False,
        "deployment_authorized": False}


def main():
    saved = read(Path("runs") / PILOT / "rl/scale10x-gate.json")
    recomputed = assess()
    if any(saved.get(k) != v for k, v in recomputed.items() if k != "generated_at"):
        raise ValueError("independent gate differs")
    audit, cleanup, ledger = (read(ROOT / n) for n in ("artifact-audit.json", "aws-cleanup.json", "compute-ledger.json"))
    result = {"kind": "stage6_scale10x_outcome", "status": "completed_negative_result", **closure(saved, audit, cleanup, ledger),
        "gate": saved, "estimated_experiment_usd": ledger["estimated_experiment_usd"],
        "estimated_cumulative_stage6_usd": ledger["prior_stage6_estimate_usd"] + ledger["estimated_experiment_usd"],
        "generated_at": now(), "evidence_sha256": {n: file_sha256(ROOT / n) for n in
            (REGISTRATION.name, "gpu-proof.json", "artifact-audit.json", "aws-cleanup.json", "compute-ledger.json")}}
    write_new(ROOT / "report.json", result)
    lines = ["# Stage 6: 10x episode-RL outcome", "", "The 320-update experiment completed, but did not qualify to replace the original SFT model.", "",
        f"Sampled decisions: {saved['training_diagnostics']['sampled_decisions']:,}.",
        f"Relative development score change: {saved['paired_score_comparison']['relative_improvement']:.3%}.",
        "Unmet checks: " + ", ".join(k for k, v in saved["checks"].items() if not v) + ".", "",
        "All adapters and metadata passed encrypted S3 read-back. The exact Stage 6 worker and root volume were removed; Stage 4 was untouched.",
        f"Estimated experiment cost: ${ledger['estimated_experiment_usd']:.2f}; AWS billing is authoritative.", "",
        "Final-test data remained untouched. Conditional replication was not applicable, not passed.",
        "This bounded negative result does not prove that a different representation, reward or training design cannot improve performance.", "",
        "Detailed paired results, guardrails and evidence hashes are in report.json.", ""]
    target = ROOT / "report.md"
    if target.exists():
        raise ValueError("refusing to overwrite report")
    target.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
