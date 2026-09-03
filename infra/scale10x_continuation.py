#!/usr/bin/env python3
"""Explicit 90 USD / 36-hour continuation; historical registered sources stay frozen."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import stage6_scale10x as original
from scripts.run_scale10x import available_training_seconds, run_process
from tetris.rl import atomic_write_json, file_sha256

ROOT = original.ROOT
PILOT = original.PILOT
PILOT_ROOT = Path("runs") / PILOT / "rl"
APPROVAL = ROOT / "runtime-approval-v1.json"
AMENDMENT = ROOT / "runtime-amendment-v1.json"
REGISTRATION = ROOT / "pilot-registration-v3.json"
LEDGER_SNAPSHOT = ROOT / "compute-ledger-before-amendment-v1.json"
BLOCK = ROOT / "pilot-block-state-v1.json"
LIMITS = {"experiment_usd": 90, "cumulative_stage6_usd": 250,
          "overall_hours": 36, "training_hours": 33,
          "evaluation_backup_reserve_seconds": 9000, "hourly_allowance_usd": 2.30}
SOURCES = ("infra/scale10x_continuation.py", "tests/test_scale10x_continuation.py",
           "infra/rl-scale10x-continuation.service", "infra/rl-scale10x-continuation.cron",
           "infra/rl-scale10x-continuation-deadline.service",
           "infra/rl-scale10x-continuation-deadline.timer")
read, write_new, now = original.read, original.write_new, original.now


def retained_correctness():
    r, p = original.validate_registration()
    left, right, manifests, _ = original.evidence(r, p, original.REGISTRATION)
    g = read(ROOT / "gpu-proof.json")
    expected = {name: True for name in original.CORRECTNESS}
    expected["pilot_projection_fits"] = False
    if (g.get("experiment") != "SCALE10X_PROOF" or g.get("status") != "not_passed"
            or g.get("checks") != expected or g.get("final_test_access") is not False
            or g.get("registration_sha256") != file_sha256(original.REGISTRATION)
            or g.get("protocol_sha256") != r["protocol_sha256"]):
        raise ValueError("amendment cannot waive correctness or rewrite the historical gate")
    paths = {str(f) for root in (left, right) for f in
             [root / "manifest.json", *sorted((root / "trajectory_batches").glob("*.json"))]}
    paths.add(str(left / "paused-manifest.json"))
    if set(g["evidence_sha256"]) != paths or any(file_sha256(Path(f)) != sha for f, sha in g["evidence_sha256"].items()):
        raise ValueError("retained GPU proof evidence changed")
    if file_sha256(left / "adapter/adapter_model.safetensors") != file_sha256(right / "adapter/adapter_model.safetensors"):
        raise ValueError("resume/control weight bytes differ")
    if g["pilot_projection"] != original.projection(r, p, manifests[0]["update_metrics"]) or g["sample_count"] != manifests[0]["sample_count"]:
        raise ValueError("measured runtime or sample accounting changed")
    d, gpu = g["positive_direction"], g["gpu"]
    if not (0 <= g["max_absolute_logprob_error"] <= r["logprob_absolute_tolerance"]
            and g["tokens_checked"] > 0 and math.isfinite(d["gradient_norm"]) and d["gradient_norm"] > 0
            and math.isfinite(d["mean_logprob_before"]) and math.isfinite(d["mean_logprob_after"])
            and d["mean_logprob_after"] > d["mean_logprob_before"] and d["restored"] is True):
        raise ValueError("invalid independent GPU probability/credit evidence")
    if not ("L40S" in gpu["name"] and 0 < gpu["peak_allocated_bytes"] <= gpu["total_bytes"]
            and math.isclose(gpu["headroom_fraction"], 1 - gpu["peak_allocated_bytes"] / gpu["total_bytes"], abs_tol=1e-12)
            and gpu["headroom_fraction"] >= r["minimum_allocated_gpu_headroom_fraction"]):
        raise ValueError("GPU identity or memory evidence differs")
    audit = read(ROOT / "proof-artifact-audit-v1.json")
    if (audit.get("status") != "passed" or {x["run_id"] for x in audit["runs"]} != {original.PROOF, original.CONTROL}
            or any(x["status"] != "passed" or not x["adapter_directory_hash_recomputed_from_s3"]
                   or x["optimizer_checkpoints_present"] for x in audit["runs"])):
        raise ValueError("both proof adapters require independent encrypted read-back")
    for row in audit["runs"]:
        mpath = Path("runs") / row["run_id"] / "rl/manifest.json"
        if row["training_manifest_sha256"] != file_sha256(mpath) or row["adapter_sha256"] != read(mpath)["output_adapter_sha256"]:
            raise ValueError("artifact audit does not bind these proof manifests")
    return r, p, g


def validate_approval(a):
    if (a.get("status") != "user_approved" or a.get("user_quote") != "okay proceed"
            or a.get("limits") != {k: LIMITS[k] for k in ("experiment_usd", "cumulative_stage6_usd", "overall_hours")}
            or a.get("instance_id") != "i-069539d2da9a2b0a2"
            or a.get("root_volume_id") != "vol-0bbe37041a8425e58"
            or a.get("original_launch_epoch") != 1788395046.0
            or a.get("absolute_deadline_epoch") != a["original_launch_epoch"] + 36 * 3600
            or a.get("iam_changes_authorized") is not False):
        raise ValueError("explicit scoped 90 USD / 36-hour approval required")


def build_amendment():
    a = read(APPROVAL)
    validate_approval(a)
    r, p, g = retained_correctness()
    old_ledger = read(LEDGER_SNAPSHOT)
    if (old_ledger["instance_id"] != a["instance_id"] or old_ledger["root_volume_id"] != a["root_volume_id"]
            or old_ledger["launch_epoch"] != a["original_launch_epoch"]
            or any(s["end_epoch"] is None for s in old_ledger["compute_sessions"])):
        raise ValueError("closed pre-amendment compute ledger required")
    if not g["generated_at"] <= a["received_at"] <= now():
        raise ValueError("approval must follow this retained proof")
    if g["pilot_projection"]["projected_seconds"] > LIMITS["training_hours"] * 3600:
        raise ValueError("measured projection still exceeds the approved training allocation")
    worst_cost = LIMITS["overall_hours"] * LIMITS["hourly_allowance_usd"] + old_ledger["extra_service_reserve_usd"]
    if worst_cost > LIMITS["experiment_usd"] or old_ledger["prior_stage6_estimate_usd"] + worst_cost > LIMITS["cumulative_stage6_usd"]:
        raise ValueError("whole experiment does not fit approved cost ceilings")
    if old_ledger["prior_stage6_estimate_usd"] + old_ledger["estimated_experiment_usd"] > 20:
        raise ValueError("registered conservative prior spend allowance is insufficient")
    return {"kind": "scale10x_runtime_budget_amendment_v1", "status": "user_approved", "recorded_at": now(),
            "approval_sha256": file_sha256(APPROVAL), "proof_registration_sha256": file_sha256(original.REGISTRATION),
            "proof_gate_sha256": file_sha256(ROOT / "gpu-proof.json"),
            "proof_artifact_audit_sha256": file_sha256(ROOT / "proof-artifact-audit-v1.json"),
            "closed_ledger_sha256": file_sha256(LEDGER_SNAPSHOT), "limits": LIMITS,
            "instance_id": a["instance_id"], "root_volume_id": a["root_volume_id"],
            "absolute_deadline_epoch": a["absolute_deadline_epoch"],
            "conservative_prior_stage_spend_usd": 20.0,
            "historical_gate_status": "not_passed", "only_amended_check": "pilot_projection_fits",
            "projected_training_seconds": g["pilot_projection"]["projected_seconds"],
            "whole_window_compute_plus_services_ceiling_usd": worst_cost,
            "scientific_recipe_unchanged": True, "promotion_rules_unchanged": True,
            "proof_training_repeated": False, "final_test_access": False,
            "source_sha256": {name: file_sha256(Path(name)) for name in SOURCES}}


def build_registration(a):
    r, _ = original.validate_registration()
    return {"kind": "scale10x_pilot_registration_v3", "status": "registered", "registered_at": now(),
            "run_id": PILOT, "amendment_sha256": file_sha256(AMENDMENT),
            "proof_registration_sha256": file_sha256(original.REGISTRATION),
            "initial_adapter": "runs/sft-v1/adapter", "recipe": {**r["pilot_recipe"], "max_training_hours": LIMITS["training_hours"]},
            "promotion": r["promotion"], "absolute_deadline_epoch": a["absolute_deadline_epoch"],
            "final_test_access": False, "research_complete": False}


def same_record(saved, expected, timestamp):
    if set(saved) != set(expected) or any(saved[k] != v for k, v in expected.items() if k != timestamp):
        raise ValueError("registered approval, limits, recipe, source or linked evidence changed")


def validate():
    a = read(AMENDMENT)
    same_record(a, build_amendment(), "recorded_at")
    q = read(REGISTRATION)
    same_record(q, build_registration(a), "registered_at")
    if not read(APPROVAL)["received_at"] <= a["recorded_at"] <= q["registered_at"] <= now():
        raise ValueError("registration timestamps are not ordered")
    return a, q


def amended_recipe(q):
    r, p = original.validate_registration()
    r = copy.deepcopy(r)
    r.update(registered_at=q["registered_at"], pilot_recipe=q["recipe"], pilot_usd=90,
             stage_usd=250, prior_stage_spend_usd=20.0, block_hours=36)
    return r, p


def training_command(q):
    r, p = amended_recipe(q)
    command = original.training_command(r, p)
    for flag, value in (("--registration-file", REGISTRATION), ("--pilot-dollar-limit", 90)):
        command[command.index(flag) + 1] = str(value)
    return command


def launch_allocation(a, ledger, current):
    if (ledger["instance_id"] != a["instance_id"] or ledger["root_volume_id"] != a["root_volume_id"]
            or ledger["instance_type"] != "g6e.2xlarge" or ledger["deadline_epoch"] != a["absolute_deadline_epoch"]
            or ledger["experiment_cap_usd"] != 90 or ledger["cumulative_stage6_cap_usd"] != 250
            or ledger.get("runtime_amendment_sha256") != file_sha256(AMENDMENT)
            or ledger.get("pilot_registration_sha256") != file_sha256(REGISTRATION)):
        raise ValueError("live worker ledger differs from the authorized continuation")
    if sum(s["end_epoch"] is None for s in ledger["compute_sessions"]) != 1 or ledger["compute_sessions"][-1]["end_epoch"] is not None:
        raise ValueError("exactly one live compute session required")
    if any(not math.isfinite(s["start_epoch"]) or s["start_epoch"] > current
           or (s["end_epoch"] is not None and (not math.isfinite(s["end_epoch"]) or not s["start_epoch"] <= s["end_epoch"] <= current))
           for s in ledger["compute_sessions"]):
        raise ValueError("compute session timestamps are invalid")
    allocation = available_training_seconds(ledger["deadline_epoch"], current,
        LIMITS["evaluation_backup_reserve_seconds"], LIMITS["training_hours"] * 3600)
    if a["projected_training_seconds"] > allocation:
        raise ValueError("projection no longer fits the absolute deadline after full evaluation/backup reserve")
    projected_cost = sum((s["end_epoch"] if s["end_epoch"] is not None else ledger["deadline_epoch"]) - s["start_epoch"]
                         for s in ledger["compute_sessions"]) / 3600 * 2.30 + ledger["extra_service_reserve_usd"]
    if projected_cost > 90 or ledger["prior_stage6_estimate_usd"] + projected_cost > 250:
        raise ValueError("live remaining workflow could exceed the approved cost cap")
    return allocation


def assess():
    from scripts.analyze_stage6 import game_metric
    from scripts.check_e2_learning import evaluation
    from scripts.check_e4_pilot import frozen_stage5, promotion_checks
    from scripts.check_recovery_pilot import fresh_summary
    from tetris.rl import paired_comparison
    a, q = validate()
    r, p = amended_recipe(q)
    m, batches = original.training_evidence(PILOT_ROOT, r, p, REGISTRATION)
    baseline_path = Path("runs/stage6-e0/rl/stress-development")
    candidate_path = PILOT_ROOT / "stress-development"
    baseline, candidate = evaluation(baseline_path, "sft", p), evaluation(candidate_path, "scale10x", p)
    if baseline["adapter_sha256"] != p["frozen_sft_adapter_sha256"] or candidate["adapter_sha256"] != m["output_adapter_sha256"]:
        raise ValueError("evaluation adapter identity differs")
    paired = paired_comparison(game_metric(baseline_path, "sft")[0], game_metric(candidate_path, "scale10x")[0], bootstrap_samples=10000, seed=6205)
    stage5 = frozen_stage5(PILOT_ROOT / "stage5", "scale10x", m["output_adapter_sha256"], p["base_model_revision"], read(p["benchmark_manifest"])["decision_rules"])
    fresh = fresh_summary(PILOT_ROOT / "fresh-recovery", "scale10x", m["output_adapter_sha256"], p, file_sha256(original.PROTOCOL))
    checks = promotion_checks(paired, candidate, stage5, r["promotion"]["min_relative_score_gain"])
    return {"experiment": "SCALE10X", "status": "passed" if all(checks.values()) else "not_passed", "checks": checks,
            "registration_sha256": file_sha256(REGISTRATION), "runtime_amendment_sha256": file_sha256(AMENDMENT),
            "training_manifest_sha256": file_sha256(PILOT_ROOT / "manifest.json"),
            "gpu_proof_sha256": a["proof_gate_sha256"], "historical_gpu_gate_status": "not_passed",
            "baseline": baseline, "candidate": candidate, "paired_score_comparison": paired, "stage5": stage5,
            "fresh_candidate": fresh, "fresh_baseline": read("runs/rl-r1-recovery-sft-seed0/rl/r1-gate.json")["fresh_baseline"],
            "training_diagnostics": original.diagnostics(batches), "final_test_access": False,
            "research_complete": False, "operations_complete": False, "deployment_authorized": False,
            "generated_at": now(), "next": "Only a qualifying pilot proceeds to registered replication; otherwise audit/report/cleanup the negative result."}


def sync_metadata():
    for run in ("stage6-scale10x-v1", PILOT):
        if (Path("runs") / run).exists():
            code = run_process([sys.executable, "scripts/sync_run_artifacts.py", "upload", "--run-id", run], ROOT / "pilot-cron-sync-v1.log", 180)
            if code:
                raise RuntimeError("metadata sync failed")


def workflow():
    a, q = validate()
    ledger = read(ROOT / "compute-ledger.json")
    deadline = ledger["deadline_epoch"]
    allocation = launch_allocation(a, ledger, time.time())
    if BLOCK.exists() or (PILOT_ROOT / "manifest.json").exists():
        raise ValueError("pilot already has state; inspect its checkpoints before any explicit continuation")
    status, failure = "incomplete", None

    def execute(phase, command, maximum):
        atomic_write_json(BLOCK, {"status": "running", "phase": phase, "updated_at": now(),
                                  "deadline_epoch": deadline, "instance_id": ledger["instance_id"]})
        code = run_process(command, ROOT / ("pilot-" + phase + "-v1.log"), min(maximum, deadline - time.time() - 600))
        if code:
            raise RuntimeError(f"{phase} exited {code}; inspect retained logs")

    try:
        write_new(ROOT / "pilot-launch-decision-v1.json", {"status": "passed", "generated_at": now(),
            "registration_sha256": file_sha256(REGISTRATION), "amendment_sha256": file_sha256(AMENDMENT),
            "historical_proof_status": "not_passed", "entry": "correctness_passed_runtime_amended",
            "projected_training_seconds": a["projected_training_seconds"], "training_allocation_seconds": allocation,
            "evaluation_backup_reserve_seconds": LIMITS["evaluation_backup_reserve_seconds"], "deadline_epoch": deadline})
        execute("training_320_updates", training_command(q), allocation)
        m = read(PILOT_ROOT / "manifest.json")
        if m["status"] != "completed" or m["completed_updates"] != 320:
            raise ValueError("pilot stopped short of 320 registered updates")
        execute("training_backup", [sys.executable, "scripts/sync_run_artifacts.py", "upload", "--run-id", PILOT, "--include-adapter", "--receipt", str(PILOT_ROOT / "training-sync-receipt.json")], 300)
        adapter = str(PILOT_ROOT / "adapter")
        execute("development", [sys.executable, "scripts/eval_stress.py", "--suite", "development", "--policies", "model", "--policy-label", "scale10x", "--adapter-dir", adapter, "--data-dirs", "data/batch1", "data/batch2", "--out-dir", str(PILOT_ROOT / "stress-development")], 3600)
        execute("stage5", [sys.executable, "scripts/eval_closed_loop.py", "--policies", "model", "--modes", "strict", "--model-label", "scale10x", "--adapter-dir", adapter, "--data-dirs", "data/batch1", "data/batch2", "--gen-batch-size", "64", "--teacher-workers", "3", "--device", "cuda", "--out-dir", str(PILOT_ROOT / "stage5")], 2700)
        execute("fresh_recovery", [sys.executable, "scripts/eval_recovery_only.py", "--registration", str(original.PROTOCOL), "--adapter-dir", adapter, "--label", "scale10x", "--out-dir", str(PILOT_ROOT / "fresh-recovery")], 900)
        execute("paired_analysis", [sys.executable, "infra/scale10x_continuation.py", "assess"], 600)
        status = read(PILOT_ROOT / "scale10x-gate.json")["status"]
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        print(failure, flush=True)
    finally:
        finished = {"status": status, "phase": "workflow_finished", "failure": failure, "finished_at": now(),
                    "deadline_epoch": deadline, "instance_id": ledger["instance_id"],
                    "research_complete": False, "operations_complete": False,
                    "note": "Independent artifact audit, report and exact cleanup still required."}
        atomic_write_json(BLOCK, finished)
        if (PILOT_ROOT / "manifest.json").exists():
            atomic_write_json(PILOT_ROOT / "block-state.json", finished)
        errors = []
        shutdown_receipt = ROOT / "pilot-shutdown-receipt-v1.json"
        for run in (PILOT, "stage6-scale10x-v1"):
            root = Path("runs") / run / "rl"
            if not root.exists():
                continue
            atomic_write_json(shutdown_receipt, {"requested_at": now(), "sync_errors": errors,
                "instance_id": ledger["instance_id"], "preserve_encrypted_ebs": True})
            receipt = root / ("sync-receipt.json" if run == PILOT else "pilot-sync-receipt-v1.json")
            command = [sys.executable, "scripts/sync_run_artifacts.py", "upload", "--run-id", run,
                       "--include-adapter", "--receipt", str(receipt)]
            try:
                for cmd, maximum in ((command, 240), (command[:-3], 60)):
                    if run_process(cmd, ROOT / "pilot-final-sync-v1.log", max(1, min(maximum, deadline - time.time() - 30))):
                        raise RuntimeError("artifact/receipt upload failed")
            except Exception as error:
                errors.append({"run": run, "error": str(error)})
        atomic_write_json(shutdown_receipt, {"requested_at": now(), "sync_errors": errors,
            "instance_id": ledger["instance_id"], "preserve_encrypted_ebs": True})
        try:
            subprocess.run(["aws", "s3", "cp", str(shutdown_receipt),
                "s3://llm-tetris-artifacts-566629888938-us-east-1/" + str(shutdown_receipt),
                "--sse", "AES256", "--only-show-errors"], timeout=20, check=False)
        except Exception as error:
            print(f"Final shutdown receipt upload failed: {error}", flush=True)
        finally:
            subprocess.run(["sudo", "/usr/sbin/shutdown", "-h", "now"], check=False)
    if failure:
        raise SystemExit(1)


def closure(gate, audit, cleanup, ledger):
    if gate["status"] != "not_passed":
        raise ValueError("a positive pilot requires the registered replication branch")
    if (audit.get("status") != "passed" or {r["run_id"] for r in audit["runs"]} != set(original.RUNS)
            or any(r["status"] != "passed" or not r["adapter_directory_hash_recomputed_from_s3"] or r["optimizer_checkpoints_present"] for r in audit["runs"])):
        raise ValueError("all three final adapters require encrypted read-back")
    if cleanup.get("status") != "passed" or not all(cleanup["checks"].values()) or cleanup["instances"] or cleanup["volumes"]:
        raise ValueError("resource cleanup is incomplete")
    if (ledger.get("instance_id") != "i-069539d2da9a2b0a2" or ledger.get("root_volume_id") != "vol-0bbe37041a8425e58"
            or ledger.get("instance_state") != "terminated" or ledger.get("root_volume_deleted") is not True
            or not ledger["stage4_untouched"] or any(s["end_epoch"] is None for s in ledger["compute_sessions"])):
        raise ValueError("exact resource cleanup and closed spending ledger required")
    if ledger["estimated_experiment_usd"] > 90 or ledger["prior_stage6_estimate_usd"] + ledger["estimated_experiment_usd"] > 250:
        raise ValueError("approved budgets exceeded")
    return {"research_complete": True, "operations_complete": True, "accepted_model": "original_sft",
            "replication": "not_applicable_no_qualifying_pilot", "final_test_access": False, "deployment_authorized": False}


def report():
    saved, recomputed = read(PILOT_ROOT / "scale10x-gate.json"), assess()
    same_record(saved, recomputed, "generated_at")
    audit, cleanup, ledger = (read(ROOT / n) for n in ("artifact-audit.json", "aws-cleanup.json", "compute-ledger.json"))
    result = {"kind": "stage6_scale10x_outcome", "status": "completed_negative_result", **closure(saved, audit, cleanup, ledger),
              "gate": saved, "generated_at": now(), "estimated_experiment_usd": ledger["estimated_experiment_usd"],
              "estimated_cumulative_stage6_usd": ledger["prior_stage6_estimate_usd"] + ledger["estimated_experiment_usd"],
              "evidence_sha256": {n: file_sha256(ROOT / n) for n in (AMENDMENT.name, REGISTRATION.name,
                  "gpu-proof.json", "proof-artifact-audit-v1.json", "artifact-audit.json", "aws-cleanup.json", "compute-ledger.json")}}
    write_new(ROOT / "report.json", result)
    target = ROOT / "report.md"
    if target.exists():
        raise ValueError("refusing to overwrite final report")
    target.write_text("\n".join(["# Stage 6: 10x episode-RL outcome", "",
        "All 320 updates and frozen evaluations completed; the candidate did not qualify to replace original SFT.", "",
        f"Sampled decisions: {saved['training_diagnostics']['sampled_decisions']:,}.",
        f"Relative development score change: {saved['paired_score_comparison']['relative_improvement']:.3%}.",
        "Unmet checks: " + ", ".join(k for k, v in saved["checks"].items() if not v) + ".", "",
        "All three adapters passed encrypted S3 read-back. The exact Stage 6 worker/root were removed; Stage 4 was untouched.",
        f"Estimated experiment cost: ${ledger['estimated_experiment_usd']:.2f}; AWS billing is authoritative.",
        "The runtime-only amendment did not alter the scientific recipe or relabel the historical time-gate failure.",
        "Final-test data remained untouched; conditional replication was not applicable, not passed.", "",
        "This bounded result does not rule out improvement from a different representation, reward or training design.", ""]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "preflight", "run", "sync", "assess", "report"))
    args = parser.parse_args()
    if args.action == "register":
        a = build_amendment()
        write_new(AMENDMENT, a)
        write_new(REGISTRATION, build_registration(a))
    elif args.action == "preflight":
        validate()
    elif args.action in ("run", "sync"):
        with (ROOT / ("workflow.lock" if args.action == "run" else "sync.lock")).open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == "run":
                try:
                    workflow()
                except BaseException:
                    # We hold the workflow lock: this cannot stop another healthy runner.
                    subprocess.run(["sudo", "/usr/sbin/shutdown", "-h", "now"], check=False)
                    raise
            else:
                sync_metadata()
    elif args.action == "assess":
        write_new(PILOT_ROOT / "scale10x-gate.json", assess())
    else:
        report()
    print(args.action + " complete", flush=True)


if __name__ == "__main__":
    main()
