#!/usr/bin/env python3
"""One explicit checkpoint-164 recovery; frozen research sources are untouched."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra import scale10x_continuation as c

ROOT, PILOT_ROOT = c.ROOT, c.PILOT_ROOT
PLAN = ROOT / "resume-plan-v1.json"
BLOCK = ROOT / "resume-block-state-v1.json"
CHECKPOINT = PILOT_ROOT / "checkpoint-164"
ARCHIVE = ROOT / "attempts/interruption-1-worker"
BUCKET = "s3://llm-tetris-artifacts-566629888938-us-east-1/"


def remaining_allocation(amendment, ledger, manifest, proof, current):
    """Use the original worst measured rate/safety factor, only for uncommitted work."""
    remaining_turns = (320 - 164) * 4 * 128
    projection = 120 + proof["worst_seconds_per_turn"] * remaining_turns * proof["safety_multiplier"]
    amended_projection = copy.deepcopy(amendment)
    amended_projection["projected_training_seconds"] = projection
    allocation = min(c.launch_allocation(amended_projection, ledger, current),
                     33 * 3600 - manifest["wall_clock_seconds"])
    if projection > allocation:
        raise ValueError("remaining checkpoint work cannot fit unchanged deadline/training cap")
    return projection, allocation


def command(q):
    return c.training_command(q) + ["--resume", str(CHECKPOINT)]


def preflight():
    a, q = c.validate()
    plan = c.read(PLAN)
    if (plan["checkpoint"] != str(CHECKPOINT) or plan["resume_update"] != 164
            or plan["final_update"] != 320 or plan["deadline_epoch"] != a["absolute_deadline_epoch"]
            or plan["registration_sha256"] != c.file_sha256(c.REGISTRATION)
            or plan["amendment_sha256"] != c.file_sha256(c.AMENDMENT)):
        raise ValueError("recovery plan differs from unchanged approval/recipe")
    for name, digest in plan["input_sha256"].items():
        if c.file_sha256(Path(name)) != digest:
            raise ValueError(f"recovery input changed: {name}")
    if BLOCK.exists() or ARCHIVE.exists():
        raise ValueError("recovery already attempted; inspect, never silently retry")
    audit = c.read(ROOT / "checkpoint-164-cpu-audit-v1.json")
    backup = c.read(ROOT / "checkpoint-164-backup-audit-v1.json")
    if audit["status"] != "passed" or not all(audit["checks"].values()) or backup["status"] != "passed":
        raise ValueError("checkpoint needs CPU integrity and independent encrypted backup")
    for name, row in audit["files"].items():
        if c.file_sha256(Path(name)) != row["sha256"]:
            raise ValueError("checkpoint bytes changed after CPU inspection")
    m = c.read(PILOT_ROOT / "manifest.json")
    if m["status"] != "failed" or m["completed_updates"] != 167 or m["sample_count"] != 54808:
        raise ValueError("not the inspected interrupted run")
    if (PILOT_ROOT / "adapter").exists() or (PILOT_ROOT / "scale10x-gate.json").exists():
        raise ValueError("unexpected final adapter/evaluation; inspect instead")
    ledger = c.read(ROOT / "compute-ledger.json")
    proof = c.read(ROOT / "gpu-proof.json")["pilot_projection"]
    projection, allocation = remaining_allocation(a, ledger, m, proof, time.time())
    timer = subprocess.check_output(["systemctl", "show", "llm-tetris-scale10x-continuation-deadline.timer",
                                    "-p", "NextElapseUSecRealtime", "-p", "ActiveState"], text=True)
    if "ActiveState=active" not in timer or "Fri 2026-09-04 12:24:06 UTC" not in timer:
        raise ValueError("original absolute deadline timer must remain active")
    conf = Path("/etc/needrestart/conf.d/llm-tetris-scale10x.conf")
    if c.file_sha256(conf) != c.file_sha256(Path("infra/rl-scale10x-needrestart.conf")):
        raise ValueError("scoped maintenance-restart exclusion not installed")
    for service in ("llm-tetris-scale10x.service", "llm-tetris-scale10x-continuation.service"):
        state = subprocess.run(["systemctl", "is-active", service], capture_output=True, text=True).stdout.strip()
        if state not in ("inactive", "failed", "unknown"):
            raise ValueError("another training runner is active")
    return q, plan, ledger, projection, allocation


def archive_initial():
    ARCHIVE.mkdir(parents=True, exist_ok=False)
    for source in sorted(PILOT_ROOT.rglob("*")):
        relative = source.relative_to(PILOT_ROOT)
        if any(part.startswith("checkpoint-") or part == "adapter" for part in relative.parts):
            continue
        if source.is_file():
            target = ARCHIVE / "pilot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in ("pilot-block-state-v1.json", "pilot-training_320_updates-v1.log", "pilot-launch-decision-v1.json",
                 "pilot-shutdown-receipt-v1.json", "compute-ledger.json"):
        source = ROOT / name
        target = ARCHIVE / "workflow" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    hashes = {str(f.relative_to(ARCHIVE)): c.file_sha256(f) for f in sorted(ARCHIVE.rglob("*")) if f.is_file()}
    c.write_new(ARCHIVE / "archive-sha256.json", hashes)
    subprocess.run(["aws", "s3", "cp", str(ARCHIVE), BUCKET + str(ARCHIVE) + "/",
                    "--recursive", "--sse", "AES256", "--only-show-errors"], timeout=300, check=True)


def check_replay(expected, completed, directory=PILOT_ROOT):
    verified = {}
    for name, digest in expected.items():
        update = int(Path(name).stem.split("-")[-1])
        if completed >= update:
            if c.file_sha256(directory / "trajectory_batches" / name) != digest:
                raise ValueError(f"replayed trajectory differs at update {update}; no silent drift")
            verified[name] = digest
    return verified


def stop_child(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_process(command, log, seconds, expected=None):
    if seconds <= 0:
        raise TimeoutError("absolute phase allocation exhausted")
    started = time.time()
    with log.open("ab") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                remaining = seconds - (time.time() - started)
                if remaining <= 0:
                    raise TimeoutError(f"phase allocation expired after {seconds:.1f}s")
                try:
                    code = process.wait(timeout=min(15, remaining))
                except subprocess.TimeoutExpired:
                    code = None
                if expected:
                    path = PILOT_ROOT / "manifest.json"
                    if path.stat().st_mtime >= started:
                        completed = c.read(path).get("completed_updates", 0)
                        verified = check_replay(expected, completed)
                        if len(verified) == len(expected):
                            receipt = ROOT / "resume-replay-audit-v1.json"
                            if not receipt.exists():
                                c.write_new(receipt, {"status": "passed", "verified_at": c.now(),
                                    "trajectory_sha256": verified, "kind": "exact_checkpoint_replay",
                                    "not_held_out_evaluation": True})
                if code is not None:
                    return code
        except BaseException:
            stop_child(process)
            raise


def workflow(q, plan, ledger, projection, allocation):
    deadline = ledger["deadline_epoch"]
    status, failure = "incomplete", None

    def execute(phase, cmd, maximum, replay=None):
        c.atomic_write_json(BLOCK, {"status": "running", "phase": phase, "updated_at": c.now(),
            "deadline_epoch": deadline, "instance_id": ledger["instance_id"], "resumed_from_update": 164})
        code = run_process(cmd, ROOT / f"resume-{phase}-v1.log", min(maximum, deadline - time.time() - 600), replay)
        if code:
            raise RuntimeError(f"{phase} exited {code}; inspect preserved evidence")

    try:
        archive_initial()
        c.write_new(ROOT / "resume-launch-decision-v1.json", {"status": "passed", "generated_at": c.now(),
            "plan_sha256": c.file_sha256(PLAN), "projected_remaining_training_seconds": projection,
            "training_allocation_seconds": allocation, "evaluation_backup_reserve_seconds": 9000,
            "deadline_epoch": deadline, "resume_update": 164, "final_update": 320,
            "original_320_step_scheduler": True, "training_command": command(q)})
        # Recompute after archiving so its runtime cannot consume evaluation reserves.
        proof = c.read(ROOT / "gpu-proof.json")["pilot_projection"]
        _, allocation = remaining_allocation(c.read(c.AMENDMENT), ledger, c.read(PILOT_ROOT / "manifest.json"), proof, time.time())
        execute("training", command(q), allocation, plan["replay_sha256"])
        m = c.read(PILOT_ROOT / "manifest.json")
        if (m["status"] != "completed" or m["completed_updates"] != 320
                or not m["resume_used"] or m["resumed_from_update"] != 164):
            raise ValueError("training is incomplete or did not resume checkpoint 164")
        check_replay(plan["replay_sha256"], 320)
        adapter = str(PILOT_ROOT / "adapter")
        execute("training_backup", [sys.executable, "scripts/sync_run_artifacts.py", "upload", "--run-id", c.PILOT,
            "--include-adapter", "--receipt", str(PILOT_ROOT / "training-sync-receipt.json")], 300)
        execute("development", [sys.executable, "scripts/eval_stress.py", "--suite", "development", "--policies", "model",
            "--policy-label", "scale10x", "--adapter-dir", adapter, "--data-dirs", "data/batch1", "data/batch2",
            "--out-dir", str(PILOT_ROOT / "stress-development")], 3600)
        execute("stage5", [sys.executable, "scripts/eval_closed_loop.py", "--policies", "model", "--modes", "strict",
            "--model-label", "scale10x", "--adapter-dir", adapter, "--data-dirs", "data/batch1", "data/batch2",
            "--gen-batch-size", "64", "--teacher-workers", "3", "--device", "cuda", "--out-dir", str(PILOT_ROOT / "stage5")], 2700)
        execute("fresh_recovery", [sys.executable, "scripts/eval_recovery_only.py", "--registration", str(c.original.PROTOCOL),
            "--adapter-dir", adapter, "--label", "scale10x", "--out-dir", str(PILOT_ROOT / "fresh-recovery")], 900)
        execute("paired_analysis", [sys.executable, "infra/scale10x_continuation.py", "assess"], 600)
        status = c.read(PILOT_ROOT / "scale10x-gate.json")["status"]
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        print(failure, flush=True)
    finally:
        finished = {"status": status, "phase": "workflow_finished", "failure": failure, "finished_at": c.now(),
            "instance_id": ledger["instance_id"], "deadline_epoch": deadline,
            "research_complete": False, "operations_complete": False, "resumed_from_update": 164,
            "note": "Independent audits, applicable replication, report and exact cleanup remain required."}
        c.atomic_write_json(BLOCK, finished)
        c.atomic_write_json(PILOT_ROOT / "block-state.json", finished)
        errors = []
        receipt_path = ROOT / "resume-shutdown-receipt-v1.json"
        try:
            for run in (c.PILOT, "stage6-scale10x-v1"):
                receipt = Path("runs") / run / "rl" / ("sync-receipt.json" if run == c.PILOT else "resume-sync-receipt-v1.json")
                cmd = [sys.executable, "scripts/sync_run_artifacts.py", "upload", "--run-id", run, "--include-adapter", "--receipt", str(receipt)]
                for sync_cmd, maximum in ((cmd, 240), (cmd[:-3], 60)):
                    try:
                        if run_process(sync_cmd, ROOT / "resume-final-sync-v1.log", max(1, min(maximum, deadline - time.time() - 30))):
                            raise RuntimeError("artifact/receipt upload failed")
                    except BaseException as error:
                        errors.append({"run": run, "error": f"{type(error).__name__}: {error}"})
            c.atomic_write_json(receipt_path, {"requested_at": c.now(), "sync_errors": errors,
                "instance_id": ledger["instance_id"], "preserve_encrypted_ebs": True})
            subprocess.run(["aws", "s3", "cp", str(receipt_path), BUCKET + str(receipt_path),
                            "--sse", "AES256", "--only-show-errors"], timeout=20, check=False)
        finally:
            subprocess.run(["sudo", "/usr/sbin/shutdown", "-h", "now"], check=False)
    if failure:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    args = parser.parse_args()
    if args.action == "preflight":
        _, _, _, projection, allocation = preflight()
        print(json.dumps({"status": "passed", "remaining_projection": projection, "training_allocation": allocation}))
        return
    # Refused duplicate/preflight invocations cannot shut down somebody else's job.
    with (ROOT / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        workflow(*preflight())


if __name__ == "__main__":
    main()
