#!/usr/bin/env python3
"""One deadline-bound proof/train/evaluation workflow on the registered EC2 worker."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.stage6_scale10x import (ROOT, REGISTRATION, PROTOCOL, PILOT, PROOF,
    CONTROL, RUNS, read, write_new, now, validate_registration, training_command, check_proof)
from tetris.rl import atomic_write_json, file_sha256


def available_training_seconds(deadline, current, reserve, maximum):
    result = min(maximum, deadline - current - reserve)
    if result <= 0:
        raise ValueError("no time remains after reserving evaluation and backup")
    return result


def run_process(command, log, seconds):
    if seconds <= 0:
        raise TimeoutError("workflow deadline reached")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError(f"process exceeded reserved window; see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-only", action="store_true")
    args = parser.parse_args()
    lock_path = ROOT / ("sync.lock" if args.sync_only else "workflow.lock")
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another instance is already active")
        if args.sync_only:
            for run in ("stage6-scale10x-v1", *RUNS):
                if (Path("runs") / run).exists():
                    result = run_process([sys.executable, "scripts/sync_run_artifacts.py", "upload", "--run-id", run], ROOT / "cron-sync.log", 180)
                    if result:
                        raise SystemExit(result)
            return
        workflow()


def workflow():
    r, p = validate_registration()
    ledger = read(ROOT / "compute-ledger.json")
    deadline = ledger["deadline_epoch"]
    if ledger["registration_sha256"] != file_sha256(REGISTRATION) or ledger["instance_type"] != r["instance_type"]:
        raise ValueError("worker ledger differs from registration")
    block = ROOT / "block-state.json"
    if block.exists() or any((Path("runs") / run / "rl/manifest.json").exists() for run in RUNS):
        raise ValueError("refusing duplicate workflow; inspect saved progress before explicit resume")
    status, failure = "failed", None

    def phase(name):
        atomic_write_json(block, {"status": "running", "phase": name, "updated_at": now(),
            "deadline_epoch": deadline, "instance_id": ledger["instance_id"]})

    def execute(name, command, limit, log=None):
        phase(name)
        seconds = min(limit, deadline - time.time() - 600)
        code = run_process(command, log or ROOT / (name + ".log"), seconds)
        if code:
            raise RuntimeError(f"{name} exited {code}; inspect its retained log")

    try:
        proof_root = Path("runs") / PROOF / "rl"
        execute("proof_first_two", training_command(r, p, proof=True, pause=True), 1800)
        paused = read(proof_root / "manifest.json")
        if paused["status"] != "paused" or paused["completed_updates"] != 2:
            raise ValueError("proof did not pause at two committed updates")
        write_new(proof_root / "paused-manifest.json", paused)
        execute("proof_resume", training_command(r, p, proof=True, resume=proof_root / "checkpoint-2"), 1800)
        execute("proof_control", training_command(r, p, proof=True, control=True), 2400)
        execute("independent_gpu_proof", [sys.executable, "scripts/stage6_scale10x.py", "gpu-proof"], 1800)
        proof = check_proof()
        allocation = available_training_seconds(deadline, time.time(), r["evaluation_reserve_seconds"], r["pilot_recipe"]["max_training_hours"] * 3600)
        if proof["pilot_projection"]["projected_seconds"] > allocation:
            raise ValueError("measured full-length training projection does not fit after evaluation reserve")
        write_new(ROOT / "launch-decision.json", {"status": "passed", "generated_at": now(),
            "proof_sha256": file_sha256(ROOT / "gpu-proof.json"), "training_allocation_seconds": allocation,
            "projected_training_seconds": proof["pilot_projection"]["projected_seconds"],
            "remaining_workflow_seconds": deadline - time.time(), "evaluation_reserve_seconds": r["evaluation_reserve_seconds"]})
        execute("training_320_updates", training_command(r, p), allocation)
        root = Path("runs") / PILOT / "rl"
        m = read(root / "manifest.json")
        if m["status"] != "completed" or m["completed_updates"] != 320:
            raise ValueError("training stopped short of the registered 320 updates")
        execute("training_backup", [sys.executable, "scripts/sync_run_artifacts.py", "upload", "--run-id", PILOT, "--include-adapter", "--receipt", str(root / "training-sync-receipt.json")], 240)
        adapter = str(root / "adapter")
        execute("development", [sys.executable, "scripts/eval_stress.py", "--suite", "development", "--policies", "model", "--policy-label", "scale10x", "--adapter-dir", adapter, "--data-dirs", "data/batch1", "data/batch2", "--out-dir", str(root / "stress-development")], 3600)
        execute("stage5", [sys.executable, "scripts/eval_closed_loop.py", "--policies", "model", "--modes", "strict", "--model-label", "scale10x", "--adapter-dir", adapter, "--data-dirs", "data/batch1", "data/batch2", "--gen-batch-size", "64", "--teacher-workers", "3", "--device", "cuda", "--out-dir", str(root / "stage5")], 2700)
        execute("fresh_recovery", [sys.executable, "scripts/eval_recovery_only.py", "--registration", str(PROTOCOL), "--adapter-dir", adapter, "--label", "scale10x", "--out-dir", str(root / "fresh-recovery")], 900)
        execute("paired_analysis", [sys.executable, "scripts/stage6_scale10x.py", "assess"], 600)
        status = read(root / "scale10x-gate.json")["status"]
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        status = "incomplete"
        print(failure, flush=True)
    finally:
        finished = {"status": status, "phase": "workflow_finished", "failure": failure,
            "finished_at": now(), "deadline_epoch": deadline, "instance_id": ledger["instance_id"],
            "research_complete": False, "operations_complete": False,
            "note": "Independent artifact audit, report and exact cleanup remain required."}
        atomic_write_json(block, finished)
        for run in RUNS:
            root = Path("runs") / run / "rl"
            if (root / "manifest.json").exists():
                atomic_write_json(root / "block-state.json", {**finished, "status": status if run == PILOT else "passed" if status in ("passed", "not_passed") else "failed"})
        sync_errors = []
        for run in (*RUNS, "stage6-scale10x-v1"):
            root = Path("runs") / run / "rl"
            if not root.exists():
                continue
            command = [sys.executable, "scripts/sync_run_artifacts.py", "upload", "--run-id", run,
                "--include-adapter", "--receipt", str(root / "sync-receipt.json")]
            try:
                if run_process(command, ROOT / "final-sync.log", max(1, min(180, deadline - time.time() - 30))):
                    raise RuntimeError("artifact upload failed")
                if run_process(command[:-3], ROOT / "final-sync.log", max(1, min(60, deadline - time.time() - 15))):
                    raise RuntimeError("receipt upload failed")
            except Exception as error:
                sync_errors.append({"run": run, "error": str(error)})
        atomic_write_json(ROOT / "shutdown-receipt.json", {"requested_at": now(), "sync_errors": sync_errors,
            "instance_id": ledger["instance_id"], "preserve_encrypted_ebs": True})
        # Stop, never terminate, until an independent observer verifies the artifacts.
        subprocess.run(["sudo", "/usr/sbin/shutdown", "-h", "now"], check=False)
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
