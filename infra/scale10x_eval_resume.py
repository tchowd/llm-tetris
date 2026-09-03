#!/usr/bin/env python3
"""Evaluation-only recovery after completed training; frozen science is unchanged."""
from __future__ import annotations

import argparse
import copy
import fcntl
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra import scale10x_continuation as c
from infra.scale10x_resume import run_process

ROOT, PILOT_ROOT = c.ROOT, c.PILOT_ROOT
PLAN = ROOT / "eval-resume-plan-v1.json"
BLOCK = ROOT / "eval-resume-block-state-v1.json"
ARCHIVE = ROOT / "attempts/evaluation-startup-1-worker"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
BUCKET = "s3://llm-tetris-artifacts-566629888938-us-east-1/"


def commands():
    adapter = str(PILOT_ROOT / "adapter")
    return [
        ("development", [sys.executable, "scripts/eval_stress.py", "--suite", "development", "--policies", "model",
            "--policy-label", "scale10x", "--adapter-dir", adapter, "--data-dirs", "data/batch1", "data/batch2",
            "--out-dir", str(PILOT_ROOT / "stress-development")], 3600),
        ("stage5", [sys.executable, "scripts/eval_closed_loop.py", "--policies", "model", "--modes", "strict",
            "--model-label", "scale10x", "--adapter-dir", adapter, "--data-dirs", "data/batch1", "data/batch2",
            "--gen-batch-size", "64", "--teacher-workers", "3", "--device", "cuda", "--out-dir", str(PILOT_ROOT / "stage5")], 2700),
        ("fresh_recovery", [sys.executable, "scripts/eval_recovery_only.py", "--registration", str(c.original.PROTOCOL),
            "--adapter-dir", adapter, "--label", "scale10x", "--out-dir", str(PILOT_ROOT / "fresh-recovery")], 900),
        ("paired_analysis", [sys.executable, "infra/scale10x_continuation.py", "assess"], 600),
    ]


def validate_training(m, cpu, backup):
    if (m.get("status") != "completed" or m.get("completed_updates") != 320 or m.get("sample_count") != 106727
            or not m.get("resume_used") or m.get("resumed_from_update") != 164):
        raise ValueError("only the inspected completed training may be evaluated")
    if (cpu.get("status") != "passed" or cpu.get("update") != 320 or cpu.get("samples") != m["sample_count"]
            or not cpu.get("checks") or not all(cpu["checks"].values())):
        raise ValueError("final checkpoint CPU integrity required")
    if (backup.get("status") != "passed" or backup.get("encryption") != "AES256"
            or not backup.get("adapter_directory_hash_recomputed_from_s3")
            or backup.get("adapter_directory_sha256") != m["output_adapter_sha256"]):
        raise ValueError("independent final-adapter backup required")


def validate_empty_failure(directory):
    m = c.read(directory / "manifest.json")
    if m.get("status") != "failed" or m.get("suite") != "development":
        raise ValueError("expected the failed development startup")
    if (directory / "games.jsonl").stat().st_size or (directory / "states.jsonl").stat().st_size:
        raise ValueError("evaluation contains results; inspect instead of restarting")
    if {p.name for p in directory.iterdir()} != {"manifest.json", "games.jsonl", "states.jsonl", "events.jsonl"}:
        raise ValueError("unexpected evaluation artifacts")


def verify_cache():
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoConfig, AutoTokenizer
    root = Path(HF_HUB_CACHE) / "models--Qwen--Qwen3-1.7B"
    if (root / "refs/main").read_text().strip() != REVISION:
        raise ValueError("offline default alias must resolve to the original pinned commit")
    config = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B", local_files_only=True)
    if config._commit_hash != REVISION:
        raise ValueError("resolved base model commit differs")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", local_files_only=True)
    if not tokenizer.encode("Stage 6 cache preflight"):
        raise ValueError("cached tokenizer failed")
    return {"base_model_revision": config._commit_hash, "tokenizer_loaded_offline": True}


def prepare_cache(root):
    from huggingface_hub.file_download import _cache_commit_hash_for_specific_revision
    ref = root / "refs/main"
    if ref.exists() and ref.read_text() != REVISION:
        raise ValueError("refusing to overwrite an existing default model reference")
    if not (root / "snapshots" / REVISION / "config.json").is_file():
        raise ValueError("original pinned snapshot missing")
    # Use the cache library's atomic reference writer: refs must have no newline.
    _cache_commit_hash_for_specific_revision(str(root), "main", REVISION)
    if ref.read_text() != REVISION:
        raise ValueError("cache reference write failed")


def preflight():
    a, q = c.validate()
    plan = c.read(PLAN)
    if (plan["deadline_epoch"] != a["absolute_deadline_epoch"] or plan["base_model_revision"] != REVISION
            or plan["registration_sha256"] != c.file_sha256(c.REGISTRATION)
            or plan["amendment_sha256"] != c.file_sha256(c.AMENDMENT)):
        raise ValueError("evaluation recovery differs from unchanged registration")
    for name, digest in plan["input_sha256"].items():
        if c.file_sha256(Path(name)) != digest:
            raise ValueError(f"evaluation recovery input changed: {name}")
    if BLOCK.exists() or ARCHIVE.exists() or (PILOT_ROOT / "scale10x-gate.json").exists():
        raise ValueError("evaluation recovery already attempted; no silent retry")
    m = c.read(PILOT_ROOT / "manifest.json")
    validate_training(m, c.read(ROOT / "final-checkpoint-cpu-audit-v1.json"), c.read(ROOT / "final-adapter-backup-audit-v1.json"))
    r, p = c.amended_recipe(q)
    c.original.training_evidence(PILOT_ROOT, r, p, c.REGISTRATION)
    validate_empty_failure(PILOT_ROOT / "stress-development")
    if any((PILOT_ROOT / name).exists() for name in ("stage5", "fresh-recovery")):
        raise ValueError("unexpected later evaluation outputs")
    ledger = c.read(ROOT / "compute-ledger.json")
    projection = copy.deepcopy(a)
    projection["projected_training_seconds"] = 0
    c.launch_allocation(projection, ledger, time.time())
    if ledger["deadline_epoch"] - time.time() < 9000:
        raise ValueError("frozen evaluation and backup reserve no longer fit")
    timer = subprocess.check_output(["systemctl", "show", "llm-tetris-scale10x-continuation-deadline.timer",
        "-p", "NextElapseUSecRealtime", "-p", "ActiveState"], text=True)
    if "ActiveState=active" not in timer or "Fri 2026-09-04 12:24:06 UTC" not in timer:
        raise ValueError("original absolute timer required")
    if c.file_sha256(Path("/etc/needrestart/conf.d/llm-tetris-scale10x-eval.conf")) != c.file_sha256(Path("infra/rl-scale10x-eval-needrestart.conf")):
        raise ValueError("scoped evaluation maintenance exclusion required")
    for service in ("llm-tetris-scale10x.service", "llm-tetris-scale10x-continuation.service", "llm-tetris-scale10x-resume-v1.service"):
        state = subprocess.run(["systemctl", "is-active", service], capture_output=True, text=True).stdout.strip()
        if state not in ("inactive", "failed", "unknown"):
            raise ValueError("prior runner still active")
    return ledger, verify_cache()


def archive_failure():
    ARCHIVE.mkdir(parents=True, exist_ok=False)
    shutil.copytree(PILOT_ROOT / "stress-development", ARCHIVE / "stress-development")
    for source in [PILOT_ROOT / "manifest.json", PILOT_ROOT / "block-state.json",
                   ROOT / "resume-block-state-v1.json", ROOT / "resume-development-v1.log",
                   ROOT / "resume-shutdown-receipt-v1.json", ROOT / "compute-ledger.json"]:
        destination = ARCHIVE / ("pilot" if source.parent == PILOT_ROOT else "workflow") / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    hashes = {str(p.relative_to(ARCHIVE)): c.file_sha256(p) for p in sorted(ARCHIVE.rglob("*")) if p.is_file()}
    c.write_new(ARCHIVE / "archive-sha256.json", hashes)
    subprocess.run(["aws", "s3", "cp", str(ARCHIVE), BUCKET + str(ARCHIVE) + "/",
        "--recursive", "--sse", "AES256", "--only-show-errors"], timeout=300, check=True)
    # Recoverable move only after encrypted archive upload. No training files move.
    shutil.move(str(PILOT_ROOT / "stress-development"), str(ARCHIVE / "original-stress-development"))


def workflow(ledger, cache):
    deadline, status, failure = ledger["deadline_epoch"], "incomplete", None
    try:
        archive_failure()
        c.write_new(ROOT / "eval-resume-launch-v1.json", {"status": "passed", "generated_at": c.now(),
            "plan_sha256": c.file_sha256(PLAN), "deadline_epoch": deadline, "cache": cache,
            "training_repeated": False, "commands": commands()})
        for phase, command, maximum in commands():
            c.atomic_write_json(BLOCK, {"status": "running", "phase": phase, "updated_at": c.now(),
                "deadline_epoch": deadline, "instance_id": ledger["instance_id"], "training_complete": True})
            if run_process(command, ROOT / f"eval-resume-{phase}-v1.log", min(maximum, deadline - time.time() - 600)):
                raise RuntimeError(f"{phase} exited nonzero; inspect preserved evidence")
        status = c.read(PILOT_ROOT / "scale10x-gate.json")["status"]
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        print(failure, flush=True)
    finally:
        finished = {"status": status, "phase": "workflow_finished", "failure": failure, "finished_at": c.now(),
            "instance_id": ledger["instance_id"], "deadline_epoch": deadline, "training_complete": True,
            "research_complete": False, "operations_complete": False,
            "note": "Independent audits, applicable replication, report and exact cleanup remain required."}
        c.atomic_write_json(BLOCK, finished)
        c.atomic_write_json(PILOT_ROOT / "block-state.json", finished)
        errors = []
        try:
            for run in (c.PILOT, "stage6-scale10x-v1"):
                receipt = Path("runs") / run / "rl" / ("sync-receipt.json" if run == c.PILOT else "eval-resume-sync-receipt-v1.json")
                cmd = [sys.executable, "scripts/sync_run_artifacts.py", "upload", "--run-id", run,
                    "--include-adapter", "--receipt", str(receipt)]
                for sync_cmd, maximum in ((cmd, 240), (cmd[:-3], 60)):
                    try:
                        if run_process(sync_cmd, ROOT / "eval-resume-final-sync-v1.log", max(1, min(maximum, deadline - time.time() - 30))):
                            raise RuntimeError("artifact upload failed")
                    except BaseException as error:
                        errors.append({"run": run, "error": f"{type(error).__name__}: {error}"})
            receipt = ROOT / "eval-resume-shutdown-receipt-v1.json"
            c.atomic_write_json(receipt, {"requested_at": c.now(), "sync_errors": errors,
                "instance_id": ledger["instance_id"], "preserve_encrypted_ebs": True})
            subprocess.run(["aws", "s3", "cp", str(receipt), BUCKET + str(receipt), "--sse", "AES256", "--only-show-errors"], timeout=20, check=False)
        finally:
            subprocess.run(["sudo", "/usr/sbin/shutdown", "-h", "now"], check=False)
    if failure:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-cache", "preflight", "run"))
    args = parser.parse_args()
    if args.action == "prepare-cache":
        from huggingface_hub.constants import HF_HUB_CACHE
        prepare_cache(Path(HF_HUB_CACHE) / "models--Qwen--Qwen3-1.7B")
        print(verify_cache())
    elif args.action == "preflight":
        ledger, cache = preflight()
        print({"status": "passed", "deadline_epoch": ledger["deadline_epoch"], "cache": cache})
    else:
        with (ROOT / "workflow.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            workflow(*preflight())


if __name__ == "__main__":
    main()
