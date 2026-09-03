"""Read-only project dashboard sources and derived status engine.

The dashboard deliberately owns no project metrics.  Every value returned
here is derived from repository artifacts, structured run events, or AWS
read APIs.  Missing and denied sources remain visible as freshness/errors;
they are never converted to zeroes.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .dashboard_thresholds import THRESHOLDS

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11+ in production
    tomllib = None


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
DATA_DIR = ROOT / "data"
STATUS_DIR = RUNS_DIR / "status"

STATUS_PRECEDENCE = {
    "failed": 7,
    "blocked": 6,
    "running": 5,
    "stale": 4,
    "ready": 3,
    "passed": 2,
    "not_started": 1,
}

STAGES = [
    {
        "number": 1,
        "slug": "game",
        "name": "Game + API",
        "short": "Engine",
        "purpose": "Freeze the deterministic game, action API, serializer, and browser contract.",
        "prerequisites": [],
        "gate": "Engine, replay, serializer, API, and web build checks pass on the current commit.",
    },
    {
        "number": 2,
        "slug": "teacher",
        "name": "Teacher",
        "short": "Teacher",
        "purpose": "Prove the two-ply Dellacherie teacher is correct, deterministic, and far above random.",
        "prerequisites": [1],
        "gate": "Feature tests and benchmark pass; teacher weights match downstream data.",
    },
    {
        "number": 3,
        "slug": "dataset",
        "name": "Dataset",
        "short": "Dataset",
        "purpose": "Generate replayable teacher traces and validate every label, split, and prompt.",
        "prerequisites": [1, 2],
        "gate": "Every batch has a passing machine-readable validation report and clean lineage.",
    },
    {
        "number": 4,
        "slug": "sft",
        "name": "Supervised fine-tune",
        "short": "SFT",
        "purpose": "Clone the teacher into Qwen3-1.7B with completion-only LoRA training.",
        "prerequisites": [3],
        "gate": "Adapter exists; parse/legality are at ceiling and held-out exact match is at least 70%.",
    },
    {
        "number": 5,
        "slug": "eval",
        "name": "Closed-loop evaluation",
        "short": "Eval",
        "purpose": "Measure the model on boards it creates, against random and teacher on shared seeds.",
        "prerequisites": [4],
        "gate": "Strict rollouts are legal, clearly above random, replayable, and survive beyond early failure.",
    },
    {
        "number": 6,
        "slug": "rl",
        "name": "RL research",
        "short": "RL",
        "purpose": "Test dense and episode-return learning without regressing format, legality, or the frozen SFT baseline.",
        "prerequisites": [5],
        "gate": "Stress-v1 is frozen, trajectory proofs pass, and replicated results support retaining either RL or SFT.",
    },
    {
        "number": 7,
        "slug": "post",
        "name": "Post-implementation",
        "short": "Release",
        "purpose": "Confirm capacity, reproducibility, retained artifacts, security, cost cleanup, and final benchmarks.",
        "prerequisites": [5],
        "gate": "Release checklist, cleanup evidence, retained adapter, and final comparison report all pass.",
    },
]

STAGE_SOURCE_PATTERNS = {
    1: ["server.py", "tetris/board.py", "tetris/engine.py", "tetris/features.py", "tetris/pieces.py", "tetris/placement.py", "tetris/serialize.py", "web/index.html", "web/package.json", "web/vite.config.js", "web/src/*.js", "web/src/*.css", "tests/test_bag.py", "tests/test_features.py", "tests/test_game_over.py", "tests/test_line_clear.py", "tests/test_placement.py", "tests/test_replay.py", "tests/test_serialize.py"],
    2: ["tetris/teacher.py", "tetris/placement.py", "tests/test_teacher_*.py"],
    6: ["STAGE6.md", "benchmarks/stress-v1/*", "infra/rl-*.sh", "infra/rl-instance.template.json", "requirements-rl.txt", "tetris/rl.py", "tetris/recovery.py", "tetris/model_policy.py", "tetris/rollout.py", "scripts/*rl*.py", "scripts/*recovery*.py", "scripts/*episode*.py", "scripts/train_sft.py", "scripts/generate_stress_manifest.py", "scripts/eval_stress.py", "scripts/analyze_stage6.py", "scripts/check_e2_learning.py", "scripts/select_e3_kl.py", "scripts/check_e4_pilot.py", "scripts/eval_closed_loop.py", "scripts/verify_stages.py", "tests/test_rl.py", "tests/test_recovery*.py", "tests/test_stress_eval.py", "tests/test_episode*.py", "tests/test_grpo_integration.py", "tests/test_stage6_analysis.py"],
    7: ["DASHBOARD.md", "infra/*", "server.py", "tetris/dashboard*.py", "web/src/*"],
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_now() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except (ValueError, TypeError):
        return None


def file_timestamp(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat().replace("+00:00", "Z")
    except OSError:
        return None


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def stage_source_fingerprint(stage: int) -> str:
    digest = hashlib.sha256()
    files: set[Path] = set()
    for pattern in STAGE_SOURCE_PATTERNS.get(stage, []):
        files.update(path for path in ROOT.glob(pattern) if path.is_file())
    for path in sorted(files):
        digest.update(relative(path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path) -> tuple[dict | list | None, str | None]:
    try:
        return json.loads(path.read_text()), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


def git_info() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    sha = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    dirty = bool(run("status", "--porcelain"))
    return {"sha": sha, "short_sha": sha[:8] if sha else None, "branch": branch, "dirty": dirty}


def issue(
    issue_id: str,
    severity: str,
    stage: int | None,
    title: str,
    *,
    observed: Any = None,
    expected: Any = None,
    evidence: str | None = None,
    detected_at: str | None = None,
    next_action: str,
    source: str = "local",
    scope: str | None = None,
) -> dict:
    return {
        "id": issue_id,
        "severity": severity,
        "stage": stage,
        "scope": scope or (f"stage:{stage}" if stage else "project"),
        "title": title,
        "observed": observed,
        "expected": expected,
        "evidence": evidence,
        "detected_at": detected_at or iso_now(),
        "next_action": next_action,
        "source": source,
        "state": "open",
    }


def progress(current: int | float | None, total: int | float | None, unit: str, label: str | None = None) -> dict:
    value = None
    if current is not None and total:
        value = min(1.0, max(0.0, float(current) / float(total)))
    return {"current": current, "total": total, "unit": unit, "value": value, "label": label or "unknown"}


def _read_events(path: Path, limit: int = 500) -> list[dict]:
    """Read a bounded tail without loading a multi-GB event file."""
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            block = 8192
            data = b""
            while end > 0 and data.count(b"\n") <= limit:
                take = min(block, end)
                end -= take
                handle.seek(end)
                data = handle.read(take) + data
        rows = []
        for raw in data.splitlines()[-limit:]:
            try:
                rows.append(json.loads(raw))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        return rows
    except OSError:
        return []


def discover_datasets() -> dict:
    batches = []
    errors = []
    if not DATA_DIR.exists():
        return {"batches": [], "totals": None, "errors": [], "updated_at": None}
    for manifest_path in sorted(DATA_DIR.glob("*/manifest.json")):
        manifest, err = read_json(manifest_path)
        if err or not isinstance(manifest, dict):
            errors.append({"source": relative(manifest_path), "message": err or "invalid manifest"})
            continue
        directory = manifest_path.parent
        validation_path = directory / "validation.json"
        validation, validation_error = read_json(validation_path)
        if validation_error == "missing":
            validation = None
            validation_error = None
        rows_path = directory / "rows.jsonl"
        games_path = directory / "games.jsonl"
        validation_ok = None
        checks_passed = checks_total = None
        if isinstance(validation, dict):
            checks = validation.get("checks", validation.get("report", {}))
            if isinstance(checks, dict):
                checks_total = len(checks)
                checks_passed = sum(1 for value in checks.values() if isinstance(value, dict) and value.get("ok"))
                validation_ok = checks_total > 0 and checks_passed == checks_total
            validation_ok = validation.get("ok", validation_ok)
        batches.append(
            {
                "id": directory.name,
                "path": relative(directory),
                "manifest": manifest,
                "games": manifest.get("num_games"),
                "rows": manifest.get("num_rows"),
                "died": manifest.get("died_count"),
                "search_depth": manifest.get("search_depth"),
                "git_sha": manifest.get("git_sha"),
                "generated_at": manifest.get("generated_at") or file_timestamp(manifest_path),
                "validation": {
                    "status": "passed" if validation_ok else ("failed" if validation_ok is False else "missing"),
                    "checks_passed": checks_passed,
                    "checks_total": checks_total,
                    "path": relative(validation_path),
                    "updated_at": file_timestamp(validation_path),
                },
                "files": {
                    "rows_bytes": rows_path.stat().st_size if rows_path.exists() else None,
                    "games_bytes": games_path.stat().st_size if games_path.exists() else None,
                    "rows_exists": rows_path.exists(),
                    "games_exists": games_path.exists(),
                },
            }
        )
    totals = None
    if batches:
        totals = {
            "batches": len(batches),
            "games": sum(b["games"] or 0 for b in batches),
            "rows": sum(b["rows"] or 0 for b in batches),
            "deaths": sum(b["died"] or 0 for b in batches),
            "validated_batches": sum(b["validation"]["status"] == "passed" for b in batches),
        }
    timestamps = [parse_timestamp(b["generated_at"]) for b in batches]
    timestamps = [stamp for stamp in timestamps if stamp]
    return {
        "batches": batches,
        "totals": totals,
        "errors": errors,
        "updated_at": max(timestamps).isoformat().replace("+00:00", "Z") if timestamps else None,
    }


def _run_from_manifest(path: Path, stage: int, kind: str) -> dict | None:
    manifest, err = read_json(path)
    if err or not isinstance(manifest, dict):
        return None
    run_dir = path.parent
    if kind == "closed_loop":
        run_dir = path.parent
        run_id = manifest.get("run_id") or (path.parent.parent.name if path.parent.name == "closed_loop" else path.parent.name)
    else:
        run_id = manifest.get("run_id") or run_dir.name
    events_path = run_dir / "events.jsonl"
    events = _read_events(events_path)
    last_event = events[-1] if events else None
    status = manifest.get("status")
    status = {"registered": "running", "completed": "passed", "stopped_budget": "failed"}.get(status, status)
    if not status:
        if last_event and last_event.get("type") == "job_failed":
            status = "failed"
        elif last_event and last_event.get("type") == "job_completed":
            status = "passed"
        elif last_event:
            stamp = parse_timestamp(last_event.get("timestamp"))
            status = "running" if stamp and (utc_now() - stamp).total_seconds() < THRESHOLDS["heartbeat_stale_seconds"] else "stale"
        else:
            status = "passed" if path.exists() else "not_started"
    metrics = None
    if stage == 4:
        metrics_path = run_dir / "open_loop_metrics.json"
        metrics, _ = read_json(metrics_path)
    elif stage == 5:
        metrics_path = run_dir / "metrics.json"
        metrics, _ = read_json(metrics_path)
    else:
        metrics_path = None
    progress_data = None
    for event in reversed(events):
        if event.get("type") in {"progress", "train_metrics", "eval_metrics"}:
            progress_data = {
                "current": event.get("current"),
                "total": event.get("total"),
                "phase": event.get("phase"),
                "metrics": event.get("metrics", {}),
                "timestamp": event.get("timestamp"),
            }
            break
    return {
        "run_id": run_id,
        "stage": stage,
        "kind": kind,
        "status": status,
        "path": relative(run_dir),
        "manifest_path": relative(path),
        "manifest": manifest,
        "metrics": metrics,
        "metrics_path": relative(metrics_path) if metrics_path else None,
        "events_path": relative(events_path),
        "events": events,
        "last_event": last_event,
        "progress": progress_data,
        "updated_at": (last_event or {}).get("timestamp") or file_timestamp(path),
        "git_sha": manifest.get("git_sha"),
        "backend": manifest.get("backend"),
        "host": manifest.get("host") or manifest.get("device"),
        "parent_run_ids": manifest.get("parent_run_ids", []),
    }


def discover_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    found: dict[tuple[int, str, str], dict] = {}
    for path in RUNS_DIR.rglob("train_manifest.json"):
        run = _run_from_manifest(path, 4, "sft")
        if run:
            found[(4, run["run_id"], run["path"])] = run
    for path in RUNS_DIR.rglob("closed_loop/manifest.json"):
        run = _run_from_manifest(path, 5, "closed_loop")
        if run:
            found[(5, run["run_id"], run["path"])] = run
    for path in RUNS_DIR.rglob("rl/manifest.json"):
        run = _run_from_manifest(path, 6, "rl")
        if run:
            found[(6, run["run_id"], run["path"])] = run
    return sorted(found.values(), key=lambda run: run.get("updated_at") or "", reverse=True)


def verification_report(stage: int) -> tuple[dict | None, Path]:
    path = STATUS_DIR / f"stage-{stage}.json"
    report, _ = read_json(path)
    return report if isinstance(report, dict) else None, path


def _stage_local_state(datasets: dict, runs: list[dict], git: dict) -> tuple[list[dict], list[dict]]:
    issues: list[dict] = []
    stages: list[dict] = []
    by_number: dict[int, dict] = {}

    for spec in STAGES:
        number = spec["number"]
        report, report_path = verification_report(number)
        status = "not_started"
        prog = progress(None, None, "checks")
        evidence: list[dict] = []
        gate_state = "unknown"
        next_action = "Review the stage plan and create its first evidence artifact."

        if report:
            report_status = report.get("status") or ("passed" if report.get("ok") else "failed")
            status = report_status if report_status in STATUS_PRECEDENCE else "stale"
            current = report.get("checks_passed")
            total = report.get("checks_total")
            prog = progress(current, total, "checks", f"{current}/{total} checks" if current is not None and total is not None else None)
            gate_state = "passed" if status == "passed" else "failed"
            evidence.append({"label": "Verification report", "path": relative(report_path), "timestamp": report.get("generated_at") or file_timestamp(report_path), "state": status})
            if report.get("git_sha") and git.get("sha") and report["git_sha"] != git["sha"]:
                status = "stale"
                gate_state = "stale"
                issues.append(issue(f"stage{number}.verification.stale", "amber", number, "Verification report is from another commit", observed=report["git_sha"][:8], expected=git["short_sha"], evidence=relative(report_path), next_action=f"Run scripts/verify_stages.py --stage {number} on the current commit."))
            elif report.get("source_fingerprint") != stage_source_fingerprint(number):
                status = "stale"
                gate_state = "stale"
                issues.append(issue(f"stage{number}.verification.source_drift", "amber", number, "Verification evidence predates relevant working-tree changes", observed=(report.get("source_fingerprint") or "missing")[:8], expected=stage_source_fingerprint(number)[:8], evidence=relative(report_path), next_action=f"Run scripts/verify_stages.py --stage {number} after the source changes."))

        if number == 1:
            required = [ROOT / "tetris/engine.py", ROOT / "server.py", ROOT / "web/src/main.js"]
            implemented = all(path.exists() for path in required)
            if not report and implemented:
                status, gate_state = "stale", "missing"
                prog = progress(3, 4, "evidence", "implementation present · verification missing")
                issues.append(issue("stage1.verification.missing", "amber", 1, "Stage 1 current-commit verification is missing", evidence=relative(report_path), next_action="Run scripts/verify_stages.py --stage 1."))
            next_action = "Run the current-commit Stage 1 verification." if status != "passed" else "Keep engine and serializer contracts frozen."

        elif number == 2:
            implemented = (ROOT / "tetris/teacher.py").exists()
            if not report and implemented:
                status, gate_state = "stale", "missing"
                prog = progress(2, 3, "evidence", "teacher present · benchmark report missing")
                issues.append(issue("stage2.verification.missing", "amber", 2, "Teacher benchmark evidence is missing", evidence=relative(report_path), next_action="Run scripts/verify_stages.py --stage 2."))
            next_action = "Regenerate the teacher correctness and benchmark report." if status != "passed" else "Confirm dataset manifests retain these teacher weights."

        elif number == 3:
            batches = datasets["batches"]
            if batches:
                validations = sum(batch["validation"]["status"] == "passed" for batch in batches)
                missing_files = [batch["id"] for batch in batches if not all((batch["files"]["rows_exists"], batch["files"]["games_exists"]))]
                prog = progress(validations, len(batches), "batches", f"{validations}/{len(batches)} validated")
                status = "passed" if validations == len(batches) and not missing_files else "stale"
                gate_state = "passed" if status == "passed" else "missing"
                for batch in batches:
                    evidence.append({"label": f"Dataset {batch['id']}", "path": f"{batch['path']}/manifest.json", "timestamp": batch["generated_at"], "state": batch["validation"]["status"]})
                    if batch["validation"]["status"] == "missing":
                        issues.append(issue(f"stage3.{batch['id']}.validation_missing", "amber", 3, f"{batch['id']} has no durable validation report", evidence=batch["validation"]["path"], next_action=f"Run validate_dataset.py --data-dir {batch['path']} --report-json {batch['validation']['path']}."))
                    if batch.get("git_sha") and git.get("sha") and batch["git_sha"] != git["sha"]:
                        issues.append(issue(f"stage3.{batch['id']}.lineage", "amber", 3, f"{batch['id']} was generated from an older commit", observed=batch["git_sha"][:8], expected=git["short_sha"], evidence=f"{batch['path']}/manifest.json", next_action="Confirm the engine/teacher contract has not changed; regenerate only if lineage is invalid."))
                if missing_files:
                    status, gate_state = "failed", "failed"
                    issues.append(issue("stage3.dataset.files_missing", "red", 3, "Dataset manifest points to missing source files", observed=missing_files, expected="games.jsonl and rows.jsonl", next_action="Restore or regenerate the affected batch."))
                next_action = "Write machine-readable validation reports for every batch." if validations < len(batches) else "Use the validated split for SFT."
            else:
                prereqs_ok = all(by_number.get(n, {}).get("status") == "passed" for n in spec["prerequisites"])
                status = "ready" if prereqs_ok else "blocked"
                next_action = "Generate and validate the smoke batch."

        elif number == 4:
            stage_runs = [run for run in runs if run["stage"] == 4]
            active = next((run for run in stage_runs if run["status"] == "running"), None)
            latest = stage_runs[0] if stage_runs else None
            if active:
                status, gate_state = "running", "pending"
                event = active.get("progress") or {}
                prog = progress(event.get("current"), event.get("total"), "steps", event.get("phase") or "training")
                next_action = "Monitor loss, GPU utilization, heartbeat, and the next held-out evaluation."
            elif latest:
                metrics = latest.get("metrics") or {}
                exact = metrics.get("exact_match")
                parse = metrics.get("parse_rate")
                legality = metrics.get("legality_rate")
                adapter_exists = (ROOT / latest["path"] / "adapter").exists()
                passed = adapter_exists and exact is not None and exact >= THRESHOLDS["stage4_exact_match"] and (parse or 0) >= THRESHOLDS["stage4_parse_rate"] and (legality or 0) >= THRESHOLDS["stage4_legality_rate"]
                status, gate_state = ("passed", "passed") if passed else ("failed", "failed")
                prog = progress(metrics.get("n"), metrics.get("n"), "eval rows", f"exact match {exact:.1%}" if exact is not None else "open-loop metrics missing")
                evidence.append({"label": "SFT run", "path": latest["manifest_path"], "timestamp": latest["updated_at"], "state": latest["status"]})
                if not adapter_exists:
                    issues.append(issue("stage4.adapter.missing", "red", 4, "Training run has no adapter artifact", evidence=latest["manifest_path"], next_action="Retrieve or regenerate the adapter before evaluation.", scope=f"run:{latest['run_id']}"))
                if exact is None:
                    issues.append(issue("stage4.open_loop.missing", "red", 4, "Open-loop gate metrics are missing", evidence=latest.get("metrics_path"), next_action="Run scripts/eval_open_loop.py against the saved adapter.", scope=f"run:{latest['run_id']}"))
                elif exact < THRESHOLDS["stage4_exact_match"]:
                    issues.append(issue("stage4.open_loop.exact_match", "red", 4, "Open-loop exact match is below gate", observed=exact, expected=f">= {THRESHOLDS['stage4_exact_match']:.0%}", evidence=latest.get("metrics_path"), next_action="Review piece and holes-bucket breakdown before retraining.", scope=f"run:{latest['run_id']}"))
                next_action = "Hand the adapter to closed-loop evaluation." if passed else "Complete or repair the Stage 4 gate."
            else:
                stage3_passed = by_number.get(3, {}).get("status") == "passed"
                status = "ready" if stage3_passed else "blocked"
                gate_state = "pending"
                prog = progress(0, 1, "run", "ready for first SFT run" if status == "ready" else "waiting for the Stage 3 gate")
                next_action = "Launch the documented Unsloth smoke test on the A10G instance." if status == "ready" else "Write passing validation reports for every Stage 3 batch first."
                if status == "ready":
                    issues.append(issue("stage4.run.not_started", "info", 4, "SFT training has not started", evidence="plan/stage-4-sft.md", next_action=next_action))

        elif number == 5:
            stage_runs = [run for run in runs if run["stage"] == 5]
            active = next((run for run in stage_runs if run["status"] == "running"), None)
            latest = stage_runs[0] if stage_runs else None
            if active:
                status, gate_state = "running", "pending"
                event = active.get("progress") or {}
                prog = progress(event.get("current"), event.get("total"), "games", event.get("phase") or "rollout")
                next_action = "Monitor strict-mode failures and seed coverage."
            elif latest and isinstance(latest.get("metrics"), dict):
                strict = ((latest["metrics"].get("model") or {}).get("strict") or {})
                lines = ((strict.get("lines") or {}).get("mean"))
                parse_fail = ((strict.get("parse_failure_rate") or {}).get("mean"))
                illegal = ((strict.get("illegal_rate") or {}).get("mean", 0))
                passed = lines is not None and lines >= THRESHOLDS["stage5_min_mean_lines"] and (parse_fail or 0) <= 0.01 and (illegal or 0) <= 0.01
                status, gate_state = ("passed", "passed") if passed else ("failed", "failed")
                games = strict.get("n_games")
                prog = progress(games, len(latest["manifest"].get("seeds", [])) or games, "games", f"strict mean {lines:.1f} lines" if lines is not None else "model strict metrics missing")
                evidence.append({"label": "Closed-loop run", "path": latest["manifest_path"], "timestamp": latest["updated_at"], "state": status})
                if not passed:
                    issues.append(issue("stage5.strict.gate", "red", 5, "Strict closed-loop rollout does not pass the survival gate", observed=lines, expected=f">= {THRESHOLDS['stage5_min_mean_lines']} mean lines with valid actions", evidence=latest.get("metrics_path"), next_action="Compare on-policy teacher match to held-out exact match and route the failure to Stage 3 or 4.", scope=f"run:{latest['run_id']}"))
                next_action = "Record the RL readiness verdict." if passed else "Diagnose the strict-mode gate failure upstream."
            else:
                stage4_passed = by_number.get(4, {}).get("status") == "passed"
                status = "ready" if stage4_passed else "blocked"
                gate_state = "pending"
                prog = progress(0, 100, "games", "waiting for a gated adapter")
                next_action = "Run the shared 100-seed random/teacher/model harness." if stage4_passed else "Pass the Stage 4 adapter gate first."

        elif number == 6:
            stage_runs = [run for run in runs if run["stage"] == 6]
            if stage_runs:
                latest = stage_runs[0]
                status = latest["status"] if latest["status"] in {"running", "failed", "stale"} else "ready"
                gate_state = "pending"
                event = latest.get("progress") or {}
                prog = progress(event.get("current"), event.get("total"), "updates", event.get("phase") or "RL")
                evidence.append({"label": "RL run", "path": latest["manifest_path"], "timestamp": latest["updated_at"], "state": status})
                next_action = "Continue the registered ladder or compare promoted checkpoints with frozen SFT."
            else:
                stage5_passed = by_number.get(5, {}).get("status") == "passed"
                status = "ready" if stage5_passed else "blocked"
                gate_state = "pending"
                prog = progress(
                    0,
                    7,
                    "experiments",
                    "stress-v1 registered · E0 control next" if stage5_passed else "awaiting Stage 5 verdict",
                )
                next_action = "Reproduce the frozen SFT control on stress-v1." if stage5_passed else "Do not start RL until strict Stage 5 passes."
            if report and report.get("research_complete") and report.get("source_fingerprint") == stage_source_fingerprint(6):
                status, gate_state = "passed", "passed"
                next_action = "Retain the policy selected by the complete research report and archive the evidence."

        elif number == 7:
            report, report_path = verification_report(7)
            if report:
                status = report.get("status", "stale")
                current, total = report.get("checks_passed"), report.get("checks_total")
                prog = progress(current, total, "checks", f"{current}/{total} release checks" if current is not None else None)
                gate_state = "passed" if status == "passed" else "failed"
                evidence.append({"label": "Release report", "path": relative(report_path), "timestamp": report.get("generated_at") or file_timestamp(report_path), "state": status})
            else:
                status = "ready" if by_number.get(5, {}).get("status") == "passed" else "blocked"
                gate_state = "missing"
                prog = progress(0, 7, "checks", "release evidence not started")
            next_action = "Run final reproducibility, retention, IAM, and cost-cleanup checks." if status != "passed" else "Archive the accepted result."

        stage_issues = [item for item in issues if item["stage"] == number]
        stage = {
            **spec,
            "status": status,
            "progress": prog,
            "gate_result": gate_state,
            "evidence": evidence,
            "next_action": next_action,
            "issue_counts": {
                "red": sum(item["severity"] == "red" for item in stage_issues),
                "amber": sum(item["severity"] == "amber" for item in stage_issues),
                "info": sum(item["severity"] == "info" for item in stage_issues),
            },
        }
        stages.append(stage)
        by_number[number] = stage

    return stages, issues


def _select_current_stage(stages: list[dict]) -> dict:
    running = next((stage for stage in stages if stage["status"] == "running"), None)
    if running:
        return running
    ready = [stage for stage in stages if stage["status"] == "ready"]
    if ready:
        return ready[0]
    actionable = [stage for stage in stages if stage["status"] in {"failed", "stale"}]
    return actionable[0] if actionable else stages[-1]


def _overall_status(stages: list[dict], issues: list[dict], active_job: dict | None) -> str:
    if any(item["severity"] == "red" for item in issues):
        return "failed"
    if active_job:
        return "running"
    if any(item["severity"] == "amber" for item in issues):
        return "attention"
    if all(stage["status"] == "passed" for stage in stages):
        return "passed"
    return "ready"


def local_snapshot() -> dict:
    git = git_info()
    datasets = discover_datasets()
    runs = discover_runs()
    stages, issues = _stage_local_state(datasets, runs, git)
    active_job = next((run for run in runs if run["status"] == "running"), None)
    current = _select_current_stage(stages)
    return {
        "project": {
            "name": "LLM Tetris",
            "git": git,
            "overall_status": _overall_status(stages, issues, active_job),
            "current_stage": current["number"],
            "current_stage_name": current["name"],
            "next_action": current["next_action"],
            "active_job": active_job,
        },
        "stages": stages,
        "issues": issues,
        "datasets": datasets,
        "runs": runs,
    }


@dataclass(frozen=True)
class DashboardConfig:
    profile: str | None
    regions: tuple[str, ...]
    project_tag: str
    gpu_quota_code: str
    log_group: str
    metrics_namespace: str
    dashboard_user: str


def load_config() -> DashboardConfig:
    path = ROOT / "infra/dashboard.toml"
    raw: dict = {}
    if path.exists() and tomllib:
        try:
            raw = tomllib.loads(path.read_text()).get("aws", {})
        except (OSError, ValueError):
            raw = {}
    regions = raw.get("regions") or [os.getenv("AWS_REGION", "us-east-1")]
    return DashboardConfig(
        profile=raw.get("profile") or os.getenv("AWS_PROFILE"),
        regions=tuple(regions),
        project_tag=raw.get("project_tag", "llm-tetris"),
        gpu_quota_code=raw.get("gpu_quota_code", "L-DB2E81BA"),
        log_group=raw.get("log_group", "/llm-tetris/jobs"),
        metrics_namespace=raw.get("metrics_namespace", "LLMTetris/Training"),
        dashboard_user=raw.get("dashboard_user", "gpu"),
    )


class TTLCache:
    def __init__(self) -> None:
        self._values: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, ttl: int, loader: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            cached = self._values.get(key)
            if cached and now - cached[0] < ttl:
                return cached[1]
        value = loader()
        with self._lock:
            self._values[key] = (now, value)
        return value


_CACHE = TTLCache()


def _aws_error(source: str, exc: Exception) -> dict:
    response = getattr(exc, "response", {}) or {}
    error = response.get("Error", {}) if isinstance(response, dict) else {}
    return {
        "source": source,
        "code": error.get("Code") or type(exc).__name__,
        "message": error.get("Message") or str(exc),
    }


def _aws_context() -> tuple[Any | None, DashboardConfig, list[dict]]:
    config = load_config()
    try:
        import boto3
        from botocore.config import Config

        session = boto3.Session(profile_name=config.profile, region_name=config.regions[0])
        client_config = Config(connect_timeout=2, read_timeout=5, retries={"max_attempts": 1})
        return (session, config, [{"client_config": client_config}])
    except Exception as exc:
        return None, config, [_aws_error("aws.session", exc)]


def _client(session: Any, service: str, region: str | None, metadata: list[dict]):
    config = metadata[0].get("client_config") if metadata else None
    return session.client(service, region_name=region, config=config)


INSTANCE_RATES = {
    "g5.xlarge": 1.006,
    "g5.2xlarge": 1.212,
    "g6.xlarge": 0.8048,
    "p4d.24xlarge": 32.7726,
}
INSTANCE_VCPUS = {"g5.xlarge": 4, "g5.2xlarge": 8, "g6.xlarge": 4, "p4d.24xlarge": 96}


def aws_resources() -> dict:
    def load() -> dict:
        session, config, metadata = _aws_context()
        if session is None:
            return {"resources": [], "errors": metadata, "identity": None}
        resources, errors = [], []
        identity = None
        try:
            identity = _client(session, "sts", config.regions[0], metadata).get_caller_identity()
        except Exception as exc:
            errors.append(_aws_error("sts:GetCallerIdentity", exc))
        for region in config.regions:
            try:
                ec2 = _client(session, "ec2", region, metadata)
                paginator = ec2.get_paginator("describe_instances")
                instances = []
                for page in paginator.paginate(Filters=[{"Name": "tag:Project", "Values": [config.project_tag]}]):
                    for reservation in page.get("Reservations", []):
                        instances.extend(reservation.get("Instances", []))
                ids = [item["InstanceId"] for item in instances]
                statuses = {}
                if ids:
                    status_response = ec2.describe_instance_status(InstanceIds=ids, IncludeAllInstances=True)
                    statuses = {item["InstanceId"]: item for item in status_response.get("InstanceStatuses", [])}
                now = utc_now()
                for item in instances:
                    tags = {tag["Key"]: tag["Value"] for tag in item.get("Tags", [])}
                    launch = item.get("LaunchTime")
                    launch = launch.astimezone(UTC) if isinstance(launch, datetime) else None
                    state = item.get("State", {}).get("Name")
                    elapsed_hours = max(0, (now - launch).total_seconds() / 3600) if launch else None
                    hourly = INSTANCE_RATES.get(item.get("InstanceType"))
                    status = statuses.get(item["InstanceId"], {})
                    resources.append(
                        {
                            "instance_id": item["InstanceId"],
                            "name": tags.get("Name"),
                            "region": region,
                            "availability_zone": item.get("Placement", {}).get("AvailabilityZone"),
                            "state": state,
                            "instance_type": item.get("InstanceType"),
                            "ami": item.get("ImageId"),
                            "public_ip": item.get("PublicIpAddress"),
                            "private_ip": item.get("PrivateIpAddress"),
                            "launch_time": launch.isoformat().replace("+00:00", "Z") if launch else None,
                            "uptime_seconds": elapsed_hours * 3600 if elapsed_hours is not None else None,
                            "hourly_rate_estimate": hourly,
                            "estimated_run_cost": elapsed_hours * hourly if elapsed_hours is not None and hourly is not None and state == "running" else None,
                            "instance_status": status.get("InstanceStatus", {}).get("Status"),
                            "system_status": status.get("SystemStatus", {}).get("Status"),
                            "scheduled_events": status.get("Events", []),
                            "volumes": [mapping.get("Ebs", {}).get("VolumeId") for mapping in item.get("BlockDeviceMappings", []) if mapping.get("Ebs")],
                            "tags": tags,
                        }
                    )
            except Exception as exc:
                errors.append(_aws_error(f"ec2:{region}", exc))
        return {"resources": resources, "errors": errors, "identity": identity}

    return _CACHE.get("aws.resources", 30, load)


def aws_logs(limit: int = 100) -> dict:
    def load() -> dict:
        session, config, metadata = _aws_context()
        if session is None:
            return {"events": [], "errors": metadata}
        errors, events = [], []
        try:
            logs = _client(session, "logs", config.regions[0], metadata)
            streams = logs.describe_log_streams(
                logGroupName=config.log_group,
                orderBy="LastEventTime",
                descending=True,
                limit=20,
            ).get("logStreams", [])
            per_stream_limit = min(max(limit, 50), 1000)
            for stream in streams:
                stream_name = stream["logStreamName"]
                response = logs.get_log_events(
                    logGroupName=config.log_group,
                    logStreamName=stream_name,
                    limit=per_stream_limit,
                    startFromHead=False,
                )
                for item in response.get("events", []):
                    message = item.get("message", "").strip()
                    parsed = None
                    try:
                        parsed = json.loads(message)
                    except json.JSONDecodeError:
                        pass
                    events.append({"timestamp": datetime.fromtimestamp(item["timestamp"] / 1000, UTC).isoformat().replace("+00:00", "Z"), "stream": stream_name, "message": message, "event": parsed})
        except Exception as exc:
            errors.append(_aws_error("cloudwatch-logs", exc))
        return {"events": sorted(events, key=lambda row: row["timestamp"], reverse=True)[:limit], "errors": errors}

    return _CACHE.get(f"aws.logs.{limit}", 10, load)


def aws_jobs() -> dict:
    logs = aws_logs(500)
    jobs: dict[str, dict] = {}
    for row in reversed(logs["events"]):
        event = row.get("event")
        if not isinstance(event, dict) or not event.get("run_id"):
            continue
        run_id = event["run_id"]
        job = jobs.setdefault(run_id, {"run_id": run_id, "stage": event.get("stage"), "events": 0})
        job.update({"last_event": event, "last_updated": event.get("timestamp") or row["timestamp"], "phase": event.get("phase"), "current": event.get("current"), "total": event.get("total"), "metrics": event.get("metrics", {})})
        job["events"] += 1
    now = utc_now()
    for job in jobs.values():
        stamp = parse_timestamp(job.get("last_updated"))
        last_type = (job.get("last_event") or {}).get("type")
        if last_type == "job_failed":
            job["status"] = "failed"
        elif last_type == "job_completed":
            job["status"] = "passed"
        elif stamp and (now - stamp).total_seconds() <= THRESHOLDS["heartbeat_stale_seconds"]:
            job["status"] = "running"
        else:
            job["status"] = "stale"
    return {"jobs": sorted(jobs.values(), key=lambda job: job.get("last_updated") or "", reverse=True), "errors": logs["errors"]}


def cloud_run(job: dict) -> dict:
    """Normalize a CloudWatch job into the run shape used by dashboard pages."""
    last_event = job.get("last_event") or {}
    return {
        "run_id": job.get("run_id"),
        "stage": job.get("stage"),
        "kind": job.get("phase") or "job",
        "backend": "AWS",
        "host": last_event.get("host") or "CloudWatch",
        "path": "cloudwatch",
        "status": job.get("status"),
        "updated_at": job.get("last_updated"),
        "progress": {
            "phase": job.get("phase"),
            "current": job.get("current"),
            "total": job.get("total"),
            "metrics": job.get("metrics", {}),
        },
        "events": [last_event] if last_event else [],
        "metrics": job.get("metrics", {}),
        "manifest": {
            "run_id": job.get("run_id"),
            "stage": job.get("stage"),
            "git_sha": last_event.get("git_sha"),
            "parent_run_ids": last_event.get("parent_run_ids", []),
        },
    }


def aws_metrics(hours: int = 1) -> dict:
    def load() -> dict:
        session, config, metadata = _aws_context()
        inventory = aws_resources()
        if session is None:
            return {"series": [], "errors": metadata + inventory.get("errors", [])}
        errors, series = [], []
        end = utc_now()
        start = end - timedelta(hours=hours)
        for resource in inventory.get("resources", []):
            if resource["state"] != "running":
                continue
            region = resource["region"]
            instance_id = resource["instance_id"]
            try:
                cloudwatch = _client(session, "cloudwatch", region, metadata)
                queries = [
                    {"Id": "cpu", "MetricStat": {"Metric": {"Namespace": "AWS/EC2", "MetricName": "CPUUtilization", "Dimensions": [{"Name": "InstanceId", "Value": instance_id}]}, "Period": 60, "Stat": "Average"}, "ReturnData": True},
                    {"Id": "network_in", "MetricStat": {"Metric": {"Namespace": "AWS/EC2", "MetricName": "NetworkIn", "Dimensions": [{"Name": "InstanceId", "Value": instance_id}]}, "Period": 60, "Stat": "Sum"}, "ReturnData": True},
                    {"Id": "network_out", "MetricStat": {"Metric": {"Namespace": "AWS/EC2", "MetricName": "NetworkOut", "Dimensions": [{"Name": "InstanceId", "Value": instance_id}]}, "Period": 60, "Stat": "Sum"}, "ReturnData": True},
                ]
                custom_metrics = {
                    "gpu": "nvidia_smi_utilization_gpu",
                    "vram": "nvidia_smi_memory_used",
                    "temp": "nvidia_smi_temperature_gpu",
                    "power": "nvidia_smi_power_draw",
                    "memory": "mem_used_percent",
                    "disk": "disk_used_percent",
                }
                for query_id, metric_name in custom_metrics.items():
                    discovered = cloudwatch.list_metrics(
                        Namespace=config.metrics_namespace,
                        MetricName=metric_name,
                        Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                    ).get("Metrics", [])
                    if discovered:
                        queries.append(
                            {
                                "Id": query_id,
                                "MetricStat": {"Metric": discovered[0], "Period": 60, "Stat": "Average"},
                                "ReturnData": True,
                            }
                        )
                response = cloudwatch.get_metric_data(MetricDataQueries=queries, StartTime=start, EndTime=end, ScanBy="TimestampAscending")
                for result in response.get("MetricDataResults", []):
                    series.append({"instance_id": instance_id, "region": region, "metric": result.get("Id"), "label": result.get("Label"), "status": result.get("StatusCode"), "points": [{"timestamp": stamp.astimezone(UTC).isoformat().replace("+00:00", "Z"), "value": value} for stamp, value in zip(result.get("Timestamps", []), result.get("Values", []))]})
            except Exception as exc:
                errors.append(_aws_error(f"cloudwatch:{region}:{instance_id}", exc))
        return {"series": series, "errors": errors + inventory.get("errors", [])}

    return _CACHE.get(f"aws.metrics.{hours}", 30, load)


def aws_costs() -> dict:
    def load() -> dict:
        session, config, metadata = _aws_context()
        if session is None:
            return {"actual": None, "forecast": None, "live": None, "errors": metadata}
        errors, actual, forecast = [], None, None
        today = utc_now().date()
        month_start = today.replace(day=1)
        tomorrow = today + timedelta(days=1)
        try:
            ce = _client(session, "ce", "us-east-1", metadata)
            response = ce.get_cost_and_usage(TimePeriod={"Start": month_start.isoformat(), "End": tomorrow.isoformat()}, Granularity="MONTHLY", Metrics=["UnblendedCost"], GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}])
            results = response.get("ResultsByTime", [])
            groups = results[0].get("Groups", []) if results else []
            actual = {"total": sum(float(group["Metrics"]["UnblendedCost"]["Amount"]) for group in groups), "unit": "USD", "by_service": [{"service": group["Keys"][0], "amount": float(group["Metrics"]["UnblendedCost"]["Amount"])} for group in groups], "period": {"start": month_start.isoformat(), "end": tomorrow.isoformat()}}
            if tomorrow.month == today.month:
                month_end = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
                forecast_response = ce.get_cost_forecast(TimePeriod={"Start": tomorrow.isoformat(), "End": month_end.isoformat()}, Metric="UNBLENDED_COST", Granularity="MONTHLY")
                forecast = {"amount": float(forecast_response["Total"]["Amount"]), "unit": forecast_response["Total"]["Unit"]}
        except Exception as exc:
            errors.append(_aws_error("cost-explorer", exc))
        inventory = aws_resources()
        running = [resource for resource in inventory.get("resources", []) if resource["state"] == "running"]
        live = {"hourly": sum(resource.get("hourly_rate_estimate") or 0 for resource in running), "run_cost": sum(resource.get("estimated_run_cost") or 0 for resource in running), "instances": len(running), "label": "estimate"}
        return {"actual": actual, "forecast": forecast, "live": live, "errors": errors + inventory.get("errors", [])}

    return _CACHE.get("aws.costs", 900, load)


def aws_credits() -> dict:
    def load() -> dict:
        session, config, metadata = _aws_context()
        if session is None:
            return {"credits": None, "errors": metadata}
        try:
            billing = _client(session, "billing", "us-east-1", metadata)
            account_id = _client(session, "sts", config.regions[0], metadata).get_caller_identity()["Account"]
            end = utc_now()
            response = billing.get_credits(
                accountId=account_id,
                startDate=end - timedelta(days=3650),
                endDate=end,
                payerAccountFlag=True,
            )
            return {"credits": response.get("credits", []), "errors": []}
        except Exception as exc:
            return {"credits": None, "errors": [_aws_error("billing:GetCredits", exc)]}

    return _CACHE.get("aws.credits", 900, load)


def aws_quotas() -> dict:
    def load() -> dict:
        session, config, metadata = _aws_context()
        if session is None:
            return {"quotas": [], "errors": metadata}
        quotas, errors = [], []
        for region in config.regions:
            try:
                service = _client(session, "service-quotas", region, metadata)
                quota = service.get_service_quota(ServiceCode="ec2", QuotaCode=config.gpu_quota_code)["Quota"]
                history = service.list_requested_service_quota_change_history_by_quota(ServiceCode="ec2", QuotaCode=config.gpu_quota_code).get("RequestedQuotas", [])
                quotas.append({"region": region, "name": quota.get("QuotaName"), "value": quota.get("Value"), "unit": quota.get("Unit"), "adjustable": quota.get("Adjustable"), "pending_requests": [{"status": item.get("Status"), "desired_value": item.get("DesiredValue"), "created": item.get("Created").isoformat() if item.get("Created") else None} for item in history if item.get("Status") == "PENDING" ]})
            except Exception as exc:
                errors.append(_aws_error(f"service-quotas:{region}", exc))
        return {"quotas": quotas, "errors": errors}

    return _CACHE.get("aws.quotas", 900, load)


def aws_alarms() -> dict:
    def load() -> dict:
        session, config, metadata = _aws_context()
        if session is None:
            return {"alarms": [], "errors": metadata}
        alarms, errors = [], []
        for region in config.regions:
            try:
                cloudwatch = _client(session, "cloudwatch", region, metadata)
                response = cloudwatch.describe_alarms(AlarmNamePrefix="llm-tetris")
                for item in [*response.get("MetricAlarms", []), *response.get("CompositeAlarms", [])]:
                    updated = item.get("StateUpdatedTimestamp")
                    alarms.append(
                        {
                            "name": item.get("AlarmName"),
                            "description": item.get("AlarmDescription"),
                            "state": item.get("StateValue"),
                            "reason": item.get("StateReason"),
                            "updated_at": updated.astimezone(UTC).isoformat().replace("+00:00", "Z") if updated else None,
                            "region": region,
                        }
                    )
            except Exception as exc:
                errors.append(_aws_error(f"cloudwatch:DescribeAlarms:{region}", exc))
        alarms.sort(key=lambda item: (item.get("state") != "ALARM", item.get("name") or ""))
        return {"alarms": alarms, "errors": errors}

    return _CACHE.get("aws.alarms", 30, load)


def aws_security() -> dict:
    def load() -> dict:
        session, config, metadata = _aws_context()
        if session is None:
            return {"principal": None, "policies": [], "warnings": [], "errors": metadata}
        errors, policies, warnings = [], [], []
        try:
            iam = _client(session, "iam", "us-east-1", metadata)
            attached = iam.list_attached_user_policies(UserName=config.dashboard_user).get("AttachedPolicies", [])
            policies = [{"name": item["PolicyName"], "arn": item["PolicyArn"]} for item in attached]
            for policy in policies:
                if policy["name"] in {"AdministratorAccess", "AmazonEC2FullAccess"}:
                    warnings.append({"severity": "red", "title": f"Over-broad policy attached: {policy['name']}", "next_action": "Replace it with the scoped read-only dashboard policy."})
            inline = iam.list_user_policies(UserName=config.dashboard_user).get("PolicyNames", [])
            policies.extend({"name": name, "arn": None, "inline": True} for name in inline)
        except Exception as exc:
            errors.append(_aws_error("iam:ListUserPolicies", exc))
        return {"principal": config.dashboard_user, "policies": policies, "warnings": warnings, "errors": errors}

    return _CACHE.get("aws.security", 900, load)


def _aws_issues(resources_payload: dict, jobs_payload: dict) -> list[dict]:
    issues = []
    jobs = jobs_payload.get("jobs", [])
    active_jobs = [job for job in jobs if job["status"] == "running"]

    def job_matches_run(job: dict, run_id: str) -> bool:
        job_run_id = job.get("run_id")
        parent_run_ids = (job.get("last_event") or {}).get("parent_run_ids") or []
        return job_run_id == run_id or run_id in parent_run_ids or bool(job_run_id and job_run_id.startswith(f"{run_id}-"))

    for resource in resources_payload.get("resources", []):
        if resource["state"] != "running":
            continue
        run_id = resource.get("tags", {}).get("RunId")
        if not run_id or not any(job_matches_run(job, run_id) for job in active_jobs):
            launch = parse_timestamp(resource.get("launch_time"))
            old_enough = launch and (utc_now() - launch).total_seconds() > THRESHOLDS["orphan_instance_seconds"]
            if old_enough:
                issues.append(issue(f"aws.instance.{resource['instance_id']}.orphan", "red", int(resource.get("tags", {}).get("Stage", 4)), "Running EC2 instance has no live job heartbeat", observed=resource["instance_id"], expected="structured event within 15 minutes", evidence=f"aws:ec2:{resource['region']}:{resource['instance_id']}", next_action="Inspect the job and terminate the instance manually if it is no longer useful.", source="aws", scope=f"resource:{resource['instance_id']}"))
        if resource.get("instance_status") not in {None, "ok"} or resource.get("system_status") not in {None, "ok"}:
            issues.append(issue(f"aws.instance.{resource['instance_id']}.health", "red", int(resource.get("tags", {}).get("Stage", 4)), "EC2 status check is unhealthy", observed={"instance": resource.get("instance_status"), "system": resource.get("system_status")}, expected="ok", evidence=f"aws:ec2:{resource['region']}:{resource['instance_id']}", next_action="Open the AWS status checks and recover or replace the instance.", source="aws"))
    for job in jobs:
        if job["status"] == "stale":
            issues.append(issue(f"aws.job.{job['run_id']}.stale", "red", job.get("stage"), "Active job heartbeat is stale", observed=job.get("last_updated"), expected="within 5 minutes", evidence=f"cloudwatch:{job['run_id']}", next_action="Inspect CloudWatch logs and the backing instance.", source="aws", scope=f"run:{job['run_id']}"))
    return issues


def _aws_alarm_issues(alarms_payload: dict) -> list[dict]:
    issues = []
    for alarm in alarms_payload.get("alarms", []):
        if alarm.get("state") != "ALARM":
            continue
        name = alarm.get("name") or "unknown"
        issues.append(
            issue(
                f"aws.alarm.{hashlib.sha1(name.encode()).hexdigest()[:8]}",
                "red",
                None,
                f"CloudWatch alarm is active: {name}",
                observed=alarm.get("reason"),
                expected="OK",
                evidence=f"aws:cloudwatch:{alarm.get('region')}:{name}",
                next_action="Inspect the tagged instance and current job before allowing additional spend.",
                source="aws",
                scope="aws:alarms",
            )
        )
    return issues


def _aws_extended_issues(
    credits_payload: dict,
    quotas_payload: dict,
    security_payload: dict,
    metrics_payload: dict,
    resources_payload: dict,
    jobs_payload: dict,
) -> list[dict]:
    issues: list[dict] = []
    panels = {
        "resources": resources_payload,
        "jobs": jobs_payload,
        "credits": credits_payload,
        "quotas": quotas_payload,
        "security": security_payload,
        "metrics": metrics_payload,
    }
    for panel, payload in panels.items():
        for err in payload.get("errors", []):
            code = err.get("code", "Unavailable")
            digest = hashlib.sha1(f"{panel}:{code}:{err.get('source')}".encode()).hexdigest()[:8]
            issues.append(
                issue(
                    f"aws.telemetry.{panel}.{digest}",
                    "amber",
                    None,
                    f"AWS {panel} telemetry is unavailable",
                    observed=code,
                    expected="read access and configured source",
                    evidence=f"aws:{err.get('source', panel)}",
                    next_action=f"Grant the scoped read action for {err.get('source', panel)} or configure the missing AWS source.",
                    source="aws",
                    scope=f"aws-panel:{panel}",
                )
            )
    for warning in security_payload.get("warnings", []):
        title = warning.get("title", "AWS security warning")
        digest = hashlib.sha1(title.encode()).hexdigest()[:8]
        issues.append(issue(f"aws.security.{digest}", warning.get("severity", "amber"), None, title, evidence="aws:iam", next_action=warning.get("next_action", "Review the configured dashboard principal."), source="aws", scope="aws:iam"))

    for credit in credits_payload.get("credits") or []:
        remaining = credit.get("remainingAmount") or {}
        amount = remaining.get("currencyAmount") if isinstance(remaining, dict) else remaining
        try:
            amount = float(amount) if amount is not None else None
        except (TypeError, ValueError):
            amount = None
        expiry = parse_timestamp(credit.get("endDate"))
        if expiry and (expiry - utc_now()).days <= 30:
            issues.append(issue(f"aws.credit.{credit.get('creditId', 'unknown')}.expiry", "red", None, "AWS credit expires within 30 days", observed=expiry.isoformat().replace("+00:00", "Z"), expected="> 30 days", evidence="aws:billing:GetCredits", next_action="Finish planned GPU work before expiry or update the cost plan.", source="aws", scope="aws:billing"))
        if isinstance(amount, (int, float)) and amount <= 0:
            issues.append(issue(f"aws.credit.{credit.get('creditId', 'unknown')}.empty", "red", None, "AWS credit is exhausted", observed=amount, expected="> projected run cost", evidence="aws:billing:GetCredits", next_action="Recalculate the training budget before launching another GPU run.", source="aws", scope="aws:billing"))

    running = [resource for resource in resources_payload.get("resources", []) if resource["state"] == "running"]
    used_vcpus = sum(INSTANCE_VCPUS.get(resource.get("instance_type"), 0) for resource in running)
    for quota in quotas_payload.get("quotas", []):
        headroom = (quota.get("value") or 0) - used_vcpus
        if headroom < 4:
            issues.append(issue(f"aws.quota.{quota.get('region')}.headroom", "amber", None, "GPU vCPU quota has insufficient headroom for the next g5.xlarge", observed=headroom, expected=">= 4 vCPUs", evidence=f"aws:service-quotas:{quota.get('region')}", next_action="Stop an unused GPU instance or request quota before the next run.", source="aws", scope="aws:quota"))

    series = metrics_payload.get("series", [])
    training_active = any(job.get("status") == "running" and job.get("phase") == "training" for job in jobs_payload.get("jobs", []))
    for metric in series:
        points = metric.get("points", [])
        if not points:
            continue
        latest = points[-1]["value"]
        if metric.get("metric") == "disk" and latest >= 90:
            issues.append(issue(f"aws.instance.{metric['instance_id']}.disk", "red", None, "Instance disk usage is above 90%", observed=latest, expected="< 90%", evidence=f"aws:cloudwatch:{metric['instance_id']}:disk", next_action="Free disk space or expand the volume before the job writes another checkpoint.", source="aws", scope=f"resource:{metric['instance_id']}"))
        elif metric.get("metric") == "disk" and latest >= 80:
            issues.append(issue(f"aws.instance.{metric['instance_id']}.disk", "amber", None, "Instance disk usage is above 80%", observed=latest, expected="< 80%", evidence=f"aws:cloudwatch:{metric['instance_id']}:disk", next_action="Inspect checkpoint and cache growth.", source="aws", scope=f"resource:{metric['instance_id']}"))
        if metric.get("metric") == "gpu" and training_active and len(points) >= 10:
            recent_average = sum(point["value"] for point in points[-10:]) / 10
            if recent_average < THRESHOLDS["gpu_low_percent"]:
                issues.append(issue(f"aws.instance.{metric['instance_id']}.gpu_low", "amber", 4, "GPU utilization stayed below 10% during training", observed=recent_average, expected=">= 10%", evidence=f"aws:cloudwatch:{metric['instance_id']}:gpu", next_action="Inspect input pipeline, batching, and process health before paying for more idle time.", source="aws", scope=f"resource:{metric['instance_id']}"))
    return issues


def envelope(data: Any, *, sources: list[dict] | None = None, errors: list[dict] | None = None) -> dict:
    errors = errors or []
    return {"generated_at": iso_now(), "partial": bool(errors), "freshness": sources or [], "errors": errors, "data": data}


def dashboard_summary(include_aws: bool = True) -> dict:
    snapshot = local_snapshot()
    aws_summary = {"resources": [], "jobs": [], "live_cost": None}
    aws_errors: list[dict] = []
    if include_aws:
        resources = aws_resources()
        jobs = aws_jobs()
        costs = aws_costs()
        credits = aws_credits()
        quotas = aws_quotas()
        security = aws_security()
        alarms = aws_alarms()
        metrics = aws_metrics()
        aws_errors = resources.get("errors", []) + jobs.get("errors", []) + costs.get("errors", []) + credits.get("errors", []) + quotas.get("errors", []) + security.get("errors", []) + alarms.get("errors", []) + metrics.get("errors", [])
        snapshot["issues"].extend(_aws_issues(resources, jobs))
        snapshot["issues"].extend(_aws_extended_issues(credits, quotas, security, metrics, resources, jobs))
        snapshot["issues"].extend(_aws_alarm_issues(alarms))
        aws_summary = {"resources": resources.get("resources", []), "jobs": jobs.get("jobs", []), "alarms": alarms.get("alarms", []), "live_cost": costs.get("live")}
        if snapshot["project"].get("active_job") is None:
            aws_active_job = next((job for job in jobs.get("jobs", []) if job.get("status") == "running"), None)
            if aws_active_job:
                active_run = cloud_run(aws_active_job)
                snapshot["project"]["active_job"] = active_run
                active_stage = next((stage for stage in snapshot["stages"] if stage["number"] == active_run.get("stage")), None)
                if active_stage and active_stage.get("status") != "passed":
                    active_stage["status"] = "running"
                    active_stage["progress"] = {
                        **active_run["progress"],
                        "value": (
                            active_run["progress"]["current"] / active_run["progress"]["total"]
                            if active_run["progress"].get("current") is not None and active_run["progress"].get("total")
                            else None
                        ),
                        "label": active_run["progress"].get("phase") or "AWS job running",
                    }
                    active_stage["next_action"] = "Monitor the live AWS job and verify its final artifact."
    severity_rank = {"red": 3, "amber": 2, "info": 1}
    snapshot["issues"].sort(key=lambda item: (-severity_rank[item["severity"]], item.get("detected_at") or ""))
    snapshot["top_issues"] = snapshot["issues"][:3]
    snapshot["aws"] = aws_summary
    snapshot["project"]["overall_status"] = _overall_status(snapshot["stages"], snapshot["issues"], snapshot["project"].get("active_job"))
    sources = [
        {"name": "repository", "updated_at": file_timestamp(ROOT / ".git/HEAD"), "state": "fresh"},
        {"name": "datasets", "updated_at": snapshot["datasets"].get("updated_at"), "state": "fresh" if snapshot["datasets"].get("updated_at") else "missing"},
        {"name": "runs", "updated_at": snapshot["runs"][0]["updated_at"] if snapshot["runs"] else None, "state": "fresh" if snapshot["runs"] else "missing"},
        {"name": "AWS", "updated_at": iso_now() if include_aws and not aws_errors else None, "state": "partial" if aws_errors else ("fresh" if include_aws else "paused")},
    ]
    return envelope(snapshot, sources=sources, errors=aws_errors)


def replay_index() -> dict[str, dict]:
    index: dict[str, dict] = {}
    if not RUNS_DIR.exists():
        return index
    for path in RUNS_DIR.rglob("closed_loop/games.jsonl"):
        try:
            with path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    digest = hashlib.sha1(f"{relative(path)}:{record.get('game_id')}".encode()).hexdigest()[:16]
                    index[digest] = {"replay_id": digest, "source": relative(path), "record": record}
        except (OSError, json.JSONDecodeError):
            continue
    return index


def replay_snapshot(replay_id: str, turn: int | None = None) -> dict | None:
    entry = replay_index().get(replay_id)
    if not entry:
        return None
    from .engine import Game

    record = entry["record"]
    game = Game(seed=record["seed"], game_id=record.get("game_id"))
    actions = record.get("actions", [])
    target = len(actions) if turn is None else min(max(0, turn), len(actions))
    for rot, x in actions[:target]:
        if game.game_over:
            break
        game.step(rot, x)
    return {
        "replay_id": replay_id,
        "source": entry["source"],
        "record": {key: record.get(key) for key in ("game_id", "seed", "policy", "mode", "pieces", "lines", "score", "died", "death_reason")},
        "turn": target,
        "total_turns": len(actions),
        "snapshot": game.snapshot(),
        "next_action": actions[target] if target < len(actions) else None,
    }
