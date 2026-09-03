#!/usr/bin/env python3
"""Write durable current-commit verification evidence for dashboard stages.

Examples:
    python scripts/verify_stages.py --stage 1
    python scripts/verify_stages.py --stage 2
    python scripts/verify_stages.py --stage 1 --stage 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from tetris.dashboard import stage_source_fingerprint


ROOT = Path(__file__).resolve().parent.parent
STATUS_DIR = ROOT / "runs/status"

STAGE_TESTS = {
    1: [
        "tests/test_bag.py",
        "tests/test_features.py",
        "tests/test_game_over.py",
        "tests/test_line_clear.py",
        "tests/test_placement.py",
        "tests/test_replay.py",
        "tests/test_serialize.py",
    ],
    2: [
        "tests/test_teacher_features.py",
        "tests/test_teacher_pick.py",
        "tests/test_teacher_benchmark.py",
    ],
    6: [
        "tests/test_rl.py",
        "tests/test_stress_eval.py",
        "tests/test_episode_rl.py",
        "tests/test_episode_proof.py",
        "tests/test_episode_runtime.py",
        "tests/test_grpo_integration.py",
        "tests/test_stage6_analysis.py",
        "tests/test_recovery.py",
        "tests/test_recovery_artifacts.py",
    ],
}


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def run_command(command: list[str], cwd: Path = ROOT) -> dict:
    started = time.time()
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return {
        "command": command,
        "ok": result.returncode == 0,
        "exit_code": result.returncode,
        "duration_seconds": time.time() - started,
        "output_tail": "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-30:]),
    }


def junit_counts(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if root.tag != "testsuite" and suites and suites[0] is root:
        suites = suites[1:]
    return {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "duration_seconds": sum(float(suite.attrib.get("time", 0)) for suite in suites),
    }


def verify_test_stage(stage: int) -> dict:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    junit_path = STATUS_DIR / f"stage-{stage}-junit.xml"
    pytest_result = run_command([sys.executable, "-m", "pytest", *STAGE_TESTS[stage], "-q", f"--junitxml={junit_path}"])
    commands = [pytest_result]
    checks = []
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "duration_seconds": 0}
    if junit_path.exists():
        counts = junit_counts(junit_path)
    checks.append({"name": "pytest", "ok": pytest_result["ok"], "detail": counts, "evidence": str(junit_path.relative_to(ROOT))})
    if stage == 1:
        build_result = run_command(["npm", "run", "build"], cwd=ROOT / "web")
        commands.append(build_result)
        checks.append({"name": "web_build", "ok": build_result["ok"], "detail": {"duration_seconds": build_result["duration_seconds"]}, "evidence": "web/dist"})
    ok = all(check["ok"] for check in checks)
    return {
        "run_id": f"verify-stage-{stage}",
        "stage": stage,
        "status": "passed" if ok else "failed",
        "ok": ok,
        "git_sha": git_sha(),
        "source_fingerprint": stage_source_fingerprint(stage),
        "checks_passed": sum(check["ok"] for check in checks),
        "checks_total": len(checks),
        "test_counts": counts,
        "checks": checks,
        "commands": commands,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def verify_release() -> dict:
    checks = [
        {"name": "adapter_retained", "ok": any((ROOT / "runs").glob("*/adapter")), "evidence": "runs/*/adapter"},
        {"name": "open_loop_metrics", "ok": any((ROOT / "runs").glob("*/open_loop_metrics.json")), "evidence": "runs/*/open_loop_metrics.json"},
        {"name": "closed_loop_metrics", "ok": any((ROOT / "runs").glob("*/closed_loop/metrics.json")), "evidence": "runs/*/closed_loop/metrics.json"},
        {"name": "stage1_verified", "ok": (STATUS_DIR / "stage-1.json").exists(), "evidence": "runs/status/stage-1.json"},
        {"name": "stage2_verified", "ok": (STATUS_DIR / "stage-2.json").exists(), "evidence": "runs/status/stage-2.json"},
        {"name": "dashboard_build", "ok": (ROOT / "web/dist/index.html").exists(), "evidence": "web/dist/index.html"},
        {"name": "dashboard_policy", "ok": (ROOT / "infra/dashboard-readonly-policy.json").exists(), "evidence": "infra/dashboard-readonly-policy.json"},
    ]
    ok = all(check["ok"] for check in checks)
    return {
        "run_id": "verify-stage-7",
        "stage": 7,
        "status": "passed" if ok else "failed",
        "ok": ok,
        "git_sha": git_sha(),
        "source_fingerprint": stage_source_fingerprint(7),
        "checks_passed": sum(check["ok"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def verify_rl_stage() -> dict:
    """Verify the local Stage 6 substrate without pretending GPU research ran."""

    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    junit_path = STATUS_DIR / "stage-6-junit.xml"
    pytest_result = run_command([sys.executable, "-m", "pytest", *STAGE_TESTS[6], "-q", f"--junitxml={junit_path}"])
    compile_result = run_command(
        [
            sys.executable,
            "-m",
            "py_compile",
            "scripts/generate_stress_manifest.py",
            "scripts/eval_stress.py",
            "scripts/train_rl.py",
            "scripts/train_episode_rl.py",
            "scripts/analyze_stage6.py",
            "scripts/check_e2_learning.py",
            "scripts/select_e3_kl.py",
            "scripts/check_e4_pilot.py",
            "scripts/train_recovery_sft.py",
            "scripts/generate_recovery_data.py",
            "scripts/audit_recovery_failures.py",
            "scripts/eval_recovery_only.py",
            "scripts/check_recovery_pilot.py",
            "scripts/check_episode_proof.py",
            "scripts/check_episode_runtime.py",
            "scripts/check_episode_pilot.py",
            "scripts/audit_recovery_artifacts.py",
            "scripts/report_recovery_outcome.py",
        ]
    )
    benchmark = ROOT / "benchmarks/stress-v1/manifest.json"
    states = ROOT / "benchmarks/stress-v1/states.jsonl"
    checks = [
        {"name": "rl_unit_tests", "ok": pytest_result["ok"], "evidence": str(junit_path.relative_to(ROOT))},
        {"name": "rl_entrypoints_compile", "ok": compile_result["ok"], "evidence": "scripts/train_rl.py"},
        {"name": "stress_v1_registered", "ok": benchmark.exists() and states.exists(), "evidence": "benchmarks/stress-v1"},
    ]
    final_reports = list((ROOT / "runs").glob("*/rl/report.json"))
    confirmation = False
    if final_reports:
        latest = max(final_reports, key=lambda path: path.stat().st_mtime)
        try:
            final_report = json.loads(latest.read_text())
            confirmation = bool(final_report.get("research_complete") and final_report.get("operations_complete"))
        except (OSError, json.JSONDecodeError):
            confirmation = False
        checks.append({"name": "research_report_complete", "ok": confirmation, "evidence": str(latest.relative_to(ROOT))})
    local_ok = all(check["ok"] for check in checks[:3])
    counts = junit_counts(junit_path) if junit_path.exists() else {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "duration_seconds": 0}
    return {
        "run_id": "verify-stage-6",
        "stage": 6,
        "status": "passed" if local_ok and confirmation else ("ready" if local_ok else "failed"),
        "ok": local_ok,
        "research_complete": confirmation,
        "git_sha": git_sha(),
        "source_fingerprint": stage_source_fingerprint(6),
        "checks_passed": sum(check["ok"] for check in checks),
        "checks_total": len(checks),
        "test_counts": counts,
        "checks": checks,
        "commands": [pytest_result, compile_result],
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, action="append", choices=[1, 2, 6, 7], required=True)
    args = parser.parse_args()
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    for stage in dict.fromkeys(args.stage):
        report = verify_release() if stage == 7 else (verify_rl_stage() if stage == 6 else verify_test_stage(stage))
        path = STATUS_DIR / f"stage-{stage}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"stage {stage}: {report['status']} ({report['checks_passed']}/{report['checks_total']}) -> {path}")
        failed = failed or not report["ok"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
