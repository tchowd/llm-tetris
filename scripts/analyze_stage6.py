#!/usr/bin/env python3
"""Create the registered paired Stage 6 comparison and research report."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from tetris.rl import bootstrap_mean_ci, file_sha256, paired_comparison


def parse_named(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, raw = value.split("=", 1)
    return label, Path(raw)


def game_metric(path: Path, label: str) -> tuple[dict[int, float], list[dict]]:
    rows = []
    with (path / "games.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("policy") == label:
                rows.append(row)
    if not rows:
        raise ValueError(f"no games for policy {label!r} in {path}")
    values = {int(row["seed"]): 100.0 * row["score"] / row["pieces"] if row["pieces"] else 0.0 for row in rows}
    return values, rows


def load_policy_metrics(path: Path, label: str) -> dict:
    report = json.loads((path / "metrics.json").read_text())
    if label not in report:
        raise ValueError(f"no metrics for policy {label!r} in {path}")
    return report[label]


def validate_evaluation(path: Path, *, suite: str, benchmark: dict, benchmark_hash: str) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("suite") != suite:
        raise ValueError(f"{path}: wrong evaluation suite")
    if manifest.get("benchmark_manifest_sha256") != benchmark_hash:
        raise ValueError(f"{path}: benchmark manifest hash differs from the registered benchmark")
    expected_cap = benchmark["long_horizon"][f"{suite}_cap"]
    if manifest.get("cap") != expected_cap or manifest.get("registered_cap") != expected_cap:
        raise ValueError(f"{path}: smoke/changed cap cannot be used for promotion")
    if manifest.get("recovery_cap") != benchmark.get("recovery_cap", 200):
        raise ValueError(f"{path}: changed recovery cap cannot be used for promotion")
    if manifest.get("seeds") != benchmark[f"{suite}_seeds"]:
        raise ValueError(f"{path}: seed list differs from the registered suite")
    if manifest.get("mode") != "strict" or manifest.get("greedy") is not True:
        raise ValueError(f"{path}: promotion requires strict greedy evaluation")
    return manifest


def stress_guardrails(metrics: dict) -> dict:
    row = metrics["long_horizon"]
    checks = {
        "parse_failures_zero": row["parse_failure_rate"]["mean"] == 0,
        "illegal_actions_zero": row["illegal_rate"]["mean"] == 0,
        "illegal_deaths_zero": row["illegal_action_deaths"] == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def stage5_gate(path: Path, policy: str, rules: dict) -> dict:
    report = json.loads(path.read_text())
    row = report[policy]["strict"]
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if manifest.get("cap") != 500 or manifest.get("seeds") != list(range(10_000_000, 10_000_100)):
        raise ValueError(f"{path}: Stage 5 gate must use the frozen 100 seeds and 500-piece cap")
    checks = {
        "mean_lines": row["lines"]["mean"] >= rules["stage5_min_mean_lines"],
        "deaths": row["deaths"] <= rules["stage5_max_deaths"],
        "parse_failures": row["parse_failure_rate"]["mean"] <= rules["stage5_max_parse_failures"],
        "illegal_actions": row["illegal_rate"]["mean"] <= rules["stage5_max_illegal_actions"],
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, default=Path("benchmarks/stress-v1/manifest.json"))
    parser.add_argument("--suite", choices=("development", "test"), required=True)
    parser.add_argument("--baseline", type=parse_named, required=True, help="LABEL=stress-eval-directory")
    parser.add_argument("--candidate", type=parse_named, action="append", required=True, help="LABEL=stress-eval-directory")
    parser.add_argument("--stage5-candidate", type=parse_named, action="append", default=[], help="POLICY=metrics.json")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=6006)
    parser.add_argument("--cleanup-report", type=Path)
    parser.add_argument("--sync-receipt", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark_manifest.read_text())
    rules = benchmark["decision_rules"]
    baseline_label, baseline_path = args.baseline
    benchmark_hash = file_sha256(args.benchmark_manifest)
    validate_evaluation(baseline_path, suite=args.suite, benchmark=benchmark, benchmark_hash=benchmark_hash)
    baseline_values, _ = game_metric(baseline_path, baseline_label)
    comparisons = {}
    per_candidate_diffs = []
    improved = 0
    training_seeds = []
    for index, (label, path) in enumerate(args.candidate):
        candidate_manifest = validate_evaluation(path, suite=args.suite, benchmark=benchmark, benchmark_hash=benchmark_hash)
        training_seeds.append(candidate_manifest.get("training_seed"))
        candidate_values, _ = game_metric(path, label)
        comparison = paired_comparison(
            baseline_values,
            candidate_values,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + index,
        )
        metrics = load_policy_metrics(path, label)
        guardrails = stress_guardrails(metrics)
        promotion = comparison["relative_improvement"] is not None and comparison["relative_improvement"] >= rules["development_relative_improvement"] and guardrails["passed"]
        comparison.update({"guardrails": guardrails, "development_promotion_passed": promotion})
        comparisons[label] = comparison
        per_candidate_diffs.append(comparison["differences"])
        improved += comparison["mean_difference"] > 0 and guardrails["passed"]

    combined_by_seed = [statistics.mean(values) for values in zip(*per_candidate_diffs)] if per_candidate_diffs else []
    combined_ci = bootstrap_mean_ci(combined_by_seed, samples=args.bootstrap_samples, seed=args.bootstrap_seed + 100)
    stage5 = {label: stage5_gate(path, label, rules) for label, path in args.stage5_candidate}
    cleanup = json.loads(args.cleanup_report.read_text()) if args.cleanup_report else None
    sync = json.loads(args.sync_receipt.read_text()) if args.sync_receipt else None
    confirmation = {
        "candidate_count": len(args.candidate),
        "training_seeds": training_seeds,
        "improved_candidate_count": improved,
        "minimum_improved": rules["confirmation_min_improved_training_seeds"],
        "combined_mean_difference": statistics.mean(combined_by_seed) if combined_by_seed else 0.0,
        "combined_bootstrap_95_ci": list(combined_ci),
        "passed": (
            args.suite == "test"
            and len(args.candidate) >= rules["confirmation_training_seed_count"]
            and None not in training_seeds
            and len(set(training_seeds)) == len(training_seeds)
            and improved >= rules["confirmation_min_improved_training_seeds"]
            and combined_ci[0] > rules["confirmation_combined_ci_must_exceed"]
            and all(row["passed"] for row in stage5.values())
            and set(stage5) == {label for label, _ in args.candidate}
        ),
    }
    report = {
        "benchmark_id": benchmark["benchmark_id"],
        "suite": args.suite,
        "primary_metric": benchmark["primary_metric"],
        "baseline": baseline_label,
        "comparisons": comparisons,
        "stage5_non_inferiority": stage5,
        "replicated_confirmation": confirmation,
        "research_complete": (
            args.suite == "test"
            and len(args.candidate) >= rules["confirmation_training_seed_count"]
            and None not in training_seeds
            and len(set(training_seeds)) == len(training_seeds)
            and set(stage5) == {label for label, _ in args.candidate}
        ),
        "operations_complete": bool(
            cleanup and cleanup.get("status") == "passed"
            and sync and sync.get("status") == "passed"
            and sync.get("direction") == "upload" and sync.get("included_adapter")
        ),
        "cleanup_report": cleanup,
        "artifact_sync_receipt": sync,
        "decision": "retain_rl" if confirmation["passed"] else "retain_sft",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Stage 6 RL research report",
        "",
        f"Suite: `{args.suite}`. Primary metric: `{benchmark['primary_metric']}`. Frozen baseline: `{baseline_label}`.",
        "",
        "| Candidate | Mean paired delta | Median delta | Bootstrap 95% CI | Relative | Guardrails |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, row in comparisons.items():
        low, high = row["bootstrap_95_ci"]
        relative = row["relative_improvement"]
        relative_text = f"{relative:.2%}" if relative is not None else "n/a"
        lines.append(
            f"| {label} | {row['mean_difference']:.4f} | {row['median_difference']:.4f} | [{low:.4f}, {high:.4f}] | {relative_text} | {'pass' if row['guardrails']['passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            f"Replicated confirmation: **{'passed' if confirmation['passed'] else 'not passed'}**. "
            f"{improved}/{len(args.candidate)} candidates improved; combined CI "
            f"[{combined_ci[0]:.4f}, {combined_ci[1]:.4f}].",
            "",
            f"Decision: **{report['decision']}**. A non-winning or null RL result is retained here as research evidence; it does not replace the frozen SFT adapter.",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_dir / 'report.json'} and {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
