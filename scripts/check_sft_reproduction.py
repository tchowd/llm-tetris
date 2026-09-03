#!/usr/bin/env python3
"""Require exact deterministic Stage 5 SFT action/outcome reproduction."""
import argparse
import json
from pathlib import Path

from tetris.rl import atomic_write_json, file_sha256


def cohort(path: Path, policy: str) -> dict:
    return {
        row["seed"]: {key: row[key] for key in ("actions", "pieces", "lines", "score", "died", "death_reason")}
        for line in (path / "games.jsonl").read_text().splitlines()
        if (row := json.loads(line)).get("policy") == policy and row["mode"] == "strict"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("runs/sft-v1/closed_loop"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--policy", default="sft")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate = cohort(args.baseline, "model"), cohort(args.candidate, args.policy)
    seeds = set(range(10_000_000, 10_000_100))
    changed = [seed for seed in sorted(seeds) if baseline.get(seed) != candidate.get(seed)]
    manifests = [json.loads((path / "manifest.json").read_text()) for path in (args.baseline, args.candidate)]
    checks = {
        "exact_seed_cohorts": set(baseline) == set(candidate) == seeds,
        "frozen_cap_and_seeds": all(row["cap"] == 500 and row["seeds"] == sorted(seeds) for row in manifests),
        "exact_actions_and_outcomes": not changed,
    }
    report = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "changed_seeds": changed,
              "baseline_games_sha256": file_sha256(args.baseline / "games.jsonl"),
              "candidate_games_sha256": file_sha256(args.candidate / "games.jsonl")}
    atomic_write_json(args.out, report)
    print(json.dumps(report), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
