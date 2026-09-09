#!/usr/bin/env python3
"""Analyze conditional revised-RL replications against the improved SFT."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from scripts.analyze_tetra17_sft import games_for, summary
from tetris.rl import file_sha256


def interval(matrix: list[list[float]], seed: int = 1703, replicates: int = 20000) -> dict:
    rng = random.Random(seed)
    flat_seed_means = [statistics.mean(row) for row in matrix]
    draws = []
    for _ in range(replicates):
        sampled_seeds = [matrix[rng.randrange(len(matrix))] for _ in matrix]
        draws.append(statistics.mean(row[rng.randrange(len(row))] for row in sampled_seeds for _ in row))
    draws.sort()
    return {"mean": statistics.mean(flat_seed_means), "ci95": [draws[int(.025 * replicates)], draws[int(.975 * replicates)]], "per_training_seed": flat_seed_means}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    r = json.loads(args.registration.read_text())
    rh = file_sha256(args.registration)
    gate = json.loads(Path("experiments/tetra17-recovery-v2/sft-gate.json").read_text())
    if gate["status"] != "passed" or not gate["conditional_rl_authorized_by_gate"]:
        raise ValueError("conditional RL was not enabled by the SFT gate")
    base = {kind: games_for(Path("experiments/tetra17-recovery-v2/evaluation/candidate"), rh, kind) for kind in ("recovery", "ordinary")}
    runs = {}
    matrix = []
    for seed in r["conditional_rl"]["training_seeds"]:
        label = f"rl-fixed-zero-{seed}"
        root = Path("experiments/tetra17-recovery-v2/evaluation") / label
        cohorts = {kind: games_for(root, rh, kind) for kind in ("recovery", "ordinary")}
        runs[label] = {kind: summary(rows) for kind, rows in cohorts.items()}
        matrix.append([int(b["death_reason"] == "cap_reached") - int(a["death_reason"] == "cap_reached") for a, b in zip(base["recovery"], cohorts["recovery"], strict=True)])
    effect = interval(matrix)
    ordinary_ok = all(v["ordinary"]["cap_reached"] == 20 and v["ordinary"]["illegal"] == 0 and v["ordinary"]["topout"] == 0 for v in runs.values())
    helps = effect["mean"] > 0 and effect["ci95"][0] >= 0 and ordinary_ok
    hurts = effect["ci95"][1] < 0 or not ordinary_ok
    report = {
        "outcome": "helps" if helps else "hurts" if hurts else "inconclusive",
        "paired_recovery_cap_effect": effect, "ordinary_gate": ordinary_ok,
        "candidate_sft": {kind: summary(rows) for kind, rows in base.items()}, "rl_runs": runs,
        "recommendation": "retain conditional RL candidate for confirmation" if helps else "retain the improved SFT endpoint without RL",
        "registration_sha256": rh, "confirmation_access": False, "final_test_access": False,
    }
    out = Path("experiments/tetra17-recovery-v2/rl-gate.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
