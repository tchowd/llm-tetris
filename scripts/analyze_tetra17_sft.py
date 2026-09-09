#!/usr/bin/env python3
"""Apply the preregistered SFT gate without accessing confirmation or final test."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from tetris.rl import file_sha256


def read(path: Path):
    return json.loads(path.read_text())


def games_for(root: Path, registration_hash: str, kind: str) -> list[dict]:
    complete = read(root / "complete.json")
    if complete["registration_sha256"] != registration_hash or complete["status"] != "completed":
        raise ValueError("evaluation identity mismatch")
    games = []
    for name, digest in sorted(complete["files_sha256"].items()):
        path = Path(name)
        if file_sha256(path) != digest:
            raise ValueError("evaluation shard changed")
        shard = read(path)
        if shard["kind"] == kind:
            games.extend(shard["games"])
    return games


def summary(games: list[dict]) -> dict:
    def mean(name):
        return statistics.mean(game[name] for game in games)
    return {
        "games": len(games), "cap_reached": sum(g["death_reason"] == "cap_reached" for g in games),
        "illegal": sum(g["death_reason"] == "illegal_action" for g in games),
        "topout": sum(g["death_reason"] == "topped_out" for g in games),
        "mean_pieces": mean("pieces"), "mean_lines": mean("lines"), "mean_score": mean("score"),
    }


def paired_interval(control: list[dict], candidate: list[dict], seed: int = 1702, replicates: int = 20000) -> dict:
    effects = [int(b["death_reason"] == "cap_reached") - int(a["death_reason"] == "cap_reached") for a, b in zip(control, candidate, strict=True)]
    rng = random.Random(seed)
    samples = [statistics.mean(effects[rng.randrange(len(effects))] for _ in effects) for _ in range(replicates)]
    samples.sort()
    return {"mean": statistics.mean(effects), "ci95": [samples[int(.025 * replicates)], samples[int(.975 * replicates)]], "paired_effects": effects}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    r = read(args.registration)
    rh = file_sha256(args.registration)
    roots = {label: Path("experiments/tetra17-recovery-v2/evaluation") / label for label in ("control", "candidate")}
    values = {label: {kind: games_for(root, rh, kind) for kind in ("recovery", "ordinary")} for label, root in roots.items()}
    summaries = {label: {kind: summary(games) for kind, games in cohorts.items()} for label, cohorts in values.items()}
    effect = paired_interval(values["control"]["recovery"], values["candidate"]["recovery"])
    open_loop = {label: read(Path("experiments/tetra17-recovery-v2/evaluation") / label / "open-loop.json") for label in roots}
    stage5 = {label: read(Path("experiments/tetra17-recovery-v2/evaluation") / label / "stage5/metrics.json")[label]["strict"] for label in roots}
    gate = r["sft_gate"]
    c, b = summaries["candidate"], summaries["control"]
    checks = {
        "recovery_absolute": c["recovery"]["cap_reached"] >= gate["recovery_min_cap_reached"],
        "recovery_effect": effect["mean"] >= gate["recovery_min_paired_improvement"],
        "recovery_uncertainty": effect["ci95"][0] > gate["recovery_paired_bootstrap_ci95_lower_gt"],
        "ordinary_survival": c["ordinary"]["cap_reached"] == gate["ordinary_required_cap_reached"] and c["ordinary"]["illegal"] <= gate["ordinary_max_illegal"] and c["ordinary"]["topout"] <= gate["ordinary_max_topout"],
        "ordinary_lines": c["ordinary"]["mean_lines"] >= gate["ordinary_min_lines_and_score_ratio"] * b["ordinary"]["mean_lines"],
        "ordinary_score": c["ordinary"]["mean_score"] >= gate["ordinary_min_lines_and_score_ratio"] * b["ordinary"]["mean_score"],
        "stage5_survival": stage5["candidate"]["cap_outs"] == gate["stage5_required_cap_reached"] and stage5["candidate"]["deaths"] <= gate["stage5_max_deaths"],
        "stage5_lines": stage5["candidate"]["lines"]["mean"] >= gate["stage5_min_lines_ratio"] * stage5["control"]["lines"]["mean"],
        "open_loop_parse": open_loop["candidate"]["parse_rate"] >= gate["open_loop_parse_min"],
        "open_loop_legality": open_loop["candidate"]["legality_rate"] >= gate["open_loop_legality_min"],
        "open_loop_exact": open_loop["candidate"]["exact_match"] >= open_loop["control"]["exact_match"] - gate["open_loop_exact_max_drop"],
    }
    report = {
        "status": "passed" if all(checks.values()) else "not_passed", "checks": checks,
        "summaries": summaries, "paired_recovery_cap_effect": effect, "open_loop": open_loop,
        "stage5": stage5, "conditional_rl_authorized_by_gate": all(checks.values()),
        "confirmation_access": False, "final_test_access": False, "registration_sha256": rh,
    }
    out = Path("experiments/tetra17-recovery-v2/sft-gate.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": checks, "paired_recovery_cap_effect": effect}, indent=2))


if __name__ == "__main__":
    main()
