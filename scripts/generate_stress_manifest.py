#!/usr/bin/env python3
"""Create the frozen Stage 6 ``stress-v1`` seeds and exact replay states.

The generated state records use ``seed + action_prefix`` as their generator
state.  Board, piece, next piece, prompt, and a state hash are retained as
redundant replay checks.  Existing manifests are never replaced unless
``--force`` is explicit.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from tetris.engine import Game
from tetris.rl import DenseRewardWeights, dense_transition, record_state, validate_seed_manifest
from tetris.rollout import default_eval_seeds

ROOT = Path(__file__).resolve().parent.parent


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def stage3_ranges(data_dir: Path) -> list[range]:
    result = []
    for path in sorted(data_dir.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        result.append(range(int(manifest["seed_start"]), int(manifest["seed_start"]) + int(manifest["num_games"])))
    return result


def choose_noisy_action(game: Game, rng: random.Random, noise_rate: float) -> tuple[int, int]:
    legal = [(item["rot"], item["x"]) for item in game.snapshot()["legal"]]
    if rng.random() < noise_rate:
        return rng.choice(legal)
    weights = DenseRewardWeights(lines=1.0, holes=1.5, aggregate_height=0.08, bumpiness=0.03)
    return max(legal, key=lambda action: (dense_transition(game, action, weights).reward, -action[0], -action[1]))


def sample_states(
    seeds: list[int],
    *,
    kind: str,
    count: int,
    sample_seed: int,
    noise_rate: float,
) -> list[dict]:
    rng = random.Random(sample_seed)
    candidates: list[dict] = []
    for seed in seeds:
        game = Game(seed)
        actions: list[list[int]] = []
        for _ in range(180):
            if game.game_over:
                break
            action = choose_noisy_action(game, rng, noise_rate)
            actions.append([action[0], action[1]])
            game.step(*action)
            if game.game_over or game.turn < 8 or game.turn % 4:
                continue
            snap = game.snapshot()
            difficulty = (
                3 * snap["holes_total"]
                + snap["max_height"]
                + 0.2 * snap["bumpiness"]
                + 0.05 * snap["aggregate_height"]
            )
            row = record_state(game, actions, state_id=f"{kind}-{seed}-{game.turn}")
            row.update(
                {
                    "kind": kind,
                    "difficulty": difficulty,
                    "features": {
                        "holes_total": snap["holes_total"],
                        "aggregate_height": snap["aggregate_height"],
                        "max_height": snap["max_height"],
                        "bumpiness": snap["bumpiness"],
                        "well_depth": sum(snap["wells"]),
                    },
                }
            )
            candidates.append(row)
    if len(candidates) < count:
        raise SystemExit(f"only sampled {len(candidates)} {kind} candidates; need {count}")
    candidates.sort(key=lambda row: (row["difficulty"], row["state_id"]))
    if kind == "recovery":
        pool = candidates[len(candidates) // 2 :]
        chosen = sorted(rng.sample(pool, count), key=lambda row: row["state_id"])
    else:
        # Quantile selection gives probes spanning easy through difficult.
        indexes = [round(i * (len(candidates) - 1) / max(1, count - 1)) for i in range(count)]
        chosen = [candidates[index] for index in indexes]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/stress-v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--training-seeds", type=int, default=256)
    parser.add_argument("--development-seeds", type=int, default=20)
    parser.add_argument("--test-seeds", type=int, default=100)
    parser.add_argument("--recovery-states", type=int, default=24)
    parser.add_argument("--probe-states", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = args.out_dir / "manifest.json"
    states_path = args.out_dir / "states.jsonl"
    if not args.force and (manifest_path.exists() or states_path.exists()):
        raise SystemExit(f"refusing to replace frozen stress-v1 files in {args.out_dir}; pass --force explicitly")

    training = list(range(20_000_000, 20_000_000 + args.training_seeds))
    development = list(range(30_000_000, 30_000_000 + args.development_seeds))
    test = list(range(40_000_000, 40_000_000 + args.test_seeds))
    recovery_sources = list(range(50_000_000, 50_000_032))
    probe_sources = list(range(60_000_000, 60_000_032))
    stage5 = default_eval_seeds(100)
    manifest = {
        "benchmark_id": "stress-v1",
        "version": 1,
        "status": "registered",
        "git_sha_at_registration": git_sha(),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "training_seeds": training,
        "development_seeds": development,
        "test_seeds": test,
        "stage5_seeds": stage5,
        "recovery_source_seeds": recovery_sources,
        "probe_source_seeds": probe_sources,
        "long_horizon": {
            "development_cap": 2000,
            "test_cap": 5000,
            "development_count": args.development_seeds,
            "test_count": args.test_seeds,
        },
        "recovery_cap": 200,
        "state_counts": {"recovery": args.recovery_states, "probe": args.probe_states},
        "primary_metric": "score_per_100_pieces",
        "gamma": 0.99,
        "budgets_usd": {"pilot": 20.0, "stage": 100.0},
        "decision_rules": {
            "stage5_min_mean_lines": 196.77,
            "stage5_max_deaths": 0,
            "stage5_max_parse_failures": 0,
            "stage5_max_illegal_actions": 0,
            "development_relative_improvement": 0.03,
            "confirmation_min_improved_training_seeds": 2,
            "confirmation_training_seed_count": 3,
            "confirmation_combined_ci_must_exceed": 0.0,
        },
        "sampling": {
            "recovery": {"policy": "noisy_dense_one_ply", "noise_rate": 0.35, "sample_seed": 6001},
            "probe": {"policy": "noisy_dense_one_ply", "noise_rate": 0.15, "sample_seed": 6002},
        },
    }
    validation = validate_seed_manifest(manifest, stage3_ranges=stage3_ranges(args.data_dir))
    states = []
    for split, source_slice, seed_offset in (("development", slice(0, 16), 0), ("test", slice(16, 32), 100)):
        for kind, sources, count, sample_seed, noise_rate in (
            ("recovery", recovery_sources, args.recovery_states, 6001, 0.35),
            ("probe", probe_sources, args.probe_states, 6002, 0.15),
        ):
            split_count = count // 2 if split == "development" else count - count // 2
            selected = sample_states(
                sources[source_slice],
                kind=kind,
                count=split_count,
                sample_seed=sample_seed + seed_offset,
                noise_rate=noise_rate,
            )
            for row in selected:
                row["split"] = split
            states.extend(selected)
    manifest["state_split_rule"] = "first 16 source seeds per kind are development; last 16 are final test"
    manifest["seed_validation"] = validation
    manifest["states_hash"] = canonical_states_hash(states)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with states_path.open("w") as handle:
        for state in states:
            handle.write(json.dumps(state, sort_keys=True) + "\n")
    print(f"wrote {manifest_path} and {len(states)} states to {states_path}")


def canonical_states_hash(states: list[dict]) -> str:
    from tetris.rl import canonical_hash

    return canonical_hash(states)


if __name__ == "__main__":
    main()
