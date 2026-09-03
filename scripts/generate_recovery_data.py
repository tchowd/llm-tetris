#!/usr/bin/env python3
"""Build a frozen training-only recovery mixture; never ingest failure boards."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tetris.engine import Game
from tetris.rl import DenseRewardWeights, atomic_write_json, dense_transition, file_sha256, record_state
from tetris.serialize import serialize_action
from tetris.teacher import pick


def jsonl(path: Path, rows: list[dict]):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def noisy_start(seed: int, rng: random.Random, *, max_turns: int, min_height: int, noise: float):
    game, actions = Game(seed=seed), []
    weights = DenseRewardWeights(lines=1, holes=1.5, aggregate_height=.08, bumpiness=.03)
    for _ in range(max_turns):
        if game.game_over:
            return None
        snapshot = game.snapshot()
        if game.turn >= 4 and snapshot["max_height"] >= min_height:
            return game, actions
        legal = [(p["rot"], p["x"]) for p in snapshot["legal"]]
        action = rng.choice(legal) if rng.random() < noise else max(
            legal, key=lambda a: (dense_transition(game, a, weights).reward, -a[0], -a[1]))
        actions.append(list(action))
        game.step(*action)
    return None


def recovery_rows(seeds: list[int], *, count: int, split: str, recipe: dict, seed: int):
    rng = random.Random(seed)
    rows, starts, used = [], [], set()
    for attempt in range(recipe["max_generation_attempts"]):
        if len(rows) >= count:
            break
        game_seed = seeds[attempt % len(seeds)]
        result = noisy_start(game_seed, rng, max_turns=recipe["noisy_prefix_max_turns"],
                             min_height=recipe["start_min_height"], noise=recipe["noisy_random_probability"])
        if result is None:
            continue
        game, prefix = result
        state = record_state(game, prefix, state_id=f"{split}-{game_seed}-{attempt}")
        if state["state_hash"] in used:
            continue
        used.add(state["state_hash"])
        state.update(split=split, kind="recovery")
        starts.append(state)
        for _ in range(recipe["teacher_continuation_cap"]):
            if game.game_over or len(rows) >= count:
                break
            snapshot = game.snapshot()
            action = pick(snapshot, snapshot["legal"])
            rows.append({"game_id": state["state_id"], "seed": game_seed, "split": split,
                         "kind": "recovery", "prompt": snapshot["prompt"],
                         "completion": serialize_action(*action), "rot": action[0], "x": action[1],
                         "state": record_state(game, prefix), "legal_count": len(snapshot["legal"]),
                         "max_height": snapshot["max_height"]})
            prefix = [*prefix, list(action)]
            game.step(*action)
        if len(starts) % 16 == 0:
            print(f"{split}: {len(rows)}/{count} rows from {len(starts)} starts", flush=True)
    if len(rows) != count:
        raise ValueError(f"insufficient {split} recovery data: {len(rows)}/{count}")
    return rows, starts


def ordinary_rows(directories: list[Path], count: int, seed: int):
    rng, rows, seen = random.Random(seed), [], 0
    for directory in directories:
        with (directory / "rows.jsonl").open() as handle:
            for line in handle:
                original = json.loads(line)
                if original["split"] != "train":
                    continue
                seen += 1
                row = {key: original[key] for key in ("game_id", "prompt", "completion", "rot", "x", "split")}
                row.update(kind="ordinary", source=str(directory), seed=original.get("seed"))
                if len(rows) < count:
                    rows.append(row)
                else:
                    index = rng.randrange(seen)
                    if index < count:
                        rows[index] = row
    if len(rows) != count:
        raise ValueError("insufficient ordinary training data")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    r = json.loads(args.registration.read_text())
    if r["protocol"] != "stage6-recovery-v1" or r["final_test_access"]:
        raise ValueError("requires development-only recovery registration")
    benchmark_path = Path(r["benchmark_manifest"])
    assert file_sha256(benchmark_path) == r["benchmark_manifest_sha256"]
    benchmark = json.loads(benchmark_path.read_text())
    train_seeds, validation_seeds = benchmark["training_seeds"][:192], benchmark["training_seeds"][192:]
    if not train_seeds or not validation_seeds or set(train_seeds) & set(validation_seeds):
        raise ValueError("invalid training/validation partition")
    d = r["data"]
    out = Path(d["data_dir"])
    if out.exists():
        raise SystemExit(f"refusing to overwrite dataset {out}")
    out.mkdir(parents=True)
    atomic_write_json(out / "generation-status.json", {"status": "running", "registration_sha256": file_sha256(args.registration)})
    recovery, starts = recovery_rows(train_seeds, count=d["recovery_train_rows"], split="train", recipe=d, seed=d["generator_seed"])
    validation, val_starts = recovery_rows(validation_seeds, count=d["recovery_validation_rows"], split="eval", recipe=d, seed=d["generator_seed"] + 1)
    if len(val_starts) < d["fresh_recovery_validation_starts"]:
        raise ValueError("insufficient independent fresh validation starts")
    normal = ordinary_rows([Path(p) for p in d["normal_data_dirs"]], d["ordinary_train_rows"], d["generator_seed"] + 2)
    rows = recovery + normal
    random.Random(d["generator_seed"] + 3).shuffle(rows)
    assert not {x["game_id"] for x in rows} & {x["game_id"] for x in validation}
    jsonl(out / "rows.jsonl", rows + validation)
    jsonl(out / "train-starts.jsonl", starts)
    jsonl(out / "validation-starts.jsonl", val_starts[:d["fresh_recovery_validation_starts"]])
    atomic_write_json(out / "training-seeds.json", train_seeds)
    atomic_write_json(out / "validation-seeds.json", validation_seeds)
    manifest = {"status": "completed", "kind": "recovery_sft_dataset", "registration_sha256": file_sha256(args.registration),
                "benchmark_sha256": file_sha256(benchmark_path), "recipe": d,
                "prefix_policy": "40% uniform legal / 60% immediate heuristic; weights lines=1 holes=1.5 height=.08 bumpiness=.03",
                "training_seeds": train_seeds, "validation_seeds": validation_seeds,
                "num_train_rows": len(rows), "num_eval_rows": len(validation),
                "num_training_starts": len(starts), "fresh_validation_starts": d["fresh_recovery_validation_starts"],
                "counts": dict(Counter(x["kind"] for x in rows)),
                "train_recovery_high_board_rows": sum(x["max_height"] >= 18 for x in recovery),
                "files_sha256": {p.name: file_sha256(p) for p in sorted(out.glob("*.json*")) if p.name != "generation-status.json"},
                "original_data_manifests_sha256": {p: file_sha256(Path(p) / "manifest.json") for p in d["normal_data_dirs"]},
                "final_test_access": False, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_write_json(out / "manifest.json", manifest)
    atomic_write_json(out / "generation-status.json", {"status": "completed"})
    print(json.dumps({k: manifest[k] for k in ("status", "num_train_rows", "num_eval_rows", "num_training_starts", "train_recovery_high_board_rows")}, indent=2))


if __name__ == "__main__":
    main()
