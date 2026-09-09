#!/usr/bin/env python3
"""Build and assemble the preregistered Tetra-1.7B recovery curriculum."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
import time
from pathlib import Path

from tetris.curriculum import difficulty_band, enriched_row, state_hash_from_snapshot
from tetris.engine import Game
from tetris.rl import DenseRewardWeights, atomic_write_json, dense_transition, file_sha256, record_state


def read(path: Path):
    return json.loads(path.read_text())


def write_jsonl(path: Path, rows) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def reservoir_ordinary(
    data_dirs: list[Path],
    count: int,
    seed: int,
    per_game_cap: int,
    excluded_states: set[str] | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    reservoir, eligible = [], 0
    game_counts = Counter()
    seen_states = set(excluded_states or ())
    for directory in data_dirs:
        with (directory / "rows.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["split"] != "train" or row.get("explored") or game_counts[row["game_id"]] >= per_game_cap:
                    continue
                game_counts[row["game_id"]] += 1
                state_hash = state_hash_from_snapshot(row)
                if state_hash in seen_states:
                    continue
                seen_states.add(state_hash)
                eligible += 1
                row["source_dataset"] = str(directory)
                if len(reservoir) < count:
                    reservoir.append(row)
                else:
                    index = rng.randrange(eligible)
                    if index < count:
                        reservoir[index] = row
    if len(reservoir) != count:
        raise ValueError(f"ordinary reservoir has {len(reservoir)}/{count} rows")
    result = []
    for row in reservoir:
        snapshot = {**row, "game_over": False, "legal": []}
        result.append(enriched_row(snapshot, split="train", kind="ordinary_retention", source_id=f"ordinary-{row['game_id']}-{row['turn']}", extra={
            "source_dataset": row["source_dataset"], "source_game_id": row["game_id"], "source_explored": False,
        }))
    return result


def exploration_action(game: Game, rng: random.Random, noise: float) -> tuple[int, int]:
    legal = [(p["rot"], p["x"]) for p in game.snapshot()["legal"]]
    if rng.random() < noise:
        return rng.choice(legal)
    weights = DenseRewardWeights(lines=1, holes=1.5, aggregate_height=.08, bumpiness=.03)
    return max(legal, key=lambda action: (dense_transition(game, action, weights).reward, -action[0], -action[1]))


def generated_components(registration: dict) -> tuple[list[dict], list[dict], list[dict]]:
    recipe = registration["dataset"]
    progressive_targets = Counter(recipe["progressive_rows_by_band"])
    safety_targets = Counter(recipe["safety_rows_by_tag"])
    progressive_counts, safety_counts = Counter(), Counter()
    rows, starts, eval_rows = [], [], []
    seen_states, prompt_labels, start_seed_used = set(), {}, set()
    rng = random.Random(recipe["generator_seed"])
    train_seeds = range(recipe["training_seed_start"], recipe["training_seed_start"] + recipe["training_seed_count"])

    for seed in train_seeds:
        if progressive_counts >= progressive_targets and safety_counts >= safety_targets and len(starts) >= recipe["on_policy_starts"]:
            break
        game, prefix = Game(seed), []
        for _ in range(recipe["exploration_cap"]):
            if game.game_over:
                break
            snapshot = game.snapshot()
            band = difficulty_band(snapshot["max_height"], snapshot["holes_total"])
            row = enriched_row(snapshot, split="train", kind="candidate", source_id=f"generated-{seed}-{game.turn}")
            label = (row["rot"], row["x"])
            state_key, prompt_key = row["state_hash"], row["prompt_hash"]
            if state_key not in seen_states and prompt_labels.get(prompt_key, label) == label:
                chosen_kind = None
                available_tags = [tag for tag in row["safety_tags"] if safety_counts[tag] < safety_targets[tag]]
                if available_tags:
                    tag = min(available_tags, key=lambda value: safety_counts[value] / safety_targets[value])
                    row["kind"], row["safety_cohort"] = "rare_safety", tag
                    safety_counts[tag] += 1
                    chosen_kind = "safety"
                elif band in progressive_targets and progressive_counts[band] < progressive_targets[band]:
                    row["kind"] = "progressive_recovery"
                    progressive_counts[band] += 1
                    chosen_kind = "progressive"
                if chosen_kind:
                    rows.append(row)
                    seen_states.add(state_key)
                    prompt_labels[prompt_key] = label
            if band in ("hard", "critical") and len(starts) < recipe["on_policy_starts"] and game.turn >= 8 and seed not in start_seed_used:
                state = record_state(game, prefix, state_id=f"on-policy-{seed}-{game.turn}")
                state.update(split="training", kind="on_policy_start", difficulty_band=band)
                starts.append(state)
                start_seed_used.add(seed)
            action = exploration_action(game, rng, recipe["exploration_random_probability"])
            prefix.append(list(action))
            game.step(*action)
    if progressive_counts != progressive_targets or safety_counts != safety_targets or len(starts) < recipe["on_policy_starts"]:
        raise ValueError({"progressive": [progressive_counts, progressive_targets], "safety": [safety_counts, safety_targets], "starts": len(starts)})

    eval_targets = Counter(recipe["progressive_eval_rows_by_band"])
    eval_counts = Counter()
    eval_rng = random.Random(recipe["generator_seed"] + 1)
    for seed in range(recipe["eval_seed_start"], recipe["eval_seed_start"] + recipe["eval_seed_count"]):
        if eval_counts >= eval_targets:
            break
        game, prefix = Game(seed), []
        for _ in range(recipe["exploration_cap"]):
            if game.game_over:
                break
            snapshot = game.snapshot()
            band = difficulty_band(snapshot["max_height"], snapshot["holes_total"])
            if band in eval_targets and eval_counts[band] < eval_targets[band]:
                row = enriched_row(snapshot, split="eval", kind="progressive_eval", source_id=f"curriculum-eval-{seed}-{game.turn}")
                if row["state_hash"] not in seen_states:
                    eval_rows.append(row)
                    seen_states.add(row["state_hash"])
                    eval_counts[band] += 1
            action = exploration_action(game, eval_rng, recipe["exploration_random_probability"])
            prefix.append(list(action))
            game.step(*action)
    if eval_counts != eval_targets:
        raise ValueError({"eval": [eval_counts, eval_targets]})
    return rows, starts[: recipe["on_policy_starts"]], eval_rows


def build_teacher(registration_path: Path) -> None:
    registration = read(registration_path)
    if registration["experiment"] != "tetra17-recovery-v2" or registration["final_test_access"]:
        raise ValueError("wrong or unsafe registration")
    out = Path(registration["dataset"]["staging_dir"])
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    generated, starts, eval_rows = generated_components(registration)
    ordinary = reservoir_ordinary(
        [Path(p) for p in registration["dataset"]["ordinary_sources"]],
        registration["dataset"]["ordinary_rows"],
        registration["dataset"]["generator_seed"] + 2,
        registration["dataset"]["ordinary_max_rows_per_game"],
        {row["state_hash"] for row in generated + eval_rows},
    )
    rows = ordinary + generated + eval_rows
    random.Random(registration["dataset"]["generator_seed"] + 3).shuffle(rows)
    write_jsonl(out / "rows.jsonl", rows)
    write_jsonl(out / "on-policy-starts.jsonl", starts)
    manifest = {
        "status": "teacher_components_complete", "registration_sha256": file_sha256(registration_path),
        "counts": dict(Counter(row["kind"] for row in rows)), "rows": len(rows), "on_policy_starts": len(starts),
        "files_sha256": {"rows.jsonl": file_sha256(out / "rows.jsonl"), "on-policy-starts.jsonl": file_sha256(out / "on-policy-starts.jsonl")},
        "final_test_access": False, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(out / "manifest.json", manifest)


def assemble(registration_path: Path) -> None:
    registration = read(registration_path)
    cfg = registration["dataset"]
    staging, on_policy, out = Path(cfg["staging_dir"]), Path(cfg["on_policy_dir"]), Path(cfg["final_dir"])
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    sm, om = read(staging / "manifest.json"), read(on_policy / "manifest.json")
    if sm["registration_sha256"] != file_sha256(registration_path) or om["registration_sha256"] != file_sha256(registration_path):
        raise ValueError("component registration mismatch")
    staged_rows = list(json.loads(line) for line in (staging / "rows.jsonl").open())
    on_rows = list(json.loads(line) for line in (on_policy / "rows.jsonl").open())
    if len(on_rows) != cfg["on_policy_rows"]:
        raise ValueError("wrong on-policy row count")
    nonordinary = [row for row in staged_rows if row["kind"] != "ordinary_retention"]
    protected_states = {row["state_hash"] for row in nonordinary + on_rows}
    ordinary = reservoir_ordinary(
        [Path(p) for p in cfg["ordinary_sources"]],
        cfg["ordinary_rows"],
        cfg["generator_seed"] + 2,
        cfg["ordinary_max_rows_per_game"],
        protected_states,
    )
    rows = nonordinary + ordinary + on_rows
    train_states, eval_states = set(), set()
    prompt_labels = {}
    for row in rows:
        target = eval_states if row["split"] == "eval" else train_states
        if row["state_hash"] in target:
            raise ValueError("duplicate state within split")
        target.add(row["state_hash"])
        label = (row["rot"], row["x"])
        if row["prompt_hash"] in prompt_labels and prompt_labels[row["prompt_hash"]] != label:
            raise ValueError("conflicting labels for serialized prompt")
        prompt_labels[row["prompt_hash"]] = label
    if train_states & eval_states:
        raise ValueError("train/eval state leakage")
    out.mkdir(parents=True)
    random.Random(cfg["generator_seed"] + 4).shuffle(rows)
    write_jsonl(out / "rows.jsonl", rows)
    manifest = {
        "status": "completed", "kind": "tetra17_recovery_curriculum_v2", "registration_sha256": file_sha256(registration_path),
        "num_train_rows": sum(r["split"] == "train" for r in rows), "num_eval_rows": sum(r["split"] == "eval" for r in rows),
        "counts": dict(Counter(r["kind"] for r in rows)), "difficulty": dict(Counter(r["difficulty_band"] for r in rows)),
        "component_manifests_sha256": {str(staging / "manifest.json"): file_sha256(staging / "manifest.json"), str(on_policy / "manifest.json"): file_sha256(on_policy / "manifest.json")},
        "ordinary_rebuilt_for_uniqueness": True,
        "ordinary_excluded_state_count": len(protected_states),
        "rows_sha256": file_sha256(out / "rows.jsonl"), "final_test_access": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(out / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--phase", choices=("teacher", "assemble"), required=True)
    args = parser.parse_args()
    (build_teacher if args.phase == "teacher" else assemble)(args.registration)


if __name__ == "__main__":
    main()
