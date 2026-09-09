#!/usr/bin/env python3
"""Collect one frozen, training-only on-policy correction round."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
import time
from pathlib import Path

from scripts.eval_stress import run_recovery_rollouts
from tetris.curriculum import enriched_row
from tetris.model_policy import build_model_policy
from tetris.rl import atomic_write_json, directory_sha256, file_sha256, parse_completion, restore_game


def read(path: Path):
    return json.loads(path.read_text())


def jsonl(path: Path, rows) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def rollout_shards(registration: dict, registration_path: Path, starts: list[dict], out: Path, device: str) -> list[dict]:
    source = Path(registration["frozen_control"]["adapter"])
    expected = registration["frozen_control"]["adapter_sha256"]
    if directory_sha256(source) != expected:
        raise ValueError("frozen Tetra-1.7B adapter changed")
    shards = out / "rollouts"
    shards.mkdir(parents=True, exist_ok=True)
    policy = None
    games = []
    for offset in range(0, len(starts), registration["dataset"]["on_policy_batch_size"]):
        shard_path = shards / f"games-{offset:05d}.json"
        shard = starts[offset : offset + registration["dataset"]["on_policy_batch_size"]]
        if not shard_path.exists():
            if policy is None:
                policy = build_model_policy(source, registration["base_model"], device, revision=registration["base_model_revision"])
            records, metrics = run_recovery_rollouts(policy, shard, cap=registration["dataset"]["on_policy_cap"], batch_size=registration["dataset"]["on_policy_batch_size"])
            atomic_write_json(shard_path, {
                "registration_sha256": file_sha256(registration_path), "adapter_sha256": expected,
                "starting_state_hashes": [s["state_hash"] for s in shard], "games": records, "metrics": metrics,
            })
        saved = read(shard_path)
        if saved["registration_sha256"] != file_sha256(registration_path) or saved["adapter_sha256"] != expected or saved["starting_state_hashes"] != [s["state_hash"] for s in shard]:
            raise ValueError("rollout shard identity mismatch")
        games.extend(saved["games"])
    return games


def correction_candidates(games: list[dict], registration: dict) -> list[dict]:
    cfg = registration["dataset"]
    candidates, seen = [], set()
    for record in games:
        start = record["starting_state"]
        game = restore_game(record["seed"], start["action_prefix"], expected=start)
        actions = record.get("actions", [])
        raws = record.get("raw_model_output", [])
        failed = record["death_reason"] in ("illegal_action", "topped_out")
        for index in range(len(raws)):
            snapshot = game.snapshot()
            model_action = parse_completion(raws[index])
            terminal_illegal = index >= len(actions)
            row = enriched_row(snapshot, split="train", kind="on_policy_correction", source_id=f"{record['game_id']}-{index}", extra={
                "frozen_model_raw_action": raws[index],
                "frozen_model_action": list(model_action) if model_action is not None else None,
                "frozen_model_terminal_illegal": terminal_illegal,
                "source_outcome": record["death_reason"],
                "failure_distance": len(actions) - index if failed else None,
            })
            if terminal_illegal:
                category = "terminal_illegal"
                regret = None
            else:
                action = tuple(actions[index])
                model_value = row["teacher_values"].get(f"{action[0]},{action[1]}")
                best_value = row["teacher_values"][f"{row['rot']},{row['x']}"]
                regret = best_value - model_value if model_value is not None else None
                if failed and len(actions) - index <= cfg["failure_context"]:
                    category = "pre_failure"
                elif regret is not None and regret >= cfg["high_regret_threshold"]:
                    category = "high_regret"
                elif action != (row["rot"], row["x"]) and row["teacher_margin"] >= cfg["clear_margin_threshold"]:
                    category = "clear_disagreement"
                else:
                    category = "matched_control"
                game.step(*action)
            row["correction_category"] = category
            row["teacher_regret_of_model_action"] = regret
            if row["state_hash"] not in seen:
                seen.add(row["state_hash"])
                candidates.append(row)
            if terminal_illegal:
                break
    return candidates


def select_rows(candidates: list[dict], target: int, seed: int) -> list[dict]:
    priority = {"terminal_illegal": 0, "pre_failure": 1, "high_regret": 2, "clear_disagreement": 3, "matched_control": 4}
    def key(row):
        tie = hashlib.sha256(f"{seed}:{row['state_hash']}".encode()).hexdigest()
        return priority[row["correction_category"]], tie
    selected = sorted(candidates, key=key)[:target]
    if len(selected) != target:
        raise ValueError(f"only {len(selected)}/{target} correction rows available")
    random.Random(seed).shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    registration = read(args.registration)
    if registration["experiment"] != "tetra17-recovery-v2" or registration["final_test_access"]:
        raise ValueError("wrong or unsafe registration")
    staging = Path(registration["dataset"]["staging_dir"])
    out = Path(registration["dataset"]["on_policy_dir"])
    out.mkdir(parents=True, exist_ok=True)
    starts = [json.loads(line) for line in (staging / "on-policy-starts.jsonl").open()]
    games = rollout_shards(registration, args.registration, starts, out, args.device)
    candidates = correction_candidates(games, registration)
    staging_states = set()
    staging_prompt_labels = {}
    with (staging / "rows.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            staging_states.add(row["state_hash"])
            staging_prompt_labels[row["prompt_hash"]] = (row["rot"], row["x"])
    candidates = [row for row in candidates if row["state_hash"] not in staging_states and staging_prompt_labels.get(row["prompt_hash"], (row["rot"], row["x"])) == (row["rot"], row["x"])]
    rows = select_rows(candidates, registration["dataset"]["on_policy_rows"], registration["dataset"]["generator_seed"] + 5)
    jsonl(out / "rows.jsonl", rows)
    manifest = {
        "status": "completed", "registration_sha256": file_sha256(args.registration),
        "frozen_adapter_sha256": registration["frozen_control"]["adapter_sha256"],
        "num_games": len(games), "num_candidates": len(candidates), "num_rows": len(rows),
        "outcomes": dict(Counter(g["death_reason"] for g in games)),
        "categories": dict(Counter(r["correction_category"] for r in rows)),
        "rows_sha256": file_sha256(out / "rows.jsonl"), "final_test_access": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
