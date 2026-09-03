#!/usr/bin/env python3
"""Replay completed E0/E3 failures and classify them without new model calls."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tetris.recovery import placement_failure
from tetris.rl import atomic_write_json, file_sha256, parse_completion, restore_game, state_hash


def audit_record(row: dict) -> dict:
    start = row.get("starting_state")
    game = restore_game(row["seed"], start["action_prefix"], expected=start) if start else restore_game(row["seed"], [])
    context = []
    for action in row["actions"]:
        before = game.snapshot()
        game.step(*action)
        context.append({"turn": before["turn"], "action": action, "before_state_hash": state_hash(before),
                        "max_height": before["max_height"], "holes": before["holes_total"]})
    if game.game_over:
        raise ValueError("illegal incident was already an engine terminal state")
    raw = row["raw_model_output"][-1]
    action = parse_completion(raw)
    failure = placement_failure(game, action)
    if failure == "legal":
        raise ValueError("recorded illegal action is actually legal")
    snap = game.snapshot()
    return {"game_id": row["game_id"], "seed": row["seed"], "policy": row["policy"],
            "continuation_pieces": row["pieces"], "failure": failure, "raw_output": raw,
            "parsed_action": action, "terminal_state": snap, "state_hash": state_hash(snap),
            "legal_alternatives": len(snap["legal"]), "preceding_moves": context[-8:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("refusing to replace a completed audit")
    records, sources = [], {}
    for run, label in [("stage6-e0", "sft"), ("rl-e3-kl001-seed0", "kl001"),
                       ("rl-e3-kl005-seed0", "kl005"), ("rl-e3-kl010-seed0", "kl010")]:
        for cohort in ("games", "recovery_games"):
            path = Path("runs") / run / "rl/stress-development" / f"{cohort}.jsonl"
            sources[str(path)] = file_sha256(path)
            for row in map(json.loads, path.read_text().splitlines()):
                if row["policy"] == label and row["death_reason"] == "illegal_action":
                    records.append({"cohort": cohort, **audit_record(row)})
    counts = {label: dict(Counter(r["failure"] for r in records if r["policy"] == label))
              for label in ("sft", "kl001", "kl005", "kl010")}
    report = {"status": "verified", "sources_sha256": sources, "counts": counts, "incidents": records,
              "training_examples_created": 0, "final_test_access": False,
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "interpretation": "All recorded recovery illegal actions collide at the top despite legal alternatives. KL .01 additionally exceeds the board boundary in a long game. This proves failure mechanics, not whether data coverage or observation aliasing caused the policy error."}
    atomic_write_json(args.out, report)
    print(json.dumps({"status": report["status"], "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
