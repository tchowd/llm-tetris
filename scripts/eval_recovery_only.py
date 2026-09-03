#!/usr/bin/env python3
"""Fresh recovery validation on the held-out training-seed partition, not test."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_stress import evaluate_states, read_states, run_recovery_rollouts
from tetris.model_policy import build_model_policy
from tetris.rl import atomic_write_json, directory_sha256, file_sha256
from tetris.rollout import teacher_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    r = json.loads(args.registration.read_text())
    data_dir = Path(r["data"]["data_dir"])
    data = json.loads((data_dir / "manifest.json").read_text())
    states_path = data_dir / "validation-starts.jsonl"
    if file_sha256(states_path) != data["files_sha256"][states_path.name]:
        raise ValueError("fresh validation states changed")
    states = read_states(states_path)
    if len(states) != r["data"]["fresh_recovery_validation_starts"] or any(
        s["split"] != "eval" or s["seed"] not in data["validation_seeds"] for s in states
    ):
        raise ValueError("fresh validation cohort is not the registered held-out partition")
    if args.out_dir.exists():
        raise ValueError("refusing to overwrite fresh validation")
    args.out_dir.mkdir(parents=True)
    adapter_hash = directory_sha256(args.adapter_dir) if args.adapter_dir else None
    registration = {"status": "running", "kind": "fresh_recovery_validation", "suite": "fresh_validation",
                    "label": args.label, "mode": "strict", "greedy": True, "cap": 200,
                    "registration_sha256": file_sha256(args.registration), "states_sha256": file_sha256(states_path),
                    "adapter_sha256": adapter_hash, "final_test_access": False}
    atomic_write_json(args.out_dir / "manifest.json", registration)
    started = time.monotonic()
    policy = build_model_policy(args.adapter_dir, r["base_model"], "cuda", revision=r["base_model_revision"]) if args.adapter_dir else teacher_policy()
    fixed, fixed_metrics = evaluate_states(policy, states, 32)
    games, metrics = run_recovery_rollouts(policy, states, cap=200, batch_size=32)
    if args.adapter_dir and directory_sha256(args.adapter_dir) != adapter_hash:
        raise ValueError("adapter changed during validation")
    for name, rows in (("games", games), ("states", fixed)):
        with (args.out_dir / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps({**row, "policy": args.label}) + "\n")
    atomic_write_json(args.out_dir / "metrics.json", {"recovery_rollouts": metrics, "fixed_states": fixed_metrics})
    atomic_write_json(args.out_dir / "manifest.json", {**registration, "status": "completed",
        "policy_metadata": getattr(policy, "metadata", {}), "wall_clock_seconds": time.monotonic() - started,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps({"label": args.label, "deaths": metrics["deaths"], "illegal_deaths": metrics["illegal_action_deaths"]}))


if __name__ == "__main__":
    main()
