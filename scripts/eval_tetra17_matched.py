#!/usr/bin/env python3
"""Sharded greedy evaluation on the registered ordinary and recovery cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.eval_stress import run_recovery_rollouts
from tetris.engine import Game
from tetris.model_policy import build_model_policy
from tetris.rl import atomic_write_json, directory_sha256, file_sha256, record_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    if registration["experiment"] != "tetra17-recovery-v2" or registration["final_test_access"]:
        raise ValueError("wrong or unsafe registration")
    if args.label == "control":
        adapter = Path(registration["frozen_control"]["adapter"])
    elif args.label == "candidate":
        adapter = Path(registration["candidate"]["output_dir"]) / "adapter"
    else:
        registered_labels = {f"rl-fixed-zero-{seed}" for seed in registration["conditional_rl"]["training_seeds"]}
        if args.label not in registered_labels or args.adapter_dir is None:
            raise ValueError("unregistered evaluation label")
        adapter = args.adapter_dir
    if args.label == "control" and directory_sha256(adapter) != registration["frozen_control"]["adapter_sha256"]:
        raise ValueError("frozen control changed")
    adapter_hash = directory_sha256(adapter)
    out = Path("experiments/tetra17-recovery-v2/evaluation") / args.label
    out.mkdir(parents=True, exist_ok=True)
    recovery_cfg = registration["evaluation"]["recovery"]
    recovery_path = Path(recovery_cfg["path"])
    if file_sha256(recovery_path) != recovery_cfg["sha256"]:
        raise ValueError("recovery development set changed")
    cohorts = {
        "recovery": ([json.loads(line) for line in recovery_path.open()], recovery_cfg["cap"]),
        "ordinary": ([record_state(Game(seed), [], state_id=f"tetra17-ordinary-{seed}") for seed in registration["evaluation"]["ordinary"]["seeds"]], registration["evaluation"]["ordinary"]["cap"]),
    }
    identity = {
        "registration_sha256": file_sha256(args.registration), "label": args.label,
        "adapter_sha256": adapter_hash, "base_model_revision": registration["base_model_revision"],
        "greedy": True, "final_test_access": False,
    }
    policy = None
    files = {}
    for kind, (states, cap) in cohorts.items():
        for offset in range(0, len(states), 32):
            path = out / f"{kind}-{offset:04d}.json"
            shard = states[offset : offset + 32]
            if not path.exists():
                if policy is None:
                    policy = build_model_policy(adapter, registration["base_model"], args.device, revision=registration["base_model_revision"])
                    if policy.metadata["base_model_revision"] != registration["base_model_revision"]:
                        raise ValueError("base model revision drift")
                games, metrics = run_recovery_rollouts(policy, shard, cap=cap, batch_size=32)
                atomic_write_json(path, {**identity, "kind": kind, "cap": cap, "starting_state_hashes": [s["state_hash"] for s in shard], "games": games, "metrics": metrics})
            saved = json.loads(path.read_text())
            if any(saved.get(k) != v for k, v in identity.items()) or saved["starting_state_hashes"] != [s["state_hash"] for s in shard]:
                raise ValueError("evaluation shard mismatch")
            files[str(path)] = file_sha256(path)
    atomic_write_json(out / "complete.json", {**identity, "status": "completed", "files_sha256": files})


if __name__ == "__main__":
    main()
