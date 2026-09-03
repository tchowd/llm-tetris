#!/usr/bin/env python3
"""Local preparation, validation and preregistered analysis for the feedback pilot.

Preparation never loads a model, reads final-test states, or calls AWS.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_stress_manifest import choose_noisy_action
from tetris.engine import Game
from tetris.recovery import load_start_bank
from tetris.rl import (atomic_write_json, directory_sha256, file_sha256,
                       record_state, restore_game)

ROOT = Path("experiments/stage6-feedback-v1")
REGISTRATION = ROOT / "registration.json"
METHODS = ("active_group", "fixed_zero")
SEEDS = (6201, 6202, 6203)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    atomic_write_json(path, value)


def schedule(seed, train_seeds, bank, updates=32):
    rng = random.Random(seed)
    result = []
    for update in range(1, updates + 1):
        start = bank[rng.randrange(len(bank))] if update % 2 == 0 else None
        environment_seed = start["seed"] if start else train_seeds[rng.randrange(len(train_seeds))]
        result.append({"update": update, "environment_seed": environment_seed,
                       "state_hash": start["state_hash"] if start else None,
                       "state_id": start["state_id"] if start else None})
    return result


def recovery_state(seed):
    """One pre-model-selected difficult state per independent source game.

    Deterministic first nonterminal state at height >=16 after >=8 decisions;
    retry prefix RNG only (never model outcomes), at most eight attempts.
    """
    for attempt in range(8):
        rng = random.Random(seed * 100 + attempt)
        game, actions = Game(seed), []
        for _ in range(160):
            if game.game_over:
                break
            action = choose_noisy_action(game, rng, .4)
            actions.append(list(action))
            game.step(*action)
            if not game.game_over and game.turn >= 8 and game.snapshot()["max_height"] >= 16:
                row = record_state(game, actions, state_id=f"feedback-dev-{seed}")
                row.update(split="development", kind="recovery", prefix_attempt=attempt)
                return row
    raise ValueError(f"no eligible recovery state for registered seed {seed}; do not replace seed silently")


def prepare():
    if REGISTRATION.exists():
        raise ValueError("registration already exists; validate it instead")
    p = read("runs/stage6-recovery-v1/rl/registration.json")
    data = Path(p["data"]["data_dir"])
    training = read(data / "training-seeds.json")
    bank = load_start_bank(data / "train-starts.jsonl", training)
    benchmark = read(p["benchmark_manifest"])
    recovery_seeds = list(range(81_000_000, 81_000_128))
    ordinary_seeds = list(range(82_000_000, 82_000_020))
    reserved_confirmation = {"recovery": [83_000_000, 83_000_256], "ordinary": [84_000_000, 84_000_100]}
    all_new = recovery_seeds + ordinary_seeds + list(range(*reserved_confirmation["recovery"])) + list(range(*reserved_confirmation["ordinary"]))
    old_seeds = {s for k, values in benchmark.items() if k.endswith("seeds") for s in values}
    old_seeds.update(read(data / "validation-seeds.json"))
    ranges = []
    stage3_manifests = []
    for path in sorted(Path("data").glob("*/manifest.json")):
        manifest = read(path)
        if "seed_start" in manifest:
            ranges.append(range(manifest["seed_start"], manifest["seed_start"] + manifest["num_games"]))
            stage3_manifests.append(path)
    if not ranges:
        raise ValueError("Stage 3 source seed ranges missing")
    if old_seeds.intersection(all_new) or any(s in span for s in all_new for span in ranges):
        raise ValueError("new evaluation seeds overlap an existing partition")
    expected_sft = p["frozen_sft_adapter_sha256"]
    if directory_sha256(Path("runs/sft-v1/adapter")) != expected_sft:
        raise ValueError("original SFT content changed")
    states = [recovery_state(seed) for seed in recovery_seeds]
    state_path = ROOT / "recovery-development.jsonl"
    with state_path.open("x") as handle:
        for state in states:
            handle.write(json.dumps(state, sort_keys=True) + "\n")
    paths = [*stage3_manifests, *Path("tetris").glob("*.py"), *Path("scripts").glob("*.py"),
             *Path("tests").glob("test*feedback*.py"), Path("tests/test_episode_rl.py"),
             Path("requirements-train.txt"), Path("requirements-rl.txt"), Path("pyproject.toml"),
             ROOT / "feedback-spec.md", ROOT / "protocol.md", ROOT / "operations.md",
             state_path, data / "training-seeds.json", data / "train-starts.jsonl",
             data / "manifest.json", Path(p["benchmark_manifest"]),
             Path("runs/sft-v1/closed_loop/manifest.json"), Path("runs/sft-v1/closed_loop/metrics.json")]
    # Hash metadata/code and our new dev states; never open sealed test states.
    r = {
        "experiment": "stage6-feedback-v1", "status": "registered_awaiting_budget_approval",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "question": "Does fixed-zero reward-to-go improve greedy recovery relative to original RL?",
        "base_model": p["base_model"], "base_model_revision": p["base_model_revision"],
        "initial_adapter": "runs/sft-v1/adapter", "initial_adapter_sha256": expected_sft,
        "tokenizer": "pinned base revision; identical across arms", "final_test_access": False,
        "deployment_authorized": False, "methods": list(METHODS), "training_seeds": list(SEEDS),
        "training_seed_file": str(data / "training-seeds.json"),
        "recovery_start_file": str(data / "train-starts.jsonl"),
        "benchmark_manifest": p["benchmark_manifest"],
        "recipe": {"updates": 32, "horizon": 128, "group_size": 4, "gamma": .99,
            "temperature": 1., "train_batch_size": 4, "learning_rate": 1e-6, "kl_beta": .05,
            "score_scale": 100., "death_penalty": 2., "illegal_penalty": 10.,
            "advantage_reward_scale": 10., "save_every": 4},
        "optimizer": {"name": "AdamW", "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01,
                      "gradient_clip_norm": 1., "scheduler": "cosine", "warmup_updates": 1, "horizon": 32},
        "sampling": {"top_p": 1., "top_k": 0, "max_new_tokens": 16},
        "schedules": {str(seed): schedule(seed, training, bank) for seed in SEEDS},
        "run_order": [{"run_id": f"rl-feedback-v1-{method}-seed{seed}", "method": method, "seed": seed}
            for i, seed in enumerate(SEEDS) for method in (METHODS if i % 2 == 0 else METHODS[::-1])],
        "evaluation": {"recovery_path": str(state_path), "recovery_seeds": recovery_seeds,
            "ordinary_seeds": ordinary_seeds, "recovery_cap": 200, "ordinary_cap": 1000,
            "batch_size": 32, "greedy": True, "mode": "strict", "checkpoint": "final update 32 only",
            "baseline": "original SFT evaluated once on exactly the same cases",
            "sealed_confirmation_seed_ranges_half_open": reserved_confirmation,
            "confirmation_access": False},
        "analysis": {"primary": "absolute recovery illegal-ending rate reduction: original minus revised",
            "minimum_useful_reduction": .10, "min_improving_seeds": 2,
            "bootstrap_replicates": 10000, "bootstrap_seed": 73220,
            "interval": "crossed paired bootstrap: resample three seed pairs and source games, 95% percentile",
            "require_lower_bound_above": 0., "max_recovery_survival_regression": 0.,
            "ordinary_min_score_ratio": .99, "ordinary_min_lines_ratio": .99,
            "ordinary_max_illegal_endings": 0, "ordinary_max_topouts": 0,
            "ordinary_max_parse_failures": 0},
        "source_and_input_sha256": {str(path): file_sha256(path) for path in sorted(set(paths))},
        "budget_approval": None,
    }
    write_new(REGISTRATION, r)
    print(f"registered six runs: {REGISTRATION}; sha256={file_sha256(REGISTRATION)}")


def validate(path=REGISTRATION):
    r = read(path)
    if r["experiment"] != "stage6-feedback-v1" or r["final_test_access"] is not False or r["evaluation"]["confirmation_access"] is not False:
        raise ValueError("not the development-only feedback pilot")
    for name, digest in r["source_and_input_sha256"].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"registered source/input changed: {name}")
    if directory_sha256(Path(r["initial_adapter"])) != r["initial_adapter_sha256"]:
        raise ValueError("original SFT changed")
    training = read(r["training_seed_file"])
    bank = load_start_bank(Path(r["recovery_start_file"]), training)
    for seed in r["training_seeds"]:
        if schedule(seed, training, bank) != r["schedules"][str(seed)]:
            raise ValueError("paired starting schedule changed")
    states = [json.loads(s) for s in Path(r["evaluation"]["recovery_path"]).read_text().splitlines()]
    if [s["seed"] for s in states] != r["evaluation"]["recovery_seeds"] or len({s["seed"] for s in states}) != len(states):
        raise ValueError("recovery cohort differs")
    for s in states:
        game = restore_game(s["seed"], s["action_prefix"], expected=s)
        if s["split"] != "development" or game.game_over:
            raise ValueError("invalid development state")
    return r


def training_command(r, run, registration=REGISTRATION):
    cmd = [sys.executable, "scripts/train_episode_rl.py", "--experiment", "FEEDBACK",
        "--question", r["question"], "--registration-file", str(registration),
        "--adapter-dir", r["initial_adapter"], "--frozen-sft-adapter-dir", r["initial_adapter"],
        "--base-model", r["base_model"], "--base-model-revision", r["base_model_revision"],
        "--benchmark-manifest", r["benchmark_manifest"],
        "--training-seeds-file", r["training_seed_file"], "--recovery-starts", r["recovery_start_file"],
        "--training-seed", str(run["seed"]), "--advantage-method", run["method"],
        "--out-dir", f"runs/{run['run_id']}/rl"]
    for key, value in r["recipe"].items():
        cmd.extend(["--" + key.replace("_", "-"), str(value)])
    # Budget values must be appended by the approved session runner.
    return cmd


def paired_interval(matrix, repetitions=10000, seed=73220):
    """Crossed resampling preserves shared evaluation games across seed pairs."""
    import numpy as np
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("complete finite paired seed-by-game matrix required")
    rng = np.random.default_rng(seed)
    nseed, ngame = values.shape
    draws = np.empty(repetitions)
    for index in range(repetitions):
        seeds = rng.integers(nseed, size=nseed)
        games = rng.integers(ngame, size=ngame)
        draws[index] = values[seeds[:, None], games].mean()
    return {"mean": float(values.mean()), "median_game_difference": float(np.median(values.mean(axis=0))),
            "per_training_seed": values.mean(axis=1).tolist(),
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "seed_pairs": nseed, "source_games": ngame}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "commands"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        r = validate()
        if args.action == "commands":
            print(json.dumps([training_command(r, run) for run in r["run_order"]], indent=2))
        else:
            print("registration, code, original SFT, schedules and development states verified")


if __name__ == "__main__":
    main()
