#!/usr/bin/env python3
"""Evaluate policies on the frozen Stage 6 stress-v1 benchmark.

Long-horizon games use the unchanged Stage 5 strict rollout harness.  The
fixed recovery/probe states are reconstructed from seed plus action prefix
and evaluated for one action without mutating the recorded source state.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

from tetris.events import EventWriter
from tetris.model_policy import build_model_policy
from tetris.rl import (
    DenseRewardWeights,
    atomic_write_json,
    dense_transition,
    directory_sha256,
    file_sha256,
    restore_game,
    record_run_failure,
    state_hash,
    validate_seed_manifest,
)
from tetris.rollout import STRICT, _teacher_best, aggregate_metrics, random_legal_policy, run_rollout, teacher_policy
from tetris.teacher import WEIGHTS

ROOT = Path(__file__).resolve().parent.parent
ACTIVE_MANIFEST: Path | None = None


def progress_policy(policy, events, *, phase: str, cap: int):
    """Observe progress without changing any prompt, batch, or decision."""
    started = time.monotonic()
    last_report = started
    turns = 0

    def pick(snapshots, teacher_infos):
        nonlocal turns, last_report
        result = policy(snapshots, teacher_infos)
        turns += len(snapshots)
        now = time.monotonic()
        if now - last_report >= 60:
            current = max(snapshot["turn"] for snapshot in snapshots) + 1
            metrics = {"turns_processed": turns, "elapsed_seconds": now - started,
                       "projected_policy_seconds": (now - started) * cap / current}
            events.emit("heartbeat", phase=phase, current=current, total=cap, metrics=metrics)
            print(f"[{phase}] turn {current}/{cap}, {turns} decisions, {now - started:.0f}s", flush=True)
            last_report = now
        return result

    pick.metadata = getattr(policy, "metadata", {})
    return pick


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def read_states(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stage3_ranges(data_dirs: list[Path]) -> list[range]:
    result = []
    for directory in data_dirs:
        path = directory / "manifest.json"
        if path.exists():
            row = json.loads(path.read_text())
            result.append(range(int(row["seed_start"]), int(row["seed_start"]) + int(row["num_games"])))
    return result


def build_policy(name: str, args):
    if name == "random":
        return random_legal_policy()
    if name == "teacher":
        # Every evaluation path already supplies the frozen teacher's scores.
        # Reuse them instead of performing the same two-ply search twice.
        return teacher_policy()
    if name == "model":
        if args.adapter_dir is None:
            raise SystemExit("--policies includes model but --adapter-dir was not supplied")
        return build_model_policy(args.adapter_dir, args.base_model, args.device)
    raise SystemExit(f"unknown policy: {name}")


def survival_and_quality(records: list[dict], diagnostics: dict[str, list[dict]], cap: int) -> dict:
    checkpoints = sorted(set([100, 500, 1000, 2000, cap]))
    checkpoints = [turn for turn in checkpoints if turn <= cap]
    rows = []
    for turn in checkpoints:
        survivors = [record for record in records if record["pieces"] >= turn]
        quality = []
        for record in survivors:
            turns = diagnostics[record["game_id"]]
            if len(turns) >= turn:
                quality.append(turns[turn - 1])

        def mean(key: str) -> float | None:
            return sum(row[key] for row in quality) / len(quality) if quality else None

        rows.append(
            {
                "turn": turn,
                "survived": len(survivors),
                "survival_rate": len(survivors) / len(records) if records else 0.0,
                "holes": mean("holes_after"),
                "aggregate_height": mean("aggregate_height"),
                "bumpiness": mean("bumpiness"),
                "well_depth": mean("well_depth"),
            }
        )
    return {"checkpoints": rows}


def evaluate_states(policy_fn, states: list[dict], batch_size: int) -> tuple[list[dict], dict]:
    results = []
    weights = DenseRewardWeights()
    for start in range(0, len(states), batch_size):
        chunk = states[start : start + batch_size]
        games = [restore_game(row["seed"], row["action_prefix"], expected=row) for row in chunk]
        snapshots = [game.snapshot() for game in games]
        for row, snap in zip(chunk, snapshots):
            if state_hash(snap) != row["state_hash"]:
                raise ValueError(f"state hash drift for {row['state_id']}")
        teacher_infos = [_teacher_best(snap, WEIGHTS) for snap in snapshots]
        outputs = policy_fn(snapshots, teacher_infos)
        for row, game, snap, teacher_info, (action, raw_text) in zip(chunk, games, snapshots, teacher_infos, outputs):
            transition = dense_transition(game, action, weights)
            legal = action is not None and action in {(item["rot"], item["x"]) for item in snap["legal"]}
            results.append(
                {
                    "state_id": row["state_id"],
                    "kind": row["kind"],
                    "seed": row["seed"],
                    "turn": row["turn"],
                    "state_hash": row["state_hash"],
                    "raw_output": raw_text,
                    "action": list(action) if action is not None else None,
                    "parsed": action is not None,
                    "legal": legal,
                    "teacher_match": action == teacher_info[0],
                    "dense_reward": transition.reward,
                    "reward_components": transition.components,
                    "terminal": transition.terminal,
                    "terminal_reason": transition.terminal_reason,
                }
            )
    summary = {}
    for kind in ("recovery", "probe"):
        rows = [row for row in results if row["kind"] == kind]
        n = len(rows)
        summary[kind] = {
            "n": n,
            "parse_rate": sum(row["parsed"] for row in rows) / n if n else 0.0,
            "legality_rate": sum(row["legal"] for row in rows) / n if n else 0.0,
            "teacher_match_rate": sum(row["teacher_match"] for row in rows) / n if n else 0.0,
            "mean_dense_reward": sum(row["dense_reward"] for row in rows) / n if n else 0.0,
        }
    return results, summary


def run_recovery_rollouts(policy_fn, states: list[dict], *, cap: int, batch_size: int) -> tuple[list[dict], dict]:
    """Strict lockstep continuation from difficult, exactly replayed boards."""

    games = [restore_game(row["seed"], row["action_prefix"], expected=row) for row in states]
    records = []
    diagnostics = {}
    for game, state in zip(games, states):
        game.game_id = state["state_id"]
        records.append({
            "game_id": game.game_id, "seed": game.seed, "mode": STRICT,
            "starting_state": state, "actions": [], "labels": [], "raw_actions": [],
            "incidents": [], "raw_model_output": [], "death_reason": None,
        })
        diagnostics[game.game_id] = []
    active = list(range(len(games)))
    while active:
        for offset in range(0, len(active), batch_size):
            indexes = active[offset : offset + batch_size]
            snapshots = [games[index].snapshot() for index in indexes]
            infos = [_teacher_best(snap, WEIGHTS) for snap in snapshots]
            outputs = policy_fn(snapshots, infos)
            for index, before, (teacher_action, values), (action, text) in zip(indexes, snapshots, infos, outputs):
                game, record = games[index], records[index]
                record["raw_model_output"].append(text)
                legal = action is not None and action in values
                if not legal:
                    record["death_reason"] = "illegal_action"
                    record["terminal_incident"] = {"parsed": action is not None, "legal": False}
                    continue
                after = game.step(*action)
                record["actions"].append(list(action))
                record["raw_actions"].append(list(action))
                record["labels"].append(list(teacher_action))
                diagnostics[game.game_id].append({
                    "turn": len(record["actions"]) - 1,
                    "parsed": True, "legal": True, "teacher_match": action == teacher_action,
                    "value_gap": values[action] - values[teacher_action],
                    "holes_before": before["holes_total"], "holes_after": after["holes_total"],
                    "holes_created": max(0, after["holes_total"] - before["holes_total"]),
                    "max_height": after["max_height"], "aggregate_height": after["aggregate_height"],
                    "bumpiness": after["bumpiness"], "well_depth": sum(after["wells"]),
                    "lines_after": after["lines"] - record["starting_state"]["lines"],
                    "score_after": after["score"] - record["starting_state"]["score"],
                })
                if game.game_over:
                    record["death_reason"] = "topped_out"
                elif len(record["actions"]) >= cap:
                    record["death_reason"] = "cap_reached"
        active = [index for index in active if records[index]["death_reason"] is None]
    for game, record in zip(games, records):
        record.update({
            "pieces": len(record["actions"]),
            "lines": game.lines - record["starting_state"]["lines"],
            "score": game.score - record["starting_state"]["score"],
            "died": record["death_reason"] in {"illegal_action", "topped_out"},
        })
    return records, aggregate_metrics(records, diagnostics)


def main() -> None:
    global ACTIVE_MANIFEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/stress-v1/manifest.json"))
    parser.add_argument("--states", type=Path, default=Path("benchmarks/stress-v1/states.jsonl"))
    parser.add_argument("--suite", choices=("development", "test"), required=True)
    parser.add_argument("--policies", default="random,teacher,model")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--policy-label", default=None, help="output label for the model policy, e.g. sft or rl-seed-1")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--gen-batch-size", type=int, default=64)
    parser.add_argument("--teacher-workers", type=int, default=1)
    parser.add_argument("--cap", type=int, default=None, help="smoke override; registered cap is used by default")
    parser.add_argument("--recovery-cap", type=int, default=None, help="smoke override; defaults to the registered recovery horizon")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    benchmark = json.loads(args.manifest.read_text())
    validation = validate_seed_manifest(benchmark, stage3_ranges=stage3_ranges(args.data_dirs))
    seeds = benchmark[f"{args.suite}_seeds"]
    registered_cap = benchmark["long_horizon"][f"{args.suite}_cap"]
    cap = args.cap or registered_cap
    recovery_cap = args.recovery_cap or benchmark.get("recovery_cap", 200)
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    if args.policy_label and "model" not in policies:
        raise SystemExit("--policy-label requires model in --policies")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "manifest.json").exists() or (args.out_dir / "games.jsonl").exists():
        raise SystemExit(f"refusing to overwrite evaluation artifacts in {args.out_dir}")
    run_id = args.out_dir.parent.parent.name if args.out_dir.parent.name == "rl" else args.out_dir.name
    events = EventWriter(args.out_dir / "events.jsonl", run_id=run_id, stage=6)
    ACTIVE_MANIFEST = args.out_dir / "manifest.json"
    atomic_write_json(ACTIVE_MANIFEST, {
        "run_id": run_id, "stage": 6, "kind": "stress_evaluation", "status": "running",
        "suite": args.suite, "seeds": seeds, "cap": cap, "recovery_cap": recovery_cap,
        "benchmark_manifest_sha256": file_sha256(args.manifest),
        "states_sha256": file_sha256(args.states),
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    events.emit("job_started", phase=f"stress-{args.suite}", current=0, total=len(policies))
    fixed_states = [row for row in read_states(args.states) if row.get("split") == args.suite]
    if not fixed_states:
        raise SystemExit(f"no fixed states registered for {args.suite}")
    metrics = {}
    policy_metadata = {}
    recovery_records_all = []
    t0 = time.time()
    with (args.out_dir / "games.jsonl").open("w") as games_handle, (args.out_dir / "states.jsonl").open("w") as state_handle:
        for index, policy_name in enumerate(policies, 1):
            label = args.policy_label if policy_name == "model" and args.policy_label else policy_name
            policy = build_policy(policy_name, args)
            policy_metadata[label] = getattr(policy, "metadata", {})
            rollout_policy = progress_policy(policy, events, phase=f"stress-{args.suite}/{label}", cap=cap)
            print(f"[{label}] running {len(seeds)} seeds at cap={cap}", flush=True)
            started = time.time()
            records, diagnostics = run_rollout(
                seeds,
                rollout_policy,
                STRICT,
                cap=cap,
                gen_batch_size=args.gen_batch_size,
                teacher_workers=args.teacher_workers,
                game_id_prefix=label,
            )
            for record in records:
                record["policy"] = label
                games_handle.write(json.dumps(record) + "\n")
            state_results, state_summary = evaluate_states(policy, fixed_states, args.gen_batch_size)
            for result in state_results:
                result["policy"] = label
                state_handle.write(json.dumps(result) + "\n")
            recovery_records, recovery_metrics = run_recovery_rollouts(
                policy, [row for row in fixed_states if row["kind"] == "recovery"],
                cap=recovery_cap, batch_size=args.gen_batch_size,
            )
            for record in recovery_records:
                record["policy"] = label
            recovery_records_all.extend(recovery_records)
            aggregate = aggregate_metrics(records, diagnostics)
            metrics[label] = {
                "long_horizon": aggregate,
                "survival_and_board_quality": survival_and_quality(records, diagnostics, cap),
                "fixed_states": state_summary,
                "recovery_rollouts": recovery_metrics,
                "wall_clock_seconds": time.time() - started,
            }
            events.emit(
                "eval_metrics",
                phase=f"stress-{args.suite}/{label}",
                current=index,
                total=len(policies),
                metrics={
                    "score_per_100_pieces": aggregate["score_per_100_pieces"]["mean"],
                    "lines_per_100_pieces": aggregate["lines_per_100_pieces"]["mean"],
                    "deaths": aggregate["deaths"],
                    "parse_failure_rate": aggregate["parse_failure_rate"]["mean"],
                    "illegal_rate": aggregate["illegal_rate"]["mean"],
                },
            )

    manifest = {
        "run_id": run_id,
        "stage": 6,
        "kind": "stress_evaluation",
        "status": "passed",
        "suite": args.suite,
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_manifest": str(args.manifest),
        "benchmark_manifest_sha256": file_sha256(args.manifest),
        "states_sha256": file_sha256(args.states),
        "seed_validation": validation,
        "seeds": seeds,
        "registered_cap": registered_cap,
        "cap": cap,
        "recovery_cap": recovery_cap,
        "greedy": True,
        "mode": STRICT,
        "policies": [args.policy_label if item == "model" and args.policy_label else item for item in policies],
        "policy_metadata": policy_metadata,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "adapter_sha256": directory_sha256(args.adapter_dir) if args.adapter_dir else None,
        "base_model": args.base_model,
        "git_sha": git_sha(),
        "host": socket.gethostname(),
        "wall_clock_seconds": time.time() - t0,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.adapter_dir and (args.adapter_dir.parent / "manifest.json").exists():
        parent_path = args.adapter_dir.parent / "manifest.json"
        parent_manifest = json.loads(parent_path.read_text())
        manifest["parent_training_manifest_sha256"] = file_sha256(parent_path)
        manifest["training_seed"] = parent_manifest.get("training_seed")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.out_dir / "recovery_games.jsonl").open("w") as handle:
        for record in recovery_records_all:
            handle.write(json.dumps(record) + "\n")
    events.emit("job_completed", phase=f"stress-{args.suite}", current=len(policies), total=len(policies), metrics={"wall_clock_seconds": time.time() - t0})
    print(f"wrote stress evaluation to {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        record_run_failure(ACTIVE_MANIFEST, error)
        raise
