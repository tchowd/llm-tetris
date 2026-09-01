#!/usr/bin/env python3
"""Stage 5: closed-loop eval. Random, teacher, and the fine-tuned model each
play real games -- driving `tetris.rollout.run_rollout`, the same batched
harness `tests/test_rollout.py`'s gate tests exercise against random/teacher
-- on one shared, fixed seed list, in both strict and assisted illegal-
action modes. See plan/stage-5-eval.md.

    python scripts/eval_closed_loop.py \
        --adapter-dir runs/sft-v1/adapter --data-dirs data/batch1 data/batch2 \
        --out-dir runs/sft-v1/closed_loop

Add `--policies random,teacher` (no --adapter-dir needed) to run only the
baselines, e.g. to reproduce Stage 2's teacher numbers without a trained
model on hand.

Requires the training extras (torch/transformers/peft) for the model
policy -- see requirements-train.txt -- but random/teacher-only runs need
nothing beyond the base project.
"""
from __future__ import annotations

import argparse
import copy
import json
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from tetris.rollout import (
    ASSISTED,
    STRICT,
    aggregate_metrics,
    default_eval_seeds,
    random_legal_policy,
    run_rollout,
    teacher_policy,
)
from tetris.events import EventWriter
from tetris.events import manifest_hashes
from tetris.teacher import WEIGHTS as LIVE_WEIGHTS


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent, text=True
        ).strip()
    except Exception:
        return None


def resolve_weights(data_dirs: list[Path]) -> dict:
    for data_dir in data_dirs:
        manifest_path = data_dir / "manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text())["teacher_weights"]
    return LIVE_WEIGHTS


def check_seed_disjointness(seeds: list[int], data_dirs: list[Path]) -> None:
    eval_seeds = set(seeds)
    for data_dir in data_dirs:
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        start = manifest["seed_start"]
        stage3_seeds = set(range(start, start + manifest["num_games"]))
        overlap = eval_seeds & stage3_seeds
        if overlap:
            raise SystemExit(f"{manifest_path}: eval seeds overlap Stage 3 seeds: {sorted(overlap)[:5]}")


def build_model_policy(adapter_dir: Path, base_model: str, device: str):
    """Loaded lazily -- only called when "model" is actually in --policies,
    so a random/teacher-only run never needs torch installed."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tetris.chat import build_generation_prompt
    from tetris.serialize import parse_action

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    model.eval()

    def pick(snapshots: list[dict], teacher_infos):
        prompts = [build_generation_prompt(tokenizer, snap["prompt"]) for snap in snapshots]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        results = []
        for text in texts:
            line = text.strip().splitlines()[0] if text.strip() else ""
            try:
                action = parse_action(line)
            except ValueError:
                action = None
            results.append((action, text))
        return results

    return pick


def build_policy(name: str, args, weights: dict):
    if name == "random":
        return random_legal_policy()
    if name == "teacher":
        return teacher_policy(weights)
    if name == "model":
        if args.adapter_dir is None:
            raise SystemExit("--policies includes 'model' but --adapter-dir was not given")
        device = args.device or _default_device()
        return build_model_policy(args.adapter_dir, args.base_model, device)
    raise SystemExit(f"unknown policy: {name}")


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@contextmanager
def event_heartbeat(events: EventWriter, *, phase: str, current: int, total: int):
    """Keep long rollout groups visibly alive between their final metrics."""
    stopped = threading.Event()

    def emit() -> None:
        while not stopped.wait(60):
            events.emit(
                "heartbeat",
                phase=phase,
                current=current,
                total=total,
                message="closed-loop rollout is still running",
            )

    thread = threading.Thread(target=emit, name="closed-loop-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=2)


def assisted_copy(records: list[dict], diagnostics: dict[str, list[dict]]) -> tuple[list[dict], dict[str, list[dict]]]:
    """Random and teacher are always legal, so strict and assisted are identical."""
    copied_records = copy.deepcopy(records)
    copied_diagnostics: dict[str, list[dict]] = {}
    for record in copied_records:
        old_game_id = record["game_id"]
        new_game_id = old_game_id.replace("-strict-", "-assisted-")
        record["game_id"] = new_game_id
        record["mode"] = ASSISTED
        copied_diagnostics[new_game_id] = copy.deepcopy(diagnostics[old_game_id])
    return copied_records, copied_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policies", default="random,teacher,model", help="comma-separated subset of random,teacher,model")
    parser.add_argument("--modes", default="strict,assisted", help="comma-separated subset of strict,assisted")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data-dirs", nargs="*", type=Path, default=[], help="Stage 3 dirs, for teacher weights + seed-disjointness check")
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--seed-offset", type=int, default=None, help="default: tetris.rollout.EVAL_SEED_OFFSET")
    parser.add_argument("--cap", type=int, default=500)
    parser.add_argument("--gen-batch-size", type=int, default=64)
    parser.add_argument("--teacher-workers", type=int, default=1, help="processes used for per-state teacher scoring")
    parser.add_argument("--device", default=None, help="default: cuda, then mps, then cpu (model policy only)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in (STRICT, ASSISTED):
            raise SystemExit(f"unknown mode: {m}")

    weights = resolve_weights(args.data_dirs)
    seeds = default_eval_seeds(args.num_seeds) if args.seed_offset is None else list(range(args.seed_offset, args.seed_offset + args.num_seeds))
    check_seed_disjointness(seeds, args.data_dirs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    games_path = args.out_dir / "games.jsonl"
    metrics_path = args.out_dir / "metrics.json"
    manifest_path = args.out_dir / "manifest.json"
    parent_run_id = args.adapter_dir.parent.name if args.adapter_dir else None
    base_run_id = args.out_dir.parent.name if args.out_dir.name == "closed_loop" else args.out_dir.name
    run_id = f"{base_run_id}-closed-loop"
    parent_run_ids = [parent_run_id] if parent_run_id else []
    events = EventWriter(args.out_dir / "events.jsonl", run_id=run_id, stage=5, lineage={"parent_run_ids": parent_run_ids})
    total_groups = len(policies) * len(modes)
    events.emit("job_started", phase="closed_loop", current=0, total=total_groups, metrics={"planned_games": total_groups * len(seeds)})

    report: dict = {}
    t0 = time.time()
    completed_groups = 0
    with games_path.open("w") as games_f:
        for policy_name in policies:
            policy_fn = build_policy(policy_name, args, weights)
            report[policy_name] = {}
            strict_legal_result = None
            for mode in modes:
                print(f"[{policy_name}/{mode}] running {len(seeds)} seeds, cap={args.cap} ...", flush=True)
                t_start = time.time()
                if mode == ASSISTED and strict_legal_result is not None:
                    records, diagnostics = assisted_copy(*strict_legal_result)
                    print("  reused strict result because this policy always emits legal actions", flush=True)
                else:
                    with event_heartbeat(
                        events,
                        phase=f"{policy_name}/{mode}",
                        current=completed_groups,
                        total=total_groups,
                    ):
                        records, diagnostics = run_rollout(
                            seeds,
                            policy_fn,
                            mode=mode,
                            cap=args.cap,
                            teacher_weights=weights,
                            gen_batch_size=args.gen_batch_size,
                            game_id_prefix=policy_name,
                            teacher_workers=args.teacher_workers,
                        )
                    if mode == STRICT and policy_name in {"random", "teacher"}:
                        strict_legal_result = (records, diagnostics)
                for rec in records:
                    rec["policy"] = policy_name
                    games_f.write(json.dumps(rec) + "\n")
                metrics = aggregate_metrics(records, diagnostics)
                report[policy_name][mode] = metrics
                completed_groups += 1
                events.emit("eval_metrics", phase=f"{policy_name}/{mode}", current=completed_groups, total=total_groups, metrics={"completed_games": completed_groups * len(seeds), "planned_games": total_groups * len(seeds), "lines_mean": metrics["lines"]["mean"], "deaths": metrics["deaths"], "parse_failure_rate": metrics["parse_failure_rate"]["mean"], "teacher_match_rate": metrics["teacher_match_rate"]["mean"]})
                elapsed = time.time() - t_start
                lines_m = metrics["lines"]
                print(
                    f"  lines: {lines_m['mean']:.1f} +/- {lines_m['se']:.1f} (median {lines_m['median']}, max {lines_m['max']}), "
                    f"deaths: {metrics['deaths']}/{metrics['n_games']} (illegal: {metrics['illegal_action_deaths']}), "
                    f"cap-outs: {metrics['cap_outs']}, "
                    f"teacher_match: {metrics['teacher_match_rate']['mean']:.2%}, "
                    f"parse_fail: {metrics['parse_failure_rate']['mean']:.2%}, "
                    f"{elapsed:.1f}s",
                    flush=True,
                )

    manifest = {
        "run_id": run_id,
        "stage": 5,
        "status": "passed",
        "host": socket.gethostname(),
        "parent_run_ids": parent_run_ids,
        "data_manifest_hashes": manifest_hashes([path / "manifest.json" for path in args.data_dirs]),
        "git_sha": _git_sha(),
        "policies": policies,
        "modes": modes,
        "seeds": seeds,
        "cap": args.cap,
        "gen_batch_size": args.gen_batch_size,
        "teacher_workers": args.teacher_workers,
        "teacher_weights": weights,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "base_model": args.base_model,
        "data_dirs": [str(d) for d in args.data_dirs],
        "wall_clock_seconds": time.time() - t0,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report["_meta"] = {
        "run_id": run_id,
        "stage": 5,
        "status": "passed",
        "git_sha": manifest["git_sha"],
        "host": manifest["host"],
        "parent_run_ids": parent_run_ids,
        "data_manifest_hashes": manifest["data_manifest_hashes"],
        "generated_at": manifest["generated_at"],
    }
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")
    events.emit("job_completed", phase="closed_loop", current=total_groups, total=total_groups, metrics={"completed_games": total_groups * len(seeds), "wall_clock_seconds": time.time() - t0}, artifacts=[str(games_path), str(metrics_path), str(manifest_path)])
    print(f"wrote {games_path}, {metrics_path}, {manifest_path}")


if __name__ == "__main__":
    main()
