#!/usr/bin/env python3
"""Stage 6 dense one-board/one-action GRPO using Hugging Face TRL.

Every run is pre-registered into ``runs/<run-id>/rl/manifest.json`` before
model loading.  A non-zero KL coefficient makes TRL copy the loaded SFT
adapter into a frozen ``ref`` adapter; this script hashes that copy before
and after training and refuses a run if it changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tetris.chat import build_generation_prompt
from tetris.engine import Game
from tetris.events import EventWriter
from tetris.rl import (
    DenseRewardWeights,
    atomic_write_json,
    check_resume_registration,
    dense_transition,
    directory_sha256,
    file_sha256,
    parse_completion,
    record_state,
    record_run_failure,
    runtime_budget,
    restore_game,
    state_hash as compute_state_hash,
    validate_seed_manifest,
    validate_entry_gate,
)

ROOT = Path(__file__).resolve().parent.parent
ACTIVE_MANIFEST: Path | None = None


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def build_state_bank(manifest: dict, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    states = []
    weights = DenseRewardWeights(lines=1.0, holes=1.5, aggregate_height=0.08, bumpiness=0.03)
    seeds = list(manifest["training_seeds"])
    rng.shuffle(seeds)
    index = 0
    while len(states) < count:
        game_seed = seeds[index % len(seeds)]
        index += 1
        game = Game(game_seed)
        actions: list[list[int]] = []
        horizon = rng.randint(4, 100)
        for _ in range(horizon):
            if game.game_over:
                break
            legal = [(item["rot"], item["x"]) for item in game.snapshot()["legal"]]
            if rng.random() < 0.2:
                action = rng.choice(legal)
            else:
                action = max(
                    legal,
                    key=lambda item: (dense_transition(game, item, weights).reward, -item[0], -item[1]),
                )
            actions.append([*action])
            game.step(*action)
        if game.game_over:
            continue
        states.append(record_state(game, actions, state_id=f"train-{game_seed}-{game.turn}-{len(states)}"))
    return states


def completion_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        final = value[-1]
        if isinstance(final, dict):
            return str(final.get("content", ""))
    return str(value or "")


def reference_hash(model) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if ".ref." not in name:
            continue
        if parameter.requires_grad:
            raise RuntimeError(f"reference parameter is trainable: {name}")
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
        count += 1
    if not count:
        raise RuntimeError("TRL did not create the frozen ref adapter; use trl==0.29.1 and beta > 0")
    return digest.hexdigest()


def save_observations(path: Path, observed: dict) -> None:
    atomic_write_json(path, {**observed, "unique_actions": sorted(observed["unique_actions"])})


def load_observations(path: Path) -> dict:
    observed = json.loads(path.read_text())
    observed["unique_actions"] = {tuple(action) for action in observed["unique_actions"]}
    return observed


def restore_policy_for_resume(model, checkpoint: Path) -> None:
    """Restore root/default LoRA even when a ref/ adapter subdirectory exists.

    Transformers 5.16's PEFT resume branch loads only adapter subdirectories
    when any exist, skipping the default adapter stored at the root. Run this
    AFTER GRPO creates its SFT reference, before Trainer restores other state.
    """
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    frozen_before = reference_hash(model)
    state = load_file(str(checkpoint / "adapter_model.safetensors"))
    set_peft_model_state_dict(model, state, adapter_name="default")
    model.set_adapter("default")
    if reference_hash(model) != frozen_before:
        raise RuntimeError("restoring the policy adapter changed the frozen reference")


def main() -> None:
    global ACTIVE_MANIFEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E1", "E2", "E3", "E4"), required=True)
    parser.add_argument("--question", required=True, help="registered research question tested by this run")
    parser.add_argument("--initialization-kind", choices=("sft", "weakened"), required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--frozen-sft-adapter-dir", type=Path, default=Path("runs/sft-v1/adapter"))
    parser.add_argument("--benchmark-manifest", type=Path, default=Path("benchmarks/stress-v1/manifest.json"))
    parser.add_argument("--stage5-manifest", type=Path, default=Path("runs/sft-v1/closed_loop/manifest.json"))
    parser.add_argument("--out-dir", type=Path, required=True, help="normally runs/<run-id>/rl")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-updates", type=int, default=50)
    parser.add_argument("--states", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-completion-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lines-weight", type=float, default=1.0)
    parser.add_argument("--holes-weight", type=float, default=1.0)
    parser.add_argument("--height-weight", type=float, default=0.05)
    parser.add_argument("--bumpiness-weight", type=float, default=0.02)
    parser.add_argument("--illegal-weight", type=float, default=10.0)
    parser.add_argument("--pilot-dollar-limit", type=float, default=20.0)
    parser.add_argument("--stage-dollar-limit", type=float, default=100.0)
    parser.add_argument("--prior-stage-spend-usd", type=float, default=0.0)
    parser.add_argument("--instance-hourly-usd", type=float, required=True)
    parser.add_argument("--max-wall-clock-hours", type=float, required=True)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--resume", nargs="?", const=True, default=False)
    parser.add_argument("--pause-after-update", type=int, help="checkpoint and pause without changing the registered scheduler horizon")
    args = parser.parse_args()
    run_started = time.time()
    manifest_path = args.out_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and not args.resume:
        raise SystemExit(f"run already registered in {manifest_path}; use --resume or a new run directory")
    if args.resume and not previous:
        raise SystemExit("resume requires this run's existing registration manifest")
    if isinstance(args.resume, str) and Path(args.resume).resolve().parent != args.out_dir.resolve():
        raise SystemExit("resume checkpoint must belong to the registered run directory")

    if args.experiment == "E1" and args.max_updates > 50:
        raise SystemExit("E1 is capped at 50 updates")
    if args.pause_after_update is not None and not 0 < args.pause_after_update < args.max_updates:
        raise SystemExit("pause update must be positive and below the registered update count")
    if args.experiment == "E2" and args.initialization_kind != "weakened":
        raise SystemExit("E2 must use a reproducible weakened adapter")
    if args.experiment in {"E3", "E4"} and args.initialization_kind != "sft":
        raise SystemExit(f"{args.experiment} must initialize from the frozen SFT adapter")
    if args.kl_beta <= 0:
        raise SystemExit("--kl-beta must be non-zero so the frozen SFT reference is active")
    if min(args.max_updates, args.states, args.batch_size, args.grad_accum) < 1 or args.group_size < 2:
        raise SystemExit("updates, states, and batch sizes must be positive; GRPO requires at least two generations")
    if args.batch_size % args.group_size:
        raise SystemExit("--batch-size must be divisible by --group-size for GRPO")
    if min(args.pilot_dollar_limit, args.stage_dollar_limit, args.max_wall_clock_hours, args.temperature) <= 0 or min(args.instance_hourly_usd, args.prior_stage_spend_usd) < 0:
        raise SystemExit("budgets, wall-clock limit, and temperature must be positive; spend/rate cannot be negative")
    if args.prior_stage_spend_usd + args.pilot_dollar_limit > args.stage_dollar_limit:
        raise SystemExit("registered pilot plus prior spend exceeds the full Stage 6 budget")
    projected_limit = args.instance_hourly_usd * args.max_wall_clock_hours
    if projected_limit > args.pilot_dollar_limit:
        raise SystemExit(f"registered wall-clock limit projects to ${projected_limit:.2f}, over pilot limit")

    benchmark = json.loads(args.benchmark_manifest.read_text())
    seed_validation = validate_seed_manifest(benchmark)
    entry_gate = validate_entry_gate(args.stage5_manifest, benchmark["stage5_seeds"])
    adapter_hash = directory_sha256(args.adapter_dir)
    frozen_sft_hash = directory_sha256(args.frozen_sft_adapter_dir)
    if args.initialization_kind == "sft" and adapter_hash != frozen_sft_hash:
        raise SystemExit("main experiments must initialize from the exact frozen SFT adapter")
    if args.initialization_kind == "weakened" and adapter_hash == frozen_sft_hash:
        raise SystemExit("the weakened-policy smoke must use a different, reproducibly trained adapter")
    reward_weights = DenseRewardWeights(
        lines=args.lines_weight,
        holes=args.holes_weight,
        aggregate_height=args.height_weight,
        bumpiness=args.bumpiness_weight,
        illegal=args.illegal_weight,
    )
    state_bank_path = args.out_dir / "state_bank.jsonl"
    if previous:
        if not state_bank_path.exists():
            raise SystemExit("cannot resume without the registered state bank")
        state_bank = [json.loads(line) for line in state_bank_path.read_text().splitlines() if line.strip()]
    else:
        state_bank = build_state_bank(benchmark, args.states, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.out_dir.parent.name if args.out_dir.name == "rl" else args.out_dir.name
    events = EventWriter(args.out_dir / "events.jsonl", run_id=run_id, stage=6)
    if not previous:
        with state_bank_path.open("w") as handle:
            for row in state_bank:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    registered = {
        "run_id": run_id,
        "stage": 6,
        "kind": "dense_grpo",
        "status": "registered",
        "experiment": args.experiment,
        "research_question": args.question,
        "initialization_kind": args.initialization_kind,
        "parent_run_ids": [args.adapter_dir.parent.name],
        "git_sha": previous.get("git_sha") if previous else git_sha(),
        "host": socket.gethostname(),
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir),
        "adapter_sha256": adapter_hash,
        "frozen_sft_adapter_sha256": frozen_sft_hash,
        "benchmark_manifest": str(args.benchmark_manifest),
        "benchmark_manifest_sha256": file_sha256(args.benchmark_manifest),
        "seed_validation": seed_validation,
        "stage5_entry_gate": entry_gate,
        "training_seed": args.seed,
        "algorithm": "TRL GRPO",
        "reward": {"formula": "lines - holes - height - bumpiness - illegal", "weights": asdict(reward_weights)},
        "kl_beta": args.kl_beta,
        "temperature": args.temperature,
        "sampling": {"temperature": args.temperature, "top_p": 1.0, "top_k": 0},
        "group_size": args.group_size,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "learning_rate": args.learning_rate,
        "max_updates": args.max_updates,
        "max_completion_length": args.max_completion_length,
        "state_bank": str(state_bank_path),
        "state_bank_sha256": file_sha256(state_bank_path),
        "num_states": len(state_bank),
        "budgets_usd": {"pilot": args.pilot_dollar_limit, "stage": args.stage_dollar_limit},
        "prior_stage_spend_usd": args.prior_stage_spend_usd,
        "instance_hourly_usd": args.instance_hourly_usd,
        "max_wall_clock_hours": args.max_wall_clock_hours,
        "projected_cost_limit_usd": projected_limit,
        "registered_at": previous["registered_at"] if previous else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if previous:
        check_resume_registration(previous, registered, [
            "experiment", "research_question", "initialization_kind", "base_model", "adapter_sha256", "frozen_sft_adapter_sha256", "stage5_entry_gate",
            "benchmark_manifest_sha256", "training_seed", "reward", "kl_beta", "temperature", "group_size",
            "batch_size", "gradient_accumulation", "learning_rate", "max_updates", "max_completion_length",
            "state_bank_sha256", "num_states", "budgets_usd", "prior_stage_spend_usd", "instance_hourly_usd", "max_wall_clock_hours",
        ])
    prior_elapsed = float(previous.get("wall_clock_seconds", 0.0)) if previous else 0.0
    budget_stopped = False
    paused = False
    latest_projection = None
    atomic_write_json(manifest_path, registered)
    ACTIVE_MANIFEST = manifest_path
    events.emit("job_started", phase="dense_grpo_initializing", current=0, total=args.max_updates)

    # Deferred heavy imports keep reward and manifest unit tests CPU-only.
    import datasets
    import peft
    import torch
    import transformers
    import trl
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    revision = previous.get("base_model_revision") if previous else None
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, revision=revision)
    model = PeftModel.from_pretrained(base, str(args.adapter_dir), is_trainable=True)
    registered["base_model_revision"] = getattr(base.config, "_commit_hash", None)
    atomic_write_json(manifest_path, registered)
    rows = [
        {
            "prompt": build_generation_prompt(tokenizer, row["prompt"]),
            "seed": row["seed"],
            "action_prefix": row["action_prefix"],
            "state_hash": row["state_hash"],
        }
        for row in state_bank
    ]
    dataset = datasets.Dataset.from_list(rows)
    observed = {"rewards": [], "unique_actions": set(), "parsed": 0, "legal": 0, "completions": 0, "tokens": 0, "group_unique_counts": [], "components": {}}
    resume_path = args.resume
    resume_step = 0
    if args.resume:
        resume_path = get_last_checkpoint(str(args.out_dir)) if args.resume is True else args.resume
        if not resume_path:
            raise RuntimeError("no checkpoint available for resume")
        checkpoint_path = Path(resume_path)
        observed = load_observations(checkpoint_path / "rl_observations.json")
        resume_step = json.loads((checkpoint_path / "trainer_state.json").read_text())["global_step"]
        if args.pause_after_update is not None and args.pause_after_update <= resume_step:
            raise RuntimeError("pause update must be later than the resumed checkpoint")

    def reward_func(completions, seed, action_prefix, state_hash: list[str], completion_ids=None, log_metric=None, log_extra=None, **kwargs):
        rewards = []
        component_rows = []
        if completion_ids is not None:
            observed["tokens"] += sum(len(row) for row in completion_ids)
        for completion, item_seed, prefix, expected_hash in zip(completions, seed, action_prefix, state_hash):
            game = restore_game(int(item_seed), prefix)
            if compute_state_hash(game.snapshot()) != expected_hash:
                raise RuntimeError("training state reconstruction drift")
            text = completion_text(completion)
            action = parse_completion(text)
            transition = dense_transition(game, action, reward_weights)
            legal = not transition.components["illegal"]
            rewards.append(transition.reward)
            component_rows.append(transition.components)
            for key, value in transition.components.items():
                observed["components"].setdefault(key, []).append(value)
            observed["rewards"].append(transition.reward)
            observed["completions"] += 1
            observed["parsed"] += action is not None
            observed["legal"] += legal
            if action is not None:
                observed["unique_actions"].add(action)
        for offset in range(0, len(completions), args.group_size):
            group = completions[offset : offset + args.group_size]
            observed["group_unique_counts"].append(len({parse_completion(completion_text(item)) for item in group}))
        if log_extra and component_rows:
            for key in component_rows[0]:
                log_extra(f"reward/{key}", [row[key] for row in component_rows])
        if log_metric and component_rows:
            for key in component_rows[0]:
                log_metric(f"reward/{key}", sum(row[key] for row in component_rows) / len(component_rows))
            log_metric("reward/parse_rate", sum(parse_completion(completion_text(value)) is not None for value in completions) / len(completions))
            log_metric("reward/legality_rate", sum(not row["illegal"] for row in component_rows) / len(component_rows))
        return rewards

    class EventsCallback(TrainerCallback):
        def on_train_begin(self, training_args, state, control, **kwargs):
            self.started_at = time.time()
            self.initial_step = state.global_step
            if state.global_step != resume_step:
                raise RuntimeError("trainer did not restore the expected global step")
            if args.resume:
                events.emit("job_resumed", phase="dense_grpo", current=state.global_step, total=args.max_updates,
                            metrics={"restored_completions": observed["completions"], "restored_tokens": observed["tokens"]})

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            events.emit("train_metrics", phase="dense_grpo", current=state.global_step, total=args.max_updates, metrics=logs or {})
            elapsed = prior_elapsed + time.time() - run_started
            atomic_write_json(manifest_path, {**registered, "status": "running", "wall_clock_seconds": elapsed, "estimated_cost_usd": elapsed / 3600 * args.instance_hourly_usd})

        def on_step_end(self, training_args, state, control, **kwargs):
            nonlocal budget_stopped, paused, latest_projection
            elapsed = prior_elapsed + time.time() - run_started
            measured_updates = state.global_step - self.initial_step
            rate = (time.time() - self.started_at) / measured_updates if measured_updates >= 3 else 0.0
            latest_projection = runtime_budget(
                elapsed_seconds=elapsed, hourly_usd=args.instance_hourly_usd,
                max_hours=args.max_wall_clock_hours, dollar_limit=args.pilot_dollar_limit,
                remaining_updates=max(0, args.max_updates - state.global_step), seconds_per_update=rate,
            )
            if latest_projection["stop"]:
                budget_stopped = True
                control.should_training_stop = True
                control.should_save = True
                events.emit("budget_stop", phase="dense_grpo", current=state.global_step,
                            total=args.max_updates, metrics=latest_projection)
            elif args.pause_after_update == state.global_step:
                paused = True
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_save(self, training_args, state, control, **kwargs):
            save_observations(args.out_dir / f"checkpoint-{state.global_step}" / "rl_observations.json", observed)
            events.emit("checkpoint_saved", phase="dense_grpo", current=state.global_step, total=args.max_updates, checkpoint=f"checkpoint-{state.global_step}")

    config = GRPOConfig(
        output_dir=str(args.out_dir),
        learning_rate=args.learning_rate,
        max_steps=args.max_updates,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.group_size,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=1.0,
        top_k=0,
        beta=args.kl_beta,
        loss_type="grpo",
        scale_rewards="group",
        num_iterations=1,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        log_completions=True,
    )
    registered["trainer_config"] = {key: value for key, value in config.to_dict().items() if key not in {"hub_token", "push_to_hub_token"}}
    registered["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "trl": trl.__version__, "datasets": datasets.__version__}
    atomic_write_json(manifest_path, registered)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[EventsCallback()],
    )
    ref_hash_before = reference_hash(trainer.model)
    if args.resume:
        restore_policy_for_resume(trainer.model, Path(resume_path))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    output = trainer.train(resume_from_checkpoint=resume_path)
    elapsed = prior_elapsed + time.time() - run_started
    ref_hash_after = reference_hash(trainer.model)
    if ref_hash_before != ref_hash_after:
        raise RuntimeError("frozen reference adapter changed during training")
    adapter_out = args.out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_out), selected_adapters=["default"])
    tokenizer.save_pretrained(str(adapter_out))
    actual_cost = elapsed / 3600 * args.instance_hourly_usd
    if budget_stopped or elapsed > args.max_wall_clock_hours * 3600 or actual_cost > args.pilot_dollar_limit:
        status = "stopped_budget"
    elif paused:
        status = "paused"
    else:
        status = "completed"
    completed = {
        **registered,
        "status": status,
        "reference_adapter_sha256_before": ref_hash_before,
        "reference_adapter_sha256_after": ref_hash_after,
        "reference_frozen": True,
        "output_adapter_sha256": directory_sha256(adapter_out),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "trl": trl.__version__,
            "datasets": datasets.__version__,
        },
        "wall_clock_seconds": elapsed,
        "estimated_cost_usd": actual_cost,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "system_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "completed_updates": trainer.state.global_step,
        "seconds_per_update": elapsed / max(1, trainer.state.global_step),
        "generated_tokens_per_second": observed["tokens"] / elapsed if elapsed else 0.0,
        "states_per_hour": observed["completions"] / args.group_size / elapsed * 3600 if elapsed else 0.0,
        "resume_used": bool(args.resume),
        "resumed_from_update": resume_step,
        "pause_after_update": args.pause_after_update,
        "runtime_projection": latest_projection,
        "train_metrics": output.metrics,
        "rollout_statistics": {
            "completions": observed["completions"],
            "generated_tokens": observed["tokens"],
            "unique_actions": len(observed["unique_actions"]),
            "mean_unique_actions_per_group": sum(observed["group_unique_counts"]) / len(observed["group_unique_counts"]) if observed["group_unique_counts"] else 0.0,
            "reward_components": {key: {"mean": sum(values) / len(values), "min": min(values), "max": max(values)} for key, values in observed["components"].items()},
            "parse_rate": observed["parsed"] / observed["completions"] if observed["completions"] else 0.0,
            "legality_rate": observed["legal"] / observed["completions"] if observed["completions"] else 0.0,
            "mean_reward": sum(observed["rewards"]) / len(observed["rewards"]) if observed["rewards"] else 0.0,
        },
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(manifest_path, completed)
    events.emit("job_completed" if status == "completed" else "job_stopped", phase="dense_grpo", current=trainer.state.global_step, total=args.max_updates, metrics={"wall_clock_seconds": elapsed, "estimated_cost_usd": actual_cost}, artifacts=[str(adapter_out), str(manifest_path)])
    print(f"saved dense GRPO adapter and manifest to {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        record_run_failure(ACTIVE_MANIFEST, error)
        raise
