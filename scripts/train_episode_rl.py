#!/usr/bin/env python3
"""Grouped multi-turn Stage 6 trainer with explicit reward-to-go.

Stock TRL GRPO is used only for one-board/one-action training.  This small
custom layer samples ``G`` trajectories from the same engine start, records
the exact completion tokens and policy/reference log-probabilities, assigns
discounted reward-to-go at each turn, and applies a selectable feedback estimator.
The default preserves normalization across the same-start active group.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
from tetris.recovery import load_start_bank, make_group_starts
from tetris.rl import (
    EpisodeRewardWeights,
    advantage_diagnostics,
    atomic_write_json,
    check_resume_registration,
    directory_sha256,
    discounted_reward_to_go,
    episode_transition,
    file_sha256,
    grouped_policy_loss,
    parse_completion,
    record_run_failure,
    runtime_budget,
    state_hash,
    trajectory_advantages,
    validate_seed_manifest,
    validate_entry_gate,
    validate_trajectory,
)

ROOT = Path(__file__).resolve().parent.parent
ACTIVE_MANIFEST: Path | None = None


def configure_execution():
    """Fail closed on nondeterministic kernels; configure before any CUDA work."""
    import torch

    workspace = ":4096:8"
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG", workspace) != workspace:
        raise ValueError("episode execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    return {"deterministic_algorithms": True, "cublas_workspace_config": workspace,
            "float32_matmul_precision": "highest", "qwen3_completion_logits_suffix": True,
            "episode_normalization": "decimal50"}


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def trim_completion(ids: list[int], *, eos_id: int | None, pad_id: int | None) -> list[int]:
    result = []
    for token in ids:
        if eos_id is not None and token == eos_id:
            result.append(int(token))
            break
        if pad_id is not None and token == pad_id:
            break
        result.append(int(token))
    return result


def sequence_logprobs(model, rows: list[dict], pad_token_id: int, temperature: float = 1.0, *, return_tokens: bool = False):
    """Differentiable mean completion log-prob for exact stored token IDs."""

    import torch

    if not rows:
        return torch.empty(0, device=next(model.parameters()).device)
    if any(not row["prompt_ids"] or not row["completion_ids"] for row in rows):
        raise ValueError("log-probability alignment requires non-empty prompt and completion token IDs")
    device = next(model.parameters()).device
    sequences = [row["prompt_ids"] + row["completion_ids"] for row in rows]
    max_length = max(len(row) for row in sequences)
    input_ids = torch.full((len(rows), max_length), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros((len(rows), max_length), dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention[index, : len(sequence)] = 1
    # Qwen supports computing only a suffix of vocabulary logits. Keep every
    # completion position, with the identical right-padded causal input.
    first_position = min(len(row["prompt_ids"]) - 1 for row in rows)
    offset = first_position if getattr(getattr(model, "config", None), "model_type", None) == "qwen3" else 0
    options = {"logits_to_keep": max_length - offset, "use_cache": False} if offset else {}
    logits = model(input_ids=input_ids, attention_mask=attention, **options).logits
    results = []
    for index, row in enumerate(rows):
        prompt_len = len(row["prompt_ids"])
        completion = torch.tensor(row["completion_ids"], dtype=torch.long, device=device)
        # Normalize only the completion positions, not the entire long board
        # prompt. Float32 is important for small policy/reference KL deltas;
        # the model itself remains bf16 on the GPU.
        token_logits = logits[index, prompt_len - 1 - offset : prompt_len - 1 - offset + len(completion)].float()
        token_logps = torch.log_softmax(token_logits / temperature, dim=-1)
        results.append(token_logps.gather(1, completion.unsqueeze(1)).squeeze(1))
    return results if return_tokens else torch.stack([row.mean() for row in results])


def sample_group(policy, reference, tokenizer, *, seed: int, group_size: int, horizon: int, temperature: float, reward_weights: EpisodeRewardWeights, start: dict | None = None) -> list[dict]:
    import torch

    games = make_group_starts(seed, group_size, start)
    trajectories = [
        {
            "episode_id": game.game_id,
            "seed": seed,
            "start_actions": start["action_prefix"] if start else [],
            "start_state": game.snapshot(),
            "steps": [],
        }
        for game in games
    ]
    active = list(range(group_size))
    policy.eval()
    reference.eval()
    for _ in range(horizon):
        if not active:
            break
        snapshots = [games[index].snapshot() for index in active]
        prompts = [build_generation_prompt(tokenizer, snap["prompt"]) for snap in snapshots]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        encoded = {key: value.to(next(policy.parameters()).device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = policy.generate(
                **encoded,
                max_new_tokens=16,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw_ids = generated[:, encoded["input_ids"].shape[1] :].detach().cpu().tolist()
        completion_ids = [
            trim_completion(row, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id) for row in raw_ids
        ]
        texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        logprob_rows = [
            {
                "prompt_ids": tokenizer(prompt, add_special_tokens=False)["input_ids"],
                "completion_ids": completion,
            }
            for prompt, completion in zip(prompts, completion_ids)
        ]
        with torch.no_grad():
            policy_tokens = sequence_logprobs(policy, logprob_rows, tokenizer.pad_token_id, temperature, return_tokens=True)
            reference_tokens = sequence_logprobs(reference, logprob_rows, tokenizer.pad_token_id, temperature, return_tokens=True)
            policy_logps = [float(row.mean().cpu()) for row in policy_tokens]
            reference_logps = [float(row.mean().cpu()) for row in reference_tokens]

        finished = []
        for batch_index, trajectory_index in enumerate(active):
            game = games[trajectory_index]
            before = game.snapshot()
            action = parse_completion(texts[batch_index])
            transition = episode_transition(game, action, reward_weights)
            legal = action is not None and action in {(p["rot"], p["x"]) for p in before["legal"]}
            after_hash = None
            if legal:
                after = game.step(*action)
                after_hash = state_hash(after)
            current_snapshot = game.snapshot()
            step = {
                "turn": before["turn"],
                "serialized_prompt": before["prompt"],
                "prompt": prompts[batch_index],
                "prompt_ids": logprob_rows[batch_index]["prompt_ids"],
                "completion_ids": completion_ids[batch_index],
                "raw_completion": texts[batch_index],
                "action": list(action) if action is not None else None,
                "parsed": action is not None,
                "legal": legal,
                "policy_logprob_at_sampling": policy_logps[batch_index],
                "reference_logprob": reference_logps[batch_index],
                "policy_token_logprobs_at_sampling": policy_tokens[batch_index].cpu().tolist(),
                "reference_token_logprobs": reference_tokens[batch_index].cpu().tolist(),
                "immediate_reward": transition.reward,
                "reward_components": transition.components,
                "before_state_hash": state_hash(before),
                "after_state_hash": after_hash,
                "terminal": transition.terminal,
                "terminal_reason": transition.terminal_reason,
                "score_after": game.score,
                "lines_after": game.lines,
                "board_quality": {
                    key: current_snapshot[key]
                    for key in ("holes_total", "aggregate_height", "bumpiness", "max_height", "wells")
                },
            }
            trajectories[trajectory_index]["steps"].append(step)
            if transition.terminal:
                finished.append(trajectory_index)
        active = [index for index in active if index not in finished]

    reward_groups = [[step["immediate_reward"] for step in trajectory["steps"]] for trajectory in trajectories]
    advantages = trajectory_advantages(reward_groups, gamma=0.99)
    for trajectory, advantage_row, game in zip(trajectories, advantages, games):
        trajectory["episode_return"] = sum(step["immediate_reward"] for step in trajectory["steps"])
        trajectory["pieces"] = len([step for step in trajectory["steps"] if step["legal"]])
        trajectory["score"] = game.score - trajectory["start_state"]["score"]
        trajectory["lines"] = game.lines - trajectory["start_state"]["lines"]
        trajectory["terminal_reason"] = trajectory["steps"][-1]["terminal_reason"] or "cap_reached"
        for step, advantage in zip(trajectory["steps"], advantage_row):
            step["advantage"] = advantage
        validation = validate_trajectory(trajectory)
        trajectory["replay_validation"] = validation
    return trajectories


def flatten_steps(trajectories: list[dict]) -> list[dict]:
    return [step for trajectory in trajectories for step in trajectory["steps"]]


def action_loss_chunk(token_rows, steps, beta, total_decisions):
    """Token mean per action, decision mean across the complete update."""
    import torch
    losses, kls = [], []
    for row, current in zip(steps, token_rows, strict=True):
        reference = torch.tensor(row["reference_token_logprobs"], dtype=current.dtype, device=current.device)
        if current.shape != reference.shape:
            raise ValueError("reference tokens do not align")
        advantages = torch.full_like(current, row["advantage"])
        losses.append(grouped_policy_loss(current, reference, advantages, beta))
        delta = reference - current
        kls.append((delta.exp() - delta - 1.0).mean())
    return torch.stack(losses).sum() / total_decisions, torch.stack(kls).sum() / total_decisions


def adapter_movement(policy, reference):
    """Coordinate movement against the unchanged SFT; no extra model copy."""
    import torch
    reference_parameters = dict(reference.named_parameters())
    squared_delta = squared_reference = 0.0
    changed = count = 0
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if "lora_" not in name:
                continue
            other = reference_parameters[name]
            delta = parameter.float() - other.float()
            squared_delta += delta.double().square().sum().item()
            squared_reference += other.double().square().sum().item()
            changed += int(torch.count_nonzero(delta).item() > 0)
            count += 1
    return {"l2": math.sqrt(squared_delta),
            "relative_l2": math.sqrt(squared_delta / squared_reference) if squared_reference else None,
            "changed_tensors": changed, "total_tensors": count}


def adapter_parameter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" in name:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path: Path, *, model, optimizer, scheduler, update: int, samples: int, rng: random.Random,
                    update_metrics: list[dict] | None = None) -> None:
    import torch
    import tempfile

    if path.exists():
        raise ValueError("refusing to overwrite a committed checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    model.save_pretrained(str(staging / "adapter"))
    state = {
        "update": update,
        "samples": samples,
        "update_metrics": update_metrics,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_rng": rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, staging / "state.pt")
    atomic_write_json(staging / "complete.json", {"update": update,
        "adapter_sha256": directory_sha256(staging / "adapter"),
        "state_sha256": file_sha256(staging / "state.pt")})
    staging.rename(path)


def load_checkpoint(path: Path, *, model, optimizer, scheduler, rng: random.Random,
                    update_metrics: list[dict] | None = None) -> tuple[int, int]:
    import torch

    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter_file = path / "adapter" / "adapter_model.safetensors"
    if (path / "complete.json").exists():
        complete = json.loads((path / "complete.json").read_text())
        if (directory_sha256(path / "adapter") != complete["adapter_sha256"]
                or file_sha256(path / "state.pt") != complete["state_sha256"]):
            raise ValueError("checkpoint completion hashes differ")
    set_peft_model_state_dict(model, load_file(str(adapter_file)), adapter_name="default")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    if update_metrics is not None:
        history = state.get("update_metrics")
        if history is None or [row["update"] for row in history] != list(range(1, int(state["update"]) + 1)):
            raise ValueError("checkpoint lacks complete committed update history")
        if sum(row["turns"] for row in history) != int(state["samples"]):
            raise ValueError("checkpoint sample count does not match committed update history")
        update_metrics[:] = history
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["update"]), int(state["samples"])


def episode_progress(*, completed: int, total: int, elapsed: float, hourly_usd: float,
                     max_hours: float, dollar_limit: float, update_metrics: list[dict],
                     pause_after: int | None = None) -> dict:
    """Decide at committed-update boundaries; never report an overrun as success."""
    rate = sum(row["seconds"] for row in update_metrics) / len(update_metrics) if len(update_metrics) >= 3 else 0.0
    projection = runtime_budget(
        elapsed_seconds=elapsed, hourly_usd=hourly_usd, max_hours=max_hours,
        dollar_limit=dollar_limit, remaining_updates=max(0, total - completed),
        seconds_per_update=rate,
    )
    if projection["stop"]:
        status = "stopped_budget"
    elif completed >= total:
        status = "completed"
    elif pause_after == completed:
        status = "paused"
    else:
        status = "running"
    return {"status": status, "runtime_projection": projection}


def main() -> None:
    global ACTIVE_MANIFEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E5", "E6", "E7", "FEEDBACK"), required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--frozen-sft-adapter-dir", type=Path, default=Path("runs/sft-v1/adapter"))
    parser.add_argument("--benchmark-manifest", type=Path, default=Path("benchmarks/stress-v1/manifest.json"))
    parser.add_argument("--stage5-manifest", type=Path, default=Path("runs/sft-v1/closed_loop/manifest.json"))
    parser.add_argument("--out-dir", type=Path, required=True, help="normally runs/<run-id>/rl")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--base-model-revision", help="pin base and tokenizer before either is loaded")
    parser.add_argument("--recovery-starts", type=Path, help="training-only replayable states; alternate empty and recovery groups")
    parser.add_argument("--training-seeds-file", type=Path, help="JSON seed list restricting the frozen training partition")
    parser.add_argument("--registration-file", type=Path, help="external immutable experiment registration")
    parser.add_argument("--approval-file", type=Path, help="new feedback-pilot budget approval annex")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--advantage-method", choices=("active_group", "fixed_zero"), default="active_group")
    parser.add_argument("--advantage-reward-scale", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--score-scale", type=float, default=100.0)
    parser.add_argument("--death-penalty", type=float, default=2.0)
    parser.add_argument("--illegal-penalty", type=float, default=10.0)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--pilot-dollar-limit", type=float, default=20.0)
    parser.add_argument("--stage-dollar-limit", type=float, default=100.0)
    parser.add_argument("--prior-stage-spend-usd", type=float, default=0.0)
    parser.add_argument("--instance-hourly-usd", type=float, required=True)
    parser.add_argument("--max-wall-clock-hours", type=float, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pause-after-update", type=int, help="checkpoint and pause without changing the registered scheduler horizon")
    args = parser.parse_args()
    if args.experiment == "FEEDBACK":
        from scripts.stage6_feedback import validate, REGISTRATION
        from scripts.eval_feedback import approved_session
        if args.approval_file is None or args.registration_file is None or args.registration_file.resolve() != REGISTRATION.resolve():
            raise ValueError("feedback pilot needs its registration and new explicit budget approval")
        approval = approved_session(args.approval_file)
        feedback_registration = validate()
        matches = [r for r in feedback_registration["run_order"]
                   if r["run_id"] == args.out_dir.parent.name and args.out_dir.name == "rl"]
        if len(matches) != 1:
            raise ValueError("unregistered feedback run directory")
        run = matches[0]
        expected = {**feedback_registration["recipe"], "advantage_method": run["method"],
                    "training_seed": run["seed"], "question": feedback_registration["question"],
                    "base_model": feedback_registration["base_model"],
                    "base_model_revision": feedback_registration["base_model_revision"]}
        if any(getattr(args, k) != v for k, v in expected.items()):
            raise ValueError("feedback recipe differs from registration")
        for flag, registered_path in (("adapter_dir", feedback_registration["initial_adapter"]),
                ("frozen_sft_adapter_dir", feedback_registration["initial_adapter"]),
                ("training_seeds_file", feedback_registration["training_seed_file"]),
                ("recovery_starts", feedback_registration["recovery_start_file"]),
                ("benchmark_manifest", feedback_registration["benchmark_manifest"])):
            if getattr(args, flag).resolve() != Path(registered_path).resolve():
                raise ValueError("feedback inputs differ from registration")
        if (args.instance_hourly_usd != approval["hourly_usd"] or args.stage_dollar_limit > approval["hard_limit_usd"]
                or args.max_wall_clock_hours > 4):
            raise ValueError("trainer exceeds approved feedback limits")
        if args.resume and not (args.resume / "complete.json").exists():
            raise ValueError("feedback resume needs a complete checkpoint")
    run_started = time.time()
    manifest_path = args.out_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and not args.resume:
        raise SystemExit(f"run already registered in {manifest_path}; use --resume or a new run directory")
    if args.resume and not previous:
        raise SystemExit("resume requires this run's existing registration manifest")
    if args.resume and args.resume.resolve().parent != args.out_dir.resolve():
        raise SystemExit("resume checkpoint must belong to the registered run directory")

    if args.experiment == "E5" and not 10 <= args.horizon <= 20:
        raise SystemExit("E5 trajectory proof must use a 10-20 turn horizon")
    if not 0 <= args.gamma <= 1:
        raise SystemExit("--gamma must be in [0, 1]")
    trajectory_advantages([], args.gamma, method=args.advantage_method, reward_scale=args.advantage_reward_scale)
    if args.kl_beta <= 0:
        raise SystemExit("--kl-beta must preserve the frozen SFT reference")
    if min(args.updates, args.horizon, args.save_every, args.train_batch_size) < 1 or args.group_size < 2:
        raise SystemExit("updates, horizon, and batch sizes must be positive; grouped RL requires at least two trajectories")
    if args.pause_after_update is not None and not 0 < args.pause_after_update < args.updates:
        raise SystemExit("pause update must be positive and earlier than the registered final update")
    if min(args.pilot_dollar_limit, args.stage_dollar_limit, args.max_wall_clock_hours, args.temperature) <= 0 or min(args.instance_hourly_usd, args.prior_stage_spend_usd) < 0:
        raise SystemExit("budgets, wall-clock limit, and temperature must be positive; spend/rate cannot be negative")
    if args.prior_stage_spend_usd + args.pilot_dollar_limit > args.stage_dollar_limit:
        raise SystemExit("registered pilot plus prior spend exceeds the full Stage 6 budget")
    projected = args.instance_hourly_usd * args.max_wall_clock_hours
    if projected > args.pilot_dollar_limit:
        raise SystemExit(f"registered wall-clock limit projects to ${projected:.2f}, over pilot limit")

    benchmark = json.loads(args.benchmark_manifest.read_text())
    seed_validation = validate_seed_manifest(benchmark)
    train_seeds = json.loads(args.training_seeds_file.read_text()) if args.training_seeds_file else benchmark["training_seeds"]
    if not train_seeds or len(set(train_seeds)) != len(train_seeds) or not set(train_seeds) <= set(benchmark["training_seeds"]):
        raise ValueError("training seeds must be a nonempty unique subset of the frozen training partition")
    start_bank = load_start_bank(args.recovery_starts, train_seeds) if args.recovery_starts else []
    entry_gate = validate_entry_gate(args.stage5_manifest, benchmark["stage5_seeds"])
    adapter_hash = directory_sha256(args.adapter_dir)
    frozen_sft_hash = directory_sha256(args.frozen_sft_adapter_dir)
    if adapter_hash != frozen_sft_hash:
        raise SystemExit("episode experiments must initialize from the exact frozen SFT adapter")
    reward_weights = EpisodeRewardWeights(args.score_scale, args.death_penalty, args.illegal_penalty)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = args.out_dir / "trajectory_batches"
    batch_dir.mkdir(exist_ok=True)
    run_id = args.out_dir.parent.name if args.out_dir.name == "rl" else args.out_dir.name
    events = EventWriter(args.out_dir / "events.jsonl", run_id=run_id, stage=6)
    registered = {
        "run_id": run_id,
        "stage": 6,
        "kind": "episode_group_rl",
        "status": "registered",
        "experiment": args.experiment,
        "research_question": args.question,
        "external_registration_sha256": file_sha256(args.registration_file) if args.registration_file else None,
        "parent_run_ids": [args.adapter_dir.parent.name],
        "git_sha": previous.get("git_sha") if previous else git_sha(),
        "host": socket.gethostname(),
        "base_model": args.base_model,
        "requested_base_model_revision": args.base_model_revision,
        "adapter_dir": str(args.adapter_dir),
        "adapter_sha256": adapter_hash,
        "frozen_sft_adapter_sha256": frozen_sft_hash,
        "reference_adapter_sha256": adapter_hash,
        "benchmark_manifest": str(args.benchmark_manifest),
        "benchmark_manifest_sha256": file_sha256(args.benchmark_manifest),
        "seed_validation": seed_validation,
        "stage5_entry_gate": entry_gate,
        "algorithm": "same-start grouped policy gradient with reward-to-go",
        "execution": configure_execution(),
        "training_seed": args.training_seed,
        "environment_training_seeds": train_seeds,
        "recovery_starts_sha256": file_sha256(args.recovery_starts) if args.recovery_starts else None,
        "starting_state_schedule": "odd empty / even recovery" if start_bank else "empty only",
        "updates": args.updates,
        "group_size": args.group_size,
        "horizon": args.horizon,
        "gamma": args.gamma,
        "advantage_method": args.advantage_method,
        "advantage_reward_scale": args.advantage_reward_scale,
        "temperature": args.temperature,
        "sampling": {"temperature": args.temperature, "top_p": 1.0, "top_k": 0, "max_new_tokens": 16},
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "kl_beta": args.kl_beta,
        "reward": {"formula": "normalized_score - death - illegal", "weights": asdict(reward_weights)},
        "budgets_usd": {"pilot": args.pilot_dollar_limit, "stage": args.stage_dollar_limit},
        "prior_stage_spend_usd": args.prior_stage_spend_usd,
        "instance_hourly_usd": args.instance_hourly_usd,
        "max_wall_clock_hours": args.max_wall_clock_hours,
        "projected_cost_limit_usd": projected,
        "registered_at": previous["registered_at"] if previous else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if previous:
        # Missing fields in historical checkpoints mean the unchanged baseline.
        previous.setdefault("advantage_method", "active_group")
        previous.setdefault("advantage_reward_scale", 10.0)
        check_resume_registration(previous, registered, [
            "experiment", "research_question", "base_model", "requested_base_model_revision", "adapter_sha256", "reference_adapter_sha256", "frozen_sft_adapter_sha256", "stage5_entry_gate",
            "environment_training_seeds", "recovery_starts_sha256", "starting_state_schedule",
            "external_registration_sha256", "execution",
            "benchmark_manifest_sha256", "training_seed", "updates", "group_size", "horizon", "gamma",
            "temperature", "learning_rate", "kl_beta", "reward", "budgets_usd", "instance_hourly_usd",
            "max_wall_clock_hours", "train_batch_size", "prior_stage_spend_usd",
            "advantage_method", "advantage_reward_scale",
        ])
    prior_elapsed = float(previous.get("wall_clock_seconds", 0.0)) if previous else 0.0
    atomic_write_json(manifest_path, registered)
    ACTIVE_MANIFEST = manifest_path
    events.emit("job_started", phase="episode_rl_initializing", current=0, total=args.updates)

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(args.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.training_seed)
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(args.training_seed)
    revision = previous.get("base_model_revision") if previous else args.base_model_revision
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy_base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, revision=revision).to("cuda")
    policy = PeftModel.from_pretrained(policy_base, str(args.adapter_dir), is_trainable=True)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_revision = getattr(policy_base.config, "_commit_hash", None)
    if args.base_model_revision and base_revision != args.base_model_revision:
        raise ValueError("loaded base model differs from the pinned revision")
    reference_base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, revision=base_revision).to("cuda")
    registered["base_model_revision"] = base_revision
    atomic_write_json(manifest_path, registered)
    reference = PeftModel.from_pretrained(reference_base, str(args.adapter_dir), is_trainable=False)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    for model in (policy, reference):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    reference_hash_before = adapter_parameter_hash(reference)
    if adapter_parameter_hash(policy) != reference_hash_before:
        raise RuntimeError("initial policy and frozen reference differ")
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, round(args.updates * 0.03)), num_training_steps=args.updates)
    start_update = 0
    sample_count = 0
    update_metrics = []
    if args.resume:
        start_update, sample_count = load_checkpoint(args.resume, model=policy, optimizer=optimizer, scheduler=scheduler, rng=rng, update_metrics=update_metrics)
        if args.pause_after_update is not None and args.pause_after_update <= start_update:
            raise ValueError("pause update must be later than the resumed checkpoint")
        if start_update > args.updates:
            raise ValueError("checkpoint exceeds the registered final update")
        events.emit("job_resumed", phase="episode_rl", current=start_update, total=args.updates, metrics={"samples": sample_count})

    stopped_for_budget = False
    paused = False
    completed_update = start_update
    last_saved_update = start_update if args.resume else None

    def progress():
        return episode_progress(
            completed=completed_update, total=args.updates,
            elapsed=prior_elapsed + time.time() - run_started,
            hourly_usd=args.instance_hourly_usd, max_hours=args.max_wall_clock_hours,
            dollar_limit=args.pilot_dollar_limit, update_metrics=update_metrics,
            pause_after=args.pause_after_update,
        )

    def persist_checkpoint():
        nonlocal last_saved_update
        checkpoint = args.out_dir / f"checkpoint-{completed_update}"
        save_checkpoint(checkpoint, model=policy, optimizer=optimizer, scheduler=scheduler,
                        update=completed_update, samples=sample_count, rng=rng, update_metrics=update_metrics)
        atomic_write_json(args.out_dir / "latest_checkpoint.json", {"path": str(checkpoint), "update": completed_update, "samples": sample_count})
        events.emit("checkpoint_saved", phase="episode_rl", current=completed_update, total=args.updates, checkpoint=str(checkpoint))
        last_saved_update = completed_update

    for update in range(start_update + 1, args.updates + 1):
        if progress()["status"] == "stopped_budget":
            stopped_for_budget = True
            break
        start = start_bank[rng.randrange(len(start_bank))] if start_bank and update % 2 == 0 else None
        environment_seed = int(start["seed"] if start else train_seeds[rng.randrange(len(train_seeds))])
        tick = time.time()
        trajectories = sample_group(
            policy,
            reference,
            tokenizer,
            seed=environment_seed,
            group_size=args.group_size,
            horizon=args.horizon,
            temperature=args.temperature,
            reward_weights=reward_weights,
            start=start,
        )
        # The helper defaults to the pre-registered value; rewrite only if a
        # caller deliberately registered another gamma.
        rewards = [[step["immediate_reward"] for step in row["steps"]] for row in trajectories]
        advantages = trajectory_advantages(rewards, args.gamma, method=args.advantage_method,
                                           reward_scale=args.advantage_reward_scale)
        for trajectory, values in zip(trajectories, advantages):
            returns = discounted_reward_to_go([step["immediate_reward"] for step in trajectory["steps"]], args.gamma)
            for step, value, reward_to_go in zip(trajectory["steps"], values, returns):
                step["advantage"] = value
                step["reward_to_go"] = reward_to_go

        batch_path = batch_dir / f"update-{update:06d}.json"
        temporary = batch_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"update": update, "environment_seed": environment_seed, "trajectories": trajectories}, indent=2) + "\n")
        temporary.replace(batch_path)

        steps = flatten_steps(trajectories)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_kl = 0.0
        chunk_size = args.train_batch_size
        for offset in range(0, len(steps), chunk_size):
            chunk = steps[offset : offset + chunk_size]
            token_rows = sequence_logprobs(policy, chunk, tokenizer.pad_token_id, args.temperature, return_tokens=True)
            loss, chunk_kl = action_loss_chunk(token_rows, chunk, args.kl_beta, len(steps))
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite policy loss")
            loss.backward()
            total_loss += float(loss.detach().cpu())
            total_kl += float(chunk_kl.detach().cpu())
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in policy.parameters() if p.requires_grad), 1.0, error_if_nonfinite=True)
        if any(p.requires_grad or p.grad is not None for p in reference.parameters()):
            raise RuntimeError("frozen reference acquired gradients")
        optimizer.step()
        scheduler.step()
        sample_count += len(steps)
        completed_update = update
        seconds = time.time() - tick
        unique_actions = {tuple(step["action"]) for step in steps if step["action"] is not None}
        metrics = {
            "update": update,
            "loss": total_loss,
            "kl": total_kl,
            "gradient_norm_before_clip": float(gradient_norm),
            "gradient_clipped": bool(gradient_norm > 1.0),
            "advantage_diagnostics": advantage_diagnostics(trajectories),
            "start_kind": "recovery" if start else "empty",
            "episode_return_variance": sum((r["episode_return"] - sum(x["episode_return"] for x in trajectories) / len(trajectories)) ** 2 for r in trajectories) / len(trajectories),
            "mean_episode_return": sum(row["episode_return"] for row in trajectories) / len(trajectories),
            "parse_rate": sum(step["parsed"] for step in steps) / len(steps),
            "legality_rate": sum(step["legal"] for step in steps) / len(steps),
            "unique_actions": len(unique_actions),
            "turns": len(steps),
            "seconds": seconds,
            "turns_per_hour": len(steps) / seconds * 3600,
        }
        update_metrics.append(metrics)
        events.emit("train_metrics", phase="episode_rl", current=update, total=args.updates, metrics=metrics)
        cumulative_elapsed = prior_elapsed + time.time() - run_started
        atomic_write_json(manifest_path, {**registered, "status": "running", "wall_clock_seconds": cumulative_elapsed, "estimated_cost_usd": cumulative_elapsed / 3600 * args.instance_hourly_usd, "completed_updates": update, "sample_count": sample_count})
        decision = progress()
        stopped_for_budget = decision["status"] == "stopped_budget"
        paused = decision["status"] == "paused"
        if update % args.save_every == 0 or update == args.updates or stopped_for_budget or paused:
            persist_checkpoint()
        if stopped_for_budget or paused:
            break

    # A stop before the next scheduled save still retains all committed updates.
    if last_saved_update != completed_update:
        persist_checkpoint()
    reference_hash_after = adapter_parameter_hash(reference)
    if reference_hash_before != reference_hash_after:
        raise RuntimeError("frozen reference adapter changed during episode training")
    adapter_out = args.out_dir / "adapter"
    policy.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))
    decision = progress()
    stopped_for_budget = stopped_for_budget or decision["status"] == "stopped_budget"
    elapsed = prior_elapsed + time.time() - run_started
    actual_cost = elapsed / 3600 * args.instance_hourly_usd
    completed = {
        **registered,
        "status": "stopped_budget" if stopped_for_budget else "paused" if paused else "completed",
        "reference_frozen": all(not parameter.requires_grad for parameter in reference.parameters()),
        "reference_weights_sha256_before": reference_hash_before,
        "reference_weights_sha256_after": reference_hash_after,
        "adapter_movement": adapter_movement(policy, reference),
        "output_adapter_sha256": directory_sha256(adapter_out),
        "sample_count": sample_count,
        "completed_updates": completed_update,
        "update_metrics": update_metrics,
        "runtime_projection": decision["runtime_projection"],
        "wall_clock_seconds": elapsed,
        "estimated_cost_usd": actual_cost,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "system_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "resume_used": bool(args.resume),
        "resumed_from_update": start_update,
        "pause_after_update": args.pause_after_update,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(manifest_path, completed)
    event_type = "budget_stop" if stopped_for_budget else "job_paused" if paused else "job_completed"
    events.emit(event_type, phase="episode_rl", current=completed["completed_updates"], total=args.updates, metrics={"wall_clock_seconds": elapsed, "estimated_cost_usd": actual_cost, "samples": sample_count}, artifacts=[str(adapter_out), str(manifest_path)])
    print(f"saved episode RL adapter and manifest to {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        record_run_failure(ACTIVE_MANIFEST, error)
        raise
