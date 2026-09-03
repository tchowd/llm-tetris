#!/usr/bin/env python3
"""Pre-register and independently verify the authorized R2 GPU trajectory proof."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_recovery_sft import validate_inputs
from tetris.chat import build_generation_prompt
from tetris.rl import (EpisodeRewardWeights, atomic_write_json, directory_sha256,
    discounted_reward_to_go, episode_transition, file_sha256, grouped_policy_loss,
    restore_game, state_hash, trajectory_advantages, validate_trajectory)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def register(protocol_path: Path, prior: float, retry_of: Path | None = None) -> dict:
    protocol = json.loads(protocol_path.read_text())
    data_dir = Path(protocol["data"]["data_dir"])
    data = validate_inputs(protocol, data_dir)
    if data["registration_sha256"] != file_sha256(protocol_path):
        raise ValueError("dataset belongs to a different protocol")
    if not math.isfinite(prior) or prior < 0 or prior + 1.05 + 20 > 100:
        raise ValueError("R2 exceeds the existing budget")
    if protocol["protocol"] != "stage6-recovery-v1" or protocol["final_test_access"]:
        raise ValueError("R2 must use the development-only recovery protocol")
    recipe = protocol["trajectory_proof"]
    result = {"experiment": "R2", "status": "registered", "registered_at": now(),
        "protocol_path": str(protocol_path), "protocol_sha256": file_sha256(protocol_path),
        "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "source_sha256": {name: file_sha256(Path(name)) for name in (
            "scripts/train_episode_rl.py", "scripts/check_episode_proof.py", "tetris/rl.py", "tetris/recovery.py")},
        "recipe": recipe, "run_id": recipe["run_id"],
        "control_run_id": "rl-r2-episode-proof-uninterrupted-seed0",
        "question": "Do exact replay, delayed credit, GPU token probabilities and resume agree?",
        "prior_stage_spend_usd": prior, "pilot_usd": 20, "stage_usd": 100,
        "hourly_usd": 1.05, "block_hours": 1, "workflow_minutes": 50,
        "logprob_absolute_tolerance": 0.0001, "resume_weights_absolute_tolerance": 0,
        "minimum_allocated_gpu_headroom_fraction": 0.15,
        "positive_direction_sgd_learning_rate": 0.001,
        "positive_direction_min_logprob_gain": 0,
        "pilot_projection_safety_multiplier": 1.5, "pilot_projection_startup_seconds": 120,
        "control": "Same four registered updates uninterrupted; compare every saved trajectory and final adapter tensor with 2+2 resumed run. No control adapter promoted.",
        "positive_direction": "On a throwaway copy, apply the actual grouped loss to exact saved completion tokens with positive synthetic delayed reward; probability must increase. Restore and hash-check weights; never export this probe.",
        "execution": {"deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
            "float32_matmul_precision": "highest", "qwen3_completion_logits_suffix": True,
            "episode_normalization": "decimal50"},
        "final_test_access": False, "research_complete": False}
    if retry_of:
        previous = json.loads(retry_of.read_text())
        failed_root = Path("runs") / previous["run_id"] / "rl"
        block = json.loads((failed_root / "block-state.json").read_text())
        if previous["protocol_sha256"] != result["protocol_sha256"] or previous["recipe"] != recipe or previous.get("retry_of_registration") or block["status"] != "failed":
            raise ValueError("only one implementation retry of the retained failed R2 is supported")
        if previous["run_id"] != recipe["run_id"]:
            raise ValueError("unexpected failed proof identity")
        diagnosis = Path("runs/stage6-recovery-v1/rl/r2-gradient-diagnosis.json")
        d = json.loads(diagnosis.read_text())
        if d["default_repeat"]["equal"] or not d["deterministic_repeat"]["equal"] or not d["adapter_unchanged"] or d["optimizer_updates"]:
            raise ValueError("deterministic-gradient repair evidence required")
        result.update(run_id="rl-r2v2-episode-proof-seed0", control_run_id="rl-r2v2-episode-proof-uninterrupted-seed0",
            retry_of_registration=str(retry_of), retry_of_registration_sha256=file_sha256(retry_of),
            failed_block_sha256=file_sha256(failed_root / "block-state.json"),
            repair_evidence=str(diagnosis), repair_evidence_sha256=file_sha256(diagnosis),
            repair="Deterministic GPU gradients, portable normalization and completion-only vocabulary projection; identical hypothesis, seeds, update recipe, tolerances and budgets.")
    return result


def validate_registration(path: Path):
    r = json.loads(path.read_text())
    p = json.loads(Path(r["protocol_path"]).read_text())
    expected = register(Path(r["protocol_path"]), r["prior_stage_spend_usd"], Path(r["retry_of_registration"]) if r.get("retry_of_registration") else None)
    if any(r.get(k) != value for k, value in expected.items() if k != "registered_at"):
        raise ValueError("R2 registration or linked evidence changed")
    if directory_sha256(Path("runs/sft-v1/adapter")) != p["frozen_sft_adapter_sha256"]:
        raise ValueError("original SFT changed")
    return r, p


def audit_batch(batch: dict, recipe: dict, training_seeds: list[int], recovery_bank: list[dict]) -> int:
    trajectories = batch["trajectories"]
    if len(trajectories) != recipe["group_size"] or batch["environment_seed"] not in training_seeds:
        raise ValueError("wrong group or training seed")
    starts = {(s["seed"], state_hash(s), json.dumps(s["action_prefix"])) for s in recovery_bank}
    start_keys = {(t["seed"], state_hash(t["start_state"]), json.dumps(t["start_actions"])) for t in trajectories}
    if len(start_keys) != 1 or any(t["seed"] != batch["environment_seed"] for t in trajectories):
        raise ValueError("trajectories do not share one exact start")
    key = next(iter(start_keys))
    if batch["update"] % 2 == 0:
        if key not in starts:
            raise ValueError("even update did not use registered recovery bank")
    elif any(t["start_actions"] or t["start_state"]["turn"] != 0 for t in trajectories):
        raise ValueError("odd update did not start empty")
    rewards = []
    for trajectory in trajectories:
        if not 1 <= len(trajectory["steps"]) <= recipe["horizon"]:
            raise ValueError("invalid trajectory horizon")
        replay = validate_trajectory(trajectory)
        if replay != trajectory["replay_validation"]:
            raise ValueError("replay evidence differs from actual replay")
        game = restore_game(trajectory["seed"], trajectory["start_actions"], expected=trajectory["start_state"])
        row_rewards = []
        for step in trajectory["steps"]:
            action = tuple(step["action"]) if step["action"] is not None else None
            transition = episode_transition(game, action, EpisodeRewardWeights())
            legal = action in {(x["rot"], x["x"]) for x in game.legal_placements()}
            if step["legal"] != legal or step["parsed"] != (action is not None):
                raise ValueError("saved validity flags are wrong")
            if transition.reward != step["immediate_reward"] or transition.components != step["reward_components"] or transition.terminal != step["terminal"] or transition.terminal_reason != step["terminal_reason"]:
                raise ValueError("reward or terminal flags do not replay")
            row_rewards.append(transition.reward)
            if legal:
                game.step(*action)
        if trajectory["score"] != game.score - trajectory["start_state"]["score"] or trajectory["lines"] != game.lines - trajectory["start_state"]["lines"]:
            raise ValueError("trajectory includes score from its starting prefix")
        if trajectory["pieces"] != sum(s["legal"] for s in trajectory["steps"]) or trajectory["episode_return"] != sum(row_rewards):
            raise ValueError("trajectory accounting differs from saved actions")
        rewards.append(row_rewards)
    advantages = trajectory_advantages(rewards, recipe["gamma"])
    for trajectory, row_rewards, values in zip(trajectories, rewards, advantages):
        returns = discounted_reward_to_go(row_rewards, recipe["gamma"])
        for step, value, reward in zip(trajectory["steps"], values, returns):
            if step["advantage"] != value or step["reward_to_go"] != reward:
                raise ValueError("delayed reward/advantage alignment is wrong")
    return sum(len(t["steps"]) for t in trajectories)


def independent_token_logprobs(model, rows, pad_id, temperature):
    """Separate cross-entropy implementation; do not call trainer's log-prob helper."""
    import torch
    sequences = [r["prompt_ids"] + r["completion_ids"] for r in rows]
    device = next(model.parameters()).device
    ids = torch.full((len(rows), max(map(len, sequences))), pad_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        ids[i, :len(sequence)] = torch.tensor(sequence, device=device)
        mask[i, :len(sequence)] = 1
    offset = min(len(r["prompt_ids"]) - 1 for r in rows) if getattr(getattr(model, "config", None), "model_type", None) == "qwen3" else 0
    options = {"logits_to_keep": ids.shape[1] - offset, "use_cache": False} if offset else {}
    logits = model(input_ids=ids, attention_mask=mask, **options).logits
    out = []
    for i, row in enumerate(rows):
        start = len(row["prompt_ids"]) - 1 - offset
        length = len(row["completion_ids"])
        selected = logits[i, start:start + length].float() / temperature
        target = torch.tensor(row["completion_ids"], device=device)
        out.append(-torch.nn.functional.cross_entropy(selected, target, reduction="none"))
    return out


def projection(r: dict, p: dict, metrics: list[dict]):
    if len(metrics) != r["recipe"]["updates"] or any(m["turns"] <= 0 or not math.isfinite(m["seconds"]) or m["seconds"] <= 0 for m in metrics):
        raise ValueError("incomplete measured throughput")
    worst = max(m["seconds"] / m["turns"] for m in metrics)
    pilot = p["episode_pilot"]
    turns = pilot["updates"] * pilot["group_size"] * pilot["horizon"]
    seconds = r["pilot_projection_startup_seconds"] + worst * turns * r["pilot_projection_safety_multiplier"]
    return {"worst_seconds_per_turn": worst, "full_length_pilot_turns": turns,
        "safety_multiplier": r["pilot_projection_safety_multiplier"], "projected_seconds": seconds,
        "projected_usd": seconds / 3600 * r["hourly_usd"],
        "within_training_limit": seconds <= pilot["max_training_hours"] * 3600}


def evidence(r: dict, p: dict, registration_path: Path | None = None):
    root = Path("runs") / r["run_id"] / "rl"
    control = Path("runs") / r["control_run_id"] / "rl"
    data = Path(p["data"]["data_dir"])
    seeds = json.loads((data / "training-seeds.json").read_text())
    bank = [json.loads(s) for s in (data / "train-starts.jsonl").read_text().splitlines()]
    manifests, batches = [], []
    recipe = r["recipe"]
    for path, resumed in ((root, True), (control, False)):
        m = json.loads((path / "manifest.json").read_text())
        expected = {k: recipe[k] for k in ("updates", "group_size", "horizon", "training_seed", "learning_rate", "kl_beta", "gamma", "train_batch_size")}
        expected.update(status="completed", experiment="E5", completed_updates=recipe["updates"],
            adapter_sha256=p["frozen_sft_adapter_sha256"], frozen_sft_adapter_sha256=p["frozen_sft_adapter_sha256"],
            benchmark_manifest_sha256=p["benchmark_manifest_sha256"], requested_base_model_revision=p["base_model_revision"],
            base_model_revision=p["base_model_revision"], environment_training_seeds=seeds,
            recovery_starts_sha256=file_sha256(data / "train-starts.jsonl"), resume_used=resumed,
            resumed_from_update=recipe["pause_after_update"] if resumed else 0, reference_frozen=True,
            temperature=1, max_wall_clock_hours=recipe["max_training_hours"], execution=r["execution"],
            research_question=r["question"], budgets_usd={"pilot": 20, "stage": 100},
            prior_stage_spend_usd=r["prior_stage_spend_usd"], instance_hourly_usd=r["hourly_usd"],
            reward={"formula": "normalized_score - death - illegal", "weights": {"score_scale": 100, "death_penalty": 2, "illegal_penalty": 10}})
        if registration_path:
            expected["external_registration_sha256"] = file_sha256(registration_path)
        if any(m.get(k) != v for k, v in expected.items()):
            raise ValueError(f"incomplete or unregistered training: {path}")
        if m["reference_weights_sha256_before"] != m["reference_weights_sha256_after"]:
            raise ValueError("reference weights changed")
        if m["registered_at"] < r["registered_at"]:
            raise ValueError("training predates the experiment registration")
        if directory_sha256(path / "adapter") != m["output_adapter_sha256"]:
            raise ValueError("output adapter does not match its manifest")
        rows = [json.loads((path / "trajectory_batches" / f"update-{i:06d}.json").read_text()) for i in range(1, recipe["updates"] + 1)]
        if len(list((path / "trajectory_batches").glob("*.json"))) != recipe["updates"]:
            raise ValueError("unexpected trajectory batch count")
        if [x["update"] for x in rows] != list(range(1, recipe["updates"] + 1)) or [x["update"] for x in m["update_metrics"]] != list(range(1, recipe["updates"] + 1)):
            raise ValueError("lost or duplicated committed updates")
        counts = [audit_batch(b, recipe, seeds, bank) for b in rows]
        if counts != [x["turns"] for x in m["update_metrics"]] or sum(counts) != m["sample_count"]:
            raise ValueError("sample accounting does not match replay")
        manifests.append(m)
        batches.append(rows)
    paused = json.loads((root / "paused-manifest.json").read_text())
    if paused["status"] != "paused" or paused["completed_updates"] != recipe["pause_after_update"] or paused["sample_count"] != sum(x["turns"] for x in manifests[0]["update_metrics"][:recipe["pause_after_update"]]):
        raise ValueError("pause did not retain exact committed progress")
    if batches[0] != batches[1]:
        raise ValueError("GPU pause/resume trajectories differ from uninterrupted control")
    return root, control, manifests, batches[0]


def gpu_proof(path: Path) -> dict:
    import torch
    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.train_episode_rl import adapter_parameter_hash, configure_execution

    r, p = validate_registration(path)
    if configure_execution() != r["execution"]:
        raise ValueError("GPU execution differs from registration")
    root, control, manifests, batches = evidence(r, p, path)
    if not torch.cuda.is_available():
        raise ValueError("R2 requires the real CUDA worker")
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    left = load_file(str(root / "adapter/adapter_model.safetensors"))
    right = load_file(str(control / "adapter/adapter_model.safetensors"))
    if set(left) != set(right) or any(not torch.equal(left[k], right[k]) for k in left):
        raise ValueError("GPU resumed adapter differs from uninterrupted control")
    tokenizer = AutoTokenizer.from_pretrained(p["base_model"], revision=p["base_model_revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    models = []
    for trainable in (True, False):
        base = AutoModelForCausalLM.from_pretrained(p["base_model"], revision=p["base_model_revision"], dtype=torch.bfloat16).to("cuda")
        if base.config._commit_hash != p["base_model_revision"]:
            raise ValueError("GPU verifier loaded a different base")
        model = PeftModel.from_pretrained(base, "runs/sft-v1/adapter", is_trainable=trainable)
        model.eval()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
        models.append(model)
    policy, reference = models
    frozen_before = adapter_parameter_hash(reference)
    max_error, tokens_checked = 0.0, 0
    for batch in batches:
        update = batch["update"]
        source = Path("runs/sft-v1/adapter") if update == 1 else root / f"checkpoint-{update - 1}/adapter"
        set_peft_model_state_dict(policy, load_file(str(source / "adapter_model.safetensors")), adapter_name="default")
        trajectories = batch["trajectories"]
        for turn in range(max(len(t["steps"]) for t in trajectories)):
            rows = [t["steps"][turn] for t in trajectories if len(t["steps"]) > turn]
            for row in rows:
                prompt = build_generation_prompt(tokenizer, row["serialized_prompt"])
                if prompt != row["prompt"] or tokenizer(prompt, add_special_tokens=False)["input_ids"] != row["prompt_ids"] or tokenizer.decode(row["completion_ids"], skip_special_tokens=True) != row["raw_completion"]:
                    raise ValueError("exact saved tokens do not match prompt/completion")
            for model, key, mean_key in ((policy, "policy_token_logprobs_at_sampling", "policy_logprob_at_sampling"), (reference, "reference_token_logprobs", "reference_logprob")):
                with torch.no_grad():
                    actual = independent_token_logprobs(model, rows, tokenizer.pad_token_id, 1)
                for row, values in zip(rows, actual):
                    saved = torch.tensor(row[key], device=values.device)
                    error = float((values - saved).abs().max())
                    if abs(float(saved.mean()) - row[mean_key]) > r["logprob_absolute_tolerance"]:
                        raise ValueError("saved mean probability does not match exact token values")
                    max_error = max(max_error, error)
                    tokens_checked += len(values)
    # Controlled delayed reward uses the same grouped loss, but is not evidence of game improvement.
    set_peft_model_state_dict(policy, load_file("runs/sft-v1/adapter/adapter_model.safetensors"), adapter_name="default")
    initial_hash = adapter_parameter_hash(policy)
    parameters = [x for x in policy.parameters() if x.requires_grad]
    saved_parameters = [x.detach().clone() for x in parameters]
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.config.use_cache = False
    policy.train()  # all dropout is disabled; enable checkpointing for the throwaway gradient
    probe = min((s for t in batches[0]["trajectories"] for s in t["steps"]), key=lambda s: s["policy_logprob_at_sampling"])
    positive = trajectory_advantages([[0, 0, 1], [0, 0, -1]], r["recipe"]["gamma"])[0][0]
    before = independent_token_logprobs(policy, [probe], tokenizer.pad_token_id, 1)[0]
    value_before = float(before.detach().mean())
    optimizer = torch.optim.SGD(parameters, lr=r["positive_direction_sgd_learning_rate"])
    loss = grouped_policy_loss(before, before.detach(), torch.full_like(before, positive), r["recipe"]["kl_beta"])
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1))
    optimizer.step()
    with torch.no_grad():
        value_after = float(independent_token_logprobs(policy, [probe], tokenizer.pad_token_id, 1)[0].mean())
        for parameter, saved in zip(parameters, saved_parameters):
            parameter.copy_(saved)
    restored = adapter_parameter_hash(policy) == initial_hash
    reference_unchanged = adapter_parameter_hash(reference) == frozen_before == manifests[0]["reference_weights_sha256_before"]
    measured = projection(r, p, manifests[0]["update_metrics"])
    peak = max(torch.cuda.max_memory_allocated(), *(m["peak_cuda_bytes"] for m in manifests))
    total = torch.cuda.get_device_properties(0).total_memory
    checks = {"all_trajectories_replayed": True, "exact_gpu_resume_trajectories": True,
        "exact_gpu_resume_adapter_tensors": True, "exact_token_alignment": tokens_checked > 0 and max_error <= r["logprob_absolute_tolerance"],
        "positive_delayed_reward_direction": math.isfinite(norm) and norm > 0 and value_after - value_before > r["positive_direction_min_logprob_gain"],
        "probe_weights_restored": restored, "reference_unchanged": reference_unchanged,
        "gpu_headroom": 1 - peak / total >= r["minimum_allocated_gpu_headroom_fraction"],
        "pilot_projection_fits": measured["within_training_limit"],
        "frozen_sft_unchanged": directory_sha256(Path("runs/sft-v1/adapter")) == p["frozen_sft_adapter_sha256"]}
    artifacts = {str(f): file_sha256(f) for base in (root, control) for f in [base / "manifest.json", *sorted((base / "trajectory_batches").glob("*.json"))]}
    artifacts[str(root / "paused-manifest.json")] = file_sha256(root / "paused-manifest.json")
    return {"experiment": "R2", "status": "passed" if all(checks.values()) else "not_passed", "checks": checks,
        "registration_sha256": file_sha256(path), "protocol_sha256": r["protocol_sha256"], "evidence_sha256": artifacts,
        "sample_count": manifests[0]["sample_count"], "tokens_checked": tokens_checked, "max_absolute_logprob_error": max_error,
        "positive_direction": {"synthetic_delayed_reward": [0, 0, 1], "advantage": positive, "gradient_norm": norm,
            "mean_logprob_before": value_before, "mean_logprob_after": value_after, "restored": restored},
        "gpu": {"name": torch.cuda.get_device_name(), "total_bytes": total, "peak_allocated_bytes": peak, "headroom_fraction": 1 - peak / total},
        "pilot_projection": measured, "verifier_seconds": time.time() - started,
        "generated_at": now(), "final_test_access": False, "research_complete": False,
        "next": "Only a fully passed R2 permits R3 registration. R1 outcome remains independent."}


def validate_proof_report(registration_path: Path, gate_path: Path):
    """Reproduce CPU-verifiable proof checks and bind the GPU-only evidence."""
    r, p = validate_registration(registration_path)
    root, control, manifests, _ = evidence(r, p, registration_path)
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != "passed" or gate.get("experiment") != "R2" or gate.get("registration_sha256") != file_sha256(registration_path) or gate.get("protocol_sha256") != r["protocol_sha256"]:
        raise ValueError("a passed, hash-bound R2 proof is required")
    expected_paths = {str(f) for base in (root, control) for f in [base / "manifest.json", *sorted((base / "trajectory_batches").glob("*.json"))]}
    expected_paths.add(str(root / "paused-manifest.json"))
    if set(gate["evidence_sha256"]) != expected_paths or any(file_sha256(Path(f)) != digest for f, digest in gate["evidence_sha256"].items()):
        raise ValueError("GPU proof evidence changed")
    if gate["pilot_projection"] != projection(r, p, manifests[0]["update_metrics"]) or gate["sample_count"] != manifests[0]["sample_count"]:
        raise ValueError("GPU proof projection or sample accounting differs")
    direction, gpu = gate["positive_direction"], gate["gpu"]
    checks = gate["checks"]
    if not checks or not all(v is True for v in checks.values()) or not gate["pilot_projection"]["within_training_limit"]:
        raise ValueError("R2 has an unmet prerequisite")
    if not (0 <= gate["max_absolute_logprob_error"] <= r["logprob_absolute_tolerance"] and gate["tokens_checked"] > 0 and direction["gradient_norm"] > 0 and math.isfinite(direction["gradient_norm"]) and direction["mean_logprob_after"] > direction["mean_logprob_before"] and direction["restored"]):
        raise ValueError("GPU probability or credit-direction evidence did not pass")
    if not math.isclose(gpu["headroom_fraction"], 1 - gpu["peak_allocated_bytes"] / gpu["total_bytes"], abs_tol=1e-12) or gpu["headroom_fraction"] < r["minimum_allocated_gpu_headroom_fraction"]:
        raise ValueError("GPU headroom evidence did not pass")
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--register-from-protocol", type=Path)
    parser.add_argument("--prior-stage-spend-usd", type=float)
    parser.add_argument("--retry-of", type=Path, help="retain the original failed proof and register its sole implementation repair")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.register_from_protocol:
        if args.registration.exists() or args.prior_stage_spend_usd is None:
            parser.error("registration must be new and prior spend supplied")
        result = register(args.register_from_protocol, args.prior_stage_spend_usd, args.retry_of)
        atomic_write_json(args.registration, result)
        return
    validate_registration(args.registration)
    if args.preflight:
        print("R2 registration and frozen inputs verified")
        return
    if not args.out or args.out.exists():
        parser.error("--out must name a new GPU proof artifact")
    result = gpu_proof(args.registration)
    atomic_write_json(args.out, result)
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
