#!/usr/bin/env python3
"""Versioned L40S verifier, preserving the previous independent GPU proof method."""
from __future__ import annotations
import math
import time
from pathlib import Path
from scripts.stage6_scale10x import validate_registration, evidence, projection, now
from scripts.check_episode_proof import independent_token_logprobs
from tetris.chat import build_generation_prompt
from tetris.rl import directory_sha256, file_sha256, grouped_policy_loss, trajectory_advantages

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
    return {"experiment": "SCALE10X_PROOF", "status": "passed" if all(checks.values()) else "not_passed", "checks": checks,
        "registration_sha256": file_sha256(path), "protocol_sha256": r["protocol_sha256"], "evidence_sha256": artifacts,
        "sample_count": manifests[0]["sample_count"], "tokens_checked": tokens_checked, "max_absolute_logprob_error": max_error,
        "positive_direction": {"synthetic_delayed_reward": [0, 0, 1], "advantage": positive, "gradient_norm": norm,
            "mean_logprob_before": value_before, "mean_logprob_after": value_after, "restored": restored},
        "gpu": {"name": torch.cuda.get_device_name(), "total_bytes": total, "peak_allocated_bytes": peak, "headroom_fraction": 1 - peak / total},
        "pilot_projection": measured, "verifier_seconds": time.time() - started,
        "generated_at": now(), "final_test_access": False, "research_complete": False,
        "next": "Only a fully passed L40S proof permits the registered 320-update training run."}



