#!/usr/bin/env python3
"""One pre-registered recovery-data continuation of frozen SFT, not a new base."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tetris.events import EventWriter
from tetris.rl import atomic_write_json, directory_sha256, file_sha256, record_run_failure, runtime_budget

ACTIVE_MANIFEST = None


def validate_inputs(registration: dict, data_dir: Path):
    if registration["protocol"] != "stage6-recovery-v1" or registration["final_test_access"]:
        raise ValueError("requires registered development-only recovery protocol")
    data = json.loads((data_dir / "manifest.json").read_text())
    if data["status"] != "completed":
        raise ValueError("recovery data is not complete")
    for name, expected in data["files_sha256"].items():
        if file_sha256(data_dir / name) != expected:
            raise ValueError(f"dataset changed: {name}")
    if set(data["training_seeds"]) & set(data["validation_seeds"]):
        raise ValueError("training/validation seed overlap")
    expected = registration["data"]
    if data["num_train_rows"] != expected["recovery_train_rows"] + expected["ordinary_train_rows"] or data["num_eval_rows"] != expected["recovery_validation_rows"]:
        raise ValueError("dataset counts differ from registration")
    return data


def main():
    global ACTIVE_MANIFEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    r = json.loads(args.registration.read_text())
    recipe, limits = r["sft"], r["budgets"]
    data_dir = Path(r["data"]["data_dir"])
    data = validate_inputs(r, data_dir)
    if data["registration_sha256"] != file_sha256(args.registration):
        raise ValueError("dataset registration changed")
    frozen = Path("runs/sft-v1/adapter")
    if directory_sha256(frozen) != r["frozen_sft_adapter_sha256"]:
        raise ValueError("frozen initialization changed")
    out = Path("runs") / recipe["run_id"] / "rl"
    manifest_path = out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and not args.resume:
        raise ValueError("refusing to overwrite training; use the registered checkpoint to resume")
    if args.resume and (previous is None or args.resume.resolve().parent != out.resolve()):
        raise ValueError("resume checkpoint must belong to this run")
    registered = {"run_id": recipe["run_id"], "stage": 6, "kind": "recovery_sft", "experiment": "R1",
                  "status": "registered", "registration_sha256": file_sha256(args.registration),
                  "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
                  "adapter_sha256": r["frozen_sft_adapter_sha256"], "base_model": r["base_model"],
                  "base_model_revision": r["base_model_revision"], "training_seed": recipe["training_seed"],
                  "recipe": recipe, "budgets": limits, "parent_run_ids": ["sft-v1"],
                  "registered_at": previous["registered_at"] if previous else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if previous:
        for key in ("registration_sha256", "dataset_manifest_sha256", "adapter_sha256", "recipe", "base_model_revision"):
            if previous[key] != registered[key]:
                raise ValueError(f"resume registration changed: {key}")
    prior_elapsed = previous.get("wall_clock_seconds", 0) if previous else 0
    out.mkdir(parents=True, exist_ok=True)
    ACTIVE_MANIFEST = manifest_path
    atomic_write_json(manifest_path, registered)
    events = EventWriter(out / "events.jsonl", run_id=recipe["run_id"], stage=6)
    events.emit("job_started", phase="recovery_sft_initializing", current=0, total=recipe["updates"])
    started = time.monotonic()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    from scripts.train_sft import SFTDataset, load_rows, make_collator

    set_seed(recipe["training_seed"])
    tokenizer = AutoTokenizer.from_pretrained(r["base_model"], revision=r["base_model_revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(r["base_model"], revision=r["base_model_revision"], dtype=torch.bfloat16).to("cuda")
    if getattr(base.config, "_commit_hash", None) != r["base_model_revision"]:
        raise ValueError("loaded base revision differs from registration")
    model = PeftModel.from_pretrained(base, str(frozen), is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    train_rows, eval_rows = load_rows([data_dir], "train"), load_rows([data_dir], "eval")
    if {x["game_id"] for x in train_rows} & {x["game_id"] for x in eval_rows}:
        raise ValueError("SFT game leakage")
    train_dataset, eval_dataset = SFTDataset(train_rows, tokenizer), SFTDataset(eval_rows, tokenizer)

    class Observe(TrainerCallback):
        def __init__(self):
            self.times = []
            self.tick = None
            self.budget_stop = False

        def on_step_begin(self, args, state, control, **kwargs):
            self.tick = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            self.times.append(time.monotonic() - self.tick)
            elapsed = prior_elapsed + time.monotonic() - started
            projection = runtime_budget(elapsed_seconds=elapsed, hourly_usd=limits["hourly_usd"],
                max_hours=recipe["max_training_hours"], dollar_limit=limits["pilot_usd"],
                remaining_updates=max(0, recipe["updates"] - state.global_step),
                seconds_per_update=sum(self.times) / len(self.times) if len(self.times) >= 3 else 0)
            self.budget_stop = projection["stop"]
            if self.budget_stop:
                control.should_training_stop = control.should_save = True
            if state.global_step % 16 == 0 or self.budget_stop:
                progress = {**registered, "status": "running", "completed_updates": state.global_step,
                            "wall_clock_seconds": elapsed, "runtime_projection": projection}
                atomic_write_json(manifest_path, progress)
                events.emit("heartbeat", phase="recovery_sft", current=state.global_step, total=recipe["updates"], metrics=projection)

        def on_log(self, args, state, control, logs=None, **kwargs):
            events.emit("train_metrics", phase="recovery_sft", current=state.global_step, total=recipe["updates"], metrics=logs or {})

        def on_save(self, args, state, control, **kwargs):
            events.emit("checkpoint_saved", phase="recovery_sft", current=state.global_step, total=recipe["updates"], checkpoint=f"checkpoint-{state.global_step}")

    callback = Observe()
    config = TrainingArguments(output_dir=str(out), max_steps=recipe["updates"],
        per_device_train_batch_size=recipe["batch_size"], per_device_eval_batch_size=recipe["batch_size"],
        gradient_accumulation_steps=recipe["gradient_accumulation"], learning_rate=recipe["learning_rate"],
        lr_scheduler_type="cosine", warmup_steps=max(1, round(recipe["updates"] * recipe["warmup_fraction"])),
        optim="adamw_torch", weight_decay=recipe["weight_decay"], max_grad_norm=1,
        logging_steps=16, eval_strategy="steps", eval_steps=128, save_strategy="steps", save_steps=64,
        save_total_limit=2, bf16=True, report_to=[], seed=recipe["training_seed"], remove_unused_columns=False)
    trainer = Trainer(model=model, args=config, train_dataset=train_dataset, eval_dataset=eval_dataset,
                      data_collator=make_collator(tokenizer.pad_token_id), processing_class=tokenizer, callbacks=[callback])
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    elapsed = prior_elapsed + time.monotonic() - started
    complete = trainer.state.global_step == recipe["updates"] and not callback.budget_stop and elapsed < recipe["max_training_hours"] * 3600
    if directory_sha256(frozen) != r["frozen_sft_adapter_sha256"]:
        raise ValueError("frozen SFT was modified")
    report = {**registered, "status": "completed" if complete else "stopped_budget", "completed_updates": trainer.state.global_step,
              "num_train_rows": len(train_rows), "num_eval_rows": len(eval_rows), "wall_clock_seconds": elapsed,
              "output_adapter_sha256": directory_sha256(out / "adapter"), "frozen_sft_unchanged": True,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "estimated_training_cost_usd": elapsed / 3600 * limits["hourly_usd"],
              "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_write_json(manifest_path, report)
    events.emit("job_completed" if complete else "budget_stop", phase="recovery_sft", current=trainer.state.global_step, total=recipe["updates"], metrics={"wall_clock_seconds": elapsed})
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        record_run_failure(ACTIVE_MANIFEST, error)
        raise
