#!/usr/bin/env python3
"""Stage 4: LoRA SFT -- clone the Stage 2 teacher into Qwen/Qwen3-1.7B.

    python scripts/train_sft.py --data-dirs data/batch1 data/batch2 --out-dir runs/sft-v1

Requires the training extras (torch/transformers/peft/accelerate), which is
why this script lives outside the base project's dependency set -- see
plan/stage-4-sft.md. Every text the model sees is built by
`tetris.chat.build_training_example`, the same assembler eval will use, so
there is exactly one place a prompt/completion pair turns into tokens.

Loss falls only on the completion + EOS (prompt and the `<think></think>`
generation prefix are masked to -100). No packing: each row is one padded
example, since rows are short (~130-150 tokens) and packing would blur the
completion mask for no throughput that matters at this scale.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from tetris.chat import build_training_example
from tetris.serialize import parse_action

BASE_MODEL = "Qwen/Qwen3-1.7B"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(data_dirs: list[Path], split: str) -> list[dict]:
    """Stream just the fields SFT needs. `game_id` is kept only so a caller
    can double-check no-leakage; the full board/features live in
    rows.jsonl for debugging, not here."""
    rows = []
    for data_dir in data_dirs:
        with (data_dir / "rows.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                if row["split"] != split:
                    continue
                rows.append(
                    {
                        "game_id": row["game_id"],
                        "prompt": row["prompt"],
                        "completion": row["completion"],
                        "rot": row["rot"],
                        "x": row["x"],
                    }
                )
    return rows


def subsample(rows: list[dict], max_rows: int | None, seed: int) -> list[dict]:
    if max_rows is None or len(rows) <= max_rows:
        return rows
    return random.Random(seed).sample(rows, max_rows)


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer):
        self.examples = [build_training_example(tokenizer, r["prompt"], r["completion"]) for r in rows]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def make_collator(pad_token_id: int):
    def collate(batch: list[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, ex in enumerate(batch):
            n = len(ex["input_ids"])
            input_ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


class GenerationExactMatchCallback(TrainerCallback):
    """Stage-4-sft.md's "Eval during training" row asks for exact match,
    not just teacher-forced loss -- loss can fall while the model just
    learns formatting. Runs real greedy generation (the eval-time decoding
    settings: greedy, stop at EOS, max_new_tokens=16) on a small fixed
    sample so this stays cheap enough to run every eval_steps.
    """

    def __init__(self, tokenizer, sample_rows: list[dict], batch_size: int = 32):
        self.tokenizer = tokenizer
        self.sample_rows = sample_rows
        self.batch_size = batch_size

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or not self.sample_rows:
            return
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        correct = 0
        parsed = 0
        with torch.no_grad():
            for i in range(0, len(self.sample_rows), self.batch_size):
                batch = self.sample_rows[i : i + self.batch_size]
                prefixes = [r["prefix"] for r in batch]
                enc = self.tokenizer(prefixes, return_tensors="pt", padding=True, padding_side="left").to(device)
                out = model.generate(
                    **enc,
                    max_new_tokens=16,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                new_tokens = out[:, enc["input_ids"].shape[1] :]
                texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                for row, text in zip(batch, texts):
                    line = text.strip().splitlines()[0] if text.strip() else ""
                    try:
                        rot, x = parse_action(line)
                        parsed += 1
                        if (rot, x) == (row["rot"], row["x"]):
                            correct += 1
                    except ValueError:
                        pass
        if was_training:
            model.train()
        n = len(self.sample_rows)
        metrics = {
            "eval_gen_parse_rate": parsed / n,
            "eval_gen_exact_match": correct / n,
        }
        print(f"[step {state.global_step}] generation eval: {metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=2000)
    parser.add_argument("--gen-eval-rows", type=int, default=64, help="rows for the periodic generation-based exact-match check")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=-1, help="override; -1 derives from --epochs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", default=None, help="default: mps if available, else cpu")
    args = parser.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    random.seed(args.seed)

    print("loading rows...")
    train_rows = subsample(load_rows(args.data_dirs, "train"), args.max_train_rows, args.seed)
    eval_rows_all = load_rows(args.data_dirs, "eval")
    eval_rows = subsample(eval_rows_all, args.max_eval_rows, args.seed)
    print(f"train rows: {len(train_rows)}, eval rows (of {len(eval_rows_all)} total): {len(eval_rows)}")

    train_game_ids = {r["game_id"] for r in train_rows}
    eval_game_ids = {r["game_id"] for r in eval_rows_all}
    leak = train_game_ids & eval_game_ids
    if leak:
        raise SystemExit(f"train/eval game_id leakage: {len(leak)} games, e.g. {sorted(leak)[:5]}")

    print(f"loading tokenizer + base model {args.base_model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    model.to(device)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("tokenizing...")
    train_dataset = SFTDataset(train_rows, tokenizer)
    eval_dataset = SFTDataset(eval_rows, tokenizer) if eval_rows else None

    gen_eval_sample = random.Random(args.seed).sample(eval_rows, min(args.gen_eval_rows, len(eval_rows))) if eval_rows else []
    for row in gen_eval_sample:
        row["prefix"] = build_training_example(tokenizer, row["prompt"], row["completion"])["prefix"]

    steps_per_epoch = math.ceil(len(train_dataset) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = max(1, round(0.03 * total_steps))
    print(f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        max_steps=total_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=(device != "cpu"),
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    callbacks = []
    if gen_eval_sample:
        callbacks.append(GenerationExactMatchCallback(tokenizer, gen_eval_sample))

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=make_collator(tokenizer.pad_token_id),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"training done in {elapsed:.1f}s")

    adapter_dir = args.out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    manifest = {
        "base_model": args.base_model,
        "data_dirs": [str(d) for d in args.data_dirs],
        "num_train_rows": len(train_rows),
        "num_eval_rows": len(eval_rows),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "seed": args.seed,
        "device": device,
        "wall_clock_seconds": elapsed,
    }
    (args.out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
