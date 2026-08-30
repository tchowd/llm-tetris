#!/usr/bin/env python3
"""Stage 4 open-loop eval: parse rate, legality rate, exact match, and value
match on held-out (`split == "eval"`) rows, using the trained LoRA adapter
to generate real completions -- not teacher-forced loss. See
plan/stage-4-sft.md's "Metrics" section.

    python scripts/eval_open_loop.py --data-dirs data/batch1 data/batch2 \
        --adapter-dir runs/sft-v1/adapter --out runs/sft-v1/open_loop_metrics.json

This is still open-loop: every board here was produced by the *teacher*,
not the model. It tells you whether the model copies the teacher on
teacher-distribution states. It says nothing about whether the model
survives boards of its own making -- that is Stage 5.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tetris.board import board_to_lists
from tetris.chat import build_generation_prompt
from tetris.placement import legal_placements_on
from tetris.serialize import parse_action
from tetris.teacher import WEIGHTS as LIVE_WEIGHTS
from tetris.teacher import value_of_placement


def load_eval_rows(data_dirs: list[Path], max_rows: int | None, seed: int) -> list[dict]:
    rows = []
    for data_dir in data_dirs:
        with (data_dir / "rows.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                if row["split"] == "eval":
                    rows.append(row)
    if max_rows is not None and len(rows) > max_rows:
        rows = random.Random(seed).sample(rows, max_rows)
    return rows


def resolve_weights(data_dirs: list[Path]) -> dict:
    for data_dir in data_dirs:
        manifest_path = data_dir / "manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text())["teacher_weights"]
    return LIVE_WEIGHTS


def holes_bucket(holes_total: int) -> str:
    if holes_total == 0:
        return "0"
    if holes_total <= 2:
        return "1-2"
    if holes_total <= 5:
        return "3-5"
    return "6+"


def score_row(row: dict, generated_text: str, weights: dict) -> dict:
    line = generated_text.strip().splitlines()[0] if generated_text.strip() else ""
    board = board_to_lists(row["board"])
    legal = legal_placements_on(board, row["piece"])
    legal_pairs = {(p["rot"], p["x"]): p for p in legal}
    best_value = max(value_of_placement(board, p, row["next"], weights) for p in legal)

    try:
        parsed = parse_action(line)
    except ValueError:
        parsed = None

    is_legal = parsed in legal_pairs if parsed is not None else False
    exact_match = parsed == (row["rot"], row["x"]) if parsed is not None else False
    value_match = False
    if is_legal:
        model_value = value_of_placement(board, legal_pairs[parsed], row["next"], weights)
        value_match = math.isclose(model_value, best_value, rel_tol=1e-9, abs_tol=1e-9)

    return {
        "piece": row["piece"],
        "holes_total": row["holes_total"],
        "parsed": parsed is not None,
        "legal": is_legal,
        "exact_match": exact_match,
        "value_match": value_match,
    }


def _bucket_stats(bucket: list[dict]) -> dict:
    m = len(bucket)
    return {
        "n": m,
        "parse_rate": sum(b["parsed"] for b in bucket) / m,
        "legality_rate": sum(b["legal"] for b in bucket) / m,
        "exact_match": sum(b["exact_match"] for b in bucket) / m,
        "value_match": sum(b["value_match"] for b in bucket) / m,
    }


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    by_piece = defaultdict(list)
    by_holes = defaultdict(list)
    for r in results:
        by_piece[r["piece"]].append(r)
        by_holes[holes_bucket(r["holes_total"])].append(r)

    overall = _bucket_stats(results) if results else {"n": 0, "parse_rate": 0.0, "legality_rate": 0.0, "exact_match": 0.0, "value_match": 0.0}
    return {
        **overall,
        "n": n,
        "by_piece": {p: _bucket_stats(rs) for p, rs in sorted(by_piece.items())},
        "by_holes_bucket": {h: _bucket_stats(rs) for h, rs in sorted(by_holes.items())},
    }


def generate_actions(tokenizer, model, prompts: list[str], device: str, batch_size: int) -> list[str]:
    texts = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, padding_side="left").to(device)
            out = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            texts.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    rows = load_eval_rows(args.data_dirs, args.max_rows, args.seed)
    print(f"evaluating on {len(rows)} held-out rows from {args.data_dirs}")
    weights = resolve_weights(args.data_dirs)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, str(args.adapter_dir)).to(device)
    model.eval()

    prompts = [build_generation_prompt(tokenizer, r["prompt"]) for r in rows]
    generated = generate_actions(tokenizer, model, prompts, device, args.batch_size)
    results = [score_row(row, text, weights) for row, text in zip(rows, generated)]

    report = aggregate(results)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
