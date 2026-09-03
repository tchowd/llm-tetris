"""Lazy model-policy loader shared by Stage 5 and Stage 6 evaluation."""
from __future__ import annotations

from pathlib import Path


def build_model_policy(adapter_dir: Path, base_model: str, device: str, *, revision: str | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .chat import build_generation_prompt
    from .serialize import parse_action

    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, revision=revision).to(device)
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

    pick.metadata = {"base_model_revision": getattr(base.config, "_commit_hash", None)}
    return pick
