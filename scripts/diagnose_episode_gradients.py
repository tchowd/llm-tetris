#!/usr/bin/env python3
"""Compare replayed GPU gradients without applying optimizer updates."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.train_episode_rl import sequence_logprobs, flatten_steps, adapter_parameter_hash
from tetris.rl import atomic_write_json, grouped_policy_loss


def main():
    out = Path(sys.argv[1])
    if out.exists():
        raise ValueError("retain prior diagnostics")
    root = Path("runs/rl-r2-episode-proof-seed0/rl")
    batch = json.loads((root / "trajectory_batches/update-000001.json").read_text())
    rows = flatten_steps(batch["trajectories"])
    rev = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=rev)
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", revision=rev, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, "runs/sft-v1/adapter", is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    before = adapter_parameter_hash(model)
    passes, gradients = [], []
    for deterministic in (False, False, True, True):
        torch.use_deterministic_algorithms(deterministic)
        model.zero_grad(set_to_none=True)
        tick = time.time()
        loss_value = 0
        for offset in range(0, len(rows), 4):
            chunk = rows[offset:offset + 4]
            values = sequence_logprobs(model, chunk, tokenizer.pad_token_id, 1, return_tokens=True)
            losses = [grouped_policy_loss(v, torch.tensor(r["reference_token_logprobs"], device=v.device), torch.full_like(v, r["advantage"]), .05) for r, v in zip(chunk, values)]
            loss = torch.stack(losses).sum() / len(rows)
            loss.backward()
            loss_value += float(loss.detach())
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1))
        gradients.append([p.grad.cpu().clone() for p in params])
        passes.append({"deterministic": deterministic, "loss": loss_value, "norm": norm, "seconds": time.time() - tick})
        print(passes[-1], flush=True)
    def compare(a, b):
        return {"equal": all(torch.equal(x, y) for x, y in zip(a, b)), "max_error": max(float((x-y).abs().max()) for x,y in zip(a,b))}
    saved = torch.load(root / "checkpoint-1/state.pt", map_location="cpu", weights_only=False)["optimizer"]["state"]
    expected = [saved[i]["exp_avg"] / .1 for i in saved]
    atomic_write_json(out, {"status": "diagnostic_only", "passes": passes,
        "default_repeat": compare(gradients[0], gradients[1]), "deterministic_repeat": compare(gradients[2], gradients[3]),
        "vs_saved_clipped_gradients": [compare(g, expected) for g in gradients],
        "adapter_unchanged": adapter_parameter_hash(model) == before, "optimizer_updates": 0,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    print(out.read_text(), flush=True)


if __name__ == "__main__":
    main()
