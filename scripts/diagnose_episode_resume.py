#!/usr/bin/env python3
"""Read-only GPU diagnosis of the retained failed R2; never train or pass its gate."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.check_episode_proof import independent_token_logprobs
from tetris.rl import atomic_write_json, file_sha256


def main():
    out = Path(sys.argv[1])
    if out.exists():
        raise ValueError("retain earlier diagnostics")
    roots = [Path("runs") / name / "rl" for name in (
        "rl-r2-episode-proof-seed0", "rl-r2-episode-proof-uninterrupted-seed0")]
    checkpoint_checks = []
    for update in (1, 2):
        a, b = [load_file(str(root / f"checkpoint-{update}/adapter/adapter_model.safetensors")) for root in roots]
        checkpoint_checks.append({"update": update, "equal": set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a),
            "max_error": max(float((a[k] - b[k]).abs().max()) for k in a)})
    revision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", revision=revision, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, "runs/sft-v1/adapter", is_trainable=True)
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    state = load_file(str(roots[0] / "checkpoint-2/adapter/adapter_model.safetensors"))
    set_peft_model_state_dict(model, state, adapter_name="default")
    actual = get_peft_model_state_dict(model)
    reload = {"equal": set(actual) == set(state) and all(torch.equal(actual[k].cpu(), state[k]) for k in state),
        "max_error": max(float((actual[k].cpu() - state[k]).abs().max()) for k in state),
        "dtypes": sorted({str(v.dtype) for v in actual.values()})}
    batches = [json.loads((root / "trajectory_batches/update-000003.json").read_text()) for root in roots]
    errors = []
    with torch.no_grad():
        for turn in range(3):
            rows = [t["steps"][turn] for t in batches[0]["trajectories"]]
            tokens = independent_token_logprobs(model, rows, tokenizer.pad_token_id, 1)
            errors.append({"turn": turn, "vs_resumed": max(float((v - torch.tensor(r["policy_token_logprobs_at_sampling"], device=v.device)).abs().max()) for r, v in zip(rows, tokens)),
                "vs_control": max(float((v - torch.tensor(t["steps"][turn]["policy_token_logprobs_at_sampling"], device=v.device)).abs().max()) for t, v in zip(batches[1]["trajectories"], tokens))})
    atomic_write_json(out, {"status": "diagnostic_only", "checkpoint_comparison": checkpoint_checks,
        "reload": reload, "token_errors": errors, "model_config_use_cache": model.config.use_cache,
        "attention_implementation": model.config._attn_implementation,
        "source_sha256": file_sha256(Path(__file__)), "training_updates_performed": 0,
        "original_r2_status": "failed", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(out.read_text())


if __name__ == "__main__":
    main()
