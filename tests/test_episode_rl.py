from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from scripts.train_episode_rl import episode_progress, load_checkpoint, save_checkpoint, sequence_logprobs, trim_completion
from tetris.rl import check_resume_registration


def test_trim_completion_keeps_eos_even_when_it_is_also_padding():
    assert trim_completion([1, 2, 9, 9], eos_id=9, pad_id=9) == [1, 2, 9]


def test_episode_execution_rejects_nondeterministic_workspace(monkeypatch):
    torch = pytest.importorskip("torch")
    from scripts.train_episode_rl import configure_execution
    previous = torch.are_deterministic_algorithms_enabled()
    precision = torch.get_float32_matmul_precision()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(ValueError, match="CUBLAS"):
        configure_execution()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    try:
        assert configure_execution()["deterministic_algorithms"]
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.set_float32_matmul_precision(precision)


def test_qwen_completion_suffix_matches_full_logits_and_gradients():
    torch = pytest.importorskip("torch")
    from scripts.check_episode_proof import independent_token_logprobs

    class Model(torch.nn.Module):
        def __init__(self, optimized):
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3" if optimized else "other")
            self.table = torch.nn.Parameter(torch.arange(60, dtype=torch.float32).reshape(12, 5) / 10)
            self.kept = None

        def forward(self, input_ids, attention_mask, logits_to_keep=0, use_cache=None):
            self.kept = logits_to_keep
            values = self.table[:input_ids.shape[1]].unsqueeze(0).expand(input_ids.shape[0], -1, -1)
            return SimpleNamespace(logits=values[:, -logits_to_keep:] if logits_to_keep else values)

    rows = [{"prompt_ids": [1]*8, "completion_ids": [2, 3]}, {"prompt_ids": [1]*6, "completion_ids": [4]}]
    full, suffix = Model(False), Model(True)
    a, b = sequence_logprobs(full, rows, 0), sequence_logprobs(suffix, rows, 0)
    assert suffix.kept == 5
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.sum().backward(); b.sum().backward()
    torch.testing.assert_close(full.table.grad, suffix.table.grad, rtol=0, atol=0)
    audited = independent_token_logprobs(suffix, rows, 0, 1)
    torch.testing.assert_close(torch.stack([x.mean() for x in audited]), b, rtol=0, atol=1e-7)


def test_resume_cannot_change_registered_configuration():
    with pytest.raises(ValueError, match="resume changes"):
        check_resume_registration({"gamma": 0.99}, {"gamma": 0.9}, ["gamma"])


def test_logprobs_align_with_exact_completion_positions_and_tokens():
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.table[: input_ids.shape[1]].unsqueeze(0).expand(input_ids.shape[0], -1, -1))

    model = TinyModel()
    rows = [
        {"prompt_ids": [1, 2], "completion_ids": [3, 0]},
        {"prompt_ids": [1], "completion_ids": [2]},
    ]
    actual = sequence_logprobs(model, rows, pad_token_id=0)
    expected_table = torch.log_softmax(model.table, dim=-1)
    assert actual[0].item() == pytest.approx((expected_table[1, 3] + expected_table[2, 0]).item() / 2)
    assert actual[1].item() == pytest.approx(expected_table[0, 2].item())
    actual.sum().backward()
    assert model.table.grad is not None


def test_checkpoint_restores_adapter_optimizer_scheduler_rng_and_accounting(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(12)
    base = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16, n_positions=16))
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
    rng = random.Random(44)
    loss = sum(parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad)
    loss.backward()
    optimizer.step()
    scheduler.step()
    checkpoint = tmp_path / "checkpoint-1"
    saved_parameters = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    history = [{"update": 1, "turns": 7, "seconds": 2.0}]
    save_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, update=1, samples=7, rng=rng, update_metrics=history)
    expected_python = rng.random()
    expected_torch = torch.rand(3)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(10)
    scheduler.step()
    restored_history = []
    update, samples = load_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, rng=rng, update_metrics=restored_history)
    assert (update, samples) == (1, 7)
    assert scheduler.last_epoch == 1
    assert restored_history == history
    assert rng.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)
    for name, parameter in model.named_parameters():
        if name in saved_parameters:
            assert torch.equal(parameter, saved_parameters[name])


def test_bfloat16_logits_are_normalized_in_float32_only_at_completion_positions(monkeypatch):
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.arange(40, dtype=torch.bfloat16).reshape(10, 4) / 10)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.table[:input_ids.shape[1]].unsqueeze(0))

    original = torch.log_softmax
    normalized_shapes = []

    def capture(tensor, *args, **kwargs):
        normalized_shapes.append((tuple(tensor.shape), tensor.dtype))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "log_softmax", capture)
    model = TinyModel()
    rows = [{"prompt_ids": [1] * 8, "completion_ids": [2, 3]}]
    actual = sequence_logprobs(model, rows, pad_token_id=0, return_tokens=True)[0]
    assert normalized_shapes == [((2, 4), torch.float32)]
    expected = original(model.table[7:9].float(), dim=-1)[[0, 1], [2, 3]]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert model.table.grad is not None


def test_episode_progress_projects_remaining_work_and_distinguishes_pause_from_completion():
    common = dict(total=10, hourly_usd=1.05, max_hours=1.0, dollar_limit=20.0)
    assert episode_progress(completed=2, elapsed=200, update_metrics=[], pause_after=2, **common)["status"] == "paused"
    decision = episode_progress(completed=3, elapsed=1200, update_metrics=[{"seconds": 400}] * 3, **common)
    assert decision["runtime_projection"]["projected_seconds"] == 4000
    assert decision["status"] == "stopped_budget"
    assert episode_progress(completed=10, elapsed=3601, update_metrics=[], **common)["status"] == "stopped_budget"
    assert episode_progress(completed=10, elapsed=3000, update_metrics=[], **common)["status"] == "completed"


@pytest.mark.parametrize("method", ["active_group", "fixed_zero"])
def test_episode_main_pause_resume_matches_uninterrupted_weights_and_samples(tmp_path, monkeypatch, method):
    """Exercise the real optimizer/checkpoint/control loop on a tiny CPU model."""
    import json
    import sys

    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
    import scripts.train_episode_rl as runner

    class CpuModel(GPT2LMHeadModel):
        def to(self, *args, **kwargs):
            if args and args[0] == "cuda":
                return self
            return super().to(*args, **kwargs)

    def make_base(*args, **kwargs):
        with torch.random.fork_rng():
            torch.manual_seed(777)
            return CpuModel(GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16, n_positions=16))

    frozen = tmp_path / "frozen" / "adapter"
    get_peft_model(make_base(), LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM")).save_pretrained(frozen)
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({"training_seeds": [123, 456], "stage5_seeds": [789]}))
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=15, save_pretrained=lambda path: None)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", make_base)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner, "validate_seed_manifest", lambda *a, **kw: {"passed": True})
    monkeypatch.setattr(runner, "validate_entry_gate", lambda *a, **kw: {"passed": True})

    def sample_tiny(policy, reference, tokenizer, **kwargs):
        rows = [{"prompt_ids": [1, 2], "completion_ids": [int(torch.randint(1, 15, ()).item())]} for _ in range(2)]
        with torch.no_grad():
            reference_tokens = sequence_logprobs(reference, rows, 0, return_tokens=True)
        return [{"episode_return": reward, "steps": [{
            **row, "immediate_reward": reward, "parsed": True, "legal": True,
            "action": [0, index], "reference_token_logprobs": tokens.tolist(),
        }]} for index, (row, tokens, reward) in enumerate(zip(rows, reference_tokens, [1.0, -1.0]))]

    monkeypatch.setattr(runner, "sample_group", sample_tiny)

    def invoke(path, *extra):
        monkeypatch.setattr(sys, "argv", ["train_episode_rl.py", "--experiment", "E5", "--question", "CPU resume proof",
            "--adapter-dir", str(frozen), "--frozen-sft-adapter-dir", str(frozen), "--benchmark-manifest", str(benchmark),
            "--out-dir", str(path), "--training-seed", "12", "--updates", "4", "--group-size", "2",
            "--save-every", "3", "--learning-rate", "0.01", "--instance-hourly-usd", "1.05",
            "--max-wall-clock-hours", "1", "--advantage-method", method, *extra])
        runner.main()
        return json.loads((path / "manifest.json").read_text())

    full, resumed = tmp_path / "full", tmp_path / "resumed"
    expected = invoke(full)
    paused = invoke(resumed, "--pause-after-update", "2")
    assert paused["status"] == "paused"
    assert paused["completed_updates"] == 2
    assert (resumed / "checkpoint-2" / "state.pt").exists()  # save-every=3 must not suppress the forced save.
    actual = invoke(resumed, "--resume", str(resumed / "checkpoint-2"))
    assert expected["status"] == actual["status"] == "completed"
    assert actual["resumed_from_update"] == 2
    assert actual["reference_weights_sha256_before"] == actual["reference_weights_sha256_after"]
    assert actual["completed_updates"] == 4
    assert actual["sample_count"] == expected["sample_count"] == 8
    assert [row["update"] for row in actual["update_metrics"]] == [1, 2, 3, 4]
    for update in range(1, 5):
        filename = f"update-{update:06d}.json"
        assert json.loads((full / "trajectory_batches" / filename).read_text()) == json.loads((resumed / "trajectory_batches" / filename).read_text())
    expected_weights = load_file(str(full / "adapter" / "adapter_model.safetensors"))
    actual_weights = load_file(str(resumed / "adapter" / "adapter_model.safetensors"))
    for name in expected_weights:
        torch.testing.assert_close(actual_weights[name], expected_weights[name], rtol=0, atol=0)
