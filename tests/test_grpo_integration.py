"""A tiny, offline TRL integration test; no model download or GPU needed."""
from __future__ import annotations

import pytest


def tiny_trainer(tmp_path, *, max_steps=1, callbacks=None):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer

    torch.manual_seed(123)
    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "board": 3, "state": 4, "left": 5, "right": 6}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.padding_side = "left"
    base = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=16, vocab_size=len(vocab), n_positions=32, eos_token_id=1, pad_token_id=0))
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    dataset = Dataset.from_dict({"prompt": ["board state"] * 4})

    completions = []

    def reward_func(completion_ids, **kwargs):
        completions.extend([list(row) for row in completion_ids])
        return [float(5 in row) for row in completion_ids]

    config = GRPOConfig(
        output_dir=str(tmp_path),
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=3,
        beta=0.05,
        loss_type="grpo",
        scale_rewards="group",
        use_cpu=True,
        bf16=False,
        report_to="none",
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        disable_tqdm=True,
        seed=123,
    )
    trainer = GRPOTrainer(model=model, args=config, reward_funcs=reward_func, train_dataset=dataset, processing_class=tokenizer, callbacks=callbacks)
    return trainer, completions


def test_trl_copies_and_preserves_pretrained_adapter_reference(tmp_path):
    from scripts.train_rl import reference_hash

    trainer, _ = tiny_trainer(tmp_path)
    before = reference_hash(trainer.model)
    trainer.train()
    after = reference_hash(trainer.model)
    assert trainer.state.global_step == 1
    assert before == after


def test_trl_resume_matches_uninterrupted_updates_and_samples(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from transformers import TrainerCallback
    from scripts.train_rl import reference_hash, restore_policy_for_resume

    class PauseAtOne(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_training_stop = True
                control.should_save = True
            return control

    first, first_samples = tiny_trainer(tmp_path / "paused", max_steps=2, callbacks=[PauseAtOne()])
    frozen_hash = reference_hash(first.model)
    first.train()
    assert first.state.global_step == 1
    checkpoint = tmp_path / "paused" / "checkpoint-1"
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
        assert (checkpoint / name).is_file()

    optimizer_verified = []

    class CheckRestoredOptimizer(TrainerCallback):
        def on_train_begin(self, args, state, control, optimizer=None, model=None, **kwargs):
            expected = torch.load(checkpoint / "optimizer.pt", weights_only=True)
            actual = optimizer.state_dict()
            assert actual["param_groups"] == expected["param_groups"]
            for parameter_id, values in expected["state"].items():
                for key, value in values.items():
                    torch.testing.assert_close(actual["state"][parameter_id][key], value, rtol=0, atol=0)
            optimizer_verified.append(True)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    torch.testing.assert_close(parameter, dict(first.model.named_parameters())[name], rtol=0, atol=0)

    resumed, resumed_samples = tiny_trainer(tmp_path / "resumed", max_steps=2, callbacks=[CheckRestoredOptimizer()])
    restore_policy_for_resume(resumed.model, checkpoint)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    full, full_samples = tiny_trainer(tmp_path / "full", max_steps=2)
    full.train()
    assert resumed.state.global_step == full.state.global_step == 2
    assert reference_hash(resumed.model) == reference_hash(full.model) == frozen_hash
    assert first_samples + resumed_samples == full_samples
    assert resumed.lr_scheduler.state_dict() == full.lr_scheduler.state_dict()
    assert optimizer_verified == [True]
    for name, parameter in resumed.model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, dict(full.model.named_parameters())[name], rtol=0, atol=0)
