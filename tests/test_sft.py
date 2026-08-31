"""Stage 4 tests (plan/stage-4-sft.md's "Tests" section -- the stage gate).

Requires the training extras (torch/transformers/peft), which live outside
the base project's dependency set -- see scripts/train_sft.py's docstring
for why (this repo's default .venv is Python 3.14, which has no PyTorch
wheels yet). Run these from the training venv:

    .venv-train/bin/python -m pytest tests/test_sft.py -v

Under the base .venv, `pytest.importorskip` below makes this file skip
cleanly instead of erroring the rest of the suite.
"""
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from tetris.chat import build_generation_prompt, build_training_example
from tetris.dataset import generate_game
from tetris.serialize import parse_action

BASE_MODEL = "Qwen/Qwen3-1.7B"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Seeds 0..99 include at least two eval-split games (63 and 94, per
# split_for_game_id) -- checked once with tetris.dataset.split_for_game_id
# so the no-leakage test below always has both splits to compare, without
# depending on any generated dataset dump existing on disk.
FIXTURE_SEEDS = range(100)
FIXTURE_MAX_PIECES = 40


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@pytest.fixture(scope="module")
def all_rows() -> list[dict]:
    rows = []
    for seed in FIXTURE_SEEDS:
        game_rows, _ = generate_game(seed, max_pieces=FIXTURE_MAX_PIECES)
        rows.extend(game_rows)
    return rows


@pytest.fixture(scope="module")
def train_rows(all_rows) -> list[dict]:
    rows = [r for r in all_rows if r["split"] == "train"]
    assert len(rows) >= 200
    return rows


@pytest.fixture(scope="module")
def eval_rows(all_rows) -> list[dict]:
    rows = [r for r in all_rows if r["split"] == "eval"]
    assert rows
    return rows


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL)


def test_template_agreement(tokenizer, all_rows):
    """The training prefix for a row must be byte-identical to the
    eval-time generation prompt for the same row (stage-4-sft.md test #1)."""
    for row in all_rows[:5]:
        example = build_training_example(tokenizer, row["prompt"], row["completion"])
        gen_prompt = build_generation_prompt(tokenizer, row["prompt"])
        assert example["prefix"] == gen_prompt
        prefix_ids = tokenizer(gen_prompt, add_special_tokens=False)["input_ids"]
        assert example["input_ids"][: len(prefix_ids)] == prefix_ids


def test_mask_correctness(tokenizer, all_rows):
    """Unmasked label positions decode to exactly the completion (plus EOS)
    and nothing else (test #2)."""
    for row in all_rows[:5]:
        example = build_training_example(tokenizer, row["prompt"], row["completion"])
        for token_id, label in zip(example["input_ids"], example["labels"]):
            assert label in (-100, token_id)
        unmasked = [tid for tid, label in zip(example["input_ids"], example["labels"]) if label != -100]
        assert unmasked[-1] == tokenizer.eos_token_id
        decoded = tokenizer.decode(unmasked[:-1])
        assert decoded == row["completion"]


def test_determinism(tokenizer, all_rows):
    """Greedy decoding on the same prompt twice gives the same string
    (test #6)."""
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).to(_device())
    model.eval()
    row = all_rows[0]
    prompt = build_generation_prompt(tokenizer, row["prompt"])
    enc = tokenizer(prompt, return_tensors="pt").to(_device())
    outputs = []
    for _ in range(2):
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outputs.append(tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True))
    assert outputs[0] == outputs[1]


def test_no_leakage_between_train_and_eval_rows(train_rows, eval_rows):
    """No eval `game_id` appears in the training file (test #5). Checked
    here against this module's generated fixture; the real dump is checked
    at dump-scale by tetris.dataset_validate.check_split_purity, and
    train_sft.py refuses to start if its own loaded train/eval sets share a
    game_id."""
    train_ids = {r["game_id"] for r in train_rows}
    eval_ids = {r["game_id"] for r in eval_rows}
    assert train_ids and eval_ids
    assert not (train_ids & eval_ids)


def test_overfit_200_rows_reaches_near_perfect_exact_match(tokenizer, train_rows):
    """stage-4-sft.md test #4: "Train on 200 rows for a few hundred steps
    and reach ~100% exact match on those same rows. If it cannot memorize
    200 cards, the loss mask or the labels are wrong -- find out in two
    minutes, not after a full run." Uses a higher LR than the production
    recipe purely to converge fast; this is a plumbing check on the loss
    mask, not a hyperparameter validation.
    """
    rows = train_rows[:200]
    assert len(rows) == 200

    device = _device()
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(device)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    examples = [build_training_example(tokenizer, r["prompt"], r["completion"]) for r in rows]
    pad_id = tokenizer.pad_token_id

    def batch_of(indices: list[int]):
        batch = [examples[i] for i in indices]
        max_len = max(len(e["input_ids"]) for e in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, e in enumerate(batch):
            n = len(e["input_ids"])
            input_ids[i, :n] = torch.tensor(e["input_ids"])
            attn[i, :n] = 1
            labels[i, :n] = torch.tensor(e["labels"])
        return input_ids.to(device), attn.to(device), labels.to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    batch_size, steps = 25, 450
    rng = random.Random(0)
    for step in range(steps):
        indices = rng.sample(range(len(rows)), batch_size)
        input_ids, attn, labels = batch_of(indices)
        opt.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        out.loss.backward()
        opt.step()
        if step % 20 == 0 or step == steps - 1:
            print(f"[overfit check] step {step}: loss={out.loss.item():.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            batch_rows = rows[i : i + 32]
            prompts = [build_generation_prompt(tokenizer, r["prompt"]) for r in batch_rows]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)
            out = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=pad_id)
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for row, text in zip(batch_rows, texts):
                line = text.strip().splitlines()[0] if text.strip() else ""
                try:
                    parsed = parse_action(line)
                except ValueError:
                    parsed = None
                correct += parsed == (row["rot"], row["x"])

    exact_match = correct / len(rows)
    assert exact_match >= 0.95, f"overfit check: only {exact_match:.2%} exact match on 200 memorized rows"
