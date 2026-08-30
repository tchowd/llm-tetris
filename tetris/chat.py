"""The one chat-template assembler (Stage 4). `serialize_prompt` (Stage 1)
is still not what the model sees once the chat template wraps it -- that
wrapper is a second formatter unless it lives in exactly one place, used by
training and by eval alike. See plan/stage-4-sft.md, "The one prompt
assembler".
"""
from __future__ import annotations

SYSTEM_PROMPT = "You are a Tetris engine. Reply with exactly one line: Action: rot=<0-3> x=<0-9>"


def build_messages(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def build_generation_prompt(tokenizer, prompt: str) -> str:
    """The exact string handed to the model at generation time. Deriving it
    from `add_generation_prompt=True` -- rather than hand-writing the
    assistant header -- is what makes Qwen3's empty `<think></think>` block
    (emitted even with `enable_thinking=False`) agree by construction with
    whatever training builds on top of this same string."""
    return tokenizer.apply_chat_template(
        build_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_training_example(tokenizer, prompt: str, completion: str) -> dict:
    """Tokenize one (prompt, completion) row for SFT.

    `input_ids` covers the full sequence (generation prefix + completion +
    EOS); `labels` masks every prefix token to -100 so the loss falls only
    on the completion, matching stage-4-sft.md's "Loss" row.
    """
    prefix = build_generation_prompt(tokenizer, prompt)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    completion_ids = completion_ids + [tokenizer.eos_token_id]
    return {
        "input_ids": prefix_ids + completion_ids,
        "labels": [-100] * len(prefix_ids) + completion_ids,
        "prefix": prefix,
    }
