"""The one serializer. Its output is what SFT and eval will see, so it must
not be duplicated anywhere else in the codebase (see stage-1-game.md)."""
from __future__ import annotations

import re

_ACTION_RE = re.compile(r"^Action: rot=([0-3]) x=([0-9])$")


def serialize_prompt(state: dict) -> str:
    heights = " ".join(str(v) for v in state["heights"])
    holes = " ".join(str(v) for v in state["holes"])
    wells = " ".join(str(v) for v in state["wells"])
    return (
        f"Piece: {state['piece']}\n"
        f"Next: {state['next']}\n"
        f"Heights: {heights}\n"
        f"Holes:   {holes}\n"
        f"Wells:   {wells}\n"
        f"Bumpiness: {state['bumpiness']}"
    )


def serialize_action(rot: int, x: int) -> str:
    return f"Action: rot={rot} x={x}"


def parse_action(text: str) -> tuple[int, int]:
    """Inverse of `serialize_action`. Rejects anything that doesn't match
    the exact contract -- a model that free-forms text is a parse failure,
    not something to guess at (stage-4-sft.md's parse-rate metric)."""
    match = _ACTION_RE.match(text.strip())
    if not match:
        raise ValueError(f"malformed action: {text!r}")
    return int(match.group(1)), int(match.group(2))
