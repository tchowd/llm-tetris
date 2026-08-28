"""The one serializer. Its output is what SFT and eval will see, so it must
not be duplicated anywhere else in the codebase (see stage-1-game.md)."""
from __future__ import annotations


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
