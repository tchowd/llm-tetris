"""Pure helpers for the Tetra-1.7B progressive SFT curriculum."""
from __future__ import annotations

import hashlib
import json

from .board import board_to_lists
from .placement import legal_placements_on
from .teacher import value_of_placement


def difficulty_band(max_height: int, holes: int) -> str:
    if max_height >= 16 or holes >= 6:
        return "critical"
    if max_height >= 13 or holes >= 3:
        return "hard"
    if max_height >= 8 or holes >= 1:
        return "moderate"
    return "low"


def board_hash(board: list[str]) -> str:
    return hashlib.sha256("\n".join(board).encode()).hexdigest()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def state_hash_from_snapshot(snapshot: dict) -> str:
    raw = json.dumps([snapshot["board"], snapshot["piece"], snapshot["next"]], separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def teacher_diagnostics(snapshot: dict) -> dict:
    board = board_to_lists(snapshot["board"])
    legal = legal_placements_on(board, snapshot["piece"])
    values = {(p["rot"], p["x"]): value_of_placement(board, p, snapshot["next"]) for p in legal}
    if not values:
        raise ValueError("cannot label terminal state")
    label = max(values, key=lambda a: (values[a], -a[0], -a[1]))
    ordered_distinct = sorted(set(values.values()), reverse=True)
    margin = ordered_distinct[0] - ordered_distinct[1] if len(ordered_distinct) > 1 else 0.0
    tolerance = 1e-9
    ties = sorted(a for a, value in values.items() if abs(value - values[label]) <= tolerance)
    placement = next(p for p in legal if (p["rot"], p["x"]) == label)
    cells = placement["cells"]
    tags = []
    if any(c == 0 for _, c in cells):
        tags.append("left_boundary")
    if any(c == 9 for _, c in cells):
        tags.append("right_boundary")
    if label[0] in (2, 3):
        tags.append("rotation_2_or_3")
    if snapshot["max_height"] >= 15:
        tags.append("top_risk")
    if snapshot["holes_total"]:
        tags.append("holes_present")
    return {
        "label": label,
        "legal_actions": [[a[0], a[1]] for a in sorted(values)],
        "teacher_values": {f"{a[0]},{a[1]}": values[a] for a in sorted(values)},
        "near_tied_best": [[a[0], a[1]] for a in ties],
        "teacher_margin": margin,
        "safety_tags": tags,
    }


def enriched_row(snapshot: dict, *, split: str, kind: str, source_id: str, extra: dict | None = None) -> dict:
    diagnostics = teacher_diagnostics(snapshot)
    rot, x = diagnostics.pop("label")
    row = {
        "game_id": source_id,
        "seed": snapshot["seed"],
        "turn": snapshot["turn"],
        "split": split,
        "kind": kind,
        "prompt": snapshot["prompt"],
        "completion": f"Action: rot={rot} x={x}",
        "rot": rot,
        "x": x,
        "piece": snapshot["piece"],
        "next": snapshot["next"],
        "board": snapshot["board"],
        "heights": snapshot["heights"],
        "holes": snapshot["holes"],
        "holes_total": snapshot["holes_total"],
        "wells": snapshot["wells"],
        "bumpiness": snapshot["bumpiness"],
        "aggregate_height": snapshot["aggregate_height"],
        "max_height": snapshot["max_height"],
        "score": snapshot["score"],
        "lines": snapshot["lines"],
        "difficulty_band": difficulty_band(snapshot["max_height"], snapshot["holes_total"]),
        "state_hash": state_hash_from_snapshot(snapshot),
        "prompt_hash": prompt_hash(snapshot["prompt"]),
        "board_hash": board_hash(snapshot["board"]),
        **diagnostics,
    }
    if extra:
        row.update(extra)
    return row
