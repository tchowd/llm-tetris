#!/usr/bin/env python3
"""Audit the frozen Tetra-1.7B SFT data and aggregate development failures.

The failure audit is descriptive only.  It never emits held-out boards or
action prefixes, so its output cannot be used as a training-state source.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Iterable

from tetris.board import board_to_lists
from tetris.features import column_heights, column_holes
from tetris.placement import legal_placements_on
from tetris.serialize import parse_action
from tetris.teacher import value_of_placement


PIECES = set("IJLOSTZ")
REQUIRED = {"game_id", "seed", "prompt", "completion", "rot", "x", "split"}


def digest(value) -> str:
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(raw.encode(), digest_size=16).hexdigest()


def difficulty_band(max_height: int, holes: int) -> str:
    """Mutually exclusive version of the preregistered progressive bands."""
    if max_height >= 16 or holes >= 6:
        return "critical"
    if max_height >= 13 or holes >= 3:
        return "hard"
    if max_height >= 8 or holes >= 1:
        return "moderate"
    return "low"


def valid_board(board) -> bool:
    return isinstance(board, list) and len(board) == 20 and all(
        isinstance(row, str) and len(row) == 10 and set(row) <= set(".IJLOSTZ#") for row in board
    )


def analyze_rows(rows: Iterable[dict], *, teacher_sample_size: int = 0, sample_seed: int = 1702) -> dict:
    """Analyze an iterable in one pass; bounded teacher sample uses reservoir sampling."""
    counts = Counter()
    missing = Counter()
    invalid = Counter()
    coverage = {name: Counter() for name in ("split", "piece", "next", "piece_next", "rotation", "x", "difficulty", "kind")}
    games_by_split: dict[str, set] = {}
    seeds_by_split: dict[str, set] = {}
    seen_rows: set[str] = set()
    state_labels: dict[str, tuple[int, int]] = {}
    prompt_boards: dict[str, str] = {}
    prompt_labels: dict[str, tuple[int, int]] = {}
    aliased_prompts: set[str] = set()
    conflicting_prompts: set[str] = set()
    conflicting_states: set[str] = set()
    samples: list[dict] = []
    rng = random.Random(sample_seed)

    for row in rows:
        counts["rows"] += 1
        absent = REQUIRED - row.keys()
        for key in absent:
            missing[key] += 1
        if absent:
            continue
        split = str(row["split"])
        coverage["split"][split] += 1
        coverage["kind"][str(row.get("kind", "original"))] += 1
        games_by_split.setdefault(split, set()).add(str(row["game_id"]))
        seeds_by_split.setdefault(split, set()).add(row["seed"])
        label = (row["rot"], row["x"])
        try:
            if parse_action(row["completion"]) != label:
                invalid["completion_label_mismatch"] += 1
        except (ValueError, TypeError):
            invalid["unparseable_completion"] += 1
        if not isinstance(row["rot"], int) or not isinstance(row["x"], int):
            invalid["non_integer_action"] += 1

        row_key = digest(row)
        if row_key in seen_rows:
            counts["exact_duplicate_rows"] += 1
        seen_rows.add(row_key)

        state = row.get("state") or {}
        piece, nxt = row.get("piece", state.get("piece")), row.get("next", state.get("next"))
        board = row.get("board", state.get("board"))
        if piece in PIECES:
            coverage["piece"][piece] += 1
        elif piece is not None:
            invalid["invalid_piece"] += 1
        if nxt in PIECES:
            coverage["next"][nxt] += 1
        elif nxt is not None:
            invalid["invalid_next"] += 1
        if piece in PIECES and nxt in PIECES:
            coverage["piece_next"][piece + nxt] += 1
        coverage["rotation"][str(row["rot"])] += 1
        coverage["x"][str(row["x"])] += 1
        max_height = row.get("max_height")
        holes = row.get("holes_total")
        if valid_board(board):
            as_lists = board_to_lists(board)
            if not isinstance(max_height, int):
                max_height = max(column_heights(as_lists))
            if not isinstance(holes, int):
                holes = sum(column_holes(as_lists))
        if isinstance(max_height, int) and isinstance(holes, int):
            coverage["difficulty"][difficulty_band(max_height, holes)] += 1

        prompt_key = digest(row["prompt"])
        if valid_board(board) and piece in PIECES and nxt in PIECES:
            board_key = digest(board)
            state_key = digest([board, piece, nxt])
            previous = state_labels.setdefault(state_key, label)
            if previous != label:
                counts["state_label_conflicts"] += 1
                conflicting_states.add(state_key)
            previous_board = prompt_boards.setdefault(prompt_key, board_key)
            if previous_board != board_key:
                counts["prompt_geometry_aliases"] += 1
                aliased_prompts.add(prompt_key)
            previous_label = prompt_labels.setdefault(prompt_key, label)
            if previous_label != label:
                counts["prompt_label_conflicts"] += 1
                conflicting_prompts.add(prompt_key)
            counts["rows_with_full_state"] += 1
            if teacher_sample_size:
                sample_row = {**row, "board": board, "piece": piece, "next": nxt}
                if len(samples) < teacher_sample_size:
                    samples.append(sample_row)
                else:
                    index = rng.randrange(counts["rows_with_full_state"])
                    if index < teacher_sample_size:
                        samples[index] = sample_row
        elif board is not None:
            invalid["invalid_board"] += 1

    overlaps = {}
    splits = sorted(games_by_split)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlaps[f"{left}:{right}"] = {
                "game_ids": len(games_by_split[left] & games_by_split[right]),
                "seeds": len(seeds_by_split[left] & seeds_by_split[right]),
            }

    teacher = Counter()
    margins = []
    for row in samples:
        board = board_to_lists(row["board"])
        legal = legal_placements_on(board, row["piece"])
        values = {(p["rot"], p["x"]): value_of_placement(board, p, row["next"]) for p in legal}
        label = (row["rot"], row["x"])
        if label not in values:
            teacher["illegal_labels"] += 1
            continue
        best_value = max(values.values())
        if values[label] == best_value:
            teacher["teacher_optimal_labels"] += 1
        else:
            teacher["teacher_suboptimal_labels"] += 1
        ordered = sorted(values.values(), reverse=True)
        margins.append(ordered[0] - ordered[1] if len(ordered) > 1 else 0.0)
        placement = next(p for p in legal if (p["rot"], p["x"]) == label)
        if any(c in (0, 9) for _, c in placement["cells"]):
            teacher["selected_edge_placements"] += 1

    return {
        "counts": dict(counts),
        "missing_required_fields": dict(missing),
        "invalid": dict(invalid),
        "coverage": {k: dict(sorted(v.items())) for k, v in coverage.items()},
        "unique_games": {k: len(v) for k, v in games_by_split.items()},
        "unique_seeds": {k: len(v) for k, v in seeds_by_split.items()},
        "split_overlap": overlaps,
        "teacher_validation": {
            "sample_size": len(samples),
            **dict(teacher),
            "mean_top_margin": sum(margins) / len(margins) if margins else None,
        },
        "identity": {
            "unique_full_states": len(state_labels),
            "unique_prompts": len(prompt_boards),
            "unique_aliased_prompts": len(aliased_prompts),
            "unique_conflicting_prompts": len(conflicting_prompts),
            "unique_conflicting_states": len(conflicting_states),
        },
    }


def read_jsonl(paths: list[Path]):
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def audit_failures(directory: Path) -> dict:
    outcomes = Counter()
    start_bands = Counter()
    pieces = Counter()
    terminal = Counter()
    files = []
    for path in sorted(directory.glob("recovery-*.json")):
        payload = json.loads(path.read_text())
        files.append(str(path))
        for game in payload["games"]:
            start = game["starting_state"]
            max_height = max(20 - i for i, row in enumerate(start["board"]) if set(row) != {"."}) if any(set(r) != {"."} for r in start["board"]) else 0
            holes = sum(int(x) for x in start["prompt"].split("Holes:", 1)[1].splitlines()[0].split())
            start_bands[difficulty_band(max_height, holes)] += 1
            pieces[start["piece"]] += 1
            outcomes[game["death_reason"]] += 1
            incident = game.get("terminal_incident") or {}
            if game["death_reason"] == "illegal_action":
                terminal["unparsed" if not incident.get("parsed") else "parsed_illegal"] += 1
            terminal["pieces_total"] += len(game.get("actions", []))
    n = sum(outcomes.values())
    return {
        "n_games": n,
        "outcomes": dict(outcomes),
        "start_difficulty": dict(start_bands),
        "start_piece": dict(sorted(pieces.items())),
        "terminal_failures": dict(terminal),
        "mean_continuation_pieces": terminal["pieces_total"] / n if n else None,
        "source_files": files,
        "held_out_state_material_emitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", nargs="+", type=Path, required=True)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--failure-dir", type=Path)
    parser.add_argument("--teacher-sample-size", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "status": "completed",
        "original": analyze_rows(read_jsonl([p / "rows.jsonl" for p in args.original]), teacher_sample_size=args.teacher_sample_size),
        "prior_recovery": analyze_rows(read_jsonl([args.recovery / "rows.jsonl"]), teacher_sample_size=min(512, args.teacher_sample_size)) if args.recovery else None,
        "development_failure_profile": audit_failures(args.failure_dir) if args.failure_dir else None,
        "final_test_access": False,
        "elapsed_seconds": time.monotonic() - started,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
