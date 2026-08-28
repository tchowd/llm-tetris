"""Stage 3 validator: prove `rows.jsonl` is not a lie about `games.jsonl`.

Implements the checks from plan/stage-3-dataset.md's "Validation" section
(the stage gate). Each `check_*` function takes already-loaded games/rows
(lists of dicts, as parsed from the JSONL files) and either returns a
report dict or raises `AssertionError` describing the first mismatch it
finds. `run_all_checks` runs every check and collects pass/fail instead of
stopping at the first failure, for a single readable report.
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from .board import WIDTH, board_to_lists
from .dataset import DEFAULT_MAX_PIECES, rebuild_rows_from_game, split_for_game_id
from .pieces import CANONICAL_ROTS
from .placement import legal_placements_on
from .serialize import serialize_prompt
from .teacher import overlay_and_clear


def rows_by_game_id(rows: list[dict]) -> dict[str, list[dict]]:
    by_game: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_game[row["game_id"]].append(row)
    for game_rows in by_game.values():
        game_rows.sort(key=lambda r: r["turn"])
    return dict(by_game)


def check_no_orphan_rows(games: list[dict], rows_by_game: dict[str, list[dict]]) -> dict:
    """#0 (implied by "every game", not numbered in the doc): every row's
    game_id must correspond to a real games.jsonl record. Every other check
    here is driven by iterating `games` -- a batch of rows under a game_id
    that never made it into games.jsonl (stale run, bad concatenation of
    two dumps) would otherwise never be looked at by any of them, even
    though it describes a game that never happened.
    """
    game_ids = {g["game_id"] for g in games}
    row_game_ids = set(rows_by_game.keys())
    orphans = sorted(row_game_ids - game_ids)
    if orphans:
        shown = orphans[:10]
        raise AssertionError(
            f"rows.jsonl has {len(orphans)} game_id(s) with no matching games.jsonl record: {shown}"
            + (" ..." if len(orphans) > len(shown) else "")
        )
    return {"distinct_game_ids": len(game_ids)}


def _rebuild_one_game(args: tuple[dict, int, dict | None]) -> tuple[str, list[dict]]:
    game, max_pieces, weights = args
    return game["game_id"], rebuild_rows_from_game(game, max_pieces=max_pieces, weights=weights)


def check_full_rebuild(
    games: list[dict],
    rows_by_game: dict[str, list[dict]],
    max_pieces: int = DEFAULT_MAX_PIECES,
    workers: int | None = None,
    weights: dict | None = None,
) -> dict:
    """#1: replay every game from seed + actions, regenerate every row,
    and demand exact equality with the stored rows.jsonl. Every game.

    This re-runs the teacher on every placement in the dataset -- as
    expensive as generating it in the first place -- so it's parallelized
    across processes by game, the same way generation is. `weights` should
    be the weights the dump was actually generated with (see
    rebuild_rows_from_game).
    """
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    checked_games = 0
    checked_rows = 0
    work = [(game, max_pieces, weights) for game in games]

    def _handle(game_id: str, rebuilt: list[dict]) -> None:
        nonlocal checked_games, checked_rows
        stored = rows_by_game.get(game_id, [])
        if len(rebuilt) != len(stored):
            raise AssertionError(
                f"game {game_id}: rebuilt {len(rebuilt)} rows but rows.jsonl has {len(stored)}"
            )
        for turn, (rebuilt_row, stored_row) in enumerate(zip(rebuilt, stored)):
            if rebuilt_row != stored_row:
                diff = {k for k in rebuilt_row if rebuilt_row.get(k) != stored_row.get(k)}
                raise AssertionError(f"game {game_id} turn {turn}: rebuilt row != stored row, differing fields: {diff}")
        checked_games += 1
        checked_rows += len(stored)

    if len(work) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for game_id, rebuilt in pool.map(_rebuild_one_game, work):
                _handle(game_id, rebuilt)
    else:
        for args in work:
            _handle(*_rebuild_one_game(args))

    return {"games": checked_games, "rows": checked_rows}


def check_labels_consistent(games: list[dict], rows_by_game: dict[str, list[dict]]) -> dict:
    """games.jsonl declares `labels` (the teacher's picks) as part of its
    schema, separate from `actions` (what was executed) -- but nothing
    else ever reads `labels` back. Verify it against rows.jsonl's own
    (rot, x) at the same turn, and verify actions == labels everywhere
    except the declared explored_turns (stage-3-dataset.md's "Game
    schema": "actions and labels are identical except at explored_turns").
    """
    checked = 0
    for game in games:
        game_id = game["game_id"]
        rows = rows_by_game.get(game_id, [])
        labels = game.get("labels", [])
        actions = game.get("actions", [])
        explored_turns = set(game.get("explored_turns", []))
        if len(labels) != len(rows):
            raise AssertionError(f"game {game_id}: {len(labels)} labels but {len(rows)} stored rows")
        for turn, (label, row) in enumerate(zip(labels, rows)):
            if list(label) != [row["rot"], row["x"]]:
                raise AssertionError(
                    f"game {game_id} turn {turn}: games.jsonl label {label} != "
                    f"rows.jsonl (rot, x) ({row['rot']}, {row['x']})"
                )
            if turn not in explored_turns and list(actions[turn]) != list(label):
                raise AssertionError(
                    f"game {game_id} turn {turn}: not in explored_turns but "
                    f"actions {actions[turn]} != labels {label}"
                )
            checked += 1
    return {"rows": checked}


def check_legality(rows: list[dict]) -> dict:
    """#2: every row's label is legal on the board stored in that row."""
    for row in rows:
        board = board_to_lists(row["board"])
        legal_pairs = {(p["rot"], p["x"]) for p in legal_placements_on(board, row["piece"])}
        if (row["rot"], row["x"]) not in legal_pairs:
            raise AssertionError(
                f"game {row['game_id']} turn {row['turn']}: label rot={row['rot']} x={row['x']} "
                "is not legal on this row's own stored board"
            )
    return {"rows": len(rows)}


def check_pre_move_consistency(
    games: list[dict],
    rows_by_game: dict[str, list[dict]],
    sample_size: int | None = None,
    rng_seed: int = 0,
) -> dict:
    """#3: row n's board plus row n's *executed* action produces row n+1's
    board. Catches a logger that snapshotted after step(). No teacher call
    is needed here (only board-overlay math), so this checks every
    consecutive pair by default rather than a small sample -- pass a
    smaller `sample_size` only if this ever proves too slow at real scale.
    """
    candidates = []
    for game in games:
        rows = rows_by_game.get(game["game_id"], [])
        for turn in range(len(rows) - 1):
            candidates.append((game, rows, turn))

    if sample_size is None or sample_size >= len(candidates):
        sample = candidates
    else:
        sample = random.Random(rng_seed).sample(candidates, sample_size)

    for game, rows, turn in sample:
        row_n, row_next = rows[turn], rows[turn + 1]
        act_rot, act_x = game["actions"][turn]
        board_n = board_to_lists(row_n["board"])
        match = next(
            (p for p in legal_placements_on(board_n, row_n["piece"]) if p["rot"] == act_rot and p["x"] == act_x),
            None,
        )
        if match is None:
            raise AssertionError(
                f"game {game['game_id']} turn {turn}: executed action rot={act_rot} x={act_x} "
                "is not legal on this row's own board"
            )
        new_board, _, _ = overlay_and_clear(board_n, match["cells"], fill=row_n["piece"])
        expected_board = ["".join(r) for r in new_board]
        if expected_board != row_next["board"]:
            raise AssertionError(
                f"game {game['game_id']} turn {turn}: row n + executed action != row n+1's board "
                "(snapshot taken after step(), not before?)"
            )
    return {"checked_pairs": len(sample), "candidate_pairs": len(candidates)}


def check_split_purity(games: list[dict], rows_by_game: dict[str, list[dict]]) -> dict:
    """#4: no game_id in both splits; report the eval fraction."""
    game_ids = [g["game_id"] for g in games]
    if len(game_ids) != len(set(game_ids)):
        raise AssertionError("duplicate game_id in games.jsonl")

    eval_games = train_games = eval_rows = train_rows = 0
    for game_id in game_ids:
        expected_split = split_for_game_id(game_id)
        rows = rows_by_game.get(game_id, [])
        splits_in_rows = {r["split"] for r in rows}
        if len(splits_in_rows) > 1:
            raise AssertionError(f"game {game_id} has rows in both splits: {splits_in_rows}")
        if splits_in_rows and next(iter(splits_in_rows)) != expected_split:
            raise AssertionError(f"game {game_id}: stored split {splits_in_rows} disagrees with split_for_game_id")
        if expected_split == "eval":
            eval_games += 1
            eval_rows += len(rows)
        else:
            train_games += 1
            train_rows += len(rows)

    total_games = eval_games + train_games
    return {
        "train_games": train_games,
        "eval_games": eval_games,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "eval_game_fraction": eval_games / total_games if total_games else 0.0,
    }


def check_prompt_identity(rows: list[dict]) -> dict:
    """#5: every stored prompt equals serialize_prompt recomputed from the
    row's own feature fields."""
    for row in rows:
        if serialize_prompt(row) != row["prompt"]:
            raise AssertionError(f"game {row['game_id']} turn {row['turn']}: prompt does not match its own feature fields")
    return {"rows": len(rows)}


def check_label_sanity(rows: list[dict]) -> dict:
    """#7: rot/x distribution isn't degenerate (e.g. "always x=0"), and
    checked per piece, not just in aggregate -- a piece with more than one
    canonical rotation (see tetris.pieces.CANONICAL_ROTS) that only ever
    appears at one rot is exactly the kind of bug aggregate counts hide.
    """
    x_counts: dict[int, int] = defaultdict(int)
    rot_counts: dict[int, int] = defaultdict(int)
    rots_per_piece: dict[str, set] = defaultdict(set)
    for row in rows:
        x_counts[row["x"]] += 1
        rot_counts[row["rot"]] += 1
        rots_per_piece[row["piece"]].add(row["rot"])

    missing_x = sorted(set(range(WIDTH)) - set(x_counts))
    if missing_x:
        raise AssertionError(f"label sanity: these x columns are never used: {missing_x}")
    if len(rot_counts) < 2:
        raise AssertionError(f"label sanity: only one rot value ever appears: {dict(rot_counts)}")

    degenerate = [
        (piece, sorted(rots_seen))
        for piece, rots_seen in rots_per_piece.items()
        if len(CANONICAL_ROTS.get(piece, [])) >= 2 and len(rots_seen) < 2
    ]
    if degenerate:
        raise AssertionError(
            f"label sanity: these pieces have >=2 canonical rotations available but only ever use one: {degenerate}"
        )

    return {
        "x_counts": dict(sorted(x_counts.items())),
        "rot_counts": dict(sorted(rot_counts.items())),
        "distinct_rots_per_piece": {p: sorted(r) for p, r in rots_per_piece.items()},
    }


def sample_rows_as_text(rows: list[dict], n: int = 20, rng_seed: int = 0) -> str:
    """#6 ("read a few thousand"): render a random sample for a human to
    actually read, board included."""
    rng = random.Random(rng_seed)
    sample = rng.sample(rows, min(n, len(rows)))
    chunks = []
    for row in sample:
        board_text = "\n".join(row["board"])
        chunks.append(
            f"game={row['game_id']} turn={row['turn']} split={row['split']} explored={row['explored']}\n"
            f"{row['prompt']}\n{row['completion']}\n"
            f"lines={row['lines']} score={row['score']}\n{board_text}"
        )
    return "\n\n---\n\n".join(chunks)


def run_all_checks(
    games: list[dict],
    rows: list[dict],
    max_pieces: int = DEFAULT_MAX_PIECES,
    workers: int | None = None,
    weights: dict | None = None,
) -> dict:
    """Run every automated check (#1,#2,#3,#4,#5,#7, plus orphan-row and
    label-consistency checks the doc's numbering doesn't separately call
    out — #6 is manual, see sample_rows_as_text). Returns
    {check_name: {"ok": bool, "detail": ...}}. `weights` should be the
    dump's own recorded weights (e.g. from its manifest.json).
    """
    by_game = rows_by_game_id(rows)
    checks = {
        "no_orphan_rows": lambda: check_no_orphan_rows(games, by_game),
        "full_rebuild": lambda: check_full_rebuild(games, by_game, max_pieces=max_pieces, workers=workers, weights=weights),
        "labels_consistent": lambda: check_labels_consistent(games, by_game),
        "legality": lambda: check_legality(rows),
        "pre_move_consistency": lambda: check_pre_move_consistency(games, by_game),
        "split_purity": lambda: check_split_purity(games, by_game),
        "prompt_identity": lambda: check_prompt_identity(rows),
        "label_sanity": lambda: check_label_sanity(rows),
    }
    report = {}
    for name, fn in checks.items():
        try:
            report[name] = {"ok": True, "detail": fn()}
        except AssertionError as exc:
            report[name] = {"ok": False, "detail": str(exc)}
    return report
