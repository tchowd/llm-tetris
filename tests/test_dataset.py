import json
import subprocess
import sys
from pathlib import Path

import pytest

from tetris import teacher as teacher_mod
from tetris.dataset import (
    generate_game,
    is_noisy_game,
    rebuild_rows_from_game,
    row_from_snapshot,
    split_for_game_id,
)
from tetris.dataset_validate import (
    check_full_rebuild,
    check_label_sanity,
    check_labels_consistent,
    check_legality,
    check_no_orphan_rows,
    check_pre_move_consistency,
    check_prompt_identity,
    check_split_purity,
    rows_by_game_id,
    run_all_checks,
)
from tetris.engine import Game
from tetris.placement import legal_placements_on
from tetris.serialize import serialize_prompt


def test_split_for_game_id_is_deterministic_and_covers_both_splits():
    ids = [str(i) for i in range(1000)]
    assert {split_for_game_id(i) for i in ids} == {"train", "eval"}
    for game_id in ids[:20]:
        assert split_for_game_id(game_id) == split_for_game_id(game_id)
    eval_fraction = sum(split_for_game_id(i) == "eval" for i in ids) / len(ids)
    assert 0.02 < eval_fraction < 0.09  # target ~5%


def test_row_from_snapshot_prompt_matches_serializer():
    g = Game(seed=1)
    snap = g.snapshot()
    row = row_from_snapshot(snap, label=(0, 3), explored=False)
    assert row["prompt"] == serialize_prompt(row)
    assert row["completion"] == "Action: rot=0 x=3"
    assert (row["rot"], row["x"]) == (0, 3)
    assert row["split"] in ("train", "eval")


def test_generate_game_labels_always_legal():
    for seed in range(10):
        rows, game_record = generate_game(seed, max_pieces=60)
        assert len(rows) == game_record["pieces"]
        for row in rows:
            board = [list(r) for r in row["board"]]
            legal_pairs = {(p["rot"], p["x"]) for p in legal_placements_on(board, row["piece"])}
            assert (row["rot"], row["x"]) in legal_pairs


def test_noisy_game_explores_every_eighth_piece_with_a_still_legal_label():
    seed = 10
    assert is_noisy_game(seed)
    rows, game_record = generate_game(seed, max_pieces=60, noisy=True)
    explored_rows = [r for r in rows if r["explored"]]
    assert explored_rows, "expected at least one explored turn in 60 pieces"
    assert all(r["turn"] % 8 == 7 for r in explored_rows)
    for turn, row in enumerate(rows):
        if not row["explored"]:
            assert game_record["actions"][turn] == game_record["labels"][turn]


def test_rebuild_rows_from_game_matches_generate_game_exactly():
    for seed in (0, 1, 5, 10, 11):  # mix of noisy (seed % 10 == 0) and clean seeds
        rows, game_record = generate_game(seed, max_pieces=80)
        assert rebuild_rows_from_game(game_record, max_pieces=80) == rows


def _make_small_dataset(seed_start=0, n=16, max_pieces=100):
    games, rows = [], []
    for seed in range(seed_start, seed_start + n):
        game_rows, game_record = generate_game(seed, max_pieces=max_pieces)
        games.append(game_record)
        rows.extend(game_rows)
    return games, rows


def test_run_all_checks_passes_on_a_freshly_generated_dataset():
    games, rows = _make_small_dataset(n=16, max_pieces=100)
    report = run_all_checks(games, rows)
    for name, result in report.items():
        assert result["ok"], f"{name} failed: {result['detail']}"


def test_check_full_rebuild_catches_a_corrupted_row():
    games, rows = _make_small_dataset(n=3, max_pieces=40)
    rows[0] = {**rows[0], "lines": rows[0]["lines"] + 999}
    with pytest.raises(AssertionError):
        check_full_rebuild(games, rows_by_game_id(rows))


def test_check_legality_catches_an_illegal_label():
    games, rows = _make_small_dataset(n=3, max_pieces=40)
    rows[0] = {**rows[0], "rot": 0, "x": 99}
    with pytest.raises(AssertionError):
        check_legality(rows)


def test_check_prompt_identity_catches_a_stale_prompt():
    games, rows = _make_small_dataset(n=3, max_pieces=40)
    rows[0] = {**rows[0], "prompt": "garbage"}
    with pytest.raises(AssertionError):
        check_prompt_identity(rows)


def test_check_pre_move_consistency_catches_a_snapshot_after_step_bug():
    games, rows = _make_small_dataset(n=3, max_pieces=40)
    by_game = rows_by_game_id(rows)
    game_id = games[0]["game_id"]
    by_game[game_id][1] = {**by_game[game_id][1], "board": ["." * 10] * 20}
    with pytest.raises(AssertionError):
        check_pre_move_consistency(games, by_game)  # default is exhaustive, not sampled


def test_check_pre_move_consistency_is_exhaustive_by_default():
    # No teacher call is needed for this check, so there's no excuse to
    # sample -- confirm every candidate pair actually gets checked.
    games, rows = _make_small_dataset(n=5, max_pieces=60)
    by_game = rows_by_game_id(rows)
    result = check_pre_move_consistency(games, by_game)
    assert result["checked_pairs"] == result["candidate_pairs"] > 0


def test_check_no_orphan_rows_catches_a_row_with_no_matching_game():
    games, rows = _make_small_dataset(n=3, max_pieces=40)
    orphan = {**rows[0], "game_id": "does-not-exist-in-games-jsonl"}
    rows = rows + [orphan]
    with pytest.raises(AssertionError):
        check_no_orphan_rows(games, rows_by_game_id(rows))


def test_check_labels_consistent_catches_a_corrupted_label():
    games, rows = _make_small_dataset(n=3, max_pieces=40)
    by_game = rows_by_game_id(rows)
    games = [dict(g) for g in games]
    games[0]["labels"] = list(games[0]["labels"])
    games[0]["labels"][0] = [3, 9]  # almost certainly not what rows.jsonl actually has
    with pytest.raises(AssertionError):
        check_labels_consistent(games, by_game)


def test_check_labels_consistent_catches_actions_labels_mismatch_outside_exploration():
    games, rows = _make_small_dataset(n=3, max_pieces=40)
    by_game = rows_by_game_id(rows)
    games = [dict(g) for g in games]
    games[0]["actions"] = list(games[0]["actions"])
    games[0]["explored_turns"] = []  # turn 0 is now claimed to be non-explored
    games[0]["actions"][0] = [3, 9]  # but no longer equals labels[0]
    with pytest.raises(AssertionError):
        check_labels_consistent(games, by_game)


def test_check_label_sanity_catches_per_piece_rotation_degeneracy():
    games, rows = _make_small_dataset(n=16, max_pieces=100)
    # Force every T placement (4 canonical rotations available) to rot=0,
    # while leaving J/L/etc. alone so the *aggregate* rot distribution
    # still looks fine -- only the per-piece check should catch this.
    rows = [{**r, "rot": 0} if r["piece"] == "T" else r for r in rows]
    with pytest.raises(AssertionError):
        check_label_sanity(rows)


def test_full_rebuild_uses_explicit_weights_not_live_default(monkeypatch):
    # A past dump must still validate against the weights it was actually
    # generated with, even after tetris.teacher.WEIGHTS is retuned later.
    # Single game -> check_full_rebuild's inline (non-multiprocess) path,
    # so the monkeypatch below is actually visible to it (a subprocess
    # worker would re-import tetris.teacher fresh and never see it).
    original_weights = dict(teacher_mod.WEIGHTS)
    games, rows = _make_small_dataset(n=1, max_pieces=40)
    by_game = rows_by_game_id(rows)

    # check_full_rebuild passes with the correct (explicit) weights...
    result = check_full_rebuild(games, by_game, weights=original_weights)
    assert result["games"] == 1

    # ...but if the live WEIGHTS is retuned and the caller forgets to pass
    # the dump's own recorded weights, the validator must not silently
    # keep agreeing with itself. Negate every weight (not just tweak one)
    # so divergence from the original picks is all but guaranteed from the
    # first move, rather than depending on a specific board state arising.
    retuned = {k: -v for k, v in original_weights.items()}
    monkeypatch.setattr(teacher_mod, "WEIGHTS", retuned)
    with pytest.raises(AssertionError):
        check_full_rebuild(games, by_game, weights=None)


def test_check_split_purity_catches_a_relabeled_row():
    games, rows = _make_small_dataset(n=3, max_pieces=40)
    by_game = rows_by_game_id(rows)
    game_id = games[0]["game_id"]
    wrong_split = "eval" if by_game[game_id][0]["split"] == "train" else "train"
    by_game[game_id][0] = {**by_game[game_id][0], "split": wrong_split}
    with pytest.raises(AssertionError):
        check_split_purity(games, by_game)


def test_cli_generate_and_validate_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = tmp_path / "smoke"

    gen = subprocess.run(
        [
            sys.executable, str(repo_root / "scripts" / "generate_dataset.py"),
            "--games", "6", "--max-pieces", "50", "--workers", "2", "--out-dir", str(out_dir),
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert gen.returncode == 0, gen.stdout + gen.stderr
    assert (out_dir / "games.jsonl").exists()
    assert (out_dir / "rows.jsonl").exists()
    assert (out_dir / "manifest.json").exists()

    val = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "validate_dataset.py"), "--data-dir", str(out_dir)],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert val.returncode == 0, val.stdout + val.stderr
    assert "all automated checks passed" in val.stdout


def test_cli_validate_uses_manifest_max_pieces_when_not_overridden(tmp_path):
    # Regression test: generate with --max-pieces above the *code's*
    # default (DEFAULT_MAX_PIECES=400) and validate without repeating
    # --max-pieces. Before this was fixed, validate_dataset.py silently
    # fell back to the code default, so rebuild_rows_from_game truncated
    # every rebuild at 400 pieces while rows.jsonl had more -- a spurious
    # length-mismatch failure on a perfectly valid dump.
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = tmp_path / "smoke"
    non_default_max_pieces = "420"  # > DEFAULT_MAX_PIECES (400)

    gen = subprocess.run(
        [
            sys.executable, str(repo_root / "scripts" / "generate_dataset.py"),
            "--games", "2", "--max-pieces", non_default_max_pieces, "--workers", "2", "--out-dir", str(out_dir),
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=180,
    )
    assert gen.returncode == 0, gen.stdout + gen.stderr

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["max_pieces"] == int(non_default_max_pieces)

    val = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "validate_dataset.py"), "--data-dir", str(out_dir)],
        cwd=repo_root, capture_output=True, text=True, timeout=180,
    )
    assert val.returncode == 0, val.stdout + val.stderr
    assert f"max_pieces={non_default_max_pieces}" in val.stdout
    assert "[FAIL] full_rebuild" not in val.stdout
