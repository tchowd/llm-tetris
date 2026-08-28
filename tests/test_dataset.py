import subprocess
import sys
from pathlib import Path

import pytest

from tetris.dataset import (
    generate_game,
    is_noisy_game,
    rebuild_rows_from_game,
    row_from_snapshot,
    split_for_game_id,
)
from tetris.dataset_validate import (
    check_full_rebuild,
    check_legality,
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
        check_pre_move_consistency(games, by_game, sample_size=10**9)


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
