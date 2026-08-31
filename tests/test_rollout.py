"""Stage 5 gate tests (plan/stage-5-eval.md's "Tests" section). No torch
required -- these exercise tetris.rollout with the random-legal and teacher
policies only. The model policy lives in scripts/eval_closed_loop.py (needs
torch, same reasoning as tests/test_sft.py) and is exercised by hand at
real-eval time, not here.
"""
from __future__ import annotations

import json
from pathlib import Path

from tetris import teacher
from tetris.engine import Game
from tetris.rollout import (
    ASSISTED,
    DEATH_ILLEGAL_ACTION,
    EVAL_SEED_OFFSET,
    STRICT,
    aggregate_metrics,
    default_eval_seeds,
    random_legal_policy,
    replay_game_log,
    run_rollout,
    teacher_policy,
)

BENCHMARK_STEPS = 120
BENCHMARK_SEEDS = range(6)


def _play_teacher_direct(seed: int, cap: int) -> dict:
    """Same loop as tests/test_teacher_benchmark.py's `_play_teacher`, kept
    independent of tetris.rollout so it's a real known-good reference."""
    g = Game(seed=seed)
    for _ in range(cap):
        if g.game_over:
            break
        snap = g.snapshot()
        rot, x = teacher.pick(snap, snap["legal"])
        g.step(rot, x)
    return {"board": ["".join(row) for row in g.board], "lines": g.lines, "score": g.score, "turn": g.turn, "game_over": g.game_over}


def test_harness_reproduces_stage_2_teacher_benchmark():
    """#1: the harness's teacher policy must reproduce Stage 2's known-good
    numbers exactly, not just clear a similar bar."""
    direct = [_play_teacher_direct(seed, BENCHMARK_STEPS) for seed in BENCHMARK_SEEDS]
    records, _ = run_rollout(list(BENCHMARK_SEEDS), teacher_policy(), mode=STRICT, cap=BENCHMARK_STEPS)

    for seed, ref, rec in zip(BENCHMARK_SEEDS, direct, records):
        assert rec["pieces"] == ref["turn"], seed
        assert rec["lines"] == ref["lines"], seed
        assert rec["score"] == ref["score"], seed
        assert rec["died"] == ref["game_over"], seed

    teacher_mean_lines = sum(r["lines"] for r in records) / len(records)
    assert teacher_mean_lines >= 15, f"teacher mean lines too low via harness: {teacher_mean_lines}"


def test_replay_reproduces_every_logged_game_with_zero_drift():
    """#2: every logged eval game rebuilds from its seed + action list with
    zero drift."""
    seeds = list(range(20, 28))
    records, _ = run_rollout(seeds, random_legal_policy(), mode=ASSISTED, cap=30)
    assert any(r["pieces"] > 0 for r in records)

    for rec in records:
        final_state, _ = replay_game_log(rec)
        assert final_state["pieces"] == rec["pieces"], rec["game_id"]
        assert final_state["lines"] == rec["lines"], rec["game_id"]
        assert final_state["score"] == rec["score"], rec["game_id"]
        assert final_state["game_over"] == rec["died"], rec["game_id"]


def test_strict_mode_kills_a_garbage_policy_immediately():
    """#3: a stub policy that emits garbage scores zero lines and one death
    per game."""

    def garbage_policy(snapshots, teacher_infos):
        return [(None, "garbage") for _ in snapshots]

    seeds = list(range(40, 45))
    records, _ = run_rollout(seeds, garbage_policy, mode=STRICT, cap=50)

    for rec in records:
        assert rec["pieces"] == 0, rec
        assert rec["lines"] == 0, rec
        assert rec["died"] is True, rec
        assert rec["death_reason"] == DEATH_ILLEGAL_ACTION, rec


def test_batching_is_a_noop_for_a_batch_invariant_policy():
    """#4: the same seed produces the same game whether run alone (batch
    size 1) or inside a larger batch. random_legal_policy is batch-invariant
    by construction (per-game RNG keyed on the game's own seed); this test
    is about the harness's own chunking/bookkeeping, not about the policy."""
    seeds = list(range(60, 70))
    cap = 25

    solo_records = []
    solo_diag = {}
    for seed in seeds:
        recs, diag = run_rollout([seed], random_legal_policy(), mode=ASSISTED, cap=cap, gen_batch_size=1)
        solo_records.extend(recs)
        solo_diag.update(diag)

    batched_records, batched_diag = run_rollout(seeds, random_legal_policy(), mode=ASSISTED, cap=cap, gen_batch_size=64)

    assert solo_records == batched_records
    assert solo_diag == batched_diag


def test_default_eval_seeds_disjoint_from_stage_3_dumps():
    """#5: no eval seed appears in the Stage 3 manifest(s)."""
    eval_seeds = set(default_eval_seeds(100))
    assert min(eval_seeds) >= 1_000_000, "eval seeds should sit far above any realistic Stage 3 seed range"

    data_dir = Path(__file__).resolve().parent.parent / "data"
    if not data_dir.exists():
        return
    for manifest_path in data_dir.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        start = manifest["seed_start"]
        stage3_seeds = set(range(start, start + manifest["num_games"]))
        overlap = eval_seeds & stage3_seeds
        assert not overlap, f"{manifest_path} overlaps eval seeds: {sorted(overlap)[:5]}"


def test_metric_arithmetic_matches_a_replayed_log():
    """#6: holes-created and value-gap recomputed from a replayed log match
    the values reported live."""
    seeds = list(range(80, 85))
    records, diagnostics = run_rollout(seeds, teacher_policy(), mode=STRICT, cap=20)

    report = aggregate_metrics(records, diagnostics)
    assert report["n_games"] == len(seeds)

    for rec in records:
        live = diagnostics[rec["game_id"]]
        _, replayed = replay_game_log(rec)
        assert len(live) == len(replayed) == len(rec["actions"])
        for live_turn, replayed_turn in zip(live, replayed):
            assert live_turn["holes_created"] == replayed_turn["holes_created"]
            assert live_turn["value_gap"] == replayed_turn["value_gap"]
            assert live_turn["teacher_match"] == replayed_turn["teacher_match"]
