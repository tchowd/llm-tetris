from __future__ import annotations

from scripts.eval_stress import build_policy, evaluate_states, progress_policy, run_recovery_rollouts, survival_and_quality
from tetris.events import EventWriter
from tetris.engine import Game
from tetris.rl import record_state
from tetris.rollout import STRICT, random_legal_policy, run_rollout, teacher_policy
from tetris.teacher import WEIGHTS


def test_fixed_state_evaluation_reconstructs_and_scores_without_model():
    game = Game(700)
    actions = []
    for _ in range(4):
        legal = game.snapshot()["legal"][0]
        action = [legal["rot"], legal["x"]]
        actions.append(action)
        game.step(*action)
    state = record_state(game, actions, state_id="probe")
    state["kind"] = "probe"
    results, summary = evaluate_states(random_legal_policy(), [state], batch_size=1)
    assert results[0]["state_id"] == "probe"
    assert results[0]["legal"]
    assert summary["probe"]["legality_rate"] == 1.0


def test_stress_metrics_include_primary_rate_and_survival_curve():
    records, diagnostics = run_rollout([800, 801], random_legal_policy(), STRICT, cap=10)
    report = survival_and_quality(records, diagnostics, cap=10)
    assert report["checkpoints"][-1]["turn"] == 10
    assert "bumpiness" in report["checkpoints"][-1]


def test_recovery_rollout_counts_only_post_start_outcomes():
    game = Game(702)
    action = game.snapshot()["legal"][0]
    prefix = [[action["rot"], action["x"]]]
    game.step(*prefix[0])
    state = record_state(game, prefix, state_id="recovery")
    state["kind"] = "recovery"
    records, metrics = run_recovery_rollouts(random_legal_policy(), [state], cap=3, batch_size=1)
    assert records[0]["pieces"] == 3
    assert len(records[0]["actions"]) == 3
    assert metrics["n_games"] == 1


def test_progress_wrapper_preserves_rollout(tmp_path):
    events = EventWriter(tmp_path / "events.jsonl", run_id="test", stage=6)
    expected = run_rollout([850, 851], random_legal_policy(), STRICT, cap=4)
    wrapped = progress_policy(random_legal_policy(), events, phase="test", cap=4)
    assert run_rollout([850, 851], wrapped, STRICT, cap=4) == expected


def test_teacher_reuses_precomputed_scores_without_changing_actions_or_metrics():
    expected = run_rollout([860, 861], teacher_policy(WEIGHTS), STRICT, cap=5)
    assert run_rollout([860, 861], build_policy("teacher", None), STRICT, cap=5) == expected

    # Deliberately omit snapshot details: a second search would fail here.
    supplied_action = (2, 3)
    assert build_policy("teacher", None)([{}], [(supplied_action, {})]) == [(supplied_action, None)]
