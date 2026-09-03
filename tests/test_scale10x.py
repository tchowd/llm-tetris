from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import stage6_scale10x as scale
from scripts.run_scale10x import available_training_seconds, run_process


def test_tenfold_target_and_full_length_projection():
    r = {"recipe": scale.recipe(4, 6102, 1), "pilot_recipe": scale.recipe(320, 6103, 9),
        "pilot_projection_startup_seconds": 120, "pilot_projection_safety_multiplier": 1.5, "hourly_usd": 2.30}
    out = scale.projection(r, {}, [{"turns": 512, "seconds": 51.2}] * 4)
    assert out["full_length_pilot_turns"] == 163840
    assert out["projected_seconds"] == pytest.approx(24696)
    assert out["within_training_limit"]
    slow = scale.projection(r, {}, [{"turns": 512, "seconds": 102.4}] * 4)
    assert not slow["within_training_limit"]


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), 0, -1])
def test_bad_throughput_rejected(seconds):
    with pytest.raises(ValueError):
        scale.projection({"recipe": {"updates": 1}}, {}, [{"turns": 1, "seconds": seconds}])


def test_runtime_preserves_evaluation_and_absolute_deadline():
    assert available_training_seconds(43200, 3600, 7800, 32400) == 31800
    assert available_training_seconds(43200, 7200, 7800, 32400) == 28200
    with pytest.raises(ValueError):
        available_training_seconds(43200, 40000, 7800, 32400)


def test_frozen_registration_refuses_overwrite(tmp_path):
    target = tmp_path / "registration.json"
    scale.write_new(target, {"updates": 320})
    with pytest.raises(ValueError):
        scale.write_new(target, {"updates": 32})
    assert json.loads(target.read_text()) == {"updates": 320}


def test_command_uses_new_budgets_and_original_sft():
    r = {"recipe": scale.recipe(4, 6102, 1), "pilot_recipe": scale.recipe(320, 6103, 9),
        "run_id": scale.PROOF, "control_run_id": scale.CONTROL, "pilot_run_id": scale.PILOT,
        "question": "pilot", "proof_question": "proof", "prior_stage_spend_usd": 16, "hourly_usd": 2.3}
    p = {"base_model_revision": "pinned", "benchmark_manifest": "frozen.json", "data": {"data_dir": "data/frozen"}}
    cmd = scale.training_command(r, p)
    pairs = dict(zip(cmd[2::2], cmd[3::2]))
    assert pairs["--updates"] == "320"
    assert pairs["--pilot-dollar-limit"] == "50"
    assert pairs["--stage-dollar-limit"] == "250"
    assert pairs["--adapter-dir"] == "runs/sft-v1/adapter"
    assert pairs["--save-every"] == "4"
    proof = scale.training_command(r, p, proof=True, pause=True)
    assert proof[-2:] == ["--pause-after-update", "2"]


def test_registration_rejects_changed_recipe(monkeypatch, tmp_path):
    expected = {"registered_at": "later", "pilot_recipe": {"updates": 320}}
    monkeypatch.setattr(scale, "build_registration", lambda: expected)
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"registered_at": "earlier", "pilot_recipe": {"updates": 32}}))
    with pytest.raises(ValueError, match="registration"):
        scale.validate_registration(path)


def test_process_timeout_does_not_report_success(tmp_path):
    import sys
    with pytest.raises(TimeoutError):
        run_process([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path / "log", .05)
