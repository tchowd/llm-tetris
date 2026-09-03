from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import scripts.check_episode_runtime as runtime
from tetris.rl import file_sha256


@pytest.fixture
def retained_proof(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    protocol, registration, gate = (Path(n) for n in ("protocol.json", "registration.json", "gate.json"))
    protocol.write_text("{}")
    registration.write_text("{}")
    r = {"protocol_path": str(protocol), "protocol_sha256": file_sha256(protocol), "recipe": {"updates": 4},
        "pilot_projection_startup_seconds": 120, "pilot_projection_safety_multiplier": 1.5,
        "hourly_usd": 1.05, "logprob_absolute_tolerance": .0001,
        "positive_direction_min_logprob_gain": 0, "minimum_allocated_gpu_headroom_fraction": .15}
    p = {"episode_pilot": {"updates": 32, "group_size": 4, "horizon": 128, "max_training_hours": 1.5}}
    manifests = [{"sample_count": 300, "update_metrics": [{"turns": 75, "seconds": 26}] * 4}] * 2
    roots = [Path("resumed"), Path("control")]
    for root in roots:
        (root / "adapter").mkdir(parents=True)
        (root / "trajectory_batches").mkdir()
        (root / "adapter/adapter_model.safetensors").write_bytes(b"same immutable adapter")
        (root / "manifest.json").write_text(json.dumps(manifests[0]))
        for i in range(4):
            (root / f"trajectory_batches/{i}.json").write_text("{}")
    (roots[0] / "paused-manifest.json").write_text("{}")
    paths = [x for root in roots for x in [root / "manifest.json", *sorted((root / "trajectory_batches").glob("*.json"))]]
    paths.append(roots[0] / "paused-manifest.json")
    g = {"experiment": "R2", "status": "not_passed", "registration_sha256": file_sha256(registration),
        "protocol_sha256": file_sha256(protocol), "final_test_access": False,
        "checks": {**{k: True for k in runtime.CORRECTNESS_CHECKS}, "pilot_projection_fits": False},
        "evidence_sha256": {str(x): file_sha256(x) for x in paths}, "sample_count": 300,
        "pilot_projection": runtime.projection(r, p, manifests[0]["update_metrics"]),
        "max_absolute_logprob_error": 0, "tokens_checked": 5000,
        "positive_direction": {"gradient_norm": 1, "mean_logprob_before": -1, "mean_logprob_after": -.99, "restored": True},
        "gpu": {"peak_allocated_bytes": 30, "total_bytes": 100, "headroom_fraction": .7},
        "generated_at": "2026-09-02T19:03:23Z"}
    gate.write_text(json.dumps(g))
    for name in runtime.SOURCES:
        path = Path(name); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
    monkeypatch.setattr(runtime, "validate_registration", lambda _: (r, p))
    monkeypatch.setattr(runtime, "evidence", lambda *args: (*roots, manifests, []))
    return registration, gate, g


def test_runtime_amendment_keeps_failed_gate_and_reserves_evaluation(retained_proof):
    registration, gate, _ = retained_proof
    original = gate.read_bytes()
    result = runtime.amendment(registration, gate)
    assert result["historical_proof_status"] == "not_passed"
    assert result["limits"]["max_training_hours"] == result["limits"]["block_hours"] == 12
    assert result["limits"]["training_timeout_minutes"] + 4 + 75 + 60 + 20 < result["limits"]["workflow_minutes"]
    assert result["limits"]["workflow_minutes"] + 8 < result["limits"]["block_hours"] * 60
    assert result["entry"]["pilot_projection"]["within_reserved_training_window"]
    amendment = Path("amendment.json"); amendment.write_text(json.dumps(result))
    assert runtime.validate_amendment(amendment, registration, gate) == result
    assert gate.read_bytes() == original
    for key, value in (("approval", {"received": False}), ("protocol_sha256", "wrong"),
                       ("limits", {**runtime.LIMITS, "pilot_usd": 200}), ("recorded_at", "2000-01-01T00:00:00Z")):
        amendment.write_text(json.dumps({**result, key: value}))
        with pytest.raises(ValueError, match="amendment"):
            runtime.validate_amendment(amendment, registration, gate)


@pytest.mark.parametrize("failure", sorted(runtime.CORRECTNESS_CHECKS) + ["extra_check"])
def test_runtime_amendment_cannot_waive_correctness(retained_proof, failure):
    registration, gate, original = retained_proof
    broken = copy.deepcopy(original); broken["checks"][failure] = False
    gate.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="correctness"):
        runtime.amendment(registration, gate)


def test_runtime_amendment_rejects_changed_evidence_and_wrong_tensor_identity(retained_proof):
    registration, gate, _ = retained_proof
    Path("control/adapter/adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="tensors"):
        runtime.amendment(registration, gate)
    Path("resumed/manifest.json").write_text("changed")
    with pytest.raises(ValueError, match="evidence"):
        runtime.amendment(registration, gate)


def test_original_strict_proof_gate_still_rejects_time_failure(retained_proof, monkeypatch):
    import scripts.check_episode_proof as proof
    registration, gate, _ = retained_proof
    monkeypatch.setattr(proof, "validate_registration", runtime.validate_registration)
    monkeypatch.setattr(proof, "evidence", runtime.evidence)
    with pytest.raises(ValueError, match="passed, hash-bound"):
        proof.validate_proof_report(registration, gate)


def test_negative_closure_accepts_only_validated_amended_entry(retained_proof):
    from scripts.report_recovery_outcome import closure_decision
    registration, gate, _ = retained_proof
    entry = runtime.amendment(registration, gate)["entry"]
    args = [{"status": "not_passed"}, entry, {"status": "not_passed"},
        {"status": "passed"}, {"status": "passed", "checks": {"clean": True}, "instances": [], "volumes": []}]
    assert closure_decision(*args)["research_complete"]
    entry["correctness_checks"]["exact_gpu_resume_adapter_tensors"] = False
    with pytest.raises(ValueError):
        closure_decision(*args)
