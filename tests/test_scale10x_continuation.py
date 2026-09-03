from __future__ import annotations

import copy
from pathlib import Path

import pytest

from infra import scale10x_continuation as c


def approved():
    return {"status": "user_approved", "user_quote": "okay proceed",
            "limits": {"experiment_usd": 90, "cumulative_stage6_usd": 250, "overall_hours": 36},
            "instance_id": "i-069539d2da9a2b0a2", "root_volume_id": "vol-0bbe37041a8425e58",
            "original_launch_epoch": 1788395046.0, "absolute_deadline_epoch": 1788524646.0,
            "iam_changes_authorized": False}


def test_explicit_approval_and_original_absolute_anchor():
    c.validate_approval(approved())
    for field, value in (("status", "proposed"), ("user_quote", ""), ("absolute_deadline_epoch", 1788528246),
                         ("iam_changes_authorized", True), ("instance_id", "another-worker")):
        a = approved()
        a[field] = value
        with pytest.raises(ValueError, match="approval"):
            c.validate_approval(a)
    a = approved()
    a["limits"]["experiment_usd"] = 100
    with pytest.raises(ValueError):
        c.validate_approval(a)


def test_only_timestamp_is_ignored_in_immutable_validation():
    c.same_record({"at": 1, "updates": 320}, {"at": 2, "updates": 320}, "at")
    for saved in ({"at": 1, "updates": 32}, {"at": 1}, {"at": 1, "updates": 320, "extra": True}):
        with pytest.raises(ValueError):
            c.same_record(saved, {"at": 2, "updates": 320}, "at")


def test_command_keeps_science_and_does_not_mutate_proof(monkeypatch):
    old = {"registered_at": "old", "pilot_recipe": c.original.recipe(320, 6103, 9),
           "pilot_run_id": c.PILOT, "question": "fixed", "prior_stage_spend_usd": 16, "hourly_usd": 2.3}
    p = {"base_model_revision": "pinned", "benchmark_manifest": "frozen", "data": {"data_dir": "frozen"}}
    frozen = copy.deepcopy(old)
    monkeypatch.setattr(c.original, "validate_registration", lambda: (old, p))
    q = {"registered_at": "new", "recipe": {**old["pilot_recipe"], "max_training_hours": 33}}
    cmd = c.training_command(q)
    options = dict(zip(cmd[2::2], cmd[3::2]))
    for flag, value in {"--updates": "320", "--group-size": "4", "--horizon": "128", "--training-seed": "6103",
                        "--learning-rate": "1e-06", "--kl-beta": "0.05", "--gamma": "0.99", "--train-batch-size": "4",
                        "--temperature": "1", "--save-every": "4", "--pilot-dollar-limit": "90",
                        "--stage-dollar-limit": "250", "--max-wall-clock-hours": "33",
                        "--adapter-dir": "runs/sft-v1/adapter", "--registration-file": str(c.REGISTRATION)}.items():
        assert options[flag] == value
    assert old == frozen
    assert "--resume" not in cmd


def live_ledger():
    return {"instance_id": "i-069539d2da9a2b0a2", "root_volume_id": "vol-0bbe37041a8425e58",
            "instance_type": "g6e.2xlarge", "deadline_epoch": 1788524646,
            "experiment_cap_usd": 90, "cumulative_stage6_cap_usd": 250,
            "runtime_amendment_sha256": "digest", "pilot_registration_sha256": "digest",
            "compute_sessions": [{"start_epoch": 1788395046, "end_epoch": 1788397414},
                                 {"start_epoch": 1788399800, "end_epoch": None}],
            "extra_service_reserve_usd": 2, "prior_stage6_estimate_usd": 15.53}


def amendment():
    return {"instance_id": "i-069539d2da9a2b0a2", "root_volume_id": "vol-0bbe37041a8425e58",
            "absolute_deadline_epoch": 1788524646, "projected_training_seconds": 109405.6373}


def test_launch_fits_and_reserves_every_phase(monkeypatch):
    monkeypatch.setattr(c, "file_sha256", lambda _: "digest")
    allocation = c.launch_allocation(amendment(), live_ledger(), 1788400000)
    assert allocation == 1788524646 - 1788400000 - 9000
    assert c.LIMITS["evaluation_backup_reserve_seconds"] >= 3600 + 2700 + 900 + 600 + 300 + 600


@pytest.mark.parametrize("change", ["deadline", "dollars", "hash", "no_session", "two_sessions", "future", "cost"])
def test_launch_rejects_drift_and_unsafe_spend(monkeypatch, change):
    monkeypatch.setattr(c, "file_sha256", lambda _: "digest")
    ledger = live_ledger()
    if change == "deadline":
        ledger["deadline_epoch"] += 1
    elif change == "dollars":
        ledger["experiment_cap_usd"] = 50
    elif change == "hash":
        ledger["runtime_amendment_sha256"] = "wrong"
    elif change == "no_session":
        ledger["compute_sessions"][-1]["end_epoch"] = 1788399900
    elif change == "two_sessions":
        ledger["compute_sessions"][0]["end_epoch"] = None
    elif change == "future":
        ledger["compute_sessions"][-1]["start_epoch"] = 1788500000
    elif change == "cost":
        ledger["extra_service_reserve_usd"] = 100
    with pytest.raises(ValueError):
        c.launch_allocation(amendment(), ledger, 1788400000)


def test_late_start_cannot_reset_deadline(monkeypatch):
    monkeypatch.setattr(c, "file_sha256", lambda _: "digest")
    with pytest.raises(ValueError, match="no longer fits"):
        c.launch_allocation(amendment(), live_ledger(), 1788500000)


def closure_inputs():
    audit = {"status": "passed", "runs": [{"run_id": run, "status": "passed",
             "adapter_directory_hash_recomputed_from_s3": True, "optimizer_checkpoints_present": False} for run in c.original.RUNS]}
    cleanup = {"status": "passed", "checks": {"all": True}, "instances": [], "volumes": []}
    ledger = {"instance_id": "i-069539d2da9a2b0a2", "root_volume_id": "vol-0bbe37041a8425e58",
              "instance_state": "terminated", "root_volume_deleted": True, "stage4_untouched": True,
              "compute_sessions": [{"end_epoch": 100}], "estimated_experiment_usd": 80, "prior_stage6_estimate_usd": 16}
    return audit, cleanup, ledger


def test_negative_closure_uses_new_cap_but_requires_all_artifacts():
    audit, cleanup, ledger = closure_inputs()
    assert c.closure({"status": "not_passed"}, audit, cleanup, ledger)["research_complete"]
    audit["runs"].pop()
    with pytest.raises(ValueError, match="three"):
        c.closure({"status": "not_passed"}, audit, cleanup, ledger)


@pytest.mark.parametrize("status", ["passed", "incomplete", "failed"])
def test_positive_or_incomplete_cannot_be_closed_as_negative(status):
    with pytest.raises(ValueError):
        c.closure({"status": status}, *closure_inputs())


def test_cleanup_and_spending_required():
    for field, value in (("instance_state", "stopped"), ("root_volume_deleted", False),
                         ("estimated_experiment_usd", 91), ("prior_stage6_estimate_usd", 200),
                         ("stage4_untouched", False), ("instance_id", "other")):
        audit, cleanup, ledger = closure_inputs()
        ledger[field] = value
        with pytest.raises(ValueError):
            c.closure({"status": "not_passed"}, audit, cleanup, ledger)


def test_short_training_stays_incomplete_and_always_stops(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    # A preexisting pilot manifest must be absent on entry, then appears after training.
    monkeypatch.setattr(c, "ROOT", root)
    monkeypatch.setattr(c, "PILOT_ROOT", pilot)
    monkeypatch.setattr(c, "BLOCK", root / "pilot-block.json")
    monkeypatch.setattr(c, "validate", lambda: ({"projected_training_seconds": 1}, {}))
    monkeypatch.setattr(c, "launch_allocation", lambda *args: 100)
    monkeypatch.setattr(c, "training_command", lambda _: ["training"])
    monkeypatch.setattr(c, "file_sha256", lambda _: "digest")
    monkeypatch.setattr(c, "write_new", lambda *args: None)
    saved, commands, process_commands = {}, [], []
    monkeypatch.setattr(c, "atomic_write_json", lambda p, v: saved.update({str(p): v}))
    monkeypatch.setattr(c, "read", lambda p: {"deadline_epoch": c.time.time() + 10000, "instance_id": "worker"}
                        if p.name == "compute-ledger.json" else {"status": "stopped_budget", "completed_updates": 4})
    monkeypatch.setattr(c, "run_process", lambda command, *args: commands.append(command) or 0)

    def system(command, **kwargs):
        process_commands.append(command)
        if command[0] == "aws":
            raise c.subprocess.TimeoutExpired(command, 20)
    monkeypatch.setattr(c.subprocess, "run", system)
    with pytest.raises(SystemExit):
        c.workflow()
    assert saved[str(c.BLOCK)]["status"] == "incomplete"
    assert "320" in saved[str(c.BLOCK)]["failure"]
    assert commands[0] == ["training"]
    assert not any("scripts/eval_stress.py" in command for command in commands)
    assert process_commands[-1] == ["sudo", "/usr/sbin/shutdown", "-h", "now"]
    assert not any("proof" in p for p in saved)
