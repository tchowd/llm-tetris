from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from infra import scale10x_resume as r
from tests.test_scale10x_continuation import live_ledger, amendment


def test_remaining_projection_preserves_rate_and_absolute_cap(monkeypatch):
    monkeypatch.setattr(r.c, "file_sha256", lambda _: "digest")
    a = amendment()
    original = copy.deepcopy(a)
    proof = {"worst_seconds_per_turn": 0.45, "safety_multiplier": 1.5}
    projection, allocation = r.remaining_allocation(a, live_ledger(), {"wall_clock_seconds": 16403}, proof, 1788426000)
    assert projection == 120 + 156 * 4 * 128 * 0.45 * 1.5
    assert allocation == 1788524646 - 1788426000 - 9000
    assert a == original
    with pytest.raises(ValueError):
        r.remaining_allocation(a, live_ledger(), {"wall_clock_seconds": 16403}, proof, 1788500000)
    with pytest.raises(ValueError):
        r.remaining_allocation(a, live_ledger(), {"wall_clock_seconds": 118000}, proof, 1788426000)


def test_command_only_appends_resume(monkeypatch):
    fixed = ["python", "train.py", "--updates", "320", "--adapter-dir", "original-sft"]
    monkeypatch.setattr(r.c, "training_command", lambda _: fixed)
    assert r.command({}) == fixed + ["--resume", str(r.CHECKPOINT)]
    assert len(fixed) == 6


def test_replay_waits_for_committed_update_and_detects_drift(tmp_path):
    batches = tmp_path / "trajectory_batches"
    batches.mkdir()
    path = batches / "update-000165.json"
    path.write_text('{"update":165}')
    expected = {path.name: r.c.file_sha256(path)}
    assert r.check_replay(expected, 164, tmp_path) == {}
    assert r.check_replay(expected, 165, tmp_path) == expected
    path.write_text('{"update":166}')
    with pytest.raises(ValueError, match="differs"):
        r.check_replay(expected, 165, tmp_path)


def test_real_subprocess_success_and_timeout(tmp_path):
    assert r.run_process([sys.executable, "-c", "pass"], tmp_path / "ok.log", 5) == 0
    started = time.time()
    with pytest.raises(TimeoutError, match="allocation"):
        r.run_process([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path / "timeout.log", 0.1)
    assert time.time() - started < 5


def test_interrupt_is_not_mislabeled_as_timeout(monkeypatch, tmp_path):
    class Process:
        def wait(self, **kwargs):
            raise KeyboardInterrupt
    monkeypatch.setattr(r.subprocess, "Popen", lambda *a, **kw: Process())
    stopped = []
    monkeypatch.setattr(r, "stop_child", lambda p: stopped.append(p))
    with pytest.raises(KeyboardInterrupt):
        r.run_process(["unused"], tmp_path / "interrupt.log", 10)
    assert len(stopped) == 1


def test_archive_preserves_metadata_not_checkpoint_and_refuses_duplicate(monkeypatch, tmp_path):
    root, pilot, archive = tmp_path / "root", tmp_path / "pilot", tmp_path / "archive"
    root.mkdir(); pilot.mkdir()
    (pilot / "manifest.json").write_text('{}')
    (pilot / "checkpoint-164").mkdir()
    (pilot / "checkpoint-164/state.pt").write_text('weights')
    for name in ("pilot-block-state-v1.json", "pilot-training_320_updates-v1.log", "pilot-launch-decision-v1.json", "pilot-shutdown-receipt-v1.json", "compute-ledger.json"):
        (root / name).write_text('preserved')
    monkeypatch.setattr(r, "ROOT", root); monkeypatch.setattr(r, "PILOT_ROOT", pilot); monkeypatch.setattr(r, "ARCHIVE", archive)
    commands = []
    monkeypatch.setattr(r.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    r.archive_initial()
    assert (archive / "pilot/manifest.json").read_text() == '{}'
    assert not (archive / "pilot/checkpoint-164").exists()
    assert (archive / "archive-sha256.json").exists()
    assert "AES256" in commands[0]
    with pytest.raises(FileExistsError):
        r.archive_initial()


@pytest.mark.parametrize("fail_sync", [False, True])
def test_incomplete_never_evaluated_always_preserves_disk_and_stops(monkeypatch, tmp_path, fail_sync):
    root, pilot = tmp_path / "root", tmp_path / "pilot"
    root.mkdir(); pilot.mkdir()
    monkeypatch.setattr(r, "ROOT", root); monkeypatch.setattr(r, "PILOT_ROOT", pilot)
    monkeypatch.setattr(r, "BLOCK", root / "block.json")
    monkeypatch.setattr(r, "archive_initial", lambda: None)
    monkeypatch.setattr(r.c, "file_sha256", lambda p: "hash")
    monkeypatch.setattr(r, "remaining_allocation", lambda *a: (1, 100))
    monkeypatch.setattr(r, "command", lambda q: ["training"])
    monkeypatch.setattr(r.c, "read", lambda p: {"pilot_projection": {}} if p.name == "gpu-proof.json" else
                        {"status": "stopped_budget", "completed_updates": 168})
    commands = []
    def run(cmd, *args):
        commands.append(cmd)
        if fail_sync and cmd != ["training"]:
            raise RuntimeError("sync failure")
        return 0
    monkeypatch.setattr(r, "run_process", run)
    system = []
    monkeypatch.setattr(r.subprocess, "run", lambda cmd, **kw: system.append(cmd))
    ledger = {"deadline_epoch": time.time() + 10000, "instance_id": "worker"}
    with pytest.raises(SystemExit):
        r.workflow({}, {"replay_sha256": {}}, ledger, 1, 100)
    state = json.loads((root / "block.json").read_text())
    assert state["status"] == "incomplete" and not state["research_complete"]
    assert not any("scripts/eval_stress.py" in cmd for cmd in commands)
    assert system[-1] == ["sudo", "/usr/sbin/shutdown", "-h", "now"]
    assert not any("terminate-instances" in cmd for cmd in system)
    receipt = json.loads((root / "resume-shutdown-receipt-v1.json").read_text())
    assert bool(receipt["sync_errors"]) == fail_sync
