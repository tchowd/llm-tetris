from __future__ import annotations

import io
import json

import pytest

from scripts.audit_recovery_artifacts import audit_run
from scripts.report_recovery_outcome import closure_decision
from tetris.rl import directory_sha256


def test_artifact_audit_checks_actual_remote_bytes_and_encryption(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "runs/example/rl"
    (root / "adapter").mkdir(parents=True)
    (root / "adapter/adapter_model.safetensors").write_bytes(b"adapter-test-fixture")
    (root / "adapter/adapter_config.json").write_text("{}")
    (root / "manifest.json").write_text(json.dumps({"status": "completed", "output_adapter_sha256": directory_sha256(root / "adapter")}))
    (root / "block-state.json").write_text(json.dumps({"status": "not_passed"}))
    (root / "sync-receipt.json").write_text(json.dumps({"status": "passed", "direction": "upload", "included_adapter": True, "optimizer_checkpoints_included": False}))
    contents = {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    class Client:
        encryption = "AES256"

        def get_paginator(self, name):
            return self

        def paginate(self, **kwargs):
            return [{"Contents": [{"Key": k} for k in contents]}]

        def get_object(self, Bucket, Key):
            return {"ServerSideEncryption": self.encryption, "Body": io.BytesIO(contents[Key])}

    client = Client()
    assert audit_run(client, "bucket", "example")["objects_verified"] == 5
    (root / "block-state.json").write_text(json.dumps({"status": "failed"}))
    contents["runs/example/rl/block-state.json"] = (root / "block-state.json").read_bytes()
    with pytest.raises(ValueError, match="finished"):
        audit_run(client, "bucket", "example")
    failed_backup = audit_run(client, "bucket", "example", allow_failed_block=True)
    assert failed_backup["status"] == "passed" and failed_backup["block_status"] == "failed"
    assert failed_backup["artifact_integrity_only"]
    (root / "block-state.json").write_text(json.dumps({"status": "not_passed"}))
    contents["runs/example/rl/block-state.json"] = (root / "block-state.json").read_bytes()
    client.encryption = None
    with pytest.raises(ValueError, match="encrypted"):
        audit_run(client, "bucket", "example")
    client.encryption = "AES256"
    contents["runs/example/rl/adapter/adapter_model.safetensors"] = b"corrupted"
    with pytest.raises(ValueError, match="content mismatch"):
        audit_run(client, "bucket", "example")


def test_negative_closure_does_not_promote_model_or_skip_required_work():
    good = [{"status": "not_passed"}, {"status": "passed"}, {"status": "not_passed"},
        {"status": "passed"}, {"status": "passed", "checks": {"clean": True}, "instances": [], "volumes": []}]
    result = closure_decision(*good)
    assert result["research_complete"] and result["operations_complete"]
    assert not result["new_model_promoted"]
    assert result["historical_e3_status"] == "not_passed"
    for index, bad in ((0, {"status": "running"}), (1, {"status": "not_passed"}),
                       (2, {"status": "passed"}), (3, {"status": "failed"}),
                       (4, {"status": "passed", "checks": {"clean": True}, "instances": ["still-stopped"]})):
        changed = list(good)
        changed[index] = bad
        with pytest.raises(ValueError):
            closure_decision(*changed)
