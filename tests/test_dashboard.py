import json

from tetris import dashboard
from tetris import events
from tetris.events import EventWriter


def configure_paths(monkeypatch, root):
    monkeypatch.setattr(dashboard, "ROOT", root)
    monkeypatch.setattr(dashboard, "DATA_DIR", root / "data")
    monkeypatch.setattr(dashboard, "RUNS_DIR", root / "runs")
    monkeypatch.setattr(dashboard, "STATUS_DIR", root / "runs/status")


def test_dataset_discovery_keeps_missing_validation_explicit(tmp_path, monkeypatch):
    configure_paths(monkeypatch, tmp_path)
    batch = tmp_path / "data/batch1"
    batch.mkdir(parents=True)
    (batch / "games.jsonl").write_text("")
    (batch / "rows.jsonl").write_text("")
    (batch / "manifest.json").write_text(json.dumps({"num_games": 2, "num_rows": 8, "died_count": 1, "git_sha": "abc", "search_depth": 2}))

    result = dashboard.discover_datasets()

    assert result["totals"] == {"batches": 1, "games": 2, "rows": 8, "deaths": 1, "validated_batches": 0}
    assert result["batches"][0]["validation"]["status"] == "missing"
    assert result["batches"][0]["files"]["rows_exists"] is True


def test_dataset_validation_report_is_counted(tmp_path, monkeypatch):
    configure_paths(monkeypatch, tmp_path)
    batch = tmp_path / "data/batch1"
    batch.mkdir(parents=True)
    (batch / "games.jsonl").write_text("")
    (batch / "rows.jsonl").write_text("")
    (batch / "manifest.json").write_text(json.dumps({"num_games": 1, "num_rows": 4, "died_count": 0}))
    (batch / "validation.json").write_text(json.dumps({"checks": {"replay": {"ok": True}, "legality": {"ok": True}}}))

    result = dashboard.discover_datasets()

    validation = result["batches"][0]["validation"]
    assert validation["status"] == "passed"
    assert validation["checks_passed"] == validation["checks_total"] == 2


def test_run_discovery_reads_progress_without_loading_whole_event_file(tmp_path, monkeypatch):
    configure_paths(monkeypatch, tmp_path)
    run = tmp_path / "runs/sft-v1"
    run.mkdir(parents=True)
    (run / "train_manifest.json").write_text(json.dumps({"run_id": "sft-v1", "stage": 4, "backend": "unsloth"}))
    writer = EventWriter(run / "events.jsonl", run_id="sft-v1", stage=4)
    writer.emit("job_started", phase="training", current=0, total=10)
    writer.emit("train_metrics", phase="training", current=4, total=10, metrics={"loss": 0.8})

    runs = dashboard.discover_runs()

    assert len(runs) == 1
    assert runs[0]["status"] == "running"
    assert runs[0]["progress"]["current"] == 4
    assert runs[0]["progress"]["metrics"]["loss"] == 0.8


def test_summary_always_contains_all_seven_stages_without_aws():
    payload = dashboard.dashboard_summary(include_aws=False)

    assert payload["partial"] is False
    assert [stage["number"] for stage in payload["data"]["stages"]] == list(range(1, 8))
    assert payload["data"]["project"]["next_action"]


def test_aws_permission_and_overbroad_policy_become_normalized_issues():
    issues = dashboard._aws_extended_issues(
        {"credits": None, "errors": [{"code": "AccessDenied", "source": "billing:GetCredits", "message": "denied"}]},
        {"quotas": [], "errors": []},
        {"warnings": [{"severity": "red", "title": "Over-broad policy attached: AdministratorAccess", "next_action": "Scope it."}], "errors": []},
        {"series": [], "errors": []},
        {"resources": [], "errors": []},
        {"jobs": [], "errors": []},
    )

    assert {(item["severity"], item["source"]) for item in issues} == {("amber", "aws"), ("red", "aws")}


def test_event_writer_pins_the_run_git_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "_git_sha", lambda: "start-sha")
    writer = EventWriter(tmp_path / "events.jsonl", run_id="run", stage=4)
    monkeypatch.setattr(events, "_git_sha", lambda: "later-sha")

    event = writer.emit("heartbeat")

    assert event["git_sha"] == "start-sha"


def test_active_cloudwatch_alarm_becomes_a_red_issue():
    issues = dashboard._aws_alarm_issues(
        {"alarms": [{"name": "llm-tetris-gpu-idle", "state": "ALARM", "reason": "idle", "region": "us-east-1"}]}
    )

    assert len(issues) == 1
    assert issues[0]["severity"] == "red"
    assert "gpu-idle" in issues[0]["title"]
