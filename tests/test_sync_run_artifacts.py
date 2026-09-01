from pathlib import Path

from scripts.sync_run_artifacts import download_run, upload_run


class FakePaginator:
    def __init__(self, keys):
        self.keys = keys

    def paginate(self, **_kwargs):
        return [{"Contents": [{"Key": key} for key in self.keys]}]


class FakeS3:
    def __init__(self, keys=()):
        self.keys = list(keys)
        self.uploads = []

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.uploads.append((Path(filename).name, bucket, key, ExtraArgs))

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator(self.keys)

    def download_file(self, _bucket, key, filename):
        Path(filename).write_text(key)


def test_upload_excludes_checkpoints_and_adapter_by_default(tmp_path):
    run = tmp_path / "runs/sft-v1"
    (run / "checkpoint-10").mkdir(parents=True)
    (run / "adapter").mkdir()
    (run / "metrics.json").write_text("{}")
    (run / "checkpoint-10/optimizer.pt").write_text("large")
    (run / "adapter/adapter_model.safetensors").write_text("weights")
    client = FakeS3()

    count = upload_run(client, bucket="bucket", run_id="sft-v1", run_dir=run, include_adapter=False)

    assert count == 1
    assert [row[2] for row in client.uploads] == ["runs/sft-v1/metrics.json"]


def test_download_rejects_traversal_and_can_include_adapter(tmp_path):
    client = FakeS3(
        [
            "runs/sft-v1/metrics.json",
            "runs/sft-v1/../outside.json",
            "runs/sft-v1/checkpoint-10/state.json",
            "runs/sft-v1/adapter/adapter_model.safetensors",
        ]
    )
    run = tmp_path / "runs/sft-v1"

    count = download_run(client, bucket="bucket", run_id="sft-v1", run_dir=run, include_adapter=True)

    assert count == 2
    assert (run / "metrics.json").exists()
    assert (run / "adapter/adapter_model.safetensors").exists()
    assert not (tmp_path / "runs/outside.json").exists()
