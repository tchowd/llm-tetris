import json

import pytest

from scripts.migrate_manifest_hashes import migrate


def test_migrate_adds_hashes_and_is_idempotent(tmp_path):
    data_dir = tmp_path / "batch"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(json.dumps({"num_games": 1}))
    (data_dir / "games.jsonl").write_text("{}\n")
    (data_dir / "rows.jsonl").write_text("{}\n")

    first = migrate(data_dir)
    second = migrate(data_dir)

    assert first["content_hashes"] == second["content_hashes"]
    assert first["lineage_migration"]["type"] == "add_content_hashes"


def test_migrate_refuses_existing_mismatch(tmp_path):
    data_dir = tmp_path / "batch"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(json.dumps({"content_hashes": {"games.jsonl": "bad", "rows.jsonl": "bad"}}))
    (data_dir / "games.jsonl").write_text("{}\n")
    (data_dir / "rows.jsonl").write_text("{}\n")

    with pytest.raises(ValueError, match="do not match"):
        migrate(data_dir)
