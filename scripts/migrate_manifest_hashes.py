#!/usr/bin/env python3
"""Add explicit content hashes to legacy dataset manifests without regenerating data."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from tetris.events import file_sha256


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def migrate(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    games_path = data_dir / "games.jsonl"
    rows_path = data_dir / "rows.jsonl"
    for required in (manifest_path, games_path, rows_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    original_bytes = manifest_path.read_bytes()
    manifest = json.loads(original_bytes)
    hashes = {
        "games.jsonl": file_sha256(games_path),
        "rows.jsonl": file_sha256(rows_path),
    }
    existing = manifest.get("content_hashes")
    if existing is not None and existing != hashes:
        raise ValueError(f"{manifest_path}: existing content_hashes do not match current files")
    if existing == hashes:
        return manifest

    manifest["content_hashes"] = hashes
    manifest["lineage_migration"] = {
        "type": "add_content_hashes",
        "original_manifest_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "git_sha": git_sha(),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, action="append", required=True)
    args = parser.parse_args()
    for data_dir in args.data_dir:
        result = migrate(data_dir)
        print(f"{data_dir}: {result['content_hashes']}")


if __name__ == "__main__":
    main()
