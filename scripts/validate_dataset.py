#!/usr/bin/env python3
"""Validate a `games.jsonl` + `rows.jsonl` dump against every check in
plan/stage-3-dataset.md's "Validation" section (the stage gate).

    python scripts/validate_dataset.py --data-dir data/smoke
    python scripts/validate_dataset.py --data-dir data/smoke --sample 20 --sample-out sample.txt

`--max-pieces` and the teacher weights default to whatever manifest.json
(written by generate_dataset.py) recorded for this dump, not this code's
current defaults -- a dump made with a non-default --max-pieces, or made
before the teacher's weights were last retuned, must still validate
against what it was actually generated with. Pass --max-pieces or
--ignore-manifest-weights to override.

Exits non-zero if any automated check fails. Check #6 ("read a few
thousand") is not automatable by design -- use --sample to produce text
for a human to actually read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tetris.dataset import DEFAULT_MAX_PIECES
from tetris.dataset_validate import run_all_checks, sample_rows_as_text


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-pieces", type=int, default=None, help="default: manifest.json's max_pieces, else the code default")
    parser.add_argument("--workers", type=int, default=None, help="processes for full_rebuild (default: cpu_count - 2)")
    parser.add_argument(
        "--ignore-manifest-weights", action="store_true",
        help="use the live teacher.WEIGHTS instead of manifest.json's recorded teacher_weights",
    )
    parser.add_argument("--sample", type=int, default=0, help="also dump N random rows as text for a human to read")
    parser.add_argument("--sample-out", type=Path, default=None, help="file to write the sample to (default: stdout)")
    args = parser.parse_args()

    games = _load_jsonl(args.data_dir / "games.jsonl")
    rows = _load_jsonl(args.data_dir / "rows.jsonl")
    print(f"loaded {len(games)} games, {len(rows)} rows from {args.data_dir}")

    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not manifest:
        print(f"warning: no manifest.json in {args.data_dir}; using code defaults for max_pieces and live teacher weights")

    max_pieces = args.max_pieces if args.max_pieces is not None else manifest.get("max_pieces", DEFAULT_MAX_PIECES)
    weights = None if args.ignore_manifest_weights else manifest.get("teacher_weights")
    print(f"max_pieces={max_pieces}, weights={'manifest' if weights else 'live teacher.WEIGHTS'}")

    report = run_all_checks(games, rows, max_pieces=max_pieces, workers=args.workers, weights=weights)

    all_ok = True
    for name, result in report.items():
        status = "PASS" if result["ok"] else "FAIL"
        print(f"[{status}] {name}: {result['detail']}")
        all_ok = all_ok and result["ok"]

    if args.sample:
        text = sample_rows_as_text(rows, n=args.sample)
        if args.sample_out:
            args.sample_out.write_text(text)
            print(f"wrote {args.sample} sampled rows to {args.sample_out} -- check #6 requires actually reading this")
        else:
            print("\n=== sample rows (check #6: read these) ===\n")
            print(text)

    if not all_ok:
        raise SystemExit("one or more validation checks FAILED -- do not train on this dump")
    print("\nall automated checks passed. Still do check #6 (read a sample) before trusting this at scale.")


if __name__ == "__main__":
    main()
