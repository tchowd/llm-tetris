#!/usr/bin/env python3
"""Validate a `games.jsonl` + `rows.jsonl` dump against every check in
plan/stage-3-dataset.md's "Validation" section (the stage gate).

    python scripts/validate_dataset.py --data-dir data/smoke
    python scripts/validate_dataset.py --data-dir data/smoke --sample 20 --sample-out sample.txt

Exits non-zero if any automated check (#1,2,3,4,5,7) fails. Check #6
("read a few thousand") is not automatable by design -- use --sample to
produce text for a human to actually read.
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
    parser.add_argument("--max-pieces", type=int, default=DEFAULT_MAX_PIECES)
    parser.add_argument("--workers", type=int, default=None, help="processes for full_rebuild (default: cpu_count - 2)")
    parser.add_argument("--sample", type=int, default=0, help="also dump N random rows as text for a human to read")
    parser.add_argument("--sample-out", type=Path, default=None, help="file to write the sample to (default: stdout)")
    args = parser.parse_args()

    games = _load_jsonl(args.data_dir / "games.jsonl")
    rows = _load_jsonl(args.data_dir / "rows.jsonl")
    print(f"loaded {len(games)} games, {len(rows)} rows from {args.data_dir}")

    report = run_all_checks(games, rows, max_pieces=args.max_pieces, workers=args.workers)

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
