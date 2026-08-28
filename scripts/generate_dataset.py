#!/usr/bin/env python3
"""Generate `games.jsonl` + `rows.jsonl` from the Stage 2 teacher.

    python scripts/generate_dataset.py --games 500 --out-dir data/smoke

Games are independent, so this parallelizes across processes by seed range
(plan/stage-3-dataset.md's "Generator" section). Row/game order within the
output files follows completion order, not seed order — the *content* is
still exactly reproducible from (seed-start, games, max-pieces), just not
the on-disk line order. Run scripts/validate_dataset.py on the output
before trusting it, and never jump straight to a 1M-row run: validate a
few-hundred-game smoke dump first.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tetris.dataset import DEFAULT_MAX_PIECES, EVAL_SPLIT_PERCENT, NOISE_SALT, generate_game
from tetris.teacher import WEIGHTS


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent, text=True).strip()
    except Exception:
        return None


def _generate_one(args: tuple[int, int]) -> tuple[list[dict], dict]:
    seed, max_pieces = args
    return generate_game(seed, max_pieces=max_pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, required=True, help="number of games to generate")
    parser.add_argument("--seed-start", type=int, default=0, help="first seed; games use seed-start .. seed-start+games-1")
    parser.add_argument("--max-pieces", type=int, default=DEFAULT_MAX_PIECES)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    games_path = args.out_dir / "games.jsonl"
    rows_path = args.out_dir / "rows.jsonl"
    manifest_path = args.out_dir / "manifest.json"

    seeds = range(args.seed_start, args.seed_start + args.games)
    t0 = time.time()

    num_rows = 0
    num_games = 0
    died_count = 0
    with games_path.open("w") as games_f, rows_path.open("w") as rows_f:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_generate_one, (seed, args.max_pieces)) for seed in seeds]
            for i, future in enumerate(as_completed(futures), 1):
                rows, game_record = future.result()
                games_f.write(json.dumps(game_record) + "\n")
                for row in rows:
                    rows_f.write(json.dumps(row) + "\n")
                num_rows += len(rows)
                num_games += 1
                died_count += game_record["died"]
                if i % max(1, args.games // 20) == 0 or i == args.games:
                    elapsed = time.time() - t0
                    print(f"  {i}/{args.games} games, {num_rows} rows, {elapsed:.1f}s elapsed", flush=True)

    elapsed = time.time() - t0
    manifest = {
        "git_sha": _git_sha(),
        "teacher_weights": WEIGHTS,
        "search_depth": 2,
        "noise_salt": NOISE_SALT,
        "eval_split_percent": EVAL_SPLIT_PERCENT,
        "seed_start": args.seed_start,
        "num_games": num_games,
        "num_rows": num_rows,
        "died_count": died_count,
        "max_pieces": args.max_pieces,
        "workers": args.workers,
        "wall_clock_seconds": elapsed,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {num_games} games / {num_rows} rows to {args.out_dir} in {elapsed:.1f}s")
    print(f"deaths: {died_count}/{num_games} (rest hit the {args.max_pieces}-piece cap)")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
