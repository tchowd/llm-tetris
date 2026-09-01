#!/usr/bin/env python3
"""Publish or retrieve dashboard run artifacts without copying checkpoints."""
from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath

import boto3


DEFAULT_BUCKET = "llm-tetris-artifacts-566629888938-us-east-1"
METADATA_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".jinja"}


def _eligible(relative: Path, include_adapter: bool) -> bool:
    if any(part.startswith("checkpoint-") or part.startswith(".") for part in relative.parts):
        return False
    if "adapter" in relative.parts:
        return include_adapter
    return relative.suffix in METADATA_SUFFIXES


def upload_run(client, *, bucket: str, run_id: str, run_dir: Path, include_adapter: bool) -> int:
    uploaded = 0
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if not _eligible(relative, include_adapter):
            continue
        key = f"runs/{run_id}/{relative.as_posix()}"
        client.upload_file(str(path), bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"})
        print(f"uploaded s3://{bucket}/{key}")
        uploaded += 1
    return uploaded


def download_run(client, *, bucket: str, run_id: str, run_dir: Path, include_adapter: bool) -> int:
    prefix = f"runs/{run_id}/"
    downloaded = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            relative_posix = PurePosixPath(item["Key"].removeprefix(prefix))
            if not relative_posix.parts or ".." in relative_posix.parts:
                continue
            relative = Path(*relative_posix.parts)
            if not _eligible(relative, include_adapter):
                continue
            destination = run_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.download")
            client.download_file(bucket, item["Key"], str(temporary))
            temporary.replace(destination)
            print(f"downloaded {destination}")
            downloaded += 1
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("upload", "download"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--bucket", default=os.environ.get("LLM_TETRIS_ARTIFACT_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--include-adapter", action="store_true", help="also transfer final adapter files; checkpoints are always excluded")
    args = parser.parse_args()

    run_dir = args.runs_dir / args.run_id
    if args.direction == "upload" and not run_dir.is_dir():
        raise SystemExit(f"run directory does not exist: {run_dir}")

    client = boto3.client("s3", region_name=args.region)
    operation = upload_run if args.direction == "upload" else download_run
    count = operation(
        client,
        bucket=args.bucket,
        run_id=args.run_id,
        run_dir=run_dir,
        include_adapter=args.include_adapter,
    )
    print(f"{args.direction} complete: {count} objects for {args.run_id}")


if __name__ == "__main__":
    main()
