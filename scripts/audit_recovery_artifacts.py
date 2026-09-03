#!/usr/bin/env python3
"""Read back completed run artifacts from encrypted S3; never modify AWS resources."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.sync_run_artifacts import DEFAULT_BUCKET, _eligible
from tetris.rl import atomic_write_json, directory_sha256, file_sha256


def audit_run(client, bucket: str, run_id: str, *, allow_failed_block: bool = False) -> dict:
    root = Path("runs") / run_id
    manifest_path = root / "rl/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    block = json.loads((root / "rl/block-state.json").read_text()) if (root / "rl/block-state.json").exists() else None
    allowed = ("passed", "not_passed", "failed") if allow_failed_block else ("passed", "not_passed")
    if manifest["status"] != "completed" or (block and block["status"] not in allowed):
        raise ValueError("only finished training/evaluation blocks can be audited")
    receipt = json.loads((root / "rl/sync-receipt.json").read_text())
    if receipt.get("status") != "passed" or receipt.get("direction") != "upload" or not receipt.get("included_adapter") or receipt.get("optimizer_checkpoints_included"):
        raise ValueError("completed final-adapter upload receipt required")
    adapter_path = root / "rl/adapter"
    expected_adapter_hash = manifest["output_adapter_sha256"]
    if directory_sha256(adapter_path) != expected_adapter_hash:
        raise ValueError("local final adapter differs from training manifest")
    prefix = f"runs/{run_id}/"
    remote_keys = {item["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix) for item in page.get("Contents", [])}
    if any(any(part.startswith("checkpoint-") or part in ("optimizer.pt", "state.pt", "scheduler.pt") for part in key.split("/")) for key in remote_keys):
        raise ValueError("optimizer checkpoint unexpectedly present in remote run")
    adapter_keys = {prefix + p.relative_to(root).as_posix() for p in adapter_path.rglob("*") if p.is_file()}
    if {key for key in remote_keys if key.startswith(prefix + "rl/adapter/")} != adapter_keys:
        raise ValueError("S3 adapter file set differs from retained local adapter")
    digests, aggregate = {}, hashlib.sha256()
    paths = sorted(p for p in root.rglob("*") if p.is_file() and _eligible(p.relative_to(root), True))
    for path in paths:
        relative = path.relative_to(root)
        # Local independent checks and download receipts are not worker-produced artifacts.
        if "local-check" in path.name or "download-receipt" in path.name:
            continue
        key = prefix + relative.as_posix()
        if key not in remote_keys:
            raise ValueError(f"missing remote artifact: {key}")
        response = client.get_object(Bucket=bucket, Key=key)
        if response.get("ServerSideEncryption") != "AES256":
            raise ValueError(f"S3 artifact is not AES256 encrypted: {key}")
        digest = hashlib.sha256()
        in_adapter = path.is_relative_to(adapter_path)
        if in_adapter:
            aggregate.update(path.relative_to(adapter_path).as_posix().encode())
            aggregate.update(b"\0")
        body = response["Body"]
        try:
            for chunk in iter(lambda: body.read(1024 * 1024), b""):
                digest.update(chunk)
                if in_adapter:
                    aggregate.update(chunk)
        finally:
            body.close()
        if in_adapter:
            aggregate.update(b"\0")
        if digest.hexdigest() != file_sha256(path):
            raise ValueError(f"remote content mismatch: {key}")
        digests[key] = digest.hexdigest()
    if aggregate.hexdigest() != expected_adapter_hash:
        raise ValueError("S3 adapter directory hash differs from training output")
    return {"run_id": run_id, "status": "passed", "block_status": block["status"] if block else None,
        "artifact_integrity_only": True, "adapter_sha256": expected_adapter_hash,
        "adapter_directory_hash_recomputed_from_s3": True, "encryption": "AES256",
        "objects_verified": len(digests), "object_sha256": digests,
        "training_manifest_sha256": file_sha256(manifest_path), "optimizer_checkpoints_present": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-failed-block", action="store_true", help="audit retained completed training from a failed proof; never mark the experiment passed")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("audit output must be a new path")
    import boto3
    client = boto3.client("s3", region_name="us-east-1")
    rows = [audit_run(client, args.bucket, run_id, allow_failed_block=args.allow_failed_block) for run_id in args.run_id]
    atomic_write_json(args.out, {"status": "passed", "kind": "encrypted_s3_content_audit", "bucket": args.bucket,
        "runs": rows, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"verified {sum(r['objects_verified'] for r in rows)} encrypted artifacts from {len(rows)} runs")


if __name__ == "__main__":
    main()
