#!/usr/bin/env python3
"""Read-only Stage 6 AWS cleanup and encryption check.

This command never stops or terminates resources. It exits non-zero while a
tagged Stage 6 instance is pending/running/stopping/stopped, when an attached
volume is unencrypted, or when an available tagged volume remains.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import boto3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    ec2 = session.client("ec2")
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Project", "Values": ["llm-tetris"]},
            {"Name": "tag:Stage", "Values": ["6"]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
        ]
    )
    instances = [item for reservation in response["Reservations"] for item in reservation["Instances"]]
    volumes_response = ec2.describe_volumes(
        Filters=[
            {"Name": "tag:Project", "Values": ["llm-tetris"]},
            {"Name": "tag:Stage", "Values": ["6"]},
        ]
    )
    volumes = volumes_response["Volumes"]
    checks = {
        "no_live_or_stopped_instances": not instances,
        "all_tagged_volumes_encrypted": all(item.get("Encrypted") for item in volumes),
        "no_available_persistent_volumes": not any(item.get("State") == "available" for item in volumes),
    }
    report = {
        "stage": 6,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "instances": [{"instance_id": item["InstanceId"], "state": item["State"]["Name"]} for item in instances],
        "volumes": [{"volume_id": item["VolumeId"], "state": item["State"], "encrypted": item.get("Encrypted")} for item in volumes],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status']}: wrote {args.out}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
