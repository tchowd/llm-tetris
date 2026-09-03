#!/usr/bin/env python3
"""Idempotent, explicitly invoked launch of the one authorized L40S worker."""
from __future__ import annotations
import argparse
import base64
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.stage6_scale10x import ROOT, REGISTRATION, PILOT, validate_registration, write_new, now
from tetris.rl import atomic_write_json, file_sha256


SUBNETS = ("subnet-094caed03deaaae73", "subnet-011c567e0b89bdea6",
    "subnet-07af2623089657057", "subnet-01df3bd9aafb4de79")


def launch_request(subnet=SUBNETS[0]):
    if subnet not in SUBNETS:
        raise ValueError("only inspected existing default-VPC subnets are eligible")
    tags = [{"Key": "Project", "Value": "llm-tetris"}, {"Key": "Stage", "Value": "6"},
        {"Key": "RunId", "Value": PILOT}, {"Key": "ManagedBy", "Value": "llm-tetris"}]
    return {"ImageId": "ami-0a4870b172edcb0f2", "InstanceType": "g6e.2xlarge",
        "MinCount": 1, "MaxCount": 1, "ClientToken": "llm-tetris-scale10x-20260902-v2-" + subnet,
        "KeyName": "gpu-training", "SubnetId": subnet,
        "SecurityGroupIds": ["sg-0a3c367cb69c4ea87"],
        "IamInstanceProfile": {"Name": "LLMTetrisTelemetryProfile"},
        "InstanceInitiatedShutdownBehavior": "stop",
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled", "HttpPutResponseHopLimit": 1},
        "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 100,
            "VolumeType": "gp3", "Encrypted": True, "DeleteOnTermination": True}}],
        "TagSpecifications": [{"ResourceType": "instance", "Tags": tags + [
            {"Key": "Name", "Value": "llm-tetris-rl-scale10x"}, {"Key": "PilotDollarLimit", "Value": "50"}]},
            {"ResourceType": "volume", "Tags": tags}],
        "UserData": Path("infra/rl-scale10x-bootstrap.sh").read_text()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--subnet-id", choices=SUBNETS, default=SUBNETS[0])
    args = parser.parse_args()
    import boto3
    from botocore.exceptions import ClientError
    r, _ = validate_registration()
    client = boto3.client("ec2", region_name="us-east-1")
    request = launch_request(args.subnet_id)
    try:
        client.run_instances(**request, DryRun=True)
    except ClientError as error:
        if error.response["Error"]["Code"] != "DryRunOperation":
            raise
    else:
        raise ValueError("unexpected dry-run response")
    if not args.launch:
        print("Dry run passed; nothing created")
        return
    ledger_path = ROOT / "compute-ledger.json"
    if ledger_path.exists():
        raise ValueError("launch already recorded; resolve exact existing instance instead of relaunching")
    active = client.describe_instances(Filters=[{"Name": "tag:Project", "Values": ["llm-tetris"]},
        {"Name": "tag:Stage", "Values": ["6"]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])
    existing = [i for reservation in active["Reservations"] for i in reservation["Instances"]]
    if existing:
        # A lost success response must be reconciled, never followed by a second launch.
        raise ValueError("a Stage 6 worker already exists; reconcile its client token and ledger before continuing")
    request_path = ROOT / ("launch-request-v2-" + args.subnet_id + ".json")
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("saved request changed; do not reuse an idempotency token")
    else:
        write_new(request_path, request)
    try:
        response = client.run_instances(**request)
    except ClientError as error:
        atomic_write_json(ROOT / ("launch-failure-" + str(__import__('time').time_ns()) + ".json"), {
            "recorded_at": now(), "request_sha256": file_sha256(request_path),
            "error_code": error.response["Error"]["Code"], "message": error.response["Error"]["Message"],
            "worker_created_confirmed": False, "registration_sha256": file_sha256(REGISTRATION)})
        raise
    instance = response["Instances"][0]
    started = instance["LaunchTime"].timestamp()
    atomic_write_json(ledger_path, {"status": "launched", "recorded_at": now(),
        "instance_id": instance["InstanceId"], "instance_type": instance["InstanceType"],
        "region": "us-east-1", "launch_epoch": started, "deadline_epoch": started + 43200,
        "registration_sha256": file_sha256(REGISTRATION), "request_sha256": file_sha256(request_path),
        "request_path": str(request_path), "subnet_id": args.subnet_id,
        "hourly_allowance_usd": r["hourly_usd"], "experiment_cap_usd": 50,
        "cumulative_stage6_cap_usd": 250, "prior_stage6_estimate_usd": 15.525172222222222,
        "extra_service_reserve_usd": 2, "root_volume_id": None,
        "estimated_experiment_usd": 0, "compute_sessions": [{"start_epoch": started, "end_epoch": None}],
        "stage4_untouched": True, "operations_complete": False})
    print(json.dumps({"instance_id": instance["InstanceId"], "deadline_epoch": started + 43200}))


if __name__ == "__main__":
    main()
