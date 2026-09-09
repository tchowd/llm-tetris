#!/usr/bin/env python3
"""Budget validation, packaging, launch, and status for recovery-v2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import base64
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time

from tetris.rl import file_sha256

REGISTRATION = Path("experiments/tetra17-recovery-v2/registration.json")
BUCKET = "llm-tetris-artifacts-566629888938-us-east-1"
PREFIX = "runs/tetra17-recovery-v2"
REGION = "us-east-1"
RATE = 2.24208


def read(path: Path):
    return json.loads(path.read_text())


def validate_approval(path: Path) -> dict:
    approval, registration = read(path), read(REGISTRATION)
    if approval.get("experiment") != "tetra17-recovery-v2" or approval.get("status") != "user_approved":
        raise ValueError("new approval for tetra17-recovery-v2 is required")
    if approval.get("registration_sha256") != file_sha256(REGISTRATION):
        raise ValueError("approval does not cover the current registration")
    if not approval.get("user_approval_text") or approval.get("hard_limit_usd", 0) <= 0:
        raise ValueError("approval needs explicit text and a positive hard limit")
    if approval.get("instance_hourly_usd") != RATE or approval.get("maximum_total_instance_hours", 0) * RATE + approval.get("ancillary_reserve_usd", 0) > approval["hard_limit_usd"]:
        raise ValueError("approved compute envelope exceeds hard limit")
    prior = approval.get("prior_instance_hours_conservative", 0)
    new = approval.get("maximum_new_instance_hours", approval.get("maximum_total_instance_hours", 0))
    cumulative = approval.get("maximum_cumulative_instance_hours", prior + new)
    if prior + new > cumulative or cumulative > approval.get("maximum_total_instance_hours", 0):
        raise ValueError("cumulative instance-hour ledger exceeds approval")
    deadline = datetime.fromisoformat(approval["absolute_deadline_utc"].replace("Z", "+00:00"))
    if deadline <= datetime.now(timezone.utc):
        raise ValueError("approval deadline expired")
    if registration["status"] != "registered_awaiting_budget_approval":
        raise ValueError("experiment is not awaiting this approval")
    return approval


def bundle(approval: Path) -> Path:
    validate_approval(approval)
    selected = [Path("pyproject.toml"), Path("requirements-train.txt"), Path("requirements-train-unsloth.txt"), Path("requirements-rl.txt"), REGISTRATION, approval,
                Path("experiments/tetra17-recovery-v2/budget-approval-v1.json"), Path("experiments/tetra17-recovery-v2/budget-approval-v2.json"),
                Path("experiments/tetra17-recovery-v2/budget-amendment-v2.json"), Path("experiments/tetra17-recovery-v2/budget-amendment-v3.json"),
                Path("experiments/tetra17-recovery-v2/registration-amendment-ops-v1.json"), Path("experiments/tetra17-recovery-v2/registration-amendment-ops-v2.json"),
                Path("experiments/tetra17-recovery-v2/registration-amendment-ops-v3.json"),
                Path("experiments/tetra17-recovery-v2/registration-amendment-ops-v4.json"),
                Path("experiments/tetra17-recovery-v2/registration-amendment-ops-v5.json"),
                Path("experiments/tetra17-recovery-v2/registration-amendment-data-v1.json"),
                Path("experiments/tetra17-recovery-v2/audit/data-and-failures.json"), Path("experiments/tetra17-recovery-v2/audit/final-dataset.json"),
                Path("experiments/stage6-feedback-v1/recovery-development.jsonl"),
                Path("benchmarks/stress-v1/manifest.json"), Path("benchmarks/stress-v1/states.jsonl"),
                Path("data/batch1/rows.jsonl"), Path("data/batch1/manifest.json"), Path("data/batch2/rows.jsonl"), Path("data/batch2/manifest.json"),
                Path("data/stage6-recovery-v1/train-starts.jsonl"), Path("data/stage6-recovery-v1/training-seeds.json")]
    for directory in (Path("tetris"), Path("scripts"), Path("tests"), Path("runs/sft-v1/adapter")):
        selected.extend(path for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    selected.extend([Path("infra/tetra17_recovery_bootstrap.sh"), Path("infra/tetra17_recovery_resume.sh"), Path("infra/tetra17_recovery_ops.py")])
    temp = Path(tempfile.mkdtemp(prefix="tetra17-recovery-v2-")) / "source.tgz"
    with tarfile.open(temp, "w:gz") as archive:
        for path in sorted(set(selected)):
            archive.add(path, arcname=str(path), recursive=False)
    return temp


def launch(approval_path: Path) -> None:
    approval = validate_approval(approval_path)
    archive = bundle(approval_path)
    subprocess.run(["aws", "s3", "cp", str(archive), f"s3://{BUCKET}/{PREFIX}/input/source.tgz", "--only-show-errors"], check=True)
    import boto3
    boto3.client("iam").put_role_policy(RoleName="LLMTetrisTelemetryRole", PolicyName="Tetra17RecoveryV2Read", PolicyDocument=json.dumps({
        "Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": ["s3:GetObject", "s3:ListBucket"],
        "Resource": [f"arn:aws:s3:::{BUCKET}", f"arn:aws:s3:::{BUCKET}/{PREFIX}/*"]}]}))
    deadline = datetime.fromisoformat(approval["absolute_deadline_utc"].replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S UTC")
    user_data = f'''#!/bin/bash
set -euo pipefail
systemd-run --unit=tetra17-bootstrap-deadline --on-calendar="{deadline}" /usr/sbin/shutdown -h now
mkdir -p /home/ubuntu/llm-tetris
chown ubuntu:ubuntu /home/ubuntu/llm-tetris
sudo -u ubuntu aws s3 cp s3://{BUCKET}/{PREFIX}/input/source.tgz /home/ubuntu/source.tgz --only-show-errors
sudo -u ubuntu tar -xzf /home/ubuntu/source.tgz -C /home/ubuntu/llm-tetris
sudo -u ubuntu bash /home/ubuntu/llm-tetris/infra/tetra17_recovery_bootstrap.sh > /home/ubuntu/llm-tetris/experiments/tetra17-recovery-v2/worker.log 2>&1
'''
    request = {
        "ImageId": "ami-0a4870b172edcb0f2", "InstanceType": "g6e.2xlarge", "MinCount": 1, "MaxCount": 1,
        "KeyName": "gpu-training", "SubnetId": "subnet-07af2623089657057", "SecurityGroupIds": ["sg-0a3c367cb69c4ea87"],
        "IamInstanceProfile": {"Name": "LLMTetrisTelemetryProfile"},
        "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 250, "VolumeType": "gp3", "DeleteOnTermination": True, "Encrypted": True}}],
        "UserData": base64.b64encode(user_data.encode()).decode(),
        "TagSpecifications": [{"ResourceType": "instance", "Tags": [{"Key": "RunId", "Value": "tetra17-recovery-v2-primary"}, {"Key": "Project", "Value": "llm-tetris"}]}],
        "InstanceInitiatedShutdownBehavior": "terminate", "ClientToken": f"tetra17-recovery-v2-{file_sha256(REGISTRATION)[:16]}",
    }
    response = boto3.client("ec2", region_name=REGION).run_instances(**request)
    receipt = {"instance_id": response["Instances"][0]["InstanceId"], "launched_at": time.time(), "registration_sha256": file_sha256(REGISTRATION), "approval_sha256": file_sha256(approval_path), "request": request}
    Path("experiments/tetra17-recovery-v2/launch-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("instance_id", "launched_at", "registration_sha256")}, indent=2))


def status() -> None:
    import boto3
    response = boto3.client("ec2", region_name=REGION).describe_instances(Filters=[{"Name": "tag:RunId", "Values": ["tetra17-recovery-v2-*"]}])
    rows = [{"id": i["InstanceId"], "state": i["State"]["Name"], "type": i["InstanceType"], "launch_time": i["LaunchTime"].isoformat()} for reservation in response["Reservations"] for i in reservation["Instances"]]
    print(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "status", "verify-worker-budget"))
    parser.add_argument("--approval", type=Path, default=Path("experiments/tetra17-recovery-v2/budget-approval.json"))
    args = parser.parse_args()
    if args.action == "launch": launch(args.approval)
    elif args.action == "status": status()
    else: print(json.dumps(validate_approval(args.approval), indent=2))


if __name__ == "__main__":
    main()
