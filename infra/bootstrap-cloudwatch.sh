#!/usr/bin/env bash
set -euo pipefail

repo_dir="${1:-/home/ubuntu/llm-tetris}"
agent_ctl="/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl"

if [[ ! -x "$agent_ctl" ]]; then
  echo "Amazon CloudWatch agent is not installed at $agent_ctl" >&2
  echo "Install the unified CloudWatch agent for this AMI, then rerun this script." >&2
  exit 1
fi

sudo "$agent_ctl" -a fetch-config -m ec2 \
  -c "file:${repo_dir}/infra/cloudwatch-agent.json" -s

aws logs put-retention-policy \
  --log-group-name /llm-tetris/jobs \
  --retention-in-days 30 \
  --region "${AWS_REGION:-us-east-1}"

echo "CloudWatch logs and 60-second GPU/system metrics are configured."
