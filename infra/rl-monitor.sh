#!/usr/bin/env bash
# Worker-local cron telemetry. Never launches training or uploads checkpoints.
set -euo pipefail
cd /home/ubuntu/llm-tetris
date -u
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
systemctl list-units 'llm-tetris-*' --no-pager
for run_dir in runs/stage6-* runs/rl-*; do
  [[ -d "$run_dir" ]] || continue
  timeout --kill-after=10s 240s .venv-rl/bin/python scripts/sync_run_artifacts.py upload \
    --run-id "${run_dir#runs/}"
done
