#!/usr/bin/env bash
# Run only the E0 controls, archive outputs, and stop the worker between blocks.
set -euo pipefail
cd /home/ubuntu/llm-tetris
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
py=.venv-rl/bin/python
mkdir -p runs/stage6-e0/rl
finish() {
  result=$?
  trap - EXIT
  set +e
  "$py" scripts/sync_run_artifacts.py upload --run-id stage6-e0 \
    --receipt runs/stage6-e0/rl/sync-receipt.json
  "$py" scripts/sync_run_artifacts.py upload --run-id stage6-e0
  sudo /usr/sbin/shutdown -h now
  exit "$result"
}
trap finish EXIT
timeout --signal=INT --kill-after=120s 2h "$py" scripts/eval_closed_loop.py \
  --policies model --modes strict --adapter-dir runs/sft-v1/adapter \
  --model-label sft --data-dirs data/batch1 data/batch2 \
  --gen-batch-size 64 --teacher-workers 3 --device cuda \
  --out-dir runs/stage6-e0/rl/stage5 > runs/stage6-e0/rl/stage5.log 2>&1
"$py" scripts/check_sft_reproduction.py --candidate runs/stage6-e0/rl/stage5 \
  --out runs/stage6-e0/rl/reproduction.json
timeout --signal=INT --kill-after=120s 3h "$py" scripts/eval_stress.py \
  --suite development --policies random,teacher,model \
  --adapter-dir runs/sft-v1/adapter --policy-label sft \
  --data-dirs data/batch1 data/batch2 --gen-batch-size 64 --teacher-workers 3 \
  --out-dir runs/stage6-e0/rl/stress-development > runs/stage6-e0/rl/stress.log 2>&1
