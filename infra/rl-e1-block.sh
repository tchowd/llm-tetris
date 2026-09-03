#!/usr/bin/env bash
# Invoke only after E0 passes; $1 is the recorded prior Stage 6 AWS spend.
set -euo pipefail
cd /home/ubuntu/llm-tetris
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
py=.venv-rl/bin/python
run_dir=runs/rl-e1-seed0/rl
mkdir -p "$run_dir"
finish() {
  result=$?
  trap - EXIT
  set +e
  "$py" scripts/sync_run_artifacts.py upload --run-id rl-e1-seed0 --include-adapter \
    --receipt "$run_dir/sync-receipt.json"
  "$py" scripts/sync_run_artifacts.py upload --run-id rl-e1-seed0 --include-adapter
  sudo /usr/sbin/shutdown -h now
  exit "$result"
}
trap finish EXIT
"$py" -c 'import json; from pathlib import Path; from scripts.analyze_stage6 import validate_evaluation; from tetris.rl import file_sha256; p=Path("benchmarks/stress-v1/manifest.json"); assert json.loads(Path("runs/stage6-e0/rl/reproduction.json").read_text())["status"] == "passed"; m=validate_evaluation(Path("runs/stage6-e0/rl/stress-development"), suite="development", benchmark=json.loads(p.read_text()), benchmark_hash=file_sha256(p)); assert m["status"] == "passed"'
common=(
  --experiment E1
  --question "Does dense GRPO fit with headroom, preserve the SFT reference, and resume optimizer/scheduler/sample accounting on A10G within one hour?"
  --initialization-kind sft --adapter-dir runs/sft-v1/adapter
  --out-dir "$run_dir" --max-updates 20 --states 128
  --group-size 4 --batch-size 4 --grad-accum 4 --save-steps 5
  --learning-rate 0.000001 --kl-beta 0.05 --seed 0
  --instance-hourly-usd 1.05 --max-wall-clock-hours 1
  --pilot-dollar-limit 20 --stage-dollar-limit 100
  --prior-stage-spend-usd "${1:?supply measured prior Stage 6 spend}"
)
timeout --signal=INT --kill-after=120s 70m "$py" scripts/train_rl.py "${common[@]}" \
  --pause-after-update 10 > "$run_dir/train.log" 2>&1
"$py" -c 'import json; from pathlib import Path; m=json.loads(Path("runs/rl-e1-seed0/rl/manifest.json").read_text()); assert m["status"] == "paused" and m["completed_updates"] == 10 and m["reference_frozen"]'
cp "$run_dir/manifest.json" "$run_dir/pause-manifest.json"
timeout --signal=INT --kill-after=120s 70m "$py" scripts/train_rl.py "${common[@]}" \
  --resume "$run_dir/checkpoint-10" >> "$run_dir/train.log" 2>&1
"$py" -c 'import json; from pathlib import Path; m=json.loads(Path("runs/rl-e1-seed0/rl/manifest.json").read_text()); assert m["status"] == "completed" and m["completed_updates"] == 20 and m["resumed_from_update"] == 10 and m["reference_frozen"]; assert m["rollout_statistics"]["completions"] == 320; assert m["peak_cuda_bytes"] < 0.9 * m["gpu_total_bytes"]'
