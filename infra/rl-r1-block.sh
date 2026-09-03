#!/usr/bin/env bash
# Authorized recovery SFT experiment; all old E3 outcomes remain unchanged.
set -euo pipefail
cd /home/ubuntu/llm-tetris
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
py=.venv-rl/bin/python
run=rl-r1-recovery-sft-seed0
out="runs/$run/rl"
protocol=runs/stage6-recovery-v1/rl
registration="$protocol/registration.json"
phase() {
  "$py" -c 'import sys,time; from pathlib import Path; from tetris.rl import atomic_write_json; atomic_write_json(Path(sys.argv[1]),{"experiment":"R1","status":"running","phase":sys.argv[2],"updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})' "$out/block-state.json" "$1"
}
if [[ ${1:-} != --workflow ]]; then
  [[ ! -e "$out/block-state.json" && ! -e "$out/manifest.json" ]]
  mkdir -p "$out"
  finish() {
    result=$?
    trap - EXIT
    set +e
    "$py" -c 'import json,sys,time; from pathlib import Path; from tetris.rl import atomic_write_json; p=Path(sys.argv[1]); m=json.loads(p.read_text()) if p.exists() else {}; code=int(sys.argv[2]); m.update(status={0:"passed",2:"not_passed",124:"timed_out"}.get(code,"failed"),exit_code=code,finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())); atomic_write_json(p,m)' "$out/block-state.json" "$result"
    timeout --kill-after=10s 6m "$py" scripts/sync_run_artifacts.py upload --run-id "$run" --include-adapter --receipt "$out/sync-receipt.json"
    timeout --kill-after=10s 60s "$py" scripts/sync_run_artifacts.py upload --run-id "$run"
    timeout --kill-after=10s 60s "$py" scripts/sync_run_artifacts.py upload --run-id stage6-recovery-v1 --receipt "$protocol/sync-receipt.json"
    timeout --kill-after=10s 60s "$py" scripts/sync_run_artifacts.py upload --run-id stage6-recovery-v1
    sudo /usr/sbin/shutdown -h now
    exit "$result"
  }
  trap finish EXIT
  "$py" -c 'import json,math; from pathlib import Path; from scripts.train_recovery_sft import validate_inputs; from tetris.rl import directory_sha256,file_sha256; r=json.loads(Path("runs/stage6-recovery-v1/rl/registration.json").read_text()); d=validate_inputs(r,Path(r["data"]["data_dir"])); assert d["registration_sha256"]==file_sha256(Path("runs/stage6-recovery-v1/rl/registration.json")); assert directory_sha256(Path("runs/sft-v1/adapter"))==r["frozen_sft_adapter_sha256"]; m=json.loads(Path("runs/stage6-aws/rl/manifest.json").read_text()); spent=m["estimated_accrued_usd"]; assert math.isfinite(spent) and spent>=0 and spent+5*1.05<=100 and 5*1.05<=20'
  phase starting
  timeout --signal=INT --kill-after=60s 290m bash "$0" --workflow > "$out/block.log" 2>&1
  exit 0
fi
phase recovery_sft_training
timeout --signal=INT --kill-after=60s 65m "$py" scripts/train_recovery_sft.py --registration "$registration" > "$out/train.log" 2>&1
timeout --kill-after=10s 4m "$py" scripts/sync_run_artifacts.py upload --run-id "$run" --include-adapter --receipt "$out/training-sync-receipt.json"
phase development_evaluation
timeout --signal=INT --kill-after=60s 75m "$py" scripts/eval_stress.py --suite development --policies model \
  --policy-label r1 --adapter-dir "$out/adapter" --data-dirs data/batch1 data/batch2 \
  --out-dir "$out/stress-development" > "$out/eval.log" 2>&1
phase stage5_non_inferiority
timeout --signal=INT --kill-after=60s 60m "$py" scripts/eval_closed_loop.py --policies model --modes strict \
  --model-label r1 --adapter-dir "$out/adapter" --data-dirs data/batch1 data/batch2 \
  --gen-batch-size 64 --teacher-workers 3 --device cuda --out-dir "$out/stage5" > "$out/stage5.log" 2>&1
phase fresh_recovery_sft_control
timeout --signal=INT --kill-after=60s 20m "$py" scripts/eval_recovery_only.py --registration "$registration" \
  --adapter-dir runs/sft-v1/adapter --label sft --out-dir "$protocol/fresh-sft" > "$out/fresh-sft.log" 2>&1
phase fresh_recovery_teacher_control
timeout --signal=INT --kill-after=60s 20m "$py" scripts/eval_recovery_only.py --registration "$registration" \
  --label teacher --out-dir "$protocol/fresh-teacher" > "$out/fresh-teacher.log" 2>&1
phase fresh_recovery_candidate
timeout --signal=INT --kill-after=60s 20m "$py" scripts/eval_recovery_only.py --registration "$registration" \
  --adapter-dir "$out/adapter" --label r1 --out-dir "$out/fresh-recovery" > "$out/fresh.log" 2>&1
phase paired_analysis
"$py" scripts/check_recovery_pilot.py --registration "$registration" --out "$out/r1-gate.json"
