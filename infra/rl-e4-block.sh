#!/usr/bin/env bash
# Fixed dense pilot, registered after independent verification of E3 selection.
set -euo pipefail
cd /home/ubuntu/llm-tetris
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
py=.venv-rl/bin/python
run=rl-e4-seed0
out="runs/$run/rl"
registration="$out/registration.json"
phase() {
  "$py" -c 'import sys,time; from pathlib import Path; from tetris.rl import atomic_write_json; atomic_write_json(Path(sys.argv[1]),{"experiment":"E4","status":"running","phase":sys.argv[2],"updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})' "$out/block-state.json" "$1"
}
if [[ ${1:-} != --workflow ]]; then
  [[ ! -e "$out/block-state.json" && ! -e "$out/manifest.json" ]]
  "$py" scripts/check_e4_pilot.py --registration "$registration" --preflight
  "$py" -c 'import json,math; from pathlib import Path; from transformers.utils.hub import cached_file; from tetris.rl import directory_sha256; r=json.loads(Path("runs/rl-e4-seed0/rl/registration.json").read_text()); ledger=json.loads(Path("runs/stage6-aws/rl/manifest.json").read_text()); spent=ledger["estimated_accrued_usd"]; assert math.isfinite(spent) and 0<=spent<=r["budgets"]["prior_stage_spend_usd"]; assert directory_sha256(Path("runs/sft-v1/adapter"))==r["frozen_sft_adapter_sha256"]; assert Path(cached_file(r["base_model"],"config.json",local_files_only=True)).parent.name==r["base_model_revision"]'
  finish() {
    result=$?
    trap - EXIT
    set +e
    "$py" -c 'import json,sys,time; from pathlib import Path; from tetris.rl import atomic_write_json; p=Path(sys.argv[1]); m=json.loads(p.read_text()) if p.exists() else {}; code=int(sys.argv[2]); m.update(status={0:"passed",2:"not_passed",124:"timed_out"}.get(code,"failed"),exit_code=code,finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())); atomic_write_json(p,m)' "$out/block-state.json" "$result"
    timeout --kill-after=10s 8m "$py" scripts/sync_run_artifacts.py upload --run-id "$run" --include-adapter --receipt "$out/sync-receipt.json"
    timeout --kill-after=10s 30s "$py" scripts/sync_run_artifacts.py upload --run-id "$run"
    sudo /usr/sbin/shutdown -h now
    exit "$result"
  }
  trap finish EXIT
  phase starting
  timeout --signal=INT --kill-after=60s 170m bash "$0" --workflow > "$out/block.log" 2>&1
  exit 0
fi
beta=$("$py" -c 'import json; print(json.load(open("runs/rl-e4-seed0/rl/registration.json"))["kl_beta"])')
prior=$("$py" -c 'import json; print(json.load(open("runs/rl-e4-seed0/rl/registration.json"))["budgets"]["prior_stage_spend_usd"]+3*1.05)')
phase dense_training
timeout --signal=INT --kill-after=60s 65m "$py" scripts/train_rl.py \
  --experiment E4 --question 'Does dense one-step learning transfer to the development stress suite without regressing frozen Stage 5?' \
  --initialization-kind sft --adapter-dir runs/sft-v1/adapter --out-dir "$out" \
  --max-updates 512 --states 256 --group-size 4 --batch-size 4 --grad-accum 4 \
  --learning-rate 0.000001 --kl-beta "$beta" --seed 0 --save-steps 32 \
  --instance-hourly-usd 1.05 --max-wall-clock-hours 1 --pilot-dollar-limit 20 \
  --stage-dollar-limit 100 --prior-stage-spend-usd "$prior" > "$out/train.log" 2>&1
"$py" -c 'import json; from pathlib import Path; from scripts.select_e3_kl import validate_training; r=json.loads(Path("runs/rl-e4-seed0/rl/registration.json").read_text()); m=json.loads(Path("runs/rl-e4-seed0/rl/manifest.json").read_text()); validate_training(m,r,r["kl_beta"],experiment="E4")'
timeout --kill-after=10s 4m "$py" scripts/sync_run_artifacts.py upload --run-id "$run" --include-adapter --receipt "$out/training-sync-receipt.json"
phase development_evaluation
timeout --signal=INT --kill-after=60s 75m "$py" scripts/eval_stress.py \
  --suite development --policies model --policy-label e4 --adapter-dir "$out/adapter" \
  --data-dirs data/batch1 data/batch2 --out-dir "$out/stress-development" > "$out/eval.log" 2>&1
phase stage5_non_inferiority
timeout --signal=INT --kill-after=60s 60m "$py" scripts/eval_closed_loop.py \
  --policies model --modes strict --model-label e4 --adapter-dir "$out/adapter" \
  --data-dirs data/batch1 data/batch2 --gen-batch-size 64 --teacher-workers 3 \
  --device cuda --out-dir "$out/stage5" > "$out/stage5.log" 2>&1
phase paired_analysis
"$py" scripts/check_e4_pilot.py --registration "$registration" --out "$out/e4-gate.json"
