#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/llm-tetris
export PYTHONPATH=/home/ubuntu/llm-tetris PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
registration=experiments/tetra17-recovery-v2/registration.json
approval=experiments/tetra17-recovery-v2/budget-approval.json
bucket=llm-tetris-artifacts-566629888938-us-east-1
prefix=runs/tetra17-recovery-v2
hard_limit=$(python3 -c 'import json;print(json.load(open("experiments/tetra17-recovery-v2/budget-approval.json"))["hard_limit_usd"])')
prior_hours=$(python3 -c 'import json;print(json.load(open("experiments/tetra17-recovery-v2/budget-approval.json")).get("prior_instance_hours_conservative",0))')
started_epoch=$(date +%s)
monitor_pid=""

sync_artifacts() {
  aws s3 sync experiments/tetra17-recovery-v2 "s3://$bucket/$prefix/experiments" --only-show-errors
  aws s3 sync runs/tetra-1.7b-recovery-v2-sft-seed0 "s3://$bucket/$prefix/runs/sft" --only-show-errors 2>/dev/null || true
  for d in runs/tetra-1.7b-recovery-v2-rl-fixed-zero-seed*/rl; do
    [ -d "$d" ] && aws s3 sync "$d" "s3://$bucket/$prefix/$d" --only-show-errors
  done
}

monitor() {
  while true; do
    sleep 900
    now=$(date +%s)
    python3 -c 'import json,sys,time; prior=float(sys.argv[3]); elapsed=int(sys.argv[1])-int(sys.argv[2])+600; rate=float(sys.argv[4]); hours=prior+elapsed/3600; json.dump({"observed_at":time.time(),"prior_instance_hours_conservative":prior,"current_worker_hours_conservative":elapsed/3600,"cumulative_instance_hours_conservative":hours,"estimated_compute_usd":hours*rate},open("experiments/tetra17-recovery-v2/live-cost.json","w"),indent=2)' "$now" "$started_epoch" "$prior_hours" 2.24208
    sync_artifacts
  done
}

finish() {
  status=$?
  set +e
  [ -n "$monitor_pid" ] && kill "$monitor_pid" 2>/dev/null
  if [ "$status" -eq 0 ]; then state=completed; else state=failed_waiting_for_recovery; fi
  python3 -c 'import json,sys,time; json.dump({"status":sys.argv[1],"recorded_at":time.time()},open("experiments/tetra17-recovery-v2/worker-state.json","w"),indent=2)' "$state"
  sync_artifacts
  aws s3 cp "$approval" "s3://$bucket/$prefix/receipts/budget-approval.json" --only-show-errors
  if [ "$status" -eq 0 ]; then sudo shutdown -h now; fi
  exit "$status"
}
trap finish EXIT

.venv-rl/bin/python infra/tetra17_recovery_ops.py verify-worker-budget --approval "$approval"
.venv-rl/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('data/tetra17-recovery-v2/manifest.json').read_text())
audit = json.loads(Path('experiments/tetra17-recovery-v2/audit/final-dataset.json').read_text())
digest = hashlib.sha256(Path('data/tetra17-recovery-v2/rows.jsonl').read_bytes()).hexdigest()
assert manifest['status'] == 'completed' and manifest['rows_sha256'] == digest
assert manifest['num_train_rows'] == 240000 and manifest['num_eval_rows'] == 4096
assert audit['status'] == 'completed' and audit['final_test_access'] is False
assert audit['original']['identity']['unique_full_states'] == 244096
assert audit['original']['identity']['unique_conflicting_states'] == 0
assert audit['original']['identity']['unique_conflicting_prompts'] == 0
PY
monitor &
monitor_pid=$!
python3 -c 'import json,time; json.dump({"status":"running","phase":"sft_or_evaluation","recorded_at":time.time()},open("experiments/tetra17-recovery-v2/worker-state.json","w"),indent=2)'

if [ ! -f runs/tetra-1.7b-recovery-v2-sft-seed0/train_manifest.json ]; then
  mkdir -p runs/tetra-1.7b-recovery-v2-sft-seed0
  resume_args=()
  latest_checkpoint=$(find runs/tetra-1.7b-recovery-v2-sft-seed0 -maxdepth 1 -type d -name 'checkpoint-*' -exec test -f '{}/trainer_state.json' ';' -print 2>/dev/null | sort -V | tail -1)
  [ -n "$latest_checkpoint" ] && resume_args=(--resume-from-checkpoint "$latest_checkpoint")
  .venv-sft/bin/python scripts/train_sft.py --backend unsloth --base-model Qwen/Qwen3-1.7B --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --data-dirs data/tetra17-recovery-v2 --out-dir runs/tetra-1.7b-recovery-v2-sft-seed0 --epochs 1 --lr 1e-4 --batch-size 16 --grad-accum 4 --seed 0 --eval-steps 250 --save-steps 250 "${resume_args[@]}"
fi
.venv-rl/bin/python infra/tetra17_recovery_ops.py verify-worker-budget --approval "$approval"

for label in control candidate; do
  if [ "$label" = control ]; then adapter=runs/sft-v1/adapter; else adapter=runs/tetra-1.7b-recovery-v2-sft-seed0/adapter; fi
  .venv-rl/bin/python scripts/eval_tetra17_matched.py --registration "$registration" --label "$label"
  .venv-rl/bin/python scripts/eval_open_loop.py --base-model Qwen/Qwen3-1.7B --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --data-dirs data/batch1 data/batch2 --adapter-dir "$adapter" --max-rows 10000 --seed 1702 --out "experiments/tetra17-recovery-v2/evaluation/$label/open-loop.json"
  .venv-rl/bin/python scripts/eval_closed_loop.py --policies model --modes strict --model-label "$label" --base-model Qwen/Qwen3-1.7B --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --adapter-dir "$adapter" --data-dirs data/batch1 data/batch2 --num-seeds 100 --cap 500 --out-dir "experiments/tetra17-recovery-v2/evaluation/$label/stage5"
  .venv-rl/bin/python scripts/eval_stress.py --suite development --policies model --policy-label "$label" --base-model Qwen/Qwen3-1.7B --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --adapter-dir "$adapter" --data-dirs data/batch1 data/batch2 --out-dir "experiments/tetra17-recovery-v2/evaluation/$label/stress-development"
  sync_artifacts
done
.venv-rl/bin/python scripts/analyze_tetra17_sft.py --registration "$registration"

if [ "$(.venv-rl/bin/python -c 'import json;print(str(json.load(open("experiments/tetra17-recovery-v2/sft-gate.json"))["conditional_rl_authorized_by_gate"]).lower())')" = true ]; then
  for seed in 7201 7202 7203; do
    out="runs/tetra-1.7b-recovery-v2-rl-fixed-zero-seed$seed/rl"
    resume_args=()
    if [ -f "$out/latest_checkpoint.json" ] && [ ! -f "$out/complete.json" ]; then
      latest=$(.venv-rl/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["path"])' "$out/latest_checkpoint.json")
      resume_args=(--resume "$latest")
    fi
    if [ ! -f "$out/complete.json" ]; then
      .venv-rl/bin/python scripts/train_episode_rl.py --experiment E7 --question "Does revised RL improve the gated recovery-v2 SFT endpoint?" --adapter-dir runs/tetra-1.7b-recovery-v2-sft-seed0/adapter --frozen-sft-adapter-dir runs/tetra-1.7b-recovery-v2-sft-seed0/adapter --benchmark-manifest benchmarks/stress-v1/manifest.json --stage5-manifest experiments/tetra17-recovery-v2/evaluation/candidate/stage5/manifest.json --recovery-starts data/stage6-recovery-v1/train-starts.jsonl --training-seeds-file data/stage6-recovery-v1/training-seeds.json --registration-file "$registration" --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --updates 32 --group-size 4 --horizon 20 --gamma .99 --advantage-method fixed_zero --advantage-reward-scale 10 --temperature 1 --learning-rate 1e-6 --kl-beta .05 --training-seed "$seed" --save-every 1 --train-batch-size 4 --pilot-dollar-limit 10 --stage-dollar-limit "$hard_limit" --instance-hourly-usd 2.24208 --max-wall-clock-hours 4 --out-dir "$out" "${resume_args[@]}"
    fi
    .venv-rl/bin/python scripts/eval_tetra17_matched.py --registration "$registration" --label "rl-fixed-zero-$seed" --adapter-dir "$out/adapter"
    sync_artifacts
  done
  .venv-rl/bin/python scripts/analyze_tetra17_rl.py --registration "$registration"
fi
