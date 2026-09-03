#!/usr/bin/env bash
# Independent episode-return pilot; strict proof or explicit runtime-only amendment.
set -euo pipefail
cd /home/ubuntu/llm-tetris
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
py=.venv-rl/bin/python
run=rl-r3-episode-seed0
out="runs/$run/rl"
registration="$out/registration.json"
protocol=runs/stage6-recovery-v1/rl/registration.json
setting() { "$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$registration" "$1"; }
phase() {
  "$py" -c 'import sys,time; from pathlib import Path; from tetris.rl import atomic_write_json; atomic_write_json(Path(sys.argv[1]),{"experiment":"R3","status":"running","phase":sys.argv[2],"updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})' "$out/block-state.json" "$1"
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
    sudo /usr/sbin/shutdown -h now
    exit "$result"
  }
  trap finish EXIT
  "$py" scripts/check_episode_pilot.py --registration "$registration" --preflight
  "$py" -c 'import json,math; from pathlib import Path; m=json.loads(Path("runs/stage6-aws/rl/manifest.json").read_text()); r=json.loads(Path("runs/rl-r3-episode-seed0/rl/registration.json").read_text()); spent=m["estimated_accrued_usd"]; assert math.isfinite(spent) and 0<=spent<=r["prior_stage_spend_usd"] and spent+r["block_hours"]*r["hourly_usd"]+20<=100'
  phase starting
  timeout --signal=INT --kill-after=60s "$(setting workflow_minutes)m" bash "$0" --workflow > "$out/block.log" 2>&1
  exit 0
fi
prior=$("$py" -c 'import json; print(json.load(open("runs/rl-r3-episode-seed0/rl/registration.json"))["prior_stage_spend_usd"])')
question=$("$py" -c 'import json; print(json.load(open("runs/rl-r3-episode-seed0/rl/registration.json"))["question"])')
training_hours=$("$py" -c 'import json; print(json.load(open("runs/rl-r3-episode-seed0/rl/registration.json"))["recipe"]["max_training_hours"])')
phase episode_training
timeout --signal=INT --kill-after=60s "$(setting training_timeout_minutes)m" "$py" scripts/train_episode_rl.py \
  --experiment E6 --question "$question" --registration-file "$registration" \
  --adapter-dir runs/sft-v1/adapter --frozen-sft-adapter-dir runs/sft-v1/adapter \
  --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --benchmark-manifest benchmarks/stress-v1/manifest.json --stage5-manifest runs/sft-v1/closed_loop/manifest.json \
  --recovery-starts data/stage6-recovery-v1/train-starts.jsonl --training-seeds-file data/stage6-recovery-v1/training-seeds.json \
  --updates 32 --group-size 4 --horizon 128 --gamma .99 --temperature 1 --learning-rate 1e-6 --kl-beta .05 \
  --score-scale 100 --death-penalty 2 --illegal-penalty 10 --training-seed 6103 --save-every 4 --train-batch-size 4 \
  --pilot-dollar-limit 20 --stage-dollar-limit 100 --prior-stage-spend-usd "$prior" --instance-hourly-usd 1.05 \
  --max-wall-clock-hours "$training_hours" --out-dir "$out" > "$out/train.log" 2>&1
"$py" -c 'import json; m=json.load(open("runs/rl-r3-episode-seed0/rl/manifest.json")); assert m["status"]=="completed" and m["completed_updates"]==32'
timeout --kill-after=10s 4m "$py" scripts/sync_run_artifacts.py upload --run-id "$run" --include-adapter --receipt "$out/training-sync-receipt.json"
phase development_evaluation
timeout --signal=INT --kill-after=60s 75m "$py" scripts/eval_stress.py --suite development --policies model \
  --policy-label r3 --adapter-dir "$out/adapter" --data-dirs data/batch1 data/batch2 \
  --out-dir "$out/stress-development" > "$out/eval.log" 2>&1
phase stage5_non_inferiority
timeout --signal=INT --kill-after=60s 60m "$py" scripts/eval_closed_loop.py --policies model --modes strict \
  --model-label r3 --adapter-dir "$out/adapter" --data-dirs data/batch1 data/batch2 \
  --gen-batch-size 64 --teacher-workers 3 --device cuda --out-dir "$out/stage5" > "$out/stage5.log" 2>&1
phase fresh_recovery
timeout --signal=INT --kill-after=60s 20m "$py" scripts/eval_recovery_only.py --registration "$protocol" \
  --adapter-dir "$out/adapter" --label r3 --out-dir "$out/fresh-recovery" > "$out/fresh.log" 2>&1
phase paired_analysis
"$py" scripts/check_episode_pilot.py --registration "$registration" --out "$out/r3-gate.json"
