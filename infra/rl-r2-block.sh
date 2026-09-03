#!/usr/bin/env bash
# R2: independently registered GPU trajectory proof, never an E3 promotion.
set -euo pipefail
cd /home/ubuntu/llm-tetris
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
py=.venv-rl/bin/python
registration=${1:-runs/rl-r2-episode-proof-seed0/rl/registration.json}
if [[ "$registration" == --workflow ]]; then registration="$2"; fi
run=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$registration")
control=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["control_run_id"])' "$registration")
out="runs/$run/rl"
phase() {
  "$py" -c 'import sys,time; from pathlib import Path; from tetris.rl import atomic_write_json; atomic_write_json(Path(sys.argv[1]),{"experiment":"R2","status":"running","phase":sys.argv[2],"updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})' "$out/block-state.json" "$1"
}
if [[ ${1:-} != --workflow ]]; then
  [[ ! -e "$out/block-state.json" && ! -e "$out/manifest.json" && ! -e "runs/$control/rl/manifest.json" ]]
  mkdir -p "$out"
  finish() {
    result=$?
    trap - EXIT
    set +e
    "$py" -c 'import json,sys,time; from pathlib import Path; from tetris.rl import atomic_write_json; p=Path(sys.argv[1]); m=json.loads(p.read_text()) if p.exists() else {}; code=int(sys.argv[2]); m.update(status={0:"passed",2:"not_passed",124:"timed_out"}.get(code,"failed"),exit_code=code,finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())); atomic_write_json(p,m)' "$out/block-state.json" "$result"
    for target in "$run" "$control"; do
      timeout --kill-after=10s 3m "$py" scripts/sync_run_artifacts.py upload --run-id "$target" --include-adapter --receipt "runs/$target/rl/sync-receipt.json"
      timeout --kill-after=10s 60s "$py" scripts/sync_run_artifacts.py upload --run-id "$target"
    done
    sudo /usr/sbin/shutdown -h now
    exit "$result"
  }
  trap finish EXIT
  "$py" scripts/check_episode_proof.py --registration "$registration" --preflight
  "$py" -c 'import json,math,sys; from pathlib import Path; m=json.loads(Path("runs/stage6-aws/rl/manifest.json").read_text()); r=json.loads(Path(sys.argv[1]).read_text()); spent=m["estimated_accrued_usd"]; assert math.isfinite(spent) and 0<=spent<=r["prior_stage_spend_usd"] and spent+1.05+20<=100' "$registration"
  phase starting
  timeout --signal=INT --kill-after=60s 50m bash "$0" --workflow "$registration" > "$out/block.log" 2>&1
  exit 0
fi
prior=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["prior_stage_spend_usd"])' "$registration")
common=(--experiment E5 --question 'Do exact replay, delayed credit, GPU token probabilities and resume agree?'
  --registration-file "$registration"
  --adapter-dir runs/sft-v1/adapter --frozen-sft-adapter-dir runs/sft-v1/adapter
  --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
  --benchmark-manifest benchmarks/stress-v1/manifest.json --stage5-manifest runs/sft-v1/closed_loop/manifest.json
  --recovery-starts data/stage6-recovery-v1/train-starts.jsonl --training-seeds-file data/stage6-recovery-v1/training-seeds.json
  --updates 4 --group-size 4 --horizon 20 --gamma .99 --temperature 1 --learning-rate 1e-6 --kl-beta .05
  --score-scale 100 --death-penalty 2 --illegal-penalty 10 --training-seed 6102 --save-every 1 --train-batch-size 4
  --pilot-dollar-limit 20 --stage-dollar-limit 100 --prior-stage-spend-usd "$prior" --instance-hourly-usd 1.05 --max-wall-clock-hours .75)
phase first_two_updates
"$py" scripts/train_episode_rl.py "${common[@]}" --out-dir "$out" --pause-after-update 2 > "$out/train-first.log" 2>&1
"$py" -c 'import json,sys; from pathlib import Path; from tetris.rl import atomic_write_json; p=Path(sys.argv[1]); m=json.loads((p/"manifest.json").read_text()); assert m["status"]=="paused" and m["completed_updates"]==2; atomic_write_json(p/"paused-manifest.json",m)' "$out"
phase resume_last_two_updates
"$py" scripts/train_episode_rl.py "${common[@]}" --out-dir "$out" --resume "$out/checkpoint-2" > "$out/train-resume.log" 2>&1
phase uninterrupted_control
mkdir -p "runs/$control/rl"
"$py" scripts/train_episode_rl.py "${common[@]}" --out-dir "runs/$control/rl" > "runs/$control/rl/train.log" 2>&1
phase independent_gpu_proof
"$py" scripts/check_episode_proof.py --registration "$registration" --out "$out/r2-gate.json" > "$out/proof.log" 2>&1
