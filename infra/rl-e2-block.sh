#!/usr/bin/env bash
# One registered weakened-policy block. $1: prior total AWS spend, conservatively rounded.
set -euo pipefail
cd /home/ubuntu/llm-tetris
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Use only the preserved, verified E0/E1 model revision, never a moving hub main.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
py=.venv-rl/bin/python
run_dir=runs/rl-e2-seed0/rl
weak_dir=runs/rl-e2-weak-sft/rl
prior=${1:?prior AWS spend is required}
registration="$run_dir/registration.json"

phase() {
  "$py" -c 'import sys,time; from pathlib import Path; from tetris.rl import atomic_write_json; atomic_write_json(Path(sys.argv[1]), {"experiment":"E2", "status":"running", "phase":sys.argv[2], "updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})' "$run_dir/block-state.json" "$1"
}

if [[ ${2:-} != --workflow ]]; then
  # Refuse duplicates or accidental restarts over partially finished evidence.
  [[ ! -e "$run_dir/block-state.json" && ! -e "$weak_dir/train_manifest.json" ]]
  "$py" -c 'import json,sys; from pathlib import Path; from transformers.utils.hub import cached_file; from tetris.rl import directory_sha256; r=json.loads(Path(sys.argv[1]).read_text()); assert json.loads(Path("runs/rl-e1-seed0/rl/e1-gate.json").read_text())["status"]=="passed"; assert float(sys.argv[2])+r["budgets"]["pilot_usd"]<=r["budgets"]["stage_usd"]; assert r["budgets"]["block_max_hours_including_sync"]*r["budgets"]["hourly_usd"]<=r["budgets"]["pilot_usd"]; assert directory_sha256(Path("runs/sft-v1/adapter"))==r["frozen_sft_adapter_sha256"]; cached=Path(cached_file(r["base_model"],"config.json",local_files_only=True)); assert cached.parent.name==r["base_model_revision"],cached; assert not Path("runs/rl-e2-seed0/rl/manifest.json").exists()' "$registration" "$prior"
  mkdir -p "$weak_dir"
  finish() {
    result=$?
    trap - EXIT
    set +e
    "$py" -c 'import json,sys,time; from pathlib import Path; from tetris.rl import atomic_write_json; p=Path(sys.argv[1]); m=json.loads(p.read_text()) if p.exists() else {}; code=int(sys.argv[2]); m.update(status={0:"passed",2:"not_passed",124:"timed_out"}.get(code,"failed"),exit_code=code,finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())); atomic_write_json(p,m)' "$run_dir/block-state.json" "$result"
    for run in rl-e2-weak-sft rl-e2-seed0; do
      timeout --kill-after=15s 8m "$py" scripts/sync_run_artifacts.py upload --run-id "$run" --include-adapter --receipt "runs/$run/rl/sync-receipt.json"
      timeout --kill-after=15s 1m "$py" scripts/sync_run_artifacts.py upload --run-id "$run"
    done
    sudo /usr/sbin/shutdown -h now
    exit "$result"
  }
  trap finish EXIT
  phase starting
  timeout --signal=INT --kill-after=120s 280m bash "$0" "$prior" --workflow > "$run_dir/block.log" 2>&1
  exit 0
fi

phase weakened_sft_training
timeout --signal=INT --kill-after=120s 60m "$py" scripts/train_sft.py \
  --data-dirs data/batch1 data/batch2 --out-dir "$weak_dir" \
  --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --backend hf --device cuda --max-train-rows 8192 --max-steps 512 \
  --batch-size 4 --grad-accum 4 --lr 0.0001 --seed 0 \
  --max-eval-rows 256 --gen-eval-rows 64 --eval-steps 256 --save-steps 128 \
  --logging-steps 10 > "$weak_dir/train.log" 2>&1
"$py" -c 'import json; from pathlib import Path; m=json.loads(Path("runs/rl-e2-weak-sft/rl/train_manifest.json").read_text()); assert m["status"]=="passed" and m["total_steps"]==512 and m["num_train_rows"]==8192 and m["seed_applied_before_model_init"]; assert m["base_model_revision"]=="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"'

phase weak_baseline_evaluation
timeout --signal=INT --kill-after=120s 75m "$py" scripts/eval_stress.py \
  --suite development --policies model --policy-label weak \
  --adapter-dir "$weak_dir/adapter" --data-dirs data/batch1 data/batch2 \
  --out-dir "$weak_dir/stress-development" > "$weak_dir/eval.log" 2>&1
"$py" scripts/check_e2_learning.py --registration "$registration" \
  --weak "$weak_dir/stress-development" --out "$run_dir/baseline-gate.json"

# Preserve weak initialization and its evaluated baseline before any RL update.
phase baseline_artifact_sync
timeout --kill-after=15s 5m "$py" scripts/sync_run_artifacts.py upload --run-id rl-e2-weak-sft --include-adapter --receipt "$weak_dir/sync-receipt.json"
timeout --kill-after=15s 2m "$py" scripts/sync_run_artifacts.py upload --run-id rl-e2-seed0

phase dense_rl_training
# Prior training budget allowance includes up to 140 minutes already spent in this block.
rl_prior=$("$py" -c 'import sys; print(float(sys.argv[1])+140/60*1.05)' "$prior")
timeout --signal=INT --kill-after=120s 65m "$py" scripts/train_rl.py \
  --experiment E2 --question 'Can dense GRPO improve the registered weakened policy on real development-state reward without validity collapse?' \
  --initialization-kind weakened --adapter-dir "$weak_dir/adapter" --out-dir "$run_dir" \
  --max-updates 512 --states 256 --group-size 4 --batch-size 4 --grad-accum 4 \
  --learning-rate 0.00001 --kl-beta 0.05 --seed 0 --save-steps 32 \
  --instance-hourly-usd 1.05 --max-wall-clock-hours 1 \
  --pilot-dollar-limit 20 --stage-dollar-limit 100 --prior-stage-spend-usd "$rl_prior" \
  > "$run_dir/train.log" 2>&1
"$py" -c 'import json; from pathlib import Path; m=json.loads(Path("runs/rl-e2-seed0/rl/manifest.json").read_text()); assert m["status"]=="completed" and m["completed_updates"]==512 and m["reference_frozen"]'

phase candidate_evaluation
timeout --signal=INT --kill-after=120s 75m "$py" scripts/eval_stress.py \
  --suite development --policies model --policy-label e2 --adapter-dir "$run_dir/adapter" \
  --data-dirs data/batch1 data/batch2 --out-dir "$run_dir/stress-development" \
  > "$run_dir/eval.log" 2>&1
phase learning_gate
"$py" scripts/check_e2_learning.py --registration "$registration" \
  --weak "$weak_dir/stress-development" --candidate "$run_dir/stress-development" \
  --baseline-gate "$run_dir/baseline-gate.json" --out "$run_dir/e2-gate.json"
