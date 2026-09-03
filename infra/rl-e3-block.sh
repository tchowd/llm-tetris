#!/usr/bin/env bash
# Fixed three-way KL comparison, only after the E2 learning gate passes.
set -euo pipefail
cd /home/ubuntu/llm-tetris
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
py=.venv-rl/bin/python
prior=${1:?prior AWS spend is required}
root=runs/stage6-e3/rl
registration="$root/registration.json"
runs=(rl-e3-kl001-seed0 rl-e3-kl005-seed0 rl-e3-kl010-seed0)
betas=(0.01 0.05 0.1)
labels=(kl001 kl005 kl010)
phase() {
  "$py" -c 'import sys,time; from pathlib import Path; from tetris.rl import atomic_write_json; atomic_write_json(Path(sys.argv[1]),{"experiment":"E3","status":"running","phase":sys.argv[2],"updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})' "$root/block-state.json" "$1"
}
if [[ ${2:-} != --workflow ]]; then
  [[ ! -e "$root/block-state.json" ]]
  for run in "${runs[@]}"; do [[ ! -e "runs/$run/rl/manifest.json" ]]; done
  "$py" -c 'import json,sys; from pathlib import Path; from transformers.utils.hub import cached_file; from tetris.rl import directory_sha256; r=json.loads(Path(sys.argv[1]).read_text()); g=json.loads(Path(r["entry_gate"]).read_text()); assert g["status"]=="passed" and all(g["checks"].values()); assert float(sys.argv[2])+20<=100; assert directory_sha256(Path("runs/sft-v1/adapter"))==r["frozen_sft_adapter_sha256"]; assert Path(cached_file(r["base_model"],"config.json",local_files_only=True)).parent.name==r["base_model_revision"]' "$registration" "$prior"
  finish() {
    result=$?
    trap - EXIT
    set +e
    "$py" -c 'import json,sys,time; from pathlib import Path; from tetris.rl import atomic_write_json; p=Path(sys.argv[1]); m=json.loads(p.read_text()) if p.exists() else {}; code=int(sys.argv[2]); m.update(status={0:"passed",2:"not_passed",124:"timed_out"}.get(code,"failed"),exit_code=code,finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())); atomic_write_json(p,m)' "$root/block-state.json" "$result"
    for run in "${runs[@]}" stage6-e3; do
      [[ -d "runs/$run" ]] || continue
      timeout --kill-after=10s 4m "$py" scripts/sync_run_artifacts.py upload --run-id "$run" --include-adapter --receipt "runs/$run/rl/sync-receipt.json"
      timeout --kill-after=10s 30s "$py" scripts/sync_run_artifacts.py upload --run-id "$run"
    done
    sudo /usr/sbin/shutdown -h now
    exit "$result"
  }
  trap finish EXIT
  phase starting
  timeout --signal=INT --kill-after=120s 340m bash "$0" "$prior" --workflow > "$root/block.log" 2>&1
  exit 0
fi
for index in 0 1 2; do
  run=${runs[$index]}; beta=${betas[$index]}; label=${labels[$index]}
  out="runs/$run/rl"
  mkdir -p "$out"
  phase "$label/dense_training"
  # Conservative prior includes up to the entire block allowance, never ignores earlier candidates.
  candidate_prior=$("$py" -c 'import sys; print(float(sys.argv[1])+6*1.05)' "$prior")
  timeout --signal=INT --kill-after=120s 65m "$py" scripts/train_rl.py \
    --experiment E3 --question 'Which registered KL preserves strict format and legal actions for the frozen SFT initialization under a fixed dense reward?' \
    --initialization-kind sft --adapter-dir runs/sft-v1/adapter --out-dir "$out" \
    --max-updates 256 --states 256 --group-size 4 --batch-size 4 --grad-accum 4 \
    --learning-rate 0.000001 --kl-beta "$beta" --seed 0 --save-steps 32 \
    --instance-hourly-usd 1.05 --max-wall-clock-hours 1 --pilot-dollar-limit 20 \
    --stage-dollar-limit 100 --prior-stage-spend-usd "$candidate_prior" > "$out/train.log" 2>&1
  "$py" -c 'import json,sys; from pathlib import Path; m=json.loads((Path(sys.argv[1])/"manifest.json").read_text()); assert m["status"]=="completed" and m["completed_updates"]==256 and m["reference_frozen"]' "$out"
  phase "$label/development_evaluation"
  timeout --signal=INT --kill-after=120s 75m "$py" scripts/eval_stress.py \
    --suite development --policies model --policy-label "$label" --adapter-dir "$out/adapter" \
    --data-dirs data/batch1 data/batch2 --out-dir "$out/stress-development" > "$out/eval.log" 2>&1
  timeout --kill-after=10s 4m "$py" scripts/sync_run_artifacts.py upload --run-id "$run" --include-adapter --receipt "$out/sync-receipt.json"
done
phase kl_selection
"$py" scripts/select_e3_kl.py --registration "$registration" --out "$root/selection.json"
