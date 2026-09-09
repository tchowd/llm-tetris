#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/llm-tetris
export PYTHONPATH=/home/ubuntu/llm-tetris PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
registration=experiments/tetra17-recovery-v2/registration.json
approval=experiments/tetra17-recovery-v2/budget-approval.json
hard_limit=$(python3 -c 'import json;print(json.load(open("experiments/tetra17-recovery-v2/budget-approval.json"))["hard_limit_usd"])')
bucket=llm-tetris-artifacts-566629888938-us-east-1
prefix=runs/tetra17-recovery-v2
started_epoch=$(date +%s)
monitor_pid=""

monitor() {
  while true; do
    sleep 900
    now=$(date +%s)
    python3 -c 'import json,sys,time; p="experiments/tetra17-recovery-v2/live-cost.json"; elapsed=int(sys.argv[1])-int(sys.argv[2])+600; rate=float(sys.argv[3]); json.dump({"observed_at":time.time(),"conservative_instance_hours":elapsed/3600,"estimated_compute_usd":elapsed/3600*rate},open(p,"w"),indent=2)' "$now" "$started_epoch" 2.24208
    aws s3 sync experiments/tetra17-recovery-v2 "s3://$bucket/$prefix/experiments" --only-show-errors
    aws s3 sync runs/tetra-1.7b-recovery-v2-sft-seed0 "s3://$bucket/$prefix/runs/sft" --only-show-errors 2>/dev/null || true
  done
}

finish() {
  status=$?
  set +e
  [ -n "$monitor_pid" ] && kill "$monitor_pid" 2>/dev/null
  .venv-rl/bin/python infra/tetra17_recovery_ops.py verify-worker-budget --approval "$approval" 2>/dev/null
  aws s3 sync experiments/tetra17-recovery-v2 "s3://$bucket/$prefix/experiments" --only-show-errors
  aws s3 sync data/tetra17-recovery-v2-staging "s3://$bucket/$prefix/data/staging" --only-show-errors
  aws s3 sync data/tetra17-recovery-v2-on-policy "s3://$bucket/$prefix/data/on-policy" --only-show-errors
  aws s3 sync data/tetra17-recovery-v2 "s3://$bucket/$prefix/data/final" --only-show-errors
  aws s3 sync runs/tetra-1.7b-recovery-v2-sft-seed0 "s3://$bucket/$prefix/runs/sft" --only-show-errors
  for d in runs/tetra-1.7b-recovery-v2-rl-fixed-zero-seed*/rl; do
    [ -d "$d" ] && aws s3 sync "$d" "s3://$bucket/$prefix/$d" --only-show-errors
  done
  aws s3 cp "$approval" "s3://$bucket/$prefix/receipts/budget-approval.json" --only-show-errors
  if [ "$status" -eq 0 ]; then
    sudo shutdown -h now
  else
    python3 -c 'import json,time; json.dump({"status":"failed_waiting_for_recovery","recorded_at":time.time()},open("experiments/tetra17-recovery-v2/worker-state.json","w"),indent=2)'
    aws s3 cp experiments/tetra17-recovery-v2/worker-state.json "s3://$bucket/$prefix/experiments/worker-state.json" --only-show-errors
  fi
  exit "$status"
}
trap finish EXIT

deadline_calendar=$(python3 -c 'import json;from datetime import datetime; value=json.load(open("experiments/tetra17-recovery-v2/budget-approval.json"))["absolute_deadline_utc"]; print(datetime.fromisoformat(value.replace("Z","+00:00")).strftime("%Y-%m-%d %H:%M:%S UTC"))')
sudo systemd-run --unit=tetra17-hard-deadline --on-calendar="$deadline_calendar" /usr/sbin/shutdown -h now
sudo python3 - <<'PY'
from pathlib import Path
import re
path = Path('/etc/apt/sources.list.d/ubuntu.sources')
if path.exists():
    source = path.read_text().replace('http://security.ubuntu.com/ubuntu', 'https://security.ubuntu.com/ubuntu')
    source = re.sub(r'http://[a-z0-9-]+\.ec2\.archive\.ubuntu\.com/ubuntu/', 'https://archive.ubuntu.com/ubuntu/', source)
    path.write_text(source)
PY
sudo apt-get update -qq
sudo apt-get -o DPkg::Lock::Timeout=300 install -y python3.12-venv python3.12-dev
python3 -m venv .venv-sft
.venv-sft/bin/python -m pip install --upgrade pip
.venv-sft/bin/python -m pip install -e . -r requirements-train-unsloth.txt
python3 -m venv .venv-rl
.venv-rl/bin/python -m pip install --upgrade pip
.venv-rl/bin/python -m pip install -e . -r requirements-rl.txt
.venv-rl/bin/python -m pip install pytest==9.1.1
.venv-sft/bin/python -m pip freeze > experiments/tetra17-recovery-v2/environment-sft.txt
.venv-rl/bin/python -m pip freeze > experiments/tetra17-recovery-v2/environment-rl.txt
.venv-rl/bin/python -c 'import torch; assert torch.cuda.is_available(); assert "L40S" in torch.cuda.get_device_name(); print(torch.__version__, torch.cuda.get_device_name())' > experiments/tetra17-recovery-v2/cuda-proof.txt
.venv-rl/bin/python infra/tetra17_recovery_ops.py verify-worker-budget --approval "$approval"
.venv-rl/bin/python -m pytest -q tests/test_curriculum.py tests/test_on_policy_curriculum.py tests/test_tetra17_data_audit.py tests/test_tetra17_curriculum_builder.py tests/test_tetra17_sft_analysis.py tests/test_feedback.py
.venv-sft/bin/python scripts/train_sft.py --backend unsloth --base-model Qwen/Qwen3-1.7B --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --data-dirs data/batch1 --out-dir /tmp/tetra17-recovery-v2-smoke --max-train-rows 32 --max-eval-rows 32 --max-steps 2 --batch-size 4 --grad-accum 1 --eval-steps 1 --save-steps 1
monitor &
monitor_pid=$!

if aws s3api head-object --bucket "$bucket" --key "$prefix/data/staging/manifest.json" >/dev/null 2>&1; then
  aws s3 sync "s3://$bucket/$prefix/data/staging" data/tetra17-recovery-v2-staging --only-show-errors
else
  .venv-rl/bin/python scripts/build_tetra17_curriculum.py --registration "$registration" --phase teacher
fi
if aws s3api head-object --bucket "$bucket" --key "$prefix/data/on-policy/manifest.json" >/dev/null 2>&1; then
  aws s3 sync "s3://$bucket/$prefix/data/on-policy" data/tetra17-recovery-v2-on-policy --only-show-errors
else
  .venv-rl/bin/python scripts/collect_tetra17_on_policy.py --registration "$registration"
fi
if aws s3api head-object --bucket "$bucket" --key "$prefix/data/final/manifest.json" >/dev/null 2>&1; then
  aws s3 sync "s3://$bucket/$prefix/data/final" data/tetra17-recovery-v2 --only-show-errors
else
  .venv-rl/bin/python scripts/build_tetra17_curriculum.py --registration "$registration" --phase assemble
fi
.venv-rl/bin/python scripts/audit_tetra17_data.py --original data/tetra17-recovery-v2 --teacher-sample-size 4000 --out experiments/tetra17-recovery-v2/audit/final-dataset.json
.venv-rl/bin/python infra/tetra17_recovery_ops.py verify-worker-budget --approval "$approval"

.venv-sft/bin/python scripts/train_sft.py --backend unsloth --base-model Qwen/Qwen3-1.7B --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --data-dirs data/tetra17-recovery-v2 --out-dir runs/tetra-1.7b-recovery-v2-sft-seed0 --epochs 1 --lr 1e-4 --batch-size 16 --grad-accum 4 --seed 0 --eval-steps 250 --save-steps 250
.venv-rl/bin/python infra/tetra17_recovery_ops.py verify-worker-budget --approval "$approval"

for label in control candidate; do
  if [ "$label" = control ]; then adapter=runs/sft-v1/adapter; else adapter=runs/tetra-1.7b-recovery-v2-sft-seed0/adapter; fi
  .venv-rl/bin/python scripts/eval_tetra17_matched.py --registration "$registration" --label "$label"
  .venv-rl/bin/python scripts/eval_open_loop.py --base-model Qwen/Qwen3-1.7B --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --data-dirs data/batch1 data/batch2 --adapter-dir "$adapter" --max-rows 10000 --seed 1702 --out "experiments/tetra17-recovery-v2/evaluation/$label/open-loop.json"
  .venv-rl/bin/python scripts/eval_closed_loop.py --policies model --modes strict --model-label "$label" --base-model Qwen/Qwen3-1.7B --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --adapter-dir "$adapter" --data-dirs data/batch1 data/batch2 --num-seeds 100 --cap 500 --out-dir "experiments/tetra17-recovery-v2/evaluation/$label/stage5"
  .venv-rl/bin/python scripts/eval_stress.py --suite development --policies model --policy-label "$label" --base-model Qwen/Qwen3-1.7B --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --adapter-dir "$adapter" --data-dirs data/batch1 data/batch2 --out-dir "experiments/tetra17-recovery-v2/evaluation/$label/stress-development"
done
.venv-rl/bin/python scripts/analyze_tetra17_sft.py --registration "$registration"

if [ "$(.venv-rl/bin/python -c 'import json;print(str(json.load(open("experiments/tetra17-recovery-v2/sft-gate.json"))["conditional_rl_authorized_by_gate"]).lower())')" = true ]; then
  for seed in 7201 7202 7203; do
    out="runs/tetra-1.7b-recovery-v2-rl-fixed-zero-seed$seed/rl"
    .venv-rl/bin/python scripts/train_episode_rl.py --experiment E7 --question "Does revised RL improve the gated recovery-v2 SFT endpoint?" --adapter-dir runs/tetra-1.7b-recovery-v2-sft-seed0/adapter --frozen-sft-adapter-dir runs/tetra-1.7b-recovery-v2-sft-seed0/adapter --benchmark-manifest benchmarks/stress-v1/manifest.json --stage5-manifest experiments/tetra17-recovery-v2/evaluation/candidate/stage5/manifest.json --recovery-starts data/stage6-recovery-v1/train-starts.jsonl --training-seeds-file data/stage6-recovery-v1/training-seeds.json --registration-file "$registration" --base-model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --updates 32 --group-size 4 --horizon 20 --gamma .99 --advantage-method fixed_zero --advantage-reward-scale 10 --temperature 1 --learning-rate 1e-6 --kl-beta .05 --training-seed "$seed" --save-every 1 --train-batch-size 4 --pilot-dollar-limit 10 --stage-dollar-limit "$hard_limit" --instance-hourly-usd 2.24208 --max-wall-clock-hours 4 --out-dir "$out"
    .venv-rl/bin/python scripts/eval_tetra17_matched.py --registration "$registration" --label "rl-fixed-zero-$seed" --adapter-dir "$out/adapter"
  done
  .venv-rl/bin/python scripts/analyze_tetra17_rl.py --registration "$registration"
fi
