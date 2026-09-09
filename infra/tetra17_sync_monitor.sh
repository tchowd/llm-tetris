#!/usr/bin/env bash
# Failure-tolerant artifact and cost monitor for the retained recovery-v2 worker.
# This is deliberately independent of the training systemd unit: observing or
# restarting this monitor must never restart the model process.
set -u

cd /home/ubuntu/llm-tetris || exit 1
bucket=llm-tetris-artifacts-566629888938-us-east-1
prefix=runs/tetra17-recovery-v2
service=tetra17-recovery-resume-v3.service
prior_hours=11.5
rate=2.24208
service_started_epoch=1788991016

sync_once() {
  now=$(date +%s)
  python3 -c 'import json,sys,time; prior=float(sys.argv[3]); elapsed=int(sys.argv[1])-int(sys.argv[2])+600; rate=float(sys.argv[4]); hours=prior+elapsed/3600; json.dump({"observed_at":time.time(),"prior_instance_hours_conservative":prior,"current_worker_hours_conservative":elapsed/3600,"cumulative_instance_hours_conservative":hours,"estimated_compute_usd":hours*rate},open("experiments/tetra17-recovery-v2/live-cost.json","w"),indent=2)' "$now" "$service_started_epoch" "$prior_hours" "$rate" || true
  aws s3 sync experiments/tetra17-recovery-v2 "s3://$bucket/$prefix/experiments" --only-show-errors || true
  aws s3 sync runs/tetra-1.7b-recovery-v2-sft-seed0 "s3://$bucket/$prefix/runs/sft" --only-show-errors || true
  for directory in runs/tetra-1.7b-recovery-v2-rl-fixed-zero-seed*/rl; do
    [ -d "$directory" ] && aws s3 sync "$directory" "s3://$bucket/$prefix/$directory" --only-show-errors || true
  done
}

while systemctl is-active --quiet "$service"; do
  sync_once
  sleep 600
done
sync_once
