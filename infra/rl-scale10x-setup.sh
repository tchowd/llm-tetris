#!/usr/bin/env bash
# Deployment only: this does not start the research service or reset a deadline.
set -euo pipefail
cd /home/ubuntu/llm-tetris
test -s runs/stage6-scale10x-v1/rl/compute-ledger.json
test ! -e runs/stage6-scale10x-v1/rl/block-state.json
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# Attach an absolute deadline before dependency downloads or GPU initialization.
scale_deadline_epoch=$(python3 -c 'import json; print(int(json.load(open("runs/stage6-scale10x-v1/rl/compute-ledger.json"))["deadline_epoch"]))')
test "$(date +%s)" -lt "$scale_deadline_epoch"
scale_deadline_calendar=$(date -u -d "@$scale_deadline_epoch" '+%Y-%m-%d %H:%M:%S UTC')
if ! sudo systemctl cat llm-tetris-scale10x-deadline.timer >/dev/null 2>&1; then
  sudo systemd-run --unit=llm-tetris-scale10x-deadline --on-calendar="$scale_deadline_calendar" /usr/sbin/shutdown -h now
fi

# This AMI's system Python lacks both venv support and the C headers used by
# Triton's CUDA helper compiler. Verify them before any research attempt.
sudo apt-get update -qq
sudo apt-get install -y python3.12-venv python3.12-dev
python3 -m venv .venv-rl
.venv-rl/bin/python -m pip install --upgrade pip
.venv-rl/bin/python -m pip install -e '.[dev]' -r requirements-rl.txt
.venv-rl/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv-rl/bin/python scripts/stage6_scale10x.py preflight
.venv-rl/bin/python -c 'import torch; assert torch.cuda.is_available(); name=torch.cuda.get_device_name(); assert "L40S" in name; print(name, torch.cuda.get_device_properties(0).total_memory, torch.__version__)'
.venv-rl/bin/python -c 'import torch; from scripts.train_episode_rl import configure_execution; configure_execution(); a=torch.ones((4,64,1),device="cuda"); b=torch.ones((4,1,300),device="cuda"); c=a@b; torch.cuda.synchronize(); assert torch.equal(c,torch.ones_like(c)); print("Triton CUDA helper compilation and exact outer-product probe passed")'
.venv-rl/bin/python -m pytest tests/test_scale10x.py tests/test_episode_proof.py tests/test_episode_runtime.py tests/test_episode_rl.py -q
sudo install -m 0644 infra/rl-scale10x.service /etc/systemd/system/llm-tetris-scale10x.service
sudo install -m 0644 infra/rl-scale10x.cron /etc/cron.d/llm-tetris-scale10x
sudo touch /var/log/llm-tetris-scale10x-sync.log
sudo chown ubuntu:ubuntu /var/log/llm-tetris-scale10x-sync.log
sudo systemctl daemon-reload
sudo systemctl enable --now cron
sudo systemctl show llm-tetris-scale10x-deadline.timer -p NextElapseUSecRealtime
echo 'Setup verified. Start llm-tetris-scale10x.service only after reviewing the timer, ledger and preflight evidence.'
