#!/usr/bin/env bash
# Deployment only. Does not start optimization or change the absolute deadline.
set -euo pipefail
cd /home/ubuntu/llm-tetris
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
proof_deadline_epoch=$(python3 -c 'import json; print(int(json.load(open("experiments/stage6-feedback-v1/gpu-proof-v1/compute-ledger.json"))["deadline_epoch"]))')
test "$(date +%s)" -lt "$proof_deadline_epoch"
proof_deadline_calendar=$(date -u -d "@$proof_deadline_epoch" '+%Y-%m-%d %H:%M:%S UTC')
sudo systemd-run --unit=feedback-proof-absolute-deadline --on-calendar="$proof_deadline_calendar" /usr/sbin/shutdown -h now
sudo apt-get update -qq
sudo apt-get install -y python3.12-venv python3.12-dev
sudo install -m 0644 infra/feedback-gpu-needrestart.conf /etc/needrestart/conf.d/feedback-gpu-proof.conf
python3 -m venv .venv-rl
.venv-rl/bin/python -m pip install --upgrade pip
.venv-rl/bin/python -m pip install -e '.[dev]' -r requirements-rl.txt
.venv-rl/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv-rl/bin/python infra/feedback_gpu_proof.py validate
.venv-rl/bin/python -c 'import torch; from scripts.train_episode_rl import configure_execution; configure_execution(); assert torch.cuda.is_available(); assert "L40S" in torch.cuda.get_device_name(); a=torch.ones((4,64,1),device="cuda"); b=torch.ones((4,1,300),device="cuda"); c=a@b; torch.cuda.synchronize(); assert torch.equal(c,torch.ones_like(c)); print("L40S CUDA compiler probe passed",torch.__version__)'
.venv-rl/bin/python -m pytest infra/test_feedback_gpu_proof.py tests/test_feedback.py tests/test_feedback_pilot.py tests/test_episode_rl.py tests/test_episode_proof.py tests/test_episode_runtime.py -q
sudo systemctl show feedback-proof-absolute-deadline.timer -p NextElapseUSecRealtime
echo 'Setup passed. Proof service has not been started.'
