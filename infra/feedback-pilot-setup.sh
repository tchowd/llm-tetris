#!/usr/bin/env bash
# Run on the assigned new worker only; does not start GPU optimization.
set -euo pipefail
cd /home/ubuntu/llm-tetris
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH=/home/ubuntu/llm-tetris
# Keep unattended package maintenance from restarting this bounded setup service.
sudo tee /etc/needrestart/conf.d/feedback-pilot.conf >/dev/null <<'EOF'
$nrconf{override_rc}->{qr(^feedback-(pilot|retained-setup)\.service$)} = 0;
EOF
sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer
pilot_deadline_calendar=$(python3 -c 'import json; from datetime import datetime; print(datetime.fromisoformat(json.load(open("experiments/stage6-feedback-v1/session/approval.json"))["absolute_deadline_utc"]).strftime("%Y-%m-%d %H:%M:%S UTC"))')
if ! systemctl is-active --quiet feedback-pilot-absolute-deadline.timer; then
  sudo systemd-run --unit=feedback-pilot-absolute-deadline --on-calendar="$pilot_deadline_calendar" /usr/sbin/shutdown -h now
fi
# The worker permits HTTPS egress; the AMI defaults to HTTP Ubuntu mirrors.
sudo python3 - <<'PY'
from pathlib import Path
import re
path = Path('/etc/apt/sources.list.d/ubuntu.sources')
source = path.read_text()
source = re.sub(r'http://[a-z0-9-]+\.ec2\.archive\.ubuntu\.com/ubuntu/', 'https://archive.ubuntu.com/ubuntu/', source)
source = source.replace('http://security.ubuntu.com/ubuntu', 'https://security.ubuntu.com/ubuntu')
path.write_text(source)
PY
sudo apt-get update -qq
sudo apt-get -o DPkg::Lock::Timeout=300 install -y python3.12-venv python3.12-dev
python3 -m venv .venv-rl
.venv-rl/bin/python -m pip install --upgrade pip
.venv-rl/bin/python -m pip install -e '.[dev]' -r requirements-rl.txt
.venv-rl/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv-rl/bin/python scripts/stage6_feedback.py validate
.venv-rl/bin/python infra/feedback_gpu_proof.py validate
.venv-rl/bin/python -c 'import torch; from scripts.train_episode_rl import configure_execution; configure_execution(); assert torch.cuda.is_available(); assert "L40S" in torch.cuda.get_device_name(); a=torch.ones((4,64,1),device="cuda"); b=torch.ones((4,1,300),device="cuda"); c=a@b; torch.cuda.synchronize(); assert torch.equal(c,torch.ones_like(c)); print("CUDA compiler proof passed",torch.__version__)'
.venv-rl/bin/python -m pytest tests/test_feedback.py tests/test_feedback_pilot.py tests/test_episode_rl.py tests/test_episode_proof.py tests/test_episode_runtime.py infra/test_feedback_gpu_proof.py infra/test_feedback_session.py -q
sudo tee /etc/systemd/system/feedback-pilot-monitor.service >/dev/null <<'EOF'
[Unit]
Description=Feedback pilot checkpoint backup and deadline watchdog
[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/llm-tetris
Environment=PYTHONPATH=/home/ubuntu/llm-tetris
ExecStart=/usr/bin/timeout --kill-after=10s 50s /home/ubuntu/llm-tetris/.venv-rl/bin/python infra/feedback_worker.py monitor
EOF
sudo tee /etc/systemd/system/feedback-pilot-monitor.timer >/dev/null <<'EOF'
[Unit]
Description=Feedback pilot minute watchdog
[Timer]
OnBootSec=1min
OnUnitActiveSec=1min
[Install]
WantedBy=timers.target
EOF
sudo tee /etc/needrestart/conf.d/feedback-pilot.conf >/dev/null <<'EOF'
$nrconf{override_rc}->{qr(^feedback-(pilot|retained-setup)\.service$)} = 0;
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now feedback-pilot-monitor.timer
sudo systemctl show feedback-pilot-absolute-deadline.timer -p NextElapseUSecRealtime
echo 'Worker setup verified; GPU workload has not started.'
