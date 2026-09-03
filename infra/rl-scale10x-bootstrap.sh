#!/usr/bin/env bash
# User data runs immediately: the hard limit starts at launch, not training.
set -euo pipefail
systemd-run --unit=llm-tetris-scale10x-budget-stop --on-active=12h /usr/sbin/shutdown -h now
