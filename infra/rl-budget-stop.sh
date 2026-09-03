#!/usr/bin/env bash
# EC2 user data: a fail-safe for the first experiment block, not a substitute
# for per-run budgets. Re-arm explicitly after each later instance start.
set -euo pipefail
systemd-run --unit=llm-tetris-budget-stop --on-active=6h /usr/sbin/shutdown -h now
