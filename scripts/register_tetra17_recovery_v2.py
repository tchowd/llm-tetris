#!/usr/bin/env python3
"""Freeze the Tetra-1.7B recovery-v2 experiment before paid execution."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

from tetris.rl import atomic_write_json, directory_sha256, file_sha256


ROOT = Path("experiments/tetra17-recovery-v2")
REGISTRATION = ROOT / "registration.json"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
FROZEN_ADAPTER = Path("runs/sft-v1/adapter")
FROZEN_HASH = "7d753d616d3b9174f489e103f37193fccda28c46b1c7ac91e2f4e87efde01171"


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    if REGISTRATION.exists():
        raise SystemExit(f"refusing to overwrite {REGISTRATION}")
    if directory_sha256(FROZEN_ADAPTER) != FROZEN_HASH:
        raise ValueError("frozen Tetra-1.7B adapter hash changed")
    audit = ROOT / "audit/data-and-failures.json"
    recovery = Path("experiments/stage6-feedback-v1/recovery-development.jsonl")
    feedback_registration = json.loads(Path("experiments/stage6-feedback-v1/registration.json").read_text())
    source_files = [
        Path("scripts/audit_tetra17_data.py"), Path("scripts/build_tetra17_curriculum.py"),
        Path("scripts/collect_tetra17_on_policy.py"), Path("scripts/train_sft.py"),
        Path("scripts/eval_open_loop.py"), Path("scripts/eval_closed_loop.py"), Path("scripts/eval_stress.py"),
        Path("scripts/eval_tetra17_matched.py"), Path("scripts/analyze_tetra17_sft.py"), Path("scripts/analyze_tetra17_rl.py"),
        Path("scripts/train_episode_rl.py"), Path("requirements-train-unsloth.txt"), Path("requirements-rl.txt"),
        Path("infra/tetra17_recovery_bootstrap.sh"), Path("infra/tetra17_recovery_ops.py"),
        Path("tetris/curriculum.py"), Path("tetris/engine.py"), Path("tetris/teacher.py"),
    ]
    registration = {
        "experiment": "tetra17-recovery-v2",
        "status": "registered_awaiting_budget_approval",
        "question": "Does a balanced progressive and one-round on-policy SFT curriculum improve Tetra-1.7B recovery while preserving ordinary play?",
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha(),
        "base_model": "Qwen/Qwen3-1.7B",
        "base_model_revision": BASE_REVISION,
        "frozen_control": {"friendly_name": "Tetra-1.7B", "adapter": str(FROZEN_ADAPTER), "adapter_sha256": FROZEN_HASH},
        "candidate": {
            "run_id": "tetra-1.7b-recovery-v2-sft-seed0", "output_dir": "runs/tetra-1.7b-recovery-v2-sft-seed0",
            "initialization": "fresh base model plus newly initialized LoRA; never continue runs/sft-v1/adapter",
            "eligible_checkpoint": "final endpoint after exactly one epoch", "training_seed": 0,
            "recipe": {"epochs": 1.0, "learning_rate": 1e-4, "batch_size": 16, "gradient_accumulation": 4,
                       "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "backend": "unsloth", "load_in_4bit": False},
        },
        "dataset": {
            "staging_dir": "data/tetra17-recovery-v2-staging", "on_policy_dir": "data/tetra17-recovery-v2-on-policy", "final_dir": "data/tetra17-recovery-v2",
            "ordinary_sources": ["data/batch1", "data/batch2"], "ordinary_rows": 96000, "ordinary_max_rows_per_game": 64,
            "progressive_rows_by_band": {"moderate": 16000, "hard": 16000, "critical": 16000},
            "safety_rows_by_tag": {"left_boundary": 6000, "right_boundary": 6000, "rotation_2_or_3": 6000, "top_risk": 6000},
            "on_policy_rows": 72000, "on_policy_starts": 1536, "on_policy_cap": 200, "on_policy_batch_size": 32,
            "failure_context": 32, "high_regret_threshold": 25.0, "clear_margin_threshold": 5.0,
            "progressive_eval_rows_by_band": {"moderate": 1024, "hard": 1024, "critical": 2048},
            "generator_seed": 1702001, "training_seed_start": 30000000, "training_seed_count": 50000,
            "eval_seed_start": 31000000, "eval_seed_count": 5000, "exploration_cap": 200, "exploration_random_probability": 0.4,
            "final_train_rows": 240000, "final_eval_rows": 4096,
            "exclusions": "No development, confirmation, final-test, or inspected failure state is a source. On-policy collection uses only 30,000,000-series training seeds.",
        },
        "evaluation": {
            "open_loop": {"source": "original Stage 3 eval split", "rows": 10000, "seed": 1702},
            "stage5": {"seeds": list(range(10000000, 10000100)), "cap": 500, "mode": "strict", "greedy": True},
            "ordinary": {"seeds": feedback_registration["evaluation"]["ordinary_seeds"], "cap": 1000},
            "recovery": {"path": str(recovery), "sha256": file_sha256(recovery), "seeds": feedback_registration["evaluation"]["recovery_seeds"], "cap": 200},
            "stress_development": {"manifest": "benchmarks/stress-v1/manifest.json", "manifest_sha256": file_sha256(Path("benchmarks/stress-v1/manifest.json"))},
            "matched_control_required": True, "raw_actions_and_per_game_records_required": True,
            "confirmation_seed_ranges_half_open": feedback_registration["evaluation"]["sealed_confirmation_seed_ranges_half_open"],
            "confirmation_access": False, "final_test_access": False,
        },
        "sft_gate": {
            "recovery_min_cap_reached": 73,
            "recovery_min_paired_improvement": 0.10,
            "recovery_paired_bootstrap_ci95_lower_gt": 0.0,
            "ordinary_required_cap_reached": 20, "ordinary_max_illegal": 0, "ordinary_max_topout": 0,
            "ordinary_min_lines_and_score_ratio": 0.99,
            "stage5_required_cap_reached": 100, "stage5_max_deaths": 0, "stage5_min_lines_ratio": 0.99,
            "open_loop_parse_min": 1.0, "open_loop_legality_min": 0.999, "open_loop_exact_max_drop": 0.02,
        },
        "conditional_rl": {
            "enabled_only_if_all_sft_gates_pass": True, "advantage_method": "fixed_zero", "updates": 32,
            "training_seeds": [7201, 7202, 7203], "initial_adapter": "candidate final endpoint",
            "reference_policy": "frozen copy of candidate final endpoint", "evaluation": "same matched development cohorts",
            "decision": "advance only if mean recovery improves, paired 95% interval excludes harm, and ordinary gates remain passed",
        },
        "audit": {"path": str(audit), "sha256": file_sha256(audit)},
        "source_sha256": {str(path): file_sha256(path) for path in source_files},
        "final_test_access": False,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REGISTRATION, registration)
    print(json.dumps({"status": registration["status"], "registration": str(REGISTRATION), "sha256": file_sha256(REGISTRATION)}, indent=2))


if __name__ == "__main__":
    main()
