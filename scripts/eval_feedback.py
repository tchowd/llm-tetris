#!/usr/bin/env python3
"""Greedy development-only pilot evaluation, resumable by fixed cohort shard."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_stress import run_recovery_rollouts
from scripts.stage6_feedback import REGISTRATION, ROOT, read, validate, write_new
from tetris.engine import Game
from tetris.model_policy import build_model_policy
from tetris.rl import directory_sha256, file_sha256, record_state


def approved_session(path, registration=REGISTRATION):
    """New explicit authority only; never consult prior Stage 6 approvals."""
    from datetime import datetime, timezone
    import math
    a = read(path)
    if (a.get("experiment") != "stage6-feedback-v1" or a.get("status") != "user_approved"
            or a.get("registration_sha256") != file_sha256(registration)
            or not a.get("user_approval_text")):
        raise ValueError("new explicit user approval for this exact registration is required")
    for key in ("hard_limit_usd", "hourly_usd", "session_hours"):
        if not isinstance(a.get(key), (float, int)) or not math.isfinite(a[key]) or a[key] <= 0:
            raise ValueError("invalid approved budget")
    deadline = datetime.fromisoformat(a["absolute_deadline_utc"].replace("Z", "+00:00"))
    if deadline <= datetime.now(timezone.utc):
        raise ValueError("approved deadline has expired")
    return a


def evaluate(label, approval_file):
    approved_session(approval_file)
    r = validate()
    runs = {run["run_id"]: run for run in r["run_order"]}
    if label != "sft" and label not in runs:
        raise ValueError("unregistered evaluation candidate")
    adapter = Path(r["initial_adapter"] if label == "sft" else f"runs/{label}/rl/adapter")
    if label != "sft":
        m = read(adapter.parent / "manifest.json")
        if (m["status"] != "completed" or m["completed_updates"] != 32
                or m["external_registration_sha256"] != file_sha256(REGISTRATION)
                or m["adapter_sha256"] != r["initial_adapter_sha256"]
                or m["advantage_method"] != runs[label]["method"]
                or m["training_seed"] != runs[label]["seed"]):
            raise ValueError("only the registered final-update candidate is eligible")
    digest = directory_sha256(adapter)
    out = ROOT / "evaluation" / label
    out.mkdir(parents=True, exist_ok=True)
    identity = {"label": label, "adapter_sha256": digest,
                "registration_sha256": file_sha256(REGISTRATION),
                "base_model_revision": r["base_model_revision"],
                "greedy": True, "final_test_access": False}
    identity_path = out / "identity.json"
    if identity_path.exists():
        if read(identity_path) != identity:
            raise ValueError("evaluation resume identity changed")
    else:
        write_new(identity_path, identity)
    states = [json.loads(s) for s in Path(r["evaluation"]["recovery_path"]).read_text().splitlines()]
    ordinary = [record_state(Game(seed), [], state_id=f"feedback-ordinary-{seed}")
                for seed in r["evaluation"]["ordinary_seeds"]]
    policy = None
    files = {}
    for kind, cohort, cap in (("recovery", states, 200), ("ordinary", ordinary, 1000)):
        for offset in range(0, len(cohort), 32):
            path = out / f"{kind}-{offset:04d}.json"
            shard = cohort[offset:offset+32]
            if not path.exists():
                approved_session(approval_file)
                if policy is None:
                    policy = build_model_policy(adapter, r["base_model"], "cuda", revision=r["base_model_revision"])
                    if policy.metadata["base_model_revision"] != r["base_model_revision"]:
                        raise ValueError("evaluation base revision changed")
                started = time.monotonic()
                games, metrics = run_recovery_rollouts(policy, shard, cap=cap, batch_size=32)
                if directory_sha256(adapter) != digest:
                    raise ValueError("adapter changed during evaluation")
                write_new(path, {**identity, "kind": kind, "cap": cap, "games": games,
                                 "metrics": metrics, "seconds": time.monotonic() - started})
            saved = read(path)
            if any(saved.get(k) != v for k, v in identity.items()) or [g["seed"] for g in saved["games"]] != [s["seed"] for s in shard]:
                raise ValueError("saved evaluation shard identity/cohort differs")
            files[str(path)] = file_sha256(path)
    if not (out / "complete.json").exists():
        write_new(out / "complete.json", {**identity, "status": "completed", "files_sha256": files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--approval-file", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.label, args.approval_file)
