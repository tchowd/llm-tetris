#!/usr/bin/env python3
"""Independent 320-update registration and evidence checks; old studies stay frozen."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.check_episode_proof import audit_batch
from scripts.train_recovery_sft import validate_inputs
from tetris.rl import atomic_write_json, directory_sha256, file_sha256

ROOT = Path("runs/stage6-scale10x-v1/rl")
REGISTRATION = ROOT / "registration-v2.json"
PROTOCOL = Path("runs/stage6-recovery-v1/rl/registration.json")
APPROVAL = ROOT / "budget-amendment-v1.json"
PILOT = "rl-scale10x-seed0"
PROOF = "rl-scale10x-proof-seed0"
CONTROL = "rl-scale10x-proof-control-seed0"
RUNS = (PROOF, CONTROL, PILOT)
HISTORICAL_HASHES = {
    "runs/stage6-recovery-v1/rl/report.json": "5a66541228de53e94cfa0c51ac2def9afaee61216029527d5d9639e1d8d9fe96",
    "scripts/train_episode_rl.py": "eab70fff9a63058ac802c9778d064efedb0aee51733bb9646e417733d9aa5e4c",
    "scripts/check_episode_proof.py": "e6b92f0c121f310cd7a2603a46ad08ec1c279772323e044706b11b0fa8ee7770",
    "tetris/rl.py": "1f6a5d834ebb7e8b9b53e3fe0cbed7362d9cd8f69fd3badf8da628c07ab012d8",
}
EXECUTION = {"deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
    "float32_matmul_precision": "highest", "qwen3_completion_logits_suffix": True,
    "episode_normalization": "decimal50"}
CORRECTNESS = {"all_trajectories_replayed", "exact_gpu_resume_trajectories",
    "exact_gpu_resume_adapter_tensors", "exact_token_alignment", "positive_delayed_reward_direction",
    "probe_weights_restored", "reference_unchanged", "gpu_headroom", "frozen_sft_unchanged"}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f"refusing to overwrite immutable artifact: {path}")
    atomic_write_json(path, value)


def recipe(updates, seed, hours):
    return {"updates": updates, "group_size": 4, "horizon": 128, "training_seed": seed,
        "learning_rate": 1e-6, "kl_beta": .05, "gamma": .99, "train_batch_size": 4,
        "score_scale": 100, "death_penalty": 2, "illegal_penalty": 10, "max_training_hours": hours}


def build_registration():
    approval, p = read(APPROVAL), read(PROTOCOL)
    if approval["status"] != "user_approved" or approval["limits"] != {
        "experiment_usd": 50, "cumulative_stage6_usd": 250, "overall_hours": 12,
        "includes_setup_proof_training_evaluation_backup_and_cleanup": True}:
        raise ValueError("explicit 50/250 USD and 12-hour approval required")
    for name, digest in HISTORICAL_HASHES.items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"historical evidence changed: {name}")
    data = validate_inputs(p, Path(p["data"]["data_dir"]))
    if data["registration_sha256"] != file_sha256(PROTOCOL):
        raise ValueError("training data is not from frozen recovery protocol")
    if directory_sha256(Path("runs/sft-v1/adapter")) != p["frozen_sft_adapter_sha256"]:
        raise ValueError("original SFT changed")
    paths = [*Path("tetris").glob("*.py"), *Path("scripts").glob("*.py"),
        Path("infra/rl-scale10x-bootstrap.sh"), Path("requirements-train.txt"),
        Path("requirements-rl.txt"), Path("pyproject.toml"),
        Path("benchmarks/stress-v1/manifest.json"), Path(p["states_path"]),
        Path("runs/sft-v1/closed_loop/manifest.json"), Path("runs/sft-v1/closed_loop/metrics.json")]
    return {"experiment": "SCALE10X", "status": "registered", "registered_at": now(),
        "prelaunch_revision": 2, "superseded_prelaunch_registration_sha256": file_sha256(ROOT / "registration.json"),
        "prelaunch_revision_reason": "Retain the failed us-east-1a capacity attempt; support safe retries and the existing default subnets in AWS-reported capacity zones. No training has run.",
        "run_id": PROOF, "control_run_id": CONTROL, "pilot_run_id": PILOT,
        "protocol_path": str(PROTOCOL), "protocol_sha256": file_sha256(PROTOCOL),
        "approval_sha256": file_sha256(APPROVAL), "historical_sha256": HISTORICAL_HASHES,
        "source_sha256": {str(f): file_sha256(f) for f in sorted(paths)},
        "dataset_manifest_sha256": file_sha256(Path(p["data"]["data_dir"]) / "manifest.json"),
        "recipe": {**recipe(4, 6102, 1), "pause_after_update": 2},
        "pilot_recipe": recipe(320, 6103, 9), "execution": EXECUTION,
        "question": "Does tenfold episode training from frozen SFT improve long-game score and recovery without legality regression?",
        "proof_question": "On L40S, do exact replay, delayed credit, GPU token probabilities and resume agree at horizon 128?",
        "instance_type": "g6e.2xlarge", "hourly_usd": 2.30,
        "prior_stage_spend_usd": 16.0, "pilot_usd": 50, "stage_usd": 250,
        "block_hours": 12, "evaluation_reserve_seconds": 7800,
        "logprob_absolute_tolerance": .0001, "resume_weights_absolute_tolerance": 0,
        "minimum_allocated_gpu_headroom_fraction": .15,
        "positive_direction_sgd_learning_rate": .001, "positive_direction_min_logprob_gain": 0,
        "pilot_projection_safety_multiplier": 1.5, "pilot_projection_startup_seconds": 120,
        "promotion": p["episode_promotion"], "final_test_access": False,
        "checkpoint_interval": 4, "optimizer_checkpoints_uploaded": False,
        "research_complete": False, "deployment_authorized": False}


def validate_registration(path=REGISTRATION):
    r = read(path)
    expected = build_registration()
    if set(r) != set(expected) or any(r[k] != v for k, v in expected.items() if k != "registered_at"):
        raise ValueError("scale registration, source, budget or frozen inputs changed")
    p = read(PROTOCOL)
    return r, p


def training_evidence(root, r, p, registration_path, *, proof=False, resumed=False):
    m = read(root / "manifest.json")
    rec = r["recipe"] if proof else r["pilot_recipe"]
    data = Path(p["data"]["data_dir"])
    seeds = read(data / "training-seeds.json")
    expected = {k: rec[k] for k in ("updates", "group_size", "horizon", "training_seed", "learning_rate", "kl_beta", "gamma", "train_batch_size")}
    expected.update(status="completed", experiment="E6", completed_updates=rec["updates"],
        external_registration_sha256=file_sha256(registration_path),
        research_question=r["proof_question"] if proof else r["question"],
        adapter_sha256=p["frozen_sft_adapter_sha256"], reference_adapter_sha256=p["frozen_sft_adapter_sha256"],
        frozen_sft_adapter_sha256=p["frozen_sft_adapter_sha256"], base_model_revision=p["base_model_revision"],
        requested_base_model_revision=p["base_model_revision"], benchmark_manifest_sha256=p["benchmark_manifest_sha256"],
        environment_training_seeds=seeds, recovery_starts_sha256=file_sha256(data / "train-starts.jsonl"),
        temperature=1, reference_frozen=True, execution=r["execution"],
        budgets_usd={"pilot": r["pilot_usd"], "stage": r["stage_usd"]},
        prior_stage_spend_usd=r["prior_stage_spend_usd"], instance_hourly_usd=r["hourly_usd"],
        max_wall_clock_hours=rec["max_training_hours"],
        reward={"formula": "normalized_score - death - illegal", "weights": {k: rec[k] for k in ("score_scale", "death_penalty", "illegal_penalty")}})
    if proof:
        expected.update(resume_used=resumed, resumed_from_update=2 if resumed else 0)
    differences = [k for k, v in expected.items() if m.get(k) != v]
    if differences or m["registered_at"] < r["registered_at"]:
        raise ValueError(f"training differs from registration: {root}: {differences}")
    if m["reference_weights_sha256_before"] != m["reference_weights_sha256_after"]:
        raise ValueError("reference changed")
    if directory_sha256(root / "adapter") != m["output_adapter_sha256"]:
        raise ValueError("output adapter hash differs")
    paths = sorted((root / "trajectory_batches").glob("*.json"))
    if len(paths) != rec["updates"]:
        raise ValueError("missing or duplicate trajectory batches")
    bank = [json.loads(s) for s in (data / "train-starts.jsonl").read_text().splitlines()]
    rows, counts = [], []
    for update, path in enumerate(paths, 1):
        batch = read(path)
        if batch["update"] != update:
            raise ValueError("committed update order differs")
        counts.append(audit_batch(batch, rec, seeds, bank))
        rows.append(batch)
    if [x["update"] for x in m["update_metrics"]] != list(range(1, rec["updates"] + 1)) or counts != [x["turns"] for x in m["update_metrics"]] or sum(counts) != m["sample_count"]:
        raise ValueError("sample or update accounting differs")
    if "L40S" not in m.get("gpu_name", ""):
        raise ValueError("training did not execute on registered L40S")
    return m, rows


def evidence(r, p, registration_path):
    root, control = (Path("runs") / run / "rl" for run in (r["run_id"], r["control_run_id"]))
    left, batches = training_evidence(root, r, p, registration_path, proof=True, resumed=True)
    right, control_batches = training_evidence(control, r, p, registration_path, proof=True)
    paused = read(root / "paused-manifest.json")
    if paused["status"] != "paused" or paused["completed_updates"] != 2 or paused["sample_count"] != sum(x["turns"] for x in left["update_metrics"][:2]):
        raise ValueError("pause evidence lost committed progress")
    if batches != control_batches:
        raise ValueError("resumed trajectories differ from uninterrupted control")
    return root, control, [left, right], batches


def projection(r, p, metrics):
    if len(metrics) != r["recipe"]["updates"] or any(m["turns"] <= 0 or not math.isfinite(m["seconds"]) or m["seconds"] <= 0 for m in metrics):
        raise ValueError("incomplete measured throughput")
    worst = max(m["seconds"] / m["turns"] for m in metrics)
    rec = r["pilot_recipe"]
    turns = rec["updates"] * rec["group_size"] * rec["horizon"]
    seconds = r["pilot_projection_startup_seconds"] + worst * turns * r["pilot_projection_safety_multiplier"]
    return {"worst_seconds_per_turn": worst, "full_length_pilot_turns": turns,
        "safety_multiplier": r["pilot_projection_safety_multiplier"], "projected_seconds": seconds,
        "projected_usd": seconds / 3600 * r["hourly_usd"],
        "within_training_limit": seconds <= rec["max_training_hours"] * 3600}


def check_proof(path=REGISTRATION):
    r, p = validate_registration(path)
    root, control, manifests, _ = evidence(r, p, path)
    g = read(ROOT / "gpu-proof.json")
    if g.get("experiment") != "SCALE10X_PROOF" or g.get("registration_sha256") != file_sha256(path) or g.get("protocol_sha256") != r["protocol_sha256"] or g.get("final_test_access") is not False:
        raise ValueError("GPU proof identity changed")
    expected_checks = {k: True for k in CORRECTNESS | {"pilot_projection_fits"}}
    if g.get("status") != "passed" or g.get("checks") != expected_checks:
        raise ValueError("GPU proof or conservative training projection did not pass")
    paths = {str(f) for base in (root, control) for f in [base / "manifest.json", *sorted((base / "trajectory_batches").glob("*.json"))]}
    paths.add(str(root / "paused-manifest.json"))
    if set(g["evidence_sha256"]) != paths or any(file_sha256(Path(f)) != sha for f, sha in g["evidence_sha256"].items()):
        raise ValueError("GPU proof artifacts changed")
    if file_sha256(root / "adapter/adapter_model.safetensors") != file_sha256(control / "adapter/adapter_model.safetensors"):
        raise ValueError("resumed adapter tensors differ")
    d, gpu = g["positive_direction"], g["gpu"]
    if not (0 <= g["max_absolute_logprob_error"] <= r["logprob_absolute_tolerance"] and g["tokens_checked"] > 0 and math.isfinite(d["gradient_norm"]) and d["gradient_norm"] > 0 and d["mean_logprob_after"] > d["mean_logprob_before"] and d["restored"] is True):
        raise ValueError("GPU probability or positive-credit proof failed")
    if not (0 < gpu["peak_allocated_bytes"] <= gpu["total_bytes"] and math.isclose(gpu["headroom_fraction"], 1 - gpu["peak_allocated_bytes"] / gpu["total_bytes"], abs_tol=1e-12) and gpu["headroom_fraction"] >= .15 and "L40S" in gpu["name"]):
        raise ValueError("GPU memory or identity proof failed")
    if g["pilot_projection"] != projection(r, p, manifests[0]["update_metrics"]) or g["sample_count"] != manifests[0]["sample_count"]:
        raise ValueError("measured projection differs")
    return g


def training_command(r, p, *, proof=False, control=False, resume=None, pause=False):
    rec = r["recipe"] if proof else r["pilot_recipe"]
    run = r["control_run_id"] if control else r["run_id"] if proof else r["pilot_run_id"]
    cmd = [sys.executable, "scripts/train_episode_rl.py", "--experiment", "E6",
        "--question", r["proof_question"] if proof else r["question"],
        "--registration-file", str(REGISTRATION), "--adapter-dir", "runs/sft-v1/adapter",
        "--frozen-sft-adapter-dir", "runs/sft-v1/adapter", "--base-model-revision", p["base_model_revision"],
        "--benchmark-manifest", p["benchmark_manifest"], "--stage5-manifest", "runs/sft-v1/closed_loop/manifest.json",
        "--recovery-starts", p["data"]["data_dir"] + "/train-starts.jsonl",
        "--training-seeds-file", p["data"]["data_dir"] + "/training-seeds.json", "--temperature", "1",
        "--save-every", "1" if proof else "4", "--out-dir", f"runs/{run}/rl"]
    for k in ("updates", "group_size", "horizon", "training_seed", "learning_rate", "kl_beta", "gamma", "train_batch_size", "score_scale", "death_penalty", "illegal_penalty"):
        cmd += ["--" + k.replace("_", "-"), str(rec[k])]
    for key, value in {"pilot-dollar-limit": 50, "stage-dollar-limit": 250,
        "prior-stage-spend-usd": r["prior_stage_spend_usd"], "instance-hourly-usd": r["hourly_usd"],
        "max-wall-clock-hours": rec["max_training_hours"]}.items():
        cmd += ["--" + key, str(value)]
    if pause:
        cmd += ["--pause-after-update", "2"]
    if resume:
        cmd += ["--resume", str(resume)]
    return cmd


def diagnostics(batches):
    steps = [s for b in batches for t in b["trajectories"] for s in t["steps"]]
    endings = {}
    for b in batches:
        for t in b["trajectories"]:
            reason = t["steps"][-1]["terminal_reason"] or "horizon_cap"
            endings[reason] = endings.get(reason, 0) + 1
    return {"trajectories": sum(len(b["trajectories"]) for b in batches),
        "sampled_decisions": len(steps), "unique_start_seeds": len({b["environment_seed"] for b in batches}),
        "illegal_sampled_decisions": sum(not s["legal"] for s in steps), "endings": endings,
        "zero_advantage_fraction": sum(s["advantage"] == 0 for s in steps) / len(steps),
        "per_update_return_variance": [statistics.pvariance(t["episode_return"] for t in b["trajectories"]) for b in batches],
        "caution": "Training is sampled at temperature 1; this is not the greedy evaluation failure rate."}


def assess(path=REGISTRATION):
    from scripts.analyze_stage6 import game_metric
    from scripts.check_e2_learning import evaluation
    from scripts.check_e4_pilot import frozen_stage5, promotion_checks
    from scripts.check_recovery_pilot import fresh_summary
    from tetris.rl import paired_comparison
    r, p = validate_registration(path)
    proof = check_proof(path)
    root = Path("runs") / PILOT / "rl"
    m, batches = training_evidence(root, r, p, path)
    base = Path("runs/stage6-e0/rl/stress-development")
    candidate_path = root / "stress-development"
    baseline, candidate = evaluation(base, "sft", p), evaluation(candidate_path, "scale10x", p)
    if baseline["adapter_sha256"] != p["frozen_sft_adapter_sha256"] or candidate["adapter_sha256"] != m["output_adapter_sha256"]:
        raise ValueError("evaluation adapter identity differs")
    comparison = paired_comparison(game_metric(base, "sft")[0], game_metric(candidate_path, "scale10x")[0], bootstrap_samples=10000, seed=6205)
    stage5 = frozen_stage5(root / "stage5", "scale10x", m["output_adapter_sha256"], p["base_model_revision"], read(p["benchmark_manifest"])["decision_rules"])
    fresh = fresh_summary(root / "fresh-recovery", "scale10x", m["output_adapter_sha256"], p, file_sha256(PROTOCOL))
    checks = promotion_checks(comparison, candidate, stage5, r["promotion"]["min_relative_score_gain"])
    return {"experiment": "SCALE10X", "status": "passed" if all(checks.values()) else "not_passed",
        "checks": checks, "registration_sha256": file_sha256(path),
        "training_manifest_sha256": file_sha256(root / "manifest.json"),
        "gpu_proof_sha256": file_sha256(ROOT / "gpu-proof.json"), "baseline": baseline,
        "candidate": candidate, "paired_score_comparison": comparison, "stage5": stage5,
        "fresh_candidate": fresh, "fresh_baseline": read("runs/rl-r1-recovery-sft-seed0/rl/r1-gate.json")["fresh_baseline"],
        "training_diagnostics": diagnostics(batches), "final_test_access": False,
        "research_complete": False, "operations_complete": False, "deployment_authorized": False,
        "generated_at": now(), "next": "Replicate only a qualifying pilot under unchanged confirmation rules and remaining budget; otherwise report the negative result after artifact audit and exact cleanup."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "preflight", "gpu-proof", "check-proof", "assess"))
    args = parser.parse_args()
    if args.action == "register":
        write_new(REGISTRATION, build_registration())
    elif args.action == "preflight":
        validate_registration()
    elif args.action == "gpu-proof":
        from scripts.prove_scale10x_gpu import gpu_proof
        result = gpu_proof(REGISTRATION)
        write_new(ROOT / "gpu-proof.json", result)
        print(json.dumps(result, indent=2))
        if result["status"] != "passed":
            raise SystemExit(2)
    elif args.action == "check-proof":
        check_proof()
    else:
        result = assess()
        write_new(Path("runs") / PILOT / "rl/scale10x-gate.json", result)
        print(json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2))
    print(args.action + " complete")


if __name__ == "__main__":
    main()
