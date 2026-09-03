#!/usr/bin/env python3
"""Complete paired analysis only; never choose a checkpoint or read final test."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.check_episode_proof import audit_batch
from scripts.stage6_feedback import ROOT, REGISTRATION, read, validate, paired_interval, write_new
from tetris.rl import advantage_diagnostics, directory_sha256, file_sha256


def evaluation_rows(r, label, digest):
    path = ROOT / "evaluation" / label / "complete.json"
    completed = read(path)
    if (completed["status"] != "completed" or completed["adapter_sha256"] != digest
            or completed["registration_sha256"] != file_sha256(REGISTRATION)):
        raise ValueError("evaluation provenance mismatch")
    rows = {"recovery": [], "ordinary": []}
    for name, sha in sorted(completed["files_sha256"].items()):
        if file_sha256(Path(name)) != sha:
            raise ValueError("evaluation shard changed")
        shard = read(name)
        if shard["adapter_sha256"] != digest or not shard["greedy"] or shard["final_test_access"]:
            raise ValueError("evaluation identity mismatch")
        rows[shard["kind"]].extend(shard["games"])
    for kind, values in rows.items():
        if [v["seed"] for v in values] != r["evaluation"][kind + "_seeds"]:
            raise ValueError("incomplete or duplicate evaluation cohort")
    return rows


def training_evidence(r, run):
    root = Path("runs") / run["run_id"] / "rl"
    m = read(root / "manifest.json")
    expected = {k: v for k, v in r["recipe"].items() if k not in {"save_every", "score_scale", "death_penalty", "illegal_penalty"}}
    expected.update(status="completed", completed_updates=32, training_seed=run["seed"],
        advantage_method=run["method"], adapter_sha256=r["initial_adapter_sha256"],
        frozen_sft_adapter_sha256=r["initial_adapter_sha256"], reference_adapter_sha256=r["initial_adapter_sha256"], reference_frozen=True,
        external_registration_sha256=file_sha256(REGISTRATION), base_model_revision=r["base_model_revision"])
    expected["reward"] = {"formula": "normalized_score - death - illegal", "weights": {
        key: r["recipe"][key] for key in ("score_scale", "death_penalty", "illegal_penalty")}}
    if any(m.get(k) != v for k, v in expected.items()):
        raise ValueError(f"training differs from registration: {run['run_id']}")
    if m["reference_weights_sha256_before"] != m["reference_weights_sha256_after"]:
        raise ValueError("reference changed")
    if (len(m["update_metrics"]) != 32 or any(not math.isfinite(x[key])
            for x in m["update_metrics"] for key in ("gradient_norm_before_clip", "loss", "kl"))):
        raise ValueError("incomplete or non-finite learning diagnostics")
    if directory_sha256(root / "adapter") != m["output_adapter_sha256"]:
        raise ValueError("candidate adapter changed")
    bank = [json.loads(s) for s in Path(r["recovery_start_file"]).read_text().splitlines()]
    seeds = read(r["training_seed_file"])
    recipe = {**r["recipe"], "advantage_method": run["method"]}
    counters = {"empty": [], "recovery": []}
    counts = []
    for update, scheduled in enumerate(r["schedules"][str(run["seed"])], 1):
        batch = read(root / "trajectory_batches" / f"update-{update:06d}.json")
        if batch["update"] != update or batch["environment_seed"] != scheduled["environment_seed"]:
            raise ValueError("paired schedule differs")
        if scheduled["state_hash"]:
            from tetris.rl import state_hash
            if state_hash(batch["trajectories"][0]["start_state"]) != scheduled["state_hash"]:
                raise ValueError("paired recovery state differs")
        counts.append(audit_batch(batch, recipe, seeds, bank))
        diagnostic = advantage_diagnostics(batch["trajectories"])
        counters["recovery" if update % 2 == 0 else "empty"].append(diagnostic)
        if m["update_metrics"][update-1]["advantage_diagnostics"] != diagnostic:
            raise ValueError("saved zero-advantage diagnostics differ")
    if counts != [x["turns"] for x in m["update_metrics"]] or sum(counts) != m["sample_count"]:
        raise ValueError("training accounting differs")
    return m, counters


def summaries(rows):
    return {kind: {"games": len(games), "illegal_endings": sum(g["death_reason"] == "illegal_action" for g in games),
        "topouts": sum(g["death_reason"] == "topped_out" for g in games),
        "survivors": sum(g["death_reason"] == "cap_reached" for g in games),
        "parse_failures": sum(g.get("terminal_incident", {}).get("parsed") is False for g in games),
        "mean_pieces": statistics.mean(g["pieces"] for g in games),
        "mean_score": statistics.mean(g["score"] for g in games),
        "mean_lines": statistics.mean(g["lines"] for g in games)} for kind, games in rows.items()}


def decision(ci, *, ordinary_ok, recovery_ok, signal_ok, analysis):
    advance = (ci["mean"] >= analysis["minimum_useful_reduction"]
        and ci["ci95"][0] > analysis["require_lower_bound_above"]
        and sum(v > 0 for v in ci["per_training_seed"]) >= analysis["min_improving_seeds"]
        and ordinary_ok and recovery_ok and signal_ok)
    return "helps" if advance else "hurts" if ci["ci95"][1] < 0 else "inconclusive", advance


def analyze():
    r = validate()
    baseline = evaluation_rows(r, "sft", r["initial_adapter_sha256"])
    rows, diagnostics = {}, {}
    for run in r["run_order"]:
        m, counters = training_evidence(r, run)
        rows[(run["method"], run["seed"])] = evaluation_rows(r, run["run_id"], m["output_adapter_sha256"])
        diagnostics[run["run_id"]] = {"by_start_kind": counters, "movement": m["adapter_movement"],
            "update_metrics": m["update_metrics"], "seconds": m["wall_clock_seconds"], "decisions": m["sample_count"]}
    primary = [[int(a["death_reason"] == "illegal_action") - int(b["death_reason"] == "illegal_action")
                for a, b in zip(rows[("active_group", seed)]["recovery"], rows[("fixed_zero", seed)]["recovery"], strict=True)]
               for seed in r["training_seeds"]]
    analysis = r["analysis"]
    ci = paired_interval(primary, analysis["bootstrap_replicates"], analysis["bootstrap_seed"])
    results = {"sft": summaries(baseline), **{f"{method}-{seed}": summaries(v) for (method, seed), v in rows.items()}}
    ordinary_ok = recovery_ok = True
    ordinary_failures = []
    for seed in r["training_seeds"]:
        candidate = results[f"fixed_zero-{seed}"]
        for other in (results["sft"], results[f"active_group-{seed}"]):
            recovery_ok &= candidate["recovery"]["survivors"] >= other["recovery"]["survivors"]
            for key in ("mean_score", "mean_lines"):
                ordinary_ok &= candidate["ordinary"][key] >= .99 * other["ordinary"][key]
        for method in r["methods"]:
            ordinary = results[f"{method}-{seed}"]["ordinary"]
            if any(ordinary[k] for k in ("illegal_endings", "topouts", "parse_failures")):
                ordinary_ok = False
                ordinary_failures.append(f"{method}-{seed}")
    signal_ok = all(d["terminal_illegal"]["effective_zero"] == 0 and
        d["terminal_illegal"]["negative"] == d["terminal_illegal"]["count"]
        for key, values in diagnostics.items() if "fixed_zero" in key
        for ds in values["by_start_kind"].values() for d in ds)
    outcome, advance = decision(ci, ordinary_ok=ordinary_ok, recovery_ok=recovery_ok, signal_ok=signal_ok, analysis=analysis)
    secondary = {}
    for kind in ("recovery", "ordinary"):
        for metric in ("pieces", "score", "lines", "survived", "topout"):
            def value(g):
                return int(g["death_reason"] == "cap_reached") if metric == "survived" else int(g["death_reason"] == "topped_out") if metric == "topout" else g[metric]
            matrix = [[value(b) - value(a) for a, b in zip(rows[("active_group", seed)][kind], rows[("fixed_zero", seed)][kind], strict=True)] for seed in r["training_seeds"]]
            secondary[f"{kind}_{metric}_revised_minus_original"] = paired_interval(matrix)
    report = {"outcome": outcome, "advance": advance, "primary": ci, "paired_primary_matrix": primary,
        "secondary_exploratory": secondary, "summary": results, "training_diagnostics": diagnostics,
        "gates": {"signal": signal_ok, "ordinary": ordinary_ok, "recovery_survival": recovery_ok},
        "ordinary_failures": ordinary_failures, "final_test_access": False,
        "registration_sha256": file_sha256(REGISTRATION),
        "caveats": "Three seed pairs give limited training-seed uncertainty; crossed bootstrap is descriptive. Secondary intervals are unadjusted. Sampled training and greedy evaluation are distinct.",
        "next": "Propose bounded independent confirmation; retain SFT." if advance else "Retain SFT. Diagnose effects and uncertainty; no automatic scaling."}
    write_new(ROOT / "report.json", report)
    (ROOT / "report.md").write_text(f"# Feedback pilot: {outcome}\n\nRecovery illegal-ending reduction: {ci['mean']:.1%}; paired 95% interval {ci['ci95']}.\n\n{report['next']}\n\n{report['caveats']}\n")
    print(json.dumps({"outcome": outcome, "primary": ci, "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    analyze()
