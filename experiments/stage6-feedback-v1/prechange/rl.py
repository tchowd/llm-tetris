"""Stage 6 reinforcement-learning primitives.

This module deliberately contains no trainer dependency.  Rewards, state
identity, trajectory replay, seed isolation, and statistical comparisons are
therefore testable with the same small dependency set as the game engine.
GPU entry points in ``scripts/train_rl.py`` and
``scripts/train_episode_rl.py`` build on these contracts.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .engine import Game
from .serialize import parse_action

Action = tuple[int, int]


def runtime_budget(*, elapsed_seconds: float, hourly_usd: float, max_hours: float,
                   dollar_limit: float, remaining_updates: int = 0,
                   seconds_per_update: float = 0.0) -> dict:
    """Conservative projection including startup and already-spent wall time."""
    values = (elapsed_seconds, hourly_usd, max_hours, dollar_limit, seconds_per_update)
    if any(not math.isfinite(value) or value < 0 for value in values) or remaining_updates < 0:
        raise ValueError("runtime budgets require finite non-negative inputs")
    projected = elapsed_seconds + remaining_updates * seconds_per_update
    return {
        "elapsed_seconds": elapsed_seconds,
        "projected_seconds": projected,
        "projected_cost_usd": projected * hourly_usd / 3600,
        "stop": projected >= max_hours * 3600 or projected * hourly_usd / 3600 >= dollar_limit,
    }


@dataclass(frozen=True)
class DenseRewardWeights:
    """Positive penalty weights for the registered one-placement reward."""

    lines: float = 1.0
    holes: float = 1.0
    aggregate_height: float = 0.05
    bumpiness: float = 0.02
    illegal: float = 10.0

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in asdict(self).values()) or self.illegal <= 0:
            raise ValueError("dense reward weights must be finite, non-negative, with a positive illegal penalty")


@dataclass(frozen=True)
class EpisodeRewardWeights:
    """Small episode objective whose step rewards sum to episode return."""

    score_scale: float = 100.0
    death_penalty: float = 2.0
    illegal_penalty: float = 10.0

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in asdict(self).values()) or self.score_scale <= 0 or self.illegal_penalty <= 0:
            raise ValueError("episode reward requires positive score scale and illegal penalty, and non-negative finite weights")


@dataclass(frozen=True)
class Transition:
    reward: float
    components: dict[str, float]
    terminal: bool
    terminal_reason: str | None
    before: dict
    after: dict | None


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    """Content hash a retained adapter without depending on mtimes."""

    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"no files to hash in {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def check_resume_registration(previous: dict, registered: dict, keys: Sequence[str]) -> None:
    """A resume may update runtime evidence, never the registered experiment."""

    changed = [key for key in keys if previous.get(key) != registered.get(key)]
    if changed:
        raise ValueError(f"resume changes pre-registered fields: {', '.join(changed)}")


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def record_run_failure(manifest_path: Path | None, error: BaseException) -> None:
    """Leave durable failed/interrupted evidence without hiding the exception."""

    if manifest_path is None or not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "status": "failed",
        "failure_type": type(error).__name__,
        "failure_message": str(error)[:2000],
        "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    atomic_write_json(manifest_path, manifest)
    from .events import EventWriter

    EventWriter(manifest_path.parent / "events.jsonl", run_id=manifest["run_id"], stage=6).emit(
        "job_failed", phase=manifest.get("kind"), message=f"{type(error).__name__}: {str(error)[:1000]}"
    )


def state_identity(snapshot: dict) -> dict:
    """The complete externally observable state used by Stage 6.

    ``seed + action_prefix`` remains the authoritative generator state.  The
    redundant fields make drift obvious without depending on ``Game``'s
    private RNG representation.
    """

    return {
        "seed": snapshot["seed"],
        "turn": snapshot["turn"],
        "piece": snapshot["piece"],
        "next": snapshot["next"],
        "board": snapshot["board"],
        "score": snapshot["score"],
        "lines": snapshot["lines"],
        "prompt": snapshot["prompt"],
    }


def state_hash(snapshot: dict) -> str:
    return canonical_hash(state_identity(snapshot))


def restore_game(seed: int, actions: Sequence[Sequence[int]], *, expected: dict | None = None) -> Game:
    """Reconstruct exact board and bag state through the public engine API."""

    game = Game(seed=seed)
    for turn, raw in enumerate(actions):
        if game.game_over:
            raise ValueError(f"action prefix continues after terminal state at turn {turn}")
        game.step(int(raw[0]), int(raw[1]))
    if expected is not None:
        actual = state_identity(game.snapshot())
        wanted = {key: expected[key] for key in actual if key in expected}
        comparable = {key: actual[key] for key in wanted}
        if comparable != wanted:
            raise ValueError("reconstructed state does not match its recorded identity")
    return game


def record_state(game: Game, actions: Sequence[Sequence[int]], *, state_id: str | None = None) -> dict:
    snap = game.snapshot()
    record = {
        "state_id": state_id or f"seed-{game.seed}-turn-{game.turn}",
        "seed": game.seed,
        "action_prefix": [[int(a[0]), int(a[1])] for a in actions],
        **state_identity(snap),
    }
    record["state_hash"] = state_hash(snap)
    return record


def parse_completion(text: str | None) -> Action | None:
    if not text:
        return None
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    try:
        return parse_action(first_line)
    except ValueError:
        return None


def dense_transition(game: Game, action: Action | None, weights: DenseRewardWeights) -> Transition:
    """Score one action on a clone; the supplied game is never mutated."""

    before = game.snapshot()
    legal = {(p["rot"], p["x"]) for p in before["legal"]}
    if action is None or action not in legal:
        components = {
            "lines_cleared": 0.0,
            "delta_holes_total": 0.0,
            "delta_aggregate_height": 0.0,
            "delta_bumpiness": 0.0,
            "illegal": 1.0,
        }
        return Transition(
            reward=-weights.illegal,
            components=components,
            terminal=True,
            terminal_reason="illegal_action",
            before=before,
            after=None,
        )

    trial = game.clone()
    after = trial.step(*action)
    components = {
        "lines_cleared": float(after["lines"] - before["lines"]),
        "delta_holes_total": float(after["holes_total"] - before["holes_total"]),
        "delta_aggregate_height": float(after["aggregate_height"] - before["aggregate_height"]),
        "delta_bumpiness": float(after["bumpiness"] - before["bumpiness"]),
        "illegal": 0.0,
    }
    reward = (
        weights.lines * components["lines_cleared"]
        - weights.holes * components["delta_holes_total"]
        - weights.aggregate_height * components["delta_aggregate_height"]
        - weights.bumpiness * components["delta_bumpiness"]
    )
    return Transition(
        reward=reward,
        components=components,
        terminal=trial.game_over,
        terminal_reason="topped_out" if trial.game_over else None,
        before=before,
        after=after,
    )


def episode_transition(game: Game, action: Action | None, weights: EpisodeRewardWeights) -> Transition:
    """Score one episode step; summing rewards yields normalized score minus penalties."""

    before = game.snapshot()
    legal = {(p["rot"], p["x"]) for p in before["legal"]}
    if action is None or action not in legal:
        components = {
            "normalized_score_delta": 0.0,
            "death_penalty": 0.0,
            "illegal_penalty": weights.illegal_penalty,
        }
        return Transition(
            reward=-weights.illegal_penalty,
            components=components,
            terminal=True,
            terminal_reason="illegal_action",
            before=before,
            after=None,
        )

    trial = game.clone()
    after = trial.step(*action)
    score_delta = (after["score"] - before["score"]) / weights.score_scale
    death = weights.death_penalty if trial.game_over else 0.0
    components = {
        "normalized_score_delta": float(score_delta),
        "death_penalty": float(death),
        "illegal_penalty": 0.0,
    }
    return Transition(
        reward=score_delta - death,
        components=components,
        terminal=trial.game_over,
        terminal_reason="topped_out" if trial.game_over else None,
        before=before,
        after=after,
    )


def discounted_reward_to_go(rewards: Sequence[float], gamma: float = 0.99) -> list[float]:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    out = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        out[index] = running
    return out


def normalize_group(values: Sequence[float], epsilon: float = 1e-8) -> list[float]:
    """Population normalization used only inside one shared-start group."""

    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance + epsilon)
    return [(value - mean) / scale for value in values]


def trajectory_advantages(reward_groups: Sequence[Sequence[float]], gamma: float = 0.99) -> list[list[float]]:
    """Normalize reward-to-go across trajectories at the same turn only."""

    returns = [discounted_reward_to_go(rewards, gamma) for rewards in reward_groups]
    advantages = [[0.0] * len(row) for row in returns]
    max_turns = max((len(row) for row in returns), default=0)
    for turn in range(max_turns):
        active = [(index, row[turn]) for index, row in enumerate(returns) if turn < len(row)]
        # Same population-normalization formula, with portable rounding for
        # exact replay across ARM/macOS and x86/Linux. Dense RL is unchanged.
        from decimal import Decimal, localcontext
        with localcontext() as context:
            context.prec = 50
            values = [Decimal.from_float(value) for _, value in active]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            scale = (variance + Decimal.from_float(1e-8)).sqrt()
            normalized = [float((value - mean) / scale) for value in values]
        for (index, _), value in zip(active, normalized):
            advantages[index][turn] = value
    return advantages


def grouped_policy_loss(policy_logprobs, reference_logprobs, advantages, beta: float):
    """GRPO-style policy loss with the non-negative k3 KL estimator.

    Tensor imports remain deferred so the base engine does not acquire a
    PyTorch dependency.
    """

    delta = reference_logprobs - policy_logprobs
    kl = delta.exp() - delta - 1.0
    return (-advantages * policy_logprobs + beta * kl).mean()


def validate_trajectory(record: dict) -> dict:
    """Replay every legal saved action and prove turn-by-turn state fidelity."""

    game = restore_game(record["seed"], record.get("start_actions", []), expected=record.get("start_state"))
    checked = 0
    terminal_reason = None
    for index, step in enumerate(record["steps"]):
        before = game.snapshot()
        if step["turn"] != before["turn"] or step["before_state_hash"] != state_hash(before):
            raise ValueError(f"trajectory drift before saved turn {step.get('turn')}")
        if "serialized_prompt" in step and step["serialized_prompt"] != before["prompt"]:
            raise ValueError("trajectory prompt does not match the replayed board")
        action = tuple(step["action"]) if step.get("action") is not None else None
        if "raw_completion" in step and parse_completion(step["raw_completion"]) != action:
            raise ValueError("saved action does not parse from its raw completion")
        if "completion_ids" in step:
            for key in ("policy_token_logprobs_at_sampling", "reference_token_logprobs"):
                if key in step and (
                    len(step[key]) != len(step["completion_ids"])
                    or any(not math.isfinite(value) for value in step[key])
                ):
                    raise ValueError("saved token log-probabilities do not align with completion IDs")
        legal = {(p["rot"], p["x"]) for p in before["legal"]}
        if action is None or action not in legal:
            if not step.get("terminal") or step.get("terminal_reason") != "illegal_action":
                raise ValueError("illegal saved action was not recorded as terminal")
            terminal_reason = "illegal_action"
            checked += 1
            if index != len(record["steps"]) - 1:
                raise ValueError("trajectory contains samples after a terminal action")
            break
        after = game.step(*action)
        if step.get("after_state_hash") != state_hash(after):
            raise ValueError(f"trajectory drift after saved turn {step.get('turn')}")
        checked += 1
        if game.game_over:
            terminal_reason = "topped_out"
            if not step.get("terminal") or step.get("terminal_reason") != "topped_out":
                raise ValueError("top-out was not recorded as terminal")
            if index != len(record["steps"]) - 1:
                raise ValueError("trajectory contains samples after top-out")
            break
    return {
        "ok": True,
        "steps_checked": checked,
        "terminal_reason": terminal_reason,
        "final_state_hash": state_hash(game.snapshot()),
    }


def validate_seed_manifest(manifest: dict, *, stage3_ranges: Iterable[range] = ()) -> dict:
    """Reject training/development/test leakage before any rollout starts."""

    required = ("training_seeds", "development_seeds", "test_seeds", "stage5_seeds")
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(f"seed manifest missing fields: {', '.join(missing)}")
    groups = {name: [int(seed) for seed in manifest[name]] for name in required}
    if any(not values for values in groups.values()):
        raise ValueError("training, development, test, and Stage 5 seed lists must be non-empty")
    optional = ("recovery_source_seeds", "probe_source_seeds")
    groups.update({name: [int(seed) for seed in manifest.get(name, [])] for name in optional})
    for name, seeds in groups.items():
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed in {name}")
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = set(groups[left]) & set(groups[right])
            if overlap:
                raise ValueError(f"seed overlap between {left} and {right}: {sorted(overlap)[:5]}")
    stage3 = set()
    for values in stage3_ranges:
        stage3.update(values)
    for name, seeds in groups.items():
        overlap = stage3 & set(seeds)
        if overlap:
            raise ValueError(f"{name} overlaps Stage 3: {sorted(overlap)[:5]}")
    return {
        "ok": True,
        "counts": {name: len(seeds) for name, seeds in groups.items()},
        "manifest_hash": canonical_hash(manifest),
    }


def validate_entry_gate(stage5_manifest_path: Path, expected_seeds: Sequence[int]) -> dict:
    """Require the frozen strict SFT evidence before any GPU training run."""

    manifest = json.loads(stage5_manifest_path.read_text())
    metrics_path = stage5_manifest_path.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text())["model"]["strict"]
    if manifest.get("seeds") != list(expected_seeds) or manifest.get("cap") != 500:
        raise ValueError("Stage 5 entry evidence does not use the frozen seed list and 500-piece cap")
    checks = {
        "no_deaths": metrics["deaths"] == 0,
        "no_parse_failures": metrics["parse_failure_rate"]["mean"] == 0,
        "no_illegal_actions": metrics["illegal_rate"]["mean"] == 0 and metrics.get("illegal_action_deaths", 0) == 0,
        "strong_play": metrics["lines"]["mean"] >= 196.77,
    }
    if not all(checks.values()):
        raise ValueError(f"frozen Stage 5 entry gate failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "manifest_sha256": file_sha256(stage5_manifest_path),
        "metrics_sha256": file_sha256(metrics_path),
    }


def bootstrap_mean_ci(values: Sequence[float], *, samples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    low = means[max(0, int(0.025 * samples) - 1)]
    high = means[min(samples - 1, int(0.975 * samples))]
    return (low, high)


def paired_comparison(
    baseline: dict[int, float],
    candidate: dict[int, float],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict:
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))
        missing_baseline = sorted(set(candidate) - set(baseline))
        raise ValueError(
            f"paired seeds differ; candidate missing {missing_candidate[:5]}, baseline missing {missing_baseline[:5]}"
        )
    seeds = sorted(baseline)
    differences = [float(candidate[item] - baseline[item]) for item in seeds]
    ci_low, ci_high = bootstrap_mean_ci(differences, samples=bootstrap_samples, seed=seed)
    mean = sum(differences) / len(differences) if differences else 0.0
    baseline_mean = sum(baseline.values()) / len(baseline) if baseline else 0.0
    return {
        "n": len(seeds),
        "seeds": seeds,
        "differences": differences,
        "mean_difference": mean,
        "median_difference": statistics.median(differences) if differences else 0.0,
        "bootstrap_95_ci": [ci_low, ci_high],
        "relative_improvement": mean / baseline_mean if baseline_mean else None,
    }


def config_dict(value: DenseRewardWeights | EpisodeRewardWeights) -> dict:
    return asdict(value)
