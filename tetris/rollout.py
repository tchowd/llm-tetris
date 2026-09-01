"""Stage 5: closed-loop rollout harness.

Random, teacher, and model policies all drive `Game.step()` through the
same batched interface -- `policy_fn(snapshots, teacher_infos) ->
[(action, raw_text), ...]` -- one call per round for every game still
alive, so an LLM policy pays for one forward pass per round instead of one
per game (stage-5-eval.md's "Speed" section: "100 games x 500 pieces
becomes ~500 batched generations instead of 50,000 single ones").
`gen_batch_size` chunks a round's alive games for memory reasons only; the
round itself always covers every alive game before the next round starts,
which is what makes that arithmetic hold. This is not a continuous-batching
scheduler that backfills a fixed working set as games die -- deliberately
simpler, and sufficient at Stage 5's ~100-seed scale.

`run_rollout` needs the teacher's 2-ply best action/value at every turn
regardless of which policy is under test (for on-policy teacher-match rate
and value gap), so it computes `teacher_infos` -- one `(best_action,
values_by_action)` pair per snapshot -- once per round and hands it to
`policy_fn`. `teacher_policy` reads straight from that instead of
re-running its own 2-ply search, so evaluating the teacher policy itself
doesn't pay for the search twice per turn.

Mirrors Stage 3's games.jsonl / replay split (tetris.dataset's
`generate_game` / `rebuild_rows_from_game`): `run_rollout` is the one place
a live game gets played and diagnosed; `replay_game_log` recomputes the same
diagnostics purely from a game record's seed + actions, so the two can be
checked against each other (tests/test_rollout.py's "replay" and "metric
arithmetic" gate tests). See plan/stage-5-eval.md.
"""
from __future__ import annotations

import math
import multiprocessing
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Callable

from .board import board_to_lists
from .engine import Game
from .teacher import WEIGHTS as DEFAULT_WEIGHTS
from .teacher import value_of_placement

Action = tuple[int, int]
TeacherInfo = tuple[Action, dict]  # (best action, {action: 2-ply value})
PolicyFn = Callable[[list[dict], list[TeacherInfo]], "list[tuple[Action | None, str | None]]"]

# Stage 3 dumps (data/{smoke,batch1,batch2} as of this writing) use seeds
# 0..3139 and grow from seed 0 upward. This offset keeps the default eval
# seed list unambiguously disjoint from any Stage 3 dump without needing to
# read a manifest at import time (stage-5-eval.md's "Seed disjointness").
EVAL_SEED_OFFSET = 10_000_000

DEATH_TOPPED_OUT = "topped_out"
DEATH_ILLEGAL_ACTION = "illegal_action"
CAP_REACHED = "cap_reached"

STRICT = "strict"
ASSISTED = "assisted"


def default_eval_seeds(n: int = 100, offset: int = EVAL_SEED_OFFSET) -> list[int]:
    return list(range(offset, offset + n))


# -- policies ---------------------------------------------------------------


def random_legal_policy() -> PolicyFn:
    """Each game gets its own `random.Random`, seeded from that game's own
    seed -- so a game's choices never depend on which other games happen to
    share its batch or on call order (stage-5-eval.md's "batching is a
    no-op" test)."""
    rngs: dict[int, random.Random] = {}

    def pick(snapshots: list[dict], teacher_infos: list[TeacherInfo]) -> list[tuple[Action | None, str | None]]:
        out = []
        for snap in snapshots:
            seed = snap["seed"]
            rng = rngs.setdefault(seed, random.Random(seed ^ 0x52414E44))
            choice = rng.choice(snap["legal"])
            out.append(((choice["rot"], choice["x"]), None))
        return out

    return pick


def teacher_policy(weights: dict | None = None) -> PolicyFn:
    """`weights=None` (the default) trusts whatever `run_rollout` itself
    already computed in `teacher_infos` for this turn -- both because it's
    the same 2-ply search either way, and to avoid the alternative footgun
    of this policy silently acting on different weights than the on-policy
    bookkeeping/labels it's compared against. Pass explicit `weights` only
    to deliberately run a policy tuned differently from the weights
    `run_rollout`'s `teacher_weights=` is judging it by."""

    def pick(snapshots: list[dict], teacher_infos: list[TeacherInfo]) -> list[tuple[Action | None, str | None]]:
        if weights is None:
            return [(action, None) for action, _values in teacher_infos]
        return [(_teacher_best(snap, weights)[0], None) for snap in snapshots]

    return pick


def _teacher_best(snap: dict, weights: dict) -> tuple[Action, dict[Action, float]]:
    """Score every legal placement once and return (best action, the score
    dict). Shared by the teacher policy and by run_rollout's on-policy
    teacher-match / value-gap bookkeeping so the 2-ply search for a given
    turn is never paid for twice. Tie-break matches `teacher.pick` exactly
    (lowest rot, then lowest x) so this is a drop-in equivalent, not an
    approximation of it."""
    board = board_to_lists(snap["board"])
    legal = snap["legal"]
    values = {(p["rot"], p["x"]): value_of_placement(board, p, snap["next"], weights) for p in legal}
    best_action = max(values, key=lambda a: (values[a], -a[0], -a[1]))
    return best_action, values


# -- rollout ------------------------------------------------------------


def run_rollout(
    seeds: list[int],
    policy_fn: PolicyFn,
    mode: str,
    cap: int = 500,
    teacher_weights: dict | None = None,
    gen_batch_size: int = 64,
    game_id_prefix: str = "eval",
    teacher_workers: int = 1,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Play every seed to death or `cap`, one lockstep round at a time.

    Returns `(game_records, diagnostics)`:

    - `game_records`: Stage 3's games.jsonl shape plus what Stage 5 adds --
      `game_id, seed, mode, actions, labels, raw_actions, incidents,
      raw_model_output, pieces, lines, score, died, death_reason`.
      `actions` is what was actually applied to the board (post-substitution
      in assisted mode); `labels` is the teacher's on-policy pick at that
      same pre-move state (not what the dataset teacher chose -- there is
      no dataset here, this is the teacher's opinion of the *model's* board);
      `raw_actions` is the policy's own parsed intent, `None` when
      unparseable; `incidents` lists the turns assisted mode substituted a
      legal move because the raw one was illegal or unparseable.
      `raw_model_output` has one entry per turn *attempted*: in strict mode,
      a death-by-illegal-action turn still gets an entry (the fatal
      output) even though that turn never makes it into `actions` -- so
      `len(raw_model_output)` can be `len(actions) + 1` there. Every other
      case (assisted mode, or a strict-mode game that topped out or hit the
      cap on its own) has the two lists the same length.
    - `diagnostics`: `game_id -> [{turn, parsed, legal, teacher_match,
      value_gap, holes_before, holes_after, holes_created, max_height,
      aggregate_height, lines_after, score_after}, ...]`, one entry per
      turn actually played, for `aggregate_metrics`.
    """
    assert mode in (STRICT, ASSISTED)
    assert gen_batch_size >= 1
    assert teacher_workers >= 1
    weights = teacher_weights if teacher_weights is not None else DEFAULT_WEIGHTS

    games = {seed: Game(seed=seed, game_id=f"{game_id_prefix}-{mode}-{seed}") for seed in seeds}
    alive = list(seeds)
    records = {
        seed: {
            "game_id": games[seed].game_id,
            "seed": seed,
            "mode": mode,
            "actions": [],
            "labels": [],
            "raw_actions": [],
            "incidents": [],
            "raw_model_output": [],
            "death_reason": None,
        }
        for seed in seeds
    }
    diagnostics: dict[str, list[dict]] = {games[seed].game_id: [] for seed in seeds}

    pool = (
        ProcessPoolExecutor(
            max_workers=teacher_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        if teacher_workers > 1
        else None
    )
    try:
        while alive:
            round_alive = list(alive)  # frozen for this round; `alive` itself shrinks below
            for i in range(0, len(round_alive), gen_batch_size):
                chunk = round_alive[i : i + gen_batch_size]
                snapshots = [games[s].snapshot() for s in chunk]
                if pool:
                    chunksize = max(1, len(snapshots) // (teacher_workers * 4))
                    teacher_infos = list(pool.map(_teacher_best, snapshots, repeat(weights), chunksize=chunksize))
                else:
                    teacher_infos = [_teacher_best(snap, weights) for snap in snapshots]
                outputs = policy_fn(snapshots, teacher_infos)
                assert len(outputs) == len(chunk), (
                    f"policy_fn returned {len(outputs)} results for {len(chunk)} snapshots"
                )

                for seed, snap, (teacher_action, values), (raw, raw_text) in zip(chunk, snapshots, teacher_infos, outputs):
                    g = games[seed]
                    rec = records[seed]
                    legal = snap["legal"]
                    legal_pairs = {(p["rot"], p["x"]) for p in legal}
                    best_value = values[teacher_action]

                    is_legal = raw is not None and raw in legal_pairs
                    incident = False
                    if is_legal:
                        taken = raw
                    elif mode == ASSISTED:
                        first = legal[0]
                        taken = (first["rot"], first["x"])
                        incident = True
                    else:
                        taken = None

                    rec["raw_model_output"].append(raw_text)

                    if taken is None:
                        rec["death_reason"] = DEATH_ILLEGAL_ACTION
                        alive.remove(seed)
                        continue

                    taken_value = values[taken]
                    holes_before = snap["holes_total"]

                    rec["actions"].append(list(taken))
                    rec["labels"].append(list(teacher_action))
                    rec["raw_actions"].append(list(raw) if raw is not None else None)
                    if incident:
                        rec["incidents"].append(snap["turn"])

                    result = g.step(*taken)

                    diagnostics[g.game_id].append(
                        {
                            "turn": snap["turn"],
                            "parsed": raw is not None,
                            "legal": is_legal,
                            "teacher_match": raw == teacher_action if raw is not None else False,
                            "value_gap": taken_value - best_value,
                            "holes_before": holes_before,
                            "holes_after": result["holes_total"],
                            "holes_created": max(0, result["holes_total"] - holes_before),
                            "max_height": result["max_height"],
                            "aggregate_height": result["aggregate_height"],
                            "lines_after": result["lines"],
                            "score_after": result["score"],
                        }
                    )

                    if g.game_over:
                        rec["death_reason"] = DEATH_TOPPED_OUT
                        alive.remove(seed)
                    elif g.turn >= cap:
                        rec["death_reason"] = CAP_REACHED
                        alive.remove(seed)
    finally:
        if pool:
            pool.shutdown()

    game_records = []
    for seed in seeds:
        g = games[seed]
        rec = records[seed]
        game_records.append(
            {
                **rec,
                "pieces": g.turn,
                "lines": g.lines,
                "score": g.score,
                "died": rec["death_reason"] in (DEATH_TOPPED_OUT, DEATH_ILLEGAL_ACTION),
            }
        )
    return game_records, diagnostics


def replay_game_log(game_record: dict, teacher_weights: dict | None = None) -> tuple[dict, list[dict]]:
    """Rebuild one game purely from its `seed` + `actions`, independent of
    everything `run_rollout` logged live. Used to prove (a) the game itself
    replays with zero drift and (b) the per-turn diagnostics in
    `aggregate_metrics` are not just whatever the live run happened to say
    (stage-5-eval.md tests #2 and #6).

    Returns `(final_state, diagnostics)` where `final_state` is
    `{pieces, lines, score, game_over}` and `diagnostics` has the same shape
    `run_rollout` produces per turn -- `raw`/`parsed`/`legal`/`incidents`
    are taken from the stored record (replay has no policy to re-ask),
    everything derived from the board (`teacher_match`, `value_gap`,
    `holes_created`, heights, ...) is recomputed from scratch.
    """
    weights = teacher_weights if teacher_weights is not None else DEFAULT_WEIGHTS
    g = Game(seed=game_record["seed"], game_id=game_record["game_id"])
    incidents = set(game_record["incidents"])
    raw_actions = game_record["raw_actions"]

    diagnostics = []
    for turn, (act_rot, act_x) in enumerate(game_record["actions"]):
        snap = g.snapshot()
        legal = snap["legal"]
        legal_pairs = {(p["rot"], p["x"]) for p in legal}
        teacher_action, values = _teacher_best(snap, weights)
        best_value = values[teacher_action]

        raw = tuple(raw_actions[turn]) if raw_actions[turn] is not None else None
        taken = (act_rot, act_x)
        assert taken in legal_pairs, f"{game_record['game_id']} turn {turn}: replayed action not legal"
        taken_value = values[taken]
        holes_before = snap["holes_total"]

        result = g.step(act_rot, act_x)
        diagnostics.append(
            {
                "turn": turn,
                "parsed": raw is not None,
                "legal": turn not in incidents,
                "teacher_match": raw == teacher_action if raw is not None else False,
                "value_gap": taken_value - best_value,
                "holes_before": holes_before,
                "holes_after": result["holes_total"],
                "holes_created": max(0, result["holes_total"] - holes_before),
                "max_height": result["max_height"],
                "aggregate_height": result["aggregate_height"],
                "lines_after": result["lines"],
                "score_after": result["score"],
            }
        )

    final_state = {"pieces": g.turn, "lines": g.lines, "score": g.score, "game_over": g.game_over}
    return final_state, diagnostics


# -- metrics ------------------------------------------------------------


def mean_se(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "se": 0.0, "n": 0}
    m = sum(values) / n
    if n < 2:
        return {"mean": m, "se": 0.0, "n": n}
    variance = sum((v - m) ** 2 for v in values) / (n - 1)
    return {"mean": m, "se": math.sqrt(variance / n), "n": n}


def aggregate_metrics(game_records: list[dict], diagnostics: dict[str, list[dict]]) -> dict:
    """One policy/mode's full metric table (stage-5-eval.md's "Metrics"
    section), mean +/- standard error over games -- each of the N seeds is
    one independent sample, matching "differences smaller than a couple of
    lines are noise" at N=100."""
    n = len(game_records)
    lines = [r["lines"] for r in game_records]
    pieces = [r["pieces"] for r in game_records]
    scores = [r["score"] for r in game_records]
    deaths = sum(r["died"] for r in game_records)
    cap_outs = sum(r["death_reason"] == CAP_REACHED for r in game_records)
    illegal_deaths = sum(r["death_reason"] == DEATH_ILLEGAL_ACTION for r in game_records)
    topped_out = sum(r["death_reason"] == DEATH_TOPPED_OUT for r in game_records)

    def per_game_rate(pred) -> list[float]:
        out = []
        for r in game_records:
            turns = diagnostics.get(r["game_id"], [])
            out.append(sum(pred(t) for t in turns) / len(turns) if turns else 0.0)
        return out

    def per_game_mean(key: str) -> list[float]:
        out = []
        for r in game_records:
            turns = diagnostics.get(r["game_id"], [])
            out.append(sum(t[key] for t in turns) / len(turns) if turns else 0.0)
        return out

    def per_game_final(key: str) -> list[float]:
        out = []
        for r in game_records:
            turns = diagnostics.get(r["game_id"], [])
            out.append(turns[-1][key] if turns else 0.0)
        return out

    return {
        "n_games": n,
        "lines": {**mean_se(lines), "median": statistics.median(lines) if lines else 0, "max": max(lines) if lines else 0, "distribution": lines},
        "pieces_survived": mean_se(pieces),
        "score": mean_se(scores),
        "deaths": deaths,
        "cap_outs": cap_outs,
        "illegal_action_deaths": illegal_deaths,
        "topped_out_deaths": topped_out,
        "holes_created_per_piece": mean_se(per_game_mean("holes_created")),
        "max_height": mean_se(per_game_final("max_height")),
        "mean_aggregate_height": mean_se(per_game_mean("aggregate_height")),
        "teacher_match_rate": mean_se(per_game_rate(lambda t: t["teacher_match"])),
        "value_gap": mean_se(per_game_mean("value_gap")),
        "parse_failure_rate": mean_se(per_game_rate(lambda t: not t["parsed"])),
        "illegal_rate": mean_se(per_game_rate(lambda t: t["parsed"] and not t["legal"])),
    }
