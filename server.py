"""Thin HTTP wrap of tetris.engine.Game. No game logic lives here.

Serves the built frontend from web/dist (run `npm run build` in web/ first).
For frontend hot-reload during development, run this server plus
`npm run dev` in web/ instead, which proxies /api to this process.
"""
from __future__ import annotations

import random

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tetris.engine import Game
from tetris import teacher
from tetris import dashboard

app = FastAPI(title="llm-tetris")

GAMES: dict[str, Game] = {}


class NewGameRequest(BaseModel):
    seed: int | None = None


class StepRequest(BaseModel):
    rot: int
    x: int


def _get_game(game_id: str) -> Game:
    game = GAMES.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"no game with id {game_id!r}")
    return game


@app.get("/api/healthz")
def healthz():
    return {"status": "ok", "games": len(GAMES)}


@app.post("/api/games")
def new_game(req: NewGameRequest):
    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    game = Game(seed=seed)
    GAMES[game.game_id] = game
    return game.snapshot()


@app.get("/api/games")
def list_games():
    return [
        {
            "game_id": game.game_id,
            "seed": game.seed,
            "turn": game.turn,
            "score": game.score,
            "lines": game.lines,
            "game_over": game.game_over,
        }
        for game in GAMES.values()
    ]


@app.get("/api/games/{game_id}")
def get_game(game_id: str):
    return _get_game(game_id).snapshot()


@app.post("/api/games/{game_id}/step")
def step(game_id: str, req: StepRequest):
    game = _get_game(game_id)
    try:
        return game.step(req.rot, req.x)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/games/{game_id}/teacher-step")
def teacher_step(game_id: str):
    """Let Stage 2's Dellacherie/El-Tetris teacher play one move, for
    spot-checking teacher games from the browser UI."""
    game = _get_game(game_id)
    if game.game_over:
        raise HTTPException(status_code=400, detail="game is over")
    snap = game.snapshot()
    rot, x = teacher.pick(snap, snap["legal"])
    result = game.step(rot, x)
    return {**result, "teacher_action": {"rot": rot, "x": x}}


# -- Read-only project operations dashboard ---------------------------------


@app.get("/api/dashboard/summary")
def dashboard_summary(include_aws: bool = True):
    return dashboard.dashboard_summary(include_aws=include_aws)


@app.get("/api/dashboard/stages")
def dashboard_stages():
    snapshot = dashboard.local_snapshot()
    return dashboard.envelope(snapshot["stages"])


@app.get("/api/dashboard/stages/{stage}")
def dashboard_stage(stage: int):
    if stage < 1 or stage > 7:
        raise HTTPException(status_code=404, detail="stage must be between 1 and 7")
    summary = dashboard.dashboard_summary()
    snapshot = summary["data"]
    item = next(value for value in snapshot["stages"] if value["number"] == stage)
    stage_runs = [value for value in snapshot["runs"] if value["stage"] == stage]
    active_job = snapshot["project"].get("active_job")
    if active_job and active_job.get("stage") == stage and active_job.get("run_id") not in {run["run_id"] for run in stage_runs}:
        stage_runs.insert(0, active_job)
    data = {
        **item,
        "issues": [value for value in snapshot["issues"] if value["stage"] == stage],
        "runs": stage_runs,
        "datasets": snapshot["datasets"]["batches"] if stage == 3 else [],
        "thresholds": dashboard.THRESHOLDS,
    }
    return dashboard.envelope(data, errors=summary["errors"])


@app.get("/api/dashboard/issues")
def dashboard_issues(
    severity: str | None = None,
    stage: int | None = None,
    source: str | None = None,
    include_aws: bool = True,
):
    payload = dashboard.dashboard_summary(include_aws=include_aws)
    items = payload["data"]["issues"]
    if severity:
        items = [item for item in items if item["severity"] == severity]
    if stage:
        items = [item for item in items if item["stage"] == stage]
    if source:
        items = [item for item in items if item["source"] == source]
    return dashboard.envelope(items, errors=payload["errors"])


@app.get("/api/dashboard/runs")
def dashboard_runs(stage: int | None = None, status: str | None = None, backend: str | None = None):
    runs = dashboard.discover_runs()
    jobs = dashboard.aws_jobs()
    local_ids = {run["run_id"] for run in runs}
    runs.extend(dashboard.cloud_run(job) for job in jobs["jobs"] if job.get("run_id") not in local_ids)
    runs.sort(key=lambda run: run.get("updated_at") or "", reverse=True)
    if stage:
        runs = [run for run in runs if run["stage"] == stage]
    if status:
        runs = [run for run in runs if run["status"] == status]
    if backend:
        runs = [run for run in runs if run.get("backend") == backend]
    return dashboard.envelope(runs, errors=jobs["errors"])


@app.get("/api/dashboard/runs/{run_id}")
def dashboard_run(run_id: str):
    matches = [run for run in dashboard.discover_runs() if run["run_id"] == run_id]
    if not matches:
        raise HTTPException(status_code=404, detail=f"no run with id {run_id!r}")
    return dashboard.envelope(matches[0])


@app.get("/api/dashboard/datasets")
def dashboard_datasets():
    data = dashboard.discover_datasets()
    return dashboard.envelope(data, errors=data["errors"])


@app.get("/api/dashboard/aws/resources")
def dashboard_aws_resources():
    payload = dashboard.aws_resources()
    return dashboard.envelope(payload["resources"], errors=payload["errors"])


@app.get("/api/dashboard/aws/jobs")
def dashboard_aws_jobs():
    payload = dashboard.aws_jobs()
    return dashboard.envelope(payload["jobs"], errors=payload["errors"])


@app.get("/api/dashboard/aws/metrics")
def dashboard_aws_metrics(hours: int = Query(default=1, ge=1, le=24)):
    payload = dashboard.aws_metrics(hours=hours)
    return dashboard.envelope(payload["series"], errors=payload["errors"])


@app.get("/api/dashboard/aws/logs")
def dashboard_aws_logs(limit: int = Query(default=100, ge=1, le=1000)):
    payload = dashboard.aws_logs(limit=limit)
    return dashboard.envelope(payload["events"], errors=payload["errors"])


@app.get("/api/dashboard/aws/costs")
def dashboard_aws_costs():
    payload = dashboard.aws_costs()
    return dashboard.envelope({key: payload[key] for key in ("actual", "forecast", "live")}, errors=payload["errors"])


@app.get("/api/dashboard/aws/credits")
def dashboard_aws_credits():
    payload = dashboard.aws_credits()
    return dashboard.envelope(payload["credits"], errors=payload["errors"])


@app.get("/api/dashboard/aws/quotas")
def dashboard_aws_quotas():
    payload = dashboard.aws_quotas()
    return dashboard.envelope(payload["quotas"], errors=payload["errors"])


@app.get("/api/dashboard/aws/alarms")
def dashboard_aws_alarms():
    payload = dashboard.aws_alarms()
    return dashboard.envelope(payload["alarms"], errors=payload["errors"])


@app.get("/api/dashboard/aws/security")
def dashboard_aws_security():
    payload = dashboard.aws_security()
    return dashboard.envelope({key: payload[key] for key in ("principal", "policies", "warnings")}, errors=payload["errors"])


@app.get("/api/dashboard/replays")
def dashboard_replays():
    items = []
    for replay_id, entry in dashboard.replay_index().items():
        record = entry["record"]
        items.append(
            {
                "replay_id": replay_id,
                "source": entry["source"],
                **{key: record.get(key) for key in ("game_id", "seed", "policy", "mode", "pieces", "lines", "score", "died", "death_reason")},
            }
        )
    return dashboard.envelope(items)


@app.get("/api/dashboard/replays/{replay_id}")
def dashboard_replay(replay_id: str, turn: int | None = Query(default=None, ge=0)):
    value = dashboard.replay_snapshot(replay_id, turn=turn)
    if value is None:
        raise HTTPException(status_code=404, detail=f"no replay with id {replay_id!r}")
    return dashboard.envelope(value)


app.mount("/assets", StaticFiles(directory="web/dist/assets", check_dir=False), name="assets")


@app.get("/{full_path:path}")
def frontend_spa(full_path: str):
    """Serve real build files when present, otherwise the SPA entrypoint.

    The fallback is what makes direct loads of /dashboard/stages/4 and
    replay links work from the production FastAPI process.
    """
    dist = dashboard.ROOT / "web/dist"
    candidate = (dist / full_path).resolve()
    if candidate.is_relative_to(dist.resolve()) and candidate.is_file():
        return FileResponse(candidate)
    index = dist / "index.html"
    if not index.exists():
        raise HTTPException(status_code=503, detail="frontend is not built; run npm run build in web/")
    return FileResponse(index)
