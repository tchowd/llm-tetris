"""Thin HTTP wrap of tetris.engine.Game. No game logic lives here.

Serves the built frontend from web/dist (run `npm run build` in web/ first).
For frontend hot-reload during development, run this server plus
`npm run dev` in web/ instead, which proxies /api to this process.
"""
from __future__ import annotations

import random

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tetris.engine import Game

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


app.mount("/", StaticFiles(directory="web/dist", html=True), name="web")
