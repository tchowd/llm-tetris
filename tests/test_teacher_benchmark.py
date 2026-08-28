import random

from tetris.engine import Game
from tetris import teacher

# 2-ply search is expensive (see plan/stage-2-teacher.md "Performance"), so
# this benchmark caps games well short of what the teacher can actually do
# (a manual 600-step run cleared 230+ lines without dying) to keep the test
# suite fast. Even at this small cap the gap over random must be enormous.
STEPS_PER_GAME = 120
NUM_SEEDS = 6


def _play_teacher(seed):
    g = Game(seed=seed)
    for _ in range(STEPS_PER_GAME):
        if g.game_over:
            break
        snap = g.snapshot()
        rot, x = teacher.pick(snap, snap["legal"])
        g.step(rot, x)
    return g.lines, g.game_over


def _play_random(seed):
    g = Game(seed=seed)
    agent_rng = random.Random(seed + 999)
    for _ in range(STEPS_PER_GAME):
        if g.game_over:
            break
        p = agent_rng.choice(g.legal_placements())
        g.step(p["rot"], p["x"])
    return g.lines, g.game_over


def test_teacher_clears_far_more_lines_than_random_and_rarely_dies():
    teacher_results = [_play_teacher(seed) for seed in range(NUM_SEEDS)]
    random_results = [_play_random(seed) for seed in range(NUM_SEEDS)]

    teacher_lines = [lines for lines, _ in teacher_results]
    teacher_deaths = sum(died for _, died in teacher_results)
    random_lines = [lines for lines, _ in random_results]

    teacher_mean = sum(teacher_lines) / len(teacher_lines)
    random_mean = sum(random_lines) / len(random_lines)

    # PLAN.md's bar: "hundreds of lines" is the target once games run long;
    # at this short, CI-friendly cap the requirement is just "a different
    # league, not a marginal improvement" over random.
    assert teacher_mean >= 15, f"teacher mean lines too low: {teacher_mean}"
    assert random_mean <= 3, f"random mean lines unexpectedly high: {random_mean}"
    assert teacher_mean >= 5 * max(random_mean, 1)
    assert teacher_deaths == 0, "teacher should not top out within so few pieces"
