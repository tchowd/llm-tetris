import json

from scripts.build_tetra17_curriculum import reservoir_ordinary
from tetris.dataset import generate_game


def test_ordinary_reservoir_is_clean_bounded_and_deterministic(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for seed in range(10, 15):
        generated, _ = generate_game(seed, max_pieces=8, noisy=False)
        for row in generated:
            row["split"] = "train"
        rows.extend(generated)
    (source / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    first = reservoir_ordinary([source], 10, 99, per_game_cap=3)
    second = reservoir_ordinary([source], 10, 99, per_game_cap=3)
    assert [row["state_hash"] for row in first] == [row["state_hash"] for row in second]
    assert len(first) == 10
    assert len({row["state_hash"] for row in first}) == 10
    assert all(row["kind"] == "ordinary_retention" and row["source_explored"] is False for row in first)
    assert max(sum(row["source_game_id"] == game for row in first) for game in {r["source_game_id"] for r in first}) <= 3


def test_ordinary_reservoir_excludes_states_and_deduplicates_sources(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    generated, _ = generate_game(22, max_pieces=8, noisy=False)
    for row in generated:
        row["split"] = "train"
    duplicate = dict(generated[0])
    duplicate["game_id"] = "duplicate-copy"
    rows = generated + [duplicate]
    (source / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    baseline = reservoir_ordinary([source], 6, 17, per_game_cap=8)
    excluded = {baseline[0]["state_hash"]}
    result = reservoir_ordinary([source], 6, 17, per_game_cap=8, excluded_states=excluded)

    assert len(result) == 6
    assert excluded.isdisjoint(row["state_hash"] for row in result)
    assert len({row["state_hash"] for row in result}) == 6
