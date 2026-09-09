from tetris.curriculum import difficulty_band, enriched_row, teacher_diagnostics
from tetris.engine import Game


def test_enriched_row_has_replayable_identity_and_legal_teacher_label():
    snapshot = Game(91).snapshot()
    row = enriched_row(snapshot, split="train", kind="progressive", source_id="fixture")
    assert row["difficulty_band"] == "low"
    assert [row["rot"], row["x"]] in row["legal_actions"]
    assert row["teacher_values"][f"{row['rot']},{row['x']}"] == max(row["teacher_values"].values())
    assert len(row["state_hash"]) == len(row["prompt_hash"]) == len(row["board_hash"]) == 64


def test_teacher_ties_use_deterministic_canonical_action():
    snapshot = Game(12).snapshot()
    first = teacher_diagnostics(snapshot)
    second = teacher_diagnostics(snapshot)
    assert first == second
    assert list(first["label"]) in first["near_tied_best"]


def test_curriculum_difficulty_boundaries():
    assert difficulty_band(7, 0) == "low"
    assert difficulty_band(8, 0) == "moderate"
    assert difficulty_band(12, 3) == "hard"
    assert difficulty_band(15, 6) == "critical"
