from scripts.audit_tetra17_data import analyze_rows, difficulty_band


def row(game, split="train", board=None, completion="Action: rot=0 x=0", rot=0, x=0):
    return {
        "game_id": game, "seed": int(game), "turn": 0, "prompt": "same",
        "completion": completion, "rot": rot, "x": x, "piece": "O", "next": "I",
        "board": board or [".........."] * 20, "max_height": 0, "holes_total": 0,
        "split": split,
    }


def test_difficulty_bands_are_progressive_and_exhaustive():
    assert difficulty_band(0, 0) == "low"
    assert difficulty_band(8, 0) == "moderate"
    assert difficulty_band(13, 0) == "hard"
    assert difficulty_band(4, 6) == "critical"


def test_audit_detects_duplicates_aliases_conflicts_and_split_leakage():
    empty = [".........."] * 20
    raised = [".........."] * 19 + ["OO........"]
    rows = [
        row("1", board=empty),
        row("1", board=empty),
        row("2", board=raised),
        row("2", board=raised, completion="Action: rot=0 x=1", x=1),
        row("1", split="eval", board=empty),
    ]
    result = analyze_rows(rows)
    assert result["counts"]["exact_duplicate_rows"] == 1
    assert result["counts"]["prompt_geometry_aliases"] >= 1
    assert result["counts"]["prompt_label_conflicts"] >= 1
    assert result["counts"]["state_label_conflicts"] >= 1
    assert result["split_overlap"]["eval:train"] == {"game_ids": 1, "seeds": 1}


def test_audit_flags_bad_completion_and_missing_required_field():
    bad = row("3", completion="nonsense")
    incomplete = row("4")
    del incomplete["x"]
    result = analyze_rows([bad, incomplete])
    assert result["invalid"]["unparseable_completion"] == 1
    assert result["missing_required_fields"]["x"] == 1
