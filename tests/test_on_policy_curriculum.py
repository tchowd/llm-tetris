from scripts.collect_tetra17_on_policy import select_rows


def test_selection_prioritizes_failures_and_is_deterministic():
    rows = []
    for index, category in enumerate(("matched_control", "pre_failure", "terminal_illegal", "high_regret")):
        rows.append({"state_hash": str(index), "correction_category": category})
    first = select_rows(rows, 2, 7)
    second = select_rows(rows, 2, 7)
    assert first == second
    assert {row["correction_category"] for row in first} == {"terminal_illegal", "pre_failure"}
