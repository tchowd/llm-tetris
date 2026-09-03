from scripts.eval_closed_loop import assisted_copy


def test_assisted_copy_relabels_legal_policy_results_without_mutating_strict():
    strict_records = [{"game_id": "teacher-strict-7", "mode": "strict", "actions": [[0, 1]]}]
    strict_diagnostics = {"teacher-strict-7": [{"turn": 0, "legal": True}]}

    assisted_records, assisted_diagnostics = assisted_copy(strict_records, strict_diagnostics)

    assert strict_records[0]["game_id"] == "teacher-strict-7"
    assert assisted_records[0]["game_id"] == "teacher-assisted-7"
    assert assisted_records[0]["mode"] == "assisted"
    assert assisted_diagnostics == {"teacher-assisted-7": [{"turn": 0, "legal": True}]}
    assert assisted_diagnostics["teacher-assisted-7"] is not strict_diagnostics["teacher-strict-7"]
