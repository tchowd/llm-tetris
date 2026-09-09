from scripts.analyze_tetra17_sft import paired_interval, summary


def game(reason, pieces=10):
    return {"death_reason": reason, "pieces": pieces, "lines": pieces // 4, "score": pieces * 10}


def test_paired_recovery_effect_uses_same_states():
    control = [game("illegal_action"), game("cap_reached")]
    candidate = [game("cap_reached"), game("cap_reached")]
    result = paired_interval(control, candidate, replicates=1000)
    assert result["mean"] == .5
    assert result["paired_effects"] == [1, 0]


def test_summary_separates_illegal_topout_and_cap():
    result = summary([game("illegal_action"), game("topped_out"), game("cap_reached")])
    assert result["illegal"] == result["topout"] == result["cap_reached"] == 1
