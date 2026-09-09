import inspect


def test_direction_measurements_preserve_singleton_batch_context(monkeypatch):
    import scripts.check_episode_proof as checker
    from infra.feedback_gpu_proof_v2 import isolated_token_logprobs
    calls = []

    def independent(model, rows, pad_id, temperature):
        calls.append((model, list(rows), pad_id, temperature))
        # A batch-sensitive implementation exposes the original confound.
        return [row['value'] + len(rows) for row in rows]

    monkeypatch.setattr(checker, 'independent_token_logprobs', independent)
    rows = [{'value': 3}, {'value': 7}]
    assert isolated_token_logprobs('model', rows, 0, 1) == [4, 8]
    assert calls == [('model', [rows[0]], 0, 1), ('model', [rows[1]], 0, 1)]


def test_training_and_final_verification_are_unchanged():
    import infra.feedback_gpu_proof as original
    import infra.feedback_gpu_proof_v2 as repaired
    for name in ('train', 'verify', 'load_models', 'coefficients', 'nested_equal'):
        assert inspect.getsource(getattr(original, name)) == inspect.getsource(getattr(repaired, name))
