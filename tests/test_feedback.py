import math

import pytest

from tetris.rl import advantage_diagnostics, episode_transition, EpisodeRewardWeights, trajectory_advantages


def test_lone_survivor_and_identical_failures():
    rewards = [[-10], [0, 0, -10]]
    old = trajectory_advantages(rewards)
    assert old[1][1:] == [0, 0]
    revised = trajectory_advantages(rewards, method="fixed_zero")
    assert revised[1] == pytest.approx([-.9801, -.99, -1])
    assert trajectory_advantages([[-10]] * 4) == [[0]] * 4
    assert trajectory_advantages([[-10]] * 4, method="fixed_zero") == [[-1]] * 4


def test_delayed_rewards_variable_lengths_and_no_failure_sign_override():
    rewards = [[], [0, 0, 10], [20, -10], [0], [2]]
    actual = trajectory_advantages(rewards, method="fixed_zero")
    assert actual[0] == []
    assert actual[1] == pytest.approx([.9801, .99, 1])
    assert actual[2] == pytest.approx([1.01, -1])
    assert actual[3:] == [[0], [.2]]
    assert trajectory_advantages([[2]] * 4, method="fixed_zero") == [[.2]] * 4
    assert trajectory_advantages([], method="fixed_zero") == []


def test_illegal_terminal_uses_actual_reward():
    from tetris.engine import Game
    for action in (None, (99, 99)):
        result = episode_transition(Game(12), action, EpisodeRewardWeights())
        assert result.terminal and result.terminal_reason == "illegal_action"
        assert trajectory_advantages([[result.reward]], method="fixed_zero") == [[-1]]


@pytest.mark.parametrize("kwargs", [{"method": "oops"}, {"reward_scale": 0}, {"reward_scale": math.nan}, {"gamma": 2}])
def test_bad_estimator_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        trajectory_advantages([[1]], **kwargs)


def test_nonfinite_rewards_rejected():
    with pytest.raises(ValueError):
        trajectory_advantages([[math.inf]])


def test_diagnostic_denominators_and_threshold():
    trajectories = [{"steps": [{"advantage": -1, "terminal_reason": "illegal_action"}]},
        {"steps": [{"advantage": 1e-11}, {"advantage": 0, "terminal_reason": "illegal_action"}]}]
    d = advantage_diagnostics(trajectories)
    assert d["all"]["count"] == 3
    assert d["all"]["exact_zero"] == 1
    assert d["all"]["effective_zero"] == 2
    assert d["terminal_illegal"]["count"] == 2
    assert d["single_active_illegal"]["effective_zero"] == 1
    assert d["terminal_topout"]["mean_abs"] is None


@pytest.mark.parametrize("advantage", [-1., 1., 0.])
def test_actual_loss_gradient_direction_at_frozen_reference(advantage):
    torch = pytest.importorskip("torch")
    from scripts.train_episode_rl import action_loss_chunk
    logits = torch.tensor([.3, -.2], requires_grad=True)
    before = logits.log_softmax(0)[0].detach().item()
    current = logits.log_softmax(0)[0:1]
    rows = [{"advantage": advantage, "reference_token_logprobs": [before]}]
    loss, _ = action_loss_chunk([current], rows, .05, 1)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    after = (logits.detach() - .1 * logits.grad).log_softmax(0)[0].item()
    if advantage < 0:
        assert after < before
    elif advantage > 0:
        assert after > before
    else:
        assert after == before


def test_chunking_and_variable_token_lengths_preserve_decision_weights():
    torch = pytest.importorskip("torch")
    from scripts.train_episode_rl import action_loss_chunk
    tokens = [torch.tensor([-1.] * n, requires_grad=True) for n in (1, 3, 2)]
    rows = [{"advantage": a, "reference_token_logprobs": [-1.] * len(t)} for a, t in zip((1., -1., 2.), tokens)]
    loss, _ = action_loss_chunk(tokens, rows, .05, 3)
    loss.backward()
    gradients = [t.grad.clone() for t in tokens]
    for t in tokens:
        t.grad = None
    for offset in (0, 2):
        part, _ = action_loss_chunk(tokens[offset:offset+2], rows[offset:offset+2], .05, 3)
        part.backward()
    for a, t, grad in zip((1., -1., 2.), tokens, gradients):
        torch.testing.assert_close(t.grad, grad, rtol=0, atol=0)
        torch.testing.assert_close(grad, torch.full_like(t, -a / (3 * len(t))))


def test_zero_coefficient_still_has_reference_regularization():
    torch = pytest.importorskip("torch")
    from scripts.train_episode_rl import action_loss_chunk
    current = torch.tensor([-.5], requires_grad=True)
    loss, _ = action_loss_chunk([current], [{"advantage": 0, "reference_token_logprobs": [-1.]}], .05, 1)
    loss.backward()
    assert current.grad.item() > 0
