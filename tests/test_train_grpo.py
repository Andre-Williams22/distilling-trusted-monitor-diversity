"""GRPO's reward, advantage and loss maths. Needs torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="pip install torch to test GRPO")


def test_reward_is_agreement_with_the_consensus_not_the_label():
    """The reward never reads ground truth -- only the debate's majority."""
    from src.train_grpo import verdict_reward

    answer = "REASONING: x\nBACKDOORED: yes\nSUSPICION: 8"
    assert verdict_reward(answer, "yes") == 1.0
    assert verdict_reward(answer, "no") == 0.0


def test_unparseable_answers_score_zero():
    """A malformed answer earns nothing, so format is pushed toward the contract."""
    from src.train_grpo import verdict_reward

    assert verdict_reward("I cannot answer that", "yes") == 0.0


def test_unanimous_groups_produce_no_gradient():
    """GRPO learns from disagreement within a group; agreement teaches nothing."""
    from src.train_grpo import group_advantages

    assert group_advantages([1.0, 1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0, 0.0]
    assert group_advantages([0.0, 0.0]) == [0.0, 0.0]


def test_split_groups_rank_correct_answers_above_wrong_ones():
    """A rewarded sample gets positive advantage, an unrewarded one negative."""
    from src.train_grpo import group_advantages

    advantages = group_advantages([1.0, 0.0])
    assert advantages[0] > 0 > advantages[1]
    assert sum(advantages) == pytest.approx(0.0)


def test_positive_advantage_pushes_probability_up():
    """Raising the policy on a good sample lowers the loss."""
    from src.train_grpo import grpo_loss

    old = torch.full((1, 3), -1.0)
    mask = torch.ones(1, 3)
    advantages = torch.tensor([1.0])
    better, _ = grpo_loss(old + 0.1, old, old, advantages, mask, 0.2, 0.0)
    worse, _ = grpo_loss(old - 0.1, old, old, advantages, mask, 0.2, 0.0)
    assert better.item() < worse.item()


def test_clipping_caps_how_far_one_step_can_move():
    """Beyond the clip range the objective stops rewarding further movement."""
    from src.train_grpo import grpo_loss

    old = torch.full((1, 3), -1.0)
    mask = torch.ones(1, 3)
    advantages = torch.tensor([1.0])
    inside, parts_inside = grpo_loss(old + 0.05, old, old, advantages, mask, 0.2, 0.0)
    outside, parts_outside = grpo_loss(old + 2.0, old, old, advantages, mask, 0.2, 0.0)
    assert parts_inside["clipped_fraction"] == 0.0
    assert parts_outside["clipped_fraction"] == 1.0
    # Clipped at exactly -(1 + epsilon) * A, not the unbounded ratio.
    assert outside.item() == pytest.approx(-1.2, abs=1e-5)


def test_kl_penalty_is_zero_at_the_reference_and_positive_away_from_it():
    """The k3 estimator is non-negative and vanishes when policy == reference."""
    from src.train_grpo import grpo_loss

    logps = torch.full((1, 3), -1.0)
    mask = torch.ones(1, 3)
    advantages = torch.zeros(1)
    _, same = grpo_loss(logps, logps, logps, advantages, mask, 0.2, 1.0)
    _, drifted = grpo_loss(logps, logps, logps - 1.0, advantages, mask, 0.2, 1.0)
    assert same["kl"] == pytest.approx(0.0, abs=1e-6)
    assert drifted["kl"] > 0


def test_padding_is_excluded_from_the_loss():
    """Masked positions contribute nothing, so padding cannot move the policy."""
    from src.train_grpo import grpo_loss

    old = torch.full((1, 4), -1.0)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    policy = old.clone()
    policy[0, 2:] = 5.0                      # nonsense in the padded positions
    advantages = torch.tensor([1.0])
    loss, parts = grpo_loss(policy, old, old, advantages, mask, 0.2, 1.0)
    assert parts["kl"] == pytest.approx(0.0, abs=1e-6)
    assert loss.item() == pytest.approx(-1.0)
