# Check likelihood ratios, channel routing, KL, entropy, and gradient propagation.

import math

import pytest
import torch

from dualcredit.objective import (
    answer_channel_term,
    channel_rows,
    dual_channel_loss,
    entropy_term,
    gather_logprob,
    kl_k3,
    response_channel_term,
)
from dualcredit.rollout import SLOT_ANSWER, SLOT_CALL, SLOT_DECISION, PositionRecord


def test_gather_logprob_selects_the_emitted_token():
    logits = torch.tensor([[0.0, 10.0], [0.0, 10.0]])
    values = gather_logprob(logits, torch.tensor([0, 1]), torch.tensor([0, 1]))
    assert values.shape == torch.Size([2])
    assert float(values[0]) < float(values[1])
    assert float(values[1]) == pytest.approx(-math.exp(-10.0), abs=1e-6)
    empty = gather_logprob(logits, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    assert empty.numel() == 0


def test_answer_channel_term_is_negative_for_positive_advantage():
    logp = torch.tensor([-1.0, -2.0])
    terms = answer_channel_term(logp, logp, torch.tensor([1.5, 0.5]), 0.2)
    assert torch.all(terms < 0)
    assert terms[0] == pytest.approx(-1.5)
    assert terms[1] == pytest.approx(-0.5)


def test_answer_channel_term_keeps_pessimistic_branch():
    old = torch.tensor([0.0])
    new = torch.tensor([1.0])
    positive = answer_channel_term(new, old, torch.tensor([1.0]), 0.2)
    assert positive[0] == pytest.approx(-1.2)
    negative = answer_channel_term(new, old, torch.tensor([-1.0]), 0.2)
    assert negative[0] == pytest.approx(math.exp(1.0))


def test_answer_channel_term_clips_large_ratios():
    old = torch.tensor([0.0])
    new = torch.tensor([3.0])
    clipped = answer_channel_term(new, old, torch.tensor([2.0]), 0.2)
    unclipped = torch.exp(new - old) * 2.0
    assert clipped[0] == pytest.approx(-1.2 * 2.0)
    assert abs(float(clipped[0])) < float(unclipped[0])


def test_response_channel_term_is_unclipped_by_default():
    old = torch.tensor([0.0])
    new = torch.tensor([3.0])
    advantage = torch.tensor([1.0])
    default = response_channel_term(new, old, advantage)
    bounded = response_channel_term(new, old, advantage, 0.2)
    assert default[0] == pytest.approx(-math.exp(3.0))
    assert bounded[0] == pytest.approx(-1.2)
    assert abs(float(default[0])) > abs(float(bounded[0]))


def test_kl_k3_is_non_negative_and_zero_at_identity():
    logp = torch.tensor([-0.4, -1.7, -3.1])
    assert torch.allclose(kl_k3(logp, logp), torch.zeros(3))
    for left, right in ((logp, torch.zeros(3)), (torch.zeros(3), logp)):
        assert torch.all(kl_k3(left, right) >= 0.0)


def test_entropy_term_bounds():
    uniform = entropy_term(torch.zeros(1, 3))
    peaked = entropy_term(torch.tensor([[20.0, 0.0, 0.0]]))
    spread = entropy_term(torch.tensor([[0.0, 1.0, 2.0]]))
    assert float(uniform[0]) == pytest.approx(math.log(3.0))
    assert float(peaked[0]) < 1e-3
    assert float(spread[0]) < float(uniform[0])
    assert bool(torch.all(entropy_term(torch.randn(6, 5)) >= 0.0))


def test_channel_rows_splits_decision_positions():
    positions = [
        PositionRecord(0, SLOT_CALL, 0, 0, 0.0, 0),
        PositionRecord(1, SLOT_DECISION, 0, 1, 0.0, 0),
        PositionRecord(2, SLOT_ANSWER, 0, 5, 0.0),
    ]
    mask = torch.tensor([False, True, False])
    answer_rows, response_rows, call_rows, decision_rows, answer_slots = channel_rows(positions, mask)
    assert answer_rows.tolist() == [0, 2]
    assert response_rows.tolist() == [1]
    assert call_rows.tolist() == [0]
    assert decision_rows.tolist() == [1]
    assert answer_slots.tolist() == [2]


def test_dual_channel_loss_reports_and_backpropagates():
    positions = [
        PositionRecord(0, SLOT_CALL, 0, 0, -0.7, 0),
        PositionRecord(1, SLOT_DECISION, 0, 1, -0.9, 0),
        PositionRecord(2, SLOT_ANSWER, 0, 5, -1.4),
    ]
    mask = torch.tensor([False, True, False])
    advantages = torch.tensor([0.4, 1.2, -0.3])
    new_call = torch.zeros(3, 2, requires_grad=True)
    new_decision = torch.zeros(3, 3, requires_grad=True)
    new_answer = torch.zeros(3, 12, requires_grad=True)
    ref_call = torch.zeros(3, 2)
    ref_decision = torch.zeros(3, 3)
    ref_answer = torch.zeros(3, 12)
    report = dual_channel_loss(
        new_call, new_decision, new_answer,
        ref_call, ref_decision, ref_answer,
        positions, advantages, mask,
        0.2, 0.02, 0.2, 0.5,
    )
    for key in ("loss", "answer_loss", "response_loss", "kl", "entropy", "new_logp", "old_logp", "answer_rows", "response_rows", "shift"):
        assert key in report
    assert torch.isfinite(report["loss"])
    assert report["answer_rows"].tolist() == [0, 2]
    assert report["response_rows"].tolist() == [1]
    assert float(report["old_logp"][1]) == pytest.approx(-0.9)
    report["loss"].backward()
    assert new_decision.grad is not None
    assert new_answer.grad is not None
    assert torch.isfinite(new_decision.grad).all()
