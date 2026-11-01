# Check quadrant conventions, normalization fallbacks, and response credit baselines.

import math
from types import SimpleNamespace

import pytest

from dualcredit.arbitration import mean_match
from dualcredit.config import Config, arm_config
from dualcredit.credit import (
    ALL_ACTIONS,
    EPSILON,
    RunningMatchBaseline,
    answer_advantages,
    cell_index,
    decided_indicator,
    group_statistics,
    match_indicator,
    necessity_indicator,
    normative_actions,
    observable_stratum,
    response_advantage,
    terminal_reward,
)


def test_terminal_reward_penalises_calls():
    assert terminal_reward(1, 0, 0.04) == pytest.approx(1.0)
    assert terminal_reward(1, 3, 0.04) == pytest.approx(1.0 - 0.12)
    assert terminal_reward(1, 3, 0.04) < terminal_reward(1, 1, 0.04)
    assert terminal_reward(0, 0, 0.04) == pytest.approx(0.0)


def test_necessity_indicator():
    assert necessity_indicator(1) == 0
    assert necessity_indicator(0) == 1


def test_cell_index_matches_quadrants():
    assert cell_index(0, 0) == 0
    assert cell_index(1, 0) == 1
    assert cell_index(0, 1) == 2
    assert cell_index(1, 1) == 3


def test_normative_actions_per_cell():
    assert normative_actions(0, 2) == (0,)
    assert normative_actions(1, 2) == (2,)
    assert normative_actions(1, 0) == ALL_ACTIONS
    assert normative_actions(2, 2) == ALL_ACTIONS
    assert normative_actions(3, 0) == (1,)


def test_match_indicator_requires_unique_target():
    assert match_indicator(0, 0, 2) == 1.0
    assert match_indicator(1, 0, 2) == 0.0
    assert match_indicator(2, 1, 2) == 1.0
    assert match_indicator(2, 1, 0) == 0.0
    assert match_indicator(0, 2, 2) == 0.0
    assert match_indicator(1, 3, 0) == 1.0
    assert match_indicator(1, None, 2) == 0.0


def test_decided_indicator():
    assert decided_indicator(0, 2) == 1
    assert decided_indicator(3, 0) == 1
    assert decided_indicator(1, 2) == 1
    assert decided_indicator(1, 0) == 0
    assert decided_indicator(2, 2) == 0
    assert decided_indicator(None, 2) == 0


def test_observable_stratum():
    assert observable_stratum(2, 2, 0.50) == 3
    assert observable_stratum(2, 2, 0.29) == 2
    assert observable_stratum(1, 2, 0.50) == 1
    assert observable_stratum(0, 2, 0.00) == 0


def test_group_statistics_pooled_and_keyed():
    values = [0.0, 2.0, 10.0, 12.0]
    keys = [0, 0, 1, 1]
    stats = group_statistics(values, keys)
    assert stats.counts["pooled"] == 4
    assert stats.counts[0] == 2
    assert stats.means["pooled"] == pytest.approx(6.0)
    assert stats.means[0] == pytest.approx(1.0)
    assert stats.means[1] == pytest.approx(11.0)
    assert stats.spreads[0] == pytest.approx(math.sqrt(2.0))
    assert stats.spreads[1] == pytest.approx(math.sqrt(2.0))
    assert stats.spreads["pooled"] == pytest.approx(math.sqrt(104.0 / 3.0))


def test_answer_advantages_pooled_has_zero_mean():
    values = [1.0, 2.0, 3.0, 4.0]
    keys = [0, 0, 1, 1]
    advantages, fallbacks = answer_advantages(values, keys, 0.5, 3, "pooled")
    assert sum(advantages) == pytest.approx(0.0, abs=1e-5)
    assert fallbacks == [1, 1, 1, 1]
    assert advantages[0] < advantages[3]


def test_answer_advantages_cell_normalisation_is_within_cell():
    values = [0.0, 2.0, 10.0, 12.0]
    keys = [0, 0, 1, 1]
    advantages, fallbacks = answer_advantages(values, keys, 0.5, 2, "cell")
    assert fallbacks == [0, 0, 0, 0]
    assert advantages[0] + advantages[1] == pytest.approx(0.0, abs=1e-5)
    assert advantages[2] + advantages[3] == pytest.approx(0.0, abs=1e-5)
    assert advantages[3] > advantages[0]


def test_answer_advantages_falls_back_below_floor():
    values = [1.0, 2.0, 100.0]
    keys = [0, 1, 1]
    advantages, fallbacks = answer_advantages(values, keys, 0.5, 2, "interpolated")
    stats = group_statistics(values, keys)
    pooled_term = (values[0] - stats.means["pooled"]) / (stats.spreads["pooled"] + EPSILON)
    assert fallbacks[0] == 1
    assert fallbacks[1] == 0
    assert advantages[0] == pytest.approx(pooled_term)


def test_answer_advantages_interpolation_mixes_terms():
    values = [0.0, 2.0, 10.0, 12.0]
    keys = [0, 0, 1, 1]
    advantages, fallbacks = answer_advantages(values, keys, 0.5, 2, "interpolated")
    stats = group_statistics(values, keys)
    expected = []
    for index, key in enumerate(keys):
        pooled_term = (values[index] - stats.means["pooled"]) / (stats.spreads["pooled"] + EPSILON)
        cell_term = (values[index] - stats.means[key]) / (stats.spreads[key] + EPSILON)
        expected.append(0.5 * pooled_term + 0.5 * cell_term)
    assert fallbacks == [0, 0, 0, 0]
    for left, right in zip(advantages, expected):
        assert left == pytest.approx(right)
    assert advantages != answer_advantages(values, keys, 1.0, 2, "interpolated")[0]


def test_mean_match_over_credits():
    assert mean_match([]) == 0.0
    credits = [SimpleNamespace(match=1.0), SimpleNamespace(match=0.0), SimpleNamespace(match=0.5)]
    assert mean_match(credits) == pytest.approx(0.5)


def test_response_advantage_formula():
    assert response_advantage(2.0, 1.0, 0.25, 3.0, True) == pytest.approx(2.0 + 3.0 * 0.75)
    assert response_advantage(2.0, 0.0, 0.25, 3.0, True) == pytest.approx(2.0 - 3.0 * 0.25)
    assert response_advantage(2.0, 1.0, 0.25, 3.0, False) == pytest.approx(2.0)


def test_running_match_baseline_is_exponential():
    baseline = RunningMatchBaseline()
    baseline.update(0.4, 0.9)
    assert baseline.baseline() == pytest.approx(0.04)
    baseline.update(0.4, 0.9)
    assert baseline.baseline() == pytest.approx(0.9 * 0.04 + 0.1 * 0.4)
    for _ in range(200):
        baseline.update(0.4, 0.9)
    assert baseline.baseline() == pytest.approx(0.4, abs=1e-3)


def test_running_match_baseline_uncentered_freezes_first_batch():
    baseline = RunningMatchBaseline(centered=False)
    baseline.update(0.3, 0.9)
    assert baseline.baseline() == pytest.approx(0.3)
    baseline.update(0.8, 0.9)
    assert baseline.baseline() == pytest.approx(0.3)


def test_arm_config_overrides():
    base = Config()
    assert arm_config(base, "ours_rec") == base
    assert arm_config(base, "ours_proxy").use_recorded_state is False
    assert arm_config(base, "grpo_clean").training_corruption_rate == 0.0
    assert arm_config(base, "grpo_clean").decision_credit is False
    assert arm_config(base, "ablate_uncentered").center_match is False
    assert arm_config(base, "ablate_uncentered").response_center is False
    assert arm_config(base, "repr_parsed").credit_source == "parsed"
    assert arm_config(base, "repr_parsed").decision_slot is True
    assert arm_config(base, "repr_instruction").decision_slot is False
    assert arm_config(base, "counterfactual").counterfactual_branches is True
    assert arm_config(base, "no_component").tool_access is False
    assert base.center_match is True


def test_arm_config_rejects_unknown_arm():
    with pytest.raises(KeyError):
        arm_config(Config(), "not_an_arm")
