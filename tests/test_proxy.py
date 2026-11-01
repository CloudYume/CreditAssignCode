import pytest

from dualcredit.config import Config
from dualcredit.perturbation import PayloadRecord
from dualcredit.proxy import NLIScorer, ProxyEstimator

BASE = Config()
SENTINEL = BASE.num_values


def build_payload(consumed_id: int, passage_ids: list, extractable_count: int, call_index: int = 0) -> PayloadRecord:
    return PayloadRecord(
        call_index=call_index,
        consumed_id=consumed_id,
        passage_ids=tuple(passage_ids),
        extractable_count=extractable_count,
        agreeing_count=sum(1 for value in passage_ids if value == consumed_id and value != SENTINEL),
        corrupted=False,
        corruption="none",
        payload_valid=int(consumed_id != SENTINEL),
    )


def test_entailment_rewards_agreement_with_the_candidate():
    scorer = NLIScorer(BASE)
    exact = [scorer.entailment(value, value, instance, 0) for instance, value in enumerate(range(0, BASE.num_values, 3))]
    apart = [scorer.entailment(value, (value + 5) % BASE.num_values, instance, 0) for instance, value in enumerate(range(0, BASE.num_values, 3))]
    assert all(0.0 <= score <= 1.0 for score in exact + apart)
    assert sum(exact) / len(exact) > sum(apart) / len(apart)
    assert scorer.entailment(5, 5, 3, 0) > scorer.entailment(5, 19, 3, 0)
    assert scorer.entailment(5, 6, 3, 0) == scorer.entailment(5, 6, 3, 0)


def test_agree_indicator_is_mostly_faithful():
    scorer = NLIScorer(BASE)
    faithful = sum(scorer.agree(7, 7, index // 4, index % 4, index) for index in range(400))
    misleading = sum(scorer.agree(7, 9, index // 4, index % 4, index) for index in range(400))
    assert faithful / 400.0 >= 0.85
    assert misleading / 400.0 <= 0.15


def test_two_factors_multiply():
    estimator = ProxyEstimator(BASE, NLIScorer(BASE))
    entailment_only = estimator.estimate(build_payload(6, [1, 2, 3], 3), 6, 4)
    assert entailment_only.entailment >= BASE.entailment_tau
    assert entailment_only.corroborated == 0
    assert entailment_only.estimated_state == 0
    assert entailment_only.decidable == 1
    both = estimator.estimate(build_payload(6, [6, 6, 6], 3), 6, 4)
    assert both.corroborated == 1
    assert both.estimated_state == 1


def test_corroboration_factor_alone_admits_consensus():
    config = Config(**{**BASE.__dict__, "proxy_agreement_factor": False})
    estimator = ProxyEstimator(config, NLIScorer(config))
    output = estimator.estimate(build_payload(6, [6, 6, 6], 3), 19, 8)
    assert output.entailment < config.entailment_tau
    assert output.corroborated == 1
    assert output.estimated_state == 1


def test_estimator_marks_dangling_payloads_invalid():
    estimator = ProxyEstimator(BASE, NLIScorer(BASE))
    output = estimator.estimate(build_payload(SENTINEL, [1, 2, 3], 3), 7, 11)
    assert output.decidable == 1
    assert output.estimated_state == 0


def test_estimator_is_undecidable_without_extractable_passages():
    estimator = ProxyEstimator(BASE, NLIScorer(BASE))
    output = estimator.estimate(build_payload(5, [SENTINEL, SENTINEL, SENTINEL], 0), 5, 12)
    assert output.decidable == 0
    assert output.estimated_state is None


def test_estimator_needs_an_elicited_candidate():
    estimator = ProxyEstimator(BASE, NLIScorer(BASE))
    output = estimator.estimate(build_payload(5, [5, 5, 5], 3), -1, 13)
    assert output.decidable == 0
    assert output.estimated_state is None
    assert output.entailment == 0.0


def test_stricter_threshold_shrinks_positives():
    counts = {}
    for tau in (0.30, 0.50, 0.70):
        config = Config(**{**BASE.__dict__, "entailment_tau": tau})
        estimator = ProxyEstimator(config, NLIScorer(config))
        counts[tau] = 0
        for instance_id in range(400):
            output = estimator.estimate(build_payload(5, [5, 5, 5], 3), 15, instance_id)
            counts[tau] += int(output.estimated_state == 1)
    assert counts[0.30] > counts[0.50]
    assert counts[0.50] >= counts[0.70]
    assert counts[0.70] >= 0
