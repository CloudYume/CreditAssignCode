import pytest
import torch

from dualcredit.config import Config
from dualcredit.data import TaskDataset, exact_match
from dualcredit.features import (
    ACTION_ADOPT,
    ACTION_OVERRIDE,
    ACTION_REQUERY,
    FeatureBuilder,
)
from dualcredit.perturbation import PerturbationEngine
from dualcredit.policy import Policy
from dualcredit.proxy import NLIScorer
from dualcredit.rollout import NO_POSITION, SLOT_ANSWER, SLOT_CALL, SLOT_DECISION, RolloutEngine

SMALL = Config(**{**Config().__dict__, "dataset_size": 60, "num_values": 12, "max_calls": 2, "hidden": 24, "adapter_rank": 4})


def build(config: Config = SMALL, seed: int = 7):
    features = FeatureBuilder(config)
    policy = Policy(config, features)
    engine = RolloutEngine(config, features, policy, PerturbationEngine(config))
    dataset = TaskDataset(config, seed=seed)
    return config, features, engine, dataset


def sample(dataset: TaskDataset, count: int = 24):
    instances = dataset.test[:count]
    seeds = [11 + instance.instance_id for instance in instances]
    return instances, seeds


def test_unaided_rollout_uses_no_calls():
    config, features, engine, dataset = build()
    instances, seeds = sample(dataset)
    result = engine.run(instances, seeds, decode="greedy", unaided=True)
    assert len(result.episodes) == len(instances)
    assert all(episode.unaided for episode in result.episodes)
    assert all(episode.num_calls == 0 for episode in result.episodes)
    assert all(position.slot == SLOT_ANSWER for position in result.positions)
    assert len(result.positions) == len(instances)


def test_assisted_rollout_emits_call_and_decision_slots():
    config, features, engine, dataset = build()
    instances, seeds = sample(dataset)
    result = engine.run(instances, seeds, decode="greedy")
    slots = {position.slot for position in result.positions}
    assert SLOT_CALL in slots
    assert SLOT_DECISION in slots
    assert SLOT_ANSWER in slots
    for episode in result.episodes:
        assert episode.num_calls >= 1
        decisions = [position for position in result.positions if position.episode_index == episode.episode_index and position.slot == SLOT_DECISION]
        assert len(decisions) == episode.num_calls
        for call in episode.calls:
            assert call.decision_position != NO_POSITION
            assert 0 <= call.call_index < config.max_calls


def test_corruption_rate_controls_payload_validity():
    config, features, engine, dataset = build()
    instances, seeds = sample(dataset, 40)
    clean = engine.run(instances, seeds, decode="greedy", corruption_rate=0.0)
    dirty = engine.run(instances, seeds, decode="greedy", corruption_rate=1.0)
    clean_valid = [call.payload.payload_valid for episode in clean.episodes for call in episode.calls]
    dirty_valid = [call.payload.payload_valid for episode in dirty.episodes for call in episode.calls]
    assert sum(clean_valid) == len(clean_valid)
    assert sum(dirty_valid) == 0
    clean_score = sum(exact_match(episode.answer_id, episode.gold_id, episode.family, config.num_values) for episode in clean.episodes)
    dirty_score = sum(exact_match(episode.answer_id, episode.gold_id, episode.family, config.num_values) for episode in dirty.episodes)
    assert clean_score > dirty_score


def test_adopted_payload_drives_the_answer():
    config, features, engine, dataset = build()
    instances, seeds = sample(dataset, 40)
    adopted = engine.run(instances, seeds, decode="greedy", corruption_rate=0.0, forced=[{0: ACTION_ADOPT}] * len(instances))
    overridden = engine.run(instances, seeds, decode="greedy", corruption_rate=0.0, forced=[{0: ACTION_OVERRIDE}] * len(instances))
    adopt_hits = 0
    adopt_total = 0
    for episode in adopted.episodes:
        payload = episode.calls[0].payload
        if payload.consumed_id >= config.num_values:
            continue
        adopt_total += 1
        adopt_hits += int(episode.answer_id == payload.consumed_id)
    override_hits = sum(int(episode.answer_id == instance.memory_candidate_id) for episode, instance in zip(overridden.episodes, instances))
    assert adopt_total > 0
    assert adopt_hits / adopt_total >= 0.5
    assert override_hits / len(instances) >= 0.5


def test_decision_budget_blocks_requery_at_last_call():
    config = Config(**{**SMALL.__dict__, "max_calls": 1})
    _, features, engine, dataset = build(config)
    instances, seeds = sample(dataset)
    result = engine.run(instances, seeds, decode="greedy")
    for episode in result.episodes:
        assert episode.num_calls == 1
        call = episode.calls[0]
        assert call.budget_remaining == 0
        assert call.credited_action != ACTION_REQUERY
    decisions = [position for position in result.positions if position.slot == SLOT_DECISION]
    assert all(position.token != ACTION_REQUERY for position in decisions)


def test_forced_actions_are_recorded_on_calls():
    config, features, engine, dataset = build()
    instances, seeds = sample(dataset)
    result = engine.run(instances, seeds, decode="greedy", forced=[{0: ACTION_OVERRIDE}] * len(instances))
    for episode in result.episodes:
        call = episode.calls[0]
        assert call.credited_action == ACTION_OVERRIDE
        assert call.emitted_action is None
        assert call.decision_position != NO_POSITION
        assert episode.calls[0].budget_remaining == config.max_calls - 1


def test_gate_selection_replaces_emitted_action():
    config, features, engine, dataset = build()
    instances, seeds = sample(dataset)
    scorer = NLIScorer(config)
    result = engine.run(instances, seeds, decode="greedy", gate="agreement", scorer=scorer)
    for episode in result.episodes:
        for call in episode.calls:
            assert call.emitted_action is None
            assert call.credited_action in (ACTION_OVERRIDE, ACTION_ADOPT, ACTION_REQUERY)
            assert call.entailment is not None


def test_position_features_align_with_positions():
    config, features, engine, dataset = build()
    instances, seeds = sample(dataset)
    result = engine.run(instances, seeds, decode="sample", generator=torch.Generator().manual_seed(5))
    assert result.position_features.shape[0] == len(result.positions)
    assert result.position_features.shape[1] == features.dimension
    assert float(result.position_features[:, features.confidence_start].abs().sum()) > 0.0


def test_reward_is_exact_match_minus_call_penalty():
    config, features, engine, dataset = build()
    instances, seeds = sample(dataset)
    result = engine.run(instances, seeds, decode="greedy")
    for episode in result.episodes:
        correct = exact_match(episode.answer_id, episode.gold_id, episode.family, config.num_values)
        assert episode.reward == pytest.approx(correct - config.alpha_call * episode.num_calls)
        assert episode.num_calls == len(episode.calls)
