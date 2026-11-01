# Exercise arm construction, short training, evaluation, reproducibility, and exports.
# SMALL deliberately bounds the work; these checks do not reproduce full paper results.

import math
from dataclasses import FrozenInstanceError

import pytest

from dualcredit.config import Config, arm_config
from dualcredit.data import TaskDataset, exact_match, id_of_numeric, numeric_of
from dualcredit.evaluation import Harness
from dualcredit.experiment import (
    ABLATION_ARMS,
    COST_ARMS,
    MAIN_ARMS,
    PROXY_ARMS,
    REPRESENTATION_ARMS,
    SHARED_KEYS,
    CELL_KEYS,
    run,
)
from dualcredit.trainer import Trainer

SMALL = Config(**{**Config().__dict__, "dataset_size": 120, "num_values": 12, "hidden": 24, "adapter_rank": 4, "episode_budget": 32, "group_size": 4, "eval_batch": 48, "updates": 3, "log_every": 100})


def fit(config: Config, seed: int | None = None) -> tuple:
    dataset = TaskDataset(config)
    trainer = Trainer(config, dataset, seed=config.seed if seed is None else seed)
    trainer.train()
    return dataset, trainer


def test_config_round_trip():
    base = Config()
    assert base.gamma == 3.0
    moved = Config(**{**SMALL.__dict__, "gamma": 1.5})
    assert moved.gamma == 1.5
    assert moved.dataset_size == SMALL.dataset_size
    with pytest.raises(FrozenInstanceError):
        object.__setattr__  # noqa: B018 placeholder guard
        base.gamma = 1.0


def test_every_arm_builds_a_trainer():
    arms = tuple(dict.fromkeys(MAIN_ARMS + ABLATION_ARMS + PROXY_ARMS + REPRESENTATION_ARMS + COST_ARMS))
    for arm in arms:
        config = arm_config(SMALL, arm)
        trainer = Trainer(config, TaskDataset(config), seed=config.seed)
        assert trainer.trainable_parameters() > 0
        assert trainer.frozen_parameters() > 0
        if arm in {"no_component", "caution_prompt", "corruption_hint", "trust_gate", "agreement_gate", "abstain_gate"}:
            assert config.updates == 0


def test_short_training_run_is_finite():
    _, trainer = fit(SMALL)
    assert len(trainer.history) == SMALL.updates
    for diagnostics in trainer.history:
        for key in ("reward", "em", "adopt", "dec_acc", "cov", "calls", "loss", "kl", "entropy", "shift"):
            assert math.isfinite(diagnostics[key])
        assert -SMALL.alpha_call * SMALL.max_calls <= diagnostics["reward"] <= 1.0
    assert trainer.assisted_episodes > 0


def test_training_is_reproducible():
    _, first = fit(SMALL, seed=17)
    _, second = fit(SMALL, seed=17)
    assert len(first.history) == len(second.history)
    for left, right in zip(first.history, second.history):
        assert left["loss"] == pytest.approx(right["loss"])
        assert left["em"] == pytest.approx(right["em"])
        assert left["adopt"] == pytest.approx(right["adopt"])


def test_no_component_arm_never_calls():
    config = arm_config(SMALL, "no_component")
    dataset, trainer = fit(config)
    assert trainer.history == []
    report = Harness(config, dataset, trainer.policy).evaluate(0.3)
    assert report["calls"] == 0.0
    assert report["credits"] == 0
    assert report["compliance"] == 0.0


def test_gate_arms_emit_valid_actions():
    for arm in ("caution_prompt", "corruption_hint", "trust_gate", "agreement_gate", "abstain_gate"):
        config = arm_config(SMALL, arm)
        dataset, trainer = fit(config)
        report = Harness(config, dataset, trainer.policy).evaluate(0.3)
        assert report["credits"] > 0
        assert report["adoption"] + report["override"] + report["requery"] == pytest.approx(100.0)
        assert 0.0 <= report["cov"] <= 100.0


def test_credit_sources_are_available():
    for source in ("emitted", "parsed", "judge"):
        config = Config(**{**SMALL.__dict__, "credit_source": source})
        dataset, trainer = fit(config)
        report = Harness(config, dataset, trainer.policy).evaluate(0.3)
        assert report["em"] >= 0.0
        assert report["adoption"] >= 0.0
    for arm in ("repr_instruction",):
        config = arm_config(SMALL, arm)
        dataset, trainer = fit(config)
        report = Harness(config, dataset, trainer.policy).evaluate(0.3)
        assert report["compliance"] == 0.0
    config = arm_config(SMALL, "repr_parsed")
    dataset, trainer = fit(config)
    report = Harness(config, dataset, trainer.policy).evaluate(0.3)
    assert report["compliance"] > 0.0
    assert report["credits"] > 0


def test_evaluation_reports_expected_keys():
    config = Config(**{**SMALL.__dict__, "updates": 1})
    dataset, trainer = fit(config)
    report = Harness(config, dataset, trainer.policy).evaluate(0.3)
    for key in SHARED_KEYS + CELL_KEYS:
        assert key in report
    for key in ("precision", "coverage", "bias"):
        assert key in report["proxy"]
    assert 0.0 <= report["em"] <= 100.0
    assert 0.0 <= report["unaided_em"] <= 100.0
    assert report["episodes"] > 0
    assert sum(report["occ_s{0}".format(cell)] for cell in range(4)) == pytest.approx(100.0)


def test_execution_family_shares_the_pipeline():
    config = Config(**{**SMALL.__dict__, "family": "execution"})
    dataset, trainer = fit(config)
    assert dataset.family == "execution"
    report = Harness(config, dataset, trainer.policy).evaluate(0.3)
    assert report["em"] >= 0.0
    assert report["credits"] > 0
    assert exact_match(4, 4, "execution", config.num_values) == 1
    assert id_of_numeric(numeric_of(5), config.num_values, 0) == 5
    assert id_of_numeric(numeric_of(7), config.num_values, 7) != 7


def test_experiment_runner_produces_a_table(tmp_path):
    config = Config(**{**SMALL.__dict__, "dataset_size": 80, "episode_budget": 16, "eval_batch": 24, "updates": 1})
    payload = run("cost", config, output_dir=str(tmp_path), families=("retrieval",), log=lambda message: None)
    assert "Arm" in payload["table"]
    assert "ours_rec" in payload["table"]
    assert len(payload["rows"]) == len(COST_ARMS)
    assert (tmp_path / "cost.json").exists()
    assert (tmp_path / "cost.txt").exists()
