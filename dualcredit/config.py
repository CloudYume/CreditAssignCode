# Immutable experiment settings and explicit per-arm overrides.
# Arm names are stable output identifiers shared by the CLI, tables, and tests.

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Config:
    # Reproducible task generation and rollout dimensions.
    seed: int = 1234
    family: str = "retrieval"
    num_values: int = 24
    dataset_size: int = 900
    train_share: float = 0.72
    max_calls: int = 3
    passages_per_call: int = 3
    training_corruption_rate: float = 0.30
    evaluation_corruption_rates: tuple = (0.0, 0.30)
    # Reward, rollout allocation, and dual-channel credit hyperparameters.
    alpha_call: float = 0.04
    group_size: int = 8
    episode_budget: int = 320
    pairing_quota: float = 0.25
    updates: int = 200
    lam: float = 0.5
    gamma: float = 3.0
    xi: float = 0.90
    cell_floor: int = 3
    clip_delta: float = 0.20
    kl_beta: float = 0.02
    entropy_beta: float = 0.80
    learning_rate: float = 5.0e-3
    # Model capacity and trainable-parameter controls.
    hidden: int = 64
    adapter_rank: int = 8
    heads_trainable: bool = True
    trunk_adapter_trainable: bool = False
    answer_head_trainable: bool = False
    # Proxy thresholds and method/ablation switches.
    entailment_tau: float = 0.50
    corroboration_depth: int = 2
    proxy_agreement_factor: bool = True
    proxy_corroboration: bool = True
    no_requery: bool = False
    normalize: str = "interpolated"
    strata: str = "cell"
    decision_credit: bool = True
    center_match: bool = True
    response_center: bool = True
    response_clip: bool = True
    use_recorded_state: bool = True
    decision_slot: bool = True
    credit_source: str = "emitted"
    pairing: bool = True
    counterfactual_branches: bool = False
    gate: str = "none"
    tool_access: bool = True
    eval_batch: int = 320
    log_every: int = 8


def arm_config(base: Config, arm: str) -> Config:
    # These are local synthetic baselines; names do not import external methods.
    overrides: dict[str, dict[str, Any]] = {
        "no_component": {"tool_access": False, "updates": 0},
        "caution_prompt": {"updates": 0, "gate": "caution"},
        "corruption_hint": {"updates": 0, "gate": "hint"},
        "grpo_clean": {"training_corruption_rate": 0.0, "normalize": "pooled", "decision_slot": True, "decision_credit": False},
        "grpo_corrupt": {"normalize": "pooled", "decision_credit": False},
        "boundary_reward": {"normalize": "pooled", "decision_credit": False, "alpha_call": 0.14},
        "shortcut_suppressed": {"normalize": "pooled", "decision_credit": False, "alpha_call": 0.10, "kl_beta": 0.12},
        "trust_gate": {"updates": 0, "gate": "trust"},
        "agreement_gate": {"updates": 0, "gate": "agreement"},
        "abstain_gate": {"updates": 0, "gate": "abstain"},
        "stratified_norm": {"normalize": "pooled", "decision_credit": False, "strata": "observable"},
        "ours_rec": {},
        "ours_proxy": {"use_recorded_state": False},
        "ablate_lam1": {"lam": 1.0},
        "ablate_lam0": {"lam": 0.0},
        "ablate_floor8": {"cell_floor": 8},
        "ablate_gamma0": {"decision_credit": False},
        "ablate_gamma05": {"gamma": 0.5},
        "ablate_uncentered": {"center_match": False, "response_center": False},
        "ablate_response_unclipped": {"response_clip": False},
        "ablate_response_raw": {"response_center": False},
        "ablate_xi095": {"xi": 0.95},
        "ablate_pairing0": {"pairing": False},
        "ablate_pairing015": {"pairing_quota": 0.15},
        "ablate_pairing050": {"pairing_quota": 0.50},
        "ablate_vartheta0": {"training_corruption_rate": 0.0},
        "ablate_vartheta050": {"training_corruption_rate": 0.50},
        "ablate_budget_j": {"max_calls": 2},
        "ablate_group4": {"group_size": 4},
        "ablate_group16": {"group_size": 16},
        "repr_parsed": {"credit_source": "parsed"},
        "repr_judge": {"credit_source": "judge"},
        "repr_instruction": {"decision_slot": False, "credit_source": "instruction"},
        "counterfactual": {"counterfactual_branches": True},
        "proxy_no_agreement": {"use_recorded_state": False, "proxy_agreement_factor": False},
        "proxy_no_corroboration": {"use_recorded_state": False, "proxy_corroboration": False},
        "proxy_kappa3": {"use_recorded_state": False, "corroboration_depth": 3},
        "proxy_tau07": {"use_recorded_state": False, "entailment_tau": 0.70},
        "proxy_tau03": {"use_recorded_state": False, "entailment_tau": 0.30},
        "proxy_budget_j": {"use_recorded_state": False, "no_requery": True},
        "proxy_single_call": {"use_recorded_state": False, "max_calls": 1},
    }
    if arm not in overrides:
        raise KeyError(arm)
    merged = dict(overrides[arm])
    if arm not in {"no_component", "caution_prompt", "corruption_hint", "trust_gate", "agreement_gate", "abstain_gate"}:
        merged.setdefault("normalize", "interpolated")
        merged.setdefault("decision_credit", True)
    return replace(base, **merged)
