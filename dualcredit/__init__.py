# Public imports for the synthetic dual-channel credit assignment experiments.

from .arbitration import CallCredit, credit_calls, group_key, judge_action, mean_match
from .config import Config, arm_config
from .credit import (
    CELL_COUNT,
    RunningMatchBaseline,
    answer_advantages,
    cell_index,
    decided_indicator,
    group_statistics,
    match_indicator,
    mean_match_per_episode,
    necessity_indicator,
    normative_actions,
    observable_stratum,
    response_advantage,
    terminal_reward,
)
from .data import Instance, TaskDataset, exact_match, id_of_numeric, numeric_of
from .evaluation import Harness
from .features import (
    ACTION_ADOPT,
    ACTION_NO_ACTION,
    ACTION_OVERRIDE,
    ACTION_REQUERY,
    ANSWER_STOP_TOKEN,
    CALL_TOKEN,
    EpisodeState,
    FeatureBuilder,
)
from .gates import gate_action
from .objective import answer_channel_term, dual_channel_loss, gather_logprob, kl_k3, response_channel_term
from .perturbation import PayloadRecord, PerturbationEngine
from .policy import Policy
from .proxy import NLIScorer, ProxyEstimator, ProxyOutput
from .rollout import CallRecord, EpisodeRecord, PositionRecord, RolloutEngine, RolloutResult, SLOT_ANSWER, SLOT_CALL, SLOT_DECISION
from .trainer import Trainer

__all__ = [
    "ACTION_ADOPT",
    "ACTION_NO_ACTION",
    "ACTION_OVERRIDE",
    "ACTION_REQUERY",
    "ANSWER_STOP_TOKEN",
    "CALL_TOKEN",
    "CELL_COUNT",
    "CallCredit",
    "CallRecord",
    "Config",
    "EpisodeRecord",
    "EpisodeState",
    "FeatureBuilder",
    "Harness",
    "Instance",
    "NLIScorer",
    "PayloadRecord",
    "PerturbationEngine",
    "Policy",
    "PositionRecord",
    "ProxyEstimator",
    "ProxyOutput",
    "RolloutEngine",
    "RolloutResult",
    "RunningMatchBaseline",
    "SLOT_ANSWER",
    "SLOT_CALL",
    "SLOT_DECISION",
    "TaskDataset",
    "Trainer",
    "answer_advantages",
    "answer_channel_term",
    "arm_config",
    "cell_index",
    "credit_calls",
    "decided_indicator",
    "dual_channel_loss",
    "exact_match",
    "gate_action",
    "gather_logprob",
    "group_key",
    "group_statistics",
    "id_of_numeric",
    "judge_action",
    "kl_k3",
    "match_indicator",
    "mean_match",
    "mean_match_per_episode",
    "necessity_indicator",
    "normative_actions",
    "numeric_of",
    "observable_stratum",
    "response_advantage",
    "response_channel_term",
    "terminal_reward",
]
