# Encode observable rollout state into fixed-width policy inputs.
# Gold answers stay in records for scoring; components() does not encode gold_id.

from dataclasses import dataclass

import torch

from .config import Config

CALL_TOKEN = 0
ANSWER_STOP_TOKEN = 1
ACTION_OVERRIDE = 0
ACTION_ADOPT = 1
ACTION_REQUERY = 2
ACTION_NO_ACTION = 3


@dataclass
class EpisodeState:
    instance_id: int
    family: str
    gold_id: int
    memory_probs: list
    memory_candidate_id: int
    memory_confidence: float
    consumed_id: int
    passage_ids: list
    agreeing_count: int
    extractable_count: int
    adopted_id: int
    last_action: int
    calls_used: int
    seed: int = 0


class FeatureBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.num_values = config.num_values
        # The extra one-hot slot represents an unresolvable/dangling payload.
        self.slot = config.num_values + 1
        self.max_calls = config.max_calls
        self.order = [
            "memory_probs",
            "candidate_onehot",
            "confidence",
            "consumed_onehot",
            "passage_mean",
            "agreeing",
            "extractable",
            "adopted_onehot",
            "last_action",
            "calls_used",
            "budget",
            "family",
            "prior_stats",
            "payload_prior",
            "consistency",
            "candidate_support",
        ]
        self.widths = {
            "memory_probs": self.num_values,
            "candidate_onehot": self.num_values,
            "confidence": 1,
            "consumed_onehot": self.slot,
            "passage_mean": self.slot,
            "agreeing": 1,
            "extractable": 1,
            "adopted_onehot": self.slot,
            "last_action": 4,
            "calls_used": self.max_calls + 1,
            "budget": 1,
            "family": 2,
            "prior_stats": 3,
            "payload_prior": 2,
            "consistency": 4,
            "candidate_support": 1,
        }
        self.starts = {}
        cursor = 0
        for key in self.order:
            self.starts[key] = cursor
            cursor += self.widths[key]
        self.dimension = cursor
        self.memory_start = self.starts["memory_probs"]
        self.candidate_start = self.starts["candidate_onehot"]
        self.confidence_start = self.starts["confidence"]
        self.consumed_start = self.starts["consumed_onehot"]
        self.passage_start = self.starts["passage_mean"]
        self.agreeing_start = self.starts["agreeing"]
        self.extractable_start = self.starts["extractable"]
        self.adopted_start = self.starts["adopted_onehot"]
        self.action_start = self.starts["last_action"]
        self.calls_start = self.starts["calls_used"]
        self.budget_start = self.starts["budget"]
        self.family_start = self.starts["family"]

    # Build features only from memory, observed payloads, actions, and remaining budget.
    def components(self, state: EpisodeState) -> dict:
        passage_mean = [0.0] * self.slot
        candidate_support = 0.0
        for value in state.passage_ids:
            passage_mean[value] += 1.0 / max(1, len(state.passage_ids))
            if value == state.memory_candidate_id:
                candidate_support += 1.0 / max(1, len(state.passage_ids))
        consumed_real = 0 <= state.consumed_id < self.num_values
        consumed_onehot = [0.0] * self.slot
        if state.consumed_id >= 0:
            consumed_onehot[state.consumed_id] = 1.0
        adopted_onehot = [0.0] * self.slot
        if state.adopted_id >= 0:
            adopted_onehot[state.adopted_id] = 1.0
        candidate_onehot = [0.0] * self.num_values
        candidate_onehot[state.memory_candidate_id] = 1.0
        last_action = [0.0] * 4
        last_action[state.last_action] = 1.0
        calls_used = [0.0] * (self.max_calls + 1)
        calls_used[min(state.calls_used, self.max_calls)] = 1.0
        ordered = sorted(state.memory_probs, reverse=True)
        top_mass = sum(ordered[: min(3, len(ordered))])
        gap = abs(state.consumed_id - state.memory_candidate_id) / self.num_values if consumed_real else 1.0
        return {
            "memory_probs": list(state.memory_probs),
            "candidate_onehot": candidate_onehot,
            "confidence": [state.memory_confidence],
            "consumed_onehot": consumed_onehot,
            "passage_mean": passage_mean,
            "agreeing": [state.agreeing_count / self.config.passages_per_call],
            "extractable": [state.extractable_count / self.config.passages_per_call],
            "adopted_onehot": adopted_onehot,
            "last_action": last_action,
            "calls_used": calls_used,
            "budget": [(self.max_calls - state.calls_used) / self.max_calls],
            "family": [1.0, 0.0] if state.family == "retrieval" else [0.0, 1.0],
            "prior_stats": [ordered[0], top_mass, state.memory_confidence],
            "payload_prior": [state.memory_probs[state.consumed_id] if consumed_real else 0.0, float(consumed_real)],
            "consistency": [
                float(consumed_real and state.consumed_id == state.memory_candidate_id),
                float(consumed_real and state.consumed_id != state.memory_candidate_id),
                gap,
                float(state.agreeing_count >= self.config.corroboration_depth),
            ],
            "candidate_support": [candidate_support],
        }

    def vector(self, state: EpisodeState) -> list:
        components = self.components(state)
        values: list[float] = []
        for key in self.order:
            values.extend(components[key])
        return values

    def stack(self, rows: list) -> torch.Tensor:
        return torch.tensor(rows, dtype=torch.float32)
