# Hand-written inference gates used by the non-training comparison arms.
# They select actions from observable state and optional simulated scorer outputs.

from .config import Config
from .features import ACTION_ADOPT, ACTION_OVERRIDE, ACTION_REQUERY, EpisodeState

CONFIDENCE_SPLIT = 0.30


def gate_action(gate: str, state: EpisodeState, config: Config, scorer) -> int:
    budget_left = state.calls_used < config.max_calls
    corroborated = state.agreeing_count >= config.corroboration_depth
    confident = state.memory_confidence >= CONFIDENCE_SPLIT
    entailment = -1.0
    if scorer is not None:
        entailment = scorer.entailment(state.consumed_id, state.memory_candidate_id, state.instance_id, max(0, state.calls_used - 1))
    if gate == "caution":
        if confident and not corroborated:
            return ACTION_OVERRIDE
        if confident or corroborated:
            return ACTION_ADOPT
        return ACTION_REQUERY if budget_left else ACTION_ADOPT
    if gate == "hint":
        if state.extractable_count < config.passages_per_call:
            return ACTION_REQUERY if budget_left else ACTION_ADOPT
        return ACTION_ADOPT if corroborated else ACTION_OVERRIDE
    if gate == "trust":
        if entailment >= config.entailment_tau:
            return ACTION_ADOPT
        return ACTION_REQUERY if budget_left else ACTION_OVERRIDE
    if gate == "agreement":
        if corroborated:
            return ACTION_ADOPT
        return ACTION_REQUERY if budget_left else ACTION_OVERRIDE
    if gate == "abstain":
        if entailment >= config.entailment_tau and corroborated:
            return ACTION_ADOPT
        if budget_left:
            return ACTION_REQUERY
        return ACTION_OVERRIDE
    return ACTION_ADOPT
