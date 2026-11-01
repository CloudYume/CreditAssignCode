from dataclasses import dataclass

from .config import Config
from .credit import cell_index, decided_indicator, match_indicator, normative_actions, observable_stratum
from .features import ACTION_OVERRIDE
from .gates import gate_action
from .rollout import EpisodeRecord
from .proxy import ProxyEstimator

NO_STATE = -1


@dataclass
class CallCredit:
    call_index: int
    action: int
    payload_state: int | None
    cell: int | None
    budget_remaining: int
    match: float
    decided: int
    entailment: float | None = None
    corroborated: int | None = None
    agreement_count: int = 0
    memory_confidence: float = 0.0
    instance_id: int = -1


def judge_action(config: Config, episode: EpisodeRecord, call, elicited_id: int, scorer) -> int:
    entailment = -1.0 if scorer is None else scorer.entailment(call.payload.consumed_id, elicited_id, episode.instance_id, call.call_index)
    corroborating = int(call.payload.agreeing_count)
    if scorer is not None:
        corroborating = 0
        for position, passage_id in enumerate(call.payload.passage_ids):
            if scorer.agree(passage_id, call.payload.consumed_id, episode.instance_id, call.call_index, position):
                corroborating += 1
    estimated_cell = cell_index(int(entailment < config.entailment_tau), int(corroborating >= config.corroboration_depth))
    targets = normative_actions(estimated_cell, int(call.budget_remaining))
    if len(targets) == 1:
        return int(targets[0])
    return int(call.credited_action if call.credited_action is not None else ACTION_OVERRIDE)


def credit_calls(
    config: Config,
    episode: EpisodeRecord,
    necessity: int | None,
    proxy: ProxyEstimator | None = None,
    elicited_id: int = -1,
    scorer=None,
    memory_confidence: float = 0.0,
) -> list:
    credits: list[CallCredit] = []
    for call in episode.calls:
        if call.credited_action is None:
            continue
        action = call.credited_action
        entailment = None
        corroborated = None
        if proxy is None:
            payload_state = int(call.payload.payload_valid)
        else:
            output = proxy.estimate(call.payload, elicited_id, episode.instance_id)
            payload_state = output.estimated_state
            entailment = output.entailment
            corroborated = output.corroborated
        cell = None if (necessity is None or payload_state is None) else cell_index(necessity, payload_state)
        credits.append(
            CallCredit(
                call_index=call.call_index,
                action=int(action),
                payload_state=payload_state,
                cell=cell,
                budget_remaining=int(call.budget_remaining),
                match=match_indicator(int(action), cell, int(call.budget_remaining)),
                decided=decided_indicator(cell, int(call.budget_remaining)),
                entailment=entailment,
                corroborated=corroborated,
                agreement_count=int(call.payload.agreeing_count),
                memory_confidence=float(memory_confidence),
                instance_id=int(episode.instance_id),
            )
        )
    if config.credit_source == "parsed" and episode.parsed_action is not None:
        for credit in credits:
            credit.action = int(episode.parsed_action)
            credit.match = match_indicator(credit.action, credit.cell, credit.budget_remaining)
            credit.decided = decided_indicator(credit.cell, credit.budget_remaining)
    if config.credit_source == "judge":
        source_calls = [call for call in episode.calls if call.credited_action is not None]
        for credit, call in zip(credits, source_calls):
            judged = judge_action(config, episode, call, elicited_id, scorer)
            credit.action = judged
            credit.match = match_indicator(judged, credit.cell, credit.budget_remaining)
            credit.decided = decided_indicator(credit.cell, credit.budget_remaining)
    return credits


def group_key(config: Config, episode: EpisodeRecord, credits: list, instance) -> int | None:
    if not credits:
        return None
    consumed = credits[-1]
    if config.strata == "observable":
        return observable_stratum(consumed.agreement_count, config.corroboration_depth, instance.memory_confidence)
    return consumed.cell


def mean_match(credits: list) -> float:
    if not credits:
        return 0.0
    return sum(credit.match for credit in credits) / len(credits)
