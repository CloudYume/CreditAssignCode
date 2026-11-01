# Estimate payload validity from candidate agreement and corroboration.
# NLIScorer is a seeded synthetic scoring model, not a pretrained NLI service.

import random
from dataclasses import dataclass

from .config import Config
from .perturbation import PayloadRecord

AGREEMENT_BASE = 0.30
AGREEMENT_BONUS = 0.52
NEIGHBOR_BONUS = 0.10
ENTAILMENT_NOISE = 0.11
AGREEMENT_CONFUSION = 0.07


@dataclass
class ProxyOutput:
    estimated_state: int | None
    entailment: float
    corroborated: int
    decidable: int


class NLIScorer:
    def __init__(self, config: Config):
        self.config = config

    def entailment(self, payload_id: int, candidate_id: int, instance_id: int, call_index: int) -> float:
        rng = random.Random("nli-{0}-{1}-{2}".format(instance_id, call_index, candidate_id))
        score = AGREEMENT_BASE + rng.gauss(0.0, ENTAILMENT_NOISE)
        if payload_id == candidate_id:
            score += AGREEMENT_BONUS
        elif abs(payload_id - candidate_id) == 1:
            score += NEIGHBOR_BONUS
        return float(min(1.0, max(0.0, score)))

    def agree(self, left_id: int, right_id: int, instance_id: int, call_index: int, position: int) -> int:
        rng = random.Random("agree-{0}-{1}-{2}".format(instance_id, call_index, position))
        observed = int(left_id == right_id)
        if rng.random() < AGREEMENT_CONFUSION:
            observed = 1 - observed
        return observed


class ProxyEstimator:
    def __init__(self, config: Config, scorer: NLIScorer, use_agreement_factor: bool | None = None, use_corroboration: bool | None = None):
        self.config = config
        self.scorer = scorer
        self.use_agreement_factor = config.proxy_agreement_factor if use_agreement_factor is None else use_agreement_factor
        self.use_corroboration = config.proxy_corroboration if use_corroboration is None else use_corroboration

    def estimate(self, payload: PayloadRecord, elicited_id: int, instance_id: int) -> ProxyOutput:
        # Missing elicitation is an unknown state, not evidence of an invalid payload.
        if elicited_id < 0:
            return ProxyOutput(estimated_state=None, entailment=0.0, corroborated=0, decidable=0)
        entailment = self.scorer.entailment(payload.consumed_id, elicited_id, instance_id, payload.call_index)
        corroborating = 0
        for position, passage_id in enumerate(payload.passage_ids):
            if self.scorer.agree(passage_id, payload.consumed_id, instance_id, payload.call_index, position):
                corroborating += 1
        corroborated = int(corroborating >= self.config.corroboration_depth)
        if payload.consumed_id == self.config.num_values:
            return ProxyOutput(estimated_state=0, entailment=entailment, corroborated=corroborated, decidable=1)
        # The proxy is positive only when each enabled factor passes.
        entailment_pass = 1 if (not self.use_agreement_factor or entailment >= self.config.entailment_tau) else 0
        corroboration_pass = 1 if (not self.use_corroboration or corroborated) else 0
        decidable = int(payload.extractable_count > 0)
        if not decidable:
            return ProxyOutput(estimated_state=None, entailment=entailment, corroborated=corroborated, decidable=0)
        return ProxyOutput(
            estimated_state=int(entailment_pass * corroboration_pass),
            entailment=entailment,
            corroborated=corroborated,
            decidable=decidable,
        )
