import random
from dataclasses import dataclass

from .config import Config
from .data import Instance, id_of_numeric, numeric_of

RETRIEVAL_CORRUPTIONS = ("neighbor_substitution", "cross_document_transplant", "dangling_reference")
EXECUTION_CORRUPTIONS = ("magnitude", "sign", "precision")
GENUINE_AGREEMENT_MASS = (0.35, 0.40, 0.25)
CORRUPT_AGREEMENT_MASS = (0.15, 0.35, 0.50)
CONSISTENT_ERROR_MASS = 0.70
EXTRACTABLE_PROB = 0.95


@dataclass(frozen=True)
class PayloadRecord:
    call_index: int
    consumed_id: int
    passage_ids: tuple
    extractable_count: int
    agreeing_count: int
    corrupted: bool
    corruption: str
    payload_valid: int


class PerturbationEngine:
    def __init__(self, config: Config):
        self.config = config
        self.sentinel = config.num_values

    def deliver(self, instance: Instance, call_index: int, rng: random.Random, corruption_rate: float) -> PayloadRecord:
        corrupted = rng.random() < corruption_rate
        corruption = "none"
        consumed_id = instance.gold_id
        if corrupted:
            corruption = self._corruption_name(instance, rng)
            consumed_id = self._corrupted_value(instance, corruption, rng)
        agreement_target = self._agreement_count(corrupted, rng)
        extractable = [rng.random() < EXTRACTABLE_PROB for _ in range(self.config.passages_per_call)]
        passage_ids: list[int] = []
        for position in range(self.config.passages_per_call):
            if not extractable[position]:
                passage_ids.append(self.sentinel)
            elif position < agreement_target:
                passage_ids.append(consumed_id)
            else:
                passage_ids.append(self._distractor(instance, consumed_id, rng))
        agreeing = sum(1 for value in passage_ids if value == consumed_id and value != self.sentinel)
        extractable_count = sum(1 for value in passage_ids if value != self.sentinel)
        valid = int(consumed_id != self.sentinel and consumed_id == instance.gold_id)
        return PayloadRecord(
            call_index=call_index,
            consumed_id=consumed_id,
            passage_ids=tuple(passage_ids),
            extractable_count=extractable_count,
            agreeing_count=agreeing,
            corrupted=corrupted,
            corruption=corruption,
            payload_valid=valid,
        )

    def _corruption_name(self, instance: Instance, rng: random.Random) -> str:
        table = RETRIEVAL_CORRUPTIONS if instance.family == "retrieval" else EXECUTION_CORRUPTIONS
        return table[rng.randrange(len(table))]

    def _corrupted_value(self, instance: Instance, corruption: str, rng: random.Random) -> int:
        num_values = self.config.num_values
        if rng.random() < CONSISTENT_ERROR_MASS and instance.memory_candidate_id != instance.gold_id:
            return instance.memory_candidate_id
        if instance.family == "retrieval":
            if corruption == "neighbor_substitution":
                return instance.neighbor_ids[rng.randrange(len(instance.neighbor_ids))]
            if corruption == "cross_document_transplant":
                pool = min(6, num_values)
                candidate = rng.randrange(pool)
                if candidate == instance.gold_id:
                    candidate = (candidate + 1) % num_values
                return candidate
            if corruption == "dangling_reference":
                return self.sentinel
            candidate = rng.randrange(num_values)
            if candidate == instance.gold_id:
                candidate = (candidate + 1) % num_values
            return candidate
        gold_numeric = numeric_of(instance.gold_id)
        if corruption == "magnitude":
            factor = 10.0 if rng.random() < 0.5 else 0.1
            altered = gold_numeric * factor
        elif corruption == "sign":
            altered = -gold_numeric
        else:
            altered = round(gold_numeric / 10.0) * 10.0
        return id_of_numeric(altered, num_values, instance.gold_id)

    def _agreement_count(self, corrupted: bool, rng: random.Random) -> int:
        mass = CORRUPT_AGREEMENT_MASS if corrupted else GENUINE_AGREEMENT_MASS
        threshold = rng.random()
        acc = 0.0
        for index, prob in enumerate(mass):
            acc += prob
            if threshold < acc:
                return index + 1
        return len(mass)

    def _distractor(self, instance: Instance, consumed_id: int, rng: random.Random) -> int:
        num_values = self.config.num_values
        candidate = rng.randrange(num_values)
        if candidate == consumed_id:
            candidate = (candidate + 1) % num_values
        return candidate
