import math
import random
from dataclasses import dataclass

from .config import Config

MEMORY_TIER_PROBS = (0.30, 0.30, 0.40)
MEMORY_TIER_CORRECT = (0.90, 0.55, 0.22)
LOGIT_NOISE = 0.55
CORRECT_BOOST = 3.40
WRONG_BOOST = 1.10
BOOST_JITTER = 0.45
VALUE_SPACING = 7.0
VALUE_OFFSET = 0.5


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def numeric_of(value_id: int) -> float:
    return value_id * VALUE_SPACING + VALUE_OFFSET


def id_of_numeric(numeric: float, num_values: int, gold_id: int) -> int:
    best_id = -1
    best_gap = float("inf")
    for candidate in range(num_values):
        gap = abs(numeric_of(candidate) - numeric)
        if gap < best_gap:
            best_gap = gap
            best_id = candidate
    if best_id == gold_id:
        shifted = (gold_id + 1) % num_values
        return shifted
    return best_id


@dataclass(frozen=True)
class Instance:
    instance_id: int
    family: str
    gold_id: int
    tier: int
    memory_logits: tuple
    memory_probs: tuple
    memory_candidate_id: int
    memory_confidence: float
    numeric_value: float
    neighbor_ids: tuple


class TaskDataset:
    def __init__(self, config: Config, seed: int | None = None, family: str | None = None):
        self.config = config
        self.family = family or config.family
        rng = random.Random(config.seed if seed is None else seed)
        self.instances = [self._build(index, rng) for index in range(config.dataset_size)]
        split = int(round(config.train_share * config.dataset_size))
        self.train = self.instances[:split]
        self.test = self.instances[split:]

    def _build(self, index: int, rng: random.Random) -> Instance:
        num_values = self.config.num_values
        tier = _draw_tier(rng)
        gold_id = rng.randrange(num_values)
        correct = rng.random() < MEMORY_TIER_CORRECT[tier]
        candidate_id = gold_id if correct else self._distractor(gold_id, rng)
        boost = (CORRECT_BOOST if correct else WRONG_BOOST) + rng.gauss(0.0, BOOST_JITTER)
        logits = [rng.gauss(0.0, LOGIT_NOISE) for _ in range(num_values)]
        logits[candidate_id] += boost
        probs = softmax(logits)
        neighbors = tuple(sorted({(gold_id + 1) % num_values, (gold_id - 1) % num_values}))
        return Instance(
            instance_id=index,
            family=self.family,
            gold_id=gold_id,
            tier=tier,
            memory_logits=tuple(logits),
            memory_probs=tuple(probs),
            memory_candidate_id=candidate_id,
            memory_confidence=probs[candidate_id],
            numeric_value=numeric_of(gold_id),
            neighbor_ids=neighbors,
        )

    def _distractor(self, gold_id: int, rng: random.Random) -> int:
        num_values = self.config.num_values
        candidate = rng.randrange(num_values)
        if candidate == gold_id:
            candidate = (candidate + 1) % num_values
        return candidate

    def instance(self, instance_id: int) -> Instance:
        return self.instances[instance_id]


def _draw_tier(rng: random.Random) -> int:
    threshold = rng.random()
    acc = 0.0
    for tier, prob in enumerate(MEMORY_TIER_PROBS):
        acc += prob
        if threshold < acc:
            return tier
    return len(MEMORY_TIER_PROBS) - 1


def exact_match(answer_id: int, gold_id: int, family: str, num_values: int) -> int:
    if answer_id < 0:
        return 0
    if family == "retrieval":
        return int(answer_id == gold_id)
    return int(abs(numeric_of(answer_id) - numeric_of(gold_id)) <= 1e-3)
