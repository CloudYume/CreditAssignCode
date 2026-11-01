# Quadrant labels, supported actions, reward normalization, and credit shaping.
# Action IDs are override=0, adopt=1, and requery=2; state IDs are s=q+2*z.

import math
from dataclasses import dataclass

CELL_COUNT = 4
ALL_ACTIONS = (0, 1, 2)
EPSILON = 1e-6


def terminal_reward(success: int, num_calls: int, alpha_call: float) -> float:
    return float(success - alpha_call * num_calls)


def necessity_indicator(unaided_success: int) -> int:
    return int(1 - int(unaided_success))


def cell_index(necessity: int, payload_state: int) -> int:
    # q=1 means unaided failure; z=1 means the consumed payload is valid.
    return int(necessity) + 2 * int(payload_state)


def normative_actions(cell: int, budget_remaining: int) -> tuple:
    # A singleton is a unique target; ALL_ACTIONS marks an undecided case.
    if cell == 0:
        return (0,)
    if cell == 1:
        # Requery is normative only while another call remains available.
        return (2,) if budget_remaining > 0 else ALL_ACTIONS
    if cell == 2:
        return ALL_ACTIONS
    if cell == 3:
        return (1,)
    raise ValueError(cell)


def match_indicator(action: int, cell: int | None, budget_remaining: int) -> float:
    if cell is None:
        return 0.0
    targets = normative_actions(cell, budget_remaining)
    return float(1 if (len(targets) == 1 and action in targets) else 0)


def decided_indicator(cell: int | None, budget_remaining: int) -> int:
    if cell is None:
        return 0
    return int(len(normative_actions(cell, budget_remaining)) == 1)


def observable_stratum(agreeing_count: int, corroboration_depth: int, memory_confidence: float) -> int:
    corroborated = int(agreeing_count >= corroboration_depth)
    confident = int(memory_confidence >= 0.30)
    return corroborated * 2 + confident


@dataclass
class GroupStats:
    counts: dict
    means: dict
    spreads: dict

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _stats(values: list) -> tuple:
    count = len(values)
    if count == 0:
        return 0, 0.0, 0.0
    mean = sum(values) / count
    if count == 1:
        return 1, mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return count, mean, math.sqrt(variance)


def group_statistics(values: list, keys: list) -> GroupStats:
    counts: dict = {}
    means: dict = {}
    spreads: dict = {}
    pooled = [values[index] for index, key in enumerate(keys) if key is not None]
    # Pooled statistics include every reward, even when its cell label is missing.
    total, mean, spread = _stats([values[index] for index in range(len(values))])
    counts["pooled"] = total
    means["pooled"] = mean
    spreads["pooled"] = spread
    grouped: dict = {}
    for index, key in enumerate(keys):
        if key is None:
            continue
        grouped.setdefault(key, []).append(values[index])
    for key, group in grouped.items():
        count, group_mean, group_spread = _stats(group)
        counts[key] = count
        means[key] = group_mean
        spreads[key] = group_spread
    return GroupStats(counts=counts, means=means, spreads=spreads)


def answer_advantages(values: list, keys: list, lam: float, floor: int, normalize: str) -> tuple:
    stats = group_statistics(values, keys)
    pooled_mean = stats.means["pooled"]
    pooled_spread = stats.spreads["pooled"]
    advantages: list[float] = []
    fallbacks: list[int] = []
    for index, key in enumerate(keys):
        value = values[index]
        pooled_term = (value - pooled_mean) / (pooled_spread + EPSILON)
        # Sparse or unavailable cells fall back to the full-batch normalized reward.
        usable = normalize != "pooled" and key is not None and stats.counts.get(key, 0) >= floor
        if not usable:
            advantages.append(pooled_term)
            fallbacks.append(1)
            continue
        cell_mean = stats.means[key]
        cell_spread = stats.spreads[key]
        cell_term = (value - cell_mean) / (cell_spread + EPSILON)
        if normalize == "cell":
            advantages.append(cell_term)
            fallbacks.append(0)
        else:
            weight = 1.0 if lam is None else lam
            # lambda=1 is pooled-only; lambda=0 is cell-only where the cell is usable.
            advantages.append(weight * pooled_term + (1.0 - weight) * cell_term)
            fallbacks.append(0)
    return advantages, fallbacks


def mean_match_per_episode(matches: list) -> float:
    if not matches:
        return 0.0
    return sum(matches) / len(matches)


class RunningMatchBaseline:
    def __init__(self, value: float = 0.0, centered: bool = True):
        self.value = value
        self.centered = centered
        self.frozen = None

    def update(self, batch_mean: float, xi: float) -> float:
        # The uncentered ablation freezes its first observed batch mean.
        if not self.centered:
            if self.frozen is None:
                self.frozen = batch_mean
            return self.frozen
        self.value = xi * self.value + (1.0 - xi) * batch_mean
        return self.value

    def baseline(self) -> float:
        if not self.centered:
            return 0.0 if self.frozen is None else self.frozen
        return self.value


def response_advantage(answer_adv: float, match: float, baseline: float, gamma: float, decision_credit: bool) -> float:
    if not decision_credit:
        return answer_adv
    return answer_adv + gamma * (match - baseline)
