# Record call -> payload decision -> answer transitions for batched episodes.
# Features and behavior log-probabilities are captured before each state update.

import random
from dataclasses import dataclass, field

import torch

from .config import Config
from .data import Instance, exact_match
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
from .perturbation import PayloadRecord, PerturbationEngine
from .policy import Policy

PHASE_CALL = 0
PHASE_DECISION = 1
PHASE_ANSWER = 2
PHASE_DONE = 3
SLOT_CALL = "call"
SLOT_DECISION = "decision"
SLOT_ANSWER = "answer"
NO_POSITION = -1


@dataclass
class PositionRecord:
    position_index: int
    slot: str
    episode_index: int
    token: int
    logprob: float
    call_index: int | None = None


@dataclass
class CallRecord:
    call_index: int
    payload: PayloadRecord
    emitted_action: int | None = None
    credited_action: int | None = None
    decision_position: int = NO_POSITION
    budget_remaining: int = 0
    elicited_candidate: int = -1
    entailment: float | None = None


@dataclass
class EpisodeRecord:
    episode_index: int
    instance_id: int
    family: str
    gold_id: int
    seed: int
    group_index: int = 0
    positions: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    answer_id: int = -1
    answer_position: int = NO_POSITION
    num_calls: int = 0
    reward: float = 0.0
    unaided: bool = False
    parsed_action: int | None = None

@dataclass
class RolloutResult:
    episodes: list
    positions: list
    position_features: torch.Tensor


class RolloutEngine:
    def __init__(self, config: Config, features: FeatureBuilder, policy: Policy, perturbation: PerturbationEngine):
        self.config = config
        self.features = features
        self.policy = policy
        self.perturbation = perturbation

    def run(
        self,
        instances: list,
        seeds: list,
        groups: list | None = None,
        decode: str = "sample",
        unaided: bool = False,
        forced: list | None = None,
        gate: str = "none",
        scorer=None,
        tool_access: bool = True,
        corruption_rate: float | None = None,
        generator: torch.Generator | None = None,
    ) -> RolloutResult:
        rate = self.config.training_corruption_rate if corruption_rate is None else corruption_rate
        states = [self._initial_state(instance, seed) for instance, seed in zip(instances, seeds)]
        # Unaided and no-tool runs bypass the call/decision stages entirely.
        phases = [PHASE_ANSWER if (unaided or not tool_access) else PHASE_CALL for _ in states]
        episodes = [
            EpisodeRecord(
                episode_index=index,
                instance_id=instance.instance_id,
                family=instance.family,
                gold_id=instance.gold_id,
                seed=seed,
                group_index=0 if groups is None else groups[index],
                unaided=unaided or not tool_access,
            )
            for index, (instance, seed) in enumerate(zip(instances, seeds))
        ]
        rows: list[list[float]] = []
        positions: list[PositionRecord] = []
        # Each tool return needs one call step and one decision step, then an answer.
        max_steps = 2 * self.config.max_calls + 2
        for _ in range(max_steps):
            active = [index for index in range(len(states)) if phases[index] != PHASE_DONE]
            if not active:
                break
            batch = self.features.stack([self.features.vector(states[index]) for index in active])
            with torch.no_grad():
                call_logits, decision_logits, answer_logits = self.policy(batch)
                _, reference_decision, _ = self.policy.reference(batch)
            for offset, episode_index in enumerate(active):
                state = states[episode_index]
                phase = phases[episode_index]
                forced_map = None if forced is None else forced[episode_index]
                if phase == PHASE_CALL:
                    phases[episode_index] = self._emit_call(
                        state=state,
                        logits=call_logits[offset].clone(),
                        episode=episodes[episode_index],
                        instance=instances[episode_index],
                        positions=positions,
                        rows=rows,
                        decode=decode,
                        generator=generator,
                        scorer=scorer,
                        corruption_rate=rate,
                    )
                elif phase == PHASE_DECISION:
                    phases[episode_index] = self._emit_decision(
                        state=state,
                        logits=decision_logits[offset].clone(),
                        reference_logits=reference_decision[offset],
                        episode=episodes[episode_index],
                        positions=positions,
                        rows=rows,
                        decode=decode,
                        gate=gate,
                        scorer=scorer,
                        forced_map=forced_map,
                        generator=generator,
                    )
                else:
                    phases[episode_index] = self._emit_answer(
                        state=state,
                        logits=answer_logits[offset],
                        episode=episodes[episode_index],
                        positions=positions,
                        rows=rows,
                        decode=decode,
                        generator=generator,
                    )
        for episode in episodes:
            episode.num_calls = len(episode.calls)
            correct = exact_match(episode.answer_id, episode.gold_id, episode.family, self.config.num_values)
            episode.reward = float(correct - self.config.alpha_call * episode.num_calls)
            episode.parsed_action = self._parse_action(episode, states[episode.episode_index])
        features_tensor = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(0, self.features.dimension)
        return RolloutResult(episodes=episodes, positions=positions, position_features=features_tensor)

    def _emit_call(self, state: EpisodeState, logits: torch.Tensor, episode: EpisodeRecord, instance: Instance, positions: list, rows: list, decode: str, generator, scorer, corruption_rate: float) -> int:
        # Assisted runs force the first call; later call slots may stop and answer.
        if state.calls_used == 0:
            logits[ANSWER_STOP_TOKEN] = float("-inf")
        token = self._select(logits, decode, generator)
        logprob = self._logprob(logits, token)
        record = PositionRecord(position_index=len(positions), slot=SLOT_CALL, episode_index=episode.episode_index, token=int(token), logprob=logprob, call_index=state.calls_used)
        positions.append(record)
        rows.append(self.features.vector(state))
        episode.positions.append(record.position_index)
        if token != CALL_TOKEN:
            return PHASE_ANSWER
        rng = random.Random("payload-{0}-{1}-{2}".format(state.instance_id, state.seed, state.calls_used))
        payload = self.perturbation.deliver(instance, state.calls_used, rng, corruption_rate)
        state.consumed_id = payload.consumed_id
        state.passage_ids = list(payload.passage_ids)
        state.agreeing_count = payload.agreeing_count
        state.extractable_count = payload.extractable_count
        state.calls_used += 1
        episode.calls.append(
            CallRecord(
                call_index=payload.call_index,
                payload=payload,
                budget_remaining=self.config.max_calls - payload.call_index - 1,
                elicited_candidate=state.memory_candidate_id,
                entailment=None if scorer is None else float(scorer.entailment(payload.consumed_id, state.memory_candidate_id, state.instance_id, payload.call_index)),
            )
        )
        return PHASE_DECISION

    def _emit_decision(self, state: EpisodeState, logits: torch.Tensor, reference_logits: torch.Tensor, episode: EpisodeRecord, positions: list, rows: list, decode: str, gate: str, scorer, forced_map: dict | None, generator) -> int:
        # Mask requery before selecting the learned action when no further call is allowed.
        if state.calls_used >= self.config.max_calls or self.config.no_requery:
            logits[ACTION_REQUERY] = float("-inf")
        call_record = episode.calls[-1]
        emitted = None
        if gate != "none":
            action = gate_action(gate, state, self.config, scorer)
        elif forced_map is not None and call_record.call_index in forced_map:
            action = forced_map[call_record.call_index]
        elif self.config.credit_source == "instruction":
            action = self._select(reference_logits, decode, generator)
        else:
            action = self._select(logits, decode, generator)
            emitted = int(action)
        logprob = self._logprob(logits, int(action))
        decision_position = NO_POSITION
        # Gates/instruction-only variants act without a learned decision-position record.
        if self.config.decision_slot and gate == "none" and self.config.credit_source != "instruction":
            record = PositionRecord(position_index=len(positions), slot=SLOT_DECISION, episode_index=episode.episode_index, token=int(action), logprob=logprob, call_index=call_record.call_index)
            positions.append(record)
            rows.append(self.features.vector(state))
            episode.positions.append(record.position_index)
            decision_position = record.position_index
        call_record.emitted_action = emitted
        call_record.credited_action = int(action)
        call_record.decision_position = decision_position
        state.last_action = int(action)
        if int(action) == ACTION_ADOPT:
            state.adopted_id = state.consumed_id
        elif int(action) == ACTION_OVERRIDE:
            state.adopted_id = -1
        if int(action) == ACTION_REQUERY and state.calls_used < self.config.max_calls:
            return PHASE_CALL
        return PHASE_ANSWER

    def _emit_answer(self, state: EpisodeState, logits: torch.Tensor, episode: EpisodeRecord, positions: list, rows: list, decode: str, generator) -> int:
        token = self._select(logits, decode, generator)
        logprob = self._logprob(logits, token)
        record = PositionRecord(position_index=len(positions), slot=SLOT_ANSWER, episode_index=episode.episode_index, token=int(token), logprob=logprob)
        positions.append(record)
        rows.append(self.features.vector(state))
        episode.positions.append(record.position_index)
        episode.answer_id = int(token)
        episode.answer_position = record.position_index
        return PHASE_DONE

    def _initial_state(self, instance: Instance, seed: int) -> EpisodeState:
        return EpisodeState(
            instance_id=instance.instance_id,
            family=instance.family,
            gold_id=instance.gold_id,
            memory_probs=list(instance.memory_probs),
            memory_candidate_id=instance.memory_candidate_id,
            memory_confidence=instance.memory_confidence,
            consumed_id=-1,
            passage_ids=[],
            agreeing_count=0,
            extractable_count=0,
            adopted_id=-1,
            last_action=ACTION_NO_ACTION,
            calls_used=0,
            seed=seed,
        )

    # This baseline infers an action from the final answer; it is not the emitted action.
    def _parse_action(self, episode: EpisodeRecord, state: EpisodeState) -> int | None:
        if not episode.calls:
            return None
        last_call = episode.calls[-1]
        if episode.answer_id == last_call.payload.consumed_id:
            return ACTION_ADOPT
        if episode.answer_id == state.memory_candidate_id:
            return ACTION_OVERRIDE
        return ACTION_REQUERY

    def _select(self, logits: torch.Tensor, decode: str, generator) -> int:
        valid = torch.isfinite(logits)
        if not bool(valid.any()):
            logits = torch.zeros_like(logits)
            valid = torch.ones_like(logits, dtype=torch.bool)
        masked = torch.where(valid, logits, torch.full_like(logits, float("-inf")))
        if decode == "greedy":
            return int(torch.argmax(masked).item())
        probabilities = torch.softmax(masked, dim=0)
        return int(torch.multinomial(probabilities, 1, generator=generator).item())

    def _logprob(self, logits: torch.Tensor, token: int) -> float:
        valid = torch.isfinite(logits)
        masked = torch.where(valid, logits, torch.full_like(logits, float("-inf")))
        return float(torch.log_softmax(masked, dim=0)[token].item())
