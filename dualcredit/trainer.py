import random
import time

import torch

from .arbitration import credit_calls, group_key, mean_match
from .config import Config
from .credit import RunningMatchBaseline, answer_advantages, response_advantage
from .data import TaskDataset, exact_match
from .features import ACTION_ADOPT, ACTION_OVERRIDE, ACTION_REQUERY, FeatureBuilder
from .objective import dual_channel_loss
from .perturbation import PerturbationEngine
from .policy import Policy
from .proxy import NLIScorer, ProxyEstimator
from .rollout import RolloutEngine, SLOT_DECISION

ACTIONS = (ACTION_OVERRIDE, ACTION_ADOPT, ACTION_REQUERY)


class Trainer:
    def __init__(self, config: Config, dataset: TaskDataset, seed: int | None = None):
        self.config = config
        self.dataset = dataset
        self.seed = config.seed if seed is None else seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        self.features = FeatureBuilder(config)
        self.policy = Policy(config, self.features)
        self.perturbation = PerturbationEngine(config)
        self.engine = RolloutEngine(config, self.features, self.policy, self.perturbation)
        self.scorer = NLIScorer(config)
        self.proxy = None if config.use_recorded_state else ProxyEstimator(config, self.scorer)
        self.optimizer = torch.optim.AdamW(self.policy.adapter_parameters(), lr=config.learning_rate, weight_decay=0.01)
        self.baseline = RunningMatchBaseline(centered=config.center_match)
        self.necessity: dict = {}
        self.history: list = []
        self.assisted_episodes = 0
        self.paired_episodes = 0
        self.elicitation_episodes = 0
        self.branch_episodes = 0
        self.elapsed = 0.0
        self.generator = torch.Generator().manual_seed(self.seed)

    def _sample(self, update_index: int) -> tuple:
        rng = random.Random(self.seed * 7919 + update_index)
        n_instances = max(1, self.config.episode_budget // self.config.group_size)
        chosen = rng.sample(self.dataset.train, n_instances)
        instances: list = []
        groups: list = []
        seeds: list = []
        for position, instance in enumerate(chosen):
            for member in range(self.config.group_size):
                instances.append(instance)
                groups.append(position)
                seeds.append(update_index * 131 + position * 17 + member)
        return chosen, instances, groups, seeds

    def _pair(self, chosen: list, update_index: int) -> int:
        if not self.config.pairing:
            return 0
        quota = int(round(self.config.pairing_quota * self.config.episode_budget))
        pending = [instance for instance in chosen if instance.instance_id not in self.necessity]
        selected = pending[:quota]
        if not selected:
            return 0
        seeds = [update_index * 311 + instance.instance_id for instance in selected]
        result = self.engine.run(selected, seeds, decode="greedy", unaided=True)
        for episode in result.episodes:
            success = exact_match(episode.answer_id, episode.gold_id, episode.family, self.config.num_values)
            self.necessity[episode.instance_id] = int(1 - success)
        self.paired_episodes += len(result.episodes)
        return len(result.episodes)


    def _needs_elicitation(self) -> bool:
        return self.proxy is not None or self.config.credit_source == "judge"


    def _elicit(self, chosen: list) -> dict:
        seeds = [7 + instance.instance_id for instance in chosen]
        result = self.engine.run(chosen, seeds, decode="greedy", unaided=True)
        self.elicitation_episodes += len(result.episodes)
        return {episode.instance_id: episode.answer_id for episode in result.episodes}

    def _branch_rewards(self, episodes: list, instances: list) -> dict:
        rewards: dict = {}
        emitted: dict = {}
        for episode in episodes:
            emitted[episode.episode_index] = {call.call_index: call.credited_action for call in episode.calls}
        indices = sorted({call_index for mapping in emitted.values() for call_index in mapping})
        for action in ACTIONS:
            for call_index in indices:
                forced: list = []
                seeds: list = []
                for episode in episodes:
                    mapping = dict(emitted[episode.episode_index])
                    mapping.pop(call_index, None)
                    mapping[call_index] = action
                    forced.append(mapping)
                    seeds.append(episode.seed)
                result = self.engine.run(instances, seeds, forced=forced, decode="greedy")
                for episode in result.episodes:
                    rewards.setdefault(episode.episode_index, {}).setdefault(call_index, {})[action] = episode.reward
                self.branch_episodes += len(result.episodes)
        return rewards

    def update(self, update_index: int) -> dict:
        started = time.perf_counter()
        chosen, instances, groups, seeds = self._sample(update_index)
        paired = self._pair(chosen, update_index)
        elicited = self._elicit(chosen) if self._needs_elicitation() else {}
        result = self.engine.run(
            instances,
            seeds,
            groups=groups,
            decode="sample",
            gate=self.config.gate,
            scorer=self.scorer,
            tool_access=self.config.tool_access,
            generator=self.generator,
        )
        episodes = result.episodes
        self.assisted_episodes += len(episodes)
        credits: dict = {}
        for episode, instance in zip(episodes, instances):
            credits[episode.episode_index] = credit_calls(
                config=self.config,
                episode=episode,
                necessity=self.necessity.get(episode.instance_id),
                proxy=self.proxy,
                elicited_id=elicited.get(episode.instance_id, -1),
                scorer=self.scorer,
                memory_confidence=instance.memory_confidence,
            )
        keys = [group_key(self.config, episode, credits[episode.episode_index], instance) for episode, instance in zip(episodes, instances)]
        rewards = [episode.reward for episode in episodes]
        advantages, fallbacks = answer_advantages(rewards, keys, self.config.lam, self.config.cell_floor, self.config.normalize)
        episode_matches = [mean_match(credits[episode.episode_index]) for episode in episodes if credits[episode.episode_index]]
        self.baseline.update(sum(episode_matches) / max(1, len(episode_matches)), self.config.xi)
        baseline_value = self.baseline.baseline()
        branch_rewards = self._branch_rewards(episodes, instances) if self.config.counterfactual_branches else None
        call_advantages: dict = {}
        for episode, answer_adv in zip(episodes, advantages):
            for credit in credits[episode.episode_index]:
                key = (episode.episode_index, credit.call_index)
                if branch_rewards is not None:
                    table = branch_rewards.get(episode.episode_index, {})
                    values = list(table.get(credit.call_index, {}).values())
                    mean_value = sum(values) / len(values) if values else 0.0
                    call_advantages[key] = table.get(credit.call_index, {}).get(credit.action, mean_value) - mean_value
                else:
                    call_advantages[key] = response_advantage(answer_adv, credit.match, baseline_value, self.config.gamma, self.config.decision_credit)
        positions = result.positions
        decision_keys: list = []
        for position in positions:
            if position.slot == SLOT_DECISION and self.config.decision_credit:
                decision_keys.append((position.episode_index, position.call_index))
        response_values = [call_advantages[key] for key in decision_keys if key in call_advantages]
        response_offset = (sum(response_values) / len(response_values)) if (self.config.response_center and response_values) else 0.0
        advantages_tensor = torch.zeros(len(positions), dtype=torch.float32)
        response_mask = torch.zeros(len(positions), dtype=torch.bool)
        for index, position in enumerate(positions):
            episode = episodes[position.episode_index]
            if position.slot == SLOT_DECISION and self.config.decision_credit:
                advantages_tensor[index] = float(call_advantages.get((episode.episode_index, position.call_index), advantages[position.episode_index])) - response_offset
                response_mask[index] = True
            else:
                advantages_tensor[index] = float(advantages[position.episode_index])
        rows = result.position_features
        new_call, new_decision, new_answer = self.policy(rows)
        with torch.no_grad():
            ref_call, ref_decision, ref_answer = self.policy.reference(rows)
        report = dual_channel_loss(
            new_call=new_call,
            new_decision=new_decision,
            new_answer=new_answer,
            ref_call=ref_call,
            ref_decision=ref_decision,
            ref_answer=ref_answer,
            positions=positions,
            advantages=advantages_tensor,
            response_mask=response_mask,
            delta=self.config.clip_delta,
            kl_beta=self.config.kl_beta,
            response_delta=self.config.clip_delta if self.config.response_clip else None,
            entropy_beta=self.config.entropy_beta if self.config.decision_credit else 0.0,
        )
        self.optimizer.zero_grad(set_to_none=True)
        if bool(report["loss"].requires_grad):
            report["loss"].backward()
            self.optimizer.step()
        successes = [int(exact_match(episode.answer_id, episode.gold_id, episode.family, self.config.num_values)) for episode in episodes]
        all_credits = [credit for episode in episodes for credit in credits[episode.episode_index]]
        self.elapsed = time.perf_counter() - started
        diagnostics = {
            "update": update_index,
            "reward": sum(rewards) / max(1, len(rewards)),
            "em": 100.0 * sum(successes) / max(1, len(successes)),
            "adopt": 100.0 * sum(int(credit.action == ACTION_ADOPT) for credit in all_credits) / max(1, len(all_credits)),
            "dec_acc": 100.0 * sum(credit.match for credit in all_credits if credit.decided) / max(1, sum(credit.decided for credit in all_credits)),
            "cov": 100.0 * sum(int(credit.decided == 0) for credit in all_credits) / max(1, len(all_credits)),
            "calls": sum(episode.num_calls for episode in episodes) / max(1, len(episodes)),
            "baseline": baseline_value,
            "fallback": 100.0 * sum(fallbacks) / max(1, len(fallbacks)),
            "loss": float(report["loss"].detach().item()),
            "kl": float(report["kl"].detach().item()),
            "entropy": float(report["entropy"].detach().item()),
            "shift": float(report["shift"].detach().item()),
            "paired": paired,
            "elapsed": self.elapsed,
        }
        self.history.append(diagnostics)
        return diagnostics

    def train(self) -> list:
        for update_index in range(self.config.updates):
            diagnostics = self.update(update_index)
            if (update_index + 1) % max(1, self.config.log_every) == 0:
                print(
                    "update {0:3d}  EM {1:6.2f}  Adm {2:6.2f}  Dec {3:6.2f}  Cov {4:6.2f}  mbar {5:6.4f}  fallback {6:6.2f}  loss {7:8.4f}  elapsed {8:6.2f}s".format(
                        update_index + 1,
                        diagnostics["em"],
                        diagnostics["adopt"],
                        diagnostics["dec_acc"],
                        diagnostics["cov"],
                        diagnostics["baseline"],
                        diagnostics["fallback"],
                        diagnostics["loss"],
                        diagnostics["elapsed"],
                    )
                )
        return self.history

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.policy.adapter_parameters())

    def frozen_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.policy.parameters() if not parameter.requires_grad)
