import time

from .arbitration import credit_calls
from .config import Config
from .data import TaskDataset, exact_match
from .features import ACTION_ADOPT, ACTION_OVERRIDE, ACTION_REQUERY, FeatureBuilder
from .perturbation import PerturbationEngine
from .policy import Policy
from .proxy import NLIScorer, ProxyEstimator
from .rollout import RolloutEngine, SLOT_DECISION

CELL_LABELS = (0, 1, 2, 3)


class Harness:
    def __init__(self, config: Config, dataset: TaskDataset, policy: Policy | None = None):
        self.config = config
        self.dataset = dataset
        self.features = FeatureBuilder(config)
        self.policy = Policy(config, self.features) if policy is None else policy
        self.perturbation = PerturbationEngine(config)
        self.engine = RolloutEngine(config, self.features, self.policy, self.perturbation)
        self.scorer = NLIScorer(config)
        self.proxy = ProxyEstimator(config, self.scorer)
        self.confidence = {instance.instance_id: instance.memory_confidence for instance in dataset.instances}
        self.elapsed = 0.0
        self.episodes = 0

    def evaluate(self, corruption_rate: float, split: str = "test", instance_limit: int | None = None, use_proxy: bool | None = None) -> dict:
        started = time.perf_counter()
        pool = self.dataset.test if split == "test" else self.dataset.train
        limit = self.config.eval_batch if instance_limit is None else instance_limit
        instances = pool[: max(1, min(limit, len(pool)))]
        unaided = self.engine.run(instances, [17 + instance.instance_id for instance in instances], decode="greedy", unaided=True)
        self.episodes += len(unaided.episodes)
        necessity: dict = {}
        elicited: dict = {}
        unaided_success: dict = {}
        for episode in unaided.episodes:
            success = exact_match(episode.answer_id, episode.gold_id, episode.family, self.config.num_values)
            necessity[episode.instance_id] = int(1 - success)
            elicited[episode.instance_id] = episode.answer_id
            unaided_success[episode.instance_id] = success
        proxy_mode = (self.config.use_recorded_state is False) if use_proxy is None else use_proxy
        active_proxy = self.proxy if proxy_mode else None
        assisted = self.engine.run(
            instances,
            [5 + instance.instance_id + int(1000 * corruption_rate) for instance in instances],
            decode="greedy",
            gate=self.config.gate,
            scorer=self.scorer,
            tool_access=self.config.tool_access,
            corruption_rate=corruption_rate,
        )
        self.episodes += len(assisted.episodes)
        episodes = assisted.episodes
        credits: dict = {}
        recorded: dict = {}
        estimated: dict = {}
        for episode in episodes:
            common = {
                "config": self.config,
                "episode": episode,
                "necessity": necessity.get(episode.instance_id),
                "elicited_id": elicited.get(episode.instance_id, -1),
                "scorer": self.scorer,
                "memory_confidence": self.confidence.get(episode.instance_id, 0.0),
            }
            credits[episode.episode_index] = credit_calls(**{**common, "proxy": active_proxy})
            recorded[episode.episode_index] = credit_calls(**{**common, "proxy": None})
            estimated[episode.episode_index] = credit_calls(**{**common, "proxy": self.proxy})
        decision_counts = sum(1 for position in assisted.positions if position.slot == SLOT_DECISION)
        call_counts = sum(episode.num_calls for episode in episodes)
        all_credits = [credit for episode in episodes for credit in credits[episode.episode_index]]
        answered = [int(exact_match(episode.answer_id, episode.gold_id, episode.family, self.config.num_values)) for episode in episodes]
        last_recorded = {episode.episode_index: (recorded[episode.episode_index][-1] if recorded[episode.episode_index] else None) for episode in episodes}
        theta_one = [
            int(exact_match(episode.answer_id, episode.gold_id, episode.family, self.config.num_values))
            for episode in episodes
            if episode.num_calls > 0 and last_recorded[episode.episode_index] is not None and last_recorded[episode.episode_index].payload_state == 1
        ]
        theta_zero = [
            int(exact_match(episode.answer_id, episode.gold_id, episode.family, self.config.num_values))
            for episode in episodes
            if episode.num_calls > 0 and last_recorded[episode.episode_index] is not None and last_recorded[episode.episode_index].payload_state == 0
        ]
        theta = 100.0 * sum(unaided_success.values()) / max(1, len(unaided_success))
        theta_one_value = 100.0 * sum(theta_one) / max(1, len(theta_one))
        theta_zero_value = 100.0 * sum(theta_zero) / max(1, len(theta_zero))
        crossing = None
        if theta_one and theta_zero and theta_one_value > theta_zero_value:
            crossing = (theta - theta_zero_value) / (theta_one_value - theta_zero_value)
        decided = [credit for credit in all_credits if credit.decided]
        report = {
            "corruption_rate": corruption_rate,
            "em": 100.0 * sum(answered) / max(1, len(answered)) if answered else 0.0,
            "unaided_em": theta,
            "theta1": theta_one_value,
            "theta0": theta_zero_value,
            "crossing": crossing,
            "adoption": 100.0 * sum(int(credit.action == ACTION_ADOPT) for credit in all_credits) / max(1, len(all_credits)) if all_credits else 0.0,
            "override": 100.0 * sum(int(credit.action == ACTION_OVERRIDE) for credit in all_credits) / max(1, len(all_credits)) if all_credits else 0.0,
            "requery": 100.0 * sum(int(credit.action == ACTION_REQUERY) for credit in all_credits) / max(1, len(all_credits)) if all_credits else 0.0,
            "dec_acc": 100.0 * sum(credit.match for credit in decided) / len(decided) if decided else 0.0,
            "cov": 100.0 * (len(all_credits) - len(decided)) / len(all_credits) if all_credits else 0.0,
            "calls": call_counts / max(1, len(episodes)),
            "compliance": 100.0 * decision_counts / call_counts if call_counts else 0.0,
            "episodes": len(episodes),
            "credits": len(all_credits),
            "elapsed": time.perf_counter() - started,
            "proxy": self._proxy_report(episodes, estimated, recorded),
        }
        report.update(self._cell_metrics(all_credits))
        self.elapsed = time.perf_counter() - started
        return report

    def _proxy_report(self, episodes: list, estimated: dict, recorded: dict) -> dict:
        pairs: list = []
        for episode in episodes:
            for left, right in zip(estimated[episode.episode_index], recorded[episode.episode_index]):
                pairs.append((left, right))
        if not pairs:
            return {"precision": 0.0, "coverage": 0.0, "bias": 0.0, "positives": 0, "scored": 0, "adoption_s1": 0.0}
        decidable = [(left, right) for (left, right) in pairs if left.payload_state is not None]
        positives = [(left, right) for (left, right) in decidable if left.payload_state == 1]
        precision = 100.0 * sum(int(right.payload_state == 1) for (_, right) in positives) / len(positives) if positives else 0.0
        coverage = 100.0 * len(decidable) / len(pairs)
        cell_one = [(left, right) for (left, right) in decidable if right.cell == 1]
        bias = 0.0
        if cell_one:
            predicted = 100.0 * sum(int(left.payload_state == 1) for (left, _) in cell_one) / len(cell_one)
            actual = 100.0 * sum(int(right.payload_state == 1) for (_, right) in cell_one) / len(cell_one)
            bias = abs(predicted - actual)
        return {
            "precision": precision,
            "coverage": coverage,
            "bias": bias,
            "positives": len(positives),
            "scored": len(pairs),
            "cell_one_calls": len(cell_one),
            "adoption_s1": 100.0 * sum(int(left.action == ACTION_ADOPT) for (left, right) in pairs if right.cell == 1) / max(1, sum(1 for (_, right) in pairs if right.cell == 1)),
        }

    def _cell_metrics(self, credits: list) -> dict:
        out: dict = {}
        total = max(1, len(credits))
        for cell in CELL_LABELS:
            group = [credit for credit in credits if credit.cell == cell]
            out["occ_s{0}".format(cell)] = 100.0 * len(group) / total
            out["adm_s{0}".format(cell)] = 100.0 * sum(int(credit.action == ACTION_ADOPT) for credit in group) / len(group) if group else 0.0
            out["ovr_s{0}".format(cell)] = 100.0 * sum(int(credit.action == ACTION_OVERRIDE) for credit in group) / len(group) if group else 0.0
            out["req_s{0}".format(cell)] = 100.0 * sum(int(credit.action == ACTION_REQUERY) for credit in group) / len(group) if group else 0.0
            out["dec_s{0}".format(cell)] = 100.0 * sum(credit.match for credit in group) / len(group) if group else 0.0
        return out
