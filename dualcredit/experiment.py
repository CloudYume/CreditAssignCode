import csv
import json
import time
from pathlib import Path

import torch

from .arbitration import credit_calls
from .config import Config, arm_config
from .credit import normative_actions
from .data import TaskDataset, exact_match
from .evaluation import Harness
from .features import FeatureBuilder
from .perturbation import PerturbationEngine
from .policy import Policy
from .rollout import SLOT_DECISION, RolloutEngine
from .trainer import Trainer

FAMILIES = ("retrieval", "execution")

MAIN_ARMS = (
    "no_component",
    "caution_prompt",
    "corruption_hint",
    "grpo_clean",
    "grpo_corrupt",
    "boundary_reward",
    "shortcut_suppressed",
    "trust_gate",
    "agreement_gate",
    "abstain_gate",
    "stratified_norm",
    "ours_rec",
    "ours_proxy",
)

ABLATION_ARMS = (
    "ours_rec",
    "ours_proxy",
    "ablate_lam1",
    "ablate_lam0",
    "ablate_floor8",
    "ablate_gamma0",
    "ablate_gamma05",
    "ablate_uncentered",
    "ablate_response_unclipped",
    "ablate_response_raw",
    "ablate_xi095",
    "ablate_pairing0",
    "ablate_pairing015",
    "ablate_pairing050",
    "ablate_vartheta0",
    "ablate_vartheta050",
    "ablate_budget_j",
    "ablate_group4",
    "ablate_group16",
)

PROXY_ARMS = (
    "ours_rec",
    "ours_proxy",
    "proxy_no_agreement",
    "proxy_no_corroboration",
    "proxy_kappa3",
    "proxy_tau07",
    "proxy_tau03",
    "proxy_budget_j",
    "proxy_single_call",
)

REPRESENTATION_ARMS = (
    "ours_rec",
    "ablate_gamma0",
    "repr_parsed",
    "repr_judge",
    "repr_instruction",
    "ablate_uncentered",
)

COST_ARMS = (
    "grpo_corrupt",
    "stratified_norm",
    "counterfactual",
    "ours_rec",
    "ours_proxy",
)

BOUNDARY_RATES = (0.0, 0.2, 0.4, 0.6)
MODE_ARMS = {
    "main": MAIN_ARMS,
    "ablation": ABLATION_ARMS,
    "proxy": PROXY_ARMS,
    "representation": REPRESENTATION_ARMS,
    "cost": COST_ARMS,
}

SHARED_KEYS = ("em", "unaided_em", "theta1", "theta0", "crossing", "adoption", "override", "requery", "dec_acc", "cov", "calls", "compliance", "episodes", "credits")
CELL_KEYS = tuple(prefix + str(cell) for cell in range(4) for prefix in ("occ_s", "adm_s", "ovr_s", "req_s", "dec_s"))
TRAIN_KEYS = ("updates", "episodes", "episodes_per_update", "assisted_episodes", "paired_episodes", "elicitation_episodes", "branch_episodes", "trainable", "frozen")


def supported_action(cell, budget_remaining: int):
    targets = normative_actions(cell, budget_remaining)
    return targets[0] if len(targets) == 1 else None


def probe_pairs(config: Config, dataset: TaskDataset, policy: Policy, corruption_rate: float, limit: int | None = None):
    features = FeatureBuilder(config)
    engine = RolloutEngine(config, features, policy, PerturbationEngine(config))
    harness = Harness(config, dataset, policy)
    pool = dataset.test
    instances = pool[: max(1, min(config.eval_batch if limit is None else limit, len(pool)))]
    seeds = [17 + instance.instance_id for instance in instances]
    unaided = engine.run(instances, seeds, decode="greedy", unaided=True)
    necessity = {episode.instance_id: int(1 - exact_match(episode.answer_id, episode.gold_id, episode.family, config.num_values)) for episode in unaided.episodes}
    elicited = {episode.instance_id: episode.answer_id for episode in unaided.episodes}
    assisted = engine.run(instances, [5 + instance.instance_id + int(1000 * corruption_rate) for instance in instances], decode="greedy", corruption_rate=corruption_rate, scorer=harness.scorer)
    pairs: list = []
    for episode in assisted.episodes:
        credits = credit_calls(
            config=config,
            episode=episode,
            necessity=necessity.get(episode.instance_id),
            proxy=None,
            elicited_id=elicited.get(episode.instance_id, -1),
            scorer=harness.scorer,
            memory_confidence=harness.confidence.get(episode.instance_id, 0.0),
        )
        decisions = [position for position in assisted.positions if position.episode_index == episode.episode_index and position.slot == SLOT_DECISION]
        for credit, position in zip(credits, decisions):
            action = supported_action(credit.cell, credit.budget_remaining)
            if action is not None:
                pairs.append((position.position_index, action))
    return assisted.position_features, pairs

def measure_shift(base: Config, arm: str, dataset: TaskDataset, corruption_rate: float) -> dict:
    config = arm_config(base, arm)
    policy = Policy(config, FeatureBuilder(config))
    rows, pairs = probe_pairs(config, dataset, policy, corruption_rate)
    with torch.no_grad():
        _, logits, _ = policy(rows)
    before = [float(logits[index, action].item()) for (index, action) in pairs]
    trainer = Trainer(config, dataset)
    trainer.policy = policy
    trainer.engine.policy = policy
    trainer.optimizer = torch.optim.AdamW(trainer.policy.adapter_parameters(), lr=config.learning_rate, weight_decay=0.0)
    trainer.train()
    with torch.no_grad():
        _, after_logits, _ = policy(rows)
    after = [float(after_logits[index, action].item()) for (index, action) in pairs]
    shift = sum(after_value - before_value for after_value, before_value in zip(after, before)) / max(1, len(pairs))
    return {"shift": shift, "decided": len(pairs), "elapsed": trainer.elapsed}


def run_arm(base: Config, arm: str, family: str) -> dict:
    config = arm_config(Config(**{**base.__dict__, "family": family}), arm)
    dataset = TaskDataset(config)
    trainer = Trainer(config, dataset, seed=config.seed)
    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started
    harness = Harness(config, dataset, trainer.policy)
    reports = {}
    for rate in config.evaluation_corruption_rates:
        report = harness.evaluate(rate)
        reports[str(rate)] = {key: report[key] for key in SHARED_KEYS + CELL_KEYS}
        reports[str(rate)]["proxy"] = report["proxy"]
    episodes = trainer.assisted_episodes + trainer.paired_episodes + trainer.elicitation_episodes + trainer.branch_episodes
    updates = max(1, config.updates)
    return {
        "arm": arm,
        "family": family,
        "config": config,
        "train": {
            "seconds": train_seconds,
            "updates": config.updates,
            "episodes": episodes,
            "episodes_per_update": episodes / updates,
            "assisted_episodes": trainer.assisted_episodes,
            "paired_episodes": trainer.paired_episodes,
            "elicitation_episodes": trainer.elicitation_episodes,
            "branch_episodes": trainer.branch_episodes,
            "trainable": trainer.trainable_parameters(),
            "frozen": trainer.frozen_parameters(),
            "mean_update_seconds": trainer.elapsed,
        },
        "reports": reports,
    }


def run_mode(base: Config, mode: str, families=FAMILIES, log=print) -> list:
    arms = MODE_ARMS[mode]
    rows: list = []
    for family in families:
        for arm in arms:
            row = run_arm(base, arm, family)
            rows.append(row)
            log("{0} {1} {2}".format(mode, family, arm))
    return rows


def clean_report(row: dict, rate: float) -> dict:
    return row["reports"][str(rate)]


def dirty_rate(base: Config) -> float:
    return max(base.evaluation_corruption_rates)


def render(columns: list, rows: list) -> str:
    widths = []
    for title, key in columns:
        width = len(title)
        for row in rows:
            width = max(width, len(_cell(row, key)))
        widths.append(width)
    header = "  ".join(title.ljust(widths[index]) for index, (title, _) in enumerate(columns))
    lines = [header, "  ".join("-" * width for width in widths)]
    for row in rows:
        lines.append("  ".join(_cell(row, key).ljust(widths[index]) for index, (_, key) in enumerate(columns)))
    return "\n".join(lines)


def _cell(row: dict, key) -> str:
    value = row.get(key, "")
    if isinstance(value, float):
        return "{0:.2f}".format(value)
    return str(value)


def main_columns(families=FAMILIES) -> list:
    columns: list = [("Arm", "arm")]
    for family in families:
        columns.append((family + " EM1", family + "_em1"))
        columns.append((family + " EM07", family + "_em07"))
        columns.append((family + " Adm s1", family + "_adm1"))
        columns.append((family + " Adm s3", family + "_adm3"))
    columns.append(("Dec acc", "dec"))
    columns.append(("Cov", "cov"))
    columns.append(("Calls", "calls"))
    return columns


def main_rows(rows: list, base: Config) -> list:
    rate = dirty_rate(base)
    grouped: dict = {}
    for row in rows:
        entry = grouped.setdefault(row["arm"], {"arm": row["arm"]})
        clean = clean_report(row, 0.0)
        dirty = clean_report(row, rate)
        entry[row["family"] + "_em1"] = clean["em"]
        entry[row["family"] + "_em07"] = dirty["em"]
        entry[row["family"] + "_adm1"] = dirty["adm_s1"]
        entry[row["family"] + "_adm3"] = dirty["adm_s3"]
        entry["dec"] = dirty["dec_acc"]
        entry["cov"] = dirty["cov"]
        entry["calls"] = dirty["calls"]
    return [grouped[arm] for arm in MAIN_ARMS if arm in grouped]


def render_main(rows: list, base: Config) -> str:
    families = tuple(dict.fromkeys(row["family"] for row in rows))
    return render(main_columns(families), main_rows(rows, base))


def render_ablation(rows: list, base: Config) -> str:
    rate = dirty_rate(base)
    reference = None
    table: list = []
    for row in rows:
        if row["family"] != FAMILIES[0]:
            continue
        clean = clean_report(row, 0.0)
        dirty = clean_report(row, rate)
        if row["arm"] == "ours_rec":
            reference = dirty["em"]
        table.append({
            "arm": row["arm"],
            "em1": clean["em"],
            "em07": dirty["em"],
            "adm1": dirty["adm_s1"],
            "adm3": dirty["adm_s3"],
            "dec": dirty["dec_acc"],
            "cov": dirty["cov"],
            "_delta": dirty["em"],
        })
    for entry in table:
        entry["delta"] = 0.0 if reference is None else entry["_delta"] - reference
    columns = [("Arm", "arm"), ("EM eta=1", "em1"), ("EM eta=.7", "em07"), ("Adm s=1", "adm1"), ("Adm s=3", "adm3"), ("Dec acc", "dec"), ("Cov", "cov"), ("Delta EM", "delta")]
    return render(columns, table)


def render_proxy(rows: list, base: Config) -> str:
    rate = dirty_rate(base)
    table: list = []
    for row in rows:
        if row["family"] != FAMILIES[0]:
            continue
        dirty = clean_report(row, rate)
        proxy = dirty["proxy"]
        table.append({
            "arm": row["arm"],
            "precision": proxy["precision"],
            "coverage": proxy["coverage"],
            "bias": proxy["bias"],
            "adm1": dirty["adm_s1"],
            "em": dirty["em"],
            "adm3": dirty["adm_s3"],
        })
    columns = [("Estimator", "arm"), ("Prec", "precision"), ("Cov", "coverage"), ("Bias", "bias"), ("Adm s=1", "adm1"), ("EM", "em"), ("Adm s=3", "adm3")]
    return render(columns, table)


def render_representation(rows: list, base: Config) -> str:
    rate = dirty_rate(base)
    table: list = []
    for row in rows:
        if row["family"] != FAMILIES[0]:
            continue
        dirty = clean_report(row, rate)
        table.append({
            "arm": row["arm"],
            "compliance": dirty["compliance"],
            "dec": dirty["dec_acc"],
            "adm1": dirty["adm_s1"],
            "adm3": dirty["adm_s3"],
            "shift": ((row["shift"]["shift"] if row["shift"].get("decided", 0) else "---") if "shift" in row else ""),
        })
    columns = [("Variant", "arm"), ("Compl.", "compliance"), ("Dec acc", "dec"), ("Adm s=1", "adm1"), ("Adm s=3", "adm3"), ("Shift", "shift")]
    return render(columns, table)


def render_cost(rows: list, base: Config) -> str:
    table: list = []
    for row in rows:
        if row["family"] != FAMILIES[0]:
            continue
        train = row["train"]
        passes = []
        if train["paired_episodes"]:
            passes.append("pairing")
        if train["elicitation_episodes"]:
            passes.append("elicitation")
        if train["branch_episodes"]:
            passes.append("branches")
        table.append({
            "arm": row["arm"],
            "episodes": train["episodes_per_update"],
            "passes": ",".join(passes) if passes else "none",
            "trainable": train["trainable"],
            "frozen": train["frozen"],
            "seconds": train["seconds"],
        })
    columns = [("Arm", "arm"), ("Episodes/update", "episodes"), ("Extra passes", "passes"), ("Trainable params", "trainable"), ("Frozen params", "frozen"), ("Train seconds", "seconds")]
    return render(columns, table)


def render_boundary(rows: list, base: Config) -> str:
    table: list = []
    for row in rows:
        entry = {"arm": row["arm"], "family": row["family"], "rate": row["boundary_rate"]}
        entry.update({key: row["report"][key] for key in ("em", "adm_s1", "adm_s3", "dec_acc", "cov", "calls")})
        table.append(entry)
    columns = [("Arm", "arm"), ("Family", "family"), ("eta corruption", "rate"), ("EM", "em"), ("Adm s=1", "adm_s1"), ("Adm s=3", "adm_s3"), ("Dec acc", "dec_acc"), ("Cov", "cov"), ("Calls", "calls")]
    return render(columns, table)


def run_boundary(base: Config, arms=("grpo_corrupt", "stratified_norm", "ours_rec", "ours_proxy"), families=FAMILIES, log=print) -> list:
    rows: list = []
    for family in families:
        for arm in arms:
            config = arm_config(Config(**{**base.__dict__, "family": family, "evaluation_corruption_rates": tuple(BOUNDARY_RATES)}), arm)
            dataset = TaskDataset(config)
            trainer = Trainer(config, dataset, seed=config.seed)
            trainer.train()
            harness = Harness(config, dataset, trainer.policy)
            for rate in BOUNDARY_RATES:
                report = harness.evaluate(rate)
                rows.append({"arm": arm, "family": family, "boundary_rate": rate, "report": report})
            log("boundary {0} {1}".format(family, arm))
    return rows



def representation_rows(base: Config, families=FAMILIES, log=print) -> list:
    rows = run_mode(base, "representation", families=families, log=log)
    for row in rows:
        family_base = Config(**{**base.__dict__, "family": row["family"]})
        family_dataset = TaskDataset(arm_config(family_base, row["arm"]))
        row["shift"] = measure_shift(family_base, row["arm"], family_dataset, dirty_rate(base))
    return rows


def to_json(payload: dict, path) -> None:
    serialisable = _plain(payload)
    Path(path).write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


def to_csv(columns: list, rows: list, path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([title for (title, _) in columns])
        for row in rows:
            writer.writerow([_cell(row, key) for (_, key) in columns])


def _plain(value):
    if isinstance(value, Config):
        return value.__dict__
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def run(mode: str, base: Config, output_dir=None, families=FAMILIES, log=print) -> dict:
    payload: dict = {"mode": mode, "families": list(families)}
    if mode == "boundary":
        rows = run_boundary(base, families=families, log=log)
        payload["boundary"] = _plain(rows)
        text = render_boundary(rows, base)
    elif mode == "representation":
        rows = representation_rows(base, families=families, log=log)
        payload["rows"] = _plain(rows)
        text = render_representation(rows, base)
    else:
        rows = run_mode(base, mode, families=families, log=log)
        payload["rows"] = _plain(rows)
        text = {
            "main": render_main,
            "ablation": render_ablation,
            "proxy": render_proxy,
            "cost": render_cost,
        }[mode](rows, base)
    payload["table"] = text
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        to_json(payload, directory / (mode + ".json"))
        (directory / (mode + ".txt")).write_text(text, encoding="utf-8")
    return payload
