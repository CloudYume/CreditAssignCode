# CLI entry point for synthetic experiments and reduced-size smoke runs.
# CLI overrides are applied after the smoke preset and before experiment-arm overrides.

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dualcredit.config import Config
from dualcredit.experiment import FAMILIES, run

MODES = ("smoke", "main", "ablation", "proxy", "representation", "cost", "boundary", "all")
SMOKE_MODES = ("main", "ablation", "proxy", "representation", "cost")


def parse_families(value: str) -> tuple:
    families = tuple(item.strip() for item in value.split(",") if item.strip())
    for family in families:
        if family not in FAMILIES:
            raise argparse.ArgumentTypeError("unknown family: " + family)
    return families


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-channel credit assignment reference experiments")
    parser.add_argument("--mode", choices=MODES, default="smoke")
    parser.add_argument("--families", type=parse_families, default=FAMILIES)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--episode-budget", type=int, default=None)
    parser.add_argument("--eval-batch", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> Config:
    config = Config()
    # Smoke uses smaller settings; it still exercises multiple experiment suites.
    if args.mode == "smoke":
        config = Config(**{**config.__dict__, "dataset_size": 200, "episode_budget": 64, "eval_batch": 64, "updates": 25, "log_every": 100})
    overrides: dict = {}
    for name, value in (
        ("seed", args.seed),
        ("dataset_size", args.dataset_size),
        ("episode_budget", args.episode_budget),
        ("eval_batch", args.eval_batch),
        ("updates", args.updates),
        ("learning_rate", args.learning_rate),
    ):
        if value is not None:
            overrides[name] = value
    if overrides:
        config = Config(**{**config.__dict__, **overrides})
    return config


def run_mode(mode: str, config: Config, families: tuple, output: str | None, quiet: bool) -> None:
    log = (lambda message: None) if quiet else print
    started = time.perf_counter()
    payload = run(mode, config, output_dir=output, families=families, log=log)
    elapsed = time.perf_counter() - started
    print("=== {0} ({1:.1f}s) ===".format(mode, elapsed))
    print(payload["table"])
    print()


def main() -> int:
    args = build_parser().parse_args()
    config = build_config(args)
    families = args.families
    # Smoke prints tables without writing the normal per-mode result directories.
    if args.mode == "smoke":
        for mode in SMOKE_MODES:
            run_mode(mode, config, families, None, args.quiet)
        return 0
    modes = ("main", "ablation", "proxy", "representation", "cost", "boundary") if args.mode == "all" else (args.mode,)
    for mode in modes:
        if args.output is None:
            output = str(Path("results") / mode)
        elif len(modes) > 1:
            output = str(Path(args.output) / mode)
        else:
            output = args.output
        run_mode(mode, config, families, output, args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
