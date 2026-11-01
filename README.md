# CreditAssignCode

![Method](https://img.shields.io/badge/Method-Dual--Channel_Credit_Assignment-orange?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red?style=flat-square)

A synthetic reference implementation of dual-channel credit assignment for *Trust the Return, or Trust Yourself? Identifiable Payload States for Agentic Reinforcement Learning*.

The implementation separates the reward for completing a task from credit for deciding how to use a tool response. It provides controlled retrieval and execution task families, corrupted payloads, explicit adoption actions, training variants, and evaluation reports.

The current policy is a small tensor model with frozen random features, configurable trainable heads, and low-rank adapters. Tasks, tool returns, and NLI-style scores are simulated locally. Running this repository does not require an API key, a pretrained language model, or an external search/execution service.

[Code walkthrough](docs/code-walkthrough.md) · [Configuration](dualcredit/config.py) · [Experiment entry point](run_experiment.py) · [Tests](tests/)

## Quick start

Use Python 3.10+ and run commands from the repository root. For a minimal CPU environment:

```bash
python -m venv .venv
```

Activate the environment in your shell:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

Install the runtime and test dependencies, then check the implementation:

```bash
python -m pip install "torch>=2.0,<3.0" "pytest>=7,<10"
python -m pytest tests -q
python run_experiment.py --help
python run_experiment.py --mode smoke
```

The [full dependency manifest](requirements.txt) pins PyTorch 2.7.1 and adds the CUDA 12.6 wheel index, test tools, and analysis libraries. Use `python -m pip install -r requirements.txt` when that environment matches your platform. The minimal installation above is sufficient for the checked-in framework and tests; the code currently creates CPU tensors and does not automatically select a GPU.

## Run experiments

`smoke` runs reduced-size versions of the main, ablation, proxy, representation, and cost suites. It prints tables without creating result files. To make an even smaller pipeline check:

```bash
python run_experiment.py --mode smoke --dataset-size 80 --episode-budget 16 --eval-batch 16 --updates 1 --seed 1234
```

For the default main comparison or a specific task family:

```bash
python run_experiment.py --mode main
python run_experiment.py --mode proxy --families retrieval --output results/proxy-check
```

Available modes:

- `main`: compare the method with no-tool, gate, reward, and policy-training variants.
- `ablation`: vary normalization, response credit, pairing, corruption, and rollout budgets.
- `proxy`: vary the simulated payload-state estimator and its evidence thresholds.
- `representation`: compare emitted, parsed, judged, and instruction-only action representations.
- `cost`: count assisted and auxiliary training episodes, parameters, and elapsed time.
- `boundary`: evaluate a fixed set of arms at several corruption rates.
- `all`: execute all six full suites.

Non-smoke modes write a readable `<mode>.txt` table and detailed `<mode>.json` records to `results/<mode>/`. `--output PATH` replaces the output directory; for `all`, each suite receives its own subdirectory. A small smoke run checks execution, not the paper's reported results.

Other CLI options include `--families retrieval,execution`, `--updates`, `--dataset-size`, `--episode-budget`, `--eval-batch`, `--seed`, and `--learning-rate`. Defaults live in `Config`; named arms then apply their explicit overrides through `arm_config()`. Review the effective configuration saved in each result record when comparing runs.

## Core concepts

The terminal reward balances task success against tool cost:

```text
R = 1[answer is correct] - alpha_call * number_of_calls
```

Two labels define the credit quadrant:

```text
q = 1 - 1[unaided answer is correct]
z = 1[consumed payload equals the correct value]
s = q + 2*z
```

- `s=0` (`q=0, z=0`): override the invalid payload.
- `s=1` (`q=1, z=0`): requery when another call remains available; otherwise there is no unique target.
- `s=2` (`q=0, z=1`): no unique target is assigned.
- `s=3` (`q=1, z=1`): adopt the valid payload.

Action IDs are `override=0`, `adopt=1`, and `requery=2`. The rollout records the decision separately from the call and final answer. Assisted runs force an initial call; unaided runs proceed directly to an answer.

The answer advantage interpolates pooled and quadrant-normalized rewards, falling back to pooled statistics for small or missing quadrants. In this implementation, normalization uses the complete update batch; recorded sampling-group IDs do not create separate normalization groups. The response advantage adds an action-match term:

```text
A_response = A_answer + gamma * (match - running_match_baseline)
```

The trainer optionally centers response advantages, then applies the separate answer and response objectives. Recorded `q` and payload validity support training/evaluation credit; they are not directly encoded in policy features. Proxy variants estimate `z` from candidate agreement and corroboration using a seeded synthetic scorer. See the [walkthrough](docs/code-walkthrough.md) for the full data flow and metric definitions.

## Repository layout and reading order

```text
CreditAssignCode/
  dualcredit/
    config.py        Settings and comparison-arm overrides
    data.py          Synthetic tasks and train/test split
    perturbation.py  Controlled payload corruption and source agreement
    features.py      Observable policy inputs and action constants
    policy.py        Tensor policy, adapters, heads, and reference path
    rollout.py       Call -> decision -> answer state machine
    gates.py         Hand-written inference comparison rules
    proxy.py         Simulated payload-state estimation
    credit.py        Quadrants, normalized rewards, and shaped advantages
    arbitration.py   Per-call credit attribution
    objective.py     Channel losses, clipping, KL, and entropy
    trainer.py       Sampling, credit construction, and optimization
    evaluation.py    Task, decision, quadrant, and proxy metrics
    experiment.py    Experiment suites and result serialization
  docs/
    code-walkthrough.md
  tests/             Unit and end-to-end regression checks
  conftest.py        Checkout import setup for pytest
  run_experiment.py  Command-line entry point
  requirements.txt   Full dependency manifest
```

Start with `config.py`, `data.py`, and `features.py`, then follow one episode through `rollout.py`. Read `credit.py` and `arbitration.py` before `trainer.py` and `objective.py`. Finish with `evaluation.py` to understand which population each reported metric describes.

## License

Original code and documentation in this repository are available under the [MIT License](LICENSE). Third-party dependencies retain their respective licenses. This software license does not assign a license to the associated manuscript or external research materials.

## Citation

If you use this implementation, cite the software repository using [CITATION.cff](CITATION.cff). Project-level attribution is used in the public metadata.

```bibtex
@software{creditassigncode,
  author = {{CreditAssignCode contributors}},
  title = {CreditAssignCode},
  year = {2026},
  url = {https://github.com/CloudYume/CreditAssignCode}
}
```

## Contact

- Official contact: [202411770@stu.neuq.edu.cn](mailto:202411770@stu.neuq.edu.cn)
- Personal contact: [2423301935@qq.com](mailto:2423301935@qq.com)
