![Method](https://img.shields.io/badge/Method-Dual--Channel_Credit_Assignment-orange?style=flat-square)
![Framework](https://img.shields.io/badge/framework-PyTorch-red?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

# CreditAssignCode

Reference implementation of the dual-channel credit assignment framework for *Credit the Response, Not the Call: Identifiable Payload States and Adoption Actions for Agentic Reinforcement Learning*.

## Project Structure

```
CreditAssignCode/
  dualcredit/
    config.py        Configuration and ablation/baseline arm definitions
    data.py          Synthetic task data (two families: retrieval / execution)
    perturbation.py  Contamination engine for tool returns (payload validity, multi-source consensus, contamination consistent with memory errors)
    features.py      State features at decision positions
    policy.py        Frozen backbone + trainable heads and low-rank adapters
    rollout.py       Three-stage (call / decision / answer) rollout state machine
    gates.py         Inference-time gating arms (caution / hint / trust / agreement / abstain)
    proxy.py         z-estimator in unlabeled scenarios (entailment × consensus)
    credit.py        Four quadrants, canonical actions, answer-channel and response-channel advantages
    arbitration.py   Maps a single call to a credit record
    objective.py     Dual-channel loss, clipping, KL and decision entropy regularization
    trainer.py       Training loop
    evaluation.py    Evaluation harness (EM, adoption rate per quadrant, decision accuracy, coverage, proxy diagnostics)
    experiment.py    Main table / ablation / proxy / representation / cost / boundary experiments
  tests/             pytest test cases
  run_experiment.py  Command-line entry point
  requirements.txt
```

## Running

```bash
conda run -n deeplearn pip install -r requirements.txt
conda run -n deeplearn pytest tests
conda run -n deeplearn python run_experiment.py --mode smoke
conda run -n deeplearn python run_experiment.py --mode main
```

`--mode` can be one of `smoke`, `main`, `ablation`, `proxy`, `representation`, `cost`, `boundary`, `all`.
Non-smoke modes will write results to `results/<mode>/` (`.txt` tables and `.json` details); use `--output` to change the path.
Common flags: `--families retrieval,execution`, `--updates`, `--dataset-size`, `--episode-budget`, `--eval-batch`, `--seed`.

## Method Highlights

- Terminal reward: `R = 1[y_hat = y*] - alpha * nu`.
- Necessity: `q = 1 - 1[y^(0) = y*]`; payload validity: `z = 1[val(p) = v*]`; four quadrants: `s = q + 2z`.
- Canonical actions: `s=0 -> override`, `s=1 -> requery` (when budget allows), `s=2 -> no unique target`, `s=3 -> adopt`.
- Answer channel advantage: normalized interpolation within groups and quadrants; response channel advantage: `A_tilde = A + gamma * (m - m_bar)`.
- Proxy estimator: `z_hat = 1[phi >= tau] * 1[corroboration realized]`.