# Code Walkthrough

[Project overview](../README.md) · [Trainer](../dualcredit/trainer.py) · [Evaluation](../dualcredit/evaluation.py)

This guide follows the checked-in synthetic implementation. Function names and result keys below match the source, so readers can trace a number or decision back to its computation.

## 1. Create a task and observable state

[`TaskDataset`](../dualcredit/data.py) generates a seeded sequence of instances and splits it into training and test subsets. Each instance contains a correct answer ID, a memory candidate, its probability distribution, and a confidence tier. Retrieval uses answer-ID equality. Execution maps IDs to numeric values before checking equality within a tolerance; it does not execute a real external program.

[`FeatureBuilder`](../dualcredit/features.py) constructs a fixed-width vector from memory probabilities, the candidate, observed passages, the consumed/adopted values, previous actions, and the remaining call budget. `EpisodeState` also carries `gold_id` for bookkeeping, but `components()` does not place that field in the policy input. The extra payload slot at `num_values` is a dangling-value sentinel.

[`PerturbationEngine.deliver()`](../dualcredit/perturbation.py) produces a consumed value and supporting passage IDs. Corruption can reinforce an incorrect memory candidate, and several passages can agree on a wrong answer. Therefore, agreement is observable evidence rather than a guarantee of payload validity. The simulator separately records `payload_valid` for credit and evaluation.

## 2. Follow an episode

[`RolloutEngine.run()`](../dualcredit/rollout.py) advances active episodes through three phases:

1. **Call.** Assisted runs force the first tool call. A tool return updates the observed payload and consumes one unit of the call budget. Later call positions can choose to stop and answer.
2. **Decision.** The policy can override, adopt, or requery. Requery is masked for learned selection when its budget is exhausted or the arm disables it. Gates, forced counterfactual actions, and instruction-only variants use separate selection paths.
3. **Answer.** The answer head selects an answer ID. Exact match and the call penalty determine the terminal reward.

An unaided or no-tool run starts directly at the answer phase. Each recorded policy position stores its slot, selected token/action, behavior log-probability, episode index, and optional call index. Its feature row is captured before the associated state transition. This alignment allows the trainer to recompute logits at the same recorded positions.

The emitted action, the action credited for training, and an action parsed from the final answer are distinct records. Gate and instruction-only actions do not create the same learned decision-position record as the default emitted-action path.

## 3. Construct labels and per-call credit

[`Trainer._pair()`](../dualcredit/trainer.py) runs selected training instances without tools and caches `q = 1 - unaided_success`. Pairing is budgeted, so an unpaired instance can have no necessity label. Proxy/judge modes can additionally elicit an unaided candidate; their extra episodes are counted separately.

[`credit_calls()`](../dualcredit/arbitration.py) combines the available necessity label with recorded or estimated payload validity. When either required label is unavailable, the quadrant is unknown. A call receives `match=1` only if its quadrant and remaining budget specify exactly one supported action and its credited action matches that target. An undecided call receives `decided=0` and `match=0`.

The default source credits the action recorded on the call. `parsed` and `judge` variants replace that attribution after the rollout, while retaining the realized trajectory. Episode normalization uses the final credited call's quadrant, or its observable stratum in the corresponding variant.

[`ProxyEstimator`](../dualcredit/proxy.py) combines a thresholded candidate-agreement score with passage corroboration. Disabled factors pass automatically for ablations. Missing elicitation or unextractable evidence can produce `estimated_state=None`; a dangling consumed value is explicitly invalid. `NLIScorer` creates reproducible noisy scores from discrete IDs and seeded randomness. It is not a pretrained entailment model.

For example, if an unaided answer fails (`q=1`) and a returned payload is invalid (`z=0`), the quadrant is `s=1`. Requery earns a match when another call remains. With no remaining call, the same quadrant has no unique supported action, so it is excluded from the denominator of overall decision accuracy.

## 4. Build the two advantage channels

[`answer_advantages()`](../dualcredit/credit.py) computes a pooled standardized reward and, where enough labels exist, a within-key standardized reward. It interpolates the two with `lam`: `lam=1` selects pooled normalization and `lam=0` selects within-key normalization. Missing keys and keys below `cell_floor` use the pooled term.

The current trainer passes all rewards in an update to this function. Although sampling records a group ID for repeated episodes of the same instance, those IDs do not partition the normalization calculation.

For each credited call, `response_advantage()` adds `gamma * (match - baseline)` to the answer advantage. The normal baseline is an exponential moving average updated from the current batch; the uncentered ablation instead freezes its first batch mean. `response_center` optionally subtracts a further mean over realized learned response positions.

When decision credit is enabled, learned decision positions receive response advantages and other positions receive answer advantages. When it is disabled, all positions use the answer channel. The counterfactual arm replaces shaped response credit with action-branch reward differences and counts its additional rollouts.

## 5. Recompute logits and optimize

[`Policy`](../dualcredit/policy.py) uses a frozen two-layer random-feature trunk, low-rank adapter paths, and separate call, decision, and answer heads. A fixed copy readout makes memory and adopted values available to answer selection. By default, call/decision base head weights are trainable; the trunk adapter and answer head are frozen by configuration.

`adapter_parameters()` returns all enabled trainable parameters, including base head weights when requested. Its name does not mean that only low-rank matrices are optimized.

[`dual_channel_loss()`](../dualcredit/objective.py) gathers the emitted token's log-probability from the appropriate head and uses the stored behavior log-probability in the importance ratio. Answer and response surrogates are averaged separately. Answer clipping and optional response clipping are controlled independently. KL terms cover answer and decision slots, while entropy regularization applies to decision logits.

The `reference()` path disables adapters and runs without gradients. It reuses the current base head weights, which can be trainable; it is not a separately stored, frozen initial-policy copy. Sampling, label construction, and stored behavior probabilities are detached. Gradients flow through the recomputed policy logits used by the loss.

## 6. Interpret evaluation outputs

[`Harness.evaluate()`](../dualcredit/evaluation.py) first evaluates unaided answers, then assisted answers on the selected split. It constructs both recorded and proxy credit for diagnostics.

- `em` and `unaided_em`: percentages of correct assisted and unaided episodes.
- `calls`: tool calls per assisted episode.
- `adoption`, `override`, and `requery`: action percentages over credited calls.
- `dec_acc`: action-match percentage over calls with a unique supported target.
- `cov`: percentage of credited calls without a unique supported target. This key is **not** the proxy coverage metric.
- `compliance`: recorded learned decision positions divided by tool calls, expressed as a percentage.
- `occ_s0` through `occ_s3`: quadrant shares over credited calls. Unknown quadrants contribute to the denominator but to none of these four numerators.
- `adm_s*`, `ovr_s*`, and `req_s*`: action percentages within the specified quadrant.
- `dec_s*`: mean match within the quadrant, including its undecided calls.
- `theta1` and `theta0`: episode accuracy conditioned on the final credited call's recorded payload validity among tool-using episodes.
- `proxy.precision`: recorded-valid share among proxy-positive calls.
- `proxy.coverage`: share of paired recorded/proxy call records for which the proxy produces a known state, including both predicted classes.
- `proxy.bias`: absolute difference between proxy-positive and recorded-valid percentages within recorded `s=1` calls.

Most empty-denominator cases are reported as zero. Inspect counts such as `episodes`, `credits`, `proxy.positives`, and `proxy.scored` before interpreting a zero as measured performance. `crossing` is computed only when both payload-validity subsets exist and `theta1 > theta0`; otherwise it is `None`.

## 7. Trace a table back to a run

[`experiment.py`](../dualcredit/experiment.py) defines stable arm IDs and saves the effective `Config`, training-cost counters, and per-corruption evaluation records for non-boundary suites. Main tables show both requested task families; several compact diagnostic tables select retrieval rows, while their detailed JSON still contains every requested family. Boundary output instead contains a `boundary` list of per-arm, per-family, per-rate reports; refer to the selected base settings and arm definitions for its configuration.

Use the JSON records for reanalysis rather than reconstructing measurements from rounded table cells. Auxiliary pairing, elicitation, and branching episodes contribute to training cost. The legacy `mean_update_seconds` field contains the trainer's most recent update duration, rather than an average over the complete history.
