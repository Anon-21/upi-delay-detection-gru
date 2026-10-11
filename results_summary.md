# Limbo Predictor: results summary

All numbers come from `results/results.json`, produced by `run_experiments.py` on **simulated** data. Nothing here is real SBI, GPay or NPCI data.

## 1. Setup

- 450,000 simulated UPI payments over 70 days, 8 anonymised banks (A to H). Bank A is a large, heavily loaded, more fragile bank.
- Temporal split, no shuffling: train 313,108 (days 0-48), validation 62,258 (days 49-58), test 74,634 (days 59-69).
- Stuck ('limbo') payments: train 3,510 (1.12%), validation 814 (1.31%), test 468 (0.63%). The test period happened to be quieter, which is a realistic drift.
- 82 input features: the last ~28 minutes of telemetry (latency, timeout rate, utilisation) for the remitter and beneficiary banks, bank identities, log amount, time of day, weekday, rent-week flag. Only information available at payment time is used.

**Model.** A multi-task neural network written in NumPy (two shared ReLU layers of 128 and 64 units, Adam, manual backprop). Head 1 outputs the probability the payment gets stuck. Head 2 outputs a distribution over how long it takes to resolve (< 1 h, 1-12 h, 12-24 h, 24-48 h, > 48 h), trained only on stuck payments. It is a feed-forward network over a flattened telemetry window, **not** a GRU: no deep-learning framework could be installed in this workspace.

## 2. Detecting payments that will get stuck (test set)

| Model | PR-AUC | ROC-AUC | Precision in top 1% of scores | Share of stuck payments caught in top 1% | Lift vs base rate |
|---|---|---|---|---|---|
| Logistic regression, no telemetry | 0.010 | 0.625 | 0.8% | 1.3% | 1.3x |
| Logistic regression | 0.157 | 0.689 | 17.6% | 28.0% | 28.0x |
| Gradient boosting | 0.163 | 0.718 | 17.0% | 27.1% | 27.1x |
| Multi-task NN (mean ± sd, 3 seeds) | 0.148 ± 0.009 | 0.696 ± 0.007 | 17.0% ± 0.1% | 27.1% ± 0.2% | 27.1 ± 0.2x |

Base rate on the test set: 0.63%.

**Uncertainty.** Bootstrap 95% intervals for test PR-AUC: logistic regression [0.124, 0.196], gradient boosting [0.126, 0.203], neural network [0.117, 0.192]. Paired difference NN minus logistic regression: [-0.028, +0.016]; NN minus gradient boosting: [-0.029, +0.012]. Both intervals include zero, so the three telemetry-based models are statistically indistinguishable here.

**What the model uses** (drop in test PR-AUC when a feature group is shuffled): telemetry ben 0.138, telemetry rem 0.016, time 0.006, banks 0.003, amount 0.002.

**Calibration.** After Platt scaling on the validation set, Brier score on test is 0.0056 against 0.0063 for always predicting the training base rate. The model slightly over-predicts; the test period had far fewer outages than training (degraded bank-minutes were 0.3% of the test period), which is the likely reason.

## 3. Predicting how long a stuck payment takes (test set, stuck payments only)

| Model | Log-loss (lower is better) | AUC for waiting > 24 h | AUC for waiting > 48 h | Brier for > 48 h |
|---|---|---|---|---|
| Prior (all stuck payments) | 1.466 | 0.500 | 0.500 | 0.0430 |
| Prior by beneficiary bank | 1.455 | 0.645 | 0.558 | 0.0438 |
| Gradient boosting | 1.141 | 0.773 | 0.886 | 0.0321 |
| Multi-task NN (3 seeds) | 1.275 ± 0.056 | 0.747 ± 0.009 | 0.793 ± 0.022 | 0.0385 ± 0.0008 |

Only 20 of 468 stuck test payments waited more than 48 h, so this part is noisy. In the highest-risk fifth the model predicted 16.3% and 16.1% actually waited more than 48 h.

Why buckets and not one number: resolution time is bimodal (a late response clears it within hours, or it waits for a reconciliation batch). A single 'expected hours' figure falls between the two modes and describes neither. My first version used that and was misleading, so it was replaced.

## 4. Replay: a hypothetical Rs 10,000 payment on Wednesday 20:00

| Scenario | Chance it gets stuck | Resolves < 1 h | Waits > 24 h | Waits > 48 h |
|---|---|---|---|---|
| Calm network (Bank A -> Bank D) | 0.1% | 56% | 1% | 0.1% |
| Beneficiary bank D degraded | 8.9% | 50% | 8% | 0.1% |
| Remitter Bank A degraded (A -> D) | 1.7% | 11% | 28% | 0.3% |
| Bank A degraded, both sides Bank A (A -> A) | 18.1% | 7% | 37% | 8.4% |

Hypothetical Rs 10,000 payment, remitter = Bank A (large, fragile), Wednesday 20:00, telemetry taken from simulated test-period minutes. Not a reconstruction of the real incident. It illustrates the kind of message the system could show; it does not claim to explain the real 58-hour delay.

In the simulator, 23% of stuck payments wait over 24 h, 7.6% over 48 h and 5.9% over 58 h. These shares come from simulator assumptions, not measurements.

## 5. Honest reading of the results

1. **Telemetry carries the signal.** Without it the model is no better than the base rate (PR-AUC 0.010 vs 0.006). With it, the top 1% of scored payments contains roughly a quarter of all stuck payments at about 27x the base rate.
2. **The deep model is not better at detection.** It ties logistic regression and gradient boosting. In this simulator the risk is a smooth function of the beneficiary bank's health, which simple models capture. The neural network's real contribution is the shared multi-task design and the resolution-time distribution.
3. **It cannot catch everything.** At 50% recall, precision is about 1-2%, because many simulated stuck payments happen while both banks look healthy (random timeouts): 43% over the whole simulation and 73% in the test period, which had very few outages. The tool is for prioritising and for messaging, not for catching every case.
4. **Resolution-time prediction beats naive priors but not gradient boosting** (log-loss 1.28 vs 1.14 vs 1.47 for the prior).

## 6. Assumptions and limitations (copy into the case study)

- All data are simulated. The simulator builds stuck payments from a latent bank-degradation signal, and telemetry is a noisy view of that signal. Models therefore partly learn the simulator's own rules; results show the method is feasible, not that it would work on SBI traffic.
- Bank parameters (latency, fragility, batch-miss rates, outage frequency) are illustrative. Bank A is assumed more fragile, inspired by public reports of higher technical-decline rates at large public-sector remitter banks; this is an assumption, not a finding.
- The mechanism follows NPCI's published rules: a short response timeout, deemed-approved settlement, and reconciliation in batch cycles. Batches run 06:00 Monday to Saturday and a payment must be ready 2 h before the cut-off. Holidays and real settlement-cycle times are not modelled.
- The 58-hour wait in your incident is not reproduced or explained by the model. The simulator can produce such waits, but the replay uses hypothetical conditions.
- Resolution-time predictions are conditional on the payment being stuck. The test set has only 468 stuck payments and 20 waits beyond 48 h, so tail estimates are noisy.
- The network is a NumPy feed-forward multi-task model, not a GRU or transformer. A deployed system would need real telemetry, privacy review and a plan for false reassurance (telling a customer 'safe, resolves soon' when it is not).

## 7. Files

- `sim.py` simulator and feature builder; `mtl_mlp.py` the network; `run_experiments.py` training and evaluation; `make_figures.py` charts; `make_summary.py` this file.
- `figures/fig1`-`fig6` PNG charts. Run order: `python run_experiments.py`, `python make_figures.py`, `python make_summary.py` (about 2 minutes on 2 CPU cores).
