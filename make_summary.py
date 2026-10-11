import json
import os
BASE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(BASE, "results", "results.json"), encoding="utf-8"))
I = R["info"]


def ms(v, f="{:.3f}"):
    return (f + " ± " + f).format(v["mean"], v["std"]) if isinstance(v, dict) else f.format(v)


L = []
w = L.append
w("# Limbo Predictor: results summary\n")
w("All numbers come from `results/results.json`, produced by `run_experiments.py` on **simulated** data. "
  "Nothing here is real SBI, GPay or NPCI data.\n")

w("## 1. Setup\n")
w(f"- {I['n_total']:,} simulated UPI payments over 70 days, 8 anonymised banks (A to H). Bank A is a large, heavily loaded, more fragile bank.")
w(f"- Temporal split, no shuffling: train {I['n_train']:,} (days 0-48), validation {I['n_val']:,} (days 49-58), test {I['n_test']:,} (days 59-69).")
w(f"- Stuck ('limbo') payments: train {I['n_limbo_train']:,} ({I['base_rate_train']:.2%}), validation {I['n_limbo_val']:,} ({I['base_rate_val']:.2%}), "
  f"test {I['n_limbo_test']:,} ({I['base_rate_test']:.2%}). The test period happened to be quieter, which is a realistic drift.")
w(f"- {I['n_features']} input features: the last ~28 minutes of telemetry (latency, timeout rate, utilisation) for the remitter and beneficiary banks, "
  "bank identities, log amount, time of day, weekday, rent-week flag. Only information available at payment time is used.\n")
w("**Model.** A multi-task neural network written in NumPy (two shared ReLU layers of 128 and 64 units, Adam, manual backprop). "
  "Head 1 outputs the probability the payment gets stuck. Head 2 outputs a distribution over how long it takes to resolve "
  "(< 1 h, 1-12 h, 12-24 h, 24-48 h, > 48 h), trained only on stuck payments. "
  "It is a feed-forward network over a flattened telemetry window, **not** a GRU: no deep-learning framework could be installed in this workspace.\n")

w("## 2. Detecting payments that will get stuck (test set)\n")
w("| Model | PR-AUC | ROC-AUC | Precision in top 1% of scores | Share of stuck payments caught in top 1% | Lift vs base rate |")
w("|---|---|---|---|---|---|")
rows = [("Logistic regression, no telemetry", R["LR (no telemetry)"]),
        ("Logistic regression", R["Logistic regression"]),
        ("Gradient boosting", R["Gradient boosting"]),
        ("Multi-task NN (mean ± sd, 3 seeds)", R["Multi-task NN (3 seeds)"])]
for n, v in rows:
    w(f"| {n} | {ms(v['pr_auc'])} | {ms(v['roc_auc'])} | {ms(v['precision_in_top1pct'], '{:.1%}')} | "
      f"{ms(v['recall_in_top1pct'], '{:.1%}')} | {ms(v['lift_in_top1pct'], '{:.1f}')}x |")
w(f"\nBase rate on the test set: {I['base_rate_test']:.2%}.\n")
b = R["bootstrap_pr_auc_95ci"]
w("**Uncertainty.** Bootstrap 95% intervals for test PR-AUC: "
  f"logistic regression [{b['Logistic regression'][0]:.3f}, {b['Logistic regression'][1]:.3f}], "
  f"gradient boosting [{b['Gradient boosting'][0]:.3f}, {b['Gradient boosting'][1]:.3f}], "
  f"neural network [{b['Multi-task NN'][0]:.3f}, {b['Multi-task NN'][1]:.3f}]. "
  f"Paired difference NN minus logistic regression: [{R['bootstrap_diff_NN_minus_LR_95ci'][0]:+.3f}, {R['bootstrap_diff_NN_minus_LR_95ci'][1]:+.3f}]; "
  f"NN minus gradient boosting: [{R['bootstrap_diff_NN_minus_GB_95ci'][0]:+.3f}, {R['bootstrap_diff_NN_minus_GB_95ci'][1]:+.3f}]. "
  "Both intervals include zero, so the three telemetry-based models are statistically indistinguishable here.\n")
pi = R["permutation_importance"]
w("**What the model uses** (drop in test PR-AUC when a feature group is shuffled): "
  + ", ".join(f"{k.replace('_', ' ')} {v['ap_drop_mean']:.3f}" for k, v in sorted(pi.items(), key=lambda kv: -kv[1]['ap_drop_mean'])) + ".\n")
c = R["calibration"]
w(f"**Calibration.** After Platt scaling on the validation set, Brier score on test is {c['brier']:.4f} against {c['brier_baseline']:.4f} for always predicting the training base rate. "
  "The model slightly over-predicts; the test period had far fewer outages than training (degraded bank-minutes were 0.3% of the test period), which is the likely reason.\n")

w("## 3. Predicting how long a stuck payment takes (test set, stuck payments only)\n")
w("| Model | Log-loss (lower is better) | AUC for waiting > 24 h | AUC for waiting > 48 h | Brier for > 48 h |")
w("|---|---|---|---|---|")
for n, v in R["eta"].items():
    w(f"| {n} | {ms(v['log_loss'])} | {ms(v['auc_over_24h'])} | {ms(v['auc_over_48h'])} | {ms(v['brier_over_48h'], '{:.4f}')} |")
t = R["tail_by_quintile"]
w(f"\nOnly {int(round(sum(o * n for o, n in zip(t['obs'], t['n']))))} of {sum(t['n'])} stuck test payments waited more than 48 h, so this part is noisy. "
  f"In the highest-risk fifth the model predicted {t['pred'][4]:.1%} and {t['obs'][4]:.1%} actually waited more than 48 h.\n")
w("Why buckets and not one number: resolution time is bimodal (a late response clears it within hours, or it waits for a reconciliation batch). "
  "A single 'expected hours' figure falls between the two modes and describes neither. My first version used that and was misleading, so it was replaced.\n")

w("## 4. Replay: a hypothetical Rs 10,000 payment on Wednesday 20:00\n")
w("| Scenario | Chance it gets stuck | Resolves < 1 h | Waits > 24 h | Waits > 48 h |")
w("|---|---|---|---|---|")
for s in R["replay"]:
    w(f"| {s['label']} | {s['p_limbo']:.1%} | {s['p_within_1h']:.0%} | {s['p_over_24h']:.0%} | {s['p_over_48h']:.1%} |")
w(f"\n{R['replay_note']} It illustrates the kind of message the system could show; it does not claim to explain the real 58-hour delay.\n")
d = R["resolution_distribution_all_limbo"]
w(f"In the simulator, {d['share_over_24h']:.0%} of stuck payments wait over 24 h, {d['share_over_48h']:.1%} over 48 h and {d['share_over_58h']:.1%} over 58 h. These shares come from simulator assumptions, not measurements.\n")

w("## 5. Honest reading of the results\n")
w("1. **Telemetry carries the signal.** Without it the model is no better than the base rate (PR-AUC 0.010 vs 0.006). With it, the top 1% of scored payments contains roughly a quarter of all stuck payments at about 27x the base rate.")
w("2. **The deep model is not better at detection.** It ties logistic regression and gradient boosting. In this simulator the risk is a smooth function of the beneficiary bank's health, which simple models capture. The neural network's real contribution is the shared multi-task design and the resolution-time distribution.")
w("3. **It cannot catch everything.** At 50% recall, precision is about 1-2%, because many simulated stuck payments happen while both banks look healthy (random timeouts): 43% over the whole simulation and 73% in the test period, which had very few outages. The tool is for prioritising and for messaging, not for catching every case.")
w("4. **Resolution-time prediction beats naive priors but not gradient boosting** (log-loss 1.28 vs 1.14 vs 1.47 for the prior).\n")

w("## 6. Assumptions and limitations (copy into the case study)\n")
for x in [
    "All data are simulated. The simulator builds stuck payments from a latent bank-degradation signal, and telemetry is a noisy view of that signal. Models therefore partly learn the simulator's own rules; results show the method is feasible, not that it would work on SBI traffic.",
    "Bank parameters (latency, fragility, batch-miss rates, outage frequency) are illustrative. Bank A is assumed more fragile, inspired by public reports of higher technical-decline rates at large public-sector remitter banks; this is an assumption, not a finding.",
    "The mechanism follows NPCI's published rules: a short response timeout, deemed-approved settlement, and reconciliation in batch cycles. Batches run 06:00 Monday to Saturday and a payment must be ready 2 h before the cut-off. Holidays and real settlement-cycle times are not modelled.",
    "The 58-hour wait in your incident is not reproduced or explained by the model. The simulator can produce such waits, but the replay uses hypothetical conditions.",
    "Resolution-time predictions are conditional on the payment being stuck. The test set has only 468 stuck payments and 20 waits beyond 48 h, so tail estimates are noisy.",
    "The network is a NumPy feed-forward multi-task model, not a GRU or transformer. A deployed system would need real telemetry, privacy review and a plan for false reassurance (telling a customer 'safe, resolves soon' when it is not).",
]:
    w(f"- {x}")
w("\n## 7. Files\n")
w("- `sim.py` simulator and feature builder; `mtl_mlp.py` the network; `run_experiments.py` training and evaluation; `make_figures.py` charts; `make_summary.py` this file.")
w("- `figures/fig1`-`fig6` PNG charts. Run order: `python run_experiments.py`, `python make_figures.py`, `python make_summary.py` (about 2 minutes on 2 CPU cores).")
open(os.path.join(BASE, "results_summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("ok")
