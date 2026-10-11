import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FixedLocator, FixedFormatter
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_curve

BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, "results")
F = os.path.join(BASE, "figures")
os.makedirs(F, exist_ok=True)
res = json.load(open(f"{R}/results.json"))
d = np.load(f"{R}/test_preds.npz")

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]     # validated ordinal ramp (steps 250-650)
ORD_TEXT = [INK, INK, "#ffffff", "#ffffff", "#ffffff"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-", "axes.axisbelow": True,
    "font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
})


def head(fig, title, sub):
    fig.text(0.05, 0.965, title, fontsize=12.5, fontweight="semibold", ha="left", va="top", color=INK)
    fig.text(0.05, 0.905, sub, fontsize=8.5, ha="left", va="top", color=INK2)


def save(fig, name):
    fig.savefig(f"{F}/{name}", dpi=200)
    plt.close(fig)


y = d["y_te"]; base = y.mean()
info = res["info"]

# ------------------------------------------------------------------ 1. precision-recall
fig, ax = plt.subplots(figsize=(7.2, 4.6))
fig.subplots_adjust(top=0.78, bottom=0.14, left=0.1, right=0.97)
series = [("Logistic regression, no telemetry", "score_lr_static", MUTED),
          ("Logistic regression", "score_lr", AQUA),
          ("Gradient boosting", "score_gb", ORANGE),
          ("Multi-task neural network", "score_nn", BLUE)]
for lab, key, col in series:
    p, r, _ = precision_recall_curve(y, d[key])
    ax.plot(r, p, color=col, lw=1.6, label=lab, solid_capstyle="round", solid_joinstyle="round")
ax.axhline(base, color=AXIS, lw=1)
ax.text(0.99, base + 0.01, f"base rate {base:.2%}", ha="right", va="bottom", fontsize=8.5, color=INK2)
ax.set_xlim(0, 1); ax.set_ylim(0, 0.6)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.grid(axis="y")
ax.set_xlabel("Recall: share of stuck payments flagged"); ax.set_ylabel("Precision: share of flags that were stuck")
ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2)
head(fig, "Telemetry is what makes stuck payments predictable",
     f"Precision-recall on {info['n_test']:,} simulated test payments ({info['n_limbo_test']} stuck). Vertical axis cut at 60%.")
save(fig, "fig1_precision_recall.png")

# ------------------------------------------------------------------ 2. calibration
cal = res["calibration"]
fig, ax = plt.subplots(figsize=(6.2, 4.6))
fig.subplots_adjust(top=0.78, bottom=0.14, left=0.12, right=0.96)
mx = max(max(cal["pred"]), max(cal["obs"])) * 1.1
ax.plot([0, mx], [0, mx], color=AXIS, lw=1)
ax.text(0.0015, 0.0262, "grey line: perfect calibration", fontsize=8.5, color=INK2, ha="left", va="bottom")
ax.plot(cal["pred"], cal["obs"], color=BLUE, lw=1.6, zorder=2)
ax.plot(cal["pred"], cal["obs"], "o", ms=8, color=BLUE, mec=SURF, mew=2, zorder=3)
ax.set_xlim(0, mx); ax.set_ylim(0, mx)
for a in (ax.xaxis, ax.yaxis):
    a.set_major_formatter(PercentFormatter(1.0, decimals=1))
ax.grid(axis="y")
ax.set_xlabel("Predicted chance a payment gets stuck"); ax.set_ylabel("Observed share that got stuck")
head(fig, "Predicted risk tracks observed risk, slightly high",
     f"10 equal groups of test payments. Brier {cal['brier']:.4f} vs {cal['brier_baseline']:.4f} for a constant guess.")
save(fig, "fig2_calibration.png")

# ------------------------------------------------------------------ 3. resolution-time distribution
h = d["res_h_all"]
bins = np.logspace(np.log10(0.05), np.log10(200), 41)
counts, _ = np.histogram(h, bins)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
fig.subplots_adjust(top=0.78, bottom=0.16, left=0.1, right=0.97)
ax.bar(bins[:-1], counts, width=np.diff(bins), align="edge", color=BLUE, edgecolor=SURF, linewidth=1.2)
ax.set_xscale("log")
ticks = [0.25, 1, 6, 12, 24, 48, 96]
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_major_formatter(FixedFormatter(["15 min", "1 h", "6 h", "12 h", "24 h", "48 h", "96 h"]))
ax.xaxis.set_minor_locator(FixedLocator([]))
ax.set_xlim(0.05, 200)
ax.axvline(58, color=INK2, lw=1)
share58 = float((h > 58).mean())
ax.set_ylim(0, counts.max() * 1.28)
ax.text(58 / 1.06, counts.max() * 1.26, f"58 h (the incident)\n{share58:.0%} of stuck payments\nwait longer than this", fontsize=8.5, color=INK2, va="top", ha="right")
ax.grid(axis="y")
ax.set_xlabel("Time until the payment resolved (log scale)"); ax.set_ylabel("Stuck payments")
head(fig, "Most stuck payments clear fast; a few wait days",
     f"Simulated resolution times for {len(h):,} stuck payments (simulator assumption, not bank data).")
save(fig, "fig3_resolution_time.png")

# ------------------------------------------------------------------ 4. tail check by quintile
tq = res["tail_by_quintile"]
n48 = int((d["ktrue"] == 4).sum())
fig, ax = plt.subplots(figsize=(7.2, 4.6))
fig.subplots_adjust(top=0.78, bottom=0.17, left=0.1, right=0.97)
xs = np.arange(5)
bw = 0.2
ax.bar(xs - bw / 2 - 0.01, tq["pred"], width=bw, color=BLUE, edgecolor=SURF, linewidth=1.5, label="Predicted")
ax.bar(xs + bw / 2 + 0.01, tq["obs"], width=bw, color=ORANGE, edgecolor=SURF, linewidth=1.5, label="Observed")
ax.text(4, max(tq["pred"][4], tq["obs"][4]) + 0.006, f"{tq['pred'][4]:.1%} predicted vs {tq['obs'][4]:.1%} observed", ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.set_ylim(0, 0.2)
ax.set_xticks(xs)
ax.set_xticklabels([f"Group {i + 1}\n(n={n})" for i, n in enumerate(tq["n"])])
ax.set_xlim(-0.6, 4.9)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.grid(axis="y")
ax.set_xlabel("Stuck payments grouped by predicted risk of waiting more than 48 h (lowest to highest)")
ax.set_ylabel("Share that waited more than 48 h")
ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
head(fig, "The long-wait risk is well calibrated where it matters",
     f"Only {n48} of {len(d['ktrue'])} stuck test payments waited over 48 h, so the lower groups are noisy.")
save(fig, "fig4_long_wait_calibration.png")

# ------------------------------------------------------------------ 5. permutation importance
imp = res["permutation_importance"]
lab = {"telemetry_ben": "Beneficiary-bank telemetry", "telemetry_rem": "Remitter-bank telemetry",
       "time": "Time of day, weekday, rent week", "banks": "Which banks", "amount": "Payment amount"}
order = sorted(imp, key=lambda k: imp[k]["ap_drop_mean"])
fig, ax = plt.subplots(figsize=(7.2, 3.6))
fig.subplots_adjust(top=0.72, bottom=0.17, left=0.33, right=0.95)
vals = [imp[k]["ap_drop_mean"] for k in order]
ax.barh(range(len(order)), vals, height=0.42, color=BLUE)
for i, v in enumerate(vals):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center", ha="left", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(order))); ax.set_yticklabels([lab[k] for k in order], color=INK2, fontsize=9)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, max(vals) * 1.15)
ax.grid(axis="x")
ax.set_xlabel("Drop in PR-AUC when that feature group is shuffled")
head(fig, "The model leans on the beneficiary bank's live health signals",
     f"Permutation importance on the test set (baseline PR-AUC {res['base_ap_seed0']:.3f}); mean of 5 shuffles.")
save(fig, "fig5_feature_importance.png")

# ------------------------------------------------------------------ 6. incident replay (stacked, ordinal)
labels = res["info"]["bucket_labels"]
rows = [("All stuck payments\n(no extra information)", res["eta_prior_train"], None)]
for s in res["replay"]:
    rows.append((s["label"].replace("Bank A degraded, both sides Bank A", "Bank A degraded, both sides").replace(" (", "\n("), s["bucket_probs"], s["p_limbo"]))
fig, ax = plt.subplots(figsize=(7.6, 4.6))
fig.subplots_adjust(top=0.74, bottom=0.12, left=0.28, right=0.97)
for i, (name, probs, pl) in enumerate(rows):
    left = 0.0
    for j, pj in enumerate(probs):
        ax.barh(i, pj, left=left, height=0.5, color=ORD[j], edgecolor=SURF, linewidth=1.5)
        if pj >= 0.07:
            ax.text(left + pj / 2, i, f"{pj:.0%}", ha="center", va="center", fontsize=8.5, color=ORD_TEXT[j])
        left += pj
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] + ("" if r[2] is None else f"\nchance of getting stuck: {r[2]:.1%}") for r in rows],
                   color=INK2, fontsize=8.5)
ax.tick_params(axis="y", length=0)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_xlabel("If the payment does get stuck: predicted time until it resolves")
ax.legend(handles=[Patch(facecolor=ORD[j], edgecolor=SURF, label=labels[j]) for j in range(5)], ncol=5,
          loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False, fontsize=8.5, labelcolor=INK2,
          handlelength=1.2, columnspacing=1.2)
head(fig, "What the model would have told you at the time of payment",
     "Hypothetical Rs 10,000 payment, Wednesday 20:00, on simulated conditions. Not a reconstruction of the real incident.")
save(fig, "fig6_incident_replay.png")
print("figures written:", sorted(os.listdir(F)))
