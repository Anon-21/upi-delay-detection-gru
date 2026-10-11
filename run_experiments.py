import json, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             brier_score_loss, log_loss)

import sim
from mtl_mlp import MultiTaskMLP, BUCKET_EDGES_H, BUCKET_LABELS, K

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# ------------------------------------------------------------------ data
W = sim.make_world(days=70, seed=7)
tx = sim.make_transactions(W, n=450_000, seed=11)
day = (tx.t // 1440).values
tr, va, te = day < 49, (day >= 49) & (day < 59), day >= 59          # temporal split, no shuffling
X = sim.build_features(W, tx.t.values, tx.rem.values, tx.ben.values, tx.amt.values)
names, groups = sim.feature_names()
y = tx.limbo.values
hours = tx.res_h.values
kk = np.digitize(np.nan_to_num(hours, nan=0.0), BUCKET_EDGES_H)       # resolution-time bucket 0..4
static_idx = np.where(np.isin(groups, ["banks", "amount", "time"]))[0]

scaler = StandardScaler().fit(X[tr])
Xs = scaler.transform(X).astype(np.float32)

info = dict(n_total=int(len(tx)), n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()),
            base_rate_train=float(y[tr].mean()), base_rate_val=float(y[va].mean()), base_rate_test=float(y[te].mean()),
            n_limbo_train=int(y[tr].sum()), n_limbo_val=int(y[va].sum()), n_limbo_test=int(y[te].sum()),
            n_features=int(X.shape[1]), bucket_labels=BUCKET_LABELS)
print(info, flush=True)


def cls_metrics(score, yy):
    ap = average_precision_score(yy, score)
    auc = roc_auc_score(yy, score)
    p, r, thr = precision_recall_curve(yy, score)
    ok = r[:-1] >= 0.5
    prec50 = float(p[:-1][ok].max()) if ok.any() else float("nan")
    thr50 = thr[ok][np.argmax(p[:-1][ok])] if ok.any() else np.inf
    k = int(0.01 * len(yy))
    top = np.argsort(-score)[:k]
    hits = yy[top].sum()
    return dict(pr_auc=float(ap), roc_auc=float(auc), precision_at_50pct_recall=prec50,
                alerts_per_1000_at_50pct_recall=float(1000 * (score >= thr50).mean()),
                recall_in_top1pct=float(hits / yy.sum()), precision_in_top1pct=float(hits / k),
                lift_in_top1pct=float((hits / k) / yy.mean()))


res, scores_te, scores_va = {}, {}, {}

# ------------------------------------------------------------------ baselines (detection)
m = LogisticRegression(max_iter=3000).fit(Xs[tr][:, static_idx], y[tr])
scores_te["LR (no telemetry)"] = m.decision_function(Xs[te][:, static_idx]); scores_va["LR (no telemetry)"] = m.decision_function(Xs[va][:, static_idx])
m = LogisticRegression(max_iter=3000).fit(Xs[tr], y[tr])
scores_te["Logistic regression"] = m.decision_function(Xs[te]); scores_va["Logistic regression"] = m.decision_function(Xs[va])
gb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                    l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
scores_te["Gradient boosting"] = gb.decision_function(X[te]); scores_va["Gradient boosting"] = gb.decision_function(X[va])

# ------------------------------------------------------------------ multi-task NN, 3 seeds
mask_all = ((y == 1) & tr).astype(np.float32)
models, mlp_runs = [], []
for seed in (0, 1, 2):
    print(f"MLP seed {seed}", flush=True)
    net = MultiTaskMLP(Xs.shape[1], seed=seed)
    net.fit(Xs[tr], y[tr].astype(np.float32), kk[tr], mask_all[tr], Xs[va], y[va],
            epochs=30, patience=5, verbose=(seed == 0))
    models.append(net)
    mlp_runs.append(cls_metrics(net.predict(Xs[te])[0], y[te]))
net = models[0]
scores_te["Multi-task NN"] = net.predict(Xs[te])[0]; scores_va["Multi-task NN"] = net.predict(Xs[va])[0]

for k in scores_te:
    res[k] = cls_metrics(scores_te[k], y[te])
    res[k]["val_pr_auc"] = float(average_precision_score(y[va], scores_va[k]))
res["Multi-task NN (3 seeds)"] = {k: dict(mean=float(np.mean([r[k] for r in mlp_runs])), std=float(np.std([r[k] for r in mlp_runs])))
                                  for k in mlp_runs[0]}
res["_mlp_runs"] = mlp_runs

# paired bootstrap on test PR-AUC (resample transactions)
rng = np.random.default_rng(1)
yt = y[te]
B = 300
boot = {k: [] for k in ("Logistic regression", "Gradient boosting", "Multi-task NN")}
for _ in range(B):
    ix = rng.integers(0, len(yt), len(yt))
    if yt[ix].sum() == 0:
        continue
    for k in boot:
        boot[k].append(average_precision_score(yt[ix], scores_te[k][ix]))
boot = {k: np.array(v) for k, v in boot.items()}
res["bootstrap_pr_auc_95ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
res["bootstrap_diff_NN_minus_LR_95ci"] = [float(np.percentile(boot["Multi-task NN"] - boot["Logistic regression"], q)) for q in (2.5, 97.5)]
res["bootstrap_diff_NN_minus_GB_95ci"] = [float(np.percentile(boot["Multi-task NN"] - boot["Gradient boosting"], q)) for q in (2.5, 97.5)]

# ------------------------------------------------------------------ calibration of P(limbo) (Platt scaling on validation set)
platt = LogisticRegression(C=1e6, max_iter=1000).fit(scores_va["Multi-task NN"][:, None], y[va])
p_te = platt.predict_proba(scores_te["Multi-task NN"][:, None])[:, 1]
bins = np.array_split(np.argsort(p_te), 10)
res["calibration"] = dict(pred=[float(p_te[b].mean()) for b in bins], obs=[float(yt[b].mean()) for b in bins],
                          brier=float(brier_score_loss(yt, p_te)),
                          brier_baseline=float(brier_score_loss(yt, np.full(te.sum(), y[tr].mean()))))

# ------------------------------------------------------------------ resolution-time buckets (limbo transactions only)
pos_tr, pos_te, pos_va = tr & (y == 1), te & (y == 1), va & (y == 1)
ktrue = kk[pos_te]
prior = np.bincount(kk[pos_tr], minlength=K) / pos_tr.sum()


def eta_metrics(P):
    over24, over48 = P[:, 3] + P[:, 4], P[:, 4]
    return dict(log_loss=float(log_loss(ktrue, np.clip(P, 1e-6, 1), labels=list(range(K)))),
                accuracy=float((P.argmax(1) == ktrue).mean()),
                auc_over_24h=float(roc_auc_score(ktrue >= 3, over24)),
                auc_over_48h=float(roc_auc_score(ktrue == 4, over48)),
                brier_over_48h=float(brier_score_loss(ktrue == 4, over48)))


eta = {}
eta["Prior (all stuck payments)"] = eta_metrics(np.tile(prior, (pos_te.sum(), 1)))
cnt = np.stack([np.bincount(kk[pos_tr & (tx.ben.values == b)], minlength=K) + 1 for b in range(sim.NB)]).astype(float)
bank_prior = cnt / cnt.sum(1, keepdims=True)
eta["Prior by beneficiary bank"] = eta_metrics(bank_prior[tx.ben.values[pos_te]])
gbc = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=6, l2_regularization=1.0,
                                     random_state=0).fit(X[pos_tr], kk[pos_tr])
eta["Gradient boosting"] = eta_metrics(gbc.predict_proba(X[pos_te]))
runs = [eta_metrics(mdl.predict(Xs[pos_te])[1]) for mdl in models]
eta["Multi-task NN (3 seeds)"] = {k: dict(mean=float(np.mean([r[k] for r in runs])), std=float(np.std([r[k] for r in runs])))
                                  for k in runs[0]}
res["eta"] = eta
res["eta_prior_train"] = prior.tolist()

# discrimination of the ">48 h" tail: actual share >48h by quintile of predicted P(>48h)
Pnn = net.predict(Xs[pos_te])[1]
order = np.argsort(Pnn[:, 4])
quint = np.array_split(order, 5)
res["tail_by_quintile"] = dict(pred=[float(Pnn[q, 4].mean()) for q in quint], obs=[float((ktrue[q] == 4).mean()) for q in quint],
                               n=[int(len(q)) for q in quint])

# ------------------------------------------------------------------ permutation importance (by feature group), NN seed 0
prng = np.random.default_rng(0)
base_ap = average_precision_score(yt, scores_te["Multi-task NN"])
imp = {}
for g in np.unique(groups):
    cols = np.where(groups == g)[0]
    drops = []
    for _ in range(5):
        Xp = Xs[te].copy()
        Xp[:, cols] = Xp[prng.permutation(len(Xp))][:, cols]
        drops.append(base_ap - average_precision_score(yt, net.predict(Xp)[0]))
    imp[g] = dict(ap_drop_mean=float(np.mean(drops)), ap_drop_std=float(np.std(drops)))
res["permutation_importance"] = imp
res["base_ap_seed0"] = float(base_ap)

# ------------------------------------------------------------------ replay scenarios (hypothetical payment on simulated telemetry)
def scenario(tc, rem, ben, amt, label):
    f = sim.build_features(W, np.array([tc]), np.array([rem]), np.array([ben]), np.array([amt]),
                           hour=np.array([20.0]), dow=np.array([2]), dom=np.array([7]))
    lg, P = net.predict(scaler.transform(f).astype(np.float32))
    P = P[0]
    return dict(label=label, remitter=sim.BANKS[rem], beneficiary=sim.BANKS[ben],
                p_limbo=float(platt.predict_proba(lg[:, None])[:, 1][0]),
                bucket_probs=[float(x) for x in P], p_within_1h=float(P[0]), p_over_24h=float(P[3] + P[4]),
                p_over_48h=float(P[4]), d_ben=float(W["d"][ben, tc]), util_ben=float(W["u"][ben, tc]))


tt = np.arange(59 * 1440, W["T"])
hh = W["hour"][tt]
evening = (hh >= 19) & (hh <= 22)
cand_ben = [(int(t_), b) for t_ in tt[evening] for b in range(1, sim.NB) if 0.5 <= W["d"][b, t_] <= 0.9]
cand_rem = [int(t_) for t_ in tt[evening] if 0.5 <= W["d"][0, t_] <= 0.9] or [int(t_) for t_ in tt if 0.5 <= W["d"][0, t_] <= 0.9]
t_deg, b_deg = cand_ben[0] if cand_ben else (None, 3)
t_calm = int(tt[evening & (W["d"][:, tt].max(axis=0) < 0.05)][0])
scen = [scenario(t_calm, 0, b_deg, 10000, "Calm network (Bank A -> Bank %s)" % sim.BANKS[b_deg])]
if t_deg is not None:
    scen.append(scenario(t_deg, 0, b_deg, 10000, "Beneficiary bank %s degraded" % sim.BANKS[b_deg]))
if cand_rem:
    scen.append(scenario(cand_rem[0], 0, b_deg, 10000, "Remitter Bank A degraded (A -> %s)" % sim.BANKS[b_deg]))
    scen.append(scenario(cand_rem[0], 0, 0, 10000, "Bank A degraded, both sides Bank A (A -> A)"))
res["replay"] = scen
res["replay_note"] = ("Hypothetical Rs 10,000 payment, remitter = Bank A (large, fragile), Wednesday 20:00, "
                      "telemetry taken from simulated test-period minutes. Not a reconstruction of the real incident.")
res["resolution_distribution_all_limbo"] = dict(
    quantiles={str(k): float(np.nanquantile(hours[y == 1], k)) for k in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
    share_over_24h=float(np.mean(hours[y == 1] > 24)), share_over_48h=float(np.mean(hours[y == 1] > 48)),
    share_over_58h=float(np.mean(hours[y == 1] > 58)))
res["info"] = info
res["runtime_s"] = float(time.time() - t0)

json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
np.savez(f"{OUT}/test_preds.npz", y_te=yt, p_te=p_te,
         score_lr_static=scores_te["LR (no telemetry)"], score_lr=scores_te["Logistic regression"],
         score_gb=scores_te["Gradient boosting"], score_nn=scores_te["Multi-task NN"],
         ktrue=ktrue, P_nn=Pnn, res_h_all=hours[y == 1], ben_te=tx.ben.values[te])
print("done in %.0fs" % (time.time() - t0), flush=True)
