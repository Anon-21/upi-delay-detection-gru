"""
GRU experiment for Limbo Predictor.

Uses the same synthetic simulator, temporal split, and evaluation metrics as
run_experiments.py. The GRU reads telemetry as a chronological sequence and
combines its representation with static transaction features.

Run from the folder containing sim.py:
    pip install torch
    python gru_experiment.py

This experiment uses synthetic data only. It does not predict real UPI incidents.
"""
import json
import os
import time
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

import sim

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

SEED = 7
BATCH_SIZE = 2048
MAX_EPOCHS = 30
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

def classification_metrics(score, y):
    ap = average_precision_score(y, score)
    auc = roc_auc_score(y, score)
    p, r, thresholds = precision_recall_curve(y, score)
    valid = r[:-1] >= 0.5
    precision50 = float(p[:-1][valid].max()) if valid.any() else float("nan")
    threshold50 = thresholds[valid][np.argmax(p[:-1][valid])] if valid.any() else np.inf
    k = max(1, int(0.01 * len(y)))
    top = np.argsort(-score)[:k]
    hits = int(y[top].sum())
    return {
        "pr_auc": float(ap),
        "roc_auc": float(auc),
        "precision_at_50pct_recall": precision50,
        "alerts_per_1000_at_50pct_recall": float(1000 * (score >= threshold50).mean()),
        "recall_in_top1pct": float(hits / max(1, y.sum())),
        "precision_in_top1pct": float(hits / k),
        "lift_in_top1pct": float((hits / k) / max(y.mean(), 1e-12)),
    }

def make_sequences(X, telemetry_cols=60, n_banks=8, n_steps=10):
    """
    Feature order from sim.build_features is:
      beneficiary: latency[10], timeout[10], utilisation[10],
      remitter:    latency[10], timeout[10], utilisation[10]
    Lag 0 is most recent (t-1); reverse to oldest -> newest for the GRU.
    """
    tele = X[:, :telemetry_cols].reshape(-1, 2, 3, n_steps)
    # (sample, bank-side, signal, lag) -> (sample, time, bank-side, signal)
    seq = tele.transpose(0, 3, 1, 2).reshape(-1, n_steps, 6)
    seq = seq[:, ::-1, :].copy()
    # All columns after telemetry are static: bank one-hot, amount, time.
    static = X[:, telemetry_cols:].copy()
    return seq.astype(np.float32), static.astype(np.float32)

class TelemetryGRU(nn.Module):
    def __init__(self, n_static, hidden_size=48, dropout=0.20):
        super().__init__()
        self.gru = nn.GRU(input_size=6, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_static, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, sequence, static):
        _, h = self.gru(sequence)
        features = torch.cat([h[-1], static], dim=1)
        return self.head(features).squeeze(1)

def main():
    start = time.time()
    print(f"Device: {DEVICE}", flush=True)
    print("Generating the same simulated dataset...", flush=True)
    world = sim.make_world(days=70, seed=7)
    tx = sim.make_transactions(world, n=450_000, seed=11)
    day = (tx.t // 1440).values
    tr = day < 49
    va = (day >= 49) & (day < 59)
    te = day >= 59

    X = sim.build_features(world, tx.t.values, tx.rem.values, tx.ben.values, tx.amt.values)
    y = tx.limbo.values.astype(np.float32)

    # Fit preprocessing on training data only.
    seq, static = make_sequences(X)
    seq_scaler = StandardScaler().fit(seq[tr].reshape(-1, seq.shape[-1]))
    seq = seq_scaler.transform(seq.reshape(-1, seq.shape[-1])).reshape(seq.shape).astype(np.float32)
    static_scaler = StandardScaler().fit(static[tr])
    static = static_scaler.transform(static).astype(np.float32)

    info = {
        "n_total": int(len(y)), "n_train": int(tr.sum()),
        "n_val": int(va.sum()), "n_test": int(te.sum()),
        "base_rate_train": float(y[tr].mean()),
        "base_rate_val": float(y[va].mean()),
        "base_rate_test": float(y[te].mean()),
        "sequence_shape_per_transaction": list(seq.shape[1:]),
        "static_features": int(static.shape[1]),
        "device": DEVICE,
    }
    print(info, flush=True)

    def loader(mask, shuffle):
        ds = TensorDataset(
            torch.from_numpy(seq[mask]),
            torch.from_numpy(static[mask]),
            torch.from_numpy(y[mask])
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=0, pin_memory=(DEVICE == "cuda"))

    train_loader = loader(tr, True)
    val_loader = loader(va, False)
    model = TelemetryGRU(n_static=static.shape[1]).to(DEVICE)
    # Weight positive examples because the simulated stuck-payment class is rare.
    pos_weight = torch.tensor([(1.0 - float(y[tr].mean())) / max(float(y[tr].mean()), 1e-8)],
                              dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_ap, best_state, wait = -1.0, None, 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for xb, sb, yb in train_loader:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, sb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        val_scores, val_y = [], []
        with torch.no_grad():
            for xb, sb, yb in val_loader:
                logits = model(xb.to(DEVICE), sb.to(DEVICE))
                val_scores.append(torch.sigmoid(logits).cpu().numpy())
                val_y.append(yb.numpy())
        val_scores = np.concatenate(val_scores)
        val_y = np.concatenate(val_y)
        val_ap = average_precision_score(val_y, val_scores)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_pr_auc": float(val_ap)})
        print(f"Epoch {epoch:02d}: loss={np.mean(losses):.4f}, val PR-AUC={val_ap:.5f}", flush=True)

        if val_ap > best_ap:
            best_ap = val_ap
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.", flush=True)
                break

    model.load_state_dict(best_state)
    model.eval()

    def predict(mask):
        out = []
        ds = TensorDataset(torch.from_numpy(seq[mask]), torch.from_numpy(static[mask]))
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        with torch.no_grad():
            for xb, sb in dl:
                out.append(torch.sigmoid(model(xb.to(DEVICE), sb.to(DEVICE))).cpu().numpy())
        return np.concatenate(out)

    val_score = predict(va)
    test_score = predict(te)
    metrics = classification_metrics(test_score, y[te].astype(int))
    metrics["val_pr_auc"] = float(average_precision_score(y[va], val_score))
    result = {
        "model": "GRU telemetry sequence + static features",
        "info": info,
        "metrics": metrics,
        "history": history,
        "note": (
            "Synthetic-data experiment only. Labels are generated by the simulator; "
            "results do not establish real-world UPI performance. Threshold decisions "
            "must be selected on validation data, not test data."
        ),
        "runtime_s": float(time.time() - start),
    }
    out_json = os.path.join(OUT, "gru_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(os.path.join(OUT, "gru_test_predictions.npz"),
                        y_test=y[te].astype(int), score_test=test_score,
                        y_val=y[va].astype(int), score_val=val_score)
    print("\nTEST METRICS")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved: {out_json}", flush=True)

if __name__ == "__main__":
    main()
