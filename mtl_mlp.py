"""
Multi-task neural network in plain NumPy (no deep learning framework is installable in this workspace).
Shared trunk (two ReLU layers) -> two heads:
  - head 1: P(transaction lands in limbo)                        (weighted binary cross-entropy)
  - head 2: distribution over resolution-time buckets, given limbo (softmax, cross-entropy,
            only on transactions that did land in limbo)
Resolution time is bimodal (late response arrives within hours, or the item waits for a reconciliation
batch), so a bucketed distribution is more honest than a single "expected hours" number.
Optimiser: Adam, with manual backpropagation.
"""
import numpy as np
from sklearn.metrics import average_precision_score

BUCKET_EDGES_H = [1, 12, 24, 48]
BUCKET_LABELS = ["< 1 h", "1-12 h", "12-24 h", "24-48 h", "> 48 h"]
K = len(BUCKET_LABELS)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MultiTaskMLP:
    def __init__(self, n_in, h1=128, h2=64, seed=0, pos_weight=4.0, lam=0.5, l2=1e-5):
        rng = np.random.default_rng(seed)
        self.p = dict(
            W1=rng.normal(0, np.sqrt(2 / n_in), (n_in, h1)), b1=np.zeros(h1),
            W2=rng.normal(0, np.sqrt(2 / h1), (h1, h2)), b2=np.zeros(h2),
            Wc=rng.normal(0, np.sqrt(1 / h2), (h2, 1)), bc=np.zeros(1),
            We=rng.normal(0, np.sqrt(1 / h2), (h2, K)), be=np.zeros(K),
        )
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0
        self.pos_weight, self.lam, self.l2 = pos_weight, lam, l2
        self.rng = rng

    def forward(self, X):
        p = self.p
        z1 = X @ p["W1"] + p["b1"]; a1 = np.maximum(z1, 0)
        z2 = a1 @ p["W2"] + p["b2"]; a2 = np.maximum(z2, 0)
        logit = (a2 @ p["Wc"] + p["bc"])[:, 0]
        probs = softmax(a2 @ p["We"] + p["be"])
        return logit, probs, (X, z1, a1, z2, a2)

    def step(self, X, y, k, mask, lr):
        p = self.p
        n = len(y)
        logit, probs, (X, z1, a1, z2, a2) = self.forward(X)
        pr = sigmoid(logit)
        w = np.where(y == 1, self.pos_weight, 1.0)
        dlogit = w * (pr - y) / n
        nm = max(mask.sum(), 1.0)
        onehot = np.eye(K)[k]
        dze = self.lam * (probs - onehot) * mask[:, None] / nm
        g = {}
        g["Wc"] = a2.T @ dlogit[:, None]; g["bc"] = dlogit.sum(keepdims=True)
        g["We"] = a2.T @ dze; g["be"] = dze.sum(0)
        da2 = dlogit[:, None] @ p["Wc"].T + dze @ p["We"].T
        dz2 = da2 * (z2 > 0)
        g["W2"] = a1.T @ dz2; g["b2"] = dz2.sum(0)
        da1 = dz2 @ p["W2"].T
        dz1 = da1 * (z1 > 0)
        g["W1"] = X.T @ dz1; g["b1"] = dz1.sum(0)
        self.t += 1
        b1c, b2c = 0.9, 0.999
        for name in p:
            grad = g[name] + (self.l2 * p[name] if name.startswith("W") else 0)
            self.m[name] = b1c * self.m[name] + (1 - b1c) * grad
            self.v[name] = b2c * self.v[name] + (1 - b2c) * grad ** 2
            mh = self.m[name] / (1 - b1c ** self.t)
            vh = self.v[name] / (1 - b2c ** self.t)
            p[name] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def predict(self, X):
        logit, probs, _ = self.forward(X)
        return logit, probs

    def fit(self, Xtr, ytr, ktr, mtr, Xva, yva, epochs=40, bs=2048, lr=1e-3, patience=6, verbose=True):
        best, best_p, bad = -1, None, 0
        n = len(ytr)
        for ep in range(epochs):
            perm = self.rng.permutation(n)
            for i in range(0, n, bs):
                j = perm[i:i + bs]
                self.step(Xtr[j], ytr[j], ktr[j], mtr[j], lr)
            logit, _ = self.predict(Xva)
            ap = average_precision_score(yva, logit)
            if verbose:
                print(f"  epoch {ep + 1:2d}  val AP = {ap:.4f}", flush=True)
            if ap > best:
                best, bad = ap, 0
                best_p = {k: v.copy() for k, v in self.p.items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        self.p = best_p
        return best
