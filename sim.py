"""
UPI "limbo" simulator.

Everything here is an ASSUMPTION-DRIVEN SIMULATION, not real bank or NPCI data.
It encodes the mechanism described in the NPCI circulars:
  - the switch waits a short time (~30 s) for the beneficiary side; on timeout the
    transaction is settled as "deemed approved" and the remitter has already been debited
  - banks reconcile such cases in batch cycles, so a miss pushes resolution to the next cycle
Parameters (bank latencies, fragility, miss rates) are illustrative. Banks are anonymised A-H;
Bank A is a large, heavily loaded, more fragile bank (an assumption inspired by public reports
of higher technical-decline rates at large public-sector remitter banks).
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

BANKS = list("ABCDEFGH")
NB = len(BANKS)
START = pd.Timestamp("2026-08-03 00:00")  # a Monday

BASE_LAT = np.array([420, 300, 280, 320, 380, 400, 260, 350], float)   # ms, median-ish
UTIL = np.array([0.80, 0.55, 0.52, 0.50, 0.62, 0.60, 0.45, 0.58])      # peak utilisation
EPI_RATE = np.array([0.40, 0.08, 0.06, 0.07, 0.15, 0.14, 0.05, 0.10])  # degradation episodes / day
FRAGILE = np.array([2.0, 0.2, 0.2, 0.3, 0.8, 0.7, 0.2, 0.5])
MISS = np.array([0.17, 0.04, 0.035, 0.04, 0.08, 0.08, 0.03, 0.06])       # P(miss a recon batch)
SHARE_R = np.array([0.30, 0.20, 0.16, 0.10, 0.08, 0.07, 0.05, 0.04])
SHARE_B = np.array([0.22, 0.20, 0.18, 0.12, 0.09, 0.08, 0.06, 0.05])
SHARE_R = SHARE_R / SHARE_R.sum()
SHARE_B = SHARE_B / SHARE_B.sum()
OK_LEVEL = 0.15           # degradation level below which a bank counts as recovered
BATCH_HOUR = 6            # recon batch at 06:00, Mon-Sat
CUTOFF_MIN = 120          # an item must be ready this long before the batch to make it


def _ar1(rng, T, phi=0.985, sigma=0.015):
    return lfilter([1.0], [1.0, -phi], rng.normal(0, sigma, T))


def make_world(days=70, seed=7):
    rng = np.random.default_rng(seed)
    T = days * 1440
    t = np.arange(T)
    hour = (t % 1440) / 60.0
    day = t // 1440
    dow = day % 7                      # day 0 = Monday
    dates = START + pd.to_timedelta(np.arange(days), unit="D")
    dom = np.repeat(dates.day.values, 1440)

    diurnal = (0.15 + 0.55 * np.exp(-0.5 * ((hour - 12.5) / 3.5) ** 2)
               + 0.55 * np.exp(-0.5 * ((hour - 20.0) / 2.5) ** 2))
    diurnal = diurnal / diurnal.max()
    mult = np.where(dom <= 5, 1.25, 1.0) * np.where(dow >= 5, 0.85, 1.0)   # rent-week spike, weekend dip
    gload = diurnal * mult

    u = np.empty((NB, T))
    for b in range(NB):
        u[b] = np.clip(UTIL[b] * gload * (1 + _ar1(rng, T)), 0, 1.3)

    # degradation episodes (latent): ramp up, plateau, ramp down; more likely when load is high
    w = gload / gload.sum()
    d = np.zeros((NB, T))
    for b in range(NB):
        n = rng.poisson(EPI_RATE[b] * days)
        for _ in range(n):
            s = int(rng.choice(T, p=w))
            ramp = int(rng.uniform(10, 45))
            dur = int(np.clip(rng.lognormal(np.log(60), 0.7), 15, 360))
            down = int(rng.uniform(5, 20))
            peak = rng.uniform(0.5, 1.0)
            shape = np.concatenate([np.linspace(0, peak, ramp, endpoint=False),
                                    np.full(dur, peak), np.linspace(peak, 0, down)])
            e = min(T, s + len(shape))
            d[b, s:e] = np.maximum(d[b, s:e], shape[: e - s])

    # observed telemetry (what a bank could actually monitor)
    lat = BASE_LAT[:, None] * (1 + 0.6 * u ** 2 + 2.5 * d) * np.exp(rng.normal(0, 0.12, (NB, T)))
    to = np.clip(0.002 + 0.30 * d ** 2 + 0.03 * np.maximum(u - 0.9, 0) + rng.normal(0, 0.002, (NB, T)), 0, 1)
    u_obs = np.clip(u + rng.normal(0, 0.02, (NB, T)), 0, 1.5)

    idx = np.where(d < OK_LEVEL, np.arange(T), T + 100000)
    next_ok = np.stack([np.minimum.accumulate(idx[b][::-1])[::-1] for b in range(NB)])

    # reconciliation batches: 06:00 Mon-Sat, extended beyond the horizon
    bt = []
    for k in range(days + 20):
        dd = START + pd.Timedelta(days=k)
        if dd.dayofweek <= 5:
            bt.append(k * 1440 + BATCH_HOUR * 60)
    return dict(T=T, days=days, hour=hour, dow=dow, dom=dom, gload=gload, u=u, d=d,
                lat=lat, to=to, u_obs=u_obs, next_ok=next_ok, batches=np.array(bt), rng=rng)


def make_transactions(world, n=450_000, seed=11):
    rng = np.random.default_rng(seed)
    T = world["T"]
    d, u, gload = world["d"], world["u"], world["gload"]
    w = gload[60:] / gload[60:].sum()
    t = 60 + rng.choice(T - 60, size=n, p=w)
    rem = rng.choice(NB, size=n, p=SHARE_R)
    ben = rng.choice(NB, size=n, p=SHARE_B)
    amt = np.clip(np.round(np.exp(rng.normal(np.log(900), 1.3, n))), 1, 100000)

    pb = np.clip(0.002 * (1 + FRAGILE[ben]) + 0.55 * d[ben, t] ** 2
                 + 0.04 * np.maximum(u[ben, t] - 0.9, 0), 0, 0.95)
    pr = np.clip(0.0008 + 0.10 * d[rem, t] ** 2, 0, 0.95)
    p = 1 - (1 - pb) * (1 - pr)
    p = p * np.clip(0.8 + 0.1 * np.clip(np.log10(amt), 0, 5), 0.8, 1.3)   # larger amounts a bit riskier
    limbo = rng.random(n) < p

    res_h = np.full(n, np.nan)
    batches = world["batches"]
    next_ok = world["next_ok"]
    for i in np.where(limbo)[0]:
        tt = int(t[i])
        culprit = ben[i] if rng.random() < pb[i] / (pb[i] + pr[i]) else rem[i]
        # fast path: a late response from the slow side arrives and auto-resolves the item (minutes to hours).
        # Less likely when the culprit bank is badly degraded, and for the fragile large bank.
        p_fast = float(np.clip(0.80 * (1 - d[culprit, tt]) - 0.15 * (culprit == 0), 0.05, 0.9))
        if rng.random() < p_fast:
            res_h[i] = float(np.exp(np.log(30) + rng.normal(0, 1.0)) / 60.0)
            continue
        nok = int(next_ok[culprit, tt])
        ready = tt + (240 if nok >= T else max(0, nok - tt))
        ready += rng.lognormal(np.log(20), 0.8)                         # queueing inside the bank
        k = int(np.searchsorted(batches, ready + CUTOFF_MIN))
        pm = min(0.85, MISS[culprit] * (1 + 1.5 * d[culprit, tt]))
        while rng.random() < pm and k < len(batches) - 2:               # missed the batch -> next working day
            k += 1
            pm *= 0.6
        t_res = batches[k] + rng.lognormal(np.log(120), 0.6)
        if rng.random() < 0.02:                                         # manual escalation / dispute path
            t_res += rng.uniform(24, 60) * 60
        res_h[i] = (t_res - tt) / 60.0

    return pd.DataFrame(dict(t=t, rem=rem, ben=ben, amt=amt, limbo=limbo.astype(int), res_h=res_h))


def feature_names(W=10):
    names, groups = [], []
    for side in ("ben", "rem"):
        for f in ("lat_rel", "timeout_pct", "util"):
            for k in range(W):
                names.append(f"{side}_{f}_lag{1 + 3 * k}")
                groups.append(f"telemetry_{side}")
    for b in BANKS:
        names.append(f"rem_is_{b}"); groups.append("banks")
    for b in BANKS:
        names.append(f"ben_is_{b}"); groups.append("banks")
    names += ["log10_amount", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "rent_week"]
    groups += ["amount", "time", "time", "time", "time", "time"]
    return names, np.array(groups)


def build_features(world, t, rem, ben, amt, hour=None, dow=None, dom=None, W=10, step=3):
    """Features available AT decision time t: telemetry from minutes t-1 ... t-(step*(W-1)+1) only."""
    t = np.asarray(t); rem = np.asarray(rem); ben = np.asarray(ben); amt = np.asarray(amt, float)
    lags = 1 + step * np.arange(W)
    tele = np.stack([world["lat"] / BASE_LAT[:, None], np.log1p(world["to"] * 100), world["u_obs"]], axis=0)
    cols = []
    for bank in (ben, rem):
        for f in range(3):
            cols.append(tele[f][bank[:, None], t[:, None] - lags[None, :]])
    tel = np.concatenate(cols, axis=1)
    oh_r = np.eye(NB)[rem]
    oh_b = np.eye(NB)[ben]
    h = world["hour"][t] if hour is None else np.broadcast_to(hour, t.shape)
    dw = world["dow"][t] if dow is None else np.broadcast_to(dow, t.shape)
    dm = world["dom"][t] if dom is None else np.broadcast_to(dom, t.shape)
    tf = np.stack([np.log10(amt), np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                   np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7), (dm <= 5).astype(float)], axis=1)
    return np.concatenate([tel, oh_r, oh_b, tf], axis=1).astype(np.float32)
