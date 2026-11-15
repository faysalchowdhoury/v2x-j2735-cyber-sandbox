"""Three-layer anomaly detector for real J2735 SPaT and BSM streams.

Layer 1 -- NEMA / J2735 protocol invariants (``nema_controller.check_stream``): per-SPaT-message flags for
conflict-matrix, transition, clearance and TimeMark violations, calibrated so a clean real day raises none.

Layer 2 -- physics-informed kinematic residuals per BSM, computed from consecutive messages of the same
sender: position residual against a constant-velocity prediction, implied longitudinal acceleration, jerk,
speed jump, and agreement between the reported acceleration (when present) and the position-derived one. Each
is turned into a one-sided z-score against clean-data scale (median / robust MAD).

Layer 3 -- a lightweight spatiotemporal scorer. It fuses the Layer-2 residuals with context features (distance
to the stop bar, the signal shown to the vehicle, speed relative to lane neighbours, local sender density) and
any Layer-1 flag on the vehicle's phase, through a logistic model (numpy; uses torch automatically when
installed) into a calibrated ``[0, 1]`` risk score per BSM. The score is trained on labelled scenarios and the
decision threshold is set on a clean validation split at a target false-alarm rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_loader import RealData, signal_at
from .nema_controller import DEFAULT_CHECKS, PHASES, TimingPlan, check_stream, estimate_timing

try:  # optional acceleration; the numpy path is the default and gives identical results
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False

PHYS = ("pos_resid", "implied_acc", "jerk", "speed_jump", "acc_mismatch", "speed_consist", "speed_consist_w", "offroad")
CTX = ("to_stopbar", "sig_go", "sig_stop", "rel_speed", "density", "stopped_on_go")
FEATURES = PHYS + CTX


# ------------------------------------------------------------------------------ Layer 2 features

def kinematic_features(bsm: pd.DataFrame) -> pd.DataFrame:
    """Per-BSM physics residuals from consecutive same-sender messages. First message of each sender is 0."""
    if len(bsm) <= 7:  # too few messages for the windowed residual; everything is "unknown" (valid=False)
        z = pd.DataFrame({c: np.zeros(len(bsm)) for c in PHYS}, index=bsm.index)
        z["_valid"] = False
        return z
    b = bsm.sort_values(["id", "t"])
    gid = b["id"].to_numpy()
    t = b["t"].to_numpy(dtype=float)
    x, y = b["x"].to_numpy(dtype=float), b["y"].to_numpy(dtype=float)
    v = b["speed"].to_numpy(dtype=float)
    hd = np.radians(b["heading"].to_numpy(dtype=float))
    same = np.r_[False, gid[1:] == gid[:-1]]
    dt = np.where(same, np.r_[np.nan, np.diff(t)], np.nan)
    dt = np.where((dt > 0.02) & (dt < 2.0), dt, np.nan)
    dx = np.r_[np.nan, np.diff(x)]
    dy = np.r_[np.nan, np.diff(y)]
    dist = np.hypot(dx, dy)
    vprev = np.r_[np.nan, v[:-1]]
    # predicted displacement from previous speed+heading over dt (constant velocity)
    pred = vprev * dt
    pos_resid = np.abs(dist - pred)
    dv = np.r_[np.nan, np.diff(v)]
    implied_acc = dv / dt
    aprev = np.r_[np.nan, implied_acc[:-1]]
    jerk = (implied_acc - aprev) / dt
    speed_jump = np.abs(dv) / dt
    a_rep = b["a_long"].to_numpy(dtype=float)
    acc_mismatch = np.where(np.isfinite(a_rep), np.abs(a_rep - implied_acc), 0.0)
    # reported speed vs. speed implied by the displacement between consecutive messages: a constant additive
    # speed offset (speed_spoof) is inconsistent here on every message, not only at the attack edges.
    speed_consist = np.abs(v - dist / dt)
    # windowed version: displacement-implied speed over ~0.6 s (6 messages) beats the per-step GPS noise that
    # swamps the single-step residual. Compared against the mean reported speed over the same window.
    K = 6
    xb = np.r_[[np.nan] * K, x[:-K]]
    yb = np.r_[[np.nan] * K, y[:-K]]
    tb = np.r_[[np.nan] * K, t[:-K]]
    gidb = np.r_[np.array([""] * K, dtype=gid.dtype), gid[:-K]]
    vbw = pd.Series(v).rolling(K, min_periods=K).mean().to_numpy()
    same_w = np.r_[np.zeros(K, bool), gid[K:] == gid[:-K]]  # same sender K messages back
    dtw = np.where(same_w, t - tb, np.nan)
    dtw = np.where((dtw > 0.2) & (dtw < 3.0), dtw, np.nan)
    distw = np.hypot(x - xb, y - yb)
    speed_consist_w = np.abs(vbw - distw / dtw)
    # off-road: lateral distance from the matched lane centreline, 0 when unmatched (inf) and capped at 20 m
    offroad = b["lane_lat"].to_numpy(dtype=float) if "lane_lat" in b else np.zeros(len(b))
    offroad = np.clip(np.nan_to_num(offroad, nan=0.0, posinf=0.0), 0.0, 20.0)
    out = pd.DataFrame({
        "pos_resid": pos_resid, "implied_acc": np.abs(implied_acc), "jerk": np.abs(jerk),
        "speed_jump": speed_jump, "acc_mismatch": acc_mismatch, "speed_consist": speed_consist,
        "speed_consist_w": speed_consist_w, "offroad": offroad,
    }, index=b.index)
    out = out.reindex(bsm.index)
    valid = np.isfinite(dt)
    out["_valid"] = pd.Series(valid, index=b.index).reindex(bsm.index).fillna(False)
    return out.fillna(0.0)


def context_features(bsm: pd.DataFrame, spat: pd.DataFrame) -> pd.DataFrame:
    """Signal- and neighbour-aware context per BSM (does not use ground-truth labels)."""
    if len(bsm) == 0:
        return pd.DataFrame({c: np.zeros(0) for c in CTX}, index=bsm.index)
    t = bsm["t"].to_numpy(dtype=float)
    phase = bsm["phase"].to_numpy()
    s = bsm["s"].to_numpy(dtype=float)
    v = bsm["speed"].to_numpy(dtype=float)
    sig = bsm["signal"].to_numpy() if "signal" in bsm else signal_at(spat, t, phase)
    to_stopbar = np.clip(np.nan_to_num(-s, nan=400.0), -50, 400)  # unmatched (nan) -> treated as far away
    sig_go = (np.isin(sig, list("Gg"))).astype(float)
    sig_stop = (sig == "R").astype(float)
    # neighbour speed and density within a 1 s time slot on the same phase
    slot = np.round(t).astype(np.int64)
    rel_speed = np.zeros(len(bsm))
    density = np.zeros(len(bsm))
    order = np.lexsort((phase, slot))
    key = slot * 16 + np.clip(phase, 0, 15)
    ks = key[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(ks)) + 1, len(ks)]
    for a, bnd in zip(bounds[:-1], bounds[1:]):
        grp = order[a:bnd]
        if phase[grp[0]] <= 0:
            continue
        vv = v[grp]
        med = np.median(vv)
        rel_speed[grp] = vv - med
        density[grp] = len(grp)
    stopped_on_go = ((v < 1.0) & (sig_go > 0) & (s < 0) & (s > -40)).astype(float)
    return pd.DataFrame({"to_stopbar": to_stopbar, "sig_go": sig_go, "sig_stop": sig_stop,
                         "rel_speed": rel_speed, "density": density, "stopped_on_go": stopped_on_go}, index=bsm.index)


# ------------------------------------------------------------------------------ Layer 1 wrapper

def protocol_flags(spat: pd.DataFrame, plan: TimingPlan, enabled: tuple[str, ...] = DEFAULT_CHECKS) -> pd.DataFrame:
    viol = check_stream(spat, plan, enabled)
    phases_hit = []
    for v in viol:
        ph = set()
        for item in v:
            for tok in item.replace("+", " ").replace("<", " ").split()[1:]:
                if tok.isdigit():
                    ph.add(int(tok))
        phases_hit.append(ph)
    return pd.DataFrame({"t": spat["t"].to_numpy(), "n_viol": [len(v) for v in viol],
                         "violations": viol, "phases": phases_hit})


# ------------------------------------------------------------------------------ Layer 3 model

# physical floor for each feature's scale, so a feature that is almost always zero on clean data (e.g.
# position residual) still maps a genuine deviation to a moderate z-score instead of saturating.
_FLOOR = {"pos_resid": 1.5, "implied_acc": 1.5, "jerk": 3.0, "speed_jump": 1.5, "acc_mismatch": 1.5, "speed_consist": 1.5,
          "speed_consist_w": 1.0, "offroad": 1.5, "to_stopbar": 30.0, "rel_speed": 2.0, "density": 2.0}


def _robust_scale(x: np.ndarray, floor: float = 1e-6) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return floor
    mad = np.median(np.abs(x - np.median(x))) * 1.4826
    p90 = np.percentile(np.abs(x - np.median(x)), 90)
    return float(max(mad, p90, floor))


@dataclass
class Detector:
    enabled_checks: tuple[str, ...] = DEFAULT_CHECKS
    plan: TimingPlan | None = None
    scale_: dict = field(default_factory=dict)
    w_: np.ndarray | None = None
    b_: float = 0.0
    mu_: np.ndarray | None = None
    sd_: np.ndarray | None = None
    threshold_: float = 0.5

    # ---- feature assembly
    def _features(self, bsm: pd.DataFrame, spat: pd.DataFrame, flags: pd.DataFrame | None) -> tuple[pd.DataFrame, np.ndarray]:
        phys = kinematic_features(bsm)
        ctx = context_features(bsm, spat)
        feats = pd.concat([phys[list(PHYS)], ctx[list(CTX)]], axis=1)
        # Layer-1 flag mapped onto each BSM: is this vehicle's phase flagged at its time?
        l1 = np.zeros(len(bsm))
        if flags is not None and flags["n_viol"].sum() > 0:
            ft = flags["t"].to_numpy()
            fph = flags["phases"].tolist()
            fn = flags["n_viol"].to_numpy()
            idx = np.clip(np.searchsorted(ft, bsm["t"].to_numpy(), side="right") - 1, 0, len(ft) - 1)
            bphase = bsm["phase"].to_numpy()
            for i, (j, p) in enumerate(zip(idx, bphase)):
                if fn[j] and (p in fph[j] or len(fph[j]) == 0):
                    l1[i] = 1.0
        return feats, l1, phys["_valid"].to_numpy().astype(bool)

    def _zscores(self, feats: pd.DataFrame) -> np.ndarray:
        z = np.zeros((len(feats), len(FEATURES)))
        for i, c in enumerate(FEATURES):
            x = feats[c].to_numpy(dtype=float)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if c in self.scale_:
                med, sc = self.scale_[c]
                z[:, i] = np.clip((x - med) / sc, -8, 12)
            else:
                z[:, i] = x
        return z

    def set_scales(self, base_bsm: pd.DataFrame, base_spat: pd.DataFrame) -> None:
        """Set robust feature scales from the clean rows of a baseline scenario."""
        kf = kinematic_features(base_bsm)
        base_feats = pd.concat([kf[list(PHYS)], context_features(base_bsm, base_spat)[list(CTX)]], axis=1)
        clean = (base_bsm["label"].to_numpy() == 0) & kf["_valid"].to_numpy().astype(bool)
        for c in PHYS:
            x = base_feats[c].to_numpy()[clean]
            self.scale_[c] = (float(np.median(x)), _robust_scale(x, _FLOOR.get(c, 1e-6)))
        for c in CTX:
            x = base_feats[c].to_numpy()[clean]
            self.scale_[c] = (float(np.median(x)), _robust_scale(x, _FLOOR[c]) if c in ("to_stopbar", "rel_speed", "density") else 1.0)

    def feature_rows(self, bsm: pd.DataFrame, spat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Z-scored feature matrix (with the Layer-1 flag column), labels, and per-row attack name, valid only."""
        flags = protocol_flags(spat, self.plan, self.enabled_checks)
        feats, l1, valid = self._features(bsm, spat, flags)
        z = self._zscores(feats)
        atk = bsm["attack"].to_numpy() if "attack" in bsm else np.array(["none"] * len(bsm))
        return np.c_[z, l1][valid], bsm["label"].to_numpy()[valid], atk[valid]

    # ---- fit
    def fit(self, scenarios, seed: int = 0) -> "Detector":
        """Fit on an iterable of (bsm, spat) scenario tables carrying ``label``; the first is the clean
        baseline (sets the robust scales). Scenarios are consumed one at a time to keep memory low. Positive
        rows are reweighted so each attack type contributes equally, so a high-volume attack (ghost queue)
        does not dominate the model over a low-volume one (kinematic)."""
        it = iter(scenarios)
        base_bsm, base_spat = next(it)
        if self.plan is None:
            self.plan = estimate_timing(base_spat)
        self.set_scales(base_bsm, base_spat)
        allf, ally, alla = [], [], []
        for bsm, spat in [(base_bsm, base_spat), *it]:
            Xi, yi, ai = self.feature_rows(bsm, spat)
            allf.append(Xi)
            ally.append(yi)
            alla.append(ai)
        X = np.vstack(allf)
        y = np.concatenate(ally).astype(float)
        atk = np.concatenate(alla)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        Xs = (X - self.mu_) / self.sd_
        # per-attack-type positive weights: each attack's positives share the same total mass
        w = np.ones(len(y))
        pos_types = [a for a in np.unique(atk) if a != "none"]
        for a in pos_types:
            m = (atk == a) & (y > 0)
            if m.sum():
                w[m] = 1.0 / m.sum()  # each attack type's positives share equal total mass
        w[y > 0] *= max((y > 0).sum(), 1) / max(w[y > 0].sum(), 1e-9)  # renormalise positive mass
        self.w_, self.b_ = _fit_logistic(Xs, y, seed, sample_weight=w)
        return self

    # ---- score
    def score_bsm(self, bsm: pd.DataFrame, spat: pd.DataFrame, flags: pd.DataFrame | None = None) -> np.ndarray:
        if flags is None:
            flags = protocol_flags(spat, self.plan, self.enabled_checks)
        feats, l1, _ = self._features(bsm, spat, flags)
        z = self._zscores(feats)
        X = (np.c_[z, l1] - self.mu_) / self.sd_
        return _sigmoid(X @ self.w_ + self.b_)

    def score_spat(self, spat: pd.DataFrame, flags: pd.DataFrame | None = None) -> np.ndarray:
        if flags is None:
            flags = protocol_flags(spat, self.plan, self.enabled_checks)
        return (flags["n_viol"].to_numpy() > 0).astype(float)

    def set_threshold(self, clean_bsm: pd.DataFrame, clean_spat: pd.DataFrame, target_fpr: float = 0.01) -> float:
        s = self.score_bsm(clean_bsm, clean_spat)
        self.threshold_ = float(np.quantile(s, 1 - target_fpr))
        return self.threshold_


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_logistic(X: np.ndarray, y: np.ndarray, seed: int = 0, iters: int = 500, lr: float = 0.5, l2: float = 1e-3,
                  sample_weight: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Class-balanced logistic regression by full-batch gradient descent (numpy; deterministic). ``sample_weight``
    additionally reweights individual rows (e.g. to balance attack types) before the class balance is applied."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    sw0 = np.ones(n) if sample_weight is None else sample_weight
    pos_mass = max((sw0 * (y > 0)).sum(), 1e-9)
    neg_mass = max((sw0 * (y <= 0)).sum(), 1e-9)
    sw = sw0 * np.where(y > 0, n / (2 * pos_mass), n / (2 * neg_mass))
    for _ in range(iters):
        p = _sigmoid(X @ w + b)
        g = (p - y) * sw
        w -= lr * (X.T @ g / n + l2 * w)
        b -= lr * g.mean()
    return w, float(b)
