"""Build scenarios from real data, train the detector and evaluate it, with a leak-free time split.

A real day is cut into disjoint time blocks: the earliest for training, the next for validation (threshold
only), the rest for testing. Scenarios inject each attack at many onsets drawn from the block, into real
SPaT/BSM, so training and test attacks come from different clock times and different real traffic.

Metrics, reported per attack type:

* attack-level detection -- an attack is "detected" if at least one of its tampered messages scores above the
  threshold; detection latency is the time from the attack's first tampered message to that alarm;
* message-level PR-AUC and recall at the operating threshold;
* clean false-alarm rate: tampered-message rate on untouched baseline scenarios, and alarms per vehicle-hour.

SPaT attacks (conflicting_green, timemark) are judged by Layer 1 (deterministic invariant flags); BSM attacks
(ghost_queue, kinematic) by the Layer 3 fused score. ``run`` sweeps seeds and writes raw results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import RealData, load_day
from .detector import Detector, protocol_flags
from .scenario_generator import ATTACKS, LABELS, VARIANTS, ScenarioConfig, generate
from .nema_controller import estimate_timing

SPAT_ATTACKS = {"conflicting_green", "timemark"}
BSM_ATTACKS = {"ghost_queue", "kinematic"}


def pr_auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    """Average-precision (area under precision-recall), no sklearn. ``w`` are per-sample weights, used to
    undo negative subsampling so precision reflects the true class balance."""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    w = np.ones(len(y)) if w is None else w
    order = np.argsort(-s)
    y, w = y[order], w[order]
    tp = np.cumsum(y * w)
    fp = np.cumsum((1 - y) * w)
    prec = tp / (tp + fp)
    rec = tp / tp[-1]
    rec = np.r_[0, rec]
    prec = np.r_[1, prec]
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


@dataclass
class Spec:
    attack: str
    variant: str


def all_specs() -> list[Spec]:
    return [Spec(a, v) for a in ATTACKS[1:] for v in VARIANTS[a]]


def _onsets(block_dur: float, n: int, rng: np.random.Generator, margin: float = 40.0) -> list[float]:
    lo, hi = margin, block_dur - margin
    return sorted(rng.uniform(lo, hi, n).tolist())


def _approach_onsets(block: RealData, n: int, rng: np.random.Generator, margin: float = 20.0) -> list[float]:
    """Onsets at the moments distinct equipped vehicles are approaching a stop bar, so a kinematic
    falsification always has a real target. Equipped-vehicle penetration is low, so these are scarce."""
    b = block.bsm
    t0 = b["t"].min()
    cand = b[(b["lane"] >= 0) & (b["speed"] > 3) & (b["s"] < -3) & (b["s"] > -250)]
    if cand.empty:
        return _onsets(block.duration, n, rng)
    first = cand.groupby("id")["t"].min()
    rel = (first - t0).to_numpy()
    rel = rel[(rel > margin) & (rel < block.duration - margin)]
    if len(rel) == 0:
        return _onsets(block.duration, n, rng)
    pick = rng.choice(rel, size=min(n, len(rel)), replace=False)
    return sorted(float(x) for x in pick)


def iter_scenarios(block: RealData, specs: list[Spec], n_onsets: int, seed: int, rng: np.random.Generator, duration: float = 10.0):
    """Yield one injected scenario at a time (streamed to keep memory low). Kinematic attacks use onsets
    anchored to real approaching vehicles; SPaT and ghost attacks use onsets spread across the block."""
    for sp in specs:
        onsets = _approach_onsets(block, n_onsets, rng) if sp.attack == "kinematic" else _onsets(block.duration, n_onsets, rng)
        for k, on in enumerate(onsets):
            sc = generate(block, ScenarioConfig(attack=sp.attack, variant=sp.variant, onset=on, duration=duration, seed=seed * 100 + k))
            yield sc


def evaluate(det: Detector, scenarios, baseline: object, rng: np.random.Generator | None = None) -> dict:
    """Per-attack-type metrics over a stream of test scenarios plus one clean baseline. Negative (clean)
    messages are subsampled for the PR-AUC pool so memory stays bounded; all positives are kept."""
    rng = rng or np.random.default_rng(0)
    thr = det.threshold_
    by_type: dict[str, dict] = {}
    for sc in scenarios:
        name = sc.config.attack
        d = by_type.setdefault(name, {"detected": 0, "n": 0, "latencies": [], "msg_y": [], "msg_s": []})
        if sc.config.attack in SPAT_ATTACKS:
            flags = protocol_flags(sc.spat, det.plan, det.enabled_checks)
            score = (flags["n_viol"].to_numpy() > 0).astype(float)
            y = sc.spat_labels
            t = sc.spat["t"].to_numpy()
        else:
            score = det.score_bsm(sc.bsm, sc.spat)
            y = sc.bsm_labels
            t = sc.bsm["t"].to_numpy()
        d.setdefault("msg_w", [])
        d["n"] += 1
        pos = y == 1
        sub = 0.15
        keep = pos | (rng.random(len(y)) < sub)  # all positives + a sample of negatives for the PR-AUC pool
        d["msg_y"].append(y[keep])
        d["msg_s"].append(score[keep])
        d["msg_w"].append(np.where(pos[keep], 1.0, 1.0 / sub))  # reweight kept negatives to full count
        if pos.sum() == 0:
            continue
        hit = (score > thr) & pos
        if hit.any():
            d["detected"] += 1
            d["latencies"].append(float(t[hit].min() - t[pos].min()))
    out = {}
    for name, d in by_type.items():
        y = np.concatenate(d["msg_y"])
        s = np.concatenate(d["msg_s"])
        wt = np.concatenate(d["msg_w"])
        lat = np.array(d["latencies"])
        out[name] = {
            "label": LABELS[name], "layer": "L1 protocol" if name in SPAT_ATTACKS else "L3 fused",
            "scenarios": d["n"], "detection_rate": d["detected"] / d["n"] if d["n"] else float("nan"),
            "median_latency_ms": float(np.median(lat) * 1000) if len(lat) else float("nan"),
            "p90_latency_ms": float(np.percentile(lat, 90) * 1000) if len(lat) else float("nan"),
            "msg_recall": float((s[y == 1] > thr).mean()) if y.sum() else float("nan"),
            "pr_auc": pr_auc(y, s, wt), "n_tampered_msgs": int(y.sum()),
        }
    # clean false alarms on the baseline
    cs = det.score_bsm(baseline.bsm, baseline.spat)
    dur_h = (baseline.bsm["t"].max() - baseline.bsm["t"].min()) / 3600
    veh_hours = baseline.bsm["id"].nunique() * dur_h
    flags = protocol_flags(baseline.spat, det.plan, det.enabled_checks)
    out["_clean"] = {
        "bsm_fpr": float((cs > thr).mean()), "bsm_alarms": int((cs > thr).sum()),
        "alarms_per_veh_hour": float((cs > thr).sum() / veh_hours) if veh_hours else float("nan"),
        "spat_fp_msgs": int((flags["n_viol"].to_numpy() > 0).sum()), "spat_msgs": int(len(baseline.spat)),
        "veh_hours": round(float(veh_hours), 2),
    }
    return out


def run_once(data: RealData, seed: int = 0, train_frac: float = 0.4, val_frac: float = 0.2,
             n_onsets_train: int = 6, n_onsets_test: int = 10, duration: float = 10.0) -> dict:
    dur = data.duration
    tr = data.window(0, dur * train_frac)
    va = data.window(dur * train_frac, dur * val_frac)
    te = data.window(dur * (train_frac + val_frac), dur * (1 - train_frac - val_frac))
    rng = np.random.default_rng(seed)
    specs = all_specs()
    plan = estimate_timing(data.spat)

    bsm_specs = [s for s in specs if s.attack in BSM_ATTACKS]

    def train_stream():
        base = generate(tr, ScenarioConfig(attack="none"))
        yield base.bsm, base.spat
        for sc in iter_scenarios(tr, bsm_specs, n_onsets_train, seed, rng, duration):
            yield sc.bsm, sc.spat

    det = Detector(plan=plan).fit(train_stream(), seed)

    base_va = generate(va, ScenarioConfig(attack="none"))
    det.set_threshold(base_va.bsm, base_va.spat, target_fpr=0.01)

    test_scen = iter_scenarios(te, specs, n_onsets_test, seed + 1, rng, duration)
    base_te = generate(te, ScenarioConfig(attack="none"))
    res = evaluate(det, test_scen, base_te, rng)
    res["_meta"] = {"seed": seed, "intersection": data.intersection, "date": data.date, "threshold": det.threshold_,
                    "train_s": round(tr.duration, 1), "test_s": round(te.duration, 1),
                    "weights": {f: float(w) for f, w in zip(list(_feat_names()), det.w_)}}
    return res


def _feat_names() -> list[str]:
    from .detector import CTX, PHYS
    return list(PHYS) + list(CTX) + ["L1_flag"]


def run(data_root: str, out: str | Path, intersection: str = "6311", date: str = "2025-02-07", seeds: int = 5,
        start_s: float = 7 * 3600, hours: float = 10.0) -> Path:
    """Train and evaluate over several seeds on a bounded window of the real day (default: a 10-hour daytime
    span, which holds the bulk of the equipped-vehicle activity and keeps per-scenario memory modest)."""
    out = Path(out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    day = load_day(data_root, intersection, date)
    data = day.window(start_s, hours * 3600)
    print(f"window: {data.duration / 3600:.1f} h, {len(data.spat):,} SPaT, {len(data.bsm):,} BSM", flush=True)
    for seed in range(seeds):
        r = run_once(data, seed=seed)
        r["_meta"]["window"] = {"start_s": start_s, "hours": hours}
        (out / "raw" / f"result_{intersection}_{date}_{seed}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(f"seed {seed} done", flush=True)
    return out
