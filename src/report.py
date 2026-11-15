"""Aggregate the per-seed benchmark results into RESULTS.md, summary.json and a results figure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ATTACK_ORDER = ["conflicting_green", "timemark", "ghost_queue", "kinematic"]


def _load(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("result_*.json"))]


def _agg(values: list[float]) -> tuple[float, float]:
    v = np.array([x for x in values if x == x], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std())


def aggregate(results: list[dict]) -> dict:
    out: dict = {"attacks": {}, "clean": {}, "n_seeds": len(results)}
    for a in ATTACK_ORDER:
        rows = [r[a] for r in results if a in r]
        if not rows:
            continue
        out["attacks"][a] = {
            "label": rows[0]["label"], "layer": rows[0]["layer"],
            "detection_rate": _agg([x["detection_rate"] for x in rows]),
            "msg_recall": _agg([x["msg_recall"] for x in rows]),
            "pr_auc": _agg([x["pr_auc"] for x in rows]),
            "median_latency_ms": _agg([x["median_latency_ms"] for x in rows]),
            "scenarios": int(np.sum([x["scenarios"] for x in rows])),
        }
    cl = [r["_clean"] for r in results]
    out["clean"] = {
        "bsm_fpr": _agg([x["bsm_fpr"] for x in cl]),
        "alarms_per_veh_hour": _agg([x["alarms_per_veh_hour"] for x in cl]),
        "spat_fp_msgs": int(np.sum([x["spat_fp_msgs"] for x in cl])),
        "spat_msgs": int(np.sum([x["spat_msgs"] for x in cl])),
    }
    out["meta"] = results[0]["_meta"]
    # mean feature weights
    ws = [r["_meta"]["weights"] for r in results]
    out["weights"] = {k: float(np.mean([w[k] for w in ws])) for k in ws[0]}
    return out


def _pm(t: tuple[float, float], scale: float = 1.0, fmt: str = ".2f") -> str:
    m, s = t
    if m != m:
        return "n/a"
    return f"{m * scale:{fmt}} ± {s * scale:{fmt}}"


def write(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    results = _load(out_dir / "raw")
    if not results:
        raise FileNotFoundError(f"no result_*.json in {out_dir / 'raw'}")
    agg = aggregate(results)
    (out_dir / "summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    m = agg["meta"]
    lines = [
        "# Benchmark results", "",
        f"Real connected-vehicle data: UDOT / Panasonic SPaT and BSM, intersection {m['intersection']} "
        f"({m.get('date')}), leak-free time split (train {m['train_s'] / 3600:.1f} h, test {m['test_s'] / 3600:.1f} h), "
        f"{agg['n_seeds']} seeds. Mean ± SD across seeds.", "",
        "## Detection by attack family", "",
        "| Attack | Detector | Scenarios | Detection rate | Msg recall | PR-AUC | Median latency (ms) |",
        "|---|---|--:|--:|--:|--:|--:|",
    ]
    for a in ATTACK_ORDER:
        if a not in agg["attacks"]:
            continue
        x = agg["attacks"][a]
        lines.append(f"| {x['label']} | {x['layer']} | {x['scenarios']} | {_pm(x['detection_rate'])} | "
                     f"{_pm(x['msg_recall'])} | {_pm(x['pr_auc'], fmt='.3f')} | {_pm(x['median_latency_ms'], fmt='.0f')} |")
    cl = agg["clean"]
    lines += [
        "", "## False alarms on untouched real data", "",
        f"* BSM false-positive rate: **{_pm(cl['bsm_fpr'], 100, '.2f')} %** (operating point calibrated to 1 % on a held-out clean split).",
        f"* Alarms per vehicle-hour: **{_pm(cl['alarms_per_veh_hour'], fmt='.1f')}**.",
        f"* Layer-1 protocol false positives: **{cl['spat_fp_msgs']} / {cl['spat_msgs']:,}** SPaT messages.",
        "", "## Learned Layer-3 feature weights (mean across seeds)", "",
        "| Feature | Weight |", "|---|--:|",
    ]
    for k, v in sorted(agg["weights"].items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"| {k} | {v:+.2f} |")
    lines += ["", "![Results](figures/results.png)", ""]
    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    _figure(agg, out_dir / "figures")
    return out_dir / "RESULTS.md"


def _figure(agg: dict, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    except Exception:  # noqa: BLE001
        pass
    fig_dir.mkdir(parents=True, exist_ok=True)
    attacks = [a for a in ATTACK_ORDER if a in agg["attacks"]]
    labels = [agg["attacks"][a]["label"].replace(" ", "\n", 1) for a in attacks]
    det = [agg["attacks"][a]["detection_rate"][0] for a in attacks]
    det_e = [agg["attacks"][a]["detection_rate"][1] for a in attacks]
    pr = [agg["attacks"][a]["pr_auc"][0] for a in attacks]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(attacks))
    ax[0].bar(x - 0.2, det, 0.4, yerr=det_e, capsize=3, color="#1565c0", label="detection rate")
    ax[0].bar(x + 0.2, pr, 0.4, color="#2e7d32", label="PR-AUC")
    ax[0].set_xticks(x, labels, fontsize=8)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_title("Detection rate and PR-AUC by attack family")
    ax[0].legend(fontsize=8)
    ax[0].axhline(1.0, color="0.7", lw=0.6, ls=":")
    w = agg["weights"]
    order = sorted(w, key=lambda k: w[k])
    ax[1].barh(range(len(order)), [w[k] for k in order], color=["#c62828" if w[k] < 0 else "#1565c0" for k in order])
    ax[1].set_yticks(range(len(order)), order, fontsize=8)
    ax[1].set_title("Layer-3 logistic feature weights")
    ax[1].axvline(0, color="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "results.png", dpi=200)
    plt.close(fig)
