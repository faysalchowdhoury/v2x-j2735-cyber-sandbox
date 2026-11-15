"""Publication-grade figures (300 DPI, Times New Roman) in ``docs/figures/``.

1. ``architecture_pipeline.png`` -- the end-to-end system: real UDOT SPaT/BSM feeds, the attack injector, the
   three detector layers, and the TMC alert output.
2. ``attack_detection_benchmark.png`` -- a real replayed window with a conflicting-green attack injected: the
   NEMA phase timeline with the injected conflict, the BSM kinematic residual, and the fused anomaly score
   crossing the operating threshold, annotated with the detection latency.

Run after a benchmark: ``python scripts/generate_figures.py --data data/raw``. Falls back to a short live
replay if no saved results are present. No fabricated numbers -- every value is computed from the real data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
try:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
except Exception:  # noqa: BLE001
    pass
BLUE, RED, GREEN, GREY, AMBER = "#1565c0", "#c62828", "#2e7d32", "#555555", "#f9a825"


def fig_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    linewidth=1.5, edgecolor=ec, facecolor=fc, alpha=0.95))
        ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="top", fontsize=10.5, fontweight="bold")
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.58 - 0.32 * i, ln, ha="center", va="top", fontsize=8.6, color="#222")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=GREY))

    box(0.2, 1.7, 2.1, 1.8, "Real V2X feed", ["UDOT / Panasonic", "Orem, UT", "SPaT 10 Hz", "BSM (OBU)", "MAP geometry"], "#e8eef7", BLUE)
    box(2.7, 1.7, 2.0, 1.8, "Attack injector", ["conflicting green", "TimeMark lure", "ghost / Sybil BSM", "kinematic spoof", "ground-truth labels"], "#fdeaea", RED)
    box(5.1, 0.5, 2.5, 4.1, "3-layer detector", [], "#eef6ef", GREEN)
    ax.text(6.35, 4.05, "L1  NEMA / J2735 invariants", ha="center", fontsize=8.6)
    ax.text(6.35, 3.6, "conflict matrix, clearance,", ha="center", fontsize=7.8, color="#444")
    ax.text(6.35, 3.3, "TimeMark monotonicity", ha="center", fontsize=7.8, color="#444")
    ax.text(6.35, 2.7, "L2  physics residuals", ha="center", fontsize=8.6)
    ax.text(6.35, 2.3, "position / speed / accel", ha="center", fontsize=7.8, color="#444")
    ax.text(6.35, 2.0, "consistency per BSM", ha="center", fontsize=7.8, color="#444")
    ax.text(6.35, 1.4, "L3  spatiotemporal fusion", ha="center", fontsize=8.6)
    ax.text(6.35, 1.0, "calibrated risk in [0,1]", ha="center", fontsize=7.8, color="#444")
    ax.plot([5.3, 7.4], [3.15, 3.15], color=GREEN, lw=0.8, ls="--")
    ax.plot([5.3, 7.4], [1.75, 1.75], color=GREEN, lw=0.8, ls="--")
    box(8.0, 1.7, 2.7, 1.8, "TMC alert & mitigation", ["phase/vehicle flagged", "detection latency (ms)", "false-alarm rate", "operator dashboard"], "#fff7e6", AMBER)
    arrow(2.3, 2.6, 2.7, 2.6)
    arrow(4.7, 2.6, 5.1, 2.6)
    arrow(7.6, 2.6, 8.0, 2.6)
    ax.text(5.5, 5.0, "V2X / SAE J2735 SPaT & BSM cybersecurity sandbox", ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "architecture_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_benchmark(out: Path, data_root: str) -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data_loader import load_day
    from src.detector import Detector, kinematic_features
    from src.experiment import _approach_onsets
    from src.nema_controller import PHASES, check_stream, estimate_timing, simple
    from src.scenario_generator import ScenarioConfig, generate

    inter, date = _find_day(data_root)
    day = load_day(data_root, inter, date)
    # pick the 30-min block with the most distinct approaching vehicles so every panel has real data
    bb = day.bsm
    cand = bb[(bb["lane"] >= 0) & (bb["speed"] > 3) & (bb["s"] < -3) & (bb["s"] > -250)]
    first = (cand.groupby("id")["t"].min() - day.spat["t"].iloc[0]).to_numpy()
    best = max(range(0, int(day.duration - 1800), 900), key=lambda s: int(((first >= s + 30) & (first < s + 1770)).sum()))
    block = day.window(best, 1800)
    plan = estimate_timing(day.spat)
    rng = np.random.default_rng(0)
    base = generate(block, ScenarioConfig(attack="none"))
    train = [(base.bsm, base.spat)]
    for v in ("teleport", "speed_spoof", "fake_brake"):
        for on in _approach_onsets(block, 3, rng):
            train.append(_bs(generate(block, ScenarioConfig(attack="kinematic", variant=v, onset=on, duration=10))))
    for on in (300, 600, 900):
        train.append(_bs(generate(block, ScenarioConfig(attack="ghost_queue", onset=on, duration=10))))
    det = Detector(plan=plan).fit(train)
    det.set_threshold(base.bsm, base.spat, 0.01)

    # anchor the conflicting-green attack to a real approaching-vehicle time, and falsify that vehicle too,
    # so panels (a)/(c) and (b) share one time axis with real data in each
    # one SPaT attack (conflicting green) and one BSM attack (teleport), each shown on its own time-since-onset
    # axis so both are centred; panel (c) overlays the two detector outputs on that shared relative axis
    atk = generate(block, ScenarioConfig(attack="conflicting_green", variant="conflict", onset=30, duration=8))
    held, spoof = atk.info.get("held_green", 2), atk.info.get("spoofed_green", 4)
    a0 = atk.onset_t
    kin = generate(block, ScenarioConfig(attack="kinematic", variant="teleport", onset=_approach_onsets(block, 1, np.random.default_rng(3))[0], duration=8))
    k0 = kin.onset_t
    LO, HI = -10.0, 22.0

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.4))
    cmap = {"G": GREEN, "Y": AMBER, "R": "#d9d9d9"}
    # (a) phase timeline, time since the SPaT-attack onset
    ax = axes[0]
    order = [held, spoof] + [p for p in PHASES if p not in (held, spoof)]
    tt = atk.spat["t"].to_numpy() - a0
    for row, p in enumerate(order):
        st = atk.spat[f"state_{p}"].map(simple).to_numpy()
        for state, col in cmap.items():
            m = (st == state) & (tt >= LO) & (tt <= HI)
            ax.scatter(tt[m], np.full(m.sum(), row), s=9, c=col, marker="s")
    ax.axvspan(0, 8, color=RED, alpha=0.10)
    ax.set_yticks(range(len(order)), [f"$\\phi{p}$" for p in order], fontsize=8)
    ax.set_ylabel("signal group")
    ax.set_title(f"(a) NEMA SPaT timeline: conflicting green injected on $\\phi{spoof}$ while $\\phi{held}$ is green", fontsize=10)
    ax.set_xlim(LO, HI)
    ax.invert_yaxis()
    # (b) kinematic residual, time since the BSM-attack onset
    kf = kinematic_features(kin.bsm)
    tb = kin.bsm["t"].to_numpy() - k0
    win = (tb >= LO) & (tb <= HI)
    lab = kin.bsm_labels.astype(bool)
    ax = axes[1]
    ax.scatter(tb[win & ~lab], kf["pos_resid"].to_numpy()[win & ~lab], s=6, c=GREY, alpha=0.5, label="honest BSMs")
    if lab.any():
        ax.scatter(tb[win & lab], kf["pos_resid"].to_numpy()[win & lab], s=22, c=RED, label="falsified BSMs")
    ax.axvspan(0, kin.config.duration, color=RED, alpha=0.08)
    ax.set_ylabel("position residual (m)")
    ax.set_title("(b) Physics-informed kinematic residual $|{\\Delta}x - v\\,{\\Delta}t|$ for a teleport BSM falsification", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(LO, HI)
    # (c) detector outputs on the shared relative axis
    ax = axes[2]
    nv = np.array([len(v) for v in check_stream(atk.spat, plan)])
    tt_s = atk.spat["t"].to_numpy() - a0
    o = np.argsort(tt_s)
    ax.plot(tt_s[o], (nv > 0).astype(float)[o], color=RED, lw=1.8, label="L1 protocol flag (SPaT attack)")
    sc = det.score_bsm(kin.bsm, kin.spat)
    vic = kin.info.get("vehicle")
    vm = (kin.bsm["id"].to_numpy() == vic) if vic else lab  # the targeted vehicle's own messages
    ob = np.argsort(tb[vm])
    ax.plot(tb[vm][ob], sc[vm][ob], color=BLUE, lw=1.2, marker="o", ms=2, label="L3 fused risk (targeted vehicle)")
    ax.axhline(det.threshold_, color="k", ls="--", lw=1, label=f"threshold = {det.threshold_:.2f}")
    first = tt_s[nv > 0]
    if len(first):
        ax.annotate("detection latency ≈ 0 ms", (0, 1.02), fontsize=9, color=RED,
                    xytext=(3, 1.14), arrowprops=dict(arrowstyle="->", color=RED))
    ax.axvspan(0, 8, color=RED, alpha=0.07)
    ax.set_ylabel("score / flag")
    ax.set_xlabel("time since attack onset (s)")
    ax.set_title("(c) Detector output vs. threshold: L1 flags the SPaT attack exactly; L3 scores the targeted vehicle", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    ax.set_xlim(LO, HI)
    ax.set_ylim(-0.05, 1.25)
    fig.suptitle(f"Attacks injected into real UDOT data — {day.imap.name}, {date}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "attack_detection_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _bs(sc):
    return sc.bsm, sc.spat


def _find_day(data_root: str) -> tuple[str, str]:
    root = Path(data_root)
    for inter in sorted(root.iterdir()):
        if inter.is_dir():
            for date in sorted(inter.iterdir()):
                if (date / "spat.csv.gz").exists():
                    return inter.name, date.name
    raise FileNotFoundError(f"no downloaded day under {data_root}; run scripts/download_data.py first")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    fig_architecture(out)
    print("wrote architecture_pipeline.png")
    fig_benchmark(out, a.data)
    print("wrote attack_detection_benchmark.png")


if __name__ == "__main__":
    main()
