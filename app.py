"""Interactive replay-and-detection sandbox over real UDOT SPaT/BSM data (Streamlit).

Pick an attack, when it starts and the detector threshold; the app replays a real window with the attack
spliced in and shows the NEMA dual-ring state with conflict alerts, a time-space diagram of the approach, the
physics / anomaly telemetry, and a raw J2735 packet diff of a tampered message against the genuine one.

Run: ``streamlit run app.py``  (add ``--server.headless true`` on a server).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import RealData, load_day  # noqa: E402
from src.detector import Detector, kinematic_features  # noqa: E402
from src.j2735_models import PHASES, bsm_json, simple, spat_json  # noqa: E402
from src.nema_controller import BARRIER, MOVEMENTS, RINGS, check_stream, conflicts, estimate_timing  # noqa: E402
from src.scenario_generator import ATTACKS, LABELS, VARIANTS, ScenarioConfig, generate  # noqa: E402

st.set_page_config(page_title="V2X J2735 Cyber Sandbox", layout="wide")
GREEN, AMBER, REDC, GREY = "#2e7d32", "#f9a825", "#c62828", "#b0b0b0"
FILL = {"G": GREEN, "Y": AMBER, "R": "#d0d0d0"}


def _find_day(root: Path) -> tuple[str, str]:
    for inter in sorted(root.iterdir()) if root.exists() else []:
        if inter.is_dir():
            for date in sorted(inter.iterdir()):
                if (date / "spat.csv.gz").exists():
                    return inter.name, date.name
    return "", ""


@st.cache_resource(show_spinner="Loading real UDOT data…")
def get_day(root: str, inter: str, date: str) -> RealData:
    return load_day(root, inter, date)


@st.cache_resource(show_spinner="Calibrating timing & detector on real data…")
def get_detector(root: str, inter: str, date: str, start: int, dur: int) -> tuple[Detector, dict]:
    day = get_day(root, inter, date)
    block = day.window(start, dur)
    plan = estimate_timing(day.spat)
    base = generate(block, ScenarioConfig(attack="none"))
    train = [(base.bsm, base.spat)]
    for v in VARIANTS["kinematic"]:
        s = generate(block, ScenarioConfig(attack="kinematic", variant=v, onset=dur * 0.3, duration=10))
        train.append((s.bsm, s.spat))
    for on in (dur * 0.2, dur * 0.5, dur * 0.7):
        g = generate(block, ScenarioConfig(attack="ghost_queue", onset=on, duration=10))
        train.append((g.bsm, g.spat))
    det = Detector(plan=plan).fit(train)
    det.set_threshold(base.bsm, base.spat, 0.01)
    return det, {"base_threshold": det.threshold_}


def ring_diagram(letters: dict[int, str], violations: list[str]) -> go.Figure:
    fig = go.Figure()
    conflict_ph = {int(t) for item in violations if item.startswith("conflict") for t in item.replace("+", " ").split()[1:] if t.isdigit()}
    for ri, ring in enumerate(RINGS):
        for ci, p in enumerate(ring):
            x, y = ci, 1 - ri
            col = FILL.get(letters[p], "#d0d0d0")
            line = dict(color=REDC, width=4) if p in conflict_ph else dict(color="#333", width=1)
            fig.add_shape(type="rect", x0=x, x1=x + 0.9, y0=y, y1=y + 0.9, fillcolor=col, line=line)
            fig.add_annotation(x=x + 0.45, y=y + 0.45, text=f"<b>φ{p}</b><br>{MOVEMENTS[p]}", showarrow=False,
                               font=dict(size=11, color="white" if letters[p] in "GY" else "#333"))
    fig.add_annotation(x=1.9, y=1.5, text="Ring 1", showarrow=False, font=dict(size=10, color="#666"))
    fig.add_annotation(x=1.9, y=-0.15, text="Ring 2", showarrow=False, font=dict(size=10, color="#666"))
    fig.add_shape(type="line", x0=2.0, x1=2.0, y0=-0.2, y1=1.9, line=dict(color="#999", dash="dash"))
    fig.add_annotation(x=2.0, y=1.95, text="barrier", showarrow=False, font=dict(size=9, color="#999"))
    fig.update_xaxes(visible=False, range=[-0.2, 4.2])
    fig.update_yaxes(visible=False, range=[-0.3, 2.1])
    fig.update_layout(height=260, margin=dict(l=4, r=4, t=4, b=4), plot_bgcolor="white")
    return fig


def main() -> None:
    st.title("V2X / SAE J2735 SPaT & BSM Cybersecurity Sandbox")
    st.caption("Replay real connected-vehicle data (UDOT / Panasonic, Orem UT), inject attacks, and watch a "
               "three-layer physics-informed detector respond. No synthetic traffic — every honest message is real.")
    root = str(ROOT / "data" / "raw")
    inter, date = _find_day(Path(root))
    if not inter:
        st.error("No data found. Run `python scripts/download_data.py` first to fetch a real day of SPaT/BSM.")
        st.stop()
    day = get_day(root, inter, date)

    with st.sidebar:
        st.header("Scenario")
        atk = st.selectbox("Attack", ATTACKS, format_func=lambda a: LABELS[a])
        variant = ""
        if len(VARIANTS[atk]) > 1:
            variant = st.selectbox("Variant", VARIANTS[atk])
        win_start = st.slider("Replay window start (s into the day)", 0, int(day.duration) - 900, 25200, 300)
        onset = st.slider("Attack onset (s into window)", 20, 880, 120, 5)
        thr_adj = st.slider("Detector sensitivity (threshold offset)", -0.4, 0.4, 0.0, 0.02,
                            help="Lower = more sensitive (more alarms); higher = fewer alarms.")
        st.markdown(f"**Intersection {inter}** — {day.imap.name}  \n**Date** {date}")

    block = day.window(win_start, 900)
    det, meta = get_detector(root, inter, date, win_start, 900)
    det.threshold_ = float(np.clip(meta["base_threshold"] + thr_adj, 0.01, 0.99))
    scen = generate(block, ScenarioConfig(attack=atk, variant=variant, onset=onset, duration=10))
    t0 = scen.spat["t"].iloc[0]
    rel = onset if np.isnan(scen.onset_t) else scen.onset_t - t0
    # state at the moment just after the attack onset
    idx = int(np.searchsorted(scen.spat["t"].to_numpy(), t0 + rel + 1.0))
    idx = min(max(idx, 0), len(scen.spat) - 1)
    row = scen.spat.iloc[idx]
    letters = {p: simple(row[f"state_{p}"]) for p in PHASES}
    plan = det.plan
    viol_all = check_stream(scen.spat, plan)
    viol_now = viol_all[idx]

    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.subheader("A · NEMA TS2 dual-ring state")
        if any(v.startswith("conflict") for v in viol_now):
            st.error("⚠ CONFLICT / PROTOCOL VIOLATION — conflicting phases showing green/clearance together")
        elif viol_now:
            st.warning("Protocol anomaly: " + ", ".join(viol_now))
        else:
            st.success("Signal state consistent with NEMA ring-and-barrier rules")
        st.plotly_chart(ring_diagram(letters, viol_now), use_container_width=True)

    with c2:
        st.subheader("B · Approach time–space diagram")
        st.plotly_chart(time_space(scen, t0, rel), use_container_width=True)

    st.subheader("C · Physics-informed residual & anomaly telemetry")
    k1, k2, k3, k4 = st.columns(4)
    sc = det.score_bsm(scen.bsm, scen.spat)
    y = scen.bsm_labels
    nviol = np.array([len(v) for v in viol_all])
    if atk in ("conflicting_green", "timemark"):
        hit = (nviol > 0) & (scen.spat_labels == 1)
        lat = 0.0 if hit.any() else float("nan")
        detected = hit.any()
        fa = int((nviol[scen.spat_labels == 0] > 0).sum())
    else:
        hit = (sc > det.threshold_) & (y == 1)
        tt = scen.bsm["t"].to_numpy()
        lat = (tt[hit].min() - tt[y == 1].min()) * 1000 if hit.any() else float("nan")
        detected = hit.any()
        fa = int((sc[y == 0] > det.threshold_).sum())
    k1.metric("Attack detected", "Yes" if detected else "No")
    k2.metric("Detection latency", "—" if np.isnan(lat) else f"{max(lat,0):.0f} ms")
    recall = float((sc[y == 1] > det.threshold_).mean()) if (atk not in ('conflicting_green', 'timemark') and y.sum()) else (1.0 if detected else 0.0)
    k3.metric("Tampered-msg recall", f"{recall:.0%}" if y.sum() else "n/a")
    k4.metric("False alarms (window)", fa)
    st.plotly_chart(telemetry(scen, sc, nviol, t0, rel, det.threshold_), use_container_width=True)

    st.subheader("D · J2735 packet inspector — spoofed vs. genuine")
    packet_inspector(scen, atk)


def time_space(scen, t0, rel) -> go.Figure:
    fig = go.Figure()
    b = scen.bsm
    on = (b["lane"] >= 0) & (b["s"] < 5) & (b["s"] > -260)
    bb = b[on]
    tt = bb["t"].to_numpy() - t0
    fig.add_trace(go.Scattergl(x=tt, y=bb["s"].to_numpy(), mode="markers",
                               marker=dict(size=4, color=GREY), name="honest BSMs", hoverinfo="skip"))
    lab = bb["label"].to_numpy() == 1
    if lab.any():
        gh = bb["ghost"].to_numpy() if "ghost" in bb else np.zeros(len(bb), bool)
        fig.add_trace(go.Scattergl(x=tt[lab & ~gh], y=bb["s"].to_numpy()[lab & ~gh], mode="markers",
                                   marker=dict(size=7, color=REDC), name="falsified"))
        if gh.any():
            fig.add_trace(go.Scattergl(x=tt[gh], y=bb["s"].to_numpy()[gh], mode="markers",
                                       marker=dict(size=7, color="#6a1b9a", symbol="x"), name="ghost/Sybil"))
    fig.add_hline(y=0, line=dict(color="#333", width=1), annotation_text="stop bar")
    fig.add_vrect(x0=rel, x1=rel + 10, fillcolor=REDC, opacity=0.08, line_width=0)
    fig.update_layout(height=300, margin=dict(l=4, r=4, t=4, b=4), xaxis_title="time in window (s)",
                      yaxis_title="distance to stop bar (m)", legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(range=[max(0, rel - 20), rel + 40])
    return fig


def telemetry(scen, sc, nviol, t0, rel, thr) -> go.Figure:
    from plotly.subplots import make_subplots
    kf = kinematic_features(scen.bsm)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Kinematic position residual (m)", "NEMA/J2735 protocol violations (count)", "Fused anomaly risk vs. threshold"))
    tb = scen.bsm["t"].to_numpy() - t0
    lab = scen.bsm_labels.astype(bool)
    fig.add_trace(go.Scattergl(x=tb[~lab], y=kf["pos_resid"].to_numpy()[~lab], mode="markers", marker=dict(size=3, color=GREY), name="honest"), 1, 1)
    if lab.any():
        fig.add_trace(go.Scattergl(x=tb[lab], y=kf["pos_resid"].to_numpy()[lab], mode="markers", marker=dict(size=6, color=REDC), name="falsified"), 1, 1)
    ts = scen.spat["t"].to_numpy() - t0
    fig.add_trace(go.Scatter(x=ts, y=nviol, mode="lines", line=dict(color=REDC), name="violations"), 2, 1)
    o = np.argsort(tb)
    fig.add_trace(go.Scattergl(x=tb[o], y=sc[o], mode="markers", marker=dict(size=3, color="#1565c0"), name="risk"), 3, 1)
    fig.add_hline(y=thr, line=dict(color="black", dash="dash"), row=3, col=1)
    for r in (1, 2, 3):
        fig.add_vrect(x0=rel, x1=rel + 10, fillcolor=REDC, opacity=0.07, line_width=0, row=r, col=1)
    fig.update_layout(height=520, margin=dict(l=4, r=4, t=28, b=4), showlegend=False)
    fig.update_xaxes(range=[max(0, rel - 15), rel + 35], title="time in window (s)", row=3, col=1)
    return fig


def packet_inspector(scen, atk) -> None:
    if atk in ("conflicting_green", "timemark"):
        lab = scen.spat["label"].to_numpy() == 1
        if not lab.any():
            st.info("No tampered SPaT message in this window — try a different onset.")
            return
        row = scen.spat[lab].iloc[len(scen.spat[lab]) // 2]
        genuine = {f"state_{p}": row[f"auth_state_{p}"] for p in PHASES}
        genuine |= {f"minEndTime_{p}": int(row[f"auth_min_{p}"]) for p in PHASES}
        spoofed = {f"state_{p}": row[f"state_{p}"] for p in PHASES}
        spoofed |= {f"minEndTime_{p}": int(row[f"min_{p}"]) for p in PHASES}
        diff = {k: (genuine[k], spoofed[k]) for k in genuine if genuine[k] != spoofed[k]}
        a, b = st.columns(2)
        a.markdown("**Genuine controller SPaT**")
        a.json(spat_json({**row, **{f"state_{p}": row[f"auth_state_{p}"] for p in PHASES},
                          **{f"min_{p}": row[f"auth_min_{p}"] for p in PHASES},
                          **{f"max_{p}": row[f"auth_max_{p}"] for p in PHASES}}, int(scen.data.intersection)))
        b.markdown("**Spoofed RSU broadcast**")
        b.json(spat_json(row, int(scen.data.intersection)))
        st.markdown("**Changed fields:** " + (", ".join(f"`{k}` {v[0]}→{v[1]}" for k, v in diff.items()) or "none"))
    else:
        lab = scen.bsm["label"].to_numpy() == 1
        if not lab.any():
            st.info("No tampered BSM in this window — try a different onset.")
            return
        rows = scen.bsm[lab]
        row = rows.iloc[len(rows) // 2]
        a, b = st.columns(2)
        a.markdown("**Genuine / expected BSM**")
        gen = row.copy()
        for c, tc in (("x", "true_x"), ("y", "true_y"), ("speed", "true_speed"), ("a_long", "true_a_long")):
            if tc in row:
                gen[c] = row[tc]
        a.json(_bsm_view(gen))
        b.markdown("**Spoofed / injected BSM**")
        b.json(_bsm_view(row))
        if bool(row.get("ghost", False)):
            st.markdown("**This is a ghost/Sybil message** — a vehicle that does not exist, forged at the stop bar.")


def _bsm_view(row) -> dict:
    return {"id": str(row["id"]), "x_m": round(float(row["x"]), 2), "y_m": round(float(row["y"]), 2),
            "speed_mps": round(float(row["speed"]), 2), "heading_deg": round(float(row["heading"]), 1),
            "a_long_mps2": None if pd.isna(row["a_long"]) else round(float(row["a_long"]), 2),
            "distance_to_stopbar_m": None if pd.isna(row["s"]) else round(float(row["s"]), 1)}


if __name__ == "__main__":
    main()
