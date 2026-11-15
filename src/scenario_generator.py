"""Inject cyberattacks into real SPaT and BSM streams, with ground-truth labels.

Each scenario takes a window of real UDOT data (``data_loader.RealData``) and returns a ``Scenario`` whose
``spat`` / ``bsm`` tables are the real messages with an attack spliced in and a per-message ``label`` (1 =
tampered) and ``attack`` column. Authentic columns are preserved (``auth_state_p`` for SPaT, ``true_*`` for
BSM) so the effect is auditable and the J2735 packet inspector can diff spoofed against genuine.

Attacks:

* ``conflicting_green`` -- a compromised RSU rebroadcasts SPaT with a phase conflicting with the current green
  set to ``protected-Movement-Allowed`` (``conflict``) or, in the ``clearance`` variant, cuts a real yellow to
  red early so a following phase is released into the clearance interval;
* ``timemark`` -- during a real green approaching its end, the advertised ``minEndTime`` is pushed several
  seconds past the real switch (and past ``maxEndTime``), the green-extension lure that keeps a CAV accelerating
  toward a red;
* ``ghost_queue`` -- forged stationary BSMs (Sybil IDs) are injected at a real approach's stop bar to fake a
  standing queue / actuation demand;
* ``kinematic`` -- a real moving vehicle's own BSMs are falsified: ``speed_spoof`` (+``mag`` m/s), ``fake_brake``
  (reported hard braking with no matching motion) or ``teleport`` (position jumped ``mag`` m downstream).

The attack starts at the first eligible real event at or after ``onset`` seconds into the window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_loader import RealData, add_signal
from .j2735_models import GO, PHASES, SHOWING, TIMEMARK_UNKNOWN, simple, timemark_of, timemark_to_epoch
from .nema_controller import conflicts, intervals

ATTACKS = ("none", "conflicting_green", "timemark", "ghost_queue", "kinematic")
VARIANTS = {"none": ("",), "conflicting_green": ("conflict", "clearance"), "timemark": ("",), "ghost_queue": ("",),
            "kinematic": ("speed_spoof", "fake_brake", "teleport")}
LABELS = {"none": "Normal baseline (real data)", "conflicting_green": "Conflicting-green SPaT spoofing",
          "timemark": "TimeMark / green-extension spoofing", "ghost_queue": "Ghost queue / Sybil BSM injection",
          "kinematic": "Kinematic BSM falsification"}


@dataclass
class ScenarioConfig:
    attack: str = "none"
    variant: str = ""
    onset: float = 60.0
    duration: float = 8.0  # attack duration (SPaT attacks) / falsification duration (kinematic), s
    magnitude: float = 25.0  # teleport metres / speed_spoof delta via _KIN_MAG
    seed: int = 0

    def __post_init__(self) -> None:
        if self.attack not in ATTACKS:
            raise ValueError(f"unknown attack {self.attack}")
        if not self.variant:
            self.variant = VARIANTS[self.attack][0]
        if self.variant not in VARIANTS[self.attack]:
            raise ValueError(f"unknown variant {self.variant!r} for {self.attack}")

    @property
    def name(self) -> str:
        return self.attack if not self.variant else f"{self.attack}:{self.variant}"


@dataclass
class Scenario:
    config: ScenarioConfig
    data: RealData
    spat: pd.DataFrame
    bsm: pd.DataFrame
    onset_t: float = float("nan")
    info: dict = field(default_factory=dict)

    @property
    def spat_labels(self) -> np.ndarray:
        return self.spat["label"].to_numpy()

    @property
    def bsm_labels(self) -> np.ndarray:
        return self.bsm["label"].to_numpy()


_KIN_MAG = {"speed_spoof": 10.0, "fake_brake": 7.0, "teleport": 25.0}


def _prep_spat(real: RealData) -> pd.DataFrame:
    sp = real.spat.copy()
    for p in PHASES:
        sp[f"auth_state_{p}"] = sp[f"state_{p}"]
        sp[f"auth_min_{p}"] = sp[f"min_{p}"]
        sp[f"auth_max_{p}"] = sp[f"max_{p}"]
    sp["label"] = 0
    sp["attack"] = "none"
    return sp


def _prep_bsm(real: RealData) -> pd.DataFrame:
    b = real.bsm.copy()
    for c in ("x", "y", "speed", "a_long"):
        b[f"true_{c}"] = b[c]
    b["label"] = 0
    b["attack"] = "none"
    b["ghost"] = False
    return b


def _mark_spat(sp: pd.DataFrame, mask: np.ndarray, name: str) -> None:
    diff = np.zeros(len(sp), dtype=bool)
    for p in PHASES:
        diff |= (sp[f"state_{p}"] != sp[f"auth_state_{p}"]) | (sp[f"min_{p}"] != sp[f"auth_min_{p}"]) | (sp[f"max_{p}"] != sp[f"auth_max_{p}"])
    sp.loc[diff, "label"] = 1
    sp.loc[diff, "attack"] = name


def generate(real: RealData, cfg: ScenarioConfig) -> Scenario:
    real = add_signal(real)
    sp = _prep_spat(real)
    bsm = _prep_bsm(real)
    t0 = sp["t"].iloc[0]
    onset_abs = t0 + cfg.onset
    rng = np.random.default_rng([cfg.seed, hash(cfg.name) % (2**31)])
    onset_t = float("nan")
    info: dict = {}

    if cfg.attack == "conflicting_green":
        onset_t = _attack_conflicting_green(sp, real, cfg, onset_abs, info)
    elif cfg.attack == "timemark":
        onset_t = _attack_timemark(sp, real, cfg, onset_abs, info)
    elif cfg.attack == "ghost_queue":
        bsm, onset_t = _attack_ghost_queue(bsm, real, cfg, onset_abs, rng, info)
    elif cfg.attack == "kinematic":
        onset_t = _attack_kinematic(bsm, real, cfg, onset_abs, rng, info)

    _mark_spat(sp, None, cfg.name)
    return Scenario(cfg, real, sp, bsm, onset_t, info)


# ---------------------------------------------------------------------------------- SPaT attacks

def _attack_conflicting_green(sp: pd.DataFrame, real: RealData, cfg: ScenarioConfig, onset_abs: float, info: dict) -> float:
    t = sp["t"].to_numpy()
    if cfg.variant == "conflict":
        # find a green phase active at/after onset, then set a conflicting phase to green for the window
        for p in (2, 6, 4, 8):
            green = (sp[f"state_{p}"].map(simple) == "G").to_numpy() & (t >= onset_abs)
            if green.any():
                ts = t[green][0]
                victim = next(q for q in (4, 8, 3, 7, 1, 5) if conflicts(p, q))
                win = (t >= ts) & (t < ts + cfg.duration)
                sp.loc[win, f"state_{victim}"] = "G"
                sp.loc[win, f"min_{victim}"] = timemark_of(ts + cfg.duration)
                sp.loc[win, f"max_{victim}"] = timemark_of(ts + cfg.duration)
                info.update(held_green=p, spoofed_green=victim, n_messages=int(win.sum()))
                return float(ts)
    else:  # clearance: truncate a real clearance and release a conflicting phase to green before red-clear ends
        for p in (2, 6, 4, 8):
            iv = intervals(sp.assign(t=t), p)
            yl = iv[(iv.state == "Y") & iv.complete & (iv.start >= onset_abs)]
            if len(yl):
                y = yl.iloc[0]
                victim = next(q for q in (4, 8, 1, 5, 3, 7) if conflicts(p, q))
                cut = y.start + 1.0  # cut the clearance to 1 s
                win = (t >= cut) & (t < y.end + 2.0)
                sp.loc[win, f"state_{victim}"] = "G"  # conflicting phase released into the (real) clearance
                sp.loc[win, f"min_{victim}"] = timemark_of(y.end + 4.0)
                sp.loc[win, f"max_{victim}"] = timemark_of(y.end + 4.0)
                info.update(clearing_phase=int(p), spoofed_green=int(victim), real_yellow_s=round(float(y.duration), 1), n_messages=int(win.sum()))
                return float(cut)
    return float("nan")


def _attack_timemark(sp: pd.DataFrame, real: RealData, cfg: ScenarioConfig, onset_abs: float, info: dict) -> float:
    t = sp["t"].to_numpy()
    for p in (2, 6):
        iv = intervals(sp.assign(t=t), p)
        gr = iv[(iv.state == "G") & iv.complete & (iv.start >= onset_abs - 20)]
        gr = gr[gr.end >= onset_abs]
        if len(gr):
            g = gr.iloc[0]
            # over the last `duration` s of the real green and into the change, advertise min/maxEndTime pushed out
            win = (t >= g.end - cfg.duration) & (t < g.end + 4.0)
            push = cfg.duration + 4.0
            sp.loc[win, f"min_{p}"] = timemark_of(t[win] + push)
            sp.loc[win, f"max_{p}"] = timemark_of(t[win] + push - 2.0)  # min later than max -> invariant break
            # keep showing green through the real change
            after = (t >= g.end) & (t < g.end + 4.0)
            sp.loc[after, f"state_{p}"] = "G"
            info.update(phase=int(p), real_switch_t=round(float(g.end - sp['t'].iloc[0]), 1), n_messages=int(win.sum()))
            return float(g.end - cfg.duration)
    return float("nan")


# ---------------------------------------------------------------------------------- BSM attacks

def _attack_ghost_queue(bsm: pd.DataFrame, real: RealData, cfg: ScenarioConfig, onset_abs: float, rng, info: dict) -> tuple[pd.DataFrame, float]:
    # pick a real approach (prefer a minor one) and stack stationary ghosts behind its stop bar
    lane = next((ln for ln in real.imap.ingress if ln.phase in (4, 8, 3, 7)), real.imap.ingress[0])
    stop = lane.xy[0]
    tv = lane.travel
    tgrid = np.arange(onset_abs, onset_abs + 60.0, 0.1)
    ghosts = []
    for j in range(4):
        tid = f"GHOST{j:02d}"
        back = 2.0 + 7.5 * j
        gx, gy = stop[0] - back * tv[0], stop[1] - back * tv[1]
        lat_d, lon_d = real.imap.to_latlon(gx, gy)
        head = np.degrees(np.arctan2(tv[0], tv[1])) % 360
        for k, tt in enumerate(tgrid):
            ex, ey = rng.normal(0, 1.0, 2)
            lad, lod = real.imap.to_latlon(gx + ex, gy + ey)
            ghosts.append({"t": tt, "id": tid, "msg_cnt": k % 128, "sec_mark": int((tt % 60) * 1000),
                           "x": gx + ex, "y": gy + ey, "speed": abs(rng.normal(0, 0.1)), "heading": head + rng.normal(0, 1),
                           "a_long": np.nan, "length": rng.uniform(4.3, 5.0), "width": rng.uniform(1.7, 1.95),
                           "lane": lane_index(real, lane), "s": -back, "lane_lat": abs(ey), "approach": lane.approach,
                           "phase": lane.phase, "maneuver": lane.maneuver, "signal": "",
                           "true_x": gx, "true_y": gy, "true_speed": 0.0, "true_a_long": np.nan,
                           "label": 1, "attack": cfg.name, "ghost": True})
    gdf = pd.DataFrame(ghosts)
    out = pd.concat([bsm, gdf], ignore_index=True).sort_values(["t", "id"]).reset_index(drop=True)
    info.update(approach_phase=int(lane.phase), n_ghosts=4, n_messages=len(gdf))
    return out, float(onset_abs)


def lane_index(real: RealData, lane) -> int:
    return real.imap.ingress.index(lane)


def _attack_kinematic(bsm: pd.DataFrame, real: RealData, cfg: ScenarioConfig, onset_abs: float, rng, info: dict) -> float:
    mag = cfg.magnitude if cfg.attack == "kinematic" and cfg.magnitude != 25.0 else _KIN_MAG[cfg.variant]
    # pick a real moving vehicle present at onset, on an ingress lane, approaching
    b = bsm
    cand = b[(b["t"] >= onset_abs) & (b["lane"] >= 0) & (b["speed"] > 3) & (b["s"] < -3) & (b["s"] > -250)].sort_values("t")
    if cand.empty:
        return float("nan")
    first = cand.iloc[0]
    vid, t_start = first["id"], float(first["t"])  # anchor on the vehicle's actual arrival, not the nominal onset
    rows = b[(b["id"] == vid) & (b["t"] >= t_start) & (b["t"] < t_start + cfg.duration)].index
    if len(rows) == 0:
        return float("nan")
    if cfg.variant == "speed_spoof":
        b.loc[rows, "speed"] = b.loc[rows, "true_speed"] + mag
    elif cfg.variant == "fake_brake":
        b.loc[rows, "a_long"] = -mag
    elif cfg.variant == "teleport":
        head = np.radians(b.loc[rows, "heading"].to_numpy())
        b.loc[rows, "x"] = b.loc[rows, "true_x"].to_numpy() + mag * np.sin(head)
        b.loc[rows, "y"] = b.loc[rows, "true_y"].to_numpy() + mag * np.cos(head)
    b.loc[rows, "label"] = 1
    b.loc[rows, "attack"] = cfg.name
    info.update(vehicle=str(vid), n_messages=int(len(rows)), magnitude=float(mag))
    return float(b.loc[rows, "t"].min())
