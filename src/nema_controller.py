"""NEMA TS2 eight-phase dual-ring model: conflict matrix, timing estimated from real SPaT, and a ring-and-barrier
state machine that checks a SPaT stream message by message.

Ring 1 serves phases 1-2 | 3-4 and ring 2 serves 5-6 | 7-8; the barrier separates the major-street side
(1, 2, 5, 6) from the minor-street side (3, 4, 7, 8). Two phases may show green or clearance together only if
they are in different rings on the same side of the barrier. At the Orem intersection used here the MAP assigns
2/6 to the State St throughs, 1/5 to the State St lefts, 4/8 to the Center St throughs and 3/7 to the Center St
lefts, so the standard matrix applies: 2 (NB through) conflicts with 1 (SB left) and every minor-street phase;
it is compatible with 5 (NB left, same direction) and 6 (SB through). ``observed_concurrency`` checks this
against real data.

``RingBarrierMonitor`` holds, per signal group, the current interval, when it began, and the advertised
TimeMarks, and reports which invariants each new message breaks. Which invariants are enforced, and with what
tolerance, is decided from clean real data (``CHECKS``; see ``scripts/validate_invariants.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .j2735_models import PHASES, SHOWING, simple, timemark_to_epoch

RINGS = ((1, 2, 3, 4), (5, 6, 7, 8))
BARRIER = ((1, 2, 5, 6), (3, 4, 7, 8))
MOVEMENTS = {1: "SB left", 2: "NB through", 3: "WB left", 4: "EB through", 5: "NB left", 6: "SB through", 7: "EB left", 8: "WB through"}


def ring_of(p: int) -> int:
    return 0 if p <= 4 else 1


def side_of(p: int) -> int:
    return 0 if p in BARRIER[0] else 1


def conflicts(p: int, q: int) -> bool:
    """True if phases p and q may never show green/clearance at the same time."""
    return p != q and (ring_of(p) == ring_of(q) or side_of(p) != side_of(q))


CONFLICT_PAIRS = tuple((p, q) for p in PHASES for q in PHASES if p < q and conflicts(p, q))


# --------------------------------------------------------------------------------------------- timing

@dataclass
class TimingPlan:
    """Per-phase interval durations estimated from clean SPaT (seconds)."""
    yellow: dict[int, float] = field(default_factory=dict)  # shortest observed yellow
    red_clear: dict[int, float] = field(default_factory=dict)  # shortest observed all-red after the yellow
    min_green: dict[int, float] = field(default_factory=dict)
    max_green: dict[int, float] = field(default_factory=dict)
    n_cycles: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: {str(p): v for p, v in getattr(self, k).items()} for k in ("yellow", "red_clear", "min_green", "max_green", "n_cycles")}

    @classmethod
    def from_dict(cls, d: dict) -> "TimingPlan":
        return cls(**{k: {int(p): v for p, v in d[k].items()} for k in ("yellow", "red_clear", "min_green", "max_green", "n_cycles")})


def intervals(spat: pd.DataFrame, p: int) -> pd.DataFrame:
    """Runs of the simplified state (G/Y/R) of one phase: state, start, end, duration, complete."""
    s = spat[f"state_{p}"].map(simple).to_numpy()
    t = spat["t"].to_numpy()
    change = np.r_[True, s[1:] != s[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(s)]
    out = pd.DataFrame({"state": s[starts], "start": t[starts], "end": np.r_[t[starts[1:]], t[-1]]})
    out["duration"] = out["end"] - out["start"]
    gap = np.r_[np.diff(t), 0]
    big = np.array([gap[a:b].max() > 1.0 if b > a else False for a, b in zip(starts, ends)])
    out["complete"] = ~big
    out.iloc[[0, -1], out.columns.get_loc("complete")] = False  # cut by the window
    return out


def estimate_timing(spat: pd.DataFrame) -> TimingPlan:
    plan = TimingPlan()
    for p in PHASES:
        iv = intervals(spat, p)
        c = iv[iv["complete"]]
        g, y = c[c["state"] == "G"]["duration"], c[c["state"] == "Y"]["duration"]
        if len(y) == 0:
            continue
        plan.yellow[p] = round(float(y.min()), 1)
        plan.min_green[p] = round(float(g.min()), 1) if len(g) else 0.0
        plan.max_green[p] = round(float(g.max()), 1) if len(g) else 0.0
        plan.n_cycles[p] = int(len(y))
        # red clearance: from this phase's yellow end to the next green of any conflicting phase
        ends = c[c["state"] == "Y"]["end"].to_numpy()
        rc = []
        for q in PHASES:
            if not conflicts(p, q):
                continue
            ivq = intervals(spat, q)
            gq = ivq[ivq["state"] == "G"]["start"].to_numpy()
            for e in ends:
                nxt = gq[gq >= e - 1e-6]
                if len(nxt) and nxt[0] - e < 10:
                    rc.append(nxt[0] - e)
        plan.red_clear[p] = round(float(min(rc)), 1) if rc else 0.0
    return plan


def observed_concurrency(spat: pd.DataFrame) -> pd.DataFrame:
    """Share of messages in which each pair of phases shows green/clearance together."""
    show = {p: spat[f"state_{p}"].isin(SHOWING).to_numpy() for p in PHASES}
    rows = [{"p": p, "q": q, "nema_conflict": conflicts(p, q), "together": float((show[p] & show[q]).mean())}
            for p in PHASES for q in PHASES if p < q]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- monitor

CHECKS = {  # name: description
    "conflict": "two conflicting phases show green/clearance at once (NEMA ring-and-barrier matrix)",
    "skip_yellow": "green followed directly by red",
    "bad_transition": "clearance followed by green, or red followed by clearance",
    "short_yellow": "clearance shorter than the shortest clean yellow for that phase",
    "short_red_clear": "a phase turns green before a conflicting phase has finished its red clearance",
    "min_after_max": "minEndTime later than maxEndTime",
    "end_before_min": "a state ends before the minEndTime it advertised",
    "overrun_max": "a state lasts past the maxEndTime it advertised",
    "rev_gap": "message counter does not advance with time",
}
# Invariants that are violated zero times on a clean real day at this intersection (see
# scripts/validate_invariants.py). The rest describe legitimate real-controller behaviour -- forcing a phase
# off without a yellow, overrunning an advertised minEndTime, a message counter that does not track time -- so
# turning them on would raise false alarms on honest data; they stay available but off by default.
DEFAULT_CHECKS = ("conflict", "bad_transition", "short_red_clear", "min_after_max", "overrun_max", "short_yellow")
NOISY_CHECKS = ("skip_yellow", "end_before_min", "rev_gap")
TOL = 0.25  # s, timing tolerance (SPaT is 10 Hz; TimeMarks have 0.1-s resolution)


class RingBarrierMonitor:
    """Stateful checker for a SPaT stream; ``step`` returns the list of broken invariants for one message."""

    def __init__(self, plan: TimingPlan, enabled: tuple[str, ...] = DEFAULT_CHECKS) -> None:
        self.plan, self.enabled = plan, set(enabled)
        self.prev_t: float | None = None
        self.prev_rev: int | None = None
        self.state = {p: None for p in PHASES}
        self.start = {p: None for p in PHASES}
        self.last_min = {p: np.nan for p in PHASES}
        self.last_max = {p: np.nan for p in PHASES}
        self.y_end = {p: -np.inf for p in PHASES}

    def step(self, t: float, rev: int, letters: dict[int, str], mn: dict[int, float], mx: dict[int, float]) -> list[str]:
        """``mn`` / ``mx``: advertised min/max end times already converted to epoch seconds (NaN = unknown)."""
        out: list[str] = []
        on = self.enabled
        L = {p: simple(letters[p]) for p in PHASES}
        fresh = self.prev_t is not None and t - self.prev_t <= 1.0
        if "conflict" in on:
            for p, q in CONFLICT_PAIRS:
                if letters[p] in SHOWING and letters[q] in SHOWING:
                    out.append(f"conflict {p}+{q}")
        if "rev_gap" in on and fresh and self.prev_rev is not None:
            steps = round((t - self.prev_t) * 10)
            if (rev - self.prev_rev) % 128 != steps % 128:
                out.append("rev_gap")
        for p in PHASES:
            old, new = self.state[p], L[p]
            if "min_after_max" in on and mn[p] == mn[p] and mx[p] == mx[p] and mn[p] > mx[p] + TOL:
                out.append(f"min_after_max {p}")
            if fresh and old is not None and old != new:
                if "skip_yellow" in on and old == "G" and new == "R":
                    out.append(f"skip_yellow {p}")
                if "bad_transition" in on and ((old == "Y" and new == "G") or (old == "R" and new == "Y")):
                    out.append(f"bad_transition {p}")
                if "end_before_min" in on and self.last_min[p] == self.last_min[p] and t < self.last_min[p] - TOL:
                    out.append(f"end_before_min {p}")
            if fresh and old == new and "overrun_max" in on and self.last_max[p] == self.last_max[p] and t > self.last_max[p] + TOL:
                out.append(f"overrun_max {p}")
            if fresh and old == "Y" and "short_yellow" in on and p in self.plan.yellow and self.start[p] is not None:
                if new != "Y" and t - self.start[p] < self.plan.yellow[p] - TOL:
                    out.append(f"short_yellow {p}")
            if fresh and new == "G" and old in ("R", "Y") and "short_red_clear" in on:
                for q in PHASES:
                    if conflicts(p, q) and q in self.plan.red_clear and t - self.y_end[q] < self.plan.red_clear[q] - TOL:
                        out.append(f"short_red_clear {p}<{q}")
            # update phase state
            if old == "Y" and new != "Y":
                self.y_end[p] = t
            if old != new or not fresh:
                self.start[p] = t if fresh else None
            self.state[p] = new
            self.last_min[p], self.last_max[p] = mn[p], mx[p]
        self.prev_t, self.prev_rev = t, rev
        return out


def check_stream(spat: pd.DataFrame, plan: TimingPlan, enabled: tuple[str, ...] = DEFAULT_CHECKS) -> list[list[str]]:
    """Run the monitor over a SPaT table (sorted by ``t``); one list of violations per message."""
    mon = RingBarrierMonitor(plan, enabled)
    cols_s = [f"state_{p}" for p in PHASES]
    cols_n = [f"min_{p}" for p in PHASES]
    cols_x = [f"max_{p}" for p in PHASES]
    T, R = spat["t"].to_numpy(dtype=float), spat["rev"].to_numpy()
    S = spat[cols_s].to_numpy()
    N = timemark_to_epoch(spat[cols_n].to_numpy(), T[:, None])
    X = timemark_to_epoch(spat[cols_x].to_numpy(), T[:, None])
    out = []
    for i in range(len(spat)):
        out.append(mon.step(float(T[i]), int(R[i]), dict(zip(PHASES, S[i])), dict(zip(PHASES, N[i])), dict(zip(PHASES, X[i]))))
    return out
