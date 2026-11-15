"""Load the downloaded real SPaT, BSM and MAP data into tidy tables keyed by a single time column.

``load_day`` returns a ``RealData`` with:

* ``imap``: the parsed ``IntersectionMap``;
* ``spat``: one row per SPaT message, ``t`` (epoch seconds), ``rev``, ``moy``, ``ms`` and ``state_p`` /
  ``min_p`` / ``max_p`` for phases 1-8, sorted by ``t`` and de-duplicated;
* ``bsm``: decoded SI BSMs with ``t``, ``id``, ``x``, ``y``, ``speed``, ``heading``, ``a_long`` and the matched
  ``lane`` / ``s`` (distance to the stop bar) / ``approach`` / ``phase`` for messages on an ingress lane.

``window`` slices a time range; ``add_signal_at`` labels each BSM with the signal state of its own phase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .j2735_models import PHASES, IntersectionMap, decode_bsm, simple

SPAT_STATE = [f"state_{p}" for p in PHASES]
SPAT_MIN = [f"min_{p}" for p in PHASES]
SPAT_MAX = [f"max_{p}" for p in PHASES]


@dataclass
class RealData:
    intersection: str
    date: str
    imap: IntersectionMap
    spat: pd.DataFrame
    bsm: pd.DataFrame

    def window(self, start_s: float, dur_s: float) -> "RealData":
        t0 = self.spat["t"].iloc[0] + start_s
        sp = self.spat[(self.spat["t"] >= t0) & (self.spat["t"] < t0 + dur_s)].reset_index(drop=True)
        bs = self.bsm[(self.bsm["t"] >= t0) & (self.bsm["t"] < t0 + dur_s)].reset_index(drop=True)
        return RealData(self.intersection, self.date, self.imap, sp, bs)

    @property
    def duration(self) -> float:
        return float(self.spat["t"].iloc[-1] - self.spat["t"].iloc[0])


def _epoch(ts: pd.Series) -> np.ndarray:
    return pd.to_datetime(ts, utc=True, format="mixed").astype("int64").to_numpy() / 1e9


def load_map(path: Path) -> IntersectionMap:
    return IntersectionMap.from_record(json.loads(path.read_text(encoding="utf-8")))


def load_day(root: str | Path, intersection: str, date: str) -> RealData:
    d = Path(root) / intersection / date
    imap = load_map(d / "map.json")
    spat = pd.read_csv(d / "spat.csv.gz")
    spat["t"] = _epoch(spat["tmstp_utc"])
    spat = spat.sort_values("t").drop_duplicates("t").reset_index(drop=True)
    for c in SPAT_STATE:
        spat[c] = spat[c].fillna("U").astype(str)
    for c in SPAT_MIN + SPAT_MAX:
        spat[c] = spat[c].fillna(36000).astype(np.int64)

    raw = pd.read_csv(d / "bsm.csv.gz")
    raw["t"] = _epoch(raw["tmstp_utc"])
    raw = raw.sort_values(["t", "temp_id"]).reset_index(drop=True)
    bsm = decode_bsm(raw, imap)
    li, s, lat = imap.map_match(bsm["x"].to_numpy(), bsm["y"].to_numpy(), bsm["heading"].to_numpy())
    bsm["lane"] = li
    bsm["s"] = s
    bsm["lane_lat"] = lat
    bsm["approach"] = [imap.ingress[i].approach if i >= 0 else -1 for i in li]
    bsm["phase"] = [imap.ingress[i].phase if i >= 0 else -1 for i in li]
    bsm["maneuver"] = [imap.ingress[i].maneuver if i >= 0 else "" for i in li]
    return RealData(intersection, date, imap, spat, bsm)


def signal_at(spat: pd.DataFrame, t: np.ndarray, phase_col: np.ndarray) -> np.ndarray:
    """Simplified signal state (G/Y/R) of each (t, phase) by taking the most recent SPaT message at or before t."""
    ts = spat["t"].to_numpy()
    idx = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
    states = {p: spat[f"state_{p}"].map(simple).to_numpy() for p in PHASES}
    out = np.full(len(t), "R", dtype="<U1")
    for p in PHASES:
        m = phase_col == p
        if m.any():
            out[m] = states[p][idx[m]]
    return out


def add_signal(data: RealData) -> RealData:
    b = data.bsm
    sig = np.full(len(b), "", dtype="<U1")
    on = b["phase"].to_numpy() > 0
    sig[on] = signal_at(data.spat, b["t"].to_numpy()[on], b["phase"].to_numpy()[on])
    b = b.copy()
    b["signal"] = sig
    return RealData(data.intersection, data.date, data.imap, data.spat, b)
