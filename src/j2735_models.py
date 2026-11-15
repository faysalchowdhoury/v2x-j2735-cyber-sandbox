"""SAE J2735 message models for real SPaT, BSM and MAP data.

* SPaT (MessageFrame 19): one row per message with ``state_p`` / ``min_p`` / ``max_p`` for signal groups 1-8
  (``minEndTime`` / ``maxEndTime`` as TimeMark). ``spat_json`` rebuilds the JER object for display.
* BSM Part I (MessageFrame 20): ``BSMcoreData`` in J2735 integer units (lat/long 1e-7 deg, speed 0.02 m/s,
  heading 0.0125 deg, accelerations 0.01 m/s^2 with 2001 = unavailable). ``decode_bsm`` converts to SI.
* MAP (MessageFrame 18): ``IntersectionMap.from_record`` parses the lane set (node offsets in cm) into
  polylines in a local east/north frame centred on the intersection reference point; ``map_match`` places a
  position on an ingress lane with its distance to the stop bar.

Event states are kept as one-letter codes: G protected green, g permissive green, Y protected clearance,
y permissive clearance, R stop-and-remain, F stop-then-proceed (flash), D dark, P pre-movement, U unavailable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TIMEMARK_UNKNOWN = 36000
PHASES = tuple(range(1, 9))
EVENT_OF = {"G": "protected-Movement-Allowed", "g": "permissive-Movement-Allowed", "Y": "protected-clearance", "y": "permissive-clearance",
            "R": "stop-And-Remain", "F": "stop-Then-Proceed", "P": "pre-Movement", "C": "caution-Conflicting-Traffic", "D": "dark", "U": "unavailable"}
GO = frozenset("Gg")  # movement allowed
CLEAR = frozenset("Yy")  # change interval
SHOWING = GO | CLEAR  # states that must never overlap for conflicting groups
ACCEL_UNAVAILABLE = 2001


def simple(letter: str) -> str:
    """Collapse to G / Y / R (anything that is not go or clearance counts as red)."""
    return "G" if letter in GO else "Y" if letter in CLEAR else "R"


# --------------------------------------------------------------------------------------------- time

def timemark_of(t_epoch: np.ndarray | float) -> np.ndarray:
    """TimeMark (tenths of a second within the UTC hour) of an epoch time in seconds."""
    return np.floor((np.asarray(t_epoch, dtype=float) % 3600) * 10 + 1e-6).astype(np.int64)


def timemark_to_epoch(tm: np.ndarray, t_epoch: np.ndarray) -> np.ndarray:
    """Epoch seconds of a TimeMark, resolved to the hour nearest the message time; NaN when unknown (36000)."""
    tm = np.asarray(tm, dtype=np.int64)
    t_epoch = np.asarray(t_epoch, dtype=float)
    now = (t_epoch % 3600) * 10
    d = (tm - now + 18000) % 36000 - 18000
    return np.where(tm >= TIMEMARK_UNKNOWN, np.nan, t_epoch + d / 10.0)


def spat_json(row: pd.Series | dict, intersection_id: int) -> dict:
    """One SPaT row as a J2735 MessageFrame (JER)."""
    states = []
    for p in PHASES:
        timing = {"minEndTime": int(row[f"min_{p}"])}
        if int(row[f"max_{p}"]) != TIMEMARK_UNKNOWN:
            timing["maxEndTime"] = int(row[f"max_{p}"])
        states.append({"signalGroup": p, "state-time-speed": [{"eventState": EVENT_OF.get(row[f"state_{p}"], "unavailable"), "timing": timing}]})
    return {"messageId": 19, "value": {"intersections": [{"id": {"id": int(intersection_id)}, "revision": int(row["rev"]),
                                                          "moy": int(row["moy"]), "timeStamp": int(row["ms"]), "states": states}]}}


# --------------------------------------------------------------------------------------------- MAP

@dataclass(frozen=True)
class Lane:
    lane_id: int
    approach: int
    groups: tuple[int, ...]  # signal groups controlling this lane's connections
    maneuver: str  # "through" | "left" | "right"
    xy: np.ndarray  # polyline, first node at the stop bar, then upstream (local metres)

    @property
    def phase(self) -> int:
        return self.groups[0]

    @property
    def travel(self) -> np.ndarray:
        """Unit direction of travel at the stop bar (towards the intersection)."""
        v = self.xy[0] - self.xy[1]
        return v / np.linalg.norm(v)


_MANEUVER = {0: "through", 1: "left", 2: "right"}


class IntersectionMap:
    def __init__(self, intersection_id: int, name: str, lat: float, lon: float, lanes: list[Lane], lane_width: float) -> None:
        self.intersection_id, self.name, self.lat, self.lon = intersection_id, name, lat, lon
        self.lanes, self.lane_width = lanes, lane_width
        self.ingress = [ln for ln in lanes if ln.groups]
        self._kx = math.radians(1) * 6378137.0 * math.cos(math.radians(lat))
        self._ky = math.radians(1) * 6378137.0

    @classmethod
    def from_record(cls, rec: dict) -> "IntersectionMap":
        lanes = []
        for ln in json.loads(rec["lane_set"]) if isinstance(rec["lane_set"], str) else rec["lane_set"]:
            pts, x, y = [], 0.0, 0.0
            for node in ln["nodeList"]["nodes"]:
                d = next(iter(node["delta"].values()))
                x, y = x + d["x"] / 100.0, y + d["y"] / 100.0
                pts.append((x, y))
            groups = tuple(sorted({c["signalGroup"] for c in ln.get("connectsTo", []) if "signalGroup" in c}))
            bits = ln.get("maneuvers") or "000"
            man = next((_MANEUVER[i] for i in range(3) if bits[i] == "1"), "through")
            lanes.append(Lane(int(ln["laneID"]), int(ln.get("ingressApproach") or ln.get("egressApproach") or 0), groups, man, np.array(pts)))
        return cls(int(rec["intersection_id"]), rec.get("intersection_name", ""), float(rec["latitude"]), float(rec["longitude"]),
                   lanes, float(rec.get("lane_width", 366)) / 100.0)

    # ------------------------------------------------------------------ geodesy
    def to_xy(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(lon_deg) - self.lon) * self._kx, (np.asarray(lat_deg) - self.lat) * self._ky

    def to_latlon(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.lat + np.asarray(y) / self._ky, self.lon + np.asarray(x) / self._kx

    # ------------------------------------------------------------------ map matching
    def map_match(self, x: np.ndarray, y: np.ndarray, heading_deg: np.ndarray | None = None,
                  max_lateral: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nearest ingress lane (index into ``self.ingress``, -1 if none), distance upstream of the stop bar
        (negative = before the bar, m) and lateral offset. With ``heading_deg`` a lane only matches if the
        vehicle travels within 60 degrees of the lane direction."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        max_lat = self.lane_width * 1.5 if max_lateral is None else max_lateral
        best = np.full(x.shape, np.inf)
        idx = np.full(x.shape, -1, dtype=np.int64)
        s_out = np.full(x.shape, np.nan)
        if heading_deg is not None:
            h = np.radians(np.asarray(heading_deg, dtype=float))
            hv = np.stack([np.sin(h), np.cos(h)], -1)
        for li, ln in enumerate(self.ingress):
            a, b = ln.xy[:-1], ln.xy[1:]  # segments from the stop bar upstream
            seg = b - a
            L = np.linalg.norm(seg, axis=1)
            cum = np.r_[0.0, np.cumsum(L)]
            p = np.stack([x, y], -1)[:, None, :]
            tt = np.clip(((p - a) * seg).sum(-1) / (L**2), 0, 1)
            proj = a + tt[..., None] * seg
            dist = np.linalg.norm(p - proj, axis=-1)
            k = dist.argmin(1)
            lat = dist[np.arange(len(x)), k]
            along = cum[k] + tt[np.arange(len(x)), k] * L[k]
            # beyond the stop bar (inside the intersection) along < 0 is allowed for a short distance
            ahead = ((p[:, 0, :] - ln.xy[0]) * ln.travel).sum(-1)
            along = np.where((k == 0) & (tt[np.arange(len(x)), 0] <= 0) & (ahead > 0), -ahead, along)
            lat = np.where(along < 0, np.abs(((p[:, 0, :] - ln.xy[0]) * np.array([-ln.travel[1], ln.travel[0]])).sum(-1)), lat)
            ok = (lat < max_lat) & (lat < best) & (along > -25)
            if heading_deg is not None:
                ok &= (hv @ ln.travel) > 0.5
            best = np.where(ok, lat, best)
            idx = np.where(ok, li, idx)
            s_out = np.where(ok, -along, s_out)
        return idx, s_out, best


# --------------------------------------------------------------------------------------------- BSM

def encode_bsm_wire(imap: IntersectionMap, t, temp_id, x, y, speed, heading, a_long, length=4.5, width=1.8) -> pd.DataFrame:
    """SI values -> a BSM table in UDOT export (wire) columns; the inverse of ``decode_bsm``. For tests and
    for the packet inspector. Acceleration ``nan`` encodes as the J2735 'unavailable' code (2001)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    lat, lon = imap.to_latlon(x, y)
    a = np.asarray(a_long, dtype=float)
    n = len(np.atleast_1d(t))
    return pd.DataFrame({
        "t": np.asarray(t, dtype=float), "temp_id": np.asarray(temp_id), "msg_count": np.arange(n) % 128,
        "sec_mark": (np.asarray(t, dtype=float) % 60 * 1000).astype(np.int64),
        "latitude": np.round(lat * 1e7).astype(np.int64), "longitude": np.round(lon * 1e7).astype(np.int64),
        "elevation": np.full(n, 1400, dtype=np.int64),
        "speed": np.clip(np.round(np.asarray(speed) / 0.02), 0, 8191).astype(np.int64),
        "heading": (np.round((np.asarray(heading) % 360) / 0.0125).astype(np.int64)) % 28800,
        "longitudinal_acceleration": np.where(np.isfinite(a), np.round(a / 0.01), ACCEL_UNAVAILABLE).astype(np.int64),
        "lateral_acceleration": np.full(n, ACCEL_UNAVAILABLE, dtype=np.int64), "yaw_rate": np.zeros(n, dtype=np.int64),
        "vehicle_length": np.round(np.full(n, length) * 100).astype(np.int64), "vehicle_width": np.round(np.full(n, width) * 100).astype(np.int64),
    })


def decode_bsm(df: pd.DataFrame, imap: IntersectionMap) -> pd.DataFrame:
    """Wire columns (UDOT export names) -> SI columns; x/y are local metres around the intersection."""
    x, y = imap.to_xy(df["latitude"].to_numpy() / 1e7, df["longitude"].to_numpy() / 1e7)
    a_long = df["longitudinal_acceleration"].to_numpy()
    out = pd.DataFrame({
        "t": df["t"].to_numpy(), "id": df["temp_id"].to_numpy(), "msg_cnt": df["msg_count"].to_numpy(), "sec_mark": df["sec_mark"].to_numpy(),
        "x": x, "y": y, "speed": df["speed"].to_numpy() * 0.02, "heading": df["heading"].to_numpy() * 0.0125,
        "a_long": np.where(a_long == ACCEL_UNAVAILABLE, np.nan, a_long * 0.01),
        "length": df["vehicle_length"].to_numpy() / 100, "width": df["vehicle_width"].to_numpy() / 100,
    }, index=df.index)
    out.loc[df["speed"].to_numpy() >= 8191, "speed"] = np.nan
    out.loc[df["heading"].to_numpy() >= 28800, "heading"] = np.nan
    return out


def bsm_json(row: pd.Series | dict) -> dict:
    """One BSM wire row as a J2735 MessageFrame (JER), Part I only."""
    return {"messageId": 20, "value": {"coreData": {
        "msgCnt": int(row["msg_count"]), "id": str(row["temp_id"]), "secMark": int(row["sec_mark"]),
        "lat": int(row["latitude"]), "long": int(row["longitude"]), "elev": int(row["elevation"]),
        "speed": int(row["speed"]), "heading": int(row["heading"]),
        "accelSet": {"long": int(row["longitudinal_acceleration"]), "lat": int(row["lateral_acceleration"]), "vert": 0, "yaw": int(row["yaw_rate"])},
        "size": {"width": int(row["vehicle_width"]), "length": int(row["vehicle_length"])},
    }}}
