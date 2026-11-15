"""Download real J2735 SPaT, BSM and MAP data from the USDOT ITS DataHub (data.transportation.gov).

Source: "Utah Connected Vehicle Data Ecosystem Program" (UDOT and Panasonic), Orem, UT, February 2025:

* SPaT  https://data.transportation.gov/d/iq8k-ytf6  (10 Hz, 3 intersections, 7 days)
* BSM   https://data.transportation.gov/d/ev64-jn8c  (deduplicated BSMs heard by the RSUs)
* MAP   https://data.transportation.gov/d/aj9z-iw7k

SPaT messages are flattened to one row per message (state, minEndTime and maxEndTime per signal group 1-8);
BSMs are kept within ``--radius`` metres of the intersection reference point. Output, per day:
``data/raw/<intersection>/<date>/spat.csv.gz``, ``bsm.csv.gz`` and ``map.json``.

Usage: ``python scripts/download_data.py --intersection 6311 --date 2025-02-07``
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

BASE = "https://data.transportation.gov/resource/{}.json"
SPAT, BSM, MAP = "iq8k-ytf6", "ev64-jn8c", "aj9z-iw7k"
CODE = {"protected-Movement-Allowed": "G", "permissive-Movement-Allowed": "g", "protected-clearance": "Y", "permissive-clearance": "y",
        "stop-And-Remain": "R", "stop-Then-Proceed": "F", "pre-Movement": "P", "caution-Conflicting-Traffic": "C", "dark": "D", "unavailable": "U"}
BSM_COLS = ["tmstp_epoch", "tmstp_utc", "temp_id", "msg_count", "sec_mark", "latitude", "longitude", "elevation", "heading", "speed",
            "longitudinal_acceleration", "lateral_acceleration", "yaw_rate", "brake_unavailable", "wheel_brakes_flags",
            "vehicle_length", "vehicle_width", "semi_major_axis_accuracy"]


def get(dataset: str, params: dict, tries: int = 6) -> list[dict]:
    url = BASE.format(dataset) + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "v2x-sandbox-downloader"}), timeout=600) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - retry any network error
            if k == tries - 1:
                raise
            print(f"  retry {k + 1}: {e}")
            time.sleep(10 * (k + 1))
    return []


def flatten_spat(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        rec = {"tmstp_utc": r["tmstp_utc"], "rev": int(r["intersection_msg_count"]), "moy": int(r["intersection_moy_utc"]),
               "ms": int(r["intersection_milliseconds"]), "fixed_time": r.get("fixed_time_operation"), "preempt": r.get("preempt_active"),
               "priority": r.get("signal_priority_active")}
        for s in json.loads(r["intersection_states"]):
            p = int(s["signalGroup"])
            e = s["state-time-speed"][0]
            tm = e.get("timing", {})
            rec[f"state_{p}"] = CODE.get(e["eventState"], "U")
            rec[f"min_{p}"] = tm.get("minEndTime", 36000)
            rec[f"max_{p}"] = tm.get("maxEndTime", 36000)
        out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intersection", default="6311")
    ap.add_argument("--date", default="2025-02-07")
    ap.add_argument("--radius", type=float, default=300.0)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--chunk-min", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out) / a.intersection / a.date
    out.mkdir(parents=True, exist_ok=True)
    m = [r for r in get(MAP, {"$limit": 10}) if r["intersection_id"] == a.intersection][0]
    (out / "map.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    lat, lon = float(m["latitude"]), float(m["longitude"])
    print(f"{a.intersection} {m['intersection_name']} ({lat}, {lon})")

    day = datetime.fromisoformat(a.date)
    parts, t = [], day
    while t < day + timedelta(days=1):
        t1 = t + timedelta(minutes=a.chunk_min)
        rows = get(SPAT, {"$select": "tmstp_utc,intersection_msg_count,intersection_moy_utc,intersection_milliseconds,intersection_states,"
                                     "fixed_time_operation,preempt_active,signal_priority_active",
                          "$where": f"intersection_id='{a.intersection}' and tmstp_utc >= '{t:%Y-%m-%dT%H:%M:%S}' and tmstp_utc < '{t1:%Y-%m-%dT%H:%M:%S}'",
                          "$order": "tmstp_utc", "$limit": 50000})
        parts.append(flatten_spat(rows))
        print(f"  SPaT {t:%H:%M} {len(rows)}", flush=True)
        t = t1
    spat = pd.concat(parts, ignore_index=True)
    spat.to_csv(out / "spat.csv.gz", index=False, compression="gzip")

    dlat = a.radius / 111320
    dlon = a.radius / (111320 * math.cos(math.radians(lat)))
    where = (f"latitude between {int((lat - dlat) * 1e7)} and {int((lat + dlat) * 1e7)} and longitude between {int((lon - dlon) * 1e7)} "
             f"and {int((lon + dlon) * 1e7)} and date_utc='{a.date}T00:00:00.000'")
    rows, off = [], 0
    while True:
        page = get(BSM, {"$select": ",".join(BSM_COLS), "$where": where, "$order": "tmstp_epoch,temp_id", "$limit": 50000, "$offset": off})
        rows += page
        off += len(page)
        print(f"  BSM {off}", flush=True)
        if len(page) < 50000:
            break
    pd.DataFrame(rows, columns=BSM_COLS).to_csv(out / "bsm.csv.gz", index=False, compression="gzip")
    print(f"done: {len(spat)} SPaT, {len(rows)} BSM -> {out}")


if __name__ == "__main__":
    main()
