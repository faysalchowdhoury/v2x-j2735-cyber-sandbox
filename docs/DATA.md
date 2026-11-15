# Data provenance

All honest messages in this sandbox are real, from the **Utah DOT / Panasonic Connected Vehicle Data Ecosystem
Program** in Orem, Utah, published openly on the USDOT ITS DataHub (`data.transportation.gov`). The program
instrumented three signalized intersections on State Street with road-side units that logged J2735 SPaT and the
BSMs broadcast by equipped on-board units, plus MAP lane geometry.

| Message | DataHub id | What `download_data.py` keeps |
|---|---|---|
| SPaT | `iq8k-ytf6` | one row per message: state, `minEndTime`, `maxEndTime` for signal groups 1–8, plus revision/time fields |
| BSM | `ev64-jn8c` | messages within `--radius` m of the intersection: id, position, speed, heading, accelerations, size |
| MAP | `aj9z-iw7k` | the intersection's lane set (node offsets, ingress/egress, signal-group connections) |

## Default selection

* **Intersection 6311 — N State St & E Center St**, a standard 8-phase dual-ring NEMA TS2 intersection whose
  MAP assigns 2/6 to the State St throughs, 1/5 to the State St lefts, 4/8 to the Center St throughs and 3/7 to
  the Center St lefts.
* **2025-02-07**, a full day (24 h) of SPaT (~862k messages) and BSM (~456k messages near the intersection).

## Properties worth knowing (measured, not assumed)

* SPaT arrives at 10 Hz (median inter-message gap 0.100 s).
* Equipped-vehicle penetration is **low and bursty**: ~150–170 distinct vehicles approach a stop bar over the
  day, 3–16 per hour. This is why the benchmark runs on a multi-hour window and anchors kinematic attacks to
  real approaching vehicles.
* The reported BSM acceleration field is "unavailable" (code 2001) ~58 % of the time, so the physics layer
  derives acceleration from position rather than trusting the field.
* Across a clean day, **no** pair of NEMA-conflicting phases is ever green at the same time, and compatible
  pairs (2+6, 2+5, 4+8) overlap as expected — the conflict matrix holds on real data, which is what makes the
  Layer-1 conflict check a zero-false-positive detector.

## Committed fixture

`tests/fixtures/6311/2025-02-07/` holds a 15-minute real slice (SPaT, BSM, MAP) so `pytest` runs with no
download. It is a verbatim excerpt of the public data above.
