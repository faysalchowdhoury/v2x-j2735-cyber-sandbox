"""Tests on real J2735 data (a committed 15-minute fixture from UDOT intersection 6311) plus deterministic
synthetic-track unit tests. Run: ``pytest``.

The headline checks required of the sandbox: the Layer-1 protocol monitor raises zero false positives on a
clean real NEMA SPaT stream, and each of the four attack families is detected when injected into real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_day
from src.detector import Detector, context_features, kinematic_features, protocol_flags
from src.j2735_models import IntersectionMap, PHASES, TIMEMARK_UNKNOWN, decode_bsm, encode_bsm_wire, timemark_of, timemark_to_epoch
from src.nema_controller import CONFLICT_PAIRS, DEFAULT_CHECKS, check_stream, conflicts, estimate_timing
from src.scenario_generator import ScenarioConfig, generate

FIX = Path(__file__).resolve().parent / "fixtures"
INTER, DATE = "6311", "2025-02-07"


@pytest.fixture(scope="module")
def day():
    return load_day(FIX, INTER, DATE)


# ----------------------------------------------------------------------------- J2735 models

def test_timemark_roundtrip() -> None:
    t = 1_700_000_000.0 + np.arange(0, 30, 0.1)
    tm = timemark_of(t)
    assert tm.min() >= 0 and tm.max() < 36000
    back = timemark_to_epoch(tm, t)
    assert np.allclose(back, t, atol=0.05)
    assert np.isnan(timemark_to_epoch(np.array([TIMEMARK_UNKNOWN]), np.array([t[0]]))[0])


def test_bsm_wire_units_roundtrip() -> None:
    # a BSM wire row decodes to the SI values it was built from
    imap = IntersectionMap.from_record(__import__("json").loads((FIX / INTER / DATE / "map.json").read_text()))
    x = np.array([10.0, -40.0]); y = np.array([5.0, -120.0])
    wire = encode_bsm_wire(imap, t=np.array([1.0, 1.1]), temp_id=np.array(["AA", "BB"]), x=x, y=y,
                           speed=np.array([12.0, 0.0]), heading=np.array([90.0, 270.0]), a_long=np.array([1.5, np.nan]))
    dec = decode_bsm(wire, imap)
    assert np.allclose(dec["x"], x, atol=0.3) and np.allclose(dec["y"], y, atol=0.3)
    assert np.allclose(dec["speed"], [12.0, 0.0], atol=0.02)
    assert abs(dec["a_long"].iloc[0] - 1.5) < 0.02 and np.isnan(dec["a_long"].iloc[1])


# ----------------------------------------------------------------------------- NEMA

def test_conflict_matrix() -> None:
    assert conflicts(2, 1) and conflicts(2, 4) and conflicts(2, 3)  # opposing-ring / cross-barrier
    assert not conflicts(2, 5) and not conflicts(2, 6)  # same-direction left and complementary through
    assert not conflicts(2, 2)
    assert len(CONFLICT_PAIRS) == 20


def test_map_parse_and_matching(day) -> None:
    assert day.imap.name and len(day.imap.ingress) >= 8
    on = day.bsm["lane"] >= 0
    assert on.sum() > 1000  # a good share of real BSMs match an ingress lane
    assert set(day.bsm.loc[on, "phase"].unique()) <= set(PHASES)


# ----------------------------------------------------------------------------- Layer 1 on clean real data

def test_clean_nema_cycles_zero_protocol_false_positives(day) -> None:
    plan = estimate_timing(day.spat)
    viol = check_stream(day.spat, plan, DEFAULT_CHECKS)
    conflict_fp = sum(1 for v in viol for item in v if item.startswith("conflict"))
    total_fp = sum(1 for v in viol if v)
    assert conflict_fp == 0  # no conflicting greens in honest data -- the core invariant
    assert total_fp == 0  # the default check set is calibrated to be clean


# ----------------------------------------------------------------------------- attacks detected

def test_conflicting_green_flagged(day) -> None:
    plan = estimate_timing(day.spat)
    sc = generate(day, ScenarioConfig(attack="conflicting_green", variant="conflict", onset=60, duration=8))
    assert sc.spat_labels.sum() > 0
    viol = check_stream(sc.spat, plan)
    flagged = sum(1 for v in viol if any(i.startswith("conflict") for i in v))
    assert flagged >= sc.spat_labels.sum() * 0.5  # the conflict window is caught message by message


def test_timemark_flagged(day) -> None:
    plan = estimate_timing(day.spat)
    sc = generate(day, ScenarioConfig(attack="timemark", onset=60, duration=6))
    if sc.spat_labels.sum() == 0:
        pytest.skip("no suitable green interval in the fixture window")
    viol = check_stream(sc.spat, plan)
    assert any(any(i.startswith("min_after_max") for i in v) for v in viol)


def test_kinematic_residual_spikes_on_teleport(day) -> None:
    sc = generate(day, ScenarioConfig(attack="kinematic", variant="teleport", onset=30, duration=8))
    if sc.bsm_labels.sum() == 0:
        pytest.skip("no approaching vehicle in the fixture window")
    kf = kinematic_features(sc.bsm)
    lab = sc.bsm_labels.astype(bool)
    assert kf["pos_resid"].to_numpy()[lab].max() > 15  # the position jump shows up strongly


def test_ghost_queue_injected_and_dense(day) -> None:
    sc = generate(day, ScenarioConfig(attack="ghost_queue", onset=60, duration=10))
    assert sc.bsm["ghost"].sum() > 0 and (sc.bsm_labels == sc.bsm["ghost"].to_numpy()).all()
    ctx = context_features(sc.bsm, sc.spat)
    assert ctx["density"].to_numpy()[sc.bsm["ghost"].to_numpy()].mean() > 1.5


# ----------------------------------------------------------------------------- end to end

def test_detector_end_to_end(day) -> None:
    plan = estimate_timing(day.spat)
    base = generate(day, ScenarioConfig(attack="none"))
    train = [(base.bsm, base.spat)]
    for v in ("teleport", "speed_spoof", "fake_brake"):
        train.append(_t(generate(day, ScenarioConfig(attack="kinematic", variant=v, onset=30, duration=10))))
    for on in (120, 300, 500):
        train.append(_t(generate(day, ScenarioConfig(attack="ghost_queue", onset=on, duration=10))))
    det = Detector(plan=plan).fit(train)
    assert np.isfinite(det.w_).all()
    det.set_threshold(base.bsm, base.spat, target_fpr=0.02)
    # clean false-positive rate stays near the target
    clean_fpr = (det.score_bsm(base.bsm, base.spat) > det.threshold_).mean()
    assert clean_fpr < 0.05
    # a conflicting-green attack is detected with zero latency via Layer 1
    atk = generate(day, ScenarioConfig(attack="conflicting_green", variant="conflict", onset=60, duration=8))
    assert (det.score_spat(atk.spat)[atk.spat_labels == 1] > 0).any()
    # a ghost queue raises the fused BSM score on the forged messages
    ghost = generate(day, ScenarioConfig(attack="ghost_queue", onset=200, duration=10))
    s = det.score_bsm(ghost.bsm, ghost.spat)
    assert (s[ghost.bsm["ghost"].to_numpy()] > det.threshold_).mean() > 0.5


def _t(sc):
    return sc.bsm, sc.spat
