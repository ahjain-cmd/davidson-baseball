"""
Count Calibration — derive pitch-call weight multipliers from historical
count run values.

For each of 12 counts (0-0 through 3-2), we compute the empirical expected
run value of plate appearances that reach that count, then derive how much a
strike or ball is worth at each count.  These transition values become the
weight multipliers used by ``_count_adjustments()`` in
``decision_engine/recommenders/pitch_call.py``.

The cache-with-fingerprint pattern matches ``decision_engine/data/priors.py``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import duckdb
import numpy as np

from config import CACHE_DIR, PARQUET_PATH, ZONE_SIDE, ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP
from decision_engine.core.shrinkage import shrink_value

# ── Standard linear weights ──────────────────────────────────────────────────
# These are loaded from historical calibration if available, falling back to
# sabermetric consensus values.

def _load_calibrated_linear_weights():
    """Attempt to load empirically-computed wOBA weights from historical calibration cache.

    Only reads from cache (won't trigger a full computation) to avoid expensive
    side effects during count calibration. Run ``calibrate_historical.py`` first
    to populate the cache.
    """
    try:
        cache_file = os.path.join(CACHE_DIR, "historical_calibration.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                blob = json.load(f)
            cal_data = blob.get("calibration", {})
            lw = cal_data.get("linear_weights")
            if lw:
                return {
                    "Strikeout": 0.00,
                    "Out": 0.00,
                    "FieldersChoice": 0.00,
                    "Sacrifice": 0.00,
                    "Walk": lw.get("bb_w", 0.32),
                    "HitByPitch": lw.get("hbp_w", 0.32),
                    "Error": lw.get("single_w", 0.47),
                    "Single": lw.get("single_w", 0.47),
                    "Double": lw.get("double_w", 0.78),
                    "Triple": lw.get("triple_w", 1.05),
                    "HomeRun": lw.get("hr_w", 1.40),
                }
    except Exception:
        pass
    return None


# Fallback values (sabermetric consensus)
_FALLBACK_LINEAR_WEIGHTS = {
    "Strikeout": 0.00,
    "Out": 0.00,
    "FieldersChoice": 0.00,
    "Sacrifice": 0.00,
    "Walk": 0.32,
    "HitByPitch": 0.32,
    "Error": 0.47,
    "Single": 0.47,
    "Double": 0.78,
    "Triple": 1.05,
    "HomeRun": 1.40,
}

# Lazy-loaded: calibrated if available, fallback otherwise
_LINEAR_WEIGHTS_CACHE = None


def _get_linear_weights():
    global _LINEAR_WEIGHTS_CACHE
    if _LINEAR_WEIGHTS_CACHE is None:
        _LINEAR_WEIGHTS_CACHE = _load_calibrated_linear_weights() or _FALLBACK_LINEAR_WEIGHTS
    return _LINEAR_WEIGHTS_CACHE


# Public alias for backward compatibility
LINEAR_WEIGHTS = _FALLBACK_LINEAR_WEIGHTS

# BB run value used for the 3-ball→walk transition (computed lazily)
def _get_bb_rv():
    return _get_linear_weights()["Walk"]

# ── Pitch-type classification (mirrors pitch_call.py) ───────────────────────
_HARD_PITCHES_SQL = "('Fastball', 'Sinker', 'Cutter', 'FourSeamFastBall', 'OneSeamFastBall', 'TwoSeamFastBall')"
_OFFSPEED_PITCHES_SQL = (
    "('Slider', 'Curveball', 'Changeup', 'Splitter', 'Sweeper', 'Knuckle Curve', "
    "'ChangeUp', 'KnuckleCurve')"
)

# Pitch-type normalization SQL (same as priors.py)
_PT_NORM = (
    "CASE "
    "WHEN TaggedPitchType IN ('Undefined','Other','Knuckleball') THEN NULL "
    "WHEN TaggedPitchType = 'FourSeamFastBall' THEN 'Fastball' "
    "WHEN TaggedPitchType IN ('OneSeamFastBall','TwoSeamFastBall') THEN 'Sinker' "
    "WHEN TaggedPitchType = 'ChangeUp' THEN 'Changeup' "
    "ELSE TaggedPitchType END"
)

# Zone SQL
_HAS_LOC = "PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL"
_IN_ZONE = f"ABS(PlateLocSide) <= {ZONE_SIDE} AND PlateLocHeight BETWEEN {ZONE_HEIGHT_BOT} AND {ZONE_HEIGHT_TOP}"
_OUT_ZONE = f"({_HAS_LOC}) AND NOT ({_IN_ZONE})"
_SWING_CALLS = "('StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay')"


@dataclass(frozen=True)
class CountWeights:
    """Calibrated weight multipliers for a single count."""
    whiff_w: float
    csw_w: float
    chase_w: float
    cmd_w: float
    hard_delta: float
    offspeed_delta: float


@dataclass
class CountCalibration:
    """Full calibration result: per-count weights + diagnostics."""
    weights: Dict[str, CountWeights]
    run_values: Dict[str, float]          # "b-s" -> RV
    strike_gain: Dict[str, float]         # "b-s" -> strike_gain
    ball_cost: Dict[str, float]           # "b-s" -> ball_cost
    chase_whiff_rate: Dict[str, float]    # "b-s" -> P(whiff|chase)
    hard_rv: Dict[str, float]             # "b-s" -> avg RV for hard pitches
    offspeed_rv: Dict[str, float]         # "b-s" -> avg RV for offspeed pitches
    pa_counts: Dict[str, int]             # "b-s" -> number of PAs through this count


# ── Cache fingerprint (same pattern as priors.py) ───────────────────────────

def _parquet_fingerprint(path: str) -> Dict:
    try:
        return {"path": path, "mtime": os.path.getmtime(path), "size": os.path.getsize(path)}
    except OSError:
        return {"path": path, "mtime": None, "size": None}


def _cache_path() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "count_calibration.json")


# ── Core computation ────────────────────────────────────────────────────────

def compute_count_calibration(parquet_path: str = PARQUET_PATH) -> CountCalibration:
    """Compute count run values and calibrated weights from Trackman parquet."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    pq = parquet_path.replace("'", "''")
    con = duckdb.connect(database=":memory:")

    # Load calibrated linear weights for run-value mapping
    lw = _get_linear_weights()

    # ── Query 1: Build PA outcome table ─────────────────────────────────────
    # Each PA gets a unique id and its terminal outcome mapped to a run value.
    con.execute(f"""
        CREATE TEMP TABLE pa_outcomes AS
        WITH pa_pitches AS (
            SELECT
                GameID || '_' || CAST(Inning AS VARCHAR) || '_'
                    || CAST(PAofInning AS VARCHAR) || '_' || Batter AS pa_id,
                CAST(Balls AS INTEGER) AS Balls,
                CAST(Strikes AS INTEGER) AS Strikes,
                PitchCall,
                KorBB,
                PlayResult,
                {_PT_NORM} AS pitch_type,
                PlateLocSide,
                PlateLocHeight,
                ExitSpeed
            FROM read_parquet('{pq}')
            WHERE PitchCall IS NOT NULL AND PitchCall != 'Undefined'
        ),
        pa_terminal AS (
            SELECT
                pa_id,
                -- Determine PA outcome: K/BB from KorBB, else from last InPlay PlayResult
                CASE
                    WHEN MAX(CASE WHEN KorBB = 'Strikeout' THEN 1 ELSE 0 END) = 1 THEN 'Strikeout'
                    WHEN MAX(CASE WHEN KorBB = 'Walk' THEN 1 ELSE 0 END) = 1 THEN 'Walk'
                    WHEN MAX(CASE WHEN PitchCall = 'HitByPitch' THEN 1 ELSE 0 END) = 1 THEN 'HitByPitch'
                    WHEN MAX(CASE WHEN PitchCall = 'InPlay' AND PlayResult = 'Single' THEN 1 ELSE 0 END) = 1 THEN 'Single'
                    WHEN MAX(CASE WHEN PitchCall = 'InPlay' AND PlayResult = 'Double' THEN 1 ELSE 0 END) = 1 THEN 'Double'
                    WHEN MAX(CASE WHEN PitchCall = 'InPlay' AND PlayResult = 'Triple' THEN 1 ELSE 0 END) = 1 THEN 'Triple'
                    WHEN MAX(CASE WHEN PitchCall = 'InPlay' AND PlayResult = 'HomeRun' THEN 1 ELSE 0 END) = 1 THEN 'HomeRun'
                    WHEN MAX(CASE WHEN PitchCall = 'InPlay' AND PlayResult = 'Error' THEN 1 ELSE 0 END) = 1 THEN 'Error'
                    WHEN MAX(CASE WHEN PitchCall = 'InPlay' AND PlayResult IN ('Out','FieldersChoice','Sacrifice') THEN 1 ELSE 0 END) = 1 THEN 'Out'
                    ELSE NULL
                END AS outcome
            FROM pa_pitches
            GROUP BY pa_id
        )
        SELECT
            p.pa_id,
            p.Balls AS balls,
            p.Strikes AS strikes,
            p.PitchCall AS pitch_call,
            p.pitch_type,
            p.PlateLocSide,
            p.PlateLocHeight,
            p.ExitSpeed,
            t.outcome,
            CASE t.outcome
                WHEN 'Strikeout' THEN {lw['Strikeout']:.4f}
                WHEN 'Out' THEN {lw['Out']:.4f}
                WHEN 'Walk' THEN {lw['Walk']:.4f}
                WHEN 'HitByPitch' THEN {lw['HitByPitch']:.4f}
                WHEN 'Error' THEN {lw['Error']:.4f}
                WHEN 'Single' THEN {lw['Single']:.4f}
                WHEN 'Double' THEN {lw['Double']:.4f}
                WHEN 'Triple' THEN {lw['Triple']:.4f}
                WHEN 'HomeRun' THEN {lw['HomeRun']:.4f}
                ELSE NULL
            END AS run_value
        FROM pa_pitches p
        JOIN pa_terminal t ON p.pa_id = t.pa_id
        WHERE t.outcome IS NOT NULL
    """)

    # ── Query 2: RV(b,s) for all 12 counts ─────────────────────────────────
    rv_rows = con.execute("""
        SELECT
            balls, strikes,
            AVG(run_value) AS rv,
            COUNT(DISTINCT pa_id) AS n_pas
        FROM pa_outcomes
        GROUP BY balls, strikes
        HAVING balls <= 3 AND strikes <= 2
        ORDER BY balls, strikes
    """).fetchall()

    run_values: Dict[str, float] = {}
    pa_counts: Dict[str, int] = {}
    for b, s, rv, n in rv_rows:
        key = f"{b}-{s}"
        run_values[key] = float(rv)
        pa_counts[key] = int(n)

    # ── Query 3: Per-count sub-metrics ──────────────────────────────────────
    # Chase-to-whiff rate at each count
    chase_rows = con.execute(f"""
        SELECT
            balls, strikes,
            SUM(CASE WHEN pitch_call = 'StrikeSwinging' THEN 1 ELSE 0 END) AS chase_whiffs,
            COUNT(*) AS chase_swings
        FROM pa_outcomes
        WHERE {_OUT_ZONE}
          AND pitch_call IN {_SWING_CALLS}
          AND balls <= 3 AND strikes <= 2
        GROUP BY balls, strikes
    """).fetchall()

    raw_chase_whiff: Dict[str, tuple] = {}
    for b, s, cw, cs in chase_rows:
        key = f"{b}-{s}"
        rate = float(cw) / float(cs) if cs > 0 else 0.0
        raw_chase_whiff[key] = (rate, int(cs))

    # Hard vs offspeed run value differential at each count
    class_rows = con.execute(f"""
        SELECT
            balls, strikes,
            CASE
                WHEN pitch_type IN ('Fastball','Sinker','Cutter') THEN 'hard'
                WHEN pitch_type IN ('Slider','Curveball','Changeup','Splitter','Sweeper','Knuckle Curve') THEN 'offspeed'
                ELSE NULL
            END AS pitch_class,
            AVG(run_value) AS class_rv,
            COUNT(DISTINCT pa_id) AS n_pas
        FROM pa_outcomes
        WHERE pitch_type IS NOT NULL
          AND balls <= 3 AND strikes <= 2
        GROUP BY balls, strikes, pitch_class
        HAVING pitch_class IS NOT NULL
    """).fetchall()

    hard_rv_raw: Dict[str, tuple] = {}    # key -> (rv, n)
    offspeed_rv_raw: Dict[str, tuple] = {}
    for b, s, pc, crv, n in class_rows:
        key = f"{b}-{s}"
        if pc == "hard":
            hard_rv_raw[key] = (float(crv), int(n))
        elif pc == "offspeed":
            offspeed_rv_raw[key] = (float(crv), int(n))

    con.close()

    # ── Derive transition values ────────────────────────────────────────────
    strike_gain: Dict[str, float] = {}
    ball_cost: Dict[str, float] = {}

    for b in range(4):
        for s in range(3):
            key = f"{b}-{s}"
            rv_here = run_values.get(key)
            if rv_here is None:
                continue

            # Strike gain: RV(b,s) - RV(b, s+1)
            # At 2 strikes, a strike = strikeout (RV=0)
            if s == 2:
                sg = rv_here - 0.0  # K run value = 0
            else:
                rv_next_s = run_values.get(f"{b}-{s+1}")
                if rv_next_s is not None:
                    sg = rv_here - rv_next_s
                else:
                    sg = rv_here * 0.3  # fallback estimate
            strike_gain[key] = sg

            # Ball cost: RV(b+1, s) - RV(b,s)
            # At 3 balls, a ball = walk (RV=0.32)
            if b == 3:
                bc = _get_bb_rv() - rv_here
            else:
                rv_next_b = run_values.get(f"{b+1}-{s}")
                if rv_next_b is not None:
                    bc = rv_next_b - rv_here
                else:
                    bc = 0.03  # fallback estimate
            ball_cost[key] = bc

    # ── Shrink sparse sub-metrics ───────────────────────────────────────────
    # Global priors for shrinkage
    all_chase_whiff = [v[0] for v in raw_chase_whiff.values() if v[1] > 0]
    global_chase_whiff = float(np.mean(all_chase_whiff)) if all_chase_whiff else 0.35

    all_hard = [v[0] for v in hard_rv_raw.values() if v[1] > 0]
    all_offspeed = [v[0] for v in offspeed_rv_raw.values() if v[1] > 0]
    global_hard_rv = float(np.mean(all_hard)) if all_hard else 0.20
    global_offspeed_rv = float(np.mean(all_offspeed)) if all_offspeed else 0.20

    chase_whiff_rate: Dict[str, float] = {}
    hard_rv: Dict[str, float] = {}
    offspeed_rv: Dict[str, float] = {}

    for key in run_values:
        # Chase whiff rate with shrinkage
        raw = raw_chase_whiff.get(key, (global_chase_whiff, 0))
        shrunk = shrink_value(raw[0], raw[1], global_chase_whiff, n_prior_equiv=200)
        chase_whiff_rate[key] = float(shrunk) if shrunk is not None else global_chase_whiff

        # Hard RV with shrinkage
        raw_h = hard_rv_raw.get(key, (global_hard_rv, 0))
        shrunk_h = shrink_value(raw_h[0], raw_h[1], global_hard_rv, n_prior_equiv=200)
        hard_rv[key] = float(shrunk_h) if shrunk_h is not None else global_hard_rv

        # Offspeed RV with shrinkage
        raw_o = offspeed_rv_raw.get(key, (global_offspeed_rv, 0))
        shrunk_o = shrink_value(raw_o[0], raw_o[1], global_offspeed_rv, n_prior_equiv=200)
        offspeed_rv[key] = float(shrunk_o) if shrunk_o is not None else global_offspeed_rv

    # ── Derive factor weights ───────────────────────────────────────────────
    raw_whiff_w: Dict[str, float] = {}
    raw_csw_w: Dict[str, float] = {}
    raw_chase_w: Dict[str, float] = {}
    raw_cmd_w: Dict[str, float] = {}
    raw_hard_delta: Dict[str, float] = {}
    raw_offspeed_delta: Dict[str, float] = {}

    for key in run_values:
        sg = strike_gain.get(key, 0.0)
        bc = ball_cost.get(key, 0.0)
        cwh = chase_whiff_rate.get(key, global_chase_whiff)

        raw_whiff_w[key] = sg
        # CSW includes fouls which don't advance at 2 strikes
        b, s = int(key[0]), int(key[2])
        csw_discount = 0.85 if s == 2 else 0.95
        raw_csw_w[key] = sg * csw_discount
        raw_chase_w[key] = sg * cwh
        raw_cmd_w[key] = sg + bc

        # Class deltas: how much better/worse is hard vs offspeed at this count
        # Negative delta = lower RV = better for pitcher
        base_rv = run_values.get(key, 0.20)
        h_rv = hard_rv.get(key, base_rv)
        o_rv = offspeed_rv.get(key, base_rv)
        # Convert to "pitch score bonus": lower RV for pitcher = better
        # hard_delta > 0 means hard pitches suppress RV better at this count
        raw_hard_delta[key] = (base_rv - h_rv) * 10.0     # scale to score-like range
        raw_offspeed_delta[key] = (base_rv - o_rv) * 10.0

    # ── Normalize weights ───────────────────────────────────────────────────
    # Normalize so max(cmd_w) ≈ 10.0
    max_cmd = max(raw_cmd_w.values()) if raw_cmd_w else 1.0
    cmd_scale = 10.0 / max_cmd if max_cmd > 0 else 1.0

    # Normalize class deltas so max(|delta|) ≈ 4.0
    max_class_delta = max(
        max(abs(v) for v in raw_hard_delta.values()) if raw_hard_delta else 1.0,
        max(abs(v) for v in raw_offspeed_delta.values()) if raw_offspeed_delta else 1.0,
    )
    class_scale = 4.0 / max_class_delta if max_class_delta > 0 else 1.0

    weights: Dict[str, CountWeights] = {}
    for key in run_values:
        weights[key] = CountWeights(
            whiff_w=round(raw_whiff_w[key] * cmd_scale, 3),
            csw_w=round(raw_csw_w[key] * cmd_scale, 3),
            chase_w=round(raw_chase_w[key] * cmd_scale, 3),
            cmd_w=round(raw_cmd_w[key] * cmd_scale, 3),
            hard_delta=round(raw_hard_delta[key] * class_scale, 3),
            offspeed_delta=round(raw_offspeed_delta[key] * class_scale, 3),
        )

    # ── Sanity checks ───────────────────────────────────────────────────────
    _assert_monotonicity(run_values)
    _assert_positive_transitions(strike_gain, ball_cost)

    return CountCalibration(
        weights=weights,
        run_values={k: round(v, 5) for k, v in run_values.items()},
        strike_gain={k: round(v, 5) for k, v in strike_gain.items()},
        ball_cost={k: round(v, 5) for k, v in ball_cost.items()},
        chase_whiff_rate={k: round(v, 5) for k, v in chase_whiff_rate.items()},
        hard_rv={k: round(v, 5) for k, v in hard_rv.items()},
        offspeed_rv={k: round(v, 5) for k, v in offspeed_rv.items()},
        pa_counts=pa_counts,
    )


def _assert_monotonicity(run_values: Dict[str, float]) -> None:
    """RV should increase with balls (for fixed strikes) and decrease with strikes."""
    for s in range(3):
        for b in range(3):
            key_lo = f"{b}-{s}"
            key_hi = f"{b+1}-{s}"
            rv_lo = run_values.get(key_lo)
            rv_hi = run_values.get(key_hi)
            if rv_lo is not None and rv_hi is not None:
                assert rv_hi >= rv_lo, (
                    f"RV should increase with balls: RV({key_hi})={rv_hi:.4f} < RV({key_lo})={rv_lo:.4f}"
                )

    for b in range(4):
        for s in range(2):
            key_lo = f"{b}-{s}"
            key_hi = f"{b}-{s+1}"
            rv_lo = run_values.get(key_lo)
            rv_hi = run_values.get(key_hi)
            if rv_lo is not None and rv_hi is not None:
                assert rv_hi <= rv_lo, (
                    f"RV should decrease with strikes: RV({key_hi})={rv_hi:.4f} > RV({key_lo})={rv_lo:.4f}"
                )


def _assert_positive_transitions(
    strike_gain: Dict[str, float], ball_cost: Dict[str, float]
) -> None:
    """Strike gains and ball costs should all be positive."""
    for key, sg in strike_gain.items():
        assert sg >= 0, f"strike_gain({key})={sg:.4f} is negative"
    for key, bc in ball_cost.items():
        assert bc >= 0, f"ball_cost({key})={bc:.4f} is negative"


# ── Cache load/save ─────────────────────────────────────────────────────────

def _serialize_calibration(cal: CountCalibration) -> dict:
    return {
        "weights": {k: asdict(v) for k, v in cal.weights.items()},
        "run_values": cal.run_values,
        "strike_gain": cal.strike_gain,
        "ball_cost": cal.ball_cost,
        "chase_whiff_rate": cal.chase_whiff_rate,
        "hard_rv": cal.hard_rv,
        "offspeed_rv": cal.offspeed_rv,
        "pa_counts": cal.pa_counts,
    }


def _deserialize_calibration(blob: dict) -> CountCalibration:
    weights = {
        k: CountWeights(**v) for k, v in blob["weights"].items()
    }
    return CountCalibration(
        weights=weights,
        run_values=blob["run_values"],
        strike_gain=blob["strike_gain"],
        ball_cost=blob["ball_cost"],
        chase_whiff_rate=blob["chase_whiff_rate"],
        hard_rv=blob["hard_rv"],
        offspeed_rv=blob["offspeed_rv"],
        pa_counts=blob["pa_counts"],
    )


def load_count_calibration(
    parquet_path: str = PARQUET_PATH, force_refresh: bool = False,
) -> Optional[CountCalibration]:
    """Load count calibration from cache, recomputing if parquet changed.

    Returns None if no parquet file exists (graceful degradation).
    """
    if not os.path.exists(parquet_path):
        return None

    cache = _cache_path()
    fp = _parquet_fingerprint(parquet_path)

    if not force_refresh and os.path.exists(cache):
        try:
            with open(cache, "r", encoding="utf-8") as f:
                blob = json.load(f)
            if blob.get("fingerprint") == fp and "calibration" in blob:
                return _deserialize_calibration(blob["calibration"])
        except Exception:
            pass

    cal = compute_count_calibration(parquet_path=parquet_path)
    try:
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(
                {"fingerprint": fp, "calibration": _serialize_calibration(cal)},
                f,
                indent=2,
            )
    except Exception:
        pass
    return cal


# ── Fallback weights (current hardcoded magic numbers) ──────────────────────

def fallback_weights() -> Dict[str, CountWeights]:
    """Encode the current hardcoded magic numbers from pitch_call.py as CountWeights.

    Used when calibration data isn't available (graceful degradation).
    """
    w: Dict[str, CountWeights] = {}

    # 0-0: CSW-forward (4.0 * csw_n, neutral cmd/whiff)
    w["0-0"] = CountWeights(whiff_w=0.0, csw_w=4.0, chase_w=0.0, cmd_w=0.0,
                            hard_delta=0.0, offspeed_delta=0.0)

    # Neutral counts (1-0, 0-1, 1-1): 2.0 * csw + 2.0 * cmd
    for key in ("1-0", "0-1", "1-1"):
        w[key] = CountWeights(whiff_w=0.0, csw_w=2.0, chase_w=0.0, cmd_w=2.0,
                              hard_delta=0.0, offspeed_delta=0.0)

    # 2-strike counts (putaway): 8.0 * whiff, offspeed +2
    # 0-2: + 4.0 * chase
    w["0-2"] = CountWeights(whiff_w=8.0, csw_w=0.0, chase_w=4.0, cmd_w=0.0,
                            hard_delta=0.0, offspeed_delta=2.0)
    w["1-2"] = CountWeights(whiff_w=8.0, csw_w=0.0, chase_w=1.5, cmd_w=0.0,
                            hard_delta=0.0, offspeed_delta=2.0)

    # Hitter's count (2-0, 2-1): 4.0 * cmd, hard +2 / offspeed -2
    w["2-0"] = CountWeights(whiff_w=0.0, csw_w=0.0, chase_w=0.0, cmd_w=4.0,
                            hard_delta=2.0, offspeed_delta=-2.0)
    w["2-1"] = CountWeights(whiff_w=0.0, csw_w=0.0, chase_w=0.0, cmd_w=4.0,
                            hard_delta=2.0, offspeed_delta=-2.0)

    # 2-2: putaway (8.0 * whiff, offspeed +2, chase 1.5)
    w["2-2"] = CountWeights(whiff_w=8.0, csw_w=0.0, chase_w=1.5, cmd_w=0.0,
                            hard_delta=0.0, offspeed_delta=2.0)

    # 3-ball counts: command-forward, hard +4 / offspeed -4
    w["3-0"] = CountWeights(whiff_w=0.0, csw_w=0.0, chase_w=0.0, cmd_w=6.0,
                            hard_delta=4.0, offspeed_delta=-4.0)
    w["3-1"] = CountWeights(whiff_w=0.0, csw_w=0.0, chase_w=0.0, cmd_w=6.0,
                            hard_delta=4.0, offspeed_delta=-4.0)

    # 3-2: leverage (4.5 * cmd + 4.5 * whiff + 2.0 * chase)
    w["3-2"] = CountWeights(whiff_w=4.5, csw_w=0.0, chase_w=2.0, cmd_w=4.5,
                            hard_delta=0.0, offspeed_delta=0.0)

    return w
