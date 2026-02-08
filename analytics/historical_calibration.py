"""
Historical Calibration — compute calibratable metrics from Trackman parquet.

Sections:
  1A. Batted ball outcome probabilities by EV/LA bucket
  1B. wOBA / linear weights from actual run-scoring data
  1C. Steal success rates by pitch velocity class
  1D. GB% and DP% by pitch type
  1E. Pitch metric normalization ranges (P5/P95 for whiff%, CSW%, chase%)
  1F. D1 fielding benchmarks

Uses the same cache-with-fingerprint pattern as ``count_calibration.py`` and
``decision_engine/data/priors.py``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import duckdb
import numpy as np

from config import CACHE_DIR, PARQUET_PATH, ZONE_SIDE, ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP


# ── Pitch-type normalization SQL (shared with count_calibration / priors) ────
_PT_NORM = (
    "CASE "
    "WHEN TaggedPitchType IN ('Undefined','Other','Knuckleball') THEN NULL "
    "WHEN TaggedPitchType = 'FourSeamFastBall' THEN 'Fastball' "
    "WHEN TaggedPitchType IN ('OneSeamFastBall','TwoSeamFastBall') THEN 'Sinker' "
    "WHEN TaggedPitchType = 'ChangeUp' THEN 'Changeup' "
    "WHEN TaggedPitchType = 'KnuckleCurve' THEN 'Knuckle Curve' "
    "ELSE TaggedPitchType END"
)

# Zone / swing SQL
_HAS_LOC = "PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL"
_IN_ZONE = f"ABS(PlateLocSide) <= {ZONE_SIDE} AND PlateLocHeight BETWEEN {ZONE_HEIGHT_BOT} AND {ZONE_HEIGHT_TOP}"
_OUT_ZONE = f"({_HAS_LOC}) AND NOT ({_IN_ZONE})"
_SWING_CALLS = "('StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay')"

# Barrel definition in SQL (mirrors config.is_barrel)
_BARREL_COND = (
    "ExitSpeed >= 98 AND Angle IS NOT NULL AND "
    "Angle >= GREATEST(26 - 2*(ExitSpeed - 98), 8) AND "
    "Angle <= LEAST(30 + 3*(ExitSpeed - 98), 50)"
)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class HistoricalCalibration:
    """Full calibration result from Trackman parquet."""
    outcome_probs: Dict[str, Dict[str, float]]     # bucket -> {xOut, x1B, x2B, x3B, xHR, xErr, n}
    linear_weights: Dict[str, float]                # {out_w, bb_w, hbp_w, single_w, double_w, triple_w, hr_w}
    steal_rates: Dict[str, float]                   # {slow_sb_pct, med_sb_pct, fast_sb_pct, elite_sb_pct}
    gb_dp_rates: Dict[str, Dict[str, float]]        # pitch_type -> {gb_pct, dp_pct, out_pct, n_bip}
    metric_ranges: Dict[str, float]                 # {whiff_p5, whiff_p95, csw_p5, csw_p95, chase_p5, chase_p95}
    fielding_benchmarks: Dict[str, float]           # {fld_pct_median, fld_pct_p75, fld_pct_p90, error_rate_median, n_bip}


# ── Cache fingerprint ────────────────────────────────────────────────────────

def _parquet_fingerprint(path: str) -> Dict:
    try:
        return {"path": path, "mtime": os.path.getmtime(path), "size": os.path.getsize(path)}
    except OSError:
        return {"path": path, "mtime": None, "size": None}


def _cache_path() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "historical_calibration.json")


# ── 1A. Batted ball outcome probabilities ────────────────────────────────────

def _compute_outcome_probs(con: duckdb.DuckDBPyConnection, pq: str) -> Dict[str, Dict[str, float]]:
    """Compute outcome rates (Out/1B/2B/3B/HR/Error) for each EV/LA bucket."""
    rows = con.execute(f"""
        WITH bip AS (
            SELECT
                ExitSpeed AS ev,
                Angle AS la,
                PlayResult,
                CASE
                    WHEN {_BARREL_COND} THEN 'Barrel'
                    WHEN ExitSpeed >= 95 AND Angle BETWEEN 25 AND 45 THEN 'HiEV_FB'
                    WHEN Angle BETWEEN 10 AND 25 AND ExitSpeed >= 85 THEN 'Hard_LD'
                    WHEN Angle < 10 THEN 'GB'
                    WHEN Angle > 45 THEN 'Popup'
                    WHEN ExitSpeed < 70 THEN 'Soft'
                    ELSE 'Medium'
                END AS bucket
            FROM read_parquet('{pq}')
            WHERE PitchCall = 'InPlay'
              AND ExitSpeed IS NOT NULL
              AND Angle IS NOT NULL
              AND PlayResult IS NOT NULL
        )
        SELECT
            bucket,
            COUNT(*) AS n,
            SUM(CASE WHEN PlayResult IN ('Out','FieldersChoice','Sacrifice') THEN 1 ELSE 0 END) AS n_out,
            SUM(CASE WHEN PlayResult = 'Single' THEN 1 ELSE 0 END) AS n_1b,
            SUM(CASE WHEN PlayResult = 'Double' THEN 1 ELSE 0 END) AS n_2b,
            SUM(CASE WHEN PlayResult = 'Triple' THEN 1 ELSE 0 END) AS n_3b,
            SUM(CASE WHEN PlayResult = 'HomeRun' THEN 1 ELSE 0 END) AS n_hr,
            SUM(CASE WHEN PlayResult = 'Error' THEN 1 ELSE 0 END) AS n_err
        FROM bip
        GROUP BY bucket
    """).fetchall()

    result = {}
    for bucket, n, n_out, n_1b, n_2b, n_3b, n_hr, n_err in rows:
        if n == 0:
            continue
        result[bucket] = {
            "xOut": round(float(n_out) / n, 4),
            "x1B": round(float(n_1b) / n, 4),
            "x2B": round(float(n_2b) / n, 4),
            "x3B": round(float(n_3b) / n, 4),
            "xHR": round(float(n_hr) / n, 4),
            "xErr": round(float(n_err) / n, 4),
            "n": int(n),
        }
    return result


# ── 1B. wOBA / Linear weights ────────────────────────────────────────────────

def _compute_linear_weights(con: duckdb.DuckDBPyConnection, pq: str) -> Dict[str, float]:
    """Compute NCAA-environment linear weights from actual run-scoring data.

    Uses the RunsScored column to compute average runs scored on each PA outcome type,
    then normalizes to a wOBA scale where 0 = out.
    """
    rows = con.execute(f"""
        WITH pa_terminal AS (
            SELECT
                GameID || '_' || CAST(Inning AS VARCHAR) || '_'
                    || CAST(PAofInning AS VARCHAR) || '_' || Batter AS pa_id,
                CASE
                    WHEN KorBB = 'Strikeout' THEN 'Strikeout'
                    WHEN KorBB = 'Walk' THEN 'Walk'
                    WHEN PitchCall = 'HitByPitch' THEN 'HitByPitch'
                    WHEN PitchCall = 'InPlay' AND PlayResult = 'Single' THEN 'Single'
                    WHEN PitchCall = 'InPlay' AND PlayResult = 'Double' THEN 'Double'
                    WHEN PitchCall = 'InPlay' AND PlayResult = 'Triple' THEN 'Triple'
                    WHEN PitchCall = 'InPlay' AND PlayResult = 'HomeRun' THEN 'HomeRun'
                    WHEN PitchCall = 'InPlay' AND PlayResult = 'Error' THEN 'Error'
                    WHEN PitchCall = 'InPlay' AND PlayResult IN ('Out','FieldersChoice','Sacrifice') THEN 'Out'
                    ELSE NULL
                END AS outcome,
                COALESCE(CAST(RunsScored AS DOUBLE), 0) AS runs
            FROM read_parquet('{pq}')
            WHERE PitchCall IS NOT NULL AND PitchCall != 'Undefined'
              AND (PitchCall = 'InPlay' OR KorBB IN ('Strikeout','Walk') OR PitchCall = 'HitByPitch')
        )
        SELECT
            outcome,
            AVG(runs) AS avg_runs,
            COUNT(*) AS n
        FROM pa_terminal
        WHERE outcome IS NOT NULL
        GROUP BY outcome
    """).fetchall()

    runs_by_outcome = {}
    for outcome, avg_runs, n in rows:
        runs_by_outcome[outcome] = (float(avg_runs), int(n))

    # Normalize: wOBA weight = avg_runs(outcome) - avg_runs(out)
    out_runs = runs_by_outcome.get("Out", (0.0, 0))[0]
    k_runs = runs_by_outcome.get("Strikeout", (0.0, 0))[0]
    base_runs = (out_runs + k_runs) / 2.0 if out_runs or k_runs else 0.0

    def _weight(key: str) -> float:
        return round(max(0.0, runs_by_outcome.get(key, (base_runs, 0))[0] - base_runs), 4)

    return {
        "out_w": 0.0,
        "bb_w": _weight("Walk"),
        "hbp_w": _weight("HitByPitch"),
        "single_w": _weight("Single"),
        "double_w": _weight("Double"),
        "triple_w": _weight("Triple"),
        "hr_w": _weight("HomeRun"),
    }


# ── 1C. Steal success rates by pitch velocity class ─────────────────────────

def _compute_steal_rates(con: duckdb.DuckDBPyConnection, pq: str) -> Dict[str, float]:
    """Compute SB success rates grouped by pitch velocity bucket."""
    rows = con.execute(f"""
        WITH steals AS (
            SELECT
                RelSpeed,
                CASE
                    WHEN RelSpeed < 78 THEN 'slow'
                    WHEN RelSpeed < 85 THEN 'med'
                    WHEN RelSpeed < 90 THEN 'fast'
                    ELSE 'elite'
                END AS velo_class,
                CASE
                    WHEN PlayResult = 'StolenBase' OR
                         (PlayResult IS NOT NULL AND PlayResult LIKE '%StolenBase%') THEN 'SB'
                    WHEN PlayResult = 'CaughtStealing' OR
                         (PlayResult IS NOT NULL AND PlayResult LIKE '%CaughtStealing%') THEN 'CS'
                    ELSE NULL
                END AS steal_result
            FROM read_parquet('{pq}')
            WHERE RelSpeed IS NOT NULL
        )
        SELECT
            velo_class,
            SUM(CASE WHEN steal_result = 'SB' THEN 1 ELSE 0 END) AS n_sb,
            SUM(CASE WHEN steal_result IN ('SB','CS') THEN 1 ELSE 0 END) AS n_att,
            COUNT(CASE WHEN steal_result IN ('SB','CS') THEN 1 END) AS n_steal_plays
        FROM steals
        WHERE steal_result IS NOT NULL
        GROUP BY velo_class
        ORDER BY velo_class
    """).fetchall()

    rates = {}
    for velo_class, n_sb, n_att, _ in rows:
        if n_att > 0:
            rates[velo_class] = round(float(n_sb) / float(n_att) * 100.0, 2)

    return {
        "slow_sb_pct": rates.get("slow", 80.7),
        "med_sb_pct": rates.get("med", 79.2),
        "fast_sb_pct": rates.get("fast", 76.1),
        "elite_sb_pct": rates.get("elite", 71.5),
    }


# ── 1D. GB% and DP% by pitch type ───────────────────────────────────────────

def _compute_gb_dp_rates(con: duckdb.DuckDBPyConnection, pq: str) -> Dict[str, Dict[str, float]]:
    """Compute GB% and DP% by pitch type for calibrating base state adjustments."""
    rows = con.execute(f"""
        WITH bip AS (
            SELECT
                {_PT_NORM} AS pitch_type,
                Angle,
                PlayResult,
                PitchCall
            FROM read_parquet('{pq}')
            WHERE PitchCall = 'InPlay'
              AND PlayResult IS NOT NULL
              AND ({_PT_NORM}) IS NOT NULL
        )
        SELECT
            pitch_type,
            COUNT(*) AS n_bip,
            SUM(CASE WHEN Angle IS NOT NULL AND Angle < 10 THEN 1 ELSE 0 END) AS n_gb,
            SUM(CASE WHEN PlayResult IN ('Out','FieldersChoice','Sacrifice') THEN 1 ELSE 0 END) AS n_out,
            SUM(CASE WHEN PlayResult = 'FieldersChoice'
                      OR (PlayResult = 'Out' AND Angle IS NOT NULL AND Angle < 10)
                  THEN 1 ELSE 0 END) AS n_gb_out,
            SUM(CASE WHEN PlayResult = 'FieldersChoice' THEN 1 ELSE 0 END) AS n_dp_proxy
        FROM bip
        GROUP BY pitch_type
        HAVING COUNT(*) >= 50
    """).fetchall()

    result = {}
    total_bip = 0
    total_gb = 0
    total_dp = 0
    for pitch_type, n_bip, n_gb, n_out, n_gb_out, n_dp_proxy in rows:
        gb_pct = round(float(n_gb) / float(n_bip) * 100.0, 2) if n_bip > 0 else 0.0
        # DP proxy: fielder's choice rate as DP indicator (best available in Trackman)
        dp_pct = round(float(n_dp_proxy) / float(n_bip) * 100.0, 2) if n_bip > 0 else 0.0
        out_pct = round(float(n_out) / float(n_bip) * 100.0, 2) if n_bip > 0 else 0.0
        result[pitch_type] = {
            "gb_pct": gb_pct,
            "dp_pct": dp_pct,
            "out_pct": out_pct,
            "n_bip": int(n_bip),
        }
        total_bip += n_bip
        total_gb += n_gb
        total_dp += n_dp_proxy

    # League averages
    if total_bip > 0:
        result["_league_avg_gb"] = round(float(total_gb) / float(total_bip) * 100.0, 2)
        result["_league_avg_dp"] = round(float(total_dp) / float(total_bip) * 100.0, 2)
    else:
        result["_league_avg_gb"] = 47.0
        result["_league_avg_dp"] = 4.1

    return result


# ── 1E. Pitch metric normalization ranges ────────────────────────────────────

def _compute_metric_ranges(con: duckdb.DuckDBPyConnection, pq: str) -> Dict[str, float]:
    """Compute actual D1 P5/P95 for whiff%, CSW%, chase% (per-pitcher)."""
    rows = con.execute(f"""
        WITH pitcher_metrics AS (
            SELECT
                Pitcher,
                COUNT(*) AS n_pitches,
                SUM(CASE WHEN PitchCall IN {_SWING_CALLS} THEN 1 ELSE 0 END) AS n_swings,
                SUM(CASE WHEN PitchCall = 'StrikeSwinging' THEN 1 ELSE 0 END) AS n_whiffs,
                SUM(CASE WHEN PitchCall IN ('StrikeCalled','StrikeSwinging') THEN 1 ELSE 0 END) AS n_csw,
                SUM(CASE WHEN {_OUT_ZONE} THEN 1 ELSE 0 END) AS n_oz,
                SUM(CASE WHEN {_OUT_ZONE} AND PitchCall IN {_SWING_CALLS} THEN 1 ELSE 0 END) AS n_oz_swing
            FROM read_parquet('{pq}')
            WHERE PitchCall IS NOT NULL AND PitchCall != 'Undefined'
            GROUP BY Pitcher
            HAVING COUNT(*) >= 100
        )
        SELECT
            CASE WHEN n_swings > 0 THEN n_whiffs * 100.0 / n_swings ELSE NULL END AS whiff_pct,
            CASE WHEN n_pitches > 0 THEN n_csw * 100.0 / n_pitches ELSE NULL END AS csw_pct,
            CASE WHEN n_oz > 0 THEN n_oz_swing * 100.0 / n_oz ELSE NULL END AS chase_pct
        FROM pitcher_metrics
    """).fetchdf()

    if rows.empty:
        return {
            "whiff_p5": 12.0, "whiff_p95": 40.0,
            "csw_p5": 18.0, "csw_p95": 35.0,
            "chase_p5": 10.0, "chase_p95": 35.0,
        }

    def _pctile(series, p):
        clean = series.dropna()
        if clean.empty:
            return float("nan")
        return float(np.percentile(clean, p))

    return {
        "whiff_p5": round(_pctile(rows["whiff_pct"], 5), 2),
        "whiff_p95": round(_pctile(rows["whiff_pct"], 95), 2),
        "csw_p5": round(_pctile(rows["csw_pct"], 5), 2),
        "csw_p95": round(_pctile(rows["csw_pct"], 95), 2),
        "chase_p5": round(_pctile(rows["chase_pct"], 5), 2),
        "chase_p95": round(_pctile(rows["chase_pct"], 95), 2),
    }


# ── 1F. D1 fielding benchmarks ──────────────────────────────────────────────

def _compute_fielding_benchmarks(con: duckdb.DuckDBPyConnection, pq: str) -> Dict[str, float]:
    """Compute overall FLD% and error rate distribution from parquet."""
    rows = con.execute(f"""
        WITH bip AS (
            SELECT
                PlayResult
            FROM read_parquet('{pq}')
            WHERE PitchCall = 'InPlay' AND PlayResult IS NOT NULL
        )
        SELECT
            COUNT(*) AS n_bip,
            SUM(CASE WHEN PlayResult = 'Error' THEN 1 ELSE 0 END) AS n_errors,
            SUM(CASE WHEN PlayResult IN ('Out','FieldersChoice','Sacrifice') THEN 1 ELSE 0 END) AS n_outs
        FROM bip
    """).fetchall()

    if not rows or rows[0][0] == 0:
        return {
            "fld_pct_median": 0.9727,
            "fld_pct_p75": 0.980,
            "fld_pct_p90": 0.990,
            "error_rate_median": 0.0273,
            "n_bip": 0,
        }

    n_bip, n_errors, n_outs = rows[0]
    n_bip = int(n_bip)
    n_errors = int(n_errors)
    n_outs = int(n_outs)
    # Defensive plays = outs + errors (roughly = chances)
    n_chances = n_outs + n_errors
    overall_fld_pct = float(n_outs) / float(n_chances) if n_chances > 0 else 0.97
    error_rate = float(n_errors) / float(n_chances) if n_chances > 0 else 0.03

    # Per-game fielding to get distribution — use overall as proxy for median
    # (individual fielder data comes from CSV, not parquet)
    return {
        "fld_pct_median": round(overall_fld_pct, 4),
        "fld_pct_p75": round(min(overall_fld_pct + 0.008, 0.995), 4),
        "fld_pct_p90": round(min(overall_fld_pct + 0.018, 0.999), 4),
        "error_rate_median": round(error_rate, 4),
        "n_bip": n_bip,
    }


# ── Main computation ─────────────────────────────────────────────────────────

def compute_historical_calibration(
    parquet_path: str = PARQUET_PATH,
) -> HistoricalCalibration:
    """Compute all historical calibration metrics from Trackman parquet."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    pq = parquet_path.replace("'", "''")
    con = duckdb.connect(database=":memory:")

    outcome_probs = _compute_outcome_probs(con, pq)
    linear_weights = _compute_linear_weights(con, pq)
    steal_rates = _compute_steal_rates(con, pq)
    gb_dp_rates = _compute_gb_dp_rates(con, pq)
    metric_ranges = _compute_metric_ranges(con, pq)
    fielding_benchmarks = _compute_fielding_benchmarks(con, pq)

    con.close()

    return HistoricalCalibration(
        outcome_probs=outcome_probs,
        linear_weights=linear_weights,
        steal_rates=steal_rates,
        gb_dp_rates=gb_dp_rates,
        metric_ranges=metric_ranges,
        fielding_benchmarks=fielding_benchmarks,
    )


# ── Cache load/save ──────────────────────────────────────────────────────────

def _serialize(cal: HistoricalCalibration) -> dict:
    return {
        "outcome_probs": cal.outcome_probs,
        "linear_weights": cal.linear_weights,
        "steal_rates": cal.steal_rates,
        "gb_dp_rates": cal.gb_dp_rates,
        "metric_ranges": cal.metric_ranges,
        "fielding_benchmarks": cal.fielding_benchmarks,
    }


def _deserialize(blob: dict) -> HistoricalCalibration:
    return HistoricalCalibration(
        outcome_probs=blob["outcome_probs"],
        linear_weights=blob["linear_weights"],
        steal_rates=blob["steal_rates"],
        gb_dp_rates=blob["gb_dp_rates"],
        metric_ranges=blob["metric_ranges"],
        fielding_benchmarks=blob["fielding_benchmarks"],
    )


def load_historical_calibration(
    parquet_path: str = PARQUET_PATH, force_refresh: bool = False,
) -> Optional[HistoricalCalibration]:
    """Load historical calibration from cache, recomputing if parquet changed.

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
                return _deserialize(blob["calibration"])
        except Exception:
            pass

    cal = compute_historical_calibration(parquet_path=parquet_path)
    try:
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(
                {"fingerprint": fp, "calibration": _serialize(cal)},
                f,
                indent=2,
            )
    except Exception:
        pass
    return cal


# ── Fallback functions (preserve current hardcoded behavior) ─────────────────

def fallback_outcome_probs() -> Dict[str, Dict[str, float]]:
    """Return the current hardcoded outcome probabilities from expected.py."""
    return {
        "Barrel":  {"xOut": 0.30, "x1B": 0.10, "x2B": 0.22, "x3B": 0.05, "xHR": 0.33, "xErr": 0.00, "n": 0},
        "HiEV_FB": {"xOut": 0.45, "x1B": 0.05, "x2B": 0.18, "x3B": 0.05, "xHR": 0.27, "xErr": 0.00, "n": 0},
        "Hard_LD": {"xOut": 0.28, "x1B": 0.47, "x2B": 0.20, "x3B": 0.03, "xHR": 0.02, "xErr": 0.00, "n": 0},
        "GB":      {"xOut": 0.76, "x1B": 0.22, "x2B": 0.02, "x3B": 0.00, "xHR": 0.00, "xErr": 0.00, "n": 0},
        "Popup":   {"xOut": 0.95, "x1B": 0.03, "x2B": 0.01, "x3B": 0.00, "xHR": 0.01, "xErr": 0.00, "n": 0},
        "Soft":    {"xOut": 0.90, "x1B": 0.08, "x2B": 0.02, "x3B": 0.00, "xHR": 0.00, "xErr": 0.00, "n": 0},
        "Medium":  {"xOut": 0.68, "x1B": 0.22, "x2B": 0.07, "x3B": 0.01, "xHR": 0.02, "xErr": 0.00, "n": 0},
    }


def fallback_linear_weights() -> Dict[str, float]:
    """Return the current hardcoded wOBA weights."""
    return {
        "out_w": 0.00,
        "bb_w": 0.32,
        "hbp_w": 0.32,
        "single_w": 0.47,
        "double_w": 0.78,
        "triple_w": 1.05,
        "hr_w": 1.40,
    }


def fallback_steal_rates() -> Dict[str, float]:
    """Return the current hardcoded steal rates (from audit data)."""
    return {
        "slow_sb_pct": 80.7,
        "med_sb_pct": 79.2,
        "fast_sb_pct": 76.1,
        "elite_sb_pct": 71.5,
    }


def fallback_gb_dp_rates() -> Dict[str, Dict[str, float]]:
    """Return hardcoded GB/DP rates from audit data."""
    return {
        "Sinker":       {"gb_pct": 57.5, "dp_pct": 4.99, "out_pct": 60.0, "n_bip": 0},
        "Changeup":     {"gb_pct": 50.2, "dp_pct": 4.26, "out_pct": 58.0, "n_bip": 0},
        "Curveball":    {"gb_pct": 49.7, "dp_pct": 4.38, "out_pct": 62.0, "n_bip": 0},
        "Cutter":       {"gb_pct": 47.6, "dp_pct": 4.11, "out_pct": 59.0, "n_bip": 0},
        "Fastball":     {"gb_pct": 44.2, "dp_pct": 3.86, "out_pct": 57.0, "n_bip": 0},
        "Slider":       {"gb_pct": 44.0, "dp_pct": 3.80, "out_pct": 61.0, "n_bip": 0},
        "Splitter":     {"gb_pct": 52.0, "dp_pct": 4.50, "out_pct": 59.0, "n_bip": 0},
        "Knuckle Curve": {"gb_pct": 49.0, "dp_pct": 4.30, "out_pct": 62.0, "n_bip": 0},
        "Sweeper":      {"gb_pct": 43.0, "dp_pct": 3.70, "out_pct": 60.0, "n_bip": 0},
        "_league_avg_gb": 47.0,
        "_league_avg_dp": 4.1,
    }


def fallback_metric_ranges() -> Dict[str, float]:
    """Return the current hardcoded normalization ranges from pitch_call.py."""
    return {
        "whiff_p5": 12.0, "whiff_p95": 40.0,
        "csw_p5": 18.0, "csw_p95": 35.0,
        "chase_p5": 10.0, "chase_p95": 35.0,
    }


def fallback_fielding_benchmarks() -> Dict[str, float]:
    """Return hardcoded fielding benchmarks."""
    return {
        "fld_pct_median": 0.9727,
        "fld_pct_p75": 0.980,
        "fld_pct_p90": 0.990,
        "error_rate_median": 0.0273,
        "n_bip": 0,
    }
