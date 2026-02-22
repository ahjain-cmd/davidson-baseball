"""Pitcher-Batter Ground Ball Positioning Model.

Combines a pitcher's movement profile with a batter's spray chart to produce
precise infield positioning recommendations, expressed as feet from 2B bag
with natural-language instructions.

Key concepts:
  - Bayesian shrinkage blends batter-specific spray with population baselines
  - Movement adjustment: more arm-side run → more pull-side GBs (small but real)
  - Weighted average across pitcher's pitch mix → final fielder positions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from data.loader import (
    query_population,
    _precompute_table_exists,
    _read_precompute_table,
)
from decision_engine.core.shrinkage import shrink_value, confidence_tier
from decision_engine.recommenders.defense_recommender import classify_shift

# ML toggle — set to True to use XGBoost GB direction model for movement adjustment
USE_ML_GB = False


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

_2B_BAG = (0.0, 127.3)
_1B_BAG = (63.6, 63.6)
_3B_BAG = (-63.6, 63.6)

_DIR_THRESHOLD = 15.0  # degrees for Pull/Center/Oppo classification
_SHRINKAGE_PRIOR_EQUIV = 30  # equivalent GBs for prior weight
_MOVEMENT_DAMPING = 0.7  # apply 70% of regression coefficient
_MAX_MOVEMENT_SHIFT_DEG = 8.0  # clamp movement adjustment
_DEG_TO_FEET = 1.9  # 1 degree ≈ 1.9 feet lateral at infield depth (~110 ft)

# Angle-based positioning constants
# Empirical shift rates from FHC data (Davidson vs UNCG 2/17):
#   Budzik (49% pull, +13% dev): SS shifted 0.8ft, 2B shifted 18.7ft
#   2B rate ≈ 1.4 ft/%, SS rate ≈ 0.06 ft/%
# Using calibrated values that produce 1-5 step differentiation:
_SS_SHIFT_RATE = 0.3         # SS barely moves (empirical: ~0.06, using 0.3 for practical min adjustment)
_2B_SHIFT_RATE = 1.2         # 2B does the heavy shifting (empirical: ~1.4)
_CORNER_SHIFT_DAMPING = 0.5  # corners shift at 50% of 2B rate
_GB_DEPTH_RATE = 0.15        # feet shallower per 1% GB rate above average (44%)
_MAX_DEPTH_ADJ = 8.0         # max depth adjustment in feet
_POP_AVG_PULL = {             # fallback if baselines unavailable
    "Right": 48.3,
    "Left": 49.5,
}

# Per-pitch-type movement → pull% rates (OLS regression on 493K GBs from trackman view).
# HB is arm-side-normalized (positive = arm-side run for all pitcher hands).
# Rates: % change in pull rate per unit of pitcher deviation from population mean.
# Tuple order: (hb_rate_per_inch, ivb_rate_per_inch, velo_rate_per_mph, rel_height_rate_per_ft, rel_side_rate_per_ft)
# Positive rate = more of that metric → higher pull%.
_MOVEMENT_PULL_RATES = {
    # RHB: n=71K FB, 65K SI, 61K SL, 69K CH, 43K CB, 6K CT
    ("Right", "Fastball"):  (-0.23, -0.24, +0.58, -1.41, +0.33),
    ("Right", "Sinker"):    (-0.18, +0.09, +0.14, -2.85, +0.81),
    ("Right", "Slider"):    (-0.24, +0.34, +0.49, -1.32, -0.31),
    ("Right", "Changeup"):  (-0.14, +0.20, +0.33, -3.12, +0.28),
    ("Right", "Curveball"): (-0.23, +0.15, +0.30, -0.48, -0.37),
    ("Right", "Cutter"):    (-0.43, +1.49, +2.58, -3.58, +0.83),
    # LHB: n=44K FB, 39K SI, 24K SL, 47K CH, 18K CB, 3K CT
    ("Left", "Fastball"):   (-0.20, -0.06, -0.29, +0.85, -0.42),
    ("Left", "Sinker"):     (+0.17, -0.22, -0.70, +2.53, -0.55),
    ("Left", "Slider"):     (-0.04, +0.17, +0.45, +1.19, -0.80),
    ("Left", "Changeup"):   (-0.09, -0.29, +0.14, +2.97, +0.07),
    ("Left", "Curveball"):  (+0.06, -0.16, +0.02, -0.19, -1.14),
    ("Left", "Cutter"):     (-0.26, +0.33, +0.99, -2.90, -2.58),
}
_MAX_MOVEMENT_PULL_ADJ = 8.0  # cap total movement pull% adjustment

# Position clamp ranges: (x_min, x_max, y_min, y_max)
_POS_CLAMPS = {
    "SS": (-80, 0, 95, 145),
    "2B": (0, 80, 95, 145),
    "3B": (-90, -30, 55, 100),
    "1B": (30, 90, 55, 100),
    "LF": (-260, -80, 180, 310),
    "CF": (-120, 120, 230, 350),
    "RF": (80, 260, 180, 310),
}

# Default fallback positions
_DEFAULT_POS = {
    "SS": (-40, 120),
    "2B": (40, 120),
    "3B": (-60, 80),
    "1B": (60, 80),
    "LF": (-180, 220),
    "CF": (0, 280),
    "RF": (180, 220),
}


# ──────────────────────────────────────────────
# DATACLASSES
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class FielderPosition:
    pos_name: str
    x: float
    y: float
    depth_ft: float  # distance from home plate
    lateral_from_2b_ft: float  # signed: negative = 3B side, positive = 1B side
    description: str  # natural-language instruction
    coverage_min_deg: float
    coverage_max_deg: float
    confidence: str  # "High", "Medium", "Low"


@dataclass
class MatchupPositioning:
    positions: Dict[str, FielderPosition]
    pitch_mix_weights: Dict[str, float]
    per_pitch_type: Dict[str, dict]  # PT → {n_batter, pull_pct, center_pct, oppo_pct, movement_shift_deg, source, ...}
    shift_type: Dict[str, str]  # from classify_shift()
    overall_pull_pct: float
    overall_center_pct: float
    overall_oppo_pct: float
    confidence: str
    gb_pct: float = 44.0  # actual batter GB%
    warnings: List[str] = field(default_factory=list)
    movement_detail: dict = field(default_factory=dict)  # per-pitch HB/IVB/Velo adjustments


# ──────────────────────────────────────────────
# DATA LOADING (cached)
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_gb_spray_baselines() -> pd.DataFrame:
    """Load population GB spray baselines per (BatterSide, PitchType)."""
    if _precompute_table_exists("gb_spray_baselines"):
        return _read_precompute_table("gb_spray_baselines")
    # Live fallback
    sql = """
        WITH gb AS (
            SELECT BatterSide, TaggedPitchType, Direction, Distance,
                   Distance * SIN(RADIANS(Direction)) AS x,
                   Distance * COS(RADIANS(Direction)) AS y,
                   PlayResult
            FROM trackman
            WHERE PitchCall = 'InPlay' AND TaggedHitType = 'GroundBall'
              AND Direction IS NOT NULL AND Distance IS NOT NULL
              AND BatterSide IN ('Left','Right') AND TaggedPitchType IS NOT NULL
        ),
        classified AS (
            SELECT *,
                CASE
                    WHEN BatterSide='Right' AND Direction < -15 THEN 'Pull'
                    WHEN BatterSide='Right' AND Direction > 15  THEN 'Oppo'
                    WHEN BatterSide='Left'  AND Direction > 15  THEN 'Pull'
                    WHEN BatterSide='Left'  AND Direction < -15 THEN 'Oppo'
                    ELSE 'Center'
                END AS FieldDir,
                CASE WHEN x < 0 THEN 'SS_side' ELSE '2B_side' END AS MiddleSide
            FROM gb
        )
        SELECT BatterSide, TaggedPitchType,
               COUNT(*) AS n_gb, AVG(Direction) AS mean_direction, STDDEV(Direction) AS std_direction,
               AVG(x) AS mean_x, AVG(y) AS mean_y,
               SUM(CASE WHEN FieldDir='Pull' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS gb_pull_pct,
               SUM(CASE WHEN FieldDir='Center' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS gb_center_pct,
               SUM(CASE WHEN FieldDir='Oppo' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS gb_oppo_pct,
               SUM(CASE WHEN PlayResult IN ('Out','Sacrifice','FieldersChoice') THEN 1 ELSE 0 END)*100.0/COUNT(*) AS out_rate,
               AVG(CASE WHEN MiddleSide='SS_side' THEN x END) AS ss_zone_mean_x,
               AVG(CASE WHEN MiddleSide='SS_side' THEN y END) AS ss_zone_mean_y,
               SUM(CASE WHEN MiddleSide='SS_side' THEN 1 ELSE 0 END) AS ss_zone_n,
               AVG(CASE WHEN MiddleSide='2B_side' THEN x END) AS b2_zone_mean_x,
               AVG(CASE WHEN MiddleSide='2B_side' THEN y END) AS b2_zone_mean_y,
               SUM(CASE WHEN MiddleSide='2B_side' THEN 1 ELSE 0 END) AS b2_zone_n,
               AVG(CASE WHEN FieldDir='Pull' THEN x END) AS pull_zone_mean_x,
               AVG(CASE WHEN FieldDir='Pull' THEN y END) AS pull_zone_mean_y,
               SUM(CASE WHEN FieldDir='Pull' THEN 1 ELSE 0 END) AS pull_zone_n,
               AVG(CASE WHEN FieldDir='Oppo' THEN x END) AS oppo_zone_mean_x,
               AVG(CASE WHEN FieldDir='Oppo' THEN y END) AS oppo_zone_mean_y,
               SUM(CASE WHEN FieldDir='Oppo' THEN 1 ELSE 0 END) AS oppo_zone_n
        FROM classified
        GROUP BY BatterSide, TaggedPitchType HAVING COUNT(*) >= 20
    """
    return query_population(sql)


@st.cache_data(show_spinner=False)
def load_pitch_movement_baselines() -> pd.DataFrame:
    """Load population movement baselines per PitchType."""
    if _precompute_table_exists("pitch_movement_baselines"):
        return _read_precompute_table("pitch_movement_baselines")
    sql = """
        SELECT TaggedPitchType,
               AVG(InducedVertBreak) AS pop_ivb_mean, STDDEV(InducedVertBreak) AS pop_ivb_std,
               AVG(CASE WHEN PitcherThrows IN ('Left','L') THEN -HorzBreak ELSE HorzBreak END) AS pop_hb_mean,
               STDDEV(CASE WHEN PitcherThrows IN ('Left','L') THEN -HorzBreak ELSE HorzBreak END) AS pop_hb_std,
               AVG(RelSpeed) AS pop_velo_mean, STDDEV(RelSpeed) AS pop_velo_std,
               AVG(RelHeight) AS pop_rel_height_mean, STDDEV(RelHeight) AS pop_rel_height_std,
               AVG(CASE WHEN PitcherThrows IN ('Left','L') THEN -RelSide ELSE RelSide END) AS pop_rel_side_mean,
               STDDEV(CASE WHEN PitcherThrows IN ('Left','L') THEN -RelSide ELSE RelSide END) AS pop_rel_side_std,
               COUNT(*) AS n
        FROM trackman
        WHERE TaggedPitchType IS NOT NULL AND TaggedPitchType NOT IN ('Other','Undefined','Knuckleball')
          AND InducedVertBreak IS NOT NULL AND HorzBreak IS NOT NULL AND RelSpeed IS NOT NULL
          AND RelHeight IS NOT NULL AND RelSide IS NOT NULL
          AND PitcherThrows IS NOT NULL
        GROUP BY TaggedPitchType HAVING COUNT(*) >= 100
    """
    return query_population(sql)


@st.cache_data(show_spinner=False)
def load_gb_movement_coefs() -> pd.DataFrame:
    """Load regression coefficients for movement-adjusted GB spray."""
    if _precompute_table_exists("gb_spray_movement_coefs"):
        return _read_precompute_table("gb_spray_movement_coefs")
    # Live fallback: compute regression coefficients from parquet
    return _compute_gb_movement_coefs_live()


def _compute_gb_movement_coefs_live() -> pd.DataFrame:
    """Compute movement → GB direction regression coefficients at runtime.

    Direction ~ hb_deviation + ivb_deviation + velo_deviation  (per BatterSide × PitchType).
    Mirrors _create_gb_spray_baselines() in precompute.py.
    """
    # First get population movement means
    pop_mov = load_pitch_movement_baselines()
    if pop_mov.empty:
        return pd.DataFrame()
    mov_means = {}
    for _, row in pop_mov.iterrows():
        pt = row.get("TaggedPitchType")
        hb_m = row.get("pop_hb_mean")
        ivb_m = row.get("pop_ivb_mean")
        velo_m = row.get("pop_velo_mean")
        if pt and pd.notna(hb_m) and pd.notna(ivb_m):
            mov_means[pt] = {
                "hb_mean": float(hb_m),
                "ivb_mean": float(ivb_m),
                "velo_mean": float(velo_m) if pd.notna(velo_m) else None,
            }

    if not mov_means:
        return pd.DataFrame()

    # Query GB pitches joined with per-pitcher movement averages
    sql = """
        WITH pitcher_avgs AS (
            SELECT
                Pitcher,
                TaggedPitchType,
                AVG(CASE WHEN PitcherThrows IN ('Left','L') THEN -HorzBreak ELSE HorzBreak END) AS pitcher_hb,
                AVG(InducedVertBreak) AS pitcher_ivb,
                AVG(RelSpeed) AS pitcher_velo,
                COUNT(*) AS pitcher_n
            FROM trackman
            WHERE TaggedPitchType IS NOT NULL
              AND HorzBreak IS NOT NULL
              AND InducedVertBreak IS NOT NULL
              AND RelSpeed IS NOT NULL
              AND PitcherThrows IS NOT NULL
            GROUP BY Pitcher, TaggedPitchType
            HAVING COUNT(*) >= 10
        )
        SELECT
            t.BatterSide,
            t.TaggedPitchType,
            t.Direction,
            pa.pitcher_hb,
            pa.pitcher_ivb,
            pa.pitcher_velo
        FROM trackman t
        JOIN pitcher_avgs pa ON t.Pitcher = pa.Pitcher AND t.TaggedPitchType = pa.TaggedPitchType
        WHERE t.PitchCall = 'InPlay'
          AND t.TaggedHitType = 'GroundBall'
          AND t.Direction IS NOT NULL
          AND t.BatterSide IN ('Left', 'Right')
          AND t.TaggedPitchType IS NOT NULL
    """
    gb_mov_df = query_population(sql)
    if gb_mov_df.empty:
        return pd.DataFrame()

    coef_rows = []
    for (bside, pt), grp in gb_mov_df.groupby(["BatterSide", "TaggedPitchType"]):
        if len(grp) < 50 or pt not in mov_means:
            continue
        pop = mov_means[pt]
        hb_dev = grp["pitcher_hb"].values - pop["hb_mean"]
        ivb_dev = grp["pitcher_ivb"].values - pop["ivb_mean"]
        velo_dev = grp["pitcher_velo"].values - pop["velo_mean"] if pop["velo_mean"] is not None else np.zeros(len(grp))
        direction = grp["Direction"].values

        A = np.column_stack([hb_dev, ivb_dev, velo_dev, np.ones(len(grp))])
        try:
            result, _, _, _ = np.linalg.lstsq(A, direction, rcond=None)
        except np.linalg.LinAlgError:
            continue
        hb_coef, ivb_coef, velo_coef, intercept = result

        ss_res = np.sum((direction - A @ result) ** 2)
        ss_tot = np.sum((direction - direction.mean()) ** 2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        coef_rows.append({
            "BatterSide": bside,
            "TaggedPitchType": pt,
            "hb_coef": float(hb_coef),
            "ivb_coef": float(ivb_coef),
            "velo_coef": float(velo_coef),
            "intercept": float(intercept),
            "r_squared": float(r_sq),
            "n": len(grp),
        })

    if not coef_rows:
        return pd.DataFrame()
    return pd.DataFrame(coef_rows)


# ──────────────────────────────────────────────
# BATTER BATTED-BALL PROFILE
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=600)
def get_batter_batted_ball_profile(
    batter: str, season_filter: Optional[Tuple[int, ...]] = None
) -> dict:
    """Query batter's batted-ball profile: GB%, FB%, LD%, pull/oppo splits on air balls.

    Returns dict with keys: n_bip, gb_pct, fb_pct, ld_pct, pu_pct,
    air_pull_pct, air_center_pct, air_oppo_pct, batter_side,
    air_pull_mean_x/y, air_center_mean_x/y, air_oppo_mean_x/y.
    """
    batter_esc = batter.replace("'", "''")
    season_clause = ""
    if season_filter:
        season_clause = f"AND EXTRACT(YEAR FROM CAST(\"Date\" AS DATE)) IN ({','.join(str(int(s)) for s in season_filter)})"

    sql = f"""
        WITH bip AS (
            SELECT
                TaggedHitType, Direction, Distance, BatterSide,
                Distance * SIN(RADIANS(Direction)) AS x,
                Distance * COS(RADIANS(Direction)) AS y
            FROM trackman
            WHERE Batter = '{batter_esc}'
              AND PitchCall = 'InPlay'
              AND TaggedHitType IS NOT NULL
              {season_clause}
        )
        SELECT
            COUNT(*) AS n_bip,
            SUM(CASE WHEN TaggedHitType='GroundBall' THEN 1 ELSE 0 END)*100.0/NULLIF(COUNT(*),0) AS gb_pct,
            SUM(CASE WHEN TaggedHitType='FlyBall' THEN 1 ELSE 0 END)*100.0/NULLIF(COUNT(*),0) AS fb_pct,
            SUM(CASE WHEN TaggedHitType='LineDrive' THEN 1 ELSE 0 END)*100.0/NULLIF(COUNT(*),0) AS ld_pct,
            SUM(CASE WHEN TaggedHitType='Popup' THEN 1 ELSE 0 END)*100.0/NULLIF(COUNT(*),0) AS pu_pct,
            -- Air ball (FB+LD) spray distribution
            SUM(CASE WHEN TaggedHitType IN ('FlyBall','LineDrive') AND Direction IS NOT NULL THEN 1 ELSE 0 END) AS n_air,
            -- For RHB: Direction<-15=Pull; for LHB: Direction>15=Pull (handled in Python)
            AVG(CASE WHEN TaggedHitType IN ('FlyBall','LineDrive') AND Direction IS NOT NULL THEN Direction END) AS air_mean_dir,
            -- Air ball centroids by x-sign
            AVG(CASE WHEN TaggedHitType IN ('FlyBall','LineDrive') AND x < -50 THEN x END) AS air_lf_mean_x,
            AVG(CASE WHEN TaggedHitType IN ('FlyBall','LineDrive') AND x < -50 THEN y END) AS air_lf_mean_y,
            SUM(CASE WHEN TaggedHitType IN ('FlyBall','LineDrive') AND x < -50 THEN 1 ELSE 0 END) AS air_lf_n,
            AVG(CASE WHEN TaggedHitType IN ('FlyBall','LineDrive') AND ABS(x) <= 50 THEN x END) AS air_cf_mean_x,
            AVG(CASE WHEN TaggedHitType IN ('FlyBall','LineDrive') AND ABS(x) <= 50 THEN y END) AS air_cf_mean_y,
            SUM(CASE WHEN TaggedHitType IN ('FlyBall','LineDrive') AND ABS(x) <= 50 THEN 1 ELSE 0 END) AS air_cf_n,
            AVG(CASE WHEN TaggedHitType IN ('FlyBall','LineDrive') AND x > 50 THEN x END) AS air_rf_mean_x,
            AVG(CASE WHEN TaggedHitType IN ('FlyBall','LineDrive') AND x > 50 THEN y END) AS air_rf_mean_y,
            SUM(CASE WHEN TaggedHitType IN ('FlyBall','LineDrive') AND x > 50 THEN 1 ELSE 0 END) AS air_rf_n,
            -- BatterSide mode
            MODE(BatterSide) AS batter_side
        FROM bip
    """
    df = query_population(sql)
    if df.empty:
        return {}
    row = df.iloc[0]
    return _safe_row_to_dict(row, df.columns)


# ──────────────────────────────────────────────
# BATTER SPRAY QUERY
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=600)
def get_batter_gb_spray_by_pitch_type(
    batter: str, batter_side: str, season_filter: Optional[Tuple[int, ...]] = None
) -> Dict[str, dict]:
    """Query batter's ground ball spray broken down by pitch type faced.

    Returns dict: PitchType → {n_gb, mean_direction, pull_pct, center_pct, oppo_pct,
                                 ss_zone_mean_x/y, b2_zone_mean_x/y, pull_zone_mean_x/y, oppo_zone_mean_x/y}
    """
    batter_esc = batter.replace("'", "''")
    season_clause = ""
    if season_filter:
        season_clause = f"AND EXTRACT(YEAR FROM CAST(\"Date\" AS DATE)) IN ({','.join(str(int(s)) for s in season_filter)})"

    pull_case = (
        "CASE WHEN Direction < -15 THEN 'Pull' WHEN Direction > 15 THEN 'Oppo' ELSE 'Center' END"
        if batter_side == "Right"
        else "CASE WHEN Direction > 15 THEN 'Pull' WHEN Direction < -15 THEN 'Oppo' ELSE 'Center' END"
    )

    sql = f"""
        WITH gb AS (
            SELECT TaggedPitchType, Direction, Distance,
                   Distance * SIN(RADIANS(Direction)) AS x,
                   Distance * COS(RADIANS(Direction)) AS y,
                   {pull_case} AS FieldDir,
                   CASE WHEN Distance * SIN(RADIANS(Direction)) < 0 THEN 'SS_side' ELSE '2B_side' END AS MiddleSide
            FROM trackman
            WHERE Batter = '{batter_esc}'
              AND PitchCall = 'InPlay' AND TaggedHitType = 'GroundBall'
              AND Direction IS NOT NULL AND Distance IS NOT NULL
              AND TaggedPitchType IS NOT NULL
              {season_clause}
        )
        SELECT TaggedPitchType,
               COUNT(*) AS n_gb,
               AVG(Direction) AS mean_direction,
               SUM(CASE WHEN FieldDir='Pull' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pull_pct,
               SUM(CASE WHEN FieldDir='Center' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS center_pct,
               SUM(CASE WHEN FieldDir='Oppo' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS oppo_pct,
               AVG(CASE WHEN MiddleSide='SS_side' THEN x END) AS ss_zone_mean_x,
               AVG(CASE WHEN MiddleSide='SS_side' THEN y END) AS ss_zone_mean_y,
               AVG(CASE WHEN MiddleSide='2B_side' THEN x END) AS b2_zone_mean_x,
               AVG(CASE WHEN MiddleSide='2B_side' THEN y END) AS b2_zone_mean_y,
               AVG(CASE WHEN FieldDir='Pull' THEN x END) AS pull_zone_mean_x,
               AVG(CASE WHEN FieldDir='Pull' THEN y END) AS pull_zone_mean_y,
               AVG(CASE WHEN FieldDir='Oppo' THEN x END) AS oppo_zone_mean_x,
               AVG(CASE WHEN FieldDir='Oppo' THEN y END) AS oppo_zone_mean_y
        FROM gb
        GROUP BY TaggedPitchType
    """
    df = query_population(sql)
    if df.empty:
        return {}
    result = {}
    for _, row in df.iterrows():
        pt = row["TaggedPitchType"]
        result[pt] = {col: v for col, v in _safe_row_to_dict(row, df.columns).items() if col != "TaggedPitchType"}
    return result


# ──────────────────────────────────────────────
# BATTER AIR BALL SPRAY (for outfielder positioning)
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=600)
def get_batter_air_spray(
    batter: str, batter_side: str, season_filter: Optional[Tuple[int, ...]] = None
) -> dict:
    """Query batter's fly ball + line drive spray for outfield positioning.

    Returns dict: {lf_mean_x, lf_mean_y, cf_mean_x, cf_mean_y,
                   rf_mean_x, rf_mean_y, n_air, pull_pct, center_pct, oppo_pct}.
    """
    batter_esc = batter.replace("'", "''")
    season_clause = ""
    if season_filter:
        season_clause = f"AND EXTRACT(YEAR FROM CAST(\"Date\" AS DATE)) IN ({','.join(str(int(s)) for s in season_filter)})"

    pull_case = (
        "CASE WHEN Direction < -15 THEN 'Pull' WHEN Direction > 15 THEN 'Oppo' ELSE 'Center' END"
        if batter_side == "Right"
        else "CASE WHEN Direction > 15 THEN 'Pull' WHEN Direction < -15 THEN 'Oppo' ELSE 'Center' END"
    )

    sql = f"""
        WITH air AS (
            SELECT Direction, Distance,
                   Distance * SIN(RADIANS(Direction)) AS x,
                   Distance * COS(RADIANS(Direction)) AS y,
                   {pull_case} AS FieldDir
            FROM trackman
            WHERE Batter = '{batter_esc}'
              AND PitchCall = 'InPlay'
              AND TaggedHitType IN ('FlyBall', 'LineDrive')
              AND Direction IS NOT NULL AND Distance IS NOT NULL
              {season_clause}
        )
        SELECT
            COUNT(*) AS n_air,
            SUM(CASE WHEN FieldDir='Pull' THEN 1 ELSE 0 END)*100.0/NULLIF(COUNT(*),0) AS pull_pct,
            SUM(CASE WHEN FieldDir='Center' THEN 1 ELSE 0 END)*100.0/NULLIF(COUNT(*),0) AS center_pct,
            SUM(CASE WHEN FieldDir='Oppo' THEN 1 ELSE 0 END)*100.0/NULLIF(COUNT(*),0) AS oppo_pct,
            -- LF zone (x < -50)
            AVG(CASE WHEN x < -50 THEN x END) AS lf_mean_x,
            AVG(CASE WHEN x < -50 THEN y END) AS lf_mean_y,
            SUM(CASE WHEN x < -50 THEN 1 ELSE 0 END) AS lf_n,
            -- CF zone (|x| <= 50)
            AVG(CASE WHEN ABS(x) <= 50 THEN x END) AS cf_mean_x,
            AVG(CASE WHEN ABS(x) <= 50 THEN y END) AS cf_mean_y,
            SUM(CASE WHEN ABS(x) <= 50 THEN 1 ELSE 0 END) AS cf_n,
            -- RF zone (x > 50)
            AVG(CASE WHEN x > 50 THEN x END) AS rf_mean_x,
            AVG(CASE WHEN x > 50 THEN y END) AS rf_mean_y,
            SUM(CASE WHEN x > 50 THEN 1 ELSE 0 END) AS rf_n
        FROM air
    """
    df = query_population(sql)
    if df.empty:
        return {}
    row = df.iloc[0]
    return _safe_row_to_dict(row, df.columns)


# ──────────────────────────────────────────────
# PITCHER MOVEMENT PROFILE
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=600)
def get_pitcher_movement_profile(
    pitcher: str, season_filter: Optional[Tuple[int, ...]] = None
) -> Dict[str, dict]:
    """Query pitcher's per-pitch-type movement profile.

    Returns dict: PitchType → {velo, ivb, hb, spin, ext, n, usage_pct}
    """
    pitcher_esc = pitcher.replace("'", "''")
    season_clause = ""
    if season_filter:
        season_clause = f"AND EXTRACT(YEAR FROM CAST(\"Date\" AS DATE)) IN ({','.join(str(int(s)) for s in season_filter)})"

    sql = f"""
        SELECT TaggedPitchType,
               AVG(RelSpeed) AS velo,
               AVG(InducedVertBreak) AS ivb,
               AVG(CASE WHEN PitcherThrows IN ('Left','L') THEN -HorzBreak ELSE HorzBreak END) AS hb,
               AVG(SpinRate) AS spin,
               AVG(Extension) AS ext,
               AVG(RelHeight) AS rel_height,
               AVG(CASE WHEN PitcherThrows IN ('Left','L') THEN -RelSide ELSE RelSide END) AS rel_side,
               COUNT(*) AS n
        FROM trackman
        WHERE Pitcher = '{pitcher_esc}'
          AND TaggedPitchType IS NOT NULL
          AND TaggedPitchType NOT IN ('Other','Undefined','Knuckleball')
          {season_clause}
        GROUP BY TaggedPitchType
    """
    df = query_population(sql)
    if df.empty:
        return {}

    total = df["n"].sum()
    result = {}
    for _, row in df.iterrows():
        pt = row["TaggedPitchType"]
        result[pt] = {
            "velo": float(row["velo"]) if pd.notna(row["velo"]) else None,
            "ivb": float(row["ivb"]) if pd.notna(row["ivb"]) else None,
            "hb": float(row["hb"]) if pd.notna(row["hb"]) else None,
            "spin": float(row["spin"]) if pd.notna(row["spin"]) else None,
            "ext": float(row["ext"]) if pd.notna(row["ext"]) else None,
            "rel_height": float(row["rel_height"]) if pd.notna(row.get("rel_height")) else None,
            "rel_side": float(row["rel_side"]) if pd.notna(row.get("rel_side")) else None,
            "n": int(row["n"]),
            "usage_pct": float(row["n"]) / total * 100 if total > 0 else 0,
        }
    return result


# ──────────────────────────────────────────────
# SHRINKAGE
# ──────────────────────────────────────────────

def _safe_float(v, default=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def shrink_gb_spray(
    batter_spray: dict,
    pop_baseline: dict,
    n_prior_equiv: int = _SHRINKAGE_PRIOR_EQUIV,
) -> dict:
    """Bayesian shrinkage of batter's GB spray toward population baseline.

    Blends all numeric metrics using shrink_value(). At 30 batter GBs → 50/50;
    at 4 GBs → ~88% population.
    """
    n_obs = _safe_float(batter_spray.get("n_gb"), 0)
    shrunk = {}
    # Map batter spray column names to population baseline column names
    # (batter SQL outputs pull_pct; population SQL outputs gb_pull_pct)
    _POP_COL_MAP = {
        "pull_pct": "gb_pull_pct",
        "center_pct": "gb_center_pct",
        "oppo_pct": "gb_oppo_pct",
    }
    metrics = [
        "mean_direction", "pull_pct", "center_pct", "oppo_pct",
        "ss_zone_mean_x", "ss_zone_mean_y", "b2_zone_mean_x", "b2_zone_mean_y",
        "pull_zone_mean_x", "pull_zone_mean_y", "oppo_zone_mean_x", "oppo_zone_mean_y",
    ]
    for m in metrics:
        obs = _safe_float(batter_spray.get(m))
        pop_key = _POP_COL_MAP.get(m, m)
        prior = _safe_float(pop_baseline.get(pop_key))
        if prior is None:
            prior = _safe_float(pop_baseline.get(m))  # fallback to same name
        shrunk[m] = shrink_value(obs, n_obs, prior, n_prior_equiv=n_prior_equiv)
    shrunk["n_gb"] = n_obs
    shrunk["batter_weight"] = n_obs / (n_obs + n_prior_equiv) if n_obs is not None else 0.0
    shrunk["pop_weight"] = 1.0 - shrunk["batter_weight"]
    return shrunk


# ──────────────────────────────────────────────
# MOVEMENT ADJUSTMENT
# ──────────────────────────────────────────────

def compute_movement_adjusted_spray(
    shrunk_spray: dict,
    pitcher_movement: dict,
    pop_movement_row: dict,
    coefs: dict,
    damping: float = _MOVEMENT_DAMPING,
) -> dict:
    """Apply movement regression to shift spray direction.

    direction_shift = damping * (hb_coef * hb_dev + ivb_coef * ivb_dev + velo_coef * velo_dev)
    Clamped to +/- MAX_MOVEMENT_SHIFT_DEG. Shifts centroids laterally.
    """
    result = dict(shrunk_spray)

    hb_coef = _safe_float(coefs.get("hb_coef"))
    ivb_coef = _safe_float(coefs.get("ivb_coef"))
    velo_coef = _safe_float(coefs.get("velo_coef"), 0.0)
    if hb_coef is None or ivb_coef is None:
        result["movement_shift_deg"] = 0.0
        return result

    pitcher_hb = _safe_float(pitcher_movement.get("hb"))
    pitcher_ivb = _safe_float(pitcher_movement.get("ivb"))
    pitcher_velo = _safe_float(pitcher_movement.get("velo"))
    pop_hb = _safe_float(pop_movement_row.get("pop_hb_mean"))
    pop_ivb = _safe_float(pop_movement_row.get("pop_ivb_mean"))
    pop_velo = _safe_float(pop_movement_row.get("pop_velo_mean"))

    if any(v is None for v in [pitcher_hb, pitcher_ivb, pop_hb, pop_ivb]):
        result["movement_shift_deg"] = 0.0
        return result

    hb_dev = pitcher_hb - pop_hb
    ivb_dev = pitcher_ivb - pop_ivb
    velo_dev = (pitcher_velo - pop_velo) if (pitcher_velo is not None and pop_velo is not None) else 0.0

    raw_shift = hb_coef * hb_dev + ivb_coef * ivb_dev + velo_coef * velo_dev
    shift_deg = np.clip(damping * raw_shift, -_MAX_MOVEMENT_SHIFT_DEG, _MAX_MOVEMENT_SHIFT_DEG)
    result["movement_shift_deg"] = float(shift_deg)

    # Shift mean direction
    if result.get("mean_direction") is not None:
        result["mean_direction"] = result["mean_direction"] + shift_deg

    # Shift centroids laterally: 1 degree ≈ 1.9 feet at infield depth
    lateral_shift_ft = shift_deg * _DEG_TO_FEET
    for key in ["ss_zone_mean_x", "b2_zone_mean_x", "pull_zone_mean_x", "oppo_zone_mean_x"]:
        if result.get(key) is not None:
            result[key] = result[key] + lateral_shift_ft

    return result


# ──────────────────────────────────────────────
# FIELDER POSITION BUILDER
# ──────────────────────────────────────────────

def _build_fielder_position(
    pos_name: str, x: float, y: float, batter_side: str, confidence: str
) -> FielderPosition:
    """Build a FielderPosition with clamping, distance calcs, and natural-language description."""
    # Clamp to valid ranges
    if pos_name in _POS_CLAMPS:
        x_min, x_max, y_min, y_max = _POS_CLAMPS[pos_name]
        x = float(np.clip(x, x_min, x_max))
        y = float(np.clip(y, y_min, y_max))

    # Enforce SS left of 2B
    # (handled at the composite level, but safety here)

    depth_ft = float(np.sqrt(x**2 + y**2))
    lateral_from_2b = x - _2B_BAG[0]  # x offset from 2B bag (negative = 3B side)

    # Coverage angle range
    coverage_half = 12.0  # degrees
    center_angle = float(np.degrees(np.arctan2(x, y))) if y > 0 else 0.0
    cov_min = center_angle - coverage_half
    cov_max = center_angle + coverage_half

    # Natural-language description
    desc = _generate_position_description(pos_name, x, y, lateral_from_2b, depth_ft, batter_side)

    return FielderPosition(
        pos_name=pos_name,
        x=round(x, 1),
        y=round(y, 1),
        depth_ft=round(depth_ft, 1),
        lateral_from_2b_ft=round(lateral_from_2b, 1),
        description=desc,
        coverage_min_deg=round(cov_min, 1),
        coverage_max_deg=round(cov_max, 1),
        confidence=confidence,
    )


def _generate_position_description(
    pos_name: str, x: float, y: float, lateral_from_2b: float,
    depth_ft: float, batter_side: str
) -> str:
    """Generate coach-friendly natural-language positioning instruction."""
    abs_lat = abs(lateral_from_2b)
    lat_dir = "3B-side" if lateral_from_2b < 0 else "1B-side"

    if pos_name in ("SS", "2B"):
        # Depth relative to standard (~120 ft)
        depth_diff = y - 120.0
        depth_str = ""
        if abs(depth_diff) > 5:
            depth_str = f", {abs(depth_diff):.0f}ft {'deeper' if depth_diff > 0 else 'shallower'}"
        elif abs(depth_diff) <= 5:
            depth_str = ", standard depth"

        if abs_lat < 5:
            return f"At the bag{depth_str}"
        return f"{abs_lat:.0f}ft to {lat_dir} of bag{depth_str}"

    elif pos_name == "3B":
        # Reference from 3B bag
        dist_from_3b = float(np.sqrt((x - _3B_BAG[0])**2 + (y - _3B_BAG[1])**2))
        if dist_from_3b < 8:
            return "Standard position near bag"
        # Direction from 3B bag
        dx = x - _3B_BAG[0]
        if dx > 5:
            return f"Shaded {abs(dx):.0f}ft toward SS hole"
        elif y < _3B_BAG[1] - 5:
            return f"Guarding {abs(y - _3B_BAG[1]):.0f}ft down the line"
        return f"{dist_from_3b:.0f}ft from bag, depth {y:.0f}ft"

    elif pos_name == "1B":
        dist_from_1b = float(np.sqrt((x - _1B_BAG[0])**2 + (y - _1B_BAG[1])**2))
        if dist_from_1b < 8:
            return "Standard position near bag"
        dx = x - _1B_BAG[0]
        if dx < -5:
            return f"Shaded {abs(dx):.0f}ft toward 2B hole"
        elif y < _1B_BAG[1] - 5:
            return f"Guarding {abs(y - _1B_BAG[1]):.0f}ft down the line"
        return f"{dist_from_1b:.0f}ft from bag, depth {y:.0f}ft"

    elif pos_name in ("LF", "CF", "RF"):
        # Outfield: depth from home and lateral offset
        standard_depth = {"LF": 220, "CF": 280, "RF": 220}
        standard_x = {"LF": -160, "CF": 0, "RF": 160}
        std_d = standard_depth.get(pos_name, 250)
        std_x = standard_x.get(pos_name, 0)
        depth_diff = y - std_d
        lat_diff = x - std_x

        parts = []
        if abs(depth_diff) > 10:
            parts.append(f"{abs(depth_diff):.0f}ft {'deeper' if depth_diff > 0 else 'shallower'}")
        else:
            parts.append("standard depth")

        if abs(lat_diff) > 10:
            lat_dir = "toward CF" if (pos_name == "LF" and lat_diff > 0) or (pos_name == "RF" and lat_diff < 0) else "toward line"
            if pos_name == "CF":
                lat_dir = "toward LF" if lat_diff < 0 else "toward RF"
            parts.append(f"shaded {abs(lat_diff):.0f}ft {lat_dir}")

        return ", ".join(parts) if parts else "Standard position"

    return f"({x:.0f}, {y:.0f})"


# ──────────────────────────────────────────────
# WEIGHTED POSITION COMPUTATION
# ──────────────────────────────────────────────

def _compute_weighted_fielder_positions(
    adjusted_sprays: Dict[str, dict],
    pitch_mix: Dict[str, float],
    batter_side: str,
    confidence: str,
    air_spray: Optional[dict] = None,
    pop_baselines: Optional[Dict[Tuple[str, str], dict]] = None,
    gb_pct: float = 44.0,
    pitcher_profile: Optional[Dict[str, dict]] = None,
    pop_movement: Optional[Dict[str, dict]] = None,
) -> Tuple[Dict[str, FielderPosition], dict]:
    """Compute fielder positions using deviation-from-standard positioning.

    Instead of using raw GB zone centroids (which cluster near home plate),
    computes lateral shifts from standard positions driven by:
    1. Pull% deviation — how much this batter pulls vs population average
    2. Movement pull% adjustment — per-pitch-type HB/IVB/Velo rates
    3. GB% depth adjustment — higher GB% batters → play slightly shallower

    Weights come from pitcher's pitch mix (usage_pct).

    Returns (positions_dict, movement_detail) where movement_detail contains
    the per-pitch and aggregate movement pull% adjustments for display.
    """
    # Normalize mix weights to sum to 1
    total_weight = sum(pitch_mix.get(pt, 0) for pt in adjusted_sprays)
    if total_weight <= 0:
        total_weight = len(adjusted_sprays) or 1
        pitch_mix = {pt: 1.0 / total_weight for pt in adjusted_sprays}

    # Step 1: Weighted pull% across pitch mix
    weighted_pull = 0.0
    w_sum = 0.0

    for pt, spray in adjusted_sprays.items():
        w = pitch_mix.get(pt, 0) / total_weight if total_weight > 0 else 0
        if w <= 0:
            continue
        w_sum += w
        pp = spray.get("pull_pct")
        weighted_pull += w * (pp if pp is not None else 33.3)

    if w_sum > 0:
        weighted_pull /= w_sum

    # Step 2: Movement pull% adjustment (HB + IVB + Velo + RelHeight + RelSide rates per pitch type)
    movement_pull_adj = 0.0
    movement_detail = {"per_pitch": {}, "total_adj": 0.0}
    if pitcher_profile and pop_movement:
        mvt_w_sum = 0.0
        for pt in adjusted_sprays:
            w = pitch_mix.get(pt, 0) / total_weight if total_weight > 0 else 0
            if w <= 0:
                continue
            rates = _MOVEMENT_PULL_RATES.get((batter_side, pt))
            if not rates:
                continue
            # Support both 3-tuple (legacy) and 5-tuple (with release metrics)
            if len(rates) == 5:
                hb_rate, ivb_rate, velo_rate, rh_rate, rs_rate = rates
            else:
                hb_rate, ivb_rate, velo_rate = rates
                rh_rate, rs_rate = 0.0, 0.0
            p_mov = pitcher_profile.get(pt, {})
            pop_mov = pop_movement.get(pt, {})
            p_hb = _safe_float(p_mov.get("hb"))
            p_ivb = _safe_float(p_mov.get("ivb"))
            p_velo = _safe_float(p_mov.get("velo"))
            p_rh = _safe_float(p_mov.get("rel_height"))
            p_rs = _safe_float(p_mov.get("rel_side"))
            pop_hb = _safe_float(pop_mov.get("pop_hb_mean"))
            pop_ivb = _safe_float(pop_mov.get("pop_ivb_mean"))
            pop_velo = _safe_float(pop_mov.get("pop_velo_mean"))
            pop_rh = _safe_float(pop_mov.get("pop_rel_height_mean"))
            pop_rs = _safe_float(pop_mov.get("pop_rel_side_mean"))

            hb_dev = (p_hb - pop_hb) if (p_hb is not None and pop_hb is not None) else 0.0
            ivb_dev = (p_ivb - pop_ivb) if (p_ivb is not None and pop_ivb is not None) else 0.0
            velo_dev = (p_velo - pop_velo) if (p_velo is not None and pop_velo is not None) else 0.0
            rh_dev = (p_rh - pop_rh) if (p_rh is not None and pop_rh is not None) else 0.0
            rs_dev = (p_rs - pop_rs) if (p_rs is not None and pop_rs is not None) else 0.0

            pt_adj = (hb_rate * hb_dev + ivb_rate * ivb_dev + velo_rate * velo_dev
                      + rh_rate * rh_dev + rs_rate * rs_dev)
            movement_pull_adj += w * pt_adj
            mvt_w_sum += w

            movement_detail["per_pitch"][pt] = {
                "hb_dev": round(hb_dev, 1), "ivb_dev": round(ivb_dev, 1),
                "velo_dev": round(velo_dev, 1),
                "rh_dev": round(rh_dev, 2), "rs_dev": round(rs_dev, 2),
                "hb_contrib": round(hb_rate * hb_dev, 2),
                "ivb_contrib": round(ivb_rate * ivb_dev, 2),
                "velo_contrib": round(velo_rate * velo_dev, 2),
                "rh_contrib": round(rh_rate * rh_dev, 2),
                "rs_contrib": round(rs_rate * rs_dev, 2),
                "pt_adj": round(pt_adj, 2), "mix_w": round(w, 3),
            }

        if mvt_w_sum > 0:
            movement_pull_adj /= mvt_w_sum
        movement_pull_adj = float(np.clip(movement_pull_adj, -_MAX_MOVEMENT_PULL_ADJ, _MAX_MOVEMENT_PULL_ADJ))

    movement_detail["total_adj"] = round(movement_pull_adj, 2)

    # Apply movement adjustment to pull%
    adjusted_pull = weighted_pull + movement_pull_adj

    # Step 3: Population average pull (weighted by same pitch mix)
    pop_pull = 0.0
    pop_w_sum = 0.0
    if pop_baselines:
        for pt in adjusted_sprays:
            w = pitch_mix.get(pt, 0) / total_weight if total_weight > 0 else 0
            if w <= 0:
                continue
            pop_key = (batter_side, pt)
            pop_bl = pop_baselines.get(pop_key, {})
            pop_pt_pull = _safe_float(pop_bl.get("gb_pull_pct")) or _safe_float(pop_bl.get("pull_pct"))
            if pop_pt_pull is not None:
                pop_pull += w * pop_pt_pull
                pop_w_sum += w
    if pop_w_sum > 0:
        pop_pull /= pop_w_sum
    else:
        pop_pull = _POP_AVG_PULL.get(batter_side, 38.0)

    # Step 4: Lateral shift computation
    pull_dev = adjusted_pull - pop_pull  # positive = more pull than average
    pull_sign = -1.0 if batter_side == "Right" else 1.0  # RHB pull = 3B side (negative x)

    # Primary mover swaps by batter side (empirical from FHC):
    #   RHB pull → 2B shifts heavy (6-8 steps), SS barely moves
    #   LHB pull → SS shifts heavy (8-10 steps), 2B barely moves
    base_signed_dev = pull_sign * pull_dev
    if batter_side == "Right":
        ss_shift = base_signed_dev * _SS_SHIFT_RATE   # 0.3 (SS barely moves for RHB)
        b2_shift = base_signed_dev * _2B_SHIFT_RATE   # 1.2 (2B is primary mover for RHB)
    else:
        ss_shift = base_signed_dev * _2B_SHIFT_RATE   # 1.2 (SS is primary mover for LHB)
        b2_shift = base_signed_dev * _SS_SHIFT_RATE   # 0.3 (2B barely moves for LHB)
    corner_shift = base_signed_dev * _2B_SHIFT_RATE * _CORNER_SHIFT_DAMPING

    # Depth adjustment: higher GB% → play shallower
    depth_adj = float(np.clip((gb_pct - 44.0) * _GB_DEPTH_RATE, -_MAX_DEPTH_ADJ, _MAX_DEPTH_ADJ))

    # Step 5: Apply to standard positions
    ss_x = _DEFAULT_POS["SS"][0] + ss_shift
    ss_y = _DEFAULT_POS["SS"][1] - depth_adj
    b2_x = _DEFAULT_POS["2B"][0] + b2_shift
    b2_y = _DEFAULT_POS["2B"][1] - depth_adj
    b3_x = _DEFAULT_POS["3B"][0] + corner_shift
    b3_y = float(_DEFAULT_POS["3B"][1])
    b1_x = _DEFAULT_POS["1B"][0] + corner_shift
    b1_y = float(_DEFAULT_POS["1B"][1])

    # Step 6: Enforce constraints
    # SS must stay on 3B side of 2B
    if ss_x >= b2_x:
        mid = (ss_x + b2_x) / 2
        ss_x = mid - 5
        b2_x = mid + 5

    # Corners shallower than middle infield
    mid_depth = min(ss_y, b2_y)
    b3_y = min(b3_y, mid_depth - 10)
    b1_y = min(b1_y, mid_depth - 10)

    positions = {
        "SS": _build_fielder_position("SS", ss_x, ss_y, batter_side, confidence),
        "2B": _build_fielder_position("2B", b2_x, b2_y, batter_side, confidence),
        "3B": _build_fielder_position("3B", b3_x, b3_y, batter_side, confidence),
        "1B": _build_fielder_position("1B", b1_x, b1_y, batter_side, confidence),
    }

    # ── Outfield positions from air ball spray data ──────────────────
    if air_spray and (air_spray.get("n_air") or 0) >= 5:
        lf_x = air_spray.get("lf_mean_x") or _DEFAULT_POS["LF"][0]
        lf_y = air_spray.get("lf_mean_y") or _DEFAULT_POS["LF"][1]
        cf_x = air_spray.get("cf_mean_x") or _DEFAULT_POS["CF"][0]
        cf_y = air_spray.get("cf_mean_y") or _DEFAULT_POS["CF"][1]
        rf_x = air_spray.get("rf_mean_x") or _DEFAULT_POS["RF"][0]
        rf_y = air_spray.get("rf_mean_y") or _DEFAULT_POS["RF"][1]

        # Shrink air ball positions toward defaults based on sample size
        n_air = air_spray.get("n_air") or 0
        air_weight = n_air / (n_air + 20)  # 20 equivalent samples for prior
        for pos_name, (raw_x, raw_y), (def_x, def_y) in [
            ("LF", (lf_x, lf_y), _DEFAULT_POS["LF"]),
            ("CF", (cf_x, cf_y), _DEFAULT_POS["CF"]),
            ("RF", (rf_x, rf_y), _DEFAULT_POS["RF"]),
        ]:
            pos_x = air_weight * raw_x + (1 - air_weight) * def_x
            pos_y = air_weight * raw_y + (1 - air_weight) * def_y
            of_conf = confidence_tier(int(n_air), low=10, high=30)
            positions[pos_name] = _build_fielder_position(
                pos_name, pos_x, pos_y, batter_side, of_conf
            )
    else:
        # Default outfield positions
        for pos_name in ("LF", "CF", "RF"):
            positions[pos_name] = _build_fielder_position(
                pos_name, _DEFAULT_POS[pos_name][0], _DEFAULT_POS[pos_name][1],
                batter_side, "Low"
            )

    return positions, movement_detail




def _safe_row_to_dict(row, columns):
    """Convert a DataFrame row to dict, coercing numeric columns to float."""
    result = {}
    for col in columns:
        val = row[col]
        if pd.isna(val) if not isinstance(val, str) else False:
            result[col] = None
        elif isinstance(val, str):
            result[col] = val
        else:
            try:
                result[col] = float(val)
            except (ValueError, TypeError):
                result[col] = val
    return result


# ──────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────

def compute_pitcher_batter_positioning(
    pitcher: str,
    batter: str,
    batter_side: str,
    pitcher_season_filter: Optional[Tuple[int, ...]] = None,
    batter_season_filter: Optional[Tuple[int, ...]] = None,
) -> MatchupPositioning:
    """Orchestrate all steps to produce pitcher-batter matchup positioning.

    1. Query batter GB spray per pitch type
    2. Load population baselines
    3. Get pitcher movement profile and pitch mix
    4. For each pitch type in pitcher's mix: shrink batter spray → apply movement adjustment
    5. Weight-average across pitch types → fielder positions
    6. Classify shift
    7. Return MatchupPositioning
    """
    warnings = []

    # Step 1: Batter spray
    batter_spray = get_batter_gb_spray_by_pitch_type(batter, batter_side, batter_season_filter)
    total_batter_gb = sum(v.get("n_gb", 0) or 0 for v in batter_spray.values())
    if total_batter_gb == 0:
        warnings.append("Batter has 0 ground balls in database. Using pure population baselines.")
    elif total_batter_gb < 10:
        warnings.append(f"Batter has only {total_batter_gb} GBs. Positions heavily regressed to population.")

    # Step 2: Population baselines
    pop_baselines_df = load_gb_spray_baselines()
    pop_baselines = {}
    for _, row in pop_baselines_df.iterrows():
        key = (row.get("BatterSide"), row.get("TaggedPitchType"))
        pop_baselines[key] = _safe_row_to_dict(row, pop_baselines_df.columns)

    pop_movement_df = load_pitch_movement_baselines()
    pop_movement = {}
    for _, row in pop_movement_df.iterrows():
        pop_movement[row["TaggedPitchType"]] = _safe_row_to_dict(row, pop_movement_df.columns)

    coefs_df = load_gb_movement_coefs()
    coefs_dict = {}
    for _, row in coefs_df.iterrows():
        key = (row.get("BatterSide"), row.get("TaggedPitchType"))
        coefs_dict[key] = _safe_row_to_dict(row, coefs_df.columns)

    # Step 3: Pitcher movement profile
    pitcher_profile = get_pitcher_movement_profile(pitcher, pitcher_season_filter)
    pitcher_not_found = len(pitcher_profile) == 0
    if pitcher_not_found:
        warnings.append("Pitcher not in database. Using equal weights across batter's observed pitch types. No movement adjustment.")

    # Determine pitch types and mix
    if pitcher_not_found:
        # Fall back to batter's observed pitch types with equal weight
        if batter_spray:
            pitch_types = list(batter_spray.keys())
        else:
            pitch_types = [k[1] for k in pop_baselines if k[0] == batter_side]
        pitch_mix = {pt: 100.0 / len(pitch_types) for pt in pitch_types} if pitch_types else {}
    else:
        pitch_types = list(pitcher_profile.keys())
        pitch_mix = {pt: pitcher_profile[pt].get("usage_pct", 0) for pt in pitch_types}

    # Step 4: Per-pitch-type shrinkage + movement adjustment
    adjusted_sprays = {}
    per_pt_summary = {}

    # ML GB path: use XGBoost model for spray prediction if enabled
    ml_spray_override = None
    if USE_ML_GB and not pitcher_not_found:
        try:
            from analytics.gb_model import predict_gb_spray
            ml_spray_override = predict_gb_spray(pitcher_profile, batter_side)
        except (FileNotFoundError, ImportError):
            pass  # Fall back to linear model

    for pt in pitch_types:
        pop_key = (batter_side, pt)
        pop_bl = pop_baselines.get(pop_key, {})
        batter_pt = batter_spray.get(pt, {})

        # Shrink
        shrunk = shrink_gb_spray(batter_pt, pop_bl)

        # Movement adjustment — use ML or linear
        if ml_spray_override and pt in ml_spray_override.get("per_pitch", {}):
            # ML path: use ML-predicted spray for this pitch type
            ml_pt = ml_spray_override["per_pitch"][pt]
            adjusted = dict(shrunk)
            # Blend ML spray with shrunk batter spray
            n_obs = shrunk.get("n_gb", 0) or 0
            ml_weight = max(0.3, 1.0 - n_obs / (n_obs + 30))  # ML has more weight when batter data sparse
            for key in ["pull_pct", "center_pct", "oppo_pct"]:
                shrunk_val = adjusted.get(key, 33.3) or 33.3
                ml_val = ml_pt.get(key, shrunk_val)
                adjusted[key] = (1 - ml_weight) * shrunk_val + ml_weight * ml_val
            adjusted["movement_shift_deg"] = 0.0  # ML already incorporates movement
        else:
            # Linear path (original)
            coef = coefs_dict.get(pop_key, {})
            pitcher_mov = pitcher_profile.get(pt, {}) if not pitcher_not_found else {}
            pop_mov = pop_movement.get(pt, {})

            if not pitcher_not_found and coef and pitcher_mov:
                adjusted = compute_movement_adjusted_spray(shrunk, pitcher_mov, pop_mov, coef)
            else:
                adjusted = dict(shrunk)
                adjusted["movement_shift_deg"] = 0.0

        adjusted_sprays[pt] = adjusted

        # Build per-PT summary
        n_batter = int(batter_pt.get("n_gb", 0) or 0)
        source = "batter+pop" if n_batter > 0 else "population"
        if ml_spray_override and pt in ml_spray_override.get("per_pitch", {}):
            source += "+ml"
        per_pt_summary[pt] = {
            "n_batter": n_batter,
            "mix_pct": pitch_mix.get(pt, 0),
            "pull_pct": adjusted.get("pull_pct"),
            "center_pct": adjusted.get("center_pct"),
            "oppo_pct": adjusted.get("oppo_pct"),
            "movement_shift_deg": adjusted.get("movement_shift_deg", 0),
            "batter_weight": adjusted.get("batter_weight", 0),
            "pop_weight": adjusted.get("pop_weight", 1),
            "source": source,
        }

    # Step 5: Weighted positions (infield + outfield)
    total_n = sum(s.get("n_gb", 0) or 0 for s in [batter_spray.get(pt, {}) for pt in pitch_types])
    conf = confidence_tier(total_n, low=15, high=40)

    # Estimate GB% — needed for depth adjustment and shift classification
    batter_profile = get_batter_batted_ball_profile(batter, batter_season_filter)
    gb_pct_est = batter_profile.get("gb_pct") if batter_profile else None
    if gb_pct_est is None or (isinstance(gb_pct_est, float) and np.isnan(gb_pct_est)):
        gb_pct_est = 44.0  # D1 population average fallback

    # Air ball spray for outfield positioning
    air_spray = get_batter_air_spray(batter, batter_side, batter_season_filter)

    if adjusted_sprays:
        positions, movement_detail = _compute_weighted_fielder_positions(
            adjusted_sprays, pitch_mix, batter_side, conf, air_spray=air_spray,
            pop_baselines=pop_baselines, gb_pct=float(gb_pct_est),
            pitcher_profile=pitcher_profile if not pitcher_not_found else None,
            pop_movement=pop_movement if not pitcher_not_found else None,
        )
    else:
        positions = {}
        movement_detail = {"per_pitch": {}, "total_adj": 0.0}

    # Step 6: Overall percentages (weighted by mix)
    overall_pull, overall_center, overall_oppo = 0.0, 0.0, 0.0
    mix_total = sum(pitch_mix.get(pt, 0) for pt in adjusted_sprays)
    for pt, spray in adjusted_sprays.items():
        w = pitch_mix.get(pt, 0) / mix_total if mix_total > 0 else 0
        pp = spray.get("pull_pct")
        cp = spray.get("center_pct")
        op = spray.get("oppo_pct")
        overall_pull += w * (pp if pp is not None else 33.3)
        overall_center += w * (cp if cp is not None else 33.3)
        overall_oppo += w * (op if op is not None else 33.3)

    # Use movement-adjusted pull% for shift classification so the label
    # matches the actual fielder positions (which include movement effect)
    mvt_adj = movement_detail.get("total_adj", 0.0)
    adjusted_overall_pull = overall_pull + mvt_adj
    adjusted_overall_oppo = overall_oppo - mvt_adj  # conserve total

    shift = classify_shift(
        pull_pct=adjusted_overall_pull,
        center_pct=overall_center,
        oppo_pct=adjusted_overall_oppo,
        gb_pct=gb_pct_est,
        gb_pull_pct=adjusted_overall_pull,
    )

    return MatchupPositioning(
        positions=positions,
        pitch_mix_weights=pitch_mix,
        per_pitch_type=per_pt_summary,
        shift_type=shift,
        overall_pull_pct=round(overall_pull, 1),
        overall_center_pct=round(overall_center, 1),
        overall_oppo_pct=round(overall_oppo, 1),
        confidence=conf,
        gb_pct=round(float(gb_pct_est), 1),
        warnings=warnings,
        movement_detail=movement_detail,
    )
