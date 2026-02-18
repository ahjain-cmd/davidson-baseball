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

# Position clamp ranges: (x_min, x_max, y_min, y_max)
_POS_CLAMPS = {
    "SS": (-80, 0, 95, 145),
    "2B": (0, 80, 95, 145),
    "3B": (-90, -30, 55, 100),
    "1B": (30, 90, 55, 100),
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
    warnings: List[str] = field(default_factory=list)


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
               COUNT(*) AS n
        FROM trackman
        WHERE TaggedPitchType IS NOT NULL AND TaggedPitchType NOT IN ('Other','Undefined','Knuckleball')
          AND InducedVertBreak IS NOT NULL AND HorzBreak IS NOT NULL AND RelSpeed IS NOT NULL
          AND PitcherThrows IS NOT NULL
        GROUP BY TaggedPitchType HAVING COUNT(*) >= 100
    """
    return query_population(sql)


@st.cache_data(show_spinner=False)
def load_gb_movement_coefs() -> pd.DataFrame:
    """Load regression coefficients for movement-adjusted GB spray."""
    if _precompute_table_exists("gb_spray_movement_coefs"):
        return _read_precompute_table("gb_spray_movement_coefs")
    return pd.DataFrame()


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
        season_clause = f"AND Season IN ({','.join(str(int(s)) for s in season_filter)})"

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
        result[pt] = {col: (float(row[col]) if pd.notna(row[col]) else None) for col in df.columns if col != "TaggedPitchType"}
    return result


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
        season_clause = f"AND Season IN ({','.join(str(int(s)) for s in season_filter)})"

    sql = f"""
        SELECT TaggedPitchType,
               AVG(RelSpeed) AS velo,
               AVG(InducedVertBreak) AS ivb,
               AVG(CASE WHEN PitcherThrows IN ('Left','L') THEN -HorzBreak ELSE HorzBreak END) AS hb,
               AVG(SpinRate) AS spin,
               AVG(Extension) AS ext,
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
    metrics = [
        "mean_direction", "pull_pct", "center_pct", "oppo_pct",
        "ss_zone_mean_x", "ss_zone_mean_y", "b2_zone_mean_x", "b2_zone_mean_y",
        "pull_zone_mean_x", "pull_zone_mean_y", "oppo_zone_mean_x", "oppo_zone_mean_y",
    ]
    for m in metrics:
        obs = _safe_float(batter_spray.get(m))
        prior = _safe_float(pop_baseline.get(m))
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

    direction_shift = damping * (hb_coef * hb_dev + ivb_coef * ivb_dev)
    Clamped to +/- MAX_MOVEMENT_SHIFT_DEG. Shifts centroids laterally.
    """
    result = dict(shrunk_spray)

    hb_coef = _safe_float(coefs.get("hb_coef"))
    ivb_coef = _safe_float(coefs.get("ivb_coef"))
    if hb_coef is None or ivb_coef is None:
        result["movement_shift_deg"] = 0.0
        return result

    pitcher_hb = _safe_float(pitcher_movement.get("hb"))
    pitcher_ivb = _safe_float(pitcher_movement.get("ivb"))
    pop_hb = _safe_float(pop_movement_row.get("pop_hb_mean"))
    pop_ivb = _safe_float(pop_movement_row.get("pop_ivb_mean"))

    if any(v is None for v in [pitcher_hb, pitcher_ivb, pop_hb, pop_ivb]):
        result["movement_shift_deg"] = 0.0
        return result

    hb_dev = pitcher_hb - pop_hb
    ivb_dev = pitcher_ivb - pop_ivb

    raw_shift = hb_coef * hb_dev + ivb_coef * ivb_dev
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

    return f"({x:.0f}, {y:.0f})"


# ──────────────────────────────────────────────
# WEIGHTED POSITION COMPUTATION
# ──────────────────────────────────────────────

def _compute_weighted_fielder_positions(
    adjusted_sprays: Dict[str, dict],
    pitch_mix: Dict[str, float],
    batter_side: str,
    confidence: str,
) -> Dict[str, FielderPosition]:
    """Compute weighted-average fielder positions from per-PT adjusted sprays.

    Weights come from pitcher's pitch mix (usage_pct).
    """
    # Normalize mix weights to sum to 1
    total_weight = sum(pitch_mix.get(pt, 0) for pt in adjusted_sprays)
    if total_weight <= 0:
        total_weight = len(adjusted_sprays) or 1
        pitch_mix = {pt: 1.0 / total_weight for pt in adjusted_sprays}

    # Weighted average of zone centroids
    ss_x, ss_y = 0.0, 0.0
    b2_x, b2_y = 0.0, 0.0
    pull_x, pull_y = 0.0, 0.0
    oppo_x, oppo_y = 0.0, 0.0
    w_sum = 0.0

    for pt, spray in adjusted_sprays.items():
        w = pitch_mix.get(pt, 0) / total_weight if total_weight > 0 else 0
        if w <= 0:
            continue
        w_sum += w

        ss_x += w * (spray.get("ss_zone_mean_x") or _DEFAULT_POS["SS"][0])
        ss_y += w * (spray.get("ss_zone_mean_y") or _DEFAULT_POS["SS"][1])
        b2_x += w * (spray.get("b2_zone_mean_x") or _DEFAULT_POS["2B"][0])
        b2_y += w * (spray.get("b2_zone_mean_y") or _DEFAULT_POS["2B"][1])
        pull_x += w * (spray.get("pull_zone_mean_x") or (_DEFAULT_POS["3B"][0] if batter_side == "Right" else _DEFAULT_POS["1B"][0]))
        pull_y += w * (spray.get("pull_zone_mean_y") or (_DEFAULT_POS["3B"][1] if batter_side == "Right" else _DEFAULT_POS["1B"][1]))
        oppo_x += w * (spray.get("oppo_zone_mean_x") or (_DEFAULT_POS["1B"][0] if batter_side == "Right" else _DEFAULT_POS["3B"][0]))
        oppo_y += w * (spray.get("oppo_zone_mean_y") or (_DEFAULT_POS["1B"][1] if batter_side == "Right" else _DEFAULT_POS["3B"][1]))

    if w_sum <= 0:
        w_sum = 1.0
    ss_x /= w_sum
    ss_y /= w_sum
    b2_x /= w_sum
    b2_y /= w_sum
    pull_x /= w_sum
    pull_y /= w_sum
    oppo_x /= w_sum
    oppo_y /= w_sum

    # Enforce: SS left of 2B (x < 2B x)
    if ss_x >= b2_x:
        mid = (ss_x + b2_x) / 2
        ss_x = mid - 5
        b2_x = mid + 5

    # Assign corners by batter side
    if batter_side == "Right":
        b3_x, b3_y = pull_x, pull_y
        b1_x, b1_y = oppo_x, oppo_y
    else:
        b3_x, b3_y = oppo_x, oppo_y
        b1_x, b1_y = pull_x, pull_y

    # Enforce: corners shallower than middle infield
    mid_depth = min(ss_y, b2_y)
    b3_y = min(b3_y, mid_depth - 10)
    b1_y = min(b1_y, mid_depth - 10)

    positions = {
        "SS": _build_fielder_position("SS", ss_x, ss_y, batter_side, confidence),
        "2B": _build_fielder_position("2B", b2_x, b2_y, batter_side, confidence),
        "3B": _build_fielder_position("3B", b3_x, b3_y, batter_side, confidence),
        "1B": _build_fielder_position("1B", b1_x, b1_y, batter_side, confidence),
    }
    return positions


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
        pop_baselines[key] = {col: (float(row[col]) if pd.notna(row[col]) else None) for col in pop_baselines_df.columns}

    pop_movement_df = load_pitch_movement_baselines()
    pop_movement = {}
    for _, row in pop_movement_df.iterrows():
        pop_movement[row["TaggedPitchType"]] = {col: (float(row[col]) if pd.notna(row[col]) else None) for col in pop_movement_df.columns}

    coefs_df = load_gb_movement_coefs()
    coefs_dict = {}
    for _, row in coefs_df.iterrows():
        key = (row.get("BatterSide"), row.get("TaggedPitchType"))
        coefs_dict[key] = {col: (float(row[col]) if pd.notna(row[col]) else None) for col in coefs_df.columns}

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

    for pt in pitch_types:
        pop_key = (batter_side, pt)
        pop_bl = pop_baselines.get(pop_key, {})
        batter_pt = batter_spray.get(pt, {})

        # Shrink
        shrunk = shrink_gb_spray(batter_pt, pop_bl)

        # Movement adjustment
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

    # Step 5: Weighted positions
    total_n = sum(s.get("n_gb", 0) or 0 for s in [batter_spray.get(pt, {}) for pt in pitch_types])
    conf = confidence_tier(total_n, low=15, high=40)

    positions = _compute_weighted_fielder_positions(
        adjusted_sprays, pitch_mix, batter_side, conf
    ) if adjusted_sprays else {}

    # Step 6: Overall percentages (weighted by mix)
    overall_pull, overall_center, overall_oppo = 0.0, 0.0, 0.0
    mix_total = sum(pitch_mix.get(pt, 0) for pt in adjusted_sprays)
    for pt, spray in adjusted_sprays.items():
        w = pitch_mix.get(pt, 0) / mix_total if mix_total > 0 else 0
        overall_pull += w * (spray.get("pull_pct") or 33.3)
        overall_center += w * (spray.get("center_pct") or 33.3)
        overall_oppo += w * (spray.get("oppo_pct") or 33.3)

    # Estimate GB% for shift classification (use total batter data or default)
    gb_pct_est = 45.0  # default — GBs only model, so assume GB-heavy context

    shift = classify_shift(
        pull_pct=overall_pull,
        center_pct=overall_center,
        oppo_pct=overall_oppo,
        gb_pct=gb_pct_est,
        gb_pull_pct=overall_pull,
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
        warnings=warnings,
    )
