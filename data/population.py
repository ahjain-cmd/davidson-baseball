"""Population-level stat queries via DuckDB — batter/pitcher rankings, stuff baselines, tunnel population."""

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

from config import (
    _name_sql,
    _SWING_CALLS_SQL,
    _CONTACT_CALLS_SQL,
    _HAS_LOC,
    ZONE_HEIGHT_BOT,
    ZONE_HEIGHT_TOP,
    ZONE_SIDE,
    MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE,
    MIN_PITCH_USAGE_PCT,
    PLATE_SIDE_MAX,
    PLATE_HEIGHT_MIN,
    PLATE_HEIGHT_MAX,
)
from data.loader import query_population, query_precompute, _precompute_table_exists, _read_precompute_table

# ── Paths for tunnel caching (mirrors app.py constants) ──────────
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARQUET_FIXED_PATH = os.path.join(_APP_DIR, "all_trackman_fixed.parquet")
PARQUET_PATH = _PARQUET_FIXED_PATH if os.path.exists(_PARQUET_FIXED_PATH) else os.path.join(_APP_DIR, "all_trackman.parquet")
CACHE_DIR = os.path.join(_APP_DIR, ".cache")
TUNNEL_BENCH_PATH = os.path.join(CACHE_DIR, "tunnel_benchmarks.json")
TUNNEL_WEIGHTS_PATH = os.path.join(CACHE_DIR, "tunnel_weights.json")


# ── Private helpers for tunnel benchmark / weight caching ──────────
def _parquet_fingerprint():
    try:
        return {"path": PARQUET_PATH, "mtime": os.path.getmtime(PARQUET_PATH), "size": os.path.getsize(PARQUET_PATH)}
    except OSError:
        return {"path": PARQUET_PATH, "mtime": None, "size": None}


def _load_tunnel_benchmarks():
    if not os.path.exists(TUNNEL_BENCH_PATH):
        return None
    try:
        with open(TUNNEL_BENCH_PATH, "r", encoding="utf-8") as f:
            blob = json.load(f)
        if blob.get("fingerprint") != _parquet_fingerprint():
            return None
        return blob.get("benchmarks")
    except Exception:
        return None


def _save_tunnel_benchmarks(benchmarks):
    os.makedirs(CACHE_DIR, exist_ok=True)
    blob = {"fingerprint": _parquet_fingerprint(), "benchmarks": benchmarks}
    with open(TUNNEL_BENCH_PATH, "w", encoding="utf-8") as f:
        json.dump(blob, f)


def _load_tunnel_weights():
    if not os.path.exists(TUNNEL_WEIGHTS_PATH):
        return None
    try:
        with open(TUNNEL_WEIGHTS_PATH, "r", encoding="utf-8") as f:
            blob = json.load(f)
        if blob.get("fingerprint") != _parquet_fingerprint():
            return None
        return blob.get("weights")
    except Exception:
        return None


def _save_tunnel_weights(weights):
    os.makedirs(CACHE_DIR, exist_ok=True)
    blob = {"fingerprint": _parquet_fingerprint(), "weights": weights}
    with open(TUNNEL_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(blob, f)


def _compute_batter_from_base(df):
    """Compute batter percentiles from pre-aggregated base stats."""
    if df.empty:
        return df
    agg = df.groupby(["Batter", "BatterTeam"]).agg(
        n_pitches=("n_pitches", "sum"),
        PA=("PA", "sum"),
        ks=("ks", "sum"),
        bbs=("bbs", "sum"),
        swings=("swings", "sum"),
        whiffs=("whiffs", "sum"),
        contacts=("contacts", "sum"),
        n_loc=("n_loc", "sum"),
        iz_count=("iz_count", "sum"),
        oz_count=("oz_count", "sum"),
        iz_swings=("iz_swings", "sum"),
        oz_swings=("oz_swings", "sum"),
        iz_contacts=("iz_contacts", "sum"),
        oz_contacts=("oz_contacts", "sum"),
        bbe=("bbe", "sum"),
        hard_hits=("hard_hits", "sum"),
        sweet_spots=("sweet_spots", "sum"),
        Barrels=("Barrels", "sum"),
        gb=("gb", "sum"),
        fb=("fb", "sum"),
        ld=("ld", "sum"),
        pu=("pu", "sum"),
        sum_ev=("sum_ev", "sum"),
        n_ev=("n_ev", "sum"),
        max_ev=("max_ev", "max"),
        sum_la=("sum_la", "sum"),
        sum_dist=("sum_dist", "sum"),
    ).reset_index()

    agg["AvgEV"] = np.where(agg["n_ev"] > 0, agg["sum_ev"] / agg["n_ev"], np.nan)
    agg["MaxEV"] = agg["max_ev"]
    agg["AvgLA"] = np.where(agg["n_ev"] > 0, agg["sum_la"] / agg["n_ev"], np.nan)
    agg["AvgDist"] = np.where(agg["n_ev"] > 0, agg["sum_dist"] / agg["n_ev"], np.nan)

    n = agg["bbe"]
    pa = agg["PA"]
    sw = agg["swings"]
    oz = agg["oz_count"]
    iz = agg["iz_count"]
    iz_sw = agg["iz_swings"]
    oz_sw = agg["oz_swings"]

    agg["HardHitPct"] = np.where(n > 0, agg["hard_hits"] / n * 100, np.nan)
    agg["BarrelPct"] = np.where(n > 0, agg["Barrels"] / n * 100, np.nan)
    agg["BarrelPA"] = np.where(pa > 0, agg["Barrels"] / pa * 100, np.nan)
    agg["SweetSpotPct"] = np.where(n > 0, agg["sweet_spots"] / n * 100, np.nan)
    agg["WhiffPct"] = np.where(sw > 0, agg["whiffs"] / sw * 100, np.nan)
    agg["KPct"] = np.where(pa > 0, agg["ks"] / pa * 100, np.nan)
    agg["BBPct"] = np.where(pa > 0, agg["bbs"] / pa * 100, np.nan)
    agg["ChasePct"] = np.where(oz > 0, agg["oz_swings"] / oz * 100, np.nan)
    agg["ChaseContact"] = np.where(oz_sw > 0, agg["oz_contacts"] / oz_sw * 100, np.nan)
    agg["ZoneSwingPct"] = np.where(iz > 0, iz_sw / iz * 100, np.nan)
    agg["ZoneContactPct"] = np.where(iz_sw > 0, agg["iz_contacts"] / iz_sw * 100, np.nan)
    agg["ZonePct"] = np.where(agg["n_loc"] > 0, agg["iz_count"] / agg["n_loc"] * 100, np.nan)
    agg["SwingPct"] = np.where(agg["n_pitches"] > 0, sw / agg["n_pitches"] * 100, np.nan)
    agg["GBPct"] = np.where(n > 0, agg["gb"] / n * 100, np.nan)
    agg["FBPct"] = np.where(n > 0, agg["fb"] / n * 100, np.nan)
    agg["LDPct"] = np.where(n > 0, agg["ld"] / n * 100, np.nan)
    agg["PUPct"] = np.where(n > 0, agg["pu"] / n * 100, np.nan)
    agg["AirPct"] = np.where(n > 0, (agg["fb"] + agg["ld"] + agg["pu"]) / n * 100, np.nan)
    agg["BBE"] = agg["bbe"]

    agg = agg[agg["PA"] >= 50].copy()
    keep = [
        "Batter", "BatterTeam", "PA", "BBE", "AvgEV", "MaxEV", "HardHitPct",
        "Barrels", "BarrelPct", "BarrelPA", "SweetSpotPct", "AvgLA", "AvgDist",
        "WhiffPct", "KPct", "BBPct", "ChasePct", "ChaseContact",
        "ZoneSwingPct", "ZoneContactPct", "ZonePct", "SwingPct",
        "GBPct", "FBPct", "LDPct", "PUPct", "AirPct",
    ]
    return agg[[c for c in keep if c in agg.columns]]


def _compute_pitcher_from_base(df):
    """Compute pitcher percentiles from pre-aggregated base stats."""
    if df.empty:
        return df
    agg = df.groupby(["Pitcher", "PitcherTeam"]).agg(
        Pitches=("Pitches", "sum"),
        PA=("PA", "sum"),
        ks=("ks", "sum"),
        bbs=("bbs", "sum"),
        swings=("swings", "sum"),
        whiffs=("whiffs", "sum"),
        n_loc=("n_loc", "sum"),
        iz_count=("iz_count", "sum"),
        oz_count=("oz_count", "sum"),
        iz_swings=("iz_swings", "sum"),
        oz_swings=("oz_swings", "sum"),
        iz_contacts=("iz_contacts", "sum"),
        bbe=("bbe", "sum"),
        hard_hits=("hard_hits", "sum"),
        n_barrels=("n_barrels", "sum"),
        gb=("gb", "sum"),
        n_ip=("n_ip", "sum"),
        sum_ev_against=("sum_ev_against", "sum"),
        n_ev_against=("n_ev_against", "sum"),
        sum_fb_velo=("sum_fb_velo", "sum"),
        n_fb_velo=("n_fb_velo", "sum"),
        max_fb_velo=("max_fb_velo", "max"),
        sum_spin=("sum_spin", "sum"),
        n_spin=("n_spin", "sum"),
        sum_ext=("sum_ext", "sum"),
        n_ext=("n_ext", "sum"),
    ).reset_index()

    agg["AvgEVAgainst"] = np.where(agg["n_ev_against"] > 0, agg["sum_ev_against"] / agg["n_ev_against"], np.nan)
    agg["AvgFBVelo"] = np.where(agg["n_fb_velo"] > 0, agg["sum_fb_velo"] / agg["n_fb_velo"], np.nan)
    agg["MaxFBVelo"] = agg["max_fb_velo"]
    agg["AvgSpin"] = np.where(agg["n_spin"] > 0, agg["sum_spin"] / agg["n_spin"], np.nan)
    agg["Extension"] = np.where(agg["n_ext"] > 0, agg["sum_ext"] / agg["n_ext"], np.nan)

    pa = agg["PA"]
    sw = agg["swings"]
    oz = agg["oz_count"]
    iz_sw = agg["iz_swings"]
    n_bat = agg["bbe"]
    n_ip = agg["n_ip"]

    agg["WhiffPct"] = np.where(sw > 0, agg["whiffs"] / sw * 100, np.nan)
    agg["KPct"] = np.where(pa > 0, agg["ks"] / pa * 100, np.nan)
    agg["BBPct"] = np.where(pa > 0, agg["bbs"] / pa * 100, np.nan)
    agg["ZonePct"] = np.where(agg["n_loc"] > 0, agg["iz_count"] / agg["n_loc"] * 100, np.nan)
    agg["ChasePct"] = np.where(oz > 0, agg["oz_swings"] / oz * 100, np.nan)
    agg["ZoneContactPct"] = np.where(iz_sw > 0, agg["iz_contacts"] / iz_sw * 100, np.nan)
    agg["HardHitAgainst"] = np.where(n_bat > 0, agg["hard_hits"] / n_bat * 100, np.nan)
    agg["BarrelPctAgainst"] = np.where(n_bat > 0, agg["n_barrels"] / n_bat * 100, np.nan)
    agg["GBPct"] = np.where(n_ip > 0, agg["gb"] / n_ip * 100, np.nan)
    agg["SwingPct"] = np.where(agg["Pitches"] > 0, sw / agg["Pitches"] * 100, np.nan)

    agg = agg[agg["Pitches"] >= 100].copy()
    keep = [
        "Pitcher", "PitcherTeam", "Pitches", "PA", "AvgFBVelo", "MaxFBVelo",
        "AvgSpin", "Extension", "WhiffPct", "KPct", "BBPct", "ZonePct",
        "ChasePct", "ZoneContactPct", "AvgEVAgainst", "HardHitAgainst",
        "BarrelPctAgainst", "GBPct", "SwingPct",
    ]
    return agg[[c for c in keep if c in agg.columns]]


# ──────────────────────────────────────────────
# Population stat functions
# ──────────────────────────────────────────────

@st.cache_data(show_spinner="Computing batter rankings...")
def compute_batter_stats_pop(season_filter=None, _version=5):
    """Compute batter stats for all D1 batters via DuckDB. Adaptive per-batter zone."""
    if _precompute_table_exists("batter_stats_pop"):
        df = _read_precompute_table("batter_stats_pop")
        if df.empty:
            return df
        if season_filter and "Season" in df.columns:
            seasons = [int(s) for s in season_filter]
            df = df[df["Season"].isin(seasons)]
        return _compute_batter_from_base(df)
    if _precompute_table_exists("trackman_pop"):
        season_clause = ""
        if season_filter:
            seasons_in = ",".join(str(int(s)) for s in season_filter)
            season_clause = f"AND Season IN ({seasons_in})"
        _bnorm = _name_sql("Batter")
        table = "trackman_pop"
        sql = f"""
        WITH batter_zones AS (
            SELECT {_bnorm} AS batter_name,
                CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                     THEN ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                     ELSE {ZONE_HEIGHT_BOT} END AS zone_bot,
                CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                     THEN ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                     ELSE {ZONE_HEIGHT_TOP} END AS zone_top
            FROM {table}
            WHERE PitchCall = 'StrikeCalled'
              AND PlateLocHeight IS NOT NULL
              AND Batter IS NOT NULL {season_clause}
            GROUP BY {_bnorm}
        ),
        raw AS (
            SELECT
                {_bnorm} AS Batter, BatterTeam,
                PitchCall, ExitSpeed, Angle, Distance, TaggedHitType,
                PlateLocSide, PlateLocHeight, BatterSide, Direction, KorBB,
                GameID || '_' || Inning || '_' || PAofInning || '_' || {_bnorm} AS pa_id,
                COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT}) AS zone_bot,
                COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP}) AS zone_top,
                CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                      AND ABS(PlateLocSide) <= {ZONE_SIDE}
                      AND PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                      AND PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP})
                     THEN 1 ELSE 0 END AS is_iz,
                CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                      AND NOT (ABS(PlateLocSide) <= {ZONE_SIDE}
                               AND PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                               AND PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP}))
                     THEN 1 ELSE 0 END AS is_oz
            FROM {table} r
            LEFT JOIN batter_zones bz ON {_bnorm} = bz.batter_name
            WHERE Batter IS NOT NULL {season_clause}
        ),
        pitch_agg AS (
            SELECT
                Batter, BatterTeam,
                COUNT(*) AS n_pitches,
                COUNT(DISTINCT pa_id) AS PA,
                COUNT(DISTINCT CASE WHEN KorBB='Strikeout' THEN pa_id END) AS ks,
                COUNT(DISTINCT CASE WHEN KorBB='Walk' THEN pa_id END) AS bbs,
                SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS swings,
                SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) AS whiffs,
                SUM(CASE WHEN PitchCall IN {_CONTACT_CALLS_SQL} THEN 1 ELSE 0 END) AS contacts,
                SUM(CASE WHEN {_HAS_LOC} THEN 1 ELSE 0 END) AS n_loc,
                SUM(is_iz) AS iz_count,
                SUM(is_oz) AS oz_count,
                SUM(CASE WHEN is_iz = 1 AND PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS iz_swings,
                SUM(CASE WHEN is_oz = 1 AND PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS oz_swings,
                SUM(CASE WHEN is_iz = 1 AND PitchCall IN {_CONTACT_CALLS_SQL} THEN 1 ELSE 0 END) AS iz_contacts,
                SUM(CASE WHEN is_oz = 1 AND PitchCall IN {_CONTACT_CALLS_SQL} THEN 1 ELSE 0 END) AS oz_contacts,
                SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN 1 ELSE 0 END) AS bbe,
                SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=95 THEN 1 ELSE 0 END) AS hard_hits,
                SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND Angle BETWEEN 8 AND 32 THEN 1 ELSE 0 END) AS sweet_spots,
                AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS AvgEV,
                MAX(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS MaxEV,
                AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN Angle END) AS AvgLA,
                AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN Distance END) AS AvgDist,
                SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=98
                    AND Angle >= GREATEST(26 - 2*(ExitSpeed-98), 8)
                    AND Angle <= LEAST(30 + 3*(ExitSpeed-98), 50) THEN 1 ELSE 0 END) AS Barrels,
                SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='GroundBall' THEN 1 ELSE 0 END) AS gb,
                SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='FlyBall' THEN 1 ELSE 0 END) AS fb,
                SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='LineDrive' THEN 1 ELSE 0 END) AS ld,
                SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='Popup' THEN 1 ELSE 0 END) AS pu
            FROM raw
            GROUP BY Batter, BatterTeam
            HAVING PA >= 50
        )
        SELECT * FROM pitch_agg
        """
        agg = query_precompute(sql)
        if agg.empty:
            return agg

        n = agg["bbe"]
        pa = agg["PA"]
        sw = agg["swings"]
        oz = agg["oz_count"]
        iz = agg["iz_count"]
        iz_sw = agg["iz_swings"]
        oz_sw = agg["oz_swings"]

        agg["HardHitPct"] = np.where(n > 0, agg["hard_hits"] / n * 100, np.nan)
        agg["BarrelPct"] = np.where(n > 0, agg["Barrels"] / n * 100, np.nan)
        agg["BarrelPA"] = np.where(pa > 0, agg["Barrels"] / pa * 100, np.nan)
        agg["SweetSpotPct"] = np.where(n > 0, agg["sweet_spots"] / n * 100, np.nan)
        agg["WhiffPct"] = np.where(sw > 0, agg["whiffs"] / sw * 100, np.nan)
        agg["KPct"] = np.where(pa > 0, agg["ks"] / pa * 100, np.nan)
        agg["BBPct"] = np.where(pa > 0, agg["bbs"] / pa * 100, np.nan)
        agg["ChasePct"] = np.where(oz > 0, agg["oz_swings"] / oz * 100, np.nan)
        agg["ChaseContact"] = np.where(oz_sw > 0, agg["oz_contacts"] / oz_sw * 100, np.nan)
        agg["ZoneSwingPct"] = np.where(iz > 0, iz_sw / iz * 100, np.nan)
        agg["ZoneContactPct"] = np.where(iz_sw > 0, agg["iz_contacts"] / iz_sw * 100, np.nan)
        agg["ZonePct"] = np.where(agg["n_loc"] > 0, agg["iz_count"] / agg["n_loc"] * 100, np.nan)
        agg["SwingPct"] = np.where(agg["n_pitches"] > 0, sw / agg["n_pitches"] * 100, np.nan)
        agg["GBPct"] = np.where(n > 0, agg["gb"] / n * 100, np.nan)
        agg["FBPct"] = np.where(n > 0, agg["fb"] / n * 100, np.nan)
        agg["LDPct"] = np.where(n > 0, agg["ld"] / n * 100, np.nan)
        agg["PUPct"] = np.where(n > 0, agg["pu"] / n * 100, np.nan)
        agg["AirPct"] = np.where(n > 0, (agg["fb"] + agg["ld"] + agg["pu"]) / n * 100, np.nan)
        agg["BBE"] = agg["bbe"]

        keep = [
            "Batter", "BatterTeam", "PA", "BBE", "AvgEV", "MaxEV", "HardHitPct",
            "Barrels", "BarrelPct", "BarrelPA", "SweetSpotPct", "AvgLA", "AvgDist",
            "WhiffPct", "KPct", "BBPct", "ChasePct", "ChaseContact",
            "ZoneSwingPct", "ZoneContactPct", "ZonePct", "SwingPct",
            "GBPct", "FBPct", "LDPct", "PUPct", "AirPct",
        ]
        return agg[[c for c in keep if c in agg.columns]]

    if _precompute_table_exists("batter_stats_pop"):
        where = None
        if season_filter:
            seasons_in = ",".join(str(int(s)) for s in season_filter)
            where = f"Season IN ({seasons_in})"
        df = _read_precompute_table("batter_stats_pop", where=where)
        required = {
            "Batter", "BatterTeam", "PA", "n_pitches", "swings", "whiffs", "contacts",
            "n_loc", "iz_count", "oz_count", "iz_swings", "oz_swings",
            "iz_contacts", "oz_contacts", "bbe", "hard_hits", "sweet_spots",
            "Barrels", "gb", "fb", "ld", "pu", "sum_ev", "n_ev", "max_ev", "sum_la", "sum_dist",
        }
        if not df.empty and required.issubset(set(df.columns)):
            return _compute_batter_from_base(df)

    season_clause = ""
    if season_filter:
        seasons_in = ",".join(str(int(s)) for s in season_filter)
        season_clause = f"AND Season IN ({seasons_in})"

    _bnorm = _name_sql("Batter")
    sql = f"""
    WITH batter_zones AS (
        SELECT {_bnorm} AS batter_name,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_BOT} END AS zone_bot,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_TOP} END AS zone_top
        FROM trackman
        WHERE PitchCall = 'StrikeCalled'
          AND PlateLocHeight IS NOT NULL
          AND Batter IS NOT NULL {season_clause}
        GROUP BY {_bnorm}
    ),
    raw AS (
        SELECT
            {_bnorm} AS Batter, BatterTeam,
            PitchCall, ExitSpeed, Angle, Distance, TaggedHitType,
            PlateLocSide, PlateLocHeight, BatterSide, Direction, KorBB,
            GameID || '_' || Inning || '_' || PAofInning || '_' || {_bnorm} AS pa_id,
            COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT}) AS zone_bot,
            COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP}) AS zone_top,
            CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                  AND ABS(PlateLocSide) <= {ZONE_SIDE}
                  AND PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                  AND PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP})
                 THEN 1 ELSE 0 END AS is_iz,
            CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                  AND NOT (ABS(PlateLocSide) <= {ZONE_SIDE}
                           AND PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                           AND PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP}))
                 THEN 1 ELSE 0 END AS is_oz
        FROM trackman r
        LEFT JOIN batter_zones bz ON {_bnorm} = bz.batter_name
        WHERE Batter IS NOT NULL {season_clause}
    ),
    pitch_agg AS (
        SELECT
            Batter, BatterTeam,
            COUNT(*) AS n_pitches,
            COUNT(DISTINCT pa_id) AS PA,
            COUNT(DISTINCT CASE WHEN KorBB='Strikeout' THEN pa_id END) AS ks,
            COUNT(DISTINCT CASE WHEN KorBB='Walk' THEN pa_id END) AS bbs,
            SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS swings,
            SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) AS whiffs,
            SUM(CASE WHEN PitchCall IN {_CONTACT_CALLS_SQL} THEN 1 ELSE 0 END) AS contacts,
            SUM(CASE WHEN {_HAS_LOC} THEN 1 ELSE 0 END) AS n_loc,
            SUM(is_iz) AS iz_count,
            SUM(is_oz) AS oz_count,
            SUM(CASE WHEN is_iz = 1 AND PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS iz_swings,
            SUM(CASE WHEN is_oz = 1 AND PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS oz_swings,
            SUM(CASE WHEN is_iz = 1 AND PitchCall IN {_CONTACT_CALLS_SQL} THEN 1 ELSE 0 END) AS iz_contacts,
            SUM(CASE WHEN is_oz = 1 AND PitchCall IN {_CONTACT_CALLS_SQL} THEN 1 ELSE 0 END) AS oz_contacts,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN 1 ELSE 0 END) AS bbe,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=95 THEN 1 ELSE 0 END) AS hard_hits,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND Angle BETWEEN 8 AND 32 THEN 1 ELSE 0 END) AS sweet_spots,
            AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS AvgEV,
            MAX(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS MaxEV,
            AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN Angle END) AS AvgLA,
            AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN Distance END) AS AvgDist,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=98
                AND Angle >= GREATEST(26 - 2*(ExitSpeed-98), 8)
                AND Angle <= LEAST(30 + 3*(ExitSpeed-98), 50) THEN 1 ELSE 0 END) AS Barrels,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='GroundBall' THEN 1 ELSE 0 END) AS gb,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='FlyBall' THEN 1 ELSE 0 END) AS fb,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='LineDrive' THEN 1 ELSE 0 END) AS ld,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='Popup' THEN 1 ELSE 0 END) AS pu
        FROM raw
        GROUP BY Batter, BatterTeam
        HAVING PA >= 50
    )
    SELECT * FROM pitch_agg
    """
    agg = query_population(sql)
    if agg.empty:
        return agg

    # Compute derived percentages — use np.where guards so zero-denominator
    # cases produce NaN (not 0%), preventing phantom 0% entries from
    # corrupting percentile distributions.
    n = agg["bbe"]
    pa = agg["PA"]
    sw = agg["swings"]
    oz = agg["oz_count"]
    iz = agg["iz_count"]
    iz_sw = agg["iz_swings"]
    oz_sw = agg["oz_swings"]

    agg["HardHitPct"] = np.where(n > 0, agg["hard_hits"] / n * 100, np.nan)
    agg["BarrelPct"] = np.where(n > 0, agg["Barrels"] / n * 100, np.nan)
    agg["BarrelPA"] = np.where(pa > 0, agg["Barrels"] / pa * 100, np.nan)
    agg["SweetSpotPct"] = np.where(n > 0, agg["sweet_spots"] / n * 100, np.nan)
    agg["WhiffPct"] = np.where(sw > 0, agg["whiffs"] / sw * 100, np.nan)
    agg["KPct"] = np.where(pa > 0, agg["ks"] / pa * 100, np.nan)
    agg["BBPct"] = np.where(pa > 0, agg["bbs"] / pa * 100, np.nan)
    agg["ChasePct"] = np.where(oz > 0, agg["oz_swings"] / oz * 100, np.nan)
    agg["ChaseContact"] = np.where(oz_sw > 0, agg["oz_contacts"] / oz_sw * 100, np.nan)
    agg["ZoneSwingPct"] = np.where(iz > 0, iz_sw / iz * 100, np.nan)
    agg["ZoneContactPct"] = np.where(iz_sw > 0, agg["iz_contacts"] / iz_sw * 100, np.nan)
    agg["ZonePct"] = np.where(agg["n_loc"] > 0, agg["iz_count"] / agg["n_loc"] * 100, np.nan)
    agg["SwingPct"] = np.where(agg["n_pitches"] > 0, sw / agg["n_pitches"] * 100, np.nan)
    agg["GBPct"] = np.where(n > 0, agg["gb"] / n * 100, np.nan)
    agg["FBPct"] = np.where(n > 0, agg["fb"] / n * 100, np.nan)
    agg["LDPct"] = np.where(n > 0, agg["ld"] / n * 100, np.nan)
    agg["PUPct"] = np.where(n > 0, agg["pu"] / n * 100, np.nan)
    agg["AirPct"] = np.where(n > 0, (agg["fb"] + agg["ld"] + agg["pu"]) / n * 100, np.nan)
    agg["BBE"] = agg["bbe"]

    keep = ["Batter", "BatterTeam", "PA", "BBE", "AvgEV", "MaxEV", "HardHitPct",
            "Barrels", "BarrelPct", "BarrelPA", "SweetSpotPct", "AvgLA", "AvgDist",
            "WhiffPct", "KPct", "BBPct", "ChasePct", "ChaseContact",
            "ZoneSwingPct", "ZoneContactPct", "ZonePct", "SwingPct",
            "GBPct", "FBPct", "LDPct", "PUPct", "AirPct"]
    return agg[[c for c in keep if c in agg.columns]]


@st.cache_data(show_spinner="Computing pitcher rankings...")
def compute_pitcher_stats_pop(season_filter=None, _version=5):
    """Compute pitcher stats for all D1 pitchers via DuckDB. Adaptive per-batter zone."""
    if _precompute_table_exists("pitcher_stats_pop"):
        df = _read_precompute_table("pitcher_stats_pop")
        if df.empty:
            return df
        if season_filter and "Season" in df.columns:
            seasons = [int(s) for s in season_filter]
            df = df[df["Season"].isin(seasons)]
        return _compute_pitcher_from_base(df)
    if _precompute_table_exists("trackman_pop"):
        season_clause = ""
        if season_filter:
            seasons_in = ",".join(str(int(s)) for s in season_filter)
            season_clause = f"AND Season IN ({seasons_in})"

        _pnorm = _name_sql("Pitcher")
        _bnorm_p = _name_sql("Batter")
        table = "trackman_pop"
        sql = f"""
        WITH batter_zones AS (
            SELECT {_bnorm_p} AS batter_name,
                CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                     THEN ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                     ELSE {ZONE_HEIGHT_BOT} END AS zone_bot,
                CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                     THEN ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                     ELSE {ZONE_HEIGHT_TOP} END AS zone_top
            FROM {table}
            WHERE PitchCall = 'StrikeCalled'
              AND PlateLocHeight IS NOT NULL
              AND Batter IS NOT NULL {season_clause}
            GROUP BY {_bnorm_p}
        ),
        raw AS (
            SELECT
                {_pnorm} AS Pitcher, PitcherTeam,
                PitchCall, ExitSpeed, Angle, TaggedHitType, TaggedPitchType,
                PlateLocSide, PlateLocHeight, RelSpeed, SpinRate, Extension, KorBB,
                GameID || '_' || Inning || '_' || PAofInning || '_' || {_bnorm_p} AS pa_id,
                CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                      AND ABS(PlateLocSide) <= {ZONE_SIDE}
                      AND PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                      AND PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP})
                     THEN 1 ELSE 0 END AS is_iz,
                CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                      AND NOT (ABS(PlateLocSide) <= {ZONE_SIDE}
                               AND PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                               AND PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP}))
                     THEN 1 ELSE 0 END AS is_oz
            FROM {table} r
            LEFT JOIN batter_zones bz ON {_bnorm_p} = bz.batter_name
            WHERE Pitcher IS NOT NULL {season_clause}
        ),
        pitch_agg AS (
            SELECT
                Pitcher, PitcherTeam,
                COUNT(*) AS Pitches,
                COUNT(DISTINCT pa_id) AS PA,
                COUNT(DISTINCT CASE WHEN KorBB='Strikeout' THEN pa_id END) AS ks,
                COUNT(DISTINCT CASE WHEN KorBB='Walk' THEN pa_id END) AS bbs,
                SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS swings,
                SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) AS whiffs,
                SUM(CASE WHEN {_HAS_LOC} THEN 1 ELSE 0 END) AS n_loc,
                SUM(is_iz) AS iz_count,
                SUM(is_oz) AS oz_count,
                SUM(CASE WHEN is_iz = 1 AND PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS iz_swings,
                SUM(CASE WHEN is_oz = 1 AND PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS oz_swings,
                SUM(CASE WHEN is_iz = 1 AND PitchCall IN {_CONTACT_CALLS_SQL} THEN 1 ELSE 0 END) AS iz_contacts,
                SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN 1 ELSE 0 END) AS bbe,
                SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=95 THEN 1 ELSE 0 END) AS hard_hits,
                AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS AvgEVAgainst,
                SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=98
                    AND Angle >= GREATEST(26 - 2*(ExitSpeed-98), 8)
                    AND Angle <= LEAST(30 + 3*(ExitSpeed-98), 50) THEN 1 ELSE 0 END) AS n_barrels,
                SUM(CASE WHEN PitchCall='InPlay' AND TaggedHitType='GroundBall' THEN 1 ELSE 0 END) AS gb,
                SUM(CASE WHEN PitchCall='InPlay' THEN 1 ELSE 0 END) AS n_ip,
                AVG(CASE WHEN TaggedPitchType IN ('Fastball','Sinker','Cutter') AND RelSpeed IS NOT NULL THEN RelSpeed END) AS AvgFBVelo,
                MAX(CASE WHEN TaggedPitchType IN ('Fastball','Sinker','Cutter') AND RelSpeed IS NOT NULL THEN RelSpeed END) AS MaxFBVelo,
                AVG(SpinRate) AS AvgSpin,
                AVG(Extension) AS Extension
            FROM raw
            GROUP BY Pitcher, PitcherTeam
            HAVING Pitches >= 100
        )
        SELECT * FROM pitch_agg
        """
        agg = query_precompute(sql)
        if agg.empty:
            return agg

        pa = agg["PA"]
        sw = agg["swings"]
        oz = agg["oz_count"]
        iz_sw = agg["iz_swings"]
        n_bat = agg["bbe"]
        n_ip = agg["n_ip"]

        agg["WhiffPct"] = np.where(sw > 0, agg["whiffs"] / sw * 100, np.nan)
        agg["KPct"] = np.where(pa > 0, agg["ks"] / pa * 100, np.nan)
        agg["BBPct"] = np.where(pa > 0, agg["bbs"] / pa * 100, np.nan)
        agg["ZonePct"] = np.where(agg["n_loc"] > 0, agg["iz_count"] / agg["n_loc"] * 100, np.nan)
        agg["ChasePct"] = np.where(oz > 0, agg["oz_swings"] / oz * 100, np.nan)
        agg["ZoneContactPct"] = np.where(iz_sw > 0, agg["iz_contacts"] / iz_sw * 100, np.nan)
        agg["HardHitAgainst"] = np.where(n_bat > 0, agg["hard_hits"] / n_bat * 100, np.nan)
        agg["BarrelPctAgainst"] = np.where(n_bat > 0, agg["n_barrels"] / n_bat * 100, np.nan)
        agg["GBPct"] = np.where(n_ip > 0, agg["gb"] / n_ip * 100, np.nan)
        agg["SwingPct"] = np.where(agg["Pitches"] > 0, sw / agg["Pitches"] * 100, np.nan)

        keep = [
            "Pitcher", "PitcherTeam", "Pitches", "PA", "AvgFBVelo", "MaxFBVelo",
            "AvgSpin", "Extension", "WhiffPct", "KPct", "BBPct", "ZonePct",
            "ChasePct", "ZoneContactPct", "AvgEVAgainst", "HardHitAgainst",
            "BarrelPctAgainst", "GBPct", "SwingPct",
        ]
        return agg[[c for c in keep if c in agg.columns]]

    if _precompute_table_exists("pitcher_stats_pop"):
        where = None
        if season_filter:
            seasons_in = ",".join(str(int(s)) for s in season_filter)
            where = f"Season IN ({seasons_in})"
        df = _read_precompute_table("pitcher_stats_pop", where=where)
        required = {
            "Pitcher", "PitcherTeam", "Pitches", "PA", "ks", "bbs", "swings", "whiffs",
            "n_loc", "iz_count", "oz_count", "iz_swings", "oz_swings", "iz_contacts",
            "bbe", "hard_hits", "n_barrels", "gb", "n_ip",
            "sum_ev_against", "n_ev_against", "sum_fb_velo", "n_fb_velo", "max_fb_velo",
            "sum_spin", "n_spin", "sum_ext", "n_ext",
        }
        if not df.empty and required.issubset(set(df.columns)):
            return _compute_pitcher_from_base(df)

    season_clause = ""
    if season_filter:
        seasons_in = ",".join(str(int(s)) for s in season_filter)
        season_clause = f"AND Season IN ({seasons_in})"

    _pnorm = _name_sql("Pitcher")
    _bnorm_p = _name_sql("Batter")
    sql = f"""
    WITH batter_zones AS (
        SELECT {_bnorm_p} AS batter_name,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_BOT} END AS zone_bot,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_TOP} END AS zone_top
        FROM trackman
        WHERE PitchCall = 'StrikeCalled'
          AND PlateLocHeight IS NOT NULL
          AND Batter IS NOT NULL {season_clause}
        GROUP BY {_bnorm_p}
    ),
    raw AS (
        SELECT
            {_pnorm} AS Pitcher, PitcherTeam,
            PitchCall, ExitSpeed, Angle, TaggedHitType, TaggedPitchType,
            PlateLocSide, PlateLocHeight, RelSpeed, SpinRate, Extension, KorBB,
            GameID || '_' || Inning || '_' || PAofInning || '_' || {_bnorm_p} AS pa_id,
            CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                  AND ABS(PlateLocSide) <= {ZONE_SIDE}
                  AND PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                  AND PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP})
                 THEN 1 ELSE 0 END AS is_iz,
            CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                  AND NOT (ABS(PlateLocSide) <= {ZONE_SIDE}
                           AND PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                           AND PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP}))
                 THEN 1 ELSE 0 END AS is_oz
        FROM trackman r
        LEFT JOIN batter_zones bz ON {_bnorm_p} = bz.batter_name
        WHERE Pitcher IS NOT NULL {season_clause}
    ),
    pitch_agg AS (
        SELECT
            Pitcher, PitcherTeam,
            COUNT(*) AS Pitches,
            COUNT(DISTINCT pa_id) AS PA,
            COUNT(DISTINCT CASE WHEN KorBB='Strikeout' THEN pa_id END) AS ks,
            COUNT(DISTINCT CASE WHEN KorBB='Walk' THEN pa_id END) AS bbs,
            SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS swings,
            SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) AS whiffs,
            SUM(CASE WHEN {_HAS_LOC} THEN 1 ELSE 0 END) AS n_loc,
            SUM(is_iz) AS iz_count,
            SUM(is_oz) AS oz_count,
            SUM(CASE WHEN is_iz = 1 AND PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS iz_swings,
            SUM(CASE WHEN is_oz = 1 AND PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS oz_swings,
            SUM(CASE WHEN is_iz = 1 AND PitchCall IN {_CONTACT_CALLS_SQL} THEN 1 ELSE 0 END) AS iz_contacts,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN 1 ELSE 0 END) AS bbe,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=95 THEN 1 ELSE 0 END) AS hard_hits,
            AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS AvgEVAgainst,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=98
                AND Angle >= GREATEST(26 - 2*(ExitSpeed-98), 8)
                AND Angle <= LEAST(30 + 3*(ExitSpeed-98), 50) THEN 1 ELSE 0 END) AS n_barrels,
            SUM(CASE WHEN PitchCall='InPlay' AND TaggedHitType='GroundBall' THEN 1 ELSE 0 END) AS gb,
            SUM(CASE WHEN PitchCall='InPlay' THEN 1 ELSE 0 END) AS n_ip,
            AVG(CASE WHEN TaggedPitchType IN ('Fastball','Sinker','Cutter') AND RelSpeed IS NOT NULL THEN RelSpeed END) AS AvgFBVelo,
            MAX(CASE WHEN TaggedPitchType IN ('Fastball','Sinker','Cutter') AND RelSpeed IS NOT NULL THEN RelSpeed END) AS MaxFBVelo,
            AVG(SpinRate) AS AvgSpin,
            AVG(Extension) AS Extension
        FROM raw
        GROUP BY Pitcher, PitcherTeam
        HAVING Pitches >= 100
    )
    SELECT * FROM pitch_agg
    """
    agg = query_population(sql)
    if agg.empty:
        return agg

    pa = agg["PA"]
    sw = agg["swings"]
    oz = agg["oz_count"]
    iz_sw = agg["iz_swings"]
    n_bat = agg["bbe"]
    n_ip = agg["n_ip"]

    agg["WhiffPct"] = np.where(sw > 0, agg["whiffs"] / sw * 100, np.nan)
    agg["KPct"] = np.where(pa > 0, agg["ks"] / pa * 100, np.nan)
    agg["BBPct"] = np.where(pa > 0, agg["bbs"] / pa * 100, np.nan)
    agg["ZonePct"] = np.where(agg["n_loc"] > 0, agg["iz_count"] / agg["n_loc"] * 100, np.nan)
    agg["ChasePct"] = np.where(oz > 0, agg["oz_swings"] / oz * 100, np.nan)
    agg["ZoneContactPct"] = np.where(iz_sw > 0, agg["iz_contacts"] / iz_sw * 100, np.nan)
    agg["HardHitAgainst"] = np.where(n_bat > 0, agg["hard_hits"] / n_bat * 100, np.nan)
    agg["BarrelPctAgainst"] = np.where(n_bat > 0, agg["n_barrels"] / n_bat * 100, np.nan)
    agg["GBPct"] = np.where(n_ip > 0, agg["gb"] / n_ip * 100, np.nan)
    agg["SwingPct"] = np.where(agg["Pitches"] > 0, sw / agg["Pitches"] * 100, np.nan)

    keep = ["Pitcher", "PitcherTeam", "Pitches", "PA", "AvgFBVelo", "MaxFBVelo",
            "AvgSpin", "Extension", "WhiffPct", "KPct", "BBPct", "ZonePct",
            "ChasePct", "ZoneContactPct", "AvgEVAgainst", "HardHitAgainst",
            "BarrelPctAgainst", "GBPct", "SwingPct"]
    return agg[[c for c in keep if c in agg.columns]]


# ──────────────────────────────────────────────
# League percentile dict builders (for scouting)
# ──────────────────────────────────────────────

@st.cache_data(show_spinner="Loading D1 batter percentiles...")
def build_league_hitters_from_local(season_filter=None):
    """Build league hitters dict for percentile context from local precomputed data.

    Returns dict matching build_tm_dict_for_league_hitters format with keys:
    rate, exit, pitch_rates, hit_types, etc.
    """
    df = compute_batter_stats_pop(season_filter=season_filter)
    if df.empty:
        return {}

    # Create a copy and rename columns to match TrueMedia API format
    out = df.copy()
    out["playerFullName"] = out["Batter"]
    out["newestTeamName"] = out["BatterTeam"]

    # Map local column names to API column names
    col_map = {
        "AvgEV": "ExitVel",
        "BarrelPct": "Barrel%",
        "HardHitPct": "HardHit%",
        "KPct": "K%",
        "BBPct": "BB%",
        "ChasePct": "Chase%",
        "WhiffPct": "SwStrk%",
        "GBPct": "Ground%",
        "FBPct": "Fly%",
        "LDPct": "Line%",
        "PUPct": "Popup%",
    }
    for old, new in col_map.items():
        if old in out.columns:
            out[new] = out[old]

    # Also add Hit95+% alias for HardHit%
    if "HardHit%" in out.columns:
        out["Hit95+%"] = out["HardHit%"]

    # Build sub-DataFrames matching the API dict structure
    def _sub(cols):
        keep = ["playerFullName", "newestTeamName"] + [c for c in cols if c in out.columns]
        return out[keep].copy()

    return {
        "rate": _sub(["PA", "K%", "BB%"]),
        "exit": _sub(["ExitVel", "Barrel%", "HardHit%", "Hit95+%"]),
        "pitch_rates": _sub(["Chase%", "SwStrk%", "SwingPct"]),
        "hit_types": _sub(["Ground%", "Fly%", "Line%", "Popup%"]),
        "hit_locations": pd.DataFrame(),
        "speed": pd.DataFrame(),
        "run_expectancy": pd.DataFrame(),
        "swing_pct": pd.DataFrame(),
        "swing_stats": pd.DataFrame(),
        "counting": pd.DataFrame(),
        "pitch_counts": pd.DataFrame(),
        "pitch_type_counts": pd.DataFrame(),
        "pitch_calls": pd.DataFrame(),
        "pitch_locations": pd.DataFrame(),
        "pitch_types": pd.DataFrame(),
        "stolen_bases": pd.DataFrame(),
        "home_runs": pd.DataFrame(),
        "expected_rate": pd.DataFrame(),
        "expected_hit_rates": pd.DataFrame(),
    }


@st.cache_data(show_spinner="Loading D1 pitcher percentiles...")
def build_league_pitchers_from_local(season_filter=None):
    """Build league pitchers dict for percentile context from local precomputed data.

    Returns dict matching build_tm_dict_for_league_pitchers format.
    """
    df = compute_pitcher_stats_pop(season_filter=season_filter)
    if df.empty:
        return {}

    # Create a copy and rename columns to match TrueMedia API format
    out = df.copy()
    out["playerFullName"] = out["Pitcher"]
    out["newestTeamName"] = out["PitcherTeam"]

    # Map local column names to API column names
    col_map = {
        "AvgFBVelo": "Vel",
        "MaxFBVelo": "MaxVel",
        "AvgSpin": "Spin",
        "KPct": "K%",
        "BBPct": "BB%",
        "ChasePct": "Chase%",
        "WhiffPct": "SwStrk%",
        "ZonePct": "InZone%",
        "AvgEVAgainst": "ExitVel",
        "BarrelPctAgainst": "Barrel%",
        "HardHitAgainst": "HardHit%",
        "GBPct": "Ground%",
    }
    for old, new in col_map.items():
        if old in out.columns:
            out[new] = out[old]

    # Build sub-DataFrames matching the API dict structure
    def _sub(cols):
        keep = ["playerFullName", "newestTeamName"] + [c for c in cols if c in out.columns]
        return out[keep].copy()

    return {
        "traditional": _sub(["Pitches", "PA", "K%", "BB%"]),
        "rate": _sub(["K%", "BB%"]),
        "movement": _sub(["Vel", "MaxVel", "Spin", "Extension"]),
        "pitch_rates": _sub(["Chase%", "SwStrk%", "InZone%", "SwingPct"]),
        "exit": _sub(["ExitVel", "Barrel%", "HardHit%"]),
        "hit_types": _sub(["Ground%"]),
        "pitch_locations": pd.DataFrame(),
        "pitch_types": pd.DataFrame(),
        "counting": pd.DataFrame(),
        "pitch_counts": pd.DataFrame(),
        "pitch_type_counts": pd.DataFrame(),
        "baserunning": pd.DataFrame(),
        "stolen_bases": pd.DataFrame(),
        "home_runs": pd.DataFrame(),
        "expected_rate": pd.DataFrame(),
        "expected_hit_rates": pd.DataFrame(),
        "hit_locations": pd.DataFrame(),
        "pitch_calls": pd.DataFrame(),
        "expected_counting": pd.DataFrame(),
        "pitching_counting": pd.DataFrame(),
        "bids": pd.DataFrame(),
    }


@st.cache_data(show_spinner=False)
def compute_stuff_baselines():
    """Pre-compute per-pitch-type mean/std for Stuff+ via DuckDB.
    Returns dict: {pitch_type: {col: (mean, std)}} and fb_velo_by_pitcher dict.
    """
    if _precompute_table_exists("stuff_baselines") and _precompute_table_exists("fb_velo_by_pitcher"):
        base_df = _read_precompute_table("stuff_baselines")
        if not base_df.empty:
            fb_df = _read_precompute_table("fb_velo_by_pitcher")
            vd_df = _read_precompute_table("velo_diff_stats") if _precompute_table_exists("velo_diff_stats") else pd.DataFrame()

            baseline_stats = {}
            for pt, grp in base_df.groupby("TaggedPitchType"):
                stats = {}
                for _, row in grp.iterrows():
                    metric = row.get("metric")
                    mean = row.get("mean")
                    std = row.get("std")
                    if metric and pd.notna(mean) and pd.notna(std):
                        stats[metric] = (float(mean), float(std))
                baseline_stats[pt] = stats

            # Require handedness-normalized HB in precompute; otherwise recompute from population
            has_hb_adj = any("HorzBreakAdj" in stats for stats in baseline_stats.values())
            if not has_hb_adj:
                base_df = pd.DataFrame()

            fb_velo_by_pitcher = {}
            if not fb_df.empty and "Pitcher" in fb_df.columns and "fb_velo" in fb_df.columns:
                fb_velo_by_pitcher = dict(zip(fb_df["Pitcher"], fb_df["fb_velo"]))

            velo_diff_stats = {}
            if not vd_df.empty:
                for _, row in vd_df.iterrows():
                    pt = row.get("TaggedPitchType")
                    mean = row.get("mean")
                    std = row.get("std")
                    if pt and pd.notna(mean) and pd.notna(std):
                        velo_diff_stats[pt] = (float(mean), float(std))

            if not base_df.empty:
                return {
                    "baseline_stats": baseline_stats,
                    "fb_velo_by_pitcher": fb_velo_by_pitcher,
                    "velo_diff_stats": velo_diff_stats,
                }

    base_cols = ["RelSpeed", "InducedVertBreak", "HorzBreakAdj", "Extension", "VertApprAngle", "SpinRate"]
    agg_exprs = []
    for col in base_cols:
        agg_exprs.append(f"AVG({col}) AS {col}_mean")
        agg_exprs.append(f"STDDEV({col}) AS {col}_std")
    agg_str = ", ".join(agg_exprs)

    sql = f"""
        SELECT pt_norm AS TaggedPitchType, {agg_str}, COUNT(*) as n
        FROM (
            SELECT *,
                CASE WHEN PitcherThrows IN ('Left','L') THEN -HorzBreak ELSE HorzBreak END AS HorzBreakAdj,
                CASE TaggedPitchType
                    WHEN 'FourSeamFastBall' THEN 'Fastball'
                    WHEN 'OneSeamFastBall' THEN 'Sinker'
                    WHEN 'TwoSeamFastBall' THEN 'Sinker'
                    WHEN 'ChangeUp' THEN 'Changeup'
                    ELSE TaggedPitchType
                END AS pt_norm
            FROM trackman
            WHERE TaggedPitchType NOT IN ('Other','Undefined','Knuckleball')
              AND RelSpeed IS NOT NULL
              AND PitcherThrows IS NOT NULL
        )
        GROUP BY pt_norm
    """
    df = query_population(sql)
    baseline_stats = {}
    for _, row in df.iterrows():
        pt = row["TaggedPitchType"]
        stats = {}
        for col in base_cols:
            m, s = row.get(f"{col}_mean"), row.get(f"{col}_std")
            if pd.notna(m) and pd.notna(s):
                stats[col] = (float(m), float(s))
        baseline_stats[pt] = stats

    # Per-pitcher fastball velo for VeloDiff
    fb_sql = """
        SELECT Pitcher, AVG(RelSpeed) AS fb_velo
        FROM trackman
        WHERE TaggedPitchType IN ('Fastball','Sinker','Cutter','FourSeamFastBall','TwoSeamFastBall','OneSeamFastBall')
          AND RelSpeed IS NOT NULL
        GROUP BY Pitcher
    """
    fb_df = query_population(fb_sql)
    fb_velo_by_pitcher = dict(zip(fb_df["Pitcher"], fb_df["fb_velo"]))

    # VeloDiff baselines for changeups/splitters
    velo_diff_stats = {}
    for pt in ["Changeup", "Splitter"]:
        # Map raw types to normalized for matching
        raw_types = [pt]
        if pt == "Changeup":
            raw_types = ["Changeup", "ChangeUp"]
        vd_sql = f"""
            SELECT t.Pitcher, t.RelSpeed, fb.fb_velo
            FROM trackman t
            JOIN (
                SELECT Pitcher, AVG(RelSpeed) AS fb_velo
                FROM trackman
                WHERE TaggedPitchType IN ('Fastball','Sinker','Cutter','FourSeamFastBall','TwoSeamFastBall','OneSeamFastBall')
                  AND RelSpeed IS NOT NULL
                GROUP BY Pitcher
            ) fb ON t.Pitcher = fb.Pitcher
            WHERE t.TaggedPitchType IN ({",".join(f"'{r}'" for r in raw_types)}) AND t.RelSpeed IS NOT NULL
        """
        try:
            vd_df = query_population(vd_sql)
            if len(vd_df) > 2:
                vd = vd_df["fb_velo"] - vd_df["RelSpeed"]
                velo_diff_stats[pt] = (float(vd.mean()), float(vd.std()))
        except Exception:
            pass

    return {"baseline_stats": baseline_stats, "fb_velo_by_pitcher": fb_velo_by_pitcher,
            "velo_diff_stats": velo_diff_stats}


def _build_tunnel_population_pop(con=None):
    """Build tunnel population from pitch-level kinematics for accurate physics.
    Returns dict: pair_type -> sorted array of raw tunnel scores."""
    if con is None and _precompute_table_exists("tunnel_population"):
        df = _read_precompute_table("tunnel_population")
        if not df.empty and {"pair_type", "score"}.issubset(set(df.columns)):
            pop = {}
            for pair, grp in df.groupby("pair_type"):
                scores = grp["score"].dropna().astype(float).tolist()
                pop[pair] = np.array(sorted(scores))
            return pop

    cols = [
        "Pitcher", "Batter", "GameID", "Inning", "PAofInning", "PitchofPA",
        "TaggedPitchType", "PitchCall",
        "RelHeight", "RelSide", "PlateLocHeight", "PlateLocSide", "RelSpeed", "Extension",
        "InducedVertBreak", "HorzBreak", "VertRelAngle", "HorzRelAngle",
        "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0",
    ]
    sql = f"""
        SELECT {', '.join(cols)}
        FROM trackman
        WHERE TaggedPitchType IS NOT NULL
          AND RelSpeed IS NOT NULL
          AND PlateLocSide BETWEEN -{PLATE_SIDE_MAX} AND {PLATE_SIDE_MAX}
          AND PlateLocHeight BETWEEN {PLATE_HEIGHT_MIN} AND {PLATE_HEIGHT_MAX}
    """
    if con is not None:
        df = con.execute(sql).fetchdf()
    else:
        df = query_population(sql)
    if df.empty:
        return {}

    # Drop low-usage pitch types per pitcher
    pt_counts = df.groupby(["Pitcher", "TaggedPitchType"]).size().rename("pt_n").reset_index()
    tot_counts = df.groupby("Pitcher").size().rename("tot_n").reset_index()
    usage = pt_counts.merge(tot_counts, on="Pitcher", how="left")
    usage["pct"] = usage["pt_n"] / usage["tot_n"] * 100
    keep = usage[usage["pct"] >= MIN_PITCH_USAGE_PCT][["Pitcher", "TaggedPitchType"]]
    df = df.merge(keep, on=["Pitcher", "TaggedPitchType"], how="inner")
    if df.empty:
        return {}

    # Sort for pitch order within PA
    sort_cols = ["GameID", "Inning", "PAofInning", "Batter", "PitchofPA"]
    df = df.sort_values([c for c in sort_cols if c in df.columns])

    # Build consecutive pairs within same PA
    grp_cols = ["GameID", "Inning", "PAofInning", "Batter"]
    prev = df.groupby(grp_cols).shift(1)
    mask = (
        df["TaggedPitchType"].notna() &
        prev["TaggedPitchType"].notna() &
        (df["TaggedPitchType"] != prev["TaggedPitchType"])
    )
    cur = df[mask].copy()
    prv = prev[mask].copy()

    if cur.empty:
        return {}

    # Physics constants
    MOUND_DIST = 60.5
    GRAVITY = 32.17
    COMMIT_TIME = 0.280

    # ── Vectorized commit/plate position computation ──
    def _compute_positions_vectorized(frame):
        """Compute commit-point and plate positions for all rows at once using numpy."""
        n = len(frame)
        commit_x = np.full(n, np.nan)
        commit_y = np.full(n, np.nan)
        plate_x = np.full(n, np.nan)
        plate_y = np.full(n, np.nan)

        # 9-param path (preferred)
        has_9p = (frame["x0"].notna() & frame["y0"].notna() &
                  frame["vx0"].notna() & frame["vy0"].notna()).values
        if has_9p.any():
            idx9 = np.where(has_9p)[0]
            a = 0.5 * frame["ay0"].values[idx9]
            b = frame["vy0"].values[idx9]
            c = frame["y0"].values[idx9]
            disc = b * b - 4 * a * c
            # Quadratic case (a != 0)
            valid_q = (a != 0) & (disc >= 0)
            # Linear case (a == 0, b != 0)
            valid_l = (a == 0) & (b != 0)
            t_total = np.full(len(idx9), np.nan)
            # Quadratic roots
            if valid_q.any():
                sq = np.sqrt(disc[valid_q])
                t1 = (-b[valid_q] - sq) / (2 * a[valid_q])
                t2 = (-b[valid_q] + sq) / (2 * a[valid_q])
                t1 = np.where(t1 > 0, t1, np.inf)
                t2 = np.where(t2 > 0, t2, np.inf)
                t_total[valid_q] = np.minimum(t1, t2)
            # Linear roots
            if valid_l.any():
                t_lin = -c[valid_l] / b[valid_l]
                t_total[valid_l] = np.where(t_lin > 0, t_lin, np.nan)
            t_total = np.where(np.isinf(t_total), np.nan, t_total)
            good = np.isfinite(t_total)
            if good.any():
                gi = idx9[good]
                tt = t_total[good]
                ct = np.maximum(0, tt - COMMIT_TIME)
                x0v = frame["x0"].values[gi]
                vx0v = frame["vx0"].values[gi]
                ax0v = frame["ax0"].values[gi]
                z0v = frame["z0"].values[gi]
                vz0v = frame["vz0"].values[gi]
                az0v = frame["az0"].values[gi]
                commit_x[gi] = -(x0v + vx0v * ct + 0.5 * ax0v * ct * ct)
                commit_y[gi] = z0v + vz0v * ct + 0.5 * az0v * ct * ct
                plate_x[gi] = -(x0v + vx0v * tt + 0.5 * ax0v * tt * tt)
                plate_y[gi] = z0v + vz0v * tt + 0.5 * az0v * tt * tt

        # IVB fallback for remaining rows
        needs_ivb = np.isnan(commit_x)
        if needs_ivb.any():
            idx_f = np.where(needs_ivb)[0]
            ext = frame["Extension"].values[idx_f].copy()
            ext = np.where(np.isnan(ext), 6.0, ext)
            velo_fps = np.maximum(frame["RelSpeed"].values[idx_f] * 5280.0 / 3600.0, 50.0)
            tt = (MOUND_DIST - ext) / velo_fps
            ivb = np.nan_to_num(frame["InducedVertBreak"].values[idx_f]) / 12.0
            hb = np.nan_to_num(frame["HorzBreak"].values[idx_f]) / 12.0
            tt2 = tt * tt
            safe_tt = np.where(tt > 0, tt, 1.0)
            safe_tt2 = np.where(tt > 0, tt2, 1.0)
            a_ivb = 2.0 * ivb / safe_tt2
            a_hb = 2.0 * hb / safe_tt2
            rel_h = frame["RelHeight"].values[idx_f]
            rel_s = frame["RelSide"].values[idx_f]
            ploc_h = frame["PlateLocHeight"].values[idx_f]
            ploc_s = frame["PlateLocSide"].values[idx_f]
            vy0 = np.where(tt > 0, (ploc_h - rel_h + 0.5 * GRAVITY * tt2 - 0.5 * a_ivb * tt2) / safe_tt, 0)
            vx0 = np.where(tt > 0, (ploc_s - rel_s - 0.5 * a_hb * tt2) / safe_tt, 0)
            ct = np.maximum(0, tt - COMMIT_TIME)
            ct2 = ct * ct
            commit_x[idx_f] = rel_s + vx0 * ct + 0.5 * a_hb * ct2
            commit_y[idx_f] = rel_h + vy0 * ct - 0.5 * GRAVITY * ct2 + 0.5 * a_ivb * ct2
            plate_x[idx_f] = rel_s + vx0 * tt + 0.5 * a_hb * tt2
            plate_y[idx_f] = rel_h + vy0 * tt - 0.5 * GRAVITY * tt2 + 0.5 * a_ivb * tt2

        return commit_x, commit_y, plate_x, plate_y

    # Compute positions for both current and previous pitch in each pair
    cur_cx, cur_cy, cur_px, cur_py = _compute_positions_vectorized(cur)
    prv_cx, prv_cy, prv_px, prv_py = _compute_positions_vectorized(prv)

    # Vectorized pair metrics
    valid = np.isfinite(cur_cx) & np.isfinite(prv_cx)
    cur_v = cur[valid].copy()
    prv_v = prv[valid].copy()
    ccx, ccy = cur_cx[valid], cur_cy[valid]
    pcx, pcy = prv_cx[valid], prv_cy[valid]
    cpx, cpy = cur_px[valid], cur_py[valid]
    ppx, ppy = prv_px[valid], prv_py[valid]

    commit_sep = np.sqrt((ccy - pcy)**2 + (ccx - pcx)**2) * 12
    plate_sep = np.sqrt((cpy - ppy)**2 + (cpx - ppx)**2) * 12
    rel_sep = np.sqrt((cur_v["RelHeight"].values - prv_v["RelHeight"].values)**2 +
                      (cur_v["RelSide"].values - prv_v["RelSide"].values)**2) * 12
    ivb_c = np.nan_to_num(cur_v["InducedVertBreak"].values)
    hb_c = np.nan_to_num(cur_v["HorzBreak"].values)
    ivb_p = np.nan_to_num(prv_v["InducedVertBreak"].values)
    hb_p = np.nan_to_num(prv_v["HorzBreak"].values)
    move_div = np.sqrt((ivb_c - ivb_p)**2 + (hb_c - hb_p)**2)
    velo_gap = np.abs(cur_v["RelSpeed"].values - prv_v["RelSpeed"].values)
    # Release angle separation
    has_ra = (cur_v["VertRelAngle"].notna().values & prv_v["VertRelAngle"].notna().values &
              cur_v["HorzRelAngle"].notna().values & prv_v["HorzRelAngle"].notna().values)
    rel_angle_sep = np.full(len(cur_v), np.nan)
    if has_ra.any():
        rel_angle_sep[has_ra] = np.sqrt(
            (cur_v["VertRelAngle"].values[has_ra] - prv_v["VertRelAngle"].values[has_ra])**2 +
            (cur_v["HorzRelAngle"].values[has_ra] - prv_v["HorzRelAngle"].values[has_ra])**2)

    # Build pair keys and collect metrics
    pt_a = prv_v["TaggedPitchType"].values.astype(str)
    pt_b = cur_v["TaggedPitchType"].values.astype(str)
    pair_keys = np.array(['/'.join(sorted([a, b])) for a, b in zip(pt_a, pt_b)])
    is_whiff = (cur_v["PitchCall"].values == "StrikeSwinging").astype(float)

    all_metrics = np.column_stack([commit_sep, rel_sep, plate_sep, move_div, velo_gap, rel_angle_sep])
    x_rows = all_metrics.tolist()
    y_rows = is_whiff.tolist()

    metrics_by_pair = {}
    for pk in np.unique(pair_keys):
        m = pair_keys == pk
        metrics_by_pair[pk] = all_metrics[m].tolist()

    if not metrics_by_pair:
        return {}

    def _fit_weights(X, y):
        # Features: commit_sep, rel_sep, plate_sep, move_div, velo_gap, rel_angle_sep
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]
        y = y[mask]
        if len(y) < 300 or len(set(y)) <= 1:
            return None
        # Downsample for speed
        max_n = 200000
        if len(y) > max_n:
            rng = np.random.default_rng(42)
            idxs = rng.choice(len(y), size=max_n, replace=False)
            X = X[idxs]
            y = y[idxs]
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0
        Xz = (X - mu) / sigma
        Xz = sm.add_constant(Xz, has_constant="add")
        model = sm.Logit(y, Xz).fit(disp=False, maxiter=50)
        coefs = model.params[1:]
        abs_coefs = np.abs(coefs)
        if abs_coefs.sum() == 0:
            return None
        w = abs_coefs / abs_coefs.sum()
        return {
            "commit": float(w[0]),
            "rel": float(w[1]),
            "plate": float(w[2]),
            "move": float(w[3]),
            "velo": float(w[4]),
            "rel_angle": float(w[5]),
        }

    # Fit global + pair-type-specific weights
    weights_blob = {"global": None, "pairs": {}}
    try:
        if x_rows and len(set(y_rows)) > 1:
            Xg = np.array(x_rows, dtype=float)
            yg = np.array(y_rows, dtype=float)
            weights_blob["global"] = _fit_weights(Xg, yg)
    except Exception:
        weights_blob["global"] = None

    # Build pair-specific weights using already-computed vectorized metrics
    try:
        pair_xy = {}
        for pk in np.unique(pair_keys):
            m = pair_keys == pk
            pair_xy[pk] = {"X": all_metrics[m].tolist(), "y": is_whiff[m].tolist()}
        for pair_key, blob in pair_xy.items():
            if len(blob["y"]) < 500 or len(set(blob["y"])) <= 1:
                continue
            w = _fit_weights(np.array(blob["X"], dtype=float), np.array(blob["y"], dtype=float))
            if w is not None:
                weights_blob["pairs"][pair_key] = w
    except Exception:
        pass

    # Default weights if regression fails
    cached = _load_tunnel_weights()
    if not weights_blob["global"]:
        weights_blob["global"] = cached.get("global") if cached else None
    if not weights_blob["global"]:
        weights_blob["global"] = {
            "commit": 0.55,
            "plate": 0.19,
            "rel": 0.10,
            "rel_angle": 0.08,
            "move": 0.06,
            "velo": 0.02,
        }
    _save_tunnel_weights(weights_blob)

    # Build benchmarks from commit_sep distribution
    pair_benchmarks = {}
    for pair_key, metrics in metrics_by_pair.items():
        if len(metrics) < 50:
            continue
        arr = np.array(metrics)
        commit_vals = arr[:, 0]
        p10, p25, p50, p75, p90 = np.percentile(commit_vals, [10, 25, 50, 75, 90])
        pair_benchmarks[pair_key] = (float(p10), float(p25), float(p50), float(p75), float(p90),
                                     float(commit_vals.mean()), float(commit_vals.std()))
    _save_tunnel_benchmarks(pair_benchmarks)

    # Build tunnel population raw score arrays
    pop = {}
    for pair_key, metrics in metrics_by_pair.items():
        bm = pair_benchmarks.get(pair_key)
        if bm is None:
            continue
        bm_p10, bm_p25, bm_p50, bm_p75, bm_p90 = bm[0], bm[1], bm[2], bm[3], bm[4]
        raw_list = []
        for commit_sep, rel_sep, plate_sep, move_div, velo_gap, rel_angle_sep in metrics:
            _anchors = [
                (bm_p90, 10), (bm_p75, 25), (bm_p50, 50),
                (bm_p25, 75), (bm_p10, 90),
            ]
            if commit_sep >= bm_p90:
                commit_pct = max(0, 10 * (1 - (commit_sep - bm_p90) / max(bm_p90, 1)))
            elif commit_sep <= bm_p10:
                commit_pct = min(100, 90 + 10 * (bm_p10 - commit_sep) / max(bm_p10, 1))
            else:
                commit_pct = 50.0
                for k in range(len(_anchors) - 1):
                    sep_hi, pct_lo = _anchors[k]
                    sep_lo, pct_hi = _anchors[k + 1]
                    if sep_lo <= commit_sep <= sep_hi:
                        frac = (sep_hi - commit_sep) / (sep_hi - sep_lo) if sep_hi != sep_lo else 0.5
                        commit_pct = pct_lo + frac * (pct_hi - pct_lo)
                        break
            plate_pct = min(100, plate_sep / 30.0 * 100)
            rel_pct = max(0, 100 - rel_sep * 12)
            if pd.notna(rel_angle_sep):
                rel_angle_pct = max(0, min(100, (1 - rel_angle_sep / 5.0) * 100))
            else:
                rel_angle_pct = 50
            move_pct = min(100, move_div / 30.0 * 100)
            velo_pct = min(100, velo_gap / 15.0 * 100)
            w_use = weights_blob["pairs"].get(pair_key) or weights_blob["global"]
            raw_tunnel = round(
                commit_pct * w_use["commit"] +
                plate_pct * w_use["plate"] +
                rel_pct * w_use["rel"] +
                rel_angle_pct * w_use["rel_angle"] +
                move_pct * w_use["move"] +
                velo_pct * w_use["velo"], 2)
            raw_list.append(raw_tunnel)
        if raw_list:
            pop[pair_key] = np.array(sorted(raw_list))

    return pop


@st.cache_data(show_spinner="Building tunnel population database...")
def build_tunnel_population_pop():
    return _build_tunnel_population_pop()
