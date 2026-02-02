"""Davidson-specific stat computations — batter & pitcher stats from Trackman data."""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

from config import (
    ZONE_HEIGHT_BOT,
    ZONE_HEIGHT_TOP,
    ZONE_SIDE,
    MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE,
    PLATE_SIDE_MAX,
    PLATE_HEIGHT_MIN,
    PLATE_HEIGHT_MAX,
    SWING_CALLS,
    CONTACT_CALLS,
    in_zone_mask,
    is_barrel_mask,
    normalize_pitch_types,
)


@st.cache_data(show_spinner=False)
def _build_batter_zones(data):
    """Build per-batter strike zone boundaries from called-strike distributions.

    Returns dict  batter_name -> (zone_bot, zone_top).
    Falls back to fixed zone (1.5-3.5) when fewer than MIN samples exist.
    """
    zones = {}
    called = data[data["PitchCall"] == "StrikeCalled"].dropna(subset=["PlateLocHeight"])
    # Guard against extreme/outlier plate locations skewing adaptive zones.
    called = called[
        called["PlateLocHeight"].between(PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX) &
        called["PlateLocSide"].between(-PLATE_SIDE_MAX, PLATE_SIDE_MAX)
    ]
    if called.empty:
        return zones
    for batter, grp in called.groupby("Batter"):
        if len(grp) >= MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE:
            zones[batter] = (
                round(grp["PlateLocHeight"].quantile(0.05), 3),
                round(grp["PlateLocHeight"].quantile(0.95), 3),
            )
    return zones


@st.cache_data(show_spinner=False)
def compute_batter_stats(data, season_filter=None):
    df = data.copy()
    if season_filter:
        df = df[df["Season"].isin(season_filter)]
    batter_zones = _build_batter_zones(df)
    _in_zone = in_zone_mask(df, batter_zones, batter_col="Batter")
    has_loc = df["PlateLocSide"].notna() & df["PlateLocHeight"].notna()
    out_zone = ~_in_zone & has_loc

    # Build unique PA identifier (robust to sampled data where PitchofPA==1 may be missing)
    _pa_id_cols = ["GameID", "Inning", "PAofInning", "Batter"]
    if all(c in df.columns for c in _pa_id_cols):
        df["_pa_id"] = (df["GameID"].astype(str) + "_" + df["Inning"].astype(str)
                        + "_" + df["PAofInning"].astype(str) + "_" + df["Batter"].astype(str))
    else:
        df["_pa_id"] = df["PitchofPA"].astype(str) + "_" + df.index.astype(str)

    # Pre-compute boolean columns for vectorized groupby
    df["_is_swing"] = df["PitchCall"].isin(SWING_CALLS)
    df["_is_whiff"] = df["PitchCall"] == "StrikeSwinging"
    df["_is_contact"] = df["PitchCall"].isin(CONTACT_CALLS)
    df["_is_inplay"] = df["PitchCall"] == "InPlay"
    df["_in_zone"] = _in_zone
    df["_out_zone"] = out_zone
    df["_has_loc"] = has_loc
    df["_iz_swing"] = _in_zone & df["_is_swing"]
    df["_oz_swing"] = out_zone & df["_is_swing"]
    df["_iz_contact"] = _in_zone & df["_is_contact"]
    df["_oz_contact"] = out_zone & df["_is_contact"]
    df["_has_ev"] = df["_is_inplay"] & df["ExitSpeed"].notna()
    df["_hard_hit"] = df["_has_ev"] & (df["ExitSpeed"] >= 95)
    if "Angle" in df.columns:
        df["_sweet_spot"] = df["_has_ev"] & df["Angle"].between(8, 32)
    else:
        df["_sweet_spot"] = False

    # Count unique PAs and deduplicated K/BB per batter
    pa_counts = df.groupby(["Batter", "BatterTeam"])["_pa_id"].nunique().reset_index(name="PA")
    k_pas = df[df["KorBB"] == "Strikeout"].groupby(["Batter", "BatterTeam"])["_pa_id"].nunique().reset_index(name="ks")
    bb_pas = df[df["KorBB"] == "Walk"].groupby(["Batter", "BatterTeam"])["_pa_id"].nunique().reset_index(name="bbs")

    # Vectorized groupby aggregation
    grp = df.groupby(["Batter", "BatterTeam"])
    agg = grp.agg(
        n_pitches=("_is_swing", "size"),
        swings=("_is_swing", "sum"),
        whiffs=("_is_whiff", "sum"),
        contacts=("_is_contact", "sum"),
        n_loc=("_has_loc", "sum"),
        iz_count=("_in_zone", "sum"),
        oz_count=("_out_zone", "sum"),
        iz_swings=("_iz_swing", "sum"),
        oz_swings=("_oz_swing", "sum"),
        iz_contacts=("_iz_contact", "sum"),
        oz_contacts=("_oz_contact", "sum"),
        bbe=("_has_ev", "sum"),
        hard_hits=("_hard_hit", "sum"),
        sweet_spots=("_sweet_spot", "sum"),
    ).reset_index()

    # Merge correct PA and K/BB counts
    agg = agg.merge(pa_counts, on=["Batter", "BatterTeam"], how="left")
    agg = agg.merge(k_pas, on=["Batter", "BatterTeam"], how="left")
    agg = agg.merge(bb_pas, on=["Batter", "BatterTeam"], how="left")
    agg["PA"] = agg["PA"].fillna(0).astype(int)
    agg["ks"] = agg["ks"].fillna(0).astype(int)
    agg["bbs"] = agg["bbs"].fillna(0).astype(int)

    # Filter to min 50 PA
    agg = agg[agg["PA"] >= 50].copy()

    # Batted ball sub-aggregations (all in-play; EV-only for EV metrics)
    batted_all = df[df["_is_inplay"]].copy()
    batted_ev = df[df["_has_ev"]].copy()
    if not batted_all.empty:
        batted_agg = batted_ev.groupby(["Batter", "BatterTeam"]).agg(
            AvgEV=("ExitSpeed", "mean"),
            MaxEV=("ExitSpeed", "max"),
            AvgLA=("Angle", "mean"),
            AvgDist=("Distance", "mean"),
        ).reset_index()
        # Barrel computation
        batted_ev["_barrel"] = is_barrel_mask(batted_ev)
        barrel_agg = batted_ev.groupby(["Batter", "BatterTeam"])["_barrel"].sum().reset_index()
        barrel_agg.columns = ["Batter", "BatterTeam", "Barrels"]
        batted_agg = batted_agg.merge(barrel_agg, on=["Batter", "BatterTeam"], how="left")
        # Hit type counts
        for ht, col_name in [("GroundBall", "gb"), ("FlyBall", "fb"), ("LineDrive", "ld"), ("Popup", "pu")]:
            if "TaggedHitType" in batted_all.columns:
                ht_counts = batted_all[batted_all["TaggedHitType"] == ht].groupby(["Batter", "BatterTeam"]).size().reset_index(name=col_name)
                batted_agg = batted_agg.merge(ht_counts, on=["Batter", "BatterTeam"], how="left")
            else:
                batted_agg[col_name] = 0
        batted_agg = batted_agg.fillna({"gb": 0, "fb": 0, "ld": 0, "pu": 0, "Barrels": 0})
        agg = agg.merge(batted_agg, on=["Batter", "BatterTeam"], how="left")
    else:
        for c in ["AvgEV", "MaxEV", "AvgLA", "AvgDist", "Barrels", "gb", "fb", "ld", "pu"]:
            agg[c] = np.nan

    # Compute derived percentages
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

    # Directional stats (pull/center/oppo) — requires per-batter side, compute for batted balls
    if not batted_all.empty and "Direction" in batted_all.columns:
        batter_side = df.groupby(["Batter", "BatterTeam"])["BatterSide"].agg(
            lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else "Right"
        ).reset_index().rename(columns={"BatterSide": "_side"})
        dir_df = batted_all.dropna(subset=["Direction"]).merge(batter_side, on=["Batter", "BatterTeam"], how="left")
        dir_df["_side"] = dir_df["_side"].fillna("Right")
        dir_df = dir_df[dir_df["Direction"].between(-90, 90)]
        dir_df["_pull"] = np.where(dir_df["_side"] == "Left", dir_df["Direction"] > 15, dir_df["Direction"] < -15)
        dir_df["_oppo"] = np.where(dir_df["_side"] == "Left", dir_df["Direction"] < -15, dir_df["Direction"] > 15)
        dir_agg = dir_df.groupby(["Batter", "BatterTeam"]).agg(
            _n_dir=("_pull", "size"), _pull=("_pull", "sum"), _oppo=("_oppo", "sum"),
        ).reset_index()
        dir_agg["PullPct"] = np.where(dir_agg["_n_dir"] > 0, dir_agg["_pull"] / dir_agg["_n_dir"] * 100, np.nan)
        dir_agg["OppoPct"] = np.where(dir_agg["_n_dir"] > 0, dir_agg["_oppo"] / dir_agg["_n_dir"] * 100, np.nan)
        dir_agg["StraightPct"] = np.where(dir_agg["_n_dir"] > 0, 100 - dir_agg["PullPct"] - dir_agg["OppoPct"], np.nan)
        agg = agg.merge(dir_agg[["Batter", "BatterTeam", "PullPct", "OppoPct", "StraightPct"]],
                        on=["Batter", "BatterTeam"], how="left")
    else:
        agg["PullPct"] = np.nan
        agg["OppoPct"] = np.nan
        agg["StraightPct"] = np.nan

    # Select output columns
    keep = ["Batter", "BatterTeam", "PA", "BBE", "AvgEV", "MaxEV", "HardHitPct",
            "Barrels", "BarrelPct", "BarrelPA", "SweetSpotPct", "AvgLA", "AvgDist",
            "WhiffPct", "KPct", "BBPct", "ChasePct", "ChaseContact",
            "ZoneSwingPct", "ZoneContactPct", "ZonePct", "SwingPct",
            "GBPct", "FBPct", "LDPct", "PUPct", "AirPct",
            "PullPct", "StraightPct", "OppoPct"]
    return agg[[c for c in keep if c in agg.columns]]


@st.cache_data(show_spinner=False)
def compute_pitcher_stats(data, season_filter=None):
    df = data.copy()
    if season_filter:
        df = df[df["Season"].isin(season_filter)]
    batter_zones = _build_batter_zones(df)
    _in_zone = in_zone_mask(df, batter_zones, batter_col="Batter")
    has_loc = df["PlateLocSide"].notna() & df["PlateLocHeight"].notna()
    out_zone = ~_in_zone & has_loc

    # Build unique PA identifier (robust to sampled data where PitchofPA==1 may be missing)
    _pa_id_cols = ["GameID", "Inning", "PAofInning", "Batter"]
    if all(c in df.columns for c in _pa_id_cols):
        df["_pa_id"] = (df["GameID"].astype(str) + "_" + df["Inning"].astype(str)
                        + "_" + df["PAofInning"].astype(str) + "_" + df["Batter"].astype(str))
    else:
        df["_pa_id"] = df["PitchofPA"].astype(str) + "_" + df.index.astype(str)

    # Pre-compute boolean columns
    df["_is_swing"] = df["PitchCall"].isin(SWING_CALLS)
    df["_is_whiff"] = df["PitchCall"] == "StrikeSwinging"
    df["_is_contact"] = df["PitchCall"].isin(CONTACT_CALLS)
    df["_is_inplay"] = df["PitchCall"] == "InPlay"
    df["_in_zone"] = _in_zone
    df["_out_zone"] = out_zone
    df["_has_loc"] = has_loc
    df["_iz_swing"] = _in_zone & df["_is_swing"]
    df["_oz_swing"] = out_zone & df["_is_swing"]
    df["_iz_contact"] = _in_zone & df["_is_contact"]
    df["_has_ev"] = df["_is_inplay"] & df["ExitSpeed"].notna()
    df["_hard_hit"] = df["_has_ev"] & (df["ExitSpeed"] >= 95)
    df["_is_fb"] = df["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter"])
    df["_fb_velo"] = np.where(df["_is_fb"] & df["RelSpeed"].notna(), df["RelSpeed"], np.nan)

    # Count unique PAs and deduplicated K/BB per pitcher
    pa_counts = df.groupby(["Pitcher", "PitcherTeam"])["_pa_id"].nunique().reset_index(name="PA")
    k_pas = df[df["KorBB"] == "Strikeout"].groupby(["Pitcher", "PitcherTeam"])["_pa_id"].nunique().reset_index(name="ks")
    bb_pas = df[df["KorBB"] == "Walk"].groupby(["Pitcher", "PitcherTeam"])["_pa_id"].nunique().reset_index(name="bbs")

    # Vectorized groupby
    grp = df.groupby(["Pitcher", "PitcherTeam"])
    agg = grp.agg(
        Pitches=("_is_swing", "size"),
        swings=("_is_swing", "sum"),
        whiffs=("_is_whiff", "sum"),
        n_loc=("_has_loc", "sum"),
        iz_count=("_in_zone", "sum"),
        oz_count=("_out_zone", "sum"),
        iz_swings=("_iz_swing", "sum"),
        oz_swings=("_oz_swing", "sum"),
        iz_contacts=("_iz_contact", "sum"),
        bbe=("_has_ev", "sum"),
        hard_hits=("_hard_hit", "sum"),
        AvgFBVelo=("_fb_velo", "mean"),
        MaxFBVelo=("_fb_velo", "max"),
        AvgSpin=("SpinRate", "mean"),
        Extension=("Extension", "mean"),
    ).reset_index()

    # Merge correct PA and K/BB counts
    agg = agg.merge(pa_counts, on=["Pitcher", "PitcherTeam"], how="left")
    agg = agg.merge(k_pas, on=["Pitcher", "PitcherTeam"], how="left")
    agg = agg.merge(bb_pas, on=["Pitcher", "PitcherTeam"], how="left")
    agg["PA"] = agg["PA"].fillna(0).astype(int)
    agg["ks"] = agg["ks"].fillna(0).astype(int)
    agg["bbs"] = agg["bbs"].fillna(0).astype(int)

    agg = agg[agg["Pitches"] >= 100].copy()

    # Batted ball sub-aggregations (all in-play; EV-only for EV metrics)
    batted_all = df[df["_is_inplay"]].copy()
    batted_ev = df[df["_has_ev"]].copy()
    if not batted_all.empty:
        batted_ev["_barrel"] = is_barrel_mask(batted_ev)
        ba = batted_ev.groupby(["Pitcher", "PitcherTeam"]).agg(
            AvgEVAgainst=("ExitSpeed", "mean"),
            n_barrels=("_barrel", "sum"),
        ).reset_index()
        agg = agg.merge(ba, on=["Pitcher", "PitcherTeam"], how="left")
    else:
        agg["AvgEVAgainst"] = np.nan
        agg["n_barrels"] = 0

    # GB count from in-play
    inplay = batted_all
    if not inplay.empty and "TaggedHitType" in inplay.columns:
        gb_counts = inplay[inplay["TaggedHitType"] == "GroundBall"].groupby(["Pitcher", "PitcherTeam"]).size().reset_index(name="gb")
        ip_counts = inplay.groupby(["Pitcher", "PitcherTeam"]).size().reset_index(name="n_ip")
        agg = agg.merge(gb_counts, on=["Pitcher", "PitcherTeam"], how="left")
        agg = agg.merge(ip_counts, on=["Pitcher", "PitcherTeam"], how="left")
        agg["gb"] = agg["gb"].fillna(0)
        agg["n_ip"] = agg["n_ip"].fillna(0)
    else:
        agg["gb"] = 0
        agg["n_ip"] = 0

    # Compute derived percentages
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


def get_percentile(value, series):
    if pd.isna(value) or series.dropna().empty:
        return np.nan
    return percentileofscore(series.dropna(), value, kind='rank')
