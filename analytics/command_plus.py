"""Command+ and pitch pair results computation."""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

from config import (
    in_zone_mask, SWING_CALLS, filter_minor_pitches,
    normalize_pitch_types, filter_davidson, MIN_PITCH_USAGE_PCT,
)
from data.loader import load_davidson_data


def _compute_command_plus(pdf, data=None):
    """Compute Command+ for each pitch type. Returns DataFrame with
    Pitch, Pitches, Loc Spread (ft), Zone%, Edge%, CSW%, Chase%, Command+.
    Returns empty DataFrame if insufficient data."""
    cmd_rows = []
    for pt in sorted(pdf["TaggedPitchType"].unique()):
        ptd = pdf[pdf["TaggedPitchType"] == pt].dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if len(ptd) < 10:
            continue
        loc_std_h = ptd["PlateLocHeight"].std()
        loc_std_s = ptd["PlateLocSide"].std()
        loc_spread = np.sqrt(loc_std_h**2 + loc_std_s**2)
        in_zone = in_zone_mask(ptd)
        zone_pct = in_zone.mean() * 100
        edge = (
            ((ptd["PlateLocSide"].abs().between(0.5, 1.1)) |
             (ptd["PlateLocHeight"].between(1.2, 1.8)) |
             (ptd["PlateLocHeight"].between(3.2, 3.8))) &
            (ptd["PlateLocSide"].abs() <= 1.5) &
            ptd["PlateLocHeight"].between(0.5, 4.5)
        )
        edge_pct = edge.mean() * 100
        csw = ptd["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
        out_zone = ~in_zone
        chase_swings = ptd[out_zone & ptd["PitchCall"].isin(SWING_CALLS)]
        chase_pct = len(chase_swings) / max(out_zone.sum(), 1) * 100
        cmd_rows.append({
            "Pitch": pt,
            "Pitches": len(ptd),
            "Loc Spread (ft)": round(loc_spread, 2),
            "Zone%": round(zone_pct, 1),
            "Edge%": round(edge_pct, 1),
            "CSW%": round(csw, 1),
            "Chase%": round(chase_pct, 1),
        })
    if not cmd_rows:
        return pd.DataFrame()
    cmd_df = pd.DataFrame(cmd_rows)
    dav_data = data if data is not None else load_davidson_data()
    all_dav = filter_davidson(dav_data, role="pitcher")
    all_dav = normalize_pitch_types(all_dav)
    cmd_scores = []
    for _, row in cmd_df.iterrows():
        pt = row["Pitch"]
        all_pt = all_dav[all_dav["TaggedPitchType"] == pt].dropna(
            subset=["PlateLocSide", "PlateLocHeight"])
        if len(all_pt) < 20:
            cmd_scores.append(100.0)
            continue
        pitcher_spreads = []
        for p, pg in all_pt.groupby("Pitcher"):
            if len(pg) < 10:
                continue
            sp = np.sqrt(pg["PlateLocHeight"].std()**2 + pg["PlateLocSide"].std()**2)
            pitcher_spreads.append(sp)
        if len(pitcher_spreads) < 3:
            cmd_scores.append(100.0)
            continue
        pctl = 100 - percentileofscore(pitcher_spreads, row["Loc Spread (ft)"], kind="rank")
        cmd_scores.append(round(100 + (pctl - 50) * 0.4, 0))
    cmd_df["Command+"] = cmd_scores
    cmd_df = cmd_df.sort_values("Command+", ascending=False)
    return cmd_df


def _compute_pitch_pair_results(pdf, data, tunnel_df=None):
    """Compute effectiveness when pitch B follows pitch A in an at-bat."""
    if pdf.empty:
        return pd.DataFrame()
    pdf = filter_minor_pitches(pdf, min_pct=MIN_PITCH_USAGE_PCT)
    if pdf.empty:
        return pd.DataFrame()

    # Sort by game, at-bat, pitch number
    sort_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning", "PitchNo"] if c in pdf.columns]
    if len(sort_cols) < 2:
        return pd.DataFrame()

    pdf_s = pdf.sort_values(sort_cols).copy()
    pdf_s["PrevPitch"] = pdf_s.groupby(["GameID", "Batter", "PAofInning"])["TaggedPitchType"].shift(1)
    pdf_s = pdf_s.dropna(subset=["PrevPitch"])

    is_whiff = pdf_s["PitchCall"] == "StrikeSwinging"
    is_swing = pdf_s["PitchCall"].isin(SWING_CALLS)
    is_csw = pdf_s["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])

    rows = []
    for (prev, curr), grp in pdf_s.groupby(["PrevPitch", "TaggedPitchType"]):
        n = len(grp)
        if n < 25:
            continue
        swings = grp[is_swing.reindex(grp.index, fill_value=False)]
        whiffs = grp[is_whiff.reindex(grp.index, fill_value=False)]
        csws = grp[is_csw.reindex(grp.index, fill_value=False)]
        batted = grp[(grp["PitchCall"] == "InPlay") & grp["ExitSpeed"].notna()]
        # Putaway% proxy: 2-strike pitches resulting in swinging/called strike
        if "Strikes" in grp.columns:
            two_strike = grp[grp["Strikes"] == 2]
            putaway = two_strike[two_strike["PitchCall"].isin(["StrikeSwinging", "StrikeCalled"])]
            putaway_pct = len(putaway) / max(len(two_strike), 1) * 100 if len(two_strike) > 0 else np.nan
        else:
            putaway_pct = np.nan
        # K%: use KorBB if available, otherwise PlayResult == Strikeout
        if "KorBB" in grp.columns:
            k_events = grp["KorBB"].isin(["Strikeout", "K"])
            k_pct = k_events.mean() * 100 if len(grp) > 0 else np.nan
        elif "PlayResult" in grp.columns:
            k_events = grp["PlayResult"].isin(["Strikeout", "K"])
            k_pct = k_events.mean() * 100 if len(grp) > 0 else np.nan
        else:
            k_pct = np.nan
        # Tunnel lookup for this pair
        tun_grade, tun_score = "-", np.nan
        if isinstance(tunnel_df, pd.DataFrame) and not tunnel_df.empty:
            tun_match = tunnel_df[
                ((tunnel_df["Pitch A"] == prev) & (tunnel_df["Pitch B"] == curr)) |
                ((tunnel_df["Pitch A"] == curr) & (tunnel_df["Pitch B"] == prev))
            ]
            if not tun_match.empty:
                tun_grade = tun_match.iloc[0]["Grade"]
                tun_score = tun_match.iloc[0]["Tunnel Score"]
        rows.append({
            "Setup Pitch": prev, "Follow Pitch": curr, "Count": n,
            "Whiff%": round(len(whiffs) / max(len(swings), 1) * 100, 1),
            "CSW%": round(len(csws) / n * 100, 1),
            "Avg EV": round(batted["ExitSpeed"].mean(), 1) if len(batted) > 0 else np.nan,
            "Putaway%": round(putaway_pct, 1) if not pd.isna(putaway_pct) else np.nan,
            "K%": round(k_pct, 1) if not pd.isna(k_pct) else np.nan,
            "Chase%": round(
                (lambda _iz=in_zone_mask(grp): len(grp[(~_iz) & grp["PitchCall"].isin(SWING_CALLS)]) /
                max(len(grp[~_iz]), 1) * 100)(), 1),
            "Tunnel": tun_grade,
            "Tunnel Score": tun_score,
        })
    return pd.DataFrame(rows).sort_values("Whiff%", ascending=False).reset_index(drop=True)
