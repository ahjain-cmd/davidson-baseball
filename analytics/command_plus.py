"""Command+ and pitch pair results computation."""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

from config import (
    in_zone_mask, SWING_CALLS, filter_minor_pitches,
    MIN_PITCH_USAGE_PCT, PLATE_SIDE_MAX, PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX,
)
from data.loader import query_population


def _build_fallback_spreads(data, pitch_types):
    """Compute per-pitcher loc spreads from in-memory Davidson data as fallback."""
    if data is None or data.empty:
        return {}
    req = {"TaggedPitchType", "Pitcher", "PlateLocSide", "PlateLocHeight"}
    if not req.issubset(data.columns):
        return {}
    loc = data.dropna(subset=["PlateLocSide", "PlateLocHeight", "TaggedPitchType"]).copy()
    loc = loc[
        loc["PlateLocSide"].between(-PLATE_SIDE_MAX, PLATE_SIDE_MAX)
        & loc["PlateLocHeight"].between(PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX)
        & loc["TaggedPitchType"].isin(pitch_types)
    ]
    if loc.empty:
        return {}
    result = {}
    for pt, grp in loc.groupby("TaggedPitchType"):
        pitcher_spreads = []
        for _, pg in grp.groupby("Pitcher"):
            if len(pg) < 10:
                continue
            s = np.sqrt(pg["PlateLocHeight"].std()**2 + pg["PlateLocSide"].std()**2)
            if np.isfinite(s):
                pitcher_spreads.append(s)
        if len(pitcher_spreads) >= 3:
            result[pt] = np.array(pitcher_spreads)
    return result


def _compute_command_plus(pdf, data=None):
    """Compute Command+ for each pitch type. Returns DataFrame with
    Pitch, Pitches, Loc Spread (ft), Zone%, Edge%, CSW%, Chase%, Command+, Percentile.
    Returns empty DataFrame if insufficient data."""
    cmd_rows = []
    for pt in sorted(pdf["TaggedPitchType"].unique()):
        ptd = pdf[pdf["TaggedPitchType"] == pt].dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
        # Filter to valid location bounds to avoid outlier bias
        ptd = ptd[
            ptd["PlateLocSide"].between(-PLATE_SIDE_MAX, PLATE_SIDE_MAX)
            & ptd["PlateLocHeight"].between(PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX)
        ]
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
    # NCAA D1 baseline via population table (DuckDB)
    pitch_types = sorted(cmd_df["Pitch"].unique())
    if pitch_types:
        pt_sql = ", ".join(f"'{p.replace(chr(39), chr(39)+chr(39))}'" for p in pitch_types)
        sql = f"""
            WITH t AS (
                SELECT Pitcher,
                       PlateLocSide,
                       PlateLocHeight,
                       CASE TaggedPitchType
                           WHEN 'FourSeamFastBall' THEN 'Fastball'
                           WHEN 'OneSeamFastBall' THEN 'Sinker'
                           WHEN 'TwoSeamFastBall' THEN 'Sinker'
                           WHEN 'ChangeUp' THEN 'Changeup'
                           ELSE TaggedPitchType
                       END AS pt_norm
                FROM trackman
                WHERE TaggedPitchType NOT IN ('Other','Undefined','Knuckleball')
                  AND PlateLocSide BETWEEN -{PLATE_SIDE_MAX} AND {PLATE_SIDE_MAX}
                  AND PlateLocHeight BETWEEN {PLATE_HEIGHT_MIN} AND {PLATE_HEIGHT_MAX}
            )
            SELECT pt_norm AS PitchType,
                   Pitcher,
                   STDDEV(PlateLocHeight) AS std_h,
                   STDDEV(PlateLocSide) AS std_s,
                   COUNT(*) AS n
            FROM t
            WHERE pt_norm IN ({pt_sql})
            GROUP BY pt_norm, Pitcher
            HAVING COUNT(*) >= 10
        """
        baseline_df = query_population(sql)
    else:
        baseline_df = pd.DataFrame()

    # Fallback: if D1 population query failed, use in-memory data
    fallback_spreads = {}
    if baseline_df.empty and data is not None:
        fallback_spreads = _build_fallback_spreads(data, pitch_types)

    cmd_scores = []
    cmd_pctls = []
    for _, row in cmd_df.iterrows():
        pt = row["Pitch"]
        my_spread = row["Loc Spread (ft)"]
        if pd.isna(my_spread):
            cmd_scores.append(100.0)
            cmd_pctls.append(50.0)
            continue

        # Try D1 population baseline first
        spreads = None
        if not baseline_df.empty:
            pt_df = baseline_df[baseline_df["PitchType"] == pt].copy()
            if not pt_df.empty:
                spreads = np.sqrt(pt_df["std_h"].astype(float)**2 + pt_df["std_s"].astype(float)**2)
                spreads = spreads.replace([np.inf, -np.inf], np.nan).dropna()
                if len(spreads) < 3:
                    spreads = None

        # Fallback to in-memory data
        if spreads is None and pt in fallback_spreads:
            spreads = fallback_spreads[pt]

        if spreads is None:
            cmd_scores.append(100.0)
            cmd_pctls.append(50.0)
            continue

        pctl = 100 - percentileofscore(spreads, my_spread, kind="rank")
        cmd_scores.append(round(100 + (pctl - 50) * 0.4, 0))
        cmd_pctls.append(round(pctl, 1))

    cmd_df["Command+"] = cmd_scores
    cmd_df["Percentile"] = cmd_pctls
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
