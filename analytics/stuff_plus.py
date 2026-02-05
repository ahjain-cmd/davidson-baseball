"""Stuff+ computation — z-score composite model."""

import streamlit as st
import pandas as pd
import numpy as np

from data.population import compute_stuff_baselines
from config import normalize_pitch_types


def _compute_stuff_plus(data, baseline=None, baselines_dict=None):
    """Compute Stuff+ for every pitch in data.
    Model: z-score composite of velo, IVB, HB, extension, VAA, spin rate
    relative to same pitch type across the BASELINE population.
    100 = average, each 10 = 1 stdev better.

    Args:
        data: DataFrame of pitches to score
        baseline: DEPRECATED — ignored when baselines_dict is provided.
        baselines_dict: Pre-computed dict from compute_stuff_baselines().
                        If None, computes from DuckDB automatically.
    """
    if data is None or len(data) == 0:
        return data
    if "StuffPlus" in data.columns:
        return data

    base_df = normalize_pitch_types(data.copy())
    df = base_df.dropna(subset=["RelSpeed", "TaggedPitchType"]).copy()
    if df.empty:
        base_df["StuffPlus"] = np.nan
        return base_df

    if baselines_dict is None:
        baselines_dict = compute_stuff_baselines()

    baseline_stats = baselines_dict["baseline_stats"]
    fb_velo_map = baselines_dict["fb_velo_by_pitcher"]
    velo_diff_stats = baselines_dict["velo_diff_stats"]
    fb_velo = pd.Series(fb_velo_map)

    weights = {
        "Fastball":       {"RelSpeed": 2.0, "InducedVertBreak": 2.5, "HorzBreak": 0.3, "Extension": 0.5, "VertApprAngle": 2.5, "SpinRate": 1.0},
        "Sinker":         {"RelSpeed": 2.5, "InducedVertBreak": -0.5, "HorzBreak": 1.5, "Extension": 0.5, "VertApprAngle": -1.5, "SpinRate": 0.8},
        "Cutter":         {"RelSpeed": 0.8, "InducedVertBreak": 0.3, "HorzBreak": -1.5, "Extension": -1.0, "VertApprAngle": -0.5, "SpinRate": 2.0},
        "Slider":         {"RelSpeed": 1.0, "InducedVertBreak": -0.5, "HorzBreak": 1.0, "Extension": 0.3, "VertApprAngle": -2.5, "SpinRate": 1.5},
        "Curveball":      {"RelSpeed": 1.5, "InducedVertBreak": -1.5, "HorzBreak": -1.5, "Extension": -1.5, "VertApprAngle": -2.0, "SpinRate": 0.5},
        "Changeup":       {"RelSpeed": 0.5, "InducedVertBreak": 1.5, "HorzBreak": 1.0, "Extension": 0.5, "VertApprAngle": -2.5, "SpinRate": 1.0, "VeloDiff": 2.0},
        "Sweeper":        {"RelSpeed": 1.5, "InducedVertBreak": -1.0, "HorzBreak": 2.0, "Extension": 0.8, "VertApprAngle": -1.5, "SpinRate": 0.5},
        "Splitter":       {"RelSpeed": 1.0, "InducedVertBreak": -2.0, "HorzBreak": 0.5, "Extension": 1.0, "VertApprAngle": -2.0, "SpinRate": -0.3, "VeloDiff": 1.5},
        "Knuckle Curve":  {"RelSpeed": 1.5, "InducedVertBreak": -1.5, "HorzBreak": -1.5, "Extension": -1.5, "VertApprAngle": -2.0, "SpinRate": 0.5},
    }
    default_w = {"RelSpeed": 1.0, "InducedVertBreak": 1.0, "HorzBreak": 1.0, "Extension": 1.0, "VertApprAngle": 1.0, "SpinRate": 1.0}

    stuff_scores = []
    # Handedness-normalized horizontal break (arm-side positive for both hands)
    if "HorzBreak" in df.columns:
        throws = df.get("PitcherThrows")
        if throws is not None:
            is_l = throws.astype(str).str.lower().str.startswith("l")
            df["_HB_ADJ"] = np.where(is_l, -df["HorzBreak"].astype(float), df["HorzBreak"].astype(float))
        else:
            df["_HB_ADJ"] = df["HorzBreak"].astype(float)

    for pt, grp in df.groupby("TaggedPitchType"):
        w = weights.get(pt, default_w)
        bstats = baseline_stats.get(pt, {})
        z_total = pd.Series(0.0, index=grp.index)
        w_total = 0.0
        for col, weight in w.items():
            if col == "VeloDiff":
                if pt not in velo_diff_stats or "Pitcher" not in grp.columns:
                    continue
                grp_fb = grp["Pitcher"].map(fb_velo)
                vd = grp_fb - grp["RelSpeed"].astype(float)
                mu, sigma = velo_diff_stats[pt]
                if sigma == 0 or pd.isna(sigma):
                    continue
                z = (vd - mu) / sigma
                z_total += z.fillna(0) * weight
                w_total += abs(weight)
                continue
            if col == "HorzBreak":
                # Use handedness-normalized baseline + values
                bkey = "HorzBreakAdj" if "HorzBreakAdj" in bstats else "HorzBreak"
                if bkey not in bstats or "_HB_ADJ" not in grp.columns:
                    continue
                mu, sigma = bstats[bkey]
                vals = grp["_HB_ADJ"].astype(float)
            else:
                if col not in grp.columns or col not in bstats:
                    continue
                mu, sigma = bstats[col]
                vals = grp[col].astype(float)
            if sigma == 0 or pd.isna(sigma) or pd.isna(mu):
                continue
            z = (vals - mu) / sigma
            z_total += z.fillna(0) * weight
            w_total += abs(weight)
        if w_total > 0:
            z_total = z_total / w_total
        grp = grp.copy()
        grp["StuffPlus"] = 100 + z_total * 10
        stuff_scores.append(grp)

    if not stuff_scores:
        base_df["StuffPlus"] = np.nan
        return base_df

    scored = pd.concat(stuff_scores, ignore_index=False)
    base_df["StuffPlus"] = scored["StuffPlus"].reindex(base_df.index)
    return base_df


@st.cache_data(show_spinner="Computing Stuff+ grades...")
def _compute_stuff_plus_all(data):
    """Cached wrapper for _compute_stuff_plus on the full Davidson dataset."""
    return _compute_stuff_plus(data)
