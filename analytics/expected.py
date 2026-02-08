"""Expected outcomes (xBA/xSLG) and zone grid data."""

import pandas as pd
import numpy as np

from config import SWING_CALLS, CONTACT_CALLS, is_barrel


# ── Lazy-loaded calibrated outcome probs & wOBA weights ─────────────────────
_OUTCOME_PROBS = None
_LINEAR_WEIGHTS = None


def _get_outcome_probs():
    global _OUTCOME_PROBS
    if _OUTCOME_PROBS is None:
        from analytics.historical_calibration import load_historical_calibration, fallback_outcome_probs
        cal = load_historical_calibration()
        _OUTCOME_PROBS = cal.outcome_probs if cal else fallback_outcome_probs()
    return _OUTCOME_PROBS


def _get_woba_weights():
    global _LINEAR_WEIGHTS
    if _LINEAR_WEIGHTS is None:
        from analytics.historical_calibration import load_historical_calibration, fallback_linear_weights
        cal = load_historical_calibration()
        _LINEAR_WEIGHTS = cal.linear_weights if cal else fallback_linear_weights()
    return _LINEAR_WEIGHTS


def _create_zone_grid_data(df, metric="swing_rate", batter_side="Right"):
    """Create 5x5 zone grid data for heatmaps.

    For RHH: negative PlateLocSide = inside, positive = outside.
    For LHH: negative PlateLocSide = outside, positive = inside.
    Returns (grid, annot, h_labels, v_labels) oriented from batter's perspective
    so column 0 = Inside for the given batter side.
    """
    h_edges = [-2, -0.83, -0.28, 0.28, 0.83, 2]
    v_edges = [0.5, 1.5, 2.17, 2.83, 3.5, 4.5]
    grid = np.full((5, 5), np.nan)
    annot = [['' for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            zone_df = df[
                (df["PlateLocSide"] >= h_edges[i]) & (df["PlateLocSide"] < h_edges[i + 1]) &
                (df["PlateLocHeight"] >= v_edges[j]) & (df["PlateLocHeight"] < v_edges[j + 1])
            ]
            if len(zone_df) < 3:
                continue
            swings = zone_df[zone_df["PitchCall"].isin(SWING_CALLS)]
            whiffs = zone_df[zone_df["PitchCall"] == "StrikeSwinging"]
            contacts = zone_df[zone_df["PitchCall"].isin(CONTACT_CALLS)]
            batted = zone_df[zone_df["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            val = np.nan
            if metric == "swing_rate":
                val = len(swings) / len(zone_df) * 100
                annot[j][i] = f"{val:.0f}%"
            elif metric == "whiff_rate":
                if len(swings) > 0:
                    val = len(whiffs) / len(swings) * 100
                    annot[j][i] = f"{val:.0f}%"
            elif metric == "contact_rate":
                if len(swings) > 0:
                    val = len(contacts) / len(swings) * 100
                    annot[j][i] = f"{val:.0f}%"
            elif metric == "avg_ev":
                if len(batted) > 0:
                    val = batted["ExitSpeed"].mean()
                    annot[j][i] = f"{val:.0f}"
            grid[j, i] = val
    # For RHH: negative PlateLocSide = inside, so col 0 = Far In
    # For LHH: negative PlateLocSide = outside, so col 0 = Far Out
    # Keep data in physical plate coordinates; flip labels to match batter perspective
    if batter_side == "Left":
        h_labels = ["Far Out", "Outside", "Middle", "Inside", "Far In"]
    else:
        h_labels = ["Far In", "Inside", "Middle", "Outside", "Far Out"]
    v_labels = ["Low+", "Low", "Mid", "High", "High+"]
    return grid, annot, h_labels, v_labels


def _compute_expected_outcomes(batted_df):
    """Compute expected outcomes based on EV/LA buckets.

    Probabilities are calibrated from D1 college Trackman data via
    ``analytics.historical_calibration``, with fallback to hardcoded values
    if the parquet is unavailable.
    """
    if batted_df.empty:
        return {}

    probs = _get_outcome_probs()
    lw = _get_woba_weights()

    outcomes = []
    for _, row in batted_df.iterrows():
        ev, la = row.get("ExitSpeed", 0), row.get("Angle", 0)
        if pd.isna(ev) or pd.isna(la):
            continue
        # Classify into bucket
        if is_barrel(ev, la):
            bucket = "Barrel"
        elif ev >= 95 and 25 <= la <= 45:
            bucket = "HiEV_FB"
        elif 10 <= la <= 25 and ev >= 85:
            bucket = "Hard_LD"
        elif la < 10:
            bucket = "GB"
        elif la > 45:
            bucket = "Popup"
        elif ev < 70:
            bucket = "Soft"
        else:
            bucket = "Medium"

        d = probs.get(bucket)
        if d is None:
            continue
        outcomes.append({
            "xOut": d["xOut"],
            "x1B": d["x1B"],
            "x2B": d["x2B"],
            "x3B": d["x3B"],
            "xHR": d["xHR"],
        })

    if not outcomes:
        return {}
    odf = pd.DataFrame(outcomes)
    # Calibrated wOBA weights from D1 run environment
    odf["xwOBAcon"] = (lw["out_w"] * odf["xOut"]
                       + lw["single_w"] * odf["x1B"]
                       + lw["double_w"] * odf["x2B"]
                       + lw["triple_w"] * odf["x3B"]
                       + lw["hr_w"] * odf["xHR"])
    return odf.mean().to_dict()
