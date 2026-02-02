"""Expected outcomes (xBA/xSLG) and zone grid data."""

import pandas as pd
import numpy as np

from config import SWING_CALLS, CONTACT_CALLS, is_barrel


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

    Probabilities calibrated for D1 college baseball (lower HR rates,
    more singles than MLB).  wOBA weights use 2024 NCAA run-environment
    scaling (slightly lower than MLB linear weights).
    """
    if batted_df.empty:
        return {}
    outcomes = []
    for _, row in batted_df.iterrows():
        ev, la = row.get("ExitSpeed", 0), row.get("Angle", 0)
        if pd.isna(ev) or pd.isna(la):
            continue
        # Barrel zone — college HR rate ~30% vs MLB ~40%
        if is_barrel(ev, la):
            outcomes.append({"xOut": 0.30, "x1B": 0.10, "x2B": 0.22, "x3B": 0.05, "xHR": 0.33})
        # High-EV fly ball
        elif ev >= 95 and 25 <= la <= 45:
            outcomes.append({"xOut": 0.45, "x1B": 0.05, "x2B": 0.18, "x3B": 0.05, "xHR": 0.27})
        # Line drive / hard-hit
        elif 10 <= la <= 25 and ev >= 85:
            outcomes.append({"xOut": 0.28, "x1B": 0.47, "x2B": 0.20, "x3B": 0.03, "xHR": 0.02})
        # Ground ball
        elif la < 10:
            outcomes.append({"xOut": 0.76, "x1B": 0.22, "x2B": 0.02, "x3B": 0.00, "xHR": 0.00})
        # Pop-up / extreme fly ball
        elif la > 45:
            outcomes.append({"xOut": 0.95, "x1B": 0.03, "x2B": 0.01, "x3B": 0.00, "xHR": 0.01})
        # Soft contact
        elif ev < 70:
            outcomes.append({"xOut": 0.90, "x1B": 0.08, "x2B": 0.02, "x3B": 0.00, "xHR": 0.00})
        # Medium contact catchall
        else:
            outcomes.append({"xOut": 0.68, "x1B": 0.22, "x2B": 0.07, "x3B": 0.01, "xHR": 0.02})
    if not outcomes:
        return {}
    odf = pd.DataFrame(outcomes)
    # NCAA-adjusted wOBA weights (lower run environment than MLB)
    odf["xwOBAcon"] = (0.0 * odf["xOut"] + 0.88 * odf["x1B"] + 1.24 * odf["x2B"]
                       + 1.56 * odf["x3B"] + 2.0 * odf["xHR"])
    return odf.mean().to_dict()
