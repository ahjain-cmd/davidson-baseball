from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from decision_engine.core.state import GameState


def _weighted_centroid(df: pd.DataFrame, fallback_x: float, fallback_y: float) -> Tuple[float, float]:
    if df is None or df.empty or len(df) < 2:
        return (fallback_x, fallback_y)
    w = df["weight"].values if "weight" in df.columns else np.ones(len(df))
    w_sum = float(np.nansum(w))
    if w_sum <= 0:
        return (fallback_x, fallback_y)
    cx = float(np.average(df["x"].values, weights=w))
    cy = float(np.average(df["y"].values, weights=w))
    return (cx, cy)


def _recommend_fielder_positions(batted_df: pd.DataFrame, batter_side: str) -> Dict[str, Tuple[float, float]]:
    """Compute recommended fielder positions based on spray data (Direction/Distance)."""
    if batted_df is None or batted_df.empty:
        return {}
    df = batted_df.dropna(subset=["Direction", "Distance"]).copy()
    if df.empty:
        return {}

    angle_rad = np.radians(df["Direction"])
    df["x"] = df["Distance"] * np.sin(angle_rad)
    df["y"] = df["Distance"] * np.cos(angle_rad)

    # Classify direction (Pull/Center/Oppo) based on batter side. Trackman: +Direction = RF.
    if batter_side == "Right":
        df["FieldDir"] = np.where(df["Direction"] < -15, "Pull", np.where(df["Direction"] > 15, "Oppo", "Center"))
    else:
        df["FieldDir"] = np.where(df["Direction"] > 15, "Pull", np.where(df["Direction"] < -15, "Oppo", "Center"))

    # Weight by damage potential: higher EV + hits weighted more
    df["weight"] = 1.0
    if "ExitSpeed" in df.columns:
        has_ev = df["ExitSpeed"].notna()
        df.loc[has_ev, "weight"] = df.loc[has_ev, "ExitSpeed"] / 85.0
    if "PlayResult" in df.columns:
        df.loc[df["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]), "weight"] *= 1.5

    # Ground balls -> infielders; air balls -> outfielders
    if "TaggedHitType" in df.columns:
        gb = df[(df["TaggedHitType"] == "GroundBall") | (df["Distance"] < 180)]
        air = df[(df["TaggedHitType"].isin(["FlyBall", "LineDrive"])) | (df["Distance"] >= 180)]
    else:
        gb = df[df["Distance"] < 180]
        air = df[df["Distance"] >= 180]

    positions: Dict[str, Tuple[float, float]] = {}

    gb_pull = gb[gb["FieldDir"] == "Pull"]
    gb_oppo = gb[gb["FieldDir"] == "Oppo"]

    # Corners
    if batter_side == "Right":
        positions["3B"] = _weighted_centroid(gb_pull, fallback_x=-60, fallback_y=80)
        positions["1B"] = _weighted_centroid(gb_oppo, fallback_x=60, fallback_y=80)
    else:
        positions["3B"] = _weighted_centroid(gb_oppo, fallback_x=-60, fallback_y=80)
        positions["1B"] = _weighted_centroid(gb_pull, fallback_x=60, fallback_y=80)

    # Middle infield (split by x sign)
    ss_df = gb[(gb["x"] < 0) & (gb["Distance"].between(50, 200))]
    b2_df = gb[(gb["x"] >= 0) & (gb["Distance"].between(50, 200))]
    positions["SS"] = _weighted_centroid(ss_df, fallback_x=-40, fallback_y=120)
    positions["2B"] = _weighted_centroid(b2_df, fallback_x=40, fallback_y=120)

    # Outfield
    of_pull = air[air["FieldDir"] == "Pull"]
    of_center = air[air["FieldDir"] == "Center"]
    of_oppo = air[air["FieldDir"] == "Oppo"]
    if batter_side == "Right":
        positions["LF"] = _weighted_centroid(of_pull, fallback_x=-180, fallback_y=220)
        positions["RF"] = _weighted_centroid(of_oppo, fallback_x=180, fallback_y=220)
    else:
        positions["LF"] = _weighted_centroid(of_oppo, fallback_x=-180, fallback_y=220)
        positions["RF"] = _weighted_centroid(of_pull, fallback_x=180, fallback_y=220)
    positions["CF"] = _weighted_centroid(of_center, fallback_x=0, fallback_y=280)
    return positions


def classify_shift(*, pull_pct: float, center_pct: float, oppo_pct: float, gb_pct: float, gb_pull_pct: Optional[float] = None) -> Dict[str, Any]:
    """Classify shift recommendation, mirroring the current defense page thresholds."""
    pull_pct = float(pull_pct) if pd.notna(pull_pct) else 0.0
    center_pct = float(center_pct) if pd.notna(center_pct) else 0.0
    oppo_pct = float(oppo_pct) if pd.notna(oppo_pct) else 0.0
    gb_pct = float(gb_pct) if pd.notna(gb_pct) else 0.0
    gb_pull_pct = float(gb_pull_pct) if gb_pull_pct is not None and pd.notna(gb_pull_pct) else None

    if pull_pct > 45 and gb_pct > 45:
        desc = f"Pull-heavy hitter ({pull_pct:.0f}% pull) with high GB rate ({gb_pct:.0f}%)."
        if gb_pull_pct is not None:
            desc += f" Ground balls go pull-side {gb_pull_pct:.0f}% of the time."
        desc += " Shift infield toward pull side."
        return {"type": "Infield Shift", "color": "#d22d49", "desc": desc}
    if pull_pct > 40:
        return {
            "type": "Shade Pull",
            "color": "#fe6100",
            "desc": f"Moderate pull tendency ({pull_pct:.0f}%). Shade middle infielders toward pull side.",
        }
    if oppo_pct > 40:
        return {
            "type": "Shade Oppo",
            "color": "#1f77b4",
            "desc": f"Oppo-oriented hitter ({oppo_pct:.0f}% opposite field). Shade defense toward opposite field.",
        }
    return {
        "type": "Standard",
        "color": "#2ca02c",
        "desc": f"Balanced spray (Pull: {pull_pct:.0f}%, Center: {center_pct:.0f}%, Oppo: {oppo_pct:.0f}%).",
    }


def _synthetic_batted_balls(
    *,
    batter_side: str,
    pull_pct: float,
    center_pct: float,
    oppo_pct: float,
    gb_pct: float,
    fb_pct: float,
    ld_pct: float,
    pu_pct: float,
    exit_vel: Optional[float],
    n: int = 220,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Normalize percentages; fall back to balanced if missing.
    def _nz(x, fallback):
        return float(x) if pd.notna(x) else float(fallback)

    pull = _nz(pull_pct, 34.0)
    ctr = _nz(center_pct, 33.0)
    opp = _nz(oppo_pct, max(0.0, 100.0 - pull - ctr))
    dir_sum = max(pull + ctr + opp, 1.0)
    dir_probs = np.array([pull, ctr, opp], dtype=float) / dir_sum

    gb = _nz(gb_pct, 44.0)
    fb = _nz(fb_pct, 32.0)
    ld = _nz(ld_pct, 18.0)
    pu = _nz(pu_pct, max(0.0, 100.0 - gb - fb - ld))
    ht_sum = max(gb + fb + ld + pu, 1.0)
    ht_probs = np.array([gb, fb, ld, pu], dtype=float) / ht_sum

    # Direction category -> representative angle (degrees).
    # Trackman: positive = RF.
    if batter_side == "Right":
        angles = {"Pull": -30.0, "Center": 0.0, "Oppo": 30.0}
    else:
        angles = {"Pull": 30.0, "Center": 0.0, "Oppo": -30.0}
    dir_labels = np.array(["Pull", "Center", "Oppo"])
    ht_labels = np.array(["GroundBall", "FlyBall", "LineDrive", "Popup"])

    dirs = rng.choice(dir_labels, size=n, p=dir_probs)
    hts = rng.choice(ht_labels, size=n, p=ht_probs)

    base_ev = float(exit_vel) if exit_vel is not None and pd.notna(exit_vel) else 85.0
    ev = np.clip(rng.normal(loc=base_ev, scale=4.5, size=n), 60.0, 110.0)

    # Simple distance model by hit type + EV scaling.
    dist = np.zeros(n, dtype=float)
    for i, ht in enumerate(hts):
        ev_scale = (ev[i] - 85.0) / 15.0  # roughly [-1, +1]
        if ht == "GroundBall":
            dist[i] = rng.uniform(40, 140) + 12.0 * ev_scale
        elif ht == "LineDrive":
            dist[i] = rng.uniform(180, 260) + 18.0 * ev_scale
        elif ht == "FlyBall":
            dist[i] = rng.uniform(230, 330) + 25.0 * ev_scale
        else:  # Popup
            dist[i] = rng.uniform(120, 210) + 10.0 * ev_scale
    dist = np.clip(dist, 20.0, 420.0)

    ang = np.array([angles[d] for d in dirs], dtype=float) + rng.normal(0.0, 6.0, size=n)
    return pd.DataFrame(
        {
            "Direction": ang,
            "Distance": dist,
            "TaggedHitType": hts,
            "ExitSpeed": ev,
            "PlayResult": "Out",  # not used heavily for positioning; keeps weights stable
        }
    )


@dataclass(frozen=True)
class DefenseRecommendation:
    shift: Dict[str, Any]
    positions: Dict[str, Dict[str, float]]  # pos -> {"x":..., "y":...}
    overlay: Dict[str, Any]


def recommend_defense_from_truemedia(
    *,
    batter_side: str,
    pull_pct: float,
    center_pct: float,
    oppo_pct: float,
    gb_pct: float,
    fb_pct: float,
    ld_pct: float,
    pu_pct: float,
    exit_vel: Optional[float],
    state: GameState,
) -> DefenseRecommendation:
    batted = _synthetic_batted_balls(
        batter_side=batter_side,
        pull_pct=pull_pct,
        center_pct=center_pct,
        oppo_pct=oppo_pct,
        gb_pct=gb_pct,
        fb_pct=fb_pct,
        ld_pct=ld_pct,
        pu_pct=pu_pct,
        exit_vel=exit_vel,
    )
    pos_xy = _recommend_fielder_positions(batted, batter_side=batter_side)
    shift = classify_shift(pull_pct=pull_pct, center_pct=center_pct, oppo_pct=oppo_pct, gb_pct=gb_pct, gb_pull_pct=None)
    positions = {p: {"x": float(x), "y": float(y)} for p, (x, y) in pos_xy.items()}
    overlay = apply_situation_overlay(state, positions)
    return DefenseRecommendation(shift=shift, positions=positions, overlay=overlay)


def apply_situation_overlay(state: GameState, positions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Apply game-state adjustments to a base defensive positioning dict."""
    overlay: Dict[str, Any] = {"type": "Standard", "notes": []}
    outs = int(state.outs)
    bases = state.bases

    # DP depth: R1, <2 outs
    if bases.on_1b and outs < 2:
        overlay["type"] = "DP Depth"
        overlay["notes"].append("Middle infielders at DP depth")
        for pos in ["SS", "2B"]:
            if pos in positions:
                positions[pos]["y"] = float(min(positions[pos]["y"], 115.0))

    # Infield in: R3, <2 outs
    if bases.on_3b and outs < 2:
        overlay["type"] = "Infield In"
        overlay["notes"].append("Cut off run at plate")
        for pos in ["1B", "2B", "SS", "3B"]:
            if pos in positions:
                positions[pos]["y"] = 85.0

    # No doubles: late innings, runner in scoring position
    if bases.risp and int(state.inning) >= 7:
        overlay["notes"].append("No doubles: OF deep, near lines")
        for pos in ["LF", "RF"]:
            if pos in positions:
                positions[pos]["y"] = float(max(positions[pos]["y"], 280.0))

    # Hold runner at 1B
    if bases.on_1b and "1B" in positions:
        overlay["notes"].append("1B holds runner")
        positions["1B"]["y"] = 65.0

    return overlay


def shift_value_delta_re(
    standard_contact_rv: float,
    shifted_contact_rv: float,
    p_in_play: float,
) -> float:
    """Express defensive shift value as ΔRE.

    Computes how the shift changes the expected run value on contact
    by altering the hit→out conversion rate.

    Parameters
    ----------
    standard_contact_rv : float
        Expected wOBA on contact with standard positioning.
    shifted_contact_rv : float
        Expected wOBA on contact with the recommended shift.
    p_in_play : float
        Probability of a ball in play for this matchup.

    Returns
    -------
    float
        ΔRE from the shift. Negative = shift saves runs (good).
    """
    return p_in_play * (shifted_contact_rv - standard_contact_rv)


def estimate_shift_contact_rv(
    *,
    shift_type: str,
    pull_pct: float,
    gb_pct: float,
    standard_contact_rv: float,
) -> float:
    """Estimate how a defensive shift changes contact run value.

    The shift improves conversion on pull-side ground balls but may
    concede hits on opposite-field contact. Net effect depends on
    pull/GB tendencies.

    Returns the adjusted contact_rv (lower = better for defense).
    """
    import pandas as pd
    pull_pct = float(pull_pct) if pd.notna(pull_pct) else 34.0
    gb_pct = float(gb_pct) if pd.notna(gb_pct) else 44.0

    if shift_type == "Standard":
        return standard_contact_rv

    # Pull tendency factor: how much of contact goes to the shifted side
    pull_factor = max(0.0, (pull_pct - 33.0) / 67.0)  # 0 at balanced, ~1 at extreme pull

    # GB factor: ground balls are most affected by infield shifts
    gb_factor = max(0.0, (gb_pct - 30.0) / 40.0)  # 0 at low GB, ~1 at extreme GB

    if shift_type == "Infield Shift":
        # Strong shift: big benefit on pull-side GBs, small cost on oppo
        benefit = 0.04 * pull_factor * gb_factor  # up to ~0.04 wOBA saved
        cost = 0.01 * (1.0 - pull_factor)  # small oppo-side cost
        return standard_contact_rv - benefit + cost

    elif shift_type == "Shade Pull":
        benefit = 0.02 * pull_factor * gb_factor
        cost = 0.005 * (1.0 - pull_factor)
        return standard_contact_rv - benefit + cost

    elif shift_type == "Shade Oppo":
        # Shade toward opposite field for oppo-oriented hitters
        oppo_factor = max(0.0, (1.0 - pull_factor))
        benefit = 0.015 * oppo_factor
        return standard_contact_rv - benefit

    return standard_contact_rv


def pitch_defense_mismatch(
    *,
    pitch_name: str,
    location_zone: Optional[str],
    defense_shift_type: str,
) -> Optional[str]:
    """Return a coach-facing warning when a pitch choice fights the defensive alignment.

    location_zone: a zone label string like "Up-Away", "Low-In", "Middle",
    or legacy labels like "glove side", "below zone".
    """
    shift = (defense_shift_type or "").strip()
    if not shift or shift == "Standard":
        return None

    zl = (location_zone or "").lower()

    # Map new 3×3 labels + legacy labels to conceptual regions
    is_away = "away" in zl or "glove" in zl
    is_low = "low" in zl or "below" in zl or "chase" in zl
    is_down_arm = "down" in zl or "arm" in zl or "low-in" in zl

    # Heuristic: breaking balls expanded away / low increase opposite-field contact risk.
    breaking = {"Slider", "Sweeper", "Curveball", "Knuckle Curve"}
    if shift in {"Infield Shift", "Shade Pull"} and pitch_name in breaking and (is_away or is_low):
        return (
            f"Pitch-defense mismatch: {pitch_name} to {location_zone} vs a pull-leaning alignment. "
            "If contact, ball may be served oppo into the shift hole."
        )

    # Heuristic: sinker/CH for GB aligns with DP/pull shades.
    if shift == "Shade Oppo" and pitch_name in {"Sinker", "Changeup", "Splitter"} and (is_down_arm or location_zone is None):
        return (
            f"Pitch-defense mismatch: {pitch_name} ground-ball bias vs an oppo shade. "
            "Consider standard or pull-side alignment if expecting GB pull."
        )

    return None

