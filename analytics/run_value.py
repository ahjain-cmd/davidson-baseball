"""Per-pitch run value (ΔRE) computation for Stuff+ and Location+ targets.

Each pitch gets a **PitchRV** based on count-state transitions:
- Non-terminal pitches: count_rv[next_count] - count_rv[current_count]
- Strikeout: -count_rv[current_count]  (out made, negative = good for pitcher)
- Walk/HBP: +linear_weights["Single"] - count_rv[current_count]
- InPlay: linear_weights[PlayResult] - count_rv[current_count]

Sign convention: negative PitchRV = good for pitcher.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

from config import CACHE_DIR


# ── Fallback count run values (from run_expectancy.py) ────────────────────────
_FALLBACK_COUNT_RV: Dict[str, float] = {
    "0-0": 0.28, "0-1": 0.21, "0-2": 0.10,
    "1-0": 0.33, "1-1": 0.27, "1-2": 0.15,
    "2-0": 0.39, "2-1": 0.34, "2-2": 0.21,
    "3-0": 0.43, "3-1": 0.40, "3-2": 0.28,
}

# ── Fallback linear weights ──────────────────────────────────────────────────
_FALLBACK_LINEAR_WEIGHTS: Dict[str, float] = {
    "Out": 0.0, "FieldersChoice": 0.0, "Sacrifice": 0.0,
    "Single": 0.47, "Double": 0.78, "Triple": 1.05, "HomeRun": 1.40,
    "Error": 0.47,
}


def _load_count_rv() -> Dict[str, float]:
    """Load count run values from cached count_calibration.json."""
    import json
    try:
        cache_file = os.path.join(CACHE_DIR, "count_calibration.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                blob = json.load(f)
            rv = blob.get("calibration", {}).get("run_values")
            if rv:
                return rv
    except Exception:
        pass
    return _FALLBACK_COUNT_RV


def _load_linear_weights() -> Dict[str, float]:
    """Load linear weights from cached historical_calibration.json."""
    import json
    try:
        cache_file = os.path.join(CACHE_DIR, "historical_calibration.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                blob = json.load(f)
            lw = blob.get("calibration", {}).get("linear_weights")
            if lw:
                return {
                    "Out": 0.0, "FieldersChoice": 0.0, "Sacrifice": 0.0,
                    "Single": lw.get("single_w", 0.47),
                    "Double": lw.get("double_w", 0.78),
                    "Triple": lw.get("triple_w", 1.05),
                    "HomeRun": lw.get("hr_w", 1.40),
                    "Error": lw.get("single_w", 0.47),
                }
    except Exception:
        pass
    return _FALLBACK_LINEAR_WEIGHTS


def compute_pitch_run_values(df: pd.DataFrame) -> pd.DataFrame:
    """Add PitchRV column to a pitch-level DataFrame.

    Required columns: Balls, Strikes, PitchCall.
    Optional columns: PlayResult, KorBB.

    Returns the same DataFrame with a new ``PitchRV`` column.
    Negative values = good for pitcher.
    """
    if df.empty or "PitchCall" not in df.columns:
        df["PitchRV"] = np.nan
        return df

    count_rv = _load_count_rv()
    lw = _load_linear_weights()

    balls = pd.to_numeric(df["Balls"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(df["Strikes"], errors="coerce").fillna(0).astype(int)
    count_key = balls.astype(str) + "-" + strikes.astype(str)
    current_rv = count_key.map(count_rv).fillna(0.28)  # default to 0-0

    pitch_call = df["PitchCall"].fillna("")
    play_result = df["PlayResult"].fillna("") if "PlayResult" in df.columns else pd.Series("", index=df.index)
    kor_bb = df["KorBB"].fillna("") if "KorBB" in df.columns else pd.Series("", index=df.index)

    rv = pd.Series(np.nan, index=df.index, dtype=float)

    # ── Terminal events ──────────────────────────────────────────────────────

    # Strikeout: -current_rv (pitcher saved those expected runs)
    is_k = kor_bb.isin(["Strikeout", "K"])
    rv[is_k] = -current_rv[is_k]

    # Walk: +lw["Single"] - current_rv (runner on, approximate)
    is_bb = kor_bb.isin(["Walk", "BB"])
    rv[is_bb] = lw["Single"] - current_rv[is_bb]

    # HBP: same as walk
    is_hbp = pitch_call == "HitByPitch"
    rv[is_hbp] = lw["Single"] - current_rv[is_hbp]

    # InPlay: linear_weights[PlayResult] - current_rv
    is_ip = pitch_call == "InPlay"
    ip_rv = play_result.map(lw).fillna(0.0)
    rv[is_ip & rv.isna()] = (ip_rv - current_rv)[is_ip & rv.isna()]

    # ── Non-terminal events (count transitions) ──────────────────────────────

    # Ball (non-walk): next count RV - current count RV
    ball_calls = {"BallCalled", "BallinDirt", "BallIntentional"}
    is_ball = pitch_call.isin(ball_calls) & rv.isna()
    next_balls = (balls + 1).clip(upper=3)
    next_count_ball = next_balls.astype(str) + "-" + strikes.astype(str)
    next_rv_ball = next_count_ball.map(count_rv).fillna(0.28)
    rv[is_ball] = (next_rv_ball - current_rv)[is_ball]

    # Called strike: next count RV - current count RV
    is_cs = (pitch_call == "StrikeCalled") & rv.isna()
    next_strikes_cs = (strikes + 1).clip(upper=2)
    next_count_cs = balls.astype(str) + "-" + next_strikes_cs.astype(str)
    next_rv_cs = next_count_cs.map(count_rv).fillna(0.21)
    rv[is_cs] = (next_rv_cs - current_rv)[is_cs]

    # Swinging strike (non-K): next count RV - current count RV
    is_ss = (pitch_call == "StrikeSwinging") & rv.isna()
    next_strikes_ss = (strikes + 1).clip(upper=2)
    next_count_ss = balls.astype(str) + "-" + next_strikes_ss.astype(str)
    next_rv_ss = next_count_ss.map(count_rv).fillna(0.21)
    rv[is_ss] = (next_rv_ss - current_rv)[is_ss]

    # Foul: if strikes < 2, count changes; if strikes == 2, no change (rv=0)
    foul_calls = {"FoulBall", "FoulBallNotFieldable", "FoulBallFieldable"}
    is_foul = pitch_call.isin(foul_calls) & rv.isna()
    foul_advances = is_foul & (strikes < 2)
    foul_no_change = is_foul & (strikes >= 2)

    next_strikes_f = (strikes + 1).clip(upper=2)
    next_count_f = balls.astype(str) + "-" + next_strikes_f.astype(str)
    next_rv_f = next_count_f.map(count_rv).fillna(0.21)
    rv[foul_advances] = (next_rv_f - current_rv)[foul_advances]
    rv[foul_no_change] = 0.0

    # Anything still NaN gets 0 (unknown pitch call)
    rv = rv.fillna(0.0)

    df = df.copy()
    df["PitchRV"] = rv
    return df
