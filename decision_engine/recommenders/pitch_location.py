from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from decision_engine.core.state import GameState


_HARD_PITCHES = {"Fastball", "Sinker", "Cutter"}

_ZONE_LABELS = {
    "up": "up in zone",
    "down": "down in zone",
    "glove": "glove side",
    "arm": "arm side",
    "chase_low": "below zone",
}


def _score_zone(z: Dict[str, Any], *, mode: str) -> float:
    wh = z.get("whiff_pct", np.nan)
    csw = z.get("csw_pct", np.nan)
    ev = z.get("ev_against", np.nan)

    wh = float(wh) if pd.notna(wh) else np.nan
    csw = float(csw) if pd.notna(csw) else np.nan
    ev = float(ev) if pd.notna(ev) else np.nan

    # Base weights: different goals in different counts.
    if mode == "putaway":
        wh_w, csw_w, ev_w = 1.2, 0.3, 0.4
    elif mode == "must_strike":
        wh_w, csw_w, ev_w = 0.2, 1.0, 0.5
    else:
        wh_w, csw_w, ev_w = 0.4, 0.8, 0.4

    s = 0.0
    if pd.notna(wh):
        s += wh_w * wh
    if pd.notna(csw):
        s += csw_w * csw
    if pd.notna(ev):
        s += ev_w * (92.0 - ev)  # lower EV is better
    return float(s)


def recommend_pitch_location(
    pitch_name: str,
    arsenal_pitch: Dict[str, Any],
    state: GameState,
) -> Optional[Dict[str, Any]]:
    """Location suggestion from the pitcher's own `zone_eff` history."""
    ze = arsenal_pitch.get("zone_eff") or {}
    if not isinstance(ze, dict) or not ze:
        return None

    b, s = state.count()
    two_strike = s == 2
    must_strike = b == 3 or (b >= 2 and b > s and not two_strike)
    mode = "neutral"
    if must_strike:
        mode = "must_strike"
    elif two_strike:
        mode = "putaway"

    best_zn = None
    best_score = -1e18
    for zn, zdata in ze.items():
        if not isinstance(zdata, dict):
            continue
        n = int(zdata.get("n", 0) or 0)
        if n < 5:
            continue
        if mode == "must_strike" and zn == "chase_low":
            continue
        score = _score_zone(zdata, mode=mode)
        # In 0-2, explicitly encourage chase expansions if supported.
        if two_strike and b == 0 and zn == "chase_low":
            score *= 1.15
        if score > best_score:
            best_score = score
            best_zn = zn

    if best_zn is None:
        return None

    out = dict(ze[best_zn])
    out.update(
        {
            "pitch": pitch_name,
            "mode": mode,
            "zone": best_zn,
            "zone_label": _ZONE_LABELS.get(best_zn, best_zn),
        }
    )
    return out

