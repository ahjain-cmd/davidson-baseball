from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def _is_bad(x: Any) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def speed_tier_from_speed_score(speed_score: Optional[float]) -> str:
    """Return runner speed tier from D1 SpeedScore."""
    if _is_bad(speed_score):
        return "NONE"
    try:
        ss = float(speed_score)
    except Exception:
        return "NONE"
    if ss >= 6.0:
        return "ELITE"
    if ss >= 4.5:
        return "FAST"
    if ss >= 3.0:
        return "AVG"
    return "SLOW"


def speed_tier_from_sb(sb: Optional[int]) -> str:
    """Fallback tiering when SpeedScore is unavailable."""
    if sb is None:
        return "NONE"
    try:
        n = int(sb)
    except Exception:
        return "NONE"
    if n >= 15:
        return "ELITE"
    if n >= 8:
        return "FAST"
    if n >= 3:
        return "AVG"
    return "SLOW"


def steal_pressure(
    runner_speed_tier: str,
    catcher_pop_time: Optional[float],
    pitcher_sb_pct: Optional[float],
    base: str,
) -> float:
    """Return a pressure score from 0.0 (no threat) to 1.0 (extreme threat)."""
    tier = (runner_speed_tier or "NONE").upper()
    speed_factor = {"ELITE": 1.0, "FAST": 0.7, "AVG": 0.35, "SLOW": 0.0, "NONE": 0.0}.get(tier, 0.0)

    # Catcher arm modifier (D1 avg PopTime ~2.03s from your data export).
    catcher_mod = 1.0
    if not _is_bad(catcher_pop_time):
        try:
            pt = float(catcher_pop_time)
        except Exception:
            pt = None
        if pt is not None:
            if pt <= 1.90:
                catcher_mod = 0.6
            elif pt <= 2.00:
                catcher_mod = 0.8
            elif pt <= 2.10:
                catcher_mod = 1.0
            elif pt <= 2.20:
                catcher_mod = 1.2
            else:
                catcher_mod = 1.4

    # Pitcher hold modifier (SB% allowed, higher is worse).
    pitcher_mod = 1.0
    if not _is_bad(pitcher_sb_pct):
        try:
            sbp = float(pitcher_sb_pct)
        except Exception:
            sbp = None
        if sbp is not None:
            if sbp > 85:
                pitcher_mod = 1.2
            elif sbp < 65:
                pitcher_mod = 0.7

    base_norm = (base or "").strip().upper()
    base_mod = 1.0 if base_norm in {"1B", "1", "FIRST"} else 0.6

    return float(min(1.0, speed_factor * catcher_mod * pitcher_mod * base_mod))


@dataclass(frozen=True)
class RunnerContext:
    """Baserunning threat context for decision engine."""

    r1_speed_tier: str = "NONE"  # "ELITE", "FAST", "AVG", "SLOW", "NONE"
    r1_speed_score: float = float("nan")
    r1_sb_count: int = 0
    r1_sb_pct: float = float("nan")

    r2_speed_tier: str = "NONE"
    r2_speed_score: float = float("nan")
    r2_sb_count: int = 0
    r2_sb_pct: float = float("nan")

    r3_speed_tier: str = "NONE"

    # Squeeze/bunt threat (batter tendency).
    batter_squeeze_count: int = 0
    batter_bunt_hit_att: int = 0

    # Our control
    catcher_pop_time: float = 2.03  # D1 average as default
    pitcher_sb_pct: float = 76.2    # D1 average from historical calibration
    pitcher_pk_rate: float = 0.0    # pickoff attempts per runner on base

    def steal_context(self, *, on_1b: bool, on_2b: bool) -> Dict[str, Any]:
        """Return a normalized steal context dict for pitch adjustments."""
        p1 = steal_pressure(self.r1_speed_tier, self.catcher_pop_time, self.pitcher_sb_pct, base="1B") if on_1b else 0.0
        p2 = steal_pressure(self.r2_speed_tier, self.catcher_pop_time, self.pitcher_sb_pct, base="2B") if on_2b else 0.0
        return {
            "pressure": float(max(p1, p2)),
            "p1": float(p1),
            "p2": float(p2),
            "on_2b_elite": bool(on_2b and (self.r2_speed_tier or "").upper() == "ELITE"),
        }

    def squeeze_context(self, *, on_3b: bool, outs: int) -> Dict[str, Any]:
        """Return squeeze/bunt threat for R3 + <2 outs."""
        if not on_3b or int(outs) >= 2:
            return {"active": False, "threat": 0.0}
        squeeze = int(self.batter_squeeze_count or 0)
        bunt = int(self.batter_bunt_hit_att or 0)
        active = (squeeze >= 2) or (bunt >= 3)
        threat = 1.0 if active else 0.0
        return {"active": active, "threat": float(threat), "squeeze": squeeze, "bunt_hit_att": bunt}

