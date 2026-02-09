from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from decision_engine.core.state import GameState


_HARD_PITCHES = {"Fastball", "Sinker", "Cutter"}
_OFFSPEED_PITCHES = {"Slider", "Curveball", "Changeup", "Splitter", "Sweeper", "Knuckle Curve"}

# ── Pitch-design zone multipliers ────────────────────────────────────────────
# (xb, yb) -> multiplier per pitch class.  Values > 1.0 = zone is natural for
# this pitch design; < 1.0 = zone fights the pitch's movement profile.
# Applied to pitcher_eff before combining with hitter_vuln so that each pitch
# retains its natural location tendency even when hitter data is present.
_PITCH_DESIGN_ZONE_MULT = {
    "Fastball": {
        (0, 2): 1.3, (1, 2): 1.3, (2, 2): 1.3,   # up = ride / miss barrels
        (0, 1): 1.0, (1, 1): 0.9, (2, 1): 1.0,   # middle = ok
        (0, 0): 0.7, (1, 0): 0.7, (2, 0): 0.7,   # low = fights design
    },
    "Sinker": {
        (0, 2): 0.7, (1, 2): 0.8, (2, 2): 0.7,   # up = fights sink
        (0, 1): 1.1, (1, 1): 1.0, (2, 1): 1.1,   # middle = ok
        (0, 0): 1.3, (1, 0): 1.3, (2, 0): 1.2,   # low = natural sink
    },
    "Slider": {
        (0, 2): 0.7, (1, 2): 0.7, (2, 2): 0.8,   # up = risky
        (0, 1): 1.1, (1, 1): 0.9, (2, 1): 0.9,   # mid-in = backdoor
        (0, 0): 1.3, (1, 0): 1.1, (2, 0): 1.0,   # low-in = sweep / bury
    },
    "Curveball": {
        (0, 2): 0.6, (1, 2): 0.6, (2, 2): 0.6,   # up = hangs
        (0, 1): 0.9, (1, 1): 1.0, (2, 1): 0.9,   # middle = ok
        (0, 0): 1.3, (1, 0): 1.4, (2, 0): 1.3,   # low = bury
    },
    "Changeup": {
        (0, 2): 0.6, (1, 2): 0.6, (2, 2): 0.6,   # up = mistake
        (0, 1): 0.9, (1, 1): 0.9, (2, 1): 1.1,   # mid-away = fade
        (0, 0): 1.2, (1, 0): 1.3, (2, 0): 1.3,   # low = natural fade
    },
}
# Hard / offspeed defaults for pitch types not in the table above
_PZM_HARD_DEFAULT = {
    (0, 2): 1.1, (1, 2): 1.1, (2, 2): 1.1,
    (0, 1): 1.0, (1, 1): 1.0, (2, 1): 1.0,
    (0, 0): 0.9, (1, 0): 0.9, (2, 0): 0.9,
}
_PZM_OFFSPEED_DEFAULT = {
    (0, 2): 0.7, (1, 2): 0.7, (2, 2): 0.7,
    (0, 1): 1.0, (1, 1): 1.0, (2, 1): 1.0,
    (0, 0): 1.2, (1, 0): 1.2, (2, 0): 1.2,
}


def _get_pzm(pitch_name: str, xb: int, yb: int) -> float:
    """Return the pitch-design zone multiplier for a given pitch and cell."""
    table = _PITCH_DESIGN_ZONE_MULT.get(pitch_name)
    if table is None:
        table = _PZM_HARD_DEFAULT if pitch_name in _HARD_PITCHES else _PZM_OFFSPEED_DEFAULT
    return table.get((xb, yb), 1.0)


# ── Legacy 5-zone labels (kept for backward compat display) ──────────────────
_ZONE_LABELS = {
    "up": "up in zone",
    "down": "down in zone",
    "glove": "glove side",
    "arm": "arm side",
    "chase_low": "below zone",
}

# ── 3×3 grid labels: (RHH label, LHH label) ─────────────────────────────────
_3x3_LABELS: Dict[Tuple[int, int], Tuple[str, str]] = {
    (0, 2): ("Up-In", "Up-Away"),
    (1, 2): ("Up-Middle", "Up-Middle"),
    (2, 2): ("Up-Away", "Up-In"),
    (0, 1): ("Mid-In", "Mid-Away"),
    (1, 1): ("Middle", "Middle"),
    (2, 1): ("Mid-Away", "Mid-In"),
    (0, 0): ("Low-In", "Low-Away"),
    (1, 0): ("Low-Middle", "Low-Middle"),
    (2, 0): ("Low-Away", "Low-In"),
}

_CHASE_LABELS = {
    "chase_low": "Chase Low",
    "chase_away": "Chase Away",
}


def _fmt_bats(bats) -> str:
    if bats is None or (isinstance(bats, float) and pd.isna(bats)):
        return "R"
    b = str(bats).strip()
    mapping = {"Right": "R", "Left": "L", "Both": "S", "Switch": "S", "R": "R", "L": "L", "S": "S"}
    return mapping.get(b, "R")


def _cell_label(xb: int, yb: int, bats: str) -> str:
    """Human-readable label for a 3×3 cell, adjusted for batter side."""
    pair = _3x3_LABELS.get((xb, yb))
    if pair is None:
        return f"({xb},{yb})"
    rhh_label, lhh_label = pair
    return lhh_label if _fmt_bats(bats) == "L" else rhh_label


# ── Pitcher effectiveness scoring (same logic as before) ─────────────────────

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
    elif mode == "hybrid_3-2":
        # 3-2 is simultaneously must-compete and high-leverage putaway.
        wh_w, csw_w, ev_w = 0.8, 0.7, 0.5
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


# ── Hitter vulnerability scoring ─────────────────────────────────────────────

def _hitter_vuln_score(xb: int, yb: int, hzv: Dict[str, Any], bats: str) -> float:
    """Interpolate a vulnerability score for a 3×3 cell from directional vulns.

    hzv keys: vuln_up, vuln_down, vuln_inside, vuln_away (0-100 each).
    Returns a value in ~0-100 range.
    """
    vu = float(hzv.get("vuln_up", 50) or 50)
    vd = float(hzv.get("vuln_down", 50) or 50)
    vi = float(hzv.get("vuln_inside", 50) or 50)
    va = float(hzv.get("vuln_away", 50) or 50)

    # Determine which column is inside / away for this batter
    b = _fmt_bats(bats)
    # PlateLocSide: + = 1B side. For RHH: 1B-side is away, for LHH: 1B-side is inside.
    # xbin 0 = left (3B side), xbin 2 = right (1B side)
    if b == "L":
        # LHH: xbin 0 = away, xbin 2 = inside
        col_vals = {0: va, 1: (vi + va) / 2, 2: vi}
    else:
        # RHH: xbin 0 = inside, xbin 2 = away
        col_vals = {0: vi, 1: (vi + va) / 2, 2: va}

    row_vals = {0: vd, 1: (vu + vd) / 2, 2: vu}

    return (row_vals[yb] + col_vals[xb]) / 2


# ── Count bonus ──────────────────────────────────────────────────────────────

def _count_bonus(xb: int, yb: int, mode: str) -> float:
    """Small bonus/penalty based on count situation and cell location."""
    # yb=0 is low (bottom row), yb=2 is high (top row); xb=1 is middle column
    is_bottom = (yb == 0)
    is_middle = (xb == 1 and yb == 1)
    is_edge = (xb in (0, 2)) or (yb in (0, 2))

    if mode == "putaway":
        # Reward low zones for putaway
        if is_bottom:
            return 15.0
        if yb == 1 and is_edge:
            return 5.0
        if is_middle:
            return -5.0
        return 0.0
    elif mode == "must_strike":
        # Suppress edges, reward middle of zone
        if is_middle:
            return 10.0
        if yb == 1:
            return 5.0
        if is_bottom:
            return -15.0
        return 0.0
    elif mode == "hybrid_3-2":
        # Blend: prefer edges / below, but not as extreme as 0-2.
        if is_bottom:
            return 8.0
        if yb == 1 and is_edge:
            return 3.0
        if is_middle:
            return -2.0
        return 0.0
    else:
        return 0.0


# ── Fallback: interpolate 5-zone → 3×3 ──────────────────────────────────────

def _fallback_3x3_from_5zone(ze: Dict[str, Dict], throws: str) -> Dict[Tuple[int, int], Dict]:
    """Map legacy 5-zone zone_eff data onto the 3×3 grid for backward compat."""
    grid: Dict[Tuple[int, int], Dict] = {}
    up = ze.get("up", {})
    down = ze.get("down", {})
    glove = ze.get("glove", {})
    arm = ze.get("arm", {})
    chase_low = ze.get("chase_low", {})

    # Determine column mapping from pitcher throws perspective
    # RHP glove = 3B side = xbin 0; RHP arm = 1B side = xbin 2
    # LHP glove = 1B side = xbin 2; LHP arm = 3B side = xbin 0
    t = str(throws).strip()
    if t.startswith("L"):
        glove_col, arm_col = 2, 0
    else:
        glove_col, arm_col = 0, 2

    def _avg_dicts(*dicts):
        """Average the metrics from multiple zone dicts."""
        result = {}
        for key in ("n", "whiff_pct", "csw_pct", "ev_against"):
            vals = [float(d.get(key, np.nan) or np.nan) for d in dicts if d]
            vals = [v for v in vals if pd.notna(v)]
            if vals:
                result[key] = sum(vals) / len(vals) if key != "n" else int(sum(vals))
            else:
                result[key] = np.nan
        return result

    # Top row
    grid[(glove_col, 2)] = _avg_dicts(up, glove) if up or glove else {}
    grid[(1, 2)] = dict(up) if up else {}
    grid[(arm_col, 2)] = _avg_dicts(up, arm) if up or arm else {}

    # Middle row
    grid[(glove_col, 1)] = dict(glove) if glove else {}
    grid[(1, 1)] = _avg_dicts(up, down, glove, arm)
    grid[(arm_col, 1)] = dict(arm) if arm else {}

    # Bottom row
    grid[(glove_col, 0)] = _avg_dicts(down, glove) if down or glove else {}
    grid[(1, 0)] = _avg_dicts(down, chase_low) if down or chase_low else {}
    grid[(arm_col, 0)] = _avg_dicts(down, arm) if down or arm else {}

    # Filter out empties and nan-n cells
    return {k: v for k, v in grid.items() if v and pd.notna(v.get("n")) and v.get("n", 0) > 0}


# ── Adjacency check for secondary location ──────────────────────────────────

def _is_adjacent(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1


# ── Count-zone adjustment ────────────────────────────────────────────────────

def _count_zone_adjustment(
    xb: int,
    yb: int,
    count_metrics: Dict[str, Dict[Tuple[int, int], Dict]],
    mode: str,
    balls: int,
    strikes: int,
) -> float:
    """Data-driven count-zone adjustment (clamped to [-20, +20]).

    Uses count-zone metrics to add bonus/penalty based on what the hitter
    actually does in the current count group at this zone.
    """
    if not count_metrics:
        return 0.0

    # Map current count to a count group
    if strikes == 2:
        cg = "two_strike"
    elif balls >= 2 and balls > strikes:
        cg = "hitter_count"
    else:
        cg = "early"

    cg_data = count_metrics.get(cg, {})
    cell = cg_data.get((xb, yb))
    if cell is None:
        return 0.0

    swing_rate = cell.get("swing_rate", 0.5)
    whiff_rate = cell.get("whiff_rate", 0.0)
    slg = cell.get("slg", 0.3)

    adj = 0.0

    if mode not in ("putaway", "must_strike", "hybrid_3-2"):
        # Early counts: reward low swing_rate zones (free called strikes)
        # and low SLG zones (safe to attack)
        adj += (1.0 - swing_rate) * 12.0  # up to +12 for zones they ignore
        adj += (1.0 - min(slg / 1.0, 1.0)) * 8.0  # up to +8 for low SLG
    elif mode == "putaway":
        # Two-strike: reward high whiff zones
        adj += whiff_rate * 15.0  # up to +15 for high whiff zones
        adj += (1.0 - min(slg / 1.0, 1.0)) * 5.0  # small bonus for low SLG
    elif mode == "must_strike":
        # Behind in count: target lowest SLG zones (minimize damage)
        adj += (1.0 - min(slg / 1.0, 1.0)) * 15.0  # up to +15 for low SLG
        adj -= swing_rate * 5.0  # mild penalty for zones they swing at (they'll make contact)
    elif mode == "hybrid_3-2":
        # 3-2 blend: whiff bonus + low SLG
        adj += whiff_rate * 10.0
        adj += (1.0 - min(slg / 1.0, 1.0)) * 10.0

    return float(np.clip(adj, -20.0, 20.0))


# ── Main recommender ─────────────────────────────────────────────────────────

def recommend_pitch_location(
    pitch_name: str,
    arsenal_pitch: Dict[str, Any],
    state: GameState,
    *,
    hitter_zone_vuln: Optional[Dict] = None,
    hitter_data: Optional[Dict] = None,
    bats: str = "R",
    throws: str = "R",
) -> Optional[Dict[str, Any]]:
    """Location suggestion combining pitcher zone effectiveness and hitter vulnerability.

    Uses a pitch-type-specific fallback chain for hitter hole scores and
    count-aware zone adjustments for data-driven count context.

    Returns a dict with primary and optional secondary location recommendations,
    including zone label, metrics, and a reason string.
    """
    # Get 3×3 zone effectiveness data (prefer fine grid, fallback to 5-zone)
    ze3 = arsenal_pitch.get("zone_eff_3x3") or {}
    if not isinstance(ze3, dict) or not ze3:
        ze5 = arsenal_pitch.get("zone_eff") or {}
        if not isinstance(ze5, dict) or not ze5:
            return None
        ze3 = _fallback_3x3_from_5zone(ze5, throws)
        if not ze3:
            return None

    # Determine count mode
    b, s = state.count()
    two_strike = s == 2
    must_strike = b == 3 or (b >= 2 and b > s and not two_strike)
    mode = "neutral"
    if b == 3 and s == 2:
        mode = "hybrid_3-2"
    elif must_strike:
        mode = "must_strike"
    elif two_strike:
        mode = "putaway"

    # Check if we have hitter vulnerability data (must have 'available' flag set)
    has_hitter = bool(
        hitter_zone_vuln
        and isinstance(hitter_zone_vuln, dict)
        and hitter_zone_vuln.get("available", False)
    )

    # ── Pitch-type hole score lookup with fallback chain ─────────────────────
    # Priority: 1. Pitch-type-specific  2. Aggregate ("ALL")  3. Directional zone_vuln  4. Pitcher-only
    hole_scores_by_pt = (hitter_data or {}).get("hole_scores_by_pt", {})
    hole_scores_agg = (hitter_data or {}).get("hole_scores_3x3", {})

    pt_holes = hole_scores_by_pt.get(pitch_name, {})
    has_pt_holes = bool(pt_holes)
    has_agg_holes = bool(hole_scores_agg)

    # Count-zone metrics
    count_metrics = (hitter_data or {}).get("count_zone_metrics", {})

    # ── Stuff+-driven weight allocation with pitch-type tier boost ───────────
    stuff_plus = arsenal_pitch.get("stuff_plus", np.nan)
    stuff_plus = float(stuff_plus) if pd.notna(stuff_plus) else 100.0
    sp_clamped = max(90.0, min(120.0, stuff_plus))

    if pitch_name in _OFFSPEED_PITCHES:
        # Secondary pitches: boost pitcher weight — command matters more for
        # breaking balls, and pitcher whiff/CSW zones reflect pitch shape.
        if has_pt_holes:
            w_hitter = float(np.interp(sp_clamped, [90, 105, 120], [0.45, 0.30, 0.15]))
        elif has_agg_holes:
            w_hitter = float(np.interp(sp_clamped, [90, 105, 120], [0.40, 0.25, 0.10]))
        elif has_hitter:
            w_hitter = 0.20
        else:
            w_hitter = 0.0
    else:
        # Hard pitches (fastball/sinker/cutter): current behavior
        if has_pt_holes:
            w_hitter = float(np.interp(sp_clamped, [90, 105, 120], [0.60, 0.45, 0.20]))
        elif has_agg_holes:
            w_hitter = float(np.interp(sp_clamped, [90, 105, 120], [0.55, 0.40, 0.15]))
        elif has_hitter:
            w_hitter = 0.30
        else:
            w_hitter = 0.0

    w_pitcher = 0.75 - w_hitter
    w_count = 0.25

    # Score all 3×3 cells
    scored: List[Tuple[Tuple[int, int], float, Dict, str]] = []
    for (xb, yb), zdata in ze3.items():
        if not isinstance(zdata, dict):
            continue
        raw_n = zdata.get("n", 0)
        n = int(raw_n) if pd.notna(raw_n) else 0
        if n < 5:
            continue

        # Pitcher effectiveness score, modulated by pitch-design zone mult
        pitcher_eff = _score_zone(zdata, mode=mode)
        pzm = _get_pzm(pitch_name, xb, yb)
        pitcher_eff_adj = pitcher_eff * pzm

        # Hitter vulnerability score — pitch-type fallback chain
        hitter_vuln = 0.0
        vuln_source = None
        pt_cell = None
        if pitch_name in _OFFSPEED_PITCHES and has_pt_holes and (xb, yb) in pt_holes:
            # Secondary pitches: use SLG + whiff from pitch-type data
            # instead of composite hole score (which conflates swing
            # vulnerability across all pitch types).
            pt_cell = pt_holes[(xb, yb)]
            raw_slg = pt_cell.get("slg", 0.4)
            raw_whiff = pt_cell.get("whiff_pct", 0.0)

            # Inverse SLG (0-100): low SLG against this pitch = attackable
            inv_slg = (1.0 - min(raw_slg / 1.5, 1.0)) * 100.0
            # Whiff rate (0-100)
            whiff_score = min(raw_whiff, 100.0)

            # 55% damage avoidance + 45% whiff generation
            secondary_vuln = 0.55 * inv_slg + 0.45 * whiff_score

            # Sample-size shrinkage toward aggregate when n is small
            n_pt = pt_cell.get("n", 0)
            shrink_n = 25
            if n_pt < shrink_n and has_agg_holes and (xb, yb) in hole_scores_agg:
                w = n_pt / (n_pt + shrink_n)
                agg_score = hole_scores_agg[(xb, yb)]
                secondary_vuln = w * secondary_vuln + (1 - w) * agg_score

            hitter_vuln = secondary_vuln
            vuln_source = "pt_sec"
        elif has_pt_holes and (xb, yb) in pt_holes:
            # Hard pitches: use composite hole score (captures swing path
            # vulnerability, whiff tendency, barrel avoidance).
            pt_cell = pt_holes[(xb, yb)]
            hitter_vuln = pt_cell["score"]
            vuln_source = "pt"
        elif has_agg_holes and (xb, yb) in hole_scores_agg:
            hitter_vuln = hole_scores_agg[(xb, yb)]
            vuln_source = "agg"
        elif has_hitter:
            hitter_vuln = _hitter_vuln_score(xb, yb, hitter_zone_vuln, bats)
            vuln_source = "dir"

        # Count-zone adjustment (data-driven layer on top of static _count_bonus)
        cb = _count_bonus(xb, yb, mode)
        cz_adj = _count_zone_adjustment(xb, yb, count_metrics, mode, b, s)
        count_score = cb + cz_adj

        # Combined score — weights shift based on stuff+ and data tier
        if has_pt_holes or has_agg_holes or has_hitter:
            combined = w_hitter * hitter_vuln + w_pitcher * pitcher_eff_adj + w_count * count_score
        else:
            combined = 0.70 * pitcher_eff_adj + 0.30 * count_score

        # In 0-2, boost bottom row (chase territory)
        if two_strike and b == 0 and yb == 0:
            combined *= 1.15

        # In must-strike, suppress bottom row
        if mode == "must_strike" and yb == 0:
            combined *= 0.75

        # Leverage-aware zone adjustment
        li = getattr(state, "leverage_index", 0.5)
        if isinstance(li, (int, float)) and li > 0.7:
            # High leverage: penalize middle-middle (fat pitch zone)
            if xb == 1 and yb == 1:
                combined *= 0.85
            # Reward edge/corner zones (safer locations)
            if xb in (0, 2) and yb in (0, 2):
                combined *= 1.08
        elif isinstance(li, (int, float)) and li < 0.3:
            # Low leverage: reward attacking the zone (middle zones)
            if xb == 1 and yb == 1:
                combined *= 1.10

        # Build reason string
        reasons = []
        if vuln_source == "pt_sec" and pt_cell is not None:
            reasons.append(f"{pitch_name} SLG {raw_slg:.3f} whiff {raw_whiff:.0f}% (n={pt_cell['n']})")
        elif vuln_source == "pt" and pt_cell is not None:
            reasons.append(f"{pitch_name} hole {hitter_vuln:.0f} (n={pt_cell['n']})")
        elif vuln_source == "agg":
            reasons.append(f"hole {hitter_vuln:.0f}")
        elif vuln_source == "dir":
            reasons.append(f"hitter vuln {hitter_vuln:.0f}")
        wh = zdata.get("whiff_pct", np.nan)
        if pd.notna(wh):
            reasons.append(f"whiff {wh:.0f}%")
        csw = zdata.get("csw_pct", np.nan)
        if pd.notna(csw):
            reasons.append(f"CSW {csw:.0f}%")
        if cz_adj != 0:
            reasons.append(f"count adj {cz_adj:+.0f}")
        if mode != "neutral":
            reasons.append(f"count: {mode}")
        reason = " + ".join(reasons)

        scored.append(((xb, yb), combined, zdata, reason))

    if not scored:
        return None

    # Sort by combined score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Primary = best
    primary_cell, primary_score, primary_data, primary_reason = scored[0]

    # Secondary = next best that is NOT adjacent to primary
    secondary = None
    for cell, score, zdata, reason in scored[1:]:
        if not _is_adjacent(primary_cell, cell):
            secondary = {
                "zone": cell,
                "zone_label": _cell_label(cell[0], cell[1], bats),
                "score": score,
                "reason": reason,
                **{k: v for k, v in zdata.items()},
            }
            break

    # Build output
    out = dict(primary_data)
    out.update({
        "pitch": pitch_name,
        "mode": mode,
        "zone": primary_cell,
        "zone_label": _cell_label(primary_cell[0], primary_cell[1], bats),
        "score": primary_score,
        "reason": primary_reason,
    })
    if secondary:
        out["secondary"] = secondary

    return out
