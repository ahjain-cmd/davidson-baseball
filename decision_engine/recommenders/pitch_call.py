from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from decision_engine.core.state import GameState


_HARD_PITCHES = {"Fastball", "Sinker", "Cutter"}
_GB_PITCHES = {"Sinker", "Changeup", "Splitter", "Curveball", "Knuckle Curve"}
_SLOW_PITCHES = {"Curveball", "Knuckle Curve"}
_MEDIUM_PITCHES = {"Changeup", "Splitter", "Slider", "Sweeper"}
_QUICK_PITCHES = {"Fastball", "Sinker", "Cutter"}


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        v = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, v))


def _norm_pct(x: Any, lo: float, hi: float) -> float:
    """Map a percent metric onto [0,1] with clipping."""
    try:
        v = float(x)
    except Exception:
        return 0.5
    if np.isnan(v):
        return 0.5
    if hi <= lo:
        return 0.5
    return float(max(0.0, min(1.0, (v - lo) / (hi - lo))))


def _cmd_norm(cmd_plus: Any) -> float:
    # Command+ is roughly centered near 100 in this codebase.
    try:
        v = float(cmd_plus)
    except Exception:
        return 0.5
    if np.isnan(v):
        return 0.5
    return float(max(0.0, min(1.0, (v - 85.0) / 30.0)))


def _count_adjustments(pitch_name: str, info: Dict[str, Any], state: GameState) -> Tuple[float, List[str]]:
    b, s = state.count()
    reasons: List[str] = []
    delta = 0.0

    is_hard = pitch_name in _HARD_PITCHES
    wh = info.get("shrunk_whiff", info.get("our_whiff"))
    csw = info.get("shrunk_csw", info.get("our_csw"))
    chase = info.get("shrunk_chase", info.get("our_chase"))
    ev = info.get("shrunk_ev", info.get("our_ev_against"))
    cmd = info.get("command_plus", np.nan)
    usage = info.get("usage", np.nan)

    wh_n = _norm_pct(wh, lo=12.0, hi=40.0)
    csw_n = _norm_pct(csw, lo=18.0, hi=35.0)
    chase_n = _norm_pct(chase, lo=10.0, hi=35.0)
    cmd_n = _cmd_norm(cmd)

    # 3-2 is the highest-leverage count: must throw a strike, but putaway still matters.
    if b == 3 and s == 2:
        delta += 4.5 * cmd_n
        delta += 4.5 * wh_n
        delta += 2.0 * chase_n
        if not np.isnan(usage):
            if usage >= 15:
                delta += 1.0
            elif usage < 5:
                delta -= 2.0
        reasons.append("3-2: leverage (command + putaway)")
        return delta, reasons

    # 3-ball counts: must-strike, command-forward.
    if b == 3:
        delta += 6.0 * cmd_n
        delta += 4.0 if is_hard else -4.0
        if not np.isnan(usage):
            if usage >= 15:
                delta += 2.0
            elif usage < 5:
                delta -= 3.0
        reasons.append("3-ball: prioritize strike probability (command/primary)")
        return delta, reasons

    # Two-strike: putaway bias.
    if s == 2:
        delta += 8.0 * wh_n
        if not is_hard:
            delta += 2.0
        # 0-2: allow waste (expand) if this pitch creates chase.
        if b == 0:
            delta += 4.0 * chase_n
            reasons.append("0-2: allow chase/waste to set up putaway")
        else:
            delta += 1.5 * chase_n
            reasons.append("2-strike: lean putaway (whiff/chase)")
        if not np.isnan(ev) and ev >= 90:
            delta -= 3.0
        return delta, reasons

    # Hitter's count (2-0, 3-1): reduce walk risk.
    if b >= 2 and b > s:
        delta += 4.0 * cmd_n
        delta += 2.0 if is_hard else -2.0
        reasons.append("behind: reduce free passes (command-forward)")
        return delta, reasons

    # First pitch: CSW/strike-forward, but avoid ultra-rare offerings.
    if b == 0 and s == 0:
        delta += 4.0 * csw_n
        if not np.isnan(usage) and usage < 5:
            delta -= 2.0
        reasons.append("0-0: steal a strike (CSW/primary)")
        return delta, reasons

    # Neutral counts: slight preference for CSW + command.
    delta += 2.0 * csw_n + 2.0 * cmd_n
    return delta, reasons


def _base_adjustments(pitch_name: str, info: Dict[str, Any], state: GameState) -> Tuple[float, List[str]]:
    bstate = state.bases
    outs = int(state.outs)
    reasons: List[str] = []
    delta = 0.0

    is_hard = pitch_name in _HARD_PITCHES
    is_gb = pitch_name in _GB_PITCHES

    wh = info.get("shrunk_whiff", info.get("our_whiff"))
    cmd = info.get("command_plus", np.nan)

    wh_n = _norm_pct(wh, lo=12.0, hi=40.0)
    cmd_n = _cmd_norm(cmd)

    # Runner on 3B (<2 outs): prioritize ground ball / contact management.
    if bstate.on_3b and outs < 2:
        if is_gb:
            delta += 10.0
        elif is_hard:
            delta -= 6.0
        reasons.append("R3, <2 outs: favor GB/contact suppression")

    # Double-play spot (R1, <2 outs): sinker/CB bias.
    if bstate.on_1b and not bstate.on_2b and not bstate.on_3b and outs < 2:
        if pitch_name in {"Sinker", "Changeup", "Splitter", "Curveball", "Knuckle Curve"}:
            delta += 8.0
            reasons.append("R1, <2 outs: DP ball bias (sinker/CH/CB)")

    # Bases loaded: K + strikes.
    if bstate.is_loaded:
        delta += 6.0 * wh_n + 4.0 * cmd_n
        reasons.append("bases loaded: K + command premium")

    # 2 outs: finish PA (whiff bias).
    if outs == 2:
        delta += 6.0 * wh_n
        reasons.append("2 outs: finish PA (whiff premium)")

    return delta, reasons


def _steal_adjustments(pitch_name: str, info: Dict[str, Any], state: GameState) -> Tuple[float, List[str]]:
    """Adjust pitch scores based on stolen base pressure (Module A)."""
    rc = getattr(state, "runner", None)
    if rc is None:
        return 0.0, []

    ctx = rc.steal_context(on_1b=bool(state.bases.on_1b), on_2b=bool(state.bases.on_2b))
    pressure = float(ctx.get("pressure", 0.0) or 0.0)
    if pressure < 0.2:
        return 0.0, []

    delta = 0.0
    reasons: List[str] = []

    if pitch_name in _SLOW_PITCHES:
        delta -= 8.0 * pressure
        reasons.append(f"steal risk: slow delivery ({pitch_name})")
    elif pitch_name in _MEDIUM_PITCHES:
        delta -= 4.0 * pressure
    elif pitch_name in _QUICK_PITCHES:
        delta += 4.0 * pressure
        if pressure >= 0.7:
            reasons.append("steal risk: quick pitch helps hold runner")

    # Elite runner on 2B: simplify arsenal (avoid rushed offspeed).
    if bool(ctx.get("on_2b_elite")) and pitch_name not in _QUICK_PITCHES:
        delta -= 3.0
        reasons.append("elite runner on 2B: simplified arsenal")

    return float(delta), reasons


def _squeeze_adjustments(pitch_name: str, info: Dict[str, Any], state: GameState) -> Tuple[float, List[str]]:
    """Situational overlay for squeeze/bunt threat (Module A)."""
    rc = getattr(state, "runner", None)
    if rc is None:
        return 0.0, []

    ctx = rc.squeeze_context(on_3b=bool(state.bases.on_3b), outs=int(state.outs))
    if not ctx.get("active"):
        return 0.0, []

    threat = float(ctx.get("threat", 0.0) or 0.0)
    if threat <= 0:
        return 0.0, []

    delta = 0.0
    reasons: List[str] = []

    # Hard to bunt: elevated fastball proxy.
    if pitch_name in {"Fastball", "Cutter"}:
        delta += 5.0 * threat
        reasons.append("squeeze alert: hard to bunt (hard/ride)")
    elif pitch_name in {"Curveball", "Knuckle Curve"}:
        delta += 3.0 * threat
        reasons.append("squeeze alert: spin/shape can disrupt bunt")
    elif pitch_name in {"Sinker"}:
        delta -= 4.0 * threat
        reasons.append("squeeze alert: avoid low sinker")
    elif pitch_name in {"Changeup"}:
        delta -= 3.0 * threat
        reasons.append("squeeze alert: avoid slow changeup")

    return float(delta), reasons


def recommend_pitch_call(
    matchup: Dict[str, Any],
    state: GameState,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Return top pitch call recommendations given a matchup + current game state."""
    if not matchup or not matchup.get("pitch_scores"):
        return []

    rows: List[Dict[str, Any]] = []
    for pitch_name, info in matchup["pitch_scores"].items():
        base_score = float(info.get("score", 50.0))
        c_delta, c_reasons = _count_adjustments(pitch_name, info, state)
        b_delta, b_reasons = _base_adjustments(pitch_name, info, state)
        sb_delta, sb_reasons = _steal_adjustments(pitch_name, info, state)
        sq_delta, sq_reasons = _squeeze_adjustments(pitch_name, info, state)
        final_score = _clamp(base_score + c_delta + b_delta + sb_delta + sq_delta)

        reasons = []
        reasons.extend(info.get("reasons") or [])
        reasons.extend(c_reasons)
        reasons.extend(b_reasons)
        reasons.extend(sb_reasons)
        reasons.extend(sq_reasons)

        rows.append(
            {
                "pitch": pitch_name,
                "score": float(round(final_score, 1)),
                "score_base": float(round(base_score, 1)),
                "adj_count": float(round(c_delta, 1)),
                "adj_base": float(round(b_delta, 1)),
                "adj_steal": float(round(sb_delta, 1)),
                "adj_squeeze": float(round(sq_delta, 1)),
                "confidence": info.get("confidence", "Low"),
                "confidence_n": info.get("confidence_n", info.get("count", 0)),
                "reasons": reasons,
                "raw": info,
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[: max(1, int(top_n))]
