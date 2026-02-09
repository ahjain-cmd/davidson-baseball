from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from decision_engine.core.state import GameState


_HARD_PITCHES = {"Fastball", "Sinker", "Cutter"}
_OFFSPEED_PITCHES = {"Slider", "Curveball", "Changeup", "Splitter", "Sweeper", "Knuckle Curve"}
_GB_PITCHES = {"Sinker", "Changeup", "Splitter", "Curveball", "Knuckle Curve"}
_SLOW_PITCHES = {"Curveball", "Knuckle Curve"}
_MEDIUM_PITCHES = {"Changeup", "Splitter", "Slider", "Sweeper"}
_QUICK_PITCHES = {"Fastball", "Sinker", "Cutter"}


# ── Calibrated weights & data (lazy-loaded singletons) ──────────────────────
_CAL = None
_HIST_CAL = None


def _sanitize_count_weights(weights):
    """Apply domain constraints to prevent calibration artifacts.

    At 3-ball counts, selection bias inflates offspeed run values because
    only elite offspeed is thrown there.  Game theory dictates hard-stuff
    dominance at those counts, so we clamp the deltas accordingly.
    At non-2-strike counts, offspeed should never receive a larger bonus
    than hard pitches.
    """
    from analytics.count_calibration import CountWeights

    result = dict(weights)
    for key in list(result.keys()):
        cw = result[key]
        b, s = int(key[0]), int(key[2])

        if b == 3 and s == 0:
            # 3-0: strongest constraint — throw fastball for a strike
            result[key] = CountWeights(
                whiff_w=cw.whiff_w, csw_w=cw.csw_w, chase_w=cw.chase_w,
                cmd_w=cw.cmd_w,
                hard_delta=max(cw.hard_delta, 3.0),
                offspeed_delta=min(cw.offspeed_delta, -3.0),
            )
        elif b == 3 and s == 1:
            # 3-1: strong constraint — heavily favor hard stuff
            result[key] = CountWeights(
                whiff_w=cw.whiff_w, csw_w=cw.csw_w, chase_w=cw.chase_w,
                cmd_w=cw.cmd_w,
                hard_delta=max(cw.hard_delta, 2.0),
                offspeed_delta=min(cw.offspeed_delta, -2.0),
            )
        elif s < 2 and cw.offspeed_delta > cw.hard_delta:
            # Non-2-strike: offspeed should never get MORE bonus than hard
            result[key] = CountWeights(
                whiff_w=cw.whiff_w, csw_w=cw.csw_w, chase_w=cw.chase_w,
                cmd_w=cw.cmd_w,
                hard_delta=cw.hard_delta,
                offspeed_delta=cw.hard_delta,
            )
    return result


def _get_count_weights():
    global _CAL
    if _CAL is None:
        from analytics.count_calibration import fallback_weights, load_count_calibration
        try:
            cal = load_count_calibration()
        except Exception:
            cal = None
        raw = cal.weights if cal else fallback_weights()
        try:
            _CAL = _sanitize_count_weights(raw)
        except Exception:
            _CAL = fallback_weights()
    return _CAL


def _get_historical_cal():
    global _HIST_CAL
    if _HIST_CAL is None:
        from analytics.historical_calibration import load_historical_calibration
        try:
            _HIST_CAL = load_historical_calibration()
        except Exception:
            _HIST_CAL = None
    return _HIST_CAL


def _get_gb_dp_rates():
    cal = _get_historical_cal()
    if cal:
        return cal.gb_dp_rates
    from analytics.historical_calibration import fallback_gb_dp_rates
    return fallback_gb_dp_rates()


def _get_steal_rates():
    cal = _get_historical_cal()
    if cal:
        return cal.steal_rates
    from analytics.historical_calibration import fallback_steal_rates
    return fallback_steal_rates()


def _get_metric_ranges():
    cal = _get_historical_cal()
    if cal:
        return cal.metric_ranges
    from analytics.historical_calibration import fallback_metric_ranges
    return fallback_metric_ranges()


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


def _count_adjustments(
    pitch_name: str, info: Dict[str, Any], state: GameState, *, hd: Dict = None,
) -> Tuple[float, List[str]]:
    b, s = state.count()
    reasons: List[str] = []
    delta = 0.0
    hd = hd or {}

    is_hard = pitch_name in _HARD_PITCHES
    is_offspeed = pitch_name in _OFFSPEED_PITCHES
    wh = info.get("shrunk_whiff", info.get("our_whiff"))
    csw = info.get("shrunk_csw", info.get("our_csw"))
    chase = info.get("shrunk_chase", info.get("our_chase"))
    ev = info.get("shrunk_ev", info.get("our_ev_against"))
    cmd = info.get("command_plus", np.nan)
    usage = info.get("usage", np.nan)

    mr = _get_metric_ranges()
    wh_n = _norm_pct(wh, lo=mr["whiff_p5"], hi=mr["whiff_p95"])
    csw_n = _norm_pct(csw, lo=mr["csw_p5"], hi=mr["csw_p95"])
    chase_n = _norm_pct(chase, lo=mr["chase_p5"], hi=mr["chase_p95"])
    cmd_n = _cmd_norm(cmd)

    # ── Calibrated count weights (unified formula for all counts) ──
    cw = _get_count_weights().get(f"{b}-{s}")
    if cw is None:
        # Defensive fallback for unexpected count values
        from analytics.count_calibration import CountWeights
        cw = CountWeights(whiff_w=2.0, csw_w=2.0, chase_w=0.0, cmd_w=2.0,
                          hard_delta=0.0, offspeed_delta=0.0)

    delta += cw.whiff_w * wh_n
    delta += cw.csw_w * csw_n
    delta += cw.chase_w * chase_n
    delta += cw.cmd_w * cmd_n
    if is_hard:
        delta += cw.hard_delta
    elif is_offspeed:
        delta += cw.offspeed_delta

    # Hard pitch command premium — scales with count leverage.
    # Well-commanded hard pitches are extra valuable at counts where you MUST
    # throw strikes (3-ball, hitter's counts).  cmd_w already tells us how
    # important command is at each count; this gives hard pitches a bonus on
    # top of the shared cmd_w * cmd_n that all pitches receive.
    if is_hard and cmd_n > 0.35:
        hard_cmd_premium = cmd_n * min(cw.cmd_w, 6.0) * 0.4
        delta += hard_cmd_premium
        if hard_cmd_premium > 1.0:
            reasons.append(f"hard cmd premium (Cmd+ {cmd:.0f}): +{hard_cmd_premium:.1f}")

    # ── Additive overlays (hitter-specific, not calibratable from our history) ──
    h_fp_swing_hard = _safe_hd(hd.get("fp_swing_hard"))
    h_fp_swing_ch = _safe_hd(hd.get("fp_swing_ch"))
    h_whiff_2k_os = _safe_hd(hd.get("whiff_2k_os"))
    h_whiff_2k_hard = _safe_hd(hd.get("whiff_2k_hard"))
    h_chase_pct = _safe_hd(hd.get("chase_pct"))
    h_bb_pct = _safe_hd(hd.get("bb_pct"))

    # Usage penalties at 3-ball: rarely-used pitch = risky.
    # Only reward high-usage hard pitches; offspeed shouldn't get a usage
    # bonus at 3-ball — the calibrated offspeed_delta already handles this.
    if b == 3 and not np.isnan(usage):
        if usage >= 15 and is_hard:
            delta += 2.0 if s < 2 else 1.0
        elif usage < 5:
            delta -= 3.0 if s < 2 else 2.0

    # Disciplined hitter at 3-ball: extra offspeed penalty (will take ball for walk)
    if b == 3 and is_offspeed and not np.isnan(h_bb_pct) and h_bb_pct > 12:
        penalty = min((h_bb_pct - 12) / 8 * 2.0, 2.0)
        delta -= penalty
        reasons.append(f"disciplined ({h_bb_pct:.0f}% BB): risky offspeed at 3-ball")

    # 0-2 extra chase emphasis: strategic intent beyond statistical weight
    if b == 0 and s == 2:
        delta += 2.0 * chase_n
        reasons.append("0-2: extra chase emphasis")

    # EV penalty at 2 strikes: continuous ramp from 85 mph (0) to 95 mph (-6.0)
    if s == 2 and not np.isnan(ev) and ev > 85:
        ev_penalty = min((ev - 85) / 5.0 * 3.0, 6.0)
        delta -= ev_penalty

    # Hitter 2K whiff: boost the pitch class the hitter actually whiffs on
    if s == 2:
        if is_offspeed and h_whiff_2k_os > 30:
            boost = min((h_whiff_2k_os - 30) / 10 * 5.0, 5.0)
            delta += boost
            reasons.append(f"2K: hitter whiffs {h_whiff_2k_os:.0f}% on OS")
        if is_hard and h_whiff_2k_hard > 35:
            boost = min((h_whiff_2k_hard - 35) / 10 * 4.0, 4.0)
            delta += boost
            reasons.append(f"2K: hitter whiffs {h_whiff_2k_hard:.0f}% on hard")

    # 0-0 hitter FP swing tendencies
    if b == 0 and s == 0:
        if is_offspeed and h_fp_swing_hard > 40:
            boost = min((h_fp_swing_hard - 40) / 15 * 3.0, 3.0)
            delta += boost
            reasons.append(f"FP: hitter attacks hard {h_fp_swing_hard:.0f}%, offspeed boost")
        if h_fp_swing_ch > 35 and pitch_name in {"Changeup", "Splitter"}:
            delta += min((h_fp_swing_ch - 35) / 15 * 2.0, 2.0)
            reasons.append(f"FP: hitter swings at CH {h_fp_swing_ch:.0f}%")

    # ── Count-group hitter performance overlay ──
    # Use the hitter's by_count data to adjust based on how they perform in
    # the current count group (ahead, behind, even).
    h_by_count = hd.get("by_count", {})
    if h_by_count:
        # Map (b, s) to count group
        _cg_map = {
            (2, 0): "ahead", (3, 0): "ahead", (3, 1): "ahead", (2, 1): "ahead",
            (0, 1): "behind", (0, 2): "behind", (1, 2): "behind",
            (1, 1): "even", (2, 2): "even",
            (3, 2): "full",
        }
        cg = _cg_map.get((b, s))
        if cg and cg in h_by_count:
            cg_data = h_by_count[cg]
            cg_n = cg_data.get("n", 0) or 0
            if cg_n >= 5:
                cg_whiff = _safe_hd(cg_data.get("whiff_pct"))
                cg_ev = _safe_hd(cg_data.get("avg_ev"))
                # Hitter struggles in this count group: boost offspeed
                if not np.isnan(cg_whiff) and cg_whiff > 30 and is_offspeed:
                    boost = min((cg_whiff - 30) / 15 * 2.0, 2.0)
                    delta += boost
                    reasons.append(f"hitter whiffs {cg_whiff:.0f}% in {cg} counts")
                # Hitter has low EV in this count group: less scary to attack
                if not np.isnan(cg_ev) and cg_ev < 82 and is_hard:
                    delta += 1.5
                    reasons.append(f"hitter avg EV {cg_ev:.0f} in {cg} counts")
                # Hitter crushes in this count: penalize hittable pitches
                if not np.isnan(cg_ev) and cg_ev > 90:
                    if is_hard:
                        delta -= 1.5
                    if not np.isnan(cg_whiff) and cg_whiff < 15:
                        delta -= 1.0
                        reasons.append(f"hitter rakes in {cg} counts (EV {cg_ev:.0f})")

    # Hitter chase data at hitter's counts: chasers still swing at offspeed.
    # But NOT at 3-ball counts — a walk is too costly to risk offspeed there.
    if b >= 2 and b > s and b < 3 and is_offspeed:
        if h_chase_pct > 28:
            boost = min((h_chase_pct - 28) / 8 * 3.0, 3.0)
            delta += boost
            reasons.append(f"chaser ({h_chase_pct:.0f}%): OS viable behind")
        elif h_chase_pct < 22 and not np.isnan(h_chase_pct):
            delta -= 1.5
            reasons.append(f"disciplined ({h_chase_pct:.0f}%): OS penalised")

    return delta, reasons


def _safe_hd(v) -> float:
    """Safely extract a hitter data value as float, defaulting to NaN."""
    try:
        f = float(v)
        return f if not np.isnan(f) else float("nan")
    except Exception:
        return float("nan")


def _base_adjustments(
    pitch_name: str, info: Dict[str, Any], state: GameState, *, hd: Dict = None,
) -> Tuple[float, List[str]]:
    bstate = state.bases
    outs = int(state.outs)
    reasons: List[str] = []
    delta = 0.0
    hd = hd or {}

    wh = info.get("shrunk_whiff", info.get("our_whiff"))
    cmd = info.get("command_plus", np.nan)

    mr = _get_metric_ranges()
    wh_n = _norm_pct(wh, lo=mr["whiff_p5"], hi=mr["whiff_p95"])
    cmd_n = _cmd_norm(cmd)

    gb_dp = _get_gb_dp_rates()
    league_avg_gb = gb_dp.get("_league_avg_gb", 47.0)
    league_avg_dp = gb_dp.get("_league_avg_dp", 4.1)
    pitch_rates = gb_dp.get(pitch_name, {})
    pitch_gb = pitch_rates.get("gb_pct", league_avg_gb)
    pitch_dp = pitch_rates.get("dp_pct", league_avg_dp)

    # Runner on 3B (<2 outs): scale GB bonus by actual GB% differential.
    if bstate.on_3b and outs < 2:
        gb_bonus = (pitch_gb / league_avg_gb - 1.0) * 10.0 if league_avg_gb > 0 else 0.0
        delta += gb_bonus
        if gb_bonus > 0.5:
            reasons.append(f"R3, <2 outs: GB bias ({pitch_name} {pitch_gb:.0f}% GB)")
        elif gb_bonus < -0.5:
            reasons.append(f"R3, <2 outs: low GB risk ({pitch_name} {pitch_gb:.0f}% GB)")

    # Double-play spot (R1, <2 outs): pitch-type-specific DP bonus.
    if bstate.on_1b and not bstate.on_2b and not bstate.on_3b and outs < 2:
        dp_bonus = (pitch_dp / league_avg_dp - 1.0) * 10.0 if league_avg_dp > 0 else 0.0
        delta += dp_bonus
        if abs(dp_bonus) > 0.3:
            reasons.append(f"R1, <2 outs: DP rate {pitch_dp:.1f}% ({pitch_name})")

        # Hitter GB%: if the batter is GB-prone, boost GB pitch types in DP spot
        h_gb = _safe_hd(hd.get("gb_pct"))
        if not np.isnan(h_gb) and h_gb > 50 and pitch_name in _GB_PITCHES:
            boost = min((h_gb - 50) / 15 * 3.0, 3.0)
            delta += boost
            reasons.append(f"GB hitter ({h_gb:.0f}%): DP boost with {pitch_name}")

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
    """Adjust pitch scores based on stolen base pressure.

    This is a situational overlay to bias toward quicker-to-the-plate options
    when there is meaningful steal pressure (runner threat + our catcher/pitcher
    control).  It is intentionally capped so it can swap borderline pitches
    but not override a dominant putaway pitch.
    """
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


def _leverage_adjustments(
    pitch_name: str, info: Dict[str, Any], state: GameState,
    wp_leverage: Optional[float] = None,
) -> Tuple[float, List[str]]:
    """Adjust pitch scoring based on game leverage.

    High leverage: favor command pitches (minimize damage).
    Low leverage: favor stuff pitches (attack aggressively).
    Uses WP-derived leverage when available, falls back to heuristic.
    """
    if wp_leverage is not None and isinstance(wp_leverage, (int, float)):
        li = float(wp_leverage)
    else:
        li = getattr(state, "leverage_index", 0.5)
    if not isinstance(li, (int, float)):
        li = 0.5

    delta = 0.0
    reasons: List[str] = []

    # Only meaningful when leverage deviates from neutral
    if abs(li - 0.5) < 0.15:
        return 0.0, []

    cmd = info.get("command_plus", np.nan)
    stuff = info.get("stuff_plus", np.nan)
    is_hard = pitch_name in _HARD_PITCHES

    if li > 0.65:
        # High leverage: command premium, penalize wild stuff
        if not np.isnan(cmd) and cmd >= 105:
            boost = min((cmd - 100) / 20 * 3.0, 3.0) * li
            delta += boost
            reasons.append(f"high leverage: command premium (Cmd+ {cmd:.0f})")
        elif not np.isnan(cmd) and cmd < 90:
            delta -= 2.0 * li
            reasons.append(f"high leverage: low command risk (Cmd+ {cmd:.0f})")

        # In high leverage, favor primary pitches (higher usage = more reliable)
        usage = info.get("usage", np.nan)
        if not np.isnan(usage) and usage < 10:
            delta -= 1.5 * li
            reasons.append("high leverage: low-usage risk")

    elif li < 0.35:
        # Low leverage: favor high-stuff pitches for development/aggression
        if not np.isnan(stuff) and stuff >= 110:
            boost = min((stuff - 100) / 20 * 2.0, 2.0) * (1.0 - li)
            delta += boost
        # Less penalty for low-usage pitches in low leverage
        usage = info.get("usage", np.nan)
        if not np.isnan(usage) and usage < 8:
            delta += 1.0  # encourage using secondary stuff in blowouts

    # Protect lead: in late innings with lead, boost command pitches
    if state.score_our is not None and state.score_opp is not None:
        our = int(state.score_our)
        opp = int(state.score_opp)
        if our > opp and state.inning >= 7 and is_hard:
            delta += 2.0
            reasons.append("protect lead: hard stuff late")

    return float(max(-5.0, min(5.0, delta))), reasons


def _hole_score_overlay(
    pitch_name: str, hd: Dict,
) -> Tuple[float, List[str]]:
    """Adjust pitch score based on hitter's pitch-type-specific hole scores.

    Higher average hole score for a pitch type means the hitter is MORE vulnerable
    to that pitch type.  Returns a delta bonus/penalty and reasons.
    """
    hole_by_pt = hd.get("hole_scores_by_pt", {})
    if not hole_by_pt:
        return 0.0, []

    # Compute mean hole score for each pitch type
    pt_means: Dict[str, float] = {}
    for pt, zones in hole_by_pt.items():
        if not zones:
            continue
        scores = [
            v["score"] if isinstance(v, dict) else float(v)
            for v in zones.values()
        ]
        if scores:
            pt_means[pt] = sum(scores) / len(scores)

    if not pt_means or pitch_name not in pt_means:
        return 0.0, []

    this_mean = pt_means[pitch_name]
    overall_mean = sum(pt_means.values()) / len(pt_means)

    # Differential: how much more/less vulnerable the hitter is to this pitch type
    # Hole scores are 0-100 (higher = more vulnerable)
    diff = this_mean - overall_mean

    # Scale: ±10 hole score difference → ±3 points adjustment (cap ±5)
    delta = diff * 0.3
    delta = max(-5.0, min(5.0, delta))

    reasons: List[str] = []
    if abs(delta) > 0.5:
        direction = "vulnerable" if delta > 0 else "resistant"
        reasons.append(f"hitter {direction} to {pitch_name} (hole {this_mean:.0f} vs avg {overall_mean:.0f})")

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


def _lookup_tunnel_score(a: str, b: str, tun_df: Optional[pd.DataFrame]) -> float:
    """Lookup tunnel score between two pitch types. Returns NaN if unavailable."""
    if not isinstance(tun_df, pd.DataFrame) or tun_df.empty:
        return np.nan
    m = tun_df[
        ((tun_df["Pitch A"] == a) & (tun_df["Pitch B"] == b))
        | ((tun_df["Pitch A"] == b) & (tun_df["Pitch B"] == a))
    ]
    if m.empty:
        return np.nan
    return float(m.iloc[0]["Tunnel Score"])


def _sequence_adjustments(
    pitch_name: str,
    state: GameState,
    tun_df: Optional[pd.DataFrame] = None,
    seq_df: Optional[pd.DataFrame] = None,
) -> Tuple[float, List[str]]:
    """Module E: Adjust pitch scores based on tunnel context with multi-pitch history.

    Uses last_pitches (up to 3) when available, falling back to last_pitch for
    backward compatibility.  Multi-pitch scoring evaluates:
    - Repetition penalty (escalating for 3+ in a row of same class)
    - Tunnel quality across the full mini-sequence
    - Pitch diversity bonus/penalty
    """
    # Build pitch history (prefer last_pitches, fall back to last_pitch)
    history: List[str] = list(getattr(state, "last_pitches", ()) or ())
    if not history:
        last = getattr(state, "last_pitch", None)
        if last:
            history = [last]
    if not history:
        return 0.0, []

    last = history[-1]  # most recent pitch
    delta = 0.0
    reasons: List[str] = []

    # ── Same-pitch repetition penalty (escalating) ──
    if pitch_name == last:
        if pitch_name in _HARD_PITCHES:
            base_penalty = -8.0
        else:
            base_penalty = -15.0
        # Escalate if we've thrown the same pitch multiple times
        consecutive = 1
        for p in reversed(history):
            if p == pitch_name:
                consecutive += 1
            else:
                break
        # Each additional repeat adds 50% more penalty
        escalation = 1.0 + 0.5 * max(0, consecutive - 2)
        delta += base_penalty * escalation
        if consecutive >= 3:
            reasons.append(f"{pitch_name} ×{consecutive}: heavy predictability penalty")
        else:
            reasons.append(f"same pitch repeated ({last}): predictability penalty")
        delta = max(-25.0, min(25.0, delta))
        return float(delta), reasons

    # ── Tunnel scoring: multi-pitch ──
    # Score tunnel from last pitch to this candidate
    tun_last = _lookup_tunnel_score(last, pitch_name, tun_df)
    if not np.isnan(tun_last):
        if tun_last >= 70:
            boost = min((tun_last - 70) / 30 * 6.0, 6.0)
            delta += boost
            reasons.append(f"tunnel {last}->{pitch_name}: {tun_last:.0f} (elite)")
        elif tun_last < 30:
            penalty = min((30 - tun_last) / 30 * 4.0, 4.0)
            delta -= penalty
            reasons.append(f"tunnel {last}->{pitch_name}: {tun_last:.0f} (poor)")

    # If we have 2+ pitch history, evaluate the full sequence tunnel
    if len(history) >= 2:
        prev2 = history[-2]
        tun_prev = _lookup_tunnel_score(prev2, last, tun_df)
        if not np.isnan(tun_prev) and not np.isnan(tun_last):
            # Average tunnel quality across the 3-pitch sequence
            avg_tun = (tun_prev + tun_last) / 2.0
            if avg_tun >= 65:
                boost = min((avg_tun - 65) / 35 * 3.0, 3.0)
                delta += boost
                reasons.append(f"3-pitch tunnel: {prev2}->{last}->{pitch_name} avg {avg_tun:.0f}")

    # ── Pitch diversity: penalize same-class clustering ──
    if len(history) >= 2:
        recent = history[-2:]  # last 2 pitches
        same_class_count = sum(
            1 for p in recent
            if (p in _HARD_PITCHES) == (pitch_name in _HARD_PITCHES)
        )
        if same_class_count == 2:
            # 3 consecutive pitches of same class (hard or offspeed)
            delta -= 3.0
            cls_name = "hard" if pitch_name in _HARD_PITCHES else "offspeed"
            reasons.append(f"3× {cls_name} class: diversity penalty")

    # ── Sequence outcome data ──
    if isinstance(seq_df, pd.DataFrame) and not seq_df.empty:
        m = seq_df[(seq_df["Setup Pitch"] == last) & (seq_df["Follow Pitch"] == pitch_name)]
        if not m.empty:
            sw = m.iloc[0].get("Whiff%", np.nan)
            if not np.isnan(sw) and sw > 25:
                boost = min((sw - 25) / 25 * 3.0, 3.0)
                delta += boost
                reasons.append(f"seq {last}->{pitch_name}: {sw:.0f}% whiff")

    # Bound total delta
    delta = max(-10.0, min(10.0, delta))
    return float(delta), reasons


def recommend_pitch_call(
    matchup: Dict[str, Any],
    state: GameState,
    top_n: int = 3,
    *,
    tun_df: Optional[pd.DataFrame] = None,
    seq_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Return top pitch call recommendations given a matchup + current game state."""
    if not matchup or not matchup.get("pitch_scores"):
        return []

    hd = matchup.get("hitter_data") or {}

    rows: List[Dict[str, Any]] = []
    for pitch_name, info in matchup["pitch_scores"].items():
        base_score = float(info.get("score", 50.0))
        c_delta, c_reasons = _count_adjustments(pitch_name, info, state, hd=hd)
        b_delta, b_reasons = _base_adjustments(pitch_name, info, state, hd=hd)
        sb_delta, sb_reasons = _steal_adjustments(pitch_name, info, state)
        sq_delta, sq_reasons = _squeeze_adjustments(pitch_name, info, state)
        se_delta, se_reasons = _sequence_adjustments(pitch_name, state, tun_df=tun_df, seq_df=seq_df)
        hs_delta, hs_reasons = _hole_score_overlay(pitch_name, hd)
        lv_delta, lv_reasons = _leverage_adjustments(pitch_name, info, state)
        final_score = _clamp(base_score + c_delta + b_delta + sb_delta + sq_delta + se_delta + hs_delta + lv_delta)

        reasons = []
        reasons.extend(info.get("reasons") or [])
        reasons.extend(c_reasons)
        reasons.extend(b_reasons)
        reasons.extend(sb_reasons)
        reasons.extend(sq_reasons)
        reasons.extend(se_reasons)
        reasons.extend(hs_reasons)
        reasons.extend(lv_reasons)

        rows.append(
            {
                "pitch": pitch_name,
                "score": float(round(final_score, 1)),
                "score_base": float(round(base_score, 1)),
                "adj_count": float(round(c_delta, 1)),
                "adj_base": float(round(b_delta, 1)),
                "adj_steal": float(round(sb_delta, 1)),
                "adj_squeeze": float(round(sq_delta, 1)),
                "adj_sequence": float(round(se_delta, 1)),
                "adj_hole_score": float(round(hs_delta, 1)),
                "adj_leverage": float(round(lv_delta, 1)),
                "confidence": info.get("confidence", "Low"),
                "confidence_n": info.get("confidence_n", info.get("count", 0)),
                "reasons": reasons,
                "raw": info,
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[: max(1, int(top_n))]


# ── RE-based pitch call recommendation ──────────────────────────────────────

_RE_CAL = None


def _get_re_calibration():
    global _RE_CAL
    if _RE_CAL is None:
        from analytics.run_expectancy import load_run_expectancy_calibration
        _RE_CAL = load_run_expectancy_calibration()
    return _RE_CAL


def recommend_pitch_call_re(
    matchup: Dict[str, Any],
    state: GameState,
    top_n: int = 3,
    *,
    tun_df: Optional[pd.DataFrame] = None,
    seq_df: Optional[pd.DataFrame] = None,
    re_cal=None,
) -> List[Dict[str, Any]]:
    """Return top pitch call recommendations scored by ΔRE (run expectancy).

    Lower ΔRE = better for pitcher. Displayed as RS/100 = -ΔRE × 100.
    """
    if not matchup or not matchup.get("pitch_scores"):
        return []

    from analytics.run_expectancy import (
        PitchOutcomeProbs,
        compute_delta_re,
        re24_lookup,
    )
    from decision_engine.core.matchup import compute_matchup_adjusted_probs

    cal = re_cal or _get_re_calibration()
    if cal is None:
        # Fallback to old scoring if no RE calibration available
        return recommend_pitch_call(matchup, state, top_n=top_n,
                                    tun_df=tun_df, seq_df=seq_df)

    hd = matchup.get("hitter_data") or {}
    b, s = state.count()
    count_str = f"{b}-{s}"
    on1b = int(bool(state.bases.on_1b))
    on2b = int(bool(state.bases.on_2b))
    on3b = int(bool(state.bases.on_3b))
    outs = int(state.outs)
    base_out = (on1b, on2b, on3b, outs)

    # Load calibrated ball/strike costs from count_calibration
    _count_ball_cost = None
    _count_strike_gain = None
    try:
        import json as _json
        from config import CACHE_DIR
        _cc_path = os.path.join(CACHE_DIR, "count_calibration.json")
        if os.path.exists(_cc_path):
            with open(_cc_path) as _ccf:
                _cc = _json.load(_ccf)
            _ccd = _cc.get("calibration", _cc)
            _count_ball_cost = _ccd.get("ball_cost")
            _count_strike_gain = _ccd.get("strike_gain")
    except Exception:
        pass

    # Load gb_dp_rates for compute_delta_re (avoids per-call file I/O)
    _gb_dp = _get_gb_dp_rates()

    # Linear weights for contact RV computation
    lw = {
        "Out": 0.0, "FieldersChoice": 0.0, "Sacrifice": 0.0,
        "Single": 0.47, "Double": 0.78, "Triple": 1.05, "HomeRun": 1.40,
        "Error": 0.47,
    }
    # Try to load from historical calibration
    hist_cal = _get_historical_cal()
    if hist_cal and hist_cal.linear_weights:
        lw["Single"] = hist_cal.linear_weights.get("single_w", 0.47)
        lw["Double"] = hist_cal.linear_weights.get("double_w", 0.78)
        lw["Triple"] = hist_cal.linear_weights.get("triple_w", 1.05)
        lw["HomeRun"] = hist_cal.linear_weights.get("hr_w", 1.40)
        lw["Error"] = hist_cal.linear_weights.get("single_w", 0.47)

    # Extract pitcher-specific batted ball profile for ΔRE computation.
    # These override the league-average hardcoded values (GB%=43%, tag_up=40%).
    _pitcher_gb_pct = None
    _pitcher_fb_pct = None
    _pitcher_scores = matchup.get("pitch_scores", {})
    _usage_weighted_gb = 0.0
    _usage_weighted_fb = 0.0
    _usage_total = 0.0
    for _pn, _pi in _pitcher_scores.items():
        _raw = _pi.get("raw", _pi) if isinstance(_pi, dict) else {}
        _u = float(_raw.get("usage", 0) or 0)
        _gb = _raw.get("gb_pct", np.nan)
        _fb = _raw.get("fb_pct", np.nan)
        if not np.isnan(_u) and _u > 0:
            if not np.isnan(_gb if isinstance(_gb, float) else float("nan")):
                _usage_weighted_gb += float(_gb) * _u
                _usage_total += _u
            if not np.isnan(_fb if isinstance(_fb, float) else float("nan")):
                _usage_weighted_fb += float(_fb) * _u
    if _usage_total > 0:
        _pitcher_gb_pct = _usage_weighted_gb / _usage_total
        _pitcher_fb_pct = _usage_weighted_fb / _usage_total if _usage_weighted_fb > 0 else None

    rows: List[Dict[str, Any]] = []
    for pitch_name, info in matchup["pitch_scores"].items():
        reasons: List[str] = []

        # 1. Get league-average outcome probs for this pitch at this count
        league_probs_obj = None
        if pitch_name in cal.outcome_probs and count_str in cal.outcome_probs[pitch_name]:
            league_probs_obj = cal.outcome_probs[pitch_name][count_str]
        else:
            # Fallback: find nearest available count by Manhattan distance
            if pitch_name in cal.outcome_probs:
                avail = cal.outcome_probs[pitch_name]
                if avail:
                    def _count_dist(c_str):
                        try:
                            cb, cs = int(c_str.split("-")[0]), int(c_str.split("-")[1])
                            return abs(cb - b) + abs(cs - s)
                        except (ValueError, IndexError):
                            return 99
                    nearest = min(avail.keys(), key=_count_dist)
                    league_probs_obj = avail[nearest]

        if league_probs_obj is None:
            # Ultimate fallback
            league_probs_obj = PitchOutcomeProbs(
                p_ball=0.38, p_called_strike=0.17, p_swinging_strike=0.10,
                p_foul=0.18, p_in_play=0.16, p_hbp=0.01,
            )

        league_probs_dict = league_probs_obj.as_dict() if hasattr(league_probs_obj, 'as_dict') else {
            "p_ball": league_probs_obj.p_ball,
            "p_called_strike": league_probs_obj.p_called_strike,
            "p_swinging_strike": league_probs_obj.p_swinging_strike,
            "p_foul": league_probs_obj.p_foul,
            "p_in_play": league_probs_obj.p_in_play,
            "p_hbp": league_probs_obj.p_hbp,
        }

        # 2. Apply matchup adjustments
        pitcher_metrics = {
            "whiff_pct": info.get("shrunk_whiff", info.get("our_whiff")),
            "csw_pct": info.get("shrunk_csw", info.get("our_csw")),
            "chase_pct": info.get("shrunk_chase", info.get("our_chase")),
            "ev_against": info.get("shrunk_ev", info.get("our_ev_against")),
            "count": info.get("count", 0),
        }
        hitter_metrics = {
            "k_pct": hd.get("k_pct"),
            "chase_pct": hd.get("chase_pct"),
            "contact_pct": hd.get("contact_pct"),
            "pa": hd.get("pa"),
        }

        adjusted = compute_matchup_adjusted_probs(
            league_probs_dict, pitcher_metrics, hitter_metrics,
        )

        adj_probs = PitchOutcomeProbs(
            p_ball=adjusted["p_ball"],
            p_called_strike=adjusted["p_called_strike"],
            p_swinging_strike=adjusted["p_swinging_strike"],
            p_foul=adjusted["p_foul"],
            p_in_play=adjusted["p_in_play"],
            p_hbp=adjusted["p_hbp"],
        )

        # Adjust contact RV for matchup
        base_crv = cal.contact_rv.get(pitch_name, 0.12)
        crv_adj = adjusted.get("contact_rv_adj", 1.0)
        adjusted_crv = base_crv * crv_adj
        adjusted_contact_rv = dict(cal.contact_rv)
        adjusted_contact_rv[pitch_name] = adjusted_crv

        # 3. Compute ΔRE
        delta_re_base = compute_delta_re(
            pitch_type=pitch_name,
            count=count_str,
            base_out_state=base_out,
            outcome_probs=adj_probs,
            re24=cal.re24,
            contact_rv=adjusted_contact_rv,
            linear_weights=lw,
            count_ball_cost=_count_ball_cost,
            count_strike_gain=_count_strike_gain,
            bip_profile=cal.bip_profiles,
            gb_dp_rates=_gb_dp,
            pitcher_gb_pct=_pitcher_gb_pct,
            pitcher_fb_pct=_pitcher_fb_pct,
        )

        # 4. Sequence adjustments — native RE: modify probs then recompute ΔRE
        se_delta_old, se_reasons = _sequence_adjustments(pitch_name, state, tun_df=tun_df, seq_df=seq_df)

        # Apply sequence effects to outcome probabilities and recompute ΔRE
        if abs(se_delta_old) > 0.5 and state.last_pitch:
            # Convert old-system delta to probability multiplier:
            # +8 old delta (max tunnel) → +15% P(ss) boost
            # -15 repetition penalty → -25% P(ss), +10% P(ball)
            ss_mult = 1.0 + se_delta_old * 0.02  # ~2% per old point
            ss_mult = max(0.6, min(1.5, ss_mult))
            ball_mult = 1.0 - se_delta_old * 0.008  # inverse, smaller
            ball_mult = max(0.85, min(1.2, ball_mult))

            seq_probs = {
                "p_ball": adjusted["p_ball"] * ball_mult,
                "p_called_strike": adjusted["p_called_strike"],
                "p_swinging_strike": adjusted["p_swinging_strike"] * ss_mult,
                "p_foul": adjusted["p_foul"],
                "p_in_play": adjusted["p_in_play"],
                "p_hbp": adjusted["p_hbp"],
            }
            # Renormalize
            seq_total = sum(seq_probs.values())
            if seq_total > 0:
                seq_probs = {k: v / seq_total for k, v in seq_probs.items()}

            seq_adj_probs = PitchOutcomeProbs(**seq_probs)
            delta_re_with_seq = compute_delta_re(
                pitch_type=pitch_name, count=count_str,
                base_out_state=base_out, outcome_probs=seq_adj_probs,
                re24=cal.re24, contact_rv=adjusted_contact_rv,
                linear_weights=lw, count_ball_cost=_count_ball_cost,
                count_strike_gain=_count_strike_gain, bip_profile=cal.bip_profiles,
                gb_dp_rates=_gb_dp,
                pitcher_gb_pct=_pitcher_gb_pct,
                pitcher_fb_pct=_pitcher_fb_pct,
            )
            delta_re_seq = delta_re_with_seq - delta_re_base
        else:
            delta_re_seq = 0.0

        # Steal/squeeze: convert old-system points to ΔRE units.
        # Principled scaling: a typical base ΔRE pitch spread is ~0.05 runs,
        # while old-system overlays span ~±10 points.  Scale factor maps
        # 10 old points ≈ 0.005 ΔRE (a meaningful but bounded adjustment).
        RE_SCALE = 0.0005
        sb_delta_old, sb_reasons = _steal_adjustments(pitch_name, info, state)
        delta_re_steal = -sb_delta_old * RE_SCALE

        sq_delta_old, sq_reasons = _squeeze_adjustments(pitch_name, info, state)
        delta_re_squeeze = -sq_delta_old * RE_SCALE

        # 4b. Hitter-specific game-theory adjustments (not captured by base ΔRE)
        # The RE computation handles mechanical count/base value, but these
        # hitter-specific tendencies need explicit modelling.
        gt_delta_old = 0.0  # accumulate in old-system points
        gt_reasons: List[str] = []
        is_hard = pitch_name in _HARD_PITCHES
        is_offspeed = pitch_name in _OFFSPEED_PITCHES

        h_bb_pct = _safe_hd(hd.get("bb_pct"))
        h_chase_pct = _safe_hd(hd.get("chase_pct"))
        h_whiff_2k_os = _safe_hd(hd.get("whiff_2k_os"))
        h_whiff_2k_hard = _safe_hd(hd.get("whiff_2k_hard"))
        h_fp_swing_hard = _safe_hd(hd.get("fp_swing_hard"))
        h_fp_swing_ch = _safe_hd(hd.get("fp_swing_ch"))
        h_gb = _safe_hd(hd.get("gb_pct"))

        # Disciplined hitter at 3-ball: offspeed penalty
        if b == 3 and is_offspeed and not np.isnan(h_bb_pct) and h_bb_pct > 12:
            penalty = min((h_bb_pct - 12) / 8 * 2.0, 2.0)
            gt_delta_old -= penalty
            gt_reasons.append(f"disciplined ({h_bb_pct:.0f}% BB): risky offspeed at 3-ball")

        # Hitter 2K whiff rates
        if s == 2:
            if is_offspeed and not np.isnan(h_whiff_2k_os) and h_whiff_2k_os > 30:
                boost = min((h_whiff_2k_os - 30) / 10 * 5.0, 5.0)
                gt_delta_old += boost
                gt_reasons.append(f"2K: hitter whiffs {h_whiff_2k_os:.0f}% on OS")
            if is_hard and not np.isnan(h_whiff_2k_hard) and h_whiff_2k_hard > 35:
                boost = min((h_whiff_2k_hard - 35) / 10 * 4.0, 4.0)
                gt_delta_old += boost
                gt_reasons.append(f"2K: hitter whiffs {h_whiff_2k_hard:.0f}% on hard")

        # 0-0 FP swing tendencies
        if b == 0 and s == 0:
            if is_offspeed and not np.isnan(h_fp_swing_hard) and h_fp_swing_hard > 40:
                boost = min((h_fp_swing_hard - 40) / 15 * 3.0, 3.0)
                gt_delta_old += boost
                gt_reasons.append(f"FP: hitter attacks hard {h_fp_swing_hard:.0f}%, offspeed boost")
            if not np.isnan(h_fp_swing_ch) and h_fp_swing_ch > 35 and pitch_name in {"Changeup", "Splitter"}:
                gt_delta_old += min((h_fp_swing_ch - 35) / 15 * 2.0, 2.0)

        # Hitter chase at hitter's counts (not 3-ball)
        if b >= 2 and b > s and b < 3 and is_offspeed:
            if not np.isnan(h_chase_pct) and h_chase_pct > 28:
                boost = min((h_chase_pct - 28) / 8 * 3.0, 3.0)
                gt_delta_old += boost
                gt_reasons.append(f"chaser ({h_chase_pct:.0f}%): OS viable behind")
            elif not np.isnan(h_chase_pct) and h_chase_pct < 22:
                gt_delta_old -= 1.5
                gt_reasons.append(f"disciplined ({h_chase_pct:.0f}%): OS penalised")

        # Count-group hitter performance overlay (RE path)
        h_by_count = hd.get("by_count", {})
        if h_by_count:
            _cg_map_re = {
                (2, 0): "ahead", (3, 0): "ahead", (3, 1): "ahead", (2, 1): "ahead",
                (0, 1): "behind", (0, 2): "behind", (1, 2): "behind",
                (1, 1): "even", (2, 2): "even",
                (3, 2): "full",
            }
            cg_re = _cg_map_re.get((b, s))
            if cg_re and cg_re in h_by_count:
                cg_data_re = h_by_count[cg_re]
                cg_n_re = cg_data_re.get("n", 0) or 0
                if cg_n_re >= 5:
                    cg_whiff_re = _safe_hd(cg_data_re.get("whiff_pct"))
                    cg_ev_re = _safe_hd(cg_data_re.get("avg_ev"))
                    if not np.isnan(cg_whiff_re) and cg_whiff_re > 30 and is_offspeed:
                        boost = min((cg_whiff_re - 30) / 15 * 2.0, 2.0)
                        gt_delta_old += boost
                        gt_reasons.append(f"hitter whiffs {cg_whiff_re:.0f}% in {cg_re} counts")
                    if not np.isnan(cg_ev_re) and cg_ev_re < 82 and is_hard:
                        gt_delta_old += 1.5
                    if not np.isnan(cg_ev_re) and cg_ev_re > 90:
                        if is_hard:
                            gt_delta_old -= 1.5
                        if not np.isnan(cg_whiff_re) and cg_whiff_re < 15:
                            gt_delta_old -= 1.0
                            gt_reasons.append(f"hitter rakes in {cg_re} counts (EV {cg_ev_re:.0f})")

        # GB hitter DP boost
        if on1b and not on2b and not on3b and outs < 2:
            if not np.isnan(h_gb) and h_gb > 50 and pitch_name in _GB_PITCHES:
                boost = min((h_gb - 50) / 15 * 3.0, 3.0)
                gt_delta_old += boost
                gt_reasons.append(f"GB hitter ({h_gb:.0f}%): DP boost with {pitch_name}")

        delta_re_gametheory = -gt_delta_old * RE_SCALE

        # 4c. Hole-score overlay — pitch-type vulnerability from zone data
        hs_delta_old, hs_reasons = _hole_score_overlay(pitch_name, hd)
        delta_re_holes = -hs_delta_old * RE_SCALE

        # 4d. Leverage adjustments (use WP leverage from state when available)
        lv_delta_old, lv_reasons = _leverage_adjustments(
            pitch_name, info, state, wp_leverage=getattr(state, "wp_leverage", None),
        )
        delta_re_leverage = -lv_delta_old * RE_SCALE

        # 5. Usage adjustment — dampen base ΔRE by reliability factor.
        # League-average outcome probs don't reflect THIS pitcher's execution
        # of a rarely-thrown pitch.  A 5% usage sinker doesn't perform like
        # the average sinker — the pitcher lacks feel, command, and deception
        # on it.  We dampen the *benefit* portion of base ΔRE proportionally.
        usage = info.get("usage", np.nan)
        delta_re_usage = 0.0
        usage_reasons: List[str] = []
        if not np.isnan(usage):
            usage_f = float(usage)

            # Reliability multiplier: sigmoid ramp with pitch-class-aware centre.
            # Fastball/Sinker: centre=15% (unusual to throw few hard pitches)
            # Slider/Curveball/Changeup: centre=10% (standard secondaries)
            # Other (Splitter, Sweeper, Cutter, Knuckle Curve): centre=8%
            if pitch_name in _HARD_PITCHES:
                sigmoid_centre = 15.0
            elif pitch_name in {"Slider", "Curveball", "Changeup"}:
                sigmoid_centre = 10.0
            else:
                sigmoid_centre = 8.0
            reliability_usage = min(1.0, max(0.0, 1.0 / (1.0 + math.exp(-0.35 * (usage_f - sigmoid_centre)))))

            # Sample-size reliability: if the pitcher has enough raw
            # observations, trust the metrics even at low usage%.
            # n_prior=150 → a 195-pitch changeup gets ~0.565 reliability.
            n_pitches = int(info.get("count", 0) or 0)
            reliability_sample = n_pitches / (n_pitches + 150) if n_pitches > 0 else 0.0

            reliability = max(reliability_usage, reliability_sample)

            if delta_re_base < 0:
                # Pitch has run-saving value; dampen by reliability
                dampened = delta_re_base * reliability
                delta_re_usage = dampened - delta_re_base  # positive = penalty
            # else: pitch is run-costing; don't help it — keep full penalty

            # Small bonus for primary pitches (hitter must respect them)
            if usage_f >= 30.0:
                delta_re_usage -= 0.002
                usage_reasons.append(f"primary pitch ({usage_f:.0f}%)")
            elif usage_f >= 20.0:
                delta_re_usage -= 0.001

            if usage_f < 8.0:
                usage_reasons.append(f"low usage ({usage_f:.0f}%): reliability {reliability:.0%}")
            elif usage_f < 15.0:
                usage_reasons.append(f"secondary ({usage_f:.0f}%)")

            # Extra penalty at 3-ball counts — walk cost makes unreliable pitches riskier
            if b == 3 and usage_f < 10.0:
                extra = 0.004 if usage_f < 5.0 else 0.002
                delta_re_usage += extra
                usage_reasons.append("3-ball: low-usage risk amplified")

        delta_re_total = (delta_re_base + delta_re_seq + delta_re_steal
                         + delta_re_squeeze + delta_re_gametheory + delta_re_holes
                         + delta_re_leverage + delta_re_usage)
        rs100 = -delta_re_total * 100.0  # runs saved per 100 pitches

        # Build reasons
        reasons.extend(info.get("reasons") or [])
        reasons.extend(se_reasons)
        reasons.extend(sb_reasons)
        reasons.extend(sq_reasons)
        reasons.extend(gt_reasons)
        reasons.extend(hs_reasons)
        reasons.extend(lv_reasons)
        reasons.extend(usage_reasons)

        # Confidence
        n_p = int(info.get("count", 0) or 0)
        from decision_engine.core.shrinkage import confidence_tier
        conf = confidence_tier(n_p)

        rows.append({
            "pitch": pitch_name,
            "delta_re": float(round(delta_re_total, 5)),
            "rs100": float(round(rs100, 2)),
            "delta_re_base": float(round(delta_re_base, 5)),
            "delta_re_seq": float(round(delta_re_seq, 5)),
            "delta_re_steal": float(round(delta_re_steal, 5)),
            "delta_re_squeeze": float(round(delta_re_squeeze, 5)),
            "delta_re_gametheory": float(round(delta_re_gametheory, 5)),
            "delta_re_holes": float(round(delta_re_holes, 5)),
            "delta_re_leverage": float(round(delta_re_leverage, 5)),
            "delta_re_usage": float(round(delta_re_usage, 5)),
            "confidence": conf,
            "confidence_n": n_p,
            "reasons": reasons,
            "raw": info,
        })

    # Sort by ΔRE ascending (most negative = best for pitcher)
    rows.sort(key=lambda r: r["delta_re"])
    return rows[: max(1, int(top_n))]


# ── WPA-based recommender ───────────────────────────────────────────────────

_WP_TABLE = None


def _get_wp_table():
    global _WP_TABLE
    if _WP_TABLE is None:
        from analytics.win_probability import load_wp_table
        _WP_TABLE = load_wp_table()
    return _WP_TABLE


def recommend_pitch_call_wpa(
    matchup: Dict[str, Any],
    state: GameState,
    top_n: int = 3,
    *,
    tun_df: Optional[pd.DataFrame] = None,
    seq_df: Optional[pd.DataFrame] = None,
    re_cal=None,
    wp_table=None,
) -> List[Dict[str, Any]]:
    """Return top pitch call recommendations scored by ΔWP (win probability).

    Parallels recommend_pitch_call_re() but uses Win Probability Added instead
    of Run Expectancy.  Lower ΔWP = better for pitcher.
    Displayed as WP±/100 = -ΔWP × 10000 (WP basis points per 100 pitches).
    """
    if not matchup or not matchup.get("pitch_scores"):
        return []

    from analytics.run_expectancy import PitchOutcomeProbs
    from analytics.win_probability import compute_delta_wp, compute_wp_leverage
    from decision_engine.core.matchup import compute_matchup_adjusted_probs

    cal = re_cal or _get_re_calibration()
    wpt = wp_table or _get_wp_table()
    if wpt is None or not wpt.wp:
        return []  # no WP table available; caller should fall back to ΔRE

    hd = matchup.get("hitter_data") or {}
    b, s = state.count()
    count_str = f"{b}-{s}"
    on1b = int(bool(state.bases.on_1b))
    on2b = int(bool(state.bases.on_2b))
    on3b = int(bool(state.bases.on_3b))
    outs = int(state.outs)
    base_out = (on1b, on2b, on3b, outs)

    # Determine score diff (home - away) and half
    score_diff = 0
    half = "top" if state.top_bottom.lower().startswith("t") else "bot"
    if state.score_our is not None and state.score_opp is not None:
        our = int(state.score_our)
        opp = int(state.score_opp)
        # "our" team is the pitching team, "opp" is the batting team.
        # Convention: if we're pitching top of inning, away team bats, we're home.
        # If we're pitching bottom, home team bats, we're away.
        if half == "top":
            # Away team is batting; we are home team pitcher
            score_diff = our - opp  # home - away
        else:
            # Home team is batting; we are away team pitcher
            score_diff = opp - our  # home - away (opp is home)
    inning = max(1, int(state.inning))

    # WP-based leverage
    wp_lev = compute_wp_leverage(wpt, inning, half, score_diff, on1b, on2b, on3b, outs)

    # Load calibrated ball/strike costs from count_calibration
    _count_ball_cost = None
    _count_strike_gain = None
    try:
        import json as _json
        from config import CACHE_DIR
        _cc_path = os.path.join(CACHE_DIR, "count_calibration.json")
        if os.path.exists(_cc_path):
            with open(_cc_path) as _ccf:
                _cc = _json.load(_ccf)
            _ccd = _cc.get("calibration", _cc)
            _count_ball_cost = _ccd.get("ball_cost")
            _count_strike_gain = _ccd.get("strike_gain")
    except Exception:
        pass

    _gb_dp = _get_gb_dp_rates()

    lw = {
        "Out": 0.0, "FieldersChoice": 0.0, "Sacrifice": 0.0,
        "Single": 0.47, "Double": 0.78, "Triple": 1.05, "HomeRun": 1.40,
        "Error": 0.47,
    }
    hist_cal = _get_historical_cal()
    if hist_cal and hist_cal.linear_weights:
        lw["Single"] = hist_cal.linear_weights.get("single_w", 0.47)
        lw["Double"] = hist_cal.linear_weights.get("double_w", 0.78)
        lw["Triple"] = hist_cal.linear_weights.get("triple_w", 1.05)
        lw["HomeRun"] = hist_cal.linear_weights.get("hr_w", 1.40)
        lw["Error"] = hist_cal.linear_weights.get("single_w", 0.47)

    # Pitcher-specific batted ball profile
    _pitcher_gb_pct = None
    _pitcher_fb_pct = None
    _pitcher_scores = matchup.get("pitch_scores", {})
    _usage_weighted_gb = 0.0
    _usage_weighted_fb = 0.0
    _usage_total = 0.0
    for _pn, _pi in _pitcher_scores.items():
        _raw = _pi.get("raw", _pi) if isinstance(_pi, dict) else {}
        _u = float(_raw.get("usage", 0) or 0)
        _gb = _raw.get("gb_pct", np.nan)
        _fb = _raw.get("fb_pct", np.nan)
        if not np.isnan(_u) and _u > 0:
            if not np.isnan(_gb if isinstance(_gb, float) else float("nan")):
                _usage_weighted_gb += float(_gb) * _u
                _usage_total += _u
            if not np.isnan(_fb if isinstance(_fb, float) else float("nan")):
                _usage_weighted_fb += float(_fb) * _u
    if _usage_total > 0:
        _pitcher_gb_pct = _usage_weighted_gb / _usage_total
        _pitcher_fb_pct = _usage_weighted_fb / _usage_total if _usage_weighted_fb > 0 else None

    RE_SCALE = 0.0005
    rows: List[Dict[str, Any]] = []

    for pitch_name, info in matchup["pitch_scores"].items():
        reasons: List[str] = []

        # 1. Get league-average outcome probs
        league_probs_obj = None
        if cal and pitch_name in cal.outcome_probs and count_str in cal.outcome_probs[pitch_name]:
            league_probs_obj = cal.outcome_probs[pitch_name][count_str]
        elif cal and pitch_name in cal.outcome_probs:
            avail = cal.outcome_probs[pitch_name]
            if avail:
                def _count_dist(c_str):
                    try:
                        cb, cs = int(c_str.split("-")[0]), int(c_str.split("-")[1])
                        return abs(cb - b) + abs(cs - s)
                    except (ValueError, IndexError):
                        return 99
                nearest = min(avail.keys(), key=_count_dist)
                league_probs_obj = avail[nearest]

        if league_probs_obj is None:
            league_probs_obj = PitchOutcomeProbs(
                p_ball=0.38, p_called_strike=0.17, p_swinging_strike=0.10,
                p_foul=0.18, p_in_play=0.16, p_hbp=0.01,
            )

        league_probs_dict = league_probs_obj.as_dict() if hasattr(league_probs_obj, 'as_dict') else {
            "p_ball": league_probs_obj.p_ball,
            "p_called_strike": league_probs_obj.p_called_strike,
            "p_swinging_strike": league_probs_obj.p_swinging_strike,
            "p_foul": league_probs_obj.p_foul,
            "p_in_play": league_probs_obj.p_in_play,
            "p_hbp": league_probs_obj.p_hbp,
        }

        # 2. Apply matchup adjustments
        pitcher_metrics = {
            "whiff_pct": info.get("shrunk_whiff", info.get("our_whiff")),
            "csw_pct": info.get("shrunk_csw", info.get("our_csw")),
            "chase_pct": info.get("shrunk_chase", info.get("our_chase")),
            "ev_against": info.get("shrunk_ev", info.get("our_ev_against")),
            "count": info.get("count", 0),
        }
        hitter_metrics = {
            "k_pct": hd.get("k_pct"),
            "chase_pct": hd.get("chase_pct"),
            "contact_pct": hd.get("contact_pct"),
            "pa": hd.get("pa"),
        }

        adjusted = compute_matchup_adjusted_probs(
            league_probs_dict, pitcher_metrics, hitter_metrics,
        )

        adj_probs = PitchOutcomeProbs(
            p_ball=adjusted["p_ball"],
            p_called_strike=adjusted["p_called_strike"],
            p_swinging_strike=adjusted["p_swinging_strike"],
            p_foul=adjusted["p_foul"],
            p_in_play=adjusted["p_in_play"],
            p_hbp=adjusted["p_hbp"],
        )

        # Adjust contact RV
        base_crv = cal.contact_rv.get(pitch_name, 0.12) if cal else 0.12
        crv_adj = adjusted.get("contact_rv_adj", 1.0)
        adjusted_crv = base_crv * crv_adj
        adjusted_contact_rv = dict(cal.contact_rv) if cal else {}
        adjusted_contact_rv[pitch_name] = adjusted_crv

        # 3. Compute ΔWP
        delta_wp_base = compute_delta_wp(
            pitch_type=pitch_name,
            count=count_str,
            base_out_state=base_out,
            outcome_probs=adj_probs,
            wp_table=wpt,
            score_diff=score_diff,
            inning=inning,
            half=half,
            contact_rv=adjusted_contact_rv,
            linear_weights=lw,
            bip_profile=cal.bip_profiles if cal else None,
            gb_dp_rates=_gb_dp,
            pitcher_gb_pct=_pitcher_gb_pct,
            pitcher_fb_pct=_pitcher_fb_pct,
            tag_up_score_pct=None,
            count_ball_cost=_count_ball_cost,
            count_strike_gain=_count_strike_gain,
        )

        # 4. Sequence adjustments (convert to ΔWP scale via ratio)
        se_delta_old, se_reasons = _sequence_adjustments(pitch_name, state, tun_df=tun_df, seq_df=seq_df)
        # Convert old-system sequence delta to ΔWP: use WP scale factor
        # ΔWP is typically ~10x smaller than ΔRE, so WP_SCALE = RE_SCALE * 0.1
        WP_SCALE = RE_SCALE * 0.1  # 0.00005
        delta_wp_seq = -se_delta_old * WP_SCALE if abs(se_delta_old) > 0.5 else 0.0

        # Steal/squeeze
        sb_delta_old, sb_reasons = _steal_adjustments(pitch_name, info, state)
        delta_wp_steal = -sb_delta_old * WP_SCALE

        sq_delta_old, sq_reasons = _squeeze_adjustments(pitch_name, info, state)
        delta_wp_squeeze = -sq_delta_old * WP_SCALE

        # Game-theory hitter adjustments
        gt_delta_old = 0.0
        gt_reasons: List[str] = []
        is_hard = pitch_name in _HARD_PITCHES
        is_offspeed = pitch_name in _OFFSPEED_PITCHES

        h_bb_pct = _safe_hd(hd.get("bb_pct"))
        h_chase_pct = _safe_hd(hd.get("chase_pct"))
        h_whiff_2k_os = _safe_hd(hd.get("whiff_2k_os"))
        h_whiff_2k_hard = _safe_hd(hd.get("whiff_2k_hard"))
        h_fp_swing_hard = _safe_hd(hd.get("fp_swing_hard"))
        h_fp_swing_ch = _safe_hd(hd.get("fp_swing_ch"))
        h_gb = _safe_hd(hd.get("gb_pct"))

        if b == 3 and is_offspeed and not np.isnan(h_bb_pct) and h_bb_pct > 12:
            penalty = min((h_bb_pct - 12) / 8 * 2.0, 2.0)
            gt_delta_old -= penalty
            gt_reasons.append(f"disciplined ({h_bb_pct:.0f}% BB): risky offspeed at 3-ball")

        if s == 2:
            if is_offspeed and not np.isnan(h_whiff_2k_os) and h_whiff_2k_os > 30:
                boost = min((h_whiff_2k_os - 30) / 10 * 5.0, 5.0)
                gt_delta_old += boost
                gt_reasons.append(f"2K: hitter whiffs {h_whiff_2k_os:.0f}% on OS")
            if is_hard and not np.isnan(h_whiff_2k_hard) and h_whiff_2k_hard > 35:
                boost = min((h_whiff_2k_hard - 35) / 10 * 4.0, 4.0)
                gt_delta_old += boost
                gt_reasons.append(f"2K: hitter whiffs {h_whiff_2k_hard:.0f}% on hard")

        if b == 0 and s == 0:
            if is_offspeed and not np.isnan(h_fp_swing_hard) and h_fp_swing_hard > 40:
                boost = min((h_fp_swing_hard - 40) / 15 * 3.0, 3.0)
                gt_delta_old += boost
                gt_reasons.append(f"FP: hitter attacks hard {h_fp_swing_hard:.0f}%, offspeed boost")
            if not np.isnan(h_fp_swing_ch) and h_fp_swing_ch > 35 and pitch_name in {"Changeup", "Splitter"}:
                gt_delta_old += min((h_fp_swing_ch - 35) / 15 * 2.0, 2.0)

        if b >= 2 and b > s and b < 3 and is_offspeed:
            if not np.isnan(h_chase_pct) and h_chase_pct > 28:
                boost = min((h_chase_pct - 28) / 8 * 3.0, 3.0)
                gt_delta_old += boost
                gt_reasons.append(f"chaser ({h_chase_pct:.0f}%): OS viable behind")
            elif not np.isnan(h_chase_pct) and h_chase_pct < 22:
                gt_delta_old -= 1.5

        # Count-group hitter performance overlay
        h_by_count = hd.get("by_count", {})
        if h_by_count:
            _cg_map = {
                (2, 0): "ahead", (3, 0): "ahead", (3, 1): "ahead", (2, 1): "ahead",
                (0, 1): "behind", (0, 2): "behind", (1, 2): "behind",
                (1, 1): "even", (2, 2): "even",
                (3, 2): "full",
            }
            cg = _cg_map.get((b, s))
            if cg and cg in h_by_count:
                cg_data = h_by_count[cg]
                cg_n = cg_data.get("n", 0) or 0
                if cg_n >= 5:
                    cg_whiff = _safe_hd(cg_data.get("whiff_pct"))
                    cg_ev = _safe_hd(cg_data.get("avg_ev"))
                    if not np.isnan(cg_whiff) and cg_whiff > 30 and is_offspeed:
                        gt_delta_old += min((cg_whiff - 30) / 15 * 2.0, 2.0)
                    if not np.isnan(cg_ev) and cg_ev < 82 and is_hard:
                        gt_delta_old += 1.5
                    if not np.isnan(cg_ev) and cg_ev > 90:
                        if is_hard:
                            gt_delta_old -= 1.5
                        if not np.isnan(cg_whiff) and cg_whiff < 15:
                            gt_delta_old -= 1.0

        if on1b and not on2b and not on3b and outs < 2:
            if not np.isnan(h_gb) and h_gb > 50 and pitch_name in _GB_PITCHES:
                boost = min((h_gb - 50) / 15 * 3.0, 3.0)
                gt_delta_old += boost
                gt_reasons.append(f"GB hitter ({h_gb:.0f}%): DP boost with {pitch_name}")

        delta_wp_gametheory = -gt_delta_old * WP_SCALE

        # Hole-score overlay
        hs_delta_old, hs_reasons = _hole_score_overlay(pitch_name, hd)
        delta_wp_holes = -hs_delta_old * WP_SCALE

        # Leverage adjustments (using WP-derived leverage)
        lv_delta_old, lv_reasons = _leverage_adjustments(pitch_name, info, state, wp_leverage=wp_lev)
        delta_wp_leverage = -lv_delta_old * WP_SCALE

        # Usage dampening (same logic, WP scale)
        usage = info.get("usage", np.nan)
        delta_wp_usage = 0.0
        usage_reasons: List[str] = []
        if not np.isnan(usage):
            usage_f = float(usage)
            if pitch_name in _HARD_PITCHES:
                sigmoid_centre = 15.0
            elif pitch_name in {"Slider", "Curveball", "Changeup"}:
                sigmoid_centre = 10.0
            else:
                sigmoid_centre = 8.0
            reliability_usage = min(1.0, max(0.0, 1.0 / (1.0 + math.exp(-0.35 * (usage_f - sigmoid_centre)))))
            n_pitches = int(info.get("count", 0) or 0)
            reliability_sample = n_pitches / (n_pitches + 150) if n_pitches > 0 else 0.0
            reliability = max(reliability_usage, reliability_sample)

            if delta_wp_base < 0:
                dampened = delta_wp_base * reliability
                delta_wp_usage = dampened - delta_wp_base
            if usage_f >= 30.0:
                delta_wp_usage -= 0.0002
            elif usage_f >= 20.0:
                delta_wp_usage -= 0.0001
            if usage_f < 8.0:
                usage_reasons.append(f"low usage ({usage_f:.0f}%): reliability {reliability:.0%}")
            if b == 3 and usage_f < 10.0:
                extra = 0.0004 if usage_f < 5.0 else 0.0002
                delta_wp_usage += extra

        delta_wp_total = (delta_wp_base + delta_wp_seq + delta_wp_steal
                         + delta_wp_squeeze + delta_wp_gametheory + delta_wp_holes
                         + delta_wp_leverage + delta_wp_usage)
        wpa_per_100 = -delta_wp_total * 10000  # WP basis points per 100 pitches

        # Build reasons
        reasons.extend(info.get("reasons") or [])
        reasons.extend(se_reasons)
        reasons.extend(sb_reasons)
        reasons.extend(sq_reasons)
        reasons.extend(gt_reasons)
        reasons.extend(hs_reasons)
        reasons.extend(lv_reasons)
        reasons.extend(usage_reasons)

        n_p = int(info.get("count", 0) or 0)
        from decision_engine.core.shrinkage import confidence_tier
        conf = confidence_tier(n_p)

        rows.append({
            "pitch": pitch_name,
            "delta_wp": float(round(delta_wp_total, 7)),
            "wpa_per_100": float(round(wpa_per_100, 2)),
            # Also include ΔRE-equivalent fields for backward compat
            "delta_re": float(round(delta_wp_total, 7)),
            "rs100": float(round(wpa_per_100, 2)),
            "delta_wp_base": float(round(delta_wp_base, 7)),
            "delta_wp_seq": float(round(delta_wp_seq, 7)),
            "delta_wp_steal": float(round(delta_wp_steal, 7)),
            "delta_wp_squeeze": float(round(delta_wp_squeeze, 7)),
            "delta_wp_gametheory": float(round(delta_wp_gametheory, 7)),
            "delta_wp_holes": float(round(delta_wp_holes, 7)),
            "delta_wp_leverage": float(round(delta_wp_leverage, 7)),
            "delta_wp_usage": float(round(delta_wp_usage, 7)),
            "wp_leverage": float(round(wp_lev, 3)),
            "confidence": conf,
            "confidence_n": n_p,
            "reasons": reasons,
            "raw": info,
        })

    # Sort by ΔWP ascending (most negative = best for pitcher)
    rows.sort(key=lambda r: r["delta_wp"])
    return rows[: max(1, int(top_n))]
