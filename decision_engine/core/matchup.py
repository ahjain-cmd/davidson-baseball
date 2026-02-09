from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from decision_engine.core.shrinkage import ShrinkageConfig, confidence_tier, shrink_value
from decision_engine.data.priors import PitchPriors

# Reuse the existing (battle-tested) composite scorer.
from _pages.scouting import _pitch_score_composite  # noqa: E402


_HARD_PITCHES = {"Fastball", "Sinker", "Cutter"}


def _norm_hand(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "?"
    s = str(x).strip().upper()
    if s in {"R", "RIGHT"}:
        return "R"
    if s in {"L", "LEFT"}:
        return "L"
    if s in {"S", "SWITCH", "BOTH"}:
        return "S"
    return "?"


def _build_hitter_data(hitter_profile: Dict[str, Any], throws: str) -> Dict[str, Any]:
    woba_key = "woba_lhp" if throws == "L" else "woba_rhp"
    woba_split = hitter_profile.get(woba_key, np.nan)
    hand_key = "lhp" if throws == "L" else "rhp"
    return {
        "pa": hitter_profile.get("pa", np.nan),
        "ops": hitter_profile.get("ops", np.nan),
        "woba": hitter_profile.get("woba", np.nan),
        "k_pct": hitter_profile.get("k_pct", np.nan),
        "bb_pct": hitter_profile.get("bb_pct", np.nan),
        "chase_pct": hitter_profile.get("chase_pct", np.nan),
        "swstrk_pct": hitter_profile.get("swstrk_pct", np.nan),
        "contact_pct": hitter_profile.get("contact_pct", np.nan),
        "swing_pct": hitter_profile.get("swing_pct", np.nan),
        "iz_swing_pct": hitter_profile.get("iz_swing_pct", np.nan),
        "p_per_pa": hitter_profile.get("p_per_pa", np.nan),
        "ev": hitter_profile.get("ev", np.nan),
        "barrel_pct": hitter_profile.get("barrel_pct", np.nan),
        "gb_pct": hitter_profile.get("gb_pct", np.nan),
        "fb_pct": hitter_profile.get("fb_pct", np.nan),
        "ld_pct": hitter_profile.get("ld_pct", np.nan),
        "pull_pct": hitter_profile.get("pull_pct", np.nan),
        "woba_split": woba_split,
        "whiff_2k_hard": hitter_profile.get(f"whiff_2k_{hand_key}_hard", np.nan),
        "whiff_2k_os": hitter_profile.get(f"whiff_2k_{hand_key}_os", np.nan),
        "fp_swing_hard": hitter_profile.get("fp_swing_hard_empty", np.nan),
        "fp_swing_ch": hitter_profile.get("fp_swing_ch_empty", np.nan),
        "swing_vs_hard": hitter_profile.get("swing_vs_hard", np.nan),
        "swing_vs_sl": hitter_profile.get("swing_vs_sl", np.nan),
        "swing_vs_cb": hitter_profile.get("swing_vs_cb", np.nan),
        "swing_vs_ch": hitter_profile.get("swing_vs_ch", np.nan),
        "high_pct": hitter_profile.get("high_pct", np.nan),
        "low_pct": hitter_profile.get("low_pct", np.nan),
        "inside_pct": hitter_profile.get("inside_pct", np.nan),
        "outside_pct": hitter_profile.get("outside_pct", np.nan),
        "zone_vuln": hitter_profile.get("zone_vuln", {}),
        "hole_scores_3x3": hitter_profile.get("hole_scores_3x3", {}),
        "hole_scores_by_pt": hitter_profile.get("hole_scores_by_pt", {}),
        "count_zone_metrics": hitter_profile.get("count_zone_metrics", {}),
        "by_count": hitter_profile.get("by_count", {}),
        "by_pitch_class": hitter_profile.get("by_pitch_class", {}),
    }


def _platoon_label(pitcher_throws: str, batter_bats: str) -> str:
    if batter_bats == "S":
        return "Switch (Neutral)"
    if pitcher_throws in {"R", "L"} and batter_bats in {"R", "L"}:
        if pitcher_throws == batter_bats:
            return "Platoon Adv"
        return "Platoon Disadv"
    return "Neutral"


def _shrunk_pitch_metrics(
    pt_data: Dict[str, Any],
    priors: Dict[str, Any],
    cfg: ShrinkageConfig,
) -> Dict[str, Any]:
    whiff_obs = pt_data.get("whiff_pct", np.nan)
    csw_obs = pt_data.get("csw_pct", np.nan)
    chase_obs = pt_data.get("chase_pct", np.nan)
    ev_obs = pt_data.get("ev_against", np.nan)
    barrel_obs = pt_data.get("barrel_pct_against", np.nan)

    whiff = shrink_value(whiff_obs, pt_data.get("n_swings"), priors.get("whiff_pct"), cfg.n_prior_equiv_whiff)
    csw = shrink_value(csw_obs, pt_data.get("count"), priors.get("csw_pct"), cfg.n_prior_equiv_csw)
    chase = shrink_value(chase_obs, pt_data.get("n_oz_pitches"), priors.get("chase_pct"), cfg.n_prior_equiv_chase)
    ev = shrink_value(ev_obs, pt_data.get("n_inplay_ev"), priors.get("ev_against"), cfg.n_prior_equiv_ev)
    barrel = shrink_value(barrel_obs, pt_data.get("n_inplay_ev"), priors.get("barrel_pct_against"), cfg.n_prior_equiv_barrel)

    return {
        "whiff_pct": whiff,
        "csw_pct": csw,
        "chase_pct": chase,
        "ev_against": ev,
        "barrel_pct_against": barrel,
    }


def score_pitcher_vs_hitter_shrunk(
    arsenal: Dict[str, Any],
    hitter_profile: Dict[str, Any],
    pitch_priors: PitchPriors,
    cfg: Optional[ShrinkageConfig] = None,
) -> Optional[Dict[str, Any]]:
    """Pitch-by-pitch scoring with small-sample shrinkage on pitcher-side metrics."""
    if arsenal is None or hitter_profile is None:
        return None

    cfg = cfg or ShrinkageConfig()
    throws = _norm_hand(arsenal.get("throws"))
    bats = _norm_hand(hitter_profile.get("bats"))
    platoon = _platoon_label(throws, bats)
    hd = _build_hitter_data(hitter_profile, throws=throws if throws in {"R", "L"} else "R")

    tun_df = arsenal.get("tunnels", pd.DataFrame())

    pitch_scores: Dict[str, Any] = {}
    for pt_name, pt_data in (arsenal.get("pitches") or {}).items():
        # Pitch-type prior fallback order: pitch type -> overall.
        prior_pt = pitch_priors.by_pitch_type.get(pt_name, pitch_priors.overall) if pitch_priors else {}
        shrunk = _shrunk_pitch_metrics(pt_data, prior_pt or {}, cfg)

        # The composite scorer reads some metrics from `arsenal_data` (notably Barrel% Against).
        # Provide a non-mutating override so shrinkage is actually reflected in the score.
        ars_data = dict(pt_data)
        if pd.notna(shrunk.get("barrel_pct_against", np.nan)):
            ars_data["barrel_pct_against"] = float(shrunk["barrel_pct_against"])
        if pd.notna(shrunk.get("ev_against", np.nan)):
            ars_data["ev_against"] = float(shrunk["ev_against"])

        pd_compat = {
            "stuff_plus": pt_data.get("stuff_plus", np.nan),
            "our_whiff": shrunk.get("whiff_pct", np.nan),
            "our_csw": shrunk.get("csw_pct", np.nan),
            "our_chase": shrunk.get("chase_pct", np.nan),
            "our_ev_against": shrunk.get("ev_against", np.nan),
        }
        comp_score = _pitch_score_composite(
            pt_name,
            pd_compat,
            hd,
            tun_df,
            platoon_label=platoon,
            arsenal_data=ars_data,
        )

        # Human-readable reasons (keep concise; UI can expand later).
        reasons = []
        stuff = pt_data.get("stuff_plus", np.nan)
        if pd.notna(stuff) and stuff >= 115:
            reasons.append(f"elite stuff ({stuff:.0f} S+)")
        whiff_obs = pt_data.get("whiff_pct", np.nan)
        if pd.notna(whiff_obs) and whiff_obs >= 35:
            reasons.append(f"high whiff ({whiff_obs:.0f}%)")
        is_hard = pt_name in _HARD_PITCHES
        hand_key = "lhp" if throws == "L" else "rhp"
        whiff_2k = hitter_profile.get(f"whiff_2k_{hand_key}_{'hard' if is_hard else 'os'}", np.nan)
        if pd.notna(whiff_2k) and whiff_2k > 35:
            reasons.append(f"hitter whiffs {whiff_2k:.0f}% on 2K {'hard' if is_hard else 'offspeed'}")
        hitter_chase = hitter_profile.get("chase_pct", np.nan)
        our_chase = pt_data.get("chase_pct", np.nan)
        if pd.notna(hitter_chase) and pd.notna(our_chase) and hitter_chase > 28 and our_chase > 30:
            reasons.append(f"chaser ({hitter_chase:.0f}%) + our chase gen ({our_chase:.0f}%)")
        ev_obs = pt_data.get("ev_against", np.nan)
        if pd.notna(ev_obs) and ev_obs > 90:
            reasons.append(f"gets hit hard ({ev_obs:.1f} EV against)")

        n_p = int(pt_data.get("count", 0) or 0)
        pitch_scores[pt_name] = {
            "score": float(round(comp_score, 1)),
            "confidence": confidence_tier(n_p),
            "confidence_n": n_p,
            "reasons": reasons,
            # Observed metrics
            "our_whiff": whiff_obs,
            "our_chase": our_chase,
            "our_csw": pt_data.get("csw_pct", np.nan),
            "our_ev_against": ev_obs,
            "barrel_pct_against": pt_data.get("barrel_pct_against", np.nan),
            "stuff_plus": stuff,
            "command_plus": pt_data.get("command_plus", np.nan),
            "usage": pt_data.get("stabilised_usage", pt_data.get("usage_pct", np.nan)),
            "velo": pt_data.get("avg_velo", np.nan),
            "eff_velo": pt_data.get("eff_velo", np.nan),
            "spin": pt_data.get("avg_spin", np.nan),
            "ivb": pt_data.get("ivb", np.nan),
            "hb": pt_data.get("hb", np.nan),
            "count": n_p,
            # Shrunk metrics used by the scorer
            "shrunk_whiff": shrunk.get("whiff_pct", np.nan),
            "shrunk_csw": shrunk.get("csw_pct", np.nan),
            "shrunk_chase": shrunk.get("chase_pct", np.nan),
            "shrunk_ev": shrunk.get("ev_against", np.nan),
            "shrunk_barrel": shrunk.get("barrel_pct_against", np.nan),
        }

    sorted_pitches = sorted(pitch_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    recommendations = []
    if sorted_pitches:
        best = sorted_pitches[0]
        recommendations.append(f"Lead with {best[0]} ({best[1]['score']:.0f})")
        offspeed = [(n, d) for n, d in sorted_pitches if n not in _HARD_PITCHES]
        if offspeed:
            recommendations.append(f"Putaway: {offspeed[0][0]} ({offspeed[0][1]['score']:.0f})")
        elif len(sorted_pitches) > 1:
            recommendations.append(f"Secondary: {sorted_pitches[1][0]} ({sorted_pitches[1][1]['score']:.0f})")

    # Usage-weighted overall score.
    overall = 50.0
    if pitch_scores:
        total_usage = float(np.nansum([v.get("usage", 0.0) or 0.0 for v in pitch_scores.values()]))
        if total_usage > 0:
            overall = float(
                np.nansum([v["score"] * (v.get("usage", 0.0) or 0.0) for v in pitch_scores.values()]) / total_usage
            )
        else:
            overall = float(np.nanmean([v["score"] for v in pitch_scores.values()]))

    return {
        "hitter": hitter_profile.get("name"),
        "pitcher": arsenal.get("name"),
        "bats": bats,
        "platoon": platoon,
        "overall_score": overall,
        "pitch_scores": pitch_scores,
        "recommendations": recommendations,
        "hitter_data": hd,
    }


# ── Matchup-adjusted outcome probabilities ──────────────────────────────────

def _logistic_adjust(p_league: float, ratio: float) -> float:
    """Adjust a probability using logistic (odds-ratio) scaling.

    ratio = pitcher_metric / league_metric, clamped [0.5, 2.0].
    """
    ratio = max(0.5, min(2.0, ratio))
    if p_league <= 0.0 or p_league >= 1.0:
        return p_league
    return p_league * ratio / (p_league * ratio + (1.0 - p_league))


def compute_matchup_adjusted_probs(
    league_probs: dict,
    pitcher_metrics: dict,
    hitter_metrics: dict,
    league_metrics: Optional[dict] = None,
) -> dict:
    """Adjust league-average pitch outcome probs using pitcher/hitter matchup data.

    Parameters
    ----------
    league_probs : dict
        Keys: p_ball, p_called_strike, p_swinging_strike, p_foul, p_in_play, p_hbp
    pitcher_metrics : dict
        Keys: whiff_pct, csw_pct, chase_pct, ev_against (optional)
    hitter_metrics : dict
        Keys: k_pct, chase_pct, contact_pct, swstrk_pct (optional)
    league_metrics : dict, optional
        Keys: whiff_pct, csw_pct, chase_pct, k_pct, contact_pct.
        Defaults to D1 league averages.

    Returns
    -------
    dict with same keys as league_probs, adjusted and renormalized.
    Also includes 'contact_rv_adj' multiplier for contact run value.
    """
    lm = league_metrics or {
        "whiff_pct": 24.0,
        "csw_pct": 27.0,
        "chase_pct": 28.0,
        "k_pct": 22.0,
        "contact_pct": 76.0,
    }

    p_ball = league_probs.get("p_ball", 0.38)
    p_cs = league_probs.get("p_called_strike", 0.17)
    p_ss = league_probs.get("p_swinging_strike", 0.10)
    p_foul = league_probs.get("p_foul", 0.18)
    p_ip = league_probs.get("p_in_play", 0.16)
    p_hbp = league_probs.get("p_hbp", 0.01)

    # Helper: shrink ratio toward 1.0 for small samples
    def _shrunk_ratio(obs, n_obs, league_val, n_prior=100):
        if obs is None or league_val is None or league_val == 0:
            return 1.0
        try:
            obs_f = float(obs)
            league_f = float(league_val)
        except (TypeError, ValueError):
            return 1.0
        if np.isnan(obs_f) or np.isnan(league_f):
            return 1.0
        raw_ratio = obs_f / league_f
        # Shrink toward 1.0 based on sample size
        try:
            n = float(n_obs) if n_obs is not None else 0.0
            if np.isnan(n):
                n = 0.0
        except (TypeError, ValueError):
            n = 0.0
        w = n / (n + n_prior)
        shrunk = w * raw_ratio + (1.0 - w) * 1.0
        return max(0.5, min(2.0, shrunk))

    # ── Pitcher adjustments ──
    p_whiff = pitcher_metrics.get("whiff_pct")
    p_csw = pitcher_metrics.get("csw_pct")
    p_chase = pitcher_metrics.get("chase_pct")
    p_n = pitcher_metrics.get("count", pitcher_metrics.get("n_pitches", 0))

    p_ss_original = p_ss  # save for cumulative cap after all adjustments

    # Whiff ratio -> adjusts P(swinging_strike)
    whiff_ratio = _shrunk_ratio(p_whiff, p_n, lm["whiff_pct"])
    p_ss = _logistic_adjust(p_ss, whiff_ratio)

    # Called-strike-only ratio -> adjusts P(called_strike)
    # CSW = called_strike% + whiff%, so called_strike_only% = CSW% - whiff%
    # Using full CSW would double-count the whiff component already applied above.
    if p_csw is not None and p_whiff is not None:
        try:
            cs_only = float(p_csw) - float(p_whiff)
            league_cs_only = float(lm["csw_pct"]) - float(lm["whiff_pct"])
            if cs_only > 0 and league_cs_only > 0:
                cs_ratio = _shrunk_ratio(cs_only, p_n, league_cs_only)
                p_cs = _logistic_adjust(p_cs, cs_ratio)
        except (TypeError, ValueError):
            pass
    elif p_csw is not None:
        # No whiff data available — fall back to full CSW ratio
        csw_ratio = _shrunk_ratio(p_csw, p_n, lm["csw_pct"])
        p_cs = _logistic_adjust(p_cs, csw_ratio)

    # ── Hitter adjustments ──
    h_k_pct = hitter_metrics.get("k_pct")
    h_chase = hitter_metrics.get("chase_pct")
    h_contact = hitter_metrics.get("contact_pct")
    h_n = hitter_metrics.get("pa", 0)

    # Hitter K% -> scales strikeout probability (affects swinging strike)
    if h_k_pct is not None and not np.isnan(float(h_k_pct or 0)):
        k_ratio = _shrunk_ratio(h_k_pct, h_n, lm["k_pct"])
        p_ss = _logistic_adjust(p_ss, k_ratio)

    # Hitter chase% -> adjusts ball/strike partition
    if h_chase is not None and not np.isnan(float(h_chase or 0)):
        chase_ratio = _shrunk_ratio(h_chase, h_n, lm["chase_pct"])
        # Higher chase = more swings on OZ = fewer balls, more whiffs
        ball_adj = 1.0 / max(chase_ratio, 0.5)  # inverse: chaser sees fewer balls
        p_ball = _logistic_adjust(p_ball, ball_adj)

    # Hitter contact% -> adjusts foul/whiff partition
    if h_contact is not None and not np.isnan(float(h_contact or 0)):
        contact_ratio = _shrunk_ratio(h_contact, h_n, lm["contact_pct"])
        # Higher contact = more fouls (instead of whiffs)
        p_foul = _logistic_adjust(p_foul, contact_ratio)
        whiff_contact_adj = 1.0 / max(contact_ratio, 0.5)
        p_ss = _logistic_adjust(p_ss, whiff_contact_adj)

    # ── Cap cumulative P(ss) adjustment to max 2.5x original ──
    # Prevents extreme compounding when pitcher whiff, hitter K%, and
    # hitter contact% adjustments all push in the same direction.
    if p_ss_original > 0:
        max_ss = p_ss_original * 2.5
        min_ss = p_ss_original * 0.3
        p_ss = max(min_ss, min(max_ss, p_ss))

    # ── Contact RV adjustment (EV-based) ──
    contact_rv_adj = 1.0
    _ev_baseline = 85.0  # D1 average EV against
    p_ev = pitcher_metrics.get("ev_against")
    if p_ev is not None and not np.isnan(float(p_ev or 0)):
        ev_ratio = float(p_ev) / _ev_baseline
        contact_rv_adj = max(0.7, min(1.5, ev_ratio))

    # ── Renormalize ──
    total = p_ball + p_cs + p_ss + p_foul + p_ip + p_hbp
    if total > 0:
        p_ball /= total
        p_cs /= total
        p_ss /= total
        p_foul /= total
        p_ip /= total
        p_hbp /= total

    return {
        "p_ball": round(p_ball, 5),
        "p_called_strike": round(p_cs, 5),
        "p_swinging_strike": round(p_ss, 5),
        "p_foul": round(p_foul, 5),
        "p_in_play": round(p_ip, 5),
        "p_hbp": round(p_hbp, 5),
        "contact_rv_adj": round(contact_rv_adj, 4),
    }
