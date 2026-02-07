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
            "usage": pt_data.get("usage_pct", np.nan),
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
