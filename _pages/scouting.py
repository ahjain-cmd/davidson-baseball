"""Opponent Scouting — game plans, pitcher/hitter reports, scoring engine."""
import math
import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import percentileofscore

from config import (
    DAVIDSON_TEAM_ID, ROSTER_2026, JERSEY, POSITION, PITCH_COLORS, NAME_MAP,
    SWING_CALLS, CONTACT_CALLS, TM_PITCH_PCT_COLS,
    ZONE_SIDE, ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP,
    PLATE_SIDE_MAX, PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX, MIN_PITCH_USAGE_PCT,
    filter_davidson, filter_minor_pitches, normalize_pitch_types, PITCH_TYPES_TO_DROP,
    in_zone_mask, is_barrel_mask, display_name, get_percentile,
    safe_mode, _is_position_player, tm_name_to_trackman,
)
from data.loader import (
    get_all_seasons, load_davidson_data, _load_truemedia,
    _tm_team, _tm_player, _safe_val, _safe_pct, _safe_num, _tm_pctile,
    _hitter_narrative, _pitcher_narrative, load_opponent_trackman,
)
from data.truemedia_api import (
    get_temp_token, fetch_all_teams, build_tm_dict_for_team, build_tm_dict_for_league_hitters,
    build_tm_dict_for_league_pitchers, fetch_team_totals_hitting, fetch_team_totals_pitching,
    fetch_team_all_pitches_trackman, fetch_hitter_count_splits,
    clear_league_cache, get_league_cache_info,
)
from data.stats import compute_batter_stats, compute_pitcher_stats, _build_batter_zones, compute_swing_path_metrics
from data.population import (
    compute_batter_stats_pop, compute_pitcher_stats_pop, compute_stuff_baselines,
    build_tunnel_population_pop,
)
from viz.layout import CHART_LAYOUT, section_header
from viz.charts import (
    add_strike_zone, add_view_badge, make_spray_chart, make_movement_profile,
    make_pitch_location_heatmap, player_header, _safe_pr, _safe_pop,
    _add_grid_zone_outline,
)
from viz.percentiles import savant_color, render_savant_percentile_section
from analytics.stuff_plus import _compute_stuff_plus, _compute_stuff_plus_all
from analytics.tunnel import _compute_tunnel_score, _build_tunnel_population
from analytics.command_plus import _compute_command_plus, _compute_pitch_pair_results
from analytics.zone_vulnerability import (
    compute_zone_swing_metrics as _zv_compute_zone_swing_metrics,
    analyze_zone_patterns as _zv_analyze_zone_patterns,
    swing_path_vulnerability as _zv_swing_path_vulnerability,
    compute_hole_scores_3x3 as _zv_compute_hole_scores_3x3,
)


def _fmt_bats(bats):
    if bats is None or (isinstance(bats, float) and pd.isna(bats)):
        return "?"
    b = str(bats).strip()
    mapping = {"Right": "R", "Left": "L", "Both": "S", "Switch": "S", "R": "R", "L": "L", "S": "S"}
    return mapping.get(b, b)


def _add_bats_badge(fig, bats):
    if fig is None:
        return fig
    label = _fmt_bats(bats)
    if label == "?":
        return fig
    fig.add_annotation(
        x=0.99, y=0.98, xref="paper", yref="paper",
        text=f"Bats: {label}",
        showarrow=False, xanchor="right", yanchor="top",
        font=dict(size=11, color="#444"),
    )
    return fig


def _prefer_truemedia_pitch_data(df):
    if df is None or df.empty:
        return df
    if "__source" in df.columns:
        tm = df[df["__source"] == "truemedia"]
        if not tm.empty:
            return tm
    return df


def _pitch_source_label(df):
    if df is None or df.empty:
        return "Trackman"
    if "__source" in df.columns and (df["__source"] == "truemedia").any():
        return "TrueMedia"
    return "Trackman"


def _filter_pitch_types_global(pitch_df, min_pct=MIN_PITCH_USAGE_PCT):
    """Remove undefined/other and globally rare pitch types (< min_pct usage)."""
    if pitch_df is None or pitch_df.empty or "TaggedPitchType" not in pitch_df.columns:
        return pitch_df
    df = normalize_pitch_types(pitch_df.copy())
    df = df.dropna(subset=["TaggedPitchType"])
    # Hard-drop undefined/unknown labels from TrueMedia payloads.
    bad_pitch_labels = {"UN", "UNK", "UNKNOWN", "UNDEFINED", "OTHER", "NONE", "NULL", "NAN", ""}
    bad_pitch_labels |= {str(v).strip().upper() for v in PITCH_TYPES_TO_DROP}
    pt_upper = df["TaggedPitchType"].astype(str).str.strip().str.upper()
    df = df[~pt_upper.isin(bad_pitch_labels)]
    if df.empty:
        return df
    vc = df["TaggedPitchType"].value_counts()
    total = vc.sum()
    if total <= 0:
        return df.iloc[0:0]
    usage_pct = vc / total * 100
    keep = usage_pct[usage_pct >= min_pct].index
    return df[df["TaggedPitchType"].isin(keep)].copy()


def _infer_hand_from_pitch_df(pitch_df, player_name, role="batter"):
    if pitch_df is None or pitch_df.empty:
        return "?"
    if role == "batter":
        df = _match_batter_trackman(pitch_df, player_name)
        col = "BatterSide"
    else:
        df = _match_pitcher_trackman(pitch_df, player_name)
        col = "PitcherThrows"
    if df.empty or col not in df.columns:
        return "?"
    series = (
        df[col]
        .replace({"R": "Right", "L": "Left", "S": "Switch", "Both": "Switch"})
        .dropna()
        .astype(str)
        .str.strip()
    )
    if series.empty:
        return "?"
    # If switch appears explicitly, honor it.
    if (series == "Switch").any():
        return "S"
    # If both sides appear in meaningful volume, treat as switch.
    counts = series.value_counts()
    if "Right" in counts and "Left" in counts:
        total = counts.sum()
        if counts["Right"] >= 10 and counts["Left"] >= 10 and (counts["Right"] / total >= 0.2) and (counts["Left"] / total >= 0.2):
            return "S"
    return counts.idxmax()


def _adjust_side_for_bats(df, side_col="PlateLocSide", bats=None):
    """Return the side coordinate (no transformation).

    We don't transform the data - instead we just place inside/away labels
    based on the batter handedness in the layout function.

    Standard catcher-view coordinates:
    - Positive PlateLocSide = first base side (catcher's right)
    - Negative PlateLocSide = third base side (catcher's left)

    For RHH: inside = third base side = negative X (left on graph)
    For LHH: inside = first base side = positive X (right on graph)
    """
    if df is None or side_col not in df.columns:
        return None
    side = pd.to_numeric(df[side_col], errors="coerce")

    # Return the side as-is - don't transform the data
    # The layout function will place inside/away labels based on batter handedness
    return side


def _rel_xbin(xb, bats):
    """Convert absolute x-bin (0=left,2=right) to hitter-relative (0=inside,2=away)."""
    label = _fmt_bats(bats)
    if label == "L":
        return 2 - xb
    return xb


def _entropy_bits(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return np.nan
    return float(-(probs * np.log2(probs)).sum())


def _pitch_type_entropy_by_count(pitch_df, min_pitches=20, min_pitch_usage_pct=0.0):
    """Compute pitch-type entropy by count and hitter hand."""
    req = {"TaggedPitchType", "Balls", "Strikes", "BatterSide"}
    if pitch_df.empty or not req.issubset(pitch_df.columns):
        return pd.DataFrame()

    df = pitch_df.dropna(subset=["TaggedPitchType", "Balls", "Strikes", "BatterSide"]).copy()
    if df.empty:
        return pd.DataFrame()
    df = normalize_pitch_types(df)
    bad_pitch_labels = {"UN", "UNK", "UNKNOWN", "UNDEFINED", "OTHER", "NONE", "NULL", "NAN", ""}
    bad_pitch_labels |= {str(v).strip().upper() for v in PITCH_TYPES_TO_DROP}
    pt_upper = df["TaggedPitchType"].astype(str).str.strip().str.upper()
    df = df[~pt_upper.isin(bad_pitch_labels)]
    if df.empty:
        return pd.DataFrame()
    if min_pitch_usage_pct and min_pitch_usage_pct > 0:
        usage = df["TaggedPitchType"].value_counts(normalize=True) * 100
        keep = usage[usage >= float(min_pitch_usage_pct)].index
        df = df[df["TaggedPitchType"].isin(keep)]
        if df.empty:
            return pd.DataFrame()

    balls_num = pd.to_numeric(df["Balls"], errors="coerce")
    strikes_num = pd.to_numeric(df["Strikes"], errors="coerce")
    side_norm = df["BatterSide"].astype(str).str.strip().str.upper().str[0]
    # Hand-specific predictability only (LHH/RHH).
    valid = balls_num.notna() & strikes_num.notna() & side_norm.isin(["L", "R"])
    df = df[valid].copy()
    if df.empty:
        return pd.DataFrame()
    balls_num = balls_num[valid].astype(int)
    strikes_num = strikes_num[valid].astype(int)
    df["count"] = balls_num.astype(str) + "-" + strikes_num.astype(str)
    df["BatterSideNorm"] = side_norm[valid]

    rows = []
    for (count, side), grp in df.groupby(["count", "BatterSideNorm"]):
        n = len(grp)
        if n < min_pitches:
            continue
        vc = grp["TaggedPitchType"].value_counts()
        top_pitch = vc.index[0]
        top_pct = vc.iloc[0] / n * 100
        probs = (vc / n).values
        k = len(vc)
        h = _entropy_bits(probs)
        h_norm = h / np.log2(k) if k > 1 and pd.notna(h) else 0.0
        predict = 1 - h_norm if pd.notna(h_norm) else np.nan
        rows.append({
            "Count": count,
            "BatterSide": side,
            "N": n,
            "Top Pitch": top_pitch,
            "Top%": top_pct,
            "Entropy": h,
            "Predictability": predict,
            "Pitch Types": k,
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["BatterSide", "Predictability", "N"], ascending=[True, False, False])


def _hitter_attack_plan(rate, exit_d, pr, ht, hl):
    """Auto-generate 'How to Attack' text for a hitter based on TrueMedia data."""
    notes = []
    chase = rate.iloc[0].get("Chase%") if not rate.empty and "Chase%" in rate.columns else None
    if chase is not None and not pd.isna(chase) and chase > 30:
        notes.append(f"High chase rate ({chase:.1f}%) — expand off the plate and below the zone")
    elif chase is not None and not pd.isna(chase) and chase < 18:
        notes.append(f"Very disciplined ({chase:.1f}% chase) — must live in the zone, compete with strikes")
    k_pct = rate.iloc[0].get("K%") if not rate.empty and "K%" in rate.columns else None
    if k_pct is not None and not pd.isna(k_pct) and k_pct > 25:
        notes.append(f"Strikeout-prone ({k_pct:.1f}% K) — use putaway secondary pitches")
    bb_pct = rate.iloc[0].get("BB%") if not rate.empty and "BB%" in rate.columns else None
    if bb_pct is not None and not pd.isna(bb_pct) and bb_pct > 12:
        notes.append(f"Patient eye ({bb_pct:.1f}% BB) — don't nibble, throw strikes early")
    gb_pct = ht.iloc[0].get("Ground%") if not ht.empty and "Ground%" in ht.columns else None
    if gb_pct is not None and not pd.isna(gb_pct) and gb_pct < 30:
        notes.append(f"Low ground ball rate ({gb_pct:.1f}%) — fly-ball hitter, keep the ball down")
    ev = exit_d.iloc[0].get("ExitVel") if not exit_d.empty and "ExitVel" in exit_d.columns else None
    if ev is not None and not pd.isna(ev) and ev > 90:
        notes.append(f"Dangerous exit velo ({ev:.1f} mph) — do not groove fastballs")
    barrel = exit_d.iloc[0].get("Barrel%") if not exit_d.empty and "Barrel%" in exit_d.columns else None
    if barrel is not None and not pd.isna(barrel) and barrel > 12:
        notes.append(f"High barrel rate ({barrel:.1f}%) — pitch to contact with movement, not velocity")
    pull = hl.iloc[0].get("HPull%") if not hl.empty and "HPull%" in hl.columns else None
    if pull is not None and not pd.isna(pull) and pull > 50:
        notes.append(f"Pull-heavy ({pull:.1f}%) — attack away, shift infield towards pull side")
    if not notes:
        notes.append("No major exploitable tendencies identified. Pitch competitively and mix locations.")
    return notes


def _pitcher_attack_plan(trad, mov, pr, ht):
    """Auto-generate 'How to Attack' text for an opposing pitcher."""
    notes = []
    era = trad.iloc[0].get("ERA") if not trad.empty and "ERA" in trad.columns else None
    fip = trad.iloc[0].get("FIP") if not trad.empty and "FIP" in trad.columns else None
    if era is not None and not pd.isna(era) and era > 5.0:
        notes.append(f"High ERA ({era:.2f}) — hittable, aggressive early in counts")
    bb9 = trad.iloc[0].get("BB/9") if not trad.empty and "BB/9" in trad.columns else None
    if bb9 is not None and not pd.isna(bb9) and bb9 > 4.0:
        notes.append(f"Control issues ({bb9:.1f} BB/9) — take pitches early, work counts")
    k9 = trad.iloc[0].get("K/9") if not trad.empty and "K/9" in trad.columns else None
    if k9 is not None and not pd.isna(k9) and k9 < 7.0:
        notes.append(f"Low strikeout stuff ({k9:.1f} K/9) — put the ball in play, don't chase")
    gb_pct = ht.iloc[0].get("Ground%") if not ht.empty and "Ground%" in ht.columns else None
    if gb_pct is not None and not pd.isna(gb_pct) and gb_pct > 50:
        notes.append(f"Ground-ball pitcher ({gb_pct:.1f}% GB) — look to elevate, hit ball in air")
    vel = mov.iloc[0].get("Vel") if not mov.empty and "Vel" in mov.columns else None
    if vel is not None and not pd.isna(vel) and vel < 88:
        notes.append(f"Below-average velocity ({vel:.1f} mph) — sit on offspeed, crush mistakes")
    elif vel is not None and not pd.isna(vel) and vel > 93:
        notes.append(f"High velocity ({vel:.1f} mph) — shorten swing, focus on timing")
    chase_pct = pr.iloc[0].get("Chase%") if not pr.empty and "Chase%" in pr.columns else None
    if chase_pct is not None and not pd.isna(chase_pct) and chase_pct < 22:
        notes.append(f"Low chase induced ({chase_pct:.1f}%) — hitters can be selective")
    if not notes:
        notes.append("Solid pitcher with no glaring weaknesses. Focus on quality at-bats and executing the game plan.")
    return notes


# ══════════════════════════════════════════════════════════════
# GAME PLAN ENGINE — Cross-reference Trackman + TrueMedia
# ══════════════════════════════════════════════════════════════

def _get_opp_hitter_profile(tm, hitter, team, pitch_df=None):
    """Extract opponent hitter vulnerability profile from TrueMedia."""
    def _lookup(table_key, default=None):
        """Team-filtered player lookup with fallback to player-only if team filter misses."""
        df = tm["hitting"].get(table_key, default if default is not None else pd.DataFrame())
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        result = _tm_player(_tm_team(df, team), hitter)
        if result.empty:
            # Fallback: skip team filter (handles combined packs with mixed team names)
            result = _tm_player(df, hitter)
        return result

    rate = _lookup("rate")
    pr = _lookup("pitch_rates")
    pt = _lookup("pitch_types")
    exit_d = _lookup("exit")
    ht = _lookup("hit_types")
    hl = _lookup("hit_locations")
    pl = _lookup("pitch_locations") if "pitch_locations" in tm["hitting"] else pd.DataFrame()
    sw = _lookup("swing_stats") if "swing_stats" in tm["hitting"] else pd.DataFrame()
    fp = _lookup("swing_pct") if "swing_pct" in tm["hitting"] else pd.DataFrame()
    bats = rate.iloc[0].get("batsHand", "?") if not rate.empty else "?"
    if bats in [None, "", "?"] or (isinstance(bats, float) and pd.isna(bats)):
        bats = _infer_hand_from_pitch_df(pitch_df, hitter, role="batter") if pitch_df is not None else "?"
    profile = {
        "name": hitter,
        "bats": bats,
        "pa": _safe_num(rate, "PA"),
        "ops": _safe_num(rate, "OPS"),
        "woba": _safe_num(rate, "WOBA"),
        "k_pct": _safe_num(rate, "K%"),
        "bb_pct": _safe_num(rate, "BB%"),
        "chase_pct": _safe_num(pr, "Chase%"),
        "swstrk_pct": _safe_num(pr, "SwStrk%"),
        "contact_pct": _safe_num(pr, "Contact%"),
        "swing_pct": _safe_num(pr, "Swing%"),
        "iz_swing_pct": _safe_num(sw, "InZoneSwing%"),
        "p_per_pa": _safe_num(pr, "P/PA"),
        "ev": _safe_num(exit_d, "ExitVel"),
        "barrel_pct": _safe_num(exit_d, "Barrel%"),
        "gb_pct": _safe_num(ht, "Ground%"),
        "fb_pct": _safe_num(ht, "Fly%"),
        "ld_pct": _safe_num(ht, "Line%"),
        "pull_pct": _safe_num(hl, "HPull%"),
        # Zone tendencies — where they see pitches
        "high_pct": _safe_num(pl, "High%"),
        "low_pct": _safe_num(pl, "Low%"),
        "inside_pct": _safe_num(pl, "Inside%"),
        "outside_pct": _safe_num(pl, "Outside%"),
        # Platoon splits
        "woba_lhp": _safe_num(sw, "wOBA LHP"),
        "woba_rhp": _safe_num(sw, "wOBA RHP"),
        # 2-strike whiff rates by hand and pitch class
        "whiff_2k_lhp_hard": _safe_num(sw, "2K Whiff vs LHP Hard"),
        "whiff_2k_lhp_os": _safe_num(sw, "2K Whiff vs LHP OS"),
        "whiff_2k_rhp_hard": _safe_num(sw, "2K Whiff vs RHP Hard"),
        "whiff_2k_rhp_os": _safe_num(sw, "2K Whiff vs RHP OS"),
        # First pitch swing rates by pitch type
        "fp_swing_hard_empty": _safe_num(sw, "1PSwing% vs Hard Empty"),
        "fp_swing_ch_empty": _safe_num(sw, "1PSwing% vs CH Empty"),
        "swing_vs_hard": _safe_num(fp, "Swing% vs Hard"),
        "swing_vs_sl": _safe_num(fp, "Swing% vs SL"),
        "swing_vs_cb": _safe_num(fp, "Swing% vs CB"),
        "swing_vs_ch": _safe_num(fp, "Swing% vs CH"),
        "pitch_type_pcts": {},
    }
    bad_pitch_labels = {"UN", "UNK", "UNKNOWN", "UNDEFINED", "OTHER", "NONE", "NULL", "NAN", ""}
    bad_pitch_labels |= {str(v).strip().upper() for v in PITCH_TYPES_TO_DROP}
    if not pt.empty:
        for tm_col, trackman_name in TM_PITCH_PCT_COLS.items():
            v = _safe_num(pt, tm_col)
            if not pd.isna(v) and v > 0 and str(trackman_name).strip().upper() not in bad_pitch_labels:
                profile["pitch_type_pcts"][trackman_name] = v

    # Zone vulnerability summary (when pitch-level data is available)
    if pitch_df is not None and not pitch_df.empty:
        from analytics.zone_vulnerability import compute_zone_vulnerability_summary
        hitter_pitches = pitch_df[pitch_df["Batter"].astype(str).str.strip() == str(hitter).strip()] if "Batter" in pitch_df.columns else pd.DataFrame()
        if len(hitter_pitches) >= 50:
            hitter_pitches_norm = normalize_pitch_types(hitter_pitches)
            profile["zone_vuln"] = compute_zone_vulnerability_summary(hitter_pitches_norm, bats)

    # Pre-computed hole scores from opponent pack (3x3 grid, 0-100)
    hole_df = tm["hitting"].get("hole_scores", pd.DataFrame())
    if isinstance(hole_df, pd.DataFrame) and not hole_df.empty:
        hitter_holes = hole_df[hole_df["playerFullName"].astype(str).str.strip() == str(hitter).strip()]
        if not hitter_holes.empty:
            has_pitcher_throws = "pitcher_throws" in hitter_holes.columns
            has_pitch_type = "pitch_type" in hitter_holes.columns

            # Helper: extract hole scores for a filtered subset
            def _extract_holes(subset):
                agg_sub = subset[subset["pitch_type"] == "ALL"] if has_pitch_type else subset
                agg_dict = {
                    (int(r["xb"]), int(r["yb"])): float(r["score"])
                    for _, r in agg_sub.iterrows()
                }
                by_pt_dict = {}
                if has_pitch_type:
                    for pt in subset["pitch_type"].unique():
                        if pt == "ALL":
                            continue
                        pt_rows = subset[subset["pitch_type"] == pt]
                        by_pt_dict[pt] = {
                            (int(r["xb"]), int(r["yb"])): {
                                "score": float(r["score"]),
                                "swing_pct": float(r.get("swing_pct", 0.5)),
                                "slg": float(r.get("slg", 0.4)),
                                "ev_mean": float(r.get("ev_mean", 85)),
                                "n": int(r.get("n", 0)),
                            }
                            for _, r in pt_rows.iterrows()
                        }
                return agg_dict, by_pt_dict

            # Aggregate (pitcher_throws="ALL" or old packs without pitcher_throws)
            if has_pitcher_throws:
                agg_rows = hitter_holes[hitter_holes["pitcher_throws"] == "ALL"]
            else:
                agg_rows = hitter_holes
            agg_3x3, agg_by_pt = _extract_holes(agg_rows)
            if agg_3x3:
                profile["hole_scores_3x3"] = agg_3x3
            if agg_by_pt:
                profile["hole_scores_by_pt"] = agg_by_pt

            # Platoon-specific hole scores (vs RHP, vs LHP)
            if has_pitcher_throws:
                by_hand = {}
                by_hand_3x3 = {}
                for hand in ("R", "L"):
                    hand_rows = hitter_holes[hitter_holes["pitcher_throws"] == hand]
                    if hand_rows.empty:
                        continue
                    h_3x3, h_by_pt = _extract_holes(hand_rows)
                    if h_by_pt:
                        by_hand[hand] = h_by_pt
                    if h_3x3:
                        by_hand_3x3[hand] = h_3x3
                if by_hand:
                    profile["hole_scores_by_hand"] = by_hand
                if by_hand_3x3:
                    profile["hole_scores_3x3_by_hand"] = by_hand_3x3

    # Count-zone metrics
    czm_df = tm["hitting"].get("count_zone_metrics", pd.DataFrame())
    if isinstance(czm_df, pd.DataFrame) and not czm_df.empty:
        hitter_czm = czm_df[czm_df["playerFullName"].astype(str).str.strip() == str(hitter).strip()]
        if not hitter_czm.empty:
            has_pt_col = "pitcher_throws" in hitter_czm.columns

            def _extract_czm(subset):
                czm_out = {}
                for cg in subset["count_group"].unique():
                    cg_rows = subset[subset["count_group"] == cg]
                    czm_out[cg] = {
                        (int(r["xb"]), int(r["yb"])): {
                            "swing_rate": float(r["swing_rate"]),
                            "whiff_rate": float(r["whiff_rate"]),
                            "slg": float(r["slg"]),
                            "n": int(r["n"]),
                        }
                        for _, r in cg_rows.iterrows()
                    }
                return czm_out

            # Aggregate count-zone metrics
            if has_pt_col:
                czm_agg = hitter_czm[hitter_czm["pitcher_throws"] == "ALL"]
            else:
                czm_agg = hitter_czm
            czm = _extract_czm(czm_agg)
            if czm:
                profile["count_zone_metrics"] = czm

            # Platoon-specific count-zone metrics
            if has_pt_col:
                czm_by_hand = {}
                for hand in ("R", "L"):
                    hand_rows = hitter_czm[hitter_czm["pitcher_throws"] == hand]
                    if hand_rows.empty:
                        continue
                    h_czm = _extract_czm(hand_rows)
                    if h_czm:
                        czm_by_hand[hand] = h_czm
                if czm_by_hand:
                    profile["count_zone_by_hand"] = czm_by_hand

    return profile


def _compute_swing_path(bdf):
    """Compute attack-angle proxy and swing path metrics (shared logic)."""
    return compute_swing_path_metrics(bdf)


def _get_our_pitcher_arsenal(data, pitcher_name, season_filter=None, tunnel_pop=None):
    """Extract Davidson pitcher arsenal from Trackman data."""
    pdf = filter_davidson(data, role="pitcher")
    pdf = pdf[pdf["Pitcher"] == pitcher_name].copy()
    if season_filter:
        pdf = pdf[pdf["Season"].isin(season_filter)]
    pdf = normalize_pitch_types(pdf)
    pdf = filter_minor_pitches(pdf)
    if len(pdf) < 20:
        return None
    throws = safe_mode(pdf["PitcherThrows"], "Right")
    cmd_df = _compute_command_plus(pdf, data)
    cmd_map = dict(zip(cmd_df["Pitch"], cmd_df["Command+"])) if not cmd_df.empty else {}
    batter_zones = _build_batter_zones(data)
    _iz = in_zone_mask(pdf, batter_zones, batter_col="Batter")
    _oz = ~_iz & pdf["PlateLocSide"].notna() & pdf["PlateLocHeight"].notna()
    arsenal = {"name": pitcher_name, "throws": throws, "total_pitches": len(pdf), "pitches": {}}
    stuff_df = _compute_stuff_plus(pdf)
    for pt_name, grp in pdf.groupby("TaggedPitchType"):
        if len(grp) < 5:
            continue
        sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
        wh = grp[grp["PitchCall"] == "StrikeSwinging"]
        csw = grp[grp["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])]
        ip = grp[grp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        oz_grp = grp[_oz.reindex(grp.index, fill_value=False)]
        oz_sw = oz_grp[oz_grp["PitchCall"].isin(SWING_CALLS)]
        iz_grp = grp[_iz.reindex(grp.index, fill_value=False)]
        stuff_plus = np.nan
        if not stuff_df.empty and "StuffPlus" in stuff_df.columns:
            sp = stuff_df[stuff_df["TaggedPitchType"] == pt_name]["StuffPlus"]
            if len(sp) > 0:
                stuff_plus = sp.mean()
        # Per-pitch zone effectiveness (where this pitch is best from Trackman)
        loc_grp = grp.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        zone_mid_h = (ZONE_HEIGHT_BOT + ZONE_HEIGHT_TOP) / 2
        zone_eff = {}
        for zn, zmask in [
            ("up", loc_grp["PlateLocHeight"] >= zone_mid_h),
            ("down", (loc_grp["PlateLocHeight"] < zone_mid_h) & (loc_grp["PlateLocHeight"] >= ZONE_HEIGHT_BOT)),
            # Catcher's-view coordinates:
            #   PlateLocSide > 0 => 1B side, PlateLocSide < 0 => 3B side
            # Glove-side (pitcher perspective):
            #   RHP: 3B side (negative), LHP: 1B side (positive)
            ("glove", loc_grp["PlateLocSide"] < 0 if throws == "Right" else loc_grp["PlateLocSide"] > 0),
            ("arm", loc_grp["PlateLocSide"] > 0 if throws == "Right" else loc_grp["PlateLocSide"] < 0),
            ("chase_low", loc_grp["PlateLocHeight"] < ZONE_HEIGHT_BOT),
        ]:
            zdf = loc_grp[zmask]
            if len(zdf) >= 5:
                z_sw = zdf[zdf["PitchCall"].isin(SWING_CALLS)]
                z_wh = zdf[zdf["PitchCall"] == "StrikeSwinging"]
                z_ip = zdf[zdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                zone_eff[zn] = {
                    "n": len(zdf), "whiff_pct": len(z_wh) / max(len(z_sw), 1) * 100 if len(z_sw) > 0 else np.nan,
                    "csw_pct": len(zdf[zdf["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])]) / len(zdf) * 100,
                    "ev_against": z_ip["ExitSpeed"].mean() if len(z_ip) > 0 else np.nan,
                }
        # Per-pitch 3×3 zone effectiveness (finer grid matching hitter analysis)
        zone_eff_3x3 = {}
        x_edges = np.array([-1.5, -0.5, 0.5, 1.5])
        y_edges = np.array([0.5, 2.0, 3.0, 4.5])
        loc_grp3 = loc_grp.copy()
        loc_grp3["xbin"] = np.clip(np.digitize(loc_grp3["PlateLocSide"], x_edges) - 1, 0, 2)
        loc_grp3["ybin"] = np.clip(np.digitize(loc_grp3["PlateLocHeight"], y_edges) - 1, 0, 2)
        for yb in range(3):
            for xb in range(3):
                zdf = loc_grp3[(loc_grp3["xbin"] == xb) & (loc_grp3["ybin"] == yb)]
                if len(zdf) >= 5:
                    z_sw = zdf[zdf["PitchCall"].isin(SWING_CALLS)]
                    z_wh = zdf[zdf["PitchCall"] == "StrikeSwinging"]
                    z_csw = zdf[zdf["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])]
                    z_ip = zdf[zdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                    zone_eff_3x3[(xb, yb)] = {
                        "n": len(zdf),
                        "whiff_pct": len(z_wh) / max(len(z_sw), 1) * 100 if len(z_sw) > 0 else np.nan,
                        "csw_pct": len(z_csw) / len(zdf) * 100,
                        "ev_against": z_ip["ExitSpeed"].mean() if len(z_ip) > 0 else np.nan,
                    }
        # Compute effective velocity estimate
        eff_velo = np.nan
        if loc_grp["RelSpeed"].notna().any() and len(loc_grp) >= 5:
            loc_adj = (loc_grp["PlateLocHeight"] - 2.5) * 1.5 + loc_grp["PlateLocSide"].abs() * (-0.5)
            eff_velo = (loc_grp["RelSpeed"] + loc_adj).mean()
        # Barrel% against
        barrels_against = int(is_barrel_mask(ip).sum()) if len(ip) > 0 else 0
        barrel_pct_against = barrels_against / max(len(ip), 1) * 100 if len(ip) > 0 else np.nan
        # Extension
        ext_val = grp["Extension"].mean() if "Extension" in grp.columns and grp["Extension"].notna().any() else np.nan
        arsenal["pitches"][pt_name] = {
            "usage_pct": len(grp) / len(pdf) * 100,
            "avg_velo": grp["RelSpeed"].mean() if grp["RelSpeed"].notna().any() else np.nan,
            "avg_spin": grp["SpinRate"].mean() if grp["SpinRate"].notna().any() else np.nan,
            "ivb": grp["InducedVertBreak"].mean() if "InducedVertBreak" in grp.columns and grp["InducedVertBreak"].notna().any() else np.nan,
            "hb": grp["HorzBreak"].mean() if "HorzBreak" in grp.columns and grp["HorzBreak"].notna().any() else np.nan,
            "whiff_pct": len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else np.nan,
            "csw_pct": len(csw) / len(grp) * 100,
            "chase_pct": len(oz_sw) / max(len(oz_grp), 1) * 100 if len(oz_grp) > 0 else np.nan,
            "ev_against": ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan,
            "barrel_pct_against": barrel_pct_against,
            # Raw sample sizes for shrinkage / confidence.
            "n_swings": int(len(sw)),
            "n_whiffs": int(len(wh)),
            "n_csw": int(len(csw)),
            "n_inplay_ev": int(len(ip)),
            "n_oz_pitches": int(len(oz_grp)),
            "n_oz_swings": int(len(oz_sw)),
            "n_barrels_against": int(barrels_against),
            "stuff_plus": stuff_plus,
            "command_plus": cmd_map.get(pt_name, np.nan),
            "count": len(grp),
            "eff_velo": eff_velo,
            "extension": ext_val,
            "zone_eff": zone_eff,
            "zone_eff_3x3": zone_eff_3x3,
        }

        # ── Count-group performance metrics (whiff, CSW, chase, EV by count group) ──
        # Uses same _COUNT_GROUPS as contextual usage below.
        from analytics.zone_vulnerability import _COUNT_GROUPS as _CG_PERF, _count_group_for as _cg_for_perf
        _cg_grp = grp.dropna(subset=["Balls", "Strikes"]).copy()
        if len(_cg_grp) >= 5:
            _cg_grp["_balls"] = pd.to_numeric(_cg_grp["Balls"], errors="coerce").astype("Int64")
            _cg_grp["_strikes"] = pd.to_numeric(_cg_grp["Strikes"], errors="coerce").astype("Int64")
            _cg_grp["_cg"] = _cg_grp.apply(lambda r: _cg_for_perf(int(r["_balls"]), int(r["_strikes"])), axis=1)
            _count_perf: dict = {}
            for _cg_name in _CG_PERF:
                _cg_rows = _cg_grp[_cg_grp["_cg"] == _cg_name]
                _n_cg = len(_cg_rows)
                if _n_cg < 5:
                    continue
                _cg_sw = _cg_rows[_cg_rows["PitchCall"].isin(SWING_CALLS)]
                _cg_wh = _cg_rows[_cg_rows["PitchCall"] == "StrikeSwinging"]
                _cg_csw = _cg_rows[_cg_rows["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])]
                _cg_oz = _cg_rows[_oz.reindex(_cg_rows.index, fill_value=False)]
                _cg_oz_sw = _cg_oz[_cg_oz["PitchCall"].isin(SWING_CALLS)]
                _cg_ip = _cg_rows[_cg_rows["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                _n_sw = len(_cg_sw)
                _n_oz = len(_cg_oz)
                _n_ip = len(_cg_ip)
                _count_perf[_cg_name] = {
                    "whiff_pct": len(_cg_wh) / max(_n_sw, 1) * 100 if _n_sw >= 3 else np.nan,
                    "csw_pct": len(_cg_csw) / _n_cg * 100,
                    "chase_pct": len(_cg_oz_sw) / max(_n_oz, 1) * 100 if _n_oz >= 3 else np.nan,
                    "ev_against": _cg_ip["ExitSpeed"].mean() if _n_ip >= 3 else np.nan,
                    "n": _n_cg,
                    "n_swings": _n_sw,
                    "n_oz_pitches": _n_oz,
                    "n_inplay_ev": _n_ip,
                }
            arsenal["pitches"][pt_name]["count_perf"] = _count_perf
        else:
            arsenal["pitches"][pt_name]["count_perf"] = {}

    # ── Contextual usage: count-group, platoon, count×platoon, transition ──
    from analytics.zone_vulnerability import _COUNT_GROUPS, _count_group_for

    # Precompute: need Balls/Strikes/BatterSide columns as numeric/normalized
    _ctx_pdf = pdf.dropna(subset=["Balls", "Strikes", "BatterSide", "TaggedPitchType"]).copy()
    _ctx_pdf["_balls"] = pd.to_numeric(_ctx_pdf["Balls"], errors="coerce").astype("Int64")
    _ctx_pdf["_strikes"] = pd.to_numeric(_ctx_pdf["Strikes"], errors="coerce").astype("Int64")
    _ctx_pdf["_side"] = _ctx_pdf["BatterSide"].astype(str).str.strip().str.upper().str[0]  # "L" or "R"
    _ctx_pdf["_cg"] = _ctx_pdf.apply(lambda r: _count_group_for(int(r["_balls"]), int(r["_strikes"])), axis=1)
    _ctx_total = len(_ctx_pdf)

    for pt_name_ctx, pt_data_ctx in arsenal["pitches"].items():
        pt_rows = _ctx_pdf[_ctx_pdf["TaggedPitchType"] == pt_name_ctx]

        # Tier 1: Count-group usage
        count_usage = {}
        for cg_name in _COUNT_GROUPS:
            cg_rows = _ctx_pdf[_ctx_pdf["_cg"] == cg_name]
            n_cg = len(cg_rows)
            if n_cg >= 10:
                count_usage[cg_name] = {
                    "pct": len(pt_rows[pt_rows["_cg"] == cg_name]) / n_cg * 100,
                    "n": n_cg,
                }
        pt_data_ctx["count_usage"] = count_usage

        # Tier 2: Platoon usage
        platoon_usage = {}
        for side in ["L", "R"]:
            side_rows = _ctx_pdf[_ctx_pdf["_side"] == side]
            n_side = len(side_rows)
            if n_side >= 15:
                platoon_usage[f"vs_{side}"] = {
                    "pct": len(pt_rows[pt_rows["_side"] == side]) / n_side * 100,
                    "n": n_side,
                }
        pt_data_ctx["platoon_usage"] = platoon_usage

        # Tier 3: Count × Platoon
        count_platoon_usage = {}
        for cg_name in _COUNT_GROUPS:
            for side in ["L", "R"]:
                cp_rows = _ctx_pdf[(_ctx_pdf["_cg"] == cg_name) & (_ctx_pdf["_side"] == side)]
                n_cp = len(cp_rows)
                if n_cp >= 8:
                    key = f"{cg_name}_vs_{side}"
                    count_platoon_usage[key] = {
                        "pct": len(pt_rows[(pt_rows["_cg"] == cg_name) & (pt_rows["_side"] == side)]) / n_cp * 100,
                        "n": n_cp,
                    }
        pt_data_ctx["count_platoon_usage"] = count_platoon_usage

    # Tier 4: Transition usage (computed once for all pitches, stored per pitch)
    # Sort by game/PA/pitch number to get sequential ordering
    _sort_cols = [c for c in ["GameID", "Batter", "PAofInning", "PitchNo"] if c in _ctx_pdf.columns]
    if len(_sort_cols) >= 2:
        _ctx_sorted = _ctx_pdf.sort_values(_sort_cols).copy()
        _ctx_sorted["_prev_pitch"] = _ctx_sorted.groupby(
            [c for c in ["GameID", "Batter", "PAofInning"] if c in _ctx_sorted.columns]
        )["TaggedPitchType"].shift(1)
        _trans = _ctx_sorted.dropna(subset=["_prev_pitch"])

        for pt_name_ctx, pt_data_ctx in arsenal["pitches"].items():
            transition_usage = {}
            for prev_pt in arsenal["pitches"]:
                after_rows = _trans[_trans["_prev_pitch"] == prev_pt]
                n_after = len(after_rows)
                if n_after >= 5:
                    transition_usage[f"after_{prev_pt}"] = {
                        "pct": len(after_rows[after_rows["TaggedPitchType"] == pt_name_ctx]) / n_after * 100,
                        "n": n_after,
                    }
            pt_data_ctx["transition_usage"] = transition_usage
    else:
        for pt_data_ctx in arsenal["pitches"].values():
            pt_data_ctx["transition_usage"] = {}

    # ── Usage stabilisation: blend current + prior season for reliability ──
    # Pitchers develop over time, so raw usage_pct is kept for display.
    # For the RE engine's reliability dampening we compute a *stabilised*
    # usage that blends current-season usage with the prior season, capping
    # the YoY swing to ±MAX_DELTA pp.  This prevents a newly-added pitch
    # (e.g. 2% sinker that was 0% last year) from being treated as if the
    # pitcher throws it regularly, while giving credit to pitches that had
    # prior-season evidence even if current-season sample is small.
    MAX_USAGE_DELTA = 8.0  # max pp change assumed realistic per season
    if "Season" in pdf.columns and arsenal["pitches"]:
        valid_seasons = sorted([int(s) for s in pdf["Season"].dropna().unique() if int(s) > 0])
        if len(valid_seasons) >= 2:
            current_season = valid_seasons[-1]
            prior_season = valid_seasons[-2]
            prior_pdf = pdf[pdf["Season"] == prior_season]
            current_pdf = pdf[pdf["Season"] == current_season]
            if len(prior_pdf) >= 30 and len(current_pdf) >= 20:
                prior_usage = {}
                for pt, g in prior_pdf.groupby("TaggedPitchType"):
                    if len(g) >= 3:
                        prior_usage[pt] = len(g) / len(prior_pdf) * 100
                current_usage = {}
                for pt, g in current_pdf.groupby("TaggedPitchType"):
                    if len(g) >= 3:
                        current_usage[pt] = len(g) / len(current_pdf) * 100
                for pt_name, pt_data in arsenal["pitches"].items():
                    prior = prior_usage.get(pt_name, 0.0)
                    cur = current_usage.get(pt_name, pt_data["usage_pct"])
                    # Cap the delta
                    delta = cur - prior
                    if abs(delta) > MAX_USAGE_DELTA:
                        stabilised = prior + MAX_USAGE_DELTA * (1 if delta > 0 else -1)
                    else:
                        stabilised = cur
                    pt_data["stabilised_usage"] = max(stabilised, 0.0)
                    pt_data["prior_season_usage"] = prior
                # Renormalize stabilised so they sum to 100%
                total_stab = sum(p.get("stabilised_usage", p["usage_pct"]) for p in arsenal["pitches"].values())
                if total_stab > 0:
                    for pt_data in arsenal["pitches"].values():
                        if "stabilised_usage" in pt_data:
                            pt_data["stabilised_usage"] = pt_data["stabilised_usage"] / total_stab * 100

    # Tunnel scores and pitch pair sequencing results
    tunnels = _compute_tunnel_score(pdf, tunnel_pop=tunnel_pop)
    arsenal["tunnels"] = tunnels
    arsenal["sequences"] = _compute_pitch_pair_results(pdf, data, tunnel_df=tunnels)
    return arsenal


def _get_opp_pitcher_profile(tm, pitcher_name, team, pitch_df=None):
    """Extract opponent pitcher profile from TrueMedia — full enrichment."""
    trad = _tm_player(_tm_team(tm["pitching"]["traditional"], team), pitcher_name)
    rate = _tm_player(_tm_team(tm["pitching"]["rate"], team), pitcher_name)
    mov = _tm_player(_tm_team(tm["pitching"]["movement"], team), pitcher_name)
    pt = _tm_player(_tm_team(tm["pitching"]["pitch_types"], team), pitcher_name)
    pr = _tm_player(_tm_team(tm["pitching"]["pitch_rates"], team), pitcher_name)
    exit_d = _tm_player(_tm_team(tm["pitching"]["exit"], team), pitcher_name)
    ht = _tm_player(_tm_team(tm["pitching"]["hit_types"], team), pitcher_name)
    pl = _tm_player(_tm_team(tm["pitching"].get("pitch_locations", pd.DataFrame()), team), pitcher_name)
    xr = _tm_player(_tm_team(tm["pitching"].get("expected_rate", pd.DataFrame()), team), pitcher_name)
    pc = _tm_player(_tm_team(tm["pitching"].get("pitch_counts", pd.DataFrame()), team), pitcher_name)
    throws = trad.iloc[0].get("throwsHand", "?") if not trad.empty else "?"
    if throws in [None, "", "?"] or (isinstance(throws, float) and pd.isna(throws)):
        throws = _infer_hand_from_pitch_df(pitch_df, pitcher_name, role="pitcher") if pitch_df is not None else "?"
    profile = {
        "name": pitcher_name,
        "throws": throws,
        # Traditional
        "era": _safe_num(trad, "ERA"), "fip": _safe_num(trad, "FIP"),
        "ip": _safe_num(trad, "IP"), "gs": _safe_num(trad, "GS"),
        "k9": _safe_num(trad, "K/9"), "bb9": _safe_num(trad, "BB/9"),
        "whip": _safe_num(trad, "WHIP"), "hr9": _safe_num(trad, "HR/9"),
        # Rate
        "xfip": _safe_num(rate, "xFIP"), "woba": _safe_num(rate, "wOBA"),
        "lob_pct": _safe_num(rate, "LOB%"), "k_pct": _safe_num(rate, "K%"),
        "bb_pct": _safe_num(rate, "BB%"), "ops_against": _safe_num(rate, "OPS"),
        # Movement
        "velo": _safe_num(mov, "Vel"), "max_velo": _safe_num(mov, "MxVel"),
        "velo_range": _safe_num(mov, "VelRange"), "spin": _safe_num(mov, "Spin"),
        "ivb": _safe_num(mov, "IndVertBrk"), "hb": _safe_num(mov, "HorzBrk"),
        "extension": _safe_num(mov, "Extension"), "eff_velo": _safe_num(mov, "EffectVel"),
        "vaa": _safe_num(mov, "VertApprAngle"),
        # Pitch rates / command
        "chase_pct": _safe_num(pr, "Chase%"), "swstrk_pct": _safe_num(pr, "SwStrk%"),
        "contact_pct": _safe_num(pr, "Contact%"), "miss_pct": _safe_num(pr, "Miss%"),
        "inzone_pct": _safe_num(pr, "InZone%"), "comploc_pct": _safe_num(pr, "CompLoc%"),
        "swing_pct": _safe_num(pr, "Swing%"), "callstrk_pct": _safe_num(pr, "CallStrk%"),
        "p_per_bf": _safe_num(pr, "P/BF"),
        # Pitch locations
        "loc_high_pct": _safe_num(pl, "High%"), "loc_vmid_pct": _safe_num(pl, "VMid%"),
        "loc_low_pct": _safe_num(pl, "Low%"), "loc_inside_pct": _safe_num(pl, "Inside%"),
        "loc_hmid_pct": _safe_num(pl, "HMid%"), "loc_outside_pct": _safe_num(pl, "Outside%"),
        "loc_uphalf_pct": _safe_num(pl, "UpHalf%"), "loc_lowhalf_pct": _safe_num(pl, "LowHalf%"),
        "loc_inhalf_pct": _safe_num(pl, "InHalf%"), "loc_outhalf_pct": _safe_num(pl, "OutHalf%"),
        # Exit data
        "ev_against": _safe_num(exit_d, "ExitVel"), "barrel_pct": _safe_num(exit_d, "Barrel%"),
        "hard_hit_pct": _safe_num(exit_d, "HardOut"), "launch_ang": _safe_num(exit_d, "LaunchAng"),
        # Hit types
        "gb_pct": _safe_num(ht, "Ground%"), "fb_pct": _safe_num(ht, "Fly%"),
        "ld_pct": _safe_num(ht, "Line%"),
        # Expected
        "xavg": _safe_num(xr, "xAVG"), "xslg": _safe_num(xr, "xSLG"),
        "xwoba": _safe_num(xr, "xWOBA"),
        # Arsenal
        "pitch_mix": {},
        # Pitch counts (raw numbers for sample size)
        "total_pitches": _safe_num(pc, "P"),
    }
    # Build pitch mix
    bad_pitch_labels = {"UN", "UNK", "UNKNOWN", "UNDEFINED", "OTHER", "NONE", "NULL", "NAN", ""}
    bad_pitch_labels |= {str(v).strip().upper() for v in PITCH_TYPES_TO_DROP}
    if not pt.empty:
        for tm_col, trackman_name in TM_PITCH_PCT_COLS.items():
            v = _safe_num(pt, tm_col)
            if not pd.isna(v) and v >= MIN_PITCH_USAGE_PCT and str(trackman_name).strip().upper() not in bad_pitch_labels:
                profile["pitch_mix"][trackman_name] = v
    # Derive: primary pitch (highest usage)
    if profile["pitch_mix"]:
        profile["primary_pitch"] = max(profile["pitch_mix"], key=profile["pitch_mix"].get)
        # Putaway candidates: offspeed/breaking pitches (non-hard)
        _hard = {"Fastball", "Sinker", "Cutter"}
        profile["putaway_candidates"] = {p: u for p, u in profile["pitch_mix"].items() if p not in _hard and u >= 5}
        # Location tendencies summary
        h = profile["loc_high_pct"]
        l = profile["loc_low_pct"]
        i = profile["loc_inside_pct"]
        o = profile["loc_outside_pct"]
        parts = []
        if not pd.isna(h) and h >= 30:
            parts.append(f"high ({h:.0f}%)")
        if not pd.isna(l) and l >= 35:
            parts.append(f"low ({l:.0f}%)")
        if not pd.isna(i) and i >= 30:
            parts.append(f"inside ({i:.0f}%)")
        if not pd.isna(o) and o >= 30:
            parts.append(f"outside ({o:.0f}%)")
        profile["location_tendency"] = ", ".join(parts) if parts else "balanced"
    else:
        profile["primary_pitch"] = None
        profile["putaway_candidates"] = {}
        profile["location_tendency"] = "unknown"
    return profile


def _get_our_hitter_profile(data, batter_name, season_filter=None):
    """Extract Davidson hitter profile from Trackman data — enriched with zones, counts, pitch-class."""
    bdf = filter_davidson(data, role="batter")
    bdf = bdf[bdf["Batter"] == batter_name].copy()
    if season_filter:
        bdf = bdf[bdf["Season"].isin(season_filter)]
    bdf = normalize_pitch_types(bdf)
    if len(bdf) < 20:
        return None
    bats = safe_mode(bdf["BatterSide"], "Right")
    batter_zones = _build_batter_zones(data)
    _iz = in_zone_mask(bdf, batter_zones, batter_col="Batter")
    _oz = ~_iz & bdf["PlateLocSide"].notna() & bdf["PlateLocHeight"].notna()
    profile = {"name": batter_name, "bats": bats, "total_pitches": len(bdf), "by_pitch_type": {}}
    for pt_name, grp in bdf.groupby("TaggedPitchType"):
        if len(grp) < 10:
            continue
        sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
        wh = grp[grp["PitchCall"] == "StrikeSwinging"]
        ip = grp[grp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        barrels = int(is_barrel_mask(ip).sum()) if len(ip) > 0 else 0
        pt_entry = {
            "seen": len(grp),
            "swing_pct": len(sw) / len(grp) * 100,
            "whiff_pct": len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else np.nan,
            "avg_ev": ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan,
            "barrel_pct": barrels / max(len(ip), 1) * 100 if len(ip) > 0 else np.nan,
            "hard_hit_pct": (len(ip[ip["ExitSpeed"] >= 95]) / max(len(ip), 1) * 100) if len(ip) > 0 else np.nan,
        }
        # Per-pitch contact depth
        if "EffectiveVelo" in grp.columns and "RelSpeed" in grp.columns:
            cd_df = ip.dropna(subset=["EffectiveVelo", "RelSpeed"])
            if len(cd_df) > 0:
                pt_entry["contact_depth"] = (cd_df["EffectiveVelo"] - cd_df["RelSpeed"]).mean()
            else:
                pt_entry["contact_depth"] = np.nan
        else:
            pt_entry["contact_depth"] = np.nan
        # Per-pitch bat speed proxy
        if "RelSpeed" in ip.columns and len(ip) > 0:
            bs_df = ip.dropna(subset=["RelSpeed"])
            if len(bs_df) > 0:
                pt_entry["bat_speed"] = ((bs_df["ExitSpeed"] - 0.2 * bs_df["RelSpeed"]) / 1.2).mean()
            else:
                pt_entry["bat_speed"] = np.nan
        else:
            pt_entry["bat_speed"] = np.nan
        # Hard-hit launch angle
        ip_la = ip.dropna(subset=["Angle"])
        hh_ip = ip_la[ip_la["ExitSpeed"] >= 95] if len(ip_la) > 0 else ip_la
        pt_entry["hard_hit_la"] = hh_ip["Angle"].median() if len(hh_ip) >= 3 else np.nan
        profile["by_pitch_type"][pt_name] = pt_entry

    all_sw = bdf[bdf["PitchCall"].isin(SWING_CALLS)]
    all_ip = bdf[bdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
    oz_pitches = bdf[_oz.reindex(bdf.index, fill_value=False)]
    oz_sw = oz_pitches[oz_pitches["PitchCall"].isin(SWING_CALLS)]
    # overall stats: EV, barrel rate, sweet spot%
    all_barrels = int(is_barrel_mask(all_ip).sum()) if len(all_ip) > 0 else 0
    all_ip_la = all_ip.dropna(subset=["Angle"]) if "Angle" in all_ip.columns else pd.DataFrame()
    sweet_spot_n = int(((all_ip_la["Angle"] >= 8) & (all_ip_la["Angle"] <= 32)).sum()) if len(all_ip_la) > 0 else 0
    profile["overall"] = {
        "avg_ev": all_ip["ExitSpeed"].mean() if len(all_ip) > 0 else np.nan,
        "barrel_pct": all_barrels / max(len(all_ip), 1) * 100 if len(all_ip) > 0 else np.nan,
        "barrel_count": all_barrels,
        "sweet_spot_pct": sweet_spot_n / max(len(all_ip_la), 1) * 100 if len(all_ip_la) > 0 else np.nan,
        "hard_hit_pct": (all_ip["ExitSpeed"] >= 87).sum() / max(len(all_ip), 1) * 100 if len(all_ip) > 0 else np.nan,
    }

    # ── Zone quadrant performance ──
    has_loc = bdf["PlateLocSide"].notna() & bdf["PlateLocHeight"].notna()
    loc_df = bdf[has_loc].copy()
    side_adj = _adjust_side_for_bats(loc_df, bats=bats)
    if side_adj is not None:
        loc_df["side_adj"] = side_adj
        side_col = "side_adj"
    else:
        side_col = "PlateLocSide"
    zone_mid_height = (ZONE_HEIGHT_BOT + ZONE_HEIGHT_TOP) / 2  # ~2.5 ft
    zone_quads = {
        "up_in":    loc_df[(loc_df["PlateLocHeight"] >= zone_mid_height) & (loc_df[side_col] <= 0)],
        "up_away":  loc_df[(loc_df["PlateLocHeight"] >= zone_mid_height) & (loc_df[side_col] > 0)],
        "down_in":  loc_df[(loc_df["PlateLocHeight"] < zone_mid_height) & (loc_df["PlateLocHeight"] >= ZONE_HEIGHT_BOT) & (loc_df[side_col] <= 0)],
        "down_away":loc_df[(loc_df["PlateLocHeight"] < zone_mid_height) & (loc_df["PlateLocHeight"] >= ZONE_HEIGHT_BOT) & (loc_df[side_col] > 0)],
        "heart":    loc_df[(loc_df["PlateLocHeight"].between(zone_mid_height - 0.4, zone_mid_height + 0.4)) & (loc_df[side_col].abs() <= 0.4)],
        "chase_up": loc_df[loc_df["PlateLocHeight"] > ZONE_HEIGHT_TOP],
        "chase_down":loc_df[loc_df["PlateLocHeight"] < ZONE_HEIGHT_BOT],
        "chase_in": loc_df[(loc_df[side_col] < -ZONE_SIDE) & loc_df["PlateLocHeight"].between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP)],
        "chase_away":loc_df[(loc_df[side_col] > ZONE_SIDE) & loc_df["PlateLocHeight"].between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP)],
    }
    profile["zones"] = {}
    for zname, zdf in zone_quads.items():
        if len(zdf) < 5:
            profile["zones"][zname] = {"n": len(zdf), "swing_pct": np.nan, "whiff_pct": np.nan, "avg_ev": np.nan}
            continue
        z_sw = zdf[zdf["PitchCall"].isin(SWING_CALLS)]
        z_wh = zdf[zdf["PitchCall"] == "StrikeSwinging"]
        z_ip = zdf[zdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        profile["zones"][zname] = {
            "n": len(zdf),
            "swing_pct": len(z_sw) / len(zdf) * 100,
            "whiff_pct": len(z_wh) / max(len(z_sw), 1) * 100 if len(z_sw) > 0 else np.nan,
            "avg_ev": z_ip["ExitSpeed"].mean() if len(z_ip) > 0 else np.nan,
        }

    # ── Count EV Grid (per-count performance) ──
    profile["count_ev_grid"] = {}
    if "Balls" in bdf.columns and "Strikes" in bdf.columns:
        cdf = bdf.dropna(subset=["Balls", "Strikes"]).copy()
        cdf["_b"] = cdf["Balls"].astype(int)
        cdf["_s"] = cdf["Strikes"].astype(int)
        for b_val in range(4):
            for s_val in range(3):
                mask = (cdf["_b"] == b_val) & (cdf["_s"] == s_val)
                g = cdf[mask]
                if len(g) < 5:
                    continue
                c_sw = g[g["PitchCall"].isin(SWING_CALLS)]
                c_ip = g[g["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                count_key = f"{b_val}-{s_val}"
                profile["count_ev_grid"][count_key] = {
                    "n": len(g),
                    "swing_pct": len(c_sw) / len(g) * 100,
                    "avg_ev": c_ip["ExitSpeed"].mean() if len(c_ip) > 0 else np.nan,
                }

    # ── Count group performance (kept for backwards compat) ──
    _count_groups = {
        "early": [(0, 0), (1, 0), (0, 1)],
        "ahead": [(2, 0), (3, 0), (3, 1), (2, 1)],
        "behind": [(0, 2), (1, 2)],
        "even": [(1, 1), (2, 2)],
        "full": [(3, 2)],
    }
    profile["by_count"] = {}
    if "Balls" in bdf.columns and "Strikes" in bdf.columns:
        cdf = bdf.dropna(subset=["Balls", "Strikes"]).copy()
        cdf["_b"] = cdf["Balls"].astype(int)
        cdf["_s"] = cdf["Strikes"].astype(int)
        for cg_name, counts in _count_groups.items():
            mask = pd.Series(False, index=cdf.index)
            for b, s in counts:
                mask |= (cdf["_b"] == b) & (cdf["_s"] == s)
            g = cdf[mask]
            if len(g) < 5:
                profile["by_count"][cg_name] = {"n": len(g), "swing_pct": np.nan, "whiff_pct": np.nan, "avg_ev": np.nan}
                continue
            c_sw = g[g["PitchCall"].isin(SWING_CALLS)]
            c_wh = g[g["PitchCall"] == "StrikeSwinging"]
            c_ip = g[g["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            profile["by_count"][cg_name] = {
                "n": len(g),
                "swing_pct": len(c_sw) / len(g) * 100,
                "whiff_pct": len(c_wh) / max(len(c_sw), 1) * 100 if len(c_sw) > 0 else np.nan,
                "avg_ev": c_ip["ExitSpeed"].mean() if len(c_ip) > 0 else np.nan,
            }

    # ── Pitch-class performance (Hard vs Offspeed) ──
    _hard_types = {"Fastball", "Sinker", "Cutter"}
    _os_types = {"Slider", "Curveball", "Changeup", "Splitter", "Sweeper", "Knuckle Curve"}
    profile["by_pitch_class"] = {}
    for cls_name, cls_types in [("hard", _hard_types), ("offspeed", _os_types)]:
        g = bdf[bdf["TaggedPitchType"].isin(cls_types)]
        if len(g) < 5:
            profile["by_pitch_class"][cls_name] = {"n": len(g), "whiff_pct": np.nan, "avg_ev": np.nan, "chase_pct": np.nan}
            continue
        cls_sw = g[g["PitchCall"].isin(SWING_CALLS)]
        cls_wh = g[g["PitchCall"] == "StrikeSwinging"]
        cls_ip = g[g["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        cls_oz = g[g.index.isin(oz_pitches.index)]
        cls_oz_sw = cls_oz[cls_oz["PitchCall"].isin(SWING_CALLS)]
        profile["by_pitch_class"][cls_name] = {
            "n": len(g),
            "whiff_pct": len(cls_wh) / max(len(cls_sw), 1) * 100 if len(cls_sw) > 0 else np.nan,
            "avg_ev": cls_ip["ExitSpeed"].mean() if len(cls_ip) > 0 else np.nan,
            "chase_pct": len(cls_oz_sw) / max(len(cls_oz), 1) * 100 if len(cls_oz) > 0 else np.nan,
        }

    sp = _compute_swing_path(bdf)
    profile["swing_path"] = sp if sp else None

    # ── Discipline Metrics ──
    iz_mask = _iz.reindex(bdf.index, fill_value=False)
    oz_mask = _oz.reindex(bdf.index, fill_value=False)
    all_swings = bdf[bdf["PitchCall"].isin(SWING_CALLS)]
    all_whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"]
    iz_pitches = bdf[iz_mask]
    iz_swings = iz_pitches[iz_pitches["PitchCall"].isin(SWING_CALLS)]
    iz_contacts = iz_pitches[iz_pitches["PitchCall"].isin(CONTACT_CALLS)]
    oz_pitches_d = bdf[oz_mask]
    oz_swings_d = oz_pitches_d[oz_pitches_d["PitchCall"].isin(SWING_CALLS)]
    # PA-based stats — count unique PAs using composite PA identifier
    _pa_cols = ["GameID", "Inning", "PAofInning", "Batter"]
    if all(c in bdf.columns for c in _pa_cols):
        pa_count = bdf.drop_duplicates(subset=_pa_cols).shape[0]
        ks = bdf[bdf["KorBB"] == "Strikeout"].drop_duplicates(subset=_pa_cols).shape[0] if "KorBB" in bdf.columns else 0
        bbs = bdf[bdf["KorBB"] == "Walk"].drop_duplicates(subset=_pa_cols).shape[0] if "KorBB" in bdf.columns else 0
    else:
        pa_count = bdf["PitchofPA"].eq(1).sum() if "PitchofPA" in bdf.columns else np.nan
        ks = len(bdf[bdf["KorBB"] == "Strikeout"]) if "KorBB" in bdf.columns else 0
        bbs = len(bdf[bdf["KorBB"] == "Walk"]) if "KorBB" in bdf.columns else 0
    profile["discipline"] = {
        "chase_pct": len(oz_swings_d) / max(len(oz_pitches_d), 1) * 100 if len(oz_pitches_d) > 0 else np.nan,
        "whiff_pct": len(all_whiffs) / max(len(all_swings), 1) * 100 if len(all_swings) > 0 else np.nan,
        "swing_pct": len(all_swings) / max(len(bdf), 1) * 100,
        "z_swing_pct": len(iz_swings) / max(len(iz_pitches), 1) * 100 if len(iz_pitches) > 0 else np.nan,
        "z_contact_pct": len(iz_contacts) / max(len(iz_swings), 1) * 100 if len(iz_swings) > 0 else np.nan,
        "k_pct": ks / max(pa_count, 1) * 100 if not pd.isna(pa_count) and pa_count > 0 else np.nan,
        "bb_pct": bbs / max(pa_count, 1) * 100 if not pd.isna(pa_count) and pa_count > 0 else np.nan,
    }

    # ── First-Pitch Approach ──
    if "PitchofPA" in bdf.columns:
        fp = bdf[bdf["PitchofPA"] == 1]
        if len(fp) >= 5:
            fp_sw = fp[fp["PitchCall"].isin(SWING_CALLS)]
            fp_wh = fp[fp["PitchCall"] == "StrikeSwinging"]
            fp_ip = fp[fp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            profile["first_pitch"] = {
                "n": len(fp),
                "swing_pct": len(fp_sw) / len(fp) * 100,
                "whiff_pct": len(fp_wh) / max(len(fp_sw), 1) * 100 if len(fp_sw) > 0 else np.nan,
                "avg_ev": fp_ip["ExitSpeed"].mean() if len(fp_ip) > 0 else np.nan,
            }
        else:
            profile["first_pitch"] = None
    else:
        profile["first_pitch"] = None

    # ── 2-Strike Adjustments ──
    if "Strikes" in bdf.columns:
        two_k = bdf[bdf["Strikes"].astype(float) >= 2]
        pre_2k = bdf[bdf["Strikes"].astype(float) < 2]
        if len(two_k) >= 10:
            tk_sw = two_k[two_k["PitchCall"].isin(SWING_CALLS)]
            tk_wh = two_k[two_k["PitchCall"] == "StrikeSwinging"]
            tk_ip = two_k[two_k["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            pre_sw = pre_2k[pre_2k["PitchCall"].isin(SWING_CALLS)]
            pre_wh = pre_2k[pre_2k["PitchCall"] == "StrikeSwinging"]
            profile["two_strike"] = {
                "n": len(two_k),
                "swing_pct": len(tk_sw) / max(len(two_k), 1) * 100,
                "whiff_pct": len(tk_wh) / max(len(tk_sw), 1) * 100 if len(tk_sw) > 0 else np.nan,
                "avg_ev": tk_ip["ExitSpeed"].mean() if len(tk_ip) > 0 else np.nan,
                "pre_2k_whiff": len(pre_wh) / max(len(pre_sw), 1) * 100 if len(pre_sw) > 0 else np.nan,
            }
        else:
            profile["two_strike"] = None
    else:
        profile["two_strike"] = None

    # ── Zone Coverage Grid (5x5) ──
    zone_grid = {}
    if len(loc_df) >= 20:
        bside = {"Right": "Right", "Left": "Left"}.get(bats, "Right")
        h_edges = [-2, -0.83, -0.28, 0.28, 0.83, 2]
        v_edges = [0.5, 1.5, 2.17, 2.83, 3.5, 4.5]
        col_labels = ["Far In", "Inside", "Middle", "Outside", "Far Out"]
        row_labels = ["Low+", "Low", "Mid", "High", "High+"]
        if bside == "Left":
            col_labels = col_labels[::-1]
        for ri in range(5):
            for ci in range(5):
                cell_mask = (
                    (loc_df["PlateLocSide"] >= h_edges[ci]) &
                    (loc_df["PlateLocSide"] < h_edges[ci + 1]) &
                    (loc_df["PlateLocHeight"] >= v_edges[ri]) &
                    (loc_df["PlateLocHeight"] < v_edges[ri + 1])
                )
                cell = loc_df[cell_mask]
                if len(cell) < 5:
                    continue
                cell_sw = cell[cell["PitchCall"].isin(SWING_CALLS)]
                cell_wh = cell[cell["PitchCall"] == "StrikeSwinging"]
                cell_con = cell[cell["PitchCall"].isin(CONTACT_CALLS)]
                cell_ip = cell[cell["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                key = f"{row_labels[ri]}_{col_labels[ci]}"
                zone_grid[key] = {
                    "n": len(cell),
                    "swing_pct": len(cell_sw) / len(cell) * 100,
                    "whiff_pct": len(cell_wh) / max(len(cell_sw), 1) * 100 if len(cell_sw) > 0 else np.nan,
                    "contact_rate": len(cell_con) / max(len(cell_sw), 1) * 100 if len(cell_sw) > 0 else np.nan,
                    "avg_ev": cell_ip["ExitSpeed"].mean() if len(cell_ip) > 0 else np.nan,
                }
    profile["zone_grid"] = zone_grid

    # ── 1A. Per-pitch-type zone_damage (6-zone map) ──
    zone_mid = (ZONE_HEIGHT_BOT + ZONE_HEIGHT_TOP) / 2  # ~2.5 ft
    for pt_name, pt_entry in profile["by_pitch_type"].items():
        pt_grp = loc_df[loc_df["TaggedPitchType"] == pt_name]
        pt_ip = pt_grp[pt_grp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        if len(pt_ip) < 10:
            pt_entry["zone_damage"] = {}
            continue
        # 6-zone: up/mid/down × in/out (relative to batter hand)
        is_rhh = bats == "Right"
        zd = {}
        for v_label, v_lo, v_hi in [("up", zone_mid + 0.5, 5.0),
                                      ("mid", zone_mid - 0.5, zone_mid + 0.5),
                                      ("down", 0.0, zone_mid - 0.5)]:
            for h_label, h_cond in [("in", pt_ip["PlateLocSide"] < 0 if is_rhh else pt_ip["PlateLocSide"] > 0),
                                     ("out", pt_ip["PlateLocSide"] >= 0 if is_rhh else pt_ip["PlateLocSide"] <= 0)]:
                cell = pt_ip[(pt_ip["PlateLocHeight"] >= v_lo) &
                             (pt_ip["PlateLocHeight"] < v_hi) & h_cond]
                if len(cell) < 5:
                    continue
                cell_barrels = int(is_barrel_mask(cell).sum()) if len(cell) > 0 else 0
                zd[f"{v_label}_{h_label}"] = {
                    "n": len(cell),
                    "avg_ev": cell["ExitSpeed"].mean(),
                    "barrels": cell_barrels,
                }
        pt_entry["zone_damage"] = zd

    # ── 1B. 2-strike whiff by pitch class (hard vs offspeed) ──
    if profile.get("two_strike") and "Strikes" in bdf.columns:
        two_k = bdf[bdf["Strikes"].astype(float) >= 2]
        if len(two_k) >= 10:
            _hard_cls = {"Fastball", "Sinker", "Cutter"}
            _os_cls = {"Slider", "Curveball", "Changeup", "Splitter", "Sweeper", "Knuckle Curve"}
            for cls_label, cls_set in [("whiff_hard", _hard_cls), ("whiff_os", _os_cls)]:
                cls_2k = two_k[two_k["TaggedPitchType"].isin(cls_set)]
                cls_2k_sw = cls_2k[cls_2k["PitchCall"].isin(SWING_CALLS)]
                cls_2k_wh = cls_2k[cls_2k["PitchCall"] == "StrikeSwinging"]
                if len(cls_2k_sw) >= 8:
                    profile["two_strike"][cls_label] = len(cls_2k_wh) / len(cls_2k_sw) * 100

    # ── 1C. Barrel zone concentration (top 3 cells by barrel count) ──
    barrel_zones = []
    if len(loc_df) >= 20 and zone_grid:
        all_ip_loc = loc_df[loc_df["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed", "Angle"])
        if len(all_ip_loc) >= 10:
            bside_1c = {"Right": "Right", "Left": "Left"}.get(bats, "Right")
            h_edges_1c = [-2, -0.83, -0.28, 0.28, 0.83, 2]
            v_edges_1c = [0.5, 1.5, 2.17, 2.83, 3.5, 4.5]
            col_labels_1c = ["Far In", "Inside", "Middle", "Outside", "Far Out"]
            row_labels_1c = ["Low+", "Low", "Mid", "High", "High+"]
            if bside_1c == "Left":
                col_labels_1c = col_labels_1c[::-1]
            cell_barrels_list = []
            for ri in range(5):
                for ci in range(5):
                    cm = ((all_ip_loc["PlateLocSide"] >= h_edges_1c[ci]) &
                          (all_ip_loc["PlateLocSide"] < h_edges_1c[ci + 1]) &
                          (all_ip_loc["PlateLocHeight"] >= v_edges_1c[ri]) &
                          (all_ip_loc["PlateLocHeight"] < v_edges_1c[ri + 1]))
                    cell_ip = all_ip_loc[cm]
                    if len(cell_ip) < 8:
                        continue
                    b_count = int(is_barrel_mask(cell_ip).sum())
                    if b_count >= 2:
                        ev_avg = cell_ip["ExitSpeed"].mean()
                        key_1c = f"{row_labels_1c[ri]}_{col_labels_1c[ci]}"
                        cell_barrels_list.append((key_1c, b_count, round(ev_avg, 1)))
            cell_barrels_list.sort(key=lambda x: x[1], reverse=True)
            barrel_zones = cell_barrels_list[:3]
    profile["barrel_zones"] = barrel_zones

    # ── 1D. Spray direction (pull/center/oppo) ──
    spray = {}
    if "Direction" in bdf.columns:
        dir_ip = bdf[(bdf["PitchCall"] == "InPlay") & bdf["Direction"].notna()].copy()
        if len(dir_ip) >= 20:
            is_rhh_spray = bats == "Right"
            if is_rhh_spray:
                pull_mask = dir_ip["Direction"] < -15
                oppo_mask = dir_ip["Direction"] > 15
            else:
                pull_mask = dir_ip["Direction"] > 15
                oppo_mask = dir_ip["Direction"] < -15
            center_mask = ~pull_mask & ~oppo_mask
            total = len(dir_ip)
            spray = {
                "pull_pct": round(pull_mask.sum() / total * 100, 1),
                "center_pct": round(center_mask.sum() / total * 100, 1),
                "oppo_pct": round(oppo_mask.sum() / total * 100, 1),
            }
    profile["spray"] = spray

    return profile



# ── Scoring Engine ──

_hard_pitches = {"Fastball", "Sinker", "Cutter"}
_swing_map = {
    "Fastball": "swing_vs_hard", "Sinker": "swing_vs_hard", "Cutter": "swing_vs_hard",
    "Slider": "swing_vs_sl", "Sweeper": "swing_vs_sl",
    "Curveball": "swing_vs_cb", "Knuckle Curve": "swing_vs_cb",
    "Changeup": "swing_vs_ch", "Splitter": "swing_vs_ch",
}


def _lookup_tunnel(a, b, tun_df):
    """Lookup tunnel score and grade for a pitch pair."""
    if not isinstance(tun_df, pd.DataFrame) or tun_df.empty:
        return np.nan, "-"
    m = tun_df[((tun_df["Pitch A"]==a)&(tun_df["Pitch B"]==b))|((tun_df["Pitch A"]==b)&(tun_df["Pitch B"]==a))]
    if m.empty:
        return np.nan, "-"
    return m.iloc[0]["Tunnel Score"], m.iloc[0]["Grade"]


def _lookup_seq(setup, follow, seq_df):
    """Lookup sequence whiff% and chase% for a pitch pair. Returns (whiff%, chase%)."""
    if not isinstance(seq_df, pd.DataFrame) or seq_df.empty:
        return np.nan, np.nan
    m = seq_df[(seq_df["Setup Pitch"]==setup)&(seq_df["Follow Pitch"]==follow)]
    if m.empty:
        return np.nan, np.nan
    return m.iloc[0]["Whiff%"], m.iloc[0].get("Chase%", np.nan)


def _score_ev(ev):
    """Convert EV against to 0-100 (lower EV = higher score)."""
    if pd.isna(ev):
        return np.nan
    return float(np.clip((95 - ev) / 15 * 100, 0, 100))


def _score_whiff(wh):
    if pd.isna(wh):
        return np.nan
    return float(np.clip(wh / 50 * 100, 0, 100))


def _score_putaway(pa):
    if pd.isna(pa):
        return np.nan
    return float(np.clip(pa / 40 * 100, 0, 100))


def _score_k(k):
    if pd.isna(k):
        return np.nan
    return float(np.clip(k / 35 * 100, 0, 100))


def _score_stuff(stuff):
    if pd.isna(stuff):
        return np.nan
    return float(np.clip((stuff - 70) / 60 * 100, 0, 100))


def _score_cmd(cmd):
    if pd.isna(cmd):
        return np.nan
    return float(np.clip((cmd - 80) / 40 * 100, 0, 100))


def _weighted_score(parts, weights):
    vals = [(p, w) for p, w in zip(parts, weights) if pd.notna(p)]
    if not vals:
        return np.nan
    s = sum(p * w for p, w in vals)
    wsum = sum(w for _, w in vals)
    return s / wsum if wsum else np.nan


def _deception_flag(tunnel):
    if pd.isna(tunnel):
        return ""
    if tunnel >= 60:
        return f"+ Deception edge (Tunnel {tunnel:.0f})"
    if tunnel <= 40:
        return f"- Deception weak (Tunnel {tunnel:.0f})"
    return ""


def _assign_tactical_tags(rows):
    if not rows:
        return rows
    whiffs = [r.get("Whiff%") for r in rows]
    ks = [r.get("K%") for r in rows]
    evs = [r.get("Avg EV") for r in rows]

    def _best_idx(vals, func=max):
        vals_clean = [v for v in vals if pd.notna(v)]
        if not vals_clean:
            return None
        target = func(vals_clean)
        for i, v in enumerate(vals):
            if pd.notna(v) and v == target:
                return i
        return None

    idx_put = _best_idx(ks, max)
    idx_ev = _best_idx(evs, min)
    idx_wh = _best_idx(whiffs, max)

    tags = {}
    for idx, label in [(idx_put, "Best putaway"), (idx_ev, "Best weak‑contact"), (idx_wh, "Best whiff")]:
        if idx is not None and idx not in tags:
            tags[idx] = label
    for i in range(len(rows)):
        if i not in tags:
            tags[i] = "Best overall"
        rows[i]["Tag"] = tags[i]
    return rows


def _filter_redundant_sequences(seqs, min_unique=3, max_keep=2):
    if not seqs:
        return []
    def _uniq_count(seq):
        pitches = [p.strip() for p in seq.split("→")]
        return len(set(pitches)), tuple(sorted(set(pitches)))
    filtered = []
    seen_sets = set()
    for s in seqs:
        uniq_cnt, uniq_set = _uniq_count(s["Seq"])
        if uniq_cnt < min_unique:
            continue
        if uniq_set in seen_sets:
            continue
        seen_sets.add(uniq_set)
        filtered.append(s)
        if len(filtered) >= max_keep:
            return filtered
    if filtered:
        return filtered
    for s in seqs:
        uniq_cnt, uniq_set = _uniq_count(s["Seq"])
        if uniq_cnt < 2:
            continue
        if uniq_set in seen_sets:
            continue
        seen_sets.add(uniq_set)
        filtered.append(s)
        if len(filtered) >= max_keep:
            break
    return filtered


def _top_tunnel_pairs(tun_df, seq_df=None, pitch_metrics=None, top_n=2):
    """Rank pitch pairs by outcomes-first score with tunnel secondary."""
    has_tunnel = isinstance(tun_df, pd.DataFrame) and not tun_df.empty
    has_seq = isinstance(seq_df, pd.DataFrame) and not seq_df.empty
    if not has_tunnel and not has_seq:
        return []
    df = tun_df.copy() if has_tunnel else pd.DataFrame()
    if has_tunnel:
        df["Tunnel Score"] = pd.to_numeric(df.get("Tunnel Score"), errors="coerce")
        df = df.dropna(subset=["Tunnel Score"])

    # Aggregate pair stats from sequence table (unordered pair)
    pair_stats = {}
    if has_seq:
        seq = seq_df.copy()
        seq["Whiff%"] = pd.to_numeric(seq.get("Whiff%"), errors="coerce")
        seq["Avg EV"] = pd.to_numeric(seq.get("Avg EV"), errors="coerce")
        seq["Putaway%"] = pd.to_numeric(seq.get("Putaway%"), errors="coerce")
        seq["K%"] = pd.to_numeric(seq.get("K%"), errors="coerce")
        seq["Count"] = pd.to_numeric(seq.get("Count"), errors="coerce").fillna(0)
        for _, r in seq.iterrows():
            key = tuple(sorted([r["Setup Pitch"], r["Follow Pitch"]]))
            stats = pair_stats.setdefault(key, {
                "whiff_sum": 0.0, "whiff_w": 0.0,
                "ev_sum": 0.0, "ev_w": 0.0,
                "put_sum": 0.0, "put_w": 0.0,
                "k_sum": 0.0, "k_w": 0.0,
                "count": 0.0,
                "count_ab": 0.0,
                "count_ba": 0.0,
            })
            w = max(float(r["Count"]), 1.0)
            stats["count"] += w
            a_key, b_key = key
            if r["Setup Pitch"] == a_key and r["Follow Pitch"] == b_key:
                stats["count_ab"] += w
            else:
                stats["count_ba"] += w
            if pd.notna(r["Whiff%"]):
                stats["whiff_sum"] += r["Whiff%"] * w; stats["whiff_w"] += w
            if pd.notna(r["Avg EV"]):
                stats["ev_sum"] += r["Avg EV"] * w; stats["ev_w"] += w
            if pd.notna(r["Putaway%"]):
                stats["put_sum"] += r["Putaway%"] * w; stats["put_w"] += w
            if pd.notna(r["K%"]):
                stats["k_sum"] += r["K%"] * w; stats["k_w"] += w

    tunnel_map = {}
    if not df.empty:
        for _, row in df.iterrows():
            key = tuple(sorted([row["Pitch A"], row["Pitch B"]]))
            tunnel_map.setdefault(key, []).append(pd.to_numeric(row.get("Tunnel Score"), errors="coerce"))

    candidates = set(pair_stats.keys()) | set(tunnel_map.keys())
    if not candidates:
        return []

    valid_pitches = set(pitch_metrics.keys()) if pitch_metrics else None
    out = []
    for key in candidates:
        a, b = key
        if valid_pitches and (a not in valid_pitches or b not in valid_pitches):
            continue
        ps = pair_stats.get(key, {})
        wh = ps.get("whiff_sum", 0) / ps.get("whiff_w", 1) if ps.get("whiff_w", 0) > 0 else np.nan
        ev = ps.get("ev_sum", 0) / ps.get("ev_w", 1) if ps.get("ev_w", 0) > 0 else np.nan
        put = ps.get("put_sum", 0) / ps.get("put_w", 1) if ps.get("put_w", 0) > 0 else np.nan
        k_pct = ps.get("k_sum", 0) / ps.get("k_w", 1) if ps.get("k_w", 0) > 0 else np.nan
        if pd.isna(k_pct):
            k_pct = put
        t_vals = tunnel_map.get(key, [])
        tunnel = float(np.nanmean(t_vals)) if t_vals else np.nan
        stuff_avg = np.nan
        cmd_avg = np.nan
        if pitch_metrics:
            stuff_avg = np.nanmean([pitch_metrics.get(a, {}).get("stuff", np.nan),
                                    pitch_metrics.get(b, {}).get("stuff", np.nan)])
            cmd_avg = np.nanmean([pitch_metrics.get(a, {}).get("cmd", np.nan),
                                  pitch_metrics.get(b, {}).get("cmd", np.nan)])

        count_ab = ps.get("count_ab", 0)
        count_ba = ps.get("count_ba", 0)
        if count_ab == count_ba:
            label_a, label_b = sorted([a, b])
        else:
            label_a, label_b = (a, b) if count_ab >= count_ba else (b, a)

        score = _weighted_score(
            [_score_whiff(wh), _score_k(k_pct), _score_ev(ev), tunnel],
            [0.35, 0.25, 0.25, 0.15],
        )
        out.append({
            "Pair": f"{label_a} → {label_b}",
            "Whiff%": wh,
            "K%": k_pct,
            "Avg EV": ev,
            "Tunnel": tunnel,
            "Stuff+": stuff_avg,
            "Cmd+": cmd_avg,
            "Score": score,
            "Pairs": ps.get("count", np.nan),
        })
    out = sorted(out, key=lambda x: (x["Score"] if pd.notna(x["Score"]) else -1), reverse=True)
    return out[:top_n]


def _top_sequences(seq_df, pitch_metrics=None, length=3, top_n=2):
    """Rank 3- or 4-pitch sequences by tunnel + whiff + K%/putaway + EV + Stuff+ + Command+."""
    if not isinstance(seq_df, pd.DataFrame) or seq_df.empty:
        return []
    df = seq_df.copy()
    df["Whiff%"] = pd.to_numeric(df.get("Whiff%"), errors="coerce")
    df["Tunnel Score"] = pd.to_numeric(df.get("Tunnel Score"), errors="coerce")
    df["Avg EV"] = pd.to_numeric(df.get("Avg EV"), errors="coerce")
    df["Putaway%"] = pd.to_numeric(df.get("Putaway%"), errors="coerce")
    df["K%"] = pd.to_numeric(df.get("K%"), errors="coerce")
    df["Count"] = pd.to_numeric(df.get("Count"), errors="coerce").fillna(0)
    df = df.dropna(subset=["Tunnel Score"])
    if df.empty:
        return []

    pair_map = {}
    valid_pitches = set(pitch_metrics.keys()) if pitch_metrics else None
    for _, row in df.iterrows():
        if valid_pitches and (
            row["Setup Pitch"] not in valid_pitches or row["Follow Pitch"] not in valid_pitches
        ):
            continue
        pair_map[(row["Setup Pitch"], row["Follow Pitch"])] = {
            "whiff": row["Whiff%"],
            "tunnel": row["Tunnel Score"],
            "ev": row["Avg EV"],
            "k": row["K%"] if pd.notna(row["K%"]) else row["Putaway%"],
            "count": row["Count"],
        }

    out_map = {}
    for (a, b), stats in pair_map.items():
        out_map.setdefault(a, []).append((b, stats))

    def _wavg(vals, wts):
        mask = [pd.notna(v) for v in vals]
        if not any(mask):
            return np.nan
        v = [val for val, m in zip(vals, mask) if m]
        w = [wt for wt, m in zip(wts, mask) if m]
        return float(np.average(v, weights=w))

    def _seq_stats(pairs):
        counts = np.array([max(p["count"], 1) for p in pairs], dtype=float)
        whiff_avg = _wavg([p["whiff"] for p in pairs], counts)
        tunnel_avg = _wavg([p["tunnel"] for p in pairs], counts)
        ev_avg = _wavg([p["ev"] for p in pairs], counts)
        k_avg = _wavg([p["k"] for p in pairs], counts)
        score = _weighted_score(
            [_score_whiff(whiff_avg), _score_k(k_avg), _score_ev(ev_avg), tunnel_avg],
            [0.35, 0.25, 0.25, 0.15],
        )
        return whiff_avg, tunnel_avg, ev_avg, k_avg, score, int(counts.sum())

    results = []
    if length == 3:
        for a, outs in out_map.items():
            for b, s1 in outs:
                for c, s2 in out_map.get(b, []):
                    wh, tn, ev, k_pct, sc, n = _seq_stats([s1, s2])
                    stuff_avg = np.nan
                    cmd_avg = np.nan
                    if pitch_metrics:
                        stuff_avg = np.nanmean([pitch_metrics.get(p, {}).get("stuff", np.nan) for p in [a, b, c]])
                        cmd_avg = np.nanmean([pitch_metrics.get(p, {}).get("cmd", np.nan) for p in [a, b, c]])
                        sc = _weighted_score(
                            [_score_whiff(wh), _score_k(k_pct), _score_ev(ev), tn],
                            [0.35, 0.25, 0.25, 0.15],
                        )
                    results.append({"Seq": f"{a} → {b} → {c}", "Whiff%": wh, "Tunnel": tn, "Avg EV": ev,
                                    "K%": k_pct, "Stuff+": stuff_avg, "Cmd+": cmd_avg, "Score": sc, "Pairs": n})
    elif length == 4:
        for a, outs in out_map.items():
            for b, s1 in outs:
                for c, s2 in out_map.get(b, []):
                    for d, s3 in out_map.get(c, []):
                        wh, tn, ev, k_pct, sc, n = _seq_stats([s1, s2, s3])
                        stuff_avg = np.nan
                        cmd_avg = np.nan
                        if pitch_metrics:
                            stuff_avg = np.nanmean([pitch_metrics.get(p, {}).get("stuff", np.nan) for p in [a, b, c, d]])
                            cmd_avg = np.nanmean([pitch_metrics.get(p, {}).get("cmd", np.nan) for p in [a, b, c, d]])
                            sc = _weighted_score(
                                [_score_whiff(wh), _score_k(k_pct), _score_ev(ev), tn],
                                [0.35, 0.25, 0.25, 0.15],
                            )
                        results.append({"Seq": f"{a} → {b} → {c} → {d}", "Whiff%": wh, "Tunnel": tn, "Avg EV": ev,
                                        "K%": k_pct, "Stuff+": stuff_avg, "Cmd+": cmd_avg, "Score": sc, "Pairs": n})
    else:
        return []

    results.sort(key=lambda x: (x["Score"] if pd.notna(x["Score"]) else -1), reverse=True)
    return results[:top_n]


def _pitch_score_composite(pt_name, pt_data, hd, tun_df, platoon_label="Neutral", arsenal_data=None):
    """Unified composite score (0-100) for one pitch vs one hitter.
    Combines all available pitcher Trackman + hitter TrueMedia factors.

    pt_data: dict with our_whiff, our_csw, our_chase, our_ev_against, stuff_plus, etc.
    hd: dict with hitter data (whiff_2k_hard, whiff_2k_os, chase_pct, k_pct, etc.)
    tun_df: DataFrame of tunnel grades
    arsenal_data: dict with per-pitch arsenal info (for EffVelo, IVB, zone_eff)
    """
    components, weights = [], []
    is_hard = pt_name in _hard_pitches

    # 1. Stuff+ (13%) — raw pitch quality: 70-130 → 0-100
    sp = pt_data.get("stuff_plus", np.nan)
    if not pd.isna(sp):
        components.append(min(max((sp - 70) / 60 * 100, 0), 100)); weights.append(13)

    # 2. Our Whiff% (10%) — 0-50% → 0-100
    wh = pt_data.get("our_whiff", pt_data.get("whiff_pct", np.nan))
    if not pd.isna(wh):
        components.append(min(wh / 50 * 100, 100)); weights.append(10)

    # 3. Our CSW% (7%) — 0-40% → 0-100
    csw = pt_data.get("our_csw", pt_data.get("csw_pct", np.nan))
    if not pd.isna(csw):
        components.append(min(csw / 40 * 100, 100)); weights.append(7)

    # 4. Their 2K Whiff Rate (8%) — matched to pitch class (hard/offspeed) + our hand
    their_2k = hd.get("whiff_2k_hard" if is_hard else "whiff_2k_os", np.nan)
    if not pd.isna(their_2k):
        components.append(min(their_2k / 40 * 100, 100)); weights.append(8)

    # 5. Chase Exploitation (8%) — our chase generation × their chase tendency
    our_chase = pt_data.get("our_chase", pt_data.get("chase_pct", np.nan))
    their_chase = hd.get("chase_pct", np.nan)
    if not pd.isna(our_chase) and not pd.isna(their_chase):
        components.append(min((our_chase * their_chase) / (30 * 30) * 100, 100)); weights.append(8)

    # 6. Their Swing Rate vs Pitch Class (5%)
    sw_key = _swing_map.get(pt_name, "")
    their_sw = hd.get(sw_key, np.nan) if sw_key else np.nan
    if not pd.isna(their_sw):
        components.append(min(their_sw / 70 * 100, 100)); weights.append(5)

    # 7. Tunnel Score (7%) — BEST tunnel score involving this pitch (not average)
    if isinstance(tun_df, pd.DataFrame) and not tun_df.empty:
        t_m = tun_df[(tun_df["Pitch A"] == pt_name) | (tun_df["Pitch B"] == pt_name)]
        if not t_m.empty:
            components.append(t_m["Tunnel Score"].max()); weights.append(10)

    # 8. EV Against (5%) — penalize pitches we get hit hard on
    ev_ag = pt_data.get("our_ev_against", pt_data.get("ev_against", np.nan))
    if not pd.isna(ev_ag):
        components.append(min(max((96 - ev_ag) / 18 * 100, 0), 100)); weights.append(5)

    # 9. K-Prone Factor (4%) — high K hitters are exploitable (equal for all pitches)
    k_pct = hd.get("k_pct", np.nan)
    if not pd.isna(k_pct):
        k_score = min(max((k_pct - 10) / 25 * 100, 0), 100)
        components.append(k_score); weights.append(4)

    # 10. Platoon Factor (4%)
    plat_score = 50
    if "Adv" in platoon_label:
        plat_score = 80
    elif "Disadv" in platoon_label:
        plat_score = 25
    components.append(plat_score); weights.append(4)

    # 11. wOBA Split (6%) — how well hitter performs vs our hand
    woba_split = hd.get("woba_split", np.nan)
    if not pd.isna(woba_split):
        components.append(min(max((0.450 - woba_split) / 0.250 * 100, 0), 100)); weights.append(6)

    # 12. Hitter Contact% (3%) — LOW contact = more exploitable
    contact = hd.get("contact_pct", np.nan)
    if not pd.isna(contact):
        components.append(min(max((95 - contact) / 35 * 100, 0), 100)); weights.append(3)

    # 13. Hitter EV + Barrel weakness (3%)
    h_ev = hd.get("ev", np.nan)
    h_brl = hd.get("barrel_pct", np.nan)
    if not pd.isna(h_ev):
        ev_weak = min(max((95 - h_ev) / 17 * 100, 0), 100)
        if not pd.isna(h_brl):
            brl_weak = min(max((15 - h_brl) / 13 * 100, 0), 100)
            components.append((ev_weak + brl_weak) / 2); weights.append(3)
        else:
            components.append(ev_weak); weights.append(3)

    # 14. IVB (4%) — fastball-only: high IVB = harder to square up
    ars_pt = arsenal_data or {}
    ivb_val = ars_pt.get("ivb", pt_data.get("ivb", np.nan))
    if is_hard and not pd.isna(ivb_val):
        # 10" IVB → 0, 16" → 50, 22"+ → 100
        components.append(min(max((ivb_val - 10) / 12 * 100, 0), 100)); weights.append(4)

    # 15. EffVelo (3%) — higher effective velocity = harder to catch up
    eff_velo = ars_pt.get("eff_velo", np.nan)
    if not pd.isna(eff_velo):
        # 82 → 0, 88 → 50, 94+ → 100
        components.append(min(max((eff_velo - 82) / 12 * 100, 0), 100)); weights.append(3)

    # 16. Our Usage (18%) — pitches we actually throw should rank higher; low-usage pitches
    #     have small samples and unreliable metrics. Steeper scaling separates mid-usage better.
    usage_pct = ars_pt.get("usage_pct", pt_data.get("usage", np.nan))
    if not pd.isna(usage_pct):
        # Steeper scaling: 0% → 0, 10% → 25, 20% → 50, 40%+ → 100
        # Floor of 30 for any pitch >= 15% usage so true secondaries aren't crushed
        # Ceiling of 15 for pitches < 5% usage (rarely thrown = unreliable)
        raw_usage = min(max(usage_pct / 40 * 100, 0), 100)
        if usage_pct < 5:
            raw_usage = min(raw_usage, 15)
        elif usage_pct >= 15:
            raw_usage = max(raw_usage, 30)
        components.append(raw_usage); weights.append(18)

    # 18. Raw Velo (3%) — 93 mph FB should score higher than 86 mph; harder to react to
    raw_velo = ars_pt.get("avg_velo", pt_data.get("velo", np.nan))
    if not pd.isna(raw_velo):
        if is_hard:
            # Hard: 82 → 0, 88 → 50, 94+ → 100
            components.append(min(max((raw_velo - 82) / 12 * 100, 0), 100)); weights.append(3)
        else:
            # Offspeed: big velo diff from hard stuff is good; 70 → 30, 78 → 55, 85+ → 80
            components.append(min(max((85 - raw_velo) / 15 * 100, 10), 90)); weights.append(2)

    # 19. Barrel% Against (3%) — low barrel rate = effective pitch, penalize hittable pitches
    brl_ag = ars_pt.get("barrel_pct_against", np.nan)
    if not pd.isna(brl_ag):
        # 0% → 100, 5% → 67, 10% → 33, 15%+ → 0
        components.append(min(max((15 - brl_ag) / 15 * 100, 0), 100)); weights.append(3)

    # 20. Horizontal Break (3%) — offspeed only: more HB = more sweep/run = harder to barrel
    hb_val = ars_pt.get("hb", pt_data.get("hb", np.nan))
    if not is_hard and not pd.isna(hb_val):
        abs_hb = abs(hb_val)
        # 2" → 0, 8" → 50, 14"+ → 100
        components.append(min(max((abs_hb - 2) / 12 * 100, 0), 100)); weights.append(3)

    # 21. InZoneSwing% (2%) — aggressive in-zone swingers are more exploitable
    iz_swing = hd.get("iz_swing_pct", np.nan)
    if not pd.isna(iz_swing):
        # 55% → 0, 65% → 50, 75%+ → 100  (high = they swing a lot in zone = exploitable)
        components.append(min(max((iz_swing - 55) / 20 * 100, 0), 100)); weights.append(2)

    # 22. Extension (2%) — longer extension = closer release = more deception
    ext = ars_pt.get("extension", np.nan)
    if not pd.isna(ext):
        # 5.0 ft → 0, 6.0 → 50, 7.0+ → 100
        components.append(min(max((ext - 5.0) / 2.0 * 100, 0), 100)); weights.append(2)

    # 23. Command+ (8%) — high command = more accurate; especially valuable for hard pitches
    #     that need precise location to avoid getting hit hard
    cmd_plus = ars_pt.get("command_plus", np.nan)
    if not pd.isna(cmd_plus):
        # 70 → 0, 100 → 50, 130+ → 100
        cmd_score = min(max((cmd_plus - 70) / 60 * 100, 0), 100)
        if is_hard:
            cmd_score = min(cmd_score * 1.2, 100)
        components.append(cmd_score); weights.append(8)

    # 24. Swing Hole Match (4%) — pitch type matched to hitter's zone vulnerability
    zone_vuln = hd.get("zone_vuln", {}) if hd else {}
    if zone_vuln.get("available"):
        _zvmap = {
            "Fastball": "vuln_up",
            "Sinker": "vuln_down",
            "Slider": lambda zv: np.nanmean([zv.get("vuln_down", np.nan), zv.get("vuln_away", np.nan)]),
            "Curveball": "vuln_chase_low",
            "Changeup": "vuln_down",
            "Sweeper": lambda zv: np.nanmean([zv.get("vuln_away", np.nan), zv.get("vuln_chase_away", np.nan)]),
            "Cutter": lambda zv: np.nanmean([zv.get("vuln_away", np.nan), zv.get("vuln_inside", np.nan)]),
            "Splitter": "vuln_chase_low",
            "Knuckle Curve": "vuln_chase_low",
        }
        zv_lookup = _zvmap.get(pt_name)
        zv_score = np.nan
        if callable(zv_lookup):
            zv_score = zv_lookup(zone_vuln)
        elif isinstance(zv_lookup, str):
            zv_score = zone_vuln.get(zv_lookup, np.nan)
        if not pd.isna(zv_score):
            components.append(min(max(zv_score, 0), 100)); weights.append(4)

    # 17. Zone Exploitation (5%) — cross our best zone with their zone weakness
    #     Formula: csw*0.6 + whiff*0.4, pitch-design multipliers (PZM),
    #     hitter exposure boosts from TrueMedia pitch location data.
    ze = ars_pt.get("zone_eff", {})
    if ze and hd:
        hitter_high = hd.get("high_pct", np.nan)
        hitter_low = hd.get("low_pct", np.nan)
        hitter_in = hd.get("inside_pct", np.nan)
        hitter_out = hd.get("outside_pct", np.nan)
        # Pitch-design zone multipliers — mirrors _get_pzm exactly (incl glove/arm)
        _ze_ivb = ars_pt.get("ivb", np.nan)
        if is_hard:
            if pt_name == "Sinker":
                _ze_pzm = {"up": 0.5, "down": 1.4, "chase_low": 1.3, "glove": 1.0, "arm": 1.0}
            elif not pd.isna(_ze_ivb) and _ze_ivb >= 16:
                _ze_pzm = {"up": 1.4, "down": 0.7, "chase_low": 0.4, "glove": 1.0, "arm": 1.0}
            elif not pd.isna(_ze_ivb) and _ze_ivb < 12:
                _ze_pzm = {"up": 0.7, "down": 1.3, "chase_low": 1.1, "glove": 1.0, "arm": 1.0}
            else:
                _ze_pzm = {"up": 1.15, "down": 0.9, "chase_low": 0.6, "glove": 1.0, "arm": 1.0}
        else:
            if pt_name in ("Curveball", "Knuckle Curve"):
                _ze_pzm = {"up": 0.2, "down": 1.2, "chase_low": 1.5, "glove": 0.8, "arm": 0.8}
            elif pt_name == "Changeup":
                _ze_pzm = {"up": 0.2, "down": 1.4, "chase_low": 1.4, "glove": 1.1, "arm": 1.0}
            else:
                _ze_pzm = {"up": 0.3, "down": 1.2, "chase_low": 1.3, "glove": 1.3, "arm": 0.8}
        # Map hitter inside/outside to pitcher arm/glove based on platoon
        _same = "Adv" in platoon_label
        _in_z = "arm" if _same else "glove"
        _out_z = "glove" if _same else "arm"
        best_exploit = 0
        for zn, zd in ze.items():
            if zd.get("n", 0) < 5:
                continue
            zone_whiff = zd.get("whiff_pct", 0) or 0
            zone_csw = zd.get("csw_pct", 0) or 0
            zone_quality = (zone_csw * 0.6 + zone_whiff * 0.4) * _ze_pzm.get(zn, 1.0)
            exposure_boost = 1.0
            if zn == "up" and not pd.isna(hitter_high) and hitter_high > 30:
                exposure_boost = 1.2
            elif zn == "down" and not pd.isna(hitter_low) and hitter_low > 35:
                exposure_boost = 1.15
            elif zn == _in_z and not pd.isna(hitter_in) and hitter_in > 28:
                exposure_boost = 1.15
            elif zn == _out_z and not pd.isna(hitter_out) and hitter_out > 28:
                exposure_boost = 1.15
            elif zn == "chase_low":
                chase_pct = hd.get("chase_pct", np.nan)
                if not pd.isna(chase_pct) and chase_pct > 28:
                    exposure_boost = 1.3
            best_exploit = max(best_exploit, zone_quality * exposure_boost)
        if best_exploit > 0:
            components.append(min(best_exploit / 40 * 100, 100)); weights.append(5)

    if not weights:
        return 50
    return sum(c * w for c, w in zip(components, weights)) / sum(weights)


def _build_3pitch_sequences(sorted_ps, hd, tun_df, seq_df):
    """Build best 3-pitch sequences: setup → bridge → putaway.
    HITTER-AWARE: Picks the putaway pitch based on the hitter's specific
    vulnerability, then finds the best setup/bridge path to get there.
    P1 (setup) must be a primary pitch (>= 15% usage) — you don't lead with
    a 10% sinker. Returns up to 3 sequences with different putaway pitches."""
    pitches = [name for name, data in sorted_ps if data.get("count", 0) >= 10]
    pitch_data = {name: data for name, data in sorted_ps if data.get("count", 0) >= 10}
    pitch_usage = {name: data.get("usage", 0) or 0 for name, data in sorted_ps}
    comp_scores = {name: data.get("score", 50) for name, data in sorted_ps}
    if len(pitches) < 2:
        return []
    # P1 candidates: must have meaningful usage (>= 15%), or fallback to top 2 by usage
    setup_candidates = [p for p in pitches if pitch_usage.get(p, 0) >= 15]
    if len(setup_candidates) < 2:
        setup_candidates = sorted(pitches, key=lambda p: pitch_usage.get(p, 0), reverse=True)[:2]

    # ── Step 1: Rank putaway candidates by hitter-specific vulnerability ──
    putaway_scores = {}
    for p in pitches:
        is_hard = p in _hard_pitches
        their_2k = hd.get("whiff_2k_hard" if is_hard else "whiff_2k_os", np.nan)
        comp = comp_scores.get(p, 50)
        whiff = pitch_data.get(p, {}).get("our_whiff", np.nan)
        chase = pitch_data.get(p, {}).get("our_chase", np.nan)
        score = comp * 0.50
        if not pd.isna(their_2k):
            score += min(their_2k / 40 * 100, 100) * 0.25
        if not pd.isna(whiff):
            score += min(whiff / 50 * 100, 100) * 0.15
        if not pd.isna(chase):
            score += min(chase / 40 * 100, 100) * 0.10
        putaway_scores[p] = score
    ranked_putaways = sorted(putaway_scores.items(), key=lambda x: x[1], reverse=True)

    # ── Step 2: For each putaway, find best setup → bridge path ──
    results = []
    for p3, p3_score in ranked_putaways:
        best_path = None
        best_path_score = -1
        for p2 in pitches:
            if p2 == p3:
                t_self, _ = _lookup_tunnel(p2, p3, tun_df)
                if pd.isna(t_self) or t_self <= 50:
                    continue
            for p1 in setup_candidates:  # P1 restricted to primary pitches
                if p1 == p2:
                    continue
                t12, g12 = _lookup_tunnel(p1, p2, tun_df)
                t23, g23 = _lookup_tunnel(p2, p3, tun_df)
                t12_bad = pd.isna(t12) or t12 < 25
                t23_bad = pd.isna(t23) or t23 < 25
                if t12_bad and t23_bad:
                    continue
                sw12, ch12 = _lookup_seq(p1, p2, seq_df)
                sw23, ch23 = _lookup_seq(p2, p3, seq_df)
                parts, wts = [], []
                # Tunnel quality (45%): t12 weight=18, t23 weight=27
                if not pd.isna(t12):
                    parts.append(t12); wts.append(18)
                if not pd.isna(t23):
                    parts.append(t23); wts.append(27)
                # Outcome effectiveness (40%): sw23=25, sw12=10, ch23=5
                if not pd.isna(sw23):
                    parts.append(min(sw23 / 50 * 100, 100)); wts.append(25)
                else:
                    parts.append(30); wts.append(25)
                if not pd.isna(sw12):
                    parts.append(min(sw12 / 50 * 100, 100)); wts.append(10)
                if not pd.isna(ch23):
                    parts.append(min(ch23 / 40 * 100, 100)); wts.append(5)
                # Pitch quality (15%): putaway composite=10, EffV gap=5
                parts.append(comp_scores.get(p3, 50)); wts.append(10)
                p1_effv = pitch_data.get(p1, {}).get("eff_velo", np.nan)
                p3_effv = pitch_data.get(p3, {}).get("eff_velo", np.nan)
                if not pd.isna(p1_effv) and not pd.isna(p3_effv):
                    gap = abs(p1_effv - p3_effv)
                    parts.append(min(25 + gap * 5, 100)); wts.append(5)
                else:
                    p1_velo = pitch_data.get(p1, {}).get("velo", np.nan)
                    p3_velo = pitch_data.get(p3, {}).get("velo", np.nan)
                    if not pd.isna(p1_velo) and not pd.isna(p3_velo):
                        gap = abs(p1_velo - p3_velo)
                        parts.append(min(25 + gap * 5, 100)); wts.append(5)
                if not wts:
                    continue
                path_score = sum(p*w for p,w in zip(parts, wts)) / sum(wts)
                if path_score > best_path_score:
                    best_path_score = path_score
                    ev_gap = abs(p1_effv - p3_effv) if not pd.isna(p1_effv) and not pd.isna(p3_effv) else np.nan
                    is_hard_p3 = p3 in _hard_pitches
                    their_2k = hd.get("whiff_2k_hard" if is_hard_p3 else "whiff_2k_os", np.nan)
                    best_path = {
                        "seq": f"{p1} → {p2} → {p3}", "p1": p1, "p2": p2, "p3": p3,
                        "score": round(p3_score * 0.35 + best_path_score * 0.65, 1),
                        "t12": t12, "t23": t23, "sw23": sw23, "their_2k": their_2k,
                        "effv_gap": ev_gap,
                    }
        if best_path:
            results.append(best_path)
        if len(results) >= 3:
            break
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _build_4pitch_sequence(top_seqs, sorted_ps, hd, tun_df, seq_df):
    """Extend the best 3-pitch sequence into a 4-pitch sequence.
    Tries appending a P4 after P3 (deeper at-bat / secondary putaway) and
    also tries prepending a P0 before P1 (early-count setup). Returns the
    best 4-pitch dict or None if fewer than 3 pitches available."""
    if not top_seqs:
        return None
    pitches = [name for name, data in sorted_ps if data.get("count", 0) >= 10]
    pitch_data = {name: data for name, data in sorted_ps if data.get("count", 0) >= 10}
    comp_scores = {name: data.get("score", 50) for name, data in sorted_ps}
    if len(pitches) < 3:
        return None

    best_4 = None
    best_4_score = -1

    for seq3 in top_seqs[:2]:  # try extending top 2 three-pitch sequences
        p1, p2, p3 = seq3["p1"], seq3["p2"], seq3["p3"]
        base_score = seq3["score"]

        # ── Option A: append P4 after P3 ──
        for p4 in pitches:
            if p4 == p3 and p4 == p2:
                continue  # avoid 3 in a row of same pitch
            t34, g34 = _lookup_tunnel(p3, p4, tun_df)
            sw34, ch34 = _lookup_seq(p3, p4, seq_df)
            parts, wts = [], []
            # Tunnel t34 (35%)
            if not pd.isna(t34):
                parts.append(t34); wts.append(35)
            # Sequence whiff sw34 (30%)
            if not pd.isna(sw34):
                parts.append(min(sw34 / 50 * 100, 100)); wts.append(30)
            else:
                parts.append(30); wts.append(30)
            # P4 putaway composite + hitter vulnerability (20%): comp=10, t2k=10
            parts.append(comp_scores.get(p4, 50)); wts.append(10)
            is_hard_p4 = p4 in _hard_pitches
            t2k_p4 = hd.get("whiff_2k_hard" if is_hard_p4 else "whiff_2k_os", np.nan)
            if not pd.isna(t2k_p4):
                parts.append(min(t2k_p4 / 40 * 100, 100)); wts.append(10)
            # Chase ch34 (15%)
            if not pd.isna(ch34):
                parts.append(min(ch34 / 40 * 100, 100)); wts.append(15)
            if not wts:
                continue
            ext_score = sum(p * w for p, w in zip(parts, wts)) / sum(wts)
            total = base_score * 0.50 + ext_score * 0.50
            if total > best_4_score:
                best_4_score = total
                best_4 = {
                    "seq": f"{p1} → {p2} → {p3} → {p4}",
                    "p1": p1, "p2": p2, "p3": p3, "p4": p4,
                    "score": round(total, 1),
                    "sw34": sw34, "t34": t34,
                    "mode": "append",
                }

        # ── Option B: prepend P0 before P1 ──
        for p0 in pitches:
            if p0 == p1 and p0 == p2:
                continue
            t01, g01 = _lookup_tunnel(p0, p1, tun_df)
            sw01, ch01 = _lookup_seq(p0, p1, seq_df)
            parts, wts = [], []
            # Tunnel t01 (35%)
            if not pd.isna(t01):
                parts.append(t01); wts.append(35)
            # Sequence whiff sw01 (25%)
            if not pd.isna(sw01):
                parts.append(min(sw01 / 50 * 100, 100)); wts.append(25)
            # P0 quality composite (25%)
            parts.append(comp_scores.get(p0, 50)); wts.append(25)
            if not wts:
                continue
            ext_score = sum(p * w for p, w in zip(parts, wts)) / sum(wts)
            total = base_score * 0.55 + ext_score * 0.45
            if total > best_4_score:
                best_4_score = total
                best_4 = {
                    "seq": f"{p0} → {p1} → {p2} → {p3}",
                    "p1": p0, "p2": p1, "p3": p2, "p4": p3,
                    "score": round(total, 1),
                    "sw34": _lookup_seq(p2, p3, seq_df)[0], "t34": _lookup_tunnel(p2, p3, tun_df)[0],
                    "mode": "prepend",
                }

    return best_4


def _score_pitcher_vs_hitter(arsenal, hitter_profile):
    """Score how well our pitcher's arsenal exploits an opponent hitter's weaknesses.
    Uses the unified _pitch_score_composite for all per-pitch and overall scoring."""
    if arsenal is None or hitter_profile is None:
        return None
    throws = "R" if arsenal["throws"] == "Right" else "L"
    bats = hitter_profile["bats"]
    platoon_label = "Neutral"
    if bats == "S":
        platoon_label = "Switch (Neutral)"
    elif (throws == "L" and bats == "L") or (throws == "R" and bats == "R"):
        platoon_label = "Platoon Adv"
    elif (throws == "L" and bats == "R") or (throws == "R" and bats == "L"):
        platoon_label = "Platoon Disadv"
    woba_key = "woba_lhp" if throws == "L" else "woba_rhp"
    woba_split = hitter_profile.get(woba_key, np.nan)
    hand_key = "lhp" if throws == "L" else "rhp"

    # Build hitter_data dict (used by composite scorer and passed to UI)
    hd = {
        "pa": hitter_profile.get("pa", np.nan), "ops": hitter_profile.get("ops", np.nan),
        "woba": hitter_profile.get("woba", np.nan),
        "k_pct": hitter_profile.get("k_pct", np.nan), "bb_pct": hitter_profile.get("bb_pct", np.nan),
        "chase_pct": hitter_profile.get("chase_pct", np.nan), "swstrk_pct": hitter_profile.get("swstrk_pct", np.nan),
        "contact_pct": hitter_profile.get("contact_pct", np.nan),
        "swing_pct": hitter_profile.get("swing_pct", np.nan),
        "iz_swing_pct": hitter_profile.get("iz_swing_pct", np.nan),
        "p_per_pa": hitter_profile.get("p_per_pa", np.nan),
        "ev": hitter_profile.get("ev", np.nan), "barrel_pct": hitter_profile.get("barrel_pct", np.nan),
        "gb_pct": hitter_profile.get("gb_pct", np.nan), "fb_pct": hitter_profile.get("fb_pct", np.nan),
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

    # Get tunnel data from arsenal
    tun_df = arsenal.get("tunnels", pd.DataFrame())

    # Score each pitch using the unified composite scorer
    pitch_scores = {}
    for pt_name, pt_data in arsenal["pitches"].items():
        is_hard = pt_name in _hard_pitches
        # Build pt_data dict compatible with composite scorer
        pd_compat = {
            "stuff_plus": pt_data.get("stuff_plus", np.nan),
            "our_whiff": pt_data.get("whiff_pct", np.nan),
            "our_csw": pt_data.get("csw_pct", np.nan),
            "our_chase": pt_data.get("chase_pct", np.nan),
            "our_ev_against": pt_data.get("ev_against", np.nan),
        }
        # Compute composite score
        comp_score = _pitch_score_composite(
            pt_name, pd_compat, hd, tun_df, platoon_label,
            arsenal_data=pt_data  # pass full arsenal pitch data for IVB/EffVelo/zone_eff
        )
        # Build reasons from notable factors
        reasons = []
        stuff = pt_data.get("stuff_plus", np.nan)
        if not pd.isna(stuff) and stuff >= 115:
            reasons.append(f"elite stuff ({stuff:.0f} S+)")
        our_whiff = pt_data.get("whiff_pct", np.nan)
        if not pd.isna(our_whiff) and our_whiff >= 35:
            reasons.append(f"high whiff ({our_whiff:.0f}%)")
        whiff_2k = hitter_profile.get(f"whiff_2k_{hand_key}_{'hard' if is_hard else 'os'}", np.nan)
        if not pd.isna(whiff_2k) and whiff_2k > 35:
            reasons.append(f"hitter whiffs {whiff_2k:.0f}% on 2K {'hard' if is_hard else 'offspeed'}")
        hitter_chase = hitter_profile.get("chase_pct", np.nan)
        our_chase = pt_data.get("chase_pct", np.nan)
        if not pd.isna(hitter_chase) and not pd.isna(our_chase) and hitter_chase > 28 and our_chase > 30:
            reasons.append(f"chaser ({hitter_chase:.0f}%) + our chase gen ({our_chase:.0f}%)")
        our_ev_against = pt_data.get("ev_against", np.nan)
        if not pd.isna(our_ev_against) and our_ev_against > 90:
            reasons.append(f"gets hit hard ({our_ev_against:.1f} EV against)")

        pitch_scores[pt_name] = {
            "score": round(comp_score, 1), "reasons": reasons,
            "our_whiff": our_whiff, "our_chase": our_chase,
            "our_csw": pt_data.get("csw_pct", np.nan),
            "our_ev_against": pt_data.get("ev_against", np.nan),
            "stuff_plus": stuff, "usage": pt_data["usage_pct"],
            "velo": pt_data["avg_velo"],
            "eff_velo": pt_data.get("eff_velo", np.nan),
            "spin": pt_data.get("avg_spin", np.nan),
            "ivb": pt_data.get("ivb", np.nan),
            "hb": pt_data.get("hb", np.nan),
            "count": pt_data.get("count", 0),
        }

    sorted_pitches = sorted(pitch_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    recommendations = []
    if sorted_pitches:
        best = sorted_pitches[0]
        recommendations.append(f"Lead with **{best[0]}** ({best[1]['score']:.0f})")
        offspeed = [(n, d) for n, d in sorted_pitches if n not in _hard_pitches]
        if offspeed:
            recommendations.append(f"Putaway: **{offspeed[0][0]}** ({offspeed[0][1]['score']:.0f})")
        elif len(sorted_pitches) > 1:
            recommendations.append(f"Secondary: **{sorted_pitches[1][0]}** ({sorted_pitches[1][1]['score']:.0f})")
    # Discipline-based notes
    hitter_bb = hitter_profile.get("bb_pct", np.nan)
    hitter_chase_pct = hitter_profile.get("chase_pct", np.nan)
    if not pd.isna(hitter_bb) and hitter_bb < 5:
        recommendations.append("Free swinger — attack the zone early")
    elif not pd.isna(hitter_bb) and hitter_bb > 14:
        recommendations.append("Patient hitter — don't nibble, pitch to contact")
    if not pd.isna(hitter_chase_pct) and hitter_chase_pct > 35:
        recommendations.append(f"Chases aggressively ({hitter_chase_pct:.0f}%) — expand early")
    gb_pct = hitter_profile.get("gb_pct", np.nan)
    if not pd.isna(gb_pct) and gb_pct > 55:
        recommendations.append(f"Ground ball hitter ({gb_pct:.0f}% GB) — pitch up in zone")
    if not pd.isna(woba_split) and woba_split >= 0.380:
        recommendations.append(f"⚠ Danger — {woba_split:.3f} wOBA vs {'LHP' if throws=='L' else 'RHP'}")
    elif not pd.isna(woba_split) and woba_split <= 0.260:
        recommendations.append(f"Exploitable — {woba_split:.3f} wOBA vs {'LHP' if throws=='L' else 'RHP'}")

    # Usage-weighted overall score from composite
    if pitch_scores:
        total_usage = sum(v["usage"] for v in pitch_scores.values())
        if total_usage > 0:
            overall = sum(v["score"] * v["usage"] for v in pitch_scores.values()) / total_usage
        else:
            overall = np.mean([v["score"] for v in pitch_scores.values()])
    else:
        overall = 50
    return {
        "hitter": hitter_profile["name"], "pitcher": arsenal["name"],
        "bats": bats, "platoon": platoon_label,
        "overall_score": overall,
        "pitch_scores": pitch_scores, "recommendations": recommendations,
        "hitter_data": hd,
    }


def _score_hitter_vs_pitcher(hitter_tm, pitcher_profile):
    """Score hitter vs pitcher using transparent per-pitch Edge system.
    Each pitch gets Advantage / Neutral / Vulnerable with readable reasons."""
    if hitter_tm is None or pitcher_profile is None:
        return None
    bats = hitter_tm["bats"]
    throws = pitcher_profile["throws"]
    platoon_factor, platoon_label = 1.0, "Neutral"
    if (throws == "R" and bats == "Left") or (throws == "L" and bats == "Right"):
        platoon_factor, platoon_label = 1.08, "Platoon Adv"
    elif (throws == "R" and bats == "Right") or (throws == "L" and bats == "Left"):
        platoon_factor, platoon_label = 0.92, "Same-Side"
    elif bats == "Switch":
        platoon_factor, platoon_label = 1.04, "Switch"

    _hard_set = {"Fastball", "Sinker", "Cutter"}
    zones = hitter_tm.get("zones", {})
    zone_grid = hitter_tm.get("zone_grid", {})

    def _timing_note(depth):
        if pd.isna(depth):
            return ""
        if depth > 0.5:
            return f"Out front +{depth:.1f}"
        if depth < -1.5:
            return f"Deep {depth:.1f}"
        return ""

    # 2A. Zone-weighted EV: cross per-pitch zone_damage with pitcher location %s
    def _zone_weighted_ev(our_pitch_data, pp):
        """Compute zone-weighted EV for a pitch type using pitcher location tendencies.
        Returns weighted EV or NaN if insufficient data.
        Only uses zone weighting when we have BOTH vertical and horizontal location data."""
        zd = our_pitch_data.get("zone_damage", {}) if our_pitch_data else {}
        if len(zd) < 2:
            return np.nan
        h_pct = pp.get("loc_high_pct", np.nan)
        vmid_pct = pp.get("loc_vmid_pct", np.nan)
        l_pct = pp.get("loc_low_pct", np.nan)
        in_pct = pp.get("loc_inside_pct", np.nan)
        out_pct = pp.get("loc_outside_pct", np.nan)
        # Only weight by location if we have real data — don't fabricate weights
        has_vert = not pd.isna(h_pct) or not pd.isna(l_pct)
        has_horiz = not pd.isna(in_pct) or not pd.isna(out_pct)
        if not has_vert and not has_horiz:
            # No location data — just average the zone EVs equally
            evs = [zdata.get("avg_ev", np.nan) for zdata in zd.values()]
            valid = [e for e in evs if not pd.isna(e)]
            return np.mean(valid) if valid else np.nan
        # Use actual location data, fill missing with uniform (33/50)
        v_h = h_pct if not pd.isna(h_pct) else 33
        v_m = vmid_pct if not pd.isna(vmid_pct) else 34
        v_l = l_pct if not pd.isna(l_pct) else 33
        h_i = in_pct if not pd.isna(in_pct) else 50
        h_o = out_pct if not pd.isna(out_pct) else 50
        zone_weights = {
            "up_in": v_h * h_i / 100, "up_out": v_h * h_o / 100,
            "mid_in": v_m * h_i / 100, "mid_out": v_m * h_o / 100,
            "down_in": v_l * h_i / 100, "down_out": v_l * h_o / 100,
        }
        weighted_ev, total_w = 0.0, 0.0
        for zkey, zdata in zd.items():
            ev = zdata.get("avg_ev", np.nan)
            if pd.isna(ev):
                continue
            w = zone_weights.get(zkey, 1.0)
            weighted_ev += ev * w
            total_w += w
        return weighted_ev / total_w if total_w > 0 else np.nan

    # 2B. Upgraded zone_cross: use per-pitch zone_damage first, fall back to zone_grid
    def _zone_cross_note(opp_pitch):
        """Cross pitcher location tendency with our per-pitch zone_damage or zone_grid."""
        h_pct = pitcher_profile.get("loc_high_pct", np.nan)
        l_pct = pitcher_profile.get("loc_low_pct", np.nan)
        # Find dominant vertical tendency
        best_tend, best_pct = "", 0
        if not pd.isna(h_pct) and h_pct > best_pct:
            best_tend, best_pct = "high", h_pct
        if not pd.isna(l_pct) and l_pct > best_pct:
            best_tend, best_pct = "low", l_pct
        if not best_tend or best_pct < 25:
            return ""
        # Map "high"/"low" to zone_damage keys
        zd_vert = "up" if best_tend == "high" else "down"
        # Try per-pitch zone_damage first
        our_pt_data = hitter_tm["by_pitch_type"].get(opp_pitch, {})
        zd = our_pt_data.get("zone_damage", {}) if our_pt_data else {}
        zd_cells = {k: v for k, v in zd.items() if k.startswith(zd_vert)}
        if zd_cells:
            best_zd_key = max(zd_cells, key=lambda k: zd_cells[k].get("avg_ev", 0)
                              if not pd.isna(zd_cells[k].get("avg_ev", np.nan)) else 0)
            best_zd = zd_cells[best_zd_key]
            cell_ev = best_zd.get("avg_ev", np.nan)
            barrels = best_zd.get("barrels", 0)
            if not pd.isna(cell_ev):
                loc_label = best_zd_key.replace("_", " ")
                pitch_label = opp_pitch.lower() + "s" if not opp_pitch.endswith("er") else opp_pitch.lower()
                if barrels >= 2:
                    return (f"They throw {pitch_label} {best_tend} ({best_pct:.0f}%) "
                            f"\u2014 we barrel {pitch_label} {loc_label} ({cell_ev:.0f} EV, {barrels} barrels)")
                ev_label = "weak" if cell_ev < 82 else ""
                return (f"They throw {pitch_label} {best_tend} ({best_pct:.0f}%) "
                        f"\u2014 we hit {loc_label} {pitch_label} at {cell_ev:.0f} EV"
                        + (f" ({ev_label})" if ev_label else ""))
        # Fall back to overall zone_grid
        match_keys = [k for k in zone_grid if best_tend.capitalize() in k.split("_")[0]]
        if not match_keys:
            return ""
        best_cell = max(match_keys, key=lambda k: zone_grid[k].get("avg_ev", 0)
                        if not pd.isna(zone_grid[k].get("avg_ev", np.nan)) else 0)
        cell_ev = zone_grid[best_cell].get("avg_ev", np.nan)
        if pd.isna(cell_ev):
            return ""
        label = best_cell.replace("_", " ").lower()
        return f"They go {best_tend} ({best_pct:.0f}%) \u2014 we hit {label} ({cell_ev:.0f} EV)"

    pitch_edges = []
    weighted_score, total_weight = 0.0, 0.0
    approach_notes = []

    for opp_pitch, opp_usage in pitcher_profile["pitch_mix"].items():
        if opp_usage < 5:
            continue
        our_data = hitter_tm["by_pitch_type"].get(opp_pitch, {})
        # 2A: Use zone-weighted EV for edge classification, keep raw for display
        our_ev_raw = our_data.get("avg_ev", np.nan) if our_data else np.nan
        our_ev_zw = _zone_weighted_ev(our_data, pitcher_profile)
        our_ev = our_ev_zw if not pd.isna(our_ev_zw) else our_ev_raw
        # Fallback to overall EV if no per-pitch data
        overall_ev = hitter_tm.get("overall", {}).get("avg_ev", np.nan)
        if pd.isna(our_ev) and not pd.isna(overall_ev):
            our_ev = overall_ev
        our_whiff = our_data.get("whiff_pct", np.nan) if our_data else np.nan
        our_barrel = our_data.get("barrel_pct", np.nan) if our_data else np.nan
        our_depth = our_data.get("contact_depth", np.nan) if our_data else np.nan
        their_whiff = pitcher_profile.get("swstrk_pct", np.nan)
        their_chase = pitcher_profile.get("chase_pct", np.nan)

        # Edge classification — barrel-aware, sample-size-sensitive
        has_pitch_data = bool(our_data)  # True if we have pitch-specific data
        ip_n = our_data.get("seen", 0) if our_data else 0
        # Use InPlay-based EV only if we have real per-pitch data (not overall fallback)
        ev_is_real = has_pitch_data and not pd.isna(our_ev_raw)
        ev_ok = ev_is_real and our_ev >= 88
        ev_great = ev_is_real and our_ev >= 90
        whiff_ok = not pd.isna(our_whiff) and our_whiff <= 25
        whiff_bad = not pd.isna(our_whiff) and our_whiff >= 35
        ev_bad = ev_is_real and our_ev < 80
        ev_weak = ev_is_real and our_ev < 85  # not terrible but not good either
        barrel_good = not pd.isna(our_barrel) and our_barrel >= 8
        their_whiff_high = not pd.isna(their_whiff) and their_whiff >= 25

        if (ev_ok and whiff_ok) or (ev_great and barrel_good):
            edge = "Advantage"
        elif (whiff_bad and ev_bad) or (ev_bad and their_whiff_high) or (not ev_is_real and whiff_bad) or (whiff_bad and ev_weak):
            edge = "Vulnerable"
        else:
            edge = "Neutral"

        # Build reason string
        if edge == "Advantage":
            parts = []
            if barrel_good:
                parts.append(f"We barrel it ({our_barrel:.0f}%)")
            if ev_is_real:
                parts.append(f"hit it hard ({our_ev:.0f} EV)")
            if not pd.isna(our_whiff):
                parts.append(f"{our_whiff:.0f}% whiff")
            reason = " and ".join(parts[:2]) if parts else "Strong matchup"
        elif edge == "Vulnerable":
            parts = []
            if whiff_bad:
                parts.append(f"High whiff ({our_whiff:.0f}%)")
            tn = _timing_note(our_depth)
            if tn:
                parts.append(f"we're {tn.lower()} on it")
            if ev_bad:
                parts.append(f"weak contact ({our_ev:.0f} EV)")
            reason = " \u2014 ".join(parts) if parts else "Vulnerable"
        else:
            ev_str = f"{our_ev:.0f} EV" if not pd.isna(our_ev) else "no data"
            wh_str = f"{our_whiff:.0f}% whiff" if not pd.isna(our_whiff) else ""
            reason_parts = [ev_str]
            if wh_str:
                reason_parts.append(wh_str)
            reason = f"Neutral \u2014 " + ", ".join(reason_parts)

        edge_val = {"Advantage": 70, "Neutral": 50, "Vulnerable": 30}[edge]
        weight = opp_usage / 100.0

        pitch_edge = {
            "pitch": opp_pitch,
            "usage": opp_usage,
            "edge": edge,
            "our_ev": our_ev,
            "our_ev_raw": our_ev_raw,
            "our_whiff": our_whiff,
            "our_barrel": our_barrel if not pd.isna(our_barrel) else np.nan,
            "their_whiff": their_whiff,
            "their_chase": their_chase,
            "timing_note": _timing_note(our_depth),
            "zone_cross": _zone_cross_note(opp_pitch),
            "reason": reason,
        }
        pitch_edges.append(pitch_edge)
        weighted_score += edge_val * weight
        total_weight += weight

    # Overall score with platoon adjustment
    raw_score = weighted_score / total_weight if total_weight > 0 else 50
    overall = max(0, min(100, raw_score * platoon_factor))

    # Build approach notes from edge data
    adv_pitches = [pe for pe in pitch_edges if pe["edge"] == "Advantage"]
    vuln_pitches = [pe for pe in pitch_edges if pe["edge"] == "Vulnerable"]
    if adv_pitches:
        # Weight by EV × usage to pick the most impactful Advantage pitch
        best = max(adv_pitches, key=lambda pe: (pe["our_ev"] if not pd.isna(pe["our_ev"]) else 80) * (pe["usage"] / 100))
        approach_notes.append(f"Sit on **{best['pitch']}** ({best['usage']:.0f}% of mix)")
    if vuln_pitches:
        worst = max(vuln_pitches, key=lambda pe: pe["our_whiff"] if not pd.isna(pe["our_whiff"]) else 0)
        approach_notes.append(f"Lay off **{worst['pitch']}** out of zone")
    unknown = [pe for pe in pitch_edges if pd.isna(pe["our_ev"]) and pe["usage"] > 10]
    if unknown:
        approach_notes.append(f"No Trackman data vs {', '.join(pe['pitch'] for pe in unknown)} \u2014 be ready")
    loc_tend = pitcher_profile.get("location_tendency", "")
    if loc_tend and loc_tend not in ("balanced", "unknown"):
        approach_notes.append(f"Lives {loc_tend}")

    # Build pitch_details dict for backward compatibility
    pitch_details = {}
    for pe in pitch_edges:
        pitch_details[pe["pitch"]] = {
            "score": {"Advantage": 70, "Neutral": 50, "Vulnerable": 30}[pe["edge"]],
            "usage": pe["usage"],
            "edge": pe["edge"],
            "our_ev": pe["our_ev"],
            "our_whiff": pe["our_whiff"],
            "our_barrel": pe["our_barrel"],
            "timing_note": pe["timing_note"],
            "zone_cross": pe["zone_cross"],
            "reason": pe["reason"],
        }

    return {
        "hitter": hitter_tm["name"], "pitcher": pitcher_profile["name"],
        "bats": bats, "platoon": platoon_label,
        "overall_score": round(overall, 1),
        "pitch_edges": pitch_edges,
        "pitch_details": pitch_details, "approach_notes": approach_notes,
    }


def _generate_at_bat_script(hitter_profile, pitcher_profile, matchup_result):
    """Generate at-bat script with tag line, 1P/ahead/2-strike plans, and pitch edges."""
    if not matchup_result or not pitcher_profile.get("pitch_mix"):
        return None
    _hard_set = {"Fastball", "Sinker", "Cutter"}
    mix = pitcher_profile["pitch_mix"]
    real_mix = {p: u for p, u in mix.items() if u >= 5}
    if not real_mix:
        return None
    zones = hitter_profile.get("zones", {})
    by_class = hitter_profile.get("by_pitch_class", {})
    pitch_edges = matchup_result.get("pitch_edges", [])
    pitch_details = matchup_result.get("pitch_details", {})
    sp = hitter_profile.get("swing_path") or {}
    disc = hitter_profile.get("discipline") or {}
    fp_data = hitter_profile.get("first_pitch") or {}
    two_k_data = hitter_profile.get("two_strike") or {}
    zone_grid = hitter_profile.get("zone_grid") or {}
    by_pt = hitter_profile.get("by_pitch_type", {})
    # 3A: New V3 fields
    barrel_zones = hitter_profile.get("barrel_zones", [])
    spray = hitter_profile.get("spray", {})
    two_k_whiff_hard = two_k_data.get("whiff_hard", np.nan) if two_k_data else np.nan
    two_k_whiff_os = two_k_data.get("whiff_os", np.nan) if two_k_data else np.nan
    count_ev = hitter_profile.get("count_ev_grid", {})

    sorted_mix = sorted(real_mix.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_mix[0] if sorted_mix else (None, 0)
    hard_pitches = [(p, u) for p, u in sorted_mix if p in _hard_set]
    os_pitches = [(p, u) for p, u in sorted_mix if p not in _hard_set]

    # ── Helpers ──
    def _best_damage_zone(zg, pp):
        """Find our best zone_grid cell, with pitcher tendency as small tiebreaker.
        EV is the primary sort — pitcher location only nudges between similar-EV zones."""
        h_pct = pp.get("loc_high_pct", np.nan)
        l_pct = pp.get("loc_low_pct", np.nan)
        candidates = []
        for key, cell in zg.items():
            if cell.get("n", 0) < 5:
                continue
            ev = cell.get("avg_ev", np.nan)
            if pd.isna(ev):
                continue
            parts = key.split("_")
            row = parts[0] if parts else ""
            # Pitcher tendency is a SMALL tiebreaker (max +3 EV boost), not dominant
            bonus = 0.0
            if "High" in row and not pd.isna(h_pct) and h_pct >= 25:
                bonus = min(h_pct / 30, 3.0)  # max 3 EV bonus
            if "Low" in row and not pd.isna(l_pct) and l_pct >= 25:
                bonus = min(l_pct / 30, 3.0)
            candidates.append((key, ev, ev + bonus))
        if not candidates:
            return None, np.nan
        best = max(candidates, key=lambda c: c[2])
        return best[0], best[1]  # return real EV, not boosted

    def _chase_vulnerability(z):
        """Check all 4 chase zones. Return list of (zone_label, whiff%) where whiff > 35%."""
        vulns = []
        for cz in ["chase_up", "chase_down", "chase_in", "chase_away"]:
            zd = z.get(cz, {})
            wh = zd.get("whiff_pct", np.nan)
            if zd.get("n", 0) >= 5 and not pd.isna(wh) and wh > 35:
                vulns.append((cz.replace("chase_", ""), wh))
        vulns.sort(key=lambda x: x[1], reverse=True)
        return vulns

    def _timing_note_str(depth):
        if pd.isna(depth):
            return ""
        if depth > 0.5:
            return f"early +{depth:.1f}"
        if depth < -1.5:
            return f"deep {depth:.1f}"
        return ""

    # ── Section 0: TAG LINE ──
    swing_type = sp.get("swing_type", "Unknown")
    bs_avg = sp.get("bat_speed_avg")
    chase = disc.get("chase_pct", np.nan)
    z_contact = disc.get("z_contact_pct", np.nan)
    if not pd.isna(chase) and not pd.isna(z_contact):
        disc_score = (100 - chase) * 0.5 + z_contact * 0.5
        disc_grade = "A" if disc_score >= 80 else "B" if disc_score >= 65 else "C" if disc_score >= 50 else "D"
    else:
        disc_grade = "-"
    overall_ev = hitter_profile.get("overall", {}).get("avg_ev", np.nan)
    overall_barrel = hitter_profile.get("overall", {}).get("barrel_pct", np.nan)
    overall_hh = hitter_profile.get("overall", {}).get("hard_hit_pct", np.nan)
    overall_ss = hitter_profile.get("overall", {}).get("sweet_spot_pct", np.nan)
    attack_angle = sp.get("attack_angle")
    tag_parts = [f"{hitter_profile['bats'][0]}HH"]
    if not pd.isna(overall_ev):
        tag_parts.append(f"{overall_ev:.0f} EV")
    if not pd.isna(overall_barrel) and overall_barrel >= 3:
        tag_parts.append(f"{overall_barrel:.0f}% barrel")
    if swing_type != "Unknown":
        tag_parts.append(swing_type)
    if bs_avg is not None:
        tag_parts.append(f"{bs_avg:.0f} mph bat")
    tag_parts.append(f"{disc_grade} discipline")
    if not pd.isna(chase):
        tag_parts.append(f"{chase:.0f}% chase")
    tag_line = " \u2022 ".join(tag_parts)

    # ── Section 1: SCOUTING THIS PITCHER ──
    scout_lines = []
    for p, u in sorted_mix[:3]:
        loc_hint = ""
        if p in _hard_set:
            h_pct = pitcher_profile.get("loc_high_pct", np.nan)
            if not pd.isna(h_pct) and h_pct >= 28:
                loc_hint = "up"
            else:
                l_pct = pitcher_profile.get("loc_low_pct", np.nan)
                loc_hint = "low" if (not pd.isna(l_pct) and l_pct >= 30) else ""
        else:
            l_pct = pitcher_profile.get("loc_low_pct", np.nan)
            loc_hint = "low" if (not pd.isna(l_pct) and l_pct >= 28) else "arm-side"
        velo = pitcher_profile.get("velo", np.nan)
        velo_str = f" {velo:.0f}mph" if not pd.isna(velo) and p == sorted_mix[0][0] else ""
        scout_lines.append(f"{p}{velo_str} ({u:.0f}%) {loc_hint}".strip())
    comploc = pitcher_profile.get("comploc_pct", np.nan)
    if not pd.isna(comploc) and comploc < 40:
        scout_lines.append(f"Low command ({comploc:.0f}%)")

    # ── Section 2: FIRST PITCH ──
    fp_pitch = primary[0]
    fp_usage = primary[1]
    callstrk = pitcher_profile.get("callstrk_pct", np.nan)
    our_fp_swing = fp_data.get("swing_pct", np.nan) if fp_data else np.nan
    our_fp_ev = fp_data.get("avg_ev", np.nan) if fp_data else np.nan
    our_1p_whiff = by_pt.get(fp_pitch, {}).get("whiff_pct", np.nan)
    our_1p_ev = by_pt.get(fp_pitch, {}).get("avg_ev", np.nan)

    # Pitcher's 1P location tendency
    h_pct = pitcher_profile.get("loc_high_pct", np.nan)
    l_pct = pitcher_profile.get("loc_low_pct", np.nan)
    fp_loc_hint = ""
    if not pd.isna(h_pct) and h_pct >= 28:
        fp_loc_hint = " up"
    elif not pd.isna(l_pct) and l_pct >= 30:
        fp_loc_hint = " low"

    fp_expect = f"{fp_pitch}{fp_loc_hint} ({fp_usage:.0f}%)"

    # Decision tree
    # Check if primary pitch is our Advantage pitch (barrel it / crush it) — override whiff gate
    fp_edge = pitch_details.get(fp_pitch, {}).get("edge", "")
    our_1p_barrel = by_pt.get(fp_pitch, {}).get("barrel_pct", np.nan)
    fp_is_advantage = fp_edge == "Advantage"
    if fp_is_advantage and not pd.isna(our_1p_ev) and our_1p_ev >= 87:
        fp_plan = "GREEN LIGHT"
        ev_str = f"{our_1p_ev:.0f} EV"
        barrel_str = f", {our_1p_barrel:.0f}% barrel" if not pd.isna(our_1p_barrel) and our_1p_barrel >= 5 else ""
        wh_str = f", {our_1p_whiff:.0f}% whiff" if not pd.isna(our_1p_whiff) else ""
        fp_detail = f"Advantage pitch \u2014 {ev_str}{barrel_str}{wh_str}"
    elif not pd.isna(our_1p_whiff) and our_1p_whiff > 30:
        fp_plan = "BE SELECTIVE"
        fp_detail = f"High whiff on {fp_pitch} ({our_1p_whiff:.0f}%) \u2014 take unless perfect"
    elif not pd.isna(our_fp_ev) and our_fp_ev > 87 and not pd.isna(our_fp_swing) and our_fp_swing >= 30:
        fp_plan = "GREEN LIGHT"
        ev_str = f"{our_1p_ev:.0f} EV" if not pd.isna(our_1p_ev) else f"{our_fp_ev:.0f} EV"
        wh_str = f", {our_1p_whiff:.0f}% whiff" if not pd.isna(our_1p_whiff) else ""
        fp_detail = f"{ev_str}{wh_str}"
    elif not pd.isna(callstrk) and callstrk > 18:
        fp_plan = "SWING IF IN ZONE"
        fp_detail = f"High called-strike rate ({callstrk:.0f}%)"
    else:
        fp_plan = "SEE IT FIRST"
        fp_detail = "Gather info, swing only if middle-middle"

    # Look zone: cross best damage zone with pitcher 1P tendency
    dmg_key, dmg_ev = _best_damage_zone(zone_grid, pitcher_profile)
    fp_look = ""
    if dmg_key and not pd.isna(dmg_ev) and dmg_ev > 85:
        fp_look = f"{dmg_key.replace('_', '-').lower()} ({dmg_ev:.0f} EV damage zone)"

    first_pitch = {
        "expect": fp_expect,
        "plan": fp_plan,
        "detail": fp_detail,
        "look": fp_look,
    }

    # ── Section 3: AHEAD IN COUNT ──
    adv_edges = [pe for pe in pitch_edges if pe["edge"] == "Advantage"]
    if adv_edges:
        best_adv = max(adv_edges, key=lambda pe: pe["our_ev"] if not pd.isna(pe["our_ev"]) else 0)
        sit_pitch = best_adv["pitch"]
        sit_ev = best_adv["our_ev"]
        sit_str = f"Sit on: {sit_pitch} \u2014 we crush it ({sit_ev:.0f} EV)" if not pd.isna(sit_ev) else f"Sit on: {sit_pitch}"
    else:
        sit_str = "Be aggressive in zone \u2014 no clear Advantage pitch"

    # Hunt zone: our best damage zone where pitcher also throws
    hunt_key, hunt_ev = _best_damage_zone(zone_grid, pitcher_profile)
    hunt_str = ""
    if hunt_key and not pd.isna(hunt_ev) and hunt_ev > 85:
        hunt_str = f"{hunt_key.replace('_', '-').lower()} ({hunt_ev:.0f} EV)"

    # 3B: Barrel hunt zone — always show top barrel zone, add pitcher overlap context
    barrel_hunt_str = ""
    if barrel_zones:
        h_pct = pitcher_profile.get("loc_high_pct", np.nan)
        l_pct = pitcher_profile.get("loc_low_pct", np.nan)
        in_pct = pitcher_profile.get("loc_inside_pct", np.nan)
        out_pct = pitcher_profile.get("loc_outside_pct", np.nan)
        # Always use top barrel zone (most barrels)
        bz_key, bz_barrels, bz_ev = barrel_zones[0]
        bz_lower = bz_key.lower()
        zone_label = bz_key.replace("_", " ")
        # Check if pitcher throws to this zone
        overlap = False
        if "high" in bz_lower and not pd.isna(h_pct) and h_pct >= 25:
            overlap = True
        if "low" in bz_lower and not pd.isna(l_pct) and l_pct >= 25:
            overlap = True
        if ("inside" in bz_lower or "far_in" in bz_lower) and not pd.isna(in_pct) and in_pct >= 25:
            overlap = True
        if ("outside" in bz_lower or "far_out" in bz_lower) and not pd.isna(out_pct) and out_pct >= 25:
            overlap = True
        if overlap:
            barrel_hunt_str = (f"Barrels come {zone_label.lower()} ({bz_barrels} barrels, {bz_ev:.0f} EV) "
                               f"\u2014 pitcher throws there. HUNT {zone_label.upper()}.")
        else:
            barrel_hunt_str = (f"Barrels come {zone_label.lower()} ({bz_barrels} barrels, {bz_ev:.0f} EV). "
                               f"HUNT {zone_label.upper()} \u2014 force your pitch.")

    # 3C: Count-aware EV callouts — AHEAD
    count_ahead_note = ""
    for ck in ["2-0", "3-1", "3-0"]:
        cd = count_ev.get(ck, {})
        cev = cd.get("avg_ev", np.nan)
        cn = cd.get("n", 0)
        if not pd.isna(cev) and cev >= 90 and cn >= 5:
            count_ahead_note = f"You mash {ck} ({cev:.0f} EV) \u2014 sit dead red"
            break

    when_ahead = {
        "sit_on": sit_str,
        "hunt_zone": hunt_str,
        "barrel_hunt": barrel_hunt_str,
        "count_note": count_ahead_note,
    }

    # ── Section 3B: WHEN BEHIND ──
    behind_note = ""
    # When behind (0-1, 0-2, 1-2), pitcher controls — what should hitter do?
    behind_pitches = []
    for p, u in sorted_mix:
        if p not in _hard_set and u >= 10:
            behind_pitches.append((p, u))
    # Add location context from pitcher profile
    low_pct_b = pitcher_profile.get("loc_low_pct", np.nan)
    loc_hint_b = ""
    if not pd.isna(low_pct_b) and low_pct_b >= 30:
        loc_hint_b = " low"
    if behind_pitches:
        bp_name, bp_usage = behind_pitches[0]
        # Check if we have an edge on this pitch
        bp_edge = pitch_details.get(bp_name, {}).get("edge", "")
        if bp_edge == "Advantage":
            behind_note = f"Expect **{bp_name}{loc_hint_b}** ({bp_usage:.0f}%) \u2014 you handle it, look to drive"
        elif bp_edge == "Vulnerable":
            behind_note = f"Expect **{bp_name}{loc_hint_b}** ({bp_usage:.0f}%) \u2014 you struggle here, protect zone"
        else:
            behind_note = f"Expect **{bp_name}{loc_hint_b}** ({bp_usage:.0f}%) \u2014 protect the zone, don't expand"
    elif hard_pitches:
        behind_note = f"Likely stays with **{hard_pitches[0][0]}** \u2014 shorten up, put it in play"

    # Count EV behind check
    count_behind_note = ""
    for ck_b in ["0-1", "1-2", "0-2"]:
        cd_b = count_ev.get(ck_b, {})
        cev_b = cd_b.get("avg_ev", np.nan)
        cn_b = cd_b.get("n", 0)
        overall_ev_b = hitter_profile.get("overall", {}).get("avg_ev", np.nan)
        if not pd.isna(cev_b) and not pd.isna(overall_ev_b) and cn_b >= 5:
            if cev_b >= 88:
                count_behind_note = f"You still hit well behind ({ck_b}: {cev_b:.0f} EV) \u2014 don't give away ABs"
                break
            elif (overall_ev_b - cev_b) > 8:
                count_behind_note = f"EV drops to {cev_b:.0f} when behind ({ck_b}) \u2014 be selectively aggressive earlier"
                break

    when_behind = {
        "note": behind_note,
        "count_note": count_behind_note,
    }

    # ── Section 4: TWO STRIKES ──
    putaway_cands = pitcher_profile.get("putaway_candidates", {})
    if putaway_cands:
        putaway_pitch = max(putaway_cands, key=putaway_cands.get)
        putaway_usage = putaway_cands[putaway_pitch]
    elif os_pitches:
        putaway_pitch, putaway_usage = os_pitches[0]
    elif hard_pitches:
        putaway_pitch, putaway_usage = hard_pitches[0]
    elif sorted_mix:
        putaway_pitch, putaway_usage = sorted_mix[-1] if len(sorted_mix) > 1 else sorted_mix[0]
    else:
        putaway_pitch, putaway_usage = "?", 0

    # Putaway location
    putaway_is_hard = putaway_pitch in _hard_set
    low_pct = pitcher_profile.get("loc_low_pct", np.nan)
    if not pd.isna(low_pct) and low_pct > 35:
        ts_loc = "low"
    elif putaway_is_hard:
        ts_loc = "up in zone"
    else:
        ts_loc = "below zone"

    ts_expect = f"{putaway_pitch} {ts_loc} ({putaway_usage:.0f}%)"

    # Chase vulnerability: check ALL 4 zones
    chase_vulns = _chase_vulnerability(zones)
    chase_warning = ""
    if chase_vulns:
        worst_cz = chase_vulns[0]
        cz_label = {"in": "inside", "away": "outside", "up": "above", "down": "below"}.get(worst_cz[0], worst_cz[0])
        chase_warning = f"\u26a0 DON'T CHASE \u2014 {worst_cz[1]:.0f}% whiff {cz_label} zone"

    # Protect plan — check BOTH classes' whiff to find true vulnerability
    hard_whiff_2k = by_class.get("hard", {}).get("whiff_pct", np.nan)
    os_whiff_2k = by_class.get("offspeed", {}).get("whiff_pct", np.nan)
    hard_adv = any(pe["edge"] == "Advantage" and pe["pitch"] in _hard_set for pe in pitch_edges)
    # Determine which class is the REAL vulnerability (regardless of putaway pitch)
    hard_vuln = not pd.isna(hard_whiff_2k) and hard_whiff_2k > 30
    os_vuln = not pd.isna(os_whiff_2k) and os_whiff_2k > 30
    if not pd.isna(chase) and chase < 22 and not pd.isna(z_contact) and z_contact > 80:
        protect = "Battle in zone, you're tough to K"
    elif not pd.isna(chase) and chase > 28:
        protect = "Shorten up, don't expand"
    elif hard_vuln and not os_vuln:
        # Vulnerable to hard stuff — gear for fastball
        if hard_adv:
            protect = f"Hard whiff is high ({hard_whiff_2k:.0f}%) but you crush it \u2014 time it up, don't be late"
        else:
            protect = f"Cheat fastball ({hard_whiff_2k:.0f}% 2K whiff), react to offspeed"
    elif os_vuln and not hard_vuln:
        # Vulnerable to offspeed — sit offspeed or cheat hard
        if hard_adv:
            protect = f"Cheat hard (Advantage), spit offspeed out of zone ({os_whiff_2k:.0f}% 2K whiff)"
        else:
            protect = f"Look for offspeed ({os_whiff_2k:.0f}% 2K whiff), fight off hard"
    elif hard_vuln and os_vuln:
        protect = "High whiff both classes \u2014 shorten up, protect zone, foul off tough pitches"
    else:
        protect = f"Shorten up, fight off {putaway_pitch}"

    # 3C: Count-aware EV callout — 2 STRIKES
    count_2k_note = ""
    overall_ev = hitter_profile.get("overall", {}).get("avg_ev", np.nan)
    for ck2 in ["0-2", "1-2"]:
        cd2 = count_ev.get(ck2, {})
        cev2 = cd2.get("avg_ev", np.nan)
        if not pd.isna(cev2) and not pd.isna(overall_ev):
            if cev2 < 80 or (overall_ev - cev2) > 8:
                count_2k_note = f"EV drops to {cev2:.0f} with 2 strikes \u2014 get your A-swing early"
                break

    # 3D: 2K pitch-class whiff note
    two_k_class_note = ""
    if not pd.isna(two_k_whiff_hard) and not pd.isna(two_k_whiff_os):
        diff = two_k_whiff_os - two_k_whiff_hard
        if abs(diff) >= 12:
            if diff > 0:
                two_k_class_note = (f"Your OS whiff with 2 strikes is {two_k_whiff_os:.0f}% "
                                    f"but hard whiff is only {two_k_whiff_hard:.0f}% "
                                    f"\u2014 look offspeed, react hard")
            else:
                two_k_class_note = (f"Your hard whiff with 2 strikes is {two_k_whiff_hard:.0f}% "
                                    f"but OS whiff is only {two_k_whiff_os:.0f}% "
                                    f"\u2014 cheat fastball, react offspeed")

    # Putaway danger ranking — rank pitcher's pitches by how dangerous they are to us with 2K
    putaway_danger = []
    for p, u in sorted_mix:
        if u < 5:
            continue
        is_hard_p = p in _hard_set
        our_2k_wh = two_k_whiff_hard if is_hard_p else two_k_whiff_os
        their_whiff_p = pitcher_profile.get("pitch_whiff", {}).get(p, np.nan)
        their_chase_p = pitcher_profile.get("pitch_chase", {}).get(p, np.nan)
        # Score: how dangerous is this pitch to us on 2 strikes?
        # Weight: our 2K vulnerability (50%) + their pitch whiff (25%) + their chase gen (25%)
        t2k = our_2k_wh / 100 if not pd.isna(our_2k_wh) else 0.25
        tw = their_whiff_p / 100 if not pd.isna(their_whiff_p) else 0.15
        tc = their_chase_p / 100 if not pd.isna(their_chase_p) else 0.10
        danger = t2k * 0.50 + tw * 0.25 + tc * 0.25
        putaway_danger.append((p, round(danger * 100, 1), our_2k_wh, is_hard_p, their_whiff_p, u))
    putaway_danger.sort(key=lambda x: x[1], reverse=True)

    two_strike = {
        "expect": ts_expect,
        "chase_warning": chase_warning,
        "protect": protect,
        "count_note": count_2k_note,
        "class_whiff_note": two_k_class_note,
        "putaway_danger": putaway_danger[:3],  # top 3 most dangerous pitches
    }

    # ── 3E: Pull tendency warning ──
    pull_warning = ""
    pull_pct = spray.get("pull_pct", 0)
    loc_outside = pitcher_profile.get("loc_outside_pct", np.nan)
    if pull_pct > 55 and not pd.isna(loc_outside) and loc_outside >= 30:
        pull_warning = f"You pull {pull_pct:.0f}% \u2014 pitcher lives outside ({loc_outside:.0f}%), stay through it."

    # ── Section 5: PITCH EDGES ──
    edge_lines = []
    sorted_edges = sorted(pitch_edges,
                          key=lambda pe: {"Advantage": 0, "Neutral": 1, "Vulnerable": 2}[pe["edge"]])
    for pe in sorted_edges:
        if pe["usage"] < 5:
            continue
        sym = "\u2713" if pe["edge"] == "Advantage" else ("\u2717" if pe["edge"] == "Vulnerable" else "\u2014")
        ev_str = f"{pe['our_ev']:.0f} EV" if not pd.isna(pe["our_ev"]) else "no data"
        wh_str = f"{pe['our_whiff']:.0f}% whiff" if not pd.isna(pe["our_whiff"]) else ""
        reason_short = pe.get("reason", "")
        # Add timing note for vulnerable pitches with depth info
        tn = pe.get("timing_note", "")
        if tn and pe["edge"] == "Vulnerable":
            reason_short += f" ({tn})"
        parts = [ev_str]
        if wh_str:
            parts.append(wh_str)
        stats_str = ", ".join(parts)
        edge_lines.append({
            "symbol": sym,
            "pitch": pe["pitch"],
            "edge": pe["edge"],
            "stats": stats_str,
            "reason": reason_short,
        })

    # ── TAG LINE data ──
    tag_line_data = {
        "name": hitter_profile["name"],
        "hand": hitter_profile["bats"],
        "swing_type": swing_type,
        "bat_speed": f"{bs_avg:.0f}" if bs_avg is not None else "-",
        "discipline_grade": disc_grade,
        "chase_pct": f"{chase:.0f}" if not pd.isna(chase) else "-",
        "tag_line": tag_line,
    }

    return {
        "tag_line": tag_line_data,
        "scout_lines": scout_lines,
        "first_pitch": first_pitch,
        "when_ahead": when_ahead,
        "when_behind": when_behind,
        "two_strike": two_strike,
        "pitch_edges": edge_lines,
        "pull_warning": pull_warning,
    }


# ── Game Plan UI ──

def _game_plan_tab(tm, team, data, pitch_df=None):
    """Game Plan tab — unified pitching + hitting plans."""
    section_header(f"Game Plan vs {team}")
    st.caption("Cross-referencing TrueMedia season data with our Trackman pitch-level data")
    seasons = get_all_seasons()
    season_filter = st.multiselect("Our Trackman Seasons", seasons, default=seasons, key="gpl_season")
    tab_pitch, tab_hit = st.tabs(["Our Pitching Plan", "Our Hitting Plan"])
    with tab_pitch:
        try:
            _pitching_plan_content(tm, team, data, season_filter, pitch_df=pitch_df)
        except Exception as e:
            st.error(f"Pitching plan error: {e}")
    with tab_hit:
        try:
            _hitting_plan_content(tm, team, data, season_filter, pitch_df=pitch_df)
        except Exception as e:
            st.error(f"Hitting plan error: {e}")


_ZONE_LABELS = {"up": "up in zone", "down": "down in zone", "glove": "glove side",
                 "arm": "arm side", "chase_low": "below zone"}

def _best_putaway_zone(pitch_name, ars_pt):
    """Find the best putaway location for a pitch using zone_eff + PZM.
    Returns (zone_label, whiff_pct) or (None, None) if no data."""
    ze = ars_pt.get("zone_eff", {})
    if not ze:
        return None, None
    ivb = ars_pt.get("ivb", np.nan)
    is_hard = pitch_name in _hard_pitches
    # Pitch-design zone multipliers (same as composite scorer)
    if is_hard:
        if pitch_name == "Sinker":
            pzm = {"up": 0.5, "down": 1.4, "chase_low": 1.3, "glove": 1.0, "arm": 1.0}
        elif not pd.isna(ivb) and ivb >= 16:
            pzm = {"up": 1.4, "down": 0.7, "chase_low": 0.4, "glove": 1.0, "arm": 1.0}
        elif not pd.isna(ivb) and ivb < 12:
            pzm = {"up": 0.7, "down": 1.3, "chase_low": 1.1, "glove": 1.0, "arm": 1.0}
        else:
            pzm = {"up": 1.15, "down": 0.9, "chase_low": 0.6, "glove": 1.0, "arm": 1.0}
    else:
        if pitch_name in ("Slider", "Sweeper"):
            pzm = {"up": 0.3, "down": 1.2, "chase_low": 1.4, "glove": 1.3, "arm": 0.8}
        elif pitch_name in ("Curveball", "Knuckle Curve"):
            pzm = {"up": 0.2, "down": 1.3, "chase_low": 1.5, "glove": 1.0, "arm": 1.0}
        elif pitch_name in ("Changeup", "Splitter"):
            pzm = {"up": 0.3, "down": 1.3, "chase_low": 1.3, "glove": 1.1, "arm": 0.9}
        else:
            pzm = {"up": 0.5, "down": 1.2, "chase_low": 1.3, "glove": 1.0, "arm": 1.0}
    # For putaway, heavily weight whiff_pct (we want swings and misses)
    best_zone, best_score = None, -1
    for zn, zdata in ze.items():
        w = zdata.get("whiff_pct", np.nan)
        n = zdata.get("n", 0)
        if pd.isna(w) or n < 5:
            continue
        # PZM-weighted whiff score
        score = w * pzm.get(zn, 1.0)
        if score > best_score:
            best_score = score
            best_zone = zn
    if best_zone is None:
        return None, None
    return _ZONE_LABELS.get(best_zone, best_zone), ze[best_zone].get("whiff_pct", np.nan)


def _pitching_plan_content(tm, team, data, season_filter, pitch_df=None):
    """Our Pitching Plan: how each of our pitchers should attack their lineup."""
    h_rate = _tm_team(tm["hitting"]["rate"], team)
    if h_rate.empty:
        st.info("No opponent hitting data found.")
        return
    dav_pitching = filter_davidson(data, role="pitcher")
    if season_filter:
        dav_pitching = dav_pitching[dav_pitching["Season"].isin(season_filter)]
    our_pitchers = sorted(dav_pitching["Pitcher"].unique())
    if not our_pitchers:
        st.warning("No Davidson pitcher data available.")
        return
    selected_pitcher = st.selectbox("Select Our Pitcher", our_pitchers,
                                     format_func=display_name, key="gpl_our_pitcher")
    tunnel_pop = build_tunnel_population_pop()
    arsenal = _get_our_pitcher_arsenal(data, selected_pitcher, season_filter, tunnel_pop=tunnel_pop)
    if arsenal is None:
        st.warning("Not enough Trackman data for this pitcher (min 100 pitches).")
        return
    throws_str = "RHP" if arsenal["throws"] == "Right" else "LHP"
    pitch_items = sorted(arsenal["pitches"].items(), key=lambda x: x[1]["usage_pct"], reverse=True)
    if not pitch_items:
        st.info("No pitch type breakdown available.")
        return
    # Prepare tunnel and sequence data for bullpen cards (no display here — see Pitching Lab)
    tunnels = arsenal.get("tunnels", pd.DataFrame())
    sequences = arsenal.get("sequences", pd.DataFrame())
    if isinstance(sequences, pd.DataFrame) and not sequences.empty:
        valid_pitches = {name for name, pt in arsenal["pitches"].items() if pt.get("count", 0) >= 10}
        sequences = sequences[
            sequences["Setup Pitch"].isin(valid_pitches) & sequences["Follow Pitch"].isin(valid_pitches)
        ]
        if "Count" in sequences.columns:
            sequences = sequences[sequences["Count"] >= 25]
    # ── Hitter-by-Hitter Plan ──
    opp_hitters = h_rate.sort_values("PA", ascending=False).head(12)
    pitch_summary = " / ".join(f"{n} {pt['usage_pct']:.0f}%" for n, pt in pitch_items[:4])
    section_header(f"{display_name(selected_pitcher)} ({throws_str}) vs Lineup")
    st.caption(f"Arsenal: {pitch_summary} — {arsenal['total_pitches']} pitches")

    # ── Best Tunnels & Sequences (Our Pitcher Trackman) ──
    pitch_metrics = {
        name: {"stuff": pt.get("stuff_plus", np.nan), "cmd": pt.get("command_plus", np.nan)}
        for name, pt in arsenal["pitches"].items()
    }
    top_pairs = _top_tunnel_pairs(tunnels, sequences, pitch_metrics, top_n=2)
    top_seq3 = _filter_redundant_sequences(
        _top_sequences(sequences, pitch_metrics, length=3, top_n=5),
        min_unique=3, max_keep=2,
    )
    top_seq4 = _filter_redundant_sequences(
        _top_sequences(sequences, pitch_metrics, length=4, top_n=5),
        min_unique=3, max_keep=2,
    )
    if top_pairs or top_seq3 or top_seq4:
        section_header("Best Pairs & Sequences (Trackman)")
        st.caption("Outcomes-first (Whiff, K/Putaway, EV) with Tunnel as a secondary signal. Details in checkbox.")
        if top_pairs:
            top_pairs = _assign_tactical_tags(top_pairs)
            rows = []
            detail = []
            for r in top_pairs:
                rows.append({
                    "Pair": r["Pair"],
                    "Score": f"{r['Score']:.1f}" if pd.notna(r["Score"]) else "-",
                    "Whiff%": f"{r['Whiff%']:.1f}" if pd.notna(r["Whiff%"]) else "-",
                    "K/Putaway%": f"{r['K%']:.1f}" if pd.notna(r["K%"]) else "-",
                    "Avg EV": f"{r['Avg EV']:.1f}" if pd.notna(r["Avg EV"]) else "-",
                    "Tag": r.get("Tag", ""),
                    "Deception": _deception_flag(r.get("Tunnel", np.nan)),
                })
                detail.append({
                    "Pair": r["Pair"],
                    "Tunnel": f"{r['Tunnel']:.1f}" if pd.notna(r["Tunnel"]) else "-",
                    "Stuff+": f"{r['Stuff+']:.0f}" if pd.notna(r["Stuff+"]) else "-",
                    "Cmd+": f"{r['Cmd+']:.0f}" if pd.notna(r["Cmd+"]) else "-",
                    "Pairs": int(r["Pairs"]) if pd.notna(r["Pairs"]) else "-",
                })
            st.markdown("**Top Pitch Pairs**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if st.checkbox(
                "Show details (Tunnel / Stuff+ / Command+ / Sample size)",
                key=f"gpl_pairs_detail_{selected_pitcher}",
            ):
                st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)
        if top_seq3:
            top_seq3 = _assign_tactical_tags(top_seq3)
            rows = []
            detail = []
            for r in top_seq3:
                rows.append({
                    "Sequence": r["Seq"],
                    "Score": f"{r['Score']:.1f}" if pd.notna(r["Score"]) else "-",
                    "Whiff%": f"{r['Whiff%']:.1f}" if pd.notna(r["Whiff%"]) else "-",
                    "K/Putaway%": f"{r['K%']:.1f}" if pd.notna(r["K%"]) else "-",
                    "Avg EV": f"{r['Avg EV']:.1f}" if pd.notna(r["Avg EV"]) else "-",
                    "Tag": r.get("Tag", ""),
                    "Deception": _deception_flag(r.get("Tunnel", np.nan)),
                })
                detail.append({
                    "Sequence": r["Seq"],
                    "Tunnel": f"{r['Tunnel']:.1f}" if pd.notna(r["Tunnel"]) else "-",
                    "Stuff+": f"{r['Stuff+']:.0f}" if pd.notna(r["Stuff+"]) else "-",
                    "Cmd+": f"{r['Cmd+']:.0f}" if pd.notna(r["Cmd+"]) else "-",
                    "Pairs": r["Pairs"],
                })
            st.markdown("**Top 3‑Pitch Sequences**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if st.checkbox(
                "Show details (Tunnel / Stuff+ / Command+ / Sample size)",
                key=f"gpl_seq3_detail_{selected_pitcher}",
            ):
                st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)
        if top_seq4:
            top_seq4 = _assign_tactical_tags(top_seq4)
            rows = []
            detail = []
            for r in top_seq4:
                rows.append({
                    "Sequence": r["Seq"],
                    "Score": f"{r['Score']:.1f}" if pd.notna(r["Score"]) else "-",
                    "Whiff%": f"{r['Whiff%']:.1f}" if pd.notna(r["Whiff%"]) else "-",
                    "K/Putaway%": f"{r['K%']:.1f}" if pd.notna(r["K%"]) else "-",
                    "Avg EV": f"{r['Avg EV']:.1f}" if pd.notna(r["Avg EV"]) else "-",
                    "Tag": r.get("Tag", ""),
                    "Deception": _deception_flag(r.get("Tunnel", np.nan)),
                })
                detail.append({
                    "Sequence": r["Seq"],
                    "Tunnel": f"{r['Tunnel']:.1f}" if pd.notna(r["Tunnel"]) else "-",
                    "Stuff+": f"{r['Stuff+']:.0f}" if pd.notna(r["Stuff+"]) else "-",
                    "Cmd+": f"{r['Cmd+']:.0f}" if pd.notna(r["Cmd+"]) else "-",
                    "Pairs": r["Pairs"],
                })
            st.markdown("**Top 4‑Pitch Sequences**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if st.checkbox(
                "Show details (Tunnel / Stuff+ / Command+ / Sample size)",
                key=f"gpl_seq4_detail_{selected_pitcher}",
            ):
                st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)
    all_matchups = []
    for _, row in opp_hitters.iterrows():
        hitter_name = row["playerFullName"]
        profile = _get_opp_hitter_profile(tm, hitter_name, team, pitch_df=pitch_df)
        matchup = _score_pitcher_vs_hitter(arsenal, profile)
        if matchup:
            all_matchups.append(matchup)
    if not all_matchups:
        st.info("No matchup data generated.")
        return
    all_matchups.sort(key=lambda x: x["overall_score"], reverse=True)
    # ── Helper for bullpen cards ──
    _hfmt = lambda v, fmt=".0f": f"{v:{fmt}}" if not pd.isna(v) else "-"

    # Pre-compute and cache sequences for each matchup (used by bullpen cards)
    for m in all_matchups:
        ps = m["pitch_scores"]
        hd = m.get("hitter_data", {})
        sorted_ps = sorted(ps.items(), key=lambda x: x[1]["score"], reverse=True)
        top_seqs = _build_3pitch_sequences(sorted_ps, hd, tunnels, sequences)
        m["_cached_seqs"] = top_seqs
    # ── Bullpen Cards ──
    st.markdown("---")
    section_header("Bullpen Cards")
    # D1-wide percentile series for all hitter tendency labels
    _all_sw = tm["hitting"].get("swing_stats", pd.DataFrame())
    _all_sp = tm["hitting"].get("swing_pct", pd.DataFrame())
    _throws_key = "RHP" if arsenal["throws"] == "Right" else "LHP"
    def _pct_series(df, col):
        return pd.to_numeric(df[col], errors="coerce").dropna() if col in df.columns else pd.Series(dtype=float)
    # 2K whiff rates (hand-matched)
    _2k_hard_series = _pct_series(_all_sw, f"2K Whiff vs {_throws_key} Hard")
    _2k_os_series = _pct_series(_all_sw, f"2K Whiff vs {_throws_key} OS")
    # 1P swing% vs hard/CH
    _fp_hard_series = _pct_series(_all_sw, "1PSwing% vs Hard Empty")
    _fp_ch_series = _pct_series(_all_sw, "1PSwing% vs CH Empty")
    # Overall swing% by pitch class
    _sw_hard_series = _pct_series(_all_sp, "Swing% vs Hard")
    _sw_sl_series = _pct_series(_all_sp, "Swing% vs SL")
    _sw_cb_series = _pct_series(_all_sp, "Swing% vs CB")
    _sw_ch_series = _pct_series(_all_sp, "Swing% vs CH")
    # Map pitch class → percentile series
    _sw_pct_map = {"swing_vs_hard": _sw_hard_series, "swing_vs_sl": _sw_sl_series,
                   "swing_vs_cb": _sw_cb_series, "swing_vs_ch": _sw_ch_series}
    # Helper: compute percentile with safety
    def _pctile(series, val):
        if pd.isna(val) or len(series) <= 10:
            return np.nan
        return percentileofscore(series, val, kind="rank")
    # Helper: label a percentile
    def _pct_label(pct):
        if pct >= 85: return "elite"
        if pct >= 70: return "above avg"
        if pct >= 60: return "exploitable"
        if pct <= 15: return "elite low"
        if pct <= 30: return "below avg"
        return "avg"

    for matchup in all_matchups:
        hd = matchup.get("hitter_data", {})
        ps = matchup["pitch_scores"]
        sorted_ps = sorted(ps.items(), key=lambda x: x[1]["score"], reverse=True)
        # Scores already computed by unified _pitch_score_composite in _score_pitcher_vs_hitter
        composites = {pt_name: round(pt_data["score"], 0) for pt_name, pt_data in sorted_ps}
        score = matchup["overall_score"]
        # Use cached sequences from summary table (avoid recomputation)
        top_seqs = matchup.get("_cached_seqs") or _build_3pitch_sequences(sorted_ps, hd, tunnels, sequences)
        # Expander label: clean and simple
        with st.expander(
            f"{'🟢' if score > 60 else '🟡' if score > 48 else '🔴'} "
            f"{display_name(matchup['hitter'])} ({matchup['bats']}) — {matchup['platoon']} — "
            f"Score: {score:.0f}"
        ):
            # ── Shared setup for bullpen card sections ──
            our_throws = arsenal["throws"]  # "Right" or "Left"
            their_bats = matchup["bats"]  # "R", "L", "S"
            same_side = (our_throws == "Right" and their_bats in ("R", "Right")) or \
                        (our_throws == "Left" and their_bats in ("L", "Left"))
            # Percentiles for all hitter tendencies
            their_2k_hard = hd.get("whiff_2k_hard", np.nan)
            their_2k_os = hd.get("whiff_2k_os", np.nan)
            pct_2k_os = _pctile(_2k_os_series, their_2k_os)
            pct_2k_hard = _pctile(_2k_hard_series, their_2k_hard)

            # ── Shared data ──
            ps_dict = dict(sorted_ps)
            real_ps = [(n, d) for n, d in sorted_ps if d.get("usage", 0) >= 10]
            if not real_ps:
                real_ps = sorted_ps[:3]
            real_hard = [(n, d) for n, d in real_ps if n in _hard_pitches]
            real_os = [(n, d) for n, d in real_ps if n not in _hard_pitches]
            primary = max(real_ps, key=lambda x: x[1].get("usage", 0) or 0) if real_ps else None
            best_hard_p = real_hard[0] if real_hard else (real_ps[0] if real_ps else None)
            their_chase_val = hd.get("chase_pct", np.nan)

            # ── RESOLVE SEQUENCE FIRST so scouting can reference it ──
            fp_hard = hd.get("fp_swing_hard", np.nan)
            fp_pct = _pctile(_fp_hard_series, fp_hard)
            fp_context = ""
            if not pd.isna(fp_hard) and not pd.isna(fp_pct):
                if fp_pct <= 25:
                    fp_context = "passive"
                elif fp_pct >= 75:
                    fp_context = "aggressive"

            # Determine the active sequence (the one that drives the game plan)
            seq_p1, seq_p2, seq_p3 = None, None, None
            g12, g23 = "-", "-"
            active_seq = None
            if top_seqs:
                active_seq = top_seqs[0]
                seq_p1, seq_p2, seq_p3 = active_seq["p1"], active_seq["p2"], active_seq["p3"]
                _, g12 = _lookup_tunnel(seq_p1, seq_p2, tunnels)
                _, g23 = _lookup_tunnel(seq_p2, seq_p3, tunnels)
                # Aggressive-1P override: swap to offspeed-starting seq
                if fp_context == "aggressive" and seq_p1 in _hard_pitches and real_os:
                    os_seqs = [s for s in top_seqs if s["p1"] not in _hard_pitches]
                    if os_seqs:
                        active_seq = os_seqs[0]
                        seq_p1, seq_p2, seq_p3 = active_seq["p1"], active_seq["p2"], active_seq["p3"]
                        _, g12 = _lookup_tunnel(seq_p1, seq_p2, tunnels)
                        _, g23 = _lookup_tunnel(seq_p2, seq_p3, tunnels)
                # Passive-1P override: force hard-pitch-starting seq (attack zone)
                elif fp_context == "passive" and seq_p1 not in _hard_pitches and real_hard:
                    hard_seqs = [s for s in top_seqs if s["p1"] in _hard_pitches]
                    if hard_seqs:
                        active_seq = hard_seqs[0]
                        seq_p1, seq_p2, seq_p3 = active_seq["p1"], active_seq["p2"], active_seq["p3"]
                        _, g12 = _lookup_tunnel(seq_p1, seq_p2, tunnels)
                        _, g23 = _lookup_tunnel(seq_p2, seq_p3, tunnels)

            # ── UNIFIED AT-BAT SCRIPT (V3 — full game-prep depth) ──
            real_ps_names = {n for n, _ in real_ps}

            if active_seq:
                st.markdown("**At-Bat Script**")

                # ── Hitter profile snapshot ──
                prof_parts = []
                h_chase = hd.get("chase_pct", np.nan)
                h_kpct = hd.get("k_pct", np.nan)
                h_bbpct = hd.get("bb_pct", np.nan)
                h_swstrk = hd.get("swstrk_pct", np.nan)
                h_contact = hd.get("contact_pct", np.nan)
                if not pd.isna(h_kpct):
                    prof_parts.append(f"K% {h_kpct:.0f}")
                if not pd.isna(h_bbpct):
                    prof_parts.append(f"BB% {h_bbpct:.0f}")
                if not pd.isna(h_chase):
                    prof_parts.append(f"Chase {h_chase:.0f}%")
                if not pd.isna(h_swstrk):
                    prof_parts.append(f"SwStr {h_swstrk:.1f}%")
                if not pd.isna(h_contact):
                    prof_parts.append(f"Contact {h_contact:.0f}%")
                # Zone tendencies
                h_high = hd.get("high_pct", np.nan)
                h_low = hd.get("low_pct", np.nan)
                h_in = hd.get("inside_pct", np.nan)
                h_out = hd.get("outside_pct", np.nan)
                zone_parts = []
                if not pd.isna(h_high) and h_high > 55:
                    zone_parts.append("high-ball hitter")
                elif not pd.isna(h_low) and h_low > 55:
                    zone_parts.append("low-ball hitter")
                if not pd.isna(h_in) and h_in > 55:
                    zone_parts.append("pulls inside")
                elif not pd.isna(h_out) and h_out > 55:
                    zone_parts.append("covers outside")
                if prof_parts:
                    zone_tag = f" — {', '.join(zone_parts)}" if zone_parts else ""
                    st.caption(f"{' | '.join(prof_parts)}{zone_tag}")

                # ── ① FIRST PITCH ──
                fp_ars = arsenal["pitches"].get(seq_p1, {})
                fp_loc, fp_whiff = _best_putaway_zone(seq_p1, fp_ars)
                fp_loc_str = f" ({fp_loc}, {fp_whiff:.0f}% whiff)" if fp_loc and not pd.isna(fp_whiff) else ""
                if fp_context == "passive":
                    st.markdown(f"**① FIRST PITCH:** {seq_p1} — attack zone{fp_loc_str}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;_Passive 1P hitter ({fp_hard:.0f}% swing, {fp_pct:.0f}th %ile) — free strike_")
                elif fp_context == "aggressive":
                    st.markdown(f"**① FIRST PITCH:** {seq_p1} — steal strike{fp_loc_str}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;_Aggressive 1P hitter ({fp_hard:.0f}% swing, {fp_pct:.0f}th %ile) — will chase_")
                else:
                    st.markdown(f"**① FIRST PITCH:** {seq_p1} — compete{fp_loc_str}")
                # Per-class swing tendencies on 1P
                fp_sw_parts = []
                for sw_label, sw_key in [("Hard", "swing_vs_hard"), ("SL", "swing_vs_sl"),
                                          ("CB", "swing_vs_cb"), ("CH", "swing_vs_ch")]:
                    sv = hd.get(sw_key, np.nan)
                    if not pd.isna(sv):
                        _sw_ser = _sw_pct_map.get(sw_key, pd.Series(dtype=float))
                        sv_pct = _pctile(_sw_ser, sv)
                        if not pd.isna(sv_pct) and (sv_pct >= 70 or sv_pct <= 25):
                            tag = "agg" if sv_pct >= 70 else "passive"
                            fp_sw_parts.append(f"{sw_label} {sv:.0f}% ({tag})")
                if fp_sw_parts:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;_Swing rates: {', '.join(fp_sw_parts)}_")

                # ── ② TOP SEQUENCES (hitter-aware 3-pitch paths) ──
                if top_seqs:
                    st.markdown("**② TOP SEQUENCES** _(hitter-aware)_")
                    for i, s in enumerate(top_seqs):
                        sw23 = s.get("sw23", np.nan)
                        t12 = s.get("t12", np.nan)
                        t23 = s.get("t23", np.nan)
                        ev_gap = s.get("effv_gap", np.nan)
                        t2k = s.get("their_2k", np.nan)
                        # Tunnel grades from lookup
                        _, g12_s = _lookup_tunnel(s["p1"], s["p2"], tunnels)
                        _, g23_s = _lookup_tunnel(s["p2"], s["p3"], tunnels)
                        # Build detail line
                        detail_parts = []
                        if g12_s not in ("-", "F", None):
                            detail_parts.append(f"P1→P2 {g12_s} tunnel")
                        if g23_s not in ("-", "F", None):
                            detail_parts.append(f"P2→P3 {g23_s} tunnel")
                        if not pd.isna(sw23):
                            detail_parts.append(f"{sw23:.0f}% whiff on P3")
                        if not pd.isna(ev_gap):
                            detail_parts.append(f"{ev_gap:.1f} mph EffV gap")
                        if not pd.isna(t2k):
                            is_hard_p3 = s["p3"] in _hard_pitches
                            detail_parts.append(f"hitter {t2k:.0f}% 2K {'hard' if is_hard_p3 else 'OS'}")
                        # Best location for putaway
                        p3_ars = arsenal["pitches"].get(s["p3"], {})
                        p3_loc, p3_whiff = _best_putaway_zone(s["p3"], p3_ars)
                        if p3_loc:
                            detail_parts.append(f"P3 loc: {p3_loc}")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**{i+1}. {s['p1']} → {s['p2']} → {s['p3']}** (Score: {s['score']:.0f})")
                        if detail_parts:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_{' | '.join(detail_parts)}_")

                # ── 4-PITCH SEQUENCE (extended at-bat) ──
                seq4 = _build_4pitch_sequence(top_seqs, sorted_ps, hd, tunnels, sequences)
                if seq4:
                    detail4 = []
                    _, g12_4 = _lookup_tunnel(seq4["p1"], seq4["p2"], tunnels)
                    _, g23_4 = _lookup_tunnel(seq4["p2"], seq4["p3"], tunnels)
                    _, g34_4 = _lookup_tunnel(seq4["p3"], seq4["p4"], tunnels)
                    for label, grade in [("P1→P2", g12_4), ("P2→P3", g23_4), ("P3→P4", g34_4)]:
                        if grade not in ("-", "F", None):
                            detail4.append(f"{label} {grade}")
                    sw34 = seq4.get("sw34", np.nan)
                    if not pd.isna(sw34):
                        detail4.append(f"{sw34:.0f}% whiff on P4")
                    p4_ars = arsenal["pitches"].get(seq4["p4"], {})
                    p4_loc, _ = _best_putaway_zone(seq4["p4"], p4_ars)
                    if p4_loc:
                        detail4.append(f"P4 loc: {p4_loc}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**4P. {seq4['seq']}** (Score: {seq4['score']:.0f})")
                    if detail4:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_{' | '.join(detail4)}_")

                # ── WHEN BEHIND (1-0, 2-0, 2-1, 3-1, 3-2) ──
                if best_hard_p:
                    bh_ars = arsenal["pitches"].get(best_hard_p[0], {})
                    bh_loc, bh_whiff = _best_putaway_zone(best_hard_p[0], bh_ars)
                    bh_loc_str = f" — {bh_loc}" if bh_loc else ""
                    bh_whiff_str = f" ({bh_whiff:.0f}% whiff)" if not pd.isna(bh_whiff) else ""
                    st.markdown(f"**WHEN BEHIND (1-0, 2-0, 3-2):** {best_hard_p[0]}{bh_loc_str}{bh_whiff_str}")
                    # Secondary behind option if we have a second hard pitch
                    if len(real_hard) >= 2:
                        bh2_name, bh2_d = real_hard[1]
                        bh2_ars = arsenal["pitches"].get(bh2_name, {})
                        bh2_loc, bh2_whiff = _best_putaway_zone(bh2_name, bh2_ars)
                        bh2_loc_str = f" — {bh2_loc}" if bh2_loc else ""
                        bh2_whiff_str = f" ({bh2_whiff:.0f}% whiff)" if not pd.isna(bh2_whiff) else ""
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;_Alt: {bh2_name}{bh2_loc_str}{bh2_whiff_str}_")

                # ── 2K APPROACH ──
                os_whiff_str = f"{their_2k_os:.0f}% OS whiff ({pct_2k_os:.0f}th %ile, {_pct_label(pct_2k_os)})" if not pd.isna(their_2k_os) and not pd.isna(pct_2k_os) else None
                hard_whiff_str = f"{their_2k_hard:.0f}% Hard whiff ({pct_2k_hard:.0f}th %ile, {_pct_label(pct_2k_hard)})" if not pd.isna(their_2k_hard) and not pd.isna(pct_2k_hard) else None
                twok_lines = []
                if os_whiff_str:
                    if not pd.isna(pct_2k_os) and pct_2k_os >= 60:
                        os_recs = []
                        for n, d in real_os:
                            ars_n = arsenal["pitches"].get(n, {})
                            loc, loc_w = _best_putaway_zone(n, ars_n)
                            loc_detail = f" {loc} ({loc_w:.0f}%)" if loc and not pd.isna(loc_w) else ""
                            os_recs.append(f"{n}{loc_detail}")
                        twok_lines.append(f"{os_whiff_str} → {' / '.join(os_recs)}" if os_recs else os_whiff_str)
                    elif not pd.isna(pct_2k_os) and pct_2k_os <= 30:
                        twok_lines.append(f"{os_whiff_str} → avoid offspeed putaway")
                    else:
                        twok_lines.append(os_whiff_str)
                if hard_whiff_str:
                    if not pd.isna(pct_2k_hard) and pct_2k_hard >= 60:
                        hard_recs = []
                        for n, d in real_hard:
                            ars_n = arsenal["pitches"].get(n, {})
                            loc, loc_w = _best_putaway_zone(n, ars_n)
                            loc_detail = f" {loc} ({loc_w:.0f}%)" if loc and not pd.isna(loc_w) else ""
                            hard_recs.append(f"{n}{loc_detail}")
                        twok_lines.append(f"{hard_whiff_str} → {' / '.join(hard_recs)}" if hard_recs else hard_whiff_str)
                    elif not pd.isna(pct_2k_hard) and pct_2k_hard <= 30:
                        twok_lines.append(f"{hard_whiff_str} → avoid fastball putaway")
                    else:
                        twok_lines.append(hard_whiff_str)
                if twok_lines:
                    st.markdown("**2K APPROACH:**")
                    for line in twok_lines:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{line}")
                    # Per-pitch putaway ranking for 2K counts
                    # Hitter's 2K whiff by class is dominant — it's the hitter's
                    # actual vulnerability, not our pitcher's general whiff rate.
                    pw_rank = []
                    for n, d in real_ps:
                        pw_w = d.get("our_whiff", 0) or 0
                        pw_ch = d.get("our_chase", 0) or 0
                        is_h = n in _hard_pitches
                        t2k_val = hd.get("whiff_2k_hard" if is_h else "whiff_2k_os", np.nan)
                        ars_n = arsenal["pitches"].get(n, {})
                        loc, _ = _best_putaway_zone(n, ars_n)
                        # Weight: hitter 2K whiff (50%) > our whiff (25%) > chase (25%)
                        pw_score = (t2k_val * 0.50 if not pd.isna(t2k_val) else 0) + pw_w * 0.25 + pw_ch * 0.25
                        pw_rank.append((n, pw_score, pw_w, pw_ch, loc))
                    pw_rank.sort(key=lambda x: x[1], reverse=True)
                    if pw_rank:
                        st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;_Putaway ranking:_")
                        for rk_name, rk_sc, rk_w, rk_ch, rk_loc in pw_rank[:4]:
                            loc_tag = f" → {rk_loc}" if rk_loc else ""
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_{rk_name}: {rk_w:.0f}% whiff, {rk_ch:.0f}% chase{loc_tag}_")

                # ── Compact context footer ──
                footer_parts = []
                if not pd.isna(their_chase_val):
                    if their_chase_val > 32:
                        best_chase_p = max(real_ps, key=lambda x: x[1].get("our_chase", 0) or 0)
                        footer_parts.append(f"Chaser ({their_chase_val:.0f}% chase) — expand early with {best_chase_p[0]}")
                    elif their_chase_val < 18:
                        footer_parts.append(f"Elite discipline ({their_chase_val:.0f}% chase) — must throw strikes")
                if not footer_parts and not pd.isna(hd.get("bb_pct", np.nan)) and hd["bb_pct"] > 14:
                    footer_parts.append(f"Patient ({hd['bb_pct']:.0f}% BB) — attack zone")
                if footer_parts:
                    st.caption(" | ".join(footer_parts))

            else:
                # Fallback: no sequences — show all available data
                st.markdown("**At-Bat Script**")
                p1_name = primary[0] if primary else (best_hard_p[0] if best_hard_p else None)
                if p1_name:
                    p1_ars = arsenal["pitches"].get(p1_name, {})
                    p1_loc, p1_whiff = _best_putaway_zone(p1_name, p1_ars)
                    p1_loc_str = f" ({p1_loc}, {p1_whiff:.0f}% whiff)" if p1_loc and not pd.isna(p1_whiff) else ""
                    st.markdown(f"**① FIRST PITCH:** {p1_name} — compete{p1_loc_str}")
                if best_hard_p:
                    bh_ars = arsenal["pitches"].get(best_hard_p[0], {})
                    bh_loc, bh_whiff = _best_putaway_zone(best_hard_p[0], bh_ars)
                    bh_loc_str = f" — {bh_loc}" if bh_loc else ""
                    bh_whiff_str = f" ({bh_whiff:.0f}% whiff)" if not pd.isna(bh_whiff) else ""
                    st.markdown(f"**WHEN BEHIND (1-0, 2-0, 3-2):** {best_hard_p[0]}{bh_loc_str}{bh_whiff_str}")


def _hitting_plan_content(tm, team, data, season_filter, pitch_df=None):
    """Our Hitting Plan V2: pitcher scouting report + actionable hitter cards."""
    p_trad = _tm_team(tm["pitching"]["traditional"], team)
    if p_trad.empty:
        st.info("No opponent pitching data found.")
        return
    pitcher_ip = {}
    for name in p_trad["playerFullName"].unique():
        ip = _safe_num(_tm_player(p_trad, name), "IP")
        pitcher_ip[name] = ip if not pd.isna(ip) else 0
    opp_pitchers = sorted(pitcher_ip.keys(), key=lambda x: pitcher_ip[x], reverse=True)
    selected_opp = st.selectbox("Select Their Pitcher", opp_pitchers, key="gpl_opp_pitcher")
    opp_profile = _get_opp_pitcher_profile(tm, selected_opp, team, pitch_df=pitch_df)
    throws_str = "RHP" if opp_profile["throws"] == "R" else ("LHP" if opp_profile["throws"] == "L" else "BHP")
    st.markdown(f"### vs {selected_opp} ({throws_str})")

    # ── A. Pitcher Scouting Report ──
    col_ars, col_stats = st.columns([3, 1])
    with col_ars:
        if opp_profile["pitch_mix"]:
            arsenal_rows = sorted(opp_profile["pitch_mix"].items(), key=lambda x: x[1], reverse=True)
            fig = go.Figure(go.Bar(
                x=[r[0] for r in arsenal_rows],
                y=[r[1] for r in arsenal_rows],
                marker_color=[PITCH_COLORS.get(r[0], "#888") for r in arsenal_rows],
                text=[f"{r[1]:.1f}%" for r in arsenal_rows], textposition="outside",
                textfont=dict(size=11, color="#000000"),
            ))
            fig.update_layout(**CHART_LAYOUT, height=250, yaxis_title="Usage %", showlegend=False,
                              yaxis=dict(range=[0, max(r[1] for r in arsenal_rows) * 1.3]))
            st.plotly_chart(fig, use_container_width=True)
    with col_stats:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("ERA", f"{opp_profile['era']:.2f}" if not pd.isna(opp_profile["era"]) else "?")
            st.metric("Velo", f"{opp_profile['velo']:.1f}" if not pd.isna(opp_profile["velo"]) else "?")
        with c2:
            st.metric("SwStr%", f"{opp_profile['swstrk_pct']:.1f}" if not pd.isna(opp_profile["swstrk_pct"]) else "?")
            st.metric("Chase%", f"{opp_profile['chase_pct']:.1f}" if not pd.isna(opp_profile["chase_pct"]) else "?")

    # "How to Attack" summary
    st.markdown("**How to Attack**")
    attack_notes = []
    loc_tend = opp_profile.get("location_tendency", "")
    h_pct = opp_profile.get("loc_high_pct", np.nan)
    l_pct = opp_profile.get("loc_low_pct", np.nan)
    # Location tendency — pick dominant direction, don't show both up AND down
    has_up = not pd.isna(h_pct) and h_pct >= 30
    has_down = not pd.isna(l_pct) and l_pct >= 35
    if has_up and has_down:
        # Both high and low — pick the stronger tendency
        if h_pct >= l_pct:
            attack_notes.append(f"Lives up in zone ({h_pct:.0f}%) \u2014 sit on elevated FB if you handle high heat")
        else:
            attack_notes.append(f"Works down ({l_pct:.0f}%) \u2014 look low, don't chase below zone")
    elif has_up:
        attack_notes.append(f"Lives up in zone ({h_pct:.0f}%) \u2014 sit on elevated FB if you handle high heat")
    elif has_down:
        attack_notes.append(f"Works down ({l_pct:.0f}%) \u2014 look low, don't chase below zone")
    comploc = opp_profile.get("comploc_pct", np.nan)
    if not pd.isna(comploc) and comploc < 40:
        attack_notes.append(f"Low command ({comploc:.0f}% CompLoc) \u2014 expect hittable mistakes")
    # Tempo advice — don't say both "be aggressive early" AND "work counts"
    low_swstr = not pd.isna(opp_profile["swstrk_pct"]) and opp_profile["swstrk_pct"] < 8
    high_bb9 = not pd.isna(opp_profile["bb9"]) and opp_profile["bb9"] > 4
    if low_swstr and high_bb9:
        attack_notes.append(f"Low whiff ({opp_profile['swstrk_pct']:.1f}% SwStr) + walks ({opp_profile['bb9']:.1f} BB/9) \u2014 hunt early, make him throw strikes")
    elif low_swstr:
        attack_notes.append(f"Low swing-and-miss ({opp_profile['swstrk_pct']:.1f}% SwStr) \u2014 be aggressive early")
    elif high_bb9:
        attack_notes.append(f"Control issues ({opp_profile['bb9']:.1f} BB/9) \u2014 work counts")
    if not pd.isna(opp_profile["ev_against"]) and opp_profile["ev_against"] > 88:
        attack_notes.append(f"Gives up hard contact ({opp_profile['ev_against']:.1f} mph EV) \u2014 look to drive")
    # Putaway pitch warning
    putaway_cands = opp_profile.get("putaway_candidates", {})
    opp_swstr = opp_profile.get("swstrk_pct", np.nan)
    if putaway_cands:
        top_putaway = max(putaway_cands, key=putaway_cands.get)
        if not pd.isna(opp_swstr) and opp_swstr > 10:
            attack_notes.append(f"{top_putaway} is putaway ({putaway_cands[top_putaway]:.0f}%) \u2014 recognize early, don't chase low")
    p_per_bf = opp_profile.get("p_per_bf", np.nan)
    if not pd.isna(p_per_bf) and p_per_bf > 4.0:
        attack_notes.append(f"Avg {p_per_bf:.1f} P/BF \u2014 be ready for deep counts")
    if opp_profile["pitch_mix"]:
        primary_p = max(opp_profile["pitch_mix"].items(), key=lambda x: x[1])
        if primary_p[1] > 55:
            attack_notes.append(f"Relies heavily on {primary_p[0]} ({primary_p[1]:.0f}%) \u2014 sit on it early")
    for note in attack_notes[:4]:
        st.markdown(f"- {note}")
    st.markdown("---")

    # ── Score all hitters ──
    dav_hitting = filter_davidson(data, role="batter")
    if season_filter:
        dav_hitting = dav_hitting[dav_hitting["Season"].isin(season_filter)]
    our_hitters = sorted(h for h in dav_hitting["Batter"].unique() if _is_position_player(h))
    all_matchups = []
    hitter_profiles = {}
    for hitter_name in our_hitters:
        hitter_tm = _get_our_hitter_profile(data, hitter_name, season_filter)
        if hitter_tm is None:
            continue
        matchup = _score_hitter_vs_pitcher(hitter_tm, opp_profile)
        if matchup:
            hitter_profiles[hitter_name] = hitter_tm
            script = _generate_at_bat_script(hitter_tm, opp_profile, matchup)
            matchup["script"] = script
            all_matchups.append(matchup)
    if not all_matchups:
        st.warning("No matchup data available.")
        return
    all_matchups.sort(key=lambda x: x["overall_score"], reverse=True)

    # ── B. Lineup Rankings ──
    section_header("Lineup Rankings")
    summary_rows = []
    for i, m in enumerate(all_matchups):
        script = m.get("script")
        tag = script.get("tag_line", {}) if script else {}
        # Best pitch (Advantage)
        adv_pitches = [pe for pe in m.get("pitch_edges", []) if pe["edge"] == "Advantage"]
        vuln_pitches = [pe for pe in m.get("pitch_edges", []) if pe["edge"] == "Vulnerable"]
        best_str = "-"
        if adv_pitches:
            bp = max(adv_pitches, key=lambda pe: pe["our_ev"] if not pd.isna(pe["our_ev"]) else 0)
            ev_s = f"{bp['our_ev']:.0f} EV" if not pd.isna(bp["our_ev"]) else ""
            best_str = f"{bp['pitch']} ({ev_s})"
        watch_str = "-"
        if vuln_pitches:
            wp = max(vuln_pitches, key=lambda pe: pe["our_whiff"] if not pd.isna(pe["our_whiff"]) else 0)
            wh_s = f"{wp['our_whiff']:.0f}% whiff" if not pd.isna(wp["our_whiff"]) else ""
            watch_str = f"{wp['pitch']} ({wh_s})"
        # Key stat: pick most distinctive attribute
        disc_g = tag.get("discipline_grade", "-")
        bs_val = tag.get("bat_speed", "-")
        key_stat = f"{disc_g} disc" if disc_g not in ("-", "C", "D") else (f"{bs_val} bat" if bs_val != "-" else "-")
        # Overall EV from profile
        h_prof = hitter_profiles.get(m["hitter"])
        ov_ev = h_prof.get("overall", {}).get("avg_ev", np.nan) if h_prof else np.nan
        ov_ev_str = f"{ov_ev:.0f}" if not pd.isna(ov_ev) else "-"
        summary_rows.append({
            "Rank": i + 1,
            "Hitter": display_name(m["hitter"]),
            "Pos": POSITION.get(m["hitter"], ""),
            "Bats": m["bats"],
            "Platoon": m["platoon"],
            "Edge": f"{m['overall_score']:.0f}",
            "Avg EV": ov_ev_str,
            "Best Pitch": best_str,
            "Watch Out": watch_str,
            "Key Stat": key_stat,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── C. Top 9 Cards ──
    section_header("Recommended Top 9")
    top9 = all_matchups[:9]
    for row_start in range(0, len(top9), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= len(top9):
                break
            m = top9[idx]
            with col:
                clr = "#2ca02c" if m["overall_score"] > 60 else "#f7c631" if m["overall_score"] > 48 else "#d22d49"
                script = m.get("script")
                tag = script.get("tag_line", {}) if script else {}
                # One-line matchup summary
                adv_p = [pe for pe in m.get("pitch_edges", []) if pe["edge"] == "Advantage"]
                vuln_p = [pe for pe in m.get("pitch_edges", []) if pe["edge"] == "Vulnerable"]
                parts = []
                if adv_p:
                    bp = max(adv_p, key=lambda pe: pe["our_ev"] if not pd.isna(pe["our_ev"]) else 0)
                    ev_tag = f" ({bp['our_ev']:.0f})" if not pd.isna(bp["our_ev"]) else ""
                    parts.append(f"Sit {bp['pitch']}{ev_tag}")
                if vuln_p:
                    wp = max(vuln_p, key=lambda pe: pe["our_whiff"] if not pd.isna(pe["our_whiff"]) else 0)
                    wh_tag = f" ({wp['our_whiff']:.0f}%wh)" if not pd.isna(wp["our_whiff"]) else ""
                    parts.append(f"protect {wp['pitch']}{wh_tag}")
                summary_line = ", ".join(parts) if parts else "Neutral matchup"
                # Tag line info for card
                tag_str = tag.get("tag_line", "")
                # Position
                pos_str = POSITION.get(m["hitter"], "")
                # Barrel hunt one-liner for card
                barrel_line = ""
                if script:
                    bh = script.get("when_ahead", {}).get("barrel_hunt", "")
                    if bh:
                        barrel_line = bh.split(".")[0]  # first sentence only
                st.markdown(
                    f'<div style="padding:10px;background:white;border-radius:8px;border:1px solid #eee;'
                    f'border-left:4px solid {clr};margin:4px 0;">'
                    f'<div style="font-size:18px;font-weight:800;color:#1a1a2e;">#{idx+1}</div>'
                    f'<div style="font-size:14px;font-weight:700;color:#1a1a2e;">'
                    f'{display_name(m["hitter"])}</div>'
                    f'<div style="font-size:11px;color:#666;">{pos_str} | {m["platoon"]} | '
                    f'Edge: {m["overall_score"]:.0f}</div>'
                    f'<div style="font-size:10px;color:#888;margin-top:2px;">{tag_str}</div>'
                    f'<div style="font-size:11px;color:#333;font-weight:600;margin-top:4px;">{summary_line}</div>'
                    + (f'<div style="font-size:10px;color:#1a7a3a;margin-top:2px;">{barrel_line}</div>'
                       if barrel_line else '')
                    + f'</div>', unsafe_allow_html=True)

    # ── D. Per-Hitter Expanders ──
    st.markdown("---")
    section_header("At-Bat Scripts")
    for m in all_matchups:
        script = m.get("script")
        with st.expander(
            f"{'🟢' if m['overall_score'] > 60 else '🟡' if m['overall_score'] > 48 else '🔴'} "
            f"{display_name(m['hitter'])} ({m['bats']}) \u2014 Edge: {m['overall_score']:.0f} | {m['platoon']}"
        ):
            if not script:
                st.caption("No Trackman data \u2014 use pitcher tendencies above.")
                continue

            tag = script.get("tag_line", {})
            scout_lines = script.get("scout_lines", [])

            # ── Tag Line ──
            st.markdown(f"**{display_name(tag.get('name', m['hitter']))}** \u2014 {tag.get('tag_line', '')}")

            # ── Hit Lab Profile (barrel zones, attack angle, sweet spot) ──
            h_prof = hitter_profiles.get(m["hitter"], {})
            lab_parts = []
            ov = h_prof.get("overall", {})
            hh_pct = ov.get("hard_hit_pct", np.nan)
            ss_pct = ov.get("sweet_spot_pct", np.nan)
            brl_ct = ov.get("barrel_count", 0)
            sp_data = h_prof.get("swing_path") or {}
            aa = sp_data.get("attack_angle")
            if not pd.isna(hh_pct):
                lab_parts.append(f"{hh_pct:.0f}% hard hit")
            if not pd.isna(ss_pct):
                lab_parts.append(f"{ss_pct:.0f}% sweet spot")
            if brl_ct and brl_ct >= 2:
                lab_parts.append(f"{brl_ct} barrels")
            if aa is not None:
                lab_parts.append(f"{aa:.1f}\u00b0 attack angle")
            bz = h_prof.get("barrel_zones", [])
            if bz:
                top_bz = bz[0]
                lab_parts.append(f"barrels {top_bz[0].replace('_', ' ').lower()} ({top_bz[1]} barrels, {top_bz[2]:.0f} EV)")
            if lab_parts:
                st.caption(" | ".join(lab_parts))

            # ── Scouting This Pitcher ──
            if scout_lines:
                st.markdown("**SCOUTING THIS PITCHER**")
                st.markdown(" | ".join(scout_lines))

            # ── At-Bat Sections ──
            c1, c2 = st.columns(2)
            fp = script.get("first_pitch", {})
            with c1:
                st.markdown("**\u2460 1ST PITCH**")
                st.markdown(f"Expect: **{fp.get('expect', '?')}**")
                st.markdown(f"Plan: **{fp.get('plan', '?')}** \u2014 {fp.get('detail', '')}")
                if fp.get("look"):
                    st.markdown(f"Look: {fp['look']}")

                st.markdown("")
                ts = script.get("two_strike", {})
                st.markdown("**\u2462 2 STRIKES**")
                st.markdown(f"He'll throw: **{ts.get('expect', '?')}**")
                if ts.get("chase_warning"):
                    st.markdown(f"**{ts['chase_warning']}**")
                st.markdown(f"Protect: {ts.get('protect', 'Shorten up')}")
                if ts.get("count_note"):
                    st.markdown(f"*{ts['count_note']}*")
                if ts.get("class_whiff_note"):
                    st.markdown(f"*{ts['class_whiff_note']}*")
                # Putaway danger ranking
                pw_danger = ts.get("putaway_danger", [])
                if pw_danger and len(pw_danger) >= 2:
                    st.markdown("**Watch for (2K):**")
                    for pw_tup in pw_danger:
                        pw_p, pw_score, pw_2k_wh, pw_is_hard = pw_tup[0], pw_tup[1], pw_tup[2], pw_tup[3]
                        pw_their_wh = pw_tup[4] if len(pw_tup) > 4 else np.nan
                        pw_usage = pw_tup[5] if len(pw_tup) > 5 else np.nan
                        parts = []
                        if not pd.isna(pw_usage):
                            parts.append(f"{pw_usage:.0f}% usage")
                        if not pd.isna(pw_their_wh):
                            parts.append(f"{pw_their_wh:.0f}% their whiff")
                        if not pd.isna(pw_2k_wh):
                            cls_lbl = "hard" if pw_is_hard else "OS"
                            parts.append(f"your {cls_lbl} 2K whiff {pw_2k_wh:.0f}%")
                        st.markdown(f"- **{pw_p}**: {', '.join(parts)}" if parts else f"- {pw_p}")

            ah = script.get("when_ahead", {})
            with c2:
                st.markdown("**\u2461 AHEAD IN COUNT**")
                st.markdown(f"{ah.get('sit_on', 'Be aggressive in zone')}")
                if ah.get("barrel_hunt"):
                    st.markdown(f"**{ah['barrel_hunt']}**")
                elif ah.get("hunt_zone"):
                    st.markdown(f"Hunt zone: {ah['hunt_zone']}")
                if ah.get("count_note"):
                    st.markdown(f"*{ah['count_note']}*")

                # ── When Behind ──
                wb = script.get("when_behind", {})
                if wb.get("note") or wb.get("count_note"):
                    st.markdown("")
                    st.markdown("**WHEN BEHIND**")
                    if wb.get("note"):
                        st.markdown(wb["note"])
                    if wb.get("count_note"):
                        st.markdown(f"*{wb['count_note']}*")

                st.markdown("")
                # ── Pitch Edges ──
                edges = script.get("pitch_edges", [])
                if edges:
                    st.markdown("**PITCH EDGES**")
                    for el in edges:
                        sym = el["symbol"]
                        edge_lbl = el["edge"]
                        line = f"{sym} **{el['pitch']}** \u2014 {edge_lbl}: {el['stats']}"
                        if el.get("reason") and edge_lbl != "Neutral":
                            line += f" ({el['reason']})"
                        st.markdown(line)

                # Pull tendency warning
                if script.get("pull_warning"):
                    st.markdown(f"**{script['pull_warning']}**")


def page_scouting(data):
    st.header("Opponent Scouting")

    # ── Season selector ──
    col_yr, col_src, col_cache = st.columns([1, 2, 1])
    with col_yr:
        season_year = st.selectbox("Season", [2026, 2025, 2024], index=0, key="sc_api_yr")
    with col_src:
        use_league_pct = st.checkbox(
            "Use NCAA D1 percentiles",
            value=True,
            key="sc_use_league_pct",
        )
    with col_cache:
        # Show cache status and refresh button
        if use_league_pct:
            cache_info = get_league_cache_info(season_year)
            if cache_info.get("hitters", {}).get("cached"):
                st.caption(f"Cached: {cache_info['hitters']['updated']}")
            if st.button("Refresh D1 Data", key="sc_refresh_cache", help="Clear cached data and fetch fresh stats from TrueMedia"):
                clear_league_cache(season_year)
                st.rerun()

    # ── Fetch teams from TrueMedia API ──
    tok = get_temp_token()
    if not tok:
        st.error("Could not authenticate with TrueMedia API. Set TM_MASTER_TOKEN in environment or Streamlit secrets.")
        return

    teams_df = fetch_all_teams(season_year)
    if teams_df.empty:
        st.warning(f"No teams found for {season_year} season.")
        return

    # Build team name -> teamId lookup
    # Use fullName (e.g. "Elon University") since it matches mostRecentTeamName in PlayerTotals
    team_name_col = None
    for c in ["fullName", "newestTeamName", "teamName", "Team", "team", "name"]:
        if c in teams_df.columns:
            team_name_col = c
            break
    team_id_col = None
    for c in ["teamId", "id", "teamID"]:
        if c in teams_df.columns:
            team_id_col = c
            break

    if team_name_col is None or team_id_col is None:
        # Fallback: show raw column names for debugging
        st.info(f"Team columns: {list(teams_df.columns)[:15]}")
        st.dataframe(teams_df.head(5))
        return

    teams_df = teams_df.dropna(subset=[team_name_col])
    team_lookup = dict(zip(teams_df[team_name_col], teams_df[team_id_col]))
    all_team_names = sorted(
        n for n in team_lookup.keys()
        if "davidson" not in str(n).lower()
    )

    if not all_team_names:
        st.info("No opponent teams found.")
        return

    # Inject Bryant combined as a top option
    _BRYANT_LABEL = "Bryant (2024-25 Combined)"
    all_team_names_with_bryant = [_BRYANT_LABEL] + all_team_names

    team = st.selectbox("Opponent", all_team_names_with_bryant, key="sc_team_api")

    if team == _BRYANT_LABEL:
        from config import BRYANT_COMBINED_TEAM_ID
        from data.bryant_combined import load_bryant_combined_pack
        team_id = BRYANT_COMBINED_TEAM_ID
        tm = load_bryant_combined_pack()
        if tm is None:
            st.warning("Bryant combined pack not built yet. Go to **Bryant Scouting** page first to build it.")
            return
    else:
        team_id = team_lookup[team]
        # ── Fetch team data via API ──
        with st.spinner(f"Loading {team} scouting data..."):
            tm = build_tm_dict_for_team(team_id, team, season_year)

    # ── v2: Local baserunning + defense intel (from CSV exports) ──
    with st.expander("Baserunning & Defense Intel (v2)", expanded=False):
        try:
            from decision_engine.data.baserunning_data import load_count_sb_squeeze, load_speed_scores
            from decision_engine.data.fielding_intel import team_defense_profile

            spd = load_speed_scores()
            if not spd.empty and "newestTeamName" in spd.columns:
                spd_t = spd[spd["newestTeamName"].astype(str).str.strip() == team].copy()
            else:
                spd_t = pd.DataFrame()

            if not spd_t.empty:
                show_cols = [c for c in ["playerFullName", "SpeedScore", "SB", "SBA", "SB%"] if c in spd_t.columns]
                st.caption("Team speed leaders (SpeedScore)")
                st.dataframe(
                    spd_t[show_cols].sort_values("SpeedScore", ascending=False).head(12),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No SpeedScore rows found for this team in local exports.")

            sq = load_count_sb_squeeze()
            if not sq.empty and "newestTeamName" in sq.columns:
                sq_t = sq[sq["newestTeamName"].astype(str).str.strip() == team].copy()
            else:
                sq_t = pd.DataFrame()
            if not sq_t.empty:
                keep = [c for c in ["playerFullName", "Squeeze #", "Bunt Hit Att", "SBA"] if c in sq_t.columns]
                threats = sq_t[(sq_t.get("Squeeze #", 0) >= 2) | (sq_t.get("Bunt Hit Att", 0) >= 3)].copy()
                st.caption("Squeeze/Bunt threats")
                if threats.empty:
                    st.write("None flagged (Squeeze # < 2 and Bunt Hit Att < 3).")
                else:
                    st.dataframe(
                        threats[keep].sort_values(["Squeeze #", "Bunt Hit Att"], ascending=False).head(20),
                        use_container_width=True,
                        hide_index=True,
                    )

            prof = team_defense_profile(team)
            starters = prof.get("starters", pd.DataFrame()) if isinstance(prof, dict) else pd.DataFrame()
            if starters is not None and not starters.empty:
                cols = [c for c in ["pos", "playerFullName", "Inn", "Chances", "E", "FLD%", "quality", "note"] if c in starters.columns]
                st.caption("Likely primary fielders by position")
                st.dataframe(starters[cols].sort_values("pos"), use_container_width=True, hide_index=True)

            of_arms = prof.get("of_arms", pd.DataFrame()) if isinstance(prof, dict) else pd.DataFrame()
            if of_arms is not None and not of_arms.empty:
                cols = [c for c in ["pos", "playerFullName", "InnOF", "OFAst", "ArmRating", "OFThrowE"] if c in of_arms.columns]
                st.caption("Outfield arms")
                st.dataframe(of_arms[cols].sort_values(["ArmRating", "OFAst"], ascending=[True, False]).head(25),
                             use_container_width=True, hide_index=True)
        except Exception as e:
            st.caption(f"Local baserunning/defense intel not available: {e}")

    # ── Load NCAA D1 context for percentiles (from TrueMedia API with disk cache) ──
    league_hitters = None
    league_pitchers = None
    if use_league_pct:
        with st.spinner("Loading NCAA D1 percentiles..."):
            try:
                # Use TrueMedia API with disk caching for full traditional stats
                # (OPS, wOBA, BA, ERA, FIP, etc. that local Trackman data lacks).
                # First load is slow (~5-10s), subsequent loads are instant from cache.
                league_hitters = build_tm_dict_for_league_hitters(season_year, allow_fallback=True)
                league_pitchers = build_tm_dict_for_league_pitchers(season_year, allow_fallback=True)
            except Exception:
                st.warning("Could not load D1-wide percentiles. Comparing against opponent team only.")

    # ── Load pitch-level data from TrueMedia GamePitchesTrackman ──
    opp_pitches = pd.DataFrame()
    try:
        opp_pitches = fetch_team_all_pitches_trackman(team_id, season_year)
    except Exception:
        opp_pitches = pd.DataFrame()

    # Show column diagnostics for debugging (collapsed)
    if not opp_pitches.empty:
        if st.checkbox("Show pitch data diagnostics", key="sc_tm_pitch_diag"):
            st.caption(f"{len(opp_pitches)} pitches loaded | Columns: {list(opp_pitches.columns)[:30]}")

    # Fallback: load opponent Trackman from local D1 parquet if API returned nothing
    if opp_pitches.empty:
        try:
            tm_names = set()
            for side_key in ["hitting", "pitching"]:
                for tbl in tm.get(side_key, {}).values():
                    if isinstance(tbl, pd.DataFrame) and "playerFullName" in tbl.columns:
                        team_df = _tm_team(tbl, team)
                        if not team_df.empty:
                            tm_names.update(team_df["playerFullName"].dropna().unique())
            if tm_names:
                trackman_names = tuple(sorted(set(
                    tm_name_to_trackman(n) for n in tm_names if n
                )))
                if trackman_names:
                    opp_pitches = load_opponent_trackman(trackman_names, tuple([season_year]))
        except Exception:
            opp_pitches = pd.DataFrame()

    # Normalize pitch types once at the source so all downstream analysis uses clean data
    if not opp_pitches.empty:
        opp_pitches = normalize_pitch_types(opp_pitches)

    # ── Load count-filtered aggregate stats ──
    count_splits = {}
    try:
        count_splits = fetch_hitter_count_splits(team_id, season_year)
    except Exception:
        count_splits = {}

    tab_overview, tab_hitters, tab_pitchers, tab_catchers = st.tabs([
        "Team Overview", "Their Hitters", "Their Pitchers", "Their Catchers"
    ])

    # ── Tab 1: Team Overview ──
    with tab_overview:
        _scouting_team_overview(tm, team, season_year)

    # ── Tab 2: Their Hitters ──
    with tab_hitters:
        _scouting_hitter_report(tm, team, opp_pitches, count_splits, league_hitters=league_hitters)

    # ── Tab 3: Their Pitchers ──
    with tab_pitchers:
        _scouting_pitcher_report(tm, team, opp_pitches, league_pitchers=league_pitchers)

    # ── Tab 4: Their Catchers ──
    with tab_catchers:
        _scouting_catcher_report(tm, team)


def _scouting_team_overview(tm, team, season_year=2026):
    """Team Overview tab — aggregate offense + pitching metrics with D1 rankings."""
    h_cnt = _tm_team(tm["hitting"]["counting"], team)
    h_rate = _tm_team(tm["hitting"]["rate"], team)
    p_trad = _tm_team(tm["pitching"]["traditional"], team)

    # Fetch D1-wide team totals from API for ranking context
    d1_h_totals = fetch_team_totals_hitting(season_year)
    d1_p_totals = fetch_team_totals_pitching(season_year)

    # Detect team name column in team totals
    def _team_name_col(df):
        for c in ["fullName", "teamName", "newestTeamName", "Team", "team", "name"]:
            if c in df.columns:
                return c
        return None

    # ── Build team-level offense aggregates from player data for this team ──
    this_team_off = {}
    if not h_cnt.empty:
        pa = h_cnt["PA"].sum() if "PA" in h_cnt.columns else 0
        ab = h_cnt["AB"].sum() if "AB" in h_cnt.columns else 0
        this_team_off = {
            "BA": h_cnt["H"].sum() / max(ab, 1) if "H" in h_cnt.columns else np.nan,
            "HR": h_cnt["HR"].sum() if "HR" in h_cnt.columns else 0,
            "SB": h_cnt["SB"].sum() if "SB" in h_cnt.columns else 0,
            "K%": h_cnt["K"].sum() / max(pa, 1) * 100 if "K" in h_cnt.columns else np.nan,
            "BB%": h_cnt["BB"].sum() / max(pa, 1) * 100 if "BB" in h_cnt.columns else np.nan,
        }

    # ── Build team-level pitching aggregates from player data ──
    this_team_pit = {}
    if not p_trad.empty:
        ip = p_trad["IP"].sum() if "IP" in p_trad.columns else 0
        this_team_pit = {
            "ERA": p_trad["ER"].sum() / max(ip, 1) * 9 if "ER" in p_trad.columns else np.nan,
            "K/9": p_trad["K"].sum() / max(ip, 1) * 9 if "K" in p_trad.columns else np.nan,
            "BB/9": p_trad["BB"].sum() / max(ip, 1) * 9 if "BB" in p_trad.columns else np.nan,
            "IP": ip,
        }

    # ── Build D1-wide ranking context from TeamTotals API ──
    d1_off_all, d1_pit_all = {}, {}
    tn_col_h = _team_name_col(d1_h_totals) if not d1_h_totals.empty else None
    if tn_col_h and not d1_h_totals.empty:
        for _, row in d1_h_totals.iterrows():
            tname = row[tn_col_h]
            pa = row.get("PA", 0) or 0
            ab = row.get("AB", 0) or 0
            d1_off_all[tname] = {
                "BA": row.get("H", 0) / max(ab, 1),
                "HR": row.get("HR", 0) or 0,
                "SB": row.get("SB", 0) or 0,
                "K%": row.get("K", 0) / max(pa, 1) * 100,
                "BB%": row.get("BB", 0) / max(pa, 1) * 100,
            }

    tn_col_p = _team_name_col(d1_p_totals) if not d1_p_totals.empty else None
    if tn_col_p and not d1_p_totals.empty:
        for _, row in d1_p_totals.iterrows():
            tname = row[tn_col_p]
            ip = row.get("IP", 0) or 0
            d1_pit_all[tname] = {
                "ERA": row.get("ERA", np.nan),
                "K/9": row.get("K/9", np.nan),
                "BB/9": row.get("BB/9", np.nan),
                "IP": ip,
            }

    # Use D1 data if available, fall back to single-team data
    if d1_off_all:
        team_off_all = d1_off_all
        # Make sure this team is represented (use our computed values)
        if team not in team_off_all and this_team_off:
            team_off_all[team] = this_team_off
    elif this_team_off:
        team_off_all = {team: this_team_off}
    else:
        team_off_all = {}

    if d1_pit_all:
        team_pit_all = d1_pit_all
        if team not in team_pit_all and this_team_pit:
            team_pit_all[team] = this_team_pit
    elif this_team_pit:
        team_pit_all = {team: this_team_pit}
    else:
        team_pit_all = {}

    n_teams = max(len(team_off_all), len(team_pit_all))
    has_d1_context = n_teams > 1

    # ── Team Narrative ──
    team_narrative = []
    if team in team_off_all:
        off = team_off_all[team]
        if has_d1_context:
            ba_rank = sum(1 for t in team_off_all.values() if t["BA"] > off["BA"]) + 1
            hr_rank = sum(1 for t in team_off_all.values() if t["HR"] > off["HR"]) + 1
            k_rank = sum(1 for t in team_off_all.values() if t["K%"] < off["K%"]) + 1
            team_narrative.append(
                f"**Offense:** {team} hits **.{int(off['BA']*1000):03d}** as a team "
                f"(#{ba_rank} of {n_teams} D1 teams), with **{off['HR']}** HR (#{hr_rank}) "
                f"and a **{off['K%']:.1f}%** K rate (#{k_rank})."
            )
        else:
            team_narrative.append(
                f"**Offense:** {team} hits **.{int(off['BA']*1000):03d}** as a team, "
                f"with **{int(off['HR'])}** HR and a **{off['K%']:.1f}%** K rate."
            )
        if off["K%"] > 22:
            team_narrative.append("This lineup is strikeout-prone — expand the zone and use putaway pitches.")
        elif off["BB%"] > 10:
            team_narrative.append("A patient lineup — throw strikes early and avoid falling behind.")
        if off["HR"] > 40:
            team_narrative.append("Power-heavy — keep the ball down and limit mistakes over the plate.")
        if off["SB"] > 50:
            team_narrative.append(f"Speed threat ({int(off['SB'])} SB) — quick deliveries and catcher readiness critical.")

    if team in team_pit_all:
        pit = team_pit_all[team]
        if has_d1_context:
            era_rank = sum(1 for t in team_pit_all.values()
                          if pd.notna(t.get("ERA")) and pd.notna(pit.get("ERA")) and t["ERA"] < pit["ERA"]) + 1
            k9_rank = sum(1 for t in team_pit_all.values()
                         if pd.notna(t.get("K/9")) and pd.notna(pit.get("K/9")) and t["K/9"] > pit["K/9"]) + 1
            team_narrative.append(
                f"**Pitching:** Staff ERA of **{pit['ERA']:.2f}** (#{era_rank} of {n_teams}), "
                f"**{pit['K/9']:.1f}** K/9 (#{k9_rank}), **{pit['BB/9']:.1f}** BB/9."
            )
        else:
            era_val = pit.get("ERA", np.nan)
            k9_val = pit.get("K/9", np.nan)
            bb9_val = pit.get("BB/9", np.nan)
            parts = []
            if pd.notna(era_val):
                parts.append(f"Staff ERA of **{era_val:.2f}**")
            if pd.notna(k9_val):
                parts.append(f"**{k9_val:.1f}** K/9")
            if pd.notna(bb9_val):
                parts.append(f"**{bb9_val:.1f}** BB/9")
            if parts:
                team_narrative.append(f"**Pitching:** {', '.join(parts)}.")
        if pd.notna(pit.get("ERA")) and pit["ERA"] > 5.0:
            team_narrative.append("Hittable staff — be aggressive and attack early.")
        elif pd.notna(pit.get("ERA")) and pit["ERA"] < 3.5:
            team_narrative.append("Elite pitching staff — need disciplined, quality at-bats.")
        if pd.notna(pit.get("BB/9")) and pit["BB/9"] > 4.0:
            team_narrative.append("Control issues — work deep counts and take walks.")

    if team_narrative:
        st.markdown("\n\n".join(team_narrative))
    else:
        st.info("No aggregate data available for this team.")

    # ── Top Players Tables ──
    col_th, col_tp = st.columns(2)
    with col_th:
        section_header("Top Hitters (by PA)")
        if not h_rate.empty:
            all_rate = tm["hitting"]["rate"]
            # Build display columns dynamically from available data
            base_cols = ["playerFullName"]
            for c in ["PA", "BA", "OBP", "SLG", "OPS"]:
                if c in h_rate.columns:
                    base_cols.append(c)
            d = h_rate[base_cols].copy()
            if "PA" in d.columns:
                d = d.sort_values("PA", ascending=False).head(10)
            else:
                d = d.head(10)
            # OPS percentile (within roster — API only has this team)
            if "OPS" in d.columns and "OPS" in all_rate.columns and len(all_rate) > 1:
                d["OPS %ile"] = d["OPS"].apply(
                    lambda x: int(percentileofscore(all_rate["OPS"].dropna(), x, kind='rank')) if pd.notna(x) else np.nan
                )
            rename_map = {"playerFullName": "Player"}
            for c in ["BA", "OBP", "SLG", "OPS"]:
                if c in d.columns:
                    d[c] = d[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
            if "OPS %ile" in d.columns:
                d["OPS %ile"] = d["OPS %ile"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
            d = d.rename(columns=rename_map)
            st.dataframe(d, use_container_width=True, hide_index=True)
    with col_tp:
        section_header("Top Pitchers (by IP)")
        if not p_trad.empty:
            base_cols = ["playerFullName"]
            for c in ["GS", "IP", "ERA", "FIP", "WHIP", "K/9"]:
                if c in p_trad.columns:
                    base_cols.append(c)
            d = p_trad[base_cols].copy()
            if "IP" in d.columns:
                d = d.sort_values("IP", ascending=False).head(10)
            else:
                d = d.head(10)
            all_trad = tm["pitching"]["traditional"]
            if "ERA" in d.columns and "ERA" in all_trad.columns and len(all_trad) > 1:
                d["ERA %ile"] = d["ERA"].apply(
                    lambda x: int(100 - percentileofscore(all_trad["ERA"].dropna(), x, kind='rank')) if pd.notna(x) else np.nan
                )
            for c in ["ERA", "FIP", "WHIP"]:
                if c in d.columns:
                    d[c] = d[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            if "K/9" in d.columns:
                d["K/9"] = d["K/9"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
            if "ERA %ile" in d.columns:
                d["ERA %ile"] = d["ERA %ile"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
            d = d.rename(columns={"playerFullName": "Player"})
            st.dataframe(d, use_container_width=True, hide_index=True)

    # ── Feature #9: Lineup Builder / Batting Order Analysis ──
    section_header("Lineup Analysis")
    h_exit = _tm_team(tm["hitting"]["exit"], team)
    h_speed = _tm_team(tm["hitting"]["speed"], team)
    h_prates = _tm_team(tm["hitting"]["pitch_rates"], team)
    h_htypes = _tm_team(tm["hitting"]["hit_types"], team)
    h_hlocs = _tm_team(tm["hitting"]["hit_locations"], team)

    if not h_rate.empty and not h_cnt.empty:
        # Build unified lineup dataframe
        lineup = h_cnt[["playerFullName", "PA", "AB", "H", "HR", "RBI", "BB", "K", "SB"]].copy()
        lineup = lineup[lineup["PA"] >= 10].copy()  # min 10 PA

        # Merge rate stats
        if not h_rate.empty:
            rate_cols = ["playerFullName"]
            for c in ["BA", "OBP", "SLG", "OPS", "ISO", "WOBA", "BABIP"]:
                if c in h_rate.columns:
                    rate_cols.append(c)
            lineup = lineup.merge(h_rate[rate_cols], on="playerFullName", how="left")
            if "WOBA" in lineup.columns:
                lineup = lineup.rename(columns={"WOBA": "wOBA"})

        # Merge exit data
        if not h_exit.empty and "ExitVel" in h_exit.columns:
            ev_cols = ["playerFullName"]
            for c in ["ExitVel", "Barrel%", "HardHit%"]:
                if c in h_exit.columns:
                    ev_cols.append(c)
            lineup = lineup.merge(h_exit[ev_cols], on="playerFullName", how="left")

        # Merge speed
        if not h_speed.empty and "SpeedScore" in h_speed.columns:
            lineup = lineup.merge(h_speed[["playerFullName", "SpeedScore"]], on="playerFullName", how="left")

        # Merge K%, BB% from rate stats (these are in rate, not pitch_rates)
        if not h_rate.empty:
            for c in ["K%", "BB%"]:
                if c in h_rate.columns and c not in lineup.columns:
                    lineup = lineup.merge(h_rate[["playerFullName", c]], on="playerFullName", how="left")
        # Merge Chase%, SwStrk% from pitch rates
        if not h_prates.empty:
            pr_cols = ["playerFullName"]
            for c in ["Chase%", "SwStrk%"]:
                if c in h_prates.columns:
                    pr_cols.append(c)
            if len(pr_cols) > 1:
                lineup = lineup.merge(h_prates[pr_cols], on="playerFullName", how="left")

        # Compute a composite "production score" for sorting
        lineup["ProdScore"] = 0.0
        if "wOBA" in lineup.columns:
            woba_med = lineup["wOBA"].median()
            lineup["ProdScore"] += lineup["wOBA"].fillna(woba_med) * 100
        elif "OPS" in lineup.columns:
            ops_med = lineup["OPS"].median()
            lineup["ProdScore"] += lineup["OPS"].fillna(ops_med) * 50
        if "ExitVel" in lineup.columns:
            lineup["ProdScore"] += lineup["ExitVel"].fillna(80) * 0.3
        if "SpeedScore" in lineup.columns:
            lineup["ProdScore"] += lineup["SpeedScore"].fillna(3) * 2

        lineup = lineup.sort_values("ProdScore", ascending=False)

        # Classify hitters into roles
        def _classify_hitter(row):
            roles = []
            if pd.notna(row.get("ISO")) and row["ISO"] >= 0.180:
                roles.append("Power")
            elif pd.notna(row.get("HR")) and row["HR"] >= 5:
                roles.append("Power")
            if pd.notna(row.get("SpeedScore")) and row["SpeedScore"] >= 5.0:
                roles.append("Speed")
            elif pd.notna(row.get("SB")) and row["SB"] >= 5:
                roles.append("Speed")
            if pd.notna(row.get("BB%")) and row["BB%"] >= 10:
                roles.append("Patient")
            if pd.notna(row.get("Chase%")) and row["Chase%"] <= 20:
                roles.append("Disciplined")
            if pd.notna(row.get("BA")) and row["BA"] >= 0.300:
                roles.append("Contact")
            elif pd.notna(row.get("K%")) and row["K%"] <= 15:
                roles.append("Contact")
            if pd.notna(row.get("ExitVel")) and row["ExitVel"] >= 90:
                roles.append("Hard-Hit")
            if not roles:
                roles.append("Utility")
            return ", ".join(roles)

        lineup["Role"] = lineup.apply(_classify_hitter, axis=1)

        # Display lineup table
        display_cols = ["playerFullName", "PA"]
        disp_names = {"playerFullName": "Player", "PA": "PA"}
        for c, name, fmt in [
            ("BA", "BA", ".3f"), ("OBP", "OBP", ".3f"), ("SLG", "SLG", ".3f"),
            ("wOBA", "wOBA", ".3f"), ("ISO", "ISO", ".3f"),
            ("HR", "HR", "d"), ("SB", "SB", "d"),
            ("ExitVel", "EV", ".1f"), ("Barrel%", "Brl%", ".1f"),
            ("SpeedScore", "Spd", ".1f"), ("Chase%", "Chase%", ".1f"),
        ]:
            if c in lineup.columns:
                display_cols.append(c)
                disp_names[c] = name
        display_cols.append("Role")
        disp_names["Role"] = "Profile"

        disp = lineup[display_cols].head(15).copy()
        disp = disp.rename(columns=disp_names)
        # Format numeric columns
        for c in disp.columns:
            if c in ["BA", "OBP", "SLG", "wOBA", "ISO"]:
                disp[c] = disp[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
            elif c in ["EV", "Spd", "Brl%", "Chase%"]:
                disp[c] = disp[c].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
            elif c in ["HR", "SB", "PA"]:
                disp[c] = disp[c].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # Lineup insights
        insights = []
        power_count = lineup[lineup.get("Role", pd.Series(dtype=str)).str.contains("Power", na=False)].shape[0] if "Role" in lineup.columns else 0
        speed_count = lineup[lineup.get("Role", pd.Series(dtype=str)).str.contains("Speed", na=False)].shape[0] if "Role" in lineup.columns else 0
        patient_count = lineup[lineup.get("Role", pd.Series(dtype=str)).str.contains("Patient|Disciplined", na=False)].shape[0] if "Role" in lineup.columns else 0

        if power_count >= 4:
            insights.append(f"⚡ **Power-heavy lineup** ({power_count} power hitters) — keep the ball down, avoid center-cut fastballs.")
        elif power_count <= 1:
            insights.append("Lacks power — pitchers can pitch to contact and let defense work.")
        if speed_count >= 3:
            insights.append(f"🏃 **Speed threat** ({speed_count} speed players) — quick deliveries, catcher awareness critical.")
        if patient_count >= 4:
            insights.append(f"👁️ **Disciplined lineup** ({patient_count} patient hitters) — must throw strikes early, avoid falling behind.")
        elif patient_count <= 1:
            insights.append("Aggressive lineup — expand the zone with offspeed, use putaway pitches.")

        if "Chase%" in lineup.columns:
            avg_chase = lineup["Chase%"].mean()
            if pd.notna(avg_chase):
                if avg_chase > 28:
                    insights.append(f"High team chase rate ({avg_chase:.1f}%) — work off the plate aggressively.")
                elif avg_chase < 20:
                    insights.append(f"Low team chase rate ({avg_chase:.1f}%) — must compete in the zone.")

        for ins in insights:
            st.markdown(ins)
    else:
        st.info("Insufficient hitting data for lineup analysis.")

    # ── Feature #10: Pitching Staff Depth Chart ──
    section_header("Pitching Staff Depth Chart")
    if not p_trad.empty:
        p_rate = _tm_team(tm["pitching"]["rate"], team)
        p_mov = _tm_team(tm["pitching"]["movement"], team)
        p_pit_rates = _tm_team(tm["pitching"]["pitch_rates"], team)
        p_exit_data = _tm_team(tm["pitching"]["exit"], team)
        p_bids_team = _tm_team(tm["pitching"]["bids"], team)

        staff = p_trad[["playerFullName", "G", "GS", "IP", "ERA", "FIP", "WHIP", "K/9", "BB/9"]].copy()
        staff = staff[staff["IP"] >= 1].copy()

        # Merge velocity from movement
        if not p_mov.empty and "Vel" in p_mov.columns:
            vel_cols = ["playerFullName"]
            for c in ["Vel", "MxVel"]:
                if c in p_mov.columns:
                    vel_cols.append(c)
            staff = staff.merge(p_mov[vel_cols], on="playerFullName", how="left")

        # Merge SwStrk% from pitch rates
        if not p_pit_rates.empty and "SwStrk%" in p_pit_rates.columns:
            staff = staff.merge(p_pit_rates[["playerFullName", "SwStrk%"]], on="playerFullName", how="left")

        # Merge ExitVel against
        if not p_exit_data.empty and "ExitVel" in p_exit_data.columns:
            staff = staff.merge(
                p_exit_data[["playerFullName", "ExitVel"]].rename(columns={"ExitVel": "EV Against"}),
                on="playerFullName", how="left"
            )

        # Classify starters vs relievers
        def _classify_pitcher(row):
            gs = row.get("GS", 0) or 0
            g = row.get("G", 0) or 0
            ip = row.get("IP", 0) or 0
            if gs >= 3 or (gs > 0 and gs / max(g, 1) > 0.5):
                return "Starter"
            elif ip >= 5 or g >= 5:
                return "Reliever"
            else:
                return "Spot/Mop-up"

        staff["Role"] = staff.apply(_classify_pitcher, axis=1)

        # Separate and display starters
        starters = staff[staff["Role"] == "Starter"].sort_values("IP", ascending=False)
        relievers = staff[staff["Role"].isin(["Reliever", "Spot/Mop-up"])].sort_values("IP", ascending=False)

        col_s, col_r = st.columns(2)

        def _format_staff_table(df, label):
            display = df.copy()
            cols_show = ["playerFullName", "G", "GS", "IP", "ERA", "FIP", "WHIP", "K/9", "BB/9"]
            col_names = {"playerFullName": "Player", "G": "G", "GS": "GS", "IP": "IP",
                         "ERA": "ERA", "FIP": "FIP", "WHIP": "WHIP", "K/9": "K/9", "BB/9": "BB/9"}
            if "Vel" in display.columns:
                cols_show.append("Vel")
                col_names["Vel"] = "Velo"
            if "SwStrk%" in display.columns:
                cols_show.append("SwStrk%")
                col_names["SwStrk%"] = "SwStr%"
            if "EV Against" in display.columns:
                cols_show.append("EV Against")
                col_names["EV Against"] = "EV Agn"

            avail = [c for c in cols_show if c in display.columns]
            display = display[avail].rename(columns=col_names)
            for c in ["ERA", "FIP", "WHIP"]:
                if c in display.columns:
                    display[c] = display[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            for c in ["K/9", "BB/9", "Velo", "SwStr%", "EV Agn"]:
                if c in display.columns:
                    display[c] = display[c].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
            for c in ["G", "GS"]:
                if c in display.columns:
                    display[c] = display[c].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
            if "IP" in display.columns:
                display["IP"] = display["IP"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
            return display

        with col_s:
            st.markdown(f"**Starters ({len(starters)})**")
            if not starters.empty:
                st.dataframe(_format_staff_table(starters, "Starters"), use_container_width=True, hide_index=True)
            else:
                st.caption("No identified starters.")

        with col_r:
            st.markdown(f"**Bullpen ({len(relievers)})**")
            if not relievers.empty:
                st.dataframe(_format_staff_table(relievers, "Relievers"), use_container_width=True, hide_index=True)
            else:
                st.caption("No identified relievers.")

        # Staff-level insights
        staff_insights = []
        if not starters.empty:
            avg_era = starters["ERA"].mean()
            avg_vel = starters["Vel"].mean() if "Vel" in starters.columns else np.nan
            if pd.notna(avg_era):
                staff_insights.append(f"Rotation ERA: **{avg_era:.2f}**")
            if pd.notna(avg_vel):
                staff_insights.append(f"Rotation avg velo: **{avg_vel:.1f}** mph")
            top_sp = starters.iloc[0]
            staff_insights.append(
                f"Ace: **{top_sp['playerFullName']}** ({top_sp['IP']:.1f} IP, {top_sp['ERA']:.2f} ERA)"
            )
        if not relievers.empty:
            avg_rel_era = relievers["ERA"].mean()
            if pd.notna(avg_rel_era):
                staff_insights.append(f"Bullpen ERA: **{avg_rel_era:.2f}**")
            if "SwStrk%" in relievers.columns:
                best_sw = relievers.loc[relievers["SwStrk%"].idxmax()] if relievers["SwStrk%"].notna().any() else None
                if best_sw is not None and pd.notna(best_sw.get("SwStrk%")):
                    staff_insights.append(
                        f"Best swing & miss: **{best_sw['playerFullName']}** ({best_sw['SwStrk%']:.1f}% SwStr)")

        if staff_insights:
            st.markdown(" | ".join(staff_insights))
    else:
        st.info("No pitching data available for staff depth chart.")

    # ── Feature #11: Team Tendencies Dashboard ──
    section_header("Team Tendencies")

    # Load additional team-level data
    h_htypes_team = _tm_team(tm["hitting"]["hit_types"], team)
    h_hlocs_team = _tm_team(tm["hitting"]["hit_locations"], team)
    h_prates_team = _tm_team(tm["hitting"]["pitch_rates"], team)
    h_exit_team = _tm_team(tm["hitting"]["exit"], team)
    h_sb_team = _tm_team(tm["hitting"]["stolen_bases"], team)
    h_speed_team = _tm_team(tm["hitting"]["speed"], team)
    p_htypes_team = _tm_team(tm["pitching"]["hit_types"], team)
    p_prates_team = _tm_team(tm["pitching"]["pitch_rates"], team)
    p_ptypes_team = _tm_team(tm["pitching"]["pitch_types"], team)

    tend_col1, tend_col2 = st.columns(2)

    # ── Offensive Tendencies ──
    with tend_col1:
        st.markdown("**Offensive Tendencies**")

        # Batted ball profile (team averages)
        if not h_htypes_team.empty:
            bb_data = []
            for lbl, col in [("GB%", "Ground%"), ("FB%", "Fly%"), ("LD%", "Line%"), ("Popup%", "Popup%")]:
                if col in h_htypes_team.columns:
                    avg = h_htypes_team[col].mean()
                    if pd.notna(avg):
                        bb_data.append({"Type": lbl, "Team Avg": f"{avg:.1f}%"})
            if bb_data:
                st.dataframe(pd.DataFrame(bb_data), use_container_width=True, hide_index=True)

        # Spray direction (team averages)
        if not h_hlocs_team.empty:
            spray_data = []
            for lbl, col in [("Pull%", "HPull%"), ("Center%", "HCtr%"), ("Oppo%", "HOppFld%")]:
                if col in h_hlocs_team.columns:
                    avg = h_hlocs_team[col].mean()
                    if pd.notna(avg):
                        spray_data.append({"Direction": lbl, "Team Avg": f"{avg:.1f}%"})
            if spray_data:
                st.dataframe(pd.DataFrame(spray_data), use_container_width=True, hide_index=True)

        # Plate discipline (team averages)
        if not h_prates_team.empty:
            disc_data = []
            for lbl, col in [("Swing%", "Swing%"), ("Contact%", "Contact%"), ("Chase%", "Chase%"),
                              ("SwStrk%", "SwStrk%"), ("InZone Swing%", "Z-Swing%")]:
                if col in h_prates_team.columns:
                    avg = h_prates_team[col].mean()
                    if pd.notna(avg):
                        disc_data.append({"Metric": lbl, "Team Avg": f"{avg:.1f}%"})
            if disc_data:
                st.dataframe(pd.DataFrame(disc_data), use_container_width=True, hide_index=True)

        # Exit velocity / quality of contact
        if not h_exit_team.empty:
            qc_data = []
            for lbl, col in [("Avg Exit Velo", "ExitVel"), ("Barrel%", "Barrel%"),
                              ("Hard Hit%", "HardHit%"), ("Sweet Spot%", "SweetSpot%")]:
                if col in h_exit_team.columns:
                    avg = h_exit_team[col].mean()
                    if pd.notna(avg):
                        fmt = f"{avg:.1f}" if "%" not in lbl else f"{avg:.1f}%"
                        if lbl == "Avg Exit Velo":
                            fmt = f"{avg:.1f} mph"
                        qc_data.append({"Metric": lbl, "Team Avg": fmt})
            if qc_data:
                st.dataframe(pd.DataFrame(qc_data), use_container_width=True, hide_index=True)

        # Baserunning tendencies
        sb_insights = []
        if not h_sb_team.empty:
            for col in ["SB2%", "SB3%"]:
                if col in h_sb_team.columns:
                    avg = h_sb_team[col].mean()
                    if pd.notna(avg):
                        sb_insights.append(f"{col}: {avg:.1f}%")
        if not h_speed_team.empty and "SpeedScore" in h_speed_team.columns:
            avg_spd = h_speed_team["SpeedScore"].mean()
            if pd.notna(avg_spd):
                sb_insights.append(f"Avg Speed Score: {avg_spd:.1f}")
        if sb_insights:
            st.caption("🏃 Baserunning: " + " | ".join(sb_insights))

    # ── Pitching Staff Tendencies ──
    with tend_col2:
        st.markdown("**Pitching Staff Tendencies**")

        # Pitch mix (team average)
        if not p_ptypes_team.empty:
            pitch_mix = []
            for lbl, col in [("Fastball", "Fastball%"), ("Sinker", "Sinker%"), ("Cutter", "Cutter%"),
                              ("Slider", "Slider%"), ("Curveball", "Curveball%"), ("Changeup", "Changeup%"),
                              ("Sweeper", "Sweeper%"), ("Splitter", "Splitter%")]:
                if col in p_ptypes_team.columns:
                    avg = p_ptypes_team[col].mean()
                    if pd.notna(avg) and avg > 0.5:
                        pitch_mix.append({"Pitch": lbl, "Usage%": f"{avg:.1f}%"})
            if pitch_mix:
                pitch_mix_df = pd.DataFrame(pitch_mix).sort_values("Usage%", ascending=False, key=lambda s: s.str.rstrip("%").astype(float))
                st.dataframe(pitch_mix_df, use_container_width=True, hide_index=True)

        # Batted ball types allowed (pitching staff)
        if not p_htypes_team.empty:
            pbb_data = []
            for lbl, col in [("GB%", "Ground%"), ("FB%", "Fly%"), ("LD%", "Line%"), ("Popup%", "Popup%")]:
                if col in p_htypes_team.columns:
                    avg = p_htypes_team[col].mean()
                    if pd.notna(avg):
                        pbb_data.append({"Type": lbl, "Staff Avg": f"{avg:.1f}%"})
            if pbb_data:
                st.dataframe(pd.DataFrame(pbb_data), use_container_width=True, hide_index=True)

        # Staff pitch discipline induced
        if not p_prates_team.empty:
            pdisc_data = []
            for lbl, col in [("Miss%", "Miss%"), ("Chase%", "Chase%"), ("SwStrk%", "SwStrk%"),
                              ("CompLoc%", "CompLoc%"), ("InZone%", "InZone%")]:
                if col in p_prates_team.columns:
                    avg = p_prates_team[col].mean()
                    if pd.notna(avg):
                        pdisc_data.append({"Metric": lbl, "Staff Avg": f"{avg:.1f}%"})
            if pdisc_data:
                st.dataframe(pd.DataFrame(pdisc_data), use_container_width=True, hide_index=True)

    # Team tendency narrative
    tend_narrative = []
    if not h_htypes_team.empty and "Ground%" in h_htypes_team.columns:
        avg_gb = h_htypes_team["Ground%"].mean()
        avg_fb = h_htypes_team["Fly%"].mean() if "Fly%" in h_htypes_team.columns else np.nan
        if pd.notna(avg_gb) and avg_gb > 48:
            tend_narrative.append(f"**Ground-ball hitting team** ({avg_gb:.1f}% GB) — infield defense matters, look for double-play opportunities.")
        elif pd.notna(avg_fb) and avg_fb > 38:
            tend_narrative.append(f"**Fly-ball hitting team** ({avg_fb:.1f}% FB) — outfield positioning critical, HR risk in elevated pitches.")

    if not h_hlocs_team.empty and "HPull%" in h_hlocs_team.columns:
        avg_pull = h_hlocs_team["HPull%"].mean()
        avg_oppo = h_hlocs_team["HOppFld%"].mean() if "HOppFld%" in h_hlocs_team.columns else np.nan
        if pd.notna(avg_pull) and avg_pull > 45:
            tend_narrative.append(f"**Pull-heavy offense** ({avg_pull:.1f}% Pull) — shift infield, work away.")
        elif pd.notna(avg_oppo) and avg_oppo > 28:
            tend_narrative.append(f"**Good opposite-field approach** ({avg_oppo:.1f}% Oppo) — this team uses the whole field.")

    if not h_prates_team.empty and "Chase%" in h_prates_team.columns:
        avg_chase = h_prates_team["Chase%"].mean()
        avg_contact = h_prates_team["Contact%"].mean() if "Contact%" in h_prates_team.columns else np.nan
        if pd.notna(avg_chase) and avg_chase > 28:
            tend_narrative.append(f"**Chase-prone lineup** ({avg_chase:.1f}% Chase) — expand zone aggressively with breaking balls.")
        elif pd.notna(avg_chase) and avg_chase < 20:
            tend_narrative.append(f"**Extremely selective lineup** ({avg_chase:.1f}% Chase) — compete in the zone, don't waste pitches.")
        if pd.notna(avg_contact) and avg_contact > 82:
            tend_narrative.append(f"**High-contact team** ({avg_contact:.1f}% Contact) — need swing-and-miss stuff or defensive excellence.")

    if not p_prates_team.empty and "SwStrk%" in p_prates_team.columns:
        avg_swstrk = p_prates_team["SwStrk%"].mean()
        if pd.notna(avg_swstrk):
            if avg_swstrk > 12:
                tend_narrative.append(f"**High-strikeout staff** ({avg_swstrk:.1f}% SwStr) — two-strike approach critical, protect the plate.")
            elif avg_swstrk < 8:
                tend_narrative.append(f"**Low swing-and-miss staff** ({avg_swstrk:.1f}% SwStr) — hittable, be aggressive early in counts.")

    if tend_narrative:
        st.markdown("---")
        st.markdown("**Key Tendencies:**")
        for t in tend_narrative:
            st.markdown(f"- {t}")


def _render_count_splits(count_splits, hitter, team):
    """Render count-specific analysis sections using TrueMedia count-filtered API data."""
    if not count_splits:
        return

    # Helper to get a single player row from a count-split DataFrame
    def _player_row(key):
        df = count_splits.get(key, pd.DataFrame())
        if df.empty or "playerFullName" not in df.columns:
            return pd.Series(dtype=float)
        match = df[df["playerFullName"] == hitter]
        if match.empty:
            return pd.Series(dtype=float)
        return match.iloc[0]

    fp = _player_row("first_pitch")
    twok = _player_row("two_strike")
    ahead = _player_row("ahead")
    behind = _player_row("behind")
    two_zero = _player_row("two_zero")
    two_one = _player_row("two_one")
    three_one = _player_row("three_one")

    # Check if we have any data at all
    has_data = any(not s.empty for s in [fp, twok, ahead, behind, two_zero, two_one, three_one])
    if not has_data:
        return

    section_header("Count Analysis")
    st.caption("Performance by count situation — plan your pitch sequencing")

    # ── First Pitch vs Two-Strike comparison ──
    if not fp.empty or not twok.empty:
        c1, c2 = st.columns(2)
        with c1:
            if not fp.empty:
                st.markdown("**First Pitch (0-0)**")
                fp_pa = fp.get("PA", fp.get("AB", 0))
                fp_metrics = []
                for lbl, col, fmt in [
                    ("AVG", "AVG", ".3f"), ("SLG", "SLG", ".3f"), ("wOBA", "WOBA", ".3f"),
                    ("Swing%", "Swing%", ".1f"), ("Whiff%", "SwStrk%", ".1f"),
                    ("Chase%", "Chase%", ".1f"), ("Exit Velo", "ExitVel", ".1f"),
                ]:
                    v = fp.get(col)
                    if v is None:
                        v = fp.get(col.replace("WOBA", "wOBA"))
                    if v is not None and not pd.isna(v):
                        fp_metrics.append({"Metric": lbl, "Value": f"{v:{fmt}}"})
                if fp_metrics:
                    if not pd.isna(fp_pa) and fp_pa > 0:
                        st.caption(f"PA: {int(fp_pa)}")
                    st.dataframe(pd.DataFrame(fp_metrics), use_container_width=True, hide_index=True)
                    # Narrative
                    fp_swing = fp.get("Swing%")
                    if fp_swing is not None and not pd.isna(fp_swing):
                        if fp_swing >= 45:
                            st.caption("Very aggressive first-pitch hitter — start offspeed or off the plate")
                        elif fp_swing <= 20:
                            st.caption("Takes first pitch — attack the zone with a strike")
        with c2:
            if not twok.empty:
                st.markdown("**Two Strikes (x-2)**")
                twok_pa = twok.get("PA", twok.get("AB", 0))
                twok_metrics = []
                for lbl, col, fmt in [
                    ("AVG", "AVG", ".3f"), ("SLG", "SLG", ".3f"), ("wOBA", "WOBA", ".3f"),
                    ("K%", "K%", ".1f"), ("Whiff%", "SwStrk%", ".1f"),
                    ("Chase%", "Chase%", ".1f"), ("Contact%", "Contact%", ".1f"),
                ]:
                    v = twok.get(col)
                    if v is None:
                        v = twok.get(col.replace("WOBA", "wOBA"))
                    if v is not None and not pd.isna(v):
                        twok_metrics.append({"Metric": lbl, "Value": f"{v:{fmt}}"})
                if twok_metrics:
                    if not pd.isna(twok_pa) and twok_pa > 0:
                        st.caption(f"PA: {int(twok_pa)}")
                    st.dataframe(pd.DataFrame(twok_metrics), use_container_width=True, hide_index=True)
                    # Narrative
                    twok_chase = twok.get("Chase%")
                    twok_whiff = twok.get("SwStrk%")
                    if twok_chase is not None and not pd.isna(twok_chase) and twok_chase >= 35:
                        st.caption(f"High chase rate with 2K ({twok_chase:.1f}%) — expand aggressively")
                    if twok_whiff is not None and not pd.isna(twok_whiff) and twok_whiff >= 35:
                        st.caption(f"Very high whiff rate with 2K ({twok_whiff:.1f}%) — swing-and-miss finisher")

    # ── Ahead vs Behind comparison ──
    if not ahead.empty or not behind.empty:
        c3, c4 = st.columns(2)
        with c3:
            if not ahead.empty:
                st.markdown("**Pitcher Ahead (balls < strikes)**")
                ahead_rows = []
                for lbl, col, fmt in [("AVG", "AVG", ".3f"), ("SLG", "SLG", ".3f"),
                                       ("Swing%", "Swing%", ".1f"), ("Chase%", "Chase%", ".1f")]:
                    v = ahead.get(col)
                    if v is not None and not pd.isna(v):
                        ahead_rows.append({"Metric": lbl, "Value": f"{v:{fmt}}"})
                if ahead_rows:
                    st.dataframe(pd.DataFrame(ahead_rows), use_container_width=True, hide_index=True)
        with c4:
            if not behind.empty:
                st.markdown("**Pitcher Behind (balls > strikes)**")
                behind_rows = []
                for lbl, col, fmt in [("AVG", "AVG", ".3f"), ("SLG", "SLG", ".3f"),
                                       ("Swing%", "Swing%", ".1f"), ("Chase%", "Chase%", ".1f")]:
                    v = behind.get(col)
                    if v is not None and not pd.isna(v):
                        behind_rows.append({"Metric": lbl, "Value": f"{v:{fmt}}"})
                if behind_rows:
                    st.dataframe(pd.DataFrame(behind_rows), use_container_width=True, hide_index=True)

    # ── Steal-Strike Counts: 2-0, 2-1, 3-1 ──
    steal_counts = [
        ("2-0", two_zero), ("2-1", two_one), ("3-1", three_one),
    ]
    steal_data = [(lbl, s) for lbl, s in steal_counts if not s.empty]
    if steal_data:
        st.markdown("**Steal-Strike Counts** — hitter-friendly counts where they look to drive")
        cols = st.columns(len(steal_data))
        for ci, (lbl, s) in enumerate(steal_data):
            with cols[ci]:
                st.markdown(f"**{lbl}**")
                rows = []
                for mlbl, col, fmt in [("AVG", "AVG", ".3f"), ("SLG", "SLG", ".3f"),
                                         ("Swing%", "Swing%", ".1f"), ("Exit Velo", "ExitVel", ".1f"),
                                         ("Barrel%", "Barrel%", ".1f")]:
                    v = s.get(col)
                    if v is not None and not pd.isna(v):
                        rows.append({"Metric": mlbl, "Value": f"{v:{fmt}}"})
                if rows:
                    pa_v = s.get("PA", s.get("AB", 0))
                    if not pd.isna(pa_v) and pa_v > 0:
                        st.caption(f"PA: {int(pa_v)}")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        # Narrative
        if not two_zero.empty:
            slg_20 = two_zero.get("SLG")
            sw_20 = two_zero.get("Swing%")
            if slg_20 is not None and not pd.isna(slg_20) and slg_20 > 0.500:
                st.caption(f"Dangerous on 2-0 ({slg_20:.3f} SLG) — do NOT groove a fastball in this count")
            if sw_20 is not None and not pd.isna(sw_20) and sw_20 < 30:
                st.caption(f"Patient on 2-0 ({sw_20:.1f}% Swing) — can be aggressive with the pitch, likely taking")


def _scouting_hitter_report(tm, team, trackman_data, count_splits=None, league_hitters=None):
    """Their Hitters tab — individual deep-dive scouting report with percentile context."""
    if count_splits is None:
        count_splits = {}
    h_rate = _tm_team(tm["hitting"]["rate"], team)
    if h_rate.empty:
        st.info("No hitting data for this team.")
        return
    # Filter to hitters with >= 50 PA for meaningful sample
    h_cnt = _tm_team(tm["hitting"]["counting"], team)
    if not h_cnt.empty and "PA" in h_cnt.columns:
        qualified = h_cnt[h_cnt["PA"] >= 50]["playerFullName"].unique()
        hitters = sorted([h for h in h_rate["playerFullName"].unique() if h in qualified])
        if not hitters:
            # Fall back to all hitters if none have 50+ PA
            hitters = sorted(h_rate["playerFullName"].unique())
    else:
        hitters = sorted(h_rate["playerFullName"].unique())
    hitter = st.selectbox("Select Hitter", hitters, key="sc_hitter")

    # Get this player's data from all tables
    rate = _tm_player(h_rate, hitter)
    cnt = _tm_player(_tm_team(tm["hitting"]["counting"], team), hitter)
    exit_d = _tm_player(_tm_team(tm["hitting"]["exit"], team), hitter)
    xrate = _tm_player(_tm_team(tm["hitting"]["expected_rate"], team), hitter)
    ht = _tm_player(_tm_team(tm["hitting"]["hit_types"], team), hitter)
    hl = _tm_player(_tm_team(tm["hitting"]["hit_locations"], team), hitter)
    pr = _tm_player(_tm_team(tm["hitting"]["pitch_rates"], team), hitter)
    pt = _tm_player(_tm_team(tm["hitting"]["pitch_types"], team), hitter)
    pl = _tm_player(_tm_team(tm["hitting"]["pitch_locations"], team), hitter)
    spd = _tm_player(_tm_team(tm["hitting"]["speed"], team), hitter)
    sb = _tm_player(_tm_team(tm["hitting"]["stolen_bases"], team), hitter)
    hrs = _tm_player(_tm_team(tm["hitting"]["home_runs"], team), hitter)
    h_pcounts = _tm_player(_tm_team(tm["hitting"]["pitch_counts"], team), hitter)
    h_ptcounts = _tm_player(_tm_team(tm["hitting"]["pitch_type_counts"], team), hitter)
    h_re = _tm_player(_tm_team(tm["hitting"]["run_expectancy"], team), hitter)
    h_swpct = _tm_player(_tm_team(tm["hitting"]["swing_pct"], team), hitter)
    h_swstats = _tm_player(_tm_team(tm["hitting"]["swing_stats"], team), hitter)

    # Roster data for percentile context (NCAA D1 if available, else team)
    def _pick_league(key):
        if isinstance(league_hitters, dict):
            df = league_hitters.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        return tm["hitting"].get(key, pd.DataFrame())

    all_h_rate = _pick_league("rate")
    all_h_exit = _pick_league("exit")
    all_h_pr = _pick_league("pitch_rates")
    all_h_ht = _pick_league("hit_types")
    all_h_hl = _pick_league("hit_locations")
    all_h_spd = _pick_league("speed")

    # Filter league data to qualified hitters (PA >= 50) for meaningful percentile comparisons
    # Without this filter, low-PA players with 0 strikeouts skew K%, BB%, etc. distributions
    MIN_PA_FOR_PCTILE = 50
    if not all_h_rate.empty and "PA" in all_h_rate.columns:
        all_h_rate_qualified = all_h_rate[all_h_rate["PA"] >= MIN_PA_FOR_PCTILE].copy()
    else:
        all_h_rate_qualified = all_h_rate
    if not all_h_pr.empty and "PA" in all_h_rate.columns:
        # pitch_rates doesn't have PA, so filter by matching players from rate
        qualified_names = set(all_h_rate_qualified["playerFullName"].dropna()) if "playerFullName" in all_h_rate_qualified.columns else set()
        if qualified_names and "playerFullName" in all_h_pr.columns:
            all_h_pr_qualified = all_h_pr[all_h_pr["playerFullName"].isin(qualified_names)].copy()
        else:
            all_h_pr_qualified = all_h_pr
    else:
        all_h_pr_qualified = all_h_pr
    all_h_re = _pick_league("run_expectancy")
    all_h_swpct = _pick_league("swing_pct")
    all_h_swstats = _pick_league("swing_stats")
    use_league = isinstance(league_hitters, dict) and isinstance(league_hitters.get("rate"), pd.DataFrame) and not league_hitters["rate"].empty

    # Header
    pos = rate.iloc[0].get("pos", "?") if not rate.empty else "?"
    bats = rate.iloc[0].get("batsHand", "?") if not rate.empty else "?"
    # Prefer TrueMedia hand; fallback to TrueMedia pitch-level if missing
    pitch_df = _prefer_truemedia_pitch_data(trackman_data)
    tm_pitch_df = pd.DataFrame()
    local_pitch_df = pd.DataFrame()
    if trackman_data is not None and not trackman_data.empty:
        if "__source" in trackman_data.columns:
            tm_pitch_df = trackman_data[trackman_data["__source"] == "truemedia"]
            local_pitch_df = trackman_data[trackman_data["__source"] != "truemedia"]
        else:
            local_pitch_df = trackman_data
    if bats in [None, "", "?"] or (isinstance(bats, float) and pd.isna(bats)):
        bats = _infer_hand_from_pitch_df(pitch_df, hitter, role="batter")
    bats_label = _fmt_bats(bats)
    bats_norm = None if bats_label == "S" else bats_label
    # Prefer TrueMedia pitch-level for larger BIP samples, fallback to local Trackman
    b_tm_tm = _match_batter_trackman(tm_pitch_df, hitter) if not tm_pitch_df.empty else pd.DataFrame()
    b_tm_local = _match_batter_trackman(local_pitch_df, hitter) if not local_pitch_df.empty else pd.DataFrame()
    b_tm = b_tm_tm if not b_tm_tm.empty else b_tm_local
    g = _safe_val(cnt, "G", "d")
    pa = _safe_val(cnt, "PA", "d")
    st.markdown(f"### {hitter}")
    st.caption(f"{pos} | Bats: {bats_label} | G: {g} | PA: {pa}")

    # ── Player Narrative ──
    narrative = _hitter_narrative(hitter, rate, exit_d, pr, ht, hl, spd,
                                  all_h_rate, all_h_exit, all_h_pr)
    st.markdown(narrative)

    n_hitters_qualified = len(all_h_rate_qualified)

    # ── Percentile Rankings (Savant-style bars) ──
    section_header("Percentile Rankings")
    if use_league:
        st.caption(f"vs. {n_hitters_qualified:,} qualified NCAA D1 hitters (50+ PA)")
    else:
        st.caption(f"vs. {len(all_h_rate):,} {team} hitters (enable NCAA D1 percentiles above)")
    hitting_metrics = [
        ("OPS", _safe_num(rate, "OPS"), _tm_pctile(rate, "OPS", all_h_rate_qualified), ".3f", True),
        ("wOBA", _safe_num(rate, "WOBA"), _tm_pctile(rate, "WOBA", all_h_rate_qualified), ".3f", True),
        ("BA", _safe_num(rate, "BA"), _tm_pctile(rate, "BA", all_h_rate_qualified), ".3f", True),
        ("SLG", _safe_num(rate, "SLG"), _tm_pctile(rate, "SLG", all_h_rate_qualified), ".3f", True),
        ("ISO", _safe_num(rate, "ISO"), _tm_pctile(rate, "ISO", all_h_rate_qualified), ".3f", True),
        ("Exit Velo", _safe_num(exit_d, "ExitVel"), _tm_pctile(exit_d, "ExitVel", all_h_exit), ".1f", True),
        ("Barrel %", _safe_num(exit_d, "Barrel%"), _tm_pctile(exit_d, "Barrel%", all_h_exit), ".1f", True),
        ("Hard Hit %", _safe_num(exit_d, "Hit95+%"), _tm_pctile(exit_d, "Hit95+%", all_h_exit), ".1f", True),
        ("K %", _safe_num(rate, "K%"), _tm_pctile(rate, "K%", all_h_rate_qualified), ".1f", False),
        ("BB %", _safe_num(rate, "BB%"), _tm_pctile(rate, "BB%", all_h_rate_qualified), ".1f", True),
        ("Chase %", _safe_num(pr, "Chase%"), _tm_pctile(pr, "Chase%", all_h_pr_qualified), ".1f", False),
        ("Whiff %", _safe_num(pr, "SwStrk%"), _tm_pctile(pr, "SwStrk%", all_h_pr_qualified), ".1f", False),
        ("Contact %", _safe_num(pr, "Contact%"), _tm_pctile(pr, "Contact%", all_h_pr_qualified), ".1f", True),
        ("Speed Score", _safe_num(spd, "SpeedScore"), _tm_pctile(spd, "SpeedScore", all_h_spd), ".1f", True),
    ]
    # Filter out metrics with nan values
    hitting_metrics = [(l, v, p, f, h) for l, v, p, f, h in hitting_metrics if not pd.isna(v)]
    render_savant_percentile_section(hitting_metrics)

    # ── Swing Path Profile (Pitch-Level) ──
    sp_source = None
    sp = None
    sp_tm = _compute_swing_path(b_tm_tm) if not b_tm_tm.empty else None
    sp_local = _compute_swing_path(b_tm_local) if not b_tm_local.empty else None
    tm_bip = sp_tm.get("n_inplay", 0) if sp_tm else 0
    local_bip = sp_local.get("n_inplay", 0) if sp_local else 0
    if sp_tm and tm_bip >= 15:
        sp = sp_tm
        sp_source = "TrueMedia"
    elif sp_local and local_bip >= 15:
        sp = sp_local
        sp_source = "Trackman"
    elif sp_tm or sp_local:
        sp = sp_tm if tm_bip >= local_bip else sp_local
        sp_source = "TrueMedia (small sample)" if sp is sp_tm else "Trackman (small sample)"

    if sp:
        section_header("Swing Path Profile (Pitch-Level)")
        st.caption(
            f"Source: {sp_source} | "
            f"{sp.get('n_hard_hit', 0)} hard-hit balls in {sp.get('n_inplay', 0)} balls in play"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Attack Angle", f"{sp['attack_angle']:.1f}°")
            st.caption(sp.get("swing_type", ""))
        with c2:
            st.metric("Path Adjust", f"{sp.get('path_adjust', 0):.1f}°/ft")
            st.caption(f"Median LA: {sp.get('avg_la_all', np.nan):.1f}°" if not pd.isna(sp.get("avg_la_all", np.nan)) else "Median LA: -")
        with c3:
            raw_aa = sp.get("attack_angle_raw", np.nan)
            st.metric("Hard-Hit LA", f"{raw_aa:.1f}°" if not pd.isna(raw_aa) else "-")
            if sp.get("bat_speed_avg") is not None:
                st.caption(f"Bat speed est: {sp.get('bat_speed_avg', 0):.1f} mph")
            if sp.get("depth_label"):
                st.caption(f"Contact depth: {sp.get('depth_label')}")

    # ── Batted Ball & Spray Profile ──
    col_bb, col_sp = st.columns(2)
    with col_bb:
        if not ht.empty:
            section_header("Batted Ball Types")
            bb_metrics = [
                ("GB %", _safe_num(ht, "Ground%"), _tm_pctile(ht, "Ground%", all_h_ht), ".1f", True),
                ("FB %", _safe_num(ht, "Fly%"), _tm_pctile(ht, "Fly%", all_h_ht), ".1f", True),
                ("LD %", _safe_num(ht, "Line%"), _tm_pctile(ht, "Line%", all_h_ht), ".1f", True),
                ("Popup %", _safe_num(ht, "Popup%"), _tm_pctile(ht, "Popup%", all_h_ht), ".1f", True),
            ]
            bb_metrics = [(l, v, p, f, h) for l, v, p, f, h in bb_metrics if not pd.isna(v)]
            render_savant_percentile_section(bb_metrics, legend=("LESS OFTEN", "AVERAGE", "MORE OFTEN"))
    with col_sp:
        # Feature #7: Spray Chart Visualization (5-zone field diagram)
        if not hl.empty:
            section_header("Spray Chart")
            farlft = _safe_num(hl, "HFarLft%")
            lftctr = _safe_num(hl, "HLftCtr%")
            deadctr = _safe_num(hl, "HDeadCtr%")
            rtctr = _safe_num(hl, "HRtCtr%")
            farrt = _safe_num(hl, "HFarRt%")
            zones = [farlft, lftctr, deadctr, rtctr, farrt]
            zone_labels = ["Far Left", "Left-Ctr", "Dead Ctr", "Right-Ctr", "Far Right"]
            if any(not pd.isna(z) for z in zones):
                # Build a polar/fan spray chart (viewed from above, catcher's perspective)
                # Angles: 0° = straight up (center field), negative = left, positive = right
                # Zone order: FarLft, LftCtr, DeadCtr, RtCtr, FarRt
                # Catcher view: FarLft on left, FarRt on right
                angles_mid = [-60, -30, 0, 30, 60]  # degrees from center field
                angle_width = 28
                fig_spray = go.Figure()

                # Draw foul lines and outfield arc for context
                fl_r = 115
                # Left field foul line (from home plate going up-right)
                fl_angle_l = math.radians(45)
                fig_spray.add_trace(go.Scatter(
                    x=[0, fl_r * math.sin(fl_angle_l)], y=[0, fl_r * math.cos(fl_angle_l)],
                    mode="lines", line=dict(color="rgba(0,0,0,0.15)", width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
                # Right field foul line
                fl_angle_r = math.radians(-45)
                fig_spray.add_trace(go.Scatter(
                    x=[0, fl_r * math.sin(fl_angle_r)], y=[0, fl_r * math.cos(fl_angle_r)],
                    mode="lines", line=dict(color="rgba(0,0,0,0.15)", width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
                # Outfield arc
                arc_pts = 30
                arc_x, arc_y = [], []
                for j in range(arc_pts + 1):
                    t = math.radians(-45 + 90 * j / arc_pts)
                    arc_x.append(fl_r * math.sin(t))
                    arc_y.append(fl_r * math.cos(t))
                fig_spray.add_trace(go.Scatter(
                    x=arc_x, y=arc_y, mode="lines",
                    line=dict(color="rgba(0,0,0,0.1)", width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))

                max_val = max((z for z in zones if not pd.isna(z)), default=1)
                for i, (z, lbl, ang) in enumerate(zip(zones, zone_labels, angles_mid)):
                    if pd.isna(z):
                        continue
                    # Radius proportional to percentage (scale to fill chart)
                    r = z / max(max_val, 1) * 105
                    theta0 = math.radians(ang - angle_width / 2)
                    theta1 = math.radians(ang + angle_width / 2)
                    # Build wedge: sin for x, cos for y (baseball coordinates)
                    n_pts = 20
                    path_x = [0]
                    path_y = [0]
                    for j in range(n_pts + 1):
                        t = theta0 + (theta1 - theta0) * j / n_pts
                        path_x.append(r * math.sin(t))
                        path_y.append(r * math.cos(t))
                    path_x.append(0)
                    path_y.append(0)
                    # Color: darker = more hits
                    intensity = min(z / max(max_val, 1), 1.0)
                    r_c = int(220 - intensity * 180)
                    g_c = int(225 - intensity * 185)
                    b_c = int(240 - intensity * 50)
                    color = f"rgba({r_c},{g_c},{b_c},0.85)"
                    fig_spray.add_trace(go.Scatter(
                        x=path_x, y=path_y, fill="toself", fillcolor=color,
                        line=dict(color="white", width=2),
                        name=lbl, text=f"{lbl}: {z:.1f}%",
                        hoverinfo="text", showlegend=False,
                    ))
                    # Label
                    lbl_r = r * 0.55
                    lbl_t = math.radians(ang)
                    fig_spray.add_annotation(
                        x=lbl_r * math.sin(lbl_t), y=lbl_r * math.cos(lbl_t),
                        text=f"<b>{z:.0f}%</b>", showarrow=False,
                        font=dict(size=12, color="#000000"), bgcolor="rgba(255,255,255,0.8)",
                    )

                # Field labels
                fig_spray.add_annotation(x=-85, y=85, text="<b>LF</b>", showarrow=False,
                                         font=dict(size=10, color="rgba(0,0,0,0.3)"))
                fig_spray.add_annotation(x=0, y=118, text="<b>CF</b>", showarrow=False,
                                         font=dict(size=10, color="rgba(0,0,0,0.3)"))
                fig_spray.add_annotation(x=85, y=85, text="<b>RF</b>", showarrow=False,
                                         font=dict(size=10, color="rgba(0,0,0,0.3)"))

                fig_spray.update_layout(
                    **CHART_LAYOUT, height=360,
                    xaxis=dict(visible=False, range=[-130, 130]),
                    yaxis=dict(visible=False, range=[-10, 130], scaleanchor="x"),
                    showlegend=False,
                )
                _add_bats_badge(fig_spray, bats_label)
                st.plotly_chart(fig_spray, use_container_width=True)
                # Pull/Center/Oppo summary
                pull = _safe_num(hl, "HPull%")
                ctr = _safe_num(hl, "HCtr%")
                oppo = _safe_num(hl, "HOppFld%")
                parts = []
                if not pd.isna(pull):
                    parts.append(f"Pull: {pull:.1f}%")
                if not pd.isna(ctr):
                    parts.append(f"Center: {ctr:.1f}%")
                if not pd.isna(oppo):
                    parts.append(f"Oppo: {oppo:.1f}%")
                if parts:
                    st.caption(" | ".join(parts))

    # ── Power Profile ──
    if not hrs.empty:
        hr_val = _safe_num(hrs, "HR")
        if not pd.isna(hr_val) and hr_val > 0:
            section_header("Power Profile")
            all_h_hrs = tm["hitting"]["home_runs"]
            hr_metrics = [
                ("HR", _safe_num(hrs, "HR"), _tm_pctile(hrs, "HR", all_h_hrs), ".0f", True),
                ("HR/FB", _safe_num(hrs, "HR/FB"), _tm_pctile(hrs, "HR/FB", all_h_hrs), ".1f", True),
                ("HR Dist", _safe_num(hrs, "HRDst"), _tm_pctile(hrs, "HRDst", all_h_hrs), ".0f", True),
                ("FB Dist", _safe_num(hrs, "FBDst"), _tm_pctile(hrs, "FBDst", all_h_hrs), ".0f", True),
            ]
            hr_metrics = [(l, v, p, f, h) for l, v, p, f, h in hr_metrics if not pd.isna(v)]
            if hr_metrics:
                render_savant_percentile_section(hr_metrics)
            # HR direction
            hr_pull = _safe_num(hrs, "HRPull")
            hr_ctr = _safe_num(hrs, "HRCtr")
            hr_opp = _safe_num(hrs, "HROpp")
            if not pd.isna(hr_pull):
                st.caption(f"HR Direction: Pull {int(hr_pull)} | Center {int(hr_ctr) if not pd.isna(hr_ctr) else 0} | Oppo {int(hr_opp) if not pd.isna(hr_opp) else 0}")

    # ── Stolen Base Detail ──
    if not sb.empty:
        sb2 = _safe_num(sb, "SB2")
        sb3 = _safe_num(sb, "SB3")
        cs2 = _safe_num(sb, "CS2")
        if not pd.isna(sb2) or not pd.isna(sb3):
            section_header("Stolen Base Breakdown")
            sb_data = []
            sb2_pct = _safe_num(sb, "SB2%")
            sb3_pct = _safe_num(sb, "SB3%")
            if not pd.isna(sb2):
                sb_data.append({"Base": "2nd", "SB": int(sb2), "CS": int(cs2) if not pd.isna(cs2) else 0,
                                "SB%": f"{sb2_pct:.1f}%" if not pd.isna(sb2_pct) else "-"})
            if not pd.isna(sb3):
                cs3 = _safe_num(sb, "CS3")
                sb_data.append({"Base": "3rd", "SB": int(sb3), "CS": int(cs3) if not pd.isna(cs3) else 0,
                                "SB%": f"{sb3_pct:.1f}%" if not pd.isna(sb3_pct) else "-"})
            if sb_data:
                st.dataframe(pd.DataFrame(sb_data), use_container_width=True, hide_index=True)

    # ── 2-Strike Whiff Profile ──
    if not h_swstats.empty:
        has_whiff = any(not pd.isna(_safe_num(h_swstats, c)) for c in
                        ["2K Whiff vs LHP Hard", "2K Whiff vs LHP OS", "2K Whiff vs RHP Hard", "2K Whiff vs RHP OS"])
        if has_whiff:
            section_header("2-Strike Whiff Profile")
            st.caption("Whiff rates in 2-strike counts — the key to putting this hitter away")

            whiff_col1, whiff_col2 = st.columns(2)

            with whiff_col1:
                st.markdown("**vs LHP (2 Strikes)**")
                lhp_data = []
                for lbl, col, color in [
                    ("Hard (FB/Sinker)", "2K Whiff vs LHP Hard", "#d22d49"),
                    ("Offspeed (SL/CB/CH)", "2K Whiff vs LHP OS", "#1dbe3a"),
                ]:
                    v = _safe_num(h_swstats, col)
                    if not pd.isna(v):
                        pct = _tm_pctile(h_swstats, col, all_h_swstats)
                        pct_str = f"{int(pct)}th" if not pd.isna(pct) else "-"
                        lhp_data.append({"Pitch Type": lbl, "Whiff%": f"{v:.1f}%", "D1 %ile": pct_str, "val": v, "color": color})
                if lhp_data:
                    st.dataframe(pd.DataFrame(lhp_data)[["Pitch Type", "Whiff%", "D1 %ile"]],
                                 use_container_width=True, hide_index=True)

            with whiff_col2:
                st.markdown("**vs RHP (2 Strikes)**")
                rhp_data = []
                for lbl, col, color in [
                    ("Hard (FB/Sinker)", "2K Whiff vs RHP Hard", "#d22d49"),
                    ("Offspeed (SL/CB/CH)", "2K Whiff vs RHP OS", "#1dbe3a"),
                ]:
                    v = _safe_num(h_swstats, col)
                    if not pd.isna(v):
                        pct = _tm_pctile(h_swstats, col, all_h_swstats)
                        pct_str = f"{int(pct)}th" if not pd.isna(pct) else "-"
                        rhp_data.append({"Pitch Type": lbl, "Whiff%": f"{v:.1f}%", "D1 %ile": pct_str, "val": v, "color": color})
                if rhp_data:
                    st.dataframe(pd.DataFrame(rhp_data)[["Pitch Type", "Whiff%", "D1 %ile"]],
                                 use_container_width=True, hide_index=True)

            # Combined whiff bar chart
            all_whiff = []
            for lbl, col, color in [
                ("vs LHP Hard", "2K Whiff vs LHP Hard", "#d22d49"),
                ("vs LHP OS", "2K Whiff vs LHP OS", "#e65730"),
                ("vs RHP Hard", "2K Whiff vs RHP Hard", "#3d7dab"),
                ("vs RHP OS", "2K Whiff vs RHP OS", "#14365d"),
            ]:
                v = _safe_num(h_swstats, col)
                if not pd.isna(v):
                    all_whiff.append({"Matchup": lbl, "Whiff%": v, "Color": color})

            if len(all_whiff) >= 2:
                aw_df = pd.DataFrame(all_whiff)
                fig_wh = go.Figure()
                fig_wh.add_trace(go.Bar(
                    x=aw_df["Matchup"], y=aw_df["Whiff%"],
                    marker_color=aw_df["Color"],
                    text=aw_df["Whiff%"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside", textfont=dict(size=11, color="#000000"),
                ))
                fig_wh.update_layout(
                    **CHART_LAYOUT, height=280,
                    yaxis=dict(title="Whiff %", range=[0, max(aw_df["Whiff%"].max() * 1.3, 40)]),
                    xaxis=dict(title=""),
                    showlegend=False,
                )
                _add_bats_badge(fig_wh, bats_label)
                st.plotly_chart(fig_wh, use_container_width=True)

            # Whiff narrative — identify the putaway pitch
            whiff_vals = {}
            for lbl, col in [("LHP Hard", "2K Whiff vs LHP Hard"), ("LHP OS", "2K Whiff vs LHP OS"),
                              ("RHP Hard", "2K Whiff vs RHP Hard"), ("RHP OS", "2K Whiff vs RHP OS")]:
                v = _safe_num(h_swstats, col)
                if not pd.isna(v):
                    whiff_vals[lbl] = v

            if whiff_vals:
                max_whiff = max(whiff_vals, key=whiff_vals.get)
                min_whiff = min(whiff_vals, key=whiff_vals.get)
                max_v = whiff_vals[max_whiff]
                min_v = whiff_vals[min_whiff]

                putaway_notes = []
                if max_v >= 30:
                    putaway_notes.append(
                        f"🔴 **Highest whiff vulnerability: {max_whiff}** ({max_v:.1f}%) — this is the putaway sequence."
                    )
                if min_v <= 12:
                    putaway_notes.append(
                        f"🟢 **Hardest to strike out with: {min_whiff}** ({min_v:.1f}%) — avoid relying on this to finish ABs."
                    )
                # Hard vs offspeed comparison
                rhp_hard = whiff_vals.get("RHP Hard")
                rhp_os = whiff_vals.get("RHP OS")
                if rhp_hard is not None and rhp_os is not None:
                    if rhp_os > rhp_hard + 10:
                        putaway_notes.append(
                            f"⚡ vs RHP: Much more vulnerable to offspeed ({rhp_os:.1f}%) than hard stuff ({rhp_hard:.1f}%) — "
                            f"establish fastball early, finish with breaking ball."
                        )
                    elif rhp_hard > rhp_os + 10:
                        putaway_notes.append(
                            f"⚡ vs RHP: More vulnerable to hard stuff ({rhp_hard:.1f}%) than offspeed ({rhp_os:.1f}%) — "
                            f"elevate fastball for the strikeout."
                        )
                lhp_hard = whiff_vals.get("LHP Hard")
                lhp_os = whiff_vals.get("LHP OS")
                if lhp_hard is not None and lhp_os is not None:
                    if lhp_os > lhp_hard + 10:
                        putaway_notes.append(
                            f"⚡ vs LHP: More vulnerable to offspeed ({lhp_os:.1f}%) than hard ({lhp_hard:.1f}%)."
                        )
                    elif lhp_hard > lhp_os + 10:
                        putaway_notes.append(
                            f"⚡ vs LHP: More vulnerable to hard stuff ({lhp_hard:.1f}%) than offspeed ({lhp_os:.1f}%)."
                        )
                for note in putaway_notes:
                    st.markdown(note)

    # ── Platoon Splits ──
    if not h_swstats.empty:
        woba_lhp = _safe_num(h_swstats, "wOBA LHP")
        woba_rhp = _safe_num(h_swstats, "wOBA RHP")
        if not pd.isna(woba_lhp) or not pd.isna(woba_rhp):
            section_header("Platoon Splits")
            split_col1, split_col2 = st.columns(2)

            # 2K whiff values per side
            whiff_lhp_hard = _safe_num(h_swstats, "2K Whiff vs LHP Hard")
            whiff_lhp_os = _safe_num(h_swstats, "2K Whiff vs LHP OS")
            whiff_rhp_hard = _safe_num(h_swstats, "2K Whiff vs RHP Hard")
            whiff_rhp_os = _safe_num(h_swstats, "2K Whiff vs RHP OS")

            with split_col1:
                st.markdown("**vs LHP**")
                if not pd.isna(woba_lhp):
                    pct_lhp = _tm_pctile(h_swstats, "wOBA LHP", all_h_swstats)
                    pct_str = f" ({int(pct_lhp)}th %ile)" if not pd.isna(pct_lhp) else ""
                    st.metric("wOBA vs LHP", f"{woba_lhp:.3f}", help=f"D1 percentile{pct_str}")
                whiff_parts_lhp = []
                if not pd.isna(whiff_lhp_hard):
                    whiff_parts_lhp.append(f"Hard whiff **{whiff_lhp_hard:.1f}%**")
                if not pd.isna(whiff_lhp_os):
                    whiff_parts_lhp.append(f"OS whiff **{whiff_lhp_os:.1f}%**")
                if whiff_parts_lhp:
                    st.caption("2K: " + ", ".join(whiff_parts_lhp))

            with split_col2:
                st.markdown("**vs RHP**")
                if not pd.isna(woba_rhp):
                    pct_rhp = _tm_pctile(h_swstats, "wOBA RHP", all_h_swstats)
                    pct_str = f" ({int(pct_rhp)}th %ile)" if not pd.isna(pct_rhp) else ""
                    st.metric("wOBA vs RHP", f"{woba_rhp:.3f}", help=f"D1 percentile{pct_str}")
                whiff_parts_rhp = []
                if not pd.isna(whiff_rhp_hard):
                    whiff_parts_rhp.append(f"Hard whiff **{whiff_rhp_hard:.1f}%**")
                if not pd.isna(whiff_rhp_os):
                    whiff_parts_rhp.append(f"OS whiff **{whiff_rhp_os:.1f}%**")
                if whiff_parts_rhp:
                    st.caption("2K: " + ", ".join(whiff_parts_rhp))

            # Actionable narrative
            narratives = []
            if not pd.isna(woba_lhp) and not pd.isna(woba_rhp):
                diff = woba_lhp - woba_rhp
                if abs(diff) >= 0.030:
                    better_side = "LHP" if diff > 0 else "RHP"
                    worse_side = "RHP" if diff > 0 else "LHP"
                    narratives.append(
                        f"⚠️ Significantly better vs {better_side} "
                        f"(.{int(max(woba_lhp, woba_rhp)*1000):03d} vs .{int(min(woba_lhp, woba_rhp)*1000):03d}). "
                        f"Opponents should use {worse_side} arms when possible."
                    )
                elif abs(diff) < 0.020:
                    narratives.append("✅ No significant platoon split — equally effective from both sides.")

            # Whiff vulnerability narratives
            for side, hard_val, os_val, opp_hard, opp_os in [
                ("LHP", whiff_lhp_hard, whiff_lhp_os, whiff_rhp_hard, whiff_rhp_os),
                ("RHP", whiff_rhp_hard, whiff_rhp_os, whiff_lhp_hard, whiff_lhp_os),
            ]:
                if not pd.isna(hard_val) and not pd.isna(opp_hard) and hard_val >= opp_hard + 10:
                    narratives.append(
                        f"⚡ Vulnerable to hard stuff from {side} with 2 strikes ({hard_val:.0f}% whiff vs {opp_hard:.0f}%)."
                    )
                if not pd.isna(os_val) and not pd.isna(opp_os) and os_val >= opp_os + 10:
                    narratives.append(
                        f"⚡ Vulnerable to offspeed from {side} with 2 strikes ({os_val:.0f}% whiff vs {opp_os:.0f}%)."
                    )

            for note in narratives:
                st.caption(note)

    # ══════════════════════════════════════════════════════════════════════════
    # COUNT-SPECIFIC ANALYSIS (from TrueMedia count-filtered API data)
    # ══════════════════════════════════════════════════════════════════════════
    _render_count_splits(count_splits, hitter, team)

    # ── Pitch-Level Overlay (prefer TrueMedia) ──
    pitch_df = _prefer_truemedia_pitch_data(trackman_data)
    src_label = _pitch_source_label(pitch_df)
    _trackman_hitter_overlay(pitch_df, hitter, bats_label, src_label)

    # ── Expected Stats (TrueMedia xrate) ──
    if not xrate.empty:
        _has_x = any(not pd.isna(_safe_num(xrate, c)) for c in ["xAVG", "xSLG", "xWOBA"])
        if _has_x:
            section_header("Expected Stats")
            st.caption("Expected stats based on quality of contact — shows over/underperformance")
            xst_rows = []
            for lbl, actual_col, expected_col in [
                ("AVG", "AVG", "xAVG"), ("SLG", "SLG", "xSLG"), ("wOBA", "wOBA", "xWOBA"),
            ]:
                act = _safe_num(xrate, actual_col)
                if pd.isna(act) and not rate.empty:
                    act = _safe_num(rate, actual_col)
                exp = _safe_num(xrate, expected_col)
                if not pd.isna(exp):
                    row = {"Stat": lbl, "Expected": f"{exp:.3f}"}
                    if not pd.isna(act):
                        delta = act - exp
                        row["Actual"] = f"{act:.3f}"
                        row["Delta"] = f"{'+' if delta >= 0 else ''}{delta:.3f}"
                    xst_rows.append(row)
            if xst_rows:
                st.dataframe(pd.DataFrame(xst_rows), use_container_width=True, hide_index=True)
                # Narrative
                xwoba_act = _safe_num(rate, "WOBA") if not rate.empty else _safe_num(xrate, "wOBA")
                xwoba_exp = _safe_num(xrate, "xWOBA")
                if not pd.isna(xwoba_act) and not pd.isna(xwoba_exp):
                    diff = xwoba_act - xwoba_exp
                    if diff > 0.020:
                        st.caption(f"Overperforming expected wOBA by {diff:.3f} — some regression likely.")
                    elif diff < -0.020:
                        st.caption(f"Underperforming expected wOBA by {abs(diff):.3f} — could be better than stats show.")

    # ── Run Expectancy (TrueMedia h_re) ──
    if not h_re.empty:
        _has_re = any(not pd.isna(_safe_num(h_re, c)) for c in ["RE24", "WPA", "Clutch"])
        if _has_re:
            section_header("Run Expectancy & Clutch")
            re_rows = []
            for lbl, col, fmt in [
                ("RE24", "RE24", ".1f"), ("WPA", "WPA", ".2f"), ("Clutch", "Clutch", ".2f"),
                ("REA", "REA", ".1f"), ("WPA/LI", "WPA/LI", ".2f"),
            ]:
                v = _safe_num(h_re, col)
                if not pd.isna(v):
                    pct = _tm_pctile(h_re, col, all_h_re)
                    pct_str = f"{int(pct)}th" if not pd.isna(pct) else "-"
                    re_rows.append({"Metric": lbl, "Value": f"{v:{fmt}}", "%ile": pct_str})
            if re_rows:
                st.dataframe(pd.DataFrame(re_rows), use_container_width=True, hide_index=True)

    # ── "Where Pitchers Attack" zone heatmap (prefer pitch-level Trackman) ──
    attack_fig = None
    if not pitch_df.empty:
        b_tm_attack = _match_batter_trackman(pitch_df, hitter)
        if not b_tm_attack.empty:
            attack_fig = _attack_zone_heatmap(b_tm_attack, "Where Pitchers Attack This Hitter", bats_norm)
    if attack_fig is not None:
        section_header("Where Pitchers Attack This Hitter")
        st.plotly_chart(attack_fig, use_container_width=True)
    elif not pl.empty:
        _has_zones = any(not pd.isna(_safe_num(pl, c)) for c in ["High%", "VMid%", "Low%", "Inside%", "HMid%", "Outside%"])
        if _has_zones:
            section_header("Where Pitchers Attack This Hitter")
            high = _safe_num(pl, "High%")
            vmid = _safe_num(pl, "VMid%")
            low = _safe_num(pl, "Low%")
            ins = _safe_num(pl, "Inside%")
            hmid = _safe_num(pl, "HMid%")
            out = _safe_num(pl, "Outside%")
            vert = [high, vmid, low]
            horiz = [ins, hmid, out]
            if all(not pd.isna(v) for v in vert) and all(not pd.isna(v) for v in horiz):
                # Orient inside/away to catcher view for LHH (inside on right)
                bats_lbl = _fmt_bats(bats_label)
                x_labels = ["Inside", "Middle", "Outside"]
                if bats_lbl == "L":
                    x_labels = ["Outside", "Middle", "Inside"]
                    horiz = horiz[::-1]
                vert_total = sum(vert)
                horiz_total = sum(horiz)
                z_matrix = []
                for v_val in vert:
                    row = []
                    for h_val in horiz:
                        cell = (v_val / max(vert_total, 1)) * (h_val / max(horiz_total, 1)) * 100
                        row.append(round(cell, 1))
                    z_matrix.append(row)
                fig_hz = go.Figure(data=go.Heatmap(
                    z=z_matrix,
                    x=x_labels,
                    y=["High", "Middle", "Low"],
                    colorscale=[[0, "#f0f4f8"], [0.5, "#3d7dab"], [1, "#14365d"]],
                    showscale=False,
                    text=[[f"{v:.1f}%" for v in row] for row in z_matrix],
                    texttemplate="%{text}",
                    textfont=dict(size=14, color="white"),
                ))
                fig_hz.add_shape(type="rect", x0=-0.5, y0=-0.5, x1=2.5, y1=2.5,
                                 line=dict(color="#000000", width=3))
                _cl = {k: v for k, v in CHART_LAYOUT.items() if k != "margin"}
                fig_hz.update_layout(**_cl, height=300,
                                     xaxis=dict(side="bottom"), yaxis=dict(autorange="reversed"),
                                     margin=dict(l=60, r=10, t=10, b=40))
                _add_bats_badge(fig_hz, bats_label)
                st.plotly_chart(fig_hz, use_container_width=True)
                # Narrative
                max_zone_val = max(v for row in z_matrix for v in row)
                if max_zone_val > 15:
                    for ri, rlbl in enumerate(["High", "Middle", "Low"]):
                        for ci, clbl in enumerate(x_labels):
                            if z_matrix[ri][ci] == max_zone_val:
                                st.caption(f"Most frequently attacked zone: **{rlbl} {clbl}** ({max_zone_val:.1f}%)")

    # ══════════════════════════════════════════════════════════
    # SWING HOLE FINDER (Definitive Holes + pitch-level summaries)
    # ══════════════════════════════════════════════════════════
    if not b_tm.empty and len(b_tm) >= 30:
        # For switch hitters, show separate analyses for LHH and RHH
        if bats_label == "S":
            # Infer batting side from PitcherThrows (switch hits opposite)
            # or use BatterSide if available
            b_tm_work = b_tm.copy()
            if "BatterSide" in b_tm_work.columns:
                b_tm_work["_batter_side"] = b_tm_work["BatterSide"].replace(
                    {"Right": "R", "Left": "L", "R": "R", "L": "L"}
                )
            elif "PitcherThrows" in b_tm_work.columns:
                # Switch hitter bats opposite: vs RHP -> L, vs LHP -> R
                b_tm_work["_batter_side"] = b_tm_work["PitcherThrows"].replace(
                    {"Right": "L", "Left": "R", "R": "L", "L": "R"}
                )
            else:
                b_tm_work["_batter_side"] = None

            # Split data by side
            lhh_data = b_tm_work[b_tm_work["_batter_side"] == "L"]
            rhh_data = b_tm_work[b_tm_work["_batter_side"] == "R"]

            n_lhh = len(lhh_data)
            n_rhh = len(rhh_data)

            if n_lhh >= 20 and n_rhh >= 20:
                # Both sides have enough data - show tabbed view
                tab_lhh, tab_rhh = st.tabs([f"🔵 vs RHP (Batting Left) — {n_lhh} pitches",
                                            f"🔴 vs LHP (Batting Right) — {n_rhh} pitches"])
                with tab_lhh:
                    _swing_hole_finder(lhh_data, hitter, "L", bats_norm="L", sp=sp, is_switch=True, pitcher_hand="R")
                with tab_rhh:
                    _swing_hole_finder(rhh_data, hitter, "R", bats_norm="R", sp=sp, is_switch=True, pitcher_hand="L")
            elif n_lhh >= 20:
                # Only LHH has enough data
                st.info(f"Switch hitter — showing LHH side only ({n_lhh} pitches vs RHP, {n_rhh} pitches vs LHP)")
                _swing_hole_finder(lhh_data, hitter, "L", bats_norm="L", sp=sp, is_switch=True, pitcher_hand="R")
            elif n_rhh >= 20:
                # Only RHH has enough data
                st.info(f"Switch hitter — showing RHH side only ({n_rhh} pitches vs LHP, {n_lhh} pitches vs RHP)")
                _swing_hole_finder(rhh_data, hitter, "R", bats_norm="R", sp=sp, is_switch=True, pitcher_hand="L")
            else:
                # Not enough data on either side, show combined
                st.info(f"Switch hitter with limited split data (LHH: {n_lhh}, RHH: {n_rhh}) — showing combined view")
                _swing_hole_finder(b_tm, hitter, bats_label, bats_norm=bats_norm, sp=sp)
        else:
            # ── Non-switch hitter: split by pitcher hand ──
            _MIN_PH = 40  # minimum pitches to show a per-hand tab
            if "PitcherThrows" in b_tm.columns:
                _pt = b_tm["PitcherThrows"].astype(str).str.strip().replace({"Right": "R", "Left": "L"})
                vs_rhp = b_tm[_pt == "R"]
                vs_lhp = b_tm[_pt == "L"]
                n_rhp, n_lhp = len(vs_rhp), len(vs_lhp)

                has_rhp = n_rhp >= _MIN_PH
                has_lhp = n_lhp >= _MIN_PH

                if has_rhp and has_lhp:
                    tab_r, tab_l, tab_all = st.tabs([
                        f"vs RHP — {n_rhp} pitches",
                        f"vs LHP — {n_lhp} pitches",
                        f"All Pitches — {len(b_tm)} pitches",
                    ])
                    with tab_r:
                        _swing_hole_finder(vs_rhp, hitter, bats_label, bats_norm=bats_norm, sp=sp, pitcher_hand="R")
                    with tab_l:
                        _swing_hole_finder(vs_lhp, hitter, bats_label, bats_norm=bats_norm, sp=sp, pitcher_hand="L")
                    with tab_all:
                        _swing_hole_finder(b_tm, hitter, bats_label, bats_norm=bats_norm, sp=sp)
                elif has_rhp:
                    tab_r, tab_all = st.tabs([
                        f"vs RHP — {n_rhp} pitches",
                        f"All Pitches — {len(b_tm)} pitches",
                    ])
                    with tab_r:
                        _swing_hole_finder(vs_rhp, hitter, bats_label, bats_norm=bats_norm, sp=sp, pitcher_hand="R")
                    with tab_all:
                        _swing_hole_finder(b_tm, hitter, bats_label, bats_norm=bats_norm, sp=sp)
                elif has_lhp:
                    tab_l, tab_all = st.tabs([
                        f"vs LHP — {n_lhp} pitches",
                        f"All Pitches — {len(b_tm)} pitches",
                    ])
                    with tab_l:
                        _swing_hole_finder(vs_lhp, hitter, bats_label, bats_norm=bats_norm, sp=sp, pitcher_hand="L")
                    with tab_all:
                        _swing_hole_finder(b_tm, hitter, bats_label, bats_norm=bats_norm, sp=sp)
                else:
                    # Neither side has enough data — show combined
                    _swing_hole_finder(b_tm, hitter, bats_label, bats_norm=bats_norm, sp=sp)
            else:
                _swing_hole_finder(b_tm, hitter, bats_label, bats_norm=bats_norm, sp=sp)


def _assign_total_bases(play_result):
    """Map Trackman PlayResult to total bases for SLG calculation."""
    mapping = {"Single": 1, "Double": 2, "Triple": 3, "HomeRun": 4}
    return mapping.get(play_result, 0)


_AB_RESULTS = {
    "Single", "Double", "Triple", "HomeRun",
    "Out", "Strikeout", "FieldersChoice", "Error",
}


def _zone_heatmap_layout(fig, title, bats=None, show_inside_away=True, is_switch=False, hitter_relative=False):
    """Apply consistent layout to zone heatmaps — large, clean, proper aspect ratio.

    When show_inside_away=True and bats is provided, adds Inside/Away labels
    to clarify the hitter-relative orientation (data is normalized so left=Inside).
    """
    _cl = {k: v for k, v in CHART_LAYOUT.items() if k != "margin"}
    fig.update_layout(
        **_cl, height=420,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14), x=0.5, xanchor="center"),
        xaxis=dict(range=[-2.0, 2.0], showgrid=False, zeroline=False,
                   showticklabels=False, constrain="domain"),
        # Maintain correct strike-zone aspect ratio (feet on X/Y)
        yaxis=dict(range=[0.0, 5.0], showgrid=False, zeroline=False,
                   showticklabels=False, constrain="domain",
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=45, r=45, t=50, b=35),
    )
    # Strike zone
    fig.add_shape(
        type="rect", x0=-ZONE_SIDE, y0=ZONE_HEIGHT_BOT, x1=ZONE_SIDE, y1=ZONE_HEIGHT_TOP,
        line=dict(color="#000000", width=3),
    )
    # Outer boundary
    fig.add_shape(
        type="rect", x0=-1.5, y0=0.5, x1=1.5, y1=4.5,
        line=dict(color="#999999", width=1, dash="dot"),
    )
    bats_label = _fmt_bats(bats) if bats is not None else "?"

    # Add position labels based on view type
    if show_inside_away and bats_label in ["L", "R"]:
        if hitter_relative:
            # Hitter-relative view: Inside always on left (data is flipped for LHH)
            left_label, right_label = "<b>← INSIDE</b>", "<b>AWAY →</b>"
            left_color, right_color = "#c0392b", "#2874a6"
        else:
            # Catcher's view: Inside/Away based on batter hand (data NOT flipped)
            # RHH: inside = 3B side (left), away = 1B side (right)
            # LHH: inside = 1B side (right), away = 3B side (left)
            if bats_label == "R":
                left_label, right_label = "<b>← INSIDE</b>", "<b>AWAY →</b>"
            else:  # LHH
                left_label, right_label = "<b>← AWAY</b>", "<b>INSIDE →</b>"
            left_color, right_color = "#c0392b", "#2874a6"

        fig.add_annotation(
            x=-1.5, y=0.2, xref="x", yref="y",
            text=left_label, showarrow=False,
            font=dict(size=12, color=left_color, family="Arial Black"),
        )
        fig.add_annotation(
            x=1.5, y=0.2, xref="x", yref="y",
            text=right_label, showarrow=False,
            font=dict(size=12, color=right_color, family="Arial Black"),
        )

        # Vertical labels
        fig.add_annotation(
            x=-1.85, y=3.75, xref="x", yref="y",
            text="<b>UP</b>", showarrow=False, textangle=-90,
            font=dict(size=10, color="#666"),
        )
        fig.add_annotation(
            x=-1.85, y=1.25, xref="x", yref="y",
            text="<b>DOWN</b>", showarrow=False, textangle=-90,
            font=dict(size=10, color="#666"),
        )
    # Keep view badge in top-left to avoid overlap with x-axis labels.
    fig.add_annotation(
        x=0.01, y=0.98, xref="paper", yref="paper",
        text="Catcher View",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(size=10, color="#666"),
        bgcolor="rgba(255,255,255,0.70)",
        bordercolor="rgba(0,0,0,0.05)",
        borderwidth=1,
        borderpad=2,
    )

    # Add handedness badge showing hitter's batting side (top-right to avoid label overlap)
    if bats_label != "?":
        badge_text = f"Bats: {bats_label} (Switch)" if is_switch else f"Bats: {bats_label}"
        fig.add_annotation(
            x=0.99, y=0.98, xref="paper", yref="paper",
            text=badge_text,
            showarrow=False, xanchor="right", yanchor="top",
            font=dict(size=11, color="#333", family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            borderpad=3,
        )
    return fig


def _attack_zone_heatmap(pitch_df, title, bats=None, min_pitches=25):
    """3x3 pitch attack heatmap using pitch-level data (inside/outside hitter-relative)."""
    loc = pitch_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if len(loc) < min_pitches:
        return None

    x_edges = np.array([-1.5, -0.5, 0.5, 1.5])
    y_edges = np.array([0.5, 2.0, 3.0, 4.5])
    loc["xbin"] = np.clip(np.digitize(loc["PlateLocSide"], x_edges) - 1, 0, 2)
    loc["ybin"] = np.clip(np.digitize(loc["PlateLocHeight"], y_edges) - 1, 0, 2)

    total = len(loc)
    grid = np.zeros((3, 3), dtype=float)
    for yi in range(3):
        for xi in range(3):
            grid[yi, xi] = (len(loc[(loc["xbin"] == xi) & (loc["ybin"] == yi)]) / max(total, 1)) * 100

    x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(3)]
    y_centers = [(y_edges[i] + y_edges[i + 1]) / 2 for i in range(3)]

    text_grid = []
    for yi in range(3):
        row = []
        for xi in range(3):
            row.append(f"{grid[yi, xi]:.1f}%")
        text_grid.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=x_centers,
        y=y_centers,
        colorscale=[[0, "#f0f4f8"], [0.5, "#3d7dab"], [1, "#14365d"]],
        showscale=False,
        text=text_grid,
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        hovertemplate="Freq: %{text}<extra></extra>",
    ))
    return _zone_heatmap_layout(fig, title, bats=bats, hitter_relative=False)


def _zone_miss_heatmap(pitch_df, title, n_bins=5):
    """Heatmap of miss% (out-of-zone) by plate location."""
    loc = pitch_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if len(loc) < 20:
        return None
    loc["is_miss"] = (
        (loc["PlateLocSide"].abs() > ZONE_SIDE) |
        (loc["PlateLocHeight"] < ZONE_HEIGHT_BOT) |
        (loc["PlateLocHeight"] > ZONE_HEIGHT_TOP)
    ).astype(int)

    x_edges = np.linspace(-1.5, 1.5, n_bins + 1)
    y_edges = np.linspace(0.5, 4.5, n_bins + 1)
    loc["xbin"] = np.clip(np.digitize(loc["PlateLocSide"], x_edges) - 1, 0, n_bins - 1)
    loc["ybin"] = np.clip(np.digitize(loc["PlateLocHeight"], y_edges) - 1, 0, n_bins - 1)

    miss_grid = np.full((n_bins, n_bins), np.nan)
    count_grid = np.zeros((n_bins, n_bins), dtype=int)
    for yi in range(n_bins):
        for xi in range(n_bins):
            cell = loc[(loc["xbin"] == xi) & (loc["ybin"] == yi)]
            count_grid[yi, xi] = len(cell)
            if len(cell) >= 3:
                miss_grid[yi, xi] = cell["is_miss"].mean() * 100

    x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(n_bins)]
    y_centers = [(y_edges[i] + y_edges[i + 1]) / 2 for i in range(n_bins)]

    text_grid = []
    for yi in range(n_bins):
        row = []
        for xi in range(n_bins):
            v = miss_grid[yi, xi]
            row.append("" if pd.isna(v) else f"<b>{v:.0f}%</b>")
        text_grid.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=miss_grid,
        x=x_centers,
        y=y_centers,
        colorscale=[
            [0.0, "#2166ac"],
            [0.20, "#67a9cf"],
            [0.40, "#d1e5f0"],
            [0.50, "#f7f7f7"],
            [0.60, "#fddbc7"],
            [0.80, "#ef8a62"],
            [1.0, "#b2182b"],
        ],
        zmin=0,
        zmax=100,
        showscale=True,
        colorbar=dict(
            title=dict(text="Miss%", font=dict(size=12)),
            len=0.6, thickness=14, x=1.02,
            tickfont=dict(size=10),
        ),
        text=text_grid,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#111111"),
        hovertemplate="Miss%: %{z:.0f}<extra></extra>",
        xgap=2, ygap=2,
    ))

    return _zone_heatmap_layout(fig, title)


def _zone_freq_heatmap(pitch_df, title, n_bins=5, hitter_relative=True, bats=None):
    """Heatmap of pitch frequency by plate location (optionally hitter-relative)."""
    loc = pitch_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if len(loc) < 20:
        return None

    x_col = "PlateLocSide"
    if hitter_relative:
        side = _adjust_side_for_bats(loc, bats=bats)
        if side is not None:
            # Flip for LHH so inside is always on the left (hitter-relative view)
            if _fmt_bats(bats) == "L":
                side = -side
            loc["side_adj"] = side
            x_col = "side_adj"

    x_edges = np.linspace(-1.5, 1.5, n_bins + 1)
    y_edges = np.linspace(0.5, 4.5, n_bins + 1)
    loc["xbin"] = np.clip(np.digitize(loc[x_col], x_edges) - 1, 0, n_bins - 1)
    loc["ybin"] = np.clip(np.digitize(loc["PlateLocHeight"], y_edges) - 1, 0, n_bins - 1)

    freq_grid = np.full((n_bins, n_bins), np.nan)
    total = len(loc)
    for yi in range(n_bins):
        for xi in range(n_bins):
            cell = loc[(loc["xbin"] == xi) & (loc["ybin"] == yi)]
            if len(cell) >= 3:
                freq_grid[yi, xi] = len(cell) / max(total, 1) * 100

    x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(n_bins)]
    y_centers = [(y_edges[i] + y_edges[i + 1]) / 2 for i in range(n_bins)]

    text_grid = []
    for yi in range(n_bins):
        row = []
        for xi in range(n_bins):
            v = freq_grid[yi, xi]
            row.append("" if pd.isna(v) else f"<b>{v:.0f}%</b>")
        text_grid.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=freq_grid,
        x=x_centers,
        y=y_centers,
        colorscale=[
            [0.0, "#f0f4f8"],
            [0.35, "#cfe0f2"],
            [0.65, "#6baed6"],
            [1.0, "#2171b5"],
        ],
        zmin=0,
        zmax=np.nanmax(freq_grid) if np.nanmax(freq_grid) > 0 else 10,
        showscale=True,
        colorbar=dict(
            title=dict(text="Freq%", font=dict(size=12)),
            len=0.6, thickness=14, x=1.02,
            tickfont=dict(size=10),
        ),
        text=text_grid,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#111111"),
        hovertemplate="Freq%: %{z:.0f}<extra></extra>",
        xgap=2, ygap=2,
    ))

    return _zone_heatmap_layout(fig, title, bats=bats, hitter_relative=hitter_relative)


def _top_zone_summary(pitch_df, bats=None):
    """Return (label, pct, total) for most frequent 3x3 zone (hitter-relative)."""
    loc = pitch_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if len(loc) < 10:
        return None
    side_adj = _adjust_side_for_bats(loc, bats=bats)
    if side_adj is None:
        side_adj = loc["PlateLocSide"]
    loc["side_adj"] = side_adj

    x_edges = np.array([-1.5, -0.5, 0.5, 1.5])
    y_edges = np.array([0.5, 2.0, 3.0, 4.5])
    loc["xbin"] = np.clip(np.digitize(loc["side_adj"], x_edges) - 1, 0, 2)
    loc["ybin"] = np.clip(np.digitize(loc["PlateLocHeight"], y_edges) - 1, 0, 2)

    grid = np.zeros((3, 3), dtype=int)
    for yi in range(3):
        for xi in range(3):
            grid[yi, xi] = len(loc[(loc["xbin"] == xi) & (loc["ybin"] == yi)])
    total = grid.sum()
    if total <= 0:
        return None
    yi, xi = np.unravel_index(np.argmax(grid), grid.shape)
    x_lbl = ["Inside", "Middle", "Outside"][xi]
    y_lbl = ["Low", "Middle", "High"][yi]
    pct = grid[yi, xi] / total * 100
    return f"{y_lbl} {x_lbl}", pct, int(total)


def _zone_slg_heatmap(pitch_df, title, n_bins=5, bats=None):
    """Build a zone SLG heatmap from pitch-level Trackman data.

    Bins pitches into an NxN grid over the plate, computes SLG in each bin
    (total bases / at-bats), and renders a blue-white-red heatmap matching
    the TrueMedia reference style.
    """
    loc = pitch_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if len(loc) < 8:
        return None

    if "PlayResult" not in loc.columns:
        return None

    ab_df = loc[loc["PlayResult"].isin(_AB_RESULTS)]
    if len(ab_df) < 5:
        return None

    ab_df = ab_df.copy()
    ab_df["TB"] = ab_df["PlayResult"].apply(_assign_total_bases)

    x_edges = np.linspace(-1.5, 1.5, n_bins + 1)
    y_edges = np.linspace(0.5, 4.5, n_bins + 1)
    x_col = "PlateLocSide"
    ab_df["xbin"] = np.clip(np.digitize(ab_df[x_col], x_edges) - 1, 0, n_bins - 1)
    ab_df["ybin"] = np.clip(np.digitize(ab_df["PlateLocHeight"], y_edges) - 1, 0, n_bins - 1)

    slg_grid = np.full((n_bins, n_bins), np.nan)
    count_grid = np.zeros((n_bins, n_bins), dtype=int)
    for yi in range(n_bins):
        for xi in range(n_bins):
            cell = ab_df[(ab_df["xbin"] == xi) & (ab_df["ybin"] == yi)]
            count_grid[yi, xi] = len(cell)
            if len(cell) >= 2:
                slg_grid[yi, xi] = cell["TB"].sum() / len(cell)

    x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(n_bins)]
    y_centers = [(y_edges[i] + y_edges[i + 1]) / 2 for i in range(n_bins)]

    slg_grid_clip = np.clip(slg_grid, 0, 1.0)

    fig = go.Figure(data=go.Heatmap(
        z=slg_grid_clip,
        x=x_centers,
        y=y_centers,
        colorscale=[
            [0.0, "#1a3399"],
            [0.15, "#4477cc"],
            [0.30, "#99bbee"],
            [0.45, "#dde8f4"],
            [0.50, "#f5f5f5"],
            [0.55, "#f4ddcc"],
            [0.70, "#ee9966"],
            [0.85, "#cc4422"],
            [1.0, "#991100"],
        ],
        zmin=0.0,
        zmax=1.0,
        showscale=True,
        colorbar=dict(
            title=dict(text="SLG", font=dict(size=12)),
            len=0.6, thickness=14, x=1.02,
            tickvals=[0.2, 0.55, 0.9],
            ticktext=[".200", ".550", ".900"],
            tickfont=dict(size=10),
        ),
        hovertemplate="SLG: %{z:.3f}<extra></extra>",
        xgap=2, ygap=2,
    ))

    return _zone_heatmap_layout(fig, title, bats=bats)


def _zone_whiff_heatmap(pitch_df, title, n_bins=5, bats=None):
    """Build a whiff-rate zone heatmap (swings that miss / total swings per zone)."""
    loc = pitch_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if "PitchCall" not in loc.columns:
        return None
    swings = loc[loc["PitchCall"].isin(SWING_CALLS)]
    if len(swings) < 10:
        return None

    swings = swings.copy()
    swings["is_whiff"] = (swings["PitchCall"] == "StrikeSwinging").astype(int)

    x_edges = np.linspace(-1.5, 1.5, n_bins + 1)
    y_edges = np.linspace(0.5, 4.5, n_bins + 1)
    x_col = "PlateLocSide"
    swings["xbin"] = np.clip(np.digitize(swings[x_col], x_edges) - 1, 0, n_bins - 1)
    swings["ybin"] = np.clip(np.digitize(swings["PlateLocHeight"], y_edges) - 1, 0, n_bins - 1)

    whiff_grid = np.full((n_bins, n_bins), np.nan)
    for yi in range(n_bins):
        for xi in range(n_bins):
            cell = swings[(swings["xbin"] == xi) & (swings["ybin"] == yi)]
            if len(cell) >= 3:
                whiff_grid[yi, xi] = cell["is_whiff"].mean() * 100

    x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(n_bins)]
    y_centers = [(y_edges[i] + y_edges[i + 1]) / 2 for i in range(n_bins)]

    text_grid = []
    for yi in range(n_bins):
        row = []
        for xi in range(n_bins):
            v = whiff_grid[yi, xi]
            if pd.isna(v):
                row.append("")
            else:
                row.append(f"<b>{v:.0f}%</b>")
        text_grid.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=whiff_grid,
        x=x_centers,
        y=y_centers,
        colorscale=[
            [0.0, "#2166ac"],
            [0.20, "#67a9cf"],
            [0.40, "#d1e5f0"],
            [0.50, "#f7f7f7"],
            [0.60, "#fddbc7"],
            [0.80, "#ef8a62"],
            [1.0, "#b2182b"],
        ],
        zmin=0,
        zmax=60,
        showscale=True,
        colorbar=dict(
            title=dict(text="Whiff%", font=dict(size=12)),
            len=0.6, thickness=14, x=1.02,
            tickfont=dict(size=10),
        ),
        text=text_grid,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#111111"),
        xgap=2, ygap=2,
    ))

    return _zone_heatmap_layout(fig, title, bats=bats)


def _zone_whiff_density_heatmap(pitch_df, title, n_bins=5, bats=None):
    """Heatmap of whiff counts by zone (swinging strikes)."""
    loc = pitch_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if "PitchCall" not in loc.columns or len(loc) < 15:
        return None
    whiffs = loc[loc["PitchCall"] == "StrikeSwinging"]
    if len(whiffs) < 5:
        return None

    x_edges = np.linspace(-1.5, 1.5, n_bins + 1)
    y_edges = np.linspace(0.5, 4.5, n_bins + 1)
    x_col = "PlateLocSide"

    whiffs["xbin"] = np.clip(np.digitize(whiffs[x_col], x_edges) - 1, 0, n_bins - 1)
    whiffs["ybin"] = np.clip(np.digitize(whiffs["PlateLocHeight"], y_edges) - 1, 0, n_bins - 1)

    count_grid = np.zeros((n_bins, n_bins), dtype=int)
    for yi in range(n_bins):
        for xi in range(n_bins):
            count_grid[yi, xi] = len(whiffs[(whiffs["xbin"] == xi) & (whiffs["ybin"] == yi)])

    x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(n_bins)]
    y_centers = [(y_edges[i] + y_edges[i + 1]) / 2 for i in range(n_bins)]

    text_grid = []
    for yi in range(n_bins):
        row = []
        for xi in range(n_bins):
            v = count_grid[yi, xi]
            row.append("" if v <= 0 else f"<b>{v}</b>")
        text_grid.append(row)

    vmax = max(count_grid.max(), 1)
    fig = go.Figure(data=go.Heatmap(
        z=count_grid,
        x=x_centers,
        y=y_centers,
        colorscale=[
            [0.0, "#f7fbff"],
            [0.30, "#c6dbef"],
            [0.60, "#6baed6"],
            [0.85, "#ef8a62"],
            [1.0, "#b2182b"],
        ],
        zmin=0,
        zmax=vmax,
        showscale=True,
        colorbar=dict(
            title=dict(text="Whiff Cnt", font=dict(size=12)),
            len=0.6, thickness=14, x=1.02,
            tickfont=dict(size=10),
        ),
        text=text_grid,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#111111"),
        xgap=2, ygap=2,
    ))

    return _zone_heatmap_layout(fig, title, bats=bats)


def _zone_swing_pct_heatmap(pitch_df, title, n_bins=5, bats=None):
    """Heatmap of swing% by zone (swings / total pitches in zone)."""
    loc = pitch_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if "PitchCall" not in loc.columns or len(loc) < 20:
        return None
    swings = loc[loc["PitchCall"].isin(SWING_CALLS)]
    if swings.empty:
        return None

    x_edges = np.linspace(-1.5, 1.5, n_bins + 1)
    y_edges = np.linspace(0.5, 4.5, n_bins + 1)
    x_col = "PlateLocSide"

    loc["xbin"] = np.clip(np.digitize(loc[x_col], x_edges) - 1, 0, n_bins - 1)
    loc["ybin"] = np.clip(np.digitize(loc["PlateLocHeight"], y_edges) - 1, 0, n_bins - 1)

    swing_grid = np.full((n_bins, n_bins), np.nan)
    for yi in range(n_bins):
        for xi in range(n_bins):
            cell = loc[(loc["xbin"] == xi) & (loc["ybin"] == yi)]
            if len(cell) >= 5:
                cell_swings = cell[cell["PitchCall"].isin(SWING_CALLS)]
                swing_grid[yi, xi] = len(cell_swings) / len(cell) * 100

    x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(n_bins)]
    y_centers = [(y_edges[i] + y_edges[i + 1]) / 2 for i in range(n_bins)]

    text_grid = []
    for yi in range(n_bins):
        row = []
        for xi in range(n_bins):
            v = swing_grid[yi, xi]
            row.append("" if pd.isna(v) else f"<b>{v:.0f}%</b>")
        text_grid.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=swing_grid,
        x=x_centers,
        y=y_centers,
        colorscale=[
            [0.0, "#2166ac"],
            [0.20, "#67a9cf"],
            [0.40, "#d1e5f0"],
            [0.50, "#f7f7f7"],
            [0.60, "#fddbc7"],
            [0.80, "#ef8a62"],
            [1.0, "#b2182b"],
        ],
        zmin=0,
        zmax=100,
        showscale=True,
        colorbar=dict(
            title=dict(text="Swing%", font=dict(size=12)),
            len=0.6, thickness=14, x=1.02,
            tickfont=dict(size=10),
        ),
        text=text_grid,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#111111"),
        xgap=2, ygap=2,
    ))

    return _zone_heatmap_layout(fig, title, bats=bats)


def _zone_ev_heatmap(pitch_df, title, n_bins=5, bats=None):
    """Heatmap of 90th percentile exit velocity by zone (balls in play)."""
    loc = pitch_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if "ExitSpeed" not in loc.columns or len(loc) < 20:
        return None
    if "PitchCall" in loc.columns:
        loc = loc[loc["PitchCall"] == "InPlay"].copy()
    if loc.empty:
        return None

    x_edges = np.linspace(-1.5, 1.5, n_bins + 1)
    y_edges = np.linspace(0.5, 4.5, n_bins + 1)
    x_col = "PlateLocSide"

    loc["xbin"] = np.clip(np.digitize(loc[x_col], x_edges) - 1, 0, n_bins - 1)
    loc["ybin"] = np.clip(np.digitize(loc["PlateLocHeight"], y_edges) - 1, 0, n_bins - 1)

    ev_grid = np.full((n_bins, n_bins), np.nan)
    for yi in range(n_bins):
        for xi in range(n_bins):
            cell = loc[(loc["xbin"] == xi) & (loc["ybin"] == yi)]
            ev_vals = pd.to_numeric(cell["ExitSpeed"], errors="coerce").dropna()
            if len(ev_vals) >= 4:
                ev_grid[yi, xi] = np.percentile(ev_vals, 90)

    x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(n_bins)]
    y_centers = [(y_edges[i] + y_edges[i + 1]) / 2 for i in range(n_bins)]

    ev_grid_clip = np.clip(ev_grid, 70, 105)
    text_grid = []
    for yi in range(n_bins):
        row = []
        for xi in range(n_bins):
            v = ev_grid[yi, xi]
            row.append("" if pd.isna(v) else f"<b>{v:.0f}</b>")
        text_grid.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=ev_grid_clip,
        x=x_centers,
        y=y_centers,
        colorscale=[
            [0.0, "#1a3399"],
            [0.20, "#5a8bd3"],
            [0.40, "#cfe0f2"],
            [0.60, "#f4ddcc"],
            [0.80, "#ee9966"],
            [1.0, "#991100"],
        ],
        zmin=70,
        zmax=105,
        showscale=True,
        colorbar=dict(
            title=dict(text="Avg EV", font=dict(size=12)),
            len=0.6, thickness=14, x=1.02,
            tickvals=[75, 85, 95, 105],
            tickfont=dict(size=10),
        ),
        text=text_grid,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#111111"),
        xgap=2, ygap=2,
    ))

    return _zone_heatmap_layout(fig, title, bats=bats)


def _swing_hole_finder(b_tm, hitter_name, bats=None, bats_norm=None, sp=None, is_switch=False, pitcher_hand=None):
    """Swing Hole Finder — comprehensive zone-based analysis of a hitter's vulnerabilities.

    Uses pitch-level data (TrueMedia GamePitchesTrackman or local Trackman) to produce:
    1. SLG zone heatmaps by pitcher hand x pitch type
    2. Whiff density heatmap (where the hitter misses most)
    3. Count-specific zone heatmaps (2-strike, first-pitch)
    4. Per-pitch vulnerability summary table
    5. Identified 'holes' with actionable coaching notes
    """
    # Header with batting side info
    bats_display = _fmt_bats(bats) if bats else "?"
    _ph_label = f", vs {'LHP' if pitcher_hand == 'L' else 'RHP'}" if pitcher_hand in ("R", "L") else ""
    if bats_display in ["L", "R"]:
        bats_full = "Left" if bats_display == "L" else "Right"
        suffix = ", Switch" if is_switch else ""
        section_header(f"Swing Hole Finder (Bats {bats_full}{suffix}{_ph_label})")
    else:
        section_header(f"Swing Hole Finder{' (' + _ph_label.lstrip(', ') + ')' if _ph_label else ''}")
    _ph_caption = f" {_ph_label.lstrip(', ')}" if _ph_label else ""
    st.caption(f"Zone-level performance from {len(b_tm)} pitches{_ph_caption}")

    # Delegate to analytics.zone_vulnerability module
    _compute_zone_swing_metrics = _zv_compute_zone_swing_metrics
    _analyze_zone_patterns = lambda zm, df, bats: _zv_analyze_zone_patterns(zm, bats)
    _swing_path_vulnerability = _zv_swing_path_vulnerability

    def _zone_label(x_bin, y_bin, bats):
        bats_label = _fmt_bats(bats)
        if bats_label in ["L", "R"]:
            x_rel = _rel_xbin(x_bin, bats_label)
            x_lbl = ["Inside", "Middle", "Away"][x_rel]
        else:
            x_lbl = ["Left", "Middle", "Right"][x_bin]
        y_lbl = ["Down", "Mid", "Up"][y_bin]
        return f"{y_lbl}-{x_lbl}"

    def _definitive_holes(df, bats, sp, min_zone_n=12):
        """Compute top definitive swing holes using shared compute_hole_scores_3x3."""
        if df.empty:
            return []

        # Normalize first so junk/undefined pitch types are excluded
        df_norm = normalize_pitch_types(df)

        scores = _zv_compute_hole_scores_3x3(df_norm, bats, sp=sp, min_zone_n=min_zone_n)
        if not scores:
            return []

        # Enrich top zones with display-level detail (whiff, slg, path_vuln, n)
        d = df_norm.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
        if d.empty:
            return []
        x_edges = np.array([-1.5, -0.5, 0.5, 1.5])
        y_edges = np.array([0.5, 2.0, 3.0, 4.5])
        d["xbin"] = np.clip(np.digitize(d["PlateLocSide"], x_edges) - 1, 0, 2)
        d["ybin"] = np.clip(np.digitize(d["PlateLocHeight"], y_edges) - 1, 0, 2)
        zone_metrics = _compute_zone_swing_metrics(df_norm, bats)

        out = []
        for (xb, yb), hole_score in scores.items():
            zdf = d[(d["xbin"] == xb) & (d["ybin"] == yb)]
            swings = zdf[zdf["PitchCall"].isin(SWING_CALLS)] if "PitchCall" in zdf.columns else pd.DataFrame()
            whiffs = zdf[zdf["PitchCall"] == "StrikeSwinging"] if "PitchCall" in zdf.columns else pd.DataFrame()
            whiff_pct = len(whiffs) / max(len(swings), 1) * 100 if len(swings) >= 5 else np.nan
            ab_df = zdf[zdf["PlayResult"].isin(_AB_RESULTS)] if "PlayResult" in zdf.columns else pd.DataFrame()
            slg = np.nan
            if len(ab_df) >= 5:
                tb = ab_df["PlayResult"].apply(_assign_total_bases).sum()
                slg = tb / len(ab_df) if len(ab_df) > 0 else np.nan
            path_vuln = _swing_path_vulnerability(zone_metrics, sp, xb, yb, bats=bats)
            out.append({
                "zone": _zone_label(xb, yb, bats),
                "score": hole_score,
                "whiff": whiff_pct,
                "slg": slg,
                "path_vuln": path_vuln,
                "n": len(zdf),
            })

        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:2]

    def _hole_score_heatmap(df, bats, sp, min_zone_n=12, is_switch_inner=False):
        """Build a 3x3 hole-score heatmap with barrel thickness overlay."""
        if df.empty:
            return None, None
        d = normalize_pitch_types(df)
        d = d.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
        if d.empty:
            return None, None
        x_edges = np.array([-1.5, -0.5, 0.5, 1.5])
        y_edges = np.array([0.5, 2.0, 3.0, 4.5])
        d["xbin"] = np.clip(np.digitize(d["PlateLocSide"], x_edges) - 1, 0, 2)
        d["ybin"] = np.clip(np.digitize(d["PlateLocHeight"], y_edges) - 1, 0, 2)

        # Use shared hole-score computation on normalized data (same population as barrel overlay)
        scores = _zv_compute_hole_scores_3x3(d, bats, sp=sp, min_zone_n=min_zone_n)

        hs_grid = np.full((3, 3), np.nan)
        barrel_grid = np.full((3, 3), np.nan)
        n_grid = np.zeros((3, 3), dtype=int)

        for yb in range(3):
            for xb in range(3):
                zdf = d[(d["xbin"] == xb) & (d["ybin"] == yb)]
                n_grid[yb, xb] = len(zdf)

                # Populate hole scores from shared computation
                if (xb, yb) in scores:
                    hs_grid[yb, xb] = scores[(xb, yb)]

                # Barrel thickness (barrel% on balls in play) — display only
                if len(zdf) >= min_zone_n:
                    ip = zdf[zdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed", "Angle"])
                    if len(ip) >= 5:
                        b_pct = is_barrel_mask(ip).mean() * 100
                        barrel_grid[yb, xb] = b_pct

        x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(3)]
        y_centers = [(y_edges[i] + y_edges[i + 1]) / 2 for i in range(3)]

        # Zone labels for context (hitter-relative inside/away)
        zone_labels = [[_zone_label(xi, yi, bats) for xi in range(3)] for yi in range(3)]

        # Find max hole score for highlighting
        max_score = float(np.nanmax(hs_grid)) if not np.all(np.isnan(hs_grid)) else 0

        text_grid = []
        for yi in range(3):
            row = []
            for xi in range(3):
                v = hs_grid[yi, xi]
                if pd.isna(v):
                    row.append("")
                elif v >= max_score - 5 and v >= 50:
                    # Highlight top hole(s) with attack indicator
                    row.append(f"<b>⚾ {v:.0f}</b>")
                else:
                    row.append(f"<b>{v:.0f}</b>")
            text_grid.append(row)

        z_min = 0
        z_max = 100.0

        # Improved colorscale: green (safe) -> yellow -> orange -> red (vulnerable/attack)
        fig = go.Figure(data=go.Heatmap(
            z=hs_grid,
            x=x_centers,
            y=y_centers,
            colorscale=[
                [0.0, "#2ecc71"],   # Green - hitter strength
                [0.30, "#f9e79f"],  # Light yellow
                [0.50, "#f5b041"],  # Orange - neutral
                [0.70, "#e74c3c"],  # Red - vulnerable
                [1.0, "#922b21"],   # Dark red - attack here
            ],
            zmin=z_min,
            zmax=z_max,
            showscale=True,
            colorbar=dict(
                title=dict(text="Attack<br>Score", font=dict(size=11)),
                len=0.5, thickness=16, x=1.02,
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["Safe", "", "Neutral", "", "ATTACK"],
                tickfont=dict(size=9),
            ),
            text=text_grid,
            texttemplate="%{text}",
            textfont=dict(size=15, color="#111111"),
            xgap=3, ygap=3,
            hovertemplate="Zone: %{customdata}<br>Score: %{z:.0f}<extra></extra>",
            customdata=[[zone_labels[yi][xi] for xi in range(3)] for yi in range(3)],
        ))

        # Barrel thickness overlay (marker size)
        bx = []
        by = []
        bs = []
        hover = []
        for yi in range(3):
            for xi in range(3):
                b = barrel_grid[yi, xi]
                if pd.isna(b):
                    continue
                bx.append(x_centers[xi])
                by.append(y_centers[yi])
                size = 4 + (b / 100) * 10
                bs.append(size)
                hover.append(f"Barrel%: {b:.1f}")
        if bx:
            fig.add_trace(go.Scatter(
                x=bx, y=by, mode="markers",
                marker=dict(size=bs, color="rgba(0,0,0,0.35)", line=dict(color="rgba(0,0,0,0.45)", width=1)),
                hovertext=hover, hoverinfo="text",
                showlegend=False,
            ))

        _hole_title = "Hole Score vs " + ("LHP" if pitcher_hand == "L" else "RHP") if pitcher_hand in ("R", "L") else "Definitive Hole Score (All Pitches)"
        fig = _zone_heatmap_layout(fig, _hole_title, bats=bats, is_switch=is_switch_inner)
        fig.update_layout(height=420)
        fig.update_xaxes(range=[-1.5, 1.5])
        fig.update_yaxes(range=[0.5, 4.5], scaleanchor="x", scaleratio=1)
        return fig, n_grid

    required = {"PlateLocSide", "PlateLocHeight", "PitchCall", "TaggedPitchType", "PitcherThrows", "PlayResult"}
    missing = required - set(b_tm.columns)
    if missing:
        st.info(f"Missing columns for swing hole analysis: {missing}")
        with st.expander("Available columns", expanded=False):
            st.caption(f"{list(b_tm.columns)[:30]}")
        return

    # Normalize pitch types once so all downstream analysis excludes junk/undefined
    b_tm_norm = normalize_pitch_types(b_tm)

    # ── Definitive Swing Holes (Outcome + Attack Angle) ──
    sp_use = sp or _compute_swing_path(b_tm_norm)
    bats_norm = bats_norm if bats_norm is not None else bats
    if _fmt_bats(bats_norm) == "S":
        bats_norm = None
    holes_def = _definitive_holes(b_tm_norm, bats_norm, sp_use)

    # ── Compute zone metrics and patterns ──
    zone_metrics = _compute_zone_swing_metrics(b_tm_norm, bats_norm)
    zone_patterns = _analyze_zone_patterns(zone_metrics, b_tm_norm, bats_norm)

    # ── Definitive Hole Score Map (Whiff + SLG + Attack Angle) ──
    fig_hole, n_grid = _hole_score_heatmap(b_tm_norm, bats_norm, sp_use, is_switch_inner=is_switch)

    # Display attack recommendations at top with swing path context
    if holes_def:
        st.markdown("### 🎯 Attack Zones")

        # Show swing path context explaining WHY these zones are targets
        context_parts = []

        if sp_use:
            aa = sp_use.get("attack_angle")
            pa = sp_use.get("path_adjust")
            cd = sp_use.get("contact_depth")
            swing_type = sp_use.get("swing_type", "Unknown")
            depth_label = sp_use.get("depth_label", "Unknown")

            if aa is not None and not pd.isna(aa):
                context_parts.append(f"**Swing:** {swing_type} ({aa:.1f}°)")

                # Path adjust determines if theoretical vulnerability applies
                can_adjust = pa is not None and not pd.isna(pa) and abs(pa) > 1.5
                if can_adjust:
                    context_parts.append(f"**Adjusts to height:** {pa:.1f}°/ft ✓")

        # Add data-driven horizontal pattern (MOST IMPORTANT)
        h_pattern = zone_patterns.get("horizontal_pattern", "balanced")
        if h_pattern != "balanced":
            context_parts.append(f"**Horizontal:** {h_pattern}")

        # Add data-driven vertical pattern
        v_pattern = zone_patterns.get("vertical_pattern", "balanced")
        if v_pattern != "balanced":
            context_parts.append(f"**Vertical:** {v_pattern}")

        # Show hard hit % breakdown if meaningful difference
        in_hh = zone_patterns.get("inside_hh_pct")
        away_hh = zone_patterns.get("away_hh_pct")
        if in_hh is not None and away_hh is not None and not pd.isna(in_hh) and not pd.isna(away_hh):
            if abs(in_hh - away_hh) > 8:
                context_parts.append(f"Hard Hit: Inside {in_hh:.0f}% vs Away {away_hh:.0f}%")

        if context_parts:
            st.caption(" | ".join(context_parts[:3]))
            if len(context_parts) > 3:
                st.caption(" | ".join(context_parts[3:]))

        cols = st.columns(len(holes_def))
        for idx, (h, col) in enumerate(zip(holes_def, cols)):
            wh = f"{h['whiff']:.0f}%" if pd.notna(h["whiff"]) else "-"
            slg = f".{int(h['slg']*1000):03d}" if pd.notna(h["slg"]) and h["slg"] < 1 else (f"{h['slg']:.3f}" if pd.notna(h["slg"]) else "-")
            pv = h.get("path_vuln")
            pv_str = f" | Path: {pv:.0f}" if pv is not None and pd.notna(pv) else ""
            with col:
                priority = "🔴 PRIMARY" if idx == 0 else "🟠 SECONDARY"
                st.markdown(f"**{priority}**")
                st.markdown(f"### {h['zone']}")
                st.caption(f"Whiff: {wh} | SLG: {slg}{pv_str} | n={h['n']}")
    else:
        st.info("Insufficient data to identify definitive attack zones.")

    if fig_hole is not None:
        st.markdown("---")
        st.markdown("**Zone Attack Map**")
        st.caption("Based on actual zone performance: whiff rate, SLG, barrel rate, LA consistency")
        st.caption("🟢 Green = hitter strength (avoid) · 🔴 Red = hitter weakness (attack)")
        st.plotly_chart(fig_hole, use_container_width=True)

        # Horizontal path context (pull/center/oppo)
        if "Direction" in b_tm.columns:
            dir_ip = b_tm[(b_tm["PitchCall"] == "InPlay") & b_tm["Direction"].notna()].copy()
            if len(dir_ip) >= 20:
                is_rhh = _fmt_bats(bats) == "R"
                if is_rhh:
                    pull_mask = dir_ip["Direction"] < -15
                    oppo_mask = dir_ip["Direction"] > 15
                else:
                    pull_mask = dir_ip["Direction"] > 15
                    oppo_mask = dir_ip["Direction"] < -15
                center_mask = ~(pull_mask | oppo_mask)
                total = len(dir_ip)
                pull_pct = pull_mask.sum() / total * 100
                oppo_pct = oppo_mask.sum() / total * 100
                center_pct = center_mask.sum() / total * 100
                if pull_pct - oppo_pct >= 12:
                    horiz_label = "Pull‑lean"
                elif oppo_pct - pull_pct >= 12:
                    horiz_label = "Oppo‑lean"
                else:
                    horiz_label = "Balanced"
                st.caption(
                    f"Horizontal path: **{horiz_label}** "
                    f"(Pull {pull_pct:.0f}% · Center {center_pct:.0f}% · Oppo {oppo_pct:.0f}%)"
                )

    # Filter out undefined/rare pitch types for all remaining pitch-level summaries
    p_use = _filter_pitch_types_global(b_tm, MIN_PITCH_USAGE_PCT)
    if p_use is None or p_use.empty:
        st.info(f"No pitch-level data after removing undefined or <{MIN_PITCH_USAGE_PCT:.0f}% usage pitches.")
        return

    # ── SLG by Pitch Type Heatmaps ──
    if "TaggedPitchType" in p_use.columns:
        pt_counts = p_use["TaggedPitchType"].value_counts()
        pt_keep = pt_counts[pt_counts >= 20].index.tolist()
        if pt_keep:
            st.markdown("**SLG by Pitch Type**")
            st.caption("Slugging by location for each pitch type (catcher view; inside/away labeled by handedness).")
            pt_list = pt_keep[:6]
            for i in range(0, len(pt_list), 3):
                cols = st.columns(3)
                for j, pt_name in enumerate(pt_list[i:i + 3]):
                    pt_df = p_use[p_use["TaggedPitchType"] == pt_name]
                    fig = _zone_slg_heatmap(pt_df, f"{pt_name} SLG", bats=bats_norm)
                    if fig is None:
                        continue
                    with cols[j]:
                        st.plotly_chart(fig, use_container_width=True)

    # ── SLG by Pitch Type vs LHP/RHP ──
    if {"TaggedPitchType", "PitcherThrows"}.issubset(p_use.columns):
        hand_map = p_use["PitcherThrows"].replace({"R": "Right", "L": "Left"}).astype(str)
        for hand_label, hand_val in [("LHP", "Left"), ("RHP", "Right")]:
            hand_df = p_use[hand_map == hand_val]
            if len(hand_df) < 25:
                continue
            pt_counts = hand_df["TaggedPitchType"].value_counts()
            pt_keep = pt_counts[pt_counts >= 15].index.tolist()
            if not pt_keep:
                continue
            st.markdown(f"**SLG vs {hand_label} by Pitch Type**")
            pt_list = pt_keep[:8]
            for i in range(0, len(pt_list), 3):
                cols = st.columns(3)
                for j, pt_name in enumerate(pt_list[i:i + 3]):
                    pt_df = hand_df[hand_df["TaggedPitchType"] == pt_name]
                    fig = _zone_slg_heatmap(pt_df, f"SLG vs {hand_label} ({pt_name})", bats=bats_norm)
                    if fig is None:
                        continue
                    with cols[j]:
                        st.plotly_chart(fig, use_container_width=True)

    # ── Zone Summary Row (Whiff%, Swing%, Avg EV) ──
    st.markdown("**Zone Summary (All Pitches)**")
    cols = st.columns(3)
    with cols[0]:
        fig = _zone_whiff_density_heatmap(p_use, "Whiff Density by Zone", bats=bats_norm)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Not enough whiff data for zone heatmap.")
    with cols[1]:
        fig = _zone_swing_pct_heatmap(p_use, "Swing% by Zone", bats=bats_norm)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Not enough swing data for zone heatmap.")
    with cols[2]:
        fig = _zone_ev_heatmap(p_use, "Avg EV by Zone", bats=bats_norm)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Not enough exit velo data for zone heatmap.")

    # ── Per‑Pitch Swing Decisions ──
    if {"TaggedPitchType", "PitchCall"}.issubset(p_use.columns):
        rows = []
        for pt_name, pt_df in p_use.groupby("TaggedPitchType"):
            n = len(pt_df)
            if n < 8:
                continue
            swings = pt_df[pt_df["PitchCall"].isin(SWING_CALLS)]
            whiffs = pt_df[pt_df["PitchCall"] == "StrikeSwinging"]
            swing_pct = len(swings) / n * 100 if n > 0 else np.nan
            whiff_pct = len(whiffs) / len(swings) * 100 if len(swings) > 0 else np.nan
            ev_vals = pd.to_numeric(pt_df.get("ExitSpeed"), errors="coerce").dropna() if "ExitSpeed" in pt_df.columns else pd.Series(dtype=float)
            avg_ev = ev_vals.mean() if len(ev_vals) > 0 else np.nan
            rows.append({
                "Pitch": pt_name,
                "N": int(n),
                "Swing%": f"{swing_pct:.1f}%" if pd.notna(swing_pct) else "-",
                "Whiff%": f"{whiff_pct:.1f}%" if pd.notna(whiff_pct) else "-",
                "Avg EV": f"{avg_ev:.1f}" if pd.notna(avg_ev) else "-",
            })
        if rows:
            st.markdown("**Per‑Pitch Swing Decisions**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Per‑Pitch SLG & Vulnerability ──
    if {"TaggedPitchType", "PitchCall"}.issubset(p_use.columns):
        rows = []
        for pt_name, pt_df in p_use.groupby("TaggedPitchType"):
            n = len(pt_df)
            if n < 8:
                continue
            swings = pt_df[pt_df["PitchCall"].isin(SWING_CALLS)]
            whiffs = pt_df[pt_df["PitchCall"] == "StrikeSwinging"]
            swing_pct = len(swings) / n * 100 if n > 0 else np.nan
            whiff_pct = len(whiffs) / len(swings) * 100 if len(swings) > 0 else np.nan
            ab_df = pt_df[pt_df["PlayResult"].isin(_AB_RESULTS)] if "PlayResult" in pt_df.columns else pd.DataFrame()
            ab = len(ab_df)
            slg = np.nan
            if ab > 0:
                tb = ab_df["PlayResult"].apply(_assign_total_bases).sum()
                slg = tb / ab
            ev_vals = pd.to_numeric(pt_df.get("ExitSpeed"), errors="coerce").dropna() if "ExitSpeed" in pt_df.columns else pd.Series(dtype=float)
            avg_ev = ev_vals.mean() if len(ev_vals) > 0 else np.nan
            rows.append({
                "Pitch": pt_name,
                "N": int(n),
                "AB": int(ab),
                "SLG": f"{slg:.3f}" if pd.notna(slg) else "-",
                "Whiff%": f"{whiff_pct:.1f}%" if pd.notna(whiff_pct) else "-",
                "Swing%": f"{swing_pct:.1f}%" if pd.notna(swing_pct) else "-",
                "Avg EV": f"{avg_ev:.1f}" if pd.notna(avg_ev) else "-",
            })
        if rows:
            st.markdown("**Per‑Pitch SLG & Vulnerability**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)



def _match_pitcher_trackman(opp_data, tm_full_name):
    """Match a TrueMedia pitcher name to rows in opponent Trackman data.
    Tries: (1) exact Trackman-format, (2) direct TrueMedia name, (3) last-name fallback."""
    if opp_data.empty or "Pitcher" not in opp_data.columns:
        return pd.DataFrame()
    trackman_name = tm_name_to_trackman(tm_full_name)
    exact = opp_data[opp_data["Pitcher"] == trackman_name]
    if not exact.empty:
        return exact
    # Try direct name match (TrueMedia GamePitchesTrackman may use "First Last")
    direct = opp_data[opp_data["Pitcher"] == tm_full_name]
    if not direct.empty:
        return direct
    last_name = tm_full_name.split()[-1] if " " in tm_full_name else tm_full_name
    fallback = opp_data[opp_data["Pitcher"].str.contains(last_name, case=False, na=False)]
    if len(fallback["Pitcher"].unique()) == 1:
        return fallback
    return pd.DataFrame()


def _match_batter_trackman(opp_data, tm_full_name):
    """Match a TrueMedia batter name to rows in opponent Trackman data.
    Tries: (1) exact Trackman-format, (2) direct TrueMedia name, (3) last-name fallback."""
    if opp_data.empty or "Batter" not in opp_data.columns:
        return pd.DataFrame()
    trackman_name = tm_name_to_trackman(tm_full_name)
    exact = opp_data[opp_data["Batter"] == trackman_name]
    if not exact.empty:
        return exact
    # Try direct name match (TrueMedia GamePitchesTrackman may use "First Last")
    direct = opp_data[opp_data["Batter"] == tm_full_name]
    if not direct.empty:
        return direct
    last_name = tm_full_name.split()[-1] if " " in tm_full_name else tm_full_name
    fallback = opp_data[opp_data["Batter"].str.contains(last_name, case=False, na=False)]
    if len(fallback["Batter"].unique()) == 1:
        return fallback
    return pd.DataFrame()


def _trackman_hitter_overlay(data, hitter_name, bats=None, source_label="Trackman"):
    """Show swing heatmap if we have pitch-level data for this hitter."""
    matches = _match_batter_trackman(data, hitter_name)
    if matches.empty or len(matches) < 10:
        return
    if "PitchCall" not in matches.columns or "PlateLocSide" not in matches.columns or "PlateLocHeight" not in matches.columns:
        return
    section_header("Pitch Location Overlay")
    st.caption(f"Pitch-level data from {len(matches)} {source_label} pitches")
    swings = matches[matches["PitchCall"].isin(SWING_CALLS)]
    loc = swings.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if not loc.empty and len(loc) >= 5:
        # Apply handedness adjustment for hitter-relative view (non-switch hitters)
        x_col = "PlateLocSide"
        bats_label = _fmt_bats(bats) if bats else None
        show_inside_away = bats_label in ["L", "R"]
        if show_inside_away:
            side_adj = _adjust_side_for_bats(loc, bats=bats)
            if side_adj is not None:
                loc["side_adj"] = side_adj
                x_col = "side_adj"
        fig = px.density_heatmap(loc, x=x_col, y="PlateLocHeight", nbinsx=12, nbinsy=12,
                                 color_continuous_scale="YlOrRd")
        add_strike_zone(fig)
        fig.update_layout(title="Swing Locations", xaxis=dict(range=[-3, 3], scaleanchor="y"),
                          yaxis=dict(range=[0, 5]),
                          height=400, coloraxis_showscale=False, **CHART_LAYOUT)
        # Add Inside/Away labels for hitter-relative orientation (non-switch only)
        if show_inside_away:
            fig.add_annotation(x=-2.5, y=0.3, text="← INSIDE", showarrow=False,
                               font=dict(size=10, color="#c0392b", family="Arial Black"))
            fig.add_annotation(x=2.5, y=0.3, text="AWAY →", showarrow=False,
                               font=dict(size=10, color="#2874a6", family="Arial Black"))
        _add_bats_badge(fig, bats)
        st.plotly_chart(fig, use_container_width=True)


def _scouting_pitcher_report(tm, team, trackman_data, league_pitchers=None):
    """Their Pitchers tab — comprehensive scouting report built for game-planning."""
    p_trad = _tm_team(tm["pitching"]["traditional"], team)
    if p_trad.empty:
        st.info("No pitching data for this team.")
        return
    # Filter to pitchers with >= 10 IP for meaningful sample
    if "IP" in p_trad.columns:
        qualified = p_trad[p_trad["IP"] >= 10]["playerFullName"].unique()
        pitchers = sorted([p for p in p_trad["playerFullName"].unique() if p in qualified])
        if not pitchers:
            # Fall back to all pitchers if none have 10+ IP
            pitchers = sorted(p_trad["playerFullName"].unique())
    else:
        pitchers = sorted(p_trad["playerFullName"].unique())
    pitcher = st.selectbox("Select Pitcher", pitchers, key="sc_pitcher")

    # ── Load all data for this pitcher ──
    trad = _tm_player(p_trad, pitcher)
    rate = _tm_player(_tm_team(tm["pitching"]["rate"], team), pitcher)
    mov = _tm_player(_tm_team(tm["pitching"]["movement"], team), pitcher)
    pt = _tm_player(_tm_team(tm["pitching"]["pitch_types"], team), pitcher)
    pr = _tm_player(_tm_team(tm["pitching"]["pitch_rates"], team), pitcher)
    exit_d = _tm_player(_tm_team(tm["pitching"]["exit"], team), pitcher)
    xrate = _tm_player(_tm_team(tm["pitching"]["expected_rate"], team), pitcher)
    ht = _tm_player(_tm_team(tm["pitching"]["hit_types"], team), pitcher)
    hl = _tm_player(_tm_team(tm["pitching"]["hit_locations"], team), pitcher)
    ploc = _tm_player(_tm_team(tm["pitching"]["pitch_locations"], team), pitcher)
    p_hr = _tm_player(_tm_team(tm["pitching"]["home_runs"], team), pitcher)
    p_ptcounts = _tm_player(_tm_team(tm["pitching"]["pitch_type_counts"], team), pitcher)
    p_cnt = _tm_player(_tm_team(tm["pitching"]["counting"], team), pitcher)

    # Roster data for percentile context (NCAA D1 if available, else team)
    def _pick_league(key):
        if isinstance(league_pitchers, dict):
            df = league_pitchers.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        return tm["pitching"].get(key, pd.DataFrame())

    all_p_trad = _pick_league("traditional")
    all_p_rate = _pick_league("rate")
    all_p_mov = _pick_league("movement")
    all_p_pr = _pick_league("pitch_rates")
    all_p_exit = _pick_league("exit")
    all_p_ht = _pick_league("hit_types")
    all_p_ploc = _pick_league("pitch_locations")
    all_p_xrate = _pick_league("expected_rate")
    all_p_hr = _pick_league("home_runs")
    use_league = isinstance(league_pitchers, dict) and isinstance(league_pitchers.get("traditional"), pd.DataFrame) and not league_pitchers["traditional"].empty

    # Filter league data to qualified pitchers (IP >= 10) for meaningful percentile comparisons
    MIN_IP_FOR_PCTILE = 10
    if not all_p_trad.empty and "IP" in all_p_trad.columns:
        all_p_trad_qualified = all_p_trad[all_p_trad["IP"] >= MIN_IP_FOR_PCTILE].copy()
        qualified_pitcher_names = set(all_p_trad_qualified["playerFullName"].dropna()) if "playerFullName" in all_p_trad_qualified.columns else set()
    else:
        all_p_trad_qualified = all_p_trad
        qualified_pitcher_names = set()
    # Filter other dataframes to match qualified pitchers
    def _filter_to_qualified(df, names):
        if df.empty or not names or "playerFullName" not in df.columns:
            return df
        return df[df["playerFullName"].isin(names)].copy()
    all_p_rate_qualified = _filter_to_qualified(all_p_rate, qualified_pitcher_names)
    all_p_mov_qualified = _filter_to_qualified(all_p_mov, qualified_pitcher_names)
    all_p_pr_qualified = _filter_to_qualified(all_p_pr, qualified_pitcher_names)
    all_p_exit_qualified = _filter_to_qualified(all_p_exit, qualified_pitcher_names)
    all_p_ht_qualified = _filter_to_qualified(all_p_ht, qualified_pitcher_names)

    # ── Header ──
    throws = trad.iloc[0].get("throwsHand", "?") if not trad.empty else "?"
    # Prefer TrueMedia hand; fallback to TrueMedia pitch-level if missing
    pitch_df = _prefer_truemedia_pitch_data(trackman_data)
    if throws in [None, "", "?"] or (isinstance(throws, float) and pd.isna(throws)):
        throws = _infer_hand_from_pitch_df(pitch_df, pitcher, role="pitcher")
    g = _safe_val(trad, "G", "d")
    gs = _safe_val(trad, "GS", "d")
    w = _safe_val(trad, "W", "d")
    l_val = _safe_val(trad, "L", "d")
    ip = _safe_val(trad, "IP")
    qs = _safe_val(trad, "QS", "d")
    sv = _safe_val(trad, "SV", "d")
    st.markdown(f"### {pitcher}")
    header_parts = [f"Throws: {throws}", f"G: {g}", f"GS: {gs}", f"{w}-{l_val}", f"IP: {ip}"]
    if qs != "-" and int(qs) > 0:
        header_parts.append(f"QS: {qs}")
    if sv != "-" and int(sv) > 0:
        header_parts.append(f"SV: {sv}")
    st.caption(" | ".join(header_parts))

    # ── Pitcher Narrative ──
    narrative = _pitcher_narrative(pitcher, trad, mov, pr, ht, exit_d,
                                   all_p_trad, all_p_mov, all_p_pr)
    st.markdown(narrative)

    n_pitchers_qualified = len(all_p_trad_qualified)

    # ══════════════════════════════════════════════════════════
    # SECTION 1: PERCENTILE RANKINGS
    # ══════════════════════════════════════════════════════════
    section_header("Percentile Rankings")
    if use_league:
        st.caption(f"vs. {n_pitchers_qualified:,} qualified NCAA D1 pitchers (10+ IP)")
    else:
        st.caption(f"vs. {len(all_p_trad):,} {team} pitchers (enable NCAA D1 percentiles above)")
    pitching_metrics = [
        ("ERA", _safe_num(trad, "ERA"), _tm_pctile(trad, "ERA", all_p_trad_qualified), ".2f", False),
        ("FIP", _safe_num(trad, "FIP"), _tm_pctile(trad, "FIP", all_p_trad_qualified), ".2f", False),
        ("xFIP", _safe_num(rate, "xFIP"), _tm_pctile(rate, "xFIP", all_p_rate_qualified), ".2f", False),
        ("WHIP", _safe_num(trad, "WHIP"), _tm_pctile(trad, "WHIP", all_p_trad_qualified), ".2f", False),
        ("K/9", _safe_num(trad, "K/9"), _tm_pctile(trad, "K/9", all_p_trad_qualified), ".1f", True),
        ("BB/9", _safe_num(trad, "BB/9"), _tm_pctile(trad, "BB/9", all_p_trad_qualified), ".1f", False),
        ("K/BB", _safe_num(trad, "K/BB"), _tm_pctile(trad, "K/BB", all_p_trad_qualified), ".2f", True),
        ("Velo", _safe_num(mov, "Vel"), _tm_pctile(mov, "Vel", all_p_mov_qualified), ".1f", True),
        ("Spin", _safe_num(mov, "Spin"), _tm_pctile(mov, "Spin", all_p_mov_qualified), ".0f", True),
        ("Extension", _safe_num(mov, "Extension"), _tm_pctile(mov, "Extension", all_p_mov_qualified), ".1f", True),
        ("Chase %", _safe_num(pr, "Chase%"), _tm_pctile(pr, "Chase%", all_p_pr_qualified), ".1f", True),
        ("SwStrk %", _safe_num(pr, "SwStrk%"), _tm_pctile(pr, "SwStrk%", all_p_pr_qualified), ".1f", True),
        ("EV Against", _safe_num(exit_d, "ExitVel"), _tm_pctile(exit_d, "ExitVel", all_p_exit_qualified), ".1f", False),
        ("Barrel %", _safe_num(exit_d, "Barrel%"), _tm_pctile(exit_d, "Barrel%", all_p_exit_qualified), ".1f", False),
        ("GB %", _safe_num(ht, "Ground%"), _tm_pctile(ht, "Ground%", all_p_ht_qualified), ".1f", True),
        ("wOBA Agn", _safe_num(rate, "wOBA"), _tm_pctile(rate, "wOBA", all_p_rate_qualified), ".3f", False),
        ("LOB %", _safe_num(rate, "LOB%"), _tm_pctile(rate, "LOB%", all_p_rate_qualified), ".1f", True),
        ("HR/9", _safe_num(trad, "HR/9"), _tm_pctile(trad, "HR/9", all_p_trad_qualified), ".2f", False),
    ]
    pitching_metrics = [(l, v, p, f, h) for l, v, p, f, h in pitching_metrics if not pd.isna(v)]
    render_savant_percentile_section(pitching_metrics)

    # ══════════════════════════════════════════════════════════
    # SECTION 2: ARSENAL & STUFF (side by side)
    # ══════════════════════════════════════════════════════════
    if not pt.empty or not mov.empty:
        section_header("Arsenal & Stuff")

        col_ars, col_stuff = st.columns([3, 2])

        with col_ars:
            if not pt.empty:
                pitch_cols = ["4Seam%", "Sink2Seam%", "Cutter%", "Slider%", "Curve%", "Change%", "Split%", "Sweeper%"]
                pitch_labels = ["4-Seam", "Sinker", "Cutter", "Slider", "Curve", "Change", "Splitter", "Sweeper"]
                count_cols = ["4Seam#", "Sink2Seam#", "Cutter#", "Slider#", "Curve#", "Change#", "Split#", "Sweeper#"]
                arsenal_rows = []
                for pct_col, cnt_col, lbl in zip(pitch_cols, count_cols, pitch_labels):
                    pct_v = pt.iloc[0].get(pct_col) if not pt.empty else None
                    cnt_v = p_ptcounts.iloc[0].get(cnt_col) if not p_ptcounts.empty and cnt_col in p_ptcounts.columns else None
                    if pct_v is not None and not pd.isna(pct_v) and pct_v >= MIN_PITCH_USAGE_PCT:
                        arsenal_rows.append({"Pitch": lbl, "Usage": pct_v,
                                             "Count": int(cnt_v) if cnt_v is not None and not pd.isna(cnt_v) else None})
                if arsenal_rows:
                    fig = go.Figure(go.Bar(
                        x=[r["Pitch"] for r in arsenal_rows],
                        y=[r["Usage"] for r in arsenal_rows],
                        marker_color=[PITCH_COLORS.get(r["Pitch"].replace("-", ""), "#888") for r in arsenal_rows],
                        text=[f"{r['Usage']:.1f}%" for r in arsenal_rows], textposition="outside",
                        textfont=dict(size=11, color="#000000"),
                    ))
                    fig.update_layout(**CHART_LAYOUT, height=280, yaxis_title="Usage %", showlegend=False,
                                      yaxis=dict(range=[0, max(r["Usage"] for r in arsenal_rows) * 1.3]))
                    st.plotly_chart(fig, use_container_width=True)

                    # Arsenal narrative
                    primary = arsenal_rows[0] if arsenal_rows else None
                    secondary = arsenal_rows[1] if len(arsenal_rows) > 1 else None
                    if primary and secondary:
                        st.caption(
                            f"Primary: **{primary['Pitch']}** ({primary['Usage']:.1f}%) | "
                            f"Secondary: **{secondary['Pitch']}** ({secondary['Usage']:.1f}%) | "
                            f"Arsenal: **{len(arsenal_rows)} pitches**"
                        )

        with col_stuff:
            if not mov.empty:
                st.markdown("**Stuff Profile**")
                stuff_data = []
                for lbl, col, fmt in [
                    ("Avg Velo", "Vel", ".1f"), ("Max Velo", "MxVel", ".1f"),
                    ("Velo Range", "VelRange", ".1f"),
                    ("Spin (rpm)", "Spin", ".0f"),
                    ("Extension", "Extension", ".1f"),
                    ("Eff. Velo", "EffectVel", ".1f"),
                    ("IVB", "IndVertBrk", ".1f"),
                    ("Horz Break", "HorzBrk", ".1f"),
                    ("VAA", "VertApprAngle", ".2f"),
                ]:
                    v = _safe_num(mov, col)
                    if not pd.isna(v):
                        pct = _tm_pctile(mov, col, all_p_mov)
                        pct_str = f"{int(pct)}" if not pd.isna(pct) else "-"
                        stuff_data.append({"Metric": lbl, "Value": f"{v:{fmt}}", "%ile": pct_str})
                if stuff_data:
                    st.dataframe(pd.DataFrame(stuff_data), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 2B: TRACKMAN MOVEMENT, HEATMAPS & ARSENAL DETAIL
    # ══════════════════════════════════════════════════════════
    p_tm_for_cmd = pd.DataFrame()
    if not trackman_data.empty:
        tm_only = _prefer_truemedia_pitch_data(trackman_data)
        src_label = _pitch_source_label(tm_only)
        p_tm_raw = _match_pitcher_trackman(tm_only, pitcher)
        p_tm = _filter_pitch_types_global(p_tm_raw, MIN_PITCH_USAGE_PCT)
        # For predictability/entropy and command maps, keep all defined pitch types
        # (drop only undefined labels; do not min-usage filter).
        p_tm_predict = _filter_pitch_types_global(p_tm_raw, min_pct=0.0)
        p_tm_for_cmd = p_tm_predict.copy() if p_tm_predict is not None else pd.DataFrame()
        if not p_tm.empty and len(p_tm) >= 20:
            # ── Movement Profile ──
            if "HorzBreak" in p_tm.columns and "InducedVertBreak" in p_tm.columns:
                section_header(f"Movement Profile ({src_label})")
                st.caption(f"Based on {len(p_tm)} pitches (filtered undefined/<{MIN_PITCH_USAGE_PCT:.0f}% usage)")
                fig_mov = make_movement_profile(p_tm)
                if fig_mov is not None:
                    st.plotly_chart(fig_mov, use_container_width=True)

            # ── Per-Pitch Location Heatmaps ──
            if "TaggedPitchType" in p_tm.columns:
                bad_pitch_labels = {"UN", "UNK", "UNKNOWN", "UNDEFINED", "OTHER", "NONE", "NULL", "NAN", ""}
                bad_pitch_labels |= {str(v).strip().upper() for v in PITCH_TYPES_TO_DROP}
                pt_clean = p_tm["TaggedPitchType"].dropna()
                pt_clean = pt_clean[~pt_clean.astype(str).str.strip().str.upper().isin(bad_pitch_labels)]
                pitch_types_avail = pt_clean.value_counts()
                pitch_types_avail = pitch_types_avail[pitch_types_avail >= 10].index.tolist()
                if pitch_types_avail:
                    section_header(f"Pitch Location Heatmaps ({src_label})")
                    n_cols = min(len(pitch_types_avail), 3)
                    cols = st.columns(n_cols)
                    for i, pt_name in enumerate(pitch_types_avail[:6]):
                        with cols[i % n_cols]:
                            pt_data = p_tm[p_tm["TaggedPitchType"] == pt_name]
                            color = PITCH_COLORS.get(pt_name, "#888")
                            fig_loc = make_pitch_location_heatmap(pt_data, pt_name, color)
                            if fig_loc is not None:
                                st.plotly_chart(fig_loc, use_container_width=True)

            # ── Arsenal Detail Table (Trackman-derived) ──
            if "TaggedPitchType" in p_tm.columns:
                pt_counts = p_tm["TaggedPitchType"].dropna().value_counts()
                pt_valid = pt_counts[pt_counts >= 5].index.tolist()
                if pt_valid:
                    section_header(f"Arsenal Detail ({src_label})")
                    ars_rows = []
                    for pt_name in pt_valid:
                        pt_df = p_tm[p_tm["TaggedPitchType"] == pt_name]
                        n = len(pt_df)
                        velo = pd.to_numeric(pt_df.get("RelSpeed"), errors="coerce").dropna() if "RelSpeed" in pt_df.columns else pd.Series(dtype=float)
                        spin = pd.to_numeric(pt_df.get("SpinRate"), errors="coerce").dropna() if "SpinRate" in pt_df.columns else pd.Series(dtype=float)
                        ivb = pd.to_numeric(pt_df.get("InducedVertBreak"), errors="coerce").dropna() if "InducedVertBreak" in pt_df.columns else pd.Series(dtype=float)
                        hb = pd.to_numeric(pt_df.get("HorzBreak"), errors="coerce").dropna() if "HorzBreak" in pt_df.columns else pd.Series(dtype=float)
                        swings = pt_df[pt_df["PitchCall"].isin(SWING_CALLS)] if "PitchCall" in pt_df.columns else pd.DataFrame()
                        whiffs = pt_df[pt_df["PitchCall"] == "StrikeSwinging"] if "PitchCall" in pt_df.columns else pd.DataFrame()
                        whiff_pct = len(whiffs) / len(swings) * 100 if len(swings) > 0 else np.nan
                        ars_rows.append({
                            "Pitch": pt_name,
                            "N": n,
                            "Velo": f"{velo.mean():.1f}" if len(velo) > 0 else "-",
                            "Spin": f"{spin.mean():.0f}" if len(spin) > 0 else "-",
                            "IVB": f"{ivb.mean():.1f}" if len(ivb) > 0 else "-",
                            "HB": f"{hb.mean():.1f}" if len(hb) > 0 else "-",
                            "Whiff%": f"{whiff_pct:.1f}%" if not pd.isna(whiff_pct) else "-",
                        })
                    if ars_rows:
                        st.dataframe(pd.DataFrame(ars_rows), use_container_width=True, hide_index=True)
            # ── Pitch Predictability by Count & Hand (Top 3 most predictable) ──
            ent_df = _pitch_type_entropy_by_count(p_tm_predict, min_pitches=20, min_pitch_usage_pct=0.0)
            if not ent_df.empty:
                section_header(f"Most Predictable Counts ({src_label})")
                st.caption("Top 3 count/hand situations where this pitcher is most predictable.")

                # Get top 3 most predictable situations
                top3 = ent_df.nlargest(3, "Predictability")

                # Display as cards with location charts
                cols = st.columns(3)
                for i, (_, row) in enumerate(top3.iterrows()):
                    with cols[i]:
                        count_str = row["Count"]
                        side = row["BatterSide"]
                        side_label = "LHH" if side == "L" else ("RHH" if side == "R" else "Both")
                        pct = row["Top%"]
                        pitch = row["Top Pitch"]
                        n = int(row["N"])

                        st.markdown(f"**{count_str} vs {side_label}**")
                        st.metric(pitch, f"{pct:.0f}%")
                        st.caption(f"n={n}")

                        # Filter pitches for this count/side and show location
                        if {"Balls", "Strikes", "BatterSide", "PlateLocSide", "PlateLocHeight"}.issubset(p_tm_predict.columns):
                            balls, strikes = int(count_str.split("-")[0]), int(count_str.split("-")[1])
                            side_map = p_tm_predict["BatterSide"].astype(str).str.strip().str.upper().str[0]
                            balls_num = pd.to_numeric(p_tm_predict["Balls"], errors="coerce")
                            strikes_num = pd.to_numeric(p_tm_predict["Strikes"], errors="coerce")
                            if side in {"L", "R"}:
                                side_mask = side_map == side
                            else:
                                side_mask = side_map.isin(["L", "R", "S"])
                            count_pitches = p_tm_predict[
                                (balls_num == balls) &
                                (strikes_num == strikes) &
                                side_mask
                            ]
                            plot_pitches = count_pitches
                            if "TaggedPitchType" in count_pitches.columns and pitch in set(count_pitches["TaggedPitchType"].dropna()):
                                top_pitch_pitches = count_pitches[count_pitches["TaggedPitchType"] == pitch]
                                if len(top_pitch_pitches) >= 8:
                                    plot_pitches = top_pitch_pitches
                            if len(plot_pitches) >= 8:
                                bats = "L" if side == "L" else ("R" if side == "R" else None)
                                fig = _attack_zone_heatmap(
                                    plot_pitches,
                                    f"{count_str} Location",
                                    bats=bats,
                                    min_pitches=8,
                                )
                                if fig is not None:
                                    fig.update_layout(height=340)
                                    st.plotly_chart(fig, use_container_width=True)
                                    if len(plot_pitches) < len(count_pitches):
                                        st.caption(f"{pitch} locations (n={len(plot_pitches)})")
                                    else:
                                        st.caption(f"All pitch locations in this count (n={len(plot_pitches)})")

        elif p_tm_raw.empty:
            st.info(f"No {src_label} pitch-level data available for this pitcher.")
        else:
            st.info(f"No {src_label} pitch-level data after removing undefined or <{MIN_PITCH_USAGE_PCT:.0f}% usage pitches.")

    # ══════════════════════════════════════════════════════════
    # SECTION 3: COMMAND PROFILE
    # ══════════════════════════════════════════════════════════
    if not ploc.empty or not pr.empty:
        section_header("Command Profile")

        cmd_col1, cmd_col2 = st.columns([3, 2])

        with cmd_col1:
            # 3x3 Heatmaps split by hitter hand (vs RHH / vs LHH)
            rendered_cmd_heatmap = False
            if not p_tm_for_cmd.empty and {"PlateLocSide", "PlateLocHeight"}.issubset(p_tm_for_cmd.columns):
                cmd_loc = p_tm_for_cmd.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()

                def _render_cmd_heatmap_side(df_side, bats_side, label):
                    if len(df_side) < 20:
                        st.info(f"Not enough pitch-level samples {label}.")
                        return False
                    x_edges = np.array([-1.5, -0.5, 0.5, 1.5])
                    y_edges = np.array([0.5, 2.0, 3.0, 4.5])
                    df_side = df_side.copy()
                    # Hitter-relative Inside/Middle/Away (normalized).
                    # PlateLocSide is catcher-view (3B-side negative -> 1B-side positive). For LHH,
                    # "inside" is 1B-side, so we flip the sign so inside is always left on the chart.
                    side = pd.to_numeric(df_side["PlateLocSide"], errors="coerce")
                    if str(bats_side).strip().upper().startswith("L"):
                        side = -side
                    df_side["xbin"] = np.clip(np.digitize(side, x_edges) - 1, 0, 2)
                    df_side["ybin"] = np.clip(np.digitize(df_side["PlateLocHeight"], y_edges) - 1, 0, 2)
                    z_matrix = []
                    total = len(df_side)
                    for yi in range(2, -1, -1):
                        row = []
                        for xi in range(3):
                            cnt = len(df_side[(df_side["xbin"] == xi) & (df_side["ybin"] == yi)])
                            row.append(round(cnt / max(total, 1) * 100, 1))
                        z_matrix.append(row)

                    # Inside is always left due to the normalization above.
                    x_labels = ["Inside", "Middle", "Away"]

                    fig_hm = go.Figure(data=go.Heatmap(
                        z=z_matrix,
                        x=x_labels,
                        y=["High", "Middle", "Low"],
                        colorscale=[[0, "#f0f4f8"], [0.5, "#3d7dab"], [1, "#14365d"]],
                        showscale=False,
                        text=[[f"{v:.1f}%" for v in row] for row in z_matrix],
                        texttemplate="%{text}",
                        textfont=dict(size=14, color="white"),
                        hovertemplate="Zone: %{y} / %{x}<br>Frequency: %{text}<extra></extra>",
                    ))
                    fig_hm.add_shape(
                        type="rect", x0=-0.5, y0=-0.5, x1=2.5, y1=2.5,
                        line=dict(color="#000000", width=3),
                    )
                    _cl = {k: v for k, v in CHART_LAYOUT.items() if k != "margin"}
                    fig_hm.update_layout(
                        **_cl, height=280,
                        xaxis=dict(side="bottom"), yaxis=dict(autorange="reversed"),
                        margin=dict(l=60, r=10, t=10, b=40),
                    )
                    st.markdown(f"**{label}**")
                    st.plotly_chart(fig_hm, use_container_width=True)
                    st.caption(f"n={total}")
                    return True

                side_map = pd.Series(index=cmd_loc.index, dtype=object)
                if "BatterSide" in cmd_loc.columns:
                    side_map = cmd_loc["BatterSide"].astype(str).str.strip().str.upper().str[0]

                cmd_r = cmd_loc[side_map == "R"] if not side_map.empty else pd.DataFrame()
                cmd_l = cmd_loc[side_map == "L"] if not side_map.empty else pd.DataFrame()

                col_rhh, col_lhh = st.columns(2)
                with col_rhh:
                    ok_r = _render_cmd_heatmap_side(cmd_r, "R", "vs RHH")
                with col_lhh:
                    ok_l = _render_cmd_heatmap_side(cmd_l, "L", "vs LHH")

                rendered_cmd_heatmap = bool(ok_r or ok_l)
                if rendered_cmd_heatmap:
                    st.caption("Hitter-relative Inside/Middle/Away frequency from pitch-level TrueMedia data (Inside is normalized to the left for both hands).")

            # Do not approximate a 3x3 joint map from API marginals (can be misleading).
            if not rendered_cmd_heatmap:
                st.info("Command heatmap unavailable by hand: pitch-level data with BatterSide and location is required.")

        with cmd_col2:
            # Command rates with percentiles
            cmd_data = []
            # From pitch rates
            for lbl, src, col in [
                ("In Zone %", pr, "InZone%"), ("CompLoc %", pr, "CompLoc%"),
                ("Chase %", pr, "Chase%"), ("SwStrk %", pr, "SwStrk%"),
                ("Miss %", pr, "Miss%"), ("FPStk %", pr, "FPStk%"),
            ]:
                v = _safe_num(src, col)
                if not pd.isna(v):
                    all_src = all_p_ploc if src is ploc else all_p_pr
                    pct = _tm_pctile(src, col, all_src)
                    pct_str = f"{int(pct)}" if not pd.isna(pct) else "-"
                    cmd_data.append({"Metric": lbl, "Value": f"{v:.1f}%", "%ile": pct_str})
            if cmd_data:
                st.dataframe(pd.DataFrame(cmd_data), use_container_width=True, hide_index=True)

            # Command narrative
            inzone = _safe_num(pr, "InZone%") if not pr.empty else np.nan
            chase = _safe_num(pr, "Chase%") if not pr.empty else np.nan
            comploc = _safe_num(pr, "CompLoc%") if not pr.empty else np.nan
            if not pd.isna(inzone) and not pd.isna(chase):
                if inzone > 55 and chase < 22:
                    st.caption("📍 Lives in the zone but doesn't expand well — sit on strikes, attack early.")
                elif inzone < 42 and chase > 30:
                    st.caption("📍 Works off the plate with high chase rate — take pitches, force strikes.")
                elif not pd.isna(comploc) and comploc > 50:
                    st.caption("📍 Plus command — hits corners consistently, be ready to swing early.")

    # ══════════════════════════════════════════════════════════
    # SECTION 3B: PLATOON SPLITS (from pitch-level data)
    # ══════════════════════════════════════════════════════════
    if trackman_data is not None and not trackman_data.empty:
        pitch_df = _prefer_truemedia_pitch_data(trackman_data)
        src_label = _pitch_source_label(pitch_df)
        p_tm_raw = _match_pitcher_trackman(pitch_df, pitcher)
        p_tm = _filter_pitch_types_global(p_tm_raw, MIN_PITCH_USAGE_PCT)
        if not p_tm.empty and "BatterSide" in p_tm.columns and len(p_tm) >= 30:
            left_pitches = p_tm[p_tm["BatterSide"] == "Left"]
            right_pitches = p_tm[p_tm["BatterSide"] == "Right"]

            # Compute per-side stats
            def _platoon_side_stats(side_df):
                stats = {}
                if side_df.empty:
                    return stats
                total = len(side_df)
                swings = side_df[side_df["PitchCall"].isin(SWING_CALLS)] if "PitchCall" in side_df.columns else pd.DataFrame()
                whiffs = side_df[side_df["PitchCall"] == "StrikeSwinging"] if "PitchCall" in side_df.columns else pd.DataFrame()
                stats["n_pitches"] = total
                if len(swings) > 0:
                    stats["whiff_pct"] = len(whiffs) / len(swings) * 100
                bip = side_df[side_df["PitchCall"] == "InPlay"] if "PitchCall" in side_df.columns else pd.DataFrame()
                stats["n_bip"] = len(bip)
                if len(bip) >= 5 and "ExitSpeed" in side_df.columns:
                    ev_vals = bip["ExitSpeed"].dropna()
                    if len(ev_vals) >= 5:
                        stats["avg_ev"] = ev_vals.mean()
                # Approximate PA: count sequences ending in InPlay, StrikeSwinging (K proxy),
                # or walk-related calls
                k_calls = ["StrikeSwinging"]
                bb_calls = ["BallCalled", "HitByPitch"]
                # Simple K/BB proxy: strikeouts = pitch 3 swinging strikes approximated
                # Just use counting of terminal events
                strikeouts = len(side_df[side_df["PitchCall"] == "StrikeSwinging"])
                walks = 0  # harder to derive from pitch-level; skip BB rate
                in_play = len(bip)
                approx_pa = strikeouts + in_play
                if "BallinDirt" in side_df["PitchCall"].values:
                    approx_pa += 0  # not a terminal event
                # Use called strike 3 as additional K source
                called_k = side_df[(side_df["PitchCall"] == "StrikeCalled") & (side_df.get("Strikes", pd.Series()) == 2)] if "Strikes" in side_df.columns else pd.DataFrame()
                total_k = strikeouts + len(called_k)
                if approx_pa + len(called_k) > 0:
                    stats["k_rate"] = total_k / (approx_pa + len(called_k)) * 100
                # Pitch usage breakdown
                if "TaggedPitchType" in side_df.columns:
                    usage = side_df["TaggedPitchType"].value_counts(normalize=True) * 100
                    stats["pitch_usage"] = usage.to_dict()
                return stats

            l_stats = _platoon_side_stats(left_pitches)
            r_stats = _platoon_side_stats(right_pitches)

            has_platoon_data = (l_stats.get("n_pitches", 0) >= 15 or r_stats.get("n_pitches", 0) >= 15)
            if has_platoon_data:
                section_header("Platoon Splits")
                st.caption(f"Derived from {src_label} pitch-level data ({l_stats.get('n_pitches', 0)} pitches vs LHH, "
                           f"{r_stats.get('n_pitches', 0)} pitches vs RHH)")

                ps_col1, ps_col2 = st.columns(2)

                for col_ctx, label, s_stats in [(ps_col1, "vs LHH", l_stats), (ps_col2, "vs RHH", r_stats)]:
                    with col_ctx:
                        st.markdown(f"**{label}**")
                        if s_stats.get("n_pitches", 0) < 15:
                            st.caption("Limited data — not enough pitches for reliable splits.")
                        else:
                            card_data = []
                            if "whiff_pct" in s_stats:
                                card_data.append({"Metric": "Whiff %", "Value": f"{s_stats['whiff_pct']:.1f}%"})
                            if "avg_ev" in s_stats:
                                card_data.append({"Metric": "Avg EV Against", "Value": f"{s_stats['avg_ev']:.1f} mph"})
                            if "k_rate" in s_stats:
                                card_data.append({"Metric": "K Rate (approx)", "Value": f"{s_stats['k_rate']:.1f}%"})
                            card_data.append({"Metric": "BIP", "Value": str(s_stats.get("n_bip", 0))})
                            if card_data:
                                st.dataframe(pd.DataFrame(card_data), use_container_width=True, hide_index=True)

                # Pitch usage comparison table
                l_usage = l_stats.get("pitch_usage", {})
                r_usage = r_stats.get("pitch_usage", {})
                all_pitches_set = sorted(set(list(l_usage.keys()) + list(r_usage.keys())))
                if all_pitches_set:
                    usage_rows = []
                    highlight_notes = []
                    for pt_name in all_pitches_set:
                        l_pct = l_usage.get(pt_name, 0)
                        r_pct = r_usage.get(pt_name, 0)
                        usage_rows.append({"Pitch": pt_name, "vs LHH": f"{l_pct:.1f}%", "vs RHH": f"{r_pct:.1f}%"})
                        if abs(l_pct - r_pct) >= 10:
                            heavier_side = "LHH" if l_pct > r_pct else "RHH"
                            heavier_pct = max(l_pct, r_pct)
                            lighter_pct = min(l_pct, r_pct)
                            highlight_notes.append(
                                f"Throws {pt_name} **{heavier_pct:.0f}%** vs {heavier_side} but only "
                                f"**{lighter_pct:.0f}%** vs {'RHH' if heavier_side == 'LHH' else 'LHH'}"
                            )
                    st.markdown("**Pitch Usage by Side**")
                    st.dataframe(pd.DataFrame(usage_rows), use_container_width=True, hide_index=True)
                    for hn in highlight_notes:
                        st.caption(f"📊 {hn}")

                # Actionable narrative
                plat_narratives = []
                l_ev = l_stats.get("avg_ev")
                r_ev = r_stats.get("avg_ev")
                if l_ev is not None and r_ev is not None and abs(l_ev - r_ev) >= 3:
                    harder_side = "LHH" if l_ev > r_ev else "RHH"
                    plat_narratives.append(
                        f"⚠️ Gives up harder contact vs {harder_side} "
                        f"({max(l_ev, r_ev):.1f} EV vs {min(l_ev, r_ev):.1f} EV)."
                    )
                l_whiff = l_stats.get("whiff_pct")
                r_whiff = r_stats.get("whiff_pct")
                if l_whiff is not None and r_whiff is not None and abs(l_whiff - r_whiff) >= 8:
                    more_whiff_side = "LHH" if l_whiff > r_whiff else "RHH"
                    plat_narratives.append(
                        f"⚡ More swings and misses vs {more_whiff_side} "
                        f"({max(l_whiff, r_whiff):.0f}% vs {min(l_whiff, r_whiff):.0f}%)."
                    )
                # Pitch usage narrative (top highlight)
                if highlight_notes:
                    top_note = highlight_notes[0]
                    # Extract pitch name for quick tip
                    plat_narratives.append(f"📋 {top_note} — look for it early in counts.")

                if not plat_narratives:
                    if l_ev is not None and r_ev is not None and l_whiff is not None and r_whiff is not None:
                        plat_narratives.append("✅ No significant platoon advantage — consistent from both sides.")

                for pn in plat_narratives:
                    st.caption(pn)

    # ══════════════════════════════════════════════════════════
    # SECTION 3C: BASERUNNING VS THIS PITCHER
    # ══════════════════════════════════════════════════════════
    _scouting_pitcher_baserunning_panel(tm, team, pitcher, trackman_data)

    # ── Pitch-Level Overlay (prefer TrueMedia) ──
    pitch_df = _prefer_truemedia_pitch_data(trackman_data) if trackman_data is not None else pd.DataFrame()
    src_label = _pitch_source_label(pitch_df)
    _trackman_pitcher_overlay(pitch_df, pitcher, src_label)


def _name_variants(name):
    """Generate normalized name variants for robust cross-source joins."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return set()
    raw = str(name).strip()
    if not raw:
        return set()

    def _clean(s):
        s = re.sub(r"[^a-zA-Z, ]+", " ", str(s)).lower()
        return re.sub(r"\s+", " ", s).strip()

    out = set()
    n = _clean(raw)
    if not n:
        return out
    out.add(n.replace(",", " "))
    out.add(re.sub(r"\s+", " ", n.replace(",", " ")).strip())

    if "," in n:
        left, right = n.split(",", 1)
        left = left.strip()
        right = right.strip()
        if left and right:
            out.add(f"{right} {left}".strip())
            out.add(f"{left} {right}".strip())
    else:
        parts = n.split()
        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]
            out.add(f"{first} {last}".strip())
            out.add(f"{last} {first}".strip())
    return {re.sub(r"\s+", " ", x).strip() for x in out if x}


def _filter_local_team_rows(df, team):
    """Filter local CSV rows to the target opponent team."""
    if df is None or df.empty:
        return pd.DataFrame()
    if "newestTeamName" not in df.columns:
        return df
    team_norm = str(team).strip().lower()
    out = df[df["newestTeamName"].astype(str).str.strip().str.lower() == team_norm].copy()
    if not out.empty:
        return out
    first_token = team_norm.split()[0] if team_norm else ""
    if not first_token:
        return df
    return df[df["newestTeamName"].astype(str).str.lower().str.contains(first_token, na=False)].copy()


def _match_local_player_rows(df, player_name):
    """Match rows for a player using name variants across common name columns."""
    if df is None or df.empty:
        return pd.DataFrame()
    keys = _name_variants(player_name)
    if not keys:
        return pd.DataFrame()

    name_cols = [c for c in ["playerFullName", "abbrevName", "player"] if c in df.columns]
    if not name_cols:
        return pd.DataFrame()

    masks = []
    for col in name_cols:
        col_keys = df[col].apply(lambda x: any(v in _name_variants(x) for v in keys))
        masks.append(col_keys)
    if not masks:
        return pd.DataFrame()

    m = masks[0]
    for extra in masks[1:]:
        m = m | extra
    return df[m].copy()


def _pitcher_steal_window_counts(pitch_df):
    """Return count-level windows where steals are most likely to succeed.

    Heuristic only: we do not know actual steal attempts by count for this pitcher here.
    We use pitch mix (fastball vs offspeed) and velo as a proxy for "run windows".
    """
    req = {"Balls", "Strikes", "TaggedPitchType", "RelSpeed"}
    if pitch_df is None or pitch_df.empty or not req.issubset(pitch_df.columns):
        return pd.DataFrame()
    df = pitch_df.copy()
    df = df.dropna(subset=["Balls", "Strikes", "TaggedPitchType"])
    if df.empty:
        return pd.DataFrame()

    df["Balls"] = pd.to_numeric(df["Balls"], errors="coerce")
    df["Strikes"] = pd.to_numeric(df["Strikes"], errors="coerce")
    df = df[df["Balls"].notna() & df["Strikes"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df["Balls"] = df["Balls"].astype(int)
    df["Strikes"] = df["Strikes"].astype(int)
    df["Count"] = df["Balls"].astype(str) + "-" + df["Strikes"].astype(str)
    df["RelSpeed"] = pd.to_numeric(df["RelSpeed"], errors="coerce")

    fb_tokens = {"FASTBALL", "FOUR-SEAM", "4-SEAM", "SINKER", "2-SEAM", "TWO-SEAM", "CUTTER"}
    pt_norm = df["TaggedPitchType"].astype(str).str.strip().str.upper()
    df["is_fastball"] = pt_norm.apply(lambda x: any(tok in x for tok in fb_tokens)).astype(int)

    agg = (
        df.groupby("Count", as_index=False)
        .agg(
            N=("Count", "size"),
            FastballPct=("is_fastball", "mean"),
            AvgVelo=("RelSpeed", "mean"),
        )
    )
    agg = agg[agg["N"] >= 12].copy()
    if agg.empty:
        return agg
    agg["FastballPct"] = agg["FastballPct"] * 100
    agg["OffspeedPct"] = 100 - agg["FastballPct"]

    # Lower fastball share + lower avg velo = better steal window.
    agg["RunWindowScore"] = (100 - agg["FastballPct"]) * 0.7 + np.clip((90 - agg["AvgVelo"]) * 5, 0, 30)
    agg = agg.sort_values(["RunWindowScore", "N"], ascending=[False, False]).head(6)
    return agg.reset_index(drop=True)


def _scouting_pitcher_baserunning_panel(tm, team, pitcher, trackman_data):
    """Pitcher-specific baserunning scouting: steal risk, pickoff risk, and count windows."""
    section_header("Baserunning vs This Pitcher")
    st.caption("Actionable steal windows for this specific pitcher (local baserunning CSVs + TrueMedia pitch-level counts).")

    p_sb = _filter_local_team_rows(_load_local_pitcher_baserunning(), team)
    p_pk = _filter_local_team_rows(_load_local_pickoffs(), team)
    c_sb = _filter_local_team_rows(_load_local_catcher_throws(), team)
    p_sb = _match_local_player_rows(p_sb, pitcher)
    p_pk = _match_local_player_rows(p_pk, pitcher)

    # Top row: 3 compact cards (keep tables out of columns so nothing gets squished).
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("**Steal Success Allowed**")
        if not p_sb.empty:
            row = p_sb.sort_values("SBOpp", ascending=False).iloc[0]
            sb = pd.to_numeric(pd.Series([row.get("SB")]), errors="coerce").iloc[0]
            cs = pd.to_numeric(pd.Series([row.get("CS")]), errors="coerce").iloc[0]
            sba = pd.to_numeric(pd.Series([row.get("SBA")]), errors="coerce").iloc[0]
            sb_pct = pd.to_numeric(pd.Series([row.get("SB%")]), errors="coerce").iloc[0]
            if pd.isna(sb_pct) and (not pd.isna(sb)) and (not pd.isna(cs)) and (sb + cs) > 0:
                sb_pct = (sb / (sb + cs)) * 100

            if pd.notna(sb_pct):
                st.metric("Runner Success%", f"{sb_pct:.1f}%")
            if pd.notna(sb) and pd.notna(cs):
                st.caption(f"SB/CS: {int(sb)} / {int(cs)}")
            if pd.notna(sba):
                st.caption(f"SBA: {int(sba)}")

            if pd.notna(sb_pct):
                if sb_pct >= 85:
                    st.caption("Signal: High steal vulnerability.")
                elif sb_pct >= 72:
                    st.caption("Signal: Moderate steal vulnerability.")
                else:
                    st.caption("Signal: Lower steal vulnerability.")
        else:
            st.caption("No pitcher SB/CS row found.")

    with col2:
        st.markdown("**Pickoff Profile**")
        if not p_pk.empty:
            row = p_pk.sort_values("PMenOn", ascending=False).iloc[0]
            men_on = pd.to_numeric(pd.Series([row.get("PMenOn")]), errors="coerce").iloc[0]
            pk_att = pd.to_numeric(pd.Series([row.get("PitcherPKAtt")]), errors="coerce").iloc[0]
            pk = pd.to_numeric(pd.Series([row.get("PitcherPK")]), errors="coerce").iloc[0]
            men_per_att = pd.to_numeric(pd.Series([row.get("P/PKAtt")]), errors="coerce").iloc[0]

            if pd.notna(pk_att):
                st.metric("Pickoff Attempts", f"{int(pk_att)}")
            if pd.notna(pk):
                st.caption(f"Pickoffs: {int(pk)}")
            if pd.notna(men_per_att):
                st.caption(f"Men-on per attempt: {men_per_att:.1f}")
            elif pd.notna(men_on) and pd.notna(pk_att) and pk_att > 0:
                st.caption(f"Men-on per attempt: {men_on / pk_att:.1f}")

            if pd.notna(men_per_att):
                if men_per_att <= 7:
                    st.caption("Signal: Aggressive pickoff threat.")
                elif men_per_att <= 14:
                    st.caption("Signal: Moderate pickoff threat.")
                else:
                    st.caption("Signal: Lower pickoff threat.")
        else:
            st.caption("No pickoff row found.")

    with col3:
        st.markdown("**Best Counts to Run**")
        p_pitch = _match_pitcher_trackman(_prefer_truemedia_pitch_data(trackman_data), pitcher) if isinstance(trackman_data, pd.DataFrame) else pd.DataFrame()
        p_pitch = _filter_pitch_types_global(p_pitch, min_pct=0.0) if not p_pitch.empty else pd.DataFrame()
        count_windows = _pitcher_steal_window_counts(p_pitch)
        if not count_windows.empty:
            # "Best" by offspeed share (simple, coach-friendly).
            top_os = count_windows.sort_values(["OffspeedPct", "N"], ascending=[False, False]).iloc[0]
            st.metric("Top Offspeed Count", f"{top_os['Count']}")
            st.caption(f"Offspeed {top_os['OffspeedPct']:.0f}% | AvgVelo {top_os['AvgVelo']:.1f} mph | n={int(top_os['N'])}")
        else:
            st.caption("No reliable count windows yet (need >=12 pitches per count).")

    # Full-width: show count windows table (avoid tiny tables inside narrow columns).
    if "count_windows" in locals() and isinstance(count_windows, pd.DataFrame) and (not count_windows.empty):
        st.markdown("**Offspeed Windows by Count**")
        show_df = count_windows[["Count", "N", "OffspeedPct", "FastballPct", "AvgVelo"]].copy()
        show_df["OffspeedPct"] = show_df["OffspeedPct"].map(lambda x: f"{x:.0f}%")
        show_df["FastballPct"] = show_df["FastballPct"].map(lambda x: f"{x:.0f}%")
        show_df["AvgVelo"] = show_df["AvgVelo"].map(lambda x: f"{x:.1f}")
        st.dataframe(show_df, use_container_width=True, hide_index=True)

    # Catcher control matters for steal success; show team context here (not pitcher-specific).
    st.markdown("**Catcher Stolen Base Control (Team)**")
    if c_sb is None or c_sb.empty:
        st.caption("No catcher SB/CS data found for this team.")
    else:
        c_df = c_sb.copy()
        if "pos" in c_df.columns:
            c_df = c_df[c_df["pos"].astype(str).str.upper().str.contains("C", na=False)].copy()
        if c_df.empty:
            st.caption("No catcher rows found for this team.")
        else:
            # Prefer 2B steal defense, but fall back to overall if needed.
            for col in ["SBOpp", "SBA", "SB", "CS", "SB2", "CS2", "SBA2", "SBOpp2", "PopTime", "PopTimeSBA2", "SB%"]:
                if col in c_df.columns:
                    c_df[col] = pd.to_numeric(c_df[col], errors="coerce")
            if "PopTimeSBA2" in c_df.columns and "PopTime" not in c_df.columns:
                c_df["PopTime"] = pd.to_numeric(c_df["PopTimeSBA2"], errors="coerce")

            sort_col = "SBOpp"
            if sort_col not in c_df.columns:
                sort_col = "SBA" if "SBA" in c_df.columns else None
            if sort_col:
                c_df = c_df.sort_values(sort_col, ascending=False)
            primary = c_df.iloc[0]

            sb = pd.to_numeric(pd.Series([primary.get("SB2", primary.get("SB"))]), errors="coerce").iloc[0]
            cs = pd.to_numeric(pd.Series([primary.get("CS2", primary.get("CS"))]), errors="coerce").iloc[0]
            sba = pd.to_numeric(pd.Series([primary.get("SBA2", primary.get("SBA"))]), errors="coerce").iloc[0]
            pop = pd.to_numeric(pd.Series([primary.get("PopTime")]), errors="coerce").iloc[0]
            succ = pd.to_numeric(pd.Series([primary.get("SB%")]), errors="coerce").iloc[0]
            if pd.isna(succ) and pd.notna(sb) and pd.notna(cs) and (sb + cs) > 0:
                succ = (sb / (sb + cs)) * 100

            name = str(primary.get("playerFullName", primary.get("player", "Catcher"))).strip()
            c_col1, c_col2 = st.columns([1, 2])
            with c_col1:
                st.caption(f"Primary: **{name}**")
                if pd.notna(succ):
                    st.metric("Runner Success% vs Catcher", f"{succ:.1f}%")
                parts = []
                if pd.notna(sb) and pd.notna(cs):
                    parts.append(f"SB/CS: {int(sb)} / {int(cs)}")
                if pd.notna(sba):
                    parts.append(f"SBA: {int(sba)}")
                if pd.notna(pop):
                    parts.append(f"Pop: {pop:.2f}s")
                if parts:
                    st.caption(" | ".join(parts))

            with c_col2:
                with st.expander("Show catcher details", expanded=False):
                    cols = [c for c in ["playerFullName", "SBA", "SB", "CS", "SB%", "PopTime"] if c in c_df.columns]
                    if cols:
                        show = c_df[cols].head(5).copy()
                        if "SB%" in show.columns:
                            show["SB%"] = show["SB%"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
                        if "PopTime" in show.columns:
                            show["PopTime"] = show["PopTime"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                        st.dataframe(show, use_container_width=True, hide_index=True)

    if (not p_sb.empty) or (not p_pk.empty):
        notes = []
        if not p_sb.empty:
            row = p_sb.sort_values("SBOpp", ascending=False).iloc[0]
            sb_pct = pd.to_numeric(pd.Series([row.get("SB%")]), errors="coerce").iloc[0]
            if pd.notna(sb_pct):
                notes.append(f"Steal success allowed: **{sb_pct:.1f}%**")
        if not p_pk.empty:
            row = p_pk.sort_values("PMenOn", ascending=False).iloc[0]
            ppkatt = pd.to_numeric(pd.Series([row.get("P/PKAtt")]), errors="coerce").iloc[0]
            if pd.notna(ppkatt):
                notes.append(f"Pickoff frequency: **1 per {ppkatt:.1f} men-on**")
        if notes:
            st.caption(" | ".join(notes))


def _trackman_pitcher_overlay(data, pitcher_name, source_label="Trackman"):
    """Show pitch-level location heatmap if we have data for this pitcher."""
    last_name = pitcher_name.split()[-1] if " " in pitcher_name else pitcher_name
    matches = data[data["Pitcher"].str.contains(last_name, case=False, na=False)]
    if matches.empty or len(matches) < 10:
        return
    section_header(f"Pitch-Level Data Overlay ({source_label})")
    st.caption(f"Pitch-level data from {source_label} ({len(matches)} pitches)")
    loc = matches.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if not loc.empty and len(loc) >= 5:
        fig = px.density_heatmap(loc, x="PlateLocSide", y="PlateLocHeight", nbinsx=12, nbinsy=12,
                                 color_continuous_scale="YlOrRd")
        add_strike_zone(fig)
        fig.update_layout(title="Pitch Locations", xaxis=dict(range=[-3, 3], scaleanchor="y"),
                          yaxis=dict(range=[0, 5]),
                          height=400, coloraxis_showscale=False, **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


def _scouting_catcher_report(tm, team):
    """Their Catchers tab — arm, framing, defense, pickoffs, pitch calling with percentile context."""
    c_def = _tm_team(tm["catching"]["defense"], team)
    c_frm = _tm_team(tm["catching"]["framing"], team)
    c_throws = _tm_team(tm["catching"]["throws"], team)
    c_pr = _tm_team(tm["catching"]["pitch_rates"], team)
    c_pt_rates = _tm_team(tm["catching"]["pitch_types_rates"], team)
    c_pt = _tm_team(tm["catching"]["pitch_types"], team)
    c_opp = _tm_team(tm["catching"]["opposing"], team)
    c_sba2 = _tm_team(tm["catching"]["sba2_throws"], team)
    c_pk = _tm_team(tm["catching"]["pickoffs"], team)
    c_pbwp = _tm_team(tm["catching"]["pb_wp"], team)
    c_pcnts = _tm_team(tm["catching"]["pitch_counts"], team)

    all_c_def = tm["catching"]["defense"]
    all_c_frm = tm["catching"]["framing"]
    all_c_throws = tm["catching"]["throws"]
    all_c_pbwp = tm["catching"]["pb_wp"]
    all_c_pr = tm["catching"]["pitch_rates"]

    # Merge available catchers
    all_catchers = set()
    for df in [c_def, c_frm, c_throws, c_pbwp, c_pk]:
        if not df.empty and "playerFullName" in df.columns:
            all_catchers.update(df["playerFullName"].unique())
    if not all_catchers:
        st.info("No catcher data for this team.")
        return

    catchers = sorted(all_catchers)
    catcher = st.selectbox("Select Catcher", catchers, key="sc_catcher")

    cd = _tm_player(c_def, catcher)
    cf = _tm_player(c_frm, catcher)
    ct = _tm_player(c_throws, catcher)
    cr = _tm_player(c_pr, catcher)
    c_ptr = _tm_player(c_pt_rates, catcher)
    c_ptc = _tm_player(c_pt, catcher)
    c_op = _tm_player(c_opp, catcher)
    c_s2 = _tm_player(c_sba2, catcher)
    c_pick = _tm_player(c_pk, catcher)
    c_pw = _tm_player(c_pbwp, catcher)
    c_pc = _tm_player(c_pcnts, catcher)

    st.markdown(f"### {catcher}")

    # ── Catcher Narrative ──
    narrative_parts = []
    pop = _safe_num(ct, "PopTime")
    throw_spd = _safe_num(ct, "CThrowSpd")
    frm_raa = _safe_num(cf, "FrmRAA")
    cs_pct = _safe_num(cd, "CS%")

    if not pd.isna(pop):
        pop_pct = _tm_pctile(ct, "PopTime", all_c_throws)
        pop_rank = 100 - pop_pct if not pd.isna(pop_pct) else np.nan
        if pop_rank >= 75:
            narrative_parts.append(f"**{catcher} has an elite arm** ({pop:.2f}s pop time, {int(pop_rank)}th percentile).")
        elif pop_rank >= 50:
            narrative_parts.append(f"**{catcher} has an average arm** ({pop:.2f}s pop time).")
        else:
            narrative_parts.append(f"**{catcher} has a below-average arm** ({pop:.2f}s pop time) — running opportunities exist.")

    if not pd.isna(frm_raa):
        frm_pct = _tm_pctile(cf, "FrmRAA", all_c_frm)
        if not pd.isna(frm_pct):
            if frm_pct >= 75:
                narrative_parts.append(f"Elite framer ({frm_raa:+.1f} FrmRAA, {int(frm_pct)}th pctile) — expect extra called strikes on borderline pitches.")
            elif frm_pct <= 25:
                narrative_parts.append(f"Below-average framer ({frm_raa:+.1f} FrmRAA) — borderline pitches may go our way.")

    if not pd.isna(cs_pct):
        cs_rank = _tm_pctile(cd, "CS%", all_c_def)
        if not pd.isna(cs_rank):
            if cs_rank >= 70:
                narrative_parts.append(f"Strong CS% ({cs_pct:.1f}%) — be selective with steal attempts.")
            elif cs_rank <= 30:
                narrative_parts.append(f"Low CS% ({cs_pct:.1f}%) — green light to run.")

    # Add PB/WP narrative
    pbwp_raa = _safe_num(c_pw, "PBWPRAA")
    if not pd.isna(pbwp_raa):
        pbwp_pct = _tm_pctile(c_pw, "PBWPRAA", all_c_pbwp)
        if not pd.isna(pbwp_pct):
            if pbwp_pct >= 75:
                narrative_parts.append(f"Elite blocker ({pbwp_raa:+.1f} PBWPRAA) — rarely lets balls get by.")
            elif pbwp_pct <= 25:
                narrative_parts.append(f"Struggles blocking ({pbwp_raa:+.1f} PBWPRAA) — extra bases available on balls in the dirt.")

    if narrative_parts:
        st.markdown(" ".join(narrative_parts))

    # ── Percentile Rankings ──
    n_catchers = max(len(all_c_throws), len(all_c_frm), len(all_c_def))
    catcher_metrics = [
        ("Pop Time", _safe_num(ct, "PopTime"), _tm_pctile(ct, "PopTime", all_c_throws), ".2f", False),
        ("Throw Velo", _safe_num(ct, "CThrowSpd"), _tm_pctile(ct, "CThrowSpd", all_c_throws), ".1f", True),
        ("FrmRAA", _safe_num(cf, "FrmRAA"), _tm_pctile(cf, "FrmRAA", all_c_frm), ".1f", True),
        ("SLAA", _safe_num(cf, "SLAA"), _tm_pctile(cf, "SLAA", all_c_frm), ".1f", True),
        ("SL+", _safe_num(cf, "SL+"), _tm_pctile(cf, "SL+", all_c_frm), ".0f", True),
        ("CS %", _safe_num(cd, "CS%"), _tm_pctile(cd, "CS%", all_c_def), ".1f", True),
        ("FldRAA", _safe_num(cd, "FldRAA"), _tm_pctile(cd, "FldRAA", all_c_def), ".1f", True),
        ("PBWPRAA", _safe_num(c_pw, "PBWPRAA"), _tm_pctile(c_pw, "PBWPRAA", all_c_pbwp), ".1f", True),
    ]
    catcher_metrics = [(l, v, p, f, h) for l, v, p, f, h in catcher_metrics if not pd.isna(v)]
    if catcher_metrics:
        section_header("Percentile Rankings")
        st.caption(f"vs. {n_catchers:,} D1 catchers")
        render_savant_percentile_section(catcher_metrics)

    # ── Arm Details + Defense Details (side by side) ──
    col_arm, col_def = st.columns(2)
    with col_arm:
        if not ct.empty:
            section_header("Arm Details")
            arm_data = []
            for lbl, col_name, fmt in [("Exchange Time", "CExchTime", ".2f"), ("Throw Velo", "CThrowSpd", ".1f"), ("Pop Time", "PopTime", ".2f")]:
                v = _safe_num(ct, col_name)
                if not pd.isna(v):
                    arm_data.append({"Metric": lbl, "Value": f"{v:{fmt}}"})
            on_tgt = _safe_num(ct, "CThrowsOnTrgt")
            total = _safe_num(ct, "CThrows")
            if not pd.isna(on_tgt) and not pd.isna(total) and total > 0:
                arm_data.append({"Metric": "On-Target %", "Value": f"{on_tgt/total*100:.1f}%"})
            if arm_data:
                st.dataframe(pd.DataFrame(arm_data), use_container_width=True, hide_index=True)
    with col_def:
        if not cd.empty:
            section_header("Defense Details")
            def_data = []
            for lbl, col_name in [("PB", "PB"), ("WP", "WP"), ("Blocks", "CatBlock")]:
                v = _safe_num(cd, col_name)
                if not pd.isna(v):
                    def_data.append({"Metric": lbl, "Value": f"{int(v)}"})
            if def_data:
                st.dataframe(pd.DataFrame(def_data), use_container_width=True, hide_index=True)

    # ── SBA2 Throw Details ──
    if not c_s2.empty:
        section_header("SBA2 Throw Breakdown")
        sba2_data = []
        for lbl, col_name, fmt in [
            ("SB2 Allowed", "SB2", "d"), ("CS2", "CS2", "d"),
            ("SBA2 Throws", "CSBA2Throws", "d"), ("On-Target", "CSBA2ThrowsOnTrgt", "d"),
            ("Off-Target", "CSBA2ThrowsOffTrgt", "d"),
            ("SBA2 Throw Velo", "CSBA2ThrowSpd", ".1f"),
            ("SBA2 Pop Time", "PopTimeSBA2", ".2f"),
            ("SBA2 Exchange", "CSBA2ExchTime", ".2f"),
            ("SBA2 Time to Base", "CSBA2TimeToBase", ".2f"),
        ]:
            v = _safe_num(c_s2, col_name)
            if not pd.isna(v):
                val_str = f"{int(v)}" if fmt == "d" else f"{v:{fmt}}"
                sba2_data.append({"Metric": lbl, "Value": val_str})
        if sba2_data:
            st.dataframe(pd.DataFrame(sba2_data), use_container_width=True, hide_index=True)

    # ── Pickoff Activity ──
    if not c_pick.empty:
        section_header("Pickoff Activity")
        pk_data = []
        for lbl, col_name in [
            ("Total PK Attempts", "CatcherPKAtt"), ("Pickoffs", "CatcherPK"), ("PK Errors", "CatcherPKErr"),
            ("PK to 1B Att", "PK1Att"), ("PK to 1B", "PK1"), ("PK1 Err", "PK1Err"),
            ("PK to 2B Att", "PK2Att"), ("PK to 2B", "PK2"), ("PK2 Err", "PK2Err"),
            ("PK to 3B Att", "PK3Att"), ("PK to 3B", "PK3"), ("PK3 Err", "PK3Err"),
        ]:
            v = _safe_num(c_pick, col_name)
            if not pd.isna(v) and v > 0:
                pk_data.append({"Metric": lbl, "Value": f"{int(v)}"})
        if pk_data:
            st.dataframe(pd.DataFrame(pk_data), use_container_width=True, hide_index=True)

    # ── Passed Balls & Wild Pitches ──
    if not c_pw.empty:
        section_header("Passed Balls & Wild Pitches")
        pw_data = []
        for lbl, col_name, fmt in [
            ("PB", "PB", "d"), ("WP", "WP", "d"), ("PB+WP", "PBWP", "d"),
            ("xPB+WP", "xPBWP", ".1f"), ("PB+WP Above Avg", "PBWPAA", ".1f"),
            ("PBWP+", "PBWP+", ".0f"), ("PBWPRAA", "PBWPRAA", ".1f"), ("PBWPWAR", "PBWPWAR", ".1f"),
        ]:
            v = _safe_num(c_pw, col_name)
            if not pd.isna(v):
                val_str = f"{int(v)}" if fmt == "d" else f"{v:{fmt}}"
                pw_data.append({"Metric": lbl, "Value": val_str})
        if pw_data:
            st.dataframe(pd.DataFrame(pw_data), use_container_width=True, hide_index=True)

    # ── Framing Details ──
    if not cf.empty:
        section_header("Framing Details")
        frm_data = []
        for lbl, col_name, fmt in [
            ("SLAA", "SLAA", ".1f"), ("SL+", "SL+", ".0f"), ("FrmRAA", "FrmRAA", ".1f"),
            ("FrmCntRAA", "FrmCntRAA", ".1f"), ("Strikes Framed", "StrkFrmd", "d"),
            ("Balls Framed", "BallFrmd", "d"),
        ]:
            v = _safe_num(cf, col_name)
            if not pd.isna(v):
                val_str = f"{int(v)}" if fmt == "d" else f"{v:{fmt}}"
                frm_data.append({"Metric": lbl, "Value": val_str})
        if frm_data:
            st.dataframe(pd.DataFrame(frm_data), use_container_width=True, hide_index=True)

    # ── Pitch Calling / Game Management ──
    if not cr.empty:
        section_header("Pitch Calling")
        st.caption("How this catcher calls the game — pitch rates when behind the plate")
        call_metrics = [
            ("InZone %", _safe_num(cr, "InZone%"), _tm_pctile(cr, "InZone%", all_c_pr), ".1f", True),
            ("Chase %", _safe_num(cr, "Chase%"), _tm_pctile(cr, "Chase%", all_c_pr), ".1f", True),
            ("Miss %", _safe_num(cr, "Miss%"), _tm_pctile(cr, "Miss%", all_c_pr), ".1f", True),
            ("SwStrk %", _safe_num(cr, "SwStrk%"), _tm_pctile(cr, "SwStrk%", all_c_pr), ".1f", True),
            ("CompLoc %", _safe_num(cr, "CompLoc%"), _tm_pctile(cr, "CompLoc%", all_c_pr), ".1f", True),
        ]
        call_metrics = [(l, v, p, f, h) for l, v, p, f, h in call_metrics if not pd.isna(v)]
        if call_metrics:
            render_savant_percentile_section(call_metrics)

    # ── Pitch Mix Called ──
    if not c_ptr.empty:
        section_header("Pitch Mix Called")
        pitch_cols = ["4Seam%", "Sink2Seam%", "Cutter%", "Slider%", "Curve%", "Change%", "Split%", "Sweeper%"]
        pitch_labels = ["4-Seam", "Sinker", "Cutter", "Slider", "Curve", "Change", "Splitter", "Sweeper"]
        vals = []
        for col_name, lbl in zip(pitch_cols, pitch_labels):
            v = _safe_num(c_ptr, col_name) if col_name in c_ptr.columns else np.nan
            if not pd.isna(v) and v > 0:
                vals.append({"Pitch": lbl, "Usage": v})
        if vals:
            vdf = pd.DataFrame(vals)
            fig = go.Figure(go.Bar(
                x=vdf["Pitch"], y=vdf["Usage"],
                marker_color=[PITCH_COLORS.get(p.replace("-", ""), "#888") for p in vdf["Pitch"]],
                text=[f"{u:.1f}%" for u in vdf["Usage"]], textposition="outside",
            ))
            fig.update_layout(**CHART_LAYOUT, height=280, yaxis_title="Usage %", showlegend=False,
                              yaxis=dict(range=[0, vdf["Usage"].max() * 1.3]))
            st.plotly_chart(fig, use_container_width=True)

    # ── Opposing Batters ──
    if not c_op.empty:
        section_header("Opposing Batters (When Catching)")
        opp_data = []
        for lbl, col_name, fmt in [
            ("AVG Against", "AVG", ".3f"), ("OBP Against", "OBP", ".3f"),
            ("SLG Against", "SLG", ".3f"), ("OPS Against", "OPS", ".3f"),
            ("H Allowed", "H", "d"), ("HR Allowed", "HR", "d"),
            ("K", "K", "d"), ("BB", "BB", "d"), ("HBP", "HBP", "d"),
        ]:
            v = _safe_num(c_op, col_name)
            if not pd.isna(v):
                val_str = f"{int(v)}" if fmt == "d" else f"{v:{fmt}}"
                opp_data.append({"Stat": lbl, "Value": val_str})
        if opp_data:
            st.dataframe(pd.DataFrame(opp_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# BASERUNNING INTELLIGENCE TAB
# ══════════════════════════════════════════════════════════════════════════════


def _find_local_csv(*candidates):
    """Find CSV in /data first, then project root, supporting legacy file names."""
    base = os.path.dirname(os.path.dirname(__file__))
    search_dirs = [os.path.join(base, "data"), base]
    for folder in search_dirs:
        for name in candidates:
            p = os.path.join(folder, name)
            if os.path.exists(p):
                return p
    return None


def _load_local_catcher_throws():
    """Load catcher throwing data from local CSV file."""
    csv_path = _find_local_csv("stolen_bases_catchers.csv", "Stolen Bases-3.csv", "Stolen Bases.csv")
    if not csv_path:
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        # Map PopTimeSBA2 to PopTime for compatibility
        if "PopTimeSBA2" in df.columns:
            df["PopTime"] = pd.to_numeric(df["PopTimeSBA2"], errors="coerce")
        for col in ["SB", "CS", "SBA", "SBOpp", "SB2", "CS2", "SB3", "CS3", "PopTime", "CThrowSpd", "CExchTime"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "SB%" in df.columns:
            df["SB%"] = df["SB%"].astype(str).str.replace("%", "", regex=False).replace("-", "")
            df["SB%"] = pd.to_numeric(df["SB%"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _load_local_pitcher_baserunning():
    """Load pitcher-level SB allowed data."""
    csv_path = _find_local_csv("stolen_bases_pitchers.csv", "Stolen Bases pitching.csv")
    if not csv_path:
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        for col in ["SB", "CS", "SBA", "SBOpp", "G", "SB2", "CS2", "SB3", "CS3", "SBAH", "SBH", "CSH"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for pct_col in ["SB%", "SB2%", "SB3%"]:
            if pct_col in df.columns:
                df[pct_col] = df[pct_col].astype(str).str.replace("%", "", regex=False).replace("-", "")
                df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _load_local_pickoffs():
    """Load pitcher pickoff tendencies."""
    csv_path = _find_local_csv("pickoffs.csv", "Pickoffs.csv")
    if not csv_path:
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        for col in [
            "PMenOn", "P/PKAtt", "PitcherPKAtt", "PitcherPK", "PitcherPKErr",
            "P1B", "P/PK1Att", "PK1Att", "PK1", "PK1Err",
            "P2B", "P/PK2Att", "PK2Att", "PK2", "PK2Err",
            "P3B", "P/PK3Att", "PK3Att", "PK3", "PK3Err",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _load_local_outfield_throws():
    """Load outfield throwing data (assists) from local CSV file."""
    import os
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "outfield_throws.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Convert numeric columns
            for col in ["OFAst", "LFAst", "CFAst", "RFAst", "OFAstTo1B", "OFAstTo2B", "OFAstTo3B", "OFAstToHome", "OFThrowE"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _load_local_stolen_bases_catcher():
    """Load catcher stolen base data (SB/CS allowed) from local CSV file."""
    import os
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stolen_bases_catchers.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Convert numeric columns
            for col in ["SB", "CS", "SBA", "SBOpp", "PopTimeSBA2", "SB2", "CS2", "SB3", "CS3"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            # Parse SB% (remove % sign, handle '-' as NaN)
            if "SB%" in df.columns:
                df["SB%"] = df["SB%"].astype(str).str.replace("%", "", regex=False).replace("-", "")
                df["SB%"] = pd.to_numeric(df["SB%"], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _load_local_stolen_bases_runners():
    """Load runner stolen base data from local CSV file."""
    import os
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stolen_bases_runners.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Convert numeric columns
            for col in ["SB", "CS", "SBA", "SBOpp", "SB2", "CS2", "SB3", "CS3", "G"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            # Parse SB% columns
            for pct_col in ["SB%", "SB2%", "SB3%"]:
                if pct_col in df.columns:
                    df[pct_col] = df[pct_col].astype(str).str.replace("%", "", regex=False).replace("-", "")
                    df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _load_local_speed_scores():
    """Load speed score data from local CSV file."""
    import os
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "speed_scores.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Convert numeric columns
            for col in ["SpeedScore", "SB", "SBA", "AB", "3B", "RS"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            # Parse SB%
            if "SB%" in df.columns:
                df["SB%"] = df["SB%"].astype(str).str.replace("%", "", regex=False).replace("-", "")
                df["SB%"] = pd.to_numeric(df["SB%"], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _load_local_running_bases():
    """Load team baserunning data (1st to 3rd, scoring, etc.) from local CSV file."""
    import os
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "running_bases.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Convert numeric columns
            for col in ["G", "1stTo3rdOpp", "1stTo3rdAtt", "1stTo3rd", "1stTo3rdOut",
                        "1stToHomeOpp", "1stToHomeAtt", "1stToHome", "1stToHomeOut",
                        "2ndToHomeOpp", "2ndToHomeAtt", "2ndToHome", "2ndToHomeOut", "ROE"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            # Parse percentage columns
            for pct_col in ["1stTo3rdAtt%", "1stTo3rdSafe%", "1stToHomeAtt%", "1stToHomeSafe%", "2ndToHomeAtt%", "2ndToHomeSafe%"]:
                if pct_col in df.columns:
                    df[pct_col] = df[pct_col].astype(str).str.replace("%", "", regex=False).replace("-", "")
                    df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _scouting_baserunning_report(tm, team, opp_pitches, season_year=2026):
    """Baserunning Intelligence tab — offensive and defensive baserunning preparation."""

    # ══════════════════════════════════════════════════════════════════════════
    # Section 1: CATCHER SCOUTING (Stealing Opportunities)
    # ══════════════════════════════════════════════════════════════════════════
    section_header("Catcher Scouting")
    st.caption("Identify opportunities to steal against their catchers")

    # Get catcher data from TrueMedia dict
    c_throws = _tm_team(tm["catching"]["throws"], team)
    c_def = _tm_team(tm["catching"]["defense"], team)
    c_sba2 = _tm_team(tm["catching"]["sba2_throws"], team)
    c_pk = _tm_team(tm["catching"]["pickoffs"], team)

    all_c_throws = tm["catching"]["throws"]
    all_c_def = tm["catching"]["defense"]

    # Fallback: load local CSV data if API data is empty
    if c_throws.empty:
        local_throws = _load_local_catcher_throws()
        if not local_throws.empty and "newestTeamName" in local_throws.columns:
            # Try exact match first
            c_throws = local_throws[local_throws["newestTeamName"] == team].copy()
            # If no match, try case-insensitive match
            if c_throws.empty:
                c_throws = local_throws[
                    local_throws["newestTeamName"].str.lower() == team.lower()
                ].copy()
            # If still no match, try partial match (contains first word of team name)
            if c_throws.empty:
                c_throws = local_throws[
                    local_throws["newestTeamName"].str.lower().str.contains(team.lower().split()[0], na=False)
                ].copy()
            all_c_throws = local_throws

    # Also load SB/CS data for catchers from local CSV
    local_catcher_sb = _load_local_stolen_bases_catcher()
    team_catcher_sb = pd.DataFrame()
    if not local_catcher_sb.empty and "newestTeamName" in local_catcher_sb.columns:
        # Try exact match first
        team_catcher_sb = local_catcher_sb[local_catcher_sb["newestTeamName"] == team].copy()
        # If no match, try case-insensitive match
        if team_catcher_sb.empty:
            team_catcher_sb = local_catcher_sb[
                local_catcher_sb["newestTeamName"].str.lower() == team.lower()
            ].copy()
        # If still no match, try partial match (contains)
        if team_catcher_sb.empty:
            team_catcher_sb = local_catcher_sb[
                local_catcher_sb["newestTeamName"].str.lower().str.contains(team.lower().split()[0], na=False)
            ].copy()

    # Merge available catchers
    all_catchers = set()
    for df in [c_throws, c_def, c_sba2, c_pk, team_catcher_sb]:
        if not df.empty and "playerFullName" in df.columns:
            all_catchers.update(df["playerFullName"].unique())

    # Initialize catcher_rows for recommendations section
    catcher_rows = []

    if all_catchers:
        catchers = sorted(all_catchers)

        # Build catcher scouting table
        for catcher in catchers:
            ct = _tm_player(c_throws, catcher)
            cd = _tm_player(c_def, catcher)
            c_s2 = _tm_player(c_sba2, catcher)
            c_sb = _tm_player(team_catcher_sb, catcher)

            pop = _safe_num(ct, "PopTime")
            throw_velo = _safe_num(ct, "CThrowSpd")
            exch = _safe_num(ct, "CExchTime")
            on_tgt = _safe_num(ct, "CThrowsOnTrgt")
            total_throws = _safe_num(ct, "CThrows")

            # Get SB/CS allowed from catcher SB data
            sb_allowed = _safe_num(c_sb, "SB")
            cs = _safe_num(c_sb, "CS")
            sb_pct = _safe_num(c_sb, "SB%")

            # Calculate on-target percentage
            on_tgt_pct = np.nan
            if not pd.isna(on_tgt) and not pd.isna(total_throws) and total_throws > 0:
                on_tgt_pct = (on_tgt / total_throws) * 100

            # Calculate pop time percentile (lower is better, so invert)
            pop_pctile = np.nan
            if not pd.isna(pop) and not all_c_throws.empty and "PopTime" in all_c_throws.columns:
                pop_pctile = _tm_pctile(ct, "PopTime", all_c_throws)
                # Invert so higher percentile = better for catcher
                if not pd.isna(pop_pctile):
                    pop_pctile = 100 - pop_pctile

            # Determine steal rating based on pop time AND actual SB% against catcher
            # SB% = success rate of runners AGAINST this catcher (higher = worse for catcher = better to steal)
            steal_rating = "Unknown"

            # First, check actual SB% if available (most reliable indicator)
            if not pd.isna(sb_pct):
                if sb_pct >= 85:
                    steal_rating = "GREEN LIGHT"
                elif sb_pct >= 75:
                    steal_rating = "Good"
                elif sb_pct >= 65:
                    steal_rating = "Caution"
                else:
                    steal_rating = "Avoid"
            # Fall back to pop time if no SB% data
            elif not pd.isna(pop):
                if pop >= 2.10:
                    steal_rating = "GREEN LIGHT"
                elif pop >= 2.00:
                    steal_rating = "Good"
                elif pop >= 1.95:
                    steal_rating = "Caution"
                else:
                    steal_rating = "Avoid"

            catcher_rows.append({
                "Catcher": catcher,
                "Pop Time": f"{pop:.2f}s" if not pd.isna(pop) else "-",
                "Throw Velo": f"{throw_velo:.1f}" if not pd.isna(throw_velo) else "-",
                "Exchange": f"{exch:.2f}s" if not pd.isna(exch) else "-",
                "SB Allowed": int(sb_allowed) if not pd.isna(sb_allowed) else "-",
                "CS": int(cs) if not pd.isna(cs) else "-",
                "SB%": f"{sb_pct:.1f}%" if not pd.isna(sb_pct) else "-",
                "Rating": steal_rating,
            })

        if catcher_rows:
            catcher_df = pd.DataFrame(catcher_rows)
            st.dataframe(catcher_df, use_container_width=True, hide_index=True)

            # Highlight weakest catcher (prioritize by SB%, then pop time)
            sb_pct_values = []
            pop_values = []
            for row in catcher_rows:
                try:
                    sb_str = row["SB%"].replace("%", "")
                    sb_pct_values.append((row["Catcher"], float(sb_str), row["SB%"]))
                except (ValueError, AttributeError):
                    pass
                try:
                    pop_str = row["Pop Time"].replace("s", "")
                    pop_values.append((row["Catcher"], float(pop_str), row["Pop Time"]))
                except (ValueError, AttributeError):
                    pass

            # Prefer SB% (higher = easier to steal on)
            if sb_pct_values:
                worst = max(sb_pct_values, key=lambda x: x[1])
                if worst[1] >= 75:
                    st.success(f"**Best steal target: {worst[0]}** (Runners succeed {worst[2]} of the time)")
            elif pop_values:
                slowest = max(pop_values, key=lambda x: x[1])
                if slowest[1] >= 2.00:
                    st.success(f"**Best steal target: {slowest[0]}** (Pop time: {slowest[2]})")
    else:
        st.info("No catcher throwing data available for this team.")

    # ══════════════════════════════════════════════════════════════════════════
    # Section 2: PITCHER STEAL VULNERABILITY (Based on actual SB/CS allowed)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    section_header("Pitcher Steal Vulnerability")
    st.caption("Which pitchers get stolen on the most?")

    # Load pitcher baserunning data from local CSV
    local_pitcher_br = _load_local_pitcher_baserunning()
    team_pitcher_br = pd.DataFrame()
    if not local_pitcher_br.empty and "newestTeamName" in local_pitcher_br.columns:
        # Try exact match first
        team_pitcher_br = local_pitcher_br[local_pitcher_br["newestTeamName"] == team].copy()
        # If no match, try case-insensitive match
        if team_pitcher_br.empty:
            team_pitcher_br = local_pitcher_br[
                local_pitcher_br["newestTeamName"].str.lower() == team.lower()
            ].copy()
        # If still no match, try partial match (contains)
        if team_pitcher_br.empty:
            team_pitcher_br = local_pitcher_br[
                local_pitcher_br["newestTeamName"].str.lower().str.contains(team.lower().split()[0], na=False)
            ].copy()

    # Initialize pitcher_rows for recommendations section
    pitcher_rows = []

    if not team_pitcher_br.empty and "playerFullName" in team_pitcher_br.columns:
        # Filter to pitchers with meaningful steal attempts against them
        team_pitcher_br = team_pitcher_br[team_pitcher_br["SBA"] >= 3].copy()

        for _, row in team_pitcher_br.iterrows():
            pitcher = row["playerFullName"]
            games = row.get("G", np.nan)
            sb_allowed = row.get("SB", np.nan)
            cs = row.get("CS", np.nan)
            sb_pct = row.get("SB%", np.nan)
            sba = row.get("SBA", np.nan)
            sb_opp = row.get("SBOpp", np.nan)

            # Calculate vulnerability rating
            rating = "No Data"
            if not pd.isna(sb_pct):
                if sb_pct >= 85:
                    rating = "HIGH VULN"
                elif sb_pct >= 75:
                    rating = "Vulnerable"
                elif sb_pct >= 65:
                    rating = "Average"
                else:
                    rating = "Good"
            elif not pd.isna(sb_allowed) and not pd.isna(cs):
                total = sb_allowed + cs
                if total > 0:
                    calc_pct = (sb_allowed / total) * 100
                    if calc_pct >= 85:
                        rating = "HIGH VULN"
                    elif calc_pct >= 75:
                        rating = "Vulnerable"
                    elif calc_pct >= 65:
                        rating = "Average"
                    else:
                        rating = "Good"

            pitcher_rows.append({
                "Pitcher": pitcher,
                "G": int(games) if not pd.isna(games) else "-",
                "SB Allowed": int(sb_allowed) if not pd.isna(sb_allowed) else "-",
                "CS": int(cs) if not pd.isna(cs) else "-",
                "SB%": f"{sb_pct:.1f}%" if not pd.isna(sb_pct) else "-",
                "SBA": int(sba) if not pd.isna(sba) else "-",
                "Rating": rating,
            })

        if pitcher_rows:
            # Sort by SB allowed (most vulnerable first)
            pitcher_rows = sorted(
                pitcher_rows,
                key=lambda x: x["SB Allowed"] if isinstance(x["SB Allowed"], int) else 0,
                reverse=True
            )

            pitcher_df = pd.DataFrame(pitcher_rows)
            st.dataframe(pitcher_df, use_container_width=True, hide_index=True)

            # Highlight most stolen-on pitcher
            vulnerable = [r for r in pitcher_rows if r["Rating"] in ["HIGH VULN", "Vulnerable"]]
            if vulnerable:
                top = vulnerable[0]
                st.success(f"**Best steal target: {top['Pitcher']}** — {top['SB Allowed']} SB allowed ({top['SB%']})")
        else:
            st.info("No pitcher baserunning data available.")
    else:
        st.info("No pitcher baserunning data available for this team.")

    # ══════════════════════════════════════════════════════════════════════════
    # Section 3: BEST COUNTS TO STEAL (Pitch Pattern Analysis)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    section_header("Best Counts to Steal")
    st.caption("Find counts with more breaking balls/changeups — slower pitches = more time to steal")

    if opp_pitches is not None and not opp_pitches.empty:
        # Analyze pitch patterns by count
        pitch_df = opp_pitches.copy()

        # Create count column (filter out rows with missing ball/strike counts)
        if "Balls" in pitch_df.columns and "Strikes" in pitch_df.columns:
            pitch_df = pitch_df.dropna(subset=["Balls", "Strikes"])
            pitch_df["Balls"] = pitch_df["Balls"].astype(int)
            pitch_df["Strikes"] = pitch_df["Strikes"].astype(int)
            pitch_df["Count"] = pitch_df["Balls"].astype(str) + "-" + pitch_df["Strikes"].astype(str)

            # Classify pitch types as fastball or offspeed
            fb_types = ["Fastball", "Four-Seam", "Sinker", "Cutter", "4-Seam", "2-Seam", "Two-Seam"]
            if "TaggedPitchType" in pitch_df.columns:
                pitch_df["IsFastball"] = pitch_df["TaggedPitchType"].apply(
                    lambda x: 1 if any(fb.lower() in str(x).lower() for fb in fb_types) else 0
                )

                # Get speed data
                if "RelSpeed" in pitch_df.columns:
                    pitch_df["RelSpeed"] = pd.to_numeric(pitch_df["RelSpeed"], errors="coerce")

                # Analyze by count
                count_analysis = pitch_df.groupby("Count").agg({
                    "IsFastball": ["sum", "count"],
                    "RelSpeed": "mean"
                }).reset_index()
                count_analysis.columns = ["Count", "FB_Count", "Total", "AvgVelo"]
                count_analysis["FB%"] = (count_analysis["FB_Count"] / count_analysis["Total"] * 100).round(1)
                count_analysis["Offspeed%"] = (100 - count_analysis["FB%"]).round(1)
                count_analysis["AvgVelo"] = count_analysis["AvgVelo"].round(1)

                # Filter to counts with enough data
                count_analysis = count_analysis[count_analysis["Total"] >= 20]

                if not count_analysis.empty:
                    # Sort by Offspeed% descending (more offspeed = slower pitches = better to steal)
                    count_analysis = count_analysis.sort_values("Offspeed%", ascending=False)

                    # Add steal recommendation based on offspeed %
                    def steal_rec(row):
                        if row["Offspeed%"] >= 50:
                            return "GREEN LIGHT"
                        elif row["Offspeed%"] >= 35:
                            return "Good"
                        elif row["Offspeed%"] >= 20:
                            return "Neutral"
                        else:
                            return "Caution"

                    count_analysis["Steal Rating"] = count_analysis.apply(steal_rec, axis=1)

                    # Display
                    display_counts = count_analysis[["Count", "Offspeed%", "AvgVelo", "Total", "Steal Rating"]].copy()
                    display_counts.columns = ["Count", "Offspeed %", "Avg Velo", "Pitches", "Rating"]
                    display_counts["Offspeed %"] = display_counts["Offspeed %"].apply(lambda x: f"{x:.1f}%")
                    display_counts["Avg Velo"] = display_counts["Avg Velo"].apply(lambda x: f"{x:.1f}")

                    st.dataframe(display_counts, use_container_width=True, hide_index=True)

                    # Highlight best counts (high offspeed %)
                    best_counts = count_analysis[count_analysis["Offspeed%"] >= 40]["Count"].tolist()

                    if best_counts:
                        st.success(f"**Best counts to steal:** {', '.join(best_counts[:4])} — High offspeed %, slower pitches")
                else:
                    st.info("Not enough pitch data to analyze count patterns.")
            else:
                st.info("No pitch type data available for count analysis.")
        else:
            st.info("No count data available in pitch data.")
    else:
        st.info("No pitch data available for count analysis.")

    # ══════════════════════════════════════════════════════════════════════════
    # Section 4: THEIR TOP BASE STEALERS (Defensive Prep)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    section_header("Their Top Base Stealers")
    st.caption("Identify their most dangerous runners — defensive preparation")

    h_cnt = _tm_team(tm["hitting"]["counting"], team)
    h_sb = _tm_team(tm["hitting"]["stolen_bases"], team) if "stolen_bases" in tm["hitting"] else pd.DataFrame()

    # Fallback: load local hitter SB data
    if h_cnt.empty or "SB" not in h_cnt.columns:
        local_hitter_sb = _load_local_stolen_bases_runners()
        if not local_hitter_sb.empty and "newestTeamName" in local_hitter_sb.columns:
            # Try to match team
            team_hitters = local_hitter_sb[local_hitter_sb["newestTeamName"] == team].copy()
            if team_hitters.empty:
                team_hitters = local_hitter_sb[
                    local_hitter_sb["newestTeamName"].str.lower() == team.lower()
                ].copy()
            if team_hitters.empty:
                team_hitters = local_hitter_sb[
                    local_hitter_sb["newestTeamName"].str.lower().str.contains(team.lower().split()[0], na=False)
                ].copy()
            if not team_hitters.empty:
                h_cnt = team_hitters

    # Use counting stats for SB data
    if not h_cnt.empty and "SB" in h_cnt.columns and "playerFullName" in h_cnt.columns:
        sb_data = h_cnt[["playerFullName", "SB"]].copy()

        # Add CS if available
        if "CS" in h_cnt.columns:
            sb_data["CS"] = h_cnt["CS"]
        else:
            sb_data["CS"] = np.nan

        # Filter to players with steal attempts
        sb_data = sb_data.dropna(subset=["SB"])
        sb_data = sb_data[sb_data["SB"] > 0]

        if not sb_data.empty:
            sb_data = sb_data.sort_values("SB", ascending=False).head(10)

            # Calculate SB%
            sb_data["SB%"] = sb_data.apply(
                lambda row: row["SB"] / (row["SB"] + row["CS"]) * 100
                if not pd.isna(row["CS"]) and (row["SB"] + row["CS"]) > 0
                else np.nan,
                axis=1
            )
            sb_data["Attempts"] = sb_data["SB"] + sb_data["CS"].fillna(0)

            # Calculate aggression rating (subjective)
            sb_data["Threat Level"] = sb_data.apply(
                lambda row: "HIGH" if row["SB"] >= 10 else ("Medium" if row["SB"] >= 5 else "Low"),
                axis=1
            )

            # Format for display
            display_rows = []
            for _, row in sb_data.iterrows():
                display_rows.append({
                    "Player": row["playerFullName"],
                    "SB": int(row["SB"]),
                    "CS": int(row["CS"]) if not pd.isna(row["CS"]) else "-",
                    "SB%": f"{row['SB%']:.1f}%" if not pd.isna(row["SB%"]) else "-",
                    "Attempts": int(row["Attempts"]),
                    "Threat": row["Threat Level"],
                })

            sb_df = pd.DataFrame(display_rows)
            st.dataframe(sb_df, use_container_width=True, hide_index=True)

            # Highlight their biggest threat
            top_stealer = sb_data.iloc[0]
            if top_stealer["SB"] >= 5:
                st.warning(f"**Watch out for {top_stealer['playerFullName']}** — {int(top_stealer['SB'])} SB on the season. Call slides, vary looks.")
        else:
            st.info("No stolen bases recorded for this team.")
    else:
        st.info("No stolen base data available for this team.")

    # ══════════════════════════════════════════════════════════════════════════
    # Section 5: OUTFIELDER ARM STRENGTH
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    section_header("Outfielder Arms")
    st.caption("Identify weak outfield arms — take extra bases on hits")

    # Load outfield throwing data
    local_of_throws = _load_local_outfield_throws()
    team_of_throws = pd.DataFrame()
    all_of_throws = local_of_throws.copy()  # For percentile calculations

    if not local_of_throws.empty and "newestTeamName" in local_of_throws.columns:
        # Try exact match first
        team_of_throws = local_of_throws[local_of_throws["newestTeamName"] == team].copy()
        # If no match, try case-insensitive match
        if team_of_throws.empty:
            team_of_throws = local_of_throws[
                local_of_throws["newestTeamName"].str.lower() == team.lower()
            ].copy()
        # If still no match, try partial match (contains)
        if team_of_throws.empty:
            team_of_throws = local_of_throws[
                local_of_throws["newestTeamName"].str.lower().str.contains(team.lower().split()[0], na=False)
            ].copy()

    of_rows = []
    if not team_of_throws.empty and "playerFullName" in team_of_throws.columns:
        for _, row in team_of_throws.iterrows():
            player = row["playerFullName"]
            pos = row.get("pos", "-")
            of_ast = row.get("OFAst", np.nan)
            lf_ast = row.get("LFAst", np.nan)
            cf_ast = row.get("CFAst", np.nan)
            rf_ast = row.get("RFAst", np.nan)
            throw_errors = row.get("OFThrowE", np.nan)

            # Determine primary position based on where they have most assists
            primary_pos = pos
            max_ast = 0
            for p, a in [("LF", lf_ast), ("CF", cf_ast), ("RF", rf_ast)]:
                if not pd.isna(a) and a > max_ast:
                    max_ast = a
                    primary_pos = p

            # Calculate arm rating based on assists (fewer assists = weaker arm)
            # Note: This is imperfect since low assists could mean good arm (runners don't test)
            # or weak arm (few opportunities). Use with other context.
            arm_rating = "-"
            if not pd.isna(of_ast):
                if of_ast >= 5:
                    arm_rating = "Strong"
                elif of_ast >= 3:
                    arm_rating = "Average"
                elif of_ast >= 1:
                    arm_rating = "Weak"
                else:
                    arm_rating = "Untested"

            of_rows.append({
                "Player": player,
                "Pos": primary_pos,
                "Total Assists": int(of_ast) if not pd.isna(of_ast) else 0,
                "LF": int(lf_ast) if not pd.isna(lf_ast) else 0,
                "CF": int(cf_ast) if not pd.isna(cf_ast) else 0,
                "RF": int(rf_ast) if not pd.isna(rf_ast) else 0,
                "Throw Errors": int(throw_errors) if not pd.isna(throw_errors) else 0,
                "Arm Rating": arm_rating,
            })

        if of_rows:
            # Sort by total assists (ascending — weak arms first)
            of_rows = sorted(of_rows, key=lambda x: x["Total Assists"])

            of_df = pd.DataFrame(of_rows)
            st.dataframe(of_df, use_container_width=True, hide_index=True)

            # Highlight weak arms
            weak_arms = [r for r in of_rows if r["Arm Rating"] in ["Weak", "Untested"] and r["Total Assists"] <= 1]
            if weak_arms:
                names = ", ".join([r["Player"] for r in weak_arms[:3]])
                st.success(f"**Potential weak arms:** {names} — consider taking extra bases")
        else:
            st.info("No outfield throwing data available.")
    else:
        st.info("No outfield throwing data available for this team.")

    # ══════════════════════════════════════════════════════════════════════════
    # Section 6: STEAL RECOMMENDATIONS (Combined Analysis)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    section_header("Steal Recommendations")
    st.caption("Combined analysis — catcher + pitcher + count = best steal opportunities")

    recommendations = []

    # Build recommendations based on catcher analysis
    if catcher_rows:
        for row in catcher_rows:
            if row["Rating"] == "GREEN LIGHT":
                if row["SB%"] != "-":
                    msg = f"**GREEN LIGHT** vs catcher **{row['Catcher']}** — Runners succeed {row['SB%']} of the time"
                else:
                    msg = f"**GREEN LIGHT** vs catcher **{row['Catcher']}** — Slow pop time ({row['Pop Time']})"
                recommendations.append({"type": "green", "message": msg})
            elif row["Rating"] == "Avoid":
                if row["SB%"] != "-":
                    msg = f"**AVOID** stealing vs catcher **{row['Catcher']}** — Only {row['SB%']} success rate"
                else:
                    msg = f"**AVOID** stealing vs catcher **{row['Catcher']}** — Quick pop time ({row['Pop Time']})"
                recommendations.append({"type": "red", "message": msg})

    # Add pitcher-specific recommendations based on SB allowed
    if pitcher_rows:
        for row in pitcher_rows:
            if row["Rating"] == "HIGH VULN":
                if row["SB%"] != "-" and row["SB Allowed"] != "-":
                    msg = f"**GREEN LIGHT** vs **{row['Pitcher']}** — {row['SB Allowed']} SB allowed ({row['SB%']} success rate)"
                elif row["SB Allowed"] != "-":
                    msg = f"**GREEN LIGHT** vs **{row['Pitcher']}** — {row['SB Allowed']} SB allowed"
                else:
                    continue  # Skip if no data
                recommendations.append({"type": "green", "message": msg})
            elif row["Rating"] == "Good":
                recommendations.append({
                    "type": "red",
                    "message": f"**CAUTION** vs **{row['Pitcher']}** — Good at controlling running game",
                })

    if recommendations:
        # Group by type
        greens = [r for r in recommendations if r["type"] == "green"]
        reds = [r for r in recommendations if r["type"] == "red"]

        if greens:
            for rec in greens[:5]:  # Limit to top 5
                st.success(rec["message"])

        if reds:
            for rec in reds[:5]:  # Limit to top 5
                st.error(rec["message"])
    else:
        st.info("Not enough data to generate specific recommendations. Scout live to assess delivery times and arm strength.")
