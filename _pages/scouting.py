"""Opponent Scouting — game plans, pitcher/hitter reports, scoring engine."""
import math
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
    filter_davidson, filter_minor_pitches, normalize_pitch_types,
    in_zone_mask, is_barrel_mask, display_name, get_percentile,
    safe_mode, _is_position_player,
)
from data.loader import (
    get_all_seasons, load_davidson_data, _load_truemedia,
    _tm_team, _tm_player, _safe_val, _safe_pct, _safe_num, _tm_pctile,
    _hitter_narrative, _pitcher_narrative,
)
from data.stats import compute_batter_stats, compute_pitcher_stats, _build_batter_zones
from data.population import compute_batter_stats_pop, compute_pitcher_stats_pop, compute_stuff_baselines, build_tunnel_population_pop
from viz.layout import CHART_LAYOUT, section_header
from viz.charts import (
    add_strike_zone, make_spray_chart, make_movement_profile,
    make_pitch_location_heatmap, player_header, _safe_pr, _safe_pop,
    _add_grid_zone_outline,
)
from viz.percentiles import savant_color, render_savant_percentile_section
from analytics.stuff_plus import _compute_stuff_plus, _compute_stuff_plus_all
from analytics.tunnel import _compute_tunnel_score, _build_tunnel_population
from analytics.command_plus import _compute_command_plus, _compute_pitch_pair_results


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

def _get_opp_hitter_profile(tm, hitter, team):
    """Extract opponent hitter vulnerability profile from TrueMedia."""
    rate = _tm_player(_tm_team(tm["hitting"]["rate"], team), hitter)
    pr = _tm_player(_tm_team(tm["hitting"]["pitch_rates"], team), hitter)
    pt = _tm_player(_tm_team(tm["hitting"]["pitch_types"], team), hitter)
    exit_d = _tm_player(_tm_team(tm["hitting"]["exit"], team), hitter)
    ht = _tm_player(_tm_team(tm["hitting"]["hit_types"], team), hitter)
    hl = _tm_player(_tm_team(tm["hitting"]["hit_locations"], team), hitter)
    pl = _tm_player(_tm_team(tm["hitting"].get("pitch_locations", pd.DataFrame()), team), hitter) if "pitch_locations" in tm["hitting"] else pd.DataFrame()
    sw = _tm_player(_tm_team(tm["hitting"].get("swing_stats", pd.DataFrame()), team), hitter) if "swing_stats" in tm["hitting"] else pd.DataFrame()
    fp = _tm_player(_tm_team(tm["hitting"].get("swing_pct", pd.DataFrame()), team), hitter) if "swing_pct" in tm["hitting"] else pd.DataFrame()
    profile = {
        "name": hitter,
        "bats": rate.iloc[0].get("batsHand", "?") if not rate.empty else "?",
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
    if not pt.empty:
        for tm_col, trackman_name in TM_PITCH_PCT_COLS.items():
            v = _safe_num(pt, tm_col)
            if not pd.isna(v) and v > 0:
                profile["pitch_type_pcts"][trackman_name] = v
    return profile


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
            ("glove", loc_grp["PlateLocSide"] > 0 if throws == "Right" else loc_grp["PlateLocSide"] < 0),
            ("arm", loc_grp["PlateLocSide"] <= 0 if throws == "Right" else loc_grp["PlateLocSide"] >= 0),
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
            "stuff_plus": stuff_plus,
            "count": len(grp),
            "eff_velo": eff_velo,
            "extension": ext_val,
            "zone_eff": zone_eff,
        }
    # Tunnel scores and pitch pair sequencing results
    arsenal["tunnels"] = _compute_tunnel_score(pdf, tunnel_pop=tunnel_pop)
    arsenal["sequences"] = _compute_pitch_pair_results(pdf, data)
    return arsenal


def _get_opp_pitcher_profile(tm, pitcher_name, team):
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
    profile = {
        "name": pitcher_name,
        "throws": trad.iloc[0].get("throwsHand", "?") if not trad.empty else "?",
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
    if not pt.empty:
        for tm_col, trackman_name in TM_PITCH_PCT_COLS.items():
            v = _safe_num(pt, tm_col)
            if not pd.isna(v) and v > 0:
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
    zone_mid_height = (ZONE_HEIGHT_BOT + ZONE_HEIGHT_TOP) / 2  # ~2.5 ft
    zone_quads = {
        "up_in":    loc_df[(loc_df["PlateLocHeight"] >= zone_mid_height) & (loc_df["PlateLocSide"] <= 0)],
        "up_away":  loc_df[(loc_df["PlateLocHeight"] >= zone_mid_height) & (loc_df["PlateLocSide"] > 0)],
        "down_in":  loc_df[(loc_df["PlateLocHeight"] < zone_mid_height) & (loc_df["PlateLocHeight"] >= ZONE_HEIGHT_BOT) & (loc_df["PlateLocSide"] <= 0)],
        "down_away":loc_df[(loc_df["PlateLocHeight"] < zone_mid_height) & (loc_df["PlateLocHeight"] >= ZONE_HEIGHT_BOT) & (loc_df["PlateLocSide"] > 0)],
        "heart":    loc_df[(loc_df["PlateLocHeight"].between(zone_mid_height - 0.4, zone_mid_height + 0.4)) & (loc_df["PlateLocSide"].abs() <= 0.4)],
        "chase_up": loc_df[loc_df["PlateLocHeight"] > ZONE_HEIGHT_TOP],
        "chase_down":loc_df[loc_df["PlateLocHeight"] < ZONE_HEIGHT_BOT],
        "chase_in": loc_df[(loc_df["PlateLocSide"] < -ZONE_SIDE) & loc_df["PlateLocHeight"].between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP)],
        "chase_away":loc_df[(loc_df["PlateLocSide"] > ZONE_SIDE) & loc_df["PlateLocHeight"].between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP)],
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

    # ── Swing Path Metrics (from hit lab) ──
    inplay_full = bdf[bdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
    inplay_la = inplay_full.dropna(subset=["Angle"])
    sp = {}
    if len(inplay_la) >= 10:
        ev_75 = inplay_la["ExitSpeed"].quantile(0.75)
        hard_hit = inplay_la[inplay_la["ExitSpeed"] >= ev_75]
        if len(hard_hit) >= 5:
            sp["attack_angle"] = hard_hit["Angle"].median()
            sp["avg_la_all"] = inplay_la["Angle"].median()
            # Swing type classification
            aa = sp["attack_angle"]
            if aa > 20:
                sp["swing_type"] = "Steep Uppercut"
            elif aa > 14:
                sp["swing_type"] = "Lift-Oriented"
            elif aa > 8:
                sp["swing_type"] = "Slight Uppercut"
            elif aa > 2:
                sp["swing_type"] = "Level"
            else:
                sp["swing_type"] = "Downward/Chopper"
            # Bat speed proxy — empirical: EV ≈ 0.2*PS + 1.2*BS
            if "RelSpeed" in hard_hit.columns:
                hh_sp = hard_hit.dropna(subset=["RelSpeed"])
                if len(hh_sp) > 0:
                    bs = (hh_sp["ExitSpeed"] - 0.2 * hh_sp["RelSpeed"]) / 1.2
                    sp["bat_speed_avg"] = bs.mean()
                    sp["bat_speed_max"] = bs.max()
            # Contact depth
            if "EffectiveVelo" in hard_hit.columns and "RelSpeed" in hard_hit.columns:
                cd_df = hard_hit.dropna(subset=["EffectiveVelo", "RelSpeed"])
                if len(cd_df) > 0:
                    depth_val = (cd_df["EffectiveVelo"] - cd_df["RelSpeed"]).mean()
                    sp["contact_depth"] = depth_val
                    sp["depth_label"] = "Out Front" if depth_val > 1.5 else ("Deep Contact" if depth_val < -1.5 else "Neutral")
                else:
                    sp["contact_depth"] = np.nan
                    sp["depth_label"] = "Unknown"
            # Path adjust: how LA changes with pitch height
            hh_loc = hard_hit.dropna(subset=["PlateLocHeight"])
            if len(hh_loc) >= 8:
                from scipy import stats as sp_stats
                slope, _, _, _, _ = sp_stats.linregress(hh_loc["PlateLocHeight"], hh_loc["Angle"])
                sp["path_adjust"] = slope  # degrees per foot
            # Per-pitch-type swing path
            sp_by_pt = {}
            for pt_name2, ptg in hard_hit.groupby("TaggedPitchType"):
                if len(ptg) < 3:
                    continue
                pt_ip = ptg.dropna(subset=["ExitSpeed"])
                sp_by_pt[pt_name2] = {
                    "hard_hit_la": ptg["Angle"].median(),
                    "hard_hit_ev": pt_ip["ExitSpeed"].mean() if len(pt_ip) > 0 else np.nan,
                }
                if "RelSpeed" in ptg.columns:
                    ptg_sp = ptg.dropna(subset=["RelSpeed"])
                    if len(ptg_sp) > 0:
                        sp_by_pt[pt_name2]["bat_speed"] = ((ptg_sp["ExitSpeed"] - 0.2 * ptg_sp["RelSpeed"]) / 1.2).mean()
            sp["by_pitch_type"] = sp_by_pt
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

    # 4. Their 2K Whiff Rate (13%) — matched to pitch class (hard/offspeed) + our hand
    their_2k = hd.get("whiff_2k_hard" if is_hard else "whiff_2k_os", np.nan)
    if not pd.isna(their_2k):
        components.append(min(their_2k / 40 * 100, 100)); weights.append(13)

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

    # 9. K-Prone Factor (4%) — high K hitters are exploitable
    k_pct = hd.get("k_pct", np.nan)
    if not pd.isna(k_pct):
        k_score = min(max((k_pct - 10) / 25 * 100, 0), 100)
        if not is_hard:
            k_score = min(k_score * 1.25, 100)
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

    # 16. Our Usage (7%) — pitches we actually throw should rank higher; low-usage pitches
    #     have small samples. But cap the penalty so quality secondaries can still surface.
    usage_pct = ars_pt.get("usage_pct", pt_data.get("usage", np.nan))
    if not pd.isna(usage_pct):
        # 0% → 0, 10% → 30, 20% → 50, 35%+ → 80, 55%+ → 100
        # Floor of 30 for any pitch >= 10% usage so secondaries aren't crushed
        raw_usage = min(max(usage_pct / 55 * 100, 0), 100)
        if usage_pct >= 10:
            raw_usage = max(raw_usage, 30)
        components.append(raw_usage); weights.append(7)

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

def _game_plan_tab(tm, team, data):
    """Game Plan tab — unified pitching + hitting plans."""
    section_header(f"Game Plan vs {team}")
    st.caption("Cross-referencing TrueMedia season data with our Trackman pitch-level data")
    seasons = get_all_seasons()
    season_filter = st.multiselect("Our Trackman Seasons", seasons, default=seasons, key="gpl_season")
    tab_pitch, tab_hit = st.tabs(["Our Pitching Plan", "Our Hitting Plan"])
    with tab_pitch:
        try:
            _pitching_plan_content(tm, team, data, season_filter)
        except Exception as e:
            st.error(f"Pitching plan error: {e}")
    with tab_hit:
        try:
            _hitting_plan_content(tm, team, data, season_filter)
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


def _pitching_plan_content(tm, team, data, season_filter):
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
    all_matchups = []
    for _, row in opp_hitters.iterrows():
        hitter_name = row["playerFullName"]
        profile = _get_opp_hitter_profile(tm, hitter_name, team)
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


def _hitting_plan_content(tm, team, data, season_filter):
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
    opp_profile = _get_opp_pitcher_profile(tm, selected_opp, team)
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

    tm = _load_truemedia()
    # Get all TrueMedia team names
    all_tm_teams = set()
    for role in ["hitting", "pitching"]:
        for key, df in tm[role].items():
            if "newestTeamName" in df.columns:
                all_tm_teams.update(df["newestTeamName"].dropna().unique())
    all_tm_teams.discard("Davidson College")
    all_tm_teams = sorted(all_tm_teams)

    if not all_tm_teams:
        st.info("No TrueMedia data found.")
        return

    team = st.selectbox("Opponent", all_tm_teams, key="sc_team_tm")

    tab_overview, tab_hitters, tab_pitchers, tab_catchers, tab_gameplan = st.tabs([
        "Team Overview", "Their Hitters", "Their Pitchers", "Their Catchers", "Game Plan"
    ])

    # ── Tab 1: Team Overview ──
    with tab_overview:
        _scouting_team_overview(tm, team)

    # ── Tab 2: Their Hitters ──
    with tab_hitters:
        _scouting_hitter_report(tm, team, data)

    # ── Tab 3: Their Pitchers ──
    with tab_pitchers:
        _scouting_pitcher_report(tm, team, data)

    # ── Tab 4: Their Catchers ──
    with tab_catchers:
        _scouting_catcher_report(tm, team)

    # ── Tab 5: Game Plan ──
    with tab_gameplan:
        _game_plan_tab(tm, team, data)


def _scouting_team_overview(tm, team):
    """Team Overview tab — aggregate offense + pitching metrics with D1 rankings."""
    h_cnt = _tm_team(tm["hitting"]["counting"], team)
    h_rate = _tm_team(tm["hitting"]["rate"], team)
    p_trad = _tm_team(tm["pitching"]["traditional"], team)

    # Build team-level aggregates across ALL D1 teams for ranking context
    all_h_cnt = tm["hitting"]["counting"]
    all_p_trad = tm["pitching"]["traditional"]

    @st.cache_data(show_spinner=False)
    def _team_agg_ranks(_h_cnt_json, _p_trad_json):
        """Compute team-level aggregates for all teams and return rank context."""
        h_cnt_all = pd.read_json(_h_cnt_json) if _h_cnt_json else pd.DataFrame()
        p_trad_all = pd.read_json(_p_trad_json) if _p_trad_json else pd.DataFrame()

        team_off = {}
        if not h_cnt_all.empty:
            for t, grp in h_cnt_all.groupby("newestTeamName"):
                pa = grp["PA"].sum()
                ab = grp["AB"].sum()
                team_off[t] = {
                    "BA": grp["H"].sum() / max(ab, 1),
                    "HR": grp["HR"].sum(),
                    "SB": grp["SB"].sum(),
                    "K%": grp["K"].sum() / max(pa, 1) * 100,
                    "BB%": grp["BB"].sum() / max(pa, 1) * 100,
                }

        team_pit = {}
        if not p_trad_all.empty:
            for t, grp in p_trad_all.groupby("newestTeamName"):
                ip = grp["IP"].sum()
                team_pit[t] = {
                    "ERA": grp["ER"].sum() / max(ip, 1) * 9 if "ER" in grp.columns else np.nan,
                    "K/9": grp["K"].sum() / max(ip, 1) * 9,
                    "BB/9": grp["BB"].sum() / max(ip, 1) * 9,
                    "IP": ip,
                }

        return team_off, team_pit

    h_json = all_h_cnt.to_json() if not all_h_cnt.empty else ""
    p_json = all_p_trad.to_json() if not all_p_trad.empty else ""
    team_off_all, team_pit_all = _team_agg_ranks(h_json, p_json)
    n_teams = max(len(team_off_all), len(team_pit_all))

    # ── Team Narrative ──
    team_narrative = []
    if team in team_off_all:
        off = team_off_all[team]
        # Rank this team's offense
        ba_rank = sum(1 for t in team_off_all.values() if t["BA"] > off["BA"]) + 1
        hr_rank = sum(1 for t in team_off_all.values() if t["HR"] > off["HR"]) + 1
        k_rank = sum(1 for t in team_off_all.values() if t["K%"] < off["K%"]) + 1  # lower is better
        team_narrative.append(
            f"**Offense:** {team} hits **.{int(off['BA']*1000):03d}** as a team "
            f"(#{ba_rank} of {n_teams} D1 teams), with **{off['HR']}** HR (#{hr_rank}) "
            f"and a **{off['K%']:.1f}%** K rate (#{k_rank})."
        )
        if off["K%"] > 22:
            team_narrative.append("This lineup is strikeout-prone — expand the zone and use putaway pitches.")
        elif off["BB%"] > 10:
            team_narrative.append("A patient lineup — throw strikes early and avoid falling behind.")
        if off["HR"] > 40:
            team_narrative.append("Power-heavy — keep the ball down and limit mistakes over the plate.")
        if off["SB"] > 50:
            team_narrative.append(f"Speed threat ({off['SB']} SB) — quick deliveries and catcher readiness critical.")

    if team in team_pit_all:
        pit = team_pit_all[team]
        era_rank = sum(1 for t in team_pit_all.values() if t["ERA"] < pit["ERA"]) + 1  # lower is better
        k9_rank = sum(1 for t in team_pit_all.values() if t["K/9"] > pit["K/9"]) + 1
        team_narrative.append(
            f"**Pitching:** Staff ERA of **{pit['ERA']:.2f}** (#{era_rank} of {n_teams}), "
            f"**{pit['K/9']:.1f}** K/9 (#{k9_rank}), **{pit['BB/9']:.1f}** BB/9."
        )
        if pit["ERA"] > 5.0:
            team_narrative.append("Hittable staff — be aggressive and attack early.")
        elif pit["ERA"] < 3.5:
            team_narrative.append("Elite pitching staff — need disciplined, quality at-bats.")
        if pit["BB/9"] > 4.0:
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
            d = h_rate[["playerFullName", "PA", "BA", "OBP", "SLG", "OPS"]].copy()
            d = d.sort_values("PA", ascending=False).head(10)
            # Add D1 OPS percentile
            d["OPS Pctile"] = d["OPS"].apply(
                lambda x: int(percentileofscore(all_rate["OPS"].dropna(), x, kind='rank')) if pd.notna(x) else np.nan
            )
            d.columns = ["Player", "PA", "BA", "OBP", "SLG", "OPS", "OPS %ile"]
            for c in ["BA", "OBP", "SLG", "OPS"]:
                d[c] = d[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
            d["OPS %ile"] = d["OPS %ile"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
            st.dataframe(d, use_container_width=True, hide_index=True)
    with col_tp:
        section_header("Top Pitchers (by IP)")
        if not p_trad.empty:
            d = p_trad[["playerFullName", "GS", "IP", "ERA", "FIP", "WHIP", "K/9"]].copy()
            d = d.sort_values("IP", ascending=False).head(10)
            # Add D1 ERA percentile (inverted — lower ERA = higher percentile)
            all_trad = tm["pitching"]["traditional"]
            d["ERA Pctile"] = d["ERA"].apply(
                lambda x: int(100 - percentileofscore(all_trad["ERA"].dropna(), x, kind='rank')) if pd.notna(x) else np.nan
            )
            d.columns = ["Player", "GS", "IP", "ERA", "FIP", "WHIP", "K/9", "ERA %ile"]
            for c in ["ERA", "FIP", "WHIP"]:
                d[c] = d[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            d["K/9"] = d["K/9"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
            d["ERA %ile"] = d["ERA %ile"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
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

        # Merge K%, BB% from pitch rates
        if not h_prates.empty:
            pr_cols = ["playerFullName"]
            for c in ["K%", "BB%", "Chase%", "SwStrk%"]:
                if c in h_prates.columns:
                    pr_cols.append(c)
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


def _scouting_hitter_report(tm, team, trackman_data):
    """Their Hitters tab — individual deep-dive scouting report with percentile context."""
    h_rate = _tm_team(tm["hitting"]["rate"], team)
    if h_rate.empty:
        st.info("No hitting data for this team.")
        return
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

    # All D1 data for percentile context
    all_h_rate = tm["hitting"]["rate"]
    all_h_exit = tm["hitting"]["exit"]
    all_h_pr = tm["hitting"]["pitch_rates"]
    all_h_ht = tm["hitting"]["hit_types"]
    all_h_hl = tm["hitting"]["hit_locations"]
    all_h_spd = tm["hitting"]["speed"]
    all_h_re = tm["hitting"]["run_expectancy"]
    all_h_swpct = tm["hitting"]["swing_pct"]
    all_h_swstats = tm["hitting"]["swing_stats"]

    # Header
    pos = rate.iloc[0].get("pos", "?") if not rate.empty else "?"
    bats = rate.iloc[0].get("batsHand", "?") if not rate.empty else "?"
    g = _safe_val(cnt, "G", "d")
    pa = _safe_val(cnt, "PA", "d")
    st.markdown(f"### {hitter}")
    st.caption(f"{pos} | Bats: {bats} | G: {g} | PA: {pa}")

    # ── Player Narrative ──
    narrative = _hitter_narrative(hitter, rate, exit_d, pr, ht, hl, spd,
                                  all_h_rate, all_h_exit, all_h_pr)
    st.markdown(narrative)

    n_hitters = len(all_h_rate)

    # ── Percentile Rankings (Savant-style bars) ──
    section_header("Percentile Rankings")
    st.caption(f"vs. {n_hitters:,} D1 hitters")
    hitting_metrics = [
        ("OPS", _safe_num(rate, "OPS"), _tm_pctile(rate, "OPS", all_h_rate), ".3f", True),
        ("wOBA", _safe_num(rate, "WOBA"), _tm_pctile(rate, "WOBA", all_h_rate), ".3f", True),
        ("BA", _safe_num(rate, "BA"), _tm_pctile(rate, "BA", all_h_rate), ".3f", True),
        ("SLG", _safe_num(rate, "SLG"), _tm_pctile(rate, "SLG", all_h_rate), ".3f", True),
        ("ISO", _safe_num(rate, "ISO"), _tm_pctile(rate, "ISO", all_h_rate), ".3f", True),
        ("Exit Velo", _safe_num(exit_d, "ExitVel"), _tm_pctile(exit_d, "ExitVel", all_h_exit), ".1f", True),
        ("Barrel %", _safe_num(exit_d, "Barrel%"), _tm_pctile(exit_d, "Barrel%", all_h_exit), ".1f", True),
        ("Hard Hit %", _safe_num(exit_d, "Hit95+%"), _tm_pctile(exit_d, "Hit95+%", all_h_exit), ".1f", True),
        ("K %", _safe_num(rate, "K%"), _tm_pctile(rate, "K%", all_h_rate), ".1f", False),
        ("BB %", _safe_num(rate, "BB%"), _tm_pctile(rate, "BB%", all_h_rate), ".1f", True),
        ("Chase %", _safe_num(pr, "Chase%"), _tm_pctile(pr, "Chase%", all_h_pr), ".1f", False),
        ("Whiff %", _safe_num(pr, "SwStrk%"), _tm_pctile(pr, "SwStrk%", all_h_pr), ".1f", False),
        ("Contact %", _safe_num(pr, "Contact%"), _tm_pctile(pr, "Contact%", all_h_pr), ".1f", True),
        ("Speed Score", _safe_num(spd, "SpeedScore"), _tm_pctile(spd, "SpeedScore", all_h_spd), ".1f", True),
    ]
    # Filter out metrics with nan values
    hitting_metrics = [(l, v, p, f, h) for l, v, p, f, h in hitting_metrics if not pd.isna(v)]
    render_savant_percentile_section(hitting_metrics)

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
                # From catcher view: FarLft = right side of chart, FarRt = left side
                angles_mid = [60, 30, 0, -30, -60]  # degrees from center field
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

    # ── Feature #6: Pitch-Type Matchup Matrix ──
    if not pt.empty:
        section_header("Pitch Type Matchup")
        st.caption("What pitches this hitter sees and how often — plan your attack around weaknesses")
        pitch_cols = ["4Seam%", "Sink2Seam%", "Cutter%", "Slider%", "Curve%", "Change%", "Split%", "Sweeper%"]
        pitch_labels = ["4-Seam", "Sinker", "Cutter", "Slider", "Curve", "Change", "Splitter", "Sweeper"]
        count_cols = ["4Seam#", "Sink2Seam#", "Cutter#", "Slider#", "Curve#", "Change#", "Split#", "Sweeper#"]
        vals = []
        for pct_col, cnt_col, lbl in zip(pitch_cols, count_cols, pitch_labels):
            pct_v = pt.iloc[0].get(pct_col) if not pt.empty else None
            cnt_v = h_ptcounts.iloc[0].get(cnt_col) if not h_ptcounts.empty and cnt_col in h_ptcounts.columns else None
            if pct_v is not None and not pd.isna(pct_v) and pct_v > 0:
                row = {"Pitch": lbl, "% Seen": f"{pct_v:.1f}%"}
                if cnt_v is not None and not pd.isna(cnt_v):
                    row["Count"] = f"{int(cnt_v)}"
                else:
                    row["Count"] = "-"
                vals.append(row)
        if vals:
            # Bar chart + table side by side
            col_chart, col_tbl = st.columns([3, 2])
            with col_chart:
                vdf = pd.DataFrame(vals)
                pct_vals = [float(v["% Seen"].rstrip("%")) for v in vals]
                fig = go.Figure(go.Bar(
                    x=[v["Pitch"] for v in vals], y=pct_vals,
                    marker_color=[PITCH_COLORS.get(v["Pitch"].replace("-", ""), "#888") for v in vals],
                    text=[v["% Seen"] for v in vals], textposition="outside",
                ))
                fig.update_layout(**CHART_LAYOUT, height=280, yaxis_title="% Seen", showlegend=False,
                                  yaxis=dict(range=[0, max(pct_vals) * 1.3]))
                st.plotly_chart(fig, use_container_width=True)
            with col_tbl:
                st.dataframe(pd.DataFrame(vals), use_container_width=True, hide_index=True)

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

    # ── Swing Tendencies by Pitch Type ──
    has_swing_data = not h_swpct.empty or not h_swstats.empty
    if has_swing_data:
        section_header("Swing Tendencies by Pitch Type")
        st.caption("How aggressively this hitter swings at different pitch types — key for pitch sequencing")

        swing_col1, swing_col2 = st.columns(2)

        with swing_col1:
            # Swing% by pitch type from 1P Swing%.csv
            swing_rates = []
            for lbl, col, color in [
                ("Fastball", "Swing% vs Hard", "#d22d49"),
                ("Slider", "Swing% vs SL", "#f7c631"),
                ("Curveball", "Swing% vs CB", "#00d1ed"),
                ("Changeup", "Swing% vs CH", "#1dbe3a"),
            ]:
                v = _safe_num(h_swpct, col)
                if not pd.isna(v):
                    pct = _tm_pctile(h_swpct, col, all_h_swpct)
                    swing_rates.append({"Pitch": lbl, "Swing%": v, "Pctile": pct, "Color": color})

            if swing_rates:
                st.markdown("**Swing Rate by Pitch Type**")
                # Bar chart
                sr_df = pd.DataFrame(swing_rates)
                fig_sw = go.Figure()
                fig_sw.add_trace(go.Bar(
                    x=sr_df["Pitch"], y=sr_df["Swing%"],
                    marker_color=sr_df["Color"],
                    text=sr_df["Swing%"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside", textfont=dict(size=11, color="#000000"),
                ))
                fig_sw.update_layout(
                    **CHART_LAYOUT, height=280,
                    yaxis=dict(title="Swing %", range=[0, max(sr_df["Swing%"].max() * 1.2, 60)]),
                    xaxis=dict(title=""),
                    showlegend=False,
                )
                st.plotly_chart(fig_sw, use_container_width=True)

                # Detail table with percentiles
                tbl_data = []
                for r in swing_rates:
                    pct_str = f"{int(r['Pctile'])}th" if not pd.isna(r["Pctile"]) else "-"
                    tbl_data.append({
                        "Pitch": r["Pitch"],
                        "Swing%": f"{r['Swing%']:.1f}%",
                        "D1 %ile": pct_str,
                    })
                st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True)

                # Narrative: identify vulnerability
                sorted_rates = sorted(swing_rates, key=lambda x: x["Swing%"], reverse=True)
                highest = sorted_rates[0]
                lowest = sorted_rates[-1]
                if highest["Swing%"] - lowest["Swing%"] > 10:
                    st.caption(
                        f"⚡ Swings most aggressively at **{highest['Pitch']}** ({highest['Swing%']:.1f}%) "
                        f"and least at **{lowest['Pitch']}** ({lowest['Swing%']:.1f}%). "
                        f"Tunnel {highest['Pitch'].lower()} look early, then use "
                        f"{lowest['Pitch'].lower()} to steal strikes."
                    )

        with swing_col2:
            # First-pitch swing tendencies
            fp_data = []
            fp_hard = _safe_num(h_swpct, "1PSwing% vs Hard Empty")
            if pd.isna(fp_hard):
                fp_hard = _safe_num(h_swstats, "1PSwing% vs Hard Empty")
            fp_ch = _safe_num(h_swstats, "1PSwing% vs CH Empty")

            if not pd.isna(fp_hard) or not pd.isna(fp_ch):
                st.markdown("**First-Pitch Aggressiveness**")
                fp_items = []
                if not pd.isna(fp_hard):
                    pct_fp_hard = _tm_pctile(h_swstats if not h_swstats.empty else h_swpct,
                                              "1PSwing% vs Hard Empty",
                                              all_h_swstats if not all_h_swstats.empty else all_h_swpct)
                    pct_str = f" ({int(pct_fp_hard)}th %ile)" if not pd.isna(pct_fp_hard) else ""
                    fp_items.append({"Situation": "1st Pitch vs Fastball (empty)", "Swing%": f"{fp_hard:.0f}%", "D1 %ile": pct_str.strip()})
                if not pd.isna(fp_ch):
                    pct_fp_ch = _tm_pctile(h_swstats, "1PSwing% vs CH Empty", all_h_swstats)
                    pct_str = f" ({int(pct_fp_ch)}th %ile)" if not pd.isna(pct_fp_ch) else ""
                    fp_items.append({"Situation": "1st Pitch vs Changeup (empty)", "Swing%": f"{fp_ch:.0f}%", "D1 %ile": pct_str.strip()})
                if fp_items:
                    st.dataframe(pd.DataFrame(fp_items), use_container_width=True, hide_index=True)

                # First-pitch narrative
                if not pd.isna(fp_hard):
                    if fp_hard >= 40:
                        st.caption(f"🔴 Very aggressive first-pitch hitter vs fastballs ({fp_hard:.0f}%) — start offspeed or off the plate.")
                    elif fp_hard <= 20:
                        st.caption(f"🟢 Patient first-pitch approach vs fastballs ({fp_hard:.0f}%) — can attack zone early with strikes.")
                    else:
                        st.caption(f"⚪ Moderate first-pitch approach ({fp_hard:.0f}%) — varies game to game.")

            # InZone Swing%, Chase%, Contact% from Swing stats
            if not h_swstats.empty:
                disc_from_sw = []
                for lbl, col in [("In-Zone Swing%", "InZoneSwing%"), ("Chase%", "Chase%"), ("Contact%", "Contact%")]:
                    v = _safe_num(h_swstats, col)
                    if not pd.isna(v):
                        pct = _tm_pctile(h_swstats, col, all_h_swstats)
                        pct_str = f"{int(pct)}th" if not pd.isna(pct) else "-"
                        disc_from_sw.append({"Metric": lbl, "Value": f"{v:.1f}%", "D1 %ile": pct_str})
                if disc_from_sw:
                    st.markdown("**Zone Discipline (Swing Stats)**")
                    st.dataframe(pd.DataFrame(disc_from_sw), use_container_width=True, hide_index=True)

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

    # ── How to Attack ──
    section_header("How to Attack")
    notes = _hitter_attack_plan(rate, exit_d, pr, ht, hl)

    # Enhance attack plan with swing data
    if not h_swpct.empty:
        sw_hard = _safe_num(h_swpct, "Swing% vs Hard")
        sw_sl = _safe_num(h_swpct, "Swing% vs SL")
        sw_cb = _safe_num(h_swpct, "Swing% vs CB")
        sw_ch = _safe_num(h_swpct, "Swing% vs CH")
        if not pd.isna(sw_hard) and not pd.isna(sw_cb):
            if sw_cb < sw_hard - 15:
                notes.append(f"Low curveball swing rate ({sw_cb:.1f}% vs {sw_hard:.1f}% FB) — use curves to steal strikes, set up fastball.")
        if not pd.isna(sw_ch) and not pd.isna(sw_hard):
            if sw_ch >= sw_hard:
                notes.append(f"Chases changeups at high rate ({sw_ch:.1f}%) — use CH as putaway pitch.")

    if not h_swstats.empty:
        fp_sw = _safe_num(h_swstats, "1PSwing% vs Hard Empty")
        if pd.isna(fp_sw):
            fp_sw = _safe_num(h_swpct, "1PSwing% vs Hard Empty")
        if not pd.isna(fp_sw) and fp_sw >= 40:
            notes.append(f"Aggressive first-pitch swinger ({fp_sw:.0f}%) — start offspeed or off the plate to steal strike one.")
        elif not pd.isna(fp_sw) and fp_sw <= 18:
            notes.append(f"Takes first pitch often ({fp_sw:.0f}%) — attack the zone early with a strike.")

        # 2K whiff recommendations
        rhp_hard = _safe_num(h_swstats, "2K Whiff vs RHP Hard")
        rhp_os = _safe_num(h_swstats, "2K Whiff vs RHP OS")
        if not pd.isna(rhp_hard) and not pd.isna(rhp_os):
            if rhp_os > rhp_hard and rhp_os > 25:
                notes.append(f"With 2 strikes (vs RHP): finish with offspeed ({rhp_os:.1f}% whiff) — breaking ball is the out-pitch.")
            elif rhp_hard > rhp_os and rhp_hard > 25:
                notes.append(f"With 2 strikes (vs RHP): finish with hard stuff ({rhp_hard:.1f}% whiff) — elevated fastball to finish.")

    for n in notes:
        st.markdown(f"- {n}")

    # ── Trackman Overlay ──
    _trackman_hitter_overlay(trackman_data, hitter)


def _trackman_hitter_overlay(data, hitter_name):
    """Show Trackman swing heatmap if we have pitch-level data for this hitter."""
    # Match by last name
    last_name = hitter_name.split()[-1] if " " in hitter_name else hitter_name
    matches = data[data["Batter"].str.contains(last_name, case=False, na=False)]
    if matches.empty or len(matches) < 10:
        return
    section_header("Trackman Data Overlay")
    st.caption(f"Pitch-level data from our Trackman system ({len(matches)} pitches)")
    swings = matches[matches["PitchCall"].isin(SWING_CALLS)]
    loc = swings.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if not loc.empty and len(loc) >= 5:
        fig = px.density_heatmap(loc, x="PlateLocSide", y="PlateLocHeight", nbinsx=12, nbinsy=12,
                                 color_continuous_scale="YlOrRd")
        add_strike_zone(fig)
        fig.update_layout(title="Swing Locations", xaxis=dict(range=[-3, 3], scaleanchor="y"),
                          yaxis=dict(range=[0, 5]),
                          height=400, coloraxis_showscale=False, **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


def _scouting_pitcher_report(tm, team, trackman_data):
    """Their Pitchers tab — comprehensive scouting report built for game-planning."""
    p_trad = _tm_team(tm["pitching"]["traditional"], team)
    if p_trad.empty:
        st.info("No pitching data for this team.")
        return
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

    # All D1 data for percentile context
    all_p_trad = tm["pitching"]["traditional"]
    all_p_rate = tm["pitching"]["rate"]
    all_p_mov = tm["pitching"]["movement"]
    all_p_pr = tm["pitching"]["pitch_rates"]
    all_p_exit = tm["pitching"]["exit"]
    all_p_ht = tm["pitching"]["hit_types"]
    all_p_ploc = tm["pitching"]["pitch_locations"]
    all_p_xrate = tm["pitching"]["expected_rate"]
    all_p_hr = tm["pitching"]["home_runs"]

    # ── Header ──
    throws = trad.iloc[0].get("throwsHand", "?") if not trad.empty else "?"
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

    n_pitchers = len(all_p_trad)

    # ══════════════════════════════════════════════════════════
    # SECTION 1: PERCENTILE RANKINGS
    # ══════════════════════════════════════════════════════════
    section_header("Percentile Rankings")
    st.caption(f"vs. {n_pitchers:,} D1 pitchers")
    pitching_metrics = [
        ("ERA", _safe_num(trad, "ERA"), _tm_pctile(trad, "ERA", all_p_trad), ".2f", False),
        ("FIP", _safe_num(trad, "FIP"), _tm_pctile(trad, "FIP", all_p_trad), ".2f", False),
        ("xFIP", _safe_num(rate, "xFIP"), _tm_pctile(rate, "xFIP", all_p_rate), ".2f", False),
        ("WHIP", _safe_num(trad, "WHIP"), _tm_pctile(trad, "WHIP", all_p_trad), ".2f", False),
        ("K/9", _safe_num(trad, "K/9"), _tm_pctile(trad, "K/9", all_p_trad), ".1f", True),
        ("BB/9", _safe_num(trad, "BB/9"), _tm_pctile(trad, "BB/9", all_p_trad), ".1f", False),
        ("K/BB", _safe_num(trad, "K/BB"), _tm_pctile(trad, "K/BB", all_p_trad), ".2f", True),
        ("Velo", _safe_num(mov, "Vel"), _tm_pctile(mov, "Vel", all_p_mov), ".1f", True),
        ("Spin", _safe_num(mov, "Spin"), _tm_pctile(mov, "Spin", all_p_mov), ".0f", True),
        ("Extension", _safe_num(mov, "Extension"), _tm_pctile(mov, "Extension", all_p_mov), ".1f", True),
        ("Chase %", _safe_num(pr, "Chase%"), _tm_pctile(pr, "Chase%", all_p_pr), ".1f", True),
        ("SwStrk %", _safe_num(pr, "SwStrk%"), _tm_pctile(pr, "SwStrk%", all_p_pr), ".1f", True),
        ("EV Against", _safe_num(exit_d, "ExitVel"), _tm_pctile(exit_d, "ExitVel", all_p_exit), ".1f", False),
        ("Barrel %", _safe_num(exit_d, "Barrel%"), _tm_pctile(exit_d, "Barrel%", all_p_exit), ".1f", False),
        ("GB %", _safe_num(ht, "Ground%"), _tm_pctile(ht, "Ground%", all_p_ht), ".1f", True),
        ("wOBA Agn", _safe_num(rate, "wOBA"), _tm_pctile(rate, "wOBA", all_p_rate), ".3f", False),
        ("LOB %", _safe_num(rate, "LOB%"), _tm_pctile(rate, "LOB%", all_p_rate), ".1f", True),
        ("HR/9", _safe_num(trad, "HR/9"), _tm_pctile(trad, "HR/9", all_p_trad), ".2f", False),
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
                    if pct_v is not None and not pd.isna(pct_v) and pct_v > 0:
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
    # SECTION 3: COMMAND PROFILE
    # ══════════════════════════════════════════════════════════
    if not ploc.empty or not pr.empty:
        section_header("Command Profile")

        cmd_col1, cmd_col2 = st.columns([3, 2])

        with cmd_col1:
            # 3x3 Heatmap
            if not ploc.empty:
                high = _safe_num(ploc, "High%")
                vmid = _safe_num(ploc, "VMid%")
                low = _safe_num(ploc, "Low%")
                inside = _safe_num(ploc, "Inside%")
                hmid = _safe_num(ploc, "HMid%")
                outside = _safe_num(ploc, "Outside%")
                vert = [high, vmid, low]
                horiz = [inside, hmid, outside]
                if all(not pd.isna(v) for v in vert) and all(not pd.isna(v) for v in horiz):
                    vert_total = sum(vert)
                    horiz_total = sum(horiz)
                    z_matrix = []
                    for v_val in vert:
                        row = []
                        for h_val in horiz:
                            cell_val = (v_val / max(vert_total, 1)) * (h_val / max(horiz_total, 1)) * 100
                            row.append(round(cell_val, 1))
                        z_matrix.append(row)
                    fig_hm = go.Figure(data=go.Heatmap(
                        z=z_matrix,
                        x=["Inside", "Middle", "Outside"],
                        y=["High", "Middle", "Low"],
                        colorscale=[[0, "#f0f4f8"], [0.5, "#3d7dab"], [1, "#14365d"]],
                        showscale=False,
                        text=[[f"{v:.1f}%" for v in row] for row in z_matrix],
                        texttemplate="%{text}",
                        textfont=dict(size=14, color="white"),
                        hovertemplate="Zone: %{y} %{x}<br>Frequency: %{text}<extra></extra>",
                    ))
                    fig_hm.add_shape(type="rect", x0=-0.5, y0=-0.5, x1=2.5, y1=2.5,
                                     line=dict(color="#000000", width=3))
                    fig_hm.update_layout(
                        **CHART_LAYOUT, height=280,
                        xaxis=dict(side="bottom"), yaxis=dict(autorange="reversed"),
                    )
                    fig_hm.update_layout(margin=dict(l=60, r=10, t=10, b=40))
                    st.plotly_chart(fig_hm, use_container_width=True)

        with cmd_col2:
            # Command rates with percentiles
            cmd_data = []
            # From pitch rates
            for lbl, src, col in [
                ("In Zone %", ploc, "InZone%"), ("CompLoc %", pr, "CompLoc%"),
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
            inzone = _safe_num(ploc, "InZone%") if not ploc.empty else np.nan
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
    # SECTION 3B: PLATOON SPLITS (from Trackman data)
    # ══════════════════════════════════════════════════════════
    if trackman_data is not None and not trackman_data.empty:
        last_name_ps = pitcher.split()[-1] if " " in pitcher else pitcher
        p_tm = trackman_data[trackman_data["Pitcher"].str.contains(last_name_ps, case=False, na=False)]
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
                st.caption(f"Derived from Trackman pitch-level data ({l_stats.get('n_pitches', 0)} pitches vs LHH, "
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
    # SECTION 4: RESULTS AGAINST (batted ball + exit velo + expected)
    # ══════════════════════════════════════════════════════════
    has_results = not exit_d.empty or not ht.empty or not xrate.empty
    if has_results:
        section_header("Results Against")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            # Batted ball & exit velo combined
            if not exit_d.empty or not ht.empty:
                st.markdown("**Contact Quality Allowed**")
                cq_data = []
                for lbl, src, col, fmt, hib in [
                    ("Exit Velo", exit_d, "ExitVel", ".1f", False),
                    ("Barrel %", exit_d, "Barrel%", ".1f", False),
                    ("Hard Hit %", exit_d, "HardOut", ".1f", False),
                    ("Launch Angle", exit_d, "LaunchAng", ".1f", None),
                    ("GB %", ht, "Ground%", ".1f", True),
                    ("FB %", ht, "Fly%", ".1f", False),
                    ("LD %", ht, "Line%", ".1f", False),
                    ("HR/FB", p_hr, "HR/FB", ".1f", False),
                ]:
                    v = _safe_num(src, col)
                    if not pd.isna(v):
                        all_src = all_p_exit if src is exit_d else (all_p_ht if src is ht else all_p_hr)
                        pct = _tm_pctile(src, col, all_src)
                        pct_str = f"{int(pct)}" if not pd.isna(pct) else "-"
                        cq_data.append({"Metric": lbl, "Value": f"{v:{fmt}}", "%ile": pct_str})
                if cq_data:
                    st.dataframe(pd.DataFrame(cq_data), use_container_width=True, hide_index=True)

        with res_col2:
            # Expected stats + spray
            if not xrate.empty:
                st.markdown("**Expected Stats Against**")
                xr_data = []
                for lbl, col in [("BA", "AVG"), ("xAVG", "xAVG"), ("SLG", "SLG"), ("xSLG", "xSLG"),
                                  ("wOBA", "wOBA"), ("xWOBA", "xWOBA"), ("BABIP", "BABIP")]:
                    v = _safe_num(xrate, col)
                    if pd.isna(v) and not rate.empty:
                        v = _safe_num(rate, col)
                    if not pd.isna(v):
                        xr_data.append({"Stat": lbl, "Value": f"{v:.3f}"})
                if xr_data:
                    st.dataframe(pd.DataFrame(xr_data), use_container_width=True, hide_index=True)

            # Spray direction
            if not hl.empty:
                spray_data = []
                for lbl, col in [("Pull %", "HPull%"), ("Center %", "HCtr%"), ("Oppo %", "HOppFld%")]:
                    v = _safe_num(hl, col)
                    if not pd.isna(v):
                        spray_data.append({"Dir": lbl, "Rate": f"{v:.1f}%"})
                if spray_data:
                    st.markdown("**Spray Against**")
                    st.dataframe(pd.DataFrame(spray_data), use_container_width=True, hide_index=True)

        # Results narrative
        ev = _safe_num(exit_d, "ExitVel") if not exit_d.empty else np.nan
        barrel = _safe_num(exit_d, "Barrel%") if not exit_d.empty else np.nan
        gb_pct = _safe_num(ht, "Ground%") if not ht.empty else np.nan
        hr_fb = _safe_num(p_hr, "HR/FB") if not p_hr.empty else np.nan

        res_notes = []
        if not pd.isna(ev):
            ev_pct = _tm_pctile(exit_d, "ExitVel", all_p_exit)
            if not pd.isna(ev_pct) and ev_pct >= 70:
                res_notes.append(f"⚠️ Hitters making hard contact ({ev:.1f} mph EV, {int(ev_pct)}th %ile) — quality of contact is a concern.")
            elif not pd.isna(ev_pct) and ev_pct <= 30:
                res_notes.append(f"✅ Suppresses hard contact ({ev:.1f} mph EV, {int(ev_pct)}th %ile) — limits damage.")
        if not pd.isna(barrel) and barrel > 8:
            res_notes.append(f"⚠️ High barrel rate allowed ({barrel:.1f}%) — vulnerable to power hitters.")
        if not pd.isna(gb_pct) and gb_pct > 50:
            res_notes.append(f"✅ Ground-ball pitcher ({gb_pct:.1f}% GB) — elevate to beat him, hit the ball in the air.")
        elif not pd.isna(gb_pct) and gb_pct < 35:
            res_notes.append(f"Fly-ball pitcher ({gb_pct:.1f}% GB) — vulnerable to HR, power approach works.")
        if not pd.isna(hr_fb) and hr_fb > 12:
            res_notes.append(f"⚠️ Gives up HR on fly balls ({hr_fb:.1f}% HR/FB) — look to drive the ball in the air.")
        for note in res_notes:
            st.markdown(note)

    # ══════════════════════════════════════════════════════════
    # SECTION 5: HOW TO ATTACK (comprehensive game plan)
    # ══════════════════════════════════════════════════════════
    section_header("How to Attack")

    # ── Arsenal Weakness Summary ──
    st.markdown("**Arsenal Weaknesses**")
    arsenal_notes = []
    if not exit_d.empty:
        ev_ag = _safe_num(exit_d, "ExitVel")
        brl = _safe_num(exit_d, "Barrel%")
        if not pd.isna(ev_ag) and ev_ag > 88:
            arsenal_notes.append(f"High EV against ({ev_ag:.1f} mph) — gets hit hard")
        if not pd.isna(brl) and brl > 8:
            arsenal_notes.append(f"High barrel rate ({brl:.1f}%) — hittable contact")
    if not pr.empty:
        swstrk = _safe_num(pr, "SwStrk%")
        if not pd.isna(swstrk) and swstrk < 8:
            arsenal_notes.append(f"Low swing-and-miss ({swstrk:.1f}% SwStr) — can be hit")
    # Identify weak pitches (high usage + poor results)
    if not pt.empty and not exit_d.empty:
        for col, name in [("4Seam%", "Fastball"), ("Sink2Seam%", "Sinker"), ("Cutter%", "Cutter"),
                           ("Slider%", "Slider"), ("Curve%", "Curveball"), ("Change%", "Changeup"),
                           ("Sweeper%", "Sweeper")]:
            usage = pt.iloc[0].get(col)
            if usage is not None and not pd.isna(usage) and usage > 15:
                arsenal_notes.append(f"{name} ({usage:.0f}% usage) — look for it early")
                break  # just flag the primary
    if not arsenal_notes:
        arsenal_notes.append("No glaring arsenal weakness — execute at-bats")
    for n in arsenal_notes:
        st.markdown(f"- {n}")

    # ── Command Profile ──
    st.markdown("**Command Profile**")
    cmd_notes = []
    if not pr.empty:
        comploc = _safe_num(pr, "CompLoc%")
        inzone = _safe_num(pr, "InZone%")
        chase_gen = _safe_num(pr, "Chase%")
        if not pd.isna(comploc):
            if comploc < 40:
                cmd_notes.append(f"Poor command ({comploc:.1f}% CompLoc) — be patient, wait for mistakes")
            elif comploc > 50:
                cmd_notes.append(f"Strong command ({comploc:.1f}% CompLoc) — have to be ready in zone")
        if not pd.isna(inzone) and not pd.isna(comploc):
            if not pd.isna(inzone) and inzone > 50:
                cmd_notes.append(f"Challenges in zone ({inzone:.0f}% InZone) — swing at strikes, don't fall behind")
            elif not pd.isna(inzone) and inzone < 40:
                cmd_notes.append(f"Pitches around zone ({inzone:.0f}% InZone) — discipline wins, take borderline pitches")
        if not pd.isna(chase_gen) and chase_gen > 30:
            cmd_notes.append(f"Generates chase ({chase_gen:.0f}%) — stay disciplined, don't expand")
    if not cmd_notes:
        cmd_notes.append("Average command profile")
    for n in cmd_notes:
        st.markdown(f"- {n}")

    # ── Location Tendencies ──
    st.markdown("**Location Tendencies**")
    loc_notes = []
    if not ploc.empty:
        h_pct = _safe_num(ploc, "High%")
        vm_pct = _safe_num(ploc, "VMid%")
        l_pct = _safe_num(ploc, "Low%")
        i_pct = _safe_num(ploc, "Inside%")
        hm_pct = _safe_num(ploc, "HMid%")
        o_pct = _safe_num(ploc, "Outside%")
        # Vertical tendency
        vert_parts = []
        if not pd.isna(h_pct) and h_pct >= 28:
            vert_parts.append(f"up ({h_pct:.0f}% High)")
        if not pd.isna(l_pct) and l_pct >= 32:
            vert_parts.append(f"down ({l_pct:.0f}% Low)")
        if not pd.isna(vm_pct) and vm_pct >= 38:
            vert_parts.append(f"middle ({vm_pct:.0f}% VMid)")
        # Horizontal tendency
        horiz_parts = []
        if not pd.isna(i_pct) and i_pct >= 28:
            horiz_parts.append(f"inside ({i_pct:.0f}%)")
        if not pd.isna(o_pct) and o_pct >= 28:
            horiz_parts.append(f"outside ({o_pct:.0f}%)")
        if vert_parts:
            loc_notes.append(f"Lives {' and '.join(vert_parts)}")
        if horiz_parts:
            loc_notes.append(f"Tends {' and '.join(horiz_parts)}")
    if not loc_notes:
        loc_notes.append("Balanced location profile")
    for n in loc_notes:
        st.markdown(f"- {n}")

    # ── Structured Team Approach ──
    st.markdown("**Team Approach**")
    approach = []
    # 1st pitch plan
    if not pt.empty:
        primary_pct = 0
        primary_name = ""
        for col, name in [("4Seam%", "fastball"), ("Sink2Seam%", "sinker"), ("Cutter%", "cutter"),
                           ("Slider%", "slider"), ("Curve%", "curveball"), ("Change%", "changeup"),
                           ("Sweeper%", "sweeper")]:
            v = pt.iloc[0].get(col)
            if v is not None and not pd.isna(v) and v > primary_pct:
                primary_pct = v
                primary_name = name
        if primary_pct > 0:
            approach.append(f"**1st Pitch**: Expect {primary_name} ({primary_pct:.0f}%)" +
                          (" — sit on it" if primary_pct > 50 else " — be ready"))
    # When ahead
    if not pr.empty:
        swstrk_v = _safe_num(pr, "SwStrk%")
        if not pd.isna(swstrk_v) and swstrk_v < 8:
            approach.append("**When Ahead**: Be aggressive — low swing-and-miss pitcher, attack in zone")
        else:
            approach.append("**When Ahead**: Look secondary, don't chase expanding zone")
    # 2-strike
    if not mov.empty:
        eff_vel = _safe_num(mov, "EffectVel")
        vel = _safe_num(mov, "Vel")
        if not pd.isna(eff_vel) and eff_vel > 93:
            approach.append(f"**2-Strike**: Shorten up — high effective velo ({eff_vel:.1f} mph)")
        elif not pd.isna(eff_vel) and eff_vel < 86:
            approach.append(f"**2-Strike**: Stay back — low effective velo ({eff_vel:.1f} mph), protect plate")
        else:
            approach.append("**2-Strike**: Shorten up, fight off putaway pitch, don't chase")
        ext = _safe_num(mov, "Extension")
        if not pd.isna(ext) and ext > 6.5:
            approach.append(f"Long extension ({ext:.1f} ft) — ball gets on you fast")
    if not approach:
        approach.append("Quality at-bats. Execute the plan.")
    for n in approach:
        st.markdown(f"- {n}")

    # ── Keep original narrative notes ──
    notes = _pitcher_attack_plan(trad, mov, pr, ht)
    # LOB% / clutch
    if not rate.empty:
        lob = _safe_num(rate, "LOB%")
        if not pd.isna(lob) and lob < 65:
            notes.append(f"Low LOB% ({lob:.1f}%) — doesn't strand runners. Rally and keep the line moving.")
        elif not pd.isna(lob) and lob > 80:
            notes.append(f"High LOB% ({lob:.1f}%) — tough in the clutch. Score in bunches when you break through.")
    if notes:
        st.markdown("**Additional Notes**")
        for n in notes:
            st.markdown(f"- {n}")

    # ── Trackman Overlay ──
    _trackman_pitcher_overlay(trackman_data, pitcher)


def _trackman_pitcher_overlay(data, pitcher_name):
    """Show Trackman location heatmap if we have pitch-level data for this pitcher."""
    last_name = pitcher_name.split()[-1] if " " in pitcher_name else pitcher_name
    matches = data[data["Pitcher"].str.contains(last_name, case=False, na=False)]
    if matches.empty or len(matches) < 10:
        return
    section_header("Trackman Data Overlay")
    st.caption(f"Pitch-level data from our Trackman system ({len(matches)} pitches)")
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





