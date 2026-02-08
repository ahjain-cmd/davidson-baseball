from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import display_name, filter_davidson
from data.population import build_tunnel_population_pop
from data.truemedia_api import fetch_all_teams

from _pages.scouting import _get_opp_hitter_profile, _get_our_pitcher_arsenal

from decision_engine.core.state import BaseState, GameState
from decision_engine.core.runner_context import RunnerContext, speed_tier_from_sb, speed_tier_from_speed_score
from decision_engine.core.matchup import score_pitcher_vs_hitter_shrunk
from decision_engine.data.opponent_pack import load_or_build_opponent_pack
from decision_engine.data.priors import load_pitch_priors
from decision_engine.data.baserunning_data import (
    get_squeeze_row,
    get_speed_row,
    pitcher_sb_summary_from_trackman,
    speed_score_and_sb_from_sources,
)
from decision_engine.recommenders.pitch_call import recommend_pitch_call, recommend_pitch_call_re
from decision_engine.recommenders.pitch_location import recommend_pitch_location
from analytics.sequencing import _build_3pitch_sequences
from config import BRYANT_COMBINED_TEAM_ID
from data.bryant_combined import load_bryant_combined_pack
from decision_engine.recommenders.defense_recommender import (
    pitch_defense_mismatch,
    recommend_defense_from_truemedia,
)


def _get_re_cal_for_defense():
    """Lazy-load RE calibration for defense ΔRE display."""
    try:
        from analytics.run_expectancy import load_run_expectancy_calibration
        return load_run_expectancy_calibration()
    except Exception:
        return None


def _safe_float(v):
    try:
        x = float(v)
        return x if not (isinstance(x, float) and np.isnan(x)) else np.nan
    except Exception:
        return np.nan


def _safe_int(v, default=0):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return int(v)
    except Exception:
        return default


def render_ingame_panel(data):
    st.title("In-Game Decision Engine")
    st.caption("Pitch call + location recommendations (count/base-aware, shrinkage, baserunning + defense overlays).")

    # ── Opponent selection ───────────────────────────────────────────────────
    with st.expander("Opponent Pack (TrueMedia)", expanded=True):
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_a:
            season_year = int(st.number_input("Season", min_value=2020, max_value=2030, value=2026, step=1))
        teams_df = fetch_all_teams(season_year=season_year)

        team_id = ""
        team_name = ""
        _BRYANT_LABEL = "Bryant (2024-25 Combined)"

        if teams_df is not None and not teams_df.empty:
            id_col = "teamId" if "teamId" in teams_df.columns else teams_df.columns[0]
            # Prefer fullName / newestTeamName (e.g. "Charlotte 49ers") over
            # teamName (which is just the nickname, e.g. "49ers").
            name_col = None
            for c in ["fullName", "newestTeamName", "mostRecentTeamName", "teamName", "name"]:
                if c in teams_df.columns:
                    name_col = c
                    break
            if name_col is None:
                for c in teams_df.columns:
                    if "name" in c.lower():
                        name_col = c
                        break
            if name_col is None:
                name_col = teams_df.columns[0]

            teams_df = teams_df[[id_col, name_col]].dropna().drop_duplicates(subset=[id_col])
            teams_df = teams_df.sort_values(name_col)

            team_options = [_BRYANT_LABEL] + teams_df[name_col].tolist()
            with col_b:
                team_name = st.selectbox("Opponent Team", team_options, index=0)
            if team_name == _BRYANT_LABEL:
                team_id = BRYANT_COMBINED_TEAM_ID
            else:
                team_id = str(teams_df[teams_df[name_col] == team_name][id_col].iloc[0])
            with col_c:
                refresh_pack = st.checkbox("Refresh pack", value=False, help="Force refresh from API and overwrite disk cache.")
        else:
            st.warning("Could not load team list from TrueMedia API. Enter team info manually.")
            team_id = st.text_input("Opponent Team ID", value="")
            team_name = st.text_input("Opponent Team Name", value="")
            refresh_pack = st.checkbox("Refresh pack", value=False)

        if not team_id or not team_name:
            st.info("Select an opponent team to proceed.")
            return

        # Load pack — special handling for Bryant combined
        if team_id == BRYANT_COMBINED_TEAM_ID:
            pack = load_bryant_combined_pack()
            if pack is None:
                st.warning("Bryant combined pack not built yet. Go to **Bryant Scouting** page first to build it.")
                return
        else:
            pack = load_or_build_opponent_pack(
                team_id=team_id,
                team_name=team_name,
                season_year=season_year,
                refresh=refresh_pack,
            )

        h_rate = pack.get("hitting", {}).get("rate")
        if h_rate is None or h_rate.empty:
            st.error("Opponent pack is empty (no hitter data). Check API token/network or cached files.")
            return
        if "playerFullName" in h_rate.columns:
            hitters = sorted(h_rate["playerFullName"].dropna().astype(str).unique().tolist())
        elif "fullName" in h_rate.columns:
            hitters = sorted(h_rate["fullName"].dropna().astype(str).unique().tolist())
        else:
            st.error("Opponent hitter table missing player name columns.")
            return

        opp_hitter = st.selectbox("Opponent Hitter", hitters, index=0)

        # Identity mapping (for baserunning lookups).
        name_to_pid = {}
        if "playerFullName" in h_rate.columns and "playerId" in h_rate.columns:
            for _, row in h_rate.dropna(subset=["playerFullName"]).iterrows():
                nm = str(row["playerFullName"]).strip()
                pid = row.get("playerId")
                try:
                    pid_int = int(pid) if pid is not None and not (isinstance(pid, float) and np.isnan(pid)) else None
                except Exception:
                    pid_int = None
                if nm and pid_int is not None:
                    name_to_pid[nm] = pid_int

    # ── Pitcher selection ────────────────────────────────────────────────────
    with st.expander("Our Pitcher", expanded=True):
        dav_pitch = filter_davidson(data, role="pitcher")
        if dav_pitch.empty:
            st.error("No Davidson pitcher data loaded.")
            return
        our_pitchers = sorted(dav_pitch["Pitcher"].dropna().astype(str).unique().tolist())
        our_pitcher = st.selectbox("Select Pitcher", our_pitchers, index=0, format_func=display_name)

        seasons = sorted([int(s) for s in dav_pitch["Season"].dropna().unique().tolist()]) if "Season" in dav_pitch.columns else []
        season_filter = st.multiselect("Seasons", seasons, default=seasons)

    # ── Game state inputs ───────────────────────────────────────────────────
    with st.expander("Game State", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 2, 2])
        with c1:
            balls = int(st.number_input("Balls", min_value=0, max_value=3, value=0, step=1))
        with c2:
            strikes = int(st.number_input("Strikes", min_value=0, max_value=2, value=0, step=1))
        with c3:
            outs = int(st.number_input("Outs", min_value=0, max_value=2, value=0, step=1))
        with c4:
            inning = int(st.number_input("Inning", min_value=1, max_value=20, value=1, step=1))
        with c5:
            last_pitch_opt = st.selectbox(
                "Last Pitch Thrown",
                ["(None)"] + sorted(
                    {"Fastball", "Sinker", "Cutter", "Slider", "Curveball",
                     "Changeup", "Splitter", "Sweeper", "Knuckle Curve"}
                ),
                index=0,
                help="The previous pitch thrown this AB (enables sequence/tunnel adjustments)",
            )
            last_pitch = None if last_pitch_opt == "(None)" else last_pitch_opt

        b1, b2, b3 = st.columns(3)
        with b1:
            on_1b = st.checkbox("1B", value=False)
        with b2:
            on_2b = st.checkbox("2B", value=False)
        with b3:
            on_3b = st.checkbox("3B", value=False)

        # Runner identity + baserunning context
        runner_ctx = RunnerContext()
        h_count = pack.get("hitting", {}).get("counting", None)
        if h_count is None:
            h_count = None

        def _counting_row(pid):
            if h_count is None or h_count.empty or pid is None:
                return None
            if "playerId" in h_count.columns:
                sub = h_count[h_count["playerId"] == pid]
                if not sub.empty:
                    return sub.iloc[0].to_dict()
            return None

        if on_1b or on_2b or on_3b:
            st.markdown("**Baserunning Context**")

        # Pitcher SB% allowed (auto from Trackman if possible)
        sb_summary = pitcher_sb_summary_from_trackman(data, our_pitcher)
        pitcher_sb_pct_default = 75.0
        if sb_summary and sb_summary.get("sb_pct") is not None and (sb_summary.get("att") or 0) >= 3:
            pitcher_sb_pct_default = float(sb_summary["sb_pct"])
        cpt_col, psb_col = st.columns([1, 1])
        with cpt_col:
            catcher_pop_time = float(st.number_input("Catcher Pop Time (s)", min_value=1.70, max_value=2.60, value=2.03, step=0.01))
        with psb_col:
            pitcher_sb_pct = float(st.number_input("Pitcher SB% Allowed", min_value=0.0, max_value=100.0, value=float(round(pitcher_sb_pct_default, 1)), step=1.0))
            if sb_summary and sb_summary.get("att"):
                st.caption(f"Trackman events: SB {sb_summary['sb']} / ATT {sb_summary['att']}")

        runner_ctx = RunnerContext(
            catcher_pop_time=catcher_pop_time,
            pitcher_sb_pct=pitcher_sb_pct,
        )

        def _runner_block(label, *, base_enabled: bool):
            if not base_enabled:
                return None, "NONE", np.nan, 0, np.nan
            opts = ["(Unknown)"] + hitters
            sel = st.selectbox(label, opts, index=0)
            if sel == "(Unknown)":
                # Allow a manual tier.
                tier = st.selectbox(f"{label} Speed Tier", ["SLOW", "AVG", "FAST", "ELITE"], index=1)
                return None, tier, np.nan, 0, np.nan

            pid = name_to_pid.get(sel)
            counting = _counting_row(pid) if pid is not None else None
            ss, sb, sbp = speed_score_and_sb_from_sources(player_id=pid, counting_row=counting)
            tier_auto = speed_tier_from_speed_score(ss) if ss is not None else speed_tier_from_sb(sb)

            # Manual override optional.
            override = st.selectbox(f"{label} Tier Override", ["Auto", "ELITE", "FAST", "AVG", "SLOW"], index=0)
            tier = tier_auto if override == "Auto" else override

            if ss is not None or sb is not None:
                cols = st.columns(3)
                with cols[0]:
                    st.metric("SpeedScore", f"{ss:.2f}" if ss is not None else "-")
                with cols[1]:
                    st.metric("SB", f"{sb}" if sb is not None else "-")
                with cols[2]:
                    st.metric("Tier", tier)
            return pid, tier, ss if ss is not None else np.nan, sb if sb is not None else 0, sbp if sbp is not None else np.nan

        r1_pid, r1_tier, r1_ss, r1_sb, r1_sbp = _runner_block("Runner on 1B", base_enabled=on_1b)
        r2_pid, r2_tier, r2_ss, r2_sb, r2_sbp = _runner_block("Runner on 2B", base_enabled=on_2b)
        _r3_pid, r3_tier, _r3_ss, _r3_sb, _r3_sbp = _runner_block("Runner on 3B", base_enabled=on_3b)

        # Batter squeeze/bunt tendencies from Count_SB_SQ.csv export.
        batter_pid = name_to_pid.get(opp_hitter)
        squeeze_row = get_squeeze_row(batter_pid)
        squeeze_ct = _safe_int(squeeze_row.get("Squeeze #") if squeeze_row else None, default=0)
        bunt_att = _safe_int(squeeze_row.get("Bunt Hit Att") if squeeze_row else None, default=0)
        if on_3b and outs < 2 and (squeeze_ct >= 2 or bunt_att >= 3):
            st.warning(f"Squeeze/Bunt Alert: squeeze={squeeze_ct}, bunt hit att={bunt_att} (R3, <2 outs)")

        runner_ctx = RunnerContext(
            r1_speed_tier=r1_tier,
            r1_speed_score=float(r1_ss) if not np.isnan(float(r1_ss)) else float("nan"),
            r1_sb_count=int(r1_sb or 0),
            r1_sb_pct=float(r1_sbp) if not np.isnan(float(r1_sbp)) else float("nan"),
            r2_speed_tier=r2_tier,
            r2_speed_score=float(r2_ss) if not np.isnan(float(r2_ss)) else float("nan"),
            r2_sb_count=int(r2_sb or 0),
            r2_sb_pct=float(r2_sbp) if not np.isnan(float(r2_sbp)) else float("nan"),
            r3_speed_tier=r3_tier,
            batter_squeeze_count=squeeze_ct,
            batter_bunt_hit_att=bunt_att,
            catcher_pop_time=catcher_pop_time,
            pitcher_sb_pct=pitcher_sb_pct,
        )

        sc1, sc2 = st.columns(2)
        with sc1:
            score_our = st.number_input("DAV", min_value=0, max_value=50, value=0, step=1)
        with sc2:
            score_opp = st.number_input("OPP", min_value=0, max_value=50, value=0, step=1)

        state = GameState(
            balls=balls,
            strikes=strikes,
            outs=outs,
            inning=inning,
            bases=BaseState(on_1b=on_1b, on_2b=on_2b, on_3b=on_3b),
            score_our=int(score_our),
            score_opp=int(score_opp),
            runner=runner_ctx,
            last_pitch=last_pitch,
        )

    # ── Compute matchup & recommendations ───────────────────────────────────
    with st.spinner("Building pitcher arsenal..."):
        tunnel_pop = build_tunnel_population_pop()
        arsenal = _get_our_pitcher_arsenal(data, our_pitcher, season_filter=season_filter, tunnel_pop=tunnel_pop)
    if arsenal is None:
        st.error("Not enough Trackman data for this pitcher/season selection.")
        return

    with st.spinner("Building opponent hitter profile..."):
        hitter_profile = _get_opp_hitter_profile(pack, opp_hitter, team_name, pitch_df=None)

    with st.spinner("Loading D1 priors (one-time, cached)..."):
        priors = load_pitch_priors()

    matchup = score_pitcher_vs_hitter_shrunk(arsenal, hitter_profile, priors)
    if matchup is None:
        st.error("Could not compute matchup.")
        return

    # Try RE-based recommendations first, fall back to composite scoring
    recs_re = recommend_pitch_call_re(
        matchup, state, top_n=3,
        tun_df=arsenal.get("tunnels"), seq_df=arsenal.get("sequences"),
    )
    recs_old = recommend_pitch_call(
        matchup, state, top_n=3,
        tun_df=arsenal.get("tunnels"), seq_df=arsenal.get("sequences"),
    )
    use_re = bool(recs_re and "delta_re" in recs_re[0])
    recs = recs_re if use_re else recs_old
    if not recs:
        st.warning("No pitch recommendations available.")
        return

    st.subheader(f"Pitch Call ({state.count_str()})")
    top_loc_zone = None
    top_loc_label = None
    top_pitch = recs[0]["pitch"] if recs else None
    hitter_zv = matchup.get("hitter_data", {}).get("zone_vuln", {})
    for i, r in enumerate(recs, start=1):
        pitch_name = r["pitch"]
        conf = r.get("confidence", "Low")
        conf_n = r.get("confidence_n", 0)

        if use_re:
            rs100 = r.get("rs100", 0.0)
            # Color coding: Green (>0.5), Yellow (±0.5), Red (<-0.5)
            if rs100 > 0.5:
                color = "🟢"
            elif rs100 < -0.5:
                color = "🔴"
            else:
                color = "🟡"
            st.markdown(f"**#{i} {pitch_name}**  |  RS/100: `{rs100:+.1f}` {color}  |  Confidence: `{conf}` (n={conf_n})")
            # ΔRE decomposition
            dre_base = r.get("delta_re_base", 0.0)
            dre_seq = r.get("delta_re_seq", 0.0)
            dre_steal = r.get("delta_re_steal", 0.0)
            dre_squeeze = r.get("delta_re_squeeze", 0.0)
            dre_usage = r.get("delta_re_usage", 0.0)
            parts = f"ΔRE: base {dre_base:+.4f}"
            if dre_seq != 0.0:
                parts += f" + seq {dre_seq:+.4f}"
            if dre_steal != 0.0:
                parts += f" + steal {dre_steal:+.4f}"
            if dre_squeeze != 0.0:
                parts += f" + squeeze {dre_squeeze:+.4f}"
            if dre_usage != 0.0:
                parts += f" + usage {dre_usage:+.4f}"
            parts += f" = {r.get('delta_re', 0.0):+.4f}"
            st.caption(parts)
        else:
            adj_seq = r.get("adj_sequence", 0.0)
            seq_tag = f"  |  Seq: `{adj_seq:+.1f}`" if adj_seq != 0.0 else ""
            st.markdown(f"**#{i} {pitch_name}**  |  Score: `{r['score']:.1f}`  |  Confidence: `{conf}` (n={conf_n}){seq_tag}")

        if r.get("reasons"):
            st.caption(" | ".join(r["reasons"][:5]))

        loc = recommend_pitch_location(
            pitch_name,
            arsenal["pitches"].get(pitch_name, {}),
            state,
            hitter_zone_vuln=hitter_zv,
            bats=matchup.get("bats", "R"),
            throws=arsenal.get("throws", "Right"),
        )
        if loc:
            if i == 1:
                top_loc_zone = loc.get("zone")
                top_loc_label = loc.get("zone_label")
            zlab = loc.get("zone_label")
            wh = loc.get("whiff_pct", float("nan"))
            csw = loc.get("csw_pct", float("nan"))
            ev = loc.get("ev_against", float("nan"))
            n = loc.get("n", 0)
            reason = loc.get("reason", "")
            st.caption(f"Location: **{zlab}** (n={n}) | whiff {wh:.0f}% | CSW {csw:.0f}% | EV {ev:.0f} | {reason}")
            sec = loc.get("secondary")
            if sec:
                sec_lab = sec.get("zone_label", "")
                sec_reason = sec.get("reason", "")
                st.caption(f"  Alt: **{sec_lab}** | {sec_reason}")

    # ── 3-Pitch Sequences ──────────────────────────────────────────────────
    with st.expander("3-Pitch Sequences", expanded=False):
        tun = arsenal.get("tunnels", pd.DataFrame())
        seq = arsenal.get("sequences", pd.DataFrame())
        sorted_ps = sorted(
            matchup.get("pitch_scores", {}).items(),
            key=lambda x: x[1].get("score", 0), reverse=True,
        )
        hd_seq = matchup.get("hitter_data") or {}
        seqs = _build_3pitch_sequences(sorted_ps, hd_seq, tun, seq)
        if seqs:
            for idx, s in enumerate(seqs[:3], start=1):
                t12 = s.get("t12", np.nan)
                t23 = s.get("t23", np.nan)
                sw23 = s.get("sw23", np.nan)
                their_2k = s.get("their_2k", np.nan)
                effv_gap = s.get("effv_gap", np.nan)
                t12_str = f"{t12:.0f}" if not np.isnan(t12) else "-"
                t23_str = f"{t23:.0f}" if not np.isnan(t23) else "-"
                sw23_str = f"{sw23:.0f}%" if not np.isnan(sw23) else "-"
                their_2k_str = f"{their_2k:.0f}%" if not np.isnan(their_2k) else "-"
                effv_str = f"{effv_gap:.1f} mph" if not np.isnan(effv_gap) else "-"
                st.markdown(
                    f"**#{idx} {s['seq']}**  |  Score: `{s['score']:.1f}`"
                )
                st.caption(
                    f"Tunnel: {t12_str} / {t23_str}  |  Whiff→P3: {sw23_str}"
                    f"  |  2K Whiff: {their_2k_str}  |  EffV Gap: {effv_str}"
                )
        else:
            st.info("Not enough arsenal data for sequence analysis.")

    # ── Defense panel (Module B) ─────────────────────────────────────────────
    with st.expander("Defense Positioning (v2)", expanded=False):
        hl = pack.get("hitting", {}).get("hit_locations", pd.DataFrame())
        ht = pack.get("hitting", {}).get("hit_types", pd.DataFrame())

        def _player_row(df, name):
            if df is None or df.empty:
                return None
            if "playerFullName" in df.columns:
                sub = df[df["playerFullName"].astype(str).str.strip() == str(name).strip()]
                if not sub.empty:
                    return sub.iloc[0].to_dict()
            if "fullName" in df.columns:
                sub = df[df["fullName"].astype(str).str.strip() == str(name).strip()]
                if not sub.empty:
                    return sub.iloc[0].to_dict()
            return None

        hl_row = _player_row(hl, opp_hitter) or {}
        ht_row = _player_row(ht, opp_hitter) or {}

        pull_pct = _safe_float(hl_row.get("HPull%"))
        ctr_pct = _safe_float(hl_row.get("HCtr%"))
        opp_pct = _safe_float(hl_row.get("HOppFld%"))
        gb_pct = _safe_float(ht_row.get("Ground%"))
        fb_pct = _safe_float(ht_row.get("Fly%"))
        ld_pct = _safe_float(ht_row.get("Line%"))
        pu_pct = _safe_float(ht_row.get("Popup%"))

        # Batter side for defense: if switch, assume opposite of pitcher throw.
        bats = str(hitter_profile.get("bats", "?")).strip().upper()
        throws = str(arsenal.get("throws", "Right")).strip().upper()
        batter_side = "Right"
        if bats in {"L", "LEFT"}:
            batter_side = "Left"
        elif bats in {"S", "SWITCH"}:
            batter_side = "Left" if throws.startswith("R") else "Right"

        defense = recommend_defense_from_truemedia(
            batter_side=batter_side,
            pull_pct=pull_pct,
            center_pct=ctr_pct,
            oppo_pct=opp_pct,
            gb_pct=gb_pct,
            fb_pct=fb_pct,
            ld_pct=ld_pct,
            pu_pct=pu_pct,
            exit_vel=hitter_profile.get("ev"),
            state=state,
        )

        st.markdown(f"**Shift:** `{defense.shift['type']}`")
        st.caption(defense.shift.get("desc", ""))
        if defense.overlay.get("notes"):
            st.caption(" | ".join(defense.overlay["notes"]))

        # Shift value in ΔRE terms
        if use_re and recs:
            from decision_engine.recommenders.defense_recommender import (
                estimate_shift_contact_rv,
                shift_value_delta_re,
            )
            top_r = recs[0]
            top_raw = top_r.get("raw", {})
            re_cal_def = _get_re_cal_for_defense()
            std_crv = re_cal_def.contact_rv.get(top_r["pitch"], 0.12) if re_cal_def else 0.12
            shifted_crv = estimate_shift_contact_rv(
                shift_type=defense.shift["type"],
                pull_pct=pull_pct,
                gb_pct=gb_pct,
                standard_contact_rv=std_crv,
            )
            # Get P(in_play) from the RE probs
            p_ip = 0.16  # fallback
            if re_cal_def and top_r["pitch"] in re_cal_def.outcome_probs:
                count_probs = re_cal_def.outcome_probs[top_r["pitch"]].get(state.count_str())
                if count_probs:
                    p_ip = count_probs.p_in_play if hasattr(count_probs, 'p_in_play') else 0.16
            shift_dre = shift_value_delta_re(std_crv, shifted_crv, p_ip)
            shift_rs100 = -shift_dre * 100.0
            if abs(shift_rs100) > 0.01:
                st.caption(f"Shift value: RS/100 {shift_rs100:+.2f} (ΔRE {shift_dre:+.4f})")

        # Compatibility warning vs top pitch recommendation.
        if top_pitch:
            warn = pitch_defense_mismatch(
                pitch_name=top_pitch,
                location_zone=top_loc_label,
                defense_shift_type=defense.shift["type"],
            )
            if warn:
                st.warning(warn)

        pos_rows = []
        for pos, xy in sorted(defense.positions.items()):
            pos_rows.append({"Pos": pos, "x": round(xy["x"], 1), "y": round(xy["y"], 1)})
        if pos_rows:
            st.dataframe(pos_rows, use_container_width=True, hide_index=True)

            # Mini field diagram (simple scatter).
            fig = go.Figure()
            xs = [defense.positions[p]["x"] for p in defense.positions]
            ys = [defense.positions[p]["y"] for p in defense.positions]
            labels = list(defense.positions.keys())
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    marker=dict(size=10, color="#111", line=dict(width=1, color="#fff")),
                    showlegend=False,
                )
            )
            # Basic diamond.
            diamond_x = [0, 63.6, 0, -63.6, 0]
            diamond_y = [0, 63.6, 127.3, 63.6, 0]
            fig.add_trace(go.Scatter(x=diamond_x, y=diamond_y, mode="lines", line=dict(color="rgba(0,0,0,0.25)"), showlegend=False))
            fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(range=[-250, 250], showgrid=False, zeroline=False, visible=False),
                yaxis=dict(range=[0, 360], showgrid=False, zeroline=False, visible=False),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Opponent fielding intel (Module C) ───────────────────────────────────
    with st.expander("Opponent Fielding Intel (v2)", expanded=False):
        starters = pack.get("defense", {}).get("starters", None)
        if starters is not None and isinstance(starters, object) and hasattr(starters, "empty") and not starters.empty:
            cols = ["pos", "playerFullName", "Inn", "Chances", "E", "FLD%", "quality", "note"]
            view = starters[[c for c in cols if c in starters.columns]].copy()
            view = view.rename(columns={"playerFullName": "Player", "quality": "Quality"})
            st.dataframe(view.sort_values(["pos"]), use_container_width=True, hide_index=True)
        of_arms = pack.get("defense", {}).get("of_arms", None)
        if of_arms is not None and hasattr(of_arms, "empty") and not of_arms.empty:
            cols = ["pos", "playerFullName", "InnOF", "OFAst", "ArmRating", "OFThrowE"]
            view = of_arms[[c for c in cols if c in of_arms.columns]].copy()
            view = view.rename(columns={"playerFullName": "Player"})
            st.dataframe(view.sort_values(["ArmRating", "OFAst"], ascending=[True, False]), use_container_width=True, hide_index=True)
