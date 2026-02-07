"""Team Overview page."""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config import (
    DAVIDSON_TEAM_ID, ROSTER_2026, PITCH_COLORS,
    SWING_CALLS, CONTACT_CALLS,
    _APP_DIR, filter_davidson, display_name, is_barrel_mask,
)
from data.loader import get_all_seasons
from data.population import compute_batter_stats_pop, compute_pitcher_stats_pop
from viz.layout import CHART_LAYOUT, section_header


def page_team(data):
    _logo_path = os.path.join(_APP_DIR, "logo_real.png")
    _celeb_path = os.path.join(_APP_DIR, "regionalchamps_copy.jpg")

    # ── Hero Banner ──
    st.markdown("""
    <style>
    .wildcats-hero {
        background: #111 !important;
        border-radius: 12px;
        padding: 28px 32px 22px 32px;
        margin-bottom: 18px;
        border: 2px solid #cc0000;
        text-align: center;
    }
    .wildcats-hero .subtitle {
        font-size: 11px; letter-spacing: 5px; color: #ccc !important;
        font-weight: 500; margin-bottom: 2px;
    }
    .wildcats-hero .title {
        font-size: 38px; font-weight: 900; color: #cc0000 !important;
        letter-spacing: 3px; line-height: 1.15;
    }
    .wildcats-hero .tagline {
        font-size: 10px; letter-spacing: 3px; color: #999 !important;
        margin-top: 4px;
    }
    .stat-card {
        background: #111 !important; border-radius: 8px; padding: 12px 8px;
        text-align: center; border: 1px solid #333;
    }
    .stat-card .val { font-size: 22px; font-weight: 800; color: #fff !important; }
    .stat-card .lbl { font-size: 10px; color: #ccc !important; letter-spacing: 1px; text-transform: uppercase; }
    .leader-card {
        background: #111 !important; border-radius: 8px; padding: 12px 16px;
        margin: 6px 0; border: 1px solid #333;
        border-left: 4px solid #cc0000;
    }
    .leader-card .cat { font-size: 11px; color: #ccc !important; letter-spacing: 1px; text-transform: uppercase; }
    .leader-card .name { font-size: 17px; font-weight: 700; color: #fff !important; }
    .leader-card .stat { font-size: 14px; font-weight: 600; color: #cc0000 !important; }
    .leader-card .note { font-size: 11px; color: #aaa !important; }
    .leader-card-alt { border-left-color: #fff !important; }
    .leader-card-alt .stat { color: #fff !important; }
    .leader-card-grn { border-left-color: #cc0000 !important; }
    .leader-card-grn .stat { color: #cc0000 !important; }
    .logo-center { display: flex; align-items: center; justify-content: center; height: 100%; min-height: 140px; }
    </style>
    """, unsafe_allow_html=True)

    # Banner row: logo | title | celebration photo
    c_logo, c_title, c_photo = st.columns([1, 3, 1.5])
    with c_logo:
        if os.path.exists(_logo_path):
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.image(_logo_path, width=110)
    with c_title:
        st.markdown("""
        <div class="wildcats-hero">
            <div class="subtitle">DAVIDSON BASEBALL</div>
            <div class="title">W.I.L.D.C.A.T.S.</div>
            <div class="tagline">Wildcat Intelligence &amp; Live Data Computing for Advanced Trackman Statistics</div>
        </div>
        """, unsafe_allow_html=True)
    with c_photo:
        if os.path.exists(_celeb_path):
            st.image(_celeb_path, use_container_width=True)

    # ── Data setup ──
    # Davidson-only, Trackman-derived rows
    dav_pitching = data[data["PitcherTeam"] == DAVIDSON_TEAM_ID].copy()
    dav_batting = data[data["BatterTeam"] == DAVIDSON_TEAM_ID].copy()
    dav_data = pd.concat([dav_pitching, dav_batting]).drop_duplicates(
        subset=["Date", "Pitcher", "Batter", "PitchNo", "Inning", "PAofInning"]
    )
    latest_date = data["Date"].max()
    dav_dates = pd.concat([dav_pitching["Date"], dav_batting["Date"]]).dropna().dt.date.nunique()
    n_pitchers = len([p for p in ROSTER_2026 if p in dav_pitching["Pitcher"].values])
    n_hitters = len([b for b in ROSTER_2026 if b in dav_batting["Batter"].values])

    # ── Top-level tabs: Dashboard + Game Log ──
    tab_dash, tab_gl = st.tabs(["Dashboard", "Game Log"])

    with tab_dash:
        _render_team_dashboard(data, dav_pitching, dav_batting, dav_data, latest_date, dav_dates, n_pitchers, n_hitters)

    with tab_gl:
        _render_game_log(data)


def _render_game_log(data):
    """Game Log tab inside Team Overview."""
    dav = data[(data["PitcherTeam"] == DAVIDSON_TEAM_ID) | (data["BatterTeam"] == DAVIDSON_TEAM_ID)]
    if dav.empty:
        st.warning("No data.")
        return
    games = dav.groupby(["Date", "GameID"]).agg(
        Stadium=("Stadium", "first"), Home=("HomeTeam", "first"),
        Away=("AwayTeam", "first"), Pitches=("PitchNo", "count"),
    ).reset_index().sort_values("Date", ascending=False)
    st.dataframe(games[["Date", "Away", "Home", "Stadium", "Pitches"]], use_container_width=True, hide_index=True)

    opts = games["GameID"].tolist()
    if opts:
        _game_labels = {row["GameID"]: f"{row['Date'].strftime('%Y-%m-%d')} {row['Away']} @ {row['Home']}"
                        for _, row in games.iterrows()}
        sel = st.selectbox("Drill into game", opts,
                           format_func=lambda g: _game_labels.get(g, str(g)), key="to_gl_game")
        gd = dav[dav["GameID"] == sel]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Davidson Pitching**")
            gp = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID]
            for p in gp["Pitcher"].unique():
                sub = gp[gp["Pitcher"] == p]
                st.markdown(f"_{display_name(p)}_ - {len(sub)} pitches")
                s = sub.groupby("TaggedPitchType").agg(
                    N=("RelSpeed", "count"), Velo=("RelSpeed", "mean"), Spin=("SpinRate", "mean")
                ).reset_index()
                s["Velo"] = s["Velo"].round(1)
                s["Spin"] = s["Spin"].round(0)
                st.dataframe(s, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Davidson Hitting**")
            gh = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID]
            ip = gh[gh["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            for b in ip["Batter"].unique():
                sub = ip[ip["Batter"] == b]
                st.markdown(f"_{display_name(b)}_ - {len(sub)} BIP, Avg: {sub['ExitSpeed'].mean():.1f}, "
                            f"Max: {sub['ExitSpeed'].max():.1f}")


def _render_team_dashboard(data, dav_pitching, dav_batting, dav_data, latest_date, dav_dates, n_pitchers, n_hitters):
    """Dashboard tab inside Team Overview."""
    # ── Quick Stats Row ──
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.markdown(f'<div class="stat-card"><div class="val">{len(dav_data):,}</div><div class="lbl">Total Pitches</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-card"><div class="val">{dav_dates}</div><div class="lbl">Davidson Games</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-card"><div class="val">{n_pitchers}</div><div class="lbl">Rostered Arms</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-card"><div class="val">{n_hitters}</div><div class="lbl">Rostered Bats</div></div>', unsafe_allow_html=True)
    with s5:
        _ld = latest_date.strftime("%b %d, %Y") if pd.notna(latest_date) else "—"
        st.markdown(f'<div class="stat-card"><div class="val" style="font-size:16px;">{_ld}</div><div class="lbl">Latest Data</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Weekly Leaders ──
    if pd.notna(latest_date):
        week_ago = latest_date - pd.Timedelta(days=7)
        recent = dav_data[dav_data["Date"] > week_ago]
    else:
        recent = pd.DataFrame()

    section_header("This Week's Leaders")

    if len(recent) < 10:
        st.info("Not enough data in the last 7 days for weekly leaders. Showing full-season leaderboards below.")
    else:
        st.caption(f"{week_ago.strftime('%b %d')} – {latest_date.strftime('%b %d, %Y')}")
        col_wh, col_wp = st.columns(2)

        # ── Hitting Leaders ──
        with col_wh:
            st.markdown("#### :red[Hitting]")
            week_bat = recent[
                (recent["BatterTeam"] == DAVIDSON_TEAM_ID) &
                (recent["Batter"].isin(ROSTER_2026))
            ].copy()
            week_inplay = week_bat[week_bat["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            if len(week_inplay) >= 5:
                # Hardest contact
                ev_leader = week_inplay.groupby("Batter")["ExitSpeed"].agg(["mean", "max", "count"]).reset_index()
                ev_leader = ev_leader[ev_leader["count"] >= 3].sort_values("mean", ascending=False)
                if len(ev_leader) > 0:
                    top = ev_leader.iloc[0]
                    st.markdown(f'<div class="leader-card">'
                                f'<div class="cat">Hardest Contact</div>'
                                f'<div class="name">{display_name(top["Batter"])}</div>'
                                f'<div class="stat">{top["mean"]:.1f} avg EV</div>'
                                f'<div class="note">{int(top["count"])} balls in play &middot; {top["max"]:.1f} max</div>'
                                f'</div>', unsafe_allow_html=True)

                # Barrel leader (only show if someone actually barreled a ball)
                _showed_barrel = False
                if "Angle" in week_inplay.columns:
                    wb = week_inplay.copy()
                    wb["is_barrel"] = is_barrel_mask(wb)
                    barrel_ct = wb.groupby("Batter")["is_barrel"].agg(["sum", "count"]).reset_index()
                    barrel_ct = barrel_ct[(barrel_ct["count"] >= 3) & (barrel_ct["sum"] > 0)].copy()
                    if len(barrel_ct) > 0:
                        barrel_ct["rate"] = barrel_ct["sum"] / barrel_ct["count"] * 100
                        barrel_ct = barrel_ct.sort_values("sum", ascending=False)
                        tb = barrel_ct.iloc[0]
                        st.markdown(f'<div class="leader-card leader-card-alt">'
                                    f'<div class="cat">Most Barrels</div>'
                                    f'<div class="name">{display_name(tb["Batter"])}</div>'
                                    f'<div class="stat">{int(tb["sum"])} barrels ({tb["rate"]:.0f}%)</div>'
                                    f'<div class="note">{int(tb["count"])} balls in play</div>'
                                    f'</div>', unsafe_allow_html=True)
                        _showed_barrel = True
                if not _showed_barrel and "Angle" in week_inplay.columns:
                    # Fallback: best sweet spot rate
                    wb2 = week_inplay.copy()
                    wb2["is_ss"] = wb2["Angle"].between(8, 32)
                    ss_ct = wb2.groupby("Batter")["is_ss"].agg(["sum", "count"]).reset_index()
                    ss_ct = ss_ct[ss_ct["count"] >= 3].copy()
                    if len(ss_ct) > 0:
                        ss_ct["rate"] = ss_ct["sum"] / ss_ct["count"] * 100
                        ss_ct = ss_ct.sort_values("rate", ascending=False)
                        ts = ss_ct.iloc[0]
                        st.markdown(f'<div class="leader-card leader-card-alt">'
                                    f'<div class="cat">Best Sweet Spot%</div>'
                                    f'<div class="name">{display_name(ts["Batter"])}</div>'
                                    f'<div class="stat">{ts["rate"]:.0f}% sweet spot</div>'
                                    f'<div class="note">{int(ts["count"])} balls in play</div>'
                                    f'</div>', unsafe_allow_html=True)

                # Max single hit
                if week_inplay["ExitSpeed"].notna().any():
                    max_ev_row = week_inplay.loc[week_inplay["ExitSpeed"].idxmax()]
                    st.markdown(f'<div class="leader-card leader-card-grn">'
                                f'<div class="cat">Hardest Single Hit</div>'
                                f'<div class="name">{display_name(max_ev_row["Batter"])}</div>'
                                f'<div class="stat">{max_ev_row["ExitSpeed"]:.1f} mph</div>'
                                f'</div>', unsafe_allow_html=True)
            else:
                st.caption("Not enough in-play data this week.")

        # ── Pitching Leaders ──
        with col_wp:
            st.markdown("#### :red[Pitching]")
            week_pit = recent[
                (recent["PitcherTeam"] == DAVIDSON_TEAM_ID) &
                (recent["Pitcher"].isin(ROSTER_2026))
            ].copy()
            if len(week_pit) >= 10:
                # Whiff leader
                pit_swings = week_pit[week_pit["PitchCall"].isin(SWING_CALLS)]
                pit_whiffs = week_pit[week_pit["PitchCall"] == "StrikeSwinging"]
                whiff_by_p = pit_swings.groupby("Pitcher").size().reset_index(name="swings")
                whiff_ct = pit_whiffs.groupby("Pitcher").size().reset_index(name="whiffs")
                whiff_by_p = whiff_by_p.merge(whiff_ct, on="Pitcher", how="left").fillna(0)
                whiff_by_p["whiff_rate"] = whiff_by_p["whiffs"] / whiff_by_p["swings"] * 100
                whiff_by_p = whiff_by_p[whiff_by_p["swings"] >= 10].sort_values("whiff_rate", ascending=False)
                if len(whiff_by_p) > 0:
                    tw = whiff_by_p.iloc[0]
                    st.markdown(f'<div class="leader-card">'
                                f'<div class="cat">Highest Whiff Rate</div>'
                                f'<div class="name">{display_name(tw["Pitcher"])}</div>'
                                f'<div class="stat">{tw["whiff_rate"]:.0f}% whiff</div>'
                                f'<div class="note">{int(tw["swings"])} swings faced</div>'
                                f'</div>', unsafe_allow_html=True)

                # Velo leader
                fb_data = week_pit[week_pit["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter"])]
                if len(fb_data) > 0:
                    fb_velo = fb_data.groupby("Pitcher")["RelSpeed"].agg(["mean", "max", "count"]).reset_index()
                    fb_velo = fb_velo[fb_velo["count"] >= 5].sort_values("max", ascending=False)
                    if len(fb_velo) > 0:
                        tv = fb_velo.iloc[0]
                        st.markdown(f'<div class="leader-card leader-card-alt">'
                                    f'<div class="cat">Top Velocity</div>'
                                    f'<div class="name">{display_name(tv["Pitcher"])}</div>'
                                    f'<div class="stat">{tv["max"]:.1f} max &middot; {tv["mean"]:.1f} avg</div>'
                                    f'<div class="note">{int(tv["count"])} fastballs thrown</div>'
                                    f'</div>', unsafe_allow_html=True)

                # Lowest EV against
                pit_inplay = recent[
                    (recent["PitcherTeam"] == DAVIDSON_TEAM_ID) &
                    (recent["Pitcher"].isin(ROSTER_2026)) &
                    (recent["PitchCall"] == "InPlay")
                ]
                pit_ev = pit_inplay.dropna(subset=["ExitSpeed"]).groupby("Pitcher")["ExitSpeed"].agg(["mean", "count"]).reset_index()
                pit_ev = pit_ev[pit_ev["count"] >= 3].sort_values("mean")
                if len(pit_ev) > 0:
                    te = pit_ev.iloc[0]
                    st.markdown(f'<div class="leader-card leader-card-grn">'
                                f'<div class="cat">Weakest Contact Allowed</div>'
                                f'<div class="name">{display_name(te["Pitcher"])}</div>'
                                f'<div class="stat">{te["mean"]:.1f} avg EV against</div>'
                                f'<div class="note">{int(te["count"])} balls in play</div>'
                                f'</div>', unsafe_allow_html=True)
            else:
                st.caption("Not enough pitching data this week.")

    st.markdown("")

    # ── Full Season Leaderboards ──
    section_header("Season Leaderboards")
    all_seasons = get_all_seasons()
    sel = st.multiselect("Season", all_seasons, default=all_seasons, key="to_s")

    tab_h, tab_p = st.tabs(["Hitting Leaderboard", "Pitching Leaderboard"])

    with tab_h:
        bs = compute_batter_stats_pop(season_filter=sel)
        dav_h = bs[(bs["BatterTeam"] == DAVIDSON_TEAM_ID) & (bs["Batter"].isin(ROSTER_2026))].copy()
        if dav_h.empty:
            st.info("No hitting data.")
        else:
            d = dav_h[["Batter", "PA", "BBE", "AvgEV", "MaxEV", "HardHitPct", "BarrelPct",
                       "SweetSpotPct", "WhiffPct", "KPct", "BBPct", "ChasePct"]].sort_values("PA", ascending=False).copy()
            d["Batter"] = d["Batter"].apply(display_name)
            d.columns = ["Batter", "PA", "BBE", "Avg EV", "Max EV", "Hard%", "Barrel%",
                         "Sweet%", "Whiff%", "K%", "BB%", "Chase%"]
            for c in d.columns[3:]:
                d[c] = d[c].round(1)
            st.dataframe(d, use_container_width=True, hide_index=True)

    with tab_p:
        ps = compute_pitcher_stats_pop(season_filter=sel)
        dav_p = ps[(ps["PitcherTeam"] == DAVIDSON_TEAM_ID) & (ps["Pitcher"].isin(ROSTER_2026))].copy()
        if dav_p.empty:
            st.info("No pitching data.")
        else:
            d = dav_p[["Pitcher", "Pitches", "PA", "AvgFBVelo", "MaxFBVelo", "AvgSpin",
                       "WhiffPct", "KPct", "BBPct", "ZonePct", "ChasePct", "AvgEVAgainst", "GBPct"]].sort_values("Pitches", ascending=False).copy()
            d["Pitcher"] = d["Pitcher"].apply(display_name)
            d.columns = ["Pitcher", "Pitches", "PA", "Avg FB", "Max FB", "Spin",
                         "Whiff%", "K%", "BB%", "Zone%", "Chase%", "EV Ag", "GB%"]
            for c in d.columns[3:]:
                d[c] = d[c].round(1)
            st.dataframe(d, use_container_width=True, hide_index=True)
