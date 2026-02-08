"""Bryant 2026 Scouting Page — multi-year combined data from TrueMedia API.

Fetches 2024+2025 data for returning Bryant players, and data from
transfer players' previous schools, then merges into a single pack.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from config import (
    BRYANT_TEAM_NAME, BRYANT_ROSTER_2026, BRYANT_JERSEY, BRYANT_POSITION,
    BRYANT_TRANSFERS, display_name,
)
from data.bryant_combined import build_bryant_combined_pack, load_bryant_combined_pack
from _pages.scouting import _get_opp_hitter_profile, _get_opp_pitcher_profile
from viz.layout import section_header


def _safe_fmt(v, fmt=".1f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:{fmt}}"


def _roster_table():
    rows = []
    for name in sorted(BRYANT_ROSTER_2026):
        prev = BRYANT_TRANSFERS.get(name)
        rows.append({
            "#": BRYANT_JERSEY.get(name, "-"),
            "Name": display_name(name, escape_html=False),
            "Pos": BRYANT_POSITION.get(name, "-"),
            "Source": prev if prev else "Bryant",
        })
    return pd.DataFrame(rows)


def _tab_team_overview(pack, n_hitters, n_pitchers):
    section_header("Bryant 2026 — Combined 2024-25 Overview")

    st.markdown("**Roster & Data Sources**")
    roster_df = _roster_table()
    st.dataframe(roster_df, use_container_width=True, hide_index=True, height=400)
    st.caption(f"Data found: {n_hitters} hitters, {n_pitchers} pitchers (from Bryant + transfer schools)")

    h_rate = pack.get("hitting", {}).get("rate")
    if h_rate is not None and not h_rate.empty:
        section_header("Team Hitting (Combined)")
        agg_cols = ["PA", "AVG", "OBP", "SLG", "OPS", "WOBA", "K%", "BB%"]
        available = [c for c in agg_cols if c in h_rate.columns]
        if available:
            totals = {}
            for c in available:
                vals = pd.to_numeric(h_rate[c], errors="coerce").dropna()
                if not vals.empty:
                    totals[c] = vals.sum() if c == "PA" else vals.mean()
            if totals:
                cols = st.columns(min(len(totals), 8))
                for i, (k, v) in enumerate(totals.items()):
                    with cols[i]:
                        fmt = ".0f" if k == "PA" else ".3f" if k in ("AVG", "OBP", "SLG", "OPS", "WOBA") else ".1f"
                        st.metric(k, f"{v:{fmt}}")

    p_trad = pack.get("pitching", {}).get("traditional")
    if p_trad is not None and not p_trad.empty:
        section_header("Team Pitching (Combined)")
        p_rate = pack.get("pitching", {}).get("rate", p_trad)
        src = p_rate if isinstance(p_rate, pd.DataFrame) and not p_rate.empty else p_trad
        agg_cols = ["IP", "ERA", "WHIP", "K%", "BB%", "FIP"]
        available = [c for c in agg_cols if c in src.columns]
        if available:
            totals = {}
            for c in available:
                vals = pd.to_numeric(src[c], errors="coerce").dropna()
                if not vals.empty:
                    totals[c] = vals.sum() if c == "IP" else vals.mean()
            if totals:
                cols = st.columns(min(len(totals), 6))
                for i, (k, v) in enumerate(totals.items()):
                    with cols[i]:
                        fmt = ".1f" if k in ("IP", "ERA", "FIP") else ".2f" if k == "WHIP" else ".1f"
                        st.metric(k, f"{v:{fmt}}")


def _tab_hitters(pack, team_name):
    section_header("Hitter Profiles")
    h_rate = pack.get("hitting", {}).get("rate")
    if h_rate is None or h_rate.empty:
        st.warning("No hitter data available.")
        return

    name_col = "playerFullName" if "playerFullName" in h_rate.columns else "fullName"
    if name_col not in h_rate.columns:
        st.error("Hitter table missing player name column.")
        return

    hitters = sorted(h_rate[name_col].dropna().astype(str).unique().tolist())
    selected = st.selectbox("Select Hitter", hitters, index=0)

    profile = _get_opp_hitter_profile(pack, selected, team_name, pitch_df=None)

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    with c1:
        st.metric("PA", _safe_fmt(profile.get("pa"), ".0f"))
    with c2:
        st.metric("OPS", _safe_fmt(profile.get("ops"), ".3f"))
    with c3:
        st.metric("wOBA", _safe_fmt(profile.get("woba"), ".3f"))
    with c4:
        st.metric("K%", _safe_fmt(profile.get("k_pct"), ".1f"))
    with c5:
        st.metric("BB%", _safe_fmt(profile.get("bb_pct"), ".1f"))
    with c6:
        st.metric("Chase%", _safe_fmt(profile.get("chase_pct"), ".1f"))
    with c7:
        st.metric("EV", _safe_fmt(profile.get("ev"), ".1f"))
    with c8:
        st.metric("Barrel%", _safe_fmt(profile.get("barrel_pct"), ".1f"))

    st.markdown("**Platoon Splits**")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.metric("wOBA vs LHP", _safe_fmt(profile.get("woba_lhp"), ".3f"))
    with pc2:
        st.metric("wOBA vs RHP", _safe_fmt(profile.get("woba_rhp"), ".3f"))

    pct_data = profile.get("pitch_type_pcts", {})
    if pct_data:
        st.markdown("**Pitch Type Seen %**")
        rows = [{"Pitch": k, "Pct": f"{v:.1f}%"} for k, v in sorted(pct_data.items(), key=lambda x: -x[1])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    zv = profile.get("zone_vuln", {})
    if zv.get("available"):
        st.markdown("**Zone Vulnerability**")
        st.caption(f"H: {zv.get('horizontal_pattern', '-')} | V: {zv.get('vertical_pattern', '-')}")
        zc1, zc2, zc3, zc4 = st.columns(4)
        with zc1:
            st.metric("Vuln Up", _safe_fmt(zv.get("vuln_up"), ".0f"))
        with zc2:
            st.metric("Vuln Down", _safe_fmt(zv.get("vuln_down"), ".0f"))
        with zc3:
            st.metric("Vuln Inside", _safe_fmt(zv.get("vuln_inside"), ".0f"))
        with zc4:
            st.metric("Vuln Away", _safe_fmt(zv.get("vuln_away"), ".0f"))


def _tab_pitchers(pack, team_name):
    section_header("Pitcher Profiles")
    p_trad = pack.get("pitching", {}).get("traditional")
    if p_trad is None or p_trad.empty:
        st.warning("No pitcher data available.")
        return

    name_col = "playerFullName" if "playerFullName" in p_trad.columns else "fullName"
    if name_col not in p_trad.columns:
        st.error("Pitcher table missing player name column.")
        return

    pitchers = sorted(p_trad[name_col].dropna().astype(str).unique().tolist())
    selected = st.selectbox("Select Pitcher", pitchers, index=0)

    profile = _get_opp_pitcher_profile(pack, selected, team_name)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("ERA", _safe_fmt(profile.get("era"), ".2f"))
    with c2:
        st.metric("IP", _safe_fmt(profile.get("ip"), ".1f"))
    with c3:
        st.metric("K%", _safe_fmt(profile.get("k_pct"), ".1f"))
    with c4:
        st.metric("BB%", _safe_fmt(profile.get("bb_pct"), ".1f"))
    with c5:
        st.metric("WHIP", _safe_fmt(profile.get("whip"), ".2f"))
    with c6:
        st.metric("FIP", _safe_fmt(profile.get("fip"), ".2f"))

    arsenal = profile.get("arsenal", {})
    if arsenal:
        st.markdown("**Arsenal**")
        rows = []
        for pt_name, pt_data in arsenal.items():
            rows.append({
                "Pitch": pt_name,
                "Usage%": _safe_fmt(pt_data.get("usage_pct"), ".1f"),
                "Velo": _safe_fmt(pt_data.get("avg_velo"), ".1f"),
                "IVB": _safe_fmt(pt_data.get("ivb"), ".1f"),
                "HB": _safe_fmt(pt_data.get("hb"), ".1f"),
                "Stuff+": _safe_fmt(pt_data.get("stuff_plus"), ".0f"),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_bryant(data):
    """Main Bryant scouting page — builds combined 2024-25 pack."""
    st.title("Bryant Scouting")
    st.caption(
        "Combined 2024 + 2025 data for the Bryant 2026 roster. "
        "Transfer players include data from their previous schools."
    )

    team_name = "Bryant (2024-25 Combined)"

    col_r, col_s = st.columns([1, 2])
    with col_r:
        refresh = st.checkbox("Force Refresh", value=False, key="bryant_refresh")
    with col_s:
        seasons = st.multiselect("Seasons to include", [2024, 2025], default=[2024, 2025], key="bryant_seasons")

    # Try cached pack first
    pack = None
    if not refresh:
        pack = load_bryant_combined_pack()
        if pack is not None:
            h_rate = pack.get("hitting", {}).get("rate")
            if h_rate is None or (isinstance(h_rate, pd.DataFrame) and h_rate.empty):
                pack = None  # stale/empty cache

    if pack is None or refresh:
        st.info("Building combined Bryant pack from TrueMedia API (this may take a minute)...")
        log_area = st.empty()
        log_lines = []

        def _progress(msg):
            log_lines.append(msg)
            log_area.text("\n".join(log_lines[-8:]))

        with st.spinner("Fetching and merging data..."):
            pack = build_bryant_combined_pack(
                refresh=True,
                seasons=sorted(seasons),
                progress_callback=_progress,
            )
        log_area.empty()

    # Count players found
    h_rate = pack.get("hitting", {}).get("rate")
    p_trad = pack.get("pitching", {}).get("traditional")
    n_hitters = len(h_rate) if isinstance(h_rate, pd.DataFrame) and not h_rate.empty else 0
    n_pitchers = len(p_trad) if isinstance(p_trad, pd.DataFrame) and not p_trad.empty else 0

    if n_hitters == 0 and n_pitchers == 0:
        st.error("No player data found. Check API connectivity and try Force Refresh.")
        return

    st.success(f"Bryant 2026 combined pack: {n_hitters} hitters, {n_pitchers} pitchers")

    tab1, tab2, tab3 = st.tabs(["Team Overview", "Hitters", "Pitchers"])
    with tab1:
        _tab_team_overview(pack, n_hitters, n_pitchers)
    with tab2:
        _tab_hitters(pack, team_name)
    with tab3:
        _tab_pitchers(pack, team_name)
