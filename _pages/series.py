"""Interactive Series Report page — combines multiple games."""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

from config import (
    DAVIDSON_TEAM_ID, JERSEY, POSITION, PITCH_COLORS,
    SWING_CALLS, CONTACT_CALLS,
    in_zone_mask, is_barrel_mask, display_name, _friendly_team_name,
)
from viz.layout import section_header
from viz.percentiles import render_savant_percentile_section
from viz.charts import make_spray_chart, make_movement_profile

from _pages.postgame import (
    _compute_pitcher_grades,
    _compute_hitter_grades,
    _compute_pitcher_percentile_metrics,
    _compute_hitter_percentile_metrics,
    _grade_at_bat,
    _compute_takeaways,
    _pg_estimate_ip,
    _pg_slug,
    _render_grade_header,
    _render_grade_cards,
    _render_radar_chart,
    _render_stuff_cmd_bars,
    _split_feedback,
    _MIN_PITCHER_PITCHES,
    _MIN_HITTER_PAS,
)


def _detect_series(games_df):
    """Auto-detect series: 2+ games vs same opponent within 4 days."""
    series_list = []
    if games_df.empty or "Date" not in games_df.columns:
        return series_list
    games_df = games_df.copy()
    games_df["_date"] = pd.to_datetime(games_df["Date"])

    # Group by opponent
    for _, grp in games_df.groupby("_opp"):
        grp = grp.sort_values("_date")
        dates = grp["_date"].tolist()
        ids = grp["GameID"].tolist()
        # Sliding window to find clusters within 4 days
        cluster = [0]
        for i in range(1, len(dates)):
            if (dates[i] - dates[cluster[0]]).days <= 4:
                cluster.append(i)
            else:
                if len(cluster) >= 2:
                    series_list.append({
                        "game_ids": [ids[j] for j in cluster],
                        "opp": grp.iloc[cluster[0]]["_opp"],
                        "start": dates[cluster[0]],
                        "end": dates[cluster[-1]],
                    })
                cluster = [i]
        if len(cluster) >= 2:
            series_list.append({
                "game_ids": [ids[j] for j in cluster],
                "opp": grp.iloc[cluster[0]]["_opp"],
                "start": dates[cluster[0]],
                "end": dates[cluster[-1]],
            })
    return series_list


def _series_record(combined_gd, game_ids):
    """Compute W-L record and per-game scores."""
    dav_wins, dav_losses = 0, 0
    game_scores = []
    for gid in game_ids:
        gd = combined_gd[combined_gd["GameID"] == gid]
        if gd.empty:
            continue
        date_str = ""
        if "Date" in gd.columns:
            d = gd["Date"].iloc[0]
            if pd.notna(d):
                date_str = pd.Timestamp(d).strftime("%b %d")
        innings = int(gd["Inning"].max()) if "Inning" in gd.columns else "?"
        dav_r, opp_r = 0, 0
        if "RunsScored" in gd.columns:
            dav_r = int(gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID]["RunsScored"].sum())
            opp_r = int(gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID]["RunsScored"].sum())
        result = "W" if dav_r > opp_r else "L"
        if dav_r > opp_r:
            dav_wins += 1
        else:
            dav_losses += 1
        home = _friendly_team_name(gd["HomeTeam"].iloc[0]) if "HomeTeam" in gd.columns else "?"
        away = _friendly_team_name(gd["AwayTeam"].iloc[0]) if "AwayTeam" in gd.columns else "?"
        opp_name = away if away != "Davidson" else home
        game_scores.append({
            "Date": date_str, "Innings": innings,
            "DAV": dav_r, "OPP": opp_r, "Result": result,
            "Opp": opp_name, "GameID": gid,
        })
    return dav_wins, dav_losses, game_scores


def _render_series_overview(combined_gd, game_ids, data):
    """Tab 1: Series Overview."""
    dav_wins, dav_losses, game_scores = _series_record(combined_gd, game_ids)

    # Series record
    opp_name = game_scores[0]["Opp"] if game_scores else "Opponent"
    st.markdown(
        f'<div style="text-align:center;padding:10px 0;">'
        f'<span style="font-size:28px;font-weight:900;">Davidson {dav_wins} - {dav_losses} {opp_name}</span>'
        f'</div>', unsafe_allow_html=True)

    # Game-by-game scores
    if game_scores:
        score_df = pd.DataFrame(game_scores)[["Date", "Innings", "DAV", "OPP", "Result"]]
        st.dataframe(score_df, use_container_width=True, hide_index=True)

    # Combined stats
    dav_pitching = combined_gd[combined_gd["PitcherTeam"] == DAVIDSON_TEAM_ID]
    dav_hitting = combined_gd[combined_gd["BatterTeam"] == DAVIDSON_TEAM_ID]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total_ip = _pg_estimate_ip(dav_pitching)
        ks = (dav_pitching["KorBB"] == "Strikeout").sum() if "KorBB" in dav_pitching.columns else 0
        bbs = (dav_pitching["KorBB"] == "Walk").sum() if "KorBB" in dav_pitching.columns else 0
        st.metric("IP", total_ip)
        st.metric("K / BB", f"{ks} / {bbs}")
    with c2:
        hits = dav_hitting["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum() if "PlayResult" in dav_hitting.columns else 0
        hrs = (dav_hitting["PlayResult"] == "HomeRun").sum() if "PlayResult" in dav_hitting.columns else 0
        st.metric("Hits", int(hits))
        st.metric("HR", int(hrs))
    with c3:
        bbe = dav_hitting[(dav_hitting["PitchCall"] == "InPlay") & dav_hitting["ExitSpeed"].notna()] if "ExitSpeed" in dav_hitting.columns else pd.DataFrame()
        avg_ev = f"{bbe['ExitSpeed'].mean():.1f}" if len(bbe) > 0 else "-"
        st.metric("Avg EV", avg_ev)
    with c4:
        loc_df = dav_hitting.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if not loc_df.empty:
            iz = in_zone_mask(loc_df)
            oz = loc_df[~iz]
            oz_sw = oz[oz["PitchCall"].isin(SWING_CALLS)]
            chase = f"{len(oz_sw)/max(len(oz),1)*100:.0f}%"
        else:
            chase = "-"
        st.metric("Chase%", chase)

    # Key takeaways
    section_header("Key Takeaways")
    pitching_bullets, hitting_bullets = _compute_takeaways(combined_gd, data)
    if pitching_bullets:
        st.markdown("**Pitching**")
        for b in pitching_bullets:
            st.markdown(f"- {b}")
    if hitting_bullets:
        st.markdown("**Hitting**")
        for b in hitting_bullets:
            st.markdown(f"- {b}")


def _render_pitching_staff(combined_gd, data):
    """Tab 2: Pitching Staff."""
    dav_pitching = combined_gd[combined_gd["PitcherTeam"] == DAVIDSON_TEAM_ID].copy()
    if dav_pitching.empty:
        st.info("No Davidson pitching data.")
        return

    # Staff summary table
    section_header("Staff Summary")
    pitchers = dav_pitching.groupby("Pitcher").size().sort_values(ascending=False).index.tolist()
    summary_rows = []
    for pitcher in pitchers:
        pdf = dav_pitching[dav_pitching["Pitcher"] == pitcher]
        n = len(pdf)
        ip = _pg_estimate_ip(pdf)
        ks = (pdf["KorBB"] == "Strikeout").sum() if "KorBB" in pdf.columns else 0
        bbs = (pdf["KorBB"] == "Walk").sum() if "KorBB" in pdf.columns else 0
        hits = pdf["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum() if "PlayResult" in pdf.columns else 0
        velo = pd.to_numeric(pdf["RelSpeed"], errors="coerce").dropna() if "RelSpeed" in pdf.columns else pd.Series(dtype=float)
        avg_v = f"{velo.mean():.1f}" if len(velo) > 0 else "-"
        max_v = f"{velo.max():.1f}" if len(velo) > 0 else "-"
        csw = f"{pdf['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging']).mean()*100:.1f}%"
        summary_rows.append({
            "Pitcher": display_name(pitcher),
            "Pitches": n, "IP": ip, "K": int(ks), "BB": int(bbs),
            "H": int(hits), "Avg Velo": avg_v, "Max Velo": max_v, "CSW%": csw,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Individual pitchers
    for pitcher in pitchers:
        pdf = dav_pitching[dav_pitching["Pitcher"] == pitcher].copy()
        n_pitches = len(pdf)
        if n_pitches < 5:
            continue
        slug = _pg_slug(pitcher)
        season_pdf = data[data["Pitcher"] == pitcher] if data is not None else pd.DataFrame()

        grades, feedback, stuff_by_pt, cmd_df = _compute_pitcher_grades(pdf, data, pitcher)
        valid_scores = [v for v in grades.values() if v is not None]
        overall = np.mean(valid_scores) if valid_scores else None
        ip_est = _pg_estimate_ip(pdf)
        small_sample = n_pitches < _MIN_PITCHER_PITCHES

        with st.expander(f"{display_name(pitcher)} ({n_pitches} pitches, ~{ip_est} IP)"):
            _render_grade_header(pitcher, n_pitches, overall, f"~{ip_est} IP", small_sample=small_sample)

            # Feedback
            if feedback:
                fb_str, fb_area = _split_feedback(feedback, grades)
                if fb_str:
                    st.markdown("**Strengths**")
                    for fb in fb_str:
                        st.markdown(f'<div style="font-size:14px;padding:2px 0 2px 12px;color:#2e7d32;">+ {fb}</div>', unsafe_allow_html=True)
                if fb_area:
                    st.markdown("**Areas to Improve**")
                    for fb in fb_area:
                        st.markdown(f'<div style="font-size:14px;padding:2px 0 2px 12px;color:#c62828;">- {fb}</div>', unsafe_allow_html=True)

            if not small_sample:
                _render_grade_cards(grades)

            # Percentile bars
            pctl_metrics = _compute_pitcher_percentile_metrics(pdf, season_pdf)
            render_savant_percentile_section(pctl_metrics, title="Combined Percentiles")

            # Stuff+/Cmd+
            if stuff_by_pt:
                _render_stuff_cmd_bars(stuff_by_pt, cmd_df, key_suffix=f"ser_pit_{slug}")

            # Movement profile
            try:
                fig_mov = make_movement_profile(pdf)
                if fig_mov:
                    st.plotly_chart(fig_mov, use_container_width=True, key=f"ser_mov_{slug}")
            except Exception:
                pass


def _render_hitting_lineup(combined_gd, data):
    """Tab 3: Hitting Lineup."""
    dav_hitting = combined_gd[combined_gd["BatterTeam"] == DAVIDSON_TEAM_ID].copy()
    if dav_hitting.empty:
        st.info("No Davidson hitting data.")
        return

    # Lineup summary table
    section_header("Lineup Summary")
    batters = dav_hitting.groupby("Batter").size().sort_values(ascending=False).index.tolist()
    summary_rows = []
    for batter in batters:
        bdf = dav_hitting[dav_hitting["Batter"] == batter]
        pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in bdf.columns]
        pa = bdf.drop_duplicates(subset=pa_cols).shape[0] if len(pa_cols) >= 2 else 0
        pr = bdf["PlayResult"] if "PlayResult" in bdf.columns else pd.Series(dtype=str)
        hits = pr.isin(["Single", "Double", "Triple", "HomeRun"]).sum()
        hrs = (pr == "HomeRun").sum()
        bbs = (bdf["KorBB"] == "Walk").sum() if "KorBB" in bdf.columns else 0
        ks = (bdf["KorBB"] == "Strikeout").sum() if "KorBB" in bdf.columns else 0
        ip = bdf[bdf["PitchCall"] == "InPlay"]
        ev_data = pd.to_numeric(ip["ExitSpeed"], errors="coerce").dropna() if "ExitSpeed" in ip.columns else pd.Series(dtype=float)
        avg_ev = f"{ev_data.mean():.1f}" if len(ev_data) > 0 else "-"
        summary_rows.append({
            "Batter": display_name(batter),
            "PA": pa, "H": int(hits), "HR": int(hrs),
            "BB": int(bbs), "K": int(ks), "Avg EV": avg_ev,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Individual hitters
    for batter in batters:
        bdf = dav_hitting[dav_hitting["Batter"] == batter].copy()
        n_pitches = len(bdf)
        if n_pitches < 3:
            continue
        slug = _pg_slug(batter)
        season_bdf = data[data["Batter"] == batter] if data is not None else pd.DataFrame()

        grades, feedback = _compute_hitter_grades(bdf, data, batter)
        valid_scores = [v for v in grades.values() if v is not None]
        overall = np.mean(valid_scores) if valid_scores else None

        pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in bdf.columns]
        n_pas = bdf.drop_duplicates(subset=pa_cols).shape[0] if len(pa_cols) >= 2 else 0
        small_sample = n_pas < _MIN_HITTER_PAS

        with st.expander(f"{display_name(batter)} ({n_pas} PA)"):
            _render_grade_header(batter, n_pitches, overall, f"{n_pas} PA", small_sample=small_sample)

            # Feedback
            if feedback:
                fb_str, fb_area = _split_feedback(feedback, grades)
                if fb_str:
                    st.markdown("**Strengths**")
                    for fb in fb_str:
                        st.markdown(f'<div style="font-size:14px;padding:2px 0 2px 12px;color:#2e7d32;">+ {fb}</div>', unsafe_allow_html=True)
                if fb_area:
                    st.markdown("**Areas to Improve**")
                    for fb in fb_area:
                        st.markdown(f'<div style="font-size:14px;padding:2px 0 2px 12px;color:#c62828;">- {fb}</div>', unsafe_allow_html=True)

            if not small_sample:
                _render_grade_cards(grades)

            # Percentile bars
            pctl_metrics = _compute_hitter_percentile_metrics(bdf, season_bdf)
            render_savant_percentile_section(pctl_metrics, title="Combined Percentiles")

            # Spray chart
            in_play = bdf[bdf["PitchCall"] == "InPlay"]
            if not in_play.empty:
                try:
                    fig_spray = make_spray_chart(in_play)
                    if fig_spray:
                        st.plotly_chart(fig_spray, use_container_width=True, key=f"ser_spray_{slug}")
                except Exception:
                    pass

            # ALL at-bat grades (no cap)
            sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in bdf.columns]
            if len(pa_cols) >= 2:
                ab_rows = []
                for pa_key, ab in bdf.groupby(pa_cols[1:]):
                    ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
                    score, letter, result = _grade_at_bat(ab_sorted, season_bdf)
                    inn = ab_sorted.iloc[0].get("Inning", "?")
                    vs_pitcher = display_name(ab_sorted.iloc[0]["Pitcher"]) if "Pitcher" in ab_sorted.columns else "?"
                    game_id = ab_sorted.iloc[0].get("GameID", "")
                    game_label = ""
                    if game_id and "Date" in ab_sorted.columns:
                        d = ab_sorted.iloc[0].get("Date")
                        if pd.notna(d):
                            game_label = pd.Timestamp(d).strftime("%m/%d")
                    ab_rows.append({
                        "Game": game_label, "Inning": inn,
                        "vs Pitcher": vs_pitcher,
                        "Pitches": len(ab_sorted),
                        "Result": result, "Score": score, "Grade": letter,
                    })
                if ab_rows:
                    ab_table = pd.DataFrame(ab_rows).sort_values(["Game", "Inning"])
                    st.markdown("**All At-Bat Grades**")
                    st.dataframe(ab_table, use_container_width=True, hide_index=True)


def _render_all_at_bats(combined_gd, data):
    """Tab 4: All At-Bats filterable table."""
    dav_hitting = combined_gd[combined_gd["BatterTeam"] == DAVIDSON_TEAM_ID].copy()
    if dav_hitting.empty:
        st.info("No Davidson hitting data.")
        return

    section_header("All At-Bats")

    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in dav_hitting.columns]
    sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in dav_hitting.columns]

    all_ab_rows = []
    if len(pa_cols) >= 2:
        batters = dav_hitting["Batter"].unique()
        for batter in batters:
            bdf = dav_hitting[dav_hitting["Batter"] == batter]
            season_bdf = data[data["Batter"] == batter] if data is not None else pd.DataFrame()
            for pa_key, ab in bdf.groupby(pa_cols[1:]):
                ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
                score, letter, result = _grade_at_bat(ab_sorted, season_bdf)
                inn = ab_sorted.iloc[0].get("Inning", "?")
                vs_pitcher = display_name(ab_sorted.iloc[0]["Pitcher"]) if "Pitcher" in ab_sorted.columns else "?"
                game_id = ab_sorted.iloc[0].get("GameID", "")
                game_label = ""
                if game_id and "Date" in ab_sorted.columns:
                    d = ab_sorted.iloc[0].get("Date")
                    if pd.notna(d):
                        game_label = pd.Timestamp(d).strftime("%m/%d")
                all_ab_rows.append({
                    "Game": game_label,
                    "Inning": inn,
                    "Batter": display_name(batter),
                    "vs Pitcher": vs_pitcher,
                    "Pitches": len(ab_sorted),
                    "Result": result,
                    "Score": score,
                    "Grade": letter,
                })

    if not all_ab_rows:
        st.info("No at-bat data available.")
        return

    ab_df = pd.DataFrame(all_ab_rows)

    # Filters
    fc1, fc2 = st.columns(2)
    with fc1:
        batter_filter = st.multiselect("Filter by Batter", sorted(ab_df["Batter"].unique()),
                                        key="ser_ab_batter_filter")
    with fc2:
        grade_filter = st.multiselect("Filter by Grade", sorted(ab_df["Grade"].unique()),
                                       key="ser_ab_grade_filter")

    filtered = ab_df.copy()
    if batter_filter:
        filtered = filtered[filtered["Batter"].isin(batter_filter)]
    if grade_filter:
        filtered = filtered[filtered["Grade"].isin(grade_filter)]

    filtered = filtered.sort_values(["Game", "Inning"])
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.caption(f"{len(filtered)} at-bats shown")


def page_series(data):
    """Series Report page — multi-game selector with combined analytics."""
    st.title("Series Report")

    # Filter to Davidson games
    dav = data[(data["PitcherTeam"] == DAVIDSON_TEAM_ID) | (data["BatterTeam"] == DAVIDSON_TEAM_ID)]
    if dav.empty:
        st.warning("No Davidson game data available.")
        return

    # Build game list — only real games (exclude private/intrasquad)
    games = dav.groupby(["Date", "GameID"]).agg(
        Home=("HomeTeam", "first"),
        Away=("AwayTeam", "first"),
        Pitches=("PitchNo", "count"),
    ).reset_index().sort_values("Date", ascending=False)

    # Filter out private / intrasquad games
    games = games[~games["GameID"].str.contains("Private", case=False, na=False)]

    if games.empty:
        st.warning("No games found.")
        return

    # Determine opponent for each game
    games["_opp"] = games.apply(
        lambda r: r["Away"] if _friendly_team_name(r["Home"]) == "Davidson" else r["Home"], axis=1)

    # Filter to current season (2026)
    games["_date"] = pd.to_datetime(games["Date"])
    games = games[games["_date"].dt.year == 2026].drop(columns=["_date"])

    if games.empty:
        st.warning("No 2026 games found.")
        return

    # Detect series
    detected_series = _detect_series(games)

    # Quick-pick buttons for detected series
    if detected_series:
        st.markdown("**Quick Select Series:**")
        qp_cols = st.columns(min(len(detected_series), 4))
        for i, series in enumerate(detected_series[:4]):
            with qp_cols[i]:
                opp_display = _friendly_team_name(series["opp"])
                start_str = series["start"].strftime("%m/%d")
                end_str = series["end"].strftime("%m/%d")
                n_games = len(series["game_ids"])
                if st.button(f"{opp_display} ({start_str}-{end_str}, {n_games}G)",
                             key=f"ser_qp_{i}"):
                    st.session_state.ser_selected_games = series["game_ids"]

    # Game multi-select
    game_labels = {}
    for _, row in games.iterrows():
        dt = row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "?"
        opp = _friendly_team_name(row["_opp"])
        game_labels[row["GameID"]] = f"{dt}  vs {opp}  ({row['Pitches']} pitches)"

    default_sel = st.session_state.get("ser_selected_games", [])
    # Ensure default selections are valid game IDs
    valid_ids = set(games["GameID"].tolist())
    default_sel = [g for g in default_sel if g in valid_ids]

    selected_games = st.multiselect(
        "Select Games", games["GameID"].tolist(),
        default=default_sel,
        format_func=lambda g: game_labels.get(g, str(g)),
        key="ser_game_select",
    )

    if not selected_games:
        st.info("Select 2 or more games to generate a series report.")
        return

    combined_gd = data[data["GameID"].isin(selected_games)].copy()
    if combined_gd.empty:
        st.warning("No data for selected games.")
        return

    # Series label for PDF
    opp_ids = combined_gd["AwayTeam"].unique().tolist() + combined_gd["HomeTeam"].unique().tolist()
    opp_names = set(_friendly_team_name(t) for t in opp_ids if _friendly_team_name(t) != "Davidson")
    opp_display = ", ".join(opp_names) if opp_names else "Opponent"
    dates = pd.to_datetime(combined_gd["Date"]).dropna()
    if not dates.empty:
        date_range = f"{dates.min().strftime('%b %d')}-{dates.max().strftime('%b %d, %Y')}"
    else:
        date_range = ""
    series_label = f"{opp_display} @ Davidson  |  {date_range}" if date_range else f"vs {opp_display}"

    # PDF export button
    col_info, col_export = st.columns([4, 1])
    with col_info:
        st.caption(f"{len(selected_games)} games selected  |  {len(combined_gd)} total pitches")
    with col_export:
        if st.button("Export Series PDF", key="ser_export_pdf"):
            with st.spinner("Building Series PDF..."):
                from generate_series_report_pdf import generate_series_pdf_bytes
                st.session_state.ser_pdf_bytes = generate_series_pdf_bytes(
                    selected_games, data, series_label)

        if st.session_state.get("ser_pdf_bytes"):
            st.download_button(
                "Download PDF", data=st.session_state.ser_pdf_bytes,
                file_name="series_report.pdf", mime="application/pdf")

    # Section selector — only renders the active section (performance)
    _ser_tabs = ["Series Overview", "Pitching Staff", "Hitting Lineup", "All At-Bats"]
    _active_ser = st.radio(
        "Section", _ser_tabs, horizontal=True, key="ser_tab_select",
        label_visibility="collapsed")

    if _active_ser == "Series Overview":
        _render_series_overview(combined_gd, selected_games, data)
    elif _active_ser == "Pitching Staff":
        _render_pitching_staff(combined_gd, data)
    elif _active_ser == "Hitting Lineup":
        _render_hitting_lineup(combined_gd, data)
    elif _active_ser == "All At-Bats":
        _render_all_at_bats(combined_gd, data)
