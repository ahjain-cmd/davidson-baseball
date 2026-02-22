"""Defensive Positioning page."""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import (
    DAVIDSON_TEAM_ID, ROSTER_2026, PITCH_COLORS,
    SWING_CALLS, _APP_DIR, DATA_ROOT,
    display_name,
)
from data.loader import get_all_seasons, query_population
from viz.layout import CHART_LAYOUT, section_header
from analytics.gb_positioning import (
    compute_pitcher_batter_positioning,
    MatchupPositioning,
)


def _compute_spray_zones(batted_df, batter_side):
    """Compute spray zone stats from batted ball data.
    Returns dict of zone -> {centroid_x, centroid_y, count, out_rate, avg_ev, avg_dist, hit_rate}."""
    df = batted_df.dropna(subset=["Direction", "Distance"]).copy()
    if df.empty:
        return {}
    angle_rad = np.radians(df["Direction"])
    df["x"] = df["Distance"] * np.sin(angle_rad)
    df["y"] = df["Distance"] * np.cos(angle_rad)

    # Classify direction: Pull/Center/Oppo based on batter side
    # Trackman: positive Direction = right field
    if batter_side == "Right":
        df["FieldDir"] = np.where(df["Direction"] < -15, "Pull",
                         np.where(df["Direction"] > 15, "Oppo", "Center"))
    else:
        df["FieldDir"] = np.where(df["Direction"] > 15, "Pull",
                         np.where(df["Direction"] < -15, "Oppo", "Center"))

    df["FieldDepth"] = np.where(df["Distance"] < 180, "IF", "OF")
    df["Zone"] = df["FieldDepth"] + "-" + df["FieldDir"]

    zones = {}
    for zone_name, zdf in df.groupby("Zone"):
        is_out = zdf["PlayResult"].isin(["Out", "Sacrifice", "FieldersChoice"]) if "PlayResult" in zdf.columns else pd.Series([False]*len(zdf))
        is_hit = zdf["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]) if "PlayResult" in zdf.columns else pd.Series([False]*len(zdf))
        zones[zone_name] = {
            "centroid_x": zdf["x"].mean(),
            "centroid_y": zdf["y"].mean(),
            "count": len(zdf),
            "out_rate": is_out.sum() / max(len(zdf), 1) * 100,
            "hit_rate": is_hit.sum() / max(len(zdf), 1) * 100,
            "avg_ev": zdf["ExitSpeed"].mean() if "ExitSpeed" in zdf.columns and zdf["ExitSpeed"].notna().any() else np.nan,
            "avg_dist": zdf["Distance"].mean(),
        }
    return zones


def _recommend_fielder_positions(batted_df, batter_side):
    """Compute recommended fielder positions based on spray data.
    Returns dict of position_name -> (x, y) in spray chart coordinates."""
    df = batted_df.dropna(subset=["Direction", "Distance"]).copy()
    if df.empty:
        return {}
    angle_rad = np.radians(df["Direction"])
    df["x"] = df["Distance"] * np.sin(angle_rad)
    df["y"] = df["Distance"] * np.cos(angle_rad)

    # Classify direction
    if batter_side == "Right":
        df["FieldDir"] = np.where(df["Direction"] < -15, "Pull",
                         np.where(df["Direction"] > 15, "Oppo", "Center"))
    else:
        df["FieldDir"] = np.where(df["Direction"] > 15, "Pull",
                         np.where(df["Direction"] < -15, "Oppo", "Center"))

    # Weight by damage potential: higher EV + hits weighted more
    df["weight"] = 1.0
    if "ExitSpeed" in df.columns:
        df.loc[df["ExitSpeed"].notna(), "weight"] = df.loc[df["ExitSpeed"].notna(), "ExitSpeed"] / 85.0
    if "PlayResult" in df.columns:
        df.loc[df["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]), "weight"] *= 1.5

    # Compute weighted centroids for each fielder position
    positions = {}
    # Ground balls -> infielders
    gb = df[(df["TaggedHitType"] == "GroundBall") | (df["Distance"] < 180)] if "TaggedHitType" in df.columns else df[df["Distance"] < 180]
    # Air balls -> outfielders
    air = df[(df["TaggedHitType"].isin(["FlyBall", "LineDrive"])) | (df["Distance"] >= 180)] if "TaggedHitType" in df.columns else df[df["Distance"] >= 180]

    pull_sign = -1 if batter_side == "Right" else 1

    # Infield positions — weighted centroids by pull/center/oppo
    gb_pull = gb[gb["FieldDir"] == "Pull"]
    gb_center = gb[gb["FieldDir"] == "Center"]
    gb_oppo = gb[gb["FieldDir"] == "Oppo"]

    # 3B: Pull side infield
    if batter_side == "Right":
        positions["3B"] = _weighted_centroid(gb_pull, fallback_x=-60, fallback_y=80)
        positions["1B"] = _weighted_centroid(gb_oppo, fallback_x=60, fallback_y=80)
    else:
        positions["3B"] = _weighted_centroid(gb_oppo, fallback_x=-60, fallback_y=80)
        positions["1B"] = _weighted_centroid(gb_pull, fallback_x=60, fallback_y=80)

    # SS and 2B: middle infield
    ss_df = gb[(gb["x"] < 0) & (gb["Distance"].between(50, 200))]
    _2b_df = gb[(gb["x"] >= 0) & (gb["Distance"].between(50, 200))]
    positions["SS"] = _weighted_centroid(ss_df, fallback_x=-40, fallback_y=120)
    positions["2B"] = _weighted_centroid(_2b_df, fallback_x=40, fallback_y=120)

    # Outfield positions
    of_pull = air[air["FieldDir"] == "Pull"]
    of_center = air[air["FieldDir"] == "Center"]
    of_oppo = air[air["FieldDir"] == "Oppo"]

    if batter_side == "Right":
        positions["LF"] = _weighted_centroid(of_pull, fallback_x=-180, fallback_y=220)
        positions["RF"] = _weighted_centroid(of_oppo, fallback_x=180, fallback_y=220)
    else:
        positions["LF"] = _weighted_centroid(of_oppo, fallback_x=-180, fallback_y=220)
        positions["RF"] = _weighted_centroid(of_pull, fallback_x=180, fallback_y=220)

    positions["CF"] = _weighted_centroid(of_center, fallback_x=0, fallback_y=280)

    return positions


def _weighted_centroid(df, fallback_x, fallback_y):
    """Compute damage-weighted centroid of batted balls."""
    if df.empty or len(df) < 2:
        return (fallback_x, fallback_y)
    w = df["weight"].values if "weight" in df.columns else np.ones(len(df))
    w_sum = w.sum()
    if w_sum == 0:
        return (fallback_x, fallback_y)
    cx = np.average(df["x"].values, weights=w)
    cy = np.average(df["y"].values, weights=w)
    return (cx, cy)


def _load_positioning_csvs(game_ids):
    """Load positioning CSVs matching given GameIDs.
    Scans DATA_ROOT parent and home directory for *_playerpositioning_FHC.csv files."""
    import re
    search_dirs = [
        os.path.dirname(DATA_ROOT),  # parent of v3
        os.path.expanduser("~"),     # home directory
        _APP_DIR,                    # app directory
    ]
    pos_files = []
    for d in search_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith("_playerpositioning_FHC.csv"):
                    pos_files.append(os.path.join(d, f))

    if not pos_files:
        return pd.DataFrame()

    frames = []
    for fp in pos_files:
        fname = os.path.basename(fp)
        # Extract GameID from filename: YYYYMMDD-Venue-Type-Num_unverified_playerpositioning_FHC.csv
        match = re.match(r"(.+?)_unverified_playerpositioning", fname)
        if not match:
            continue
        file_game_id = match.group(1)
        if game_ids is not None and file_game_id not in game_ids:
            continue
        try:
            df = pd.read_csv(fp)
            df["_GameID"] = file_game_id
            frames.append(df)
        except Exception:
            continue

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _draw_defensive_field(batted_df, recommended=None, actual_positions=None, height=500,
                          color_by="PlayResult", title=""):
    """Draw baseball field with spray data and defensive positions overlaid."""
    fig = go.Figure()

    # Grass fill
    theta_grass = np.linspace(-np.pi / 4, np.pi / 4, 80)
    grass_r = 400
    grass_x = [0] + list(grass_r * np.sin(theta_grass)) + [0]
    grass_y = [0] + list(grass_r * np.cos(theta_grass)) + [0]
    fig.add_trace(go.Scatter(x=grass_x, y=grass_y, mode="lines",
                             fill="toself", fillcolor="rgba(76,160,60,0.06)",
                             line=dict(color="rgba(76,160,60,0.15)", width=1), showlegend=False,
                             hoverinfo="skip"))

    # Infield diamond
    diamond_x = [0, 63.6, 0, -63.6, 0]
    diamond_y = [0, 63.6, 127.3, 63.6, 0]
    fig.add_trace(go.Scatter(x=diamond_x, y=diamond_y, mode="lines",
                             line=dict(color="rgba(160,120,60,0.25)", width=1), showlegend=False,
                             fill="toself", fillcolor="rgba(160,120,60,0.06)", hoverinfo="skip"))

    # Foul lines
    fl = 350
    fig.add_trace(go.Scatter(x=[0, -fl * np.sin(np.pi / 4)], y=[0, fl * np.cos(np.pi / 4)],
                             mode="lines", line=dict(color="rgba(0,0,0,0.12)", width=1),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, fl * np.sin(np.pi / 4)], y=[0, fl * np.cos(np.pi / 4)],
                             mode="lines", line=dict(color="rgba(0,0,0,0.12)", width=1),
                             showlegend=False, hoverinfo="skip"))

    # Batted balls
    if batted_df is not None and not batted_df.empty:
        spray = batted_df.dropna(subset=["Direction", "Distance"]).copy()
        if not spray.empty:
            angle_rad = np.radians(spray["Direction"])
            spray["x"] = spray["Distance"] * np.sin(angle_rad)
            spray["y"] = spray["Distance"] * np.cos(angle_rad)

            if color_by == "PlayResult" and "PlayResult" in spray.columns:
                result_colors = {"Out": "#999", "Single": "#2ca02c", "Double": "#f7c631",
                                 "Triple": "#fe6100", "HomeRun": "#d22d49",
                                 "Error": "#9467bd", "FieldersChoice": "#bbb",
                                 "Sacrifice": "#bbb"}
                for res, clr in result_colors.items():
                    sub = spray[spray["PlayResult"] == res]
                    if sub.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=sub["x"], y=sub["y"], mode="markers",
                        marker=dict(size=6, color=clr, opacity=0.7,
                                    line=dict(width=0.3, color="white")),
                        name=res,
                        hovertemplate="EV: %{customdata[0]:.1f}<br>Dist: %{customdata[1]:.0f}ft<extra></extra>",
                        customdata=sub[["ExitSpeed", "Distance"]].fillna(0).values,
                    ))
            else:
                ev_vals = spray["ExitSpeed"].fillna(80) if "ExitSpeed" in spray.columns else pd.Series([80]*len(spray))
                fig.add_trace(go.Scatter(
                    x=spray["x"], y=spray["y"], mode="markers",
                    marker=dict(size=6, color=ev_vals,
                                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                cmin=60, cmax=105, showscale=True,
                                colorbar=dict(title="EV", len=0.6),
                                line=dict(width=0.3, color="white")),
                    name="Batted Balls", showlegend=False,
                ))

    # Recommended positions (star markers)
    if recommended:
        pos_colors = {"1B": "#d62728", "2B": "#ff7f0e", "SS": "#2ca02c",
                      "3B": "#1f77b4", "LF": "#9467bd", "CF": "#8c564b", "RF": "#e377c2"}
        for pos_name, (px, py) in recommended.items():
            fig.add_trace(go.Scatter(
                x=[px], y=[py], mode="markers+text",
                marker=dict(size=18, color=pos_colors.get(pos_name, "#333"),
                            symbol="star", line=dict(width=2, color="white")),
                text=[pos_name], textposition="top center",
                textfont=dict(size=11, color=pos_colors.get(pos_name, "#333")),
                name=f"Rec: {pos_name}", showlegend=False,
                hovertemplate=f"<b>{pos_name}</b><br>Recommended<br>({px:.0f}, {py:.0f})<extra></extra>",
            ))

    # Actual positions (diamond markers)
    if actual_positions:
        for pos_name, (px, py) in actual_positions.items():
            fig.add_trace(go.Scatter(
                x=[px], y=[py], mode="markers+text",
                marker=dict(size=14, color="#000000", symbol="diamond",
                            line=dict(width=2, color="white")),
                text=[pos_name], textposition="bottom center",
                textfont=dict(size=9, color="#000000"),
                name=f"Actual: {pos_name}", showlegend=False,
                hovertemplate=f"<b>{pos_name}</b><br>Actual Position<br>({px:.0f}, {py:.0f})<extra></extra>",
            ))

    fig.update_layout(
        xaxis=dict(range=[-300, 300], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-15, 400], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", fixedrange=True),
        height=height, margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#000000", family="Inter, Arial, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5,
                    font=dict(size=10, color="#000000"), bgcolor="rgba(0,0,0,0)"),
        title=dict(text=title, font=dict(size=14)),
    )
    return fig


def _apply_game_filter(df, game_mode):
    """Filter DataFrame by game vs scrimmage. DAV_WIL vs DAV_WIL = scrimmage."""
    if game_mode == "Games Only":
        return df[~((df.get("PitcherTeam", pd.Series()) == DAVIDSON_TEAM_ID) &
                     (df.get("BatterTeam", pd.Series()) == DAVIDSON_TEAM_ID))].copy()
    elif game_mode == "Scrimmages Only":
        return df[((df.get("PitcherTeam", pd.Series()) == DAVIDSON_TEAM_ID) &
                    (df.get("BatterTeam", pd.Series()) == DAVIDSON_TEAM_ID))].copy()
    return df  # "All"


def page_defensive_positioning(data):
    st.markdown('<div class="section-header">Defensive Positioning</div>', unsafe_allow_html=True)
    st.caption("Spray-based defensive alignment optimizer — see where batted balls land and where fielders should stand")

    # Mode selection
    mode = st.radio("Analysis Mode", ["Scout Batter", "Team Tendencies"], horizontal=True, key="dp_mode")

    # ── Use the pre-loaded `data` DataFrame (fast!) instead of scanning parquet ──
    # `data` contains all Davidson-involved games from load_davidson_data()
    if data is None or data.empty:
        st.error("No data loaded.")
        return

    # Season + Game/Scrimmage filters (shared)
    col_s, col_g = st.columns([1, 1])
    with col_s:
        all_seasons = sorted(data["Season"].dropna().unique().tolist()) if "Season" in data.columns else []
        season_filter = st.multiselect("Season", all_seasons, default=all_seasons, key="dp_season")
    with col_g:
        game_mode = st.selectbox("Game Type", ["Games Only", "All", "Scrimmages Only"], key="dp_game_type")

    # Apply season filter
    fdata = data.copy()
    if season_filter and "Season" in fdata.columns:
        fdata = fdata[fdata["Season"].isin(season_filter)]
    fdata = _apply_game_filter(fdata, game_mode)

    if mode == "Scout Batter":
        # Build batter list from pre-loaded data (instant, no parquet scan)
        inplay = fdata[fdata["PitchCall"] == "InPlay"].copy() if "PitchCall" in fdata.columns else pd.DataFrame()
        if inplay.empty:
            st.warning("No batted ball data with current filters.")
            return

        # Deduplicate: get unique (Batter, BatterTeam) combos
        batter_teams = (
            inplay[["Batter", "BatterTeam"]]
            .dropna(subset=["Batter"])
            .drop_duplicates()
            .query("Batter != '' and Batter != ', '")
            .sort_values("Batter")
        )
        teams_available = sorted(batter_teams["BatterTeam"].dropna().unique().tolist())
        teams_list = ["All Teams"] + teams_available

        col_t, col_b = st.columns([1, 2])
        with col_t:
            team_filter = st.selectbox("Filter by Team", teams_list, key="dp_team_filter")
        with col_b:
            if team_filter != "All Teams":
                bt_filtered = batter_teams[batter_teams["BatterTeam"] == team_filter]
            else:
                bt_filtered = batter_teams
            batters = bt_filtered["Batter"].tolist()
            if not batters:
                st.warning("No batters found for this team.")
                return
            batter = st.selectbox("Select Batter", batters, format_func=display_name, key="dp_batter")

        bdf = fdata[fdata["Batter"] == batter]
        label = display_name(batter)
    else:
        teams_available = sorted(fdata["BatterTeam"].dropna().unique().tolist()) if "BatterTeam" in fdata.columns else []
        if not teams_available:
            st.warning("No team data with current filters.")
            return
        team = st.selectbox("Select Team", teams_available, key="dp_team")
        bdf = fdata[fdata["BatterTeam"] == team]
        label = team

    if bdf.empty or "PitchCall" not in bdf.columns:
        st.warning(f"No pitch data found for {label}.")
        return

    batted = bdf[bdf["PitchCall"] == "InPlay"].copy()
    if len(batted) < 10:
        st.warning(f"Not enough batted balls for {label} (need 10+, have {len(batted)}).")
        return

    # Determine batter side
    batter_side = "Right"
    if "BatterSide" in bdf.columns and bdf["BatterSide"].notna().any():
        side_mode = bdf["BatterSide"].mode()
        if len(side_mode) > 0:
            batter_side = side_mode.iloc[0]
    b_str = {"Right": "R", "Left": "L"}.get(batter_side, batter_side)

    st.markdown(
        f'<div style="background:#f8f8f8;padding:10px 16px;border-radius:8px;margin:8px 0;border:1px solid #eee;">'
        f'<span style="font-size:16px;font-weight:800;color:#1a1a2e !important;">{label}</span>'
        f'<span style="margin-left:12px;font-size:13px;color:#666 !important;">Bats: {b_str} | '
        f'{len(batted)} batted balls</span></div>', unsafe_allow_html=True)

    tab_spray, tab_shift, tab_sit, tab_actual, tab_matchup = st.tabs([
        "Spray & Positioning", "Shift Analysis", "Situational", "Actual Positioning",
        "Pitcher-Batter Matchup",
    ])

    # ─── Tab 1: Spray Tendencies & Optimal Positioning ──────
    with tab_spray:
        section_header("Spray Tendencies & Optimal Positioning")

        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            ht_options = ["All"] + sorted(batted["TaggedHitType"].dropna().unique().tolist()) if "TaggedHitType" in batted.columns else ["All"]
            hit_type_filter = st.selectbox("Hit Type", ht_options, key="dp_ht")
        with col_f2:
            pt_options = ["All"] + sorted(batted["TaggedPitchType"].dropna().unique().tolist()) if "TaggedPitchType" in batted.columns else ["All"]
            pitch_type_filter = st.selectbox("Pitch Type Faced", pt_options, key="dp_pt")

        filt = batted.copy()
        if hit_type_filter != "All" and "TaggedHitType" in filt.columns:
            filt = filt[filt["TaggedHitType"] == hit_type_filter]
        if pitch_type_filter != "All" and "TaggedPitchType" in filt.columns:
            filt = filt[filt["TaggedPitchType"] == pitch_type_filter]

        if len(filt) < 5:
            st.info("Not enough batted balls with current filters.")
        else:
            # Compute recommended positions
            recommended = _recommend_fielder_positions(filt, batter_side)

            col_field, col_stats = st.columns([3, 2])
            with col_field:
                fig_field = _draw_defensive_field(filt, recommended=recommended, height=520,
                                                  title="Spray Chart + Recommended Fielder Positions")
                st.plotly_chart(fig_field, use_container_width=True)
                st.caption("★ Stars = recommended fielder positions (weighted by hit frequency × damage)")

            with col_stats:
                # Zone breakdown table
                section_header("Zone Breakdown")
                zones = _compute_spray_zones(filt, batter_side)
                zone_rows = []
                for zname in ["IF-Pull", "IF-Center", "IF-Oppo", "OF-Pull", "OF-Center", "OF-Oppo"]:
                    z = zones.get(zname, {})
                    if not z:
                        continue
                    zone_rows.append({
                        "Zone": zname,
                        "Count": z["count"],
                        "Hit%": f"{z['count']/len(filt)*100:.1f}%" if len(filt) > 0 else "-",
                        "Out%": f"{z['out_rate']:.1f}%",
                        "H Rate": f"{z['hit_rate']:.1f}%",
                        "Avg EV": f"{z['avg_ev']:.1f}" if not pd.isna(z.get("avg_ev", np.nan)) else "-",
                        "Avg Dist": f"{z['avg_dist']:.0f} ft",
                    })
                if zone_rows:
                    st.dataframe(pd.DataFrame(zone_rows).set_index("Zone"), use_container_width=True)

                # Pull/Center/Oppo summary
                section_header("Directional Summary")
                spray_df = filt.dropna(subset=["Direction"]).copy()
                if not spray_df.empty:
                    if batter_side == "Right":
                        spray_df["Dir"] = np.where(spray_df["Direction"] < -15, "Pull",
                                          np.where(spray_df["Direction"] > 15, "Oppo", "Center"))
                    else:
                        spray_df["Dir"] = np.where(spray_df["Direction"] > 15, "Pull",
                                          np.where(spray_df["Direction"] < -15, "Oppo", "Center"))

                    dir_rows = []
                    for d in ["Pull", "Center", "Oppo"]:
                        dd = spray_df[spray_df["Dir"] == d]
                        if dd.empty:
                            continue
                        gb_pct = len(dd[dd["TaggedHitType"] == "GroundBall"]) / max(len(dd), 1) * 100 if "TaggedHitType" in dd.columns else np.nan
                        ld_pct = len(dd[dd["TaggedHitType"] == "LineDrive"]) / max(len(dd), 1) * 100 if "TaggedHitType" in dd.columns else np.nan
                        fb_pct = len(dd[dd["TaggedHitType"] == "FlyBall"]) / max(len(dd), 1) * 100 if "TaggedHitType" in dd.columns else np.nan
                        dir_rows.append({
                            "Direction": d,
                            "BBE": len(dd),
                            "%": f"{len(dd)/len(spray_df)*100:.1f}%",
                            "Avg EV": f"{dd['ExitSpeed'].mean():.1f}" if dd["ExitSpeed"].notna().any() else "-",
                            "GB%": f"{gb_pct:.0f}%" if not pd.isna(gb_pct) else "-",
                            "LD%": f"{ld_pct:.0f}%" if not pd.isna(ld_pct) else "-",
                            "FB%": f"{fb_pct:.0f}%" if not pd.isna(fb_pct) else "-",
                        })
                    if dir_rows:
                        st.dataframe(pd.DataFrame(dir_rows).set_index("Direction"), use_container_width=True)

    # ─── Tab 2: Shift Analysis ──────────────────────
    with tab_shift:
        section_header("Shift Analysis & Recommendations")

        spray_all = batted.dropna(subset=["Direction"]).copy()
        if len(spray_all) < 10:
            st.info("Not enough batted balls for shift analysis.")
        else:
            if batter_side == "Right":
                spray_all["Dir"] = np.where(spray_all["Direction"] < -15, "Pull",
                                   np.where(spray_all["Direction"] > 15, "Oppo", "Center"))
            else:
                spray_all["Dir"] = np.where(spray_all["Direction"] > 15, "Pull",
                                   np.where(spray_all["Direction"] < -15, "Oppo", "Center"))

            total = len(spray_all)
            pull_pct = len(spray_all[spray_all["Dir"] == "Pull"]) / total * 100
            center_pct = len(spray_all[spray_all["Dir"] == "Center"]) / total * 100
            oppo_pct = len(spray_all[spray_all["Dir"] == "Oppo"]) / total * 100
            gb_pct = len(spray_all[spray_all["TaggedHitType"] == "GroundBall"]) / total * 100 if "TaggedHitType" in spray_all.columns else 0
            gb_pull = spray_all[(spray_all["TaggedHitType"] == "GroundBall") & (spray_all["Dir"] == "Pull")] if "TaggedHitType" in spray_all.columns else pd.DataFrame()
            n_gb = len(spray_all[spray_all["TaggedHitType"] == "GroundBall"]) if "TaggedHitType" in spray_all.columns else 0
            gb_pull_pct = len(gb_pull) / n_gb * 100 if n_gb > 0 else 0

            # Shift recommendation
            if pull_pct > 45 and gb_pct > 45:
                shift_rec = "Infield Shift"
                shift_color = "#d22d49"
                shift_desc = (f"Pull-heavy hitter ({pull_pct:.0f}% pull) with high GB rate ({gb_pct:.0f}%). "
                              f"Ground balls go pull-side {gb_pull_pct:.0f}% of the time. "
                              f"Shift infield toward pull side.")
            elif pull_pct > 40:
                shift_rec = "Shade Pull"
                shift_color = "#fe6100"
                shift_desc = (f"Moderate pull tendency ({pull_pct:.0f}%). "
                              f"Shade middle infielders slightly toward pull side, don't full shift.")
            elif oppo_pct > 40:
                shift_rec = "Shade Oppo"
                shift_color = "#1f77b4"
                shift_desc = (f"Oppo-oriented hitter ({oppo_pct:.0f}% opposite field). "
                              f"Shade defense toward opposite field.")
            else:
                shift_rec = "Standard"
                shift_color = "#2ca02c"
                shift_desc = (f"Balanced spray (Pull: {pull_pct:.0f}%, Center: {center_pct:.0f}%, Oppo: {oppo_pct:.0f}%). "
                              f"Use standard defensive alignment.")

            st.markdown(
                f'<div style="padding:16px;background:white;border-radius:10px;border-left:6px solid {shift_color};'
                f'border:1px solid #eee;margin:8px 0;">'
                f'<div style="font-size:20px;font-weight:900;color:{shift_color} !important;">{shift_rec}</div>'
                f'<div style="font-size:13px;color:#333 !important;margin-top:4px;">{shift_desc}</div>'
                f'</div>', unsafe_allow_html=True)

            # Spray distribution cards
            col_p, col_c, col_o = st.columns(3)
            for col, dir_name, pct in [(col_p, "Pull", pull_pct), (col_c, "Center", center_pct), (col_o, "Oppo", oppo_pct)]:
                with col:
                    dir_df = spray_all[spray_all["Dir"] == dir_name]
                    hits = dir_df[dir_df["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"])] if "PlayResult" in dir_df.columns else pd.DataFrame()
                    outs = dir_df[dir_df["PlayResult"].isin(["Out", "Sacrifice", "FieldersChoice"])] if "PlayResult" in dir_df.columns else pd.DataFrame()
                    ev_str = f"{dir_df['ExitSpeed'].mean():.1f}" if dir_df["ExitSpeed"].notna().any() else "-"
                    st.metric(dir_name, f"{pct:.1f}%",
                              delta=f"EV: {ev_str} | Hits: {len(hits)} | Outs: {len(outs)}")

            # Hit outcome by direction
            section_header("Hit Outcome by Field Third")
            outcome_rows = []
            for d in ["Pull", "Center", "Oppo"]:
                dd = spray_all[spray_all["Dir"] == d]
                if dd.empty or "PlayResult" not in dd.columns:
                    continue
                n = len(dd)
                outs = dd["PlayResult"].isin(["Out", "Sacrifice", "FieldersChoice"]).sum()
                singles = (dd["PlayResult"] == "Single").sum()
                xbh = dd["PlayResult"].isin(["Double", "Triple", "HomeRun"]).sum()
                outcome_rows.append({
                    "Direction": d,
                    "BBE": n,
                    "Out%": f"{outs/n*100:.1f}%",
                    "Single%": f"{singles/n*100:.1f}%",
                    "XBH%": f"{xbh/n*100:.1f}%",
                    "Avg EV": f"{dd['ExitSpeed'].mean():.1f}" if dd["ExitSpeed"].notna().any() else "-",
                    "Avg Dist": f"{dd['Distance'].mean():.0f}ft" if dd["Distance"].notna().any() else "-",
                })
            if outcome_rows:
                st.dataframe(pd.DataFrame(outcome_rows).set_index("Direction"), use_container_width=True)

            # GB vs FB directional tendencies
            if "TaggedHitType" in spray_all.columns:
                section_header("Ground Ball vs Fly Ball Directional Tendencies")
                st.caption("Ground balls are typically more pull-heavy — the shift exploits this")
                ht_dir_rows = []
                for ht in ["GroundBall", "LineDrive", "FlyBall"]:
                    ht_df = spray_all[spray_all["TaggedHitType"] == ht]
                    if len(ht_df) < 3:
                        continue
                    ht_n = len(ht_df)
                    ht_dir_rows.append({
                        "Hit Type": {"GroundBall": "Ground Ball", "LineDrive": "Line Drive", "FlyBall": "Fly Ball"}.get(ht, ht),
                        "Count": ht_n,
                        "Pull%": f"{len(ht_df[ht_df['Dir']=='Pull'])/ht_n*100:.1f}%",
                        "Center%": f"{len(ht_df[ht_df['Dir']=='Center'])/ht_n*100:.1f}%",
                        "Oppo%": f"{len(ht_df[ht_df['Dir']=='Oppo'])/ht_n*100:.1f}%",
                        "Avg EV": f"{ht_df['ExitSpeed'].mean():.1f}" if ht_df["ExitSpeed"].notna().any() else "-",
                    })
                if ht_dir_rows:
                    st.dataframe(pd.DataFrame(ht_dir_rows).set_index("Hit Type"), use_container_width=True)

                # Stacked bar chart: direction by hit type
                fig_stack = go.Figure()
                for d, clr in [("Pull", "#d22d49"), ("Center", "#f7c631"), ("Oppo", "#1f77b4")]:
                    vals = []
                    cats = []
                    for ht in ["GroundBall", "LineDrive", "FlyBall"]:
                        ht_df = spray_all[spray_all["TaggedHitType"] == ht]
                        if len(ht_df) < 3:
                            continue
                        vals.append(len(ht_df[ht_df["Dir"] == d]) / len(ht_df) * 100)
                        cats.append({"GroundBall": "GB", "LineDrive": "LD", "FlyBall": "FB"}.get(ht, ht))
                    fig_stack.add_trace(go.Bar(x=cats, y=vals, name=d, marker_color=clr))
                fig_stack.update_layout(**CHART_LAYOUT, height=300, barmode="group",
                                        yaxis_title="% of Hit Type", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_stack, use_container_width=True)

    # ─── Tab 3: Situational Positioning ──────────────
    with tab_sit:
        section_header("Situational Positioning Adjustments")
        st.caption("How spray tendencies change by count, pitch type, and situation")

        spray_sit = batted.dropna(subset=["Direction"]).copy()
        if len(spray_sit) < 15:
            st.info("Not enough data for situational analysis.")
        else:
            if batter_side == "Right":
                spray_sit["Dir"] = np.where(spray_sit["Direction"] < -15, "Pull",
                                   np.where(spray_sit["Direction"] > 15, "Oppo", "Center"))
            else:
                spray_sit["Dir"] = np.where(spray_sit["Direction"] > 15, "Pull",
                                   np.where(spray_sit["Direction"] < -15, "Oppo", "Center"))

            # By count situation
            section_header("Spray by Count Situation")
            if "Balls" in spray_sit.columns and "Strikes" in spray_sit.columns:
                spray_sit_c = spray_sit.dropna(subset=["Balls", "Strikes"]).copy()
                spray_sit_c["Balls"] = spray_sit_c["Balls"].astype(int)
                spray_sit_c["Strikes"] = spray_sit_c["Strikes"].astype(int)
                spray_sit_c["Situation"] = "Even"
                spray_sit_c.loc[spray_sit_c["Balls"] < spray_sit_c["Strikes"], "Situation"] = "Pitcher Ahead"
                spray_sit_c.loc[spray_sit_c["Balls"] > spray_sit_c["Strikes"], "Situation"] = "Hitter Ahead"

                sit_rows = []
                for sit in ["Pitcher Ahead", "Even", "Hitter Ahead"]:
                    sdf = spray_sit_c[spray_sit_c["Situation"] == sit]
                    if len(sdf) < 5:
                        continue
                    n = len(sdf)
                    sit_rows.append({
                        "Situation": sit,
                        "BBE": n,
                        "Pull%": f"{len(sdf[sdf['Dir']=='Pull'])/n*100:.1f}%",
                        "Center%": f"{len(sdf[sdf['Dir']=='Center'])/n*100:.1f}%",
                        "Oppo%": f"{len(sdf[sdf['Dir']=='Oppo'])/n*100:.1f}%",
                        "Avg EV": f"{sdf['ExitSpeed'].mean():.1f}" if sdf["ExitSpeed"].notna().any() else "-",
                        "GB%": f"{len(sdf[sdf['TaggedHitType']=='GroundBall'])/n*100:.1f}%" if "TaggedHitType" in sdf.columns else "-",
                    })
                if sit_rows:
                    st.dataframe(pd.DataFrame(sit_rows).set_index("Situation"), use_container_width=True)

            # 2-strike approach
            section_header("2-Strike Approach")
            if "Strikes" in spray_sit.columns:
                pre2k = spray_sit[spray_sit["Strikes"].fillna(0).astype(int) < 2]
                with2k = spray_sit[spray_sit["Strikes"].fillna(0).astype(int) == 2]
                two_k_rows = []
                for label_2k, df_2k in [("< 2 Strikes", pre2k), ("2 Strikes", with2k)]:
                    if len(df_2k) < 5:
                        continue
                    n = len(df_2k)
                    two_k_rows.append({
                        "Count": label_2k,
                        "BBE": n,
                        "Pull%": f"{len(df_2k[df_2k['Dir']=='Pull'])/n*100:.1f}%",
                        "Center%": f"{len(df_2k[df_2k['Dir']=='Center'])/n*100:.1f}%",
                        "Oppo%": f"{len(df_2k[df_2k['Dir']=='Oppo'])/n*100:.1f}%",
                        "Avg EV": f"{df_2k['ExitSpeed'].mean():.1f}" if df_2k["ExitSpeed"].notna().any() else "-",
                        "GB%": f"{len(df_2k[df_2k['TaggedHitType']=='GroundBall'])/n*100:.1f}%" if "TaggedHitType" in df_2k.columns else "-",
                    })
                if two_k_rows:
                    st.dataframe(pd.DataFrame(two_k_rows).set_index("Count"), use_container_width=True)
                    # Insight
                    if len(pre2k) >= 5 and len(with2k) >= 5:
                        pre_pull = len(pre2k[pre2k["Dir"] == "Pull"]) / len(pre2k) * 100
                        with_pull = len(with2k[with2k["Dir"] == "Pull"]) / len(with2k) * 100
                        diff = with_pull - pre_pull
                        if abs(diff) > 5:
                            direction = "more oppo" if diff < 0 else "more pull-heavy"
                            st.info(f"With 2 strikes, this batter goes **{direction}** ({diff:+.1f}% pull change). "
                                    f"Adjust positioning accordingly.")

            # By pitch type faced
            section_header("Spray by Pitch Type Faced")
            if "TaggedPitchType" in spray_sit.columns:
                pt_types = spray_sit["TaggedPitchType"].value_counts()
                pt_types = pt_types[pt_types >= 5].index.tolist()
                if pt_types:
                    n_cols = min(len(pt_types), 3)
                    pt_cols = st.columns(n_cols)
                    for idx, pt in enumerate(pt_types[:6]):
                        pt_df = spray_sit[spray_sit["TaggedPitchType"] == pt]
                        with pt_cols[idx % n_cols]:
                            fig_pt = _draw_defensive_field(pt_df, height=300, title=pt, color_by="EV")
                            st.plotly_chart(fig_pt, use_container_width=True)
                            n = len(pt_df)
                            pull_p = len(pt_df[pt_df["Dir"] == "Pull"]) / max(n, 1) * 100
                            st.caption(f"n={n} | Pull: {pull_p:.0f}% | "
                                       f"EV: {pt_df['ExitSpeed'].mean():.1f}" if pt_df["ExitSpeed"].notna().any() else f"n={n} | Pull: {pull_p:.0f}%")

    # ─── Tab 4: Actual Positioning Data ──────────────
    with tab_actual:
        section_header("Actual Fielder Positioning (Camera Data)")
        st.caption("When field-home-camera positioning data is available, see where fielders were actually standing")

        # Try to load positioning files
        game_ids = bdf["GameID"].dropna().unique().tolist() if "GameID" in bdf.columns else []
        pos_data = _load_positioning_csvs(game_ids if game_ids else None)

        if pos_data.empty:
            st.info("No positioning CSV data found for the selected games. "
                    "Positioning data requires *_playerpositioning_FHC.csv files matching the game IDs.")
            st.markdown(
                '<div style="padding:12px;background:#f8f8f8;border-radius:8px;border:1px solid #eee;margin-top:8px;">'
                '<div style="font-size:13px;font-weight:700;color:#1a1a2e !important;">How Positioning Data Works</div>'
                '<div style="font-size:12px;color:#555 !important;">Trackman\'s field-home camera captures all 7 fielder '
                'positions (1B, 2B, 3B, SS, LF, CF, RF) at pitch release. When CSV files are available, this tab shows '
                'actual fielder positions overlaid on the batter\'s spray chart, along with shift detection and gap analysis.</div>'
                '</div>', unsafe_allow_html=True)
        else:
            st.success(f"Loaded positioning data: {len(pos_data)} pitches from {pos_data['_GameID'].nunique()} game(s)")

            # FHC coordinate system (verified from data):
            #   X = depth from home plate (feet): 1B~89, SS~141, CF~315
            #   Z = lateral position (feet): negative=3B/LF side, positive=1B/RF side
            # Spray chart coordinate system:
            #   x = lateral (negative=left/3B, positive=right/1B)
            #   y = depth from home plate
            # Mapping: FHC_Z → spray_x, FHC_X → spray_y
            pos_cols = ["1B", "2B", "3B", "SS", "LF", "CF", "RF"]
            has_pos = all(f"{p}_PositionAtReleaseX" in pos_data.columns for p in pos_cols)

            if has_pos:
                if pos_data[f"1B_PositionAtReleaseX"].notna().any():
                    # Compute average positions across all pitches (shows general alignment)
                    avg_positions = {}
                    for p in pos_cols:
                        x_col = f"{p}_PositionAtReleaseX"
                        z_col = f"{p}_PositionAtReleaseZ"
                        if pos_data[x_col].notna().any():
                            avg_x = pos_data[z_col].mean()   # FHC Z → spray x (lateral)
                            avg_y = pos_data[x_col].mean()   # FHC X → spray y (depth)
                            avg_positions[p] = (avg_x, avg_y)

                    # Draw field with actual positions + spray data + recommended
                    recommended = _recommend_fielder_positions(batted, batter_side)
                    fig_actual = _draw_defensive_field(batted, recommended=recommended,
                                                       actual_positions=avg_positions,
                                                       height=550, title="Actual vs Recommended Positioning")
                    st.plotly_chart(fig_actual, use_container_width=True)
                    st.caption("★ Stars = recommended positions | ◆ Diamonds = actual average positions")

                    # Gap analysis table
                    if avg_positions and recommended:
                        section_header("Position Gap Analysis")
                        gap_rows = []
                        for p in pos_cols:
                            if p in avg_positions and p in recommended:
                                ax, ay = avg_positions[p]
                                rx, ry = recommended[p]
                                gap = np.sqrt((ax - rx)**2 + (ay - ry)**2)
                                gap_rows.append({
                                    "Position": p,
                                    "Actual (x, y)": f"({ax:.0f}, {ay:.0f})",
                                    "Recommended (x, y)": f"({rx:.0f}, {ry:.0f})",
                                    "Gap (ft)": f"{gap:.1f}",
                                    "Direction": "Shift pull-ward" if (rx - ax) * (-1 if batter_side == "Right" else 1) > 5
                                                 else "Shift oppo-ward" if (rx - ax) * (-1 if batter_side == "Right" else 1) < -5
                                                 else "Well-positioned",
                                })
                        if gap_rows:
                            st.dataframe(pd.DataFrame(gap_rows).set_index("Position"), use_container_width=True)

                # Shift detection summary
                if "DetectedShift" in pos_data.columns:
                    section_header("Detected Shift Usage")
                    shift_counts = pos_data["DetectedShift"].value_counts()
                    shift_rows = []
                    for shift_type, count in shift_counts.items():
                        shift_rows.append({
                            "Shift Type": shift_type,
                            "Pitches": count,
                            "Usage%": f"{count/len(pos_data)*100:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(shift_rows).set_index("Shift Type"), use_container_width=True)
            else:
                st.warning("Positioning coordinate columns not found in the CSV data.")

    # ─── Tab 5: Pitcher-Batter Matchup ──────────────
    with tab_matchup:
        section_header("Pitcher-Batter Matchup Positioning")
        st.caption("Movement-adjusted, shrinkage-blended positioning — factors in the pitcher's pitch mix and movement profile")

        if mode != "Scout Batter":
            st.info("Pitcher-Batter Matchup is only available in Scout Batter mode.")
        else:
            # Pitcher selector — use pre-loaded data (fast)
            pitchers = sorted(
                data["Pitcher"].dropna()
                .loc[lambda s: (s != '') & (s != ', ')]
                .unique().tolist()
            ) if "Pitcher" in data.columns else []
            if not pitchers:
                st.warning("No pitcher data found in database.")
            else:
                col_p1, col_p2 = st.columns([2, 1])
                with col_p1:
                    sel_pitcher = st.selectbox("Select Pitcher", pitchers, format_func=display_name, key="dp_matchup_pitcher")
                with col_p2:
                    pitcher_seasons = sorted(data["Season"].dropna().unique().tolist()) if "Season" in data.columns else []
                    pitcher_season_filter = st.multiselect(
                        "Pitcher Season", pitcher_seasons, default=pitcher_seasons, key="dp_matchup_p_season"
                    )

                pitcher_sf = tuple(pitcher_season_filter) if pitcher_season_filter else None
                batter_sf = tuple(season_filter) if season_filter else None

                matchup = compute_pitcher_batter_positioning(
                    pitcher=sel_pitcher,
                    batter=batter,
                    batter_side=batter_side,
                    pitcher_season_filter=pitcher_sf,
                    batter_season_filter=batter_sf,
                )

                # Warnings
                for w in matchup.warnings:
                    st.warning(w)

                # Shift recommendation banner
                shift = matchup.shift_type
                shift_color = shift.get("color", "#2ca02c")
                conf_colors = {"High": "#2ca02c", "Medium": "#fe6100", "Low": "#d22d49"}
                conf_color = conf_colors.get(matchup.confidence, "#888")
                st.markdown(
                    f'<div style="padding:16px;background:white;border-radius:10px;border-left:6px solid {shift_color};'
                    f'border:1px solid #eee;margin:8px 0;">'
                    f'<div style="font-size:20px;font-weight:900;color:{shift_color} !important;">{shift.get("type", "Standard")}</div>'
                    f'<div style="font-size:13px;color:#333 !important;margin-top:4px;">{shift.get("desc", "")}</div>'
                    f'<div style="font-size:12px;color:#888 !important;margin-top:6px;">'
                    f'GB Pull: {matchup.overall_pull_pct:.1f}% | Center: {matchup.overall_center_pct:.1f}% | Oppo: {matchup.overall_oppo_pct:.1f}%'
                    f' | GB%: <b style="color:#333 !important;">{matchup.gb_pct:.1f}%</b>'
                    f' | Confidence: <b style="color:{conf_color} !important;">{matchup.confidence}</b></div>'
                    f'</div>', unsafe_allow_html=True)

                if matchup.positions:
                    # Two-column layout: field + position descriptions
                    col_field_m, col_desc_m = st.columns([3, 2])
                    with col_field_m:
                        # Build recommended positions dict for _draw_defensive_field
                        rec_positions = {
                            p: (pos.x, pos.y) for p, pos in matchup.positions.items()
                        }
                        # Show all batted balls (GB + air) for full-field context
                        fig_matchup = _draw_defensive_field(
                            batted, recommended=rec_positions, height=520,
                            title=f"Matchup: {display_name(batter)} vs {display_name(sel_pitcher)}"
                        )
                        st.plotly_chart(fig_matchup, use_container_width=True)
                        st.caption("All batted balls shown. Stars = matchup-adjusted positions (7 fielders).")

                    with col_desc_m:
                        section_header("Infield Position Instructions")
                        pos_rows = []
                        for pos_name in ["3B", "SS", "2B", "1B"]:
                            fp = matchup.positions.get(pos_name)
                            if not fp:
                                continue
                            pos_rows.append({
                                "Position": pos_name,
                                "Instruction": fp.description,
                                "Depth": f"{fp.depth_ft:.0f} ft",
                                "From 2B Bag": f"{fp.lateral_from_2b_ft:+.0f} ft",
                                "Confidence": fp.confidence,
                            })
                        if pos_rows:
                            st.dataframe(pd.DataFrame(pos_rows).set_index("Position"), use_container_width=True)

                        # Outfield instructions
                        of_rows = []
                        for pos_name in ["LF", "CF", "RF"]:
                            fp = matchup.positions.get(pos_name)
                            if not fp:
                                continue
                            of_rows.append({
                                "Position": pos_name,
                                "Instruction": fp.description,
                                "Depth": f"{fp.depth_ft:.0f} ft",
                                "Confidence": fp.confidence,
                            })
                        if of_rows:
                            section_header("Outfield Instructions")
                            st.dataframe(pd.DataFrame(of_rows).set_index("Position"), use_container_width=True)

                # Pitch mix contribution table
                section_header("Pitch Mix Contribution")
                mix_rows = []
                for pt, summary in sorted(matchup.per_pitch_type.items(), key=lambda x: -(x[1].get("mix_pct") or 0)):
                    mv_shift = summary.get("movement_shift_deg", 0)
                    mv_str = f"{mv_shift:+.1f}\u00b0" if abs(mv_shift) > 0.05 else "-"
                    mix_rows.append({
                        "Pitch Type": pt,
                        "Mix%": f"{summary.get('mix_pct', 0):.1f}%",
                        "Batter GBs": summary.get("n_batter", 0),
                        "Pull%": f"{summary.get('pull_pct', 0):.1f}%" if summary.get("pull_pct") is not None else "-",
                        "Ctr%": f"{summary.get('center_pct', 0):.1f}%" if summary.get("center_pct") is not None else "-",
                        "Opp%": f"{summary.get('oppo_pct', 0):.1f}%" if summary.get("oppo_pct") is not None else "-",
                        "Mvmt Shift": mv_str,
                        "Source": summary.get("source", "-"),
                    })
                if mix_rows:
                    st.dataframe(pd.DataFrame(mix_rows).set_index("Pitch Type"), use_container_width=True)

                # Expander: Pitcher Movement Profile
                with st.expander("Pitcher Movement Profile"):
                    from analytics.gb_positioning import get_pitcher_movement_profile
                    p_profile = get_pitcher_movement_profile(sel_pitcher, pitcher_sf)
                    if p_profile:
                        prof_rows = []
                        for pt, prof in sorted(p_profile.items(), key=lambda x: -x[1].get("usage_pct", 0)):
                            prof_rows.append({
                                "Pitch": pt,
                                "Velo": f"{prof['velo']:.1f}" if prof.get("velo") else "-",
                                "IVB": f"{prof['ivb']:.1f}" if prof.get("ivb") else "-",
                                "HB": f"{prof['hb']:.1f}" if prof.get("hb") else "-",
                                "Spin": f"{prof['spin']:.0f}" if prof.get("spin") else "-",
                                "Ext": f"{prof['ext']:.1f}" if prof.get("ext") else "-",
                                "Usage%": f"{prof['usage_pct']:.1f}%",
                            })
                        st.dataframe(pd.DataFrame(prof_rows).set_index("Pitch"), use_container_width=True)
                    else:
                        st.info("No pitch data found for this pitcher.")

                # Expander: Shrinkage Details & Confidence
                with st.expander("Confidence & Shrinkage Details"):
                    st.caption("Shows how much the model trusts the batter's data vs population for each pitch type")

                    # Visual confidence gauge
                    total_gb = sum(s.get("n_batter", 0) for s in matchup.per_pitch_type.values())
                    has_movement = any(abs(s.get("movement_shift_deg", 0)) > 0.05 for s in matchup.per_pitch_type.values())

                    gauge_items = [
                        ("Sample Size", total_gb, 40, "ground balls"),
                        ("Movement Adj.", 1 if has_movement else 0, 1, "active" if has_movement else "inactive"),
                    ]
                    gauge_cols = st.columns(len(gauge_items))
                    for gc, (label, val, target, unit) in zip(gauge_cols, gauge_items):
                        with gc:
                            pct = min(val / target * 100, 100) if target > 0 else 0
                            bar_color = "#2ca02c" if pct >= 75 else "#fe6100" if pct >= 40 else "#d22d49"
                            st.markdown(
                                f'<div style="text-align:center;">'
                                f'<div style="font-size:11px;color:#888 !important;">{label}</div>'
                                f'<div style="background:#eee;border-radius:4px;height:8px;margin:4px 0;">'
                                f'<div style="background:{bar_color};height:8px;border-radius:4px;width:{pct:.0f}%;"></div></div>'
                                f'<div style="font-size:12px;font-weight:700;color:#333 !important;">{val} {unit}</div>'
                                f'</div>', unsafe_allow_html=True)

                    st.markdown("")

                    shrink_rows = []
                    for pt, summary in matchup.per_pitch_type.items():
                        bw = summary.get("batter_weight", 0)
                        pw = summary.get("pop_weight", 1)
                        mv = summary.get("movement_shift_deg", 0)
                        shrink_rows.append({
                            "Pitch Type": pt,
                            "Batter GBs": summary.get("n_batter", 0),
                            "Batter Weight": f"{bw * 100:.0f}%",
                            "Pop Weight": f"{pw * 100:.0f}%",
                            "Mvmt Shift": f"{mv:+.1f}\u00b0" if abs(mv) > 0.05 else "-",
                        })
                    if shrink_rows:
                        st.dataframe(pd.DataFrame(shrink_rows).set_index("Pitch Type"), use_container_width=True)

                    # Interpretation
                    if total_gb < 10:
                        st.info("Very few batter ground balls — positions heavily rely on population averages. "
                                "Increase confidence by adding more game data.")
                    elif total_gb < 30:
                        st.info("Moderate sample size — positions blend batter data with population baselines. "
                                "Recommendations will stabilize with more data.")
                    else:
                        st.success(f"Good sample size ({total_gb} GBs) — positions primarily reflect this batter's actual tendencies.")
