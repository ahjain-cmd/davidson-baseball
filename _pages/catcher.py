"""Catcher Analytics page."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import (
    DAVIDSON_TEAM_ID, ROSTER_2026, JERSEY, POSITION, PITCH_COLORS,
    SWING_CALLS, CONTACT_CALLS,
    in_zone_mask, display_name, get_percentile,
)
from data.loader import get_all_seasons
from viz.layout import CHART_LAYOUT, section_header
from viz.charts import player_header, make_pitch_location_heatmap
from viz.percentiles import render_savant_percentile_section


def page_catcher(data):
    st.header("Catcher Analytics")
    catching = data[
        (data["CatcherTeam"] == DAVIDSON_TEAM_ID) & (data["Catcher"].isin(ROSTER_2026))
    ] if "Catcher" in data.columns and "CatcherTeam" in data.columns else pd.DataFrame()

    if catching.empty:
        st.warning("No catcher data found. Ensure the data has Catcher/CatcherTeam columns.")
        return

    catchers = sorted(catching["Catcher"].unique())
    c1, c2 = st.columns([2, 3])
    with c1:
        selected = st.selectbox("Select Catcher", catchers, key="cat_c")
    with c2:
        all_seasons = get_all_seasons()
        sel_seasons = st.multiselect("Season", all_seasons, default=all_seasons, key="cat_s")

    cdf = catching[(catching["Catcher"] == selected) & (catching["Season"].isin(sel_seasons))]
    if cdf.empty:
        st.info("No data for this catcher in selected seasons.")
        return

    jersey = JERSEY.get(selected, "")
    pos = POSITION.get(selected, "C")
    player_header(selected, jersey, pos,
                  f"{pos}  |  Davidson Wildcats",
                  f"{len(cdf)} pitches received  |  "
                  f"Seasons: {', '.join(str(int(s)) for s in sorted(sel_seasons))}")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        # Pop Time
        section_header("Pop Time & Exchange")
        pop = cdf.dropna(subset=["PopTime"])
        exc = cdf.dropna(subset=["ExchangeTime"]) if "ExchangeTime" in cdf.columns else pd.DataFrame()

        if len(pop) > 0:
            # Get all catchers' avg pop time for context
            all_catchers_pop = data.dropna(subset=["PopTime"]) if "PopTime" in data.columns else pd.DataFrame()
            catcher_avgs = all_catchers_pop.groupby("Catcher")["PopTime"].mean() if not all_catchers_pop.empty else pd.Series()
            catcher_avgs = catcher_avgs[all_catchers_pop.groupby("Catcher").size() >= 5] if not all_catchers_pop.empty else pd.Series()

            p_pop = pop["PopTime"].mean()
            pop_pct = get_percentile(p_pop, catcher_avgs) if len(catcher_avgs) > 0 else np.nan
            pop_metrics = [("Avg Pop Time", p_pop, pop_pct, ".2f", False)]

            if len(exc) > 0:
                all_catchers_exc = data.dropna(subset=["ExchangeTime"]) if "ExchangeTime" in data.columns else pd.DataFrame()
                catcher_exc_avgs = all_catchers_exc.groupby("Catcher")["ExchangeTime"].mean() if not all_catchers_exc.empty else pd.Series()
                catcher_exc_avgs = catcher_exc_avgs[all_catchers_exc.groupby("Catcher").size() >= 5] if not all_catchers_exc.empty else pd.Series()
                p_exc = exc["ExchangeTime"].mean()
                exc_pct = get_percentile(p_exc, catcher_exc_avgs) if len(catcher_exc_avgs) > 0 else np.nan
                pop_metrics.append(("Avg Exchange", p_exc, exc_pct, ".2f", False))

            render_savant_percentile_section(pop_metrics, None)
            st.caption(f"Based on {len(pop)} recorded throws. Lower is better.")

            # Pop time distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=pop["PopTime"], nbinsx=20,
                marker_color="#cc0000", opacity=0.7,
            ))
            fig.add_vline(x=p_pop, line_dash="solid", line_color="#333",
                          annotation_text=f"Avg: {p_pop:.2f}s",
                          annotation_position="top right")
            if len(catcher_avgs) > 0:
                db_avg = catcher_avgs.mean()
                fig.add_vline(x=db_avg, line_dash="dash", line_color="#999",
                              annotation_text=f"DB Avg: {db_avg:.2f}s",
                              annotation_position="top left")
            fig.update_layout(
                xaxis_title="Pop Time (s)", yaxis_title="Count",
                height=300, **CHART_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, key="cat_pop_dist")
        else:
            st.info("No pop time data recorded.")

    with col2:
        # Framing analysis
        section_header("Receiving / Framing")
        # Called strikes on pitches outside the zone
        loc_data = cdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if len(loc_data) > 0:
            _iz = in_zone_mask(loc_data)
            out_zone_pitches = loc_data[~_iz]
            called_strikes_out = out_zone_pitches[out_zone_pitches["PitchCall"] == "StrikeCalled"]
            frame_rate = len(called_strikes_out) / max(len(out_zone_pitches), 1) * 100

            # Context: all catchers' framing rate
            all_loc = data.dropna(subset=["PlateLocSide", "PlateLocHeight"])
            if "Catcher" in all_loc.columns:
                all_in_zone = in_zone_mask(all_loc)
                all_out = all_loc[~all_in_zone]
                catcher_frame_rates = []
                for c, grp in all_out.groupby("Catcher"):
                    if len(grp) < 50:
                        continue
                    cs = grp[grp["PitchCall"] == "StrikeCalled"]
                    catcher_frame_rates.append(len(cs) / max(len(grp), 1) * 100)
                frame_pct = get_percentile(frame_rate, pd.Series(catcher_frame_rates)) if catcher_frame_rates else np.nan
            else:
                frame_pct = np.nan

            render_savant_percentile_section(
                [("Frame Rate", frame_rate, frame_pct, ".1f", True)], None,
            )
            st.caption(f"Called strikes on {len(out_zone_pitches)} out-of-zone pitches")

            # Framing heatmap — where does this catcher get extra strikes?
            if len(called_strikes_out) >= 3:
                fig = make_pitch_location_heatmap(called_strikes_out, "Framed Strikes", "#1a7a1a", height=350)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="cat_frame_hm")
        else:
            st.info("No location data for framing analysis.")

    # Catcher receiving by pitcher
    st.markdown("---")
    section_header("Performance by Pitcher Caught")
    pitcher_rows = []
    for pitcher, grp in cdf.groupby("Pitcher"):
        if len(grp) < 20:
            continue
        sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
        wh = grp[grp["PitchCall"] == "StrikeSwinging"]
        loc = grp.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        in_z = in_zone_mask(loc)
        out_z = loc[~in_z]
        cs_out = out_z[out_z["PitchCall"] == "StrikeCalled"]
        pitcher_rows.append({
            "Pitcher": pitcher,
            "Pitches": len(grp),
            "Whiff%": round(len(wh) / max(len(sw), 1) * 100, 1),
            "Frame%": round(len(cs_out) / max(len(out_z), 1) * 100, 1) if len(out_z) > 0 else None,
        })
    if pitcher_rows:
        st.dataframe(pd.DataFrame(pitcher_rows).sort_values("Pitches", ascending=False),
                      use_container_width=True, hide_index=True)
