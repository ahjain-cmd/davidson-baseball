"""Player Development page."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config import (
    DAVIDSON_TEAM_ID, ROSTER_2026, PITCH_COLORS,
    filter_davidson, filter_minor_pitches,
)
from viz.layout import CHART_LAYOUT
from analytics.stuff_plus import _compute_stuff_plus


def page_development(data):
    st.header("Player Development")
    role = st.radio("View", ["Pitcher", "Hitter"], horizontal=True)

    if role == "Pitcher":
        pdf = filter_davidson(data, "pitcher")
        if pdf.empty:
            st.warning("No data.")
            return
        player = st.selectbox("Pitcher", sorted(pdf["Pitcher"].unique()), key="dv_p")
        pdata = pdf[pdf["Pitcher"] == player]
        metric = st.selectbox("Metric", ["RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak",
                                          "Extension", "VertApprAngle", "EffectiveVelo"], key="dv_mp")
        pts = sorted(pdata["TaggedPitchType"].dropna().unique())
        # Default to pitch types with >= 10% usage
        total = len(pdata["TaggedPitchType"].dropna())
        if total > 0:
            counts = pdata["TaggedPitchType"].dropna().value_counts()
            default_pts = sorted(counts[counts / total >= 0.10].index)
        else:
            default_pts = pts
        sel_pt = st.multiselect("Pitch Types", pts, default=default_pts, key="dv_pt")
        pdata = pdata[pdata["TaggedPitchType"].isin(sel_pt)]
        if pdata[metric].notna().any():
            daily = pdata.groupby(["Date", "TaggedPitchType"])[metric].agg(["mean", "count"]).reset_index()
            daily.columns = ["Date", "PitchType", "Value", "Count"]
            fig = px.scatter(daily, x="Date", y="Value", color="PitchType", size="Count",
                             color_discrete_map=PITCH_COLORS, trendline="lowess",
                             labels={"Value": metric})
            # Add per-pitch-type historical average lines
            for pt in sel_pt:
                pt_vals = pdata.loc[pdata["TaggedPitchType"] == pt, metric].dropna()
                if len(pt_vals) > 0:
                    fig.add_hline(
                        y=pt_vals.mean(),
                        line_dash="dot",
                        line_color=PITCH_COLORS.get(pt, "#777"),
                        opacity=0.5,
                        annotation_text=f"{pt} avg: {pt_vals.mean():.1f}",
                        annotation_font_size=10,
                        annotation_font_color=PITCH_COLORS.get(pt, "#777"),
                    )
            fig.update_layout(height=400, **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

            window = st.slider("Rolling window", 10, 100, 25, key="dv_wp")
            pdata_s = pdata.sort_values("Date")
            for pt in sel_pt:
                sub = pdata_s[pdata_s["TaggedPitchType"] == pt][metric].dropna()
                if len(sub) >= window:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(len(sub))), y=sub.rolling(window).mean(),
                                             name=pt, line=dict(color=PITCH_COLORS.get(pt, "#777"))))
                    fig.update_layout(title=f"{pt} Rolling {window} {metric}",
                                      xaxis_title="Pitch #", yaxis_title=metric, height=220,
                                      **CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            # Rolling Stuff+
            st.subheader("Rolling Stuff+")
            pdata_stuff = _compute_stuff_plus(pdata_s.copy())
            if "StuffPlus" in pdata_stuff.columns and pdata_stuff["StuffPlus"].notna().any():
                stuff_fig = go.Figure()
                has_trace = False
                for pt in sel_pt:
                    sub = pdata_stuff[pdata_stuff["TaggedPitchType"] == pt]["StuffPlus"].dropna()
                    if len(sub) >= window:
                        stuff_fig.add_trace(go.Scatter(
                            x=list(range(len(sub))),
                            y=sub.rolling(window).mean(),
                            name=pt,
                            line=dict(color=PITCH_COLORS.get(pt, "#777")),
                        ))
                        has_trace = True
                if has_trace:
                    stuff_fig.add_hline(y=100, line_dash="dash", line_color="gray",
                                        annotation_text="Avg (100)")
                    stuff_fig.update_layout(
                        title=f"Rolling {window} Stuff+",
                        xaxis_title="Pitch #",
                        yaxis_title="Stuff+",
                        height=350,
                        **CHART_LAYOUT,
                    )
                    st.plotly_chart(stuff_fig, use_container_width=True)
                else:
                    st.info("Not enough pitches per type for rolling Stuff+.")
            else:
                st.info("Stuff+ could not be computed for this pitcher.")
    else:
        hdf = filter_davidson(data, "batter")
        if hdf.empty:
            st.warning("No data.")
            return
        player = st.selectbox("Hitter", sorted(hdf["Batter"].unique()), key="dv_h")
        batted = hdf[(hdf["Batter"] == player) & (hdf["PitchCall"] == "InPlay")].dropna(subset=["ExitSpeed"])
        metric = st.selectbox("Metric", ["ExitSpeed", "Angle", "Distance"], key="dv_mh")
        if batted[metric].notna().any():
            daily = batted.groupby("Date")[metric].agg(["mean", "count"]).reset_index()
            daily.columns = ["Date", "Value", "Count"]
            fig = px.scatter(daily, x="Date", y="Value", size="Count", trendline="lowess",
                             labels={"Value": metric})
            fig.update_layout(height=350, **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
