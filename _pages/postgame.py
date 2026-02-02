"""Postgame Report page."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import (
    DAVIDSON_TEAM_ID, ROSTER_2026, JERSEY, POSITION, PITCH_COLORS,
    SWING_CALLS, CONTACT_CALLS,
    ZONE_SIDE, ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP,
    in_zone_mask, is_barrel_mask, display_name,
)
from viz.layout import CHART_LAYOUT, section_header
from viz.charts import add_strike_zone, make_spray_chart, make_movement_profile, player_header
from analytics.stuff_plus import _compute_stuff_plus
from analytics.command_plus import _compute_command_plus

BALL_RADIUS = 0.12  # ~1.45 in ≈ baseball radius in feet

def _pg_slug(name):
    """Create a key-safe slug from a player name."""
    return name.replace(" ", "").replace(",", "").replace(".", "")


def _pg_estimate_ip(pdf):
    """Estimate innings pitched from pitch-level data."""
    outs = 0
    if "OutsOnPlay" in pdf.columns:
        outs += pd.to_numeric(pdf["OutsOnPlay"], errors="coerce").fillna(0).sum()
    if "KorBB" in pdf.columns:
        outs += len(pdf[pdf["KorBB"] == "Strikeout"])
        outs -= len(pdf[(pdf["KorBB"] == "Strikeout") & (pdf.get("OutsOnPlay", pd.Series(dtype=float)).fillna(0) > 0)]) if "OutsOnPlay" in pdf.columns else 0
    full = int(outs // 3)
    part = int(outs % 3)
    return f"{full}.{part}"


def _pg_count_state(balls, strikes):
    """Classify count from pitcher's perspective: Ahead, Behind, Even."""
    if pd.isna(balls) or pd.isna(strikes):
        return "Even"
    b, s = int(balls), int(strikes)
    if s > b:
        return "Ahead"
    elif b > s:
        return "Behind"
    return "Even"


def _pg_inning_group(inning):
    """Classify inning into Early/Mid/Late."""
    if pd.isna(inning):
        return "Early"
    i = int(inning)
    if i <= 3:
        return "Early (1-3)"
    elif i <= 6:
        return "Mid (4-6)"
    return "Late (7+)"


def _pg_count_leverage(balls, strikes):
    """Classify count leverage: High if 3-ball or 2-strike count, else Medium/Low."""
    if pd.isna(balls) or pd.isna(strikes):
        return "Low"
    b, s = int(balls), int(strikes)
    if b == 3 or s == 2:
        return "High"
    if b == 2 or s == 1:
        return "Medium"
    return "Low"


def _pg_pitch_sequence_text(ab_df):
    """Build a compact text description of a pitch sequence for an at-bat."""
    parts = []
    for _, row in ab_df.iterrows():
        pt = row.get("TaggedPitchType", "?")
        velo = row.get("RelSpeed", np.nan)
        call = row.get("PitchCall", "?")
        v_str = f" {velo:.0f}" if pd.notna(velo) else ""
        call_short = {"StrikeCalled": "SC", "BallCalled": "BC", "StrikeSwinging": "SS",
                      "FoulBall": "F", "FoulBallNotFieldable": "F", "FoulBallFieldable": "F",
                      "InPlay": "IP", "HitByPitch": "HBP", "BallIntentional": "IB"}.get(call, call[:3] if isinstance(call, str) else "?")
        parts.append(f"{pt}{v_str} ({call_short})")
    return " → ".join(parts)


def _pg_mini_location_plot(ab_df, key_suffix=""):
    """Create a small location scatter for a single at-bat with numbered pitches."""
    loc = ab_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if loc.empty:
        return None
    loc = loc.reset_index(drop=True)
    loc["PitchNum"] = range(1, len(loc) + 1)
    fig = go.Figure()
    for _, row in loc.iterrows():
        pt = row.get("TaggedPitchType", "Other")
        color = PITCH_COLORS.get(pt, "#aaa")
        fig.add_trace(go.Scatter(
            x=[row["PlateLocSide"]], y=[row["PlateLocHeight"]],
            mode="markers+text", text=[str(int(row["PitchNum"]))],
            textposition="top center", textfont=dict(size=9, color="#000000"),
            marker=dict(size=10, color=color, line=dict(width=1, color="white")),
            showlegend=False,
            hovertemplate=f"#{int(row['PitchNum'])} {pt}<br>{row.get('PitchCall','')}<extra></extra>",
        ))
    add_strike_zone(fig)
    fig.update_layout(
        xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, scaleanchor="y"),
        yaxis=dict(range=[0, 5], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        height=220, margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ── Umpire Report ──
def _postgame_umpire(gd):
    """Render the Umpire Report tab for a single game."""
    section_header("Umpire Report")

    called = gd[gd["PitchCall"].isin(["StrikeCalled", "BallCalled"])].copy()
    called = called.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if called.empty:
        st.info("No called pitch location data available for this game.")
        return

    # Fixed rulebook zone for umpire evaluation (no batter-adaptive)
    iz = (called["PlateLocSide"].abs() <= ZONE_SIDE) & \
         called["PlateLocHeight"].between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP)
    is_strike = called["PitchCall"] == "StrikeCalled"
    called["InZone"] = iz
    called["Correct"] = (is_strike & iz) | (~is_strike & ~iz)
    called["Gifted"] = is_strike & ~iz
    called["Missed"] = ~is_strike & iz

    # ── 3b/3c: Accuracy scatter + metrics ──
    col_scatter, col_metrics = st.columns([2, 1])

    with col_scatter:
        section_header("Called Pitch Accuracy")
        correct = called[called["Correct"]]
        incorrect = called[~called["Correct"]]
        fig = go.Figure()
        if not correct.empty:
            fig.add_trace(go.Scatter(
                x=correct["PlateLocSide"], y=correct["PlateLocHeight"],
                mode="markers", marker=dict(size=7, color="#2ca02c", symbol="circle", opacity=0.7),
                name="Correct",
                customdata=correct[["PitchCall", "Batter", "Pitcher", "Inning"]].fillna("").values if all(c in correct.columns for c in ["PitchCall","Batter","Pitcher","Inning"]) else None,
                hovertemplate="%{customdata[0]}<br>Batter: %{customdata[1]}<br>Pitcher: %{customdata[2]}<br>Inn: %{customdata[3]}<extra></extra>" if all(c in correct.columns for c in ["PitchCall","Batter","Pitcher","Inning"]) else None,
            ))
        if not incorrect.empty:
            fig.add_trace(go.Scatter(
                x=incorrect["PlateLocSide"], y=incorrect["PlateLocHeight"],
                mode="markers", marker=dict(size=9, color="#d62728", symbol="x", opacity=0.85),
                name="Incorrect",
                customdata=incorrect[["PitchCall", "Batter", "Pitcher", "Inning"]].fillna("").values if all(c in incorrect.columns for c in ["PitchCall","Batter","Pitcher","Inning"]) else None,
                hovertemplate="%{customdata[0]}<br>Batter: %{customdata[1]}<br>Pitcher: %{customdata[2]}<br>Inn: %{customdata[3]}<extra></extra>" if all(c in incorrect.columns for c in ["PitchCall","Batter","Pitcher","Inning"]) else None,
            ))
        add_strike_zone(fig)
        fig.update_layout(
            xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, title="", fixedrange=True, scaleanchor="y"),
            yaxis=dict(range=[0, 5], showgrid=False, zeroline=False, title="", fixedrange=True),
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#000000", family="Inter, Arial, sans-serif"),
            margin=dict(l=20, r=10, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True, key="pg_ump_accuracy_scatter")

    with col_metrics:
        section_header("Accuracy Metrics")
        total = len(called)
        n_correct = called["Correct"].sum()
        n_strikes = is_strike.sum()
        n_balls = (~is_strike).sum()
        strike_correct = ((is_strike & iz).sum() / max(n_strikes, 1)) * 100
        ball_correct = ((~is_strike & ~iz).sum() / max(n_balls, 1)) * 100
        gifted = called["Gifted"].sum()
        missed = called["Missed"].sum()

        st.metric("Overall Accuracy", f"{n_correct / max(total, 1) * 100:.1f}%", f"{n_correct}/{total}")
        st.metric("Called Strike Accuracy", f"{strike_correct:.1f}%", f"{int((is_strike & iz).sum())}/{n_strikes}")
        st.metric("Called Ball Accuracy", f"{ball_correct:.1f}%", f"{int((~is_strike & ~iz).sum())}/{n_balls}")
        st.metric("Gifted Strikes", f"{int(gifted)}", help="Called strike outside zone")
        st.metric("Missed Strikes", f"{int(missed)}", help="Called ball inside zone")

    # ── 3d: Umpire's effective zone ──
    section_header("Umpire's Effective Zone")
    cs_locs = called[called["PitchCall"] == "StrikeCalled"]
    if len(cs_locs) >= 5:
        fig_ez = go.Figure()
        fig_ez.add_trace(go.Histogram2dContour(
            x=cs_locs["PlateLocSide"], y=cs_locs["PlateLocHeight"],
            colorscale=[[0, "rgba(255,255,255,0)"], [0.3, "rgba(200,60,60,0.3)"],
                        [0.6, "rgba(200,60,60,0.5)"], [1.0, "rgba(200,60,60,0.8)"]],
            showscale=False, ncontours=8,
            contours=dict(showlines=True, coloring="fill"),
            line=dict(width=0.5, color="rgba(150,150,150,0.3)"),
        ))
        fig_ez.add_shape(type="rect", x0=-ZONE_SIDE, x1=ZONE_SIDE,
                         y0=ZONE_HEIGHT_BOT, y1=ZONE_HEIGHT_TOP,
                         line=dict(color="#333", width=2, dash="dash"),
                         fillcolor="rgba(0,0,0,0)")
        fig_ez.update_layout(
            xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, title="", fixedrange=True, scaleanchor="y"),
            yaxis=dict(range=[0, 5], showgrid=False, zeroline=False, title="", fixedrange=True),
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#000000", family="Inter, Arial, sans-serif"),
            margin=dict(l=20, r=10, t=30, b=20),
        )
        st.plotly_chart(fig_ez, use_container_width=True, key="pg_ump_eff_zone")
    else:
        st.info("Not enough called strikes for effective zone visualization.")

    # ── 3e: Breakdowns ──
    section_header("Breakdowns")
    bd_c1, bd_c2 = st.columns(2)

    # By count state
    with bd_c1:
        st.markdown("**By Count State** (Pitcher POV)")
        if "Balls" in called.columns and "Strikes" in called.columns:
            called["_CountState"] = called.apply(lambda r: _pg_count_state(r.get("Balls"), r.get("Strikes")), axis=1)
            rows = []
            for state in ["Ahead", "Even", "Behind"]:
                sub = called[called["_CountState"] == state]
                if sub.empty:
                    continue
                rows.append({
                    "Count": state, "Pitches": len(sub),
                    "Accuracy%": f"{sub['Correct'].mean()*100:.1f}",
                    "Gifted": int(sub["Gifted"].sum()),
                    "Missed": int(sub["Missed"].sum()),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Count data not available.")

    # By inning group
    with bd_c2:
        st.markdown("**By Inning Group**")
        if "Inning" in called.columns:
            called["_InnGrp"] = called["Inning"].apply(_pg_inning_group)
            rows = []
            for grp_name in ["Early (1-3)", "Mid (4-6)", "Late (7+)"]:
                sub = called[called["_InnGrp"] == grp_name]
                if sub.empty:
                    continue
                rows.append({
                    "Innings": grp_name, "Pitches": len(sub),
                    "Accuracy%": f"{sub['Correct'].mean()*100:.1f}",
                    "Gifted": int(sub["Gifted"].sum()),
                    "Missed": int(sub["Missed"].sum()),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Inning data not available.")

    # By batter side
    if "BatterSide" in called.columns:
        section_header("By Batter Side")
        bs_c1, bs_c2 = st.columns(2)
        for side, col in [("Right", bs_c1), ("Left", bs_c2)]:
            side_df = called[called["BatterSide"] == side]
            if side_df.empty:
                continue
            with col:
                st.markdown(f"**{side}-Handed Hitters** ({len(side_df)} calls, "
                            f"{side_df['Correct'].mean()*100:.1f}% accuracy)")
                fig_s = go.Figure()
                sc = side_df[side_df["Correct"]]
                si = side_df[~side_df["Correct"]]
                if not sc.empty:
                    fig_s.add_trace(go.Scatter(x=sc["PlateLocSide"], y=sc["PlateLocHeight"],
                                              mode="markers", marker=dict(size=6, color="#2ca02c", opacity=0.6),
                                              name="Correct", showlegend=False))
                if not si.empty:
                    fig_s.add_trace(go.Scatter(x=si["PlateLocSide"], y=si["PlateLocHeight"],
                                              mode="markers", marker=dict(size=8, color="#d62728", symbol="x", opacity=0.8),
                                              name="Incorrect", showlegend=False))
                add_strike_zone(fig_s)
                fig_s.update_layout(
                    xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, fixedrange=True, scaleanchor="y"),
                    yaxis=dict(range=[0, 5], showgrid=False, zeroline=False, fixedrange=True),
                    height=280, plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(l=10, r=10, t=10, b=10),
                    font=dict(color="#000000", family="Inter, Arial, sans-serif"),
                )
                st.plotly_chart(fig_s, use_container_width=True, key=f"pg_ump_side_{side}")

    # By pitcher
    if "Pitcher" in called.columns:
        section_header("By Pitcher")
        rows = []
        for p, sub in called.groupby("Pitcher"):
            rows.append({
                "Pitcher": display_name(p), "Pitches": len(sub),
                "Accuracy%": f"{sub['Correct'].mean()*100:.1f}",
                "Gifted": int(sub["Gifted"].sum()),
                "Missed": int(sub["Missed"].sum()),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("Pitches", ascending=False), use_container_width=True, hide_index=True)

    # ── 3f: Shadow zone analysis ──
    section_header("Shadow Zone Analysis")
    shadow = (
        (called["PlateLocSide"].abs().between(ZONE_SIDE - BALL_RADIUS, ZONE_SIDE + BALL_RADIUS)) |
        (called["PlateLocHeight"].between(ZONE_HEIGHT_BOT - BALL_RADIUS, ZONE_HEIGHT_BOT + BALL_RADIUS)) |
        (called["PlateLocHeight"].between(ZONE_HEIGHT_TOP - BALL_RADIUS, ZONE_HEIGHT_TOP + BALL_RADIUS))
    )
    shadow_df = called[shadow]
    non_shadow_df = called[~shadow]
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Shadow Zone Pitches", len(shadow_df))
    with sc2:
        if len(shadow_df) > 0:
            st.metric("Shadow Zone Accuracy", f"{shadow_df['Correct'].mean()*100:.1f}%")
        else:
            st.metric("Shadow Zone Accuracy", "N/A")
    with sc3:
        if len(non_shadow_df) > 0:
            st.metric("Non-Shadow Accuracy", f"{non_shadow_df['Correct'].mean()*100:.1f}%")
        else:
            st.metric("Non-Shadow Accuracy", "N/A")

    # ── 3g: Impact metrics ──
    section_header("Impact Metrics — Missed Calls by Leverage")
    if "Balls" in called.columns and "Strikes" in called.columns:
        called["_Leverage"] = called.apply(lambda r: _pg_count_leverage(r.get("Balls"), r.get("Strikes")), axis=1)
        gifted_df = called[called["Gifted"]]
        missed_df = called[called["Missed"]]
        imp_rows = []
        for lev in ["High", "Medium", "Low"]:
            imp_rows.append({
                "Leverage": lev,
                "Gifted Strikes": int(gifted_df[gifted_df["_Leverage"] == lev].shape[0]),
                "Missed Strikes": int(missed_df[missed_df["_Leverage"] == lev].shape[0]),
            })
        st.dataframe(pd.DataFrame(imp_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Count data not available for leverage analysis.")


# ── Pitcher Report ──
def _postgame_pitchers(gd, data):
    """Render the Pitcher Report tab for a single game."""
    section_header("Pitcher Report — Davidson Arms")

    dav_pitching = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID].copy()
    if dav_pitching.empty:
        st.info("No Davidson pitching data for this game.")
        return

    pitchers = dav_pitching.groupby("Pitcher").size().sort_values(ascending=False).index.tolist()

    for idx, pitcher in enumerate(pitchers):
        pdf = dav_pitching[dav_pitching["Pitcher"] == pitcher].copy()
        n_pitches = len(pdf)
        jersey = JERSEY.get(pitcher, "")
        pos = POSITION.get(pitcher, "P")
        ip_est = _pg_estimate_ip(pdf)
        ks = len(pdf[pdf["KorBB"] == "Strikeout"]) if "KorBB" in pdf.columns else 0
        bbs = len(pdf[pdf["KorBB"] == "Walk"]) if "KorBB" in pdf.columns else 0
        hits = len(pdf[pdf["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"])]) if "PlayResult" in pdf.columns else 0

        player_header(pitcher, jersey, pos,
                      f"{n_pitches} pitches · ~{ip_est} IP",
                      f"K: {ks}  BB: {bbs}  H: {hits}")

        with st.expander(f"Details — {display_name(pitcher)}", expanded=(idx == 0)):
            _pg_pitcher_detail(pdf, data, pitcher)


def _pg_pitcher_detail(pdf, data, pitcher):
    """Render detailed pitcher breakdown inside an expander."""
    slug = _pg_slug(pitcher)
    col_left, col_right = st.columns(2)

    with col_left:
        # Pitch mix table
        section_header("Pitch Mix")
        if "TaggedPitchType" in pdf.columns:
            mix_rows = []
            total = len(pdf)
            for pt, grp in pdf.groupby("TaggedPitchType"):
                row = {"Pitch": pt, "N": len(grp), "Usage%": f"{len(grp)/total*100:.1f}"}
                if "RelSpeed" in grp.columns:
                    v = grp["RelSpeed"].dropna()
                    row["Avg Velo"] = f"{v.mean():.1f}" if len(v) > 0 else "-"
                    row["Max Velo"] = f"{v.max():.1f}" if len(v) > 0 else "-"
                if "SpinRate" in grp.columns:
                    s = grp["SpinRate"].dropna()
                    row["Avg Spin"] = f"{s.mean():.0f}" if len(s) > 0 else "-"
                if "InducedVertBreak" in grp.columns:
                    ivb = grp["InducedVertBreak"].dropna()
                    row["Avg IVB"] = f"{ivb.mean():.1f}" if len(ivb) > 0 else "-"
                if "HorzBreak" in grp.columns:
                    hb = grp["HorzBreak"].dropna()
                    row["Avg HB"] = f"{hb.mean():.1f}" if len(hb) > 0 else "-"
                mix_rows.append(row)
            if mix_rows:
                st.dataframe(pd.DataFrame(mix_rows).sort_values("N", ascending=False), use_container_width=True, hide_index=True)

        # Stuff+
        section_header("Stuff+")
        stuff = _compute_stuff_plus(pdf)
        if "StuffPlus" in stuff.columns and "TaggedPitchType" in stuff.columns:
            sp_summary = stuff.groupby("TaggedPitchType")["StuffPlus"].mean().round(0).reset_index()
            sp_summary.columns = ["Pitch", "Stuff+"]
            st.dataframe(sp_summary.sort_values("Stuff+", ascending=False), use_container_width=True, hide_index=True)

        # Command+
        section_header("Command+")
        cmd = _compute_command_plus(pdf, data)
        if not cmd.empty:
            st.dataframe(cmd, use_container_width=True, hide_index=True)

    with col_right:
        # Pitch location scatter
        section_header("Pitch Locations")
        loc = pdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if not loc.empty and "TaggedPitchType" in loc.columns:
            fig_loc = go.Figure()
            for pt in sorted(loc["TaggedPitchType"].unique()):
                sub = loc[loc["TaggedPitchType"] == pt]
                color = PITCH_COLORS.get(pt, "#aaa")
                hover_data = []
                for _, row in sub.iterrows():
                    v = f"{row['RelSpeed']:.1f}" if pd.notna(row.get("RelSpeed")) else "?"
                    r = row.get("PitchCall", "?")
                    hover_data.append(f"{pt} {v}mph<br>{r}")
                fig_loc.add_trace(go.Scatter(
                    x=sub["PlateLocSide"], y=sub["PlateLocHeight"],
                    mode="markers", marker=dict(size=7, color=color, opacity=0.8,
                                                line=dict(width=0.5, color="white")),
                    name=pt, text=hover_data, hoverinfo="text",
                ))
            add_strike_zone(fig_loc)
            fig_loc.update_layout(
                xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, title="", fixedrange=True, scaleanchor="y"),
                yaxis=dict(range=[0, 5], showgrid=False, zeroline=False, title="", fixedrange=True),
                height=380, plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color="#000000", family="Inter, Arial, sans-serif"),
                margin=dict(l=15, r=10, t=25, b=15),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
            )
            st.plotly_chart(fig_loc, use_container_width=True, key=f"pg_pit_loc_{slug}")

        # Movement profile
        section_header("Movement Profile")
        fig_mov = make_movement_profile(pdf, height=380)
        if fig_mov:
            st.plotly_chart(fig_mov, use_container_width=True, key=f"pg_pit_mov_{slug}")
        else:
            st.caption("Not enough movement data.")

    # Full width — Whiff & Chase table
    section_header("Whiff & Chase by Pitch Type")
    if "TaggedPitchType" in pdf.columns:
        wc_rows = []
        for pt, grp in pdf.groupby("TaggedPitchType"):
            n = len(grp)
            swings = grp[grp["PitchCall"].isin(SWING_CALLS)]
            whiffs = grp[grp["PitchCall"] == "StrikeSwinging"]
            loc_grp = grp.dropna(subset=["PlateLocSide", "PlateLocHeight"])
            if len(loc_grp) > 0:
                oz = ~in_zone_mask(loc_grp)
                oz_pitches = len(loc_grp[oz])
                oz_swings = loc_grp[oz & loc_grp["PitchCall"].isin(SWING_CALLS)]
                chase_pct = len(oz_swings) / max(oz_pitches, 1) * 100
            else:
                chase_pct = np.nan
            csw = grp["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).sum()
            wc_rows.append({
                "Pitch": pt, "Pitches": n,
                "Swings": len(swings),
                "Whiff%": f"{len(whiffs)/max(len(swings),1)*100:.1f}" if len(swings) > 0 else "-",
                "Chase%": f"{chase_pct:.1f}" if pd.notna(chase_pct) else "-",
                "CSW%": f"{csw/n*100:.1f}",
            })
        if wc_rows:
            st.dataframe(pd.DataFrame(wc_rows).sort_values("Pitches", ascending=False), use_container_width=True, hide_index=True)

    # Release point scatter
    if "RelSide" in pdf.columns and "RelHeight" in pdf.columns:
        section_header("Release Point")
        rel = pdf.dropna(subset=["RelSide", "RelHeight"])
        if not rel.empty and "TaggedPitchType" in rel.columns:
            fig_rel = go.Figure()
            for pt in sorted(rel["TaggedPitchType"].unique()):
                sub = rel[rel["TaggedPitchType"] == pt]
                color = PITCH_COLORS.get(pt, "#aaa")
                fig_rel.add_trace(go.Scatter(
                    x=sub["RelSide"], y=sub["RelHeight"],
                    mode="markers", marker=dict(size=6, color=color, opacity=0.7),
                    name=pt,
                ))
            fig_rel.update_layout(
                xaxis_title="Release Side (ft)", yaxis_title="Release Height (ft)",
                height=300, plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color="#000000", family="Inter, Arial, sans-serif"),
                margin=dict(l=50, r=10, t=25, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
            )
            st.plotly_chart(fig_rel, use_container_width=True, key=f"pg_pit_rel_{slug}")

    # Key at-bats: K and BB
    section_header("Key At-Bats (K & BB)")
    if "KorBB" in pdf.columns:
        pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in pdf.columns]
        sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in pdf.columns]
        key_abs = pdf[pdf["KorBB"].isin(["Strikeout", "Walk"])]
        if not key_abs.empty and len(pa_cols) >= 2:
            for _, pa_key in key_abs.drop_duplicates(subset=pa_cols).iterrows():
                mask = pd.Series(True, index=pdf.index)
                for c in pa_cols:
                    mask = mask & (pdf[c] == pa_key[c])
                ab = pdf[mask].sort_values(sort_cols) if sort_cols else pdf[mask]
                if ab.empty:
                    continue
                result = pa_key.get("KorBB", "?")
                batter = display_name(pa_key["Batter"]) if "Batter" in pa_key.index else "?"
                inn = pa_key.get("Inning", "?")
                st.markdown(f"**Inn {inn}** vs {batter} — **{result}** ({len(ab)} pitches)")
                ab_c1, ab_c2 = st.columns([2, 1])
                with ab_c1:
                    st.caption(_pg_pitch_sequence_text(ab))
                with ab_c2:
                    fig_ab = _pg_mini_location_plot(ab)
                    if fig_ab:
                        ab_key = f"pg_pit_kab_{slug}_{inn}_{_pg_slug(str(batter))}"
                        st.plotly_chart(fig_ab, use_container_width=True, key=ab_key)


# ── Hitter Report ──
def _postgame_hitters(gd, data):
    """Render the Hitter Report tab for a single game."""
    section_header("Hitter Report — Davidson Bats")

    dav_hitting = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID].copy()
    if dav_hitting.empty:
        st.info("No Davidson batting data for this game.")
        return

    batters = dav_hitting.groupby("Batter").size().sort_values(ascending=False).index.tolist()

    for idx, batter in enumerate(batters):
        bdf = dav_hitting[dav_hitting["Batter"] == batter].copy()
        n_pitches = len(bdf)
        jersey = JERSEY.get(batter, "")
        pos = POSITION.get(batter, "")

        # Count PAs
        pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in bdf.columns]
        pa = bdf.drop_duplicates(subset=pa_cols).shape[0] if len(pa_cols) >= 2 else 0
        hits = len(bdf[bdf["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"])]) if "PlayResult" in bdf.columns else 0
        bbs = len(bdf[bdf["KorBB"] == "Walk"]) if "KorBB" in bdf.columns else 0
        ks = len(bdf[bdf["KorBB"] == "Strikeout"]) if "KorBB" in bdf.columns else 0
        bbe = len(bdf[(bdf["PitchCall"] == "InPlay") & bdf["ExitSpeed"].notna()]) if "ExitSpeed" in bdf.columns else 0

        player_header(batter, jersey, pos,
                      f"{n_pitches} pitches seen · {pa} PA",
                      f"H: {hits}  BB: {bbs}  K: {ks}  BBE: {bbe}")

        with st.expander(f"Details — {display_name(batter)}", expanded=(idx == 0)):
            _pg_hitter_detail(bdf, data, batter)


def _pg_hitter_detail(bdf, data, batter):
    """Render detailed hitter breakdown inside an expander."""
    slug = _pg_slug(batter)
    col_left, col_right = st.columns(2)

    with col_left:
        # Plate discipline metrics
        section_header("Plate Discipline")
        loc_df = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if not loc_df.empty:
            iz = in_zone_mask(loc_df)
            in_zone_df = loc_df[iz]
            out_zone_df = loc_df[~iz]
            swings = loc_df[loc_df["PitchCall"].isin(SWING_CALLS)]
            iz_swings = in_zone_df[in_zone_df["PitchCall"].isin(SWING_CALLS)]
            oz_swings = out_zone_df[out_zone_df["PitchCall"].isin(SWING_CALLS)]
            whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"]
            iz_contacts = in_zone_df[in_zone_df["PitchCall"].isin(CONTACT_CALLS)]

            d_c1, d_c2 = st.columns(2)
            with d_c1:
                st.metric("Zone Swing%", f"{len(iz_swings)/max(len(in_zone_df),1)*100:.1f}%")
                st.metric("Chase%", f"{len(oz_swings)/max(len(out_zone_df),1)*100:.1f}%")
                st.metric("SwStr%", f"{len(whiffs)/max(len(bdf),1)*100:.1f}%")
            with d_c2:
                st.metric("Whiff%", f"{len(whiffs)/max(len(swings),1)*100:.1f}%" if len(swings) > 0 else "N/A")
                st.metric("Zone Contact%", f"{len(iz_contacts)/max(len(iz_swings),1)*100:.1f}%" if len(iz_swings) > 0 else "N/A")
        else:
            st.caption("No location data available.")

        # By pitch type table
        section_header("By Pitch Type")
        if "TaggedPitchType" in bdf.columns:
            pt_rows = []
            for pt, grp in bdf.groupby("TaggedPitchType"):
                n = len(grp)
                sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
                wh = grp[grp["PitchCall"] == "StrikeSwinging"]
                loc_g = grp.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                chase_pct = np.nan
                if len(loc_g) > 0:
                    oz_g = loc_g[~in_zone_mask(loc_g)]
                    if len(oz_g) > 0:
                        chase_pct = len(oz_g[oz_g["PitchCall"].isin(SWING_CALLS)]) / len(oz_g) * 100
                contact = grp[grp["PitchCall"].isin(CONTACT_CALLS)]
                avg_ev = contact["ExitSpeed"].mean() if "ExitSpeed" in contact.columns and len(contact) > 0 else np.nan
                pt_rows.append({
                    "Pitch": pt, "N": n,
                    "Swing%": f"{len(sw)/n*100:.1f}",
                    "Whiff%": f"{len(wh)/max(len(sw),1)*100:.1f}" if len(sw) > 0 else "-",
                    "Chase%": f"{chase_pct:.1f}" if pd.notna(chase_pct) else "-",
                    "Avg EV": f"{avg_ev:.1f}" if pd.notna(avg_ev) else "-",
                })
            if pt_rows:
                st.dataframe(pd.DataFrame(pt_rows).sort_values("N", ascending=False), use_container_width=True, hide_index=True)

    with col_right:
        # Batted ball quality
        section_header("Batted Ball Quality")
        in_play = bdf[(bdf["PitchCall"] == "InPlay")].copy()
        bbe_df = in_play.dropna(subset=["ExitSpeed"]) if "ExitSpeed" in in_play.columns else pd.DataFrame()
        if not bbe_df.empty:
            ev = bbe_df["ExitSpeed"]
            q_c1, q_c2 = st.columns(2)
            with q_c1:
                st.metric("Avg EV", f"{ev.mean():.1f} mph")
                st.metric("Max EV", f"{ev.max():.1f} mph")
                st.metric("BBE", len(bbe_df))
            with q_c2:
                if "Angle" in bbe_df.columns:
                    la = bbe_df["Angle"].dropna()
                    st.metric("Avg LA", f"{la.mean():.1f}°" if len(la) > 0 else "N/A")
                hh = (ev >= 95).mean() * 100
                st.metric("Hard Hit%", f"{hh:.1f}%")
                if "ExitSpeed" in bbe_df.columns and "Angle" in bbe_df.columns:
                    barrel = is_barrel_mask(bbe_df).mean() * 100
                    st.metric("Barrel%", f"{barrel:.1f}%")
        else:
            st.caption("No batted ball data.")

        # Spray chart
        section_header("Spray Chart")
        if not in_play.empty:
            fig_spray = make_spray_chart(in_play, height=320)
            if fig_spray:
                st.plotly_chart(fig_spray, use_container_width=True, key=f"pg_hit_spray_{slug}")
            else:
                st.caption("No spray chart data.")

    # Full width — Key at-bats
    section_header("Key At-Bats")
    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in bdf.columns]
    sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in bdf.columns]
    if len(pa_cols) >= 2:
        # Identify key ABs: HR, XBH, K, BB, long PAs (6+ pitches)
        pa_groups = []
        for pa_key, ab in bdf.groupby(pa_cols[1:]):  # group within game
            if not isinstance(pa_key, tuple):
                pa_key = (pa_key,)
            ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
            is_key = False
            result_label = ""
            if "PlayResult" in ab.columns:
                if ab["PlayResult"].eq("HomeRun").any():
                    is_key, result_label = True, "HR"
                elif ab["PlayResult"].isin(["Double", "Triple"]).any():
                    is_key, result_label = True, ab[ab["PlayResult"].isin(["Double", "Triple"])]["PlayResult"].iloc[0]
            if "KorBB" in ab.columns:
                if ab["KorBB"].eq("Strikeout").any():
                    is_key, result_label = True, result_label or "K"
                elif ab["KorBB"].eq("Walk").any():
                    is_key, result_label = True, result_label or "BB"
            if len(ab) >= 6:
                is_key = True
                result_label = result_label or f"{len(ab)}-pitch PA"
            if is_key:
                pa_groups.append((ab_sorted, result_label))

        if pa_groups:
            for ab, result_label in pa_groups:
                inn = ab.iloc[0].get("Inning", "?")
                pitcher_name = display_name(ab.iloc[0]["Pitcher"]) if "Pitcher" in ab.columns else "?"
                st.markdown(f"**Inn {inn}** vs {pitcher_name} — **{result_label}** ({len(ab)} pitches)")
                ab_c1, ab_c2 = st.columns([2, 1])
                with ab_c1:
                    st.caption(_pg_pitch_sequence_text(ab))
                with ab_c2:
                    fig_ab = _pg_mini_location_plot(ab)
                    if fig_ab:
                        ab_key = f"pg_hit_kab_{slug}_{inn}_{_pg_slug(str(ab.iloc[0].get('Pitcher','')))}"
                        st.plotly_chart(fig_ab, use_container_width=True, key=ab_key)
        else:
            st.caption("No notable at-bats (HR, XBH, K, BB, or 6+ pitch PA).")
    else:
        st.caption("PA identification columns not available.")


# ── Main postgame page ──
def page_postgame(data):
    """Postgame Report page — single game selector with Umpire, Pitcher, Hitter tabs."""
    st.title("Postgame Report")

    # Filter to Davidson games
    dav = data[(data["PitcherTeam"] == DAVIDSON_TEAM_ID) | (data["BatterTeam"] == DAVIDSON_TEAM_ID)]
    if dav.empty:
        st.warning("No Davidson game data available.")
        return

    # Build game list
    games = dav.groupby(["Date", "GameID"]).agg(
        Home=("HomeTeam", "first"),
        Away=("AwayTeam", "first"),
        Pitches=("PitchNo", "count"),
    ).reset_index().sort_values("Date", ascending=False)

    if games.empty:
        st.warning("No games found.")
        return

    game_labels = {}
    for _, row in games.iterrows():
        dt = row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "?"
        game_labels[row["GameID"]] = f"{dt}  {row['Away']} @ {row['Home']}  ({row['Pitches']} pitches)"

    sel_game = st.selectbox("Select Game", games["GameID"].tolist(),
                            format_func=lambda g: game_labels.get(g, str(g)),
                            key="pg_game_select")

    gd = data[data["GameID"] == sel_game]
    if gd.empty:
        st.warning("No data for selected game.")
        return

    tab_ump, tab_pit, tab_hit = st.tabs(["Umpire Report", "Pitcher Report", "Hitter Report"])
    with tab_ump:
        _postgame_umpire(gd)
    with tab_pit:
        _postgame_pitchers(gd, data)
    with tab_hit:
        _postgame_hitters(gd, data)
