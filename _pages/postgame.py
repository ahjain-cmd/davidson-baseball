"""Postgame Report page."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import percentileofscore

from config import (
    DAVIDSON_TEAM_ID, ROSTER_2026, JERSEY, POSITION, PITCH_COLORS,
    SWING_CALLS, CONTACT_CALLS,
    ZONE_SIDE, ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP,
    PLATE_SIDE_MAX, PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX,
    in_zone_mask, is_barrel_mask, display_name, get_percentile,
    _friendly_team_name,
)
from viz.layout import CHART_LAYOUT, section_header
from viz.charts import add_strike_zone, make_spray_chart, make_movement_profile, player_header, _add_grid_zone_outline
from viz.percentiles import render_savant_percentile_section, savant_color
from analytics.stuff_plus import _compute_stuff_plus
from analytics.command_plus import _compute_command_plus
from analytics.expected import _create_zone_grid_data

BALL_RADIUS = 0.12  # ~1.45 in ≈ baseball radius in feet

_GRADE_THRESHOLDS = {
    "A+": 95, "A": 85, "A-": 80,
    "B+": 75, "B": 65, "B-": 60,
    "C+": 55, "C": 45, "C-": 40,
    "D": 30, "F": 0,
}


def _letter_grade(score):
    """Convert a 0-100 score to a letter grade."""
    for grade, threshold in _GRADE_THRESHOLDS.items():
        if score >= threshold:
            return grade
    return "F"


def _grade_color(grade):
    """Return color for letter grade."""
    if grade.startswith("A"):
        return "#1dbe3a"
    if grade.startswith("B"):
        return "#2d7fc1"
    if grade.startswith("C"):
        return "#f7c631"
    if grade.startswith("D"):
        return "#fe6100"
    return "#d22d49"


# ── Soft Tier System ─────────────────────────────────────────────────────────
_TIER_THRESHOLDS = {"Strength": 70, "Average": 40}  # else "Needs Work"

_MIN_PITCHER_PITCHES = 20
_MIN_HITTER_PAS = 3


def _tier_label(score):
    """Convert a 0-100 score to a soft tier label."""
    if pd.isna(score) or score is None:
        return "N/A"
    if score >= 70:
        return "Strength"
    if score >= 40:
        return "Average"
    return "Needs Work"


def _tier_color(tier):
    """Return color for tier badge."""
    return {"Strength": "#2e7d32", "Average": "#f9a825", "Needs Work": "#c62828"}.get(tier, "#9e9e9e")


def _tier_icon(tier):
    """Return icon character for tier badge."""
    return {"Strength": "+", "Average": "~", "Needs Work": "-"}.get(tier, "")

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
        height=280, margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ── Game Summary ──
def _postgame_summary(gd):
    """Render a quick game summary section at the top."""
    section_header("Game Summary")

    home = gd["HomeTeam"].iloc[0] if "HomeTeam" in gd.columns else "?"
    away = gd["AwayTeam"].iloc[0] if "AwayTeam" in gd.columns else "?"
    total_pitches = len(gd)
    innings = int(gd["Inning"].max()) if "Inning" in gd.columns else "?"

    dav_pitching = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID]
    dav_hitting = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID]
    opp_pitching = gd[gd["PitcherTeam"] != DAVIDSON_TEAM_ID]

    # Davidson pitching stats
    dav_ks = len(dav_pitching[dav_pitching["KorBB"] == "Strikeout"]) if "KorBB" in dav_pitching.columns else 0
    dav_bbs = len(dav_pitching[dav_pitching["KorBB"] == "Walk"]) if "KorBB" in dav_pitching.columns else 0
    dav_hits_allowed = len(dav_pitching[dav_pitching["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"])]) if "PlayResult" in dav_pitching.columns else 0

    # Davidson hitting stats
    dav_hits = len(dav_hitting[dav_hitting["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"])]) if "PlayResult" in dav_hitting.columns else 0
    dav_hr = len(dav_hitting[dav_hitting["PlayResult"] == "HomeRun"]) if "PlayResult" in dav_hitting.columns else 0
    dav_hit_bbs = len(dav_hitting[dav_hitting["KorBB"] == "Walk"]) if "KorBB" in dav_hitting.columns else 0
    dav_hit_ks = len(dav_hitting[dav_hitting["KorBB"] == "Strikeout"]) if "KorBB" in dav_hitting.columns else 0

    # Batted ball quality
    dav_bbe = dav_hitting[(dav_hitting["PitchCall"] == "InPlay") & dav_hitting["ExitSpeed"].notna()] if "ExitSpeed" in dav_hitting.columns else pd.DataFrame()
    avg_ev = dav_bbe["ExitSpeed"].mean() if len(dav_bbe) > 0 else np.nan
    max_ev = dav_bbe["ExitSpeed"].max() if len(dav_bbe) > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"**{away} @ {home}**")
        st.caption(f"{innings} innings · {total_pitches} total pitches")
    with c2:
        st.markdown("**Davidson Pitching**")
        st.caption(f"K: {dav_ks}  BB: {dav_bbs}  H: {dav_hits_allowed}")
    with c3:
        st.markdown("**Davidson Hitting**")
        st.caption(f"H: {dav_hits}  HR: {dav_hr}  BB: {dav_hit_bbs}  K: {dav_hit_ks}")
    with c4:
        st.markdown("**Batted Ball Quality**")
        ev_str = f"Avg EV: {avg_ev:.1f}" if pd.notna(avg_ev) else "Avg EV: -"
        mx_str = f"Max EV: {max_ev:.1f}" if pd.notna(max_ev) else "Max EV: -"
        st.caption(f"{ev_str}  {mx_str}  BBE: {len(dav_bbe)}")

    st.markdown("---")


# ── Umpire Report ──
def _postgame_umpire(gd):
    """Render the Umpire Report tab for a single game."""
    section_header("Umpire Report")

    called = gd[gd["PitchCall"].isin(["StrikeCalled", "BallCalled"])].copy()
    # Ensure numeric locations and remove invalid/out-of-range tracking
    called["PlateLocSide"] = pd.to_numeric(called.get("PlateLocSide"), errors="coerce")
    called["PlateLocHeight"] = pd.to_numeric(called.get("PlateLocHeight"), errors="coerce")
    called = called.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    valid_loc = (
        called["PlateLocSide"].between(-PLATE_SIDE_MAX, PLATE_SIDE_MAX) &
        called["PlateLocHeight"].between(PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX)
    )
    dropped = int((~valid_loc).sum())
    called = called.loc[valid_loc].copy()
    if called.empty:
        st.info("No called pitch location data available for this game.")
        return
    if dropped > 0:
        st.caption(f"Excluded {dropped} called pitches with invalid tracking locations.")

    # Fixed rulebook zone for umpire evaluation (no batter-adaptive)
    iz = in_zone_mask(called)
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

    # ── 3h: Missed Calls Impact on Davidson ──
    section_header("Missed Calls — Davidson Impact")
    incorrect = called[~called["Correct"]].copy()
    if not incorrect.empty and "PitcherTeam" in incorrect.columns:
        dav_pitching_mc = incorrect[incorrect["PitcherTeam"] == DAVIDSON_TEAM_ID]
        dav_hitting_mc = incorrect[incorrect["BatterTeam"] == DAVIDSON_TEAM_ID] if "BatterTeam" in incorrect.columns else pd.DataFrame()

        mc_c1, mc_c2 = st.columns(2)
        with mc_c1:
            st.markdown("**When Davidson is Pitching**")
            if not dav_pitching_mc.empty:
                gifted_against = dav_pitching_mc[dav_pitching_mc["Gifted"]]
                missed_for = dav_pitching_mc[dav_pitching_mc["Missed"]]
                st.caption(f"Gifted strikes (help): {len(gifted_against)}  |  "
                           f"Missed strikes (hurt): {len(missed_for)}")
                net = len(gifted_against) - len(missed_for)
                label = "net favorable" if net > 0 else "net unfavorable" if net < 0 else "neutral"
                st.metric("Net Impact", f"{net:+d} calls", label)
            else:
                st.caption("No missed calls when Davidson was pitching.")

        with mc_c2:
            st.markdown("**When Davidson is Hitting**")
            if not dav_hitting_mc.empty:
                gifted_vs = dav_hitting_mc[dav_hitting_mc["Gifted"]]
                missed_vs = dav_hitting_mc[dav_hitting_mc["Missed"]]
                st.caption(f"Gifted strikes (hurt): {len(gifted_vs)}  |  "
                           f"Missed strikes (help): {len(missed_vs)}")
                net = len(missed_vs) - len(gifted_vs)
                label = "net favorable" if net > 0 else "net unfavorable" if net < 0 else "neutral"
                st.metric("Net Impact", f"{net:+d} calls", label)
            else:
                st.caption("No missed calls when Davidson was hitting.")
    else:
        st.caption("No missed calls data available.")


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

    # First-Pitch Strike% & Count Performance
    fp_c1, fp_c2 = st.columns(2)
    with fp_c1:
        section_header("First-Pitch Strike%")
        if "Balls" in pdf.columns and "Strikes" in pdf.columns:
            first_pitches = pdf[(pdf["Balls"] == 0) & (pdf["Strikes"] == 0)]
            if len(first_pitches) > 0:
                fp_strikes = first_pitches[first_pitches["PitchCall"].isin(
                    ["StrikeCalled", "StrikeSwinging", "FoulBall", "FoulBallNotFieldable",
                     "FoulBallFieldable", "InPlay"])]
                fps_pct = len(fp_strikes) / len(first_pitches) * 100
                st.metric("First-Pitch Strike%", f"{fps_pct:.1f}%",
                          f"{len(fp_strikes)}/{len(first_pitches)} PAs")
                if "TaggedPitchType" in first_pitches.columns:
                    fp_mix = first_pitches["TaggedPitchType"].value_counts()
                    fp_rows = [{"Pitch": pt, "N": n, "Strike%": f"{first_pitches[(first_pitches['TaggedPitchType']==pt) & first_pitches['PitchCall'].isin(['StrikeCalled','StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay'])].shape[0]/max(n,1)*100:.1f}"} for pt, n in fp_mix.items()]
                    st.dataframe(pd.DataFrame(fp_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No first-pitch data available.")
        else:
            st.caption("Count data not available.")

    with fp_c2:
        section_header("Count Performance")
        if "Balls" in pdf.columns and "Strikes" in pdf.columns:
            pdf["_CountStr"] = pdf["Balls"].astype(int).astype(str) + "-" + pdf["Strikes"].astype(int).astype(str)
            count_rows = []
            for count_val, grp in pdf.groupby("_CountStr"):
                n = len(grp)
                if n < 2:
                    continue
                swings = grp[grp["PitchCall"].isin(SWING_CALLS)]
                whiffs = grp[grp["PitchCall"] == "StrikeSwinging"]
                csw = grp["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).sum()
                count_rows.append({
                    "Count": count_val, "N": n,
                    "CSW%": f"{csw/n*100:.1f}",
                    "Whiff%": f"{len(whiffs)/max(len(swings),1)*100:.1f}" if len(swings) > 0 else "-",
                    "InPlay": int(grp[grp["PitchCall"] == "InPlay"].shape[0]),
                })
            if count_rows:
                st.dataframe(pd.DataFrame(count_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Count data not available.")

    # Velocity Trend by Inning
    section_header("Velocity Trend by Inning")
    if "Inning" in pdf.columns and "RelSpeed" in pdf.columns and "TaggedPitchType" in pdf.columns:
        velo_df = pdf.dropna(subset=["Inning", "RelSpeed"])
        if len(velo_df) > 3:
            fig_velo = go.Figure()
            for pt in sorted(velo_df["TaggedPitchType"].unique()):
                sub = velo_df[velo_df["TaggedPitchType"] == pt]
                inn_avg = sub.groupby("Inning")["RelSpeed"].mean().reset_index()
                color = PITCH_COLORS.get(pt, "#aaa")
                fig_velo.add_trace(go.Scatter(
                    x=inn_avg["Inning"], y=inn_avg["RelSpeed"],
                    mode="lines+markers", name=pt,
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                ))
            fig_velo.update_layout(
                xaxis_title="Inning", yaxis_title="Avg Velocity (mph)",
                xaxis=dict(dtick=1),
                height=320, plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color="#000000", family="Inter, Arial, sans-serif"),
                margin=dict(l=50, r=10, t=25, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
            )
            st.plotly_chart(fig_velo, use_container_width=True, key=f"pg_pit_velo_{slug}")
        else:
            st.caption("Not enough data for velocity trend.")

    # Best Pitch Sequences (single-game analysis)
    section_header("Best Pitch Sequences")
    if "TaggedPitchType" in pdf.columns and "PitchCall" in pdf.columns:
        # Build consecutive pitch pairs from actual game data
        sort_cols_seq = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in pdf.columns]
        pdf_sorted = pdf.sort_values(sort_cols_seq) if sort_cols_seq else pdf
        pa_id_cols = [c for c in ["Batter", "Inning", "PAofInning"] if c in pdf.columns]
        pair_rows = []
        if len(pa_id_cols) >= 2:
            for _, ab in pdf_sorted.groupby(pa_id_cols):
                ab = ab.sort_values(sort_cols_seq) if sort_cols_seq else ab
                types = ab["TaggedPitchType"].tolist()
                calls = ab["PitchCall"].tolist()
                for i in range(len(types) - 1):
                    pair_rows.append({
                        "Setup": types[i], "Follow": types[i + 1],
                        "FollowCall": calls[i + 1],
                    })
        if pair_rows:
            pairs_df = pd.DataFrame(pair_rows)
            seq_summary = []
            for (setup, follow), grp in pairs_df.groupby(["Setup", "Follow"]):
                n = len(grp)
                if n < 3:
                    continue
                whiffs = (grp["FollowCall"] == "StrikeSwinging").sum()
                csw = grp["FollowCall"].isin(["StrikeCalled", "StrikeSwinging"]).sum()
                seq_summary.append({
                    "Sequence": f"{setup} → {follow}",
                    "N": n,
                    "Whiff%": round(whiffs / n * 100, 1),
                    "CSW%": round(csw / n * 100, 1),
                })
            if seq_summary:
                seq_df = pd.DataFrame(seq_summary).sort_values("CSW%", ascending=False)
                st.dataframe(seq_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Not enough repeated sequences (min 3) in this outing.")
        else:
            st.caption("Could not identify pitch sequences.")

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
                ab_c1, ab_c2 = st.columns([1, 2])
                with ab_c1:
                    fig_ab = _pg_mini_location_plot(ab)
                    if fig_ab:
                        ab_key = f"pg_pit_kab_{slug}_{inn}_{_pg_slug(str(batter))}"
                        st.plotly_chart(fig_ab, use_container_width=True, key=ab_key)
                with ab_c2:
                    st.caption(_pg_pitch_sequence_text(ab))


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

    # Full width — Pitch Locations Seen
    section_header("Pitch Locations Seen")
    loc_all = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if not loc_all.empty and "TaggedPitchType" in loc_all.columns:
        pt_types = sorted(loc_all["TaggedPitchType"].unique())
        n_types = len(pt_types)
        if n_types > 0:
            loc_cols = st.columns(min(n_types, 4))
            for i, pt in enumerate(pt_types):
                with loc_cols[i % min(n_types, 4)]:
                    pt_locs = loc_all[loc_all["TaggedPitchType"] == pt]
                    color = PITCH_COLORS.get(pt, "#aaa")
                    fig_pt = go.Figure()
                    swung = pt_locs[pt_locs["PitchCall"].isin(SWING_CALLS)]
                    took = pt_locs[~pt_locs["PitchCall"].isin(SWING_CALLS)]
                    if not took.empty:
                        fig_pt.add_trace(go.Scatter(
                            x=took["PlateLocSide"], y=took["PlateLocHeight"],
                            mode="markers", marker=dict(size=7, color=color, opacity=0.5, symbol="circle"),
                            name="Took", showlegend=False,
                        ))
                    if not swung.empty:
                        fig_pt.add_trace(go.Scatter(
                            x=swung["PlateLocSide"], y=swung["PlateLocHeight"],
                            mode="markers", marker=dict(size=9, color=color, opacity=0.9, symbol="diamond",
                                                        line=dict(width=1, color="white")),
                            name="Swung", showlegend=False,
                        ))
                    add_strike_zone(fig_pt)
                    fig_pt.update_layout(
                        title=dict(text=f"{pt} ({len(pt_locs)})", font=dict(size=12)),
                        xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False,
                                   showticklabels=False, fixedrange=True, scaleanchor="y"),
                        yaxis=dict(range=[0, 5], showgrid=False, zeroline=False,
                                   showticklabels=False, fixedrange=True),
                        height=260, plot_bgcolor="white", paper_bgcolor="white",
                        margin=dict(l=5, r=5, t=30, b=5),
                    )
                    st.plotly_chart(fig_pt, use_container_width=True, key=f"pg_hit_ploc_{slug}_{pt}")
        st.caption("Circles = took, Diamonds = swung")
    else:
        st.caption("No pitch location data available.")

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
                ab_c1, ab_c2 = st.columns([1, 2])
                with ab_c1:
                    fig_ab = _pg_mini_location_plot(ab)
                    if fig_ab:
                        ab_key = f"pg_hit_kab_{slug}_{inn}_{_pg_slug(str(ab.iloc[0].get('Pitcher','')))}"
                        st.plotly_chart(fig_ab, use_container_width=True, key=ab_key)
                with ab_c2:
                    st.caption(_pg_pitch_sequence_text(ab))
        else:
            st.caption("No notable at-bats (HR, XBH, K, BB, or 6+ pitch PA).")
    else:
        st.caption("PA identification columns not available.")


# ── Grading helpers ──

def _score_linear(val, lo, hi):
    """Map a value linearly from [lo, hi] -> [0, 100], clamped."""
    if pd.isna(val):
        return None
    return float(np.clip((val - lo) / max(hi - lo, 1e-9) * 100, 0, 100))


def _render_grade_header(name, n_pitches, overall, label_extra="", small_sample=False):
    """Render player name + overall tier badge header."""
    if small_sample:
        tier = "Small Sample"
        badge_color = "#9e9e9e"
        badge_text = "Small Sample"
    elif overall is not None:
        tier = _tier_label(overall)
        badge_color = _tier_color(tier)
        badge_text = f"{_tier_icon(tier)} {tier}"
    else:
        tier = "N/A"
        badge_color = "#9e9e9e"
        badge_text = "N/A"
    extra = f" · {label_extra}" if label_extra else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;padding:8px 0;">'
        f'<span style="font-size:18px;font-weight:700;">{display_name(name)}</span>'
        f'<span style="display:inline-block;padding:3px 12px;border-radius:12px;'
        f'background:{badge_color};color:white;font-size:12px;font-weight:700;'
        f'letter-spacing:0.5px;">{badge_text}</span>'
        f'<span style="font-size:12px;color:#888;">{n_pitches} pitches{extra}</span>'
        f'</div>', unsafe_allow_html=True)


def _render_grade_cards(grades):
    """Render compact tier pill badges in a horizontal row."""
    if not grades:
        return
    pills_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;padding:4px 0;">'
    for cat, score in grades.items():
        if score is None:
            pills_html += (
                f'<div style="display:flex;align-items:center;gap:4px;padding:3px 10px;'
                f'border-radius:10px;background:#f0f0f0;border:1px solid #ddd;">'
                f'<span style="font-size:10px;color:#888;">{cat}</span>'
                f'<span style="font-size:10px;font-weight:700;color:#bbb;">N/A</span>'
                f'</div>'
            )
            continue
        tier = _tier_label(score)
        color = _tier_color(tier)
        icon = _tier_icon(tier)
        pills_html += (
            f'<div style="display:flex;align-items:center;gap:4px;padding:3px 10px;'
            f'border-radius:10px;background:{color}18;border:1px solid {color}40;">'
            f'<span style="font-size:10px;color:#555;">{cat}</span>'
            f'<span style="font-size:11px;font-weight:700;color:{color};">{icon}</span>'
            f'</div>'
        )
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)


def _render_radar_chart(grades, key_suffix=""):
    """Render a Scatterpolar radar chart with tier-based coloring."""
    valid = {k: v for k, v in grades.items() if v is not None}
    if len(valid) < 3:
        st.caption("Insufficient Data")
        return
    cats = list(valid.keys())
    vals = [valid[c] for c in cats]
    overall = np.mean(vals)
    tier = _tier_label(overall)
    tier_c = _tier_color(tier)
    # Create rgba fill from tier color
    _color_map = {
        "#2e7d32": ("rgba(46,125,50,0.20)", "#2e7d32"),
        "#f9a825": ("rgba(249,168,37,0.20)", "#f9a825"),
        "#c62828": ("rgba(198,40,40,0.20)", "#c62828"),
    }
    fill_color, line_color = _color_map.get(tier_c, ("rgba(158,158,158,0.20)", "#9e9e9e"))
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2),
        marker=dict(size=5, color=line_color),
        name="Performance",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[25, 50, 75, 100],
                            tickfont=dict(size=8), gridcolor="#e0e0e0"),
            angularaxis=dict(tickfont=dict(size=10, color="#333")),
            bgcolor="white",
        ),
        height=250, margin=dict(l=40, r=40, t=15, b=15),
        paper_bgcolor="white", showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"pg_radar_{key_suffix}")


def _render_stuff_cmd_bars(stuff_by_pt, cmd_df, key_suffix=""):
    """Render horizontal grouped bar chart of Stuff+ and Command+ by pitch type."""
    pitch_types = sorted(stuff_by_pt.keys())
    cmd_map = {}
    if cmd_df is not None and not cmd_df.empty and "Pitch" in cmd_df.columns and "Command+" in cmd_df.columns:
        cmd_map = dict(zip(cmd_df["Pitch"], cmd_df["Command+"]))
    if not pitch_types:
        return
    stuff_vals = [stuff_by_pt.get(pt, 100) for pt in pitch_types]
    cmd_vals = [cmd_map.get(pt, 100) for pt in pitch_types]

    def _bar_color(v):
        if v > 110:
            return "#d22d49"
        if v < 90:
            return "#2d7fc1"
        return "#9e9e9e"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=pitch_types, x=stuff_vals, orientation="h", name="Stuff+",
        marker=dict(color=[_bar_color(v) for v in stuff_vals]),
        text=[f"{v:.0f}" for v in stuff_vals], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        y=pitch_types, x=cmd_vals, orientation="h", name="Command+",
        marker=dict(color=[_bar_color(v) for v in cmd_vals], opacity=0.7),
        text=[f"{v:.0f}" for v in cmd_vals], textposition="outside",
    ))
    fig.update_layout(
        barmode="group", height=max(180, 40 * len(pitch_types) + 50),
        xaxis=dict(title="Score", range=[50, max(max(stuff_vals + cmd_vals, default=100) + 15, 130)]),
        yaxis=dict(title=""),
        margin=dict(l=10, r=40, t=20, b=30),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        font=dict(color="#000000", family="Inter, Arial, sans-serif"),
    )
    fig.add_vline(x=100, line_dash="dash", line_color="#aaa", line_width=1)
    st.plotly_chart(fig, use_container_width=True, key=f"pg_stuffcmd_{key_suffix}")


# ── Pitcher grading logic ──

def _compute_pitcher_grades(pdf, data, pitcher):
    """Compute 6 pitcher grade dimensions and context-aware feedback."""
    grades = {}
    feedback = []
    n_pitches = len(pdf)

    # Season data for this pitcher
    season_pdf = data[data["Pitcher"] == pitcher] if data is not None else pd.DataFrame()

    # --- Stuff ---
    stuff_df = _compute_stuff_plus(pdf)
    stuff_by_pt = {}
    if "StuffPlus" in stuff_df.columns:
        avg_stuff = stuff_df["StuffPlus"].mean()
        grades["Stuff"] = _score_linear(avg_stuff, 70, 130)
        for pt, grp in stuff_df.groupby("TaggedPitchType"):
            stuff_by_pt[pt] = grp["StuffPlus"].mean()
    else:
        grades["Stuff"] = None

    # Season Stuff+ comparison
    if not season_pdf.empty:
        season_stuff = _compute_stuff_plus(season_pdf)
        if "StuffPlus" in season_stuff.columns:
            season_avg_stuff = season_stuff["StuffPlus"].mean()
            diff = avg_stuff - season_avg_stuff
            if abs(diff) >= 2:
                direction = "above" if diff > 0 else "below"
                feedback.append(f"Stuff+ was {abs(diff):.0f} points {direction} season average ({season_avg_stuff:.0f}).")

    # --- Command ---
    cmd_df = _compute_command_plus(pdf, data)
    if not cmd_df.empty and "Command+" in cmd_df.columns:
        avg_cmd = cmd_df["Command+"].mean()
        grades["Command"] = _score_linear(avg_cmd, 80, 120)
    else:
        grades["Command"] = None

    # --- Deception (FPS% + count management) ---
    if "Balls" in pdf.columns and "Strikes" in pdf.columns:
        fps_pct = 50.0
        ahead_pct = 50.0
        fp = pdf[(pdf["Balls"] == 0) & (pdf["Strikes"] == 0)]
        if len(fp) > 0:
            fp_strikes = fp[fp["PitchCall"].isin(
                ["StrikeCalled", "StrikeSwinging", "FoulBall",
                 "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"])]
            fps_pct = len(fp_strikes) / len(fp) * 100
        # Pitches thrown when ahead (strikes > balls)
        count_df = pdf.dropna(subset=["Balls", "Strikes"])
        if len(count_df) > 0:
            ahead = count_df[count_df["Strikes"].astype(int) > count_df["Balls"].astype(int)]
            ahead_pct = len(ahead) / len(count_df) * 100
        deception_score = fps_pct * 0.6 + ahead_pct * 0.4
        grades["Deception"] = min(100, max(0, deception_score))
    else:
        grades["Deception"] = None

    # FPS feedback
    n_pas = 0
    if "Balls" in pdf.columns and "Strikes" in pdf.columns:
        fp = pdf[(pdf["Balls"] == 0) & (pdf["Strikes"] == 0)]
        n_pas = len(fp)
    if n_pas > 0:
        fp_strikes = fp[fp["PitchCall"].isin(
            ["StrikeCalled", "StrikeSwinging", "FoulBall",
             "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"])]
        fps_game = len(fp_strikes) / n_pas * 100
        season_fps = 50.0
        if not season_pdf.empty and "Balls" in season_pdf.columns and "Strikes" in season_pdf.columns:
            sfp = season_pdf[(season_pdf["Balls"] == 0) & (season_pdf["Strikes"] == 0)]
            if len(sfp) > 0:
                sfp_str = sfp[sfp["PitchCall"].isin(
                    ["StrikeCalled", "StrikeSwinging", "FoulBall",
                     "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"])]
                season_fps = len(sfp_str) / len(sfp) * 100
        context = "above" if fps_game > season_fps else "below"
        feedback.append(f"Got ahead in {len(fp_strikes)}/{n_pas} PAs ({fps_game:.0f}% FPS) — {context} season rate ({season_fps:.0f}%).")

    # --- Swing & Miss (CSW%) ---
    csw_pct = pdf["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
    grades["Swing & Miss"] = _score_linear(csw_pct, 15, 35)
    csw_tier = _tier_label(grades["Swing & Miss"])
    feedback.append(f"CSW% at {csw_pct:.1f}% ({csw_tier.lower()}).")

    # Best/worst pitch by CSW%
    if "TaggedPitchType" in pdf.columns:
        pt_csw = []
        for pt, grp in pdf.groupby("TaggedPitchType"):
            if len(grp) >= 5:
                c = grp["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
                pt_csw.append((pt, c, len(grp)))
        if pt_csw:
            pt_csw.sort(key=lambda x: x[1], reverse=True)
            best = pt_csw[0]
            feedback.append(f"Best pitch: {best[0]} at {best[1]:.0f}% CSW ({best[2]} thrown).")
            if len(pt_csw) > 1:
                worst = pt_csw[-1]
                if worst[1] < best[1] - 5:
                    feedback.append(f"Struggled with {worst[0]} ({worst[1]:.0f}% CSW).")

    # --- Pitch Execution (release point consistency) ---
    if "RelHeight" in pdf.columns and "RelSide" in pdf.columns:
        rel_df = pdf.dropna(subset=["RelHeight", "RelSide"])
        if len(rel_df) >= 5:
            combined_std = np.sqrt(rel_df["RelHeight"].std()**2 + rel_df["RelSide"].std()**2)
            # Lower std = better. Typical range: 0.05 (elite) to 0.3 (poor)
            grades["Execution"] = _score_linear(combined_std, 0.3, 0.05)  # inverted: lower is better
        else:
            grades["Execution"] = None
    else:
        grades["Execution"] = None

    # --- Results (K%, inverted BB%, inverted barrel%) ---
    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in pdf.columns]
    pas_faced = pdf.drop_duplicates(subset=pa_cols).shape[0] if len(pa_cols) >= 2 else max(n_pitches // 4, 1)
    has_korbb = "KorBB" in pdf.columns
    ks = len(pdf[pdf["KorBB"] == "Strikeout"]) if has_korbb else 0
    bbs = len(pdf[pdf["KorBB"] == "Walk"]) if has_korbb else 0
    k_pct = ks / max(pas_faced, 1) * 100
    bb_pct = bbs / max(pas_faced, 1) * 100
    bbe = pdf[(pdf["PitchCall"] == "InPlay") & pdf["ExitSpeed"].notna()] if "ExitSpeed" in pdf.columns else pd.DataFrame()
    k_score = _score_linear(k_pct, 10, 40) if has_korbb else None
    bb_score = _score_linear(bb_pct, 15, 0) if has_korbb else None  # inverted: lower BB% is better
    if len(bbe) > 0:
        barrel_pct = is_barrel_mask(bbe).mean() * 100
        barrel_score = _score_linear(barrel_pct, 15, 0)  # inverted
    else:
        barrel_score = None
    # Compute Results grade from available sub-scores (reweight proportionally)
    sub = [(k_score, 0.4), (bb_score, 0.3), (barrel_score, 0.3)]
    valid_sub = [(s, w) for s, w in sub if s is not None]
    if valid_sub:
        total_w = sum(w for _, w in valid_sub)
        grades["Results"] = sum(s * w for s, w in valid_sub) / total_w
    else:
        grades["Results"] = None

    # Velo trend feedback
    if "Inning" in pdf.columns and "RelSpeed" in pdf.columns:
        fb_pitches = pdf[pdf["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter"])].dropna(subset=["RelSpeed", "Inning"])
        if len(fb_pitches) >= 5:
            last_inn = fb_pitches["Inning"].max()
            peak_velo = fb_pitches.groupby("Inning")["RelSpeed"].mean().max()
            last_inn_velo = fb_pitches[fb_pitches["Inning"] == last_inn]["RelSpeed"].mean()
            if peak_velo - last_inn_velo > 2:
                feedback.append(f"Velo dropped {peak_velo - last_inn_velo:.1f} mph from peak in final inning — possible fatigue.")

    return grades, feedback, stuff_by_pt, cmd_df


def _compute_pitcher_percentile_metrics(pdf, season_pdf):
    """Compute game-day metrics vs season for Savant-style percentile bars."""
    metrics = []
    # Avg FB Velo
    fb = pdf[pdf["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter"])].dropna(subset=["RelSpeed"])
    game_fb_velo = fb["RelSpeed"].mean() if len(fb) > 0 else np.nan
    season_fb = season_pdf[season_pdf["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter"])].dropna(subset=["RelSpeed"]) if not season_pdf.empty else pd.DataFrame()
    fb_pctl = np.nan
    if not season_fb.empty and pd.notna(game_fb_velo):
        game_avgs = season_fb.groupby("GameID")["RelSpeed"].mean() if "GameID" in season_fb.columns else pd.Series(dtype=float)
        if len(game_avgs) >= 3:
            fb_pctl = percentileofscore(game_avgs.dropna(), game_fb_velo, kind="rank")
    metrics.append(("Avg FB Velo", game_fb_velo, fb_pctl, ".1f", True))

    # Whiff%
    swings = pdf[pdf["PitchCall"].isin(SWING_CALLS)]
    whiffs = pdf[pdf["PitchCall"] == "StrikeSwinging"]
    game_whiff = len(whiffs) / max(len(swings), 1) * 100 if len(swings) > 0 else np.nan
    whiff_pctl = np.nan
    if not season_pdf.empty and pd.notna(game_whiff) and "GameID" in season_pdf.columns:
        def _game_whiff(g):
            s = g[g["PitchCall"].isin(SWING_CALLS)]
            w = g[g["PitchCall"] == "StrikeSwinging"]
            return len(w) / max(len(s), 1) * 100 if len(s) > 0 else np.nan
        game_whiffs = season_pdf.groupby("GameID").apply(_game_whiff).dropna()
        if len(game_whiffs) >= 3:
            whiff_pctl = percentileofscore(game_whiffs, game_whiff, kind="rank")
    metrics.append(("Whiff%", game_whiff, whiff_pctl, ".1f", True))

    # CSW%
    game_csw = pdf["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
    csw_pctl = np.nan
    if not season_pdf.empty and "GameID" in season_pdf.columns:
        game_csws = season_pdf.groupby("GameID").apply(
            lambda g: g["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
        ).dropna()
        if len(game_csws) >= 3:
            csw_pctl = percentileofscore(game_csws, game_csw, kind="rank")
    metrics.append(("CSW%", game_csw, csw_pctl, ".1f", True))

    # Chase%
    loc_df = pdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    game_chase = np.nan
    if len(loc_df) > 0:
        iz = in_zone_mask(loc_df)
        oz = loc_df[~iz]
        if len(oz) > 0:
            oz_sw = oz[oz["PitchCall"].isin(SWING_CALLS)]
            game_chase = len(oz_sw) / len(oz) * 100
    chase_pctl = np.nan
    if not season_pdf.empty and pd.notna(game_chase) and "GameID" in season_pdf.columns:
        def _game_chase(g):
            l = g.dropna(subset=["PlateLocSide", "PlateLocHeight"])
            if len(l) == 0:
                return np.nan
            iz = in_zone_mask(l)
            o = l[~iz]
            if len(o) == 0:
                return np.nan
            return len(o[o["PitchCall"].isin(SWING_CALLS)]) / len(o) * 100
        game_chases = season_pdf.groupby("GameID").apply(_game_chase).dropna()
        if len(game_chases) >= 3:
            chase_pctl = percentileofscore(game_chases, game_chase, kind="rank")
    metrics.append(("Chase%", game_chase, chase_pctl, ".1f", True))

    # K%
    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in pdf.columns]
    pas_faced = pdf.drop_duplicates(subset=pa_cols).shape[0] if len(pa_cols) >= 2 else 1
    ks = len(pdf[pdf["KorBB"] == "Strikeout"]) if "KorBB" in pdf.columns else 0
    game_k_pct = ks / max(pas_faced, 1) * 100
    k_pctl = np.nan
    if not season_pdf.empty and "GameID" in season_pdf.columns and "KorBB" in season_pdf.columns:
        def _game_kpct(g):
            pc = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in g.columns]
            pa = g.drop_duplicates(subset=pc).shape[0] if len(pc) >= 2 else 1
            return len(g[g["KorBB"] == "Strikeout"]) / max(pa, 1) * 100
        game_kpcts = season_pdf.groupby("GameID").apply(_game_kpct).dropna()
        if len(game_kpcts) >= 3:
            k_pctl = percentileofscore(game_kpcts, game_k_pct, kind="rank")
    metrics.append(("K%", game_k_pct, k_pctl, ".1f", True))

    # BB%
    bbs = len(pdf[pdf["KorBB"] == "Walk"]) if "KorBB" in pdf.columns else 0
    game_bb_pct = bbs / max(pas_faced, 1) * 100
    bb_pctl = np.nan
    if not season_pdf.empty and "GameID" in season_pdf.columns and "KorBB" in season_pdf.columns:
        def _game_bbpct(g):
            pc = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in g.columns]
            pa = g.drop_duplicates(subset=pc).shape[0] if len(pc) >= 2 else 1
            return len(g[g["KorBB"] == "Walk"]) / max(pa, 1) * 100
        game_bbpcts = season_pdf.groupby("GameID").apply(_game_bbpct).dropna()
        if len(game_bbpcts) >= 3:
            bb_pctl = 100 - percentileofscore(game_bbpcts, game_bb_pct, kind="rank")
    metrics.append(("BB%", game_bb_pct, bb_pctl, ".1f", False))

    return metrics


# ── Hitter grading logic ──

def _compute_hitter_grades(bdf, data, batter):
    """Compute 6 hitter grade dimensions and personalized feedback."""
    grades = {}
    feedback = []
    n_pitches = len(bdf)

    season_bdf = data[data["Batter"] == batter] if data is not None else pd.DataFrame()
    loc_df = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])

    # --- Discipline (Chase% inverted + BB%) ---
    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in bdf.columns]
    n_pas = bdf.drop_duplicates(subset=pa_cols).shape[0] if len(pa_cols) >= 2 else max(n_pitches // 4, 1)
    if len(loc_df) > 0:
        chase_pct = 50.0
        bbs = len(bdf[bdf["KorBB"] == "Walk"]) if "KorBB" in bdf.columns else 0
        bb_pct = bbs / max(n_pas, 1) * 100
        bb_score = min(100, bb_pct * 5)  # 20% BB = 100
        iz = in_zone_mask(loc_df)
        out_zone_df = loc_df[~iz]
        if len(out_zone_df) > 0:
            oz_swings = out_zone_df[out_zone_df["PitchCall"].isin(SWING_CALLS)]
            chase_pct = len(oz_swings) / len(out_zone_df) * 100
        chase_score = min(100, max(0, 100 - chase_pct * 2.5))
        grades["Discipline"] = chase_score * 0.6 + bb_score * 0.4
        disc_tier = _tier_label(grades["Discipline"])
        feedback.append(f"Chase rate at {chase_pct:.0f}% ({disc_tier.lower()}).")
    else:
        grades["Discipline"] = None

    # --- Power (Avg EV, hard hit%, barrel%) ---
    in_play = bdf[bdf["PitchCall"] == "InPlay"]
    bbe_df = in_play.dropna(subset=["ExitSpeed"]) if "ExitSpeed" in bdf.columns else pd.DataFrame()
    if len(bbe_df) > 0:
        avg_ev = bbe_df["ExitSpeed"].mean()
        hh_pct = (bbe_df["ExitSpeed"] >= 95).mean() * 100
        barrel_pct = is_barrel_mask(bbe_df).mean() * 100 if "Angle" in bbe_df.columns else 0
        ev_score = _score_linear(avg_ev, 70, 95)
        hh_score = _score_linear(hh_pct, 10, 60)
        barrel_score = _score_linear(barrel_pct, 0, 25)
        grades["Power"] = ev_score * 0.4 + hh_score * 0.3 + barrel_score * 0.3
        pwr_tier = _tier_label(grades["Power"])
        feedback.append(f"Power: {avg_ev:.1f} mph avg EV, {hh_pct:.0f}% hard hit ({pwr_tier.lower()}).")
    else:
        grades["Power"] = None

    # --- Contact Quality (sweet spot% + solid+barrel%) ---
    if len(bbe_df) > 0 and "Angle" in bbe_df.columns:
        la = pd.to_numeric(bbe_df["Angle"], errors="coerce")
        sweet_spot = la.between(8, 32).mean() * 100
        solid = (bbe_df["ExitSpeed"] >= 85).mean() * 100
        sp_score = _score_linear(sweet_spot, 20, 60)
        solid_score = _score_linear(solid, 30, 80)
        grades["Contact Quality"] = sp_score * 0.5 + solid_score * 0.5
    else:
        grades["Contact Quality"] = None

    # --- Bat-to-Ball (whiff% inverted + zone contact%) ---
    swings = bdf[bdf["PitchCall"].isin(SWING_CALLS)]
    whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"]
    if len(swings) > 0:
        whiff_pct = len(whiffs) / len(swings) * 100
        whiff_score = min(100, max(0, 100 - whiff_pct * 3))
        zone_contact_pct = 50.0
        if len(loc_df) > 0:
            iz = in_zone_mask(loc_df)
            iz_df = loc_df[iz]
            iz_swings = iz_df[iz_df["PitchCall"].isin(SWING_CALLS)]
            iz_contacts = iz_df[iz_df["PitchCall"].isin(CONTACT_CALLS)]
            if len(iz_swings) > 0:
                zone_contact_pct = len(iz_contacts) / len(iz_swings) * 100
        zc_score = _score_linear(zone_contact_pct, 60, 95)
        grades["Bat-to-Ball"] = whiff_score * 0.6 + zc_score * 0.4
    else:
        grades["Bat-to-Ball"] = None

    # --- Zone Awareness (swing rate in top EV zones + take rate in worst whiff zones) ---
    season_loc = season_bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"]) if not season_bdf.empty else pd.DataFrame()
    za_score = None
    if len(season_loc) >= 30 and len(loc_df) > 0:
        h_edges = [-0.83, -0.28, 0.28, 0.83]
        v_edges = [1.5, 2.17, 2.83, 3.5]
        ev_zones = []
        whiff_zones = []
        for vi in range(3):
            for hi in range(3):
                zdf = season_loc[
                    (season_loc["PlateLocSide"] >= h_edges[hi]) &
                    (season_loc["PlateLocSide"] < h_edges[hi + 1]) &
                    (season_loc["PlateLocHeight"] >= v_edges[vi]) &
                    (season_loc["PlateLocHeight"] < v_edges[vi + 1])
                ]
                if "ExitSpeed" in zdf.columns:
                    bbe_z = zdf[(zdf["PitchCall"] == "InPlay") & zdf["ExitSpeed"].notna()]
                    if len(bbe_z) >= 3:
                        ev_zones.append((vi, hi, bbe_z["ExitSpeed"].mean()))
                sw_z = zdf[zdf["PitchCall"].isin(SWING_CALLS)]
                wh_z = zdf[zdf["PitchCall"] == "StrikeSwinging"]
                if len(sw_z) >= 5:
                    whiff_zones.append((vi, hi, len(wh_z) / len(sw_z) * 100))

        # Swing rate in best EV zones
        top_ev_swing_rate = 50.0
        if ev_zones:
            ev_zones.sort(key=lambda x: x[2], reverse=True)
            top_ev = ev_zones[:3]
            game_in_top = pd.DataFrame()
            for vi, hi, _ in top_ev:
                zm = (
                    (loc_df["PlateLocSide"] >= h_edges[hi]) &
                    (loc_df["PlateLocSide"] < h_edges[hi + 1]) &
                    (loc_df["PlateLocHeight"] >= v_edges[vi]) &
                    (loc_df["PlateLocHeight"] < v_edges[vi + 1])
                )
                game_in_top = pd.concat([game_in_top, loc_df[zm]])
            game_in_top = game_in_top.drop_duplicates()
            if len(game_in_top) > 0:
                top_swings = game_in_top[game_in_top["PitchCall"].isin(SWING_CALLS)]
                top_ev_swing_rate = len(top_swings) / len(game_in_top) * 100
                best_ev_val = ev_zones[0][2]
                feedback.append(f"Swung at {top_ev_swing_rate:.0f}% of pitches in best EV zones (season avg: {best_ev_val:.0f} mph).")

        # Take rate in worst whiff zones
        take_bad_rate = 50.0
        if whiff_zones:
            whiff_zones.sort(key=lambda x: x[2], reverse=True)
            worst_whiff = whiff_zones[:3]
            game_in_bad = pd.DataFrame()
            for vi, hi, _ in worst_whiff:
                zm = (
                    (loc_df["PlateLocSide"] >= h_edges[hi]) &
                    (loc_df["PlateLocSide"] < h_edges[hi + 1]) &
                    (loc_df["PlateLocHeight"] >= v_edges[vi]) &
                    (loc_df["PlateLocHeight"] < v_edges[vi + 1])
                )
                game_in_bad = pd.concat([game_in_bad, loc_df[zm]])
            game_in_bad = game_in_bad.drop_duplicates()
            if len(game_in_bad) > 0:
                took = game_in_bad[~game_in_bad["PitchCall"].isin(SWING_CALLS)]
                take_bad_rate = len(took) / len(game_in_bad) * 100

        ev_swing_score = _score_linear(top_ev_swing_rate, 30, 80)
        take_score = _score_linear(take_bad_rate, 30, 80)
        za_score = ev_swing_score * 0.6 + take_score * 0.4
    grades["Zone Awareness"] = za_score

    # --- Pitch Recognition (whiff% by pitch group) ---
    pr_score = None
    if "TaggedPitchType" in bdf.columns and len(swings) > 0:
        fb_types = {"Fastball", "Sinker", "Cutter"}
        brk_types = {"Slider", "Curveball", "Sweeper", "Knuckle Curve"}
        os_types = {"Changeup", "Splitter"}
        group_whiffs = {}
        for group_name, types in [("FB", fb_types), ("BRK", brk_types), ("OS", os_types)]:
            g_pitches = bdf[bdf["TaggedPitchType"].isin(types)]
            g_swings = g_pitches[g_pitches["PitchCall"].isin(SWING_CALLS)]
            g_whiffs = g_pitches[g_pitches["PitchCall"] == "StrikeSwinging"]
            if len(g_swings) >= 3:
                group_whiffs[group_name] = len(g_whiffs) / len(g_swings) * 100
        if group_whiffs:
            # Only average groups that have actual data (no hardcoded defaults)
            weights = {"FB": 1.0, "BRK": 1.2, "OS": 1.2}
            total_w = sum(weights[g] for g in group_whiffs)
            avg_whiff = sum(group_whiffs[g] * weights[g] for g in group_whiffs) / total_w
            pr_score = min(100, max(0, 100 - avg_whiff * 2.5))
    grades["Pitch Recognition"] = pr_score

    # Pitch type performance feedback
    if "TaggedPitchType" in bdf.columns and "ExitSpeed" in bdf.columns:
        pt_damage = []
        for pt, grp in bdf.groupby("TaggedPitchType"):
            ip = grp[(grp["PitchCall"] == "InPlay") & grp["ExitSpeed"].notna()]
            sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
            wh = grp[grp["PitchCall"] == "StrikeSwinging"]
            wh_pct = len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else np.nan
            ev_val = ip["ExitSpeed"].mean() if len(ip) >= 1 else np.nan
            if len(grp) >= 3:
                pt_damage.append((pt, ev_val, wh_pct, len(ip)))
        if pt_damage:
            # Best by EV
            with_ev = [x for x in pt_damage if pd.notna(x[1]) and x[3] >= 1]
            if with_ev:
                with_ev.sort(key=lambda x: x[1], reverse=True)
                best = with_ev[0]
                feedback.append(f"Did most damage on {best[0]} ({best[1]:.0f} mph avg EV).")
            # Worst by whiff
            with_whiff = [x for x in pt_damage if pd.notna(x[2]) and x[2] > 30]
            if with_whiff:
                with_whiff.sort(key=lambda x: x[2], reverse=True)
                worst = with_whiff[0]
                feedback.append(f"Struggled with {worst[0]} ({worst[2]:.0f}% whiff rate).")

    # Count performance feedback
    if "Balls" in bdf.columns and "Strikes" in bdf.columns:
        count_cats = {"Ahead": [], "Even": [], "Behind": []}
        for _, row in bdf.iterrows():
            state = _pg_count_state(row.get("Balls"), row.get("Strikes"))
            count_cats[state].append(row)
        best_state = None
        best_ev = 0
        for state, rows in count_cats.items():
            rdf = pd.DataFrame(rows)
            if len(rdf) < 3:
                continue
            ip = rdf[(rdf["PitchCall"] == "InPlay") & rdf["ExitSpeed"].notna()] if "ExitSpeed" in rdf.columns else pd.DataFrame()
            if len(ip) > 0:
                ev_val = ip["ExitSpeed"].mean()
                if ev_val > best_ev:
                    best_ev = ev_val
                    best_state = state
        if best_state and best_ev > 0:
            feedback.append(f"Best results in {best_state.lower()} counts ({best_ev:.0f} mph avg EV).")

    # Approach assessment vs season
    if not season_bdf.empty and len(loc_df) > 0:
        season_loc_s = season_bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if len(season_loc_s) >= 30:
            season_swings = season_loc_s[season_loc_s["PitchCall"].isin(SWING_CALLS)]
            season_swing_pct = len(season_swings) / len(season_loc_s) * 100
            game_swings = loc_df[loc_df["PitchCall"].isin(SWING_CALLS)]
            game_swing_pct = len(game_swings) / len(loc_df) * 100
            diff = game_swing_pct - season_swing_pct
            if diff > 5:
                feedback.append(f"More aggressive than season norms (+{diff:.0f}% swing rate).")
            elif diff < -5:
                feedback.append(f"More passive than season norms ({diff:.0f}% swing rate).")

    return grades, feedback


def _compute_hitter_percentile_metrics(bdf, season_bdf):
    """Compute game-day metrics vs season for Savant-style percentile bars."""
    metrics = []
    in_play = bdf[bdf["PitchCall"] == "InPlay"]
    bbe_df = in_play.dropna(subset=["ExitSpeed"]) if "ExitSpeed" in bdf.columns else pd.DataFrame()

    # Avg EV
    game_ev = bbe_df["ExitSpeed"].mean() if len(bbe_df) > 0 else np.nan
    ev_pctl = np.nan
    if not season_bdf.empty and pd.notna(game_ev) and "GameID" in season_bdf.columns:
        def _game_ev(g):
            ip = g[(g["PitchCall"] == "InPlay") & g["ExitSpeed"].notna()]
            return ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan
        game_evs = season_bdf.groupby("GameID").apply(_game_ev).dropna()
        if len(game_evs) >= 3:
            ev_pctl = percentileofscore(game_evs, game_ev, kind="rank")
    metrics.append(("Avg EV", game_ev, ev_pctl, ".1f", True))

    # Whiff%
    swings = bdf[bdf["PitchCall"].isin(SWING_CALLS)]
    whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"]
    game_whiff = len(whiffs) / max(len(swings), 1) * 100 if len(swings) > 0 else np.nan
    whiff_pctl = np.nan
    if not season_bdf.empty and pd.notna(game_whiff) and "GameID" in season_bdf.columns:
        def _game_whiff_b(g):
            s = g[g["PitchCall"].isin(SWING_CALLS)]
            w = g[g["PitchCall"] == "StrikeSwinging"]
            return len(w) / max(len(s), 1) * 100 if len(s) > 0 else np.nan
        game_whiffs = season_bdf.groupby("GameID").apply(_game_whiff_b).dropna()
        if len(game_whiffs) >= 3:
            whiff_pctl = 100 - percentileofscore(game_whiffs, game_whiff, kind="rank")
    metrics.append(("Whiff%", game_whiff, whiff_pctl, ".1f", False))

    # Chase%
    loc_df = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    game_chase = np.nan
    if len(loc_df) > 0:
        iz = in_zone_mask(loc_df)
        oz = loc_df[~iz]
        if len(oz) > 0:
            oz_sw = oz[oz["PitchCall"].isin(SWING_CALLS)]
            game_chase = len(oz_sw) / len(oz) * 100
    chase_pctl = np.nan
    if not season_bdf.empty and pd.notna(game_chase) and "GameID" in season_bdf.columns:
        def _game_chase_b(g):
            l = g.dropna(subset=["PlateLocSide", "PlateLocHeight"])
            if len(l) == 0:
                return np.nan
            iz = in_zone_mask(l)
            o = l[~iz]
            if len(o) == 0:
                return np.nan
            return len(o[o["PitchCall"].isin(SWING_CALLS)]) / len(o) * 100
        game_chases = season_bdf.groupby("GameID").apply(_game_chase_b).dropna()
        if len(game_chases) >= 3:
            chase_pctl = 100 - percentileofscore(game_chases, game_chase, kind="rank")
    metrics.append(("Chase%", game_chase, chase_pctl, ".1f", False))

    # Hard Hit%
    game_hh = (bbe_df["ExitSpeed"] >= 95).mean() * 100 if len(bbe_df) > 0 else np.nan
    hh_pctl = np.nan
    if not season_bdf.empty and pd.notna(game_hh) and "GameID" in season_bdf.columns:
        def _game_hh(g):
            ip = g[(g["PitchCall"] == "InPlay") & g["ExitSpeed"].notna()]
            return (ip["ExitSpeed"] >= 95).mean() * 100 if len(ip) > 0 else np.nan
        game_hhs = season_bdf.groupby("GameID").apply(_game_hh).dropna()
        if len(game_hhs) >= 3:
            hh_pctl = percentileofscore(game_hhs, game_hh, kind="rank")
    metrics.append(("Hard Hit%", game_hh, hh_pctl, ".1f", True))

    # Barrel%
    game_barrel = is_barrel_mask(bbe_df).mean() * 100 if len(bbe_df) > 0 and "Angle" in bbe_df.columns else np.nan
    barrel_pctl = np.nan
    if not season_bdf.empty and pd.notna(game_barrel) and "GameID" in season_bdf.columns:
        def _game_barrel(g):
            ip = g[(g["PitchCall"] == "InPlay") & g["ExitSpeed"].notna()]
            return is_barrel_mask(ip).mean() * 100 if len(ip) > 0 and "Angle" in ip.columns else np.nan
        game_barrels = season_bdf.groupby("GameID").apply(_game_barrel).dropna()
        if len(game_barrels) >= 3:
            barrel_pctl = percentileofscore(game_barrels, game_barrel, kind="rank")
    metrics.append(("Barrel%", game_barrel, barrel_pctl, ".1f", True))

    return metrics


# ── At-bat grading (Phase 4) ──

def _grade_at_bat(ab_df, season_bdf):
    """Grade a single plate appearance. Returns (score, letter, result_text)."""
    score = 50  # base
    result_text = ""

    # Determine result
    play_result = ab_df["PlayResult"].iloc[-1] if "PlayResult" in ab_df.columns else ""
    kor_bb = ab_df["KorBB"].iloc[-1] if "KorBB" in ab_df.columns else ""
    if pd.isna(play_result):
        play_result = ""
    if pd.isna(kor_bb):
        kor_bb = ""

    if play_result == "HomeRun":
        score += 30
        result_text = "HR"
    elif play_result in ("Double", "Triple"):
        score += 25
        result_text = play_result
    elif play_result == "Single":
        score += 15
        result_text = "1B"
    elif kor_bb == "Walk":
        score += 15
        result_text = "BB"
    elif kor_bb == "Strikeout":
        result_text = "K"
    elif play_result == "Out" or play_result == "FieldersChoice":
        result_text = "Out"
    elif play_result == "Sacrifice":
        result_text = "Sac"
        score += 5
    elif ab_df["PitchCall"].iloc[-1] == "InPlay":
        result_text = "IP"
    else:
        result_text = play_result or "?"

    # Bonuses
    n_pitches = len(ab_df)
    if n_pitches >= 6:
        score += 15  # long PA

    # Quality contact
    ip = ab_df[(ab_df["PitchCall"] == "InPlay") & ab_df["ExitSpeed"].notna()] if "ExitSpeed" in ab_df.columns else pd.DataFrame()
    if len(ip) > 0:
        max_ev = ip["ExitSpeed"].max()
        if max_ev >= 95:
            score += 15
        elif max_ev < 70:
            score -= 10  # weak contact

        # Barrel bonus
        if "Angle" in ip.columns and is_barrel_mask(ip).any():
            score += 20

        # Popup penalty
        if "TaggedHitType" in ip.columns and (ip["TaggedHitType"] == "Popup").any():
            score -= 10

    # K on <= 3 pitches penalty
    if kor_bb == "Strikeout" and n_pitches <= 3:
        score -= 20

    # Chased first pitch out of zone penalty
    if len(ab_df) >= 1:
        first_pitch = ab_df.iloc[0]
        if (pd.notna(first_pitch.get("PlateLocSide")) and pd.notna(first_pitch.get("PlateLocHeight"))):
            fp_df = ab_df.iloc[:1].copy()
            fp_iz = in_zone_mask(fp_df)
            if not fp_iz.iloc[0] and first_pitch.get("PitchCall") in SWING_CALLS:
                score -= 15

    # Zone awareness bonus: swing in top damage zone and contact
    if not season_bdf.empty and len(ip) > 0:
        season_loc = season_bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if len(season_loc) >= 30:
            h_edges = [-0.83, -0.28, 0.28, 0.83]
            v_edges = [1.5, 2.17, 2.83, 3.5]
            ev_zones = []
            for vi in range(3):
                for hi in range(3):
                    zdf = season_loc[
                        (season_loc["PlateLocSide"] >= h_edges[hi]) &
                        (season_loc["PlateLocSide"] < h_edges[hi + 1]) &
                        (season_loc["PlateLocHeight"] >= v_edges[vi]) &
                        (season_loc["PlateLocHeight"] < v_edges[vi + 1])
                    ]
                    if "ExitSpeed" in zdf.columns:
                        bbe_z = zdf[(zdf["PitchCall"] == "InPlay") & zdf["ExitSpeed"].notna()]
                        if len(bbe_z) >= 3:
                            ev_zones.append((vi, hi, bbe_z["ExitSpeed"].mean()))
            if ev_zones:
                ev_zones.sort(key=lambda x: x[2], reverse=True)
                top3 = ev_zones[:3]
                for _, row in ip.iterrows():
                    ps, ph = row.get("PlateLocSide"), row.get("PlateLocHeight")
                    if pd.isna(ps) or pd.isna(ph):
                        continue
                    for vi, hi, _ in top3:
                        if (h_edges[hi] <= ps < h_edges[hi + 1] and
                                v_edges[vi] <= ph < v_edges[vi + 1]):
                            score += 10
                            break
                    break  # only check first contact

    score = max(0, min(100, score))
    return score, _letter_grade(score), result_text


# ── Coach Takeaways (Phase 5) ──

def _compute_takeaways(gd, data):
    """Pure computation — returns (pitching_bullets, hitting_bullets)."""
    dav_pitching = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID]
    dav_hitting = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID]

    pitching_bullets = []
    hitting_bullets = []

    # Pitching takeaways
    if not dav_pitching.empty:
        # FB velo summary
        fb = dav_pitching[dav_pitching["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter"])].dropna(subset=["RelSpeed"])
        if len(fb) > 0:
            avg_v = fb["RelSpeed"].mean()
            max_v = fb["RelSpeed"].max()
            season_pit = data[data["PitcherTeam"] == DAVIDSON_TEAM_ID] if data is not None else pd.DataFrame()
            season_fb = season_pit[season_pit["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter"])].dropna(subset=["RelSpeed"]) if not season_pit.empty else pd.DataFrame()
            if len(season_fb) > 0:
                season_avg = season_fb["RelSpeed"].mean()
                diff = avg_v - season_avg
                comp = f" ({diff:+.1f} vs season avg)" if abs(diff) >= 0.5 else ""
            else:
                comp = ""
            pitching_bullets.append(f"FB velocity: {avg_v:.1f} avg / {max_v:.1f} max{comp}")

        # K/BB ratio
        pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in dav_pitching.columns]
        pas = dav_pitching.drop_duplicates(subset=pa_cols).shape[0] if len(pa_cols) >= 2 else 1
        ks = len(dav_pitching[dav_pitching["KorBB"] == "Strikeout"]) if "KorBB" in dav_pitching.columns else 0
        bbs = len(dav_pitching[dav_pitching["KorBB"] == "Walk"]) if "KorBB" in dav_pitching.columns else 0
        ratio = f"{ks}/{bbs}" if bbs > 0 else f"{ks}/0"
        pitching_bullets.append(f"K/BB: {ratio} ({ks/max(pas,1)*100:.0f}% K rate, {bbs/max(pas,1)*100:.0f}% BB rate)")

        # Best/worst pitch overall by CSW%
        if "TaggedPitchType" in dav_pitching.columns:
            pt_csw = []
            for pt, grp in dav_pitching.groupby("TaggedPitchType"):
                if len(grp) >= 10:
                    c = grp["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
                    pt_csw.append((pt, c))
            if pt_csw:
                pt_csw.sort(key=lambda x: x[1], reverse=True)
                pitching_bullets.append(f"Best pitch by CSW%: {pt_csw[0][0]} ({pt_csw[0][1]:.0f}%)")
                if len(pt_csw) > 1:
                    pitching_bullets.append(f"Weakest pitch: {pt_csw[-1][0]} ({pt_csw[-1][1]:.0f}%)")

    # Hitting takeaways
    if not dav_hitting.empty:
        # Team batted ball quality
        bbe = dav_hitting[(dav_hitting["PitchCall"] == "InPlay") & dav_hitting["ExitSpeed"].notna()] if "ExitSpeed" in dav_hitting.columns else pd.DataFrame()
        if len(bbe) > 0:
            hh_pct = (bbe["ExitSpeed"] >= 95).mean() * 100
            barrel_pct = is_barrel_mask(bbe).mean() * 100 if "Angle" in bbe.columns else 0
            hitting_bullets.append(f"Team hard-hit rate: {hh_pct:.0f}% ({len(bbe)} BBE). Barrel rate: {barrel_pct:.0f}%.")

            # Best contact hitter
            best_ev = 0
            best_hitter = ""
            for batter, grp in bbe.groupby("Batter"):
                if len(grp) >= 2:
                    avg = grp["ExitSpeed"].mean()
                    if avg > best_ev:
                        best_ev = avg
                        best_hitter = batter
            if best_hitter:
                hitting_bullets.append(f"Best contact: {display_name(best_hitter)} ({best_ev:.1f} mph avg EV).")

        # Team chase rate
        loc_df = dav_hitting.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if len(loc_df) > 0:
            iz = in_zone_mask(loc_df)
            oz = loc_df[~iz]
            if len(oz) > 0:
                team_chase = len(oz[oz["PitchCall"].isin(SWING_CALLS)]) / len(oz) * 100
                assessment = "aggressive" if team_chase > 30 else "disciplined" if team_chase < 20 else "solid"
                hitting_bullets.append(f"Team chase rate: {team_chase:.0f}% — {assessment} approach.")

        # Clutch PAs: best PA grades with runners on / 2 outs
        pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in dav_hitting.columns]
        sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in dav_hitting.columns]
        if len(pa_cols) >= 2 and "Outs" in dav_hitting.columns:
            clutch_pas = []
            for pa_key, ab in dav_hitting.groupby(pa_cols[1:]):
                ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
                first_pitch = ab_sorted.iloc[0]
                outs_val = first_pitch.get("Outs", 0)
                runners = False
                for base_col in ["Runner1B", "Runner2B", "Runner3B"]:
                    if base_col in first_pitch.index and pd.notna(first_pitch.get(base_col)) and first_pitch.get(base_col) != "":
                        runners = True
                        break
                if runners or (pd.notna(outs_val) and int(outs_val) >= 2):
                    season_bdf = data[data["Batter"] == first_pitch.get("Batter")] if data is not None else pd.DataFrame()
                    score, letter, result = _grade_at_bat(ab_sorted, season_bdf)
                    if score >= 70:
                        clutch_pas.append((display_name(first_pitch.get("Batter", "?")), letter, result, score))
            if clutch_pas:
                clutch_pas.sort(key=lambda x: x[3], reverse=True)
                top = clutch_pas[0]
                hitting_bullets.append(f"Clutch: {top[0]} earned {top[1]} ({top[2]}) in high-leverage spot.")

    return pitching_bullets, hitting_bullets


def _generate_takeaways(gd, data):
    """Streamlit rendering wrapper."""
    section_header("Coach Takeaways")
    pitching_bullets, hitting_bullets = _compute_takeaways(gd, data)
    if pitching_bullets:
        st.markdown("**Pitching**")
        for b in pitching_bullets:
            st.caption(f"- {b}")
    if hitting_bullets:
        st.markdown("**Hitting**")
        for b in hitting_bullets:
            st.caption(f"- {b}")


def _split_feedback(feedback, grades):
    """Split feedback into Strengths and Areas to Improve based on content."""
    strengths = []
    areas = []
    for fb in feedback:
        # Heuristic: if feedback mentions struggle, low, dropped, below, poor -> areas to improve
        lower = fb.lower()
        if any(w in lower for w in ["struggled", "dropped", "below", "poor", "weak", "passive", "more aggressive"]):
            areas.append(fb)
        else:
            strengths.append(fb)
    return strengths, areas


# ── Grades & Feedback ──
def _postgame_grades(gd, data):
    """Render the Grades & Feedback tab — feedback-first layout with tier badges."""
    section_header("Grades & Feedback")

    dav_pitching = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID].copy()
    dav_hitting = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID].copy()

    # ── Pitcher Grades ──
    if not dav_pitching.empty:
        section_header("Pitcher Grades")
        pitchers = dav_pitching.groupby("Pitcher").size().sort_values(ascending=False).index.tolist()
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

            # 1. Tier Summary Banner
            _render_grade_header(pitcher, n_pitches, overall, f"~{ip_est} IP", small_sample=small_sample)

            # 2. Key Feedback (primary content — prominent)
            if feedback:
                fb_strengths, fb_areas = _split_feedback(feedback, grades)
                if fb_strengths:
                    st.markdown('**Strengths**')
                    for fb in fb_strengths:
                        st.markdown(f'<div style="font-size:14px;padding:2px 0 2px 12px;color:#2e7d32;">+ {fb}</div>', unsafe_allow_html=True)
                if fb_areas:
                    st.markdown('**Areas to Improve**')
                    for fb in fb_areas:
                        st.markdown(f'<div style="font-size:14px;padding:2px 0 2px 12px;color:#c62828;">- {fb}</div>', unsafe_allow_html=True)
                if not fb_strengths and not fb_areas:
                    for fb in feedback:
                        st.markdown(f'<div style="font-size:14px;padding:2px 0 2px 12px;">~ {fb}</div>', unsafe_allow_html=True)

            if not small_sample:
                # 3. Tier Breakdown — compact pills
                _render_grade_cards(grades)

            # 4. Percentile Bars
            pctl_metrics = _compute_pitcher_percentile_metrics(pdf, season_pdf)
            render_savant_percentile_section(pctl_metrics, title="Game vs Season Percentiles")

            # 5. Stuff+/Cmd+ Bars
            if stuff_by_pt:
                _render_stuff_cmd_bars(stuff_by_pt, cmd_df, key_suffix=f"pit_{slug}")

            # 6. Radar Chart — smaller, supporting visual
            if not small_sample:
                _render_radar_chart(grades, key_suffix=f"pit_{slug}")

            st.markdown("---")

    # ── Hitter Grades ──
    if not dav_hitting.empty:
        section_header("Hitter Grades")
        batters = dav_hitting.groupby("Batter").size().sort_values(ascending=False).index.tolist()
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

            # 1. Tier Summary Banner
            _render_grade_header(batter, n_pitches, overall, f"{n_pas} PA", small_sample=small_sample)

            # 2. Key Feedback (primary content — prominent)
            if feedback:
                fb_strengths, fb_areas = _split_feedback(feedback, grades)
                if fb_strengths:
                    st.markdown('**Strengths**')
                    for fb in fb_strengths:
                        st.markdown(f'<div style="font-size:14px;padding:2px 0 2px 12px;color:#2e7d32;">+ {fb}</div>', unsafe_allow_html=True)
                if fb_areas:
                    st.markdown('**Areas to Improve**')
                    for fb in fb_areas:
                        st.markdown(f'<div style="font-size:14px;padding:2px 0 2px 12px;color:#c62828;">- {fb}</div>', unsafe_allow_html=True)
                if not fb_strengths and not fb_areas:
                    for fb in feedback:
                        st.markdown(f'<div style="font-size:14px;padding:2px 0 2px 12px;">~ {fb}</div>', unsafe_allow_html=True)

            if not small_sample:
                # 3. Tier Breakdown — compact pills
                _render_grade_cards(grades)

            # 4. Percentile Bars
            pctl_metrics = _compute_hitter_percentile_metrics(bdf, season_bdf)
            render_savant_percentile_section(pctl_metrics, title="Game vs Season Percentiles")

            # 5. Radar Chart — smaller, supporting visual
            if not small_sample:
                _render_radar_chart(grades, key_suffix=f"hit_{slug}")

            # 6. At-Bat Grades Table
            sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in bdf.columns]
            if len(pa_cols) >= 2:
                ab_rows = []
                for pa_key, ab in bdf.groupby(pa_cols[1:]):
                    ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
                    score, letter, result = _grade_at_bat(ab_sorted, season_bdf)
                    inn = ab_sorted.iloc[0].get("Inning", "?")
                    vs_pitcher = display_name(ab_sorted.iloc[0]["Pitcher"]) if "Pitcher" in ab_sorted.columns else "?"
                    ab_rows.append({
                        "Inning": inn,
                        "vs Pitcher": vs_pitcher,
                        "Pitches": len(ab_sorted),
                        "Result": result,
                        "Score": score,
                        "Grade": letter,
                    })
                if ab_rows:
                    ab_table = pd.DataFrame(ab_rows).sort_values("Inning")
                    st.markdown("**At-Bat Grades**")
                    st.dataframe(ab_table, use_container_width=True, hide_index=True)

            st.markdown("---")

    # ── Coach Takeaways ──
    _generate_takeaways(gd, data)


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

    col_sel, col_btn = st.columns([5, 1])
    with col_sel:
        sel_game = st.selectbox("Select Game", games["GameID"].tolist(),
                                format_func=lambda g: game_labels.get(g, str(g)),
                                key="pg_game_select")

    gd = data[data["GameID"] == sel_game]
    if gd.empty:
        st.warning("No data for selected game.")
        return

    # PDF export: game label for filename / header
    _sel_row = games[games["GameID"] == sel_game].iloc[0] if sel_game in games["GameID"].values else None
    if _sel_row is not None:
        _dt = _sel_row["Date"].strftime("%Y-%m-%d") if pd.notna(_sel_row["Date"]) else "?"
        _away_name = _friendly_team_name(_sel_row['Away']) if 'Away' in _sel_row.index else _sel_row.get('Away', '?')
        _home_name = _friendly_team_name(_sel_row['Home']) if 'Home' in _sel_row.index else _sel_row.get('Home', '?')
        pdf_game_label = f"{_away_name} @ {_home_name}  |  {_dt}"
    else:
        pdf_game_label = sel_game

    with col_btn:
        if st.button("Export PDF", key="pg_gen_pdf"):
            with st.spinner("Building PDF report..."):
                from generate_postgame_report_pdf import generate_postgame_pdf_bytes
                st.session_state.pg_pdf_bytes = generate_postgame_pdf_bytes(gd, data, pdf_game_label)
                st.session_state.pg_pdf_game = sel_game

        if st.session_state.get("pg_pdf_bytes") and st.session_state.get("pg_pdf_game") == sel_game:
            st.download_button(
                "Download PDF", data=st.session_state.pg_pdf_bytes,
                file_name=f"postgame_{sel_game}.pdf", mime="application/pdf")

    _postgame_summary(gd)

    tab_ump, tab_pit, tab_hit, tab_grades = st.tabs(
        ["Umpire Report", "Pitcher Report", "Hitter Report", "Grades & Feedback"])
    with tab_ump:
        _postgame_umpire(gd)
    with tab_pit:
        _postgame_pitchers(gd, data)
    with tab_hit:
        _postgame_hitters(gd, data)
    with tab_grades:
        _postgame_grades(gd, data)
