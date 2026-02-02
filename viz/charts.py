"""Chart builders — spray chart, movement profile, pitch heatmap, strike zone helpers."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from config import ZONE_SIDE, ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP, PITCH_COLORS, filter_minor_pitches, display_name


def strike_zone_rect():
    return dict(type="rect", x0=-ZONE_SIDE, x1=ZONE_SIDE, y0=ZONE_HEIGHT_BOT, y1=ZONE_HEIGHT_TOP,
                line=dict(color="#333", width=2), fillcolor="rgba(0,0,0,0)")


def add_strike_zone(fig, label=True):
    fig.add_shape(strike_zone_rect())
    if label:
        fig.add_annotation(
            x=0, y=ZONE_HEIGHT_BOT - 0.18, text="Catcher's View",
            showarrow=False, font=dict(size=9, color="#999"),
            xanchor="center", yanchor="top",
        )
    return fig


def _add_grid_zone_outline(fig, color="#333", width=3):
    """Add a rectangular strike zone outline to a 5x5 categorical grid heatmap.

    The inner 3x3 (indices 1-3) is the strike zone.  Uses a simple rectangle
    matching the style of the continuous-axis heatmaps (contact rate, damage).
    """
    fig.add_shape(
        type="rect",
        x0=0.5, y0=0.5, x1=3.5, y1=3.5,
        line=dict(color=color, width=width),
        fillcolor="rgba(0,0,0,0)",
    )
    return fig


def make_spray_chart(in_play_df, height=360):
    spray = in_play_df.dropna(subset=["Direction", "Distance"]).copy()
    spray = spray[spray["Direction"].between(-90, 90)]
    if spray.empty:
        return None
    angle_rad = np.radians(spray["Direction"])
    spray["x"] = spray["Distance"] * np.sin(angle_rad)
    spray["y"] = spray["Distance"] * np.cos(angle_rad)

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

    # Infield
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

    # Hit type colors
    ht_colors = {"GroundBall": "#d62728", "LineDrive": "#2ca02c", "FlyBall": "#1f77b4", "Popup": "#ff7f0e"}
    ht_names = {"GroundBall": "GB", "LineDrive": "LD", "FlyBall": "FB", "Popup": "PU"}

    for ht in ["GroundBall", "LineDrive", "FlyBall", "Popup"]:
        sub = spray[spray["TaggedHitType"] == ht]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers",
            marker=dict(size=7, color=ht_colors.get(ht, "#777"), opacity=0.8,
                        line=dict(width=0.5, color="white")),
            name=ht_names.get(ht, ht),
            hovertemplate="EV: %{customdata[0]:.1f}<br>Dist: %{customdata[1]:.0f}ft<extra></extra>",
            customdata=sub[["ExitSpeed", "Distance"]].fillna(0).values,
        ))

    other = spray[~spray["TaggedHitType"].isin(["GroundBall", "LineDrive", "FlyBall", "Popup"])]
    if not other.empty:
        fig.add_trace(go.Scatter(x=other["x"], y=other["y"], mode="markers",
                                 marker=dict(size=5, color="#bbb", opacity=0.5), name="Other"))

    fig.update_layout(
        xaxis=dict(range=[-300, 300], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-15, 400], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", fixedrange=True),
        height=height, margin=dict(l=0, r=0, t=5, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#000000", family="Inter, Arial, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5,
                    font=dict(size=10, color="#000000"), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def player_header(name, jersey, pos, detail_line, stat_line):
    st.markdown(
        f'<div class="player-header-dark" style="background:linear-gradient(135deg,#000000,#1a1a1a,#000000);'
        f'padding:20px 28px;border-radius:10px;margin-bottom:16px;border:1px solid rgba(255,255,255,0.1);">'
        f'<div style="display:flex;align-items:center;gap:20px;">'
        f'<div class="ph-jersey" style="font-size:48px;font-weight:900;line-height:1;'
        f'font-family:Inter,sans-serif;text-shadow:0 2px 4px rgba(0,0,0,0.3);">#{jersey}</div>'
        f'<div>'
        f'<div class="ph-name" style="font-size:30px;font-weight:800;margin:0;'
        f'font-family:Inter,sans-serif;text-shadow:0 1px 3px rgba(0,0,0,0.3);">{display_name(name)}</div>'
        f'<div class="ph-detail" style="font-size:14px;margin-top:3px;font-family:Inter,sans-serif;">'
        f'{detail_line}</div>'
        f'<div class="ph-stat" style="font-size:13px;margin-top:2px;font-family:Inter,sans-serif;">'
        f'{stat_line}</div>'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )


def _safe_pr(pr, key):
    """Safely get a value from a player record Series, returning NaN if missing."""
    if pr is None:
        return np.nan
    try:
        v = pr[key]
        return v if pd.notna(v) else np.nan
    except (KeyError, TypeError, IndexError):
        return np.nan


def _safe_pop(all_stats, key):
    """Safely get a column from population stats DataFrame for percentile calc."""
    if all_stats is None or (isinstance(all_stats, pd.DataFrame) and all_stats.empty):
        return pd.Series(dtype=float)
    if key in all_stats.columns:
        return all_stats[key]
    return pd.Series(dtype=float)


def make_movement_profile(pdf, height=520):
    """Create a Baseball Savant-style movement profile with concentric circles."""
    mov = pdf.dropna(subset=["HorzBreak", "InducedVertBreak"])
    if mov.empty:
        return None
    # Filter to main pitches only
    mov = filter_minor_pitches(mov)
    if mov.empty:
        return None

    fig = go.Figure()

    # Concentric circles at 6, 12, 18, 24 inches
    for r in [6, 12, 18, 24]:
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=r * np.cos(theta), y=r * np.sin(theta), mode="lines",
            line=dict(color="#d0d8e0", width=1), showlegend=False, hoverinfo="skip",
        ))
        fig.add_annotation(x=0, y=r, text=f'{r}"', showarrow=False,
                           font=dict(size=9, color="#999"), yshift=8)

    # Crosshairs
    fig.add_hline(y=0, line_color="#bbb", line_width=1)
    fig.add_vline(x=0, line_color="#bbb", line_width=1)

    # Negate HorzBreak for Savant convention:
    # Trackman: positive HB = glove-side (toward 3B for RHP)
    # Savant chart: LEFT = toward 1B (arm-side), RIGHT = toward 3B (glove-side)
    # So we negate HB so that glove-side (positive trackman) plots LEFT (negative x)
    mov = mov.copy()
    mov["HB_plot"] = -mov["HorzBreak"]

    # Plot each pitch type as cluster
    pitch_types = sorted(mov["TaggedPitchType"].dropna().unique())
    for pt in pitch_types:
        sub = mov[mov["TaggedPitchType"] == pt]
        if len(sub) < 3:
            continue
        color = PITCH_COLORS.get(pt, "#aaa")
        fig.add_trace(go.Scatter(
            x=sub["HB_plot"], y=sub["InducedVertBreak"], mode="markers",
            marker=dict(size=7, color=color, opacity=0.7, line=dict(width=0.5, color="white")),
            name=pt,
            hovertemplate=f"{pt}<br>HB: %{{customdata:.1f}}\"<br>IVB: %{{y:.1f}}\"<extra></extra>",
            customdata=sub["HorzBreak"],
        ))

    # Axis labels — Savant convention: 1B ◄ MOVES TOWARD ► 3B
    fig.add_annotation(x=0, y=27, text="MORE RISE", showarrow=False,
                       font=dict(size=9, color="#666", family="Inter"), yshift=5)
    fig.add_annotation(x=0, y=-27, text="MORE DROP", showarrow=False,
                       font=dict(size=9, color="#666", family="Inter"), yshift=-5)
    fig.add_annotation(x=-27, y=0, text="1B SIDE", showarrow=False,
                       font=dict(size=9, color="#666", family="Inter"), xshift=-10)
    fig.add_annotation(x=27, y=0, text="3B SIDE", showarrow=False,
                       font=dict(size=9, color="#666", family="Inter"), xshift=10)
    # Direction arrow
    fig.add_annotation(x=0, y=29, text="1B ◄  MOVES TOWARD  ► 3B", showarrow=False,
                       font=dict(size=8, color="#999", family="Inter"), yshift=12)

    max_r = 28
    fig.update_layout(
        xaxis=dict(range=[-max_r, max_r], showgrid=False, zeroline=False,
                   showticklabels=False, title="", fixedrange=True, scaleanchor="y"),
        yaxis=dict(range=[-max_r, max_r], showgrid=False, zeroline=False,
                   showticklabels=False, title="", fixedrange=True),
        height=height, plot_bgcolor="#f0f4f8", paper_bgcolor="white",
        font=dict(color="#000000", family="Inter, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(size=11, color="#000000")),
    )
    return fig


def make_pitch_location_heatmap(pitch_data, title, color, height=380):
    """Create a Savant-style location heatmap with blue-white-red density."""
    loc = pitch_data.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if len(loc) < 3:
        return None

    fig = go.Figure()

    # Density contour — blue (sparse) → white → red (dense), matching Savant style
    fig.add_trace(go.Histogram2dContour(
        x=loc["PlateLocSide"], y=loc["PlateLocHeight"],
        colorscale=[
            (0.0, "rgba(255,255,255,0)"),
            (0.15, "rgba(173,203,227,0.6)"),
            (0.3, "rgba(120,170,210,0.7)"),
            (0.45, "rgba(180,200,220,0.6)"),
            (0.55, "rgba(230,220,220,0.6)"),
            (0.7, "rgba(230,180,180,0.7)"),
            (0.85, "rgba(220,120,120,0.8)"),
            (1.0, "rgba(200,60,60,0.9)"),
        ],
        showscale=False, ncontours=12,
        contours=dict(showlines=True, coloring="fill"),
        line=dict(width=0.5, color="rgba(150,150,150,0.3)"),
        hoverinfo="skip",
    ))

    # Strike zone outer box
    fig.add_shape(type="rect", x0=-ZONE_SIDE, x1=ZONE_SIDE, y0=ZONE_HEIGHT_BOT, y1=ZONE_HEIGHT_TOP,
                  line=dict(color="#333", width=2), fillcolor="rgba(0,0,0,0)")

    # Home plate
    plate_x = [-0.71, 0.71, 0.71, 0, -0.71, -0.71]
    plate_y = [0.15, 0.15, 0.0, -0.2, 0.0, 0.15]
    fig.add_trace(go.Scatter(x=plate_x, y=plate_y, mode="lines",
                             fill="toself", fillcolor="rgba(220,220,220,0.5)",
                             line=dict(color="#aaa", width=1.5), showlegend=False, hoverinfo="skip"))

    fig.update_layout(
        xaxis=dict(range=[-2.2, 2.2], showgrid=False, zeroline=False, showticklabels=False,
                   title="", fixedrange=True, scaleanchor="y"),
        yaxis=dict(range=[-0.5, 4.8], showgrid=False, zeroline=False, showticklabels=False,
                   title="", fixedrange=True),
        height=height, margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#000000", family="Inter, Arial, sans-serif"),
    )
    return fig
