"""Hitting pages — Hitter Card, Hitting Overview, Swing Decision Lab, Hitting Lab."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import percentileofscore

from config import (
    DAVIDSON_TEAM_ID, ROSTER_2026, JERSEY, POSITION, PITCH_COLORS,
    SWING_CALLS, CONTACT_CALLS, ZONE_SIDE, ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP,
    PLATE_SIDE_MAX, PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX,
    filter_davidson, filter_minor_pitches, normalize_pitch_types,
    in_zone_mask, is_barrel_mask, display_name, get_percentile, _is_position_player,
)
from data.loader import get_all_seasons, _load_truemedia, _tm_player, _safe_val, _safe_pct, _safe_num, _tm_pctile
from data.stats import compute_batter_stats, _build_batter_zones, compute_swing_path_metrics
from data.population import compute_batter_stats_pop
from viz.layout import CHART_LAYOUT, section_header
from viz.charts import (
    add_strike_zone, make_spray_chart, player_header, _safe_pr, _safe_pop,
    _add_grid_zone_outline,
)
from viz.percentiles import savant_color, render_savant_percentile_section
from analytics.expected import _compute_expected_outcomes, _create_zone_grid_data

# These are needed by functions extracted from app.py
from config import safe_mode, _SWING_CALLS_SQL
from data.loader import query_population


_CURRENT_BATS = None


def _set_current_bats(bats):
    global _CURRENT_BATS
    _CURRENT_BATS = bats


def _add_bats_badge(fig, bats):
    if fig is None:
        return fig
    if bats is None or (isinstance(bats, float) and pd.isna(bats)):
        return fig
    b = str(bats).strip()
    mapping = {"Right": "R", "Left": "L", "Switch": "S", "Both": "S", "R": "R", "L": "L", "S": "S"}
    label = mapping.get(b, b)
    if not label or label == "?":
        return fig
    fig.add_annotation(
        x=0.99, y=0.98, xref="paper", yref="paper",
        text=f"Bats: {label}",
        showarrow=False, xanchor="right", yanchor="top",
        font=dict(size=11, color="#444"),
    )
    return fig


def _plotly_chart_bats(fig, **kwargs):
    if fig is None:
        return
    _add_bats_badge(fig, _CURRENT_BATS)
    if "use_container_width" not in kwargs:
        kwargs["use_container_width"] = True
    st.plotly_chart(fig, **kwargs)


def _hitter_card_content(data, batter, season_filter, bdf, batted, pr, all_batter_stats):
    """Render a single-page Hitter Card — simple, actionable summary."""
    all_stats = all_batter_stats
    in_play = bdf[bdf["PitchCall"] == "InPlay"]
    swings = bdf[bdf["PitchCall"].isin(SWING_CALLS)]
    whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"]
    _bs = safe_mode(bdf["BatterSide"], "Right") if "BatterSide" in bdf.columns else "Right"

    # ── ROW 1: Percentile Rankings + Spray Chart ──
    pc_col1, pc_col2 = st.columns([1, 1], gap="medium")
    with pc_col1:
        batting_metrics = [
            ("Avg EV", _safe_pr(pr, "AvgEV"), get_percentile(_safe_pr(pr, "AvgEV"), _safe_pop(all_stats, "AvgEV")), ".1f", True),
            ("Max EV", _safe_pr(pr, "MaxEV"), get_percentile(_safe_pr(pr, "MaxEV"), _safe_pop(all_stats, "MaxEV")), ".1f", True),
            ("Barrel %", _safe_pr(pr, "BarrelPct"), get_percentile(_safe_pr(pr, "BarrelPct"), _safe_pop(all_stats, "BarrelPct")), ".1f", True),
            ("Hard Hit %", _safe_pr(pr, "HardHitPct"), get_percentile(_safe_pr(pr, "HardHitPct"), _safe_pop(all_stats, "HardHitPct")), ".1f", True),
            ("Sweet Spot %", _safe_pr(pr, "SweetSpotPct"), get_percentile(_safe_pr(pr, "SweetSpotPct"), _safe_pop(all_stats, "SweetSpotPct")), ".1f", True),
            ("K %", _safe_pr(pr, "KPct"), get_percentile(_safe_pr(pr, "KPct"), _safe_pop(all_stats, "KPct")), ".1f", False),
            ("BB %", _safe_pr(pr, "BBPct"), get_percentile(_safe_pr(pr, "BBPct"), _safe_pop(all_stats, "BBPct")), ".1f", True),
            ("Whiff %", _safe_pr(pr, "WhiffPct"), get_percentile(_safe_pr(pr, "WhiffPct"), _safe_pop(all_stats, "WhiffPct")), ".1f", False),
            ("Chase %", _safe_pr(pr, "ChasePct"), get_percentile(_safe_pr(pr, "ChasePct"), _safe_pop(all_stats, "ChasePct")), ".1f", False),
            ("Z-Contact %", _safe_pr(pr, "ZoneContactPct"), get_percentile(_safe_pr(pr, "ZoneContactPct"), _safe_pop(all_stats, "ZoneContactPct")), ".1f", True),
        ]
        render_savant_percentile_section(batting_metrics, "Percentile Rankings")
        st.caption(f"vs. {len(all_stats)} batters in database (min 50 PA)")

    with pc_col2:
        section_header("Spray Chart")
        fig_spray = make_spray_chart(in_play, height=420)
        if fig_spray:
            _plotly_chart_bats(fig_spray, use_container_width=True, key="hc_spray")
        else:
            st.info("No batted ball data.")

        # Batted ball profile cards underneath (like pitcher usage/velo cards)
        bb_items = [
            ("Pull", f"{pr['PullPct']:.0f}%" if not pd.isna(pr.get('PullPct')) else "-", "#e63946"),
            ("Center", f"{pr['StraightPct']:.0f}%" if not pd.isna(pr.get('StraightPct')) else "-", "#457b9d"),
            ("Oppo", f"{pr['OppoPct']:.0f}%" if not pd.isna(pr.get('OppoPct')) else "-", "#2a9d8f"),
            ("GB", f"{pr['GBPct']:.0f}%" if not pd.isna(pr.get('GBPct')) else "-", "#d62728"),
            ("FB/LD", f"{(pr.get('FBPct', 0) or 0) + (pr.get('LDPct', 0) or 0):.0f}%" if not pd.isna(pr.get('FBPct')) else "-", "#1f77b4"),
        ]
        bb_cols = st.columns(len(bb_items))
        for idx, (name, val, color) in enumerate(bb_items):
            with bb_cols[idx]:
                st.markdown(
                    f'<div style="text-align:center;padding:6px 4px;border-radius:6px;'
                    f'border-top:3px solid {color};background:#f9f9f9;">'
                    f'<div style="font-weight:bold;font-size:13px;color:{color};">{name}</div>'
                    f'<div style="font-size:12px;color:#555;">{val}</div>'
                    f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── ROW 2: Damage Heatmap + Whiff Density + Swing Probability ──
    section_header("Where They Whiff & Where They Crush")
    loc_data = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    whiff_loc = loc_data[loc_data["PitchCall"] == "StrikeSwinging"]
    contacts_loc = loc_data[loc_data["PitchCall"].isin(CONTACT_CALLS)]
    batted_loc = loc_data[(loc_data["PitchCall"] == "InPlay")].dropna(subset=["ExitSpeed"])

    loc_cols = st.columns(3)

    # 1) Damage Heatmap (Avg EV) — contour with barrel stars
    with loc_cols[0]:
        section_header("Damage Heatmap (Avg EV)")
        if len(batted_loc) >= 5:
            fig_dmg = go.Figure()
            fig_dmg.add_trace(go.Histogram2dContour(
                x=batted_loc["PlateLocSide"], y=batted_loc["PlateLocHeight"],
                z=batted_loc["ExitSpeed"], histfunc="avg",
                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                contours=dict(showlines=False), ncontours=12, showscale=True,
                colorbar=dict(title="Avg EV", len=0.8),
            ))
            barrel_loc = batted_loc[is_barrel_mask(batted_loc)]
            if not barrel_loc.empty:
                fig_dmg.add_trace(go.Scatter(
                    x=barrel_loc["PlateLocSide"], y=barrel_loc["PlateLocHeight"],
                    mode="markers", marker=dict(size=12, color="#d22d49", symbol="star",
                                                 line=dict(width=1, color="white")),
                    name="Barrels", hovertemplate="EV: %{customdata[0]:.1f}<extra></extra>",
                    customdata=barrel_loc[["ExitSpeed"]].values))
            add_strike_zone(fig_dmg)
            fig_dmg.update_layout(**CHART_LAYOUT, height=350,
                                   xaxis=dict(range=[-1.8, 1.8], title="Horizontal", scaleanchor="y"),
                                   yaxis=dict(range=[0.5, 4.5], title="Vertical"),
                                   legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10)))
            fig_dmg.update_xaxes(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
            fig_dmg.update_yaxes(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
            _plotly_chart_bats(fig_dmg, use_container_width=True, key="hc_damage")
        else:
            st.caption("Not enough batted ball data")
        st.caption(f"{len(batted_loc)} batted balls")

    # 2) Whiff Zone Map — 5x5 grid heatmap with whiff% per zone
    with loc_cols[1]:
        section_header("Whiff Zone Map")
        grid_whiff, annot_whiff, h_lbl, v_lbl = _create_zone_grid_data(bdf, metric="whiff_rate", batter_side=_bs)
        if not np.isnan(grid_whiff).all():
            fig_wz = go.Figure(data=go.Heatmap(
                z=grid_whiff, text=annot_whiff, texttemplate="%{text}",
                x=h_lbl, y=v_lbl,
                colorscale=[[0, "#2ca02c"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                zmin=0, zmax=60, showscale=True,
                colorbar=dict(title="Whiff%", len=0.8),
                textfont=dict(size=13, color="white"),
            ))
            _add_grid_zone_outline(fig_wz)
            fig_wz.update_layout(**CHART_LAYOUT, height=350)
            fig_wz.update_xaxes(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
            fig_wz.update_yaxes(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
            _plotly_chart_bats(fig_wz, use_container_width=True, key="hc_whiff_zone")
        else:
            st.caption("Not enough swing data for whiff zones")

    # 3) Swing Probability Contour — P(Swing) by location
    with loc_cols[2]:
        section_header("Swing Probability")
        all_with_loc = loc_data.copy()
        all_with_loc["is_swing"] = all_with_loc["PitchCall"].isin(SWING_CALLS).astype(int)
        if len(all_with_loc) >= 20:
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Histogram2dContour(
                x=all_with_loc["PlateLocSide"], y=all_with_loc["PlateLocHeight"],
                z=all_with_loc["is_swing"],
                histfunc="avg",
                colorscale=[[0, "#2ca02c"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                contours=dict(showlabels=True, labelfont=dict(size=10)),
                nbinsx=12, nbinsy=12,
                showscale=True,
                colorbar=dict(title="P(Swing)", len=0.8),
            ))
            add_strike_zone(fig_prob)
            fig_prob.update_layout(**CHART_LAYOUT, height=350,
                                    xaxis=dict(range=[-1.8, 1.8], title="Horizontal", scaleanchor="y"),
                                    yaxis=dict(range=[0.5, 4.5], title="Vertical"))
            fig_prob.update_xaxes(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
            fig_prob.update_yaxes(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
            _plotly_chart_bats(fig_prob, use_container_width=True, key="hc_swing_prob")
        else:
            st.caption("Not enough pitch data")
        st.caption(f"{len(all_with_loc)} pitches")

    st.markdown("---")

    # ── ROW 3: Best Count Performance + Pitch Type Performance ──
    col_counts, col_pitch = st.columns([1, 1], gap="medium")

    with col_counts:
        section_header("Count Performance")
        if "Balls" in bdf.columns and "Strikes" in bdf.columns:
            bdf_counts = bdf.dropna(subset=["Balls", "Strikes"]).copy()
            bdf_counts["Count"] = bdf_counts["Balls"].astype(int).astype(str) + "-" + bdf_counts["Strikes"].astype(int).astype(str)
            all_data_counts = data.dropna(subset=["Balls", "Strikes"]).copy()
            all_data_counts["Count"] = all_data_counts["Balls"].astype(int).astype(str) + "-" + all_data_counts["Strikes"].astype(int).astype(str)

            def _cs(sub):
                sw = sub[sub["PitchCall"].isin(SWING_CALLS)]
                wh = sub[sub["PitchCall"] == "StrikeSwinging"]
                ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                return {
                    "n": len(sub), "swings": len(sw),
                    "whiff_pct": len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else 0,
                    "ev": ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan,
                }

            db_avgs = {cnt: _cs(grp) for cnt, grp in all_data_counts.groupby("Count")}
            player_cs = {cnt: _cs(grp) for cnt, grp in bdf_counts.groupby("Count")}

            # Avg EV heatmap by count
            balls_range = [0, 1, 2, 3]
            strikes_range = [0, 1, 2]
            z_vals, hover_text, annotations = [], [], []
            for s in strikes_range:
                row_z, row_hover = [], []
                for b in balls_range:
                    cnt = f"{b}-{s}"
                    ps = player_cs.get(cnt, {})
                    db = db_avgs.get(cnt, {})
                    p_ev = ps.get("ev", np.nan)
                    db_ev = db.get("ev", np.nan)
                    n = ps.get("n", 0)
                    p_whiff = ps.get("whiff_pct", np.nan)
                    if n >= 3 and not pd.isna(p_ev):
                        diff = p_ev - db_ev if not pd.isna(db_ev) else 0
                        row_z.append(diff)
                        row_hover.append(
                            f"<b>{cnt}</b> ({n} pitches)<br>"
                            f"Avg EV: <b>{p_ev:.1f} mph</b><br>"
                            f"DB Avg: {db_ev:.1f} mph<br>"
                            f"Whiff%: {p_whiff:.1f}%"
                        )
                        annotations.append(dict(
                            x=balls_range.index(b), y=strikes_range.index(s),
                            text=f"<b>{p_ev:.0f}</b><br><span style='font-size:9px'>W: {p_whiff:.0f}%</span>",
                            showarrow=False, font=dict(size=13, color="white", family="Inter"),
                        ))
                    else:
                        row_z.append(np.nan)
                        row_hover.append(f"{cnt}<br>Not enough data")
                        annotations.append(dict(
                            x=balls_range.index(b), y=strikes_range.index(s),
                            text="—", showarrow=False, font=dict(size=13, color="#aaa"),
                        ))
                z_vals.append(row_z)
                hover_text.append(row_hover)

            fig = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=[f"{b} Balls" for b in balls_range],
                y=[f"{s} Strikes" for s in strikes_range],
                colorscale=[
                    [0.0, "#cc0000"], [0.35, "#e8a0a0"], [0.5, "#f0f0f0"],
                    [0.65, "#a0d4a0"], [1.0, "#1a7a1a"],
                ],
                zmid=0, showscale=False,
                hovertext=hover_text, hoverinfo="text",
                xgap=3, ygap=3,
            ))
            fig.update_layout(
                annotations=annotations,
                xaxis=dict(side="top", tickfont=dict(size=12, color="#000000")),
                yaxis=dict(tickfont=dict(size=12, color="#000000"), autorange="reversed"),
                height=250, margin=dict(l=80, r=20, t=40, b=20),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, Arial, sans-serif"),
            )
            _plotly_chart_bats(fig, use_container_width=True, key="hc_count_grid")
            st.caption("EV (bold) + Whiff% per count. Green = EV above DB avg, Red = below.")
        else:
            st.caption("Count data not available.")

    with col_pitch:
        section_header("Performance vs Pitch Types")
        pitch_groups = {
            "Fastball": ["Fastball", "Sinker", "Cutter"],
            "Breaking": ["Slider", "Curveball", "Sweeper", "Knuckle Curve"],
            "Offspeed": ["Changeup", "Splitter"],
        }
        pt_rows = []
        for group_name, types in pitch_groups.items():
            sub = bdf[bdf["TaggedPitchType"].isin(types)]
            if len(sub) < 3:
                continue
            sub_swings = sub[sub["PitchCall"].isin(SWING_CALLS)]
            sub_whiffs = sub[sub["PitchCall"] == "StrikeSwinging"]
            sub_ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            whiff_pct = len(sub_whiffs) / max(len(sub_swings), 1) * 100
            avg_ev = sub_ip["ExitSpeed"].mean() if len(sub_ip) > 0 else np.nan
            hard_pct = len(sub_ip[sub_ip["ExitSpeed"] >= 95]) / max(len(sub_ip), 1) * 100 if len(sub_ip) > 0 else np.nan
            pt_rows.append({
                "Pitch Group": group_name,
                "#": len(sub),
                "Seen %": round(len(sub) / len(bdf) * 100, 1),
                "Whiff%": round(whiff_pct, 1),
                "Avg EV": round(avg_ev, 1) if not pd.isna(avg_ev) else None,
                "Hard%": round(hard_pct, 1) if not pd.isna(hard_pct) else None,
            })
        if pt_rows:
            st.dataframe(pd.DataFrame(pt_rows), use_container_width=True, hide_index=True)

        # Detailed per-pitch breakdown
        bdf_filt = filter_minor_pitches(bdf)
        detail_rows = []
        for pt in sorted(bdf_filt["TaggedPitchType"].dropna().unique()):
            sub = bdf_filt[bdf_filt["TaggedPitchType"] == pt]
            if len(sub) < 3:
                continue
            sub_sw = sub[sub["PitchCall"].isin(SWING_CALLS)]
            sub_wh = sub[sub["PitchCall"] == "StrikeSwinging"]
            sub_ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            detail_rows.append({
                "Pitch": pt,
                "#": len(sub),
                "%": round(len(sub) / len(bdf) * 100, 1),
                "Whiff%": round(len(sub_wh) / max(len(sub_sw), 1) * 100, 1),
                "EV": round(sub_ip["ExitSpeed"].mean(), 1) if len(sub_ip) > 0 else None,
                "LA": round(sub_ip["Angle"].mean(), 1) if len(sub_ip) > 0 and sub_ip["Angle"].notna().any() else None,
            })
        if detail_rows:
            with st.expander("Detailed Pitch Type Breakdown"):
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── ROW 4: Swing Path Summary + EV vs LA Scatter ──
    col_swing, col_evla = st.columns([1, 1], gap="medium")

    with col_swing:
        section_header("Swing Path Profile")
        sp = compute_swing_path_metrics(bdf)
        if sp is not None:
            # Display as metric cards
            cards = [
                ("Attack Angle", f"{sp['attack_angle']:.1f}\u00b0", sp.get("swing_type", "")),
                ("Path Adjust", f"{sp['path_adjust']:.1f}\u00b0/ft", "Low = flat, High = adaptable"),
                ("Median LA", f"{sp['avg_la_all']:.1f}\u00b0", f"Hard-hit: {sp['attack_angle_raw']:.1f}\u00b0"),
            ]
            if not pd.isna(sp.get("bat_speed_avg", np.nan)):
                cards.append(("Bat Speed (est)", f"{sp['bat_speed_avg']:.1f} mph", "Top-25% EV proxy"))
            if sp.get("depth_label"):
                cards.append(("Contact Depth", sp["depth_label"], "Where bat meets ball"))

            for label, val, desc in cards:
                st.markdown(
                    f'<div style="display:flex;align-items:center;margin:4px 0;padding:8px 12px;'
                    f'background:#f9f9f9;border-radius:6px;border-left:3px solid #e63946;">'
                    f'<div style="flex:1;">'
                    f'<div style="font-size:11px;color:#888;text-transform:uppercase;">{label}</div>'
                    f'<div style="font-size:18px;font-weight:700;color:#1a1a2e;">{val}</div>'
                    f'</div>'
                    f'<div style="font-size:11px;color:#888;text-align:right;">{desc}</div>'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.info("Not enough batted ball data for swing path analysis (need 10+).")

    with col_evla:
        section_header("Exit Velo vs Launch Angle")
        ev_la = in_play.dropna(subset=["ExitSpeed", "Angle"])
        if not ev_la.empty:
            fig = px.scatter(
                ev_la, x="Angle", y="ExitSpeed", color="TaggedHitType",
                color_discrete_map={"GroundBall": "#d62728", "LineDrive": "#2ca02c",
                                    "FlyBall": "#1f77b4", "Popup": "#ff7f0e"},
                opacity=0.7, labels={"Angle": "Launch Angle", "ExitSpeed": "Exit Velo"},
            )
            _bz_ev = np.linspace(98, max(ev_la["ExitSpeed"].max() + 2, 105), 40)
            _bz_la_lo = np.clip(26 - 2 * (_bz_ev - 98), 8, 26)
            _bz_la_hi = np.clip(30 + 3 * (_bz_ev - 98), 30, 50)
            fig.add_trace(go.Scatter(
                x=np.concatenate([_bz_la_lo, _bz_la_hi[::-1]]),
                y=np.concatenate([_bz_ev, _bz_ev[::-1]]),
                fill="toself", fillcolor="rgba(230,57,70,0.06)",
                line=dict(color="#e63946", width=1.5, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_annotation(x=20, y=_bz_ev[-1], text="BARREL ZONE",
                               showarrow=False, font=dict(size=9, color="#e63946"))
            fig.update_layout(height=350, showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                          font=dict(size=9, color="#000000")),
                              **CHART_LAYOUT)
            _plotly_chart_bats(fig, use_container_width=True, key="hc_ev_la")
        else:
            st.info("No exit velo / launch angle data.")

    # ── ROW 5: Platoon Splits ──
    st.markdown("---")
    section_header("Platoon Splits")
    if "PitcherThrows" in bdf.columns:
        split_metrics = []
        for side, label in [("Right", "vs RHP"), ("Left", "vs LHP")]:
            side_df = bdf[bdf["PitcherThrows"] == side]
            if len(side_df) < 5:
                continue
            s_sw = side_df[side_df["PitchCall"].isin(SWING_CALLS)]
            s_wh = side_df[side_df["PitchCall"] == "StrikeSwinging"]
            s_ip = side_df[side_df["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            s_whiff = len(s_wh) / max(len(s_sw), 1) * 100
            s_ev = s_ip["ExitSpeed"].mean() if len(s_ip) > 0 else np.nan
            s_hard = len(s_ip[s_ip["ExitSpeed"] >= 95]) / max(len(s_ip), 1) * 100 if len(s_ip) > 0 else np.nan

            # Percentiles vs all batters in same split (via DuckDB)
            adf = query_population(f"""
                SELECT
                    Batter,
                    CASE WHEN SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) > 0
                        THEN SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) * 100.0
                             / SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END)
                        ELSE NULL END AS whiff,
                    AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS ev
                FROM trackman
                WHERE PitcherThrows = '{side}'
                GROUP BY Batter
                HAVING COUNT(*) >= 10
            """)
            w_pct = get_percentile(s_whiff, adf["whiff"]) if not adf.empty else 50
            e_pct = get_percentile(s_ev, adf["ev"]) if not adf.empty and not pd.isna(s_ev) else np.nan

            split_metrics.append((f"{label} Whiff%", s_whiff, w_pct, ".1f", False))
            if not pd.isna(e_pct):
                split_metrics.append((f"{label} Avg EV", s_ev, e_pct, ".1f", True))

        if split_metrics:
            render_savant_percentile_section(split_metrics, None)
            st.caption("Percentile vs. all batters in DB (min 10 pitches in that split)")
    else:
        st.caption("No pitcher-throws data available for splits.")


def _hitting_overview(data, batter, season_filter, bdf, batted, pr, all_batter_stats):
    """Content from the original Hitter Card, rendered inside the Overview tab."""
    in_play = bdf[bdf["PitchCall"] == "InPlay"]
    all_stats = all_batter_stats  # alias for brevity in this section

    # ── ROW 1: Percentile Rankings + Spray Chart + Batted Ball Stats ──
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        # ── PERCENTILE RANKINGS (Savant-style) ──
        batting_metrics = [
            ("Avg EV", _safe_pr(pr, "AvgEV"), get_percentile(_safe_pr(pr, "AvgEV"), _safe_pop(all_stats, "AvgEV")), ".1f", True),
            ("Max EV", _safe_pr(pr, "MaxEV"), get_percentile(_safe_pr(pr, "MaxEV"), _safe_pop(all_stats, "MaxEV")), ".1f", True),
            ("Barrel %", _safe_pr(pr, "BarrelPct"), get_percentile(_safe_pr(pr, "BarrelPct"), _safe_pop(all_stats, "BarrelPct")), ".1f", True),
            ("Hard Hit %", _safe_pr(pr, "HardHitPct"), get_percentile(_safe_pr(pr, "HardHitPct"), _safe_pop(all_stats, "HardHitPct")), ".1f", True),
            ("Sweet Spot %", _safe_pr(pr, "SweetSpotPct"), get_percentile(_safe_pr(pr, "SweetSpotPct"), _safe_pop(all_stats, "SweetSpotPct")), ".1f", True),
            ("Avg LA", _safe_pr(pr, "AvgLA"), get_percentile(_safe_pr(pr, "AvgLA"), _safe_pop(all_stats, "AvgLA")), ".1f", True),
            ("K %", _safe_pr(pr, "KPct"), get_percentile(_safe_pr(pr, "KPct"), _safe_pop(all_stats, "KPct")), ".1f", False),
            ("BB %", _safe_pr(pr, "BBPct"), get_percentile(_safe_pr(pr, "BBPct"), _safe_pop(all_stats, "BBPct")), ".1f", True),
            ("Whiff %", _safe_pr(pr, "WhiffPct"), get_percentile(_safe_pr(pr, "WhiffPct"), _safe_pop(all_stats, "WhiffPct")), ".1f", False),
            ("Chase %", _safe_pr(pr, "ChasePct"), get_percentile(_safe_pr(pr, "ChasePct"), _safe_pop(all_stats, "ChasePct")), ".1f", False),
        ]
        render_savant_percentile_section(batting_metrics, "Percentile Rankings")
        st.caption(f"vs. {len(all_stats)} batters in database (min 50 PA)")

    with col2:
        section_header("Hits Spray Chart")
        fig = make_spray_chart(in_play, height=420)
        if fig:
            _plotly_chart_bats(fig, use_container_width=True, key="hitter_spray")
        else:
            st.info("No batted ball data.")

    # ── ROW 2: Batting Stats Table + Quality of Contact ──
    st.markdown("---")
    col3, col4 = st.columns([1, 1], gap="medium")

    with col3:
        section_header("Statcast Batting Statistics")
        stats_df = pd.DataFrame([{
            "PA": int(_safe_pr(pr, "PA") or 0),
            "BBE": int(_safe_pr(pr, "BBE") or 0),
            "Barrels": int(_safe_pr(pr, "Barrels") or 0),
            "Barrel%": round(_safe_pr(pr, "BarrelPct"), 1) if not pd.isna(_safe_pr(pr, "BarrelPct")) else None,
            "Brl/PA": round(_safe_pr(pr, "BarrelPA"), 1) if not pd.isna(_safe_pr(pr, "BarrelPA")) else None,
            "Avg EV": round(_safe_pr(pr, "AvgEV"), 1) if not pd.isna(_safe_pr(pr, "AvgEV")) else None,
            "Max EV": round(_safe_pr(pr, "MaxEV"), 1) if not pd.isna(_safe_pr(pr, "MaxEV")) else None,
            "Avg LA": round(_safe_pr(pr, "AvgLA"), 1) if not pd.isna(_safe_pr(pr, "AvgLA")) else None,
            "Sweet%": round(_safe_pr(pr, "SweetSpotPct"), 1) if not pd.isna(_safe_pr(pr, "SweetSpotPct")) else None,
            "Hard%": round(_safe_pr(pr, "HardHitPct"), 1) if not pd.isna(_safe_pr(pr, "HardHitPct")) else None,
            "K%": round(_safe_pr(pr, "KPct") or 0, 1),
            "BB%": round(_safe_pr(pr, "BBPct") or 0, 1),
        }])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Batted Ball Profile
        section_header("Batted Ball Profile")
        bb_df = pd.DataFrame([{
            "GB%": round(_safe_pr(pr, "GBPct"), 1) if not pd.isna(_safe_pr(pr, "GBPct")) else None,
            "Air%": round(_safe_pr(pr, "AirPct"), 1) if not pd.isna(_safe_pr(pr, "AirPct")) else None,
            "FB%": round(_safe_pr(pr, "FBPct"), 1) if not pd.isna(_safe_pr(pr, "FBPct")) else None,
            "LD%": round(_safe_pr(pr, "LDPct"), 1) if not pd.isna(_safe_pr(pr, "LDPct")) else None,
            "PU%": round(_safe_pr(pr, "PUPct"), 1) if not pd.isna(_safe_pr(pr, "PUPct")) else None,
            "Pull%": round(_safe_pr(pr, "PullPct"), 1) if not pd.isna(_safe_pr(pr, "PullPct")) else None,
            "Cent%": round(_safe_pr(pr, "StraightPct"), 1) if not pd.isna(_safe_pr(pr, "StraightPct")) else None,
            "Oppo%": round(_safe_pr(pr, "OppoPct"), 1) if not pd.isna(_safe_pr(pr, "OppoPct")) else None,
        }])
        st.dataframe(bb_df, use_container_width=True, hide_index=True)

    with col4:
        section_header("Plate Discipline")
        disc_df = pd.DataFrame([{
            "Pitches": len(bdf),
            "Zone%": round(_safe_pr(pr, "ZonePct"), 1) if not pd.isna(_safe_pr(pr, "ZonePct")) else None,
            "Z-Swing%": round(_safe_pr(pr, "ZoneSwingPct"), 1) if not pd.isna(_safe_pr(pr, "ZoneSwingPct")) else None,
            "Z-Contact%": round(_safe_pr(pr, "ZoneContactPct"), 1) if not pd.isna(_safe_pr(pr, "ZoneContactPct")) else None,
            "Chase%": round(_safe_pr(pr, "ChasePct") or 0, 1),
            "Chase Ct%": round(_safe_pr(pr, "ChaseContact"), 1) if not pd.isna(_safe_pr(pr, "ChaseContact")) else None,
            "Swing%": round(_safe_pr(pr, "SwingPct") or 0, 1),
            "Whiff%": round(_safe_pr(pr, "WhiffPct") or 0, 1),
            "K%": round(_safe_pr(pr, "KPct") or 0, 1),
            "BB%": round(_safe_pr(pr, "BBPct") or 0, 1),
        }])
        st.dataframe(disc_df, use_container_width=True, hide_index=True)

        # Quality of Contact
        section_header("Quality of Contact")
        if len(batted) > 0:
            weak = len(batted[batted["ExitSpeed"] < 70]) / len(batted) * 100
            topped = len(batted[batted["Angle"] < -10]) / len(batted) * 100 if batted["Angle"].notna().any() else 0
            under = len(batted[batted["Angle"] > 40]) / len(batted) * 100 if batted["Angle"].notna().any() else 0
            flare = len(batted[(batted["ExitSpeed"].between(70, 88)) & (batted["Angle"].between(8, 32))]) / len(batted) * 100 if batted["Angle"].notna().any() else 0
            solid = len(batted[(batted["ExitSpeed"].between(89, 97)) & (batted["Angle"].between(8, 32))]) / len(batted) * 100 if batted["Angle"].notna().any() else 0
            barrel = _safe_pr(pr, "BarrelPct") if not pd.isna(_safe_pr(pr, "BarrelPct")) else 0
            qoc_df = pd.DataFrame([{
                "Weak%": round(weak, 1),
                "Topped%": round(topped, 1),
                "Under%": round(under, 1),
                "Flare%": round(flare, 1),
                "Solid%": round(solid, 1),
                "Barrel%": round(barrel, 1),
                "Brl/PA": round(_safe_pr(pr, "BarrelPA"), 1) if not pd.isna(_safe_pr(pr, "BarrelPA")) else None,
            }])
            st.dataframe(qoc_df, use_container_width=True, hide_index=True)

    # ── ROW 3: Rolling EV + EV vs LA + Swing Heatmap ──
    st.markdown("---")
    col5, col6, col7 = st.columns([1, 1, 1], gap="medium")

    with col5:
        section_header("Rolling Exit Velocity")
        batted_sorted = in_play.dropna(subset=["ExitSpeed"]).sort_values("Date")
        if len(batted_sorted) >= 5:
            w = min(15, len(batted_sorted))
            batted_sorted = batted_sorted.copy()
            batted_sorted["Roll"] = batted_sorted["ExitSpeed"].rolling(w, min_periods=3).mean()
            db_avg = _safe_pop(all_stats, "AvgEV").mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(batted_sorted))), y=batted_sorted["Roll"],
                mode="lines", line=dict(color="#e63946", width=2.5), showlegend=False,
                hovertemplate="BIP #%{x}<br>Rolling EV: %{y:.1f} mph<extra></extra>",
            ))
            fig.add_hline(y=db_avg, line_dash="dash", line_color="#9e9e9e",
                          annotation_text=f"DB Avg: {db_avg:.1f}",
                          annotation_position="bottom right",
                          annotation_font=dict(size=10, color="#666"))
            fig.update_layout(xaxis_title="Batted Ball #", yaxis_title="EV (mph)",
                              height=300, **CHART_LAYOUT)
            _plotly_chart_bats(fig, use_container_width=True, key="hitter_roll_ev")
        else:
            st.info("Not enough batted balls for rolling chart.")

    with col6:
        section_header("Exit Velo vs Launch Angle")
        ev_la = in_play.dropna(subset=["ExitSpeed", "Angle"])
        if not ev_la.empty:
            fig = px.scatter(
                ev_la, x="Angle", y="ExitSpeed", color="TaggedHitType",
                color_discrete_map={"GroundBall": "#d62728", "LineDrive": "#2ca02c",
                                    "FlyBall": "#1f77b4", "Popup": "#ff7f0e"},
                opacity=0.7, labels={"Angle": "Launch Angle", "ExitSpeed": "Exit Velo"},
            )
            # Barrel zone — trace actual curved boundary
            _bz_ev = np.linspace(98, max(ev_la["ExitSpeed"].max() + 2, 105), 40)
            _bz_la_lo = np.clip(26 - 2 * (_bz_ev - 98), 8, 26)
            _bz_la_hi = np.clip(30 + 3 * (_bz_ev - 98), 30, 50)
            fig.add_trace(go.Scatter(
                x=np.concatenate([_bz_la_lo, _bz_la_hi[::-1]]),
                y=np.concatenate([_bz_ev, _bz_ev[::-1]]),
                fill="toself", fillcolor="rgba(230,57,70,0.06)",
                line=dict(color="#e63946", width=1.5, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_annotation(x=20, y=_bz_ev[-1], text="BARREL ZONE",
                               showarrow=False, font=dict(size=9, color="#e63946"))
            fig.update_layout(height=300, showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                          font=dict(size=9, color="#000000")),
                              **CHART_LAYOUT)
            _plotly_chart_bats(fig, use_container_width=True, key="hitter_ev_la")
        else:
            st.info("No exit velo / launch angle data.")

    with col7:
        section_header("Swing Decisions (Heatmap)")
        loc = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        swing_loc = loc[loc["PitchCall"].isin(SWING_CALLS)]
        if not swing_loc.empty:
            fig = px.density_heatmap(
                swing_loc, x="PlateLocSide", y="PlateLocHeight",
                nbinsx=14, nbinsy=14, color_continuous_scale="YlOrRd",
            )
            add_strike_zone(fig)
            fig.update_layout(
                xaxis=dict(range=[-2.5, 2.5], scaleanchor="y", showticklabels=False, title=""),
                yaxis=dict(range=[0, 5], showticklabels=False, title=""),
                height=300, margin=dict(l=0, r=0, t=5, b=0),
                coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color="#000000", family="Inter, Arial, sans-serif"),
            )
            _plotly_chart_bats(fig, use_container_width=True, key="hitter_swing_hm")
        else:
            st.info("No swing location data.")

    # ── ROW 4: Pitch Tracking by Type ──
    st.markdown("---")
    section_header("Pitch Tracking (Performance vs Pitch Types)")
    pitch_groups = {
        "Fastball": ["Fastball", "Sinker", "Cutter"],
        "Breaking": ["Slider", "Curveball", "Sweeper", "Knuckle Curve"],
        "Offspeed": ["Changeup", "Splitter"],
    }

    pt_rows = []
    for group_name, types in pitch_groups.items():
        sub = bdf[bdf["TaggedPitchType"].isin(types)]
        if len(sub) < 3:
            continue
        sub_swings = sub[sub["PitchCall"].isin(SWING_CALLS)]
        sub_whiffs = sub[sub["PitchCall"] == "StrikeSwinging"]
        sub_ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        _pa_cols = ["GameID", "Inning", "PAofInning", "Batter"]
        if all(c in sub.columns for c in _pa_cols):
            sub_pa = sub.drop_duplicates(subset=_pa_cols).shape[0]
        else:
            sub_pa = int(sub["PitchofPA"].eq(1).sum())
        sub_ks = len(sub[sub["KorBB"] == "Strikeout"].drop_duplicates(subset=_pa_cols)) if all(c in sub.columns for c in _pa_cols) else len(sub[sub["KorBB"] == "Strikeout"])
        pt_rows.append({
            "Pitch Group": group_name,
            "#": len(sub),
            "%": round(len(sub) / len(bdf) * 100, 1),
            "PA": sub_pa,
            "SO": sub_ks,
            "BBE": len(sub_ip),
            "EV": round(sub_ip["ExitSpeed"].mean(), 1) if len(sub_ip) > 0 else None,
            "LA": round(sub_ip["Angle"].mean(), 1) if len(sub_ip) > 0 and sub_ip["Angle"].notna().any() else None,
            "Whiff%": round(len(sub_whiffs) / max(len(sub_swings), 1) * 100, 1),
        })
    if pt_rows:
        st.dataframe(pd.DataFrame(pt_rows), use_container_width=True, hide_index=True)

    # Per-pitch type breakdown (only main pitches)
    bdf_filtered = filter_minor_pitches(bdf)
    pt_detail_rows = []
    for pt in sorted(bdf_filtered["TaggedPitchType"].dropna().unique()):
        sub = bdf_filtered[bdf_filtered["TaggedPitchType"] == pt]
        if len(sub) < 3:
            continue
        sub_swings = sub[sub["PitchCall"].isin(SWING_CALLS)]
        sub_whiffs = sub[sub["PitchCall"] == "StrikeSwinging"]
        sub_ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        pt_detail_rows.append({
            "Pitch": pt,
            "#": len(sub),
            "%": round(len(sub) / len(bdf) * 100, 1),
            "BBE": len(sub_ip),
            "EV": round(sub_ip["ExitSpeed"].mean(), 1) if len(sub_ip) > 0 else None,
            "LA": round(sub_ip["Angle"].mean(), 1) if len(sub_ip) > 0 and sub_ip["Angle"].notna().any() else None,
            "Whiff%": round(len(sub_whiffs) / max(len(sub_swings), 1) * 100, 1),
        })
    if pt_detail_rows:
        with st.expander("Detailed Pitch Type Breakdown"):
            st.dataframe(pd.DataFrame(pt_detail_rows), use_container_width=True, hide_index=True)

    # ── COUNT-BASED PERFORMANCE (Visual Heatmap Grid) ──
    st.markdown("---")
    section_header("Count-Based Performance")
    if "Balls" in bdf.columns and "Strikes" in bdf.columns:
        # Compute DB-wide averages per count for context
        all_data_counts = data.dropna(subset=["Balls", "Strikes"]).copy()
        all_data_counts["Count"] = all_data_counts["Balls"].astype(int).astype(str) + "-" + all_data_counts["Strikes"].astype(int).astype(str)
        bdf_counts = bdf.dropna(subset=["Balls", "Strikes"]).copy()
        bdf_counts["Count"] = bdf_counts["Balls"].astype(int).astype(str) + "-" + bdf_counts["Strikes"].astype(int).astype(str)

        def _count_stats(sub):
            sw = sub[sub["PitchCall"].isin(SWING_CALLS)]
            wh = sub[sub["PitchCall"] == "StrikeSwinging"]
            ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            return {
                "n": len(sub), "swings": len(sw),
                "swing_pct": len(sw) / max(len(sub), 1) * 100,
                "whiff_pct": len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else 0,
                "ev": ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan,
                "bbe": len(ip),
            }

        # DB averages per count
        db_count_avgs = {}
        for cnt, grp in all_data_counts.groupby("Count"):
            db_count_avgs[cnt] = _count_stats(grp)

        # Player stats per count
        player_count_stats = {}
        for cnt, grp in bdf_counts.groupby("Count"):
            player_count_stats[cnt] = _count_stats(grp)

        # Visual count grid — 4 balls x 3 strikes, color-coded by Whiff% vs DB avg
        st.markdown(
            '<p style="font-size:12px;color:#666;margin-bottom:4px;">'
            'Each cell shows this hitter\'s Whiff% in that count. '
            '<span style="color:#1a7a1a;font-weight:700;">Green = better than DB avg</span> (lower whiff), '
            '<span style="color:#cc0000;font-weight:700;">Red = worse</span> (higher whiff). '
            'Hover labels show the full context.</p>',
            unsafe_allow_html=True,
        )

        count_metric = st.radio("Count Grid Metric", ["Whiff%", "Swing%", "Avg EV"], horizontal=True, key="hc_count_metric")
        metric_key = {"Whiff%": "whiff_pct", "Swing%": "swing_pct", "Avg EV": "ev"}[count_metric]
        # For whiff and swing, lower is better for hitter; for EV, higher is better
        higher_better = count_metric == "Avg EV"

        # Build heatmap grid
        balls_range = [0, 1, 2, 3]
        strikes_range = [0, 1, 2]
        z_vals = []
        hover_text = []
        for s in strikes_range:
            row_z = []
            row_hover = []
            for b in balls_range:
                cnt = f"{b}-{s}"
                ps = player_count_stats.get(cnt, {})
                db = db_count_avgs.get(cnt, {})
                p_val = ps.get(metric_key, np.nan)
                db_val = db.get(metric_key, np.nan)
                n = ps.get("n", 0)
                if n < 3 or pd.isna(p_val):
                    row_z.append(np.nan)
                    row_hover.append(f"{cnt}<br>Not enough data")
                else:
                    # Compute difference vs DB (positive = better for hitter)
                    if higher_better:
                        diff = p_val - db_val if not pd.isna(db_val) else 0
                    else:
                        diff = db_val - p_val if not pd.isna(db_val) else 0
                    row_z.append(diff)
                    db_str = f"{db_val:.1f}" if not pd.isna(db_val) else "N/A"
                    verdict = "BETTER" if diff > 0 else ("WORSE" if diff < 0 else "AVERAGE")
                    v_color = "#1a7a1a" if diff > 0 else ("#cc0000" if diff < 0 else "#666")
                    row_hover.append(
                        f"<b>{cnt}</b> ({n} pitches)<br>"
                        f"{count_metric}: <b>{p_val:.1f}{'%' if metric_key != 'ev' else ' mph'}</b><br>"
                        f"DB Avg: {db_str}{'%' if metric_key != 'ev' else ' mph'}<br>"
                        f"<span style='color:{v_color}'>{verdict} ({diff:+.1f})</span>"
                    )
            z_vals.append(row_z)
            hover_text.append(row_hover)

        # Build annotations with actual values
        annotations = []
        for si, s in enumerate(strikes_range):
            for bi, b in enumerate(balls_range):
                cnt = f"{b}-{s}"
                ps = player_count_stats.get(cnt, {})
                db = db_count_avgs.get(cnt, {})
                p_val = ps.get(metric_key, np.nan)
                db_val = db.get(metric_key, np.nan)
                n = ps.get("n", 0)
                if n >= 3 and not pd.isna(p_val):
                    annotations.append(dict(
                        x=bi, y=si, text=f"<b>{p_val:.1f}</b><br><span style='font-size:9px'>DB: {db_val:.1f}</span>" if not pd.isna(db_val) else f"<b>{p_val:.1f}</b>",
                        showarrow=False, font=dict(size=13, color="white", family="Inter"),
                    ))
                else:
                    annotations.append(dict(
                        x=bi, y=si, text="\u2014", showarrow=False,
                        font=dict(size=13, color="#aaa", family="Inter"),
                    ))

        fig = go.Figure(data=go.Heatmap(
            z=z_vals,
            x=[f"{b} Balls" for b in balls_range],
            y=[f"{s} Strikes" for s in strikes_range],
            colorscale=[
                [0.0, "#cc0000"], [0.35, "#e8a0a0"], [0.5, "#f0f0f0"],
                [0.65, "#a0d4a0"], [1.0, "#1a7a1a"],
            ],
            zmid=0, showscale=False,
            hovertext=hover_text, hoverinfo="text",
            xgap=3, ygap=3,
        ))
        fig.update_layout(
            annotations=annotations,
            xaxis=dict(side="top", tickfont=dict(size=12, color="#000000")),
            yaxis=dict(tickfont=dict(size=12, color="#000000"), autorange="reversed"),
            height=250, margin=dict(l=80, r=20, t=40, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter, Arial, sans-serif"),
        )
        _plotly_chart_bats(fig, use_container_width=True, key="hitter_count_grid")

        # Count-state summary with percentile bars
        st.markdown("")
        bdf_st = bdf.dropna(subset=["Balls", "Strikes"]).copy()
        bdf_st["CountState"] = "Even"
        bdf_st.loc[bdf_st["Strikes"] > bdf_st["Balls"], "CountState"] = "Behind"
        bdf_st.loc[bdf_st["Balls"] > bdf_st["Strikes"], "CountState"] = "Ahead"
        all_st = data.dropna(subset=["Balls", "Strikes"]).copy()
        all_st["CountState"] = "Even"
        all_st.loc[all_st["Strikes"] > all_st["Balls"], "CountState"] = "Behind"
        all_st.loc[all_st["Balls"] > all_st["Strikes"], "CountState"] = "Ahead"

        # Compute per-batter stats by count state for percentiles
        def _batter_count_state_stats(full_data, state):
            sub = full_data[full_data["CountState"] == state] if state else full_data
            rows = []
            for batter, grp in sub.groupby("Batter"):
                if len(grp) < 10:
                    continue
                sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
                wh = grp[grp["PitchCall"] == "StrikeSwinging"]
                ip = grp[grp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                rows.append({
                    "Batter": batter,
                    "whiff_pct": len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else np.nan,
                    "swing_pct": len(sw) / max(len(grp), 1) * 100,
                    "ev": ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan,
                })
            return pd.DataFrame(rows)

        state_metrics = []
        for state in ["Ahead", "Even", "Behind"]:
            p_sub = bdf_st[bdf_st["CountState"] == state]
            if len(p_sub) < 5:
                continue
            p_stats = _count_stats(p_sub)
            all_state_df = _batter_count_state_stats(data.dropna(subset=["Balls", "Strikes"]).assign(
                CountState=lambda x: np.where(x["Strikes"] > x["Balls"], "Behind",
                                    np.where(x["Balls"] > x["Strikes"], "Ahead", "Even"))
            ), state)
            whiff_pct = get_percentile(p_stats["whiff_pct"], all_state_df["whiff_pct"]) if not all_state_df.empty else 50
            ev_pct = get_percentile(p_stats["ev"], all_state_df["ev"]) if not all_state_df.empty and not pd.isna(p_stats["ev"]) else np.nan
            state_metrics.append(
                (f"{state} Whiff%", p_stats["whiff_pct"], whiff_pct, ".1f", False)
            )
            if not pd.isna(ev_pct):
                state_metrics.append(
                    (f"{state} EV", p_stats["ev"], ev_pct, ".1f", True)
                )

        if state_metrics:
            render_savant_percentile_section(state_metrics, "Count-State Performance (Percentile)")
            st.caption("Percentile vs. all batters in DB with 10+ pitches in that count state")

    else:
        st.caption("Count data not available.")

    # ── SITUATIONAL SPLITS (Percentile-Based Visual) ──
    st.markdown("---")
    section_header("Situational Splits")

    def _compute_split_percentile(player_df, all_data, split_col, split_val, filter_fn=None):
        """Compute player stats and percentile rank for a split."""
        if filter_fn:
            p_sub = filter_fn(player_df)
            a_sub = filter_fn(all_data)
        else:
            p_sub = player_df[player_df[split_col] == split_val] if split_col else player_df
            a_sub = all_data[all_data[split_col] == split_val] if split_col else all_data
        if len(p_sub) < 5:
            return None

        p_sw = p_sub[p_sub["PitchCall"].isin(SWING_CALLS)]
        p_wh = p_sub[p_sub["PitchCall"] == "StrikeSwinging"]
        p_ip = p_sub[p_sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])

        # All batters in this split
        batter_rows = []
        for batter, grp in a_sub.groupby("Batter"):
            if len(grp) < 10:
                continue
            sw = grp[grp["PitchCall"].isin(SWING_CALLS)]
            wh = grp[grp["PitchCall"] == "StrikeSwinging"]
            ip = grp[grp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            batter_rows.append({
                "whiff": len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else np.nan,
                "ev": ip["ExitSpeed"].mean() if len(ip) > 0 else np.nan,
                "chase": (lambda _iz=in_zone_mask(grp): len(grp[(~_iz) & grp["PlateLocSide"].notna()][grp["PitchCall"].isin(SWING_CALLS)]) / max(len(grp[(~_iz) & grp["PlateLocSide"].notna()]), 1) * 100)() if grp["PlateLocSide"].notna().any() else np.nan,
            })
        all_df = pd.DataFrame(batter_rows)

        p_whiff = len(p_wh) / max(len(p_sw), 1) * 100 if len(p_sw) > 0 else np.nan
        p_ev = p_ip["ExitSpeed"].mean() if len(p_ip) > 0 else np.nan

        return {
            "n": len(p_sub), "bbe": len(p_ip),
            "whiff": p_whiff,
            "whiff_pct": get_percentile(p_whiff, all_df["whiff"]) if not all_df.empty and not pd.isna(p_whiff) else np.nan,
            "ev": p_ev,
            "ev_pct": get_percentile(p_ev, all_df["ev"]) if not all_df.empty and not pd.isna(p_ev) else np.nan,
        }

    sit_metrics = []

    # vs RHP / LHP
    if "PitcherThrows" in bdf.columns:
        for side, label in [("Right", "vs RHP"), ("Left", "vs LHP")]:
            res = _compute_split_percentile(bdf, data, "PitcherThrows", side)
            if res:
                sit_metrics.append((f"{label} Whiff%", res["whiff"], res["whiff_pct"], ".1f", False))
                if not pd.isna(res["ev_pct"]):
                    sit_metrics.append((f"{label} Avg EV", res["ev"], res["ev_pct"], ".1f", True))

    # By inning group
    if "Inning" in bdf.columns:
        for label, lo, hi in [("Early Inn.", 1, 3), ("Mid Inn.", 4, 6), ("Late Inn.", 7, 20)]:
            res = _compute_split_percentile(bdf, data, None, None,
                filter_fn=lambda df, l=lo, h=hi: df[df["Inning"].between(l, h)])
            if res:
                sit_metrics.append((f"{label} Whiff%", res["whiff"], res["whiff_pct"], ".1f", False))
                if not pd.isna(res["ev_pct"]):
                    sit_metrics.append((f"{label} Avg EV", res["ev"], res["ev_pct"], ".1f", True))

    if sit_metrics:
        render_savant_percentile_section(sit_metrics, None)
        st.caption("Percentile vs. all batters in DB (min 10 pitches in that split)")


# ──────────────────────────────────────────────
# SWING DECISION LAB
# ──────────────────────────────────────────────

def _swing_decision_lab(data, batter, season_filter, bdf, batted, pr, all_batter_stats):
    """Render the Swing Decision Lab — plate discipline, decision maps,
    pitch-type breakdowns, count-state analysis, and recommendations."""

    all_stats = all_batter_stats
    # Use adaptive per-batter zone (matches population computation)
    batter_zones = _build_batter_zones(data)
    iz = in_zone_mask(bdf, batter_zones, batter_col="Batter")
    oz = ~iz & bdf["PlateLocSide"].notna() & bdf["PlateLocHeight"].notna()
    swings = bdf[bdf["PitchCall"].isin(SWING_CALLS)]
    whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"]
    contacts = bdf[bdf["PitchCall"].isin(CONTACT_CALLS)]
    in_zone_pitches = bdf[iz]
    out_zone_pitches = bdf[oz]
    swings_iz = in_zone_pitches[in_zone_pitches["PitchCall"].isin(SWING_CALLS)]
    swings_oz = out_zone_pitches[out_zone_pitches["PitchCall"].isin(SWING_CALLS)]
    contacts_iz = in_zone_pitches[in_zone_pitches["PitchCall"].isin(CONTACT_CALLS)]
    contacts_oz = out_zone_pitches[out_zone_pitches["PitchCall"].isin(CONTACT_CALLS)]

    zone_swing_pct = len(swings_iz) / max(len(in_zone_pitches), 1) * 100
    chase_pct = len(swings_oz) / max(len(out_zone_pitches), 1) * 100
    whiff_pct = len(whiffs) / max(len(swings), 1) * 100
    zone_contact_pct = len(contacts_iz) / max(len(swings_iz), 1) * 100 if len(swings_iz) > 0 else np.nan
    chase_contact_pct = len(contacts_oz) / max(len(swings_oz), 1) * 100 if len(swings_oz) > 0 else np.nan

    # ═══════════════════════════════════════════
    # SECTION A: Decision Profile Summary
    # ═══════════════════════════════════════════
    section_header("Decision Profile")
    st.caption("Plate discipline metrics vs. all hitters in the database.")

    disc_metrics = [
        ("Zone Swing%", zone_swing_pct, get_percentile(zone_swing_pct, _safe_pop(all_stats, "ZoneSwingPct")), ".1f", True),
        ("Chase%", chase_pct, get_percentile(chase_pct, _safe_pop(all_stats, "ChasePct")), ".1f", False),
        ("Whiff%", whiff_pct, get_percentile(whiff_pct, _safe_pop(all_stats, "WhiffPct")), ".1f", False),
        ("Z-Contact%", zone_contact_pct, get_percentile(zone_contact_pct, _safe_pop(all_stats, "ZoneContactPct")), ".1f", True),
        ("Chase Contact%", chase_contact_pct,
         get_percentile(chase_contact_pct, _safe_pop(all_stats, "ChaseContact")) if not pd.isna(chase_contact_pct) else np.nan,
         ".1f", True),
    ]
    render_savant_percentile_section(disc_metrics)

    # Decision Score composite
    zs_pctl = get_percentile(zone_swing_pct, _safe_pop(all_stats, "ZoneSwingPct"))
    ch_pctl = get_percentile(chase_pct, _safe_pop(all_stats, "ChasePct"))
    zc_pctl = get_percentile(zone_contact_pct, _safe_pop(all_stats, "ZoneContactPct")) if not pd.isna(zone_contact_pct) else 50
    # For chase, lower is better → invert percentile
    ch_pctl_good = (100 - ch_pctl) if not pd.isna(ch_pctl) else 50
    zs_pctl_safe = zs_pctl if not pd.isna(zs_pctl) else 50
    zc_pctl_safe = zc_pctl if not pd.isna(zc_pctl) else 50
    decision_score = round(zs_pctl_safe * 0.35 + ch_pctl_good * 0.40 + zc_pctl_safe * 0.25, 0)

    # Verdict — nuanced based on the specific metric profile
    high_chase = chase_pct > 32 if not pd.isna(chase_pct) else False
    low_chase = chase_pct < 25 if not pd.isna(chase_pct) else False
    passive_zone = zone_swing_pct < 60
    aggressive_zone = zone_swing_pct > 72
    high_zc = zone_contact_pct > 85 if not pd.isna(zone_contact_pct) else False
    low_zc = zone_contact_pct < 75 if not pd.isna(zone_contact_pct) else False

    if decision_score >= 80:
        if low_chase and aggressive_zone:
            verdict = "Elite plate discipline \u2014 hunts strikes and lays off everything else"
        elif low_chase and high_zc:
            verdict = "Exceptional pitch recognition with elite zone contact"
        else:
            verdict = "Top-tier decision-maker \u2014 controls the at-bat consistently"
    elif decision_score >= 65:
        if high_chase:
            verdict = f"Strong zone attacker but chases too much ({chase_pct:.0f}% chase rate)"
        elif passive_zone:
            verdict = f"Great eye but could be more aggressive on hittable pitches ({zone_swing_pct:.0f}% zone swing)"
        elif low_zc:
            verdict = "Good pitch selection, needs to barrel more pitches in the zone"
        else:
            verdict = "Above-average approach \u2014 makes pitchers work for outs"
    elif decision_score >= 50:
        if high_chase and passive_zone:
            verdict = "Swings at the wrong pitches \u2014 chases out of zone, takes in zone"
        elif high_chase:
            verdict = f"Expanding the zone too often ({chase_pct:.0f}% chase) \u2014 hurting the at-bat"
        elif passive_zone:
            verdict = f"Leaving hittable pitches in the zone ({zone_swing_pct:.0f}% zone swing)"
        else:
            verdict = "Middle-of-the-pack approach \u2014 no glaring weakness but no strength either"
    elif decision_score >= 35:
        if high_chase and low_zc:
            verdict = "Chasing off the plate and struggling to connect in the zone"
        elif aggressive_zone and high_chase:
            verdict = f"Free swinger \u2014 aggressive everywhere, {chase_pct:.0f}% chase rate"
        elif passive_zone:
            verdict = "Too passive \u2014 watching too many hittable pitches go by"
        else:
            verdict = "Below-average pitch selection \u2014 approach needs refinement"
    else:
        if high_chase and passive_zone:
            verdict = "Inverted approach \u2014 chasing balls while taking strikes"
        elif high_chase:
            verdict = f"Expands aggressively ({chase_pct:.0f}% chase) \u2014 pitchers exploiting the zone"
        else:
            verdict = "Significant swing-decision issues \u2014 needs mechanical or recognition work"

    ds_color = "#22c55e" if decision_score >= 70 else "#3b82f6" if decision_score >= 55 else "#f59e0b" if decision_score >= 40 else "#ef4444"
    st.markdown(
        f'<div style="padding:10px 16px;border-radius:8px;border-left:5px solid {ds_color};'
        f'background:{ds_color}10;margin:8px 0;">'
        f'<span style="font-size:20px;font-weight:bold;color:{ds_color};">Decision Score: {decision_score:.0f}</span>'
        f'<span style="font-size:13px;color:#555;margin-left:12px;">{verdict}</span>'
        f'</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # SECTION B: Swing Decision Zone Maps
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Swing Decision Zone Maps")
    st.caption(
        "Where should this hitter swing vs. where they actually swing? Mismatch = coaching opportunity. "
        "Should Swing score blends EV (45%), contact rate (35%), and in‑play rate (20%), "
        "then shrinks toward 50 with small samples (based on swing count)."
    )

    loc_data = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    _bs = safe_mode(bdf["BatterSide"], "Right") if "BatterSide" in bdf.columns else "Right"
    is_rhh = _bs != "Left"
    if len(loc_data) >= 30:
        # Build 5x5 grid
        h_edges = np.linspace(-1.5, 1.5, 6)
        v_edges = np.linspace(0.5, 4.5, 6)

        should_swing = np.full((5, 5), np.nan)
        actually_swings = np.full((5, 5), np.nan)
        mismatch = np.full((5, 5), np.nan)

        for vi in range(5):
            for hi in range(5):
                mask = (
                    (loc_data["PlateLocSide"] >= h_edges[hi]) &
                    (loc_data["PlateLocSide"] < h_edges[hi + 1]) &
                    (loc_data["PlateLocHeight"] >= v_edges[vi]) &
                    (loc_data["PlateLocHeight"] < v_edges[vi + 1])
                )
                cell = loc_data[mask]
                if len(cell) < 3:
                    continue
                cell_swings = cell[cell["PitchCall"].isin(SWING_CALLS)]
                swing_rate = len(cell_swings) / len(cell) * 100
                actually_swings[vi, hi] = swing_rate

                cell_ip = cell[(cell["PitchCall"] == "InPlay") & cell["ExitSpeed"].notna()]
                cell_whiffs = cell[cell["PitchCall"] == "StrikeSwinging"]
                if len(cell_swings) >= 2:
                    contact_rate = 1 - len(cell_whiffs) / len(cell_swings) if len(cell_swings) > 0 else 0
                    inplay_rate = len(cell_ip) / len(cell_swings) if len(cell_swings) > 0 else 0
                    avg_ev = cell_ip["ExitSpeed"].mean() if len(cell_ip) >= 2 else 70
                    ev_score = min(max((avg_ev - 70) / 30 * 100, 0), 100) if not pd.isna(avg_ev) else 30
                    contact_score = contact_rate * 100
                    inplay_score = inplay_rate * 100
                    raw_should = ev_score * 0.45 + contact_score * 0.35 + inplay_score * 0.20
                    shrink_w = min(len(cell_swings) / 15, 1.0)
                    should_score = 50 * (1 - shrink_w) + raw_should * shrink_w
                    should_swing[vi, hi] = should_score
                    mismatch[vi, hi] = swing_rate - should_score

        # Hide cells outside the strike zone (only show center 3x3)
        keep_mask = np.zeros((5, 5), dtype=bool)
        keep_mask[1:4, 1:4] = True
        for arr in (should_swing, actually_swings, mismatch):
            arr[~keep_mask] = np.nan

        # Label columns relative to batter handedness.
        # Trackman: col 0 = most negative PlateLocSide = 3B side.
        # RHH: 3B side = inside → labels left-to-right: In … Away
        # LHH: 3B side = away  → labels left-to-right: Away … In
        if is_rhh:
            h_labels = ["In", "In-Mid", "Mid", "Away-Mid", "Away"]
            h_labels_in = ["In", "Mid", "Away"]
        else:
            h_labels = ["Away", "Away-Mid", "Mid", "In-Mid", "In"]
            h_labels_in = ["Away", "Mid", "In"]
        v_labels = ["Low", "Low-Mid", "Mid", "Up-Mid", "Up"]
        v_labels_in = ["Low", "Mid", "Up"]

        # Use only the in-zone 3x3 for display
        should_in = should_swing[1:4, 1:4]
        actual_in = actually_swings[1:4, 1:4]
        mismatch_in = mismatch[1:4, 1:4]

        map1, map2, map3 = st.columns(3)
        with map1:
            st.markdown("**Should Swing**")
            st.caption("Green = good outcomes when swinging here")
            fig_should = px.imshow(
                np.flipud(should_in), text_auto=".0f",
                color_continuous_scale=[[0, "#ef4444"], [0.5, "#fbbf24"], [1, "#22c55e"]],
                x=h_labels_in, y=list(reversed(v_labels_in)),
                labels=dict(color="Score"), aspect="auto",
            )
            _add_grid_zone_outline(fig_should)
            fig_should.update_layout(height=320, coloraxis_showscale=False, **CHART_LAYOUT)
            _plotly_chart_bats(fig_should, use_container_width=True, key="sdl_should")

        with map2:
            st.markdown("**Actually Swings**")
            st.caption("Swing rate by zone cell")
            fig_actual = px.imshow(
                np.flipud(actual_in), text_auto=".0f",
                color_continuous_scale="YlOrRd",
                x=h_labels_in, y=list(reversed(v_labels_in)),
                labels=dict(color="Swing%"), aspect="auto",
            )
            _add_grid_zone_outline(fig_actual)
            fig_actual.update_layout(height=320, coloraxis_showscale=False, **CHART_LAYOUT)
            _plotly_chart_bats(fig_actual, use_container_width=True, key="sdl_actual")

        with map3:
            st.markdown("**Mismatch**")
            st.caption("Red = swings too much, Blue = should swing more")
            mm_rounded = np.round(mismatch_in)
            mm_text = []
            for row in np.flipud(mm_rounded):
                row_text = []
                for v in row:
                    row_text.append(f"{v:+.0f}" if not np.isnan(v) else "")
                mm_text.append(row_text)
            fig_mm = px.imshow(
                np.flipud(mm_rounded),
                color_continuous_scale=[[0, "#3b82f6"], [0.5, "white"], [1, "#ef4444"]],
                x=h_labels_in, y=list(reversed(v_labels_in)),
                labels=dict(color="Over/Under"), aspect="auto",
                zmin=-50, zmax=50,
            )
            fig_mm.update_traces(text=mm_text, texttemplate="%{text}")
            _add_grid_zone_outline(fig_mm)
            fig_mm.update_layout(height=320, coloraxis_showscale=False, **CHART_LAYOUT)
            _plotly_chart_bats(fig_mm, use_container_width=True, key="sdl_mismatch")

        # Summary blurb: where to swing more/less (largest mismatches)
        cells = []
        for vi in range(3):
            for hi in range(3):
                val = mismatch_in[vi, hi]
                if pd.isna(val):
                    continue
                cells.append({
                    "v": v_labels_in[vi],
                    "h": h_labels_in[hi],
                    "m": float(val),
                })
        if cells:
            more = sorted([c for c in cells if c["m"] < 0], key=lambda x: x["m"])[:2]
            less = sorted([c for c in cells if c["m"] > 0], key=lambda x: -x["m"])[:2]

            def _fmt_zone(c):
                return f'{c["v"]} / {c["h"]} ({c["m"]:+.0f})'

            more_txt = ", ".join(_fmt_zone(c) for c in more) if more else "None"
            less_txt = ", ".join(_fmt_zone(c) for c in less) if less else "None"
            st.markdown(
                f'<div style="padding:12px 16px;margin-top:8px;font-size:14px;line-height:1.5;'
                f'background:#f8fafc;border-radius:8px;border-left:4px solid #334155;">'
                f'<b>Summary:</b><br>'
                f'• <b>Swing more</b> in: {more_txt}<br>'
                f'• <b>Swing less</b> in: {less_txt}'
                f'</div>', unsafe_allow_html=True
            )
    else:
        st.info("Not enough location data for zone maps (need 30+ pitches).")

    # ═══════════════════════════════════════════
    # SECTION C: Pitch-Type Decision Breakdown
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Pitch-Type Decision Breakdown")
    st.caption("How does this hitter decide against each pitch type? In-zone vs. out-of-zone splits.")

    pitch_types_seen = sorted(bdf["TaggedPitchType"].dropna().unique())
    # Compute team averages for comparison
    all_dav_bat = filter_davidson(data, "batter")
    all_dav_bat = all_dav_bat[all_dav_bat["Season"].isin(season_filter)]
    team_batter_zones = _build_batter_zones(all_dav_bat)

    pt_rows = []
    flags = []
    for pt in pitch_types_seen:
        pt_d = bdf[bdf["TaggedPitchType"] == pt]
        if len(pt_d) < 5:
            continue
        pt_iz = pt_d[in_zone_mask(pt_d, batter_zones, "Batter")]
        pt_oz_mask = ~in_zone_mask(pt_d, batter_zones, "Batter") & pt_d["PlateLocSide"].notna()
        pt_oz = pt_d[pt_oz_mask]
        pt_sw = pt_d[pt_d["PitchCall"].isin(SWING_CALLS)]
        pt_wh = pt_d[pt_d["PitchCall"] == "StrikeSwinging"]
        pt_sw_oz = pt_oz[pt_oz["PitchCall"].isin(SWING_CALLS)]
        pt_sw_iz = pt_iz[pt_iz["PitchCall"].isin(SWING_CALLS)]
        pt_ip = pt_d[(pt_d["PitchCall"] == "InPlay") & pt_d["ExitSpeed"].notna()]

        swing_pct = len(pt_sw) / max(len(pt_d), 1) * 100
        whiff_pt = len(pt_wh) / max(len(pt_sw), 1) * 100 if len(pt_sw) > 0 else 0
        chase_pt = len(pt_sw_oz) / max(len(pt_oz), 1) * 100 if len(pt_oz) > 0 else 0
        zone_swing_pt = len(pt_sw_iz) / max(len(pt_iz), 1) * 100 if len(pt_iz) > 0 else 0
        take_iz_pct = 100 - zone_swing_pt
        avg_ev_pt = pt_ip["ExitSpeed"].mean() if len(pt_ip) >= 2 else np.nan

        pt_rows.append({
            "Pitch": pt, "Seen": len(pt_d),
            "Swing%": round(swing_pct, 1),
            "Whiff%": round(whiff_pt, 1),
            "Chase%": round(chase_pt, 1),
            "Z-Swing%": round(zone_swing_pt, 1),
            "Z-Take%": round(take_iz_pct, 1),
            "Avg EV": round(avg_ev_pt, 1) if not pd.isna(avg_ev_pt) else np.nan,
        })

        # Compute team averages for this pitch type to flag outliers
        team_pt = all_dav_bat[all_dav_bat["TaggedPitchType"] == pt]
        if len(team_pt) >= 20:
            team_oz_mask = ~in_zone_mask(team_pt, team_batter_zones, "Batter") & team_pt["PlateLocSide"].notna()
            team_oz = team_pt[team_oz_mask]
            team_sw_oz = team_oz[team_oz["PitchCall"].isin(SWING_CALLS)]
            team_chase = len(team_sw_oz) / max(len(team_oz), 1) * 100

            team_iz = team_pt[in_zone_mask(team_pt, team_batter_zones, "Batter")]
            team_sw_iz = team_iz[team_iz["PitchCall"].isin(SWING_CALLS)]
            team_z_swing = len(team_sw_iz) / max(len(team_iz), 1) * 100

            if chase_pt > team_chase + 10 and len(pt_oz) >= 5:
                flags.append(f"Chases **{pt}** {chase_pt:.0f}% of the time (team avg: {team_chase:.0f}%)")
            if take_iz_pct > 30 and take_iz_pct > (100 - team_z_swing) + 10 and len(pt_iz) >= 5:
                ev_str = f", avg EV when swinging: {avg_ev_pt:.0f} mph" if not pd.isna(avg_ev_pt) else ""
                flags.append(f"Takes in-zone **{pt}** {take_iz_pct:.0f}% of the time \u2014 missing hittable pitches{ev_str}")

    if pt_rows:
        pt_df = pd.DataFrame(pt_rows)
        st.dataframe(pt_df, use_container_width=True, hide_index=True)

    if flags:
        st.markdown("**Flags:**")
        for f in flags:
            st.markdown(
                f'<div style="padding:4px 12px;margin:2px 0;font-size:12px;'
                f'background:#fff8e1;border-radius:4px;border-left:3px solid #f59e0b;">'
                f'{f}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # SECTION D: Count-State Decisions
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Count-State Decisions")

    # Ahead vs Behind metrics
    ahead_counts = [("1", "0"), ("2", "0"), ("2", "1"), ("3", "0"), ("3", "1")]
    behind_counts = [("0", "1"), ("0", "2"), ("1", "2")]

    def _count_metrics(count_list, label):
        frames = []
        for b, s in count_list:
            cd = bdf[(bdf["Balls"].astype(str) == b) & (bdf["Strikes"].astype(str) == s)]
            frames.append(cd)
        combined = pd.concat(frames) if frames else pd.DataFrame()
        if len(combined) < 5:
            return None
        c_sw = combined[combined["PitchCall"].isin(SWING_CALLS)]
        c_wh = combined[combined["PitchCall"] == "StrikeSwinging"]
        c_iz = combined[in_zone_mask(combined, batter_zones, "Batter")]
        c_oz_mask = ~in_zone_mask(combined, batter_zones, "Batter") & combined["PlateLocSide"].notna()
        c_oz = combined[c_oz_mask]
        c_sw_oz = c_oz[c_oz["PitchCall"].isin(SWING_CALLS)]
        c_ip = combined[(combined["PitchCall"] == "InPlay") & combined["ExitSpeed"].notna()]
        return {
            "label": label, "n": len(combined),
            "swing_pct": len(c_sw) / max(len(combined), 1) * 100,
            "chase_pct": len(c_sw_oz) / max(len(c_oz), 1) * 100 if len(c_oz) > 0 else 0,
            "whiff_pct": len(c_wh) / max(len(c_sw), 1) * 100 if len(c_sw) > 0 else 0,
            "avg_ev": c_ip["ExitSpeed"].mean() if len(c_ip) >= 2 else np.nan,
        }

    ahead_m = _count_metrics(ahead_counts, "Ahead (1-0, 2-0, 2-1, 3-0, 3-1)")
    behind_m = _count_metrics(behind_counts, "Behind (0-1, 0-2, 1-2)")

    if ahead_m and behind_m:
        cc1, cc2 = st.columns(2)
        for col, m, color in [(cc1, ahead_m, "#22c55e"), (cc2, behind_m, "#ef4444")]:
            with col:
                ev_str = f"{m['avg_ev']:.1f} mph" if not pd.isna(m['avg_ev']) else "-"
                st.markdown(
                    f'<div style="padding:10px 14px;border-radius:8px;border-left:4px solid {color};'
                    f'background:{color}10;margin:4px 0;">'
                    f'<div style="font-size:14px;font-weight:bold;">{m["label"]}</div>'
                    f'<div style="font-size:12px;color:#555;margin-top:4px;">'
                    f'Swing% <b>{m["swing_pct"]:.0f}</b> &middot; '
                    f'Chase% <b>{m["chase_pct"]:.0f}</b> &middot; '
                    f'Whiff% <b>{m["whiff_pct"]:.0f}</b> &middot; '
                    f'Avg EV <b>{ev_str}</b>'
                    f'</div><div style="font-size:11px;color:#888;">({m["n"]} pitches)</div>'
                    f'</div>', unsafe_allow_html=True)

        # Chase expansion: compare 0-0 vs 0-2 swing zones
        st.markdown("**Zone Expansion: 0-0 vs 0-2**")
        st.caption("How much does the swing zone expand when behind?")
        cnt_00 = bdf[(bdf["Balls"].astype(str) == "0") & (bdf["Strikes"].astype(str) == "0")]
        cnt_02 = bdf[(bdf["Balls"].astype(str) == "0") & (bdf["Strikes"].astype(str) == "2")]
        if len(cnt_00) >= 10 and len(cnt_02) >= 10:
            exp1, exp2 = st.columns(2)
            for col_exp, cnt_d, cnt_label in [(exp1, cnt_00, "0-0"), (exp2, cnt_02, "0-2")]:
                with col_exp:
                    cnt_loc = cnt_d.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                    cnt_sw = cnt_loc[cnt_loc["PitchCall"].isin(SWING_CALLS)]
                    if len(cnt_sw) >= 3:
                        fig_exp = go.Figure(go.Histogram2d(
                            x=cnt_sw["PlateLocSide"], y=cnt_sw["PlateLocHeight"],
                            xbins=dict(start=-2.5, end=2.5, size=5.0/8),
                            ybins=dict(start=0, end=5, size=5.0/8),
                            colorscale="YlOrRd", showscale=False,
                        ))
                        add_strike_zone(fig_exp)
                        fig_exp.update_layout(
                            xaxis=dict(range=[-2.5, 2.5], title="", showticklabels=False,
                                       scaleanchor="y", scaleratio=1),
                            yaxis=dict(range=[0, 5], title="", showticklabels=False),
                            height=350, margin=dict(l=5, r=5, t=5, b=5),
                            plot_bgcolor="white", paper_bgcolor="white",
                        )
                        _plotly_chart_bats(fig_exp, use_container_width=True, key=f"sdl_exp_{cnt_label}")
                        # Compute chase rate for this count
                        cnt_iz = cnt_d[in_zone_mask(cnt_d, batter_zones, "Batter")]
                        cnt_oz_m = ~in_zone_mask(cnt_d, batter_zones, "Batter") & cnt_d["PlateLocSide"].notna()
                        cnt_oz = cnt_d[cnt_oz_m]
                        cnt_ch = len(cnt_oz[cnt_oz["PitchCall"].isin(SWING_CALLS)]) / max(len(cnt_oz), 1) * 100
                        st.caption(f"{cnt_label}: Chase {cnt_ch:.0f}% ({len(cnt_d)} pitches)")
                    else:
                        st.caption(f"{cnt_label}: not enough swings")
        else:
            st.info("Not enough data at 0-0 or 0-2 counts.")

    # Two-strike approach
    st.markdown("**Two-Strike Approach**")
    two_strike = bdf[pd.to_numeric(bdf["Strikes"], errors='coerce').fillna(0).astype(int) == 2]
    if len(two_strike) >= 10:
        ts_rows = []
        for pt in sorted(two_strike["TaggedPitchType"].dropna().unique()):
            pt_2s = two_strike[two_strike["TaggedPitchType"] == pt]
            if len(pt_2s) < 3:
                continue
            pt_2s_sw = pt_2s[pt_2s["PitchCall"].isin(SWING_CALLS)]
            pt_2s_wh = pt_2s[pt_2s["PitchCall"] == "StrikeSwinging"]
            pt_2s_oz_m = ~in_zone_mask(pt_2s, batter_zones, "Batter") & pt_2s["PlateLocSide"].notna()
            pt_2s_oz = pt_2s[pt_2s_oz_m]
            pt_2s_ch = pt_2s_oz[pt_2s_oz["PitchCall"].isin(SWING_CALLS)]
            pt_2s_ip = pt_2s[(pt_2s["PitchCall"] == "InPlay") & pt_2s["ExitSpeed"].notna()]
            ts_rows.append({
                "Pitch": pt, "N": len(pt_2s),
                "Whiff%": round(len(pt_2s_wh) / max(len(pt_2s_sw), 1) * 100, 1) if len(pt_2s_sw) > 0 else 0,
                "Chase%": round(len(pt_2s_ch) / max(len(pt_2s_oz), 1) * 100, 1) if len(pt_2s_oz) > 0 else 0,
                "Avg EV": round(pt_2s_ip["ExitSpeed"].mean(), 1) if len(pt_2s_ip) >= 2 else np.nan,
            })
        if ts_rows:
            ts_df = pd.DataFrame(ts_rows).sort_values("Whiff%", ascending=False)
            st.dataframe(ts_df, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough two-strike data.")

    # ═══════════════════════════════════════════
    # SECTION E: Swing Timing & Quality
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Swing Quality & Contact")

    # EV by zone location heatmap
    ip_loc = batted.dropna(subset=["PlateLocSide", "PlateLocHeight", "ExitSpeed"])
    if len(ip_loc) >= 15:
        st.markdown("**Exit Velocity by Location**")
        st.caption("Where does this hitter generate the most damage?")
        h_edges_ev = np.linspace(-1.5, 1.5, 6)
        v_edges_ev = np.linspace(0.5, 4.5, 6)
        ev_grid = np.full((5, 5), np.nan)
        for vi in range(5):
            for hi in range(5):
                mask = (
                    (ip_loc["PlateLocSide"] >= h_edges_ev[hi]) &
                    (ip_loc["PlateLocSide"] < h_edges_ev[hi + 1]) &
                    (ip_loc["PlateLocHeight"] >= v_edges_ev[vi]) &
                    (ip_loc["PlateLocHeight"] < v_edges_ev[vi + 1])
                )
                cell_ip = ip_loc[mask]
                if len(cell_ip) >= 2:
                    ev_grid[vi, hi] = cell_ip["ExitSpeed"].mean()
        if is_rhh:
            h_labels_ev = ["In", "In-Mid", "Mid", "Away-Mid", "Away"]
        else:
            h_labels_ev = ["Away", "Away-Mid", "Mid", "In-Mid", "In"]
        v_labels_ev = ["Low", "Low-Mid", "Mid", "Up-Mid", "Up"]
        fig_ev_zone = px.imshow(
            np.flipud(ev_grid), text_auto=".0f",
            color_continuous_scale=[[0, "#3b82f6"], [0.5, "white"], [1, "#ef4444"]],
            x=h_labels_ev, y=list(reversed(v_labels_ev)),
            labels=dict(color="Avg EV"), aspect="auto",
            zmin=70, zmax=100,
        )
        _add_grid_zone_outline(fig_ev_zone)
        fig_ev_zone.update_layout(height=350, **CHART_LAYOUT)
        _plotly_chart_bats(fig_ev_zone, use_container_width=True, key="sdl_ev_zone")

    # Bat speed proxy by pitch type (if enough hard hit data)
    hard_hit = batted[batted["ExitSpeed"] >= 80].copy()
    if len(hard_hit) >= 10 and "RelSpeed" in hard_hit.columns:
        hard_hit["BatSpeedProxy"] = (hard_hit["ExitSpeed"] - 0.2 * hard_hit["RelSpeed"]) / 1.2
        bs_by_pt = hard_hit.groupby("TaggedPitchType")["BatSpeedProxy"].agg(["mean", "count"]).reset_index()
        bs_by_pt = bs_by_pt[bs_by_pt["count"] >= 3].sort_values("mean", ascending=False)
        if not bs_by_pt.empty:
            st.markdown("**Estimated Bat Speed by Pitch Type**")
            st.caption("Higher = squaring up; lower = late/fooled. Based on EV >= 80 mph contact.")
            fig_bs = go.Figure()
            for _, row_bs in bs_by_pt.iterrows():
                pc = PITCH_COLORS.get(row_bs["TaggedPitchType"], "#888")
                fig_bs.add_trace(go.Bar(
                    x=[row_bs["TaggedPitchType"]], y=[row_bs["mean"]],
                    marker_color=pc, text=[f'{row_bs["mean"]:.1f}'],
                    textposition="outside", name=row_bs["TaggedPitchType"],
                    showlegend=False,
                ))
            fig_bs.update_layout(
                height=300, yaxis_title="Est. Bat Speed (mph)", **CHART_LAYOUT,
                yaxis=dict(range=[min(bs_by_pt["mean"].min() - 5, 55), bs_by_pt["mean"].max() + 5]),
            )
            _plotly_chart_bats(fig_bs, use_container_width=True, key="sdl_bat_speed")

    # ═══════════════════════════════════════════
    # SECTION F: Actionable Recommendations
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Recommendations")

    recs = []

    # Chase-related recommendations
    if not pd.isna(chase_pct):
        # Overall chase rate
        behind_chase = behind_m["chase_pct"] if behind_m else 0
        ahead_chase = ahead_m["chase_pct"] if ahead_m else 0
        if behind_m and ahead_m and behind_chase > ahead_chase + 12:
            recs.append(
                f"Tighten the zone with 2 strikes \u2014 chase rate jumps from "
                f"{ahead_chase:.0f}% when ahead to {behind_chase:.0f}% when behind")

    # Pitch-type specific chasing
    for f_text in flags:
        if "Chases" in f_text:
            recs.append(f_text.replace("**", ""))

    # Zone passivity
    if zone_swing_pct < 62:
        # Find the pitch type they take the most in-zone
        best_take_pt = None
        best_take_rate = 0
        for row_pt in pt_rows:
            if row_pt["Z-Take%"] > best_take_rate and row_pt["Seen"] >= 10:
                best_take_rate = row_pt["Z-Take%"]
                best_take_pt = row_pt["Pitch"]
        if best_take_pt and best_take_rate > 30:
            ev_val = next((r["Avg EV"] for r in pt_rows if r["Pitch"] == best_take_pt), np.nan)
            ev_str = f", avg EV when swinging: {ev_val:.0f} mph" if not pd.isna(ev_val) else ""
            recs.append(
                f"Attack the in-zone {best_take_pt} more \u2014 taking {best_take_rate:.0f}% of "
                f"hittable {best_take_pt}s{ev_str}")

    # Whiff-related
    if whiff_pct > 30:
        worst_whiff_pt = max(pt_rows, key=lambda r: r["Whiff%"]) if pt_rows else None
        if worst_whiff_pt and worst_whiff_pt["Whiff%"] > 35:
            recs.append(
                f"High whiff rate on {worst_whiff_pt['Pitch']} ({worst_whiff_pt['Whiff%']:.0f}%) "
                f"\u2014 consider shortening the swing or looking for it earlier in the count")

    if not recs:
        recs.append("No major red flags \u2014 continue current approach and focus on consistency")

    for i, rec in enumerate(recs[:4]):
        rec_color = "#f59e0b" if i < len(flags) else "#3b82f6"
        st.markdown(
            f'<div style="padding:6px 14px;margin:3px 0;font-size:13px;'
            f'border-radius:6px;border-left:4px solid {rec_color};background:{rec_color}08;">'
            f'{rec}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE: HITTING (merged Hitter Card + Hitters Lab)
# ──────────────────────────────────────────────
def page_hitting(data):
    hitting = filter_davidson(data, "batter")
    if hitting.empty:
        st.warning("No hitting data found.")
        return

    batters = sorted(hitting["Batter"].unique())
    c1, c2 = st.columns([2, 3])
    with c1:
        batter = st.selectbox("Select Hitter", batters, format_func=display_name, key="hitting_batter")
    with c2:
        all_seasons = get_all_seasons()
        season_filter = st.multiselect("Season", all_seasons, default=all_seasons, key="hitting_season")

    bdf = hitting[(hitting["Batter"] == batter) & (hitting["Season"].isin(season_filter))]
    if len(bdf) < 20:
        st.warning("Not enough pitches (need 20+) to analyze.")
        return

    all_batter_stats = compute_batter_stats_pop(season_filter=season_filter)
    pr = None
    if all_batter_stats.empty:
        st.info("Population stats unavailable \u2014 showing team data only.")
        all_batter_stats = pd.DataFrame()
    else:
        if batter in all_batter_stats["Batter"].values:
            pr = all_batter_stats[all_batter_stats["Batter"] == batter].iloc[0]

    batted = bdf[bdf["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])

    if pr is None:
        pr_local = compute_batter_stats(bdf, season_filter=None)
        if not pr_local.empty:
            pr = pr_local.iloc[0]

    # Player header
    jersey = JERSEY.get(batter, "")
    pos = POSITION.get(batter, "")
    side = safe_mode(bdf["BatterSide"], "")
    bats = {"Right": "R", "Left": "L", "Switch": "S"}.get(side, side)
    _set_current_bats(bats)
    pa_val = int(_safe_pr(pr, "PA") or 0) if pr is not None and "PA" in pr else 0
    bbe_val = int(_safe_pr(pr, "BBE") or 0) if pr is not None and "BBE" in pr else 0
    player_header(batter, jersey, pos,
                  f"{pos}  |  Bats: {bats}  |  Davidson Wildcats",
                  f"{pa_val} PA  |  {bbe_val} Batted Balls  |  "
                  f"Seasons: {', '.join(str(int(s)) for s in sorted(season_filter))}")

    tab_card, tab_sdl = st.tabs(["Hitter Card", "Swing Decision Lab"])
    with tab_card:
        _hitter_card_content(data, batter, season_filter, bdf, batted, pr, all_batter_stats)
    with tab_sdl:
        _swing_decision_lab(data, batter, season_filter, bdf, batted, pr, all_batter_stats)


def _hitting_lab_content(data, batter, season_filter, bdf, batted, pr, all_batter_stats,
                         tab_quality, tab_discipline, tab_coverage, tab_approach,
                         tab_pitch_type, tab_spray, tab_swing):
    """Render the 7 Hitters Lab tabs. Called from page_hitting()."""
    side = safe_mode(bdf["BatterSide"], "")
    _iz_mask = in_zone_mask(bdf)
    out_zone_mask = ~_iz_mask & bdf["PlateLocSide"].notna() & bdf["PlateLocHeight"].notna()

    # ─── Tab 1: Batted Ball Quality ─────────────────────
    with tab_quality:
        section_header("Batted Ball Quality Grades")
        if len(batted) < 3:
            st.info("Not enough batted ball data (need 3+ BBE).")
        else:
            col_pct, col_scatter = st.columns([1, 2])
            with col_pct:
                metrics_data = []
                for label, col_name, fmt, hib in [
                    ("Avg EV", "AvgEV", ".1f", True), ("Max EV", "MaxEV", ".1f", True),
                    ("Barrel%", "BarrelPct", ".1f", True), ("Hard-Hit%", "HardHitPct", ".1f", True),
                    ("Sweet Spot%", "SweetSpotPct", ".1f", True), ("Avg LA", "AvgLA", ".1f", True),
                    ("K%", "KPct", ".1f", False), ("BB%", "BBPct", ".1f", True),
                ]:
                    val = pr.get(col_name, np.nan)
                    pct = get_percentile(val, all_batter_stats[col_name].dropna()) if not pd.isna(val) else np.nan
                    metrics_data.append((label, val, pct, fmt, hib))
                render_savant_percentile_section(metrics_data, title="Percentile Rankings vs All Hitters")

            with col_scatter:
                section_header("Exit Velocity vs Launch Angle")
                bp = batted.copy()
                conditions = [
                    is_barrel_mask(bp),
                    bp["ExitSpeed"] >= 95,
                    bp["ExitSpeed"].between(80, 95),
                ]
                bp["Quality"] = np.select(conditions, ["Barrel", "Hard-Hit", "Medium"], default="Weak")
                q_colors = {"Barrel": "#d22d49", "Hard-Hit": "#fe6100", "Medium": "#f7c631", "Weak": "#aaaaaa"}
                fig_ev = px.scatter(bp, x="Angle", y="ExitSpeed", color="Quality",
                                    color_discrete_map=q_colors,
                                    labels={"Angle": "Launch Angle", "ExitSpeed": "Exit Velocity (mph)"})
                _bz_ev2 = np.linspace(98, max(batted["ExitSpeed"].max() + 2, 105), 40)
                _bz_la_lo2 = np.clip(26 - 2 * (_bz_ev2 - 98), 8, 26)
                _bz_la_hi2 = np.clip(30 + 3 * (_bz_ev2 - 98), 30, 50)
                fig_ev.add_trace(go.Scatter(
                    x=np.concatenate([_bz_la_lo2, _bz_la_hi2[::-1]]),
                    y=np.concatenate([_bz_ev2, _bz_ev2[::-1]]),
                    fill="toself", fillcolor="rgba(210,45,73,0.08)",
                    line=dict(color="rgba(210,45,73,0.3)", width=1, dash="dash"),
                    showlegend=False, hoverinfo="skip",
                ))
                fig_ev.add_annotation(x=20, y=_bz_ev2[-1], text="Barrel Zone",
                                       font=dict(size=9, color="#d22d49"), showarrow=False)
                fig_ev.update_layout(**CHART_LAYOUT, height=400,
                                      legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10)))
                _plotly_chart_bats(fig_ev, use_container_width=True)

            col_ev_dist, col_la_dist = st.columns(2)
            with col_ev_dist:
                section_header("Exit Velocity Distribution")
                all_batted = query_population("SELECT ExitSpeed FROM trackman WHERE PitchCall='InPlay' AND ExitSpeed IS NOT NULL")
                fig_ev_hist = go.Figure()
                fig_ev_hist.add_trace(go.Histogram(
                    x=all_batted["ExitSpeed"], name="All Hitters",
                    marker_color="rgba(158,158,158,0.45)", nbinsx=30,
                    histnorm="probability density",
                ))
                fig_ev_hist.add_trace(go.Histogram(
                    x=batted["ExitSpeed"], name=display_name(batter),
                    marker_color="rgba(210,45,73,0.55)", nbinsx=25,
                    histnorm="probability density",
                ))
                fig_ev_hist.update_layout(
                    **CHART_LAYOUT, height=320, barmode="overlay",
                    xaxis_title="Exit Velocity (mph)", yaxis_title="Density",
                    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10)),
                )
                _plotly_chart_bats(fig_ev_hist, use_container_width=True)

            with col_la_dist:
                section_header("Launch Angle Distribution")
                la_bins = [(-90, -10, "Topped", "#d62728"), (-10, 8, "Ground Ball", "#ff7f0e"),
                           (8, 32, "Sweet Spot", "#2ca02c"), (32, 50, "Fly Ball", "#1f77b4"), (50, 90, "Popup", "#9467bd")]
                fig_la = go.Figure()
                for lo, hi, lbl, clr in la_bins:
                    subset = batted[batted["Angle"].between(lo, hi)]
                    fig_la.add_trace(go.Bar(x=[lbl], y=[len(subset)], name=lbl, marker_color=clr,
                                            text=[f"{len(subset)/len(batted)*100:.0f}%"], textposition="outside"))
                fig_la.update_layout(**CHART_LAYOUT, height=320, showlegend=False,
                                      yaxis_title="Count", xaxis_title="Launch Angle Zone", bargap=0.15)
                _plotly_chart_bats(fig_la, use_container_width=True)

            section_header("Expected Outcomes (EV/LA Model)")
            xo = _compute_expected_outcomes(batted)
            if xo:
                xo_cols = st.columns(6)
                for i, (k, lbl, clr) in enumerate([
                    ("xOut", "xOut%", "#9e9e9e"), ("x1B", "x1B%", "#2ca02c"), ("x2B", "x2B%", "#1f77b4"),
                    ("x3B", "x3B%", "#ff7f0e"), ("xHR", "xHR%", "#d22d49"), ("xwOBAcon", "xwOBAcon", "#6a0dad"),
                ]):
                    with xo_cols[i]:
                        val = xo.get(k, 0)
                        fmt_val = f"{val*100:.1f}%" if k != "xwOBAcon" else f"{val:.3f}"
                        st.markdown(
                            f'<div style="text-align:center;padding:12px;background:white;border-radius:8px;'
                            f'border:1px solid #eee;">'
                            f'<div style="font-size:24px;font-weight:800;color:{clr} !important;">{fmt_val}</div>'
                            f'<div style="font-size:11px;font-weight:600;color:#666 !important;text-transform:uppercase;">'
                            f'{lbl}</div></div>', unsafe_allow_html=True)

    # ─── Tab 2: Plate Discipline ────────────────────────
    with tab_discipline:
        section_header("Plate Discipline Overview")
        disc_metrics = []
        for label, col_name, fmt, hib in [
            ("Chase%", "ChasePct", ".1f", False), ("Whiff%", "WhiffPct", ".1f", False),
            ("K%", "KPct", ".1f", False), ("BB%", "BBPct", ".1f", True),
            ("Swing%", "SwingPct", ".1f", True), ("Z-Swing%", "ZoneSwingPct", ".1f", True),
            ("Z-Contact%", "ZoneContactPct", ".1f", True),
        ]:
            val = pr.get(col_name, np.nan)
            pct = get_percentile(val, all_batter_stats[col_name].dropna()) if not pd.isna(val) else np.nan
            disc_metrics.append((label, val, pct, fmt, hib))

        col_disc_pct, col_disc_grid = st.columns([1, 2])
        with col_disc_pct:
            render_savant_percentile_section(disc_metrics, title="Discipline Percentiles")
        with col_disc_grid:
            section_header("Swing Rate by Zone")
            grid_swing, annot_swing, h_labels, v_labels = _create_zone_grid_data(bdf, metric="swing_rate", batter_side=side)
            fig_grid = go.Figure(data=go.Heatmap(
                z=grid_swing, text=annot_swing, texttemplate="%{text}",
                x=h_labels, y=v_labels,
                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                zmin=0, zmax=100, showscale=True,
                colorbar=dict(title="Swing%", len=0.8),
            ))
            _add_grid_zone_outline(fig_grid)
            fig_grid.update_layout(**CHART_LAYOUT, height=380, xaxis=dict(side="bottom"))
            _plotly_chart_bats(fig_grid, use_container_width=True)

        col_ev_grid, col_chase = st.columns(2)
        with col_ev_grid:
            section_header("Avg EV by Zone")
            grid_ev, annot_ev, h_labels_ev, v_labels_ev = _create_zone_grid_data(bdf, metric="avg_ev", batter_side=side)
            fig_ev_grid = go.Figure(data=go.Heatmap(
                z=grid_ev, text=annot_ev, texttemplate="%{text}",
                x=h_labels_ev, y=v_labels_ev,
                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                zmin=60, zmax=100, showscale=True,
                colorbar=dict(title="EV", len=0.8),
            ))
            _add_grid_zone_outline(fig_ev_grid)
            fig_ev_grid.update_layout(**CHART_LAYOUT, height=380, xaxis=dict(side="bottom"))
            _plotly_chart_bats(fig_ev_grid, use_container_width=True)

        with col_chase:
            section_header("Chase Locations")
            chase_df = bdf[out_zone_mask].copy()
            if not chase_df.empty:
                chase_df["Outcome"] = "Taken"
                chase_df.loc[chase_df["PitchCall"] == "StrikeSwinging", "Outcome"] = "Swing & Miss"
                chase_df.loc[chase_df["PitchCall"].isin(["FoulBall", "FoulBallNotFieldable", "FoulBallFieldable"]), "Outcome"] = "Foul"
                chase_df.loc[chase_df["PitchCall"] == "InPlay", "Outcome"] = "In Play"
                chase_colors = {"Taken": "#aaaaaa", "Swing & Miss": "#d22d49", "Foul": "#f7c631", "In Play": "#2ca02c"}
                fig_chase = px.scatter(chase_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]),
                                        x="PlateLocSide", y="PlateLocHeight", color="Outcome",
                                        color_discrete_map=chase_colors, opacity=0.6,
                                        labels={"PlateLocSide": "Horizontal", "PlateLocHeight": "Vertical"})
                add_strike_zone(fig_chase)
                fig_chase.update_layout(**CHART_LAYOUT, height=380,
                                         xaxis=dict(range=[-2.5, 2.5], scaleanchor="y"), yaxis=dict(range=[0, 5]),
                                         legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10)))
                _plotly_chart_bats(fig_chase, use_container_width=True)
            else:
                st.info("No out-of-zone data available.")

        section_header("Pitch Recognition by Type")
        pt_disc_rows = []
        for pt in sorted(bdf["TaggedPitchType"].dropna().unique()):
            pt_df = bdf[bdf["TaggedPitchType"] == pt]
            if len(pt_df) < 5:
                continue
            pt_sw = pt_df[pt_df["PitchCall"].isin(SWING_CALLS)]
            pt_wh = pt_df[pt_df["PitchCall"] == "StrikeSwinging"]
            pt_ct = pt_df[pt_df["PitchCall"].isin(CONTACT_CALLS)]
            pt_oz = pt_df[(~in_zone_mask(pt_df)) & pt_df["PlateLocSide"].notna()]
            pt_ch = pt_oz[pt_oz["PitchCall"].isin(SWING_CALLS)]
            pt_disc_rows.append({
                "Pitch Type": pt, "Seen": len(pt_df),
                "Swing%": f"{len(pt_sw)/len(pt_df)*100:.1f}",
                "Whiff%": f"{len(pt_wh)/max(len(pt_sw),1)*100:.1f}" if len(pt_sw) > 0 else "-",
                "Contact%": f"{len(pt_ct)/max(len(pt_sw),1)*100:.1f}" if len(pt_sw) > 0 else "-",
                "Chase%": f"{len(pt_ch)/max(len(pt_oz),1)*100:.1f}" if len(pt_oz) > 0 else "-",
            })
        if pt_disc_rows:
            st.dataframe(pd.DataFrame(pt_disc_rows).set_index("Pitch Type"), use_container_width=True)

    # ─── Tab 3: Zone Coverage ───────────────────────────
    with tab_coverage:
        section_header("Zone Coverage Analysis")
        if len(batted) < 5:
            st.info("Not enough batted ball data for zone coverage analysis.")
        else:
            col_contact_hz, col_damage_hz = st.columns(2)
            with col_contact_hz:
                section_header("Contact Rate Heatmap")
                contact_loc = bdf[bdf["PitchCall"].isin(CONTACT_CALLS)].dropna(subset=["PlateLocSide", "PlateLocHeight"])
                if not contact_loc.empty:
                    fig_contact = go.Figure()
                    fig_contact.add_trace(go.Histogram2dContour(
                        x=contact_loc["PlateLocSide"], y=contact_loc["PlateLocHeight"],
                        colorscale=[[0, "rgba(255,255,255,0)"], [0.3, "#a8d5e2"], [0.6, "#f7c631"], [1, "#d22d49"]],
                        contours=dict(showlines=False), ncontours=15, showscale=False))
                    fig_contact.add_trace(go.Scatter(
                        x=contact_loc["PlateLocSide"], y=contact_loc["PlateLocHeight"],
                        mode="markers", marker=dict(size=4, color="#2ca02c", opacity=0.3),
                        showlegend=False, hoverinfo="skip"))
                    add_strike_zone(fig_contact)
                    fig_contact.update_layout(**CHART_LAYOUT, height=400,
                                               xaxis=dict(range=[-2.5, 2.5], title="Horizontal", scaleanchor="y"),
                                               yaxis=dict(range=[0, 5], title="Vertical"))
                    _plotly_chart_bats(fig_contact, use_container_width=True)

            with col_damage_hz:
                section_header("Damage Heatmap (Avg EV)")
                batted_loc = batted.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                if not batted_loc.empty:
                    fig_damage = go.Figure()
                    fig_damage.add_trace(go.Histogram2dContour(
                        x=batted_loc["PlateLocSide"], y=batted_loc["PlateLocHeight"],
                        z=batted_loc["ExitSpeed"], histfunc="avg",
                        colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                        contours=dict(showlines=False), ncontours=12, showscale=True,
                        colorbar=dict(title="Avg EV", len=0.8)))
                    barrel_loc = batted_loc[is_barrel_mask(batted_loc)]
                    if not barrel_loc.empty:
                        fig_damage.add_trace(go.Scatter(
                            x=barrel_loc["PlateLocSide"], y=barrel_loc["PlateLocHeight"],
                            mode="markers", marker=dict(size=12, color="#d22d49", symbol="star",
                                                         line=dict(width=1, color="white")),
                            name="Barrels", hovertemplate="EV: %{customdata[0]:.1f}<extra></extra>",
                            customdata=barrel_loc[["ExitSpeed"]].values))
                    add_strike_zone(fig_damage)
                    fig_damage.update_layout(**CHART_LAYOUT, height=400,
                                              xaxis=dict(range=[-2.5, 2.5], title="Horizontal", scaleanchor="y"),
                                              yaxis=dict(range=[0, 5], title="Vertical"),
                                              legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10)))
                    _plotly_chart_bats(fig_damage, use_container_width=True)

            col_whiff_hz, _ = st.columns(2)
            with col_whiff_hz:
                section_header("Whiff Zone Map")
                grid_whiff, annot_whiff, h_lbl, v_lbl = _create_zone_grid_data(bdf, metric="whiff_rate", batter_side=side)
                fig_wz = go.Figure(data=go.Heatmap(
                    z=grid_whiff, text=annot_whiff, texttemplate="%{text}",
                    x=h_lbl, y=v_lbl,
                    colorscale=[[0, "#2ca02c"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                    zmin=0, zmax=60, showscale=True,
                    colorbar=dict(title="Whiff%", len=0.8)))
                _add_grid_zone_outline(fig_wz)
                fig_wz.update_layout(**CHART_LAYOUT, height=380)
                _plotly_chart_bats(fig_wz, use_container_width=True)

            section_header("Damage by Pitch Type & Location")
            top_pts = [pt for pt in sorted(bdf["TaggedPitchType"].dropna().unique())
                       if len(bdf[bdf["TaggedPitchType"] == pt]) >= 10][:4]
            if top_pts:
                pt_cols = st.columns(len(top_pts))
                for idx, pt in enumerate(top_pts):
                    with pt_cols[idx]:
                        pt_b = bdf[(bdf["TaggedPitchType"] == pt) & (bdf["PitchCall"] == "InPlay")].dropna(
                            subset=["ExitSpeed", "PlateLocSide", "PlateLocHeight"])
                        st.markdown(f'<div style="text-align:center;font-weight:700;font-size:13px;color:#1a1a2e !important;'
                                    f'margin-bottom:4px;">{pt} ({len(pt_b)} BBE)</div>', unsafe_allow_html=True)
                        if len(pt_b) >= 3:
                            fig_pt = go.Figure()
                            fig_pt.add_trace(go.Scatter(
                                x=pt_b["PlateLocSide"], y=pt_b["PlateLocHeight"], mode="markers",
                                marker=dict(size=8, color=pt_b["ExitSpeed"],
                                            colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                            cmin=60, cmax=100,
                                            showscale=(idx == len(top_pts) - 1),
                                            colorbar=dict(title="EV", len=0.8) if idx == len(top_pts) - 1 else None,
                                            line=dict(width=0.5, color="white")),
                                hovertemplate="EV: %{marker.color:.1f}<extra></extra>"))
                            add_strike_zone(fig_pt)
                            fig_pt.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                                  font=dict(size=11, color="#000000", family="Inter, Arial, sans-serif"),
                                                  height=300,
                                                  xaxis=dict(range=[-2.5, 2.5], showticklabels=False,
                                                             scaleanchor="y", fixedrange=True),
                                                  yaxis=dict(range=[0, 5], showticklabels=False,
                                                             fixedrange=True),
                                                  margin=dict(l=5, r=5, t=5, b=5))
                            _plotly_chart_bats(fig_pt, use_container_width=True)
                        else:
                            st.caption("Not enough data")

    # ─── Tab 4: Approach Analysis ───────────────────────
    with tab_approach:
        section_header("Count-Based Approach")
        bdf_c = bdf.dropna(subset=["Balls", "Strikes"]).copy()
        bdf_c["Balls"] = bdf_c["Balls"].astype(int)
        bdf_c["Strikes"] = bdf_c["Strikes"].astype(int)

        count_grid_ev = np.full((4, 3), np.nan)
        count_grid_sw = np.full((4, 3), np.nan)
        annot_ev = [['' for _ in range(3)] for _ in range(4)]
        annot_sw = [['' for _ in range(3)] for _ in range(4)]
        for b in range(4):
            for s in range(3):
                cd = bdf_c[(bdf_c["Balls"] == b) & (bdf_c["Strikes"] == s)]
                if len(cd) < 3:
                    continue
                cb = cd[cd["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                cs = cd[cd["PitchCall"].isin(SWING_CALLS)]
                if len(cb) > 0:
                    ev_v = cb["ExitSpeed"].mean()
                    count_grid_ev[b, s] = ev_v
                    annot_ev[b][s] = f"{ev_v:.0f}"
                sw_v = len(cs) / len(cd) * 100
                count_grid_sw[b, s] = sw_v
                annot_sw[b][s] = f"{sw_v:.0f}%"

        col_evc, col_swc = st.columns(2)
        with col_evc:
            section_header("Avg EV by Count")
            fig_evc = go.Figure(data=go.Heatmap(
                z=count_grid_ev, text=annot_ev, texttemplate="%{text}",
                x=["0 Strikes", "1 Strike", "2 Strikes"], y=["0 Balls", "1 Ball", "2 Balls", "3 Balls"],
                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                zmin=70, zmax=100, showscale=True, colorbar=dict(title="EV", len=0.8)))
            fig_evc.update_layout(**CHART_LAYOUT, height=320)
            _plotly_chart_bats(fig_evc, use_container_width=True)
        with col_swc:
            section_header("Swing% by Count")
            fig_swc = go.Figure(data=go.Heatmap(
                z=count_grid_sw, text=annot_sw, texttemplate="%{text}",
                x=["0 Strikes", "1 Strike", "2 Strikes"], y=["0 Balls", "1 Ball", "2 Balls", "3 Balls"],
                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                zmin=0, zmax=100, showscale=True, colorbar=dict(title="Swing%", len=0.8)))
            fig_swc.update_layout(**CHART_LAYOUT, height=320)
            _plotly_chart_bats(fig_swc, use_container_width=True)

        col_fp, col_2k = st.columns(2)
        with col_fp:
            section_header("First Pitch Performance")
            fp = bdf[bdf["PitchofPA"] == 1] if "PitchofPA" in bdf.columns else pd.DataFrame()
            if not fp.empty and len(fp) >= 5:
                fp_sw = fp[fp["PitchCall"].isin(SWING_CALLS)]
                fp_wh = fp[fp["PitchCall"] == "StrikeSwinging"]
                fp_bt = fp[fp["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                for lbl, val in [("1st Pitch Swing%", len(fp_sw)/len(fp)*100),
                                  ("1st Pitch Whiff%", len(fp_wh)/max(len(fp_sw),1)*100 if len(fp_sw) > 0 else 0),
                                  ("1st Pitch Avg EV", fp_bt["ExitSpeed"].mean() if len(fp_bt) > 0 else np.nan),
                                  ("1st Pitch BBE", float(len(fp_bt)))]:
                    fv = f"{val:.1f}" if not pd.isna(val) else "-"
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:8px 12px;'
                        f'background:white;border-radius:6px;margin:4px 0;border:1px solid #eee;">'
                        f'<span style="font-size:12px;font-weight:600;color:#1a1a2e !important;">{lbl}</span>'
                        f'<span style="font-size:14px;font-weight:800;color:#d22d49 !important;">{fv}</span></div>',
                        unsafe_allow_html=True)
            else:
                st.info("Not enough first-pitch data.")

        with col_2k:
            section_header("2-Strike Adjustments")
            early = bdf_c[bdf_c["Strikes"] < 2]
            two_k = bdf_c[bdf_c["Strikes"] == 2]
            if len(early) >= 10 and len(two_k) >= 10:
                rows_2k = []
                for lbl, fn in [
                    ("Swing%", lambda d: len(d[d["PitchCall"].isin(SWING_CALLS)])/max(len(d),1)*100),
                    ("Whiff%", lambda d: len(d[d["PitchCall"]=="StrikeSwinging"])/max(len(d[d["PitchCall"].isin(SWING_CALLS)]),1)*100 if len(d[d["PitchCall"].isin(SWING_CALLS)])>0 else 0),
                ]:
                    ev = fn(early)
                    tv = fn(two_k)
                    rows_2k.append({"Metric": lbl, "<2 Strikes": f"{ev:.1f}%", "2 Strikes": f"{tv:.1f}%", "Change": f"{tv-ev:+.1f}%"})
                st.dataframe(pd.DataFrame(rows_2k).set_index("Metric"), use_container_width=True)
            else:
                st.info("Not enough data for 2-strike analysis.")

        section_header("At-Bat Length Outcomes")
        if "PitchofPA" in bdf.columns:
            pa_id_cols = [c for c in ["GameID", "PAofInning", "Inning", "Batter"] if c in bdf.columns]
            if len(pa_id_cols) >= 2:
                pa_lens = bdf.groupby(pa_id_cols)["PitchofPA"].max()
                length_rows = []
                for lo, hi, lbl in [(1, 3, "1-3 pitches"), (4, 6, "4-6 pitches"), (7, 20, "7+ pitches")]:
                    pa_ids = pa_lens[(pa_lens >= lo) & (pa_lens <= hi)]
                    n_pa = len(pa_ids)
                    if n_pa == 0:
                        continue
                    pa_sub = bdf.set_index(pa_id_cols).loc[pa_ids.index].reset_index()
                    ks = pa_sub[pa_sub["KorBB"] == "Strikeout"].drop_duplicates(subset=pa_id_cols) if "KorBB" in pa_sub.columns else pd.DataFrame()
                    bbs = pa_sub[pa_sub["KorBB"] == "Walk"].drop_duplicates(subset=pa_id_cols) if "KorBB" in pa_sub.columns else pd.DataFrame()
                    bbe = pa_sub[pa_sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                    length_rows.append({
                        "PA Length": lbl, "PA": n_pa,
                        "K%": f"{len(ks)/n_pa*100:.1f}%",
                        "BB%": f"{len(bbs)/n_pa*100:.1f}%",
                        "Avg EV": f"{bbe['ExitSpeed'].mean():.1f}" if len(bbe) > 0 else "-",
                    })
                if length_rows:
                    st.dataframe(pd.DataFrame(length_rows).set_index("PA Length"), use_container_width=True)

    # ─── Tab 5: Pitch Type Performance ──────────────────
    with tab_pitch_type:
        section_header("Performance by Pitch Type")
        pt_rows = []
        for pt in sorted(bdf["TaggedPitchType"].dropna().unique()):
            ptd = bdf[bdf["TaggedPitchType"] == pt]
            if len(ptd) < 5:
                continue
            pt_sw = ptd[ptd["PitchCall"].isin(SWING_CALLS)]
            pt_wh = ptd[ptd["PitchCall"] == "StrikeSwinging"]
            pt_ct = ptd[ptd["PitchCall"].isin(CONTACT_CALLS)]
            pt_bt = ptd[ptd["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            pt_br = pt_bt[is_barrel_mask(pt_bt)] if len(pt_bt) > 0 else pd.DataFrame()
            pt_rows.append({
                "Pitch Type": pt, "Seen": len(ptd),
                "Seen%": len(ptd)/len(bdf)*100,
                "Swing%": len(pt_sw)/len(ptd)*100,
                "Whiff%": len(pt_wh)/max(len(pt_sw),1)*100 if len(pt_sw) > 0 else 0,
                "Contact%": len(pt_ct)/max(len(pt_sw),1)*100 if len(pt_sw) > 0 else 0,
                "BBE": len(pt_bt),
                "Avg EV": pt_bt["ExitSpeed"].mean() if len(pt_bt) > 0 else np.nan,
                "Max EV": pt_bt["ExitSpeed"].max() if len(pt_bt) > 0 else np.nan,
                "Barrel%": len(pt_br)/max(len(pt_bt),1)*100 if len(pt_bt) > 0 else 0,
            })

        if pt_rows:
            pt_df = pd.DataFrame(pt_rows)
            disp = pt_df.copy()
            for c in ["Seen%", "Swing%", "Whiff%", "Contact%", "Barrel%"]:
                disp[c] = disp[c].map(lambda x: f"{x:.1f}%")
            for c in ["Avg EV", "Max EV"]:
                disp[c] = disp[c].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
            st.dataframe(disp.set_index("Pitch Type"), use_container_width=True)

            col_evb, col_whb = st.columns(2)
            with col_evb:
                section_header("Avg EV by Pitch Type")
                ch = pt_df.dropna(subset=["Avg EV"]).sort_values("Avg EV", ascending=True)
                if not ch.empty:
                    colors = [PITCH_COLORS.get(p, "#aaa") for p in ch["Pitch Type"]]
                    fig_pe = go.Figure(go.Bar(
                        y=ch["Pitch Type"], x=ch["Avg EV"], orientation="h", marker_color=colors,
                        text=ch["Avg EV"].map(lambda x: f"{x:.1f}"), textposition="outside"))
                    fig_pe.update_layout(**CHART_LAYOUT, height=max(200, len(ch)*40+60),
                                          xaxis_title="Avg Exit Velocity (mph)",
                                          xaxis=dict(range=[60, ch["Avg EV"].max()+8]))
                    _plotly_chart_bats(fig_pe, use_container_width=True)
            with col_whb:
                section_header("Whiff% by Pitch Type")
                ch2 = pt_df.sort_values("Whiff%", ascending=True)
                colors = [PITCH_COLORS.get(p, "#aaa") for p in ch2["Pitch Type"]]
                fig_pw = go.Figure(go.Bar(
                    y=ch2["Pitch Type"], x=ch2["Whiff%"], orientation="h", marker_color=colors,
                    text=ch2["Whiff%"].map(lambda x: f"{x:.1f}%"), textposition="outside"))
                fig_pw.update_layout(**CHART_LAYOUT, height=max(200, len(ch2)*40+60),
                                      xaxis_title="Whiff%", xaxis=dict(range=[0, max(ch2["Whiff%"].max()+10, 40)]))
                _plotly_chart_bats(fig_pw, use_container_width=True)

            section_header("Pitch Movement vs Damage")
            bwm = bdf[(bdf["PitchCall"] == "InPlay")].dropna(subset=["ExitSpeed", "HorzBreak", "InducedVertBreak"])
            if len(bwm) >= 5:
                fig_mv = px.scatter(bwm, x="HorzBreak", y="InducedVertBreak", color="ExitSpeed",
                                     size="ExitSpeed",
                                     color_continuous_scale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                     hover_data={"TaggedPitchType": True, "ExitSpeed": ":.1f"},
                                     labels={"HorzBreak": "Horizontal Break (in)", "InducedVertBreak": "Induced Vert Break (in)", "ExitSpeed": "EV"})
                fig_mv.update_layout(**CHART_LAYOUT, height=400)
                _plotly_chart_bats(fig_mv, use_container_width=True)
        else:
            st.info("Not enough pitch type data.")

    # ─── Tab 6: Spray Lab ──────────────────────────────
    with tab_spray:
        section_header("Spray Chart Analysis")
        in_play = bdf[bdf["PitchCall"] == "InPlay"].copy()
        if len(in_play) < 5:
            st.info("Not enough in-play data for spray analysis.")
        else:
            col_spray, col_table = st.columns([2, 1])
            with col_spray:
                spray_data = in_play.dropna(subset=["Direction", "Distance"]).copy()
                if not spray_data.empty:
                    angle_rad = np.radians(spray_data["Direction"])
                    spray_data["x"] = spray_data["Distance"] * np.sin(angle_rad)
                    spray_data["y"] = spray_data["Distance"] * np.cos(angle_rad)
                    fig_sp = go.Figure()
                    theta_g = np.linspace(-np.pi/4, np.pi/4, 80)
                    gr = 400
                    fig_sp.add_trace(go.Scatter(x=[0]+list(gr*np.sin(theta_g))+[0],
                                                y=[0]+list(gr*np.cos(theta_g))+[0], mode="lines",
                                                fill="toself", fillcolor="rgba(76,160,60,0.06)",
                                                line=dict(color="rgba(76,160,60,0.15)", width=1),
                                                showlegend=False, hoverinfo="skip"))
                    fig_sp.add_trace(go.Scatter(x=[0,63.6,0,-63.6,0], y=[0,63.6,127.3,63.6,0], mode="lines",
                                                line=dict(color="rgba(160,120,60,0.25)", width=1),
                                                fill="toself", fillcolor="rgba(160,120,60,0.06)",
                                                showlegend=False, hoverinfo="skip"))
                    fl = 350
                    for sx in [-1, 1]:
                        fig_sp.add_trace(go.Scatter(x=[0, sx*fl*np.sin(np.pi/4)], y=[0, fl*np.cos(np.pi/4)],
                                                    mode="lines", line=dict(color="rgba(0,0,0,0.12)", width=1),
                                                    showlegend=False, hoverinfo="skip"))
                    ev_vals = spray_data["ExitSpeed"].fillna(0)
                    fig_sp.add_trace(go.Scatter(
                        x=spray_data["x"], y=spray_data["y"], mode="markers",
                        marker=dict(size=8, color=ev_vals,
                                    colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                    cmin=60, cmax=100, showscale=True, colorbar=dict(title="EV", len=0.6),
                                    line=dict(width=0.5, color="white")),
                        hovertemplate="EV: %{marker.color:.1f} mph<br>Dist: %{customdata[0]:.0f}ft<extra></extra>",
                        customdata=spray_data[["Distance"]].values, showlegend=False))
                    fig_sp.update_layout(
                        xaxis=dict(range=[-300, 300], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
                        yaxis=dict(range=[-15, 400], showgrid=False, zeroline=False, showticklabels=False,
                                   scaleanchor="x", fixedrange=True),
                        height=450, margin=dict(l=0, r=0, t=5, b=0),
                        plot_bgcolor="white", paper_bgcolor="white")
                    _plotly_chart_bats(fig_sp, use_container_width=True)

            with col_table:
                section_header("Pull / Center / Oppo")
                bd = in_play.dropna(subset=["Direction", "ExitSpeed"]).copy()
                if not bd.empty:
                    bs = safe_mode(bdf["BatterSide"], "Right")
                    if bs == "Left":
                        pull_m = bd["Direction"] > 15
                        oppo_m = bd["Direction"] < -15
                    else:
                        pull_m = bd["Direction"] < -15
                        oppo_m = bd["Direction"] > 15
                    center_m = ~pull_m & ~oppo_m
                    dir_rows = []
                    for lbl, mask in [("Pull", pull_m), ("Center", center_m), ("Oppo", oppo_m)]:
                        sub = bd[mask]
                        ns = len(sub)
                        if ns == 0:
                            dir_rows.append({"Dir": lbl, "BBE": 0, "%": "-", "Avg EV": "-", "Max EV": "-"})
                            continue
                        gb_n = len(sub[sub["TaggedHitType"] == "GroundBall"])
                        ld_n = len(sub[sub["TaggedHitType"] == "LineDrive"])
                        fb_n = len(sub[sub["TaggedHitType"] == "FlyBall"])
                        brr = int(is_barrel_mask(sub).sum()) if sub["Angle"].notna().any() else 0
                        dir_rows.append({
                            "Dir": lbl, "BBE": ns,
                            "%": f"{ns/len(bd)*100:.0f}%",
                            "Avg EV": f"{sub['ExitSpeed'].mean():.1f}",
                            "Max EV": f"{sub['ExitSpeed'].max():.1f}",
                            "GB%": f"{gb_n/ns*100:.0f}%",
                            "LD%": f"{ld_n/ns*100:.0f}%",
                            "FB%": f"{fb_n/ns*100:.0f}%",
                            "Barrel": brr,
                        })
                    st.dataframe(pd.DataFrame(dir_rows).set_index("Dir"), use_container_width=True)

            col_la_dir, col_gb = st.columns(2)
            with col_la_dir:
                section_header("Launch Angle by Direction")
                bla = in_play.dropna(subset=["Direction", "Angle"]).copy()
                if not bla.empty:
                    bs = safe_mode(bdf["BatterSide"], "Right")
                    if bs == "Left":
                        bla["Field"] = np.where(bla["Direction"] > 15, "Pull",
                                                 np.where(bla["Direction"] < -15, "Oppo", "Center"))
                    else:
                        bla["Field"] = np.where(bla["Direction"] < -15, "Pull",
                                                 np.where(bla["Direction"] > 15, "Oppo", "Center"))
                    fig_ld = px.box(bla, x="Field", y="Angle",
                                     category_orders={"Field": ["Pull", "Center", "Oppo"]},
                                     color="Field", color_discrete_map={"Pull": "#d22d49", "Center": "#9e9e9e", "Oppo": "#1f77b4"},
                                     labels={"Angle": "Launch Angle", "Field": ""})
                    fig_ld.add_shape(type="rect", x0=-0.5, x1=2.5, y0=8, y1=32,
                                      fillcolor="rgba(29,190,58,0.08)", line=dict(width=0))
                    fig_ld.add_annotation(x=2.3, y=20, text="Sweet Spot", font=dict(size=9, color="#2ca02c"), showarrow=False)
                    fig_ld.update_layout(**CHART_LAYOUT, height=350, showlegend=False)
                    _plotly_chart_bats(fig_ld, use_container_width=True)
            with col_gb:
                section_header("GB% by Pitch Type")
                gb_rows = []
                for pt in sorted(bdf["TaggedPitchType"].dropna().unique()):
                    pt_ip = in_play[in_play["TaggedPitchType"] == pt]
                    if len(pt_ip) < 3:
                        continue
                    gb_n = len(pt_ip[pt_ip["TaggedHitType"] == "GroundBall"])
                    gb_rows.append({"Pitch Type": pt, "GB%": gb_n/len(pt_ip)*100, "BBE": len(pt_ip)})
                if gb_rows:
                    gdf = pd.DataFrame(gb_rows).sort_values("GB%", ascending=True)
                    colors = [PITCH_COLORS.get(p, "#aaa") for p in gdf["Pitch Type"]]
                    fig_gb = go.Figure(go.Bar(
                        y=gdf["Pitch Type"], x=gdf["GB%"], orientation="h", marker_color=colors,
                        text=gdf["GB%"].map(lambda x: f"{x:.0f}%"), textposition="outside"))
                    fig_gb.update_layout(**CHART_LAYOUT, height=max(200, len(gdf)*40+60),
                                          xaxis_title="Ground Ball %", xaxis=dict(range=[0, 100]))
                    _plotly_chart_bats(fig_gb, use_container_width=True)
