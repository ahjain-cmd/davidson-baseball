"""Generate a comprehensive postgame PDF report.

Produces a multi-page landscape PDF (11 x 8.5) with:
  1. Cover / Game Summary
  2. Umpire Report
  3. Pitching Staff Summary
  4..N. Individual Pitcher pages
  N+1. Hitting Lineup Summary
  N+2..M. Individual Hitter pages
  M+1. Coach Takeaways

All rendering uses matplotlib (no Streamlit dependency).
"""

import io
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from config import (
    DAVIDSON_TEAM_ID, JERSEY, POSITION,
    PITCH_COLORS, SWING_CALLS, CONTACT_CALLS,
    ZONE_SIDE, ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP,
    PLATE_SIDE_MAX, PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX,
    in_zone_mask, is_barrel_mask, display_name, normalize_pitch_types,
    filter_minor_pitches, _friendly_team_name,
)
from viz.percentiles import savant_color
from analytics.stuff_plus import _compute_stuff_plus
from analytics.command_plus import _compute_command_plus, _compute_pitch_pair_results
from analytics.expected import _create_zone_grid_data
from analytics.zone_vulnerability import compute_zone_swing_metrics
from analytics.tunnel import _compute_tunnel_score

# Import computation helpers from postgame page (no Streamlit dependency)
from _pages.postgame import (
    _compute_pitcher_grades,
    _compute_hitter_grades,
    _compute_pitcher_percentile_metrics,
    _compute_hitter_percentile_metrics,
    _compute_historical_stuff_cmd_distributions,
    _grade_at_bat,
    _letter_grade,
    _grade_color,
    _pg_estimate_ip,
    _pg_build_pa_pitch_rows,
    _score_linear,
    _compute_takeaways,
    _split_feedback,
    _compute_call_grade,
    _ZONE_X_EDGES,
    _ZONE_Y_EDGES,
    _MIN_PITCHER_PITCHES,
    _MIN_HITTER_PAS,
)

# ── Style Constants ──────────────────────────────────────────────────────────
_HDR_BG = "#1a1a2e"
_SECTION_BG = "#2c3e50"
_ALT_ROW = "#f0f3f7"
_WHITE = "#ffffff"
_DARK = "#1a1a2e"
_FIG_SIZE = (11, 8.5)


# ── Shared Helpers ───────────────────────────────────────────────────────────

def _header_bar(fig, gs_slot, text):
    """Draw a dark header bar spanning the given gridspec slot."""
    ax = fig.add_subplot(gs_slot)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_facecolor(_HDR_BG)
    ax.patch.set_alpha(1.0)
    ax.text(0.5, 0.5, text, fontsize=13, fontweight="bold",
            color="white", va="center", ha="center", transform=ax.transAxes)
    return ax


def _draw_zone_rect(ax):
    """Draw the strike zone rectangle."""
    ax.add_patch(Rectangle(
        (-ZONE_SIDE, ZONE_HEIGHT_BOT),
        2 * ZONE_SIDE, ZONE_HEIGHT_TOP - ZONE_HEIGHT_BOT,
        linewidth=1.5, edgecolor="#333", facecolor="none", zorder=5))


def _styled_table(ax, rows, col_labels, col_w, fontsize=7.5, row_height=1.4):
    """Create a styled table with alternating row colors."""
    cw_sum = sum(col_w)
    col_w = [w / cw_sum for w in col_w]
    tbl = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="upper center", cellLoc="center",
        colWidths=col_w,
        colColours=[_SECTION_BG] * len(col_labels))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, row_height)
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor(_SECTION_BG)
        cell.set_text_props(color="white", fontweight="bold", fontsize=fontsize - 0.5)
    for i in range(len(rows)):
        for j in range(len(col_labels)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(_ALT_ROW if i % 2 == 0 else _WHITE)
            cell.set_edgecolor("#ddd")
            if j == 0:
                cell.get_text().set_ha("left")
                cell.set_text_props(fontweight="bold", fontsize=fontsize - 0.5)
    return tbl


# ── Chart Helpers ────────────────────────────────────────────────────────────

def _mpl_radar_chart(fig, gs_slot, grades):
    """Polar axes filled radar chart with tier-based coloring."""
    valid = {k: v for k, v in grades.items() if v is not None}
    if len(valid) < 3:
        ax = fig.add_subplot(gs_slot)
        ax.axis("off")
        ax.text(0.5, 0.5, "Insufficient Data", fontsize=9, color="#999",
                ha="center", va="center", transform=ax.transAxes)
        return
    cats = list(valid.keys())
    vals = [valid[c] for c in cats]
    n = len(cats)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals_closed = vals + [vals[0]]
    angles_closed = angles + [angles[0]]

    fill_color = (0.13, 0.39, 0.68, 0.20)
    line_color = "#2163ae"

    ax = fig.add_subplot(gs_slot, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(cats, fontsize=7, color="#333")
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=6, color="#aaa")
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.5)
    ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.5)
    ax.spines["polar"].set_color("#e0e0e0")

    ax.fill(angles_closed, vals_closed, color=fill_color)
    ax.plot(angles_closed, vals_closed, color=line_color, linewidth=2)
    ax.scatter(angles, vals, color=line_color, s=20, zorder=10)


def _mpl_percentile_bars(ax, metrics):
    """Savant-style horizontal percentile bars — clean flat design."""
    n = len(metrics)
    if n == 0:
        ax.axis("off")
        return

    # Use 0-150 scale: 0-28 for label, 30-130 for bar track, 135-150 for value
    ax.set_xlim(0, 150)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axis("off")

    bar_h = 0.30
    track_h = 0.08
    track_left = 30
    track_right = 130
    track_w = track_right - track_left

    for idx, (label, val, pct, fmt, hib) in enumerate(reversed(metrics)):
        y = idx
        effective_pct = pct if hib else (100 - pct) if not pd.isna(pct) else None
        display_val = f"{val:{fmt}}" if not pd.isna(val) else "-"

        # Label (left, inside axes)
        ax.text(28, y, label.upper(), fontsize=6.5, fontweight="bold",
                color=_DARK, va="center", ha="right")
        # Value (right, inside axes)
        ax.text(135, y, display_val, fontsize=6.5, fontweight="bold",
                color=_DARK, va="center", ha="left")

        if effective_pct is None or pd.isna(pct):
            # N/A row — thin gray track with "N/A" text
            ax.barh(y, track_w, left=track_left, height=track_h,
                    color="#e0e0e0", zorder=1)
            ax.text(track_left + track_w / 2, y, "N/A", fontsize=7,
                    color="#999", va="center", ha="center", zorder=6)
            continue

        # Gray track background
        ax.barh(y, track_w, left=track_left, height=track_h,
                color="#e0e0e0", zorder=1)

        # Colored marker (rounded rect)
        color = savant_color(pct, hib)
        marker_w = 8
        marker_x = track_left + max(min(effective_pct, 100), 0) / 100 * track_w
        marker = FancyBboxPatch(
            (marker_x - marker_w / 2, y - bar_h / 2), marker_w, bar_h,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor=color, edgecolor="white", linewidth=1.2, zorder=5,
        )
        ax.add_patch(marker)
        ax.text(marker_x, y, f"{int(round(effective_pct))}",
                fontsize=7, fontweight="bold", color="white",
                va="center", ha="center", zorder=6)


def _mpl_stuff_cmd_bars(ax, stuff_by_pt, cmd_df,
                        season_stuff_by_pt=None, season_cmd_by_pt=None):
    """Horizontal grouped bar chart of Stuff+ and Command+ by pitch type.

    When season distributions provided with ≥10 data points, shows percentiles
    vs own history (0-100 scale, 50th ref line). Otherwise raw scores.
    """
    pitch_types = sorted(stuff_by_pt.keys())
    cmd_map = {}
    if cmd_df is not None and not cmd_df.empty and "Pitch" in cmd_df.columns and "Command+" in cmd_df.columns:
        cmd_map = dict(zip(cmd_df["Pitch"], cmd_df["Command+"]))
    if not pitch_types:
        ax.axis("off")
        return

    stuff_vals = [stuff_by_pt.get(pt, 100) for pt in pitch_types]
    cmd_vals = [cmd_map.get(pt, 100) for pt in pitch_types]
    y = np.arange(len(pitch_types))
    bar_h = 0.35

    # Determine if we can use historical percentiles
    use_pctl = False
    if season_stuff_by_pt and season_cmd_by_pt:
        has_stuff = any(len(season_stuff_by_pt.get(pt, [])) >= 10 for pt in pitch_types)
        has_cmd = any(len(season_cmd_by_pt.get(pt, [])) >= 10 for pt in pitch_types)
        use_pctl = has_stuff and has_cmd

    if use_pctl:
        display_stuff = []
        display_cmd = []
        for pt, sv, cv in zip(pitch_types, stuff_vals, cmd_vals):
            hist_s = season_stuff_by_pt.get(pt)
            hist_c = season_cmd_by_pt.get(pt)
            if hist_s is not None and len(hist_s) >= 10:
                display_stuff.append(percentileofscore(hist_s, sv, kind="rank"))
            else:
                display_stuff.append(50.0)
            if hist_c is not None and len(hist_c) >= 10:
                display_cmd.append(percentileofscore(hist_c, cv, kind="rank"))
            else:
                display_cmd.append(50.0)
        x_lim = (0, 108)
        ref_line = 50
        x_label = "Percentile"
        title_text = "Stuff+ / Command+ (vs Own History)"
    else:
        display_stuff = stuff_vals
        display_cmd = cmd_vals
        x_lim = (50, max(max(stuff_vals + cmd_vals, default=100) + 15, 130))
        ref_line = 100
        x_label = "Score"
        title_text = "Stuff+ / Command+"

    def _bar_color_pctl(v):
        if v >= 75:
            return "#d22d49"
        if v <= 25:
            return "#2d7fc1"
        return "#9e9e9e"

    def _bar_color_raw(v):
        if v > 110:
            return "#d22d49"
        if v < 90:
            return "#2d7fc1"
        return "#9e9e9e"

    _bar_color = _bar_color_pctl if use_pctl else _bar_color_raw

    ax.barh(y + bar_h / 2, display_stuff, bar_h, label="Stuff+",
            color=[_bar_color(v) for v in display_stuff])
    ax.barh(y - bar_h / 2, display_cmd, bar_h, label="Command+",
            color=[_bar_color(v) for v in display_cmd], alpha=0.7)
    ax.axvline(ref_line, color="#aaa", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(pitch_types, fontsize=7)
    ax.set_xlim(*x_lim)
    ax.set_xlabel(x_label, fontsize=7)
    ax.legend(fontsize=6, loc="lower right")
    ax.tick_params(axis="x", labelsize=6)
    for i, (sv, cv) in enumerate(zip(display_stuff, display_cmd)):
        ax.text(sv + 1, i + bar_h / 2, f"{sv:.0f}", fontsize=6, va="center")
        ax.text(cv + 1, i - bar_h / 2, f"{cv:.0f}", fontsize=6, va="center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title_text, fontsize=8, fontweight="bold", color=_DARK, loc="left", pad=3)


def _mpl_grade_header(ax, name, n_pitches, overall=None, label_extra="", small_sample=False):
    """Player name banner."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor("#000000")
    jersey = JERSEY.get(name, "")
    pos = POSITION.get(name, "")
    dname = display_name(name, escape_html=False)
    extra = f" | {label_extra}" if label_extra else ""
    jersey_str = f"#{jersey}  " if jersey else ""
    ax.text(0.03, 0.55, f"{jersey_str}{dname}", fontsize=18, fontweight="900",
            color="white", va="center", ha="left", transform=ax.transAxes,
            clip_on=False)
    ax.text(0.03, 0.18, f"{pos} | {n_pitches} pitches{extra}",
            fontsize=9, color="#aaa", va="center", ha="left", transform=ax.transAxes,
            clip_on=False)


def _mpl_grade_cards(ax, grades):
    """Placeholder — grade cards removed."""
    ax.axis("off")


def _mpl_zone_scatter(ax, pdf):
    """Pitch location scatter colored by pitch type with zone rectangle."""
    loc = pdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(0.5, 4.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#ddd")
    _draw_zone_rect(ax)
    if not loc.empty and "TaggedPitchType" in loc.columns:
        for pt in loc["TaggedPitchType"].unique():
            sub = loc[loc["TaggedPitchType"] == pt]
            ax.scatter(sub["PlateLocSide"], sub["PlateLocHeight"],
                       c=PITCH_COLORS.get(pt, "#aaa"), s=18, alpha=0.8,
                       edgecolors="white", linewidths=0.3, zorder=10, label=pt)
    ax.legend(fontsize=5, loc="upper right", framealpha=0.7)
    ax.set_title("Pitch Locations", fontsize=8, fontweight="bold", color=_DARK, pad=3)


def _mpl_pa_zone_plot(ax, ab_df):
    """Numbered pitch locations for a single PA, colored by pitch type.

    Pitch numbers match the original PA sequence (1-based), so numbers
    stay consistent with the pitch table even when some pitches lack location.
    """
    # Assign original sequence numbers BEFORE filtering
    ab_numbered = ab_df.copy()
    ab_numbered["_OrigPitchNum"] = range(1, len(ab_numbered) + 1)
    loc = ab_numbered.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(0.5, 4.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#ddd")
    _draw_zone_rect(ax)
    if loc.empty:
        ax.text(0.5, 0.5, "No location\ndata", fontsize=8, color="#999",
                ha="center", va="center", transform=ax.transAxes)
        return
    for _, row in loc.iterrows():
        pnum = int(row["_OrigPitchNum"])
        pt = row.get("TaggedPitchType", "Other")
        color = PITCH_COLORS.get(pt, "#aaa")
        ax.scatter(row["PlateLocSide"], row["PlateLocHeight"],
                   c=color, s=120, alpha=0.85, edgecolors="white",
                   linewidths=0.5, zorder=10)
        ax.text(row["PlateLocSide"], row["PlateLocHeight"], str(pnum),
                fontsize=6, fontweight="bold", color="white",
                ha="center", va="center", zorder=11)


def _mpl_movement_profile(ax, pdf):
    """IVB x HB scatter with concentric circles."""
    mov = pdf.dropna(subset=["HorzBreak", "InducedVertBreak"])
    mov = filter_minor_pitches(mov)
    if mov.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No movement data", fontsize=8, color="#999",
                ha="center", va="center", transform=ax.transAxes)
        return
    mov = mov.copy()
    # Catcher-view: negate HB for RHP
    throws = mov.get("PitcherThrows")
    if throws is not None and not throws.dropna().empty:
        is_lhp = throws.astype(str).str.lower().str.startswith("l")
        hb_sign = np.where(is_lhp, 1.0, -1.0)
    else:
        hb_sign = -1.0
    mov["_HB"] = mov["HorzBreak"].astype(float) * hb_sign
    mov["_IVB"] = mov["InducedVertBreak"].astype(float)

    for r in [6, 12, 18, 24]:
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="#d0d8e0", linewidth=0.7)
        ax.annotate(f'{r}"', xy=(0, r), fontsize=6, color="#999", ha="center", va="bottom")
    ax.axhline(0, color="#bbb", linewidth=0.5)
    ax.axvline(0, color="#bbb", linewidth=0.5)
    for pt in sorted(mov["TaggedPitchType"].unique()):
        sub = mov[mov["TaggedPitchType"] == pt]
        color = PITCH_COLORS.get(pt, "#aaa")
        ax.scatter(sub["_HB"], sub["_IVB"], c=color, s=18, alpha=0.8,
                   edgecolors="white", linewidths=0.3, zorder=10, label=pt)
    lim = 28
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal Break (in)", fontsize=6)
    ax.set_ylabel("Induced Vert Break (in)", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.legend(fontsize=5, loc="lower right", framealpha=0.7)
    ax.set_title("Movement Profile", fontsize=8, fontweight="bold", color=_DARK, pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _mpl_spray_chart(ax, in_play_df):
    """Direction/Distance spray chart with field outline."""
    spray = in_play_df.dropna(subset=["Direction", "Distance"]).copy()
    spray = spray[spray["Direction"].between(-90, 90)]
    if spray.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No spray data", fontsize=8, color="#999",
                ha="center", va="center", transform=ax.transAxes)
        return
    angle_rad = np.radians(spray["Direction"])
    spray["x"] = spray["Distance"] * np.sin(angle_rad)
    spray["y"] = spray["Distance"] * np.cos(angle_rad)

    # Grass fill
    theta = np.linspace(-np.pi / 4, np.pi / 4, 80)
    grass_r = 400
    grass_x = np.concatenate([[0], grass_r * np.sin(theta), [0]])
    grass_y = np.concatenate([[0], grass_r * np.cos(theta), [0]])
    ax.fill(grass_x, grass_y, color="rgba(76,160,60,0.06)" if False else "#e8f5e3",
            alpha=0.4, zorder=0)

    # Diamond
    diamond_x = [0, 63.6, 0, -63.6, 0]
    diamond_y = [0, 63.6, 127.3, 63.6, 0]
    ax.fill(diamond_x, diamond_y, color="#f5e6d0", alpha=0.3, zorder=1)
    ax.plot(diamond_x, diamond_y, color="#c8a872", linewidth=0.7, zorder=2)

    # Foul lines
    fl = 350
    ax.plot([0, -fl * np.sin(np.pi / 4)], [0, fl * np.cos(np.pi / 4)],
            color="#ccc", linewidth=0.5, zorder=1)
    ax.plot([0, fl * np.sin(np.pi / 4)], [0, fl * np.cos(np.pi / 4)],
            color="#ccc", linewidth=0.5, zorder=1)

    ht_colors = {"GroundBall": "#d62728", "LineDrive": "#2ca02c",
                 "FlyBall": "#1f77b4", "Popup": "#ff7f0e"}
    ht_names = {"GroundBall": "GB", "LineDrive": "LD",
                "FlyBall": "FB", "Popup": "PU"}
    for ht in ["GroundBall", "LineDrive", "FlyBall", "Popup"]:
        sub = spray[spray.get("TaggedHitType", pd.Series(dtype=str)) == ht] if "TaggedHitType" in spray.columns else pd.DataFrame()
        if sub.empty:
            continue
        ax.scatter(sub["x"], sub["y"], c=ht_colors.get(ht, "#777"),
                   s=18, alpha=0.8, edgecolors="white", linewidths=0.3,
                   label=ht_names.get(ht, ht), zorder=10)
    # Other hits
    if "TaggedHitType" in spray.columns:
        other = spray[~spray["TaggedHitType"].isin(["GroundBall", "LineDrive", "FlyBall", "Popup"])]
        if not other.empty:
            ax.scatter(other["x"], other["y"], c="#bbb", s=12, alpha=0.5, zorder=5)

    ax.set_xlim(-300, 300)
    ax.set_ylim(-15, 400)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(fontsize=5, loc="upper right", framealpha=0.7)
    ax.set_title("Spray Chart", fontsize=8, fontweight="bold", color=_DARK, pad=3)


def _mpl_damage_heatmap(ax, grid, annot, h_labels, v_labels, game_swings_xy=None):
    """Heatmap with diamond markers for game swings."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dmg",
        ["#2d7fc1", "#f7f7f7", "#d22d49"])
    masked = np.ma.masked_invalid(grid)
    ax.imshow(masked, aspect="auto", cmap=cmap, origin="upper",
              vmin=np.nanmin(grid) if np.any(np.isfinite(grid)) else 0,
              vmax=np.nanmax(grid) if np.any(np.isfinite(grid)) else 100)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            txt = annot[i][j] if i < len(annot) and j < len(annot[i]) else ""
            if txt:
                ax.text(j, i, txt, fontsize=6, ha="center", va="center", color=_DARK)
    ax.set_xticks(range(len(h_labels)))
    ax.set_xticklabels(h_labels, fontsize=5)
    ax.set_yticks(range(len(v_labels)))
    ax.set_yticklabels(v_labels, fontsize=5)
    ax.tick_params(axis="both", length=2, pad=1)
    # Strike zone outline (inner 3x3)
    ax.add_patch(Rectangle((0.5, 0.5), 3, 3, linewidth=2,
                            edgecolor="#333", facecolor="none", zorder=5))
    # Diamond markers for game swings
    if game_swings_xy:
        for xi, yi in game_swings_xy:
            ax.scatter(xi, yi, marker="D", s=50, facecolors="white",
                       edgecolors="#333", linewidths=1.5, zorder=10)
    ax.set_title("Damage Zones (Avg EV)", fontsize=7, fontweight="bold",
                 color=_DARK, pad=3)


def _mpl_best_zone_heatmap(ax, bdf, bats):
    """Matplotlib 3x3 best-zone heatmap for PDF hitter pages."""
    from scipy.stats import percentileofscore as _pctile

    loc_df = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if len(loc_df) < 30:
        ax.axis("off")
        ax.text(0.5, 0.5, "Not enough data", fontsize=7, color="#999",
                ha="center", va="center", transform=ax.transAxes)
        return

    zone_metrics = compute_zone_swing_metrics(bdf, bats)
    if zone_metrics is None:
        ax.axis("off")
        return

    # Gather raw values for percentile ranking
    ev_vals, barrel_vals, whiff_vals = [], [], []
    for yb in range(3):
        for xb in range(3):
            m = zone_metrics.get((xb, yb), {})
            ev_vals.append(m.get("ev_mean", np.nan))
            barrel_vals.append(m.get("barrel_pct", np.nan))
            zdf = loc_df[(np.clip(np.digitize(loc_df["PlateLocSide"], _ZONE_X_EDGES) - 1, 0, 2) == xb) &
                         (np.clip(np.digitize(loc_df["PlateLocHeight"], _ZONE_Y_EDGES) - 1, 0, 2) == yb)]
            swings = zdf[zdf["PitchCall"].isin(SWING_CALLS)] if "PitchCall" in zdf.columns else pd.DataFrame()
            whiffs = zdf[zdf["PitchCall"] == "StrikeSwinging"] if "PitchCall" in zdf.columns else pd.DataFrame()
            whiff_pct = len(whiffs) / max(len(swings), 1) * 100 if len(swings) >= 3 else np.nan
            whiff_vals.append(whiff_pct)

    ev_arr, barrel_arr, whiff_arr = np.array(ev_vals), np.array(barrel_vals), np.array(whiff_vals)

    def _prank(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return np.full_like(arr, 50.0)
        return np.array([_pctile(valid, v, kind="rank") if not np.isnan(v) else np.nan for v in arr])

    ev_p, barrel_p, whiff_p = _prank(ev_arr), _prank(barrel_arr), _prank(whiff_arr)

    grid = np.full((3, 3), np.nan)
    idx = 0
    for yb in range(3):
        for xb in range(3):
            parts, total_w = [], 0.0
            if not np.isnan(ev_p[idx]):
                parts.append(0.45 * ev_p[idx]); total_w += 0.45
            if not np.isnan(barrel_p[idx]):
                parts.append(0.30 * barrel_p[idx]); total_w += 0.30
            if not np.isnan(whiff_p[idx]):
                parts.append(0.25 * (100 - whiff_p[idx])); total_w += 0.25
            grid[2 - yb, xb] = sum(parts) / total_w if total_w > 0 and parts else np.nan
            idx += 1

    cmap = plt.cm.RdYlGn
    masked = np.ma.masked_invalid(grid)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Text overlay
    idx = 0
    for yb in range(3):
        for xb in range(3):
            m = zone_metrics.get((xb, yb), {})
            r, c = 2 - yb, xb
            v = grid[r, c]
            if not np.isnan(v):
                ev_str = f"{m.get('ev_mean', 0):.0f}" if m.get("n_contact", 0) >= 5 and "ev_mean" in m else ""
                bp_str = f"{m.get('barrel_pct', 0):.0f}%" if m.get("n_contact", 0) >= 5 and "barrel_pct" in m else ""
                txt = f"{ev_str}\n{bp_str}" if ev_str and bp_str else ev_str or bp_str
                brightness = v / 100.0
                color = "white" if brightness > 0.7 or brightness < 0.15 else "black"
                ax.text(c, r, txt, ha="center", va="center", fontsize=6,
                        fontweight="bold", color=color)
            idx += 1

    # Zone styling
    for x in [0.5, 1.5]:
        ax.axvline(x, color="white", lw=1.5, zorder=2)
    for y in [0.5, 1.5]:
        ax.axhline(y, color="white", lw=1.5, zorder=2)
    ax.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False,
                            edgecolor="#555", lw=1.2, ls="--", zorder=3))
    ax.add_patch(Rectangle((0.17, -0.17), 1.66, 2.0, fill=False,
                            edgecolor="black", lw=2.5, zorder=4))
    ax.set_yticks([0, 2])
    ax.set_yticklabels(["UP", "DOWN"], fontsize=6, fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=2)
    b = bats[0].upper() if isinstance(bats, str) and bats else "R"
    if b == "R":
        left_lbl, right_lbl = "IN", "AWAY"
    elif b == "L":
        left_lbl, right_lbl = "AWAY", "IN"
    else:
        left_lbl, right_lbl = "L", "R"
    ax.set_xticks([0, 2])
    ax.set_xticklabels([left_lbl, right_lbl], fontsize=5.5, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_title("Best Hitting Zones", fontsize=7, fontweight="bold", color=_DARK, pad=3)


def _mpl_call_grade_box(ax, grade_info):
    """Compact call grade box for PDF pitcher pages."""
    ax.axis("off")
    if grade_info is None:
        ax.text(0.5, 0.5, "N/A", fontsize=8, color="#999",
                ha="center", va="center", transform=ax.transAxes)
        return

    # Grade letter
    gl = grade_info["grade_letter"]
    gc = _grade_color(gl)
    ax.text(0.05, 0.92, f"Call Grade: {gl}", fontsize=9, fontweight="bold",
            color=gc, va="top", ha="left", transform=ax.transAxes)
    ax.text(0.05, 0.78,
            f"Seq Util: {grade_info['seq_util_score']:.0f}  |  Loc Exec: {grade_info['loc_exec_score']:.0f}",
            fontsize=6.5, color=_DARK, va="top", ha="left", transform=ax.transAxes)

    # Top pairs
    y = 0.65
    if grade_info["top_pairs"]:
        ax.text(0.05, y, "Top Pairs:", fontsize=6.5, fontweight="bold",
                color=_DARK, va="top", ha="left", transform=ax.transAxes)
        y -= 0.10
        for p in grade_info["top_pairs"][:2]:
            whiff = f"{p.get('Whiff%', 0):.0f}%" if pd.notna(p.get("Whiff%")) else "-"
            ax.text(0.08, y, f"{p.get('Pair','?')}  Whiff:{whiff}",
                    fontsize=6, color=_DARK, va="top", ha="left", transform=ax.transAxes)
            y -= 0.09

    # Best locations
    if grade_info["best_locations"]:
        ax.text(0.05, y, "Best Loc:", fontsize=6.5, fontweight="bold",
                color=_DARK, va="top", ha="left", transform=ax.transAxes)
        y -= 0.09
        for pt, loc in list(grade_info["best_locations"].items())[:4]:
            ax.text(0.08, y, f"{pt}: {loc}", fontsize=5.5, color=_DARK,
                    va="top", ha="left", transform=ax.transAxes)
            y -= 0.08


def _mpl_ab_grades_table(ax, ab_rows):
    """At-bat grades as a styled table."""
    if not ab_rows:
        ax.axis("off")
        return
    col_labels = ["Inn", "vs Pitcher", "P", "Result", "Score", "Grade"]
    rows = []
    for r in ab_rows:
        rows.append([
            str(r.get("Inning", "?")),
            str(r.get("vs Pitcher", "?"))[:18],
            str(r.get("Pitches", "?")),
            str(r.get("Result", "?")),
            str(r.get("Score", "?")),
            str(r.get("Grade", "?")),
        ])
    ax.axis("off")
    _styled_table(ax, rows, col_labels,
                  [0.08, 0.30, 0.07, 0.15, 0.12, 0.12],
                  fontsize=6, row_height=1.2)
    ax.set_title("At-Bat Grades", fontsize=7, fontweight="bold",
                 color=_DARK, loc="left", pad=2)


def _mpl_feedback(ax, feedback):
    """Feedback bullets split into Strengths / Areas to Improve."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if not feedback:
        ax.text(0.02, 0.85, "No specific notes.", fontsize=7, color="#999",
                va="top", ha="left", transform=ax.transAxes)
        return
    fb_str, fb_area = _split_feedback(feedback, {})
    y = 0.95
    if fb_str:
        ax.text(0.02, y, "Strengths", fontsize=8, fontweight="bold",
                color="#2e7d32", va="top", ha="left", transform=ax.transAxes)
        y -= 0.08
        for fb in fb_str[:3]:
            ax.text(0.04, y, f"+ {fb}", fontsize=6.5, va="top", ha="left",
                    color=_DARK, transform=ax.transAxes, wrap=True)
            y -= 0.12
    if fb_area:
        ax.text(0.02, y, "Areas to Improve", fontsize=8, fontweight="bold",
                color="#c62828", va="top", ha="left", transform=ax.transAxes)
        y -= 0.08
        for fb in fb_area[:3]:
            ax.text(0.04, y, f"- {fb}", fontsize=6.5, va="top", ha="left",
                    color=_DARK, transform=ax.transAxes, wrap=True)
            y -= 0.12
    if not fb_str and not fb_area:
        text = "\n".join(f"~ {fb}" for fb in feedback[:6])
        ax.text(0.02, 0.92, "Feedback", fontsize=8, fontweight="bold",
                color=_DARK, va="top", ha="left", transform=ax.transAxes)
        ax.text(0.02, 0.78, text, fontsize=6.5, va="top", ha="left",
                color=_DARK, transform=ax.transAxes, linespacing=1.6,
                wrap=True)


# ── Page Renderers ───────────────────────────────────────────────────────────

def _render_cover_page(gd, game_label):
    """Cover / Game Summary page."""
    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(3, 1, figure=fig,
        height_ratios=[0.10, 0.25, 0.65],
        hspace=0.06, top=0.97, bottom=0.03, left=0.03, right=0.97)

    _header_bar(fig, outer[0], f"POSTGAME REPORT  |  {game_label}")

    # Matchup info
    ax_info = fig.add_subplot(outer[1])
    ax_info.axis("off")

    home_id = gd["HomeTeam"].iloc[0] if "HomeTeam" in gd.columns else "?"
    away_id = gd["AwayTeam"].iloc[0] if "AwayTeam" in gd.columns else "?"
    home = _friendly_team_name(home_id)
    away = _friendly_team_name(away_id)
    total_pitches = len(gd)
    innings = int(gd["Inning"].max()) if "Inning" in gd.columns else "?"

    dav_pitching = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID]
    dav_hitting = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID]

    dav_ks = (dav_pitching["KorBB"] == "Strikeout").sum() if "KorBB" in dav_pitching.columns else 0
    dav_bbs = (dav_pitching["KorBB"] == "Walk").sum() if "KorBB" in dav_pitching.columns else 0
    dav_hits_allowed = dav_pitching["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum() if "PlayResult" in dav_pitching.columns else 0

    dav_hits = dav_hitting["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum() if "PlayResult" in dav_hitting.columns else 0
    dav_hr = (dav_hitting["PlayResult"] == "HomeRun").sum() if "PlayResult" in dav_hitting.columns else 0
    dav_hit_bbs = (dav_hitting["KorBB"] == "Walk").sum() if "KorBB" in dav_hitting.columns else 0
    dav_hit_ks = (dav_hitting["KorBB"] == "Strikeout").sum() if "KorBB" in dav_hitting.columns else 0

    dav_bbe = dav_hitting[(dav_hitting["PitchCall"] == "InPlay") & dav_hitting["ExitSpeed"].notna()] if "ExitSpeed" in dav_hitting.columns else pd.DataFrame()
    avg_ev = dav_bbe["ExitSpeed"].mean() if len(dav_bbe) > 0 else np.nan
    max_ev = dav_bbe["ExitSpeed"].max() if len(dav_bbe) > 0 else np.nan

    # Estimate game score from runs scored per inning
    _score_line = ""
    if "RunsScored" in gd.columns:
        dav_runs = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID]["RunsScored"].sum()
        opp_runs = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID]["RunsScored"].sum()
        dav_runs = int(dav_runs) if pd.notna(dav_runs) else 0
        opp_runs = int(opp_runs) if pd.notna(opp_runs) else 0
        _score_line = f"   |   Davidson {dav_runs} - {opp_runs} {away if away != 'Davidson' else home}"

    ax_info.text(0.5, 0.80, f"{away}  @  {home}", fontsize=24, fontweight="900",
                 color=_DARK, va="center", ha="center", transform=ax_info.transAxes)
    ax_info.text(0.5, 0.52, f"{innings} innings  |  {total_pitches} total pitches{_score_line}",
                 fontsize=12, color="#555", va="center", ha="center", transform=ax_info.transAxes)

    # Stat columns — 2x2 grid for better vertical fill
    stats_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[2],
        wspace=0.20, hspace=0.25)

    _stat_fontsize = 11

    # Top-left: Davidson Pitching
    ax1 = fig.add_subplot(stats_gs[0, 0])
    ax1.axis("off")
    ax1.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#f7f9fc", edgecolor="#dde3ea", linewidth=0.8,
        transform=ax1.transAxes, zorder=0))
    ax1.text(0.5, 0.92, "PITCHING", fontsize=10, fontweight="bold",
             color=_DARK, va="top", ha="center", transform=ax1.transAxes)
    pit_text = (
        f"Strikeouts:    {dav_ks}\n"
        f"Walks:         {dav_bbs}\n"
        f"Hits Allowed:  {dav_hits_allowed}\n"
        f"K/BB Ratio:    {dav_ks}/{dav_bbs}"
    )
    ax1.text(0.12, 0.72, pit_text, fontsize=_stat_fontsize, va="top", ha="left",
             color=_DARK, transform=ax1.transAxes, family="monospace", linespacing=2.0)

    # Top-right: Davidson Hitting
    ax2 = fig.add_subplot(stats_gs[0, 1])
    ax2.axis("off")
    ax2.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#f7f9fc", edgecolor="#dde3ea", linewidth=0.8,
        transform=ax2.transAxes, zorder=0))
    ax2.text(0.5, 0.92, "HITTING", fontsize=10, fontweight="bold",
             color=_DARK, va="top", ha="center", transform=ax2.transAxes)
    hit_text = (
        f"Hits:       {dav_hits}\n"
        f"Home Runs:  {dav_hr}\n"
        f"Walks:      {dav_hit_bbs}\n"
        f"Strikeouts: {dav_hit_ks}"
    )
    ax2.text(0.12, 0.72, hit_text, fontsize=_stat_fontsize, va="top", ha="left",
             color=_DARK, transform=ax2.transAxes, family="monospace", linespacing=2.0)

    # Bottom-left: Batted Ball Quality
    ax3 = fig.add_subplot(stats_gs[1, 0])
    ax3.axis("off")
    ax3.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#f7f9fc", edgecolor="#dde3ea", linewidth=0.8,
        transform=ax3.transAxes, zorder=0))
    ax3.text(0.5, 0.92, "BATTED BALL", fontsize=10, fontweight="bold",
             color=_DARK, va="top", ha="center", transform=ax3.transAxes)
    ev_str = f"{avg_ev:.1f}" if pd.notna(avg_ev) else "-"
    mx_str = f"{max_ev:.1f}" if pd.notna(max_ev) else "-"
    hh_pct = (dav_bbe["ExitSpeed"] >= 95).mean() * 100 if len(dav_bbe) > 0 else 0
    bbq_text = (
        f"Avg EV:    {ev_str} mph\n"
        f"Max EV:    {mx_str} mph\n"
        f"Hard Hit%: {hh_pct:.0f}%\n"
        f"BBE:       {len(dav_bbe)}"
    )
    ax3.text(0.12, 0.72, bbq_text, fontsize=_stat_fontsize, va="top", ha="left",
             color=_DARK, transform=ax3.transAxes, family="monospace", linespacing=2.0)

    # Bottom-right: Plate Discipline
    ax4 = fig.add_subplot(stats_gs[1, 1])
    ax4.axis("off")
    ax4.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#f7f9fc", edgecolor="#dde3ea", linewidth=0.8,
        transform=ax4.transAxes, zorder=0))
    ax4.text(0.5, 0.92, "DISCIPLINE", fontsize=10, fontweight="bold",
             color=_DARK, va="top", ha="center", transform=ax4.transAxes)
    loc_df = dav_hitting.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if not loc_df.empty:
        iz = in_zone_mask(loc_df)
        iz_df = loc_df[iz]
        oz_df = loc_df[~iz]
        swings = loc_df[loc_df["PitchCall"].isin(SWING_CALLS)]
        iz_swings = iz_df[iz_df["PitchCall"].isin(SWING_CALLS)]
        oz_swings = oz_df[oz_df["PitchCall"].isin(SWING_CALLS)]
        whiffs = dav_hitting[dav_hitting["PitchCall"] == "StrikeSwinging"]
        z_sw = len(iz_swings) / max(len(iz_df), 1) * 100
        ch = len(oz_swings) / max(len(oz_df), 1) * 100
        wh = len(whiffs) / max(len(swings), 1) * 100 if len(swings) > 0 else 0
        disc_text = (
            f"Zone Swing%: {z_sw:.1f}%\n"
            f"Chase%:      {ch:.1f}%\n"
            f"Whiff%:      {wh:.1f}%\n"
            f"SwStr%:      {len(whiffs)/max(len(dav_hitting),1)*100:.1f}%"
        )
    else:
        disc_text = "No location data."
    ax4.text(0.12, 0.72, disc_text, fontsize=_stat_fontsize, va="top", ha="left",
             color=_DARK, transform=ax4.transAxes, family="monospace", linespacing=2.0)

    return fig


def _prepare_umpire_data(gd):
    """Prepare called pitch DataFrame with Correct/Gifted/Missed flags."""
    called = gd[gd["PitchCall"].isin(["StrikeCalled", "BallCalled"])].copy()
    called["PlateLocSide"] = pd.to_numeric(called.get("PlateLocSide"), errors="coerce")
    called["PlateLocHeight"] = pd.to_numeric(called.get("PlateLocHeight"), errors="coerce")
    called = called.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if called.empty:
        return None
    iz = in_zone_mask(called)
    is_strike = called["PitchCall"] == "StrikeCalled"
    called["Correct"] = (is_strike & iz) | (~is_strike & ~iz)
    called["Gifted"] = is_strike & ~iz
    called["Missed"] = ~is_strike & iz
    return called


def _render_umpire_page(called, game_label):
    """Umpire report page (adapted from existing generate_postgame_pdf)."""
    if called is None or called.empty:
        return None

    is_strike = called["PitchCall"] == "StrikeCalled"
    iz = in_zone_mask(called)

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(3, 1, figure=fig,
        height_ratios=[0.08, 0.52, 0.40],
        hspace=0.08, top=0.97, bottom=0.03, left=0.03, right=0.97)

    _header_bar(fig, outer[0], f"POSTGAME REPORT  |  {game_label}  |  UMPIRE")

    top_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
        width_ratios=[1.2, 0.8], wspace=0.08)

    # Scatter
    ax_scatter = fig.add_subplot(top_gs[0, 0])
    ax_scatter.set_xlim(-2.5, 2.5)
    ax_scatter.set_ylim(0, 5)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xticks([])
    ax_scatter.set_yticks([])
    for sp in ax_scatter.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#ddd")
    _draw_zone_rect(ax_scatter)
    ax_scatter.set_title("CALLED PITCH ACCURACY", fontsize=9, fontweight="bold",
                         color=_DARK, loc="left", pad=4)

    correct = called[called["Correct"]]
    incorrect = called[~called["Correct"]]
    if not correct.empty:
        ax_scatter.scatter(correct["PlateLocSide"], correct["PlateLocHeight"],
                           c="#2ca02c", s=22, alpha=0.6, edgecolors="white",
                           linewidths=0.3, zorder=10, label="Correct")
    if not incorrect.empty:
        ax_scatter.scatter(incorrect["PlateLocSide"], incorrect["PlateLocHeight"],
                           c="#d62728", s=35, alpha=0.8, marker="x",
                           linewidths=1.2, zorder=11, label="Incorrect")
    ax_scatter.legend(loc="upper right", fontsize=7, framealpha=0.8)

    # Metrics
    ax_metrics = fig.add_subplot(top_gs[0, 1])
    ax_metrics.axis("off")
    ax_metrics.set_title("ACCURACY METRICS", fontsize=9, fontweight="bold",
                         color=_DARK, loc="left", pad=4)

    total = len(called)
    n_correct = int(called["Correct"].sum())
    n_strikes = int(is_strike.sum())
    n_balls = int((~is_strike).sum())
    strike_correct = (is_strike & iz).sum() / max(n_strikes, 1) * 100
    ball_correct = (~is_strike & ~iz).sum() / max(n_balls, 1) * 100
    gifted = int(called["Gifted"].sum())
    missed = int(called["Missed"].sum())

    metrics_text = (
        f"Overall:  {n_correct / max(total, 1) * 100:.1f}%  ({n_correct}/{total})\n\n"
        f"Strike:   {strike_correct:.1f}%  ({int((is_strike & iz).sum())}/{n_strikes})\n\n"
        f"Ball:     {ball_correct:.1f}%  ({int((~is_strike & ~iz).sum())}/{n_balls})\n\n"
        f"Gifted:   {gifted}  (K called OZ)\n\n"
        f"Missed:   {missed}  (B called IZ)"
    )
    ax_metrics.text(0.05, 0.90, metrics_text, fontsize=8.5, va="top", ha="left",
                    color=_DARK, transform=ax_metrics.transAxes,
                    family="monospace", linespacing=1.5)

    # Bottom breakdowns
    bot_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.15)

    # By count
    ax_count = fig.add_subplot(bot_gs[0, 0])
    ax_count.axis("off")
    ax_count.set_title("BY COUNT STATE", fontsize=8, fontweight="bold",
                       color=_DARK, loc="left", pad=2)
    if "Balls" in called.columns and "Strikes" in called.columns:
        b_col = pd.to_numeric(called["Balls"], errors="coerce")
        s_col = pd.to_numeric(called["Strikes"], errors="coerce")
        count_rows = []
        for state_name, mask in [
            ("Ahead", s_col > b_col), ("Even", b_col == s_col), ("Behind", b_col > s_col)
        ]:
            sub = called[mask]
            if sub.empty:
                continue
            count_rows.append([state_name, str(len(sub)),
                               f"{sub['Correct'].mean()*100:.1f}",
                               str(int(sub['Gifted'].sum())),
                               str(int(sub['Missed'].sum()))])
        if count_rows:
            _styled_table(ax_count, count_rows, ["Count", "Pitches", "Acc%", "Gifted", "Missed"],
                          [0.20, 0.18, 0.20, 0.18, 0.18], fontsize=7, row_height=1.4)

    # By side
    ax_side = fig.add_subplot(bot_gs[0, 1])
    ax_side.axis("off")
    ax_side.set_title("BY BATTER SIDE", fontsize=8, fontweight="bold",
                      color=_DARK, loc="left", pad=2)
    if "BatterSide" in called.columns:
        side_rows = []
        for side in ["Right", "Left"]:
            sub = called[called["BatterSide"] == side]
            if sub.empty:
                continue
            side_rows.append([side, str(len(sub)),
                              f"{sub['Correct'].mean()*100:.1f}",
                              str(int(sub['Gifted'].sum())),
                              str(int(sub['Missed'].sum()))])
        if side_rows:
            _styled_table(ax_side, side_rows, ["Side", "Pitches", "Acc%", "Gifted", "Missed"],
                          [0.20, 0.18, 0.20, 0.18, 0.18], fontsize=7, row_height=1.4)

    # By pitcher
    ax_pitcher = fig.add_subplot(bot_gs[0, 2])
    ax_pitcher.axis("off")
    ax_pitcher.set_title("BY PITCHER", fontsize=8, fontweight="bold",
                         color=_DARK, loc="left", pad=2)
    if "Pitcher" in called.columns:
        pit_rows = []
        for p, sub in called.groupby("Pitcher"):
            pit_rows.append([display_name(p, escape_html=False), str(len(sub)),
                             f"{sub['Correct'].mean()*100:.1f}",
                             str(int(sub['Gifted'].sum())),
                             str(int(sub['Missed'].sum()))])
        pit_rows.sort(key=lambda r: int(r[1]), reverse=True)
        if pit_rows:
            _styled_table(ax_pitcher, pit_rows, ["Pitcher", "Pitches", "Acc%", "Gifted", "Missed"],
                          [0.30, 0.15, 0.18, 0.15, 0.15], fontsize=7, row_height=1.4)

    return fig


def _call_leverage_score(row):
    """Compute a simple leverage score for a missed/gifted call."""
    score = 1.0
    b = int(row.get("Balls", 0) or 0)
    s = int(row.get("Strikes", 0) or 0)
    # Count leverage
    if b == 3 and s == 2:
        score *= 2.0       # full count
    elif b == 3 or s == 2:
        score *= 1.5       # payoff counts
    # Inning weight
    inn = int(row.get("Inning", 1) or 1)
    if inn >= 7:
        score *= 1.5
    elif inn >= 4:
        score *= 1.2
    # Runners
    risp = any(
        pd.notna(row.get(c)) and str(row.get(c)).strip() not in ("", " ")
        for c in ["Runner2B", "Runner3B"]
    )
    runners_on = risp or (
        pd.notna(row.get("Runner1B")) and str(row.get("Runner1B")).strip() not in ("", " ")
    )
    if risp:
        score *= 1.5
    elif runners_on:
        score *= 1.2
    # Outs
    if int(row.get("Outs", 0) or 0) == 2:
        score *= 1.3
    return score


def _render_umpire_page2(called, game_label):
    """Umpire report page 2: By Batter Side scatter + Biggest Leverage Missed Calls."""
    if called is None or called.empty:
        return None

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(3, 1, figure=fig,
        height_ratios=[0.08, 0.45, 0.47],
        hspace=0.08, top=0.97, bottom=0.03, left=0.03, right=0.97)

    _header_bar(fig, outer[0], f"POSTGAME REPORT  |  {game_label}  |  UMPIRE")

    # ── Row 1: By Batter Side scatter plots ──
    scatter_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.12)

    for idx, side in enumerate(["Right", "Left"]):
        ax = fig.add_subplot(scatter_gs[0, idx])
        side_df = called[called.get("BatterSide", pd.Series(dtype=str)) == side] if "BatterSide" in called.columns else pd.DataFrame()
        if side_df.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No {side}-Handed Data", fontsize=9, color="#999",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        n_calls = len(side_df)
        acc = side_df["Correct"].mean() * 100

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color("#ddd")
        _draw_zone_rect(ax)

        label = "RHH" if side == "Right" else "LHH"
        ax.set_title(f"{label} — {n_calls} calls, {acc:.1f}% accuracy",
                     fontsize=8, fontweight="bold", color=_DARK, loc="left", pad=4)

        sc = side_df[side_df["Correct"]]
        si = side_df[~side_df["Correct"]]
        if not sc.empty:
            ax.scatter(sc["PlateLocSide"], sc["PlateLocHeight"],
                       c="#2ca02c", s=22, alpha=0.6, edgecolors="white",
                       linewidths=0.3, zorder=10, label="Correct")
        if not si.empty:
            ax.scatter(si["PlateLocSide"], si["PlateLocHeight"],
                       c="#d62728", s=35, alpha=0.8, marker="x",
                       linewidths=1.2, zorder=11, label="Incorrect")
        ax.legend(loc="upper right", fontsize=6, framealpha=0.8)

    # ── Row 2: Biggest Leverage Missed Calls table ──
    ax_table = fig.add_subplot(outer[2])
    ax_table.axis("off")
    ax_table.set_title("BIGGEST LEVERAGE MISSED CALLS", fontsize=9, fontweight="bold",
                       color=_DARK, loc="left", pad=4)

    incorrect = called[~called["Correct"]].copy()
    if incorrect.empty:
        ax_table.text(0.5, 0.5, "No missed calls this game!", fontsize=10, color="#2ca02c",
                      ha="center", va="center", transform=ax_table.transAxes, fontweight="bold")
    else:
        incorrect["LeverageScore"] = incorrect.apply(_call_leverage_score, axis=1)
        incorrect = incorrect.sort_values("LeverageScore", ascending=False).head(5)

        table_rows = []
        for _, row in incorrect.iterrows():
            inn = str(int(row.get("Inning", 0) or 0))
            b = int(row.get("Balls", 0) or 0)
            s = int(row.get("Strikes", 0) or 0)
            count = f"{b}-{s}"
            outs = str(int(row.get("Outs", 0) or 0))
            batter = display_name(row.get("Batter", ""), escape_html=False) if row.get("Batter") else ""
            pitcher = display_name(row.get("Pitcher", ""), escape_html=False) if row.get("Pitcher") else ""
            call_type = "Gifted Strike" if row.get("Gifted", False) else "Missed Strike"
            lev = row["LeverageScore"]
            lev_label = "High" if lev > 2.5 else ("Med" if lev > 1.5 else "Low")
            table_rows.append([inn, count, outs, batter, pitcher, call_type, lev_label])

        if table_rows:
            _styled_table(ax_table, table_rows,
                          ["Inn", "Count", "Outs", "Batter", "Pitcher", "Call Type", "Leverage"],
                          [0.08, 0.10, 0.08, 0.22, 0.22, 0.18, 0.12],
                          fontsize=7.5, row_height=1.6)

    return fig


def _render_pitching_summary_page(gd, game_label):
    """Pitching staff summary page."""
    dav_pitching = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID].copy()
    if dav_pitching.empty:
        return None

    pitchers = dav_pitching.groupby("Pitcher").size().sort_values(ascending=False).index.tolist()

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    n_pitchers = len(pitchers)
    outer = gridspec.GridSpec(3, 1, figure=fig,
        height_ratios=[0.08, 0.28, 0.64],
        hspace=0.08, top=0.97, bottom=0.03, left=0.03, right=0.97)

    _header_bar(fig, outer[0], f"POSTGAME REPORT  |  {game_label}  |  PITCHING")

    # Summary table
    ax_table = fig.add_subplot(outer[1])
    ax_table.axis("off")
    ax_table.set_title("STAFF SUMMARY", fontsize=9, fontweight="bold",
                       color=_DARK, loc="left", pad=4)

    col_labels = ["Pitcher", "Pitches", "IP", "K", "BB", "H", "Avg Velo", "Max Velo"]
    rows = []
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
        jersey = JERSEY.get(pitcher, "")
        dname = display_name(pitcher, escape_html=False)
        label = f"#{jersey} {dname}" if jersey else dname
        rows.append([label, str(n), ip, str(ks), str(bbs), str(hits), avg_v, max_v])

    _styled_table(ax_table, rows, col_labels,
                  [0.22, 0.09, 0.08, 0.07, 0.07, 0.07, 0.10, 0.10],
                  fontsize=8, row_height=1.5)

    # Per-pitcher panels: donut + zone scatter (up to 4)
    n_cols = min(n_pitchers, 4)
    inner = gridspec.GridSpecFromSubplotSpec(2, max(n_cols, 1), subplot_spec=outer[2],
        hspace=0.25, wspace=0.30)

    for idx, pitcher in enumerate(pitchers[:4]):
        pdf = dav_pitching[dav_pitching["Pitcher"] == pitcher]
        dname = display_name(pitcher, escape_html=False)

        # Donut
        ax_donut = fig.add_subplot(inner[0, idx])
        mix = pdf["TaggedPitchType"].dropna().value_counts()
        if len(mix) > 0:
            colors = [PITCH_COLORS.get(pt, "#aaa") for pt in mix.index]
            wedges, _ = ax_donut.pie(
                mix.values, labels=None, colors=colors,
                wedgeprops=dict(width=0.33, edgecolor="white", linewidth=1),
                startangle=90)
            legend_parts = [f"{pt} {n}" for pt, n in zip(mix.index, mix.values)]
            ax_donut.legend(wedges, legend_parts, loc="center", fontsize=5.5, frameon=False)
        ax_donut.set_title(dname, fontsize=8, fontweight="bold", color=_DARK, pad=3)

        # Zone scatter
        ax_zone = fig.add_subplot(inner[1, idx])
        _mpl_zone_scatter(ax_zone, pdf)

    if n_pitchers > 4:
        fig.text(0.5, 0.01,
                 f"(+ {n_pitchers - 4} additional pitcher{'s' if n_pitchers - 4 > 1 else ''} not shown)",
                 fontsize=6, ha="center", color="#999")

    return fig


def _render_pitcher_page(pdf, data, pitcher, game_label):
    """Individual pitcher detail page."""
    n_pitches = len(pdf)
    if n_pitches < 5:
        return None

    season_pdf = data[data["Pitcher"] == pitcher] if data is not None else pd.DataFrame()
    grades, feedback, stuff_by_pt, cmd_df = _compute_pitcher_grades(pdf, data, pitcher)
    valid_scores = [v for v in grades.values() if v is not None]
    overall = np.mean(valid_scores) if valid_scores else None
    ip_est = _pg_estimate_ip(pdf)

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    # GridSpec: 5 rows x 2 cols
    outer = gridspec.GridSpec(5, 2, figure=fig,
        height_ratios=[0.10, 0.07, 0.28, 0.25, 0.30],
        hspace=0.18, wspace=0.15,
        top=0.97, bottom=0.05, left=0.04, right=0.96)

    small_sample = n_pitches < _MIN_PITCHER_PITCHES

    # Row 0: Header (span both cols)
    ax_hdr = fig.add_subplot(outer[0, :])
    _mpl_grade_header(ax_hdr, pitcher, n_pitches, overall, f"~{ip_est} IP", small_sample=small_sample)

    # Row 1: Grade cards (span both cols)
    ax_cards = fig.add_subplot(outer[1, :])
    _mpl_grade_cards(ax_cards, grades)

    # Row 2 left: Pitch mix table + zone scatter
    mix_zone_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2, 0],
        wspace=0.15)

    # Pitch mix table
    ax_mix = fig.add_subplot(mix_zone_gs[0, 0])
    ax_mix.axis("off")
    ax_mix.set_title("Pitch Mix", fontsize=8, fontweight="bold", color=_DARK, loc="left", pad=2)
    if "TaggedPitchType" in pdf.columns:
        mix_rows = []
        total = len(pdf)
        for pt, grp in pdf.groupby("TaggedPitchType"):
            v = grp["RelSpeed"].dropna() if "RelSpeed" in grp.columns else pd.Series(dtype=float)
            avg_v = f"{v.mean():.1f}" if len(v) > 0 else "-"
            mix_rows.append([pt, str(len(grp)), f"{len(grp)/total*100:.0f}%", avg_v])
        mix_rows.sort(key=lambda r: int(r[1]), reverse=True)
        if mix_rows:
            _styled_table(ax_mix, mix_rows, ["Pitch", "N", "%", "Velo"],
                          [0.30, 0.15, 0.20, 0.20], fontsize=6.5, row_height=1.3)

    # Zone scatter
    ax_zone = fig.add_subplot(mix_zone_gs[0, 1])
    _mpl_zone_scatter(ax_zone, pdf)

    # Row 2 right: Movement profile
    ax_mov = fig.add_subplot(outer[2, 1])
    _mpl_movement_profile(ax_mov, pdf)

    # Row 3 left: Radar
    _mpl_radar_chart(fig, outer[3, 0], grades)

    # Row 3 right: Stuff+/Cmd+ bars (percentiles vs own history)
    ax_bars = fig.add_subplot(outer[3, 1])
    season_stuff_dist, season_cmd_dist = _compute_historical_stuff_cmd_distributions(season_pdf, data)
    _mpl_stuff_cmd_bars(ax_bars, stuff_by_pt, cmd_df,
                        season_stuff_by_pt=season_stuff_dist,
                        season_cmd_by_pt=season_cmd_dist)

    # Row 4 left: Percentile bars
    pctl_metrics = _compute_pitcher_percentile_metrics(pdf, season_pdf)
    ax_pctl = fig.add_subplot(outer[4, 0])
    _mpl_percentile_bars(ax_pctl, pctl_metrics)
    ax_pctl.set_title("Game vs Historical Percentiles", fontsize=7, fontweight="bold",
                      color=_DARK, loc="left", pad=2)

    # Row 4 right: Call grade (top) + feedback (bottom)
    row4_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[4, 1],
        height_ratios=[0.45, 0.55], hspace=0.12)

    # Call grade box
    ax_cg = fig.add_subplot(row4_right[0])
    if n_pitches >= _MIN_PITCHER_PITCHES:
        try:
            grade_info = _compute_call_grade(pdf, data, pitcher)
            _mpl_call_grade_box(ax_cg, grade_info)
        except Exception:
            ax_cg.axis("off")
    else:
        ax_cg.axis("off")

    # Feedback
    ax_fb = fig.add_subplot(row4_right[1])
    _mpl_feedback(ax_fb, feedback)

    return fig


def _render_hitting_summary_page(gd, game_label):
    """Hitting lineup summary page."""
    dav_hitting = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID].copy()
    if dav_hitting.empty:
        return None

    batters = dav_hitting.groupby("Batter").size().sort_values(ascending=False).index.tolist()

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(4, 1, figure=fig,
        height_ratios=[0.08, 0.42, 0.22, 0.28],
        hspace=0.10, top=0.97, bottom=0.03, left=0.03, right=0.97)

    _header_bar(fig, outer[0], f"POSTGAME REPORT  |  {game_label}  |  HITTING")

    # Lineup table
    ax_table = fig.add_subplot(outer[1])
    ax_table.axis("off")
    ax_table.set_title("LINEUP SUMMARY", fontsize=9, fontweight="bold",
                       color=_DARK, loc="left", pad=4)

    col_labels = ["Batter", "PA", "H", "2B", "3B", "HR", "BB", "K", "Avg EV", "Max EV", "HH%"]
    rows = []
    for batter in batters:
        bdf = dav_hitting[dav_hitting["Batter"] == batter]
        pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in bdf.columns]
        pa = bdf.drop_duplicates(subset=pa_cols).shape[0] if len(pa_cols) >= 2 else 0
        pr = bdf["PlayResult"] if "PlayResult" in bdf.columns else pd.Series(dtype=str)
        hits = pr.isin(["Single", "Double", "Triple", "HomeRun"]).sum()
        doubles = (pr == "Double").sum()
        triples = (pr == "Triple").sum()
        hrs = (pr == "HomeRun").sum()
        bbs = (bdf["KorBB"] == "Walk").sum() if "KorBB" in bdf.columns else 0
        ks = (bdf["KorBB"] == "Strikeout").sum() if "KorBB" in bdf.columns else 0
        in_play = bdf[bdf["PitchCall"] == "InPlay"]
        ev_data = pd.to_numeric(in_play["ExitSpeed"], errors="coerce").dropna() if "ExitSpeed" in in_play.columns else pd.Series(dtype=float)
        avg_ev = f"{ev_data.mean():.1f}" if len(ev_data) > 0 else "-"
        max_ev = f"{ev_data.max():.1f}" if len(ev_data) > 0 else "-"
        hh_pct = f"{(ev_data >= 95).mean() * 100:.0f}%" if len(ev_data) > 0 else "-"
        jersey = JERSEY.get(batter, "")
        dname = display_name(batter, escape_html=False)
        label = f"#{jersey} {dname}" if jersey else dname
        rows.append([label, str(pa), str(hits), str(doubles), str(triples),
                     str(hrs), str(bbs), str(ks), avg_ev, max_ev, hh_pct])

    _styled_table(ax_table, rows, col_labels,
                  [0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.08],
                  fontsize=7, row_height=1.3)

    # Team discipline
    metrics_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.15)

    ax_disc = fig.add_subplot(metrics_gs[0, 0])
    ax_disc.axis("off")
    ax_disc.set_title("PLATE DISCIPLINE", fontsize=8, fontweight="bold",
                      color=_DARK, loc="left", pad=2)
    loc_df = dav_hitting.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if not loc_df.empty:
        iz = in_zone_mask(loc_df)
        iz_df = loc_df[iz]
        oz_df = loc_df[~iz]
        swings = loc_df[loc_df["PitchCall"].isin(SWING_CALLS)]
        iz_swings = iz_df[iz_df["PitchCall"].isin(SWING_CALLS)]
        oz_swings = oz_df[oz_df["PitchCall"].isin(SWING_CALLS)]
        whiffs = dav_hitting[dav_hitting["PitchCall"] == "StrikeSwinging"]
        z_sw = len(iz_swings) / max(len(iz_df), 1) * 100
        ch = len(oz_swings) / max(len(oz_df), 1) * 100
        wh = len(whiffs) / max(len(swings), 1) * 100 if len(swings) > 0 else 0
        swstr = len(whiffs) / max(len(dav_hitting), 1) * 100
        disc_text = (
            f"Zone Swing%:  {z_sw:.1f}%\n"
            f"Chase%:       {ch:.1f}%\n"
            f"Whiff%:       {wh:.1f}%\n"
            f"SwStr%:       {swstr:.1f}%"
        )
    else:
        disc_text = "No location data available."
    ax_disc.text(0.05, 0.85, disc_text, fontsize=8.5, va="top", ha="left",
                 color=_DARK, transform=ax_disc.transAxes,
                 family="monospace", linespacing=1.6)

    # BBQ
    ax_bb = fig.add_subplot(metrics_gs[0, 1])
    ax_bb.axis("off")
    ax_bb.set_title("BATTED BALL QUALITY", fontsize=8, fontweight="bold",
                    color=_DARK, loc="left", pad=2)
    in_play_all = dav_hitting[dav_hitting["PitchCall"] == "InPlay"]
    ev_all = pd.to_numeric(in_play_all["ExitSpeed"], errors="coerce").dropna() if "ExitSpeed" in in_play_all.columns else pd.Series(dtype=float)
    if len(ev_all) > 0:
        la_all = pd.to_numeric(in_play_all["Angle"], errors="coerce").dropna() if "Angle" in in_play_all.columns else pd.Series(dtype=float)
        avg_la = f"{la_all.mean():.1f}" if len(la_all) > 0 else "-"
        barrel_pct = "-"
        if "ExitSpeed" in in_play_all.columns and "Angle" in in_play_all.columns:
            bbe = in_play_all.dropna(subset=["ExitSpeed", "Angle"])
            if len(bbe) > 0:
                barrel_pct = f"{is_barrel_mask(bbe).mean() * 100:.1f}%"
        bb_text = (
            f"Avg EV:    {ev_all.mean():.1f} mph\n"
            f"Max EV:    {ev_all.max():.1f} mph\n"
            f"Hard Hit%: {(ev_all >= 95).mean() * 100:.1f}%  (>=95 mph)\n"
            f"Avg LA:    {avg_la} deg\n"
            f"Barrel%:   {barrel_pct}"
        )
    else:
        bb_text = "No batted ball data."
    ax_bb.text(0.05, 0.85, bb_text, fontsize=8.5, va="top", ha="left",
               color=_DARK, transform=ax_bb.transAxes,
               family="monospace", linespacing=1.6)

    # Key at-bats
    ax_abs = fig.add_subplot(outer[3])
    ax_abs.axis("off")
    ax_abs.set_title("KEY AT-BATS", fontsize=8, fontweight="bold",
                     color=_DARK, loc="left", pad=2)
    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in dav_hitting.columns]
    sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in dav_hitting.columns]
    ab_lines = []
    if len(pa_cols) >= 2:
        for batter in batters:
            bdf = dav_hitting[dav_hitting["Batter"] == batter]
            for pa_key, ab in bdf.groupby(pa_cols[1:]):
                ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
                is_key = False
                result_label = ""
                if "PlayResult" in ab.columns:
                    if ab["PlayResult"].eq("HomeRun").any():
                        is_key, result_label = True, "HR"
                    elif ab["PlayResult"].isin(["Double", "Triple"]).any():
                        is_key = True
                        result_label = ab[ab["PlayResult"].isin(["Double", "Triple"])]["PlayResult"].iloc[0]
                if "KorBB" in ab.columns:
                    if ab["KorBB"].eq("Walk").any():
                        is_key, result_label = True, result_label or "BB"
                if len(ab) >= 6:
                    is_key = True
                    result_label = result_label or f"{len(ab)}-pitch PA"
                if is_key:
                    inn = ab_sorted.iloc[0].get("Inning", "?")
                    pitcher_name = display_name(ab_sorted.iloc[0]["Pitcher"], escape_html=False) if "Pitcher" in ab.columns else "?"
                    batter_name = display_name(batter, escape_html=False)
                    line = f"Inn {inn} | {batter_name} vs {pitcher_name} -- {result_label} ({len(ab)}p)"
                    ab_lines.append((int(inn) if pd.notna(inn) else 99, line))

    ab_lines.sort(key=lambda x: x[0])
    if ab_lines:
        text = "\n".join(f"  {line}" for _, line in ab_lines[:12])
    else:
        text = "No notable at-bats (HR, XBH, BB, or 6+ pitch PA)."
    ax_abs.text(0.02, 0.95, text, fontsize=6.5, va="top", ha="left",
                color=_DARK, transform=ax_abs.transAxes,
                linespacing=1.5, fontfamily="monospace")

    return fig


def _render_hitter_page(bdf, data, batter, game_label):
    """Individual hitter detail page."""
    n_pitches = len(bdf)
    if n_pitches < 3:
        return None

    season_bdf = data[data["Batter"] == batter] if data is not None else pd.DataFrame()
    grades, feedback = _compute_hitter_grades(bdf, data, batter)
    valid_scores = [v for v in grades.values() if v is not None]
    overall = np.mean(valid_scores) if valid_scores else None

    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in bdf.columns]
    n_pas = bdf.drop_duplicates(subset=pa_cols).shape[0] if len(pa_cols) >= 2 else 0
    small_sample = n_pas < _MIN_HITTER_PAS

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    # GridSpec: 5 rows x 2 cols
    outer = gridspec.GridSpec(5, 2, figure=fig,
        height_ratios=[0.10, 0.07, 0.28, 0.25, 0.30],
        hspace=0.18, wspace=0.15,
        top=0.97, bottom=0.05, left=0.04, right=0.96)

    # Row 0: Header
    ax_hdr = fig.add_subplot(outer[0, :])
    _mpl_grade_header(ax_hdr, batter, n_pitches, overall, f"{n_pas} PA", small_sample=small_sample)

    # Row 1: Grade cards
    ax_cards = fig.add_subplot(outer[1, :])
    _mpl_grade_cards(ax_cards, grades)

    # Row 2 left: Discipline + BBQ metrics
    ax_disc = fig.add_subplot(outer[2, 0])
    ax_disc.axis("off")
    ax_disc.set_title("Plate Discipline & BBQ", fontsize=8, fontweight="bold",
                      color=_DARK, loc="left", pad=3)

    loc_df = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    disc_lines = []
    if not loc_df.empty:
        iz = in_zone_mask(loc_df)
        iz_df = loc_df[iz]
        oz_df = loc_df[~iz]
        swings = loc_df[loc_df["PitchCall"].isin(SWING_CALLS)]
        iz_swings = iz_df[iz_df["PitchCall"].isin(SWING_CALLS)]
        oz_swings = oz_df[oz_df["PitchCall"].isin(SWING_CALLS)]
        whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"]
        disc_lines.append(f"Zone Swing%: {len(iz_swings)/max(len(iz_df),1)*100:.1f}%")
        disc_lines.append(f"Chase%:      {len(oz_swings)/max(len(oz_df),1)*100:.1f}%")
        disc_lines.append(f"Whiff%:      {len(whiffs)/max(len(swings),1)*100:.1f}%" if len(swings) > 0 else "Whiff%: N/A")
    in_play = bdf[bdf["PitchCall"] == "InPlay"]
    bbe_df = in_play.dropna(subset=["ExitSpeed"]) if "ExitSpeed" in bdf.columns else pd.DataFrame()
    if len(bbe_df) > 0:
        ev = bbe_df["ExitSpeed"]
        disc_lines.append(f"Avg EV:      {ev.mean():.1f} mph")
        disc_lines.append(f"Max EV:      {ev.max():.1f} mph")
        disc_lines.append(f"Hard Hit%:   {(ev >= 95).mean()*100:.0f}%")
        if "Angle" in bbe_df.columns:
            barrel = is_barrel_mask(bbe_df).mean() * 100
            disc_lines.append(f"Barrel%:     {barrel:.0f}%")
    else:
        disc_lines.append("No batted ball data.")
    ax_disc.text(0.05, 0.92, "\n".join(disc_lines), fontsize=7.5, va="top", ha="left",
                 color=_DARK, transform=ax_disc.transAxes,
                 family="monospace", linespacing=1.5)

    # Row 2 right: Spray chart
    ax_spray = fig.add_subplot(outer[2, 1])
    if not in_play.empty:
        _mpl_spray_chart(ax_spray, in_play)
    else:
        ax_spray.axis("off")
        ax_spray.text(0.5, 0.5, "No in-play data", fontsize=8, color="#999",
                      ha="center", va="center", transform=ax_spray.transAxes)

    # Row 3 left: Radar
    _mpl_radar_chart(fig, outer[3, 0], grades)

    # Row 3 right: Best Zone Heatmap (replaces damage heatmap)
    ax_heat = fig.add_subplot(outer[3, 1])
    batter_side = bdf["BatterSide"].iloc[0] if "BatterSide" in bdf.columns and len(bdf) > 0 else "Right"
    if pd.isna(batter_side):
        batter_side = "Right"
    season_source = season_bdf if not season_bdf.empty and len(season_bdf) >= 30 else bdf
    try:
        _mpl_best_zone_heatmap(ax_heat, season_source, batter_side)
    except Exception:
        # Fallback to old damage heatmap
        ax_heat.clear()
        season_loc = season_bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"]) if not season_bdf.empty else pd.DataFrame()
        if len(season_loc) >= 20:
            grid, annot, h_labels, v_labels = _create_zone_grid_data(
                season_loc, metric="avg_ev", batter_side=batter_side)
            game_loc = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
            game_swings = game_loc[game_loc["PitchCall"].isin(SWING_CALLS)]
            sw_xy = []
            h_edges = [-2, -0.83, -0.28, 0.28, 0.83, 2]
            v_edges_map = [0.5, 1.5, 2.17, 2.83, 3.5, 4.5]
            for _, row in game_swings.iterrows():
                ps, ph = row["PlateLocSide"], row["PlateLocHeight"]
                xi = next((i for i in range(5) if h_edges[i] <= ps < h_edges[i + 1]), None)
                yi = next((i for i in range(5) if v_edges_map[i] <= ph < v_edges_map[i + 1]), None)
                if xi is not None and yi is not None:
                    sw_xy.append((xi, yi))
            _mpl_damage_heatmap(ax_heat, grid, annot, h_labels, v_labels, sw_xy)
        else:
            ax_heat.axis("off")
            ax_heat.text(0.5, 0.5, "Not enough historical data\nfor damage heatmap",
                         fontsize=7, color="#999", ha="center", va="center",
                         transform=ax_heat.transAxes)

    # Row 4 left: Percentile bars
    pctl_metrics = _compute_hitter_percentile_metrics(bdf, season_bdf)
    ax_pctl = fig.add_subplot(outer[4, 0])
    _mpl_percentile_bars(ax_pctl, pctl_metrics)
    ax_pctl.set_title("Game vs Historical Percentiles", fontsize=7, fontweight="bold",
                      color=_DARK, loc="left", pad=2)

    # Row 4 right: At-bat grades (top) + feedback (bottom)
    row4_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[4, 1],
        height_ratios=[0.55, 0.45], hspace=0.15)

    ax_ab = fig.add_subplot(row4_right[0])
    sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in bdf.columns]
    ab_rows = []
    if len(pa_cols) >= 2:
        for pa_key, ab in bdf.groupby(pa_cols[1:]):
            ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
            score, letter, result = _grade_at_bat(ab_sorted, season_bdf)
            inn = ab_sorted.iloc[0].get("Inning", "?")
            vs_pitcher = display_name(ab_sorted.iloc[0]["Pitcher"], escape_html=False) if "Pitcher" in ab_sorted.columns else "?"
            ab_rows.append({
                "Inning": inn, "vs Pitcher": vs_pitcher,
                "Pitches": len(ab_sorted), "Result": result,
                "Score": score, "Grade": letter,
            })
    if ab_rows:
        _mpl_ab_grades_table(ax_ab, ab_rows)
    else:
        ax_ab.axis("off")
        ax_ab.text(0.5, 0.5, "No at-bats", fontsize=8, color="#999",
                   ha="center", va="center", transform=ax_ab.transAxes)

    ax_fb = fig.add_subplot(row4_right[1])
    _mpl_feedback(ax_fb, feedback)

    return fig


def _render_pa_breakdown_pages(player_df, player_name, game_label, role="pitcher"):
    """Render PA breakdown pages. ~3 PAs per page.

    Returns: list of Figure objects.
    """
    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in player_df.columns]
    sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in player_df.columns]

    if len(pa_cols) < 2:
        return []

    # Collect PAs sorted by inning
    pa_list = []
    for pa_key, ab in player_df.groupby(pa_cols[1:]):
        ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
        inn = ab_sorted.iloc[0].get("Inning", "?")
        pa_list.append((inn, ab_sorted))
    pa_list.sort(key=lambda x: (int(x[0]) if pd.notna(x[0]) and str(x[0]).isdigit() else 99))

    if not pa_list:
        return []

    PAS_PER_PAGE = 3
    figures = []
    dname = display_name(player_name, escape_html=False)

    for page_start in range(0, len(pa_list), PAS_PER_PAGE):
        page_pas = pa_list[page_start:page_start + PAS_PER_PAGE]
        n_pas = len(page_pas)

        fig = plt.figure(figsize=_FIG_SIZE)
        fig.patch.set_facecolor("white")

        # Height ratios: header + up to 3 PA rows
        h_ratios = [0.08] + [0.30] * n_pas
        # Pad remaining space if fewer than 3 PAs
        if n_pas < PAS_PER_PAGE:
            h_ratios.append(1.0 - 0.08 - 0.30 * n_pas)

        outer = gridspec.GridSpec(len(h_ratios), 1, figure=fig,
            height_ratios=h_ratios, hspace=0.12,
            top=0.97, bottom=0.03, left=0.04, right=0.96)

        # Header
        _header_bar(fig, outer[0], f"POSTGAME | {game_label} | {dname} — PA Details")

        for pa_i, (inn, ab_sorted) in enumerate(page_pas):
            row_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1 + pa_i],
                wspace=0.08, width_ratios=[0.6, 0.4])

            # Determine opponent and result
            if role == "hitter":
                opponent = display_name(ab_sorted.iloc[0]["Pitcher"], escape_html=False) if "Pitcher" in ab_sorted.columns else "?"
            else:
                opponent = display_name(ab_sorted.iloc[0]["Batter"], escape_html=False) if "Batter" in ab_sorted.columns else "?"

            last = ab_sorted.iloc[-1]
            result = "?"
            if last.get("KorBB") == "Strikeout":
                result = "K"
            elif last.get("KorBB") == "Walk":
                result = "BB"
            elif last.get("PitchCall") == "HitByPitch":
                result = "HBP"
            elif pd.notna(last.get("PlayResult")) and last.get("PlayResult") not in ("Undefined", ""):
                result = last["PlayResult"]
            elif last.get("PitchCall") == "InPlay":
                result = "InPlay"

            n_pitches_pa = len(ab_sorted)
            pa_header = f"Inn {inn} vs {opponent} — {result} ({n_pitches_pa}p)"

            # Left: pitch table
            ax_table = fig.add_subplot(row_gs[0, 0])
            ax_table.axis("off")
            ax_table.set_title(pa_header, fontsize=7.5, fontweight="bold",
                              color=_DARK, loc="left", pad=2)

            pitch_rows = _pg_build_pa_pitch_rows(ab_sorted)
            if pitch_rows:
                table_data = [
                    [str(r["#"]), r["Count"], r["Type"], r["Velo"], r["Call"], r["EV"], r["LA"]]
                    for r in pitch_rows
                ]
                _styled_table(ax_table, table_data,
                             ["#", "Count", "Type", "Velo", "Call", "EV", "LA"],
                             [0.06, 0.10, 0.18, 0.12, 0.12, 0.12, 0.10],
                             fontsize=6.5, row_height=1.3)

            # Right: zone plot
            ax_zone = fig.add_subplot(row_gs[0, 1])
            _mpl_pa_zone_plot(ax_zone, ab_sorted)

        figures.append(fig)

    return figures


def _render_takeaways_page(gd, data, game_label):
    """Coach takeaways page."""
    pitching_bullets, hitting_bullets = _compute_takeaways(gd, data)

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(3, 1, figure=fig,
        height_ratios=[0.08, 0.46, 0.46],
        hspace=0.08, top=0.97, bottom=0.03, left=0.03, right=0.97)

    _header_bar(fig, outer[0], f"POSTGAME REPORT  |  {game_label}  |  COACH TAKEAWAYS")

    # Pitching takeaways
    ax_pit = fig.add_subplot(outer[1])
    ax_pit.axis("off")
    ax_pit.set_title("PITCHING", fontsize=10, fontweight="bold",
                     color=_DARK, loc="left", pad=4)
    if pitching_bullets:
        text = "\n".join(f"  - {b}" for b in pitching_bullets)
    else:
        text = "  No pitching takeaways available."
    ax_pit.text(0.03, 0.85, text, fontsize=9, va="top", ha="left",
                color=_DARK, transform=ax_pit.transAxes,
                linespacing=1.8, fontfamily="sans-serif")

    # Hitting takeaways
    ax_hit = fig.add_subplot(outer[2])
    ax_hit.axis("off")
    ax_hit.set_title("HITTING", fontsize=10, fontweight="bold",
                     color=_DARK, loc="left", pad=4)
    if hitting_bullets:
        text = "\n".join(f"  - {b}" for b in hitting_bullets)
    else:
        text = "  No hitting takeaways available."
    ax_hit.text(0.03, 0.85, text, fontsize=9, va="top", ha="left",
                color=_DARK, transform=ax_hit.transAxes,
                linespacing=1.8, fontfamily="sans-serif")

    return fig


# ── Public API ───────────────────────────────────────────────────────────────

def generate_postgame_pdf_bytes(gd, data, game_label) -> bytes:
    """Generate the full postgame report PDF in memory.

    Returns raw bytes suitable for st.download_button.
    """
    buf = io.BytesIO()
    pages_saved = 0

    with PdfPages(buf) as pdf:
        # Page 1: Cover
        try:
            fig = _render_cover_page(gd, game_label)
            if fig:
                pdf.savefig(fig)
                plt.close(fig)
                pages_saved += 1
        except Exception:
            traceback.print_exc()

        # Page 2-3: Umpire
        try:
            called = _prepare_umpire_data(gd)
            fig = _render_umpire_page(called, game_label)
            if fig:
                pdf.savefig(fig)
                plt.close(fig)
                pages_saved += 1
            fig2 = _render_umpire_page2(called, game_label)
            if fig2:
                pdf.savefig(fig2)
                plt.close(fig2)
                pages_saved += 1
        except Exception:
            traceback.print_exc()

        # Pitching summary
        try:
            fig = _render_pitching_summary_page(gd, game_label)
            if fig:
                pdf.savefig(fig)
                plt.close(fig)
                pages_saved += 1
        except Exception:
            traceback.print_exc()

        # Pages 4..N: Individual pitchers + PA breakdowns
        dav_pitching = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID].copy()
        if not dav_pitching.empty:
            pitchers = dav_pitching.groupby("Pitcher").size().sort_values(ascending=False).index.tolist()
            for pitcher in pitchers:
                try:
                    pitcher_pdf = dav_pitching[dav_pitching["Pitcher"] == pitcher].copy()
                    fig = _render_pitcher_page(pitcher_pdf, data, pitcher, game_label)
                    if fig:
                        pdf.savefig(fig)
                        plt.close(fig)
                        pages_saved += 1
                    # PA breakdown pages for this pitcher
                    pa_figs = _render_pa_breakdown_pages(pitcher_pdf, pitcher, game_label, role="pitcher")
                    for pa_fig in pa_figs:
                        pdf.savefig(pa_fig)
                        plt.close(pa_fig)
                        pages_saved += 1
                except Exception:
                    traceback.print_exc()

        # Page N+1: Hitting summary
        try:
            fig = _render_hitting_summary_page(gd, game_label)
            if fig:
                pdf.savefig(fig)
                plt.close(fig)
                pages_saved += 1
        except Exception:
            traceback.print_exc()

        # Pages N+2..M: Individual hitters + PA breakdowns
        dav_hitting = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID].copy()
        if not dav_hitting.empty:
            batters = dav_hitting.groupby("Batter").size().sort_values(ascending=False).index.tolist()
            for batter in batters:
                try:
                    batter_df = dav_hitting[dav_hitting["Batter"] == batter].copy()
                    fig = _render_hitter_page(batter_df, data, batter, game_label)
                    if fig:
                        pdf.savefig(fig)
                        plt.close(fig)
                        pages_saved += 1
                    # PA breakdown pages for this hitter
                    pa_figs = _render_pa_breakdown_pages(batter_df, batter, game_label, role="hitter")
                    for pa_fig in pa_figs:
                        pdf.savefig(pa_fig)
                        plt.close(pa_fig)
                        pages_saved += 1
                except Exception:
                    traceback.print_exc()

        # Final page: Coach takeaways
        try:
            fig = _render_takeaways_page(gd, data, game_label)
            if fig:
                pdf.savefig(fig)
                plt.close(fig)
                pages_saved += 1
        except Exception:
            traceback.print_exc()

    buf.seek(0)
    return buf.read()


def generate_umpire_pdf_bytes(gd, game_label) -> bytes:
    """Generate a standalone umpire report PDF in memory.

    Returns raw bytes suitable for st.download_button.
    """
    buf = io.BytesIO()
    called = _prepare_umpire_data(gd)
    with PdfPages(buf) as pdf:
        try:
            fig = _render_umpire_page(called, game_label)
            if fig:
                pdf.savefig(fig)
                plt.close(fig)
            fig2 = _render_umpire_page2(called, game_label)
            if fig2:
                pdf.savefig(fig2)
                plt.close(fig2)
        except Exception:
            traceback.print_exc()
    buf.seek(0)
    return buf.read()
