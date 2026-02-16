"""Generate a Series Report PDF combining multiple games.

Produces a multi-page landscape PDF (11 x 8.5) with:
  1. Cover / Series Summary (with record, score table, stats strip, key takeaways)
  2. Coach Takeaways (pitching + hitting bullets)
  3. Pitching Staff Summary
  4..N. Individual Pitcher pages (20+ pitches only)
  N+1. Hitting Lineup Summary
  N+2..M. Individual Hitter pages (5+ PAs only)

Key differences from single-game report:
  - No letter grades or radar charts
  - Flat Savant-style percentile bars (thin track + rounded markers)
  - Pitch Locations Seen (per pitch type, circles=took, diamonds=swung)
  - Pitch-type results table (swing%, whiff%, avg EV)
  - Count leverage breakdowns
  - Velo distribution sparkline (pitchers)
  - Page numbers
  - Season delta columns
"""

import io
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
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
from analytics.command_plus import _compute_command_plus
from analytics.expected import _create_zone_grid_data
from analytics.zone_vulnerability import compute_zone_swing_metrics

from _pages.postgame import (
    _compute_pitcher_grades,
    _compute_hitter_grades,
    _compute_pitcher_percentile_metrics,
    _compute_hitter_percentile_metrics,
    _grade_at_bat,
    _letter_grade,
    _grade_color,
    _pg_estimate_ip,
    _pg_build_pa_pitch_rows,
    _score_linear,
    _compute_takeaways,
    _split_feedback,
    _MIN_PITCHER_PITCHES,
    _MIN_HITTER_PAS,
    _ZONE_X_EDGES,
    _ZONE_Y_EDGES,
)

# ── Style Constants ──────────────────────────────────────────────────────────
_HDR_BG = "#1a1a2e"
_SECTION_BG = "#2c3e50"
_ALT_ROW = "#f0f3f7"
_WHITE = "#ffffff"
_DARK = "#1a1a2e"
_FIG_SIZE = (11, 8.5)
_SAVANT_CMAP = LinearSegmentedColormap.from_list(
    "savant", ["#14365d", "#3d7dab", "#9e9e9e", "#ee7e1e", "#be0000"])

# ── Page numbering ───────────────────────────────────────────────────────────
_page_num = 0
_total_pages = 0


def _add_page_number(fig):
    global _page_num
    _page_num += 1
    fig.text(0.98, 0.01, f"Page {_page_num}", fontsize=6, color="#999",
             ha="right", va="bottom")


# ── Shared Helpers ───────────────────────────────────────────────────────────

def _header_bar(fig, gs_slot, text):
    ax = fig.add_subplot(gs_slot)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_facecolor(_HDR_BG)
    ax.patch.set_alpha(1.0)
    ax.text(0.5, 0.5, text, fontsize=13, fontweight="bold",
            color="white", va="center", ha="center", transform=ax.transAxes)
    return ax


def _draw_zone_rect(ax):
    ax.add_patch(Rectangle(
        (-ZONE_SIDE, ZONE_HEIGHT_BOT),
        2 * ZONE_SIDE, ZONE_HEIGHT_TOP - ZONE_HEIGHT_BOT,
        linewidth=1.5, edgecolor="#333", facecolor="none", zorder=5))


def _styled_table(ax, rows, col_labels, col_w, fontsize=7.5, row_height=1.4):
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


# ── Player Header (no letter grade) ─────────────────────────────────────────

def _player_header(ax, name, n_pitches, label_extra=""):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor("#000000")
    jersey = JERSEY.get(name, "")
    pos = POSITION.get(name, "")
    dname = display_name(name, escape_html=False)
    extra = f" | {label_extra}" if label_extra else ""
    jersey_str = f"#{jersey}  " if jersey else ""
    ax.text(0.03, 0.55, f"{jersey_str}{dname}", fontsize=20, fontweight="900",
            color="white", va="center", ha="left", transform=ax.transAxes, clip_on=False)
    ax.text(0.03, 0.18, f"{pos} | {n_pitches} pitches{extra}",
            fontsize=9, color="#aaa", va="center", ha="left", transform=ax.transAxes, clip_on=False)


# ── Gradient Percentile Bars ─────────────────────────────────────────────────

def _flat_percentile_bars(ax, metrics):
    """Savant-style horizontal percentile bars — clean flat design."""
    n = len(metrics)
    if n == 0:
        ax.axis("off")
        return

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

        ax.text(28, y, label.upper(), fontsize=6.5, fontweight="bold",
                color=_DARK, va="center", ha="right")
        ax.text(135, y, display_val, fontsize=6.5, fontweight="bold",
                color=_DARK, va="center", ha="left")

        if effective_pct is None or pd.isna(pct):
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


# ── Zone Scatter ─────────────────────────────────────────────────────────────

def _zone_scatter(ax, pdf):
    loc = pdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(0.5, 4.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#ddd")
    _draw_zone_rect(ax)
    if not loc.empty and "TaggedPitchType" in loc.columns:
        for pt in loc["TaggedPitchType"].unique():
            sub = loc[loc["TaggedPitchType"] == pt]
            ax.scatter(sub["PlateLocSide"], sub["PlateLocHeight"],
                       c=PITCH_COLORS.get(pt, "#aaa"), s=18, alpha=0.8,
                       edgecolors="white", linewidths=0.3, zorder=10, label=pt)
    ax.legend(fontsize=5, loc="upper right", framealpha=0.7)
    ax.set_title("Pitch Locations", fontsize=7, fontweight="bold", color=_DARK, pad=3)


# ── Pitch Locations Seen (hitter view) ───────────────────────────────────────

def _pitch_locations_seen(fig, gs_slot, bdf):
    """Per-pitch-type zone scatters: circles=took, diamonds=swung."""
    loc = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if loc.empty or "TaggedPitchType" not in loc.columns:
        ax = fig.add_subplot(gs_slot)
        ax.axis("off")
        ax.text(0.5, 0.5, "No location data", fontsize=7, color="#999",
                ha="center", va="center", transform=ax.transAxes)
        return

    pitch_types = loc["TaggedPitchType"].dropna().value_counts()
    pitch_types = pitch_types[pitch_types >= 1].head(6)  # max 6 types
    n_types = len(pitch_types)
    if n_types == 0:
        ax = fig.add_subplot(gs_slot)
        ax.axis("off")
        return

    n_cols = min(n_types, 3)
    n_rows = (n_types + n_cols - 1) // n_cols
    inner = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs_slot,
        hspace=0.35, wspace=0.25)

    for i, (pt, count) in enumerate(pitch_types.items()):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(inner[row, col])
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(0.5, 4.5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#ddd")
        _draw_zone_rect(ax)

        sub = loc[loc["TaggedPitchType"] == pt]
        color = PITCH_COLORS.get(pt, "#aaa")
        took = sub[~sub["PitchCall"].isin(SWING_CALLS)]
        swung = sub[sub["PitchCall"].isin(SWING_CALLS)]

        if not took.empty:
            ax.scatter(took["PlateLocSide"], took["PlateLocHeight"],
                       marker="o", c=color, s=35, alpha=0.6,
                       edgecolors="white", linewidths=0.5, zorder=10)
        if not swung.empty:
            ax.scatter(swung["PlateLocSide"], swung["PlateLocHeight"],
                       marker="D", c=color, s=42, alpha=0.9,
                       edgecolors="white", linewidths=0.5, zorder=11)

        ax.set_title(f"{pt} ({count})", fontsize=7, fontweight="bold",
                     color=_DARK, pad=2)

    # Legend in last subplot space or below
    if n_types < n_rows * n_cols:
        ax_leg = fig.add_subplot(inner[n_rows - 1, n_cols - 1])
        ax_leg.axis("off")
        ax_leg.scatter([], [], marker="o", c="#666", s=22, label="Took")
        ax_leg.scatter([], [], marker="D", c="#666", s=28, label="Swung")
        ax_leg.legend(fontsize=6, loc="center", frameon=False)


# ── Pitch-Type Results Table ─────────────────────────────────────────────────

def _pitch_type_results_table(ax, bdf):
    """Show swing%, whiff%, avg EV per pitch type faced."""
    ax.axis("off")
    loc = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if "TaggedPitchType" not in bdf.columns:
        ax.text(0.5, 0.5, "No pitch type data", fontsize=7, color="#999",
                ha="center", va="center", transform=ax.transAxes)
        return

    rows = []
    for pt, grp in bdf.groupby("TaggedPitchType"):
        if pd.isna(pt) or len(grp) < 2:
            continue
        n = len(grp)
        swings = grp[grp["PitchCall"].isin(SWING_CALLS)]
        whiffs = grp[grp["PitchCall"] == "StrikeSwinging"]
        swing_pct = len(swings) / n * 100
        whiff_pct = len(whiffs) / max(len(swings), 1) * 100 if len(swings) > 0 else 0
        in_play = grp[grp["PitchCall"] == "InPlay"]
        ev_data = pd.to_numeric(in_play["ExitSpeed"], errors="coerce").dropna() if "ExitSpeed" in in_play.columns else pd.Series(dtype=float)
        avg_ev = f"{ev_data.mean():.1f}" if len(ev_data) > 0 else "-"
        rows.append([pt, str(n), f"{swing_pct:.0f}%", f"{whiff_pct:.0f}%", avg_ev])

    rows.sort(key=lambda r: int(r[1]), reverse=True)
    rows = rows[:4]  # Cap at 4 to prevent overflow
    if rows:
        ax.set_title("vs Pitch Type", fontsize=7, fontweight="bold",
                     color=_DARK, loc="left", pad=2)
        _styled_table(ax, rows, ["Pitch", "N", "Swing%", "Whiff%", "Avg EV"],
                      [0.25, 0.12, 0.18, 0.18, 0.18], fontsize=6, row_height=1.2)


# ── Count Breakdown ──────────────────────────────────────────────────────────

def _count_breakdown_table(ax, pdf, role="hitter"):
    """Performance by count state (ahead/even/behind)."""
    ax.axis("off")
    if "Balls" not in pdf.columns or "Strikes" not in pdf.columns:
        return
    b = pd.to_numeric(pdf["Balls"], errors="coerce")
    s = pd.to_numeric(pdf["Strikes"], errors="coerce")

    rows = []
    for label, mask in [("Ahead", s > b), ("Even", s == b), ("Behind", b > s)]:
        sub = pdf[mask]
        if sub.empty:
            continue
        n = len(sub)
        swings = sub[sub["PitchCall"].isin(SWING_CALLS)]
        whiffs = sub[sub["PitchCall"] == "StrikeSwinging"]
        swing_pct = len(swings) / n * 100
        whiff_pct = len(whiffs) / max(len(swings), 1) * 100 if len(swings) > 0 else 0
        if role == "hitter":
            ip = sub[sub["PitchCall"] == "InPlay"]
            ev = pd.to_numeric(ip["ExitSpeed"], errors="coerce").dropna() if "ExitSpeed" in ip.columns else pd.Series(dtype=float)
            ev_str = f"{ev.mean():.1f}" if len(ev) > 0 else "-"
            rows.append([label, str(n), f"{swing_pct:.0f}%", f"{whiff_pct:.0f}%", ev_str])
        else:
            csw = sub["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
            rows.append([label, str(n), f"{swing_pct:.0f}%", f"{csw:.0f}%", f"{whiff_pct:.0f}%"])

    if rows:
        ax.set_title("By Count", fontsize=7, fontweight="bold",
                     color=_DARK, loc="left", pad=2)
        if role == "hitter":
            _styled_table(ax, rows, ["Count", "N", "Swing%", "Whiff%", "Avg EV"],
                          [0.20, 0.12, 0.18, 0.18, 0.18], fontsize=6, row_height=1.2)
        else:
            _styled_table(ax, rows, ["Count", "N", "Swing%", "CSW%", "Whiff%"],
                          [0.20, 0.12, 0.18, 0.18, 0.18], fontsize=6, row_height=1.2)


# ── Velo Sparkline ───────────────────────────────────────────────────────────

def _velo_sparkline(ax, pdf):
    """Small velo-over-time sparkline for primary pitch type."""
    if "RelSpeed" not in pdf.columns or "TaggedPitchType" not in pdf.columns:
        ax.axis("off")
        return
    primary_pt = pdf["TaggedPitchType"].dropna().value_counts()
    if primary_pt.empty:
        ax.axis("off")
        return
    pt = primary_pt.index[0]
    velo = pd.to_numeric(pdf[pdf["TaggedPitchType"] == pt]["RelSpeed"], errors="coerce").dropna()
    if len(velo) < 3:
        ax.axis("off")
        return

    color = PITCH_COLORS.get(pt, "#d22d49")
    x = range(len(velo))
    ax.plot(x, velo.values, color=color, linewidth=1.2, alpha=0.8)
    ax.fill_between(x, velo.values, alpha=0.1, color=color)
    ax.axhline(velo.mean(), color="#aaa", linewidth=0.5, linestyle="--")
    ax.set_xlim(0, len(velo) - 1)
    ax.set_ylim(velo.min() - 2, velo.max() + 2)
    ax.tick_params(labelsize=5, length=2)
    ax.set_ylabel("mph", fontsize=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{pt} Velo ({velo.mean():.1f} avg)", fontsize=6,
                 fontweight="bold", color=_DARK, pad=2)


# ── Movement Profile ─────────────────────────────────────────────────────────

def _movement_profile(ax, pdf):
    mov = pdf.dropna(subset=["HorzBreak", "InducedVertBreak"])
    mov = filter_minor_pitches(mov)
    if mov.empty:
        ax.axis("off")
        return
    mov = mov.copy()
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
        ax.annotate(f'{r}"', xy=(0, r), fontsize=5, color="#999", ha="center", va="bottom")
    ax.axhline(0, color="#bbb", linewidth=0.5)
    ax.axvline(0, color="#bbb", linewidth=0.5)
    for pt in sorted(mov["TaggedPitchType"].unique()):
        sub = mov[mov["TaggedPitchType"] == pt]
        ax.scatter(sub["_HB"], sub["_IVB"], c=PITCH_COLORS.get(pt, "#aaa"),
                   s=18, alpha=0.8, edgecolors="white", linewidths=0.3, zorder=10, label=pt)
    lim = 28
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Horz Break (in)", fontsize=5, labelpad=1)
    ax.set_ylabel("Induced Vert Break (in)", fontsize=5, labelpad=1)
    ax.tick_params(labelsize=4)
    ax.legend(fontsize=4, loc="lower right", framealpha=0.7)
    ax.set_title("Movement Profile", fontsize=7, fontweight="bold", color=_DARK, pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Spray Chart ──────────────────────────────────────────────────────────────

def _spray_chart(ax, in_play_df):
    spray = in_play_df.dropna(subset=["Direction", "Distance"]).copy()
    spray = spray[spray["Direction"].between(-90, 90)]
    if spray.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No spray data", fontsize=7, color="#999",
                ha="center", va="center", transform=ax.transAxes)
        return
    angle_rad = np.radians(spray["Direction"])
    spray["x"] = spray["Distance"] * np.sin(angle_rad)
    spray["y"] = spray["Distance"] * np.cos(angle_rad)
    theta = np.linspace(-np.pi / 4, np.pi / 4, 80)
    grass_r = 400
    ax.fill(np.concatenate([[0], grass_r * np.sin(theta), [0]]),
            np.concatenate([[0], grass_r * np.cos(theta), [0]]),
            color="#e8f5e3", alpha=0.4, zorder=0)
    diamond_x = [0, 63.6, 0, -63.6, 0]
    diamond_y = [0, 63.6, 127.3, 63.6, 0]
    ax.fill(diamond_x, diamond_y, color="#f5e6d0", alpha=0.3, zorder=1)
    ax.plot(diamond_x, diamond_y, color="#c8a872", linewidth=0.7, zorder=2)
    fl = 350
    ax.plot([0, -fl * np.sin(np.pi/4)], [0, fl * np.cos(np.pi/4)], color="#ccc", linewidth=0.5)
    ax.plot([0, fl * np.sin(np.pi/4)], [0, fl * np.cos(np.pi/4)], color="#ccc", linewidth=0.5)
    ht_colors = {"GroundBall": "#d62728", "LineDrive": "#2ca02c", "FlyBall": "#1f77b4", "Popup": "#ff7f0e"}
    ht_names = {"GroundBall": "GB", "LineDrive": "LD", "FlyBall": "FB", "Popup": "PU"}
    for ht in ["GroundBall", "LineDrive", "FlyBall", "Popup"]:
        sub = spray[spray.get("TaggedHitType", pd.Series(dtype=str)) == ht] if "TaggedHitType" in spray.columns else pd.DataFrame()
        if not sub.empty:
            ax.scatter(sub["x"], sub["y"], c=ht_colors.get(ht, "#777"),
                       s=18, alpha=0.8, edgecolors="white", linewidths=0.3,
                       label=ht_names.get(ht, ht), zorder=10)
    ax.set_xlim(-300, 300); ax.set_ylim(-15, 400)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(fontsize=4, loc="upper right", framealpha=0.7)
    ax.set_title("Spray Chart", fontsize=7, fontweight="bold", color=_DARK, pad=2)


# ── Stuff+/Command+ Bars ────────────────────────────────────────────────────

def _build_stuff_population(data):
    """Build per-pitch-type Stuff+ population: dict of pitch_type -> array of
    per-pitcher average Stuff+ scores across all pitchers in the database."""
    stuff_pop = {}
    if data is None or data.empty:
        return stuff_pop
    stuff_df = _compute_stuff_plus(data.copy())
    if "StuffPlus" in stuff_df.columns:
        pitcher_pt = stuff_df.dropna(subset=["StuffPlus"]).groupby(
            ["Pitcher", "TaggedPitchType"])["StuffPlus"].mean().reset_index()
        for pt, grp in pitcher_pt.groupby("TaggedPitchType"):
            if len(grp) >= 3:
                stuff_pop[pt] = grp["StuffPlus"].values
    return stuff_pop


def _stuff_cmd_bars(ax, stuff_by_pt, cmd_df, stuff_pop=None):
    pitch_types = sorted(stuff_by_pt.keys())
    # Command+ already has a Percentile column from _compute_command_plus
    cmd_pctl_map = {}
    cmd_map = {}
    if cmd_df is not None and not cmd_df.empty and "Pitch" in cmd_df.columns:
        if "Percentile" in cmd_df.columns:
            cmd_pctl_map = dict(zip(cmd_df["Pitch"], cmd_df["Percentile"]))
        if "Command+" in cmd_df.columns:
            cmd_map = dict(zip(cmd_df["Pitch"], cmd_df["Command+"]))
    if not pitch_types:
        ax.axis("off")
        return

    # Convert Stuff+ raw scores to percentiles vs population
    stuff_pctls, cmd_pctls = [], []
    for pt in pitch_types:
        sv = stuff_by_pt.get(pt, 100)
        if stuff_pop and pt in stuff_pop and len(stuff_pop[pt]) >= 3:
            stuff_pctls.append(percentileofscore(stuff_pop[pt], sv, kind="rank"))
        else:
            # Fallback: approximate from z-score (100=mean, 10=1sd)
            from scipy.stats import norm
            stuff_pctls.append(norm.cdf((sv - 100) / 10) * 100)
        # Use Command+ Percentile directly if available
        if pt in cmd_pctl_map:
            cmd_pctls.append(cmd_pctl_map[pt])
        else:
            cv = cmd_map.get(pt, 100)
            from scipy.stats import norm
            cmd_pctls.append(norm.cdf((cv - 100) / 10) * 100)

    y = np.arange(len(pitch_types))
    bar_h = 0.35

    def _pctl_color(pctl):
        if pctl >= 75:
            return "#d22d49"
        if pctl <= 25:
            return "#2d7fc1"
        return "#9e9e9e"

    ax.barh(y + bar_h/2, stuff_pctls, bar_h, label="Stuff+",
            color=[_pctl_color(v) for v in stuff_pctls])
    ax.barh(y - bar_h/2, cmd_pctls, bar_h, label="Command+",
            color=[_pctl_color(v) for v in cmd_pctls], alpha=0.7)
    ax.axvline(50, color="#aaa", linestyle="--", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(pitch_types, fontsize=6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile", fontsize=5, labelpad=2)
    ax.legend(fontsize=5, loc="lower right")
    ax.tick_params(axis="x", labelsize=5)
    for i, (sv, cv) in enumerate(zip(stuff_pctls, cmd_pctls)):
        ax.text(min(sv + 1, 93), i + bar_h/2, f"{sv:.0f}", fontsize=5, va="center")
        ax.text(min(cv + 1, 93), i - bar_h/2, f"{cv:.0f}", fontsize=5, va="center")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("Stuff+ / Command+  (Percentile)", fontsize=7, fontweight="bold", color=_DARK, loc="left", pad=2)


# ── Best Swing Zones Heatmap ─────────────────────────────────────────────────

def _mpl_best_zone_heatmap(ax_heat, ax_key, bdf, bats):
    """Matplotlib 3x3 best-zone heatmap with description key for PDF hitter pages.

    ax_heat: axes for the 3x3 heatmap grid
    ax_key:  axes for the color key / description below the heatmap
    """
    from scipy.stats import percentileofscore as _pctile

    loc_df = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if len(loc_df) < 30:
        ax_heat.axis("off")
        ax_heat.text(0.5, 0.5, "Not enough data\n(need 30+ pitches)",
                     fontsize=7, color="#999",
                     ha="center", va="center", transform=ax_heat.transAxes)
        ax_key.axis("off")
        return

    zone_metrics = compute_zone_swing_metrics(bdf, bats)
    if zone_metrics is None:
        ax_heat.axis("off")
        ax_key.axis("off")
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
    ax_heat.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Text overlay — labeled EV and Brl%
    idx = 0
    for yb in range(3):
        for xb in range(3):
            m = zone_metrics.get((xb, yb), {})
            r, c = 2 - yb, xb
            v = grid[r, c]
            if not np.isnan(v):
                has_stats = m.get("n_contact", 0) >= 5
                ev_str = f"EV {m.get('ev_mean', 0):.0f}" if has_stats and "ev_mean" in m else ""
                bp_str = f"Brl {m.get('barrel_pct', 0):.0f}%" if has_stats and "barrel_pct" in m else ""
                txt = f"{ev_str}\n{bp_str}" if ev_str and bp_str else ev_str or bp_str
                brightness = v / 100.0
                color = "white" if brightness > 0.7 or brightness < 0.15 else "black"
                ax_heat.text(c, r, txt, ha="center", va="center", fontsize=5.5,
                             fontweight="bold", color=color)
            idx += 1

    # Zone styling
    for x in [0.5, 1.5]:
        ax_heat.axvline(x, color="white", lw=1.5, zorder=2)
    for y in [0.5, 1.5]:
        ax_heat.axhline(y, color="white", lw=1.5, zorder=2)
    ax_heat.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False,
                                edgecolor="#555", lw=1.2, ls="--", zorder=3))
    ax_heat.add_patch(Rectangle((0.17, -0.17), 1.66, 2.0, fill=False,
                                edgecolor="black", lw=2.5, zorder=4))
    ax_heat.set_yticks([0, 2])
    ax_heat.set_yticklabels(["UP", "DOWN"], fontsize=6, fontweight="bold")
    ax_heat.tick_params(axis="y", length=0, pad=2)
    b = bats[0].upper() if isinstance(bats, str) and bats else "R"
    if b == "R":
        left_lbl, right_lbl = "IN", "AWAY"
    elif b == "L":
        left_lbl, right_lbl = "AWAY", "IN"
    else:
        left_lbl, right_lbl = "L", "R"
    ax_heat.set_xticks([0, 2])
    ax_heat.set_xticklabels([left_lbl, right_lbl], fontsize=5.5, fontweight="bold")
    ax_heat.tick_params(axis="x", length=0, pad=2)
    ax_heat.set_title("Best Hitting Zones  (Season)", fontsize=7,
                      fontweight="bold", color=_DARK, pad=3)

    # ── Key / description below the heatmap ──
    ax_key.set_xlim(0, 1); ax_key.set_ylim(0, 1)
    ax_key.axis("off")

    # Color gradient bar
    gradient = np.linspace(0, 100, 256).reshape(1, -1)
    ax_key.imshow(gradient, cmap=cmap, aspect="auto",
                  extent=[0.15, 0.85, 0.68, 0.88], zorder=5)
    ax_key.text(0.14, 0.78, "Cold", fontsize=5, color="#c62828",
                fontweight="bold", va="center", ha="right", transform=ax_key.transAxes)
    ax_key.text(0.86, 0.78, "Hot", fontsize=5, color="#2e7d32",
                fontweight="bold", va="center", ha="left", transform=ax_key.transAxes)

    # Description lines
    n_bbe = sum(m.get("n_contact", 0) for m in zone_metrics.values())
    ax_key.text(0.50, 0.48, f"Season data  |  {len(loc_df)} pitches  |  {n_bbe} batted balls",
                fontsize=4.5, color="#666", va="top", ha="center", transform=ax_key.transAxes)
    ax_key.text(0.50, 0.24,
                "Score = 45% Exit Velo + 30% Barrel% + 25% Contact%",
                fontsize=4.5, color="#888", va="top", ha="center", transform=ax_key.transAxes)


# ── Feedback ─────────────────────────────────────────────────────────────────

def _feedback_block(ax, feedback):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    if not feedback:
        ax.text(0.02, 0.90, "No specific notes.", fontsize=5.5, color="#999",
                va="top", ha="left", transform=ax.transAxes)
        return
    fb_str, fb_area = _split_feedback(feedback, {})
    y = 0.97
    if fb_str:
        ax.text(0.02, y, "Strengths", fontsize=6.5, fontweight="bold",
                color="#2e7d32", va="top", ha="left", transform=ax.transAxes)
        y -= 0.12
        for fb in fb_str[:3]:
            ax.text(0.04, y, f"+ {fb}", fontsize=5, va="top", ha="left",
                    color=_DARK, transform=ax.transAxes)
            y -= 0.12
    if fb_area:
        ax.text(0.02, y, "Areas to Improve", fontsize=6.5, fontweight="bold",
                color="#c62828", va="top", ha="left", transform=ax.transAxes)
        y -= 0.12
        for fb in fb_area[:2]:
            ax.text(0.04, y, f"- {fb}", fontsize=5, va="top", ha="left",
                    color=_DARK, transform=ax.transAxes)
            y -= 0.12
    if not fb_str and not fb_area:
        text = "\n".join(f"~ {fb}" for fb in feedback[:4])
        ax.text(0.02, 0.95, "Feedback", fontsize=6.5, fontweight="bold",
                color=_DARK, va="top", ha="left", transform=ax.transAxes)
        ax.text(0.02, 0.80, text, fontsize=5, va="top", ha="left",
                color=_DARK, transform=ax.transAxes, linespacing=1.5)


# ── Season Delta Text ────────────────────────────────────────────────────────

def _season_delta_block(ax, series_df, season_df, role="hitter"):
    """Show series stats vs season averages with arrows."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.95, "Series vs Season", fontsize=7, fontweight="bold",
            color=_DARK, va="top", ha="left", transform=ax.transAxes)

    lines = []
    if role == "hitter":
        # Series stats
        in_play = series_df[series_df["PitchCall"] == "InPlay"]
        ev = pd.to_numeric(in_play["ExitSpeed"], errors="coerce").dropna() if "ExitSpeed" in in_play.columns else pd.Series(dtype=float)
        swings = series_df[series_df["PitchCall"].isin(SWING_CALLS)]
        whiffs = series_df[series_df["PitchCall"] == "StrikeSwinging"]

        # Season stats
        s_ip = season_df[season_df["PitchCall"] == "InPlay"]
        s_ev = pd.to_numeric(s_ip["ExitSpeed"], errors="coerce").dropna() if "ExitSpeed" in s_ip.columns else pd.Series(dtype=float)
        s_swings = season_df[season_df["PitchCall"].isin(SWING_CALLS)]
        s_whiffs = season_df[season_df["PitchCall"] == "StrikeSwinging"]

        if len(ev) > 0 and len(s_ev) > 0:
            d = ev.mean() - s_ev.mean()
            arrow = "+" if d > 0 else ""
            lines.append(f"Avg EV: {ev.mean():.1f}  ({arrow}{d:.1f} vs season)")
        if len(swings) > 0 and len(s_swings) > 0:
            w = len(whiffs) / len(swings) * 100
            sw = len(s_whiffs) / len(s_swings) * 100
            d = w - sw
            arrow = "+" if d > 0 else ""
            lines.append(f"Whiff%: {w:.1f}%  ({arrow}{d:.1f}% vs season)")
        loc = series_df.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        s_loc = season_df.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if len(loc) > 0 and len(s_loc) > 0:
            iz = in_zone_mask(loc)
            oz_sw = loc[~iz & loc["PitchCall"].isin(SWING_CALLS)]
            oz = loc[~iz]
            ch = len(oz_sw) / max(len(oz), 1) * 100
            s_iz = in_zone_mask(s_loc)
            s_oz_sw = s_loc[~s_iz & s_loc["PitchCall"].isin(SWING_CALLS)]
            s_oz = s_loc[~s_iz]
            s_ch = len(s_oz_sw) / max(len(s_oz), 1) * 100
            d = ch - s_ch
            arrow = "+" if d > 0 else ""
            lines.append(f"Chase%: {ch:.1f}%  ({arrow}{d:.1f}% vs season)")
    else:
        # Pitcher
        velo = pd.to_numeric(series_df["RelSpeed"], errors="coerce").dropna() if "RelSpeed" in series_df.columns else pd.Series(dtype=float)
        s_velo = pd.to_numeric(season_df["RelSpeed"], errors="coerce").dropna() if "RelSpeed" in season_df.columns else pd.Series(dtype=float)
        if len(velo) > 0 and len(s_velo) > 0:
            d = velo.mean() - s_velo.mean()
            arrow = "+" if d > 0 else ""
            lines.append(f"Avg Velo: {velo.mean():.1f}  ({arrow}{d:.1f} vs season)")
        csw = series_df["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
        s_csw = season_df["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
        d = csw - s_csw
        arrow = "+" if d > 0 else ""
        lines.append(f"CSW%: {csw:.1f}%  ({arrow}{d:.1f}% vs season)")

    text = "\n".join(f"  {l}" for l in lines)
    ax.text(0.02, 0.78, text, fontsize=6, va="top", ha="left",
            color=_DARK, transform=ax.transAxes, family="monospace", linespacing=1.6)


# ── AB Grades Table ──────────────────────────────────────────────────────────

def _ab_grades_table(ax, ab_rows):
    if not ab_rows:
        ax.axis("off")
        return
    col_labels = ["Game", "Inn", "vs Pitcher", "P", "Result", "Score"]
    rows = []
    for r in ab_rows:
        rows.append([
            str(r.get("Game", ""))[:6],
            str(r.get("Inning", "?")),
            str(r.get("vs Pitcher", "?"))[:16],
            str(r.get("Pitches", "?")),
            str(r.get("Result", "?")),
            str(r.get("Score", "?")),
        ])
    ax.axis("off")
    _styled_table(ax, rows, col_labels,
                  [0.14, 0.07, 0.28, 0.07, 0.22, 0.10],
                  fontsize=5.5, row_height=1.0)
    ax.set_title("At-Bat Results", fontsize=7, fontweight="bold",
                 color=_DARK, loc="left", pad=2)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_cover(combined_gd, game_ids, series_label, data=None):
    """Series cover page — redesigned with series record, score table, stats strip, and takeaways."""
    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")
    outer = gridspec.GridSpec(5, 1, figure=fig,
        height_ratios=[0.08, 0.18, 0.18, 0.10, 0.46],
        hspace=0.06, top=0.97, bottom=0.03, left=0.03, right=0.97)

    _header_bar(fig, outer[0], f"SERIES REPORT  |  {series_label}")

    # ── Row 1: Series record prominently ──
    ax_record = fig.add_subplot(outer[1])
    ax_record.axis("off")
    home_id = combined_gd["HomeTeam"].iloc[0] if "HomeTeam" in combined_gd.columns else "?"
    away_id = combined_gd["AwayTeam"].iloc[0] if "AwayTeam" in combined_gd.columns else "?"
    home = _friendly_team_name(home_id)
    away = _friendly_team_name(away_id)
    opp_name = away if away != "Davidson" else home

    # Compute series record from per-game RunsScored
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
        if dav_r > opp_r:
            dav_wins += 1
            result = "W"
        else:
            dav_losses += 1
            result = "L"
        game_scores.append((date_str, innings, dav_r, opp_r, result))

    # Sort games chronologically
    game_scores.sort(key=lambda x: x[0])

    ax_record.text(0.5, 0.70, f"DAVIDSON  {dav_wins}  -  {dav_losses}  {opp_name.upper()}",
                   fontsize=28, fontweight="900", color=_DARK,
                   va="center", ha="center", transform=ax_record.transAxes)
    ax_record.text(0.5, 0.25, f"{len(game_ids)}-Game Series  |  {len(combined_gd)} total pitches",
                   fontsize=11, color="#555", va="center", ha="center", transform=ax_record.transAxes)

    # ── Row 2: Per-game score table ──
    ax_scores = fig.add_subplot(outer[2])
    ax_scores.axis("off")
    if game_scores:
        score_rows = []
        for date_str, innings, dav_r, opp_r, result in game_scores:
            score_rows.append([date_str, f"{innings} inn", f"DAV {dav_r} - {opp_r} {opp_name[:3].upper()}", result])
        _styled_table(ax_scores, score_rows, ["Date", "Innings", "Score", "W/L"],
                      [0.20, 0.20, 0.40, 0.12], fontsize=8, row_height=1.5)

    # ── Row 3: Compact one-row stats strip ──
    dav_pitching = combined_gd[combined_gd["PitcherTeam"] == DAVIDSON_TEAM_ID]
    dav_hitting = combined_gd[combined_gd["BatterTeam"] == DAVIDSON_TEAM_ID]

    dav_ks = (dav_pitching["KorBB"] == "Strikeout").sum() if "KorBB" in dav_pitching.columns else 0
    dav_bbs = (dav_pitching["KorBB"] == "Walk").sum() if "KorBB" in dav_pitching.columns else 0
    dav_hits = dav_hitting["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum() if "PlayResult" in dav_hitting.columns else 0
    dav_hr = (dav_hitting["PlayResult"] == "HomeRun").sum() if "PlayResult" in dav_hitting.columns else 0
    dav_bbe = dav_hitting[(dav_hitting["PitchCall"] == "InPlay") & dav_hitting["ExitSpeed"].notna()] if "ExitSpeed" in dav_hitting.columns else pd.DataFrame()
    avg_ev = dav_bbe["ExitSpeed"].mean() if len(dav_bbe) > 0 else np.nan
    avg_ev_str = f"{avg_ev:.1f}" if pd.notna(avg_ev) else "-"

    loc_df = dav_hitting.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    chase_str = "-"
    if not loc_df.empty:
        iz = in_zone_mask(loc_df)
        oz = loc_df[~iz]
        oz_sw = oz[oz["PitchCall"].isin(SWING_CALLS)]
        chase_str = f"{len(oz_sw)/max(len(oz),1)*100:.0f}%"

    ax_strip = fig.add_subplot(outer[3])
    ax_strip.axis("off")
    strip_items = [
        ("K", str(dav_ks)), ("BB", str(dav_bbs)),
        ("H", str(dav_hits)), ("HR", str(dav_hr)),
        ("Avg EV", f"{avg_ev_str}"), ("Chase%", chase_str),
    ]
    n_items = len(strip_items)
    for i, (lbl, val) in enumerate(strip_items):
        x = (i + 0.5) / n_items
        ax_strip.text(x, 0.68, val, fontsize=14, fontweight="900",
                      color=_DARK, va="center", ha="center", transform=ax_strip.transAxes)
        ax_strip.text(x, 0.22, lbl, fontsize=7, fontweight="bold",
                      color="#888", va="center", ha="center", transform=ax_strip.transAxes)

    # ── Row 4: Top takeaway bullets ──
    ax_takeaways = fig.add_subplot(outer[4])
    ax_takeaways.axis("off")

    pitching_bullets, hitting_bullets = [], []
    try:
        pitching_bullets, hitting_bullets = _compute_takeaways(combined_gd, data)
    except Exception:
        pass

    y_pos = 0.95
    ax_takeaways.text(0.03, y_pos, "KEY TAKEAWAYS", fontsize=10, fontweight="bold",
                      color=_DARK, va="top", ha="left", transform=ax_takeaways.transAxes)
    y_pos -= 0.08

    # Show top 3 pitching + top 3 hitting
    if pitching_bullets:
        ax_takeaways.text(0.03, y_pos, "PITCHING", fontsize=8, fontweight="bold",
                          color=_SECTION_BG, va="top", ha="left", transform=ax_takeaways.transAxes)
        y_pos -= 0.06
        for bullet in pitching_bullets[:3]:
            ax_takeaways.text(0.05, y_pos, f"- {bullet}", fontsize=7.5, color=_DARK,
                              va="top", ha="left", transform=ax_takeaways.transAxes)
            y_pos -= 0.06

    y_pos -= 0.03
    if hitting_bullets:
        ax_takeaways.text(0.03, y_pos, "HITTING", fontsize=8, fontweight="bold",
                          color=_SECTION_BG, va="top", ha="left", transform=ax_takeaways.transAxes)
        y_pos -= 0.06
        for bullet in hitting_bullets[:3]:
            ax_takeaways.text(0.05, y_pos, f"- {bullet}", fontsize=7.5, color=_DARK,
                              va="top", ha="left", transform=ax_takeaways.transAxes)
            y_pos -= 0.06

    _add_page_number(fig)
    return fig


def _render_takeaways_page(combined_gd, data, series_label):
    """Coach takeaways page — aggregated across all games in the series."""
    pitching_bullets, hitting_bullets = _compute_takeaways(combined_gd, data)

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(3, 1, figure=fig,
        height_ratios=[0.08, 0.46, 0.46],
        hspace=0.08, top=0.97, bottom=0.03, left=0.03, right=0.97)

    _header_bar(fig, outer[0], f"SERIES REPORT  |  {series_label}  |  COACH TAKEAWAYS")

    # Pitching takeaways
    ax_pit = fig.add_subplot(outer[1])
    ax_pit.axis("off")
    ax_pit.set_title("PITCHING", fontsize=11, fontweight="bold",
                     color=_DARK, loc="left", pad=6)
    if pitching_bullets:
        y = 0.88
        for b in pitching_bullets:
            ax_pit.text(0.04, y, f"- {b}", fontsize=9.5, va="top", ha="left",
                        color=_DARK, transform=ax_pit.transAxes, fontfamily="sans-serif")
            y -= 0.14
    else:
        ax_pit.text(0.04, 0.85, "No pitching takeaways available.", fontsize=9,
                    color="#999", va="top", ha="left", transform=ax_pit.transAxes)

    # Hitting takeaways
    ax_hit = fig.add_subplot(outer[2])
    ax_hit.axis("off")
    ax_hit.set_title("HITTING", fontsize=11, fontweight="bold",
                     color=_DARK, loc="left", pad=6)
    if hitting_bullets:
        y = 0.88
        for b in hitting_bullets:
            ax_hit.text(0.04, y, f"- {b}", fontsize=9.5, va="top", ha="left",
                        color=_DARK, transform=ax_hit.transAxes, fontfamily="sans-serif")
            y -= 0.14
    else:
        ax_hit.text(0.04, 0.85, "No hitting takeaways available.", fontsize=9,
                    color="#999", va="top", ha="left", transform=ax_hit.transAxes)

    _add_page_number(fig)
    return fig


def _render_pitcher_page(pdf, data, pitcher, series_label, stuff_pop=None):
    """Individual pitcher page with velo sparkline and count breakdown."""
    n_pitches = len(pdf)
    if n_pitches < 20:
        return None
    season_pdf = data[data["Pitcher"] == pitcher] if data is not None else pd.DataFrame()
    _grades, feedback, stuff_by_pt, cmd_df = _compute_pitcher_grades(pdf, data, pitcher)
    ip_est = _pg_estimate_ip(pdf)

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    # 4 rows x 2 cols (radar removed)
    outer = gridspec.GridSpec(4, 2, figure=fig,
        height_ratios=[0.08, 0.28, 0.30, 0.34],
        hspace=0.20, wspace=0.15,
        top=0.97, bottom=0.05, left=0.07, right=0.96)

    # Row 0: Header
    ax_hdr = fig.add_subplot(outer[0, :])
    _player_header(ax_hdr, pitcher, n_pitches, f"~{ip_est} IP")

    # Row 1 left: Pitch mix + zone
    mix_zone = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0], wspace=0.15)
    ax_mix = fig.add_subplot(mix_zone[0, 0])
    ax_mix.axis("off")
    ax_mix.set_title("Pitch Mix", fontsize=7, fontweight="bold", color=_DARK, loc="left", pad=2)
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
                          [0.30, 0.15, 0.20, 0.20], fontsize=6, row_height=1.2)
    ax_zone = fig.add_subplot(mix_zone[0, 1])
    _zone_scatter(ax_zone, pdf)

    # Row 1 right: Movement profile
    ax_mov = fig.add_subplot(outer[1, 1])
    _movement_profile(ax_mov, pdf)

    # Row 2 left: Stuff+/Cmd+ (percentile)
    ax_bars = fig.add_subplot(outer[2, 0])
    _stuff_cmd_bars(ax_bars, stuff_by_pt, cmd_df, stuff_pop=stuff_pop)

    # Row 2 right: Velo sparkline + Count breakdown
    r2_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2, 1],
        height_ratios=[0.45, 0.55], hspace=0.45)
    ax_velo = fig.add_subplot(r2_right[0])
    _velo_sparkline(ax_velo, pdf)
    ax_count = fig.add_subplot(r2_right[1])
    _count_breakdown_table(ax_count, pdf, role="pitcher")

    # Row 3 left: Percentile bars
    pctl = _compute_pitcher_percentile_metrics(pdf, season_pdf)
    ax_pctl = fig.add_subplot(outer[3, 0])
    _flat_percentile_bars(ax_pctl, pctl)

    # Row 3 right: Feedback + season delta
    r3_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[3, 1],
        height_ratios=[0.55, 0.45], hspace=0.1)
    ax_fb = fig.add_subplot(r3_right[0])
    _feedback_block(ax_fb, feedback)
    ax_delta = fig.add_subplot(r3_right[1])
    _season_delta_block(ax_delta, pdf, season_pdf, role="pitcher")

    _add_page_number(fig)
    return fig


def _render_hitter_page(bdf, data, batter, series_label, game_ids=None):
    """Individual hitter page with pitch locations seen, pitch-type table, and best swing zones."""
    n_pitches = len(bdf)
    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in bdf.columns]
    n_pas = bdf.drop_duplicates(subset=pa_cols).shape[0] if len(pa_cols) >= 2 else 0
    if n_pas < 3:
        return None
    season_bdf = data[data["Batter"] == batter] if data is not None else pd.DataFrame()
    _grades, feedback = _compute_hitter_grades(bdf, data, batter)

    # Build AB rows across all games
    sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in bdf.columns]
    ab_rows = []
    if len(pa_cols) >= 2:
        for pa_key, ab in bdf.groupby(pa_cols[1:]):
            ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
            score, letter, result = _grade_at_bat(ab_sorted, season_bdf)
            inn = ab_sorted.iloc[0].get("Inning", "?")
            vs_pitcher = display_name(ab_sorted.iloc[0]["Pitcher"], escape_html=False) if "Pitcher" in ab_sorted.columns else "?"
            game_id = ab_sorted.iloc[0].get("GameID", "")
            game_label = ""
            if game_id and "Date" in ab_sorted.columns:
                d = ab_sorted.iloc[0].get("Date")
                if pd.notna(d):
                    game_label = pd.Timestamp(d).strftime("%m/%d")
            if result == "Undefined":
                result = ab_sorted.iloc[-1].get("PitchCall", "Out") if len(ab_sorted) > 0 else "Out"
            ab_rows.append({
                "Game": game_label, "Inning": inn, "vs Pitcher": vs_pitcher,
                "Pitches": len(ab_sorted), "Result": result, "Score": score,
            })

    # Cap at-bats for PDF layout (Streamlit shows all)
    _MAX_AB_PDF = 8
    ab_display = ab_rows[:_MAX_AB_PDF]
    ab_overflow = max(0, len(ab_rows) - _MAX_AB_PDF)

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    # 4 rows x 2 cols — redesigned layout
    outer = gridspec.GridSpec(4, 2, figure=fig,
        height_ratios=[0.07, 0.25, 0.35, 0.33],
        hspace=0.14, wspace=0.12,
        top=0.97, bottom=0.04, left=0.04, right=0.96)

    # Row 0: Header
    ax_hdr = fig.add_subplot(outer[0, :])
    _player_header(ax_hdr, batter, n_pitches, f"{n_pas} PA")

    # Row 1: 3-column layout (Discipline+Feedback | Best Zones | Spray)
    r1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, :],
        width_ratios=[0.35, 0.30, 0.35], wspace=0.10)

    # Left: Discipline + Feedback (stacked vertically)
    r1_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=r1[0],
        height_ratios=[0.50, 0.50], hspace=0.08)

    ax_disc = fig.add_subplot(r1_left[0])
    ax_disc.axis("off")
    ax_disc.set_title("Plate Discipline & BBQ", fontsize=7, fontweight="bold",
                      color=_DARK, loc="left", pad=2)
    loc_df = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    disc_lines = []
    if not loc_df.empty:
        iz = in_zone_mask(loc_df)
        iz_sw = loc_df[iz & loc_df["PitchCall"].isin(SWING_CALLS)]
        oz_sw = loc_df[~iz & loc_df["PitchCall"].isin(SWING_CALLS)]
        swings = loc_df[loc_df["PitchCall"].isin(SWING_CALLS)]
        whiffs = bdf[bdf["PitchCall"] == "StrikeSwinging"]
        disc_lines.append(f"Zone Swing%: {len(iz_sw)/max(len(loc_df[iz]),1)*100:.1f}%")
        disc_lines.append(f"Chase%:      {len(oz_sw)/max(len(loc_df[~iz]),1)*100:.1f}%")
        disc_lines.append(f"Whiff%:      {len(whiffs)/max(len(swings),1)*100:.1f}%" if len(swings) > 0 else "Whiff%: N/A")
    in_play = bdf[bdf["PitchCall"] == "InPlay"]
    bbe_df = in_play.dropna(subset=["ExitSpeed"]) if "ExitSpeed" in bdf.columns else pd.DataFrame()
    if len(bbe_df) > 0:
        ev = bbe_df["ExitSpeed"]
        disc_lines.append(f"Avg EV:      {ev.mean():.1f} mph")
        disc_lines.append(f"Max EV:      {ev.max():.1f} mph")
        disc_lines.append(f"Hard Hit%:   {(ev >= 95).mean()*100:.0f}%")
    else:
        disc_lines.append("No batted ball data.")
    ax_disc.text(0.05, 0.95, "\n".join(disc_lines), fontsize=6.5, va="top", ha="left",
                 color=_DARK, transform=ax_disc.transAxes, family="monospace", linespacing=1.4)

    # Feedback (Strengths / Areas) below discipline stats
    ax_fb = fig.add_subplot(r1_left[1])
    _feedback_block(ax_fb, feedback)

    # Center: Best Swing Zones (season-long data) — heatmap + key
    r1_center = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=r1[1],
        height_ratios=[0.70, 0.30], hspace=0.22)
    ax_zones = fig.add_subplot(r1_center[0])
    ax_zones_key = fig.add_subplot(r1_center[1])
    bats = season_bdf["BatterSide"].iloc[0] if "BatterSide" in season_bdf.columns and len(season_bdf) > 0 else "Right"
    _mpl_best_zone_heatmap(ax_zones, ax_zones_key, season_bdf, bats)

    # Right: Spray chart
    ax_spray = fig.add_subplot(r1[2])
    if not in_play.empty:
        _spray_chart(ax_spray, in_play)
    else:
        ax_spray.axis("off")
        ax_spray.text(0.5, 0.5, "No in-play data", fontsize=7, color="#999",
                      ha="center", va="center", transform=ax_spray.transAxes)

    # Row 2 left: Pitch Locations Seen (bigger)
    _pitch_locations_seen(fig, outer[2, 0], bdf)

    # Row 2 right: Pitch-type results + Count breakdown
    r2_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2, 1],
        height_ratios=[0.55, 0.45], hspace=0.50)
    ax_pt = fig.add_subplot(r2_right[0])
    _pitch_type_results_table(ax_pt, bdf)
    ax_count = fig.add_subplot(r2_right[1])
    _count_breakdown_table(ax_count, bdf, role="hitter")

    # Row 3 left: Percentile bars + Season delta
    r3_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[3, 0],
        height_ratios=[0.65, 0.35], hspace=0.10)
    pctl = _compute_hitter_percentile_metrics(bdf, season_bdf)
    ax_pctl = fig.add_subplot(r3_left[0])
    _flat_percentile_bars(ax_pctl, pctl)
    ax_delta = fig.add_subplot(r3_left[1])
    _season_delta_block(ax_delta, bdf, season_bdf, role="hitter")

    # Row 3 right: AB grades table
    ax_ab = fig.add_subplot(outer[3, 1])
    if ab_display:
        _ab_grades_table(ax_ab, ab_display)
        if ab_overflow > 0:
            ax_ab.text(0.98, 0.02, f"+{ab_overflow} more", fontsize=5, color="#999",
                       ha="right", va="bottom", transform=ax_ab.transAxes)
    else:
        ax_ab.axis("off")

    _add_page_number(fig)
    return fig


def _render_pitching_summary(combined_gd, series_label):
    """Pitching staff summary."""
    dav_pitching = combined_gd[combined_gd["PitcherTeam"] == DAVIDSON_TEAM_ID].copy()
    if dav_pitching.empty:
        return None
    pitchers = dav_pitching.groupby("Pitcher").size().sort_values(ascending=False).index.tolist()

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")
    outer = gridspec.GridSpec(2, 1, figure=fig,
        height_ratios=[0.08, 0.92],
        hspace=0.08, top=0.97, bottom=0.03, left=0.03, right=0.97)
    _header_bar(fig, outer[0], f"SERIES REPORT  |  {series_label}  |  PITCHING")

    ax_table = fig.add_subplot(outer[1])
    ax_table.axis("off")
    ax_table.set_title("STAFF SUMMARY", fontsize=9, fontweight="bold",
                       color=_DARK, loc="left", pad=4)
    col_labels = ["Pitcher", "Pitches", "IP", "K", "BB", "H", "Avg Velo", "Max Velo", "CSW%"]
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
        csw = f"{pdf['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging']).mean()*100:.1f}%"
        jersey = JERSEY.get(pitcher, "")
        dname = display_name(pitcher, escape_html=False)
        label = f"#{jersey} {dname}" if jersey else dname
        rows.append([label, str(n), ip, str(ks), str(bbs), str(hits), avg_v, max_v, csw])
    _styled_table(ax_table, rows, col_labels,
                  [0.20, 0.08, 0.07, 0.06, 0.06, 0.06, 0.09, 0.09, 0.09],
                  fontsize=7.5, row_height=1.5)
    _add_page_number(fig)
    return fig


def _render_hitting_summary(combined_gd, series_label):
    """Hitting lineup summary."""
    dav_hitting = combined_gd[combined_gd["BatterTeam"] == DAVIDSON_TEAM_ID].copy()
    if dav_hitting.empty:
        return None
    batters = dav_hitting.groupby("Batter").size().sort_values(ascending=False).index.tolist()

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")
    outer = gridspec.GridSpec(2, 1, figure=fig,
        height_ratios=[0.08, 0.92],
        hspace=0.08, top=0.97, bottom=0.03, left=0.03, right=0.97)
    _header_bar(fig, outer[0], f"SERIES REPORT  |  {series_label}  |  HITTING")

    ax_table = fig.add_subplot(outer[1])
    ax_table.axis("off")
    ax_table.set_title("LINEUP SUMMARY", fontsize=9, fontweight="bold",
                       color=_DARK, loc="left", pad=4)
    col_labels = ["Batter", "PA", "H", "2B", "HR", "BB", "K", "Avg EV", "Max EV", "HH%"]
    rows = []
    for batter in batters:
        bdf = dav_hitting[dav_hitting["Batter"] == batter]
        pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in bdf.columns]
        pa = bdf.drop_duplicates(subset=pa_cols).shape[0] if len(pa_cols) >= 2 else 0
        pr = bdf["PlayResult"] if "PlayResult" in bdf.columns else pd.Series(dtype=str)
        hits = pr.isin(["Single", "Double", "Triple", "HomeRun"]).sum()
        doubles = (pr == "Double").sum()
        hrs = (pr == "HomeRun").sum()
        bbs = (bdf["KorBB"] == "Walk").sum() if "KorBB" in bdf.columns else 0
        ks = (bdf["KorBB"] == "Strikeout").sum() if "KorBB" in bdf.columns else 0
        ip = bdf[bdf["PitchCall"] == "InPlay"]
        ev_data = pd.to_numeric(ip["ExitSpeed"], errors="coerce").dropna() if "ExitSpeed" in ip.columns else pd.Series(dtype=float)
        avg_ev = f"{ev_data.mean():.1f}" if len(ev_data) > 0 else "-"
        max_ev = f"{ev_data.max():.1f}" if len(ev_data) > 0 else "-"
        hh_pct = f"{(ev_data >= 95).mean()*100:.0f}%" if len(ev_data) > 0 else "-"
        jersey = JERSEY.get(batter, "")
        dname = display_name(batter, escape_html=False)
        label = f"#{jersey} {dname}" if jersey else dname
        rows.append([label, str(pa), str(hits), str(doubles), str(hrs),
                     str(bbs), str(ks), avg_ev, max_ev, hh_pct])
    _styled_table(ax_table, rows, col_labels,
                  [0.20, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.10, 0.10, 0.08],
                  fontsize=7, row_height=1.3)
    _add_page_number(fig)
    return fig


# ── Per-Hitter AB Review Pages ───────────────────────────────────────────────

def _mpl_pa_zone_plot(ax, ab_df):
    """Numbered pitch locations for a single PA, colored by pitch type."""
    ab_numbered = ab_df.copy()
    ab_numbered["_OrigPitchNum"] = range(1, len(ab_numbered) + 1)
    loc = ab_numbered.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(0.5, 4.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#ddd")
    _draw_zone_rect(ax)
    # 3x3 grid lines
    third_x = ZONE_SIDE / 3
    third_y = (ZONE_HEIGHT_TOP - ZONE_HEIGHT_BOT) / 3
    for x in [-third_x, third_x]:
        ax.plot([x, x], [ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP],
                color="#ccc", linewidth=0.5, linestyle=":", zorder=4)
    for i in [1, 2]:
        y = ZONE_HEIGHT_BOT + third_y * i
        ax.plot([-ZONE_SIDE, ZONE_SIDE], [y, y],
                color="#ccc", linewidth=0.5, linestyle=":", zorder=4)
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


def _render_hitter_ab_pages(bdf, data, batter, series_label, game_ids):
    """Per-AB review pages for a hitter across a series. ~3 ABs per page.
    Returns list of Figure objects."""
    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in bdf.columns]
    sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in bdf.columns]
    if len(pa_cols) < 2:
        return []

    season_bdf = data[data["Batter"] == batter] if data is not None and not data.empty else pd.DataFrame()

    pa_list = []
    for pa_key, ab in bdf.groupby(pa_cols[1:]):
        ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
        inn = ab_sorted.iloc[0].get("Inning", "?")
        game_id = ab_sorted.iloc[0].get("GameID", "")
        game_date_str = ""
        if "Date" in ab_sorted.columns:
            d = ab_sorted.iloc[0].get("Date")
            if pd.notna(d):
                game_date_str = pd.Timestamp(d).strftime("%m/%d")
        pa_list.append((inn, ab_sorted, game_date_str))
    pa_list.sort(key=lambda x: (x[2], int(x[0]) if pd.notna(x[0]) and str(x[0]).isdigit() else 99))

    if not pa_list:
        return []

    dname = display_name(batter, escape_html=False)
    figures = []
    PAS_PER_PAGE = 3

    for page_start in range(0, len(pa_list), PAS_PER_PAGE):
        page_pas = pa_list[page_start:page_start + PAS_PER_PAGE]
        n_pas_page = len(page_pas)

        fig = plt.figure(figsize=_FIG_SIZE)
        fig.patch.set_facecolor("white")

        h_ratios = [0.08] + [0.28] * n_pas_page
        remaining = 1.0 - sum(h_ratios)
        if remaining > 0.01:
            h_ratios.append(remaining)
        n_rows = len(h_ratios)

        page_outer = gridspec.GridSpec(n_rows, 1, figure=fig,
            height_ratios=h_ratios, hspace=0.10,
            top=0.97, bottom=0.03, left=0.04, right=0.96)

        _header_bar(fig, page_outer[0],
                    f"SERIES AB REVIEW  |  {dname}  |  {series_label}")

        for pa_i, (inn, ab_sorted, game_date_str) in enumerate(page_pas):
            row_gs = gridspec.GridSpecFromSubplotSpec(1, 2,
                subplot_spec=page_outer[1 + pa_i],
                wspace=0.08, width_ratios=[0.6, 0.4])

            vs_pitcher = display_name(ab_sorted.iloc[0]["Pitcher"], escape_html=False) if "Pitcher" in ab_sorted.columns else "?"
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

            score, letter, _ = _grade_at_bat(ab_sorted, season_bdf)
            date_prefix = f"[{game_date_str}] " if game_date_str else ""
            pa_header = f"{date_prefix}Inn {inn} vs {vs_pitcher} — {result} ({len(ab_sorted)}p) [{letter}]"

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

            # Swing decision annotation below the table
            loc = ab_sorted.dropna(subset=["PlateLocSide", "PlateLocHeight"])
            notes = []
            if not loc.empty:
                iz = in_zone_mask(loc)
                oz_swings = loc[~iz & loc["PitchCall"].isin(SWING_CALLS)]
                iz_pitches = loc[iz]
                iz_takes = iz_pitches[~iz_pitches["PitchCall"].isin(SWING_CALLS)]
                if len(oz_swings) > 0:
                    notes.append(f"Chased {len(oz_swings)}x outside zone")
                if len(iz_pitches) > 0 and len(iz_takes) > 0:
                    take_pct = len(iz_takes) / len(iz_pitches) * 100
                    if take_pct > 50:
                        notes.append(f"Took {take_pct:.0f}% in-zone")
            ip = ab_sorted[ab_sorted["PitchCall"] == "InPlay"]
            if "ExitSpeed" in ip.columns and not ip.empty:
                ev = ip["ExitSpeed"].dropna()
                if not ev.empty:
                    notes.append(f"EV: {ev.iloc[0]:.0f} mph")
            if notes:
                ax_table.text(0.02, 0.02, " | ".join(notes),
                             fontsize=5.5, color="#666", va="bottom", ha="left",
                             transform=ax_table.transAxes)

            # Right: zone plot
            ax_zone = fig.add_subplot(row_gs[0, 1])
            _mpl_pa_zone_plot(ax_zone, ab_sorted)

        _add_page_number(fig)
        figures.append(fig)

    return figures


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_series_pdf_bytes(game_ids, data, series_label) -> bytes:
    """Generate the series report PDF combining multiple games.

    Args:
        game_ids: list of GameID strings to combine
        data: full DataFrame with all season data
        series_label: e.g. "Bryant @ Davidson  |  Feb 13-14, 2026"

    Returns:
        Raw PDF bytes.
    """
    global _page_num
    _page_num = 0

    combined_gd = data[data["GameID"].isin(game_ids)].copy()
    if combined_gd.empty:
        return b""

    buf = io.BytesIO()

    with PdfPages(buf) as pdf:
        # 1. Cover (redesigned)
        try:
            fig = _render_cover(combined_gd, game_ids, series_label, data=data)
            if fig:
                pdf.savefig(fig); plt.close(fig)
        except Exception:
            traceback.print_exc()

        # 2. Coach Takeaways (NEW)
        try:
            fig = _render_takeaways_page(combined_gd, data, series_label)
            if fig:
                pdf.savefig(fig); plt.close(fig)
        except Exception:
            traceback.print_exc()

        # 3. Pitching Summary
        try:
            fig = _render_pitching_summary(combined_gd, series_label)
            if fig:
                pdf.savefig(fig); plt.close(fig)
        except Exception:
            traceback.print_exc()

        # 4. Individual Pitcher pages (20+ pitches only)
        dav_pitching = combined_gd[combined_gd["PitcherTeam"] == DAVIDSON_TEAM_ID].copy()
        if not dav_pitching.empty:
            # Build Stuff+ population distribution once for percentile conversion
            try:
                stuff_pop = _build_stuff_population(data)
            except Exception:
                stuff_pop = {}
            pitchers = dav_pitching.groupby("Pitcher").size().sort_values(ascending=False).index.tolist()
            for pitcher in pitchers:
                try:
                    pitcher_df = dav_pitching[dav_pitching["Pitcher"] == pitcher].copy()
                    fig = _render_pitcher_page(pitcher_df, data, pitcher, series_label,
                                              stuff_pop=stuff_pop)
                    if fig:
                        pdf.savefig(fig); plt.close(fig)
                except Exception:
                    traceback.print_exc()

        # 5. Hitting Summary
        try:
            fig = _render_hitting_summary(combined_gd, series_label)
            if fig:
                pdf.savefig(fig); plt.close(fig)
        except Exception:
            traceback.print_exc()

        # 6. Individual Hitter pages + AB review pages (5+ PAs only)
        dav_hitting = combined_gd[combined_gd["BatterTeam"] == DAVIDSON_TEAM_ID].copy()
        if not dav_hitting.empty:
            batters = dav_hitting.groupby("Batter").size().sort_values(ascending=False).index.tolist()
            for batter in batters:
                try:
                    batter_df = dav_hitting[dav_hitting["Batter"] == batter].copy()
                    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in batter_df.columns]
                    n_pas = batter_df.drop_duplicates(subset=pa_cols).shape[0] if len(pa_cols) >= 2 else 0
                    # Summary page (3+ PAs)
                    fig = _render_hitter_page(batter_df, data, batter, series_label, game_ids)
                    if fig:
                        pdf.savefig(fig); plt.close(fig)
                    # AB review pages (5+ PAs only to keep PDF fast)
                    if n_pas >= 5:
                        ab_figs = _render_hitter_ab_pages(batter_df, data, batter, series_label, game_ids)
                        for ab_fig in ab_figs:
                            pdf.savefig(ab_fig); plt.close(ab_fig)
                except Exception:
                    traceback.print_exc()

    buf.seek(0)
    return buf.read()
