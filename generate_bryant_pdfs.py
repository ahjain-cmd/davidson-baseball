"""Generate one-page PDF scouting reports for each Bryant hitter.

Usage:
    python generate_bryant_pdfs.py

Outputs to  bryant_scouting_pdfs/  (one PDF per hitter + combined PDF).
"""
import sys, os, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from data.bryant_combined import (
    load_bryant_combined_pack,
    build_bryant_combined_pack,
    load_bryant_pitches,
    load_bryant_pitcher_pitches,
)
from data.loader import load_davidson_data, _tm_team, _tm_player, _safe_num
from config import (
    BRYANT_JERSEY,
    BRYANT_POSITION,
    SWING_CALLS,
    ZONE_SIDE,
    ZONE_HEIGHT_BOT,
    ZONE_HEIGHT_TOP,
    normalize_pitch_types,
    tm_name_to_trackman,
    ROSTER_2026,
    POSITION,
    JERSEY,
)
from analytics.zone_vulnerability import compute_hole_scores_3x3
from _pages.scouting import (
    _get_our_pitcher_arsenal,
    _get_opp_hitter_profile,
    _get_our_hitter_profile,
    _get_opp_pitcher_profile,
    _score_pitcher_vs_hitter,
    _score_hitter_vs_pitcher,
)
from decision_engine.recommenders.defense_recommender import classify_shift
from decision_engine.data.baserunning_data import (
    load_speed_scores,
    load_stolen_bases_hitters,
    load_stolen_bases_catchers,
    load_stolen_bases_pitchers,
    load_pickoffs,
    load_count_sb_squeeze,
)

# ── Constants ────────────────────────────────────────────────────────────────
_TEAM = "Bryant (2024-25 Combined)"
_AB_RESULTS = {"Single", "Double", "Triple", "HomeRun", "Out", "Strikeout", "FieldersChoice", "Error"}
_TB_MAP = {"Single": 1, "Double": 2, "Triple": 3, "HomeRun": 4}
X_EDGES = np.array([-1.5, -0.5, 0.5, 1.5])
Y_EDGES = np.array([0.5, 2.0, 3.0, 4.5])

# Colormaps
_HOLE_CMAP = LinearSegmentedColormap.from_list("holes", ["#2ecc71", "#f1c40f", "#e74c3c"])
_SLG_CMAP = LinearSegmentedColormap.from_list("slg", ["#3498db", "#ffffff", "#e74c3c"])
_HEAT_CMAP = LinearSegmentedColormap.from_list("heat", ["#ffffff", "#3498db", "#e74c3c"])
_SWING_CMAP = LinearSegmentedColormap.from_list("swing", ["#3498db", "#ffffff", "#e74c3c"])
_EV_CMAP = LinearSegmentedColormap.from_list("ev", ["#3498db", "#e74c3c"])

# Pure-pitcher positions to skip
_PITCHER_ONLY = {"RHP", "LHP"}

# ── Davidson pitcher role ordering ───────────────────────────────────────
_DAV_STARTERS = ["Perkins, Wilson", "Cavanaugh, Cooper", "Vokal, Jacob"]

_DAV_RELIEVERS = [
    "Hall, Edward", "Yochum, Simon", "Hoyt, Ivan", "Champey, Brycen",
    "Furr, Keely", "Jones, Parker", "Marenghi, Will", "Papciak, Will",
    "Taggart, Carson", "Smith, Daniel", "Wille, Tyler",
]

_DAV_INJURED = ["Banks, Will", "Whelan, Thomas", "Hamilton, Matthew"]

_DAV_DNP = {"Hultquist, Henry", "Ban, Jason", "Ludwig, Landon"}

# Alias mapping: some names appear differently in arsenal dicts vs role lists
_PITCHER_ALIASES = {
    "Hall, Ed": "Hall, Edward",
    "Hamilton, Matt": "Hamilton, Matthew",
}
_PITCHER_ALIASES_REV = {v: k for k, v in _PITCHER_ALIASES.items()}

# Bryant pitcher throw-hand overrides (last name → "L"/"R")
_BRYANT_PITCHER_THROWS = {
    "Vining": "L",
    "Galusha": "R",
    "Flaherty": "R",
    "Dressler": "R",
}


def _resolve_pitcher_name(name, arsenal_dict):
    """Return the key in *arsenal_dict* that matches *name* (handles aliases)."""
    if name in arsenal_dict:
        return name
    alias = _PITCHER_ALIASES.get(name) or _PITCHER_ALIASES_REV.get(name)
    if alias and alias in arsenal_dict:
        return alias
    return None


def _order_pitchers_by_role(dav_arsenals):
    """Return [(arsenal_key, role_tag), ...] ordered: starters → relievers → others → injured.

    Skips DNP entirely.  Only includes pitchers that have an arsenal in *dav_arsenals*.
    """
    ordered = []
    seen = set()

    for name in _DAV_STARTERS:
        key = _resolve_pitcher_name(name, dav_arsenals)
        if key and key not in seen:
            ordered.append((key, "starter"))
            seen.add(key)

    for name in _DAV_RELIEVERS:
        key = _resolve_pitcher_name(name, dav_arsenals)
        if key and key not in seen:
            ordered.append((key, "reliever"))
            seen.add(key)

    # Others with arsenals not in any list and not DNP
    all_listed = set()
    for lst in [_DAV_STARTERS, _DAV_RELIEVERS, _DAV_INJURED]:
        for n in lst:
            all_listed.add(n)
            a = _PITCHER_ALIASES.get(n)
            if a:
                all_listed.add(a)
            a2 = _PITCHER_ALIASES_REV.get(n)
            if a2:
                all_listed.add(a2)
    dnp_expanded = set()
    for n in _DAV_DNP:
        dnp_expanded.add(n)
        a = _PITCHER_ALIASES.get(n)
        if a:
            dnp_expanded.add(a)
        a2 = _PITCHER_ALIASES_REV.get(n)
        if a2:
            dnp_expanded.add(a2)

    # Skip unlisted pitchers — only show starters, relievers, and injured

    for name in _DAV_INJURED:
        key = _resolve_pitcher_name(name, dav_arsenals)
        if key and key not in seen:
            ordered.append((key, "injured"))
            seen.add(key)

    return ordered


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_bats(bats) -> str:
    if bats is None or (isinstance(bats, float) and pd.isna(bats)):
        return "?"
    b = str(bats).strip()
    return {"Right": "R", "Left": "L", "Both": "S", "Switch": "S"}.get(b, b)


def _bin_pitch(row):
    """Assign 3x3 zone bins to a pitch."""
    x = row.get("PlateLocSide", np.nan)
    y = row.get("PlateLocHeight", np.nan)
    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan
    xb = int(np.clip(np.digitize(x, X_EDGES) - 1, 0, 2))
    yb = int(np.clip(np.digitize(y, Y_EDGES) - 1, 0, 2))
    return xb, yb


def _compute_slg_grid(pdf):
    """Compute SLG in each 3x3 zone cell. Returns 3x3 array (nan where no data)."""
    grid = np.full((3, 3), np.nan)
    n_grid = np.zeros((3, 3), dtype=int)
    if pdf.empty:
        return grid, n_grid
    d = pdf.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if d.empty:
        return grid, n_grid
    d["xb"] = np.clip(np.digitize(d["PlateLocSide"], X_EDGES) - 1, 0, 2).astype(int)
    d["yb"] = np.clip(np.digitize(d["PlateLocHeight"], Y_EDGES) - 1, 0, 2).astype(int)
    ab = d[d["PlayResult"].isin(_AB_RESULTS)] if "PlayResult" in d.columns else pd.DataFrame()
    if ab.empty:
        return grid, n_grid
    for yb in range(3):
        for xb in range(3):
            cell = ab[(ab["xb"] == xb) & (ab["yb"] == yb)]
            n_grid[yb, xb] = len(cell)
            if len(cell) >= 3:
                tb = cell["PlayResult"].map(lambda r: _TB_MAP.get(r, 0)).sum()
                grid[yb, xb] = tb / len(cell)
    return grid, n_grid


def _roster_key(tm_name):
    """Convert TrueMedia 'First Last' to roster 'Last, First' for BRYANT_JERSEY/POSITION lookup."""
    return tm_name_to_trackman(tm_name)


def _is_hitter(name):
    """True if player is a position player (not a pure pitcher)."""
    pos = BRYANT_POSITION.get(_roster_key(name), "")
    return pos not in _PITCHER_ONLY


# ── Load pitches ─────────────────────────────────────────────────────────────

def _load_all_pitches():
    """Load and merge hitter + pitcher pitches, normalize, dedup."""
    h = load_bryant_pitches()
    p = load_bryant_pitcher_pitches()
    frames = [df for df in [h, p] if not df.empty]
    if not frames:
        return pd.DataFrame()
    pitches = pd.concat(frames, ignore_index=True)
    if "trackmanPitchUID" in pitches.columns:
        has_uid = pitches["trackmanPitchUID"].notna()
        dedup_cols = ["trackmanPitchUID"]
        if "Batter" in pitches.columns:
            dedup_cols.append("Batter")
        pitches = pd.concat([
            pitches[has_uid].drop_duplicates(subset=dedup_cols, keep="first"),
            pitches[~has_uid],
        ], ignore_index=True)
    pitches = normalize_pitch_types(pitches)
    return pitches


# ── Drawing functions ────────────────────────────────────────────────────────

def _draw_header(ax, hitter, rate_row, cnt_row):
    """Draw header bar with jersey, name, position, bats, PA."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Force the background patch to render
    ax.patch.set_facecolor("#1a1a2e")
    ax.patch.set_alpha(1.0)

    rk = _roster_key(hitter)
    jersey = BRYANT_JERSEY.get(rk, "")
    pos = BRYANT_POSITION.get(rk, "")
    bats = _fmt_bats(rate_row.iloc[0].get("batsHand", "?") if not rate_row.empty else "?")
    pa = int(_safe_num(rate_row, "PA", 0))

    parts = hitter.split(", ")
    display = f"{parts[1]} {parts[0]}" if len(parts) == 2 else hitter

    text = f"#{jersey}  {display.upper()}"
    details = f"|  {pos}  |  Bats: {bats}  |  {pa} PA"

    ax.text(0.02, 0.5, text, fontsize=16, fontweight="bold", color="white",
            va="center", ha="left", transform=ax.transAxes)
    ax.text(0.98, 0.5, details, fontsize=11, color="#cccccc",
            va="center", ha="right", transform=ax.transAxes)


def _draw_summary(ax, summary_text):
    """Draw AI-generated summary text."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#f0f0f5")
    ax.text(0.02, 0.95, summary_text, fontsize=8.5, va="top", ha="left",
            wrap=True, transform=ax.transAxes, style="italic",
            fontfamily="sans-serif", color="#222222")


def _draw_batted_ball(ax, ht_row):
    """Horizontal bar chart for GB%, FB%, LD%, Popup%."""
    ax.set_title("Batted Ball Profile", fontsize=9, fontweight="bold", pad=4)
    cats = ["Ground%", "Fly%", "Line%", "Popup%"]
    labels = ["GB%", "FB%", "LD%", "Pop%"]
    vals = [_safe_num(ht_row, c, 0) for c in cats]
    colors = ["#e74c3c" if v > 40 else "#f39c12" if v > 25 else "#95a5a6" for v in vals]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, vals, color=colors, height=0.6, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 65)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_spray_chart(ax, hl_row):
    """Polar wedge spray chart with 5 directional zones."""
    ax.set_title("Spray Chart", fontsize=9, fontweight="bold", pad=4)
    # Use percentages from hit_locations
    zone_cols = ["HFarLft%", "HLftCtr%", "HDeadCtr%", "HRtCtr%", "HFarRt%"]
    zone_labels = ["Far L", "L-C", "Center", "R-C", "Far R"]
    vals = [_safe_num(hl_row, c, 0) for c in zone_cols]

    # Draw fan diagram — scale up to fill the subplot
    R_MAX = 1.6  # max wedge radius
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.22, 1.3)
    ax.axis("off")

    # Foul lines
    import matplotlib.patches as patches
    foul_l_angle = np.radians(135)
    foul_r_angle = np.radians(45)
    ax.plot([0, 1.15 * np.cos(foul_l_angle)], [0, 1.15 * np.sin(foul_l_angle)],
            "k-", lw=1, alpha=0.4)
    ax.plot([0, 1.15 * np.cos(foul_r_angle)], [0, 1.15 * np.sin(foul_r_angle)],
            "k-", lw=1, alpha=0.4)

    # Outfield arc
    arc_angles = np.linspace(foul_r_angle, foul_l_angle, 50)
    ax.plot(1.1 * np.cos(arc_angles), 1.1 * np.sin(arc_angles), "k-", lw=0.8, alpha=0.3)

    # Wedges: 5 equal-angle zones from 135° (left field) to 45° (right field)
    wedge_edges = np.linspace(135, 45, 6)
    max_val = max(vals) if max(vals) > 0 else 1
    for i, v in enumerate(vals):
        a1 = np.radians(wedge_edges[i])
        a2 = np.radians(wedge_edges[i + 1])
        r = 0.4 + 0.65 * (v / max_val) if max_val > 0 else 0.4
        theta = np.linspace(a1, a2, 20)
        xs = np.concatenate([[0], r * np.cos(theta), [0]])
        ys = np.concatenate([[0], r * np.sin(theta), [0]])
        intensity = v / max_val if max_val > 0 else 0
        color = plt.cm.Reds(0.2 + 0.6 * intensity)
        ax.fill(xs, ys, color=color, alpha=0.7, edgecolor="white", lw=1.5)
        # Label
        mid_a = np.radians((wedge_edges[i] + wedge_edges[i + 1]) / 2)
        lx = (r * 0.55) * np.cos(mid_a)
        ly = (r * 0.55) * np.sin(mid_a)
        if v > 0:
            ax.text(lx, ly, f"{v:.0f}%", fontsize=8, ha="center", va="center",
                    fontweight="bold", color="black")

    # Field position labels
    ax.text(-0.95, 0.95, "LF", fontsize=7, ha="center", color="#777")
    ax.text(0, 1.18, "CF", fontsize=7, ha="center", color="#777")
    ax.text(0.95, 0.95, "RF", fontsize=7, ha="center", color="#777")

    # Direction labels
    pull = _safe_num(hl_row, "HPull%", 0)
    ctr = _safe_num(hl_row, "HCtr%", 0)
    oppo = _safe_num(hl_row, "HOppFld%", 0)
    ax.text(0, -0.18, f"Pull {pull:.0f}%  |  Ctr {ctr:.0f}%  |  Oppo {oppo:.0f}%",
            fontsize=7.5, ha="center", va="top", color="#555")


def _draw_hole_heatmap(ax, holes, bats, title):
    """3x3 heatmap of hole scores (green=safe, red=attackable)."""
    ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
    grid = np.full((3, 3), np.nan)
    for (xb, yb), score in holes.items():
        grid[2 - yb, xb] = score  # flip y so top row = high zone

    ax.imshow(grid, cmap=_HOLE_CMAP, vmin=20, vmax=80, aspect="auto")
    # Draw strike zone border
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor="black", lw=2))

    for yb in range(3):
        for xb in range(3):
            v = grid[yb, xb]
            if not np.isnan(v):
                ax.text(xb, yb, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        fontweight="bold", color="black" if v < 60 else "white")

    # Labels
    b = _fmt_bats(bats)
    if b == "R":
        ax.set_xlabel("← Inside    Away →", fontsize=6, labelpad=2)
    elif b == "L":
        ax.set_xlabel("← Away    Inside →", fontsize=6, labelpad=2)
    else:
        ax.set_xlabel("← Left    Right →", fontsize=6, labelpad=2)
    ax.set_ylabel("Low → High", fontsize=6, labelpad=2)
    ax.set_xticks([])
    ax.set_yticks([])


def _style_zone_grid(ax, bats, cmap, vmin, vmax, cb_label=""):
    """Apply the standard zone-grid style: dashed outer border, bold strike-zone
    rectangle, white gridlines, colorbar, UP/DOWN + AWAY/INSIDE labels."""
    # White gridlines between cells
    for x in [0.5, 1.5]:
        ax.axvline(x, color="white", lw=1.5, zorder=2)
    for y in [0.5, 1.5]:
        ax.axhline(y, color="white", lw=1.5, zorder=2)

    # Dashed outer border (full charted area)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False,
                                edgecolor="#555", lw=1.2, ls="--", zorder=3))

    # Bold strike-zone rectangle (mapped from real zone coordinates)
    # Zone: |x|≤0.83 → imshow x 0.17–1.83;  y 1.5–3.5 → imshow y -0.17–1.83
    ax.add_patch(plt.Rectangle((0.17, -0.17), 1.66, 2.0, fill=False,
                                edgecolor="black", lw=2.5, zorder=4))

    # Y-axis: UP / DOWN
    ax.set_yticks([0, 2])
    ax.set_yticklabels(["UP", "DOWN"], fontsize=6, fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=2)

    # X-axis: AWAY / INSIDE (handedness-aware)
    b = _fmt_bats(bats) if not isinstance(bats, str) else bats
    if b == "R":
        left_lbl, right_lbl = "← INSIDE", "AWAY →"
    elif b == "L":
        left_lbl, right_lbl = "← AWAY", "INSIDE →"
    else:
        left_lbl, right_lbl = "← L", "R →"
    ax.set_xticks([0, 2])
    ax.set_xticklabels([left_lbl, right_lbl], fontsize=5.5, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=15)
    cb.ax.tick_params(labelsize=5)
    if cb_label:
        cb.set_label(cb_label, fontsize=5.5, labelpad=2)


def _draw_slg_heatmap(ax, grid, n_grid, title, bats="?"):
    """3x3 SLG heatmap (blue-white-red) with zone-style rendering."""
    ax.set_title(title, fontsize=7, fontweight="bold", pad=2)
    display = np.where(n_grid >= 3, grid, np.nan)
    display = display[::-1]  # flip y so top row = high zone

    ax.imshow(display, cmap=_SLG_CMAP, vmin=0, vmax=1.0, aspect="auto")
    for yr in range(3):
        for xr in range(3):
            v = display[yr, xr]
            if not np.isnan(v):
                ax.text(xr, yr, f".{int(v * 1000):03d}", ha="center", va="center",
                        fontsize=6.5, fontweight="bold",
                        color="white" if v > 0.7 or v < 0.15 else "black")
    _style_zone_grid(ax, bats, _SLG_CMAP, 0, 1.0, cb_label="SLG")


def _draw_zone_metric(ax, pitches_df, metric, title, cmap, vmin, vmax, fmt_fn, bats="?", cb_label=""):
    """Generic 3x3 zone metric heatmap with zone-style rendering."""
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=3)
    grid = np.full((3, 3), np.nan)

    if not pitches_df.empty:
        d = pitches_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
        if not d.empty:
            d["xb"] = np.clip(np.digitize(d["PlateLocSide"], X_EDGES) - 1, 0, 2).astype(int)
            d["yb"] = np.clip(np.digitize(d["PlateLocHeight"], Y_EDGES) - 1, 0, 2).astype(int)
            for yb in range(3):
                for xb in range(3):
                    cell = d[(d["xb"] == xb) & (d["yb"] == yb)]
                    val = metric(cell)
                    if val is not None:
                        grid[2 - yb, xb] = val

    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for yr in range(3):
        for xr in range(3):
            v = grid[yr, xr]
            if not np.isnan(v):
                txt = fmt_fn(v)
                brightness = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                color = "white" if brightness > 0.7 or brightness < 0.15 else "black"
                ax.text(xr, yr, txt, ha="center", va="center", fontsize=6.5,
                        fontweight="bold", color=color)
    _style_zone_grid(ax, bats, cmap, vmin, vmax, cb_label=cb_label)


def _whiff_count(cell):
    if len(cell) < 5:
        return None
    return len(cell[cell["PitchCall"] == "StrikeSwinging"])


def _swing_pct(cell):
    if len(cell) < 5:
        return None
    swings = cell[cell["PitchCall"].isin(SWING_CALLS)]
    return len(swings) / len(cell) * 100


def _two_strike_whiff_pct(cell):
    """Whiff% on pitches with 2 strikes."""
    if "Strikes" not in cell.columns or "PitchCall" not in cell.columns:
        return None
    strikes = pd.to_numeric(cell["Strikes"], errors="coerce")
    two_k = cell[strikes == 2]
    if len(two_k) < 5:
        return None
    swings = two_k[two_k["PitchCall"].isin(SWING_CALLS)]
    if len(swings) < 3:
        return None
    whiffs = two_k[two_k["PitchCall"] == "StrikeSwinging"]
    return len(whiffs) / len(swings) * 100


def _ev_p90(cell):
    ip = cell[cell["PitchCall"] == "InPlay"] if "PitchCall" in cell.columns else pd.DataFrame()
    if len(ip) < 3 or "ExitSpeed" not in ip.columns:
        return None
    evs = pd.to_numeric(ip["ExitSpeed"], errors="coerce").dropna()
    if len(evs) < 3:
        return None
    return float(evs.quantile(0.9))


# ── AI Summary Generator ────────────────────────────────────────────────────

def _zone_desc(xb, yb, bats):
    """Human-readable zone label relative to batter handedness."""
    v_labels = {0: "down", 1: "middle", 2: "up"}
    if bats == "R":
        h_labels = {0: "inside", 1: "middle", 2: "away"}
    elif bats == "L":
        h_labels = {0: "away", 1: "middle", 2: "inside"}
    else:
        h_labels = {0: "left", 1: "middle", 2: "right"}
    h = h_labels[xb]
    v = v_labels[yb]
    if h == "middle" and v == "middle":
        return "middle-middle"
    if h == "middle":
        return v
    if v == "middle":
        return h
    return f"{v}-{h}"


def _hand_slg(pitches_df, hand_val):
    """Compute SLG against a specific pitcher hand from pitch-level data."""
    if pitches_df.empty or "PitcherThrows" not in pitches_df.columns:
        return np.nan
    sub = pitches_df[pitches_df["PitcherThrows"].isin(
        ["Right", "R"] if hand_val == "R" else ["Left", "L"]
    )]
    ab = sub[sub["PlayResult"].isin(_AB_RESULTS)] if "PlayResult" in sub.columns else pd.DataFrame()
    if len(ab) < 10:
        return np.nan
    tb = ab["PlayResult"].map(lambda r: _TB_MAP.get(r, 0)).sum()
    return tb / len(ab)


def _hand_k_rate(pitches_df, hand_val):
    """Compute K rate against a specific pitcher hand from pitch-level data."""
    if pitches_df.empty or "PitcherThrows" not in pitches_df.columns:
        return np.nan
    sub = pitches_df[pitches_df["PitcherThrows"].isin(
        ["Right", "R"] if hand_val == "R" else ["Left", "L"]
    )]
    ab = sub[sub["PlayResult"].isin(_AB_RESULTS)] if "PlayResult" in sub.columns else pd.DataFrame()
    if len(ab) < 10:
        return np.nan
    ks = len(ab[ab["PlayResult"] == "Strikeout"])
    return ks / len(ab) * 100


def _top_hole(holes, bats):
    """Return (zone_desc, score) for the highest hole score, or (None, None)."""
    if not holes:
        return None, None
    best = max(holes, key=holes.get)
    return _zone_desc(best[0], best[1], bats), holes[best]


def _generate_hitter_summary(name, rate_row, cnt_row, ht_row, hl_row, pr_row,
                              pitches_df, holes_all, holes_rhp, holes_lhp):
    """Rule-based natural language scouting summary (4-6 sentences).

    Incorporates platoon splits, zone vulnerability by pitcher hand,
    batted ball tendencies, and approach metrics for maximum coaching utility.
    """
    ba = _safe_num(rate_row, "AVG", np.nan)
    slg = _safe_num(rate_row, "SLG", np.nan)
    obp = _safe_num(rate_row, "OBP", np.nan)
    k_pct = _safe_num(rate_row, "K%", np.nan)
    bb_pct = _safe_num(rate_row, "BB%", np.nan)
    pa = _safe_num(rate_row, "PA", 0)

    chase = _safe_num(pr_row, "Chase%", np.nan)
    swstrk = _safe_num(pr_row, "SwStrk%", np.nan)
    contact = _safe_num(pr_row, "Contact%", np.nan)

    gb = _safe_num(ht_row, "Ground%", np.nan)
    fb = _safe_num(ht_row, "Fly%", np.nan)
    ld = _safe_num(ht_row, "Line%", np.nan)
    popup = _safe_num(ht_row, "Popup%", np.nan)
    pull = _safe_num(hl_row, "HPull%", np.nan)
    oppo = _safe_num(hl_row, "HOppFld%", np.nan)

    iso = (slg - ba) if pd.notna(slg) and pd.notna(ba) else np.nan
    bats = _fmt_bats(rate_row.iloc[0].get("batsHand", "?") if not rate_row.empty else "?")

    # Platoon SLG/K from pitch data
    slg_v_r = _hand_slg(pitches_df, "R")
    slg_v_l = _hand_slg(pitches_df, "L")
    k_v_r = _hand_k_rate(pitches_df, "R")
    k_v_l = _hand_k_rate(pitches_df, "L")

    sentences = []

    # ── S1: Profile + batted ball ──
    traits = []
    if pd.notna(k_pct):
        if k_pct > 25:
            traits.append("high-strikeout")
        elif k_pct < 15:
            traits.append("contact-oriented")
    if pd.notna(bb_pct) and bb_pct > 12:
        traits.append("patient")
    if pd.notna(iso):
        if iso > 0.200:
            traits.append("plus-power")
        elif iso > 0.140:
            traits.append("moderate power")
        elif iso < 0.080:
            traits.append("limited power")
    profile = ", ".join(traits) if traits else "average-profile"

    stat_bits = []
    if pd.notna(ba):
        stat_bits.append(f".{int(ba * 1000):03d}")
    if pd.notna(obp):
        stat_bits.append(f".{int(obp * 1000):03d} OBP")
    if pd.notna(slg):
        stat_bits.append(f".{int(slg * 1000):03d} SLG")
    stat_str = "/".join(stat_bits[:1]) + (f", {', '.join(stat_bits[1:])}" if len(stat_bits) > 1 else "")

    bb_note = ""
    if pd.notna(k_pct) and pd.notna(bb_pct):
        bb_note = f", {k_pct:.0f}% K / {bb_pct:.0f}% BB"

    s1 = f"{profile.capitalize()} hitter ({stat_str}{bb_note})."

    # Batted ball flavor
    bb_parts = []
    if pd.notna(gb) and gb > 48:
        bb_parts.append(f"heavy ground-ball tendency ({gb:.0f}% GB)")
    elif pd.notna(fb) and fb > 40:
        bb_parts.append(f"fly-ball oriented ({fb:.0f}% FB)")
    if pd.notna(ld) and ld > 22:
        bb_parts.append(f"quality line-drive rate ({ld:.0f}% LD)")
    elif pd.notna(ld) and ld < 12:
        bb_parts.append(f"low line-drive rate ({ld:.0f}% LD)")
    if pd.notna(popup) and popup > 15:
        bb_parts.append(f"pop-up prone ({popup:.0f}%)")
    if pd.notna(pull) and pull > 45:
        bb_parts.append(f"pull-heavy ({pull:.0f}% pull)")
    elif pd.notna(oppo) and oppo > 30:
        bb_parts.append(f"uses the whole field ({oppo:.0f}% oppo)")

    if bb_parts:
        s1 += " " + "; ".join(bb_parts).capitalize() + "."
    sentences.append(s1)

    # ── S2: Platoon splits ──
    platoon_parts = []
    if pd.notna(slg_v_r) and pd.notna(slg_v_l):
        diff = slg_v_r - slg_v_l
        r_str = f".{int(slg_v_r * 1000):03d}"
        l_str = f".{int(slg_v_l * 1000):03d}"
        if abs(diff) > 0.150:
            weaker = "RHP" if slg_v_r < slg_v_l else "LHP"
            stronger = "LHP" if weaker == "RHP" else "RHP"
            platoon_parts.append(
                f"Significant platoon split: {r_str} SLG vs RHP, {l_str} vs LHP — "
                f"much more dangerous vs {stronger}"
            )
        elif abs(diff) > 0.060:
            better = "LHP" if slg_v_l > slg_v_r else "RHP"
            platoon_parts.append(f"Hits {better} slightly better ({r_str} vs R, {l_str} vs L)")
        else:
            platoon_parts.append(f"Balanced platoon splits ({r_str} vs R, {l_str} vs L)")
    elif pd.notna(slg_v_r):
        platoon_parts.append(f".{int(slg_v_r * 1000):03d} SLG vs RHP (limited LHP data)")
    if pd.notna(k_v_r) and pd.notna(k_v_l) and abs(k_v_r - k_v_l) > 8:
        higher = "RHP" if k_v_r > k_v_l else "LHP"
        platoon_parts.append(f"strikes out more vs {higher} ({k_v_r:.0f}% vs R, {k_v_l:.0f}% vs L)")
    if platoon_parts:
        sentences.append(". ".join(platoon_parts) + ".")

    # ── S3: Zone vulnerability by hand ──
    vuln_parts = []
    rhp_desc, rhp_score = _top_hole(holes_rhp, bats)
    lhp_desc, lhp_score = _top_hole(holes_lhp, bats)
    all_desc, all_score = _top_hole(holes_all, bats)

    if rhp_desc and rhp_score and rhp_score > 45:
        vuln_parts.append(f"vs RHP: biggest hole is {rhp_desc} ({rhp_score:.0f})")
    if lhp_desc and lhp_score and lhp_score > 45:
        vuln_parts.append(f"vs LHP: biggest hole is {lhp_desc} ({lhp_score:.0f})")
    if not vuln_parts and all_desc and all_score and all_score > 45:
        vuln_parts.append(f"primary hole is {all_desc} ({all_score:.0f})")

    approach_bits = []
    if pd.notna(chase) and chase > 32:
        approach_bits.append(f"expands the zone ({chase:.0f}% chase)")
    elif pd.notna(chase) and chase < 22:
        approach_bits.append(f"selective eye ({chase:.0f}% chase)")
    if pd.notna(swstrk) and swstrk > 12:
        approach_bits.append(f"whiff-prone in zone ({swstrk:.0f}% SwStrk)")

    if vuln_parts or approach_bits:
        sentences.append((" — ".join(vuln_parts) + (". " if vuln_parts else "") +
                          "; ".join(approach_bits)).strip().capitalize() + ".")

    # ── S4: Gameplan vs RHP ──
    gp_r = []
    if rhp_desc and rhp_score and rhp_score > 50:
        loc = rhp_desc
        if "away" in loc and "down" in loc:
            gp_r.append("attack down-and-away with breaking balls")
        elif "away" in loc:
            gp_r.append("work the outer third")
        elif "up" in loc and "inside" in loc:
            gp_r.append("elevated fastball inside is the primary weapon")
        elif "up" in loc:
            gp_r.append("go up in the zone")
        elif "down" in loc:
            gp_r.append("work down in the zone")
        elif "inside" in loc:
            gp_r.append("attack inside")
        else:
            gp_r.append(f"target the {loc} zone")
    if pd.notna(chase) and chase > 32:
        gp_r.append("use secondary early to steal strikes and expand off the plate")
    if pd.notna(pull) and pull > 50:
        gp_r.append("pitch away to neutralize pull power")
    if pd.notna(gb) and gb > 48:
        gp_r.append("stay elevated to avoid ground-ball damage")
    if gp_r:
        sentences.append("vs RHP: " + ", ".join(gp_r) + ".")

    # ── S5: Gameplan vs LHP ──
    gp_l = []
    if lhp_desc and lhp_score and lhp_score > 50:
        loc = lhp_desc
        if "away" in loc and "down" in loc:
            gp_l.append("attack down-and-away with offspeed")
        elif "away" in loc:
            gp_l.append("work the outer third")
        elif "inside" in loc:
            gp_l.append("go after him inside")
        elif "up" in loc:
            gp_l.append("elevate in the zone")
        elif "down" in loc:
            gp_l.append("stay down")
        else:
            gp_l.append(f"target {loc}")
    if pd.notna(slg_v_l) and slg_v_l < 0.250:
        gp_l.append("limited damage vs LHP — be aggressive in the zone")
    elif pd.notna(slg_v_l) and slg_v_l > 0.500:
        gp_l.append("dangerous vs LHP — pitch carefully")
    if gp_l:
        sentences.append("vs LHP: " + ", ".join(gp_l) + ".")

    # ── Fallback if we have very little ──
    if len(sentences) < 2:
        if pd.notna(k_pct) and k_pct < 15:
            sentences.append("Pitch to contact, rely on defense, avoid free bases.")
        else:
            sentences.append("Mix locations and pitch types to keep him off balance.")

    return " ".join(sentences)


# ── Main page renderer ──────────────────────────────────────────────────────

def _render_hitter_page(hitter, pack, pitches):
    """Build one-page matplotlib figure for a hitter. Returns Figure or None."""
    h_rate = _tm_player(_tm_team(pack["hitting"]["rate"], _TEAM), hitter)
    h_cnt = _tm_player(_tm_team(pack["hitting"].get("counting", pd.DataFrame()), _TEAM), hitter)
    h_ht = _tm_player(_tm_team(pack["hitting"].get("hit_types", pd.DataFrame()), _TEAM), hitter)
    h_hl = _tm_player(_tm_team(pack["hitting"].get("hit_locations", pd.DataFrame()), _TEAM), hitter)
    h_pr = _tm_player(_tm_team(pack["hitting"].get("pitch_rates", pd.DataFrame()), _TEAM), hitter)

    pa = _safe_num(h_rate, "PA", 0)
    if pa < 5:
        return None

    bats_raw = h_rate.iloc[0].get("batsHand", "?") if not h_rate.empty else "?"
    bats = _fmt_bats(bats_raw)

    # Pitch-level data for this hitter
    hitter_pitches = pd.DataFrame()
    if not pitches.empty and "Batter" in pitches.columns:
        hitter_pitches = pitches[pitches["Batter"] == hitter].copy()

    # Hole scores
    holes_all = compute_hole_scores_3x3(hitter_pitches, bats) if not hitter_pitches.empty else {}
    holes_rhp = {}
    holes_lhp = {}
    if not hitter_pitches.empty and "PitcherThrows" in hitter_pitches.columns:
        rhp_df = hitter_pitches[hitter_pitches["PitcherThrows"].isin(["Right", "R"])]
        lhp_df = hitter_pitches[hitter_pitches["PitcherThrows"].isin(["Left", "L"])]
        if len(rhp_df) >= 30:
            holes_rhp = compute_hole_scores_3x3(rhp_df, bats)
        if len(lhp_df) >= 30:
            holes_lhp = compute_hole_scores_3x3(lhp_df, bats)

    # AI Summary
    summary = _generate_hitter_summary(hitter, h_rate, h_cnt, h_ht, h_hl, h_pr,
                                        hitter_pitches, holes_all, holes_rhp, holes_lhp)

    # ── Figure layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    gs = gridspec.GridSpec(
        7, 1, figure=fig,
        height_ratios=[0.4, 0.75, 1.7, 1.6, 1.6, 1.6, 1.6],
        hspace=0.22, top=0.97, bottom=0.02, left=0.04, right=0.96,
    )

    # Row 0: Header
    ax_header = fig.add_subplot(gs[0])
    _draw_header(ax_header, hitter, h_rate, h_cnt)

    # Row 1: Summary
    ax_summary = fig.add_subplot(gs[1])
    _draw_summary(ax_summary, summary)

    # Row 2: Batted Ball + Spray Chart
    gs_row2 = gs[2].subgridspec(1, 2, wspace=0.3)
    ax_bb = fig.add_subplot(gs_row2[0, 0])
    _draw_batted_ball(ax_bb, h_ht)
    ax_spray = fig.add_subplot(gs_row2[0, 1])
    _draw_spray_chart(ax_spray, h_hl)

    # Row 3: Hole Score vs RHP + vs LHP
    gs_row3 = gs[3].subgridspec(1, 2, wspace=0.4)
    ax_hole_r = fig.add_subplot(gs_row3[0, 0])
    if holes_rhp:
        _draw_hole_heatmap(ax_hole_r, holes_rhp, bats, "Hole Score vs RHP")
    elif holes_all:
        _draw_hole_heatmap(ax_hole_r, holes_all, bats, "Hole Score (All)")
    else:
        ax_hole_r.set_title("Hole Score vs RHP", fontsize=8, fontweight="bold")
        ax_hole_r.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                       fontsize=9, color="#999", transform=ax_hole_r.transAxes)
        ax_hole_r.axis("off")

    ax_hole_l = fig.add_subplot(gs_row3[0, 1])
    if holes_lhp:
        _draw_hole_heatmap(ax_hole_l, holes_lhp, bats, "Hole Score vs LHP")
    else:
        ax_hole_l.set_title("Hole Score vs LHP", fontsize=8, fontweight="bold")
        ax_hole_l.text(0.5, 0.5, "n < 30 vs LHP" if not holes_lhp else "Insufficient data",
                       ha="center", va="center", fontsize=9, color="#999",
                       transform=ax_hole_l.transAxes)
        ax_hole_l.axis("off")

    # Row 4: SLG by pitch type (top 4 overall)
    gs_row4 = gs[4].subgridspec(1, 4, wspace=0.55)
    _draw_slg_by_pitch_type_row(fig, gs_row4, hitter_pitches, "SLG by Pitch Type", bats=bats)

    # Row 5: SLG by pitch type per hand (top 3 vs RHP, spacer, top 3 vs LHP)
    gs_row5 = gs[5].subgridspec(1, 7, wspace=0.3,
                                 width_ratios=[1, 1, 1, 0.15, 1, 1, 1])
    _draw_slg_hand_splits(fig, gs_row5, hitter_pitches, bats=bats)

    # Row 6: Zone Summary (Whiff Density, Swing%, 2-Strike Whiff%)
    gs_row6 = gs[6].subgridspec(1, 3, wspace=0.55)
    ax_whiff = fig.add_subplot(gs_row6[0, 0])
    _draw_zone_metric(ax_whiff, hitter_pitches, _whiff_count,
                      "Whiff Density", _SWING_CMAP, 0, 15, lambda v: f"{int(v)}",
                      bats=bats, cb_label="Whiff Cnt")

    ax_swing = fig.add_subplot(gs_row6[0, 1])
    _draw_zone_metric(ax_swing, hitter_pitches, _swing_pct,
                      "Swing% by Zone", _SWING_CMAP, 0, 100, lambda v: f"{v:.0f}%",
                      bats=bats, cb_label="Swing%")

    ax_2k = fig.add_subplot(gs_row6[0, 2])
    _draw_zone_metric(ax_2k, hitter_pitches, _two_strike_whiff_pct,
                      "2-Strike Whiff%", _SWING_CMAP, 0, 60, lambda v: f"{v:.0f}%",
                      bats=bats, cb_label="Whiff%")

    return fig


def _draw_slg_by_pitch_type_row(fig, gs_row, hitter_pitches, row_label, bats="?"):
    """Draw up to 4 SLG heatmaps for top pitch types by count."""
    if hitter_pitches.empty or "TaggedPitchType" not in hitter_pitches.columns:
        for i in range(4):
            ax = fig.add_subplot(gs_row[0, i])
            ax.axis("off")
            if i == 0:
                ax.text(0.5, 0.5, "No pitch data", ha="center", va="center",
                        fontsize=9, color="#999", transform=ax.transAxes)
        return

    counts = hitter_pitches["TaggedPitchType"].dropna().value_counts()
    top_types = counts[counts >= 15].head(4).index.tolist()

    for i in range(4):
        ax = fig.add_subplot(gs_row[0, i])
        if i < len(top_types):
            pt = top_types[i]
            pt_df = hitter_pitches[hitter_pitches["TaggedPitchType"] == pt]
            grid, n_grid = _compute_slg_grid(pt_df)
            _draw_slg_heatmap(ax, grid, n_grid, f"{pt} (n={len(pt_df)})", bats=bats)
        else:
            ax.axis("off")


def _draw_slg_hand_splits(fig, gs_row, hitter_pitches, bats="?"):
    """Draw SLG grids split by pitcher hand.

    Expects gs_row to be a 1x7 subgridspec (cols 0-2 = vs R, 3 = spacer, 4-6 = vs L).
    """
    if hitter_pitches.empty or "PitcherThrows" not in hitter_pitches.columns:
        for i in range(7):
            ax = fig.add_subplot(gs_row[0, i])
            ax.axis("off")
        return

    rhp = hitter_pitches[hitter_pitches["PitcherThrows"].isin(["Right", "R"])]
    lhp = hitter_pitches[hitter_pitches["PitcherThrows"].isin(["Left", "L"])]

    # Spacer column
    ax_spacer = fig.add_subplot(gs_row[0, 3])
    ax_spacer.axis("off")

    # vs RHP — columns 0-2
    if len(rhp) >= 25 and "TaggedPitchType" in rhp.columns:
        rhp_counts = rhp["TaggedPitchType"].dropna().value_counts()
        rhp_tops = rhp_counts[rhp_counts >= 10].head(3).index.tolist()
        for j in range(3):
            ax = fig.add_subplot(gs_row[0, j])
            if j < len(rhp_tops):
                pt = rhp_tops[j]
                pt_df = rhp[rhp["TaggedPitchType"] == pt]
                grid, n_grid = _compute_slg_grid(pt_df)
                _draw_slg_heatmap(ax, grid, n_grid, f"vs R: {pt}", bats=bats)
            else:
                ax.axis("off")
    else:
        for j in range(3):
            ax = fig.add_subplot(gs_row[0, j])
            ax.axis("off")
            if j == 1:
                ax.text(0.5, 0.5, "vs RHP: n < 25", ha="center", va="center",
                        fontsize=8, color="#999", transform=ax.transAxes)

    # vs LHP — columns 4-6
    if len(lhp) >= 25 and "TaggedPitchType" in lhp.columns:
        lhp_counts = lhp["TaggedPitchType"].dropna().value_counts()
        lhp_tops = lhp_counts[lhp_counts >= 10].head(3).index.tolist()
        for j in range(3):
            ax = fig.add_subplot(gs_row[0, 4 + j])
            if j < len(lhp_tops):
                pt = lhp_tops[j]
                pt_df = lhp[lhp["TaggedPitchType"] == pt]
                grid, n_grid = _compute_slg_grid(pt_df)
                _draw_slg_heatmap(ax, grid, n_grid, f"vs L: {pt}", bats=bats)
            else:
                ax.axis("off")
    else:
        for j in range(3):
            ax = fig.add_subplot(gs_row[0, 4 + j])
            ax.axis("off")
            if j == 1:
                ax.text(0.5, 0.5, "vs LHP: n < 25", ha="center", va="center",
                        fontsize=8, color="#999", transform=ax.transAxes)


# ══════════════════════════════════════════════════════════════════════════════
# ██  PITCHER  PDF  FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

from config import STUFF_WEIGHTS, STUFF_WEIGHTS_DEFAULT, PITCH_COLORS, TM_PITCH_PCT_COLS, filter_minor_pitches, PITCH_TYPES_TO_DROP
from decision_engine.data.baserunning_data import (
    load_stolen_bases_catchers, load_stolen_bases_pitchers, load_pickoffs,
    get_pitcher_sb_row_by_name, get_pickoffs_row_by_name,
)

_LOC_CMAP = LinearSegmentedColormap.from_list("loc", ["#e8f0fe", "#6fa8dc", "#1a3d6f"])
_MIN_ARSENAL_USAGE = 5.0  # minimum % to include in arsenal table


def _style_zone_compact(ax, bats=None):
    """Compact zone-grid style for pitcher heatmaps: no colorbar, minimal labels."""
    for x in [0.5, 1.5]:
        ax.axvline(x, color="white", lw=1, zorder=2)
    for y in [0.5, 1.5]:
        ax.axhline(y, color="white", lw=1, zorder=2)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False,
                                edgecolor="#888", lw=0.8, ls="--", zorder=3))
    ax.add_patch(plt.Rectangle((0.17, -0.17), 1.66, 2.0, fill=False,
                                edgecolor="black", lw=2, zorder=4))
    ax.set_yticks([])
    b = _fmt_bats(bats) if bats and not isinstance(bats, str) else (bats or "?")
    if b == "R":
        left_lbl, right_lbl = "← IN", "AWAY →"
    elif b == "L":
        left_lbl, right_lbl = "← AWAY", "IN →"
    else:
        left_lbl, right_lbl = "", ""
    if left_lbl:
        ax.set_xticks([0, 2])
        ax.set_xticklabels([left_lbl, right_lbl], fontsize=5, color="#666")
        ax.tick_params(axis="x", length=0, pad=1)
    else:
        ax.set_xticks([])


def _entropy_bits(probs):
    """Shannon entropy in bits."""
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return np.nan
    return float(-(probs * np.log2(probs)).sum())


def _is_pitcher_player(name):
    """True if player has a pitcher position (RHP or LHP anywhere in position string)."""
    pos = BRYANT_POSITION.get(_roster_key(name), "")
    return "RHP" in pos or "LHP" in pos


def _pitcher_throws(name):
    """Return 'R' or 'L' based on position."""
    pos = BRYANT_POSITION.get(_roster_key(name), "")
    if "LHP" in pos:
        return "L"
    return "R"


def _load_pitcher_pitches():
    """Load pitcher-side pitch-level data, normalize pitch types."""
    pp = load_bryant_pitcher_pitches()
    if pp.empty:
        return pp
    pp = normalize_pitch_types(pp)
    return pp


# ── Arsenal & Stuff+ ────────────────────────────────────────────────────────

def _compute_team_stuff_baselines(all_pp):
    """Compute mean/std for each metric per pitch type across all team pitchers.
    Returns dict of {pitch_type: {metric: (mean, std)}}."""
    if all_pp.empty or "TaggedPitchType" not in all_pp.columns:
        return {}
    baselines = {}
    for pt, grp in all_pp.groupby("TaggedPitchType"):
        stats = {}
        for col in ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension", "VertApprAngle"]:
            if col in grp.columns:
                vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                if len(vals) >= 10:
                    stats[col] = (float(vals.mean()), float(vals.std()))
        baselines[pt] = stats
    return baselines


def _compute_pitcher_stuff_plus(pitcher_pp, baselines):
    """Compute average Stuff+ per pitch type for a single pitcher.
    Returns dict of {pitch_type: stuff_plus_score}."""
    if pitcher_pp.empty:
        return {}
    result = {}
    for pt, grp in pitcher_pp.groupby("TaggedPitchType"):
        if pt not in baselines or len(grp) < 5:
            continue
        w = STUFF_WEIGHTS.get(pt, STUFF_WEIGHTS_DEFAULT)
        bstats = baselines[pt]
        z_scores = []
        w_total = 0.0
        for col, weight in w.items():
            if col == "VeloDiff":
                continue  # skip for simplicity in team-relative
            if col == "HorzBreak":
                actual_col = "HorzBreak"
            else:
                actual_col = col
            if actual_col not in grp.columns or col not in bstats:
                continue
            mu, sigma = bstats[col]
            if sigma == 0 or pd.isna(sigma):
                continue
            vals = pd.to_numeric(grp[actual_col], errors="coerce").dropna()
            if len(vals) < 3:
                continue
            z = (vals.mean() - mu) / sigma
            z_scores.append(z * weight)
            w_total += abs(weight)
        if w_total > 0:
            avg_z = sum(z_scores) / w_total
            result[pt] = 100 + avg_z * 10
        else:
            result[pt] = np.nan
    return result


def _build_arsenal_data(pitcher_pp, pitch_types_row, all_pp):
    """Build arsenal table data from pitch-level and pack data.
    Returns list of dicts sorted by usage desc."""
    arsenal = []
    baselines = _compute_team_stuff_baselines(all_pp)
    stuff_scores = _compute_pitcher_stuff_plus(pitcher_pp, baselines)

    if not pitcher_pp.empty and "TaggedPitchType" in pitcher_pp.columns:
        total = len(pitcher_pp[pitcher_pp["TaggedPitchType"].notna()])
        counts = pitcher_pp["TaggedPitchType"].value_counts()
        for pt, cnt in counts.items():
            usage_pct = cnt / total * 100 if total > 0 else 0
            if cnt < 5 or not pt or usage_pct < _MIN_ARSENAL_USAGE:
                continue
            grp = pitcher_pp[pitcher_pp["TaggedPitchType"] == pt]
            row = {"pitch": pt, "n": cnt, "usage": cnt / total * 100 if total > 0 else 0}
            for col, key in [("RelSpeed", "vel"), ("InducedVertBreak", "ivb"),
                             ("HorzBreak", "hb"), ("SpinRate", "spin"), ("Extension", "ext")]:
                vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                row[key] = float(vals.mean()) if len(vals) >= 3 else np.nan
            row["stuff"] = stuff_scores.get(pt, np.nan)
            arsenal.append(row)
    elif not pitch_types_row.empty:
        # Fallback: use pack pitch_types percentages (no pitch-level data)
        for pct_col, pt_name in TM_PITCH_PCT_COLS.items():
            usage = _safe_num(pitch_types_row, pct_col, 0)
            if usage >= _MIN_ARSENAL_USAGE:
                arsenal.append({"pitch": pt_name, "n": 0, "usage": usage,
                                "vel": np.nan, "ivb": np.nan, "hb": np.nan,
                                "spin": np.nan, "ext": np.nan, "stuff": np.nan})

    arsenal.sort(key=lambda r: r["usage"], reverse=True)
    return arsenal


def _draw_arsenal_table(ax, arsenal):
    """Draw arsenal table as a formatted matplotlib table."""
    ax.axis("off")
    ax.set_title("Arsenal & Stuff", fontsize=10, fontweight="bold", pad=6, loc="left")

    if not arsenal:
        ax.text(0.5, 0.5, "No arsenal data available", ha="center", va="center",
                fontsize=9, color="#999", transform=ax.transAxes)
        return

    cols = ["Pitch", "Usage", "Vel", "IVB", "HB", "Spin", "Ext", "Stuff+"]
    rows = []
    row_colors = []
    for r in arsenal[:7]:  # max 7 pitch types
        color = PITCH_COLORS.get(r["pitch"], "#aaaaaa")
        row_colors.append(color)
        vel_s = f"{r['vel']:.1f}" if pd.notna(r['vel']) else "—"
        ivb_s = f'{r["ivb"]:.1f}"' if pd.notna(r["ivb"]) else "—"
        hb_s = f'{r["hb"]:.1f}"' if pd.notna(r["hb"]) else "—"
        spin_s = f"{r['spin']:.0f}" if pd.notna(r["spin"]) else "—"
        ext_s = f"{r['ext']:.1f}'" if pd.notna(r["ext"]) else "—"
        stuff_s = f"{r['stuff']:.0f}" if pd.notna(r["stuff"]) else "—"
        rows.append([r["pitch"], f"{r['usage']:.0f}%", vel_s, ivb_s, hb_s, spin_s, ext_s, stuff_s])

    table = ax.table(cellText=rows, colLabels=cols, loc="center",
                     cellLoc="center", colColours=["#e8e8e8"] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", fontsize=7.5)
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)

    # Style data rows — color indicator on pitch name cell
    for i, (row_data, color) in enumerate(zip(rows, row_colors)):
        for j in range(len(cols)):
            cell = table[i + 1, j]
            if j == 0:
                cell.set_facecolor(color + "30")  # light tint
                cell.set_text_props(fontweight="bold", color=color)
            else:
                cell.set_facecolor("#ffffff" if i % 2 == 0 else "#f5f5f5")


# ── Pitch Location Heatmaps ─────────────────────────────────────────────────

def _compute_location_grid(pitches_sub):
    """Compute pitch frequency in each 3x3 zone cell. Returns (3x3 pct array, 3x3 count)."""
    grid = np.zeros((3, 3))
    if pitches_sub.empty:
        return grid, grid.astype(int)
    d = pitches_sub.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    if d.empty:
        return grid, grid.astype(int)
    d["xb"] = np.clip(np.digitize(d["PlateLocSide"], X_EDGES) - 1, 0, 2).astype(int)
    d["yb"] = np.clip(np.digitize(d["PlateLocHeight"], Y_EDGES) - 1, 0, 2).astype(int)
    total = len(d)
    n_grid = np.zeros((3, 3), dtype=int)
    for yb in range(3):
        for xb in range(3):
            cnt = len(d[(d["xb"] == xb) & (d["yb"] == yb)])
            n_grid[yb, xb] = cnt
            grid[yb, xb] = cnt / total * 100 if total > 0 else 0
    return grid, n_grid


def _draw_location_heatmap(ax, pitcher_pp_pt, title, throws="R"):
    """3x3 pitch location heatmap for a single pitch type (compact blue style)."""
    ax.set_title(title, fontsize=7, fontweight="bold", pad=2)
    grid_pct, n_grid = _compute_location_grid(pitcher_pp_pt)
    display = grid_pct[::-1]  # flip y so top = high

    ax.imshow(display, cmap=_LOC_CMAP, vmin=0, vmax=25, aspect="auto")
    for yr in range(3):
        for xr in range(3):
            v = display[yr, xr]
            if v > 0.5:
                ax.text(xr, yr, f"{v:.0f}%", ha="center", va="center",
                        fontsize=6.5, fontweight="bold",
                        color="white" if v > 15 else "#1a3d6f")

    bats_equiv = "L" if throws == "R" else "R"
    _style_zone_compact(ax, bats_equiv)


def _draw_pitcher_locations_row(fig, gs_slot, pitcher_pp, throws="R"):
    """Draw up to 4 pitch location heatmaps for top pitch types, centered."""
    if pitcher_pp.empty or "TaggedPitchType" not in pitcher_pp.columns:
        gs_row = gs_slot.subgridspec(1, 1)
        ax = fig.add_subplot(gs_row[0, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, "No pitch-level data", ha="center", va="center",
                fontsize=9, color="#999", transform=ax.transAxes)
        return

    counts = pitcher_pp["TaggedPitchType"].dropna().value_counts()
    top_types = counts[counts >= 10].head(4).index.tolist()
    n = len(top_types)

    if n == 0:
        gs_row = gs_slot.subgridspec(1, 1)
        ax = fig.add_subplot(gs_row[0, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, "No pitch types with n >= 10", ha="center", va="center",
                fontsize=9, color="#999", transform=ax.transAxes)
        return

    if n >= 4:
        gs_row = gs_slot.subgridspec(1, 4, wspace=0.3)
        col_indices = list(range(4))
    else:
        # Center n heatmaps with padding on each side
        pad = (4 - n) / 2.0
        ratios = [pad] + [1] * n + [pad]
        gs_row = gs_slot.subgridspec(1, n + 2, wspace=0.3, width_ratios=ratios)
        col_indices = list(range(1, n + 1))
        ax_l = fig.add_subplot(gs_row[0, 0]); ax_l.axis("off")
        ax_r = fig.add_subplot(gs_row[0, -1]); ax_r.axis("off")

    for i, col in enumerate(col_indices):
        ax = fig.add_subplot(gs_row[0, col])
        if i < n:
            pt = top_types[i]
            pt_df = pitcher_pp[pitcher_pp["TaggedPitchType"] == pt]
            _draw_location_heatmap(ax, pt_df, f"{pt} (n={len(pt_df)})", throws=throws)
        else:
            ax.axis("off")


# ── Count Predictability ─────────────────────────────────────────────────────

def _compute_count_predictability_by_hand(pitcher_pp, min_pitches=20):
    """Find the most predictable count/hand combos using entropy-based scoring.
    Returns list of dicts sorted by predictability (highest first), matching
    the computation in the scouting page."""
    req = {"TaggedPitchType", "Balls", "Strikes", "BatterSide"}
    if pitcher_pp.empty or not req.issubset(pitcher_pp.columns):
        return []

    df = pitcher_pp.dropna(subset=["TaggedPitchType", "Balls", "Strikes", "BatterSide"]).copy()
    if df.empty:
        return []
    df = normalize_pitch_types(df)
    bad_labels = {"UN", "UNK", "UNKNOWN", "UNDEFINED", "OTHER", "NONE", "NULL", "NAN", ""}
    bad_labels |= {str(v).strip().upper() for v in PITCH_TYPES_TO_DROP}
    pt_upper = df["TaggedPitchType"].astype(str).str.strip().str.upper()
    df = df[~pt_upper.isin(bad_labels)]
    if df.empty:
        return []

    balls_num = pd.to_numeric(df["Balls"], errors="coerce")
    strikes_num = pd.to_numeric(df["Strikes"], errors="coerce")
    side_norm = df["BatterSide"].astype(str).str.strip().str.upper().str[0]
    valid = balls_num.notna() & strikes_num.notna() & side_norm.isin(["L", "R"])
    df = df[valid].copy()
    if df.empty:
        return []
    balls_num = balls_num[valid].astype(int)
    strikes_num = strikes_num[valid].astype(int)
    df["count"] = balls_num.astype(str) + "-" + strikes_num.astype(str)
    df["BatterSideNorm"] = side_norm[valid]

    results = []
    for (count, side), grp in df.groupby(["count", "BatterSideNorm"]):
        n = len(grp)
        if n < min_pitches:
            continue
        vc = grp["TaggedPitchType"].value_counts()
        top_pitch = vc.index[0]
        top_pct = vc.iloc[0] / n * 100
        top_n = int(vc.iloc[0])
        second_pitch = vc.index[1] if len(vc) > 1 else "—"
        second_pct = vc.iloc[1] / n * 100 if len(vc) > 1 else 0

        probs = (vc / n).values
        k = len(vc)
        h = _entropy_bits(probs)
        h_norm = h / np.log2(k) if k > 1 and pd.notna(h) else 0.0
        predict = 1 - h_norm if pd.notna(h_norm) else np.nan

        label_side = "vs LHH" if side == "L" else "vs RHH"
        results.append({
            "label": f"{count} {label_side}",
            "count": count,
            "hand": label_side,
            "side": side,
            "n": n,
            "top_pitch": top_pitch,
            "top_pct": top_pct,
            "top_n": top_n,
            "second_pitch": second_pitch,
            "second_pct": second_pct,
            "predictability": predict,
        })

    results.sort(key=lambda r: (r.get("predictability", 0), r["n"]), reverse=True)
    return results


def _draw_count_predictability(fig, gs_slot, pitcher_pp):
    """Draw top 3 most predictable count/hand combos with location heatmaps."""
    combos = _compute_count_predictability_by_hand(pitcher_pp)
    if not combos:
        gs_inner = gs_slot.subgridspec(1, 3, wspace=0.25)
        ax = fig.add_subplot(gs_inner[0, 0])
        ax.axis("off")
        ax.set_title("Most Predictable Counts", fontsize=9, fontweight="bold", pad=4)
        ax.text(0.5, 0.5, "No count data", ha="center", va="center",
                fontsize=9, color="#999", transform=ax.transAxes)
        for i in range(1, 3):
            ax2 = fig.add_subplot(gs_inner[0, i])
            ax2.axis("off")
        return

    top3 = combos[:3]
    n_combos = len(top3)

    if n_combos >= 3:
        gs_inner = gs_slot.subgridspec(2, 3, wspace=0.25, hspace=0.05,
                                        height_ratios=[0.4, 1.0])
        col_indices = list(range(3))
    else:
        # Center n_combos items with padding on each side
        pad = (3 - n_combos) / 2.0
        ratios = [pad] + [1] * n_combos + [pad]
        gs_inner = gs_slot.subgridspec(2, n_combos + 2, wspace=0.25, hspace=0.05,
                                        height_ratios=[0.4, 1.0], width_ratios=ratios)
        col_indices = list(range(1, n_combos + 1))
        for row in range(2):
            ax_pad_l = fig.add_subplot(gs_inner[row, 0]); ax_pad_l.axis("off")
            ax_pad_r = fig.add_subplot(gs_inner[row, -1]); ax_pad_r.axis("off")

    for i, col in enumerate(col_indices):
        ax_text = fig.add_subplot(gs_inner[0, col])
        ax_text.axis("off")
        ax_hm = fig.add_subplot(gs_inner[1, col])

        r = top3[i]
        color = PITCH_COLORS.get(r["top_pitch"], "#333")

        if i == 0:
            ax_text.set_title("Most Predictable Counts", fontsize=9, fontweight="bold",
                              pad=3, loc="left")

        # Compact: "0-0 vs RHH — Fastball 52% (36/69)"
        ax_text.text(0.5, 0.85, r["label"], fontsize=8, fontweight="bold",
                     ha="center", va="top", transform=ax_text.transAxes)
        ax_text.text(0.5, 0.42, f"{r['top_pitch']} {r['top_pct']:.0f}%",
                     fontsize=12, fontweight="bold",
                     ha="center", va="top", transform=ax_text.transAxes, color=color)
        ax_text.text(0.5, 0.05, f"{r['top_n']}/{r['n']}",
                     fontsize=6.5, ha="center", va="top", transform=ax_text.transAxes, color="#888")

        # Location heatmap
        balls_num = pd.to_numeric(pitcher_pp["Balls"], errors="coerce")
        strikes_num = pd.to_numeric(pitcher_pp["Strikes"], errors="coerce")
        cnt_parts = r["count"].split("-")
        b_val, s_val = int(cnt_parts[0]), int(cnt_parts[1])
        side_val = r["side"]
        side_mask = pitcher_pp["BatterSide"].astype(str).str.strip().str.upper().str[0] == side_val
        count_mask = (balls_num == b_val) & (strikes_num == s_val) & side_mask
        count_pitches = pitcher_pp[count_mask].copy()
        count_pitches = normalize_pitch_types(count_pitches)
        top_pt_pitches = count_pitches[count_pitches["TaggedPitchType"] == r["top_pitch"]]

        if len(top_pt_pitches) >= 8:
            grid_pct, _ = _compute_location_grid(top_pt_pitches)
            display = grid_pct[::-1]
            ax_hm.imshow(display, cmap=_LOC_CMAP, vmin=0, vmax=25, aspect="auto")
            for yr in range(3):
                for xr in range(3):
                    v = display[yr, xr]
                    if v > 0.5:
                        ax_hm.text(xr, yr, f"{v:.0f}%", ha="center", va="center",
                                   fontsize=6, fontweight="bold",
                                   color="white" if v > 15 else "#1a3d6f")
            _style_zone_compact(ax_hm, side_val)
        else:
            ax_hm.axis("off")
            ax_hm.text(0.5, 0.5, "n < 8", ha="center", va="center",
                       fontsize=7, color="#999", transform=ax_hm.transAxes)


# ── 2-Strike Tendencies ─────────────────────────────────────────────────────

def _two_strike_hand_summary(two_k_sub):
    """Compute 2-strike stats for a hand-filtered subset."""
    if len(two_k_sub) < 8:
        return None
    pt_counts = two_k_sub["TaggedPitchType"].dropna().value_counts()
    total_pt = pt_counts.sum()
    mix = [(pt, cnt / total_pt * 100) for pt, cnt in pt_counts.items() if cnt / total_pt * 100 >= 1]

    has_loc = two_k_sub.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    zone_pct = np.nan
    chase_rate = np.nan
    if len(has_loc) > 0:
        in_zone = has_loc[
            (has_loc["PlateLocSide"].abs() <= ZONE_SIDE) &
            (has_loc["PlateLocHeight"].between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP))
        ]
        zone_pct = len(in_zone) / len(has_loc) * 100
        out_zone = has_loc[~(
            (has_loc["PlateLocSide"].abs() <= ZONE_SIDE) &
            (has_loc["PlateLocHeight"].between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP))
        )]
        oz_swings = out_zone[out_zone["PitchCall"].isin(SWING_CALLS)]
        chase_rate = len(oz_swings) / len(out_zone) * 100 if len(out_zone) > 0 else np.nan

    swings = two_k_sub[two_k_sub["PitchCall"].isin(SWING_CALLS)]
    whiffs = two_k_sub[two_k_sub["PitchCall"] == "StrikeSwinging"]
    whiff_rate = len(whiffs) / len(swings) * 100 if len(swings) > 0 else np.nan

    ks_by_pt = {}
    for pt_name in pt_counts.index[:5]:
        pt_sub = two_k_sub[two_k_sub["TaggedPitchType"] == pt_name]
        ks_by_pt[pt_name] = len(pt_sub[pt_sub["PitchCall"] == "StrikeSwinging"]) + \
                             len(pt_sub[pt_sub["PitchCall"] == "StrikeCalled"])
    putaway = max(ks_by_pt, key=ks_by_pt.get) if ks_by_pt else "—"

    return {"n": len(two_k_sub), "mix": mix, "zone_pct": zone_pct,
            "whiff": whiff_rate, "chase": chase_rate, "putaway": putaway}


def _draw_two_strike_tendencies(fig, gs_slot, pitcher_pp):
    """Draw 2-strike tendencies split by batter hand, with putaway pitch heatmaps."""
    if pitcher_pp.empty or "Strikes" not in pitcher_pp.columns:
        gs_inner = gs_slot.subgridspec(1, 2, wspace=0.25)
        ax = fig.add_subplot(gs_inner[0, 0])
        ax.axis("off")
        ax.set_title("2-Strike Tendencies", fontsize=9, fontweight="bold", pad=4)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=9, color="#999", transform=ax.transAxes)
        ax2 = fig.add_subplot(gs_inner[0, 1])
        ax2.axis("off")
        return

    strikes = pd.to_numeric(pitcher_pp["Strikes"], errors="coerce")
    two_k = pitcher_pp[strikes == 2]

    # Layout: [text_L, heatmap_L, spacer, text_R, heatmap_R]
    gs_inner = gs_slot.subgridspec(1, 5, wspace=0.08,
                                    width_ratios=[1.0, 0.75, 0.05, 1.0, 0.75])
    ax_spacer = fig.add_subplot(gs_inner[0, 2])
    ax_spacer.axis("off")

    for col_idx, (side, label, bats_side) in enumerate([("Left", "vs LHH", "L"), ("Right", "vs RHH", "R")]):
        text_col = col_idx * 3
        hm_col = col_idx * 3 + 1
        ax_text = fig.add_subplot(gs_inner[0, text_col])
        ax_hm = fig.add_subplot(gs_inner[0, hm_col])
        ax_text.axis("off")

        if col_idx == 0:
            ax_text.set_title("2-Strike Tendencies", fontsize=9, fontweight="bold",
                              pad=3, loc="left")

        sub = two_k[two_k["BatterSide"].isin([side, side[0]])] if "BatterSide" in two_k.columns else pd.DataFrame()
        summary = _two_strike_hand_summary(sub)

        if summary is None:
            ax_text.text(0.5, 0.5, f"{label}: n < 10", ha="center", va="center",
                         fontsize=8, color="#999", transform=ax_text.transAxes)
            ax_hm.axis("off")
            continue

        y = 0.92
        dy = 0.10

        # Header
        ax_text.text(0.05, y, f"{label} (n={summary['n']})", fontsize=8.5, fontweight="bold",
                     transform=ax_text.transAxes, va="top")
        y -= dy * 1.1

        # Pitch mix — compact
        mix_parts = [f"{pt} {pct:.0f}%" for pt, pct in summary["mix"][:4]]
        ax_text.text(0.05, y, " | ".join(mix_parts), fontsize=6, transform=ax_text.transAxes,
                     va="top", color="#333")
        y -= dy * 1.3

        # Stats on one line each
        stat_parts = []
        if pd.notna(summary["zone_pct"]):
            stat_parts.append(f"Zone: {summary['zone_pct']:.0f}%")
        if pd.notna(summary["whiff"]):
            stat_parts.append(f"Whiff: {summary['whiff']:.0f}%")
        if pd.notna(summary["chase"]):
            stat_parts.append(f"Chase: {summary['chase']:.0f}%")
        if stat_parts:
            w_color = "#e74c3c" if pd.notna(summary["whiff"]) and summary["whiff"] > 30 else "#333"
            ax_text.text(0.05, y, "  |  ".join(stat_parts), fontsize=6.5,
                         color=w_color, transform=ax_text.transAxes, va="top")
            y -= dy

        # Putaway pitch
        pw_color = PITCH_COLORS.get(summary["putaway"], "#333")
        ax_text.text(0.05, y, f"Putaway: {summary['putaway']}", fontsize=7,
                     fontweight="bold", color=pw_color, transform=ax_text.transAxes, va="top")

        # Heatmap: putaway pitch location
        putaway_df = sub[sub["TaggedPitchType"] == summary["putaway"]]
        if len(putaway_df) >= 8:
            grid_pct, _ = _compute_location_grid(putaway_df)
            display = grid_pct[::-1]
            ax_hm.imshow(display, cmap=_LOC_CMAP, vmin=0, vmax=25, aspect="auto")
            for yr in range(3):
                for xr in range(3):
                    v = display[yr, xr]
                    if v > 0.5:
                        ax_hm.text(xr, yr, f"{v:.0f}%", ha="center", va="center",
                                   fontsize=6, fontweight="bold",
                                   color="white" if v > 15 else "#1a3d6f")
            _style_zone_compact(ax_hm, bats_side)
            ax_hm.set_title(f"2K {summary['putaway']}", fontsize=6.5, fontweight="bold", pad=2)
        else:
            ax_hm.axis("off")
            ax_hm.text(0.5, 0.5, "n < 8", ha="center", va="center",
                       fontsize=7, color="#999", transform=ax_hm.transAxes)


# ── Baserunning Panel ────────────────────────────────────────────────────────

def _draw_baserunning_panel(fig, gs_slot, pitcher_pp, pitcher_name):
    """Draw 3-column baserunning panel: Steal Success, Pickoff Profile, Best Counts to Run."""
    gs_inner = gs_slot.subgridspec(1, 3, wspace=0.2)

    # Get CSV data
    sb_row = get_pitcher_sb_row_by_name(pitcher_name)
    pk_row = get_pickoffs_row_by_name(pitcher_name)

    # ── Column 1: Steal Success Allowed ──
    ax1 = fig.add_subplot(gs_inner[0, 0])
    ax1.axis("off")
    ax1.set_title("Baserunning vs Pitcher", fontsize=9, fontweight="bold", pad=4, loc="left")

    if sb_row:
        sba = int(sb_row.get("SBA", 0) or 0)
        sb = int(sb_row.get("SB", 0) or 0)
        cs = int(sb_row.get("CS", 0) or 0)
        sb_pct = sb / sba * 100 if sba > 0 else 0
    elif not pitcher_pp.empty:
        sba_2b = pitcher_pp["stolenBaseAttempt2B"].sum() if "stolenBaseAttempt2B" in pitcher_pp.columns else 0
        sb_2b = pitcher_pp["stolenBase2B"].sum() if "stolenBase2B" in pitcher_pp.columns else 0
        sba_3b = pitcher_pp["stolenBaseAttempt3B"].sum() if "stolenBaseAttempt3B" in pitcher_pp.columns else 0
        sb_3b = pitcher_pp["stolenBase3B"].sum() if "stolenBase3B" in pitcher_pp.columns else 0
        sba = int(sba_2b + sba_3b)
        sb = int(sb_2b + sb_3b)
        cs = sba - sb
        sb_pct = sb / sba * 100 if sba > 0 else 0
    else:
        sba, sb, cs, sb_pct = 0, 0, 0, 0

    ax1.text(0.05, 0.92, "Steal Success Allowed", fontsize=8, fontweight="bold",
             transform=ax1.transAxes, va="top")

    ax1.text(0.05, 0.72, "Runner Success%", fontsize=7, color="#666",
             transform=ax1.transAxes, va="top")
    pct_color = "#e74c3c" if sb_pct > 75 else "#2ecc71" if sb_pct < 50 else "#f39c12"
    ax1.text(0.05, 0.56, f"{sb_pct:.0f}%", fontsize=18, fontweight="bold",
             color=pct_color, transform=ax1.transAxes, va="top")

    ax1.text(0.05, 0.32, f"SB/CS: {sb} / {cs}", fontsize=7.5,
             transform=ax1.transAxes, va="top")
    ax1.text(0.05, 0.22, f"SBA: {sba}", fontsize=7.5,
             transform=ax1.transAxes, va="top")

    signal = "Higher steal vulnerability." if sb_pct > 70 else "Lower steal vulnerability." if sb_pct < 50 else "Average steal vulnerability."
    ax1.text(0.05, 0.08, f"Signal: {signal}", fontsize=6.5, color="#999",
             style="italic", transform=ax1.transAxes, va="top")

    # ── Column 2: Pickoff Profile ──
    ax2 = fig.add_subplot(gs_inner[0, 1])
    ax2.axis("off")

    ax2.text(0.05, 0.92, "Pickoff Profile", fontsize=8, fontweight="bold",
             transform=ax2.transAxes, va="top")

    if pk_row:
        pk_att = int(pk_row.get("PitcherPKAtt", 0) or 0)
        pk_success = int(pk_row.get("PitcherPK", 0) or 0)
        men_on = int(pk_row.get("PMenOn", 0) or 0)
        men_per_att = men_on / pk_att if pk_att > 0 else 0
    else:
        pk_att, pk_success, men_on, men_per_att = 0, 0, 0, 0

    ax2.text(0.05, 0.72, "Pickoff Attempts", fontsize=7, color="#666",
             transform=ax2.transAxes, va="top")
    ax2.text(0.05, 0.56, f"{pk_att}", fontsize=18, fontweight="bold",
             color="#1a1a2e", transform=ax2.transAxes, va="top")

    ax2.text(0.05, 0.32, f"Pickoffs: {pk_success}", fontsize=7.5,
             transform=ax2.transAxes, va="top")
    if men_per_att > 0:
        ax2.text(0.05, 0.22, f"Men-on per attempt: {men_per_att:.1f}", fontsize=7.5,
                 transform=ax2.transAxes, va="top")

    pk_signal = "Higher pickoff threat." if pk_att > 20 else "Lower pickoff threat."
    ax2.text(0.05, 0.08, f"Signal: {pk_signal}", fontsize=6.5, color="#999",
             style="italic", transform=ax2.transAxes, va="top")

    # ── Column 3: Best Counts to Run ──
    ax3 = fig.add_subplot(gs_inner[0, 2])
    ax3.axis("off")

    ax3.text(0.05, 0.92, "Best Counts to Run", fontsize=8, fontweight="bold",
             transform=ax3.transAxes, va="top")

    if not pitcher_pp.empty and "Balls" in pitcher_pp.columns:
        # Find the count with highest offspeed %
        _OFFSPEED = {"Slider", "Curveball", "Changeup", "Splitter", "Sweeper", "Knuckle Curve"}
        balls = pd.to_numeric(pitcher_pp["Balls"], errors="coerce")
        strikes = pd.to_numeric(pitcher_pp["Strikes"], errors="coerce")

        best_count = None
        best_os_pct = 0
        best_os_vel = 0
        best_n = 0
        for b in range(4):
            for s in range(3):
                sub = pitcher_pp[(balls == b) & (strikes == s)]
                pt_sub = sub[sub["TaggedPitchType"].notna()]
                if len(pt_sub) < 10:
                    continue
                os_mask = pt_sub["TaggedPitchType"].isin(_OFFSPEED)
                os_pct = os_mask.sum() / len(pt_sub) * 100
                os_vel = pd.to_numeric(pt_sub.loc[os_mask, "RelSpeed"], errors="coerce").mean() if os_mask.any() else np.nan
                if os_pct > best_os_pct:
                    best_os_pct = os_pct
                    best_count = f"{b}-{s}"
                    best_os_vel = os_vel
                    best_n = len(pt_sub)

        if best_count:
            ax3.text(0.05, 0.72, "Top Offspeed Count", fontsize=7, color="#666",
                     transform=ax3.transAxes, va="top")
            ax3.text(0.05, 0.56, best_count, fontsize=18, fontweight="bold",
                     color="#1a1a2e", transform=ax3.transAxes, va="top")

            vel_s = f" | AvgVelo {best_os_vel:.1f} mph" if pd.notna(best_os_vel) else ""
            ax3.text(0.05, 0.32, f"Offspeed {best_os_pct:.0f}%{vel_s}", fontsize=7,
                     transform=ax3.transAxes, va="top")
            ax3.text(0.05, 0.22, f"n={best_n}", fontsize=7, color="#999",
                     transform=ax3.transAxes, va="top")
    else:
        ax3.text(0.5, 0.5, "No count data", ha="center", va="center",
                 fontsize=8, color="#999", transform=ax3.transAxes)


# ── Platoon Splits Panel ────────────────────────────────────────────────────

def _draw_platoon_panel(ax, pitcher_pp, trad_row, rate_row):
    """Draw platoon split stats for the pitcher."""
    ax.axis("off")
    ax.set_title("Platoon Splits", fontsize=9, fontweight="bold", pad=4, loc="left")

    # Compute from pitch-level data
    splits = {}
    if not pitcher_pp.empty and "BatterSide" in pitcher_pp.columns:
        for side, label in [("Right", "vs RHB"), ("Left", "vs LHB")]:
            sub = pitcher_pp[pitcher_pp["BatterSide"].isin([side, side[0]])]
            if len(sub) < 15:
                continue
            ab = sub[sub["PlayResult"].isin(_AB_RESULTS)] if "PlayResult" in sub.columns else pd.DataFrame()
            n_ab = len(ab)
            if n_ab < 10:
                continue
            tb = ab["PlayResult"].map(lambda r: _TB_MAP.get(r, 0)).sum()
            hits = ab["PlayResult"].isin({"Single", "Double", "Triple", "HomeRun"}).sum()
            ks = (ab["PlayResult"] == "Strikeout").sum()
            ba = hits / n_ab if n_ab > 0 else 0
            slg = tb / n_ab if n_ab > 0 else 0
            k_rate = ks / n_ab * 100 if n_ab > 0 else 0
            # Whiff rate
            swings = sub[sub["PitchCall"].isin(SWING_CALLS)]
            whiffs = sub[sub["PitchCall"] == "StrikeSwinging"]
            whiff_r = len(whiffs) / len(swings) * 100 if len(swings) > 0 else 0
            splits[label] = {
                "n": len(sub), "ab": n_ab, "ba": ba, "slg": slg,
                "k_rate": k_rate, "whiff": whiff_r
            }

    if not splits:
        # Fallback to pack data
        era = _safe_num(trad_row, "ERA", np.nan)
        k_rate = _safe_num(rate_row, "K%", np.nan)
        bb_rate = _safe_num(rate_row, "BB%", np.nan)
        woba = _safe_num(rate_row, "wOBA", np.nan) or _safe_num(rate_row, "WOBA", np.nan)
        y = 0.7
        ax.text(0.05, y, "Overall (no split data):", fontsize=8, fontweight="bold",
                transform=ax.transAxes, va="top")
        y -= 0.15
        parts = []
        if pd.notna(k_rate):
            parts.append(f"K%: {k_rate:.1f}%")
        if pd.notna(bb_rate):
            parts.append(f"BB%: {bb_rate:.1f}%")
        if pd.notna(woba):
            parts.append(f"wOBA: .{int(woba * 1000):03d}")
        ax.text(0.08, y, "  |  ".join(parts), fontsize=7.5,
                transform=ax.transAxes, va="top")
        return

    # Draw splits table
    cols = ["Split", "n", "BA", "SLG", "K%", "Whiff%"]
    rows = []
    for label, s in splits.items():
        rows.append([
            label, str(s["n"]),
            f".{int(s['ba'] * 1000):03d}", f".{int(s['slg'] * 1000):03d}",
            f"{s['k_rate']:.0f}%", f"{s['whiff']:.0f}%",
        ])

    table = ax.table(cellText=rows, colLabels=cols, loc="center",
                     cellLoc="center", colColours=["#e8e8e8"] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.4)

    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=7)

    for i in range(len(rows)):
        for j in range(len(cols)):
            cell = table[i + 1, j]
            cell.set_facecolor("#ffffff" if i % 2 == 0 else "#f5f5f5")


# ── Pitcher AI Summary ──────────────────────────────────────────────────────

def _generate_pitcher_summary(name, trad_row, rate_row, pr_row, ht_row,
                               pitch_types_row, pitcher_pp, arsenal):
    """Rule-based pitcher scouting summary (4-6 sentences)."""
    ip = _safe_num(trad_row, "IP", 0)
    era = _safe_num(trad_row, "ERA", np.nan)
    fip = _safe_num(trad_row, "FIP", np.nan)
    whip = _safe_num(trad_row, "WHIP", np.nan)
    k9 = _safe_num(trad_row, "K/9", np.nan)
    bb9 = _safe_num(trad_row, "BB/9", np.nan)
    k_pct = _safe_num(rate_row, "K%", np.nan)
    bb_pct = _safe_num(rate_row, "BB%", np.nan)
    woba = _safe_num(rate_row, "wOBA", np.nan) or _safe_num(rate_row, "WOBA", np.nan)
    ba_against = _safe_num(rate_row, "BA", np.nan)
    gs = _safe_num(trad_row, "GS", 0)
    g = _safe_num(trad_row, "G", 0)

    chase = _safe_num(pr_row, "Chase%", np.nan)
    swstrk = _safe_num(pr_row, "SwStrk%", np.nan)
    contact = _safe_num(pr_row, "Contact%", np.nan)
    fpstk = _safe_num(pr_row, "FPStk%", np.nan)

    gb = _safe_num(ht_row, "Ground%", np.nan)
    fb = _safe_num(ht_row, "Fly%", np.nan)

    throws = _pitcher_throws(name)
    sentences = []

    # ── S1: Profile ──
    role = "Starter" if gs > g * 0.4 and gs > 2 else "Reliever" if g > 3 else "Pitcher"
    hand = "Left-handed" if throws == "L" else "Right-handed"

    stat_parts = []
    if pd.notna(era):
        stat_parts.append(f"{era:.2f} ERA")
    if pd.notna(fip):
        stat_parts.append(f"{fip:.2f} FIP")
    if pd.notna(whip):
        stat_parts.append(f"{whip:.2f} WHIP")
    stat_str = ", ".join(stat_parts)

    pitch_note = ""
    if arsenal:
        primary = arsenal[0]["pitch"]
        vel = arsenal[0].get("vel")
        if pd.notna(vel):
            pitch_note = f" who works off a {vel:.0f} mph {primary.lower()}"

    s1 = f"{hand} {role.lower()}{pitch_note} ({ip:.1f} IP, {stat_str})."
    sentences.append(s1)

    # ── S2: Arsenal overview ──
    if arsenal and len(arsenal) > 1:
        pitch_desc = []
        for a in arsenal[:4]:
            vel_s = f" ({a['vel']:.0f} mph)" if pd.notna(a.get("vel")) else ""
            pitch_desc.append(f"{a['pitch']}{vel_s} {a['usage']:.0f}%")
        sentences.append("Arsenal: " + ", ".join(pitch_desc) + ".")

    # ── S3: Approach / tendencies ──
    tendency_parts = []
    if pd.notna(k_pct) and pd.notna(bb_pct):
        if k_pct > 25:
            tendency_parts.append(f"high-strikeout arm ({k_pct:.0f}% K)")
        elif k_pct < 15:
            tendency_parts.append(f"low-K pitcher ({k_pct:.0f}% K) — pitch to contact")
        if bb_pct < 6:
            tendency_parts.append(f"excellent command ({bb_pct:.0f}% BB)")
        elif bb_pct > 12:
            tendency_parts.append(f"walks too many ({bb_pct:.0f}% BB)")
    if pd.notna(swstrk) and swstrk > 12:
        tendency_parts.append(f"generates in-zone whiffs ({swstrk:.0f}% SwStrk)")
    if pd.notna(chase) and chase > 32:
        tendency_parts.append(f"gets batters to chase ({chase:.0f}%)")
    if pd.notna(gb) and gb > 50:
        tendency_parts.append(f"ground-ball heavy ({gb:.0f}% GB)")
    elif pd.notna(fb) and fb > 42:
        tendency_parts.append(f"fly-ball pitcher ({fb:.0f}% FB)")
    if pd.notna(fpstk) and fpstk > 65:
        tendency_parts.append(f"gets ahead early ({fpstk:.0f}% FPStk)")
    elif pd.notna(fpstk) and fpstk < 50:
        tendency_parts.append(f"falls behind often ({fpstk:.0f}% FPStk)")

    if tendency_parts:
        sentences.append("; ".join(tendency_parts).capitalize() + ".")

    # ── S4: Platoon splits from pitch data ──
    if not pitcher_pp.empty and "BatterSide" in pitcher_pp.columns:
        for side, label in [("Right", "RHB"), ("Left", "LHB")]:
            sub = pitcher_pp[pitcher_pp["BatterSide"].isin([side, side[0]])]
            ab = sub[sub["PlayResult"].isin(_AB_RESULTS)] if "PlayResult" in sub.columns else pd.DataFrame()
            if len(ab) >= 15:
                tb = ab["PlayResult"].map(lambda r: _TB_MAP.get(r, 0)).sum()
                slg = tb / len(ab)
                ks = (ab["PlayResult"] == "Strikeout").sum()
                k_r = ks / len(ab) * 100
                ba = ab["PlayResult"].isin({"Single", "Double", "Triple", "HomeRun"}).sum() / len(ab)

        # Compare sides
        r_sub = pitcher_pp[pitcher_pp["BatterSide"].isin(["Right", "R"])]
        l_sub = pitcher_pp[pitcher_pp["BatterSide"].isin(["Left", "L"])]
        r_ab = r_sub[r_sub["PlayResult"].isin(_AB_RESULTS)] if "PlayResult" in r_sub.columns else pd.DataFrame()
        l_ab = l_sub[l_sub["PlayResult"].isin(_AB_RESULTS)] if "PlayResult" in l_sub.columns else pd.DataFrame()
        if len(r_ab) >= 15 and len(l_ab) >= 15:
            r_slg = r_ab["PlayResult"].map(lambda r: _TB_MAP.get(r, 0)).sum() / len(r_ab)
            l_slg = l_ab["PlayResult"].map(lambda r: _TB_MAP.get(r, 0)).sum() / len(l_ab)
            r_ba = r_ab["PlayResult"].isin({"Single", "Double", "Triple", "HomeRun"}).sum() / len(r_ab)
            l_ba = l_ab["PlayResult"].isin({"Single", "Double", "Triple", "HomeRun"}).sum() / len(l_ab)
            diff = abs(r_slg - l_slg)
            if diff > 0.150:
                weaker = "RHB" if r_slg > l_slg else "LHB"
                stronger = "LHB" if weaker == "RHB" else "RHB"
                sentences.append(
                    f"Significant platoon split — {weaker} hit .{int(min(r_slg,l_slg)*1000):03d} SLG, "
                    f"{stronger} slug .{int(max(r_slg,l_slg)*1000):03d}. "
                    f"More vulnerable to {stronger}."
                )
            elif diff > 0.060:
                better = "RHB" if r_slg > l_slg else "LHB"
                sentences.append(
                    f"Slight platoon lean — {better} hit him slightly harder "
                    f"(.{int(r_slg*1000):03d} vs R, .{int(l_slg*1000):03d} vs L)."
                )

    # ── S5: How to attack ──
    attack = []
    if pd.notna(fpstk) and fpstk > 60:
        attack.append("gets ahead, so be aggressive early in counts")
    elif pd.notna(fpstk) and fpstk < 50:
        attack.append("falls behind — be patient and work counts")
    if pd.notna(bb_pct) and bb_pct > 10:
        attack.append("prone to walks, take pitches")
    if pd.notna(contact) and contact > 82:
        attack.append("low whiff rate — look to put the ball in play")
    if arsenal:
        primary = arsenal[0]["pitch"]
        if primary == "Fastball" and arsenal[0].get("vel") and arsenal[0]["vel"] < 88:
            attack.append(f"fastball is hittable ({arsenal[0]['vel']:.0f} mph)")
        if len(arsenal) >= 2:
            secondary = arsenal[1]["pitch"]
            attack.append(f"look for the {secondary.lower()} as the primary secondary")
    if pd.notna(gb) and gb > 50:
        attack.append("try to elevate — he's a ground-ball pitcher")
    if attack:
        sentences.append("How to attack: " + ", ".join(attack) + ".")

    if len(sentences) < 2:
        sentences.append("Limited data — approach with balanced gameplan.")

    return " ".join(sentences)


# ── Pitcher Header ───────────────────────────────────────────────────────────

def _draw_pitcher_header(ax, pitcher, trad_row):
    """Draw header bar for pitcher PDF."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_facecolor("#1a1a2e")
    ax.patch.set_alpha(1.0)

    rk = _roster_key(pitcher)
    jersey = BRYANT_JERSEY.get(rk, "")
    pos = BRYANT_POSITION.get(rk, "")
    ip = _safe_num(trad_row, "IP", 0)
    era = _safe_num(trad_row, "ERA", np.nan)

    parts = pitcher.split(", ")
    display = f"{parts[1]} {parts[0]}" if len(parts) == 2 else pitcher

    text = f"#{jersey}  {display.upper()}"
    era_s = f"{era:.2f}" if pd.notna(era) else "—"
    details = f"|  {pos}  |  {ip:.1f} IP  |  {era_s} ERA"

    ax.text(0.02, 0.5, text, fontsize=16, fontweight="bold", color="white",
            va="center", ha="left", transform=ax.transAxes)
    ax.text(0.98, 0.5, details, fontsize=11, color="#cccccc",
            va="center", ha="right", transform=ax.transAxes)


# ── Catcher SB Control (Team-Level) ─────────────────────────────────────────

_BRYANT_CATCHERS = [
    {
        "name": "C. Papetti (C - BRY)",
        "CThrowsOffTrgt": "3 (72%)",  "off_pctl": 72,
        "SB3%": "80.0% (67%)",        "sb3_pctl": 67,
        "SB2%": "70.0% (81%)",        "sb2_pctl": 81,
        "CThrowSpd": "77.0 (52%)",    "spd_pctl": 52,
        "CThrowsOnTrgt": "5 (11%)",   "on_pctl": 11,
        "PopTimeSBA2": "2.03 (71%)",   "pop_pctl": 71,
    },
    {
        "name": "B. Durand (C - BRY)",
        "CThrowsOffTrgt": "4 (60%)",  "off_pctl": 60,
        "SB3%": "100.0% (35%)",       "sb3_pctl": 35,
        "SB2%": "81.3% (29%)",        "sb2_pctl": 29,
        "CThrowSpd": "79.4 (82%)",    "spd_pctl": 82,
        "CThrowsOnTrgt": "10 (26%)",  "on_pctl": 26,
        "PopTimeSBA2": "1.91 (98%)",   "pop_pctl": 98,
    },
]


def _catcher_cell_color(pctl, higher_is_better=True):
    """Return background color based on percentile — green=good, red=bad for catchers."""
    if higher_is_better:
        if pctl >= 75:
            return "#d4edda"  # green
        elif pctl >= 50:
            return "#fff9c4"  # yellow
        elif pctl >= 25:
            return "#fce4d6"  # light red/orange
        else:
            return "#f8d7da"  # red
    else:
        # Lower is better (e.g. SB% allowed, throws off target)
        if pctl >= 75:
            return "#f8d7da"
        elif pctl >= 50:
            return "#fce4d6"
        elif pctl >= 25:
            return "#fff9c4"
        else:
            return "#d4edda"


_CATCHER_IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Screenshot 2026-02-11 at 1.08.48\u202fPM.png")


def _draw_catcher_sb_panel(ax, all_pitcher_pp):
    """Draw catcher team control by embedding the screenshot image."""
    ax.set_title("Catcher Team Control", fontsize=9, fontweight="bold",
                 pad=4, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if os.path.exists(_CATCHER_IMG_PATH):
        from PIL import Image as _PILImage
        pil_img = _PILImage.open(_CATCHER_IMG_PATH)
        # Use high-quality resampling to avoid blur at smaller display size
        img_arr = np.asarray(pil_img)
        ax.imshow(img_arr, aspect="equal", interpolation="none")
        ax.set_xlim(0, img_arr.shape[1])
        ax.set_ylim(img_arr.shape[0], 0)
    else:
        ax.text(0.5, 0.5, "Catcher image not found", ha="center", va="center",
                fontsize=9, color="#999", transform=ax.transAxes)


# ── Main Pitcher Page Renderer ───────────────────────────────────────────────

def _render_pitcher_page(pitcher, pack, all_pitcher_pp):
    """Build one-page matplotlib figure for a pitcher. Returns Figure or None."""
    p_trad = _tm_player(_tm_team(pack["pitching"]["traditional"], _TEAM), pitcher)
    p_rate = _tm_player(_tm_team(pack["pitching"].get("rate", pd.DataFrame()), _TEAM), pitcher)
    p_pr = _tm_player(_tm_team(pack["pitching"].get("pitch_rates", pd.DataFrame()), _TEAM), pitcher)
    p_ht = _tm_player(_tm_team(pack["pitching"].get("hit_types", pd.DataFrame()), _TEAM), pitcher)
    p_pt = _tm_player(_tm_team(pack["pitching"].get("pitch_types", pd.DataFrame()), _TEAM), pitcher)

    ip = _safe_num(p_trad, "IP", 0)
    bf = _safe_num(p_trad, "BF", 0)
    if ip < 1 and bf < 3:
        return None

    throws = _pitcher_throws(pitcher)

    # Pitcher pitch-level data
    pitcher_pp = pd.DataFrame()
    if not all_pitcher_pp.empty and "Pitcher" in all_pitcher_pp.columns:
        pitcher_pp = all_pitcher_pp[all_pitcher_pp["Pitcher"] == pitcher].copy()

    # Build arsenal
    arsenal = _build_arsenal_data(pitcher_pp, p_pt, all_pitcher_pp)

    # AI Summary
    summary = _generate_pitcher_summary(pitcher, p_trad, p_rate, p_pr, p_ht, p_pt,
                                         pitcher_pp, arsenal)

    # ── Figure layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    gs = gridspec.GridSpec(
        8, 1, figure=fig,
        height_ratios=[0.4, 0.65, 1.15, 1.3, 1.7, 1.4, 1.15, 1.05],
        hspace=0.3, top=0.97, bottom=0.02, left=0.05, right=0.95,
    )

    # Row 0: Header
    ax_header = fig.add_subplot(gs[0])
    _draw_pitcher_header(ax_header, pitcher, p_trad)

    # Row 1: Summary
    ax_summary = fig.add_subplot(gs[1])
    _draw_summary(ax_summary, summary)

    # Row 2: Arsenal Table
    ax_arsenal = fig.add_subplot(gs[2])
    _draw_arsenal_table(ax_arsenal, arsenal)

    # Row 3: Pitch Location Heatmaps (up to 4, centered)
    _draw_pitcher_locations_row(fig, gs[3], pitcher_pp, throws=throws)

    # Row 4: Most Predictable Counts (top 3 count/hand combos)
    _draw_count_predictability(fig, gs[4], pitcher_pp)

    # Row 5: 2-Strike Tendencies (split by hand)
    _draw_two_strike_tendencies(fig, gs[5], pitcher_pp)

    # Row 6: Baserunning vs Pitcher (3 columns: steal, pickoff, best counts)
    _draw_baserunning_panel(fig, gs[6], pitcher_pp, pitcher)

    # Row 7: Catcher SB Control (table) + Platoon Splits
    gs_row7 = gs[7].subgridspec(1, 2, wspace=0.3)
    ax_catcher = fig.add_subplot(gs_row7[0, 0])
    _draw_catcher_sb_panel(ax_catcher, all_pitcher_pp)
    ax_platoon = fig.add_subplot(gs_row7[0, 1])
    _draw_platoon_panel(ax_platoon, pitcher_pp, p_trad, p_rate)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ██  MATCHUP STRATEGY  ONE-PAGER
# ══════════════════════════════════════════════════════════════════════════════


def _score_to_grade(score):
    """Convert 0-100 matchup score to letter grade."""
    if score >= 72:
        return "A+"
    if score >= 65:
        return "A"
    if score >= 60:
        return "B+"
    if score >= 55:
        return "B"
    if score >= 50:
        return "C"
    if score >= 42:
        return "D"
    return "F"


def _grade_color(grade):
    """Return color for a letter grade."""
    return {
        "A+": "#1a7a2f", "A": "#2ecc71", "B+": "#82c91e", "B": "#f5c542",
        "C": "#f5a623", "D": "#e74c3c", "F": "#a31515",
    }.get(grade, "#999999")


def _edge_label(score):
    """Convert overall_score to ADV / NEUT / VULN label."""
    if score >= 60:
        return "ADV"
    if score >= 45:
        return "NEUT"
    return "VULN"


def _edge_color(label):
    """Return color for an edge label."""
    return {"ADV": "#1a7a2f", "NEUT": "#f5a623", "VULN": "#e74c3c"}.get(label, "#999")


def _score_all_pitching_matchups(dav_arsenals, bryant_hitter_profiles):
    """Score all (our pitcher × their hitter) matchups.

    Returns dict of {hitter_name: [(pitcher_name, result_dict), ...]} sorted by
    overall_score desc within each hitter.
    """
    matchups = {}
    for h_name, h_prof in bryant_hitter_profiles.items():
        results = []
        for p_name, ars in dav_arsenals.items():
            try:
                res = _score_pitcher_vs_hitter(ars, h_prof)
                if res and "overall_score" in res:
                    results.append((p_name, res))
            except Exception:
                pass
        results.sort(key=lambda x: x[1]["overall_score"], reverse=True)
        if results:
            matchups[h_name] = results
    return matchups


def _score_all_offensive_matchups(dav_hitter_profiles, bryant_pitcher_profiles):
    """Score all (our hitter × their pitcher) matchups.

    Returns dict of {pitcher_name: [(hitter_name, result_dict), ...]} sorted by
    overall_score desc within each pitcher.
    """
    matchups = {}
    for bp_name, bp_prof in bryant_pitcher_profiles.items():
        results = []
        for h_name, h_prof in dav_hitter_profiles.items():
            try:
                res = _score_hitter_vs_pitcher(h_prof, bp_prof)
                if res and "overall_score" in res:
                    results.append((h_name, res))
            except Exception:
                pass
        results.sort(key=lambda x: x[1]["overall_score"], reverse=True)
        if results:
            matchups[bp_name] = results
    return matchups


def _compute_offensive_edge(hitter_prof, pitcher_prof):
    """Continuous 0-100 offensive edge score using Trackman pitch-type cross-reference.

    Cross-references our hitter's per-pitch-type Trackman metrics (EV, whiff%,
    barrel%, hard-hit%, contact depth) against the opponent pitcher's arsenal
    (pitch_mix %).  Usage-weighted, with adjustments for platoon, pitcher
    command/vulnerability, zone coverage, and count leverage.

    50 = neutral.  >60 = advantage.  <40 = disadvantage.
    """
    if not hitter_prof or not pitcher_prof:
        return None

    by_pt = hitter_prof.get("by_pitch_type", {})
    overall = hitter_prof.get("overall", {})
    zones = hitter_prof.get("zones", {})
    by_count = hitter_prof.get("by_count", {})
    pitch_mix = pitcher_prof.get("pitch_mix", {})

    if not pitch_mix:
        return None

    # ── 1. Per-pitch cross-reference (usage-weighted) ──────────────────
    components = []
    total_usage = 0

    for pt_name, usage_pct in pitch_mix.items():
        if usage_pct < 5:
            continue

        h_data = by_pt.get(pt_name)
        weight = usage_pct / 100.0
        total_usage += weight

        if h_data and h_data.get("seen", 0) >= 10:
            factors = []

            # EV component (40% of pitch score) — higher = better for hitter
            ev = h_data.get("avg_ev")
            if pd.notna(ev) and ev > 0:
                # 78→20, 85→45, 90→60, 95→75, 100→90
                ev_score = np.clip((ev - 78) / 22 * 70 + 20, 15, 90)
                factors.append(("ev", ev_score, 0.35))

            # Whiff component (25%) — lower = better for hitter
            whiff = h_data.get("whiff_pct")
            if pd.notna(whiff):
                # 0%→82, 15%→60, 30%→38, 50%→15
                whiff_score = np.clip(82 - whiff * 1.34, 15, 85)
                factors.append(("whiff", whiff_score, 0.25))

            # Barrel component (15%) — higher = better
            barrel = h_data.get("barrel_pct")
            if pd.notna(barrel):
                # 0%→30, 5%→45, 10%→60, 20%→80
                barrel_score = np.clip(30 + barrel * 2.5, 20, 85)
                factors.append(("barrel", barrel_score, 0.15))

            # Hard-hit% component (10%)
            hh = h_data.get("hard_hit_pct")
            if pd.notna(hh):
                # 0→30, 20→46, 40→62, 60→78
                hh_score = np.clip(30 + hh * 0.8, 20, 82)
                factors.append(("hh", hh_score, 0.10))

            # Contact depth component (7.5%) — more negative = out in front = good
            cd = h_data.get("contact_depth")
            if pd.notna(cd):
                # -3→70 (well out front), 0→50 (on time), +3→30 (late)
                cd_score = np.clip(50 - cd * 6.67, 25, 80)
                factors.append(("cd", cd_score, 0.075))

            # Swing% component (7.5%) — moderate swing rate is best
            sw_pct = h_data.get("swing_pct")
            if pd.notna(sw_pct):
                # sweet spot ~55-65%, penalize extremes
                swing_dist = abs(sw_pct - 60)
                sw_score = np.clip(70 - swing_dist * 1.0, 30, 75)
                factors.append(("sw", sw_score, 0.075))

            if factors:
                w_total = sum(w for _, _, w in factors)
                pitch_score = sum(s * w for _, s, w in factors) / w_total
            else:
                pitch_score = 50
        else:
            # No pitch-specific data — use overall quality, dampened
            ov_ev = overall.get("avg_ev")
            ov_barrel = overall.get("barrel_pct")
            sub_factors = []
            if pd.notna(ov_ev) and ov_ev > 0:
                sub_factors.append(np.clip((ov_ev - 78) / 22 * 70 + 20, 20, 80))
            if pd.notna(ov_barrel):
                sub_factors.append(np.clip(30 + ov_barrel * 2.5, 25, 75))
            if sub_factors:
                pitch_score = np.mean(sub_factors) * 0.65 + 50 * 0.35  # dampen
            else:
                pitch_score = 50

        components.append((pitch_score, weight))

    if not components or total_usage <= 0:
        return None

    raw = sum(s * w for s, w in components) / total_usage

    # ── 2. Platoon adjustment (±6-8%) ─────────────────────────────────
    bats = hitter_prof.get("bats", "")
    throws = pitcher_prof.get("throws", "")
    if bats and throws:
        b = bats[0] if len(bats) > 1 else bats
        t = throws[0] if len(throws) > 1 else throws
        if (b == "L" and t == "R") or (b == "R" and t == "L"):
            raw *= 1.07
        elif b == t:
            raw *= 0.93
        elif b in ("S", "B"):
            raw *= 1.03

    # ── 3. Pitcher vulnerability adjustments ──────────────────────────
    # High walk rate — hitters benefit from free bases / hitter's counts
    bb_pct = pitcher_prof.get("bb_pct")
    if pd.notna(bb_pct) and bb_pct > 8:
        raw += (bb_pct - 8) * 0.6

    # High EV-against — pitcher is hittable
    ev_against = pitcher_prof.get("ev_against")
    if pd.notna(ev_against) and ev_against > 87:
        raw += (ev_against - 87) * 0.8

    # Low chase rate — pitcher can't get hitters to expand
    chase = pitcher_prof.get("chase_pct")
    if pd.notna(chase) and chase < 25:
        raw += (25 - chase) * 0.3

    # High K rate — pitcher is tough
    k_pct = pitcher_prof.get("k_pct")
    if pd.notna(k_pct) and k_pct > 22:
        raw -= (k_pct - 22) * 0.5

    # Low contact% — pitcher creates whiffs
    contact = pitcher_prof.get("contact_pct")
    if pd.notna(contact) and contact < 75:
        raw -= (75 - contact) * 0.3

    # ── 4. Zone coverage bonus ────────────────────────────────────────
    # If pitcher goes to zones where our hitter is strong
    loc_high = pitcher_prof.get("loc_uphalf_pct")
    loc_low = pitcher_prof.get("loc_lowhalf_pct")
    loc_in = pitcher_prof.get("loc_inhalf_pct")
    loc_out = pitcher_prof.get("loc_outhalf_pct")

    if zones:
        # Check if pitcher throws to our hitter's strong zones
        up_in = zones.get("up_in", {})
        up_away = zones.get("up_away", {})
        down_in = zones.get("down_in", {})
        down_away = zones.get("down_away", {})

        # Find our hitter's strongest zone by EV
        zone_evs = {}
        for zn, zd in zones.items():
            if zn.startswith("chase"):
                continue
            zev = zd.get("avg_ev")
            if pd.notna(zev) and zd.get("n", 0) >= 5:
                zone_evs[zn] = zev

        if zone_evs:
            best_zone = max(zone_evs, key=zone_evs.get)
            best_ev = zone_evs[best_zone]
            # Bonus if pitcher throws to our hot zone
            if "up" in best_zone and pd.notna(loc_high) and loc_high > 30:
                raw += min((loc_high - 30) * 0.15, 3)
            if "down" in best_zone and pd.notna(loc_low) and loc_low > 40:
                raw += min((loc_low - 40) * 0.15, 3)

    # ── 5. Count leverage bonus ───────────────────────────────────────
    # If our hitter performs well when ahead in the count
    ahead = by_count.get("ahead", {})
    behind = by_count.get("behind", {})
    if ahead and behind:
        ahead_ev = ahead.get("avg_ev")
        behind_ev = behind.get("avg_ev")
        if pd.notna(ahead_ev) and pd.notna(behind_ev) and ahead_ev > behind_ev + 3:
            # Our hitter is a count-leverager — bonus if pitcher walks people
            if pd.notna(bb_pct) and bb_pct > 8:
                raw += 1.5

    return float(np.clip(raw, 15, 85))


def _pick_top_note(result_dict, max_len=28):
    """Extract the most useful short note from a matchup result."""
    recs = result_dict.get("recommendations", [])
    if recs:
        note = recs[0]
        return (note[:max_len - 1] + "…") if len(note) > max_len else note
    notes = result_dict.get("approach_notes", [])
    if notes:
        note = notes[0]
        return (note[:max_len - 1] + "…") if len(note) > max_len else note
    return ""


# ── Drawing functions for matchup page ────────────────────────────────────


def _draw_matchup_header(ax):
    """Full-width navy header bar for matchup strategy page."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_facecolor("#1a1a2e")
    ax.patch.set_alpha(1.0)

    ax.text(0.5, 0.5,
            "DAVIDSON  vs  BRYANT   ·   GAME-DAY MATCHUP STRATEGY   ·   Feb 2026",
            fontsize=14, fontweight="bold", color="white",
            va="center", ha="center", transform=ax.transAxes)


def _draw_pitching_matchups(ax, pitching_matchups, pack, dav_arsenals,
                            offensive_matchups=None, bryant_pitcher_profiles=None):
    """Draw the 'OUR PITCHING vs THEIR LINEUP' table."""
    ax.axis("off")
    ax.set_title("OUR PITCHING vs THEIR LINEUP", fontsize=10, fontweight="bold",
                 pad=4, loc="left", color="#1a1a2e")

    if not pitching_matchups:
        ax.text(0.5, 0.5, "No pitching matchup data available", ha="center",
                va="center", fontsize=9, color="#999", transform=ax.transAxes)
        return

    # Sort hitters by PA desc
    h_rate = _tm_team(pack["hitting"]["rate"], _TEAM)
    hitter_pa = {}
    if not h_rate.empty:
        for _, row in h_rate.iterrows():
            name = row.get("playerFullName", "")
            pa = row.get("PA", 0)
            try:
                pa = float(pa) if pd.notna(pa) else 0
            except Exception:
                pa = 0
            hitter_pa[name] = pa

    sorted_hitters = sorted(pitching_matchups.keys(),
                            key=lambda h: hitter_pa.get(h, 0), reverse=True)

    # Build table rows (no NOTE column)
    cols = ["#", "HITTER", "B", "BEST PITCHER", "GRD"]
    rows = []
    grade_colors = []
    for h_name in sorted_hitters[:11]:
        best_pitcher, best_res = pitching_matchups[h_name][0]
        score = best_res["overall_score"]
        grade = _score_to_grade(score)
        bats = best_res.get("bats", "?")
        if bats and len(bats) > 1:
            bats = bats[0]

        rk = _roster_key(h_name)
        jersey = str(BRYANT_JERSEY.get(rk, ""))

        # Shorten pitcher name
        p_parts = best_pitcher.split(", ")
        p_display = p_parts[0] if len(p_parts) >= 1 else best_pitcher

        # Shorten hitter name
        h_parts = h_name.split(", ")
        h_display = h_parts[0] if len(h_parts) >= 1 else h_name

        rows.append([jersey, h_display, bats, p_display, grade])
        grade_colors.append(_grade_color(grade))

    if not rows:
        ax.text(0.5, 0.5, "No matchup data", ha="center", va="center",
                fontsize=9, color="#999", transform=ax.transAxes)
        return

    col_widths = [0.08, 0.28, 0.08, 0.35, 0.12]
    table = ax.table(cellText=rows, colLabels=cols, loc="upper center",
                     cellLoc="center", colWidths=col_widths,
                     colColours=["#e8e8e8"] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.25)

    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=6.5)

    for i in range(len(rows)):
        for j in range(len(cols)):
            cell = table[i + 1, j]
            cell.set_facecolor("#ffffff" if i % 2 == 0 else "#f5f5f5")
            if j == 4:  # Grade column — color the background
                cell.set_facecolor(grade_colors[i] + "30")
                cell.set_text_props(fontweight="bold", color=grade_colors[i])

    # Below table: DP BEST (top 2), HIGH LEV (top 2), BEST vs RELIEVERS
    y_bottom = 0.01

    # DP BEST — top 2 pitchers by GB-inducing arsenal
    dp_candidates = []
    for p_name, ars in dav_arsenals.items():
        pitches = ars.get("pitches", {})
        total_n = sum(p.get("count", 0) for p in pitches.values())
        if total_n < 30:
            continue
        gb_proxy = 0
        for pt_name, pt_data in pitches.items():
            if any(k in pt_name.lower() for k in ["sinker", "changeup", "cutter"]):
                gb_proxy += pt_data.get("usage_pct", 0)
        dp_candidates.append((p_name, gb_proxy))
    dp_candidates.sort(key=lambda x: x[1], reverse=True)

    dp_top2 = dp_candidates[:2]
    if dp_top2:
        dp_names = []
        for p_name, gb_pct in dp_top2:
            parts = p_name.split(", ")
            display = f"{parts[1]} {parts[0]}" if len(parts) == 2 else p_name
            dp_names.append(display)
        ax.text(0.02, y_bottom + 0.12,
                f"DP: {', '.join(dp_names)}",
                fontsize=6.5, color="#1a1a2e", fontweight="bold",
                transform=ax.transAxes, va="bottom")

    # HIGH LEV — top 2 pitchers by avg score across best hitters
    highlev_candidates = []
    for p_name in dav_arsenals:
        scores = []
        for h_name, matchup_list in pitching_matchups.items():
            for mp_name, mp_res in matchup_list:
                if mp_name == p_name:
                    scores.append(mp_res["overall_score"])
                    break
        if len(scores) >= 3:
            avg = np.mean(sorted(scores, reverse=True)[:4])
            highlev_candidates.append((p_name, avg))
    highlev_candidates.sort(key=lambda x: x[1], reverse=True)

    hl_top2 = highlev_candidates[:2]
    if hl_top2:
        hl_names = []
        for p_name, avg in hl_top2:
            parts = p_name.split(", ")
            display = f"{parts[1]} {parts[0]}" if len(parts) == 2 else p_name
            hl_names.append(f"{display} ({avg:.0f})")
        ax.text(0.02, y_bottom + 0.06,
                f"HIGH LEV: {', '.join(hl_names)}",
                fontsize=6.5, color="#1a1a2e", fontweight="bold",
                transform=ax.transAxes, va="bottom")

    # BEST HITTERS vs THEIR RELIEVERS
    if offensive_matchups and bryant_pitcher_profiles:
        reliever_lines = []
        for throw_label, throw_side in [("vs RHP", "Right"), ("vs LHP", "Left")]:
            best_h = None
            best_score = 0
            for bp_name, bp_prof in bryant_pitcher_profiles.items():
                throws = bp_prof.get("throws", "?")
                if throws not in (throw_side, throw_side[0]):
                    continue
                gs = bp_prof.get("gs", 0) or 0
                if gs > 2:
                    continue  # skip starters
                if bp_name not in offensive_matchups:
                    continue
                for h_name, res in offensive_matchups[bp_name]:
                    if res["overall_score"] > best_score:
                        best_score = res["overall_score"]
                        best_h = h_name
            if best_h:
                parts = best_h.split(", ")
                h_disp = f"{parts[1]} {parts[0]}" if len(parts) == 2 else best_h
                edge = _edge_label(best_score)
                reliever_lines.append(f"{throw_label} relief: {h_disp} ({edge})")
        if reliever_lines:
            ax.text(0.02, y_bottom,
                    "PH  " + "   |   ".join(reliever_lines),
                    fontsize=6.5, color="#1a1a2e", fontweight="bold",
                    transform=ax.transAxes, va="bottom")


def _draw_offensive_matchups(ax, offensive_matchups, dav_hitter_profiles,
                             bryant_pitcher_profiles):
    """Draw the 'OUR LINEUP vs THEIR PITCHING' table."""
    ax.axis("off")
    ax.set_title("OUR LINEUP vs THEIR PITCHING", fontsize=10, fontweight="bold",
                 pad=4, loc="left", color="#1a1a2e")

    if not offensive_matchups or not bryant_pitcher_profiles:
        ax.text(0.5, 0.5, "No offensive matchup data available", ha="center",
                va="center", fontsize=9, color="#999", transform=ax.transAxes)
        return

    # Find their likely starter (most IP)
    starter_name = None
    starter_ip = 0
    for bp_name, bp_prof in bryant_pitcher_profiles.items():
        ip = bp_prof.get("ip", 0) or 0
        gs = bp_prof.get("gs", 0) or 0
        if gs > 0 and ip > starter_ip:
            starter_ip = ip
            starter_name = bp_name

    if not starter_name:
        # fallback: just use first pitcher with matchups
        starter_name = next(iter(offensive_matchups), None)

    if not starter_name or starter_name not in offensive_matchups:
        ax.text(0.5, 0.5, "No starter matchup data", ha="center",
                va="center", fontsize=9, color="#999", transform=ax.transAxes)
        return

    s_parts = starter_name.split(", ")
    s_display = f"{s_parts[1]} {s_parts[0]}" if len(s_parts) == 2 else starter_name

    starter_matchups = offensive_matchups[starter_name]

    # Build table
    cols = ["#", "HITTER", "B", "EDGE", "NOTE"]
    rows = []
    edge_colors = []
    for h_name, res in starter_matchups[:11]:
        score = res["overall_score"]
        edge = _edge_label(score)
        bats = res.get("bats", "?")
        if bats and len(bats) > 1:
            bats = bats[0]

        jersey = str(JERSEY.get(h_name, ""))
        h_parts = h_name.split(", ")
        h_display = h_parts[0] if len(h_parts) >= 1 else h_name

        note = _pick_top_note(res)
        rows.append([jersey, h_display, bats, edge, note])
        edge_colors.append(_edge_color(edge))

    if not rows:
        ax.text(0.5, 0.5, "No matchup data", ha="center", va="center",
                fontsize=9, color="#999", transform=ax.transAxes)
        return

    # Label
    ax.text(0.02, 0.97, f"vs {s_display.upper()} (Starter)",
            fontsize=7.5, fontweight="bold", color="#444",
            transform=ax.transAxes, va="top")

    col_widths = [0.07, 0.22, 0.06, 0.10, 0.45]
    table = ax.table(cellText=rows, colLabels=cols, loc="center",
                     cellLoc="center", colWidths=col_widths,
                     colColours=["#e8e8e8"] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)

    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=6.5)

    for i in range(len(rows)):
        for j in range(len(cols)):
            cell = table[i + 1, j]
            cell.set_facecolor("#ffffff" if i % 2 == 0 else "#f5f5f5")
            if j == 3:  # Edge column
                cell.set_facecolor(edge_colors[i] + "25")
                cell.set_text_props(fontweight="bold", color=edge_colors[i])
            if j == 4:  # Note column
                cell.get_text().set_ha("left")
                cell.set_text_props(fontsize=6)

    # Pinch hit section below table
    y_ph = 0.02
    # Find best PH vs RHP and LHP relievers
    for throw_label, throw_side in [("vs RHP relief", "Right"), ("vs LHP relief", "Left")]:
        best_ph = None
        best_ph_score = 0
        best_ph_note = ""
        for bp_name, bp_prof in bryant_pitcher_profiles.items():
            throws = bp_prof.get("throws", "?")
            if throws not in (throw_side, throw_side[0]):
                continue
            gs = bp_prof.get("gs", 0) or 0
            if gs > 2:
                continue  # skip starters, want relievers
            if bp_name not in offensive_matchups:
                continue
            for h_name, res in offensive_matchups[bp_name]:
                if res["overall_score"] > best_ph_score:
                    best_ph_score = res["overall_score"]
                    best_ph = h_name
                    best_ph_note = _pick_top_note(res, max_len=35)

        if best_ph:
            ph_parts = best_ph.split(", ")
            ph_display = f"{ph_parts[1]} {ph_parts[0]}" if len(ph_parts) == 2 else best_ph
            edge = _edge_label(best_ph_score)
            ax.text(0.02, y_ph,
                    f"PH {throw_label}: {ph_display} ({edge}) — {best_ph_note}",
                    fontsize=6, color="#333", transform=ax.transAxes, va="bottom")
            y_ph += 0.05


def _draw_defensive_positioning(ax, pack):
    """Draw compact defensive positioning table for Bryant hitters."""
    ax.axis("off")
    ax.set_title("DEFENSIVE POSITIONING", fontsize=10, fontweight="bold",
                 pad=4, loc="left", color="#1a1a2e")

    h_hl = _tm_team(pack["hitting"].get("hit_locations", pd.DataFrame()), _TEAM)
    h_ht = _tm_team(pack["hitting"].get("hit_types", pd.DataFrame()), _TEAM)
    h_rate = _tm_team(pack["hitting"]["rate"], _TEAM)

    if h_hl.empty or h_ht.empty:
        ax.text(0.5, 0.5, "No batted ball data", ha="center", va="center",
                fontsize=9, color="#999", transform=ax.transAxes)
        return

    hitters_list = sorted(h_rate["playerFullName"].unique()) if not h_rate.empty else []

    cols = ["#", "HITTER", "PULL", "GB%", "REC"]
    rows = []
    rec_colors = []

    for h_name in hitters_list:
        rk = _roster_key(h_name)
        pos = BRYANT_POSITION.get(rk, "")
        if pos in _PITCHER_ONLY:
            continue

        hl_row = _tm_player(h_hl, h_name)
        ht_row = _tm_player(h_ht, h_name)

        pull = _safe_num(hl_row, "HPull%", np.nan)
        ctr = _safe_num(hl_row, "HCtr%", np.nan)
        oppo = _safe_num(hl_row, "HOppFld%", np.nan)
        gb = _safe_num(ht_row, "Ground%", np.nan)

        if pd.isna(pull) or pd.isna(gb):
            continue

        shift_result = classify_shift(
            pull_pct=pull, center_pct=ctr if pd.notna(ctr) else 33,
            oppo_pct=oppo if pd.notna(oppo) else 33,
            gb_pct=gb,
        )

        rec_type = shift_result["type"]
        rec_color = shift_result["color"]

        # Only show non-standard recommendations (or show all with standard dimmed)
        jersey = str(BRYANT_JERSEY.get(rk, ""))
        h_parts = h_name.split(", ")
        h_display = h_parts[0] if len(h_parts) >= 1 else h_name

        # Short rec label
        short_rec = {"Infield Shift": "SHIFT", "Shade Pull": "SHADE PULL",
                     "Shade Oppo": "SHADE OPPO", "Standard": "STD"}.get(rec_type, rec_type)

        rows.append([jersey, h_display, f"{pull:.0f}%", f"{gb:.0f}%", short_rec])
        rec_colors.append(rec_color)

    if not rows:
        ax.text(0.5, 0.5, "No positioning data", ha="center", va="center",
                fontsize=9, color="#999", transform=ax.transAxes)
        return

    col_widths = [0.08, 0.28, 0.14, 0.14, 0.26]
    table = ax.table(cellText=rows, colLabels=cols, loc="upper center",
                     cellLoc="center", colWidths=col_widths,
                     colColours=["#e8e8e8"] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.15)

    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=6)

    for i in range(len(rows)):
        for j in range(len(cols)):
            cell = table[i + 1, j]
            cell.set_facecolor("#ffffff" if i % 2 == 0 else "#f5f5f5")
            if j == 4:  # Rec column
                cell.set_text_props(fontweight="bold", color=rec_colors[i], fontsize=6.5)
                if rows[i][4] == "STD":
                    cell.set_text_props(color="#aaaaaa", fontweight="normal", fontsize=6.5)


def _draw_baserunning_intel(ax, pack):
    """Draw baserunning intelligence panel (their threats + our running vs them)."""
    ax.axis("off")
    ax.set_title("BASERUNNING INTELLIGENCE", fontsize=10, fontweight="bold",
                 pad=4, loc="left", color="#1a1a2e")

    y = 0.92

    # ── THEIR SPEED THREATS ──
    ax.text(0.02, y, "THEIR SPEED THREATS", fontsize=8, fontweight="bold",
            color="#1a1a2e", transform=ax.transAxes, va="top")
    y -= 0.06

    speed_df = load_speed_scores()
    sb_hitters_df = load_stolen_bases_hitters()

    bryant_speed = []
    if not speed_df.empty and "newestTeamName" in speed_df.columns:
        bc = speed_df[speed_df["newestTeamName"].str.contains("Bryant", case=False, na=False)]
        for _, row in bc.iterrows():
            name = str(row.get("playerFullName", "?"))
            spd = row.get("SpeedScore", np.nan)
            try:
                spd = float(spd)
            except Exception:
                spd = np.nan
            if pd.notna(spd) and spd >= 4.5:
                # Look up SB data
                sb_count = 0
                sb_pct = np.nan
                if not sb_hitters_df.empty and "playerFullName" in sb_hitters_df.columns:
                    sb_row = sb_hitters_df[sb_hitters_df["playerFullName"] == name]
                    if not sb_row.empty:
                        sb_count = int(sb_row.iloc[0].get("SB", 0) or 0)
                        sba = int(sb_row.iloc[0].get("SBA", 0) or 0)
                        sb_pct = sb_count / sba * 100 if sba > 0 else np.nan

                alert = "HIGH" if spd >= 6.0 else "MED"
                alert_color = "#e74c3c" if alert == "HIGH" else "#f5a623"
                bryant_speed.append({
                    "name": name, "spd": spd, "sb": sb_count,
                    "sb_pct": sb_pct, "alert": alert, "alert_color": alert_color,
                })

    # Also check SB leaders who may not have speed scores
    if not sb_hitters_df.empty and "newestTeamName" in sb_hitters_df.columns:
        bc_sb = sb_hitters_df[sb_hitters_df["newestTeamName"].str.contains("Bryant", case=False, na=False)]
        existing_names = {s["name"] for s in bryant_speed}
        for _, row in bc_sb.iterrows():
            name = str(row.get("playerFullName", "?"))
            if name in existing_names:
                continue
            sb_count = int(row.get("SB", 0) or 0)
            if sb_count >= 3:
                sba = int(row.get("SBA", 0) or 0)
                sb_pct = sb_count / sba * 100 if sba > 0 else np.nan
                bryant_speed.append({
                    "name": name, "spd": np.nan, "sb": sb_count,
                    "sb_pct": sb_pct, "alert": "MED", "alert_color": "#f5a623",
                })

    bryant_speed.sort(key=lambda s: (s["spd"] if pd.notna(s["spd"]) else 0), reverse=True)

    if bryant_speed:
        for s in bryant_speed[:5]:
            spd_s = f"{s['spd']:.1f}" if pd.notna(s["spd"]) else "—"
            sb_pct_s = f"{s['sb_pct']:.0f}%" if pd.notna(s["sb_pct"]) else "—"
            short_name = s["name"].split(", ")[0] if ", " in s["name"] else s["name"]
            line = f"  {short_name:<12s}  Spd:{spd_s}  SB:{s['sb']}  SB%:{sb_pct_s}"
            ax.text(0.02, y, line, fontsize=6, fontfamily="monospace",
                    color="#333", transform=ax.transAxes, va="top")
            ax.text(0.92, y, s["alert"], fontsize=6, fontweight="bold",
                    color=s["alert_color"], transform=ax.transAxes, va="top", ha="right")
            y -= 0.05
    else:
        ax.text(0.04, y, "No speed data available", fontsize=6.5, color="#999",
                transform=ax.transAxes, va="top")
        y -= 0.06

    y -= 0.04

    # ── OUR RUNNING vs THEM ──
    ax.text(0.02, y, "OUR RUNNING vs THEM", fontsize=8, fontweight="bold",
            color="#1a1a2e", transform=ax.transAxes, va="top")
    y -= 0.06

    catchers_df = load_stolen_bases_catchers()
    bryant_catchers = []
    if not catchers_df.empty and "newestTeamName" in catchers_df.columns:
        bc = catchers_df[catchers_df["newestTeamName"].str.contains("Bryant", case=False, na=False)]
        for _, r in bc.iterrows():
            name = str(r.get("playerFullName", "?"))
            sba = int(r.get("SBA", 0) or 0)
            sb = int(r.get("SB", 0) or 0)
            sb_pct = sb / sba * 100 if sba > 0 else np.nan
            pop = r.get("PopTimeSBA2", np.nan)
            try:
                pop = float(pop)
            except Exception:
                pop = np.nan

            if pd.notna(sb_pct) or pd.notna(pop):
                if pd.notna(sb_pct) and sb_pct > 78:
                    signal = "RUN"
                    sig_color = "#2ecc71"
                elif pd.notna(pop) and pop > 2.05:
                    signal = "RUN"
                    sig_color = "#2ecc71"
                elif pd.notna(pop) and pop < 1.92:
                    signal = "HOLD"
                    sig_color = "#e74c3c"
                else:
                    signal = "MAYBE"
                    sig_color = "#f5a623"
                bryant_catchers.append({
                    "name": name, "pop": pop, "sb_pct": sb_pct,
                    "signal": signal, "sig_color": sig_color,
                })

    if bryant_catchers:
        for c in bryant_catchers[:3]:
            pop_s = f"{c['pop']:.2f}" if pd.notna(c["pop"]) else "—"
            sb_pct_s = f"{c['sb_pct']:.0f}%" if pd.notna(c["sb_pct"]) else "—"
            short_name = c["name"].split(", ")[0] if ", " in c["name"] else c["name"]
            line = f"  {short_name:<12s}  Pop:{pop_s}  SB%:{sb_pct_s}"
            ax.text(0.02, y, line, fontsize=6, fontfamily="monospace",
                    color="#333", transform=ax.transAxes, va="top")
            ax.text(0.92, y, c["signal"], fontsize=6, fontweight="bold",
                    color=c["sig_color"], transform=ax.transAxes, va="top", ha="right")
            y -= 0.05
    else:
        ax.text(0.04, y, "No catcher data available", fontsize=6.5, color="#999",
                transform=ax.transAxes, va="top")


def _build_pitching_matrix(dav_arsenals, bryant_hitter_profiles):
    """Build full cross-product matrix: score[hitter][pitcher] = overall_score.

    Returns matrix_dict only.  Caller provides ordering via _order_pitchers_by_role().
    matrix_dict is {hitter: {pitcher: result_dict}}.
    """
    matrix = {}

    for h_name, h_prof in bryant_hitter_profiles.items():
        matrix[h_name] = {}
        for p_name, ars in dav_arsenals.items():
            try:
                res = _score_pitcher_vs_hitter(ars, h_prof)
                if res and "overall_score" in res:
                    matrix[h_name][p_name] = res
            except Exception:
                pass

    return matrix


def _build_offensive_matrix(dav_hitter_profiles, bryant_pitcher_profiles):
    """Build full cross-product matrix using continuous Trackman-based scoring.

    Returns (matrix_dict, pitcher_names_sorted, hitter_avg_scores).
    matrix_dict is {hitter: {pitcher: score_float}}.
    """
    matrix = {}
    hitter_avg_scores = {}

    for h_name, h_prof in dav_hitter_profiles.items():
        matrix[h_name] = {}
        for bp_name, bp_prof in bryant_pitcher_profiles.items():
            try:
                score = _compute_offensive_edge(h_prof, bp_prof)
                if score is not None:
                    matrix[h_name][bp_name] = score
            except Exception:
                pass

    # Compute avg score per hitter
    for h_name in matrix:
        scores = list(matrix[h_name].values())
        if scores:
            hitter_avg_scores[h_name] = np.mean(scores)

    # Sort pitchers by IP (starters first)
    sorted_pitchers = sorted(
        bryant_pitcher_profiles.keys(),
        key=lambda p: bryant_pitcher_profiles[p].get("ip", 0) or 0,
        reverse=True,
    )

    return matrix, sorted_pitchers, hitter_avg_scores


def _grade_bg_color(grade):
    """Return a lighter background color for grade cells in the matrix."""
    return {
        "A+": "#c6f0c6", "A": "#d4f5d4", "B+": "#e8f5cc",
        "B": "#fef3c7", "C": "#fde8c8", "D": "#fcd5d5", "F": "#f5c6c6",
    }.get(grade, "#f0f0f0")


def _edge_bg_color(score):
    """Return background color for an offensive edge cell."""
    if score >= 65:
        return "#c6f0c6"
    if score >= 58:
        return "#d4f5d4"
    if score >= 52:
        return "#e8f5cc"
    if score >= 46:
        return "#fef3c7"
    if score >= 40:
        return "#fde8c8"
    return "#fcd5d5"


def _short_name(full_name):
    """'Last, First' → 'Last' or 'First Last' → last word."""
    if ", " in full_name:
        return full_name.split(", ")[0]
    parts = full_name.split()
    return parts[-1] if parts else full_name


def _display_name(full_name):
    """'Last, First' → 'F. Last'."""
    if ", " in full_name:
        parts = full_name.split(", ")
        first_init = parts[1][0] if len(parts) > 1 and parts[1] else ""
        return f"{first_init}. {parts[0]}" if first_init else parts[0]
    return full_name


# ── Pitching Matrix Page ───────────────────────────────────────────────


def _render_pitching_matrix_page(pack, pitching_matrix, ordered_pitchers,
                                 dav_arsenals, bryant_hitter_profiles,
                                 pitching_matchups, offensive_matrix,
                                 bryant_pitcher_profiles,
                                 dav_hitter_profiles=None):
    """Full-page color-coded pitching matchup matrix.

    Rows = Bryant hitters, Columns = Davidson pitchers (ordered by role).
    Cells = letter grade with color background.
    ordered_pitchers = [(arsenal_key, role_tag), ...] from _order_pitchers_by_role().
    """
    if not pitching_matrix or not ordered_pitchers:
        return None

    # Sort hitters by PA desc
    h_rate_pa = {}
    for h_name in bryant_hitter_profiles:
        prof = bryant_hitter_profiles[h_name]
        h_rate_pa[h_name] = prof.get("pa", 0) or 0
    sorted_hitters = sorted(pitching_matrix.keys(),
                            key=lambda h: h_rate_pa.get(h, 0), reverse=True)

    # Use all ordered pitchers (no truncation)
    top_pitchers = [p for p, _ in ordered_pitchers]
    role_tags = [r for _, r in ordered_pitchers]
    n_pitchers = len(top_pitchers)
    n_hitters = len(sorted_hitters)

    if n_pitchers == 0 or n_hitters == 0:
        return None

    fig = plt.figure(figsize=(11, 8.5))
    gs = gridspec.GridSpec(3, 1, figure=fig,
                          height_ratios=[0.35, 6.5, 0.55],
                          hspace=0.02,
                          top=0.98, bottom=0.01, left=0.03, right=0.97)

    # Row 0: Header + subtitle combined
    ax_header = fig.add_subplot(gs[0])
    ax_header.set_xlim(0, 1); ax_header.set_ylim(0, 1)
    ax_header.set_xticks([]); ax_header.set_yticks([])
    for spine in ax_header.spines.values():
        spine.set_visible(False)
    ax_header.patch.set_facecolor("#1a1a2e"); ax_header.patch.set_alpha(1.0)
    ax_header.text(0.5, 0.62,
                   "OUR PITCHING vs THEIR LINEUP   \u00b7   MATCHUP MATRIX",
                   fontsize=14, fontweight="bold", color="white",
                   va="center", ha="center", transform=ax_header.transAxes)
    ax_header.text(0.5, 0.18,
                   "Grade = matchup edge (A+ best, F worst)  |  "
                   "Pitchers by role: S=Starter | Injured in gray  |  "
                   "Hitters sorted by PA",
                   fontsize=7, color="#aaa", ha="center", va="center",
                   transform=ax_header.transAxes)

    # Row 1: Matrix table
    ax_matrix = fig.add_subplot(gs[1])
    ax_matrix.axis("off")

    # Build column header labels — last name + role suffix
    pitcher_short = []
    for p, role in ordered_pitchers:
        last = _short_name(p)
        if role == "starter":
            pitcher_short.append(f"{last}\n(S)")
        elif role == "injured":
            pitcher_short.append(f"{last}\n(INJ)")
        else:
            pitcher_short.append(last)

    col_labels = ["#", "HITTER", "B"] + pitcher_short
    n_cols = len(col_labels)

    rows_data = []
    row_cell_colors = []

    for h_name in sorted_hitters:
        rk = _roster_key(h_name)
        jersey = str(BRYANT_JERSEY.get(rk, ""))
        bats = bryant_hitter_profiles[h_name].get("bats", "?")
        if bats and len(bats) > 1:
            bats = bats[0]

        h_display = _short_name(h_name)

        row = [jersey, h_display, bats]
        colors = ["#ffffff", "#ffffff", "#ffffff"]

        for idx, p_name in enumerate(top_pitchers):
            is_injured = role_tags[idx] == "injured"
            res = pitching_matrix[h_name].get(p_name)
            if res:
                score = res["overall_score"]
                grade = _score_to_grade(score)
                row.append(grade)
                if is_injured:
                    colors.append("#e0e0e0")  # muted gray for injured
                else:
                    colors.append(_grade_bg_color(grade))
            else:
                row.append("\u2014")
                colors.append("#e0e0e0" if is_injured else "#f0f0f0")

        rows_data.append(row)
        row_cell_colors.append(colors)

    # Column widths — narrowed fixed cols for 20+ pitcher columns
    fixed_w = [0.035, 0.09, 0.03]
    remaining = 1.0 - sum(fixed_w) - 0.01
    pitcher_col_w = remaining / max(n_pitchers, 1)
    col_widths = fixed_w + [pitcher_col_w] * n_pitchers

    table = ax_matrix.table(cellText=rows_data, colLabels=col_labels,
                            bbox=[0, 0, 1, 1],  # fill entire axes — no dead space
                            cellLoc="center",
                            colWidths=col_widths,
                            colColours=["#e8e8e8"] * n_cols)
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    # Role-based header colors
    _HDR_COLORS = {
        "starter": "#1a5276",   # blue-green tint
        "reliever": "#2c3e50",  # standard dark
        "injured": "#7f8c8d",   # gray
    }
    for j in range(n_cols):
        cell = table[0, j]
        if j < 3:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=6.5)
        else:
            p_idx = j - 3
            role = role_tags[p_idx] if p_idx < len(role_tags) else "reliever"
            cell.set_facecolor(_HDR_COLORS.get(role, "#2c3e50"))
            cell.set_text_props(color="white", fontweight="bold", fontsize=5.5)

    # Style data cells
    for i in range(len(rows_data)):
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_facecolor(row_cell_colors[i][j])
            if j >= 3:
                p_idx = j - 3
                is_injured = p_idx < len(role_tags) and role_tags[p_idx] == "injured"
                grade_text = rows_data[i][j]
                if grade_text != "\u2014":
                    gc = _grade_color(grade_text)
                    if is_injured:
                        gc = "#999999"
                    cell.set_text_props(fontweight="bold", color=gc, fontsize=7)
                else:
                    cell.set_text_props(color="#999", fontsize=7)

    # Row 2: Footer with DP, HIGH LEV, PH recommendations
    ax_footer = fig.add_subplot(gs[2])
    ax_footer.axis("off")

    # Build set of unavailable pitchers (injured + DNP) for footer filtering
    _unavailable = set()
    for lst in [_DAV_INJURED, list(_DAV_DNP)]:
        for n in lst:
            _unavailable.add(n)
            a = _PITCHER_ALIASES.get(n)
            if a:
                _unavailable.add(a)
            a2 = _PITCHER_ALIASES_REV.get(n)
            if a2:
                _unavailable.add(a2)

    y = 0.92
    # DP BEST (top 2) — exclude injured/DNP
    dp_candidates = []
    for p_name, ars in dav_arsenals.items():
        if p_name in _unavailable:
            continue
        pitches_d = ars.get("pitches", {})
        total_n = sum(p.get("count", 0) for p in pitches_d.values())
        if total_n < 30:
            continue
        gb_proxy = 0
        for pt_name, pt_data in pitches_d.items():
            if any(k in pt_name.lower() for k in ["sinker", "changeup", "cutter"]):
                gb_proxy += pt_data.get("usage_pct", 0)
        dp_candidates.append((p_name, gb_proxy))
    dp_candidates.sort(key=lambda x: x[1], reverse=True)
    dp_names = [_display_name(p) for p, _ in dp_candidates[:2]]
    if dp_names:
        ax_footer.text(0.02, y,
                       f"DP SITUATION:  {', '.join(dp_names)}  (GB-heavy arsenal)",
                       fontsize=8, fontweight="bold", color="#1a1a2e",
                       transform=ax_footer.transAxes, va="top")
    y -= 0.18

    # HIGH LEV (top 2) — exclude injured/DNP
    highlev = []
    for p_name in dav_arsenals:
        if p_name in _unavailable:
            continue
        scores = []
        for h_name in pitching_matrix:
            if p_name in pitching_matrix[h_name]:
                scores.append(pitching_matrix[h_name][p_name]["overall_score"])
        if len(scores) >= 3:
            avg = np.mean(sorted(scores, reverse=True)[:4])
            highlev.append((p_name, avg))
    highlev.sort(key=lambda x: x[1], reverse=True)
    hl_names = [f"{_display_name(p)} ({avg:.0f})" for p, avg in highlev[:2]]
    if hl_names:
        ax_footer.text(0.02, y,
                       f"HIGH LEVERAGE:  {', '.join(hl_names)}",
                       fontsize=8, fontweight="bold", color="#1a1a2e",
                       transform=ax_footer.transAxes, va="top")
    y -= 0.18

    # PH recommendations (using continuous offensive edge scores)
    if offensive_matrix and bryant_pitcher_profiles:
        ph_parts = []
        for throw_label, throw_side in [("vs RHP", "Right"), ("vs LHP", "Left")]:
            best_h, best_score = None, 0
            for bp_name, bp_prof in bryant_pitcher_profiles.items():
                throws = bp_prof.get("throws", "?")
                if throws not in (throw_side, throw_side[0]):
                    continue
                gs_count = bp_prof.get("gs", 0) or 0
                if gs_count > 2:
                    continue
                for h_name, h_scores in offensive_matrix.items():
                    score = h_scores.get(bp_name)
                    if score is not None and score > best_score:
                        best_score = score
                        best_h = h_name
            if best_h:
                edge = _edge_label(best_score)
                ph_parts.append(f"{throw_label}: {_display_name(best_h)} ({edge}, {best_score:.0f})")
        if ph_parts:
            ax_footer.text(0.02, y,
                           f"PINCH HIT:  {'   |   '.join(ph_parts)}",
                           fontsize=8, fontweight="bold", color="#1a1a2e",
                           transform=ax_footer.transAxes, va="top")
    y -= 0.18

    # Legend
    legend_grades = ["A+", "A", "B+", "B", "C", "D", "F"]
    x_leg = 0.02
    ax_footer.text(x_leg, y, "LEGEND:", fontsize=6.5, fontweight="bold",
                   color="#333", transform=ax_footer.transAxes, va="top")
    x_leg += 0.07
    for g in legend_grades:
        ax_footer.text(x_leg, y, f" {g} ", fontsize=6.5, fontweight="bold",
                       color=_grade_color(g),
                       bbox=dict(facecolor=_grade_bg_color(g), edgecolor="#ccc",
                                 boxstyle="round,pad=0.15", lw=0.5),
                       transform=ax_footer.transAxes, va="top")
        x_leg += 0.045

    ax_footer.text(x_leg + 0.02, y, "  A+ = strong advantage for us,  F = disadvantage",
                   fontsize=6, color="#777", transform=ax_footer.transAxes, va="top")

    # Grade methodology note
    y_meth = y - 0.16
    ax_footer.text(0.02, y_meth,
        "METHODOLOGY:  Per-pitch Trackman cross-reference of pitcher arsenal vs hitter pitch-type "
        "vulnerabilities.  Weights: Exit Velo 35%, Whiff Rate 25%, Barrel% 15%, Hard Hit% 10%, "
        "Contact Depth 7.5%, Swing% 7.5% + platoon adjustment.  "
        "Grades reflect relative matchup edge, not absolute pitcher quality.  "
        "Larger pitch samples yield more reliable grades.",
        fontsize=5, color="#999", transform=ax_footer.transAxes, va="top",
        fontstyle="italic")

    return fig


# ── Offensive Matrix Page ──────────────────────────────────────────────


def _render_offensive_matrix_page(offensive_matrix, sorted_bp_pitchers,
                                  hitter_avg_scores,
                                  dav_hitter_profiles, bryant_pitcher_profiles):
    """Full-page offensive matchup matrix.

    Rows = Davidson hitters (sorted by avg edge), Columns = Bryant pitchers.
    Cells = continuous 0-100 score with color background.
    """
    if not offensive_matrix or not sorted_bp_pitchers:
        return None

    # Sort hitters by avg score descending
    sorted_hitters = sorted(
        [h for h in offensive_matrix if offensive_matrix[h]],
        key=lambda h: hitter_avg_scores.get(h, 0),
        reverse=True,
    )
    if not sorted_hitters:
        return None

    # Limit to top 12 pitchers by IP
    top_pitchers = sorted_bp_pitchers[:12]
    n_pitchers = len(top_pitchers)

    fig = plt.figure(figsize=(11, 8.5))
    gs = gridspec.GridSpec(4, 1, figure=fig,
                          height_ratios=[0.3, 0.15, 5.0, 0.5],
                          hspace=0.08,
                          top=0.97, bottom=0.02, left=0.03, right=0.97)

    # Row 0: Header
    ax_header = fig.add_subplot(gs[0])
    ax_header.set_xlim(0, 1); ax_header.set_ylim(0, 1)
    ax_header.set_xticks([]); ax_header.set_yticks([])
    for spine in ax_header.spines.values():
        spine.set_visible(False)
    ax_header.patch.set_facecolor("#1a1a2e"); ax_header.patch.set_alpha(1.0)
    ax_header.text(0.5, 0.5,
                   "OUR LINEUP vs THEIR PITCHING   \u00b7   MATCHUP MATRIX",
                   fontsize=14, fontweight="bold", color="white",
                   va="center", ha="center", transform=ax_header.transAxes)

    # Row 1: Subtitle
    ax_sub = fig.add_subplot(gs[1])
    ax_sub.axis("off")
    ax_sub.text(0.5, 0.5,
                "Hitters ranked by avg edge  |  Pitchers sorted by IP  |  "
                "Per-pitch Trackman cross-reference (EV, whiff, barrel, HH%, contact depth, swing%)  |  "
                "Platoon + pitcher vulnerability adjusted",
                fontsize=6.5, color="#666", ha="center", va="center",
                transform=ax_sub.transAxes)

    # Row 2: Full-width matrix
    ax_matrix = fig.add_subplot(gs[2])
    ax_matrix.axis("off")

    # Pitcher short names for columns
    pitcher_short = []
    for p in top_pitchers:
        bp_prof = bryant_pitcher_profiles.get(p, {})
        throws = bp_prof.get("throws", "?")
        if throws and len(throws) > 1:
            throws = throws[0]
        # Manual overrides
        _last = p.split(",")[0].strip() if "," in p else p.split()[-1]
        if _last in _BRYANT_PITCHER_THROWS:
            throws = _BRYANT_PITCHER_THROWS[_last]
        parts = p.split(", ")
        if len(parts) == 2:
            pitcher_short.append(f"{parts[1][0]}.{parts[0][:6]}\n({throws})")
        else:
            pitcher_short.append(f"{_short_name(p)[:6]}\n({throws})")

    col_labels = ["#", "HITTER", "B"] + pitcher_short + ["AVG"]
    n_cols = len(col_labels)

    rows_data = []
    row_cell_colors = []

    for h_name in sorted_hitters:
        # Strip switch-hitter suffix for jersey lookup
        base_name = re.sub(r" \([RL]\)$", "", h_name)
        jersey = str(JERSEY.get(base_name, ""))
        h_disp = _short_name(h_name)
        bats = ""
        h_prof = dav_hitter_profiles.get(h_name)
        if h_prof:
            bats = h_prof.get("bats", "")
            if bats and len(bats) > 1:
                bats = bats[0]

        row = [jersey, h_disp, bats]
        colors = ["#ffffff", "#ffffff", "#ffffff"]

        scores_for_avg = []
        for bp_name in top_pitchers:
            score = offensive_matrix[h_name].get(bp_name)
            if score is not None:
                scores_for_avg.append(score)
                row.append(f"{score:.0f}")
                colors.append(_edge_bg_color(score))
            else:
                row.append("\u2014")
                colors.append("#f0f0f0")

        # AVG column
        avg = np.mean(scores_for_avg) if scores_for_avg else 0
        row.append(f"{avg:.0f}")
        colors.append(_edge_bg_color(avg))

        rows_data.append(row)
        row_cell_colors.append(colors)

    # Column widths
    fixed_w = [0.04, 0.10, 0.03]
    avg_w = [0.055]
    remaining = 1.0 - sum(fixed_w) - sum(avg_w) - 0.01
    p_col_w = remaining / max(n_pitchers, 1)
    col_widths = fixed_w + [p_col_w] * n_pitchers + avg_w

    table = ax_matrix.table(cellText=rows_data, colLabels=col_labels,
                            loc="upper center", cellLoc="center",
                            colWidths=col_widths,
                            colColours=["#e8e8e8"] * n_cols)
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=5.5)

    # Style data cells
    for i in range(len(rows_data)):
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_facecolor(row_cell_colors[i][j])
            if j >= 3 and j < 3 + n_pitchers:
                score_text = rows_data[i][j]
                if score_text != "\u2014":
                    try:
                        sv = float(score_text)
                        ec = _edge_color(_edge_label(sv))
                        cell.set_text_props(fontweight="bold", color=ec, fontsize=7)
                    except ValueError:
                        pass
            elif j == n_cols - 1:  # AVG column
                try:
                    sv = float(rows_data[i][j])
                    ec = _edge_color(_edge_label(sv))
                    cell.set_text_props(fontweight="bold", color=ec, fontsize=7.5)
                except ValueError:
                    pass
            elif j == 1:  # hitter name left-align
                cell.get_text().set_ha("left")

    # Row 3: Legend
    ax_legend = fig.add_subplot(gs[3])
    ax_legend.axis("off")
    x = 0.02
    ax_legend.text(x, 0.7, "LEGEND:", fontsize=7, fontweight="bold",
                   color="#333", transform=ax_legend.transAxes, va="center")
    x += 0.06
    for label, score_ex, color in [("60+ ADV", 65, "#1a7a2f"),
                                    ("45-59 NEUT", 50, "#f5a623"),
                                    ("<45 VULN", 35, "#e74c3c")]:
        ax_legend.text(x, 0.7, f" {label} ", fontsize=7, fontweight="bold",
                       color=color,
                       bbox=dict(facecolor=_edge_bg_color(score_ex),
                                 edgecolor="#ccc", boxstyle="round,pad=0.15", lw=0.5),
                       transform=ax_legend.transAxes, va="center")
        x += 0.10
    ax_legend.text(x + 0.02, 0.7,
                   "Score = Trackman cross-reference (EV 35%, Whiff 25%, Barrel 15%, HH 10%, "
                   "Depth 7.5%, Swing 7.5%) + platoon + pitcher vulnerability",
                   fontsize=5.5, color="#777",
                   transform=ax_legend.transAxes, va="center")

    return fig


# ── Team Overview Data Functions ──────────────────────────────────────


def _get_bryant_speed_threats():
    """Return list of Bryant speed threats from speed_scores data."""
    df = load_speed_scores()
    if df.empty:
        return []
    if "newestTeamName" in df.columns:
        df = df[df["newestTeamName"] == "Bryant University"]
    elif "newestTeamAbbrevName" in df.columns:
        df = df[df["newestTeamAbbrevName"] == "BRY"]
    else:
        return []

    threats = []
    for _, row in df.iterrows():
        speed = row.get("SpeedScore", 0) or 0
        sb = row.get("SB", 0) or 0
        sba = row.get("SBA", 0) or 0
        try:
            speed = float(speed)
            sb = int(float(sb))
            sba = int(float(sba))
        except (ValueError, TypeError):
            continue
        if speed >= 4.5 or sb > 0:
            cs = sba - sb
            sb_pct = (sb / sba * 100) if sba > 0 else 0
            name = row.get("playerFullName", row.get("abbrevName", "?"))
            threats.append({
                "name": name,
                "speed_score": speed,
                "sb": sb,
                "cs": cs,
                "sb_pct": sb_pct,
            })
    threats.sort(key=lambda x: x["speed_score"], reverse=True)
    return threats


def _get_bryant_bunt_threats():
    """Return list of Bryant bunt/squeeze threats from count_sb_squeeze data."""
    df = load_count_sb_squeeze()
    if df.empty:
        return []
    if "newestTeamName" in df.columns:
        df = df[df["newestTeamName"] == "Bryant University"]
    elif "newestTeamAbbrevName" in df.columns:
        df = df[df["newestTeamAbbrevName"] == "BRY"]
    else:
        return []

    threats = []
    for _, row in df.iterrows():
        squeeze = row.get("Squeeze #", 0) or 0
        bunt_att = row.get("Bunt Hit Att", 0) or 0
        try:
            squeeze = int(float(squeeze))
            bunt_att = int(float(bunt_att))
        except (ValueError, TypeError):
            continue
        if squeeze >= 2 or bunt_att >= 3:
            name = row.get("playerFullName", row.get("abbrevName", "?"))
            threats.append({
                "name": name,
                "squeeze": squeeze,
                "bunt_hit_att": bunt_att,
            })
    threats.sort(key=lambda x: x["squeeze"] + x["bunt_hit_att"], reverse=True)
    return threats


def _hitter_profile_tags(prof, ba, slg, hr, sb):
    """Generate profile tag string for a hitter based on stats."""
    tags = []
    iso = (slg - ba) if pd.notna(slg) and pd.notna(ba) else np.nan
    if (pd.notna(iso) and iso > .170) or (pd.notna(hr) and int(hr) >= 5):
        tags.append("Power")
    if pd.notna(sb) and int(sb) >= 5:
        tags.append("Speed")
    bb_pct = prof.get("bb_pct", np.nan) if prof else np.nan
    k_pct = prof.get("k_pct", np.nan) if prof else np.nan
    chase = prof.get("chase_pct", np.nan) if prof else np.nan
    if pd.notna(bb_pct) and bb_pct >= 10:
        tags.append("Patient")
    if pd.notna(k_pct) and k_pct <= 18:
        tags.append("Contact")
    if pd.notna(chase) and chase <= 25:
        tags.append("Disc")
    return ", ".join(tags) if tags else "-"


def _render_team_overview_page(pack, bryant_pitcher_profiles, bryant_hitter_profiles):
    """Build Page 3: Comprehensive Bryant team overview.

    - LINEUP ANALYSIS: all hitters by PA with stats + profile tags + narrative
    - PITCHING STAFF: all pitchers by IP
    - SPEED + BUNT threats
    """
    fig = plt.figure(figsize=(11, 8.5))

    # Layout: 4 rows × 2 cols
    #   Row 0 (both): Header bar
    #   Row 1 (both): LINEUP ANALYSIS table
    #   Row 2 (both): Narrative blurbs
    #   Row 3 (left):  PITCHING STAFF table
    #   Row 3 (right): Speed + Bunt threats (sub-split vertically)
    outer = gridspec.GridSpec(4, 2, figure=fig,
        height_ratios=[0.12, 2.0, 0.22, 2.5],
        width_ratios=[0.57, 0.43],
        hspace=0.06, wspace=0.04,
        top=0.97, bottom=0.02, left=0.02, right=0.98)

    # ── Header ──────────────────────────────────────────────────────────
    ax_header = fig.add_subplot(outer[0, :])
    ax_header.set_xlim(0, 1); ax_header.set_ylim(0, 1)
    ax_header.set_xticks([]); ax_header.set_yticks([])
    for spine in ax_header.spines.values():
        spine.set_visible(False)
    ax_header.patch.set_facecolor("#1a1a2e"); ax_header.patch.set_alpha(1.0)
    ax_header.text(0.5, 0.5,
                   "BRYANT UNIVERSITY \u00b7 TEAM OVERVIEW (24-25)",
                   fontsize=14, fontweight="bold", color="white",
                   va="center", ha="center", transform=ax_header.transAxes)

    # ── Prepare pack data ───────────────────────────────────────────────
    team = _TEAM
    h_rate = _tm_team(pack["hitting"]["rate"], team)
    h_cnt = _tm_team(pack["hitting"]["counting"], team)
    p_trad = _tm_team(pack["pitching"]["traditional"], team)

    # ── LINEUP ANALYSIS ─────────────────────────────────────────────────
    ax_lineup = fig.add_subplot(outer[1, :])
    ax_lineup.axis("off")
    ax_lineup.set_title("LINEUP ANALYSIS", fontsize=9, fontweight="bold",
                        color="#1a1a2e", loc="left", pad=2)

    lineup_labels = ["Player", "PA", "BA", "OBP", "SLG", "wOBA", "ISO",
                     "HR", "SB", "EV", "Brl%", "Chase%", "Profile"]
    lineup_rows = []
    lineup_colors = []
    tag_counts = {"Power": 0, "Speed": 0, "Patient": 0, "Contact": 0, "Disc": 0}

    hitters_sorted = sorted(
        bryant_hitter_profiles.items(),
        key=lambda x: x[1].get("pa", 0) or 0,
        reverse=True)

    for h_name, prof in hitters_sorted:
        pa = int(prof.get("pa", 0) or 0)

        # Rate table lookups
        pr = _tm_player(h_rate, h_name)
        ba = _safe_num(pr, "BA")
        obp = _safe_num(pr, "OBP")
        slg = _safe_num(pr, "SLG")
        woba = prof.get("woba", np.nan)
        iso = (slg - ba) if pd.notna(slg) and pd.notna(ba) else np.nan

        # Counting table lookups
        pc = _tm_player(h_cnt, h_name)
        hr = int(_safe_num(pc, "HR", 0))
        sb = int(_safe_num(pc, "SB", 0))

        # Profile stats
        ev = prof.get("ev", np.nan)
        brl = prof.get("barrel_pct", np.nan)
        chase = prof.get("chase_pct", np.nan)

        # Profile tags
        tags = _hitter_profile_tags(prof, ba, slg, hr, sb)
        for t in tag_counts:
            if t in tags:
                tag_counts[t] += 1

        def _fmt_rate(v):
            return f".{int(v * 1000):03d}" if pd.notna(v) else "-"

        row = [
            h_name,
            str(pa),
            _fmt_rate(ba), _fmt_rate(obp), _fmt_rate(slg), _fmt_rate(woba), _fmt_rate(iso),
            str(hr), str(sb),
            f"{ev:.1f}" if pd.notna(ev) else "-",
            f"{brl:.1f}" if pd.notna(brl) else "-",
            f"{chase:.1f}" if pd.notna(chase) else "-",
            tags,
        ]
        lineup_rows.append(row)
        bg = "#f7f7f7" if len(lineup_rows) % 2 == 0 else "#ffffff"
        lineup_colors.append([bg] * len(lineup_labels))

    if lineup_rows:
        col_w = [0.125, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05,
                 0.035, 0.035, 0.05, 0.045, 0.05, 0.22]
        cw_sum = sum(col_w)
        col_w = [w / cw_sum for w in col_w]

        lt = ax_lineup.table(
            cellText=lineup_rows, colLabels=lineup_labels,
            loc="upper center", cellLoc="center",
            colWidths=col_w,
            colColours=["#e8e8e8"] * len(lineup_labels))
        lt.auto_set_font_size(False)
        lt.set_fontsize(6)
        lt.scale(1, 1.25)

        for j in range(len(lineup_labels)):
            cell = lt[0, j]
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=5.5)

        for i in range(len(lineup_rows)):
            for j in range(len(lineup_labels)):
                cell = lt[i + 1, j]
                cell.set_facecolor(lineup_colors[i][j])
                if j == 0:
                    cell.get_text().set_ha("left")
                    cell.set_text_props(fontsize=5.5, fontweight="bold")
                elif j == len(lineup_labels) - 1:
                    cell.get_text().set_ha("left")
                    cell.set_text_props(fontsize=5, color="#555")

    # ── NARRATIVE BLURBS ────────────────────────────────────────────────
    ax_narr = fig.add_subplot(outer[2, :])
    ax_narr.axis("off")

    # Team aggregates
    agg_parts = []
    if not h_cnt.empty:
        _pa = h_cnt["PA"].sum() if "PA" in h_cnt.columns else 0
        _ab = h_cnt["AB"].sum() if "AB" in h_cnt.columns else 0
        _ba = h_cnt["H"].sum() / max(_ab, 1) if "H" in h_cnt.columns else np.nan
        _hr = int(h_cnt["HR"].sum()) if "HR" in h_cnt.columns else 0
        _sb = int(h_cnt["SB"].sum()) if "SB" in h_cnt.columns else 0
        _kp = h_cnt["K"].sum() / max(_pa, 1) * 100 if "K" in h_cnt.columns else np.nan
        _bbp = h_cnt["BB"].sum() / max(_pa, 1) * 100 if "BB" in h_cnt.columns else np.nan
        if pd.notna(_ba):
            agg_parts.append(f"Team: .{int(_ba*1000):03d} BA")
        agg_parts.append(f"{_hr} HR")
        agg_parts.append(f"{_sb} SB")
        if pd.notna(_kp):
            agg_parts.append(f"{_kp:.1f}% K")
        if pd.notna(_bbp):
            agg_parts.append(f"{_bbp:.1f}% BB")
    if not p_trad.empty:
        _ip = p_trad["IP"].sum() if "IP" in p_trad.columns else 0
        _era = p_trad["ER"].sum() / max(_ip, 1) * 9 if "ER" in p_trad.columns else np.nan
        _k9 = p_trad["K"].sum() / max(_ip, 1) * 9 if "K" in p_trad.columns else np.nan
        _bb9 = p_trad["BB"].sum() / max(_ip, 1) * 9 if "BB" in p_trad.columns else np.nan
        if pd.notna(_era):
            agg_parts.append(f"Staff ERA {_era:.2f}")
        if pd.notna(_k9):
            agg_parts.append(f"{_k9:.1f} K/9")
        if pd.notna(_bb9):
            agg_parts.append(f"{_bb9:.1f} BB/9")

    x_narr = 0.01
    if agg_parts:
        ax_narr.text(x_narr, 0.85, " | ".join(agg_parts),
                     fontsize=6.5, fontweight="bold", color="#1a1a2e",
                     transform=ax_narr.transAxes, va="top")

    # Profile-based blurbs
    blurbs = []
    if tag_counts.get("Power", 0) >= 4:
        blurbs.append(f"Power-heavy lineup ({tag_counts['Power']} power hitters) "
                      "\u2014 keep the ball down, avoid center-cut fastballs.")
    if tag_counts.get("Speed", 0) >= 4:
        blurbs.append(f"Speed threat ({tag_counts['Speed']} speed players) "
                      "\u2014 quick deliveries, catcher awareness critical.")
    n_patient = tag_counts.get("Patient", 0) + tag_counts.get("Disc", 0)
    if n_patient >= 6:
        blurbs.append(f"Disciplined lineup ({n_patient} patient hitters) "
                      "\u2014 must throw strikes early, avoid falling behind.")
    if not h_cnt.empty:
        if pd.notna(_kp) and _kp > 22:
            blurbs.append("Strikeout-prone \u2014 expand the zone, use putaway pitches.")
    if not p_trad.empty:
        if pd.notna(_era) and _era > 5.0:
            blurbs.append("Hittable staff \u2014 be aggressive, attack early in counts.")
        elif pd.notna(_era) and _era < 3.5:
            blurbs.append("Elite pitching staff \u2014 need disciplined, quality ABs.")
        if pd.notna(_bb9) and _bb9 > 4.0:
            blurbs.append("Control issues \u2014 work deep counts, take walks.")

    if blurbs:
        ax_narr.text(x_narr, 0.35, "   ".join(blurbs),
                     fontsize=5.5, color="#444", transform=ax_narr.transAxes,
                     va="top")

    # ── PITCHING STAFF (left column of row 3) ───────────────────────────
    ax_pitch = fig.add_subplot(outer[3, 0])
    ax_pitch.axis("off")
    ax_pitch.set_title("PITCHING STAFF", fontsize=9, fontweight="bold",
                       color="#1a1a2e", loc="left", pad=2)

    pitch_labels = ["Player", "T", "G", "GS", "IP", "ERA", "FIP", "WHIP", "K/9"]
    pitch_rows = []
    pitch_row_colors = []

    if not p_trad.empty:
        p_sorted = p_trad.sort_values("IP", ascending=False) if "IP" in p_trad.columns else p_trad
        for _, row in p_sorted.iterrows():
            pname = row.get("playerFullName", "?")
            throws = "?"
            if pname in bryant_pitcher_profiles:
                throws = bryant_pitcher_profiles[pname].get("throws", "?")
            if throws and len(throws) > 1:
                throws = throws[0]
            # Manual overrides
            _last = pname.split(",")[0].strip() if "," in pname else pname.split()[-1]
            if _last in _BRYANT_PITCHER_THROWS:
                throws = _BRYANT_PITCHER_THROWS[_last]

            g = row.get("G", 0)
            gs_v = row.get("GS", 0)
            ip = row.get("IP", 0)
            era = row.get("ERA", np.nan)
            fip = row.get("FIP", np.nan)
            whip = row.get("WHIP", np.nan)
            k9 = row.get("K/9", np.nan)

            try:
                g = int(float(g)) if pd.notna(g) else 0
            except (ValueError, TypeError):
                g = 0
            try:
                gs_v = int(float(gs_v)) if pd.notna(gs_v) else 0
            except (ValueError, TypeError):
                gs_v = 0

            p_row = [
                pname,
                throws,
                str(g), str(gs_v),
                f"{float(ip):.1f}" if pd.notna(ip) else "-",
                f"{float(era):.2f}" if pd.notna(era) else "-",
                f"{float(fip):.2f}" if pd.notna(fip) else "-",
                f"{float(whip):.2f}" if pd.notna(whip) else "-",
                f"{float(k9):.1f}" if pd.notna(k9) else "-",
            ]
            pitch_rows.append(p_row)
            bg = "#f7f7f7" if len(pitch_rows) % 2 == 0 else "#ffffff"
            pitch_row_colors.append([bg] * len(pitch_labels))

    if pitch_rows:
        pcol_w = [0.22, 0.04, 0.06, 0.06, 0.09, 0.09, 0.09, 0.09, 0.09]
        pcw_sum = sum(pcol_w)
        pcol_w = [w / pcw_sum for w in pcol_w]

        pt = ax_pitch.table(
            cellText=pitch_rows, colLabels=pitch_labels,
            loc="upper center", cellLoc="center",
            colWidths=pcol_w,
            colColours=["#e8e8e8"] * len(pitch_labels))
        pt.auto_set_font_size(False)
        pt.set_fontsize(5.5)
        pt.scale(1, 1.15)

        for j in range(len(pitch_labels)):
            cell = pt[0, j]
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=5)

        for i in range(len(pitch_rows)):
            for j in range(len(pitch_labels)):
                cell = pt[i + 1, j]
                cell.set_facecolor(pitch_row_colors[i][j])
                if j == 0:
                    cell.get_text().set_ha("left")
                    cell.set_text_props(fontsize=5.5, fontweight="bold")

    # ── SPEED + BUNT THREATS (right column of row 3, split vertically) ──
    inner_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[3, 1],
        height_ratios=[1.2, 1.0], hspace=0.08)

    # Speed threats
    ax_speed = fig.add_subplot(inner_right[0])
    ax_speed.axis("off")
    ax_speed.set_title("SPEED THREATS", fontsize=8, fontweight="bold",
                       color="#1a1a2e", loc="left", pad=2)

    speed_threats = _get_bryant_speed_threats()
    if speed_threats:
        sp_labels = ["Name", "Speed", "SB/CS", "SB%"]
        sp_rows = []
        for t in speed_threats[:8]:
            sp_rows.append([
                t["name"],
                f"{t['speed_score']:.1f}",
                f"{t['sb']}/{t['cs']}",
                f"{t['sb_pct']:.0f}%",
            ])
        sp_tbl = ax_speed.table(
            cellText=sp_rows, colLabels=sp_labels,
            loc="upper center", cellLoc="center",
            colWidths=[0.38, 0.18, 0.22, 0.18],
            colColours=["#e8e8e8"] * 4)
        sp_tbl.auto_set_font_size(False)
        sp_tbl.set_fontsize(5.5)
        sp_tbl.scale(1, 1.2)
        for j in range(4):
            cell = sp_tbl[0, j]
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=5)
        for i in range(len(sp_rows)):
            for j in range(4):
                bg = "#f7f7f7" if i % 2 == 1 else "#ffffff"
                sp_tbl[i + 1, j].set_facecolor(bg)
    else:
        ax_speed.text(0.02, 0.80, "No speed data available.", fontsize=7,
                      color="#999", transform=ax_speed.transAxes, va="top")

    # Bunt / Squeeze threats
    ax_bunt = fig.add_subplot(inner_right[1])
    ax_bunt.axis("off")
    ax_bunt.set_title("BUNT / SQUEEZE THREATS", fontsize=8,
                      fontweight="bold", color="#1a1a2e",
                      loc="left", pad=2)

    bunt_threats = _get_bryant_bunt_threats()
    if bunt_threats:
        bn_labels = ["Name", "Squeeze #", "Bunt Hit Att"]
        bn_rows = []
        for t in bunt_threats[:6]:
            bn_rows.append([
                t["name"], str(t["squeeze"]), str(t["bunt_hit_att"]),
            ])
        bn_tbl = ax_bunt.table(
            cellText=bn_rows, colLabels=bn_labels,
            loc="upper center", cellLoc="center",
            colWidths=[0.42, 0.27, 0.27],
            colColours=["#e8e8e8"] * 3)
        bn_tbl.auto_set_font_size(False)
        bn_tbl.set_fontsize(5.5)
        bn_tbl.scale(1, 1.2)
        for j in range(3):
            cell = bn_tbl[0, j]
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=5)
        for i in range(len(bn_rows)):
            for j in range(3):
                bg = "#f7f7f7" if i % 2 == 1 else "#ffffff"
                bn_tbl[i + 1, j].set_facecolor(bg)
    else:
        ax_bunt.text(0.02, 0.80, "No bunt/squeeze data available.", fontsize=7,
                     color="#999", transform=ax_bunt.transAxes, va="top")

    return fig


# ── Overview Page (Defense + Baserunning) ──────────────────────────────


def _render_overview_page(pack):
    """Build the defense + baserunning overview page. Returns Figure or None."""
    fig = plt.figure(figsize=(11, 8.5))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                          height_ratios=[0.3, 5.0],
                          width_ratios=[1, 1],
                          hspace=0.10, wspace=0.08,
                          top=0.97, bottom=0.02, left=0.03, right=0.97)

    # Row 0: Header
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_xlim(0, 1); ax_header.set_ylim(0, 1)
    ax_header.set_xticks([]); ax_header.set_yticks([])
    for spine in ax_header.spines.values():
        spine.set_visible(False)
    ax_header.patch.set_facecolor("#1a1a2e"); ax_header.patch.set_alpha(1.0)
    ax_header.text(0.5, 0.5,
                   "DAVIDSON vs BRYANT   ·   DEFENSE & BASERUNNING",
                   fontsize=14, fontweight="bold", color="white",
                   va="center", ha="center", transform=ax_header.transAxes)

    # Row 1, Left: Defensive Positioning
    ax_defense = fig.add_subplot(gs[1, 0])
    _draw_defensive_positioning(ax_defense, pack)

    # Row 1, Right: Baserunning Intelligence
    ax_baserun = fig.add_subplot(gs[1, 1])
    _draw_baserunning_intel(ax_baserun, pack)

    return fig


# ── Main entry point ────────────────────────────────────────────────────────

def main():
    print("Loading Bryant combined pack...")
    pack = load_bryant_combined_pack()
    if pack is None:
        print("  Pack not cached — building from API (this may take a few minutes)...")
        pack = build_bryant_combined_pack(progress_callback=lambda msg: print(f"    {msg}"))
    if pack is None:
        print("ERROR: Could not load or build Bryant combined pack.")
        sys.exit(1)

    print("Loading pitch-level data...")
    pitches = _load_all_pitches()
    print(f"  {len(pitches)} total pitches loaded")

    h_rate = _tm_team(pack["hitting"]["rate"], _TEAM)
    if h_rate.empty:
        print("ERROR: No hitters found in pack.")
        sys.exit(1)

    hitters = sorted(h_rate["playerFullName"].unique())
    print(f"Found {len(hitters)} players in hitting rate table")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bryant_scouting_pdfs")
    os.makedirs(out_dir, exist_ok=True)

    all_pdf_path = os.path.join(out_dir, "bryant_all_hitters.pdf")
    all_pdf = PdfPages(all_pdf_path)
    generated = 0

    for hitter in hitters:
        # Skip pure pitchers
        if not _is_hitter(hitter):
            print(f"  Skipping {hitter} (pitcher)")
            continue

        print(f"  Generating: {hitter}...")
        try:
            fig = _render_hitter_page(hitter, pack, pitches)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        if fig is None:
            print(f"    Skipped (< 5 PA)")
            continue

        safe_name = hitter.replace(", ", "_").replace(" ", "_")
        fig_path = os.path.join(out_dir, f"{safe_name}.pdf")
        fig.savefig(fig_path, bbox_inches="tight")
        all_pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        generated += 1
        print(f"    Saved → {fig_path}")

    all_pdf.close()
    print(f"\nDone! Generated {generated} hitter scouting reports.")
    print(f"  Individual PDFs: {out_dir}/")
    print(f"  Combined PDF:    {all_pdf_path}")

    # ── Pitcher PDFs ──────────────────────────────────────────────────
    print("\n--- Generating PITCHER scouting reports ---")
    pitcher_pitches = _load_pitcher_pitches()
    print(f"  {len(pitcher_pitches)} pitcher pitch-level rows loaded")

    p_trad = _tm_team(pack["pitching"].get("traditional", pd.DataFrame()), _TEAM)
    if p_trad.empty:
        print("No pitchers found in pack.")
    else:
        pitchers = sorted(p_trad["playerFullName"].unique())
        print(f"Found {len(pitchers)} pitchers in pitching traditional table")

        pitcher_pdf_path = os.path.join(out_dir, "bryant_all_pitchers.pdf")
        pitcher_pdf = PdfPages(pitcher_pdf_path)
        p_generated = 0

        for pitcher in pitchers:
            print(f"  Generating: {pitcher}...")
            try:
                fig = _render_pitcher_page(pitcher, pack, pitcher_pitches)
            except Exception as e:
                import traceback
                print(f"    ERROR: {e}")
                traceback.print_exc()
                continue

            if fig is None:
                print(f"    Skipped (< 1 IP)")
                continue

            safe_name = pitcher.replace(", ", "_").replace(" ", "_")
            fig_path = os.path.join(out_dir, f"P_{safe_name}.pdf")
            fig.savefig(fig_path, bbox_inches="tight", dpi=300)
            pitcher_pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)
            p_generated += 1
            print(f"    Saved → {fig_path}")

        pitcher_pdf.close()
        print(f"\nDone! Generated {p_generated} pitcher scouting reports.")
        print(f"  Individual PDFs: {out_dir}/P_*.pdf")
        print(f"  Combined PDF:    {pitcher_pdf_path}")

    # ── Matchup Strategy One-Pager ──────────────────────────────────────
    print("\n--- Generating MATCHUP STRATEGY page ---")
    try:
        dav_data = load_davidson_data()
        print(f"  Davidson data: {len(dav_data)} rows")
    except Exception as e:
        print(f"  WARNING: Could not load Davidson data: {e}")
        dav_data = pd.DataFrame()

    # Build Davidson pitcher arsenals
    dav_arsenals = {}
    if not dav_data.empty:
        for name in ROSTER_2026:
            pos = POSITION.get(name, "")
            if "RHP" not in pos and "LHP" not in pos:
                continue
            try:
                ars = _get_our_pitcher_arsenal(dav_data, name)
                if ars:
                    dav_arsenals[name] = ars
            except Exception:
                pass
    print(f"  Davidson pitcher arsenals: {len(dav_arsenals)}")

    # Build Bryant hitter profiles
    bryant_hitter_profiles = {}
    for h in hitters:
        rk = _roster_key(h)
        pos = BRYANT_POSITION.get(rk, "")
        if pos in _PITCHER_ONLY:
            continue
        try:
            prof = _get_opp_hitter_profile(pack, h, _TEAM, pitch_df=pitches)
            if prof:
                bryant_hitter_profiles[h] = prof
        except Exception:
            pass
    print(f"  Bryant hitter profiles: {len(bryant_hitter_profiles)}")

    # Build Davidson hitter profiles
    dav_hitter_profiles = {}
    if not dav_data.empty:
        for name in ROSTER_2026:
            pos = POSITION.get(name, "")
            if "RHP" in pos or "LHP" in pos:
                continue
            try:
                prof = _get_our_hitter_profile(dav_data, name)
                if prof:
                    dav_hitter_profiles[name] = prof
            except Exception:
                pass
    print(f"  Davidson hitter profiles: {len(dav_hitter_profiles)}")

    # ── Vannoy switch-hitter split ──────────────────────────────────────
    _VANNOY = "Vannoy, Matthew"
    if _VANNOY in dav_hitter_profiles and not dav_data.empty:
        from config import filter_davidson
        vannoy_batter_df = filter_davidson(dav_data, role="batter")
        vannoy_batter_df = vannoy_batter_df[vannoy_batter_df["Batter"] == _VANNOY]

        # Right-side profile: exclude his Left-side rows
        dav_data_vannoy_R = dav_data[
            ~((dav_data["Batter"] == _VANNOY) & (dav_data["BatterSide"] == "Left"))
        ]
        try:
            prof_r = _get_our_hitter_profile(dav_data_vannoy_R, _VANNOY)
            if prof_r:
                prof_r["bats"] = "R"
                prof_r["name"] = f"{_VANNOY} (R)"
                dav_hitter_profiles[f"{_VANNOY} (R)"] = prof_r
        except Exception:
            pass

        # Left-side profile: exclude his Right-side rows
        dav_data_vannoy_L = dav_data[
            ~((dav_data["Batter"] == _VANNOY) & (dav_data["BatterSide"] == "Right"))
        ]
        try:
            prof_l = _get_our_hitter_profile(dav_data_vannoy_L, _VANNOY)
            if prof_l:
                prof_l["bats"] = "L"
                prof_l["name"] = f"{_VANNOY} (L)"
                dav_hitter_profiles[f"{_VANNOY} (L)"] = prof_l
        except Exception:
            pass

        # Remove the combined profile
        del dav_hitter_profiles[_VANNOY]
        print(f"  Vannoy split into R/L profiles")

    # Build Bryant pitcher profiles
    bryant_pitcher_profiles = {}
    bp_trad = _tm_team(pack["pitching"].get("traditional", pd.DataFrame()), _TEAM)
    bp_pitchers = sorted(bp_trad["playerFullName"].unique()) if not bp_trad.empty else []
    for bp in bp_pitchers:
        try:
            prof = _get_opp_pitcher_profile(pack, bp, _TEAM,
                                            pitch_df=pitcher_pitches)
            if prof:
                bryant_pitcher_profiles[bp] = prof
        except Exception:
            pass
    # Apply throws overrides
    for bp_name, prof in bryant_pitcher_profiles.items():
        _last = bp_name.split(",")[0].strip() if "," in bp_name else bp_name.split()[-1]
        if _last in _BRYANT_PITCHER_THROWS:
            prof["throws"] = _BRYANT_PITCHER_THROWS[_last]
    print(f"  Bryant pitcher profiles: {len(bryant_pitcher_profiles)}")

    # Score all matchups — full cross-product matrices
    pitching_matchups = _score_all_pitching_matchups(dav_arsenals, bryant_hitter_profiles)

    pitching_matrix = _build_pitching_matrix(dav_arsenals, bryant_hitter_profiles)
    ordered_pitchers = _order_pitchers_by_role(dav_arsenals)
    offensive_matrix, sorted_bp_pitchers, hitter_avg_scores = _build_offensive_matrix(
        dav_hitter_profiles, bryant_pitcher_profiles)

    print(f"  Pitching matrix: {len(pitching_matrix)} hitters × {len(ordered_pitchers)} pitchers")
    print(f"  Offensive matrix: {len(offensive_matrix)} hitters × {len(sorted_bp_pitchers)} pitchers")

    matchup_path = os.path.join(out_dir, "bryant_matchup_strategy.pdf")
    matchup_pdf = PdfPages(matchup_path)
    pages_saved = 0

    # Page 1: Pitching matchup matrix
    fig1 = _render_pitching_matrix_page(
        pack, pitching_matrix, ordered_pitchers,
        dav_arsenals, bryant_hitter_profiles,
        pitching_matchups, offensive_matrix, bryant_pitcher_profiles,
        dav_hitter_profiles=dav_hitter_profiles)
    if fig1:
        matchup_pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)
        pages_saved += 1

    # Page 2: Offensive matchup matrix
    fig2 = _render_offensive_matrix_page(
        offensive_matrix, sorted_bp_pitchers, hitter_avg_scores,
        dav_hitter_profiles, bryant_pitcher_profiles)
    if fig2:
        matchup_pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)
        pages_saved += 1

    # Page 3: Team overview
    fig3 = _render_team_overview_page(pack, bryant_pitcher_profiles, bryant_hitter_profiles)
    if fig3:
        matchup_pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)
        pages_saved += 1

    matchup_pdf.close()
    if pages_saved:
        print(f"  Saved {pages_saved} pages → {matchup_path}")
    else:
        print("  No matchup pages generated (insufficient data).")

    print("\n=== All done! ===")


if __name__ == "__main__":
    main()
