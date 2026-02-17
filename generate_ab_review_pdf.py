"""Generate an AB Review & Game Call Grades PDF.

Produces a multi-page landscape PDF (11 x 8.5) with:
  1. Cover Page
  2-4. Starting Pitcher Deep Dive (call grade, sequences, inning progression, vs opponent)
  5-N. Per-Batter AB Review Pages (~3 ABs per page)
  N+1. Relief Pitcher Sections
  N+2. Hitter AB Review with zone overlays

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
    in_zone_mask, is_barrel_mask, display_name,
    filter_minor_pitches, _friendly_team_name,
)
from analytics.zone_vulnerability import compute_zone_swing_metrics, compute_hole_scores_3x3
from analytics.tunnel import _compute_tunnel_score
from analytics.command_plus import _compute_command_plus, _compute_pitch_pair_results
from analytics.stuff_plus import _compute_stuff_plus

from _pages.postgame import (
    _compute_call_grade,
    _pg_build_pa_pitch_rows,
    _pg_estimate_ip,
    _ZONE_X_EDGES,
    _ZONE_Y_EDGES,
    _zone_desc,
    _letter_grade,
    _grade_color,
    _MIN_PITCHER_PITCHES,
)
from _pages.pitching import (
    _build_pitch_metric_map,
    _rank_pairs,
    _rank_sequences_from_pdf,
)
from generate_postgame_report_pdf import (
    _header_bar,
    _draw_zone_rect,
    _styled_table,
    _mpl_pa_zone_plot,
    _mpl_best_zone_heatmap,
    _mpl_call_grade_box,
)

# ── Style Constants ──────────────────────────────────────────────────────────
_HDR_BG = "#1a1a2e"
_SECTION_BG = "#2c3e50"
_ALT_ROW = "#f0f3f7"
_WHITE = "#ffffff"
_DARK = "#1a1a2e"
_FIG_SIZE = (11, 8.5)


# ── Cover Page ───────────────────────────────────────────────────────────────

def _render_cover(gd, game_label):
    """Cover page: AB REVIEW & GAME CALL GRADES."""
    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(3, 1, figure=fig,
        height_ratios=[0.12, 0.50, 0.38],
        hspace=0.06, top=0.97, bottom=0.03, left=0.03, right=0.97)

    _header_bar(fig, outer[0], "AB REVIEW & GAME CALL GRADES")

    ax_info = fig.add_subplot(outer[1])
    ax_info.axis("off")

    home_id = gd["HomeTeam"].iloc[0] if "HomeTeam" in gd.columns else "?"
    away_id = gd["AwayTeam"].iloc[0] if "AwayTeam" in gd.columns else "?"
    home = _friendly_team_name(home_id)
    away = _friendly_team_name(away_id)
    total_pitches = len(gd)
    innings = int(gd["Inning"].max()) if "Inning" in gd.columns else "?"

    _score_line = ""
    if "RunsScored" in gd.columns:
        dav_runs = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID]["RunsScored"].sum()
        opp_runs = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID]["RunsScored"].sum()
        dav_runs = int(dav_runs) if pd.notna(dav_runs) else 0
        opp_runs = int(opp_runs) if pd.notna(opp_runs) else 0
        _score_line = f"\nDavidson {dav_runs} - {opp_runs} Opponent"

    ax_info.text(0.5, 0.70, f"{away}  @  {home}", fontsize=28, fontweight="900",
                 color=_DARK, va="center", ha="center", transform=ax_info.transAxes)
    ax_info.text(0.5, 0.45, f"{game_label}{_score_line}",
                 fontsize=14, color="#555", va="center", ha="center", transform=ax_info.transAxes)
    ax_info.text(0.5, 0.20, f"{innings} innings  |  {total_pitches} total pitches",
                 fontsize=11, color="#888", va="center", ha="center", transform=ax_info.transAxes)

    # Summary stats
    ax_stats = fig.add_subplot(outer[2])
    ax_stats.axis("off")
    dav_pitching = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID]
    dav_hitting = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID]
    n_pitchers = dav_pitching["Pitcher"].nunique() if "Pitcher" in dav_pitching.columns else 0
    n_batters = dav_hitting["Batter"].nunique() if "Batter" in dav_hitting.columns else 0
    ax_stats.text(0.5, 0.70,
                  f"Davidson Pitchers: {n_pitchers}  |  Davidson Batters: {n_batters}",
                  fontsize=12, color=_DARK, va="center", ha="center", transform=ax_stats.transAxes)

    return fig


# ── Starting Pitcher Deep Dive ───────────────────────────────────────────────

def _render_sp_call_grade_page(pitcher_pdf, data, pitcher, game_label):
    """Page 1 of SP deep dive: Call Grade Overview."""
    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(4, 1, figure=fig,
        height_ratios=[0.08, 0.30, 0.32, 0.30],
        hspace=0.12, top=0.97, bottom=0.03, left=0.04, right=0.96)

    dname = display_name(pitcher, escape_html=False)
    ip_est = _pg_estimate_ip(pitcher_pdf)
    n_pitches = len(pitcher_pdf)
    ks = (pitcher_pdf["KorBB"] == "Strikeout").sum() if "KorBB" in pitcher_pdf.columns else 0
    bbs = (pitcher_pdf["KorBB"] == "Walk").sum() if "KorBB" in pitcher_pdf.columns else 0
    er = 0  # Not reliably computable from pitch data

    _header_bar(fig, outer[0],
                f"SP DEEP DIVE  |  {dname}  |  {ip_est} IP, {ks}K, {bbs}BB  |  {game_label}")

    # Call grade
    grade_info = _compute_call_grade(pitcher_pdf, data, pitcher)

    ax_grade = fig.add_subplot(outer[1])
    _mpl_call_grade_box(ax_grade, grade_info)

    # Top pairs table
    ax_pairs = fig.add_subplot(outer[2])
    ax_pairs.axis("off")
    if grade_info and grade_info["top_pairs"]:
        ax_pairs.set_title("Top Pitch Pairs & Sequences", fontsize=9, fontweight="bold",
                           color=_DARK, loc="left", pad=4)
        pair_rows = []
        for p in grade_info["top_pairs"]:
            csw = f"{p.get('CSW%', 0):.1f}" if pd.notna(p.get("CSW%")) else "-"
            avg_ev = f"{p.get('Avg EV', 0):.1f}" if pd.notna(p.get("Avg EV")) else "-"
            chase = f"{p.get('Chase%', 0):.1f}" if pd.notna(p.get("Chase%")) else "-"
            pair_rows.append([p.get("Pair", "?"), csw, avg_ev, chase])
        if grade_info["top_sequences"]:
            for s in grade_info["top_sequences"]:
                csw = f"{s.get('CSW%', 0):.1f}" if pd.notna(s.get("CSW%")) else "-"
                chase = f"{s.get('Chase%', 0):.1f}" if pd.notna(s.get("Chase%")) else "-"
                pair_rows.append([s.get("Seq", "?"), csw, "-", chase])
        _styled_table(ax_pairs, pair_rows,
                      ["Pair / Sequence", "CSW%", "Avg EV", "Chase%"],
                      [0.40, 0.15, 0.15, 0.15],
                      fontsize=7.5, row_height=1.5)
    else:
        ax_pairs.text(0.5, 0.5, "Insufficient data for pair analysis",
                      fontsize=9, color="#999", ha="center", va="center",
                      transform=ax_pairs.transAxes)

    # Best location per pitch type
    ax_loc = fig.add_subplot(outer[3])
    ax_loc.axis("off")
    if grade_info and grade_info["best_locations"]:
        ax_loc.set_title("Best Location by Pitch Type", fontsize=9, fontweight="bold",
                         color=_DARK, loc="left", pad=4)
        loc_rows = [[pt, loc] for pt, loc in grade_info["best_locations"].items()]
        if loc_rows:
            _styled_table(ax_loc, loc_rows,
                          ["Pitch Type", "Best Zone (CSW%)"],
                          [0.35, 0.55],
                          fontsize=7.5, row_height=1.5)
    else:
        ax_loc.text(0.5, 0.5, "Insufficient location data",
                    fontsize=9, color="#999", ha="center", va="center",
                    transform=ax_loc.transAxes)

    return fig


def _render_sp_inning_progression(pitcher_pdf, game_label, pitcher):
    """Page 2: Inning-by-inning progression for SP."""
    dname = display_name(pitcher, escape_html=False)

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(3, 1, figure=fig,
        height_ratios=[0.08, 0.50, 0.42],
        hspace=0.10, top=0.97, bottom=0.03, left=0.04, right=0.96)

    _header_bar(fig, outer[0], f"INNING PROGRESSION  |  {dname}  |  {game_label}")

    # Inning-by-inning table
    ax_table = fig.add_subplot(outer[1])
    ax_table.axis("off")

    if "Inning" not in pitcher_pdf.columns:
        ax_table.text(0.5, 0.5, "No inning data available", fontsize=10, color="#999",
                      ha="center", va="center", transform=ax_table.transAxes)
        return fig

    innings = sorted(pitcher_pdf["Inning"].dropna().unique())
    table_rows = []
    velo_by_inning = {}
    csw_calls = ["StrikeCalled", "StrikeSwinging"]

    for inn in innings:
        inn_df = pitcher_pdf[pitcher_pdf["Inning"] == inn]
        n = len(inn_df)
        # Pitch mix
        mix = inn_df["TaggedPitchType"].value_counts() if "TaggedPitchType" in inn_df.columns else pd.Series(dtype=int)
        mix_str = ", ".join(f"{pt}:{cnt}" for pt, cnt in mix.head(4).items()) if not mix.empty else "-"
        # Avg velo
        velo = inn_df["RelSpeed"].dropna() if "RelSpeed" in inn_df.columns else pd.Series(dtype=float)
        avg_velo = f"{velo.mean():.1f}" if len(velo) > 0 else "-"
        velo_by_inning[inn] = velo.mean() if len(velo) > 0 else np.nan
        # CSW%
        csw_n = len(inn_df[inn_df["PitchCall"].isin(csw_calls)]) if "PitchCall" in inn_df.columns else 0
        csw_pct = f"{csw_n / n * 100:.0f}%" if n > 0 else "-"
        # Zone%
        loc = inn_df.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if len(loc) > 0:
            iz = in_zone_mask(loc)
            zone_pct = f"{iz.mean() * 100:.0f}%"
        else:
            zone_pct = "-"

        table_rows.append([str(int(inn)), str(n), mix_str, avg_velo, csw_pct, zone_pct])

    ax_table.set_title("Inning-by-Inning Breakdown", fontsize=9, fontweight="bold",
                       color=_DARK, loc="left", pad=4)
    if table_rows:
        _styled_table(ax_table, table_rows,
                      ["Inn", "Pitches", "Mix", "Avg Velo", "CSW%", "Zone%"],
                      [0.06, 0.08, 0.38, 0.12, 0.10, 0.10],
                      fontsize=7, row_height=1.4)

    # Fatigue detection
    ax_notes = fig.add_subplot(outer[2])
    ax_notes.axis("off")
    ax_notes.set_title("Fatigue Indicators", fontsize=9, fontweight="bold",
                       color=_DARK, loc="left", pad=4)

    notes = []
    valid_velos = [(inn, v) for inn, v in velo_by_inning.items() if not np.isnan(v)]
    if len(valid_velos) >= 3:
        early_avg = np.mean([v for inn, v in valid_velos[:2]])
        late_velos = valid_velos[2:]
        for inn, v in late_velos:
            drop = early_avg - v
            if drop > 1.5:
                notes.append(f"Velo drop of {drop:.1f} mph in inning {int(inn)} (early avg: {early_avg:.1f})")

    if not notes:
        notes.append("No significant fatigue indicators detected.")

    ax_notes.text(0.05, 0.85, "\n".join(f"  - {n}" for n in notes),
                  fontsize=9, va="top", ha="left", color=_DARK,
                  transform=ax_notes.transAxes, linespacing=1.8)

    return fig


def _render_sp_vs_opponent(pitcher_pdf, data, pitcher, game_label):
    """Page 3: vs Opponent Scouting overlay."""
    dname = display_name(pitcher, escape_html=False)

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(2, 1, figure=fig,
        height_ratios=[0.08, 0.92],
        hspace=0.06, top=0.97, bottom=0.03, left=0.04, right=0.96)

    _header_bar(fig, outer[0], f"VS OPPONENT SCOUTING  |  {dname}  |  {game_label}")

    ax_main = fig.add_subplot(outer[1])
    ax_main.axis("off")

    # Try loading Bryant pack for opponent data
    opp_data_available = False
    pack = None
    try:
        from data.bryant_combined import load_bryant_combined_pack
        pack = load_bryant_combined_pack()
        if pack and "hitting" in pack:
            opp_data_available = True
    except Exception:
        pass

    if not opp_data_available or pack is None:
        ax_main.text(0.5, 0.5, "No opponent scouting data available.\n"
                     "Build Bryant combined pack for opponent overlay.",
                     fontsize=11, color="#999", ha="center", va="center",
                     transform=ax_main.transAxes)
        return fig

    # Per-batter analysis
    if "Batter" not in pitcher_pdf.columns:
        ax_main.text(0.5, 0.5, "No batter data available.", fontsize=10, color="#999",
                     ha="center", va="center", transform=ax_main.transAxes)
        return fig

    batters_faced = pitcher_pdf.groupby("Batter").size().sort_values(ascending=False).index.tolist()
    rows = []
    for batter in batters_faced[:12]:
        batter_name = display_name(batter, escape_html=False)
        batter_pitches = pitcher_pdf[pitcher_pdf["Batter"] == batter]
        n = len(batter_pitches)

        # Check if we have hole data for this batter
        batter_side = batter_pitches["BatterSide"].iloc[0] if "BatterSide" in batter_pitches.columns else "Right"
        if pd.isna(batter_side):
            batter_side = "Right"

        hole_info = "N/A"
        attacked_pct = "N/A"

        # Try to find batter in Bryant pack
        hitting_pack = pack.get("hitting", {})
        batter_df = None
        for key, val in hitting_pack.items():
            if isinstance(val, pd.DataFrame) and not val.empty:
                if "Batter" in val.columns and batter in val["Batter"].values:
                    batter_df = val[val["Batter"] == batter]
                    break

        if batter_df is not None and not batter_df.empty:
            try:
                holes = compute_hole_scores_3x3(batter_df, batter_side)
                if holes:
                    best_hole = max(holes.items(), key=lambda x: x[1])
                    hole_zone = _zone_desc(best_hole[0][0], best_hole[0][1], batter_side)
                    hole_info = f"{hole_zone} ({best_hole[1]:.0f})"

                    # Check if pitcher attacked this zone
                    loc_pitches = batter_pitches.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                    if len(loc_pitches) > 0:
                        loc_pitches = loc_pitches.copy()
                        loc_pitches["_xb"] = np.clip(np.digitize(loc_pitches["PlateLocSide"], _ZONE_X_EDGES) - 1, 0, 2)
                        loc_pitches["_yb"] = np.clip(np.digitize(loc_pitches["PlateLocHeight"], _ZONE_Y_EDGES) - 1, 0, 2)
                        top_holes = sorted(holes.items(), key=lambda x: x[1], reverse=True)[:2]
                        hole_zones = [h[0] for h in top_holes]
                        in_hole = sum(len(loc_pitches[(loc_pitches["_xb"] == z[0]) & (loc_pitches["_yb"] == z[1])]) for z in hole_zones)
                        attacked_pct = f"{in_hole / len(loc_pitches) * 100:.0f}%"
            except Exception:
                pass

        result = "?"
        last = batter_pitches.iloc[-1]
        if last.get("KorBB") == "Strikeout":
            result = "K"
        elif last.get("KorBB") == "Walk":
            result = "BB"
        elif pd.notna(last.get("PlayResult")) and last.get("PlayResult") not in ("Undefined", ""):
            result = str(last["PlayResult"])

        rows.append([batter_name[:20], str(n), result, hole_info, attacked_pct])

    ax_main.set_title("Batters Faced — Opponent Weakness Analysis", fontsize=9,
                      fontweight="bold", color=_DARK, loc="left", pad=4)
    if rows:
        _styled_table(ax_main, rows,
                      ["Batter", "Pitches", "Result", "Top Hole (Score)", "Attacked%"],
                      [0.25, 0.08, 0.10, 0.30, 0.12],
                      fontsize=7, row_height=1.3)
    else:
        ax_main.text(0.5, 0.5, "No batters faced.", fontsize=10, color="#999",
                     ha="center", va="center", transform=ax_main.transAxes)

    return fig


# ── Per-Batter AB Review Pages ───────────────────────────────────────────────

def _get_scouting_note(batter, batter_side, zone_xb, zone_yb, pack):
    """Get a brief scouting note for the zone targeted."""
    if pack is None:
        return "N/A"
    hitting_pack = pack.get("hitting", {})
    batter_df = None
    for key, val in hitting_pack.items():
        if isinstance(val, pd.DataFrame) and not val.empty:
            if "Batter" in val.columns and batter in val["Batter"].values:
                batter_df = val[val["Batter"] == batter]
                break
    if batter_df is None or batter_df.empty:
        return "N/A"
    try:
        holes = compute_hole_scores_3x3(batter_df, batter_side)
        if (zone_xb, zone_yb) in holes:
            score = holes[(zone_xb, zone_yb)]
            zone_name = _zone_desc(zone_xb, zone_yb, batter_side)
            if score >= 60:
                return f"Hole: {score:.0f} ({zone_name})"
            elif score <= 30:
                return f"Strength: {score:.0f} ({zone_name})"
            else:
                return f"Neutral: {score:.0f} ({zone_name})"
    except Exception:
        pass
    return "N/A"


def _grade_ab_call(ab_df, batter, batter_side, pack):
    """Grade an AB's pitch selection based on opponent weakness targeting.
    Returns letter grade A-F."""
    if pack is None or len(ab_df) < 2:
        return "N/A"

    hitting_pack = pack.get("hitting", {})
    batter_df = None
    for key, val in hitting_pack.items():
        if isinstance(val, pd.DataFrame) and not val.empty:
            if "Batter" in val.columns and batter in val["Batter"].values:
                batter_df = val[val["Batter"] == batter]
                break
    if batter_df is None or batter_df.empty:
        return "N/A"

    try:
        holes = compute_hole_scores_3x3(batter_df, batter_side)
        if not holes:
            return "N/A"

        loc = ab_df.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
        if loc.empty:
            return "N/A"

        loc["_xb"] = np.clip(np.digitize(loc["PlateLocSide"], _ZONE_X_EDGES) - 1, 0, 2)
        loc["_yb"] = np.clip(np.digitize(loc["PlateLocHeight"], _ZONE_Y_EDGES) - 1, 0, 2)

        top_holes = sorted(holes.items(), key=lambda x: x[1], reverse=True)[:3]
        hole_zones = {h[0] for h in top_holes}

        targeted = sum(1 for _, r in loc.iterrows() if (int(r["_xb"]), int(r["_yb"])) in hole_zones)
        pct = targeted / len(loc) * 100

        if pct >= 60:
            return "A"
        elif pct >= 45:
            return "B"
        elif pct >= 30:
            return "C"
        elif pct >= 15:
            return "D"
        else:
            return "F"
    except Exception:
        return "N/A"


def _render_ab_review_pages(pitcher_pdf, game_label, pitcher, pack=None, role="pitcher"):
    """Render per-AB review pages for a pitcher. ~3 ABs per page.

    Returns: list of Figure objects.
    """
    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in pitcher_pdf.columns]
    sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in pitcher_pdf.columns]

    if len(pa_cols) < 2:
        return []

    pa_list = []
    for pa_key, ab in pitcher_pdf.groupby(pa_cols[1:]):
        ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
        inn = ab_sorted.iloc[0].get("Inning", "?")
        pa_list.append((inn, ab_sorted))
    pa_list.sort(key=lambda x: (int(x[0]) if pd.notna(x[0]) and str(x[0]).isdigit() else 99))

    if not pa_list:
        return []

    PAS_PER_PAGE = 3
    figures = []
    dname = display_name(pitcher, escape_html=False)

    for page_start in range(0, len(pa_list), PAS_PER_PAGE):
        page_pas = pa_list[page_start:page_start + PAS_PER_PAGE]
        n_pas = len(page_pas)

        fig = plt.figure(figsize=_FIG_SIZE)
        fig.patch.set_facecolor("white")

        h_ratios = [0.08] + [0.30] * n_pas
        if n_pas < PAS_PER_PAGE:
            h_ratios.append(1.0 - 0.08 - 0.30 * n_pas)

        page_outer = gridspec.GridSpec(len(h_ratios), 1, figure=fig,
            height_ratios=h_ratios, hspace=0.12,
            top=0.97, bottom=0.03, left=0.04, right=0.96)

        _header_bar(fig, page_outer[0], f"AB REVIEW | {game_label} | {dname}")

        for pa_i, (inn, ab_sorted) in enumerate(page_pas):
            # 3-column layout: pitch_table (55%) | zone_plot (25%) | scouting_note (20%)
            row_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=page_outer[1 + pa_i],
                wspace=0.08, width_ratios=[0.55, 0.25, 0.20])

            if role == "pitcher":
                opponent = display_name(ab_sorted.iloc[0]["Batter"], escape_html=False) if "Batter" in ab_sorted.columns else "?"
                batter = ab_sorted.iloc[0].get("Batter", "?")
            else:
                opponent = display_name(ab_sorted.iloc[0]["Pitcher"], escape_html=False) if "Pitcher" in ab_sorted.columns else "?"
                batter = ab_sorted.iloc[0].get("Batter", "?") if role == "hitter" else "?"

            batter_side = ab_sorted.iloc[0].get("BatterSide", "Right")
            if pd.isna(batter_side):
                batter_side = "Right"

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

            # AB verdict
            if role == "pitcher":
                ab_grade = _grade_ab_call(ab_sorted, batter, batter_side, pack)
                grade_str = f" [{ab_grade}]" if ab_grade != "N/A" else ""
            else:
                grade_str = ""

            pa_header = f"Inn {inn} vs {opponent} — {result} ({n_pitches_pa}p){grade_str}"

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

            # Middle: zone plot
            ax_zone = fig.add_subplot(row_gs[0, 1])
            _mpl_pa_zone_plot(ax_zone, ab_sorted)

            # Right: scouting note
            ax_scout = fig.add_subplot(row_gs[0, 2])
            ax_scout.axis("off")

            # Get scouting info for the primary zone targeted
            loc = ab_sorted.dropna(subset=["PlateLocSide", "PlateLocHeight"])
            if len(loc) > 0 and pack is not None:
                loc = loc.copy()
                loc["_xb"] = np.clip(np.digitize(loc["PlateLocSide"], _ZONE_X_EDGES) - 1, 0, 2)
                loc["_yb"] = np.clip(np.digitize(loc["PlateLocHeight"], _ZONE_Y_EDGES) - 1, 0, 2)
                # Most targeted zone
                zone_counts = loc.groupby(["_xb", "_yb"]).size()
                if not zone_counts.empty:
                    top_zone = zone_counts.idxmax()
                    note = _get_scouting_note(batter, batter_side, int(top_zone[0]), int(top_zone[1]), pack)
                else:
                    note = "N/A"
            else:
                note = "N/A"

            ax_scout.text(0.1, 0.70, "Scouting", fontsize=7, fontweight="bold",
                         color=_DARK, va="top", ha="left", transform=ax_scout.transAxes)
            ax_scout.text(0.1, 0.50, note, fontsize=6.5, color=_DARK,
                         va="top", ha="left", transform=ax_scout.transAxes, wrap=True)

        figures.append(fig)

    return figures


# ── Relief Pitcher Section ───────────────────────────────────────────────────

def _render_reliever_page(pitcher_pdf, data, pitcher, game_label, pack=None):
    """Compact reliever page: call grade, top pair, key ABs."""
    dname = display_name(pitcher, escape_html=False)
    n_pitches = len(pitcher_pdf)
    ip_est = _pg_estimate_ip(pitcher_pdf)

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(3, 1, figure=fig,
        height_ratios=[0.08, 0.35, 0.57],
        hspace=0.10, top=0.97, bottom=0.03, left=0.04, right=0.96)

    _header_bar(fig, outer[0],
                f"RELIEVER  |  {dname}  |  {ip_est} IP, {n_pitches}p  |  {game_label}")

    # Call grade
    grade_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.15)

    ax_grade = fig.add_subplot(grade_gs[0, 0])
    if n_pitches >= _MIN_PITCHER_PITCHES:
        try:
            grade_info = _compute_call_grade(pitcher_pdf, data, pitcher)
            _mpl_call_grade_box(ax_grade, grade_info)
        except Exception:
            ax_grade.axis("off")
            ax_grade.text(0.5, 0.5, "Grade N/A", fontsize=9, color="#999",
                         ha="center", va="center", transform=ax_grade.transAxes)
    else:
        ax_grade.axis("off")
        ax_grade.text(0.5, 0.5, f"< {_MIN_PITCHER_PITCHES} pitches\nNo call grade",
                     fontsize=9, color="#999", ha="center", va="center",
                     transform=ax_grade.transAxes)

    # Pitch mix summary
    ax_mix = fig.add_subplot(grade_gs[0, 1])
    ax_mix.axis("off")
    ax_mix.set_title("Pitch Mix", fontsize=8, fontweight="bold", color=_DARK, loc="left", pad=2)
    if "TaggedPitchType" in pitcher_pdf.columns:
        mix_rows = []
        total = len(pitcher_pdf)
        for pt, grp in pitcher_pdf.groupby("TaggedPitchType"):
            v = grp["RelSpeed"].dropna() if "RelSpeed" in grp.columns else pd.Series(dtype=float)
            avg_v = f"{v.mean():.1f}" if len(v) > 0 else "-"
            mix_rows.append([pt, str(len(grp)), f"{len(grp)/total*100:.0f}%", avg_v])
        mix_rows.sort(key=lambda r: int(r[1]), reverse=True)
        if mix_rows:
            _styled_table(ax_mix, mix_rows, ["Pitch", "N", "%", "Velo"],
                          [0.30, 0.15, 0.20, 0.20], fontsize=6.5, row_height=1.3)

    # Key ABs
    ax_abs = fig.add_subplot(outer[2])
    ax_abs.axis("off")
    ax_abs.set_title("Key At-Bats", fontsize=8, fontweight="bold", color=_DARK, loc="left", pad=2)

    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in pitcher_pdf.columns]
    sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in pitcher_pdf.columns]
    ab_lines = []
    if len(pa_cols) >= 2:
        for pa_key, ab in pitcher_pdf.groupby(pa_cols[1:]):
            ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
            inn = ab_sorted.iloc[0].get("Inning", "?")
            batter_name = display_name(ab_sorted.iloc[0]["Batter"], escape_html=False) if "Batter" in ab_sorted.columns else "?"
            last = ab_sorted.iloc[-1]
            result = "?"
            if last.get("KorBB") == "Strikeout":
                result = "K"
            elif last.get("KorBB") == "Walk":
                result = "BB"
            elif pd.notna(last.get("PlayResult")) and last.get("PlayResult") not in ("Undefined", ""):
                result = str(last["PlayResult"])
            ab_lines.append(f"Inn {inn} | {batter_name} — {result} ({len(ab_sorted)}p)")

    if ab_lines:
        text = "\n".join(f"  {line}" for line in ab_lines[:10])
    else:
        text = "No at-bats recorded."
    ax_abs.text(0.02, 0.92, text, fontsize=7.5, va="top", ha="left",
                color=_DARK, transform=ax_abs.transAxes,
                linespacing=1.6, fontfamily="monospace")

    return fig


# ── Hitter AB Review ─────────────────────────────────────────────────────────

def _render_hitter_ab_review_pages(bdf, data, batter, game_label):
    """Hitter AB review pages: per-AB table + season zone heatmap + swing decisions."""
    pa_cols = [c for c in ["GameID", "Batter", "Inning", "PAofInning"] if c in bdf.columns]
    sort_cols = [c for c in ["Inning", "PAofInning", "PitchNo"] if c in bdf.columns]

    if len(pa_cols) < 2:
        return []

    season_bdf = data[data["Batter"] == batter] if data is not None and not data.empty else pd.DataFrame()
    batter_side = bdf["BatterSide"].iloc[0] if "BatterSide" in bdf.columns and len(bdf) > 0 else "Right"
    if pd.isna(batter_side):
        batter_side = "Right"

    pa_list = []
    for pa_key, ab in bdf.groupby(pa_cols[1:]):
        ab_sorted = ab.sort_values(sort_cols) if sort_cols else ab
        inn = ab_sorted.iloc[0].get("Inning", "?")
        pa_list.append((inn, ab_sorted))
    pa_list.sort(key=lambda x: (int(x[0]) if pd.notna(x[0]) and str(x[0]).isdigit() else 99))

    if not pa_list:
        return []

    dname = display_name(batter, escape_html=False)
    figures = []

    # Page 1: Zone heatmap + swing decision summary + first ~2 ABs
    PAS_PER_PAGE = 2  # fewer per page since we include zone heatmap

    for page_start in range(0, len(pa_list), PAS_PER_PAGE):
        page_pas = pa_list[page_start:page_start + PAS_PER_PAGE]
        n_pas_page = len(page_pas)
        is_first_page = page_start == 0

        fig = plt.figure(figsize=_FIG_SIZE)
        fig.patch.set_facecolor("white")

        if is_first_page:
            h_ratios = [0.08, 0.30] + [0.30] * n_pas_page
            if n_pas_page < PAS_PER_PAGE:
                h_ratios.append(1.0 - sum(h_ratios))
            n_rows = len(h_ratios)
        else:
            h_ratios = [0.08] + [0.30] * n_pas_page
            if n_pas_page < PAS_PER_PAGE:
                h_ratios.append(1.0 - sum(h_ratios))
            n_rows = len(h_ratios)

        page_outer = gridspec.GridSpec(n_rows, 1, figure=fig,
            height_ratios=h_ratios, hspace=0.10,
            top=0.97, bottom=0.03, left=0.04, right=0.96)

        _header_bar(fig, page_outer[0], f"HITTER AB REVIEW | {dname} | {game_label}")

        content_row_start = 1

        if is_first_page:
            # Zone heatmap + swing decision summary
            zone_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=page_outer[1],
                wspace=0.15, width_ratios=[0.5, 0.5])

            # Season best zone heatmap (mini)
            ax_zone = fig.add_subplot(zone_gs[0, 0])
            season_source = season_bdf if not season_bdf.empty and len(season_bdf) >= 30 else bdf
            try:
                _mpl_best_zone_heatmap(ax_zone, season_source, batter_side)
            except Exception:
                ax_zone.axis("off")
                ax_zone.text(0.5, 0.5, "Zone data N/A", fontsize=8, color="#999",
                            ha="center", va="center", transform=ax_zone.transAxes)

            # Swing decision summary
            ax_decisions = fig.add_subplot(zone_gs[0, 1])
            ax_decisions.axis("off")
            ax_decisions.set_title("Swing Decisions", fontsize=8, fontweight="bold",
                                   color=_DARK, loc="left", pad=3)

            # Compute: swings in best zones vs chases
            zone_metrics = compute_zone_swing_metrics(season_source, batter_side)
            decision_text = []
            if zone_metrics:
                loc_game = bdf.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
                if not loc_game.empty:
                    loc_game["_xb"] = np.clip(np.digitize(loc_game["PlateLocSide"], _ZONE_X_EDGES) - 1, 0, 2)
                    loc_game["_yb"] = np.clip(np.digitize(loc_game["PlateLocHeight"], _ZONE_Y_EDGES) - 1, 0, 2)
                    game_swings = loc_game[loc_game["PitchCall"].isin(SWING_CALLS)]
                    iz = in_zone_mask(loc_game)
                    oz_swings = loc_game[(~iz) & loc_game["PitchCall"].isin(SWING_CALLS)]
                    decision_text.append(f"Total swings: {len(game_swings)}")
                    decision_text.append(f"In-zone swings: {len(game_swings[iz.reindex(game_swings.index, fill_value=False)])}")
                    decision_text.append(f"Chases: {len(oz_swings)}")
                    chase_pct = len(oz_swings) / max(len(loc_game[~iz]), 1) * 100
                    decision_text.append(f"Chase%: {chase_pct:.1f}%")
            else:
                decision_text.append("Insufficient data for swing analysis.")

            ax_decisions.text(0.05, 0.85, "\n".join(decision_text),
                              fontsize=8, va="top", ha="left", color=_DARK,
                              transform=ax_decisions.transAxes,
                              family="monospace", linespacing=1.6)

            content_row_start = 2

        # AB rows
        for pa_i, (inn, ab_sorted) in enumerate(page_pas):
            row_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=page_outer[content_row_start + pa_i],
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

            pa_header = f"Inn {inn} vs {vs_pitcher} — {result} ({len(ab_sorted)}p)"

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

            ax_zone = fig.add_subplot(row_gs[0, 1])
            _mpl_pa_zone_plot(ax_zone, ab_sorted)

        figures.append(fig)

    return figures


# ── Public API ───────────────────────────────────────────────────────────────

def generate_ab_review_pdf_bytes(gd, data, game_label) -> bytes:
    """Generate the AB Review & Game Call Grades PDF in memory.

    Returns raw bytes suitable for st.download_button.
    """
    buf = io.BytesIO()
    pages_saved = 0

    # Try loading Bryant pack once for scouting data
    pack = None
    try:
        from data.bryant_combined import load_bryant_combined_pack
        pack = load_bryant_combined_pack()
    except Exception:
        pass

    dav_pitching = gd[gd["PitcherTeam"] == DAVIDSON_TEAM_ID].copy()
    dav_hitting = gd[gd["BatterTeam"] == DAVIDSON_TEAM_ID].copy()

    with PdfPages(buf) as pdf:
        # Page 1: Cover
        try:
            fig = _render_cover(gd, game_label)
            if fig:
                pdf.savefig(fig)
                plt.close(fig)
                pages_saved += 1
        except Exception:
            traceback.print_exc()

        # SP Deep Dive: pitchers with >= 40 pitches (or longest-outing pitcher)
        if not dav_pitching.empty and "Pitcher" in dav_pitching.columns:
            pitchers = dav_pitching.groupby("Pitcher").size().sort_values(ascending=False)
            sp_pitchers = [p for p, n in pitchers.items() if n >= 40]

            # If no pitcher has >=40 pitches, use the longest-outing pitcher
            if not sp_pitchers and not pitchers.empty:
                sp_pitchers = [pitchers.index[0]]

            relievers = [p for p in pitchers.index if p not in sp_pitchers and pitchers[p] >= 15]

            # SP pages
            for pitcher in sp_pitchers:
                pitcher_pdf = dav_pitching[dav_pitching["Pitcher"] == pitcher].copy()
                try:
                    fig = _render_sp_call_grade_page(pitcher_pdf, data, pitcher, game_label)
                    if fig:
                        pdf.savefig(fig)
                        plt.close(fig)
                        pages_saved += 1
                except Exception:
                    traceback.print_exc()

                try:
                    fig = _render_sp_inning_progression(pitcher_pdf, game_label, pitcher)
                    if fig:
                        pdf.savefig(fig)
                        plt.close(fig)
                        pages_saved += 1
                except Exception:
                    traceback.print_exc()

                try:
                    fig = _render_sp_vs_opponent(pitcher_pdf, data, pitcher, game_label)
                    if fig:
                        pdf.savefig(fig)
                        plt.close(fig)
                        pages_saved += 1
                except Exception:
                    traceback.print_exc()

                # Per-batter AB review pages
                try:
                    ab_figs = _render_ab_review_pages(pitcher_pdf, game_label, pitcher,
                                                      pack=pack, role="pitcher")
                    for ab_fig in ab_figs:
                        pdf.savefig(ab_fig)
                        plt.close(ab_fig)
                        pages_saved += 1
                except Exception:
                    traceback.print_exc()

            # Reliever pages
            for pitcher in relievers:
                pitcher_pdf = dav_pitching[dav_pitching["Pitcher"] == pitcher].copy()
                try:
                    fig = _render_reliever_page(pitcher_pdf, data, pitcher, game_label, pack=pack)
                    if fig:
                        pdf.savefig(fig)
                        plt.close(fig)
                        pages_saved += 1
                except Exception:
                    traceback.print_exc()

        # Hitter AB Review pages
        if not dav_hitting.empty and "Batter" in dav_hitting.columns:
            batters = dav_hitting.groupby("Batter").size().sort_values(ascending=False).index.tolist()
            for batter in batters:
                batter_df = dav_hitting[dav_hitting["Batter"] == batter].copy()
                if len(batter_df) < 3:
                    continue
                try:
                    hitter_figs = _render_hitter_ab_review_pages(batter_df, data, batter, game_label)
                    for hfig in hitter_figs:
                        pdf.savefig(hfig)
                        plt.close(hfig)
                        pages_saved += 1
                except Exception:
                    traceback.print_exc()

    buf.seek(0)
    return buf.read()
