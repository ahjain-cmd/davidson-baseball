"""Pitching pages — Pitcher Card, Pitching Overview, Pitch Lab, Game Planning."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import percentileofscore

from config import (
    DAVIDSON_TEAM_ID, ROSTER_2026, JERSEY, POSITION, PITCH_COLORS,
    SWING_CALLS, CONTACT_CALLS, ZONE_SIDE, ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP,
    PLATE_SIDE_MAX, PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX, MIN_PITCH_USAGE_PCT,
    filter_davidson, filter_minor_pitches, normalize_pitch_types,
    in_zone_mask, is_barrel_mask, display_name, get_percentile,
)
from data.loader import get_all_seasons, _load_truemedia, _tm_player, _safe_val, _safe_pct, _safe_num, _tm_pctile
from data.stats import compute_pitcher_stats, _build_batter_zones
from data.population import compute_pitcher_stats_pop, compute_stuff_baselines
from viz.layout import CHART_LAYOUT, section_header
from viz.charts import (
    add_strike_zone, make_spray_chart, make_movement_profile,
    make_pitch_location_heatmap, player_header, _safe_pr, _safe_pop,
    _add_grid_zone_outline,
)
from viz.percentiles import savant_color, render_savant_percentile_section
from analytics.stuff_plus import _compute_stuff_plus, _compute_stuff_plus_all
from analytics.tunnel import _compute_tunnel_score, _build_tunnel_population, _load_tunnel_benchmarks
from analytics.command_plus import _compute_command_plus, _compute_pitch_pair_results
from analytics.expected import _create_zone_grid_data

# Import helpers that live in app.py
from config import safe_mode, _SWING_CALLS_SQL
from data.loader import query_population
from data.population import build_tunnel_population_pop


def _score_linear(val, lo, hi, invert=False):
    if pd.isna(val) or hi == lo:
        return np.nan
    if invert:
        val = hi - (val - lo)
        lo, hi = lo, hi
    return float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))


def _score_ev(ev):
    # Lower EV is better. Map ~80-95 to 100-0.
    return _score_linear(95 - ev, 0, 15)


def _score_whiff(wh):
    return _score_linear(wh, 0, 50)


def _score_k(k):
    return _score_linear(k, 0, 35)


def _score_stuff(stuff):
    return _score_linear(stuff, 70, 130)


def _score_cmd(cmd):
    return _score_linear(cmd, 80, 120)


def _weighted_score(parts, weights):
    vals = [(p, w) for p, w in zip(parts, weights) if pd.notna(p)]
    if not vals:
        return np.nan
    s = sum(p * w for p, w in vals)
    wsum = sum(w for _, w in vals)
    return s / wsum if wsum else np.nan


def _build_pitch_metric_map(pdf, stuff_df=None, cmd_df=None):
    stuff_map = {}
    if isinstance(stuff_df, pd.DataFrame) and not stuff_df.empty and "StuffPlus" in stuff_df.columns:
        stuff_map = stuff_df.groupby("TaggedPitchType")["StuffPlus"].mean().to_dict()
    cmd_map = {}
    if isinstance(cmd_df, pd.DataFrame) and not cmd_df.empty:
        cmd_map = dict(zip(cmd_df["Pitch"], cmd_df["Command+"]))
    return {pt: {"stuff": stuff_map.get(pt, np.nan), "cmd": cmd_map.get(pt, np.nan)}
            for pt in pdf["TaggedPitchType"].dropna().unique()}


def _pair_stats(pair_df, a, b):
    if not isinstance(pair_df, pd.DataFrame) or pair_df.empty:
        return {}
    sub = pair_df[((pair_df["Setup Pitch"] == a) & (pair_df["Follow Pitch"] == b)) |
                  ((pair_df["Setup Pitch"] == b) & (pair_df["Follow Pitch"] == a))].copy()
    if sub.empty:
        return {}
    sub["Count"] = pd.to_numeric(sub.get("Count"), errors="coerce").fillna(0)
    w = sub["Count"].where(sub["Count"] > 0, 1)
    count_ab = float(w[(sub["Setup Pitch"] == a) & (sub["Follow Pitch"] == b)].sum())
    count_ba = float(w[(sub["Setup Pitch"] == b) & (sub["Follow Pitch"] == a)].sum())
    def _wavg(col):
        vals = pd.to_numeric(sub.get(col), errors="coerce")
        if vals is None or vals.dropna().empty:
            return np.nan
        return float(np.average(vals.fillna(0), weights=w))
    return {
        "whiff": _wavg("Whiff%"),
        "k": _wavg("K%"),
        "ev": _wavg("Avg EV"),
        "putaway": _wavg("Putaway%"),
        "count": float(w.sum()),
        "count_ab": count_ab,
        "count_ba": count_ba,
    }


def _rank_pairs(tunnel_df, pair_df, pitch_metrics, top_n=2):
    """Rank pitch pairs using outcomes-first score; include same-pitch pairs."""
    if not isinstance(pair_df, pd.DataFrame) or pair_df.empty:
        return []
    # candidate pairs from pair_df (includes same-pitch)
    candidates = set()
    for _, r in pair_df.iterrows():
        candidates.add((r["Setup Pitch"], r["Follow Pitch"]))
    # add tunnel pairs
    if isinstance(tunnel_df, pd.DataFrame) and not tunnel_df.empty:
        for _, r in tunnel_df.iterrows():
            candidates.add((r["Pitch A"], r["Pitch B"]))

    tunnel_map = {}
    if isinstance(tunnel_df, pd.DataFrame) and not tunnel_df.empty:
        for _, r in tunnel_df.iterrows():
            key = tuple(sorted([r["Pitch A"], r["Pitch B"]]))
            tunnel_map[key] = pd.to_numeric(r.get("Tunnel Score"), errors="coerce")

    best_by_key = {}
    for a, b in candidates:
        if a not in pitch_metrics or b not in pitch_metrics:
            continue
        ps = _pair_stats(pair_df, a, b)
        pm_a = pitch_metrics.get(a, {})
        pm_b = pitch_metrics.get(b, {})
        stuff_avg = np.nanmean([pm_a.get("stuff", np.nan), pm_b.get("stuff", np.nan)])
        cmd_avg = np.nanmean([pm_a.get("cmd", np.nan), pm_b.get("cmd", np.nan)])
        tunnel = tunnel_map.get(tuple(sorted([a, b])), np.nan)
        whiff = ps.get("whiff", np.nan)
        k_pct = ps.get("k", np.nan)
        if pd.isna(k_pct):
            k_pct = ps.get("putaway", np.nan)
        ev = ps.get("ev", np.nan)
        count_ab = ps.get("count_ab", 0)
        count_ba = ps.get("count_ba", 0)
        if count_ab == count_ba:
            label_a, label_b = sorted([a, b])
        else:
            label_a, label_b = (a, b) if count_ab >= count_ba else (b, a)
        score = _weighted_score(
            [_score_whiff(whiff), _score_k(k_pct), _score_ev(ev), tunnel],
            [0.35, 0.25, 0.25, 0.15],
        )
        row = {
            "Pair": f"{label_a} → {label_b}",
            "Tunnel": tunnel,
            "Whiff%": whiff,
            "K%": k_pct,
            "Avg EV": ev,
            "Stuff+": stuff_avg,
            "Cmd+": cmd_avg,
            "Score": score,
            "Pairs": ps.get("count", np.nan),
            "_key": tuple(sorted([a, b])),
        }
        key = row["_key"]
        prev = best_by_key.get(key)
        if prev is None:
            best_by_key[key] = row
            continue
        prev_score = prev.get("Score", np.nan)
        if pd.isna(prev_score) or (pd.notna(score) and score > prev_score):
            best_by_key[key] = row
        elif pd.notna(score) and pd.isna(prev_score):
            best_by_key[key] = row

    out = list(best_by_key.values())
    out = sorted(out, key=lambda x: (x["Score"] if pd.notna(x["Score"]) else -1), reverse=True)
    for r in out:
        r.pop("_key", None)

    # Prioritize diverse pairs: ensure at least one different-pitch pair in top results
    # Split into different-pitch and same-pitch pairs
    different_pairs = [r for r in out if " → " in r["Pair"] and r["Pair"].split(" → ")[0] != r["Pair"].split(" → ")[1]]
    same_pairs = [r for r in out if " → " in r["Pair"] and r["Pair"].split(" → ")[0] == r["Pair"].split(" → ")[1]]

    # Build result: prioritize different-pitch pairs, but include same-pitch if score is significantly higher
    result = []
    if different_pairs:
        result.append(different_pairs[0])  # Best different-pitch pair
        # Add second best: either another different pair or a same pair if it scores much higher
        remaining_diff = different_pairs[1:] if len(different_pairs) > 1 else []
        if same_pairs and remaining_diff:
            # Include same pair only if it scores > 10 points higher than next different pair
            if same_pairs[0].get("Score", 0) > remaining_diff[0].get("Score", 0) + 10:
                result.append(same_pairs[0])
            else:
                result.append(remaining_diff[0])
        elif same_pairs:
            result.append(same_pairs[0])
        elif remaining_diff:
            result.append(remaining_diff[0])
    elif same_pairs:
        # Only same-pitch pairs available
        result = same_pairs[:top_n]

    return result[:top_n]


def _rank_sequences(pair_df, pitch_metrics, length=3, top_n=2):
    if not isinstance(pair_df, pd.DataFrame) or pair_df.empty:
        return []
    df = pair_df.copy()
    df["Count"] = pd.to_numeric(df.get("Count"), errors="coerce").fillna(0)
    df["Whiff%"] = pd.to_numeric(df.get("Whiff%"), errors="coerce")
    df["K%"] = pd.to_numeric(df.get("K%"), errors="coerce")
    df["Putaway%"] = pd.to_numeric(df.get("Putaway%"), errors="coerce")
    df["Avg EV"] = pd.to_numeric(df.get("Avg EV"), errors="coerce")
    df["Tunnel Score"] = pd.to_numeric(df.get("Tunnel Score"), errors="coerce")
    df = df.dropna(subset=["Tunnel Score"])
    if df.empty:
        return []

    pair_map = {}
    valid_pitches = set(pitch_metrics.keys())
    for _, r in df.iterrows():
        if r["Setup Pitch"] not in valid_pitches or r["Follow Pitch"] not in valid_pitches:
            continue
        pair_map[(r["Setup Pitch"], r["Follow Pitch"])] = {
            "whiff": r["Whiff%"],
            "k": r["K%"] if pd.notna(r["K%"]) else r["Putaway%"],
            "ev": r["Avg EV"],
            "tunnel": r["Tunnel Score"],
            "count": r["Count"],
        }

    out_map = {}
    for (a, b), stats in pair_map.items():
        out_map.setdefault(a, []).append((b, stats))

    def _wavg(vals, wts):
        mask = [pd.notna(v) for v in vals]
        if not any(mask):
            return np.nan
        v = [val for val, m in zip(vals, mask) if m]
        w = [wt for wt, m in zip(wts, mask) if m]
        return float(np.average(v, weights=w))

    def _seq_stats(pairs):
        counts = [max(p["count"], 1) for p in pairs]
        whiff_avg = _wavg([p["whiff"] for p in pairs], counts)
        k_avg = _wavg([p["k"] for p in pairs], counts)
        ev_avg = _wavg([p["ev"] for p in pairs], counts)
        tunnel_avg = _wavg([p["tunnel"] for p in pairs], counts)
        return whiff_avg, k_avg, ev_avg, tunnel_avg, int(np.sum(counts))

    results = []
    if length == 3:
        for a, outs in out_map.items():
            for b, s1 in outs:
                for c, s2 in out_map.get(b, []):
                    wh, k, ev, tn, n = _seq_stats([s1, s2])
                    pitches = [a, b, c]
                    stuff_avg = np.nanmean([pitch_metrics.get(p, {}).get("stuff", np.nan) for p in pitches])
                    cmd_avg = np.nanmean([pitch_metrics.get(p, {}).get("cmd", np.nan) for p in pitches])
                    score = _weighted_score(
                        [_score_whiff(wh), _score_k(k), _score_ev(ev), tn],
                        [0.35, 0.25, 0.25, 0.15],
                    )
                    results.append({"Seq": f"{a} → {b} → {c}", "Tunnel": tn, "Whiff%": wh,
                                    "K%": k, "Avg EV": ev, "Stuff+": stuff_avg,
                                    "Cmd+": cmd_avg, "Score": score, "Pairs": n})
    elif length == 4:
        for a, outs in out_map.items():
            for b, s1 in outs:
                for c, s2 in out_map.get(b, []):
                    for d, s3 in out_map.get(c, []):
                        wh, k, ev, tn, n = _seq_stats([s1, s2, s3])
                        pitches = [a, b, c, d]
                        stuff_avg = np.nanmean([pitch_metrics.get(p, {}).get("stuff", np.nan) for p in pitches])
                        cmd_avg = np.nanmean([pitch_metrics.get(p, {}).get("cmd", np.nan) for p in pitches])
                        score = _weighted_score(
                            [_score_whiff(wh), _score_k(k), _score_ev(ev), tn],
                            [0.35, 0.25, 0.25, 0.15],
                        )
                        results.append({"Seq": f"{a} → {b} → {c} → {d}", "Tunnel": tn, "Whiff%": wh,
                                        "K%": k, "Avg EV": ev, "Stuff+": stuff_avg,
                                        "Cmd+": cmd_avg, "Score": score, "Pairs": n})
    else:
        return []
    results.sort(key=lambda x: (x["Score"] if pd.notna(x["Score"]) else -1), reverse=True)
    return results[:top_n]


def _filter_redundant_sequences(seqs, min_unique=3, max_keep=2):
    """Drop sequences that are just repeats of the same 2 pitches unless needed."""
    if not seqs:
        return []
    def _uniq_count(seq):
        pitches = [p.strip() for p in seq.split("→")]
        return len(set(pitches)), tuple(sorted(set(pitches)))
    filtered = []
    seen_sets = set()
    for s in seqs:
        uniq_cnt, uniq_set = _uniq_count(s["Seq"])
        if uniq_cnt < min_unique:
            continue
        if uniq_set in seen_sets:
            continue
        seen_sets.add(uniq_set)
        filtered.append(s)
        if len(filtered) >= max_keep:
            return filtered
    if filtered:
        return filtered
    # Fallback: allow 2-pitch loops if nothing else exists, still de-duplicate
    for s in seqs:
        uniq_cnt, uniq_set = _uniq_count(s["Seq"])
        if uniq_cnt < 2:
            continue
        if uniq_set in seen_sets:
            continue
        seen_sets.add(uniq_set)
        filtered.append(s)
        if len(filtered) >= max_keep:
            break
    return filtered


def _deception_flag(tunnel):
    if pd.isna(tunnel):
        return ""
    if tunnel >= 60:
        return f"+ Deception edge (Tunnel {tunnel:.0f})"
    if tunnel <= 40:
        return f"- Deception weak (Tunnel {tunnel:.0f})"
    return ""


def _assign_tactical_tags(rows):
    if not rows:
        return rows
    whiffs = [r.get("Whiff%") for r in rows]
    ks = [r.get("K%") for r in rows]
    evs = [r.get("Avg EV") for r in rows]

    def _best_idx(vals, func=max):
        vals_clean = [v for v in vals if pd.notna(v)]
        if not vals_clean:
            return None
        target = func(vals_clean)
        for i, v in enumerate(vals):
            if pd.notna(v) and v == target:
                return i
        return None

    idx_put = _best_idx(ks, max)
    idx_ev = _best_idx(evs, min)
    idx_wh = _best_idx(whiffs, max)

    tags = {}
    for idx, label in [(idx_put, "Best putaway"), (idx_ev, "Best weak‑contact"), (idx_wh, "Best whiff")]:
        if idx is not None and idx not in tags:
            tags[idx] = label
    for i in range(len(rows)):
        if i not in tags:
            tags[i] = "Best overall"
        rows[i]["Tag"] = tags[i]
    return rows


def _pitching_overview(data, pitcher, season_filter, pdf, pdf_raw, pr, all_pitcher_stats):
    """Content from the original Pitcher Card, rendered inside the Overview tab."""
    all_stats = all_pitcher_stats  # alias for brevity
    total_pitches = len(pdf)

    # ── ROW 1: Percentile Rankings + Movement Profile ──
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        p_metrics = [
            ("FB Velo", _safe_pr(pr, "AvgFBVelo"), get_percentile(_safe_pr(pr, "AvgFBVelo"), _safe_pop(all_stats, "AvgFBVelo")), ".1f", True),
            ("Avg EV Against", _safe_pr(pr, "AvgEVAgainst"), get_percentile(_safe_pr(pr, "AvgEVAgainst"), _safe_pop(all_stats, "AvgEVAgainst")), ".1f", False),
            ("Chase %", _safe_pr(pr, "ChasePct"), get_percentile(_safe_pr(pr, "ChasePct"), _safe_pop(all_stats, "ChasePct")), ".1f", True),
            ("Whiff %", _safe_pr(pr, "WhiffPct"), get_percentile(_safe_pr(pr, "WhiffPct"), _safe_pop(all_stats, "WhiffPct")), ".1f", True),
            ("K %", _safe_pr(pr, "KPct"), get_percentile(_safe_pr(pr, "KPct"), _safe_pop(all_stats, "KPct")), ".1f", True),
            ("BB %", _safe_pr(pr, "BBPct"), get_percentile(_safe_pr(pr, "BBPct"), _safe_pop(all_stats, "BBPct")), ".1f", False),
            ("Barrel %", _safe_pr(pr, "BarrelPctAgainst"), get_percentile(_safe_pr(pr, "BarrelPctAgainst"), _safe_pop(all_stats, "BarrelPctAgainst")), ".1f", False),
            ("Hard Hit %", _safe_pr(pr, "HardHitAgainst"), get_percentile(_safe_pr(pr, "HardHitAgainst"), _safe_pop(all_stats, "HardHitAgainst")), ".1f", False),
            ("GB %", _safe_pr(pr, "GBPct"), get_percentile(_safe_pr(pr, "GBPct"), _safe_pop(all_stats, "GBPct")), ".1f", True),
            ("Extension", _safe_pr(pr, "Extension"), get_percentile(_safe_pr(pr, "Extension"), _safe_pop(all_stats, "Extension")), ".1f", True),
        ]
        render_savant_percentile_section(p_metrics, "Percentile Rankings")
        st.caption(f"vs. {len(all_stats)} pitchers in database (min 100 pitches)")

    with col2:
        section_header("Movement Profile (Induced Break)")
        fig = make_movement_profile(pdf, height=500)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="pitcher_movement")
        else:
            st.info("No movement data.")

    # ── ROW 2: Pitch Arsenal Table ──
    st.markdown("---")
    section_header("Pitch Arsenal")
    arsenal_rows = []
    main_pitches = sorted(pdf["TaggedPitchType"].dropna().unique())
    for pt in main_pitches:
        sub = pdf[pdf["TaggedPitchType"] == pt]
        n = len(sub)
        sub_swings = sub[sub["PitchCall"].isin(SWING_CALLS)]
        sub_whiffs = sub[sub["PitchCall"] == "StrikeSwinging"]
        sub_ip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
        row = {
            "Pitch": pt,
            "#": n,
            "Use%": round(n / total_pitches * 100, 1),
            "Velo": round(sub["RelSpeed"].mean(), 1) if sub["RelSpeed"].notna().any() else None,
            "Max": round(sub["RelSpeed"].max(), 1) if sub["RelSpeed"].notna().any() else None,
            "Spin": int(round(sub["SpinRate"].mean())) if sub["SpinRate"].notna().any() else None,
            "IVB": round(sub["InducedVertBreak"].mean(), 1) if sub["InducedVertBreak"].notna().any() else None,
            "HB": round(sub["HorzBreak"].mean(), 1) if sub["HorzBreak"].notna().any() else None,
            "Ext.": round(sub["Extension"].mean(), 1) if sub["Extension"].notna().any() else None,
            "Whiff%": round(len(sub_whiffs) / max(len(sub_swings), 1) * 100, 1),
            "EV Ag": round(sub_ip["ExitSpeed"].mean(), 1) if len(sub_ip) > 0 else None,
        }
        if "Tilt" in sub.columns and sub["Tilt"].notna().any():
            row["Tilt"] = safe_mode(sub["Tilt"], None)
        if "ZoneSpeed" in sub.columns and sub["ZoneSpeed"].notna().any():
            row["Zone Velo"] = round(sub["ZoneSpeed"].mean(), 1)
        if "VertApprAngle" in sub.columns and sub["VertApprAngle"].notna().any():
            row["VAA"] = round(sub["VertApprAngle"].mean(), 1)
        arsenal_rows.append(row)
    if arsenal_rows:
        st.dataframe(pd.DataFrame(arsenal_rows), use_container_width=True, hide_index=True)

    # Arsenal summary text
    arsenal_summary = ", ".join(
        f"{r['Pitch']} ({r['Use%']}%)" for r in arsenal_rows
    )
    st.markdown(
        f'<p style="font-size:13px;color:#555;margin-top:4px;">'
        f'{display_name(pitcher)} relies on {len(main_pitches)} pitches: {arsenal_summary}</p>',
        unsafe_allow_html=True,
    )

    # ── ROW 3: Per-Pitch Location Heatmaps (Savant style) ──
    st.markdown("---")
    section_header("Pitch Locations by Type")

    # Arrange pitch heatmaps in a grid (up to 4 per row)
    n_pitches = len(main_pitches)
    cols_per_row = min(n_pitches, 4)
    for row_start in range(0, n_pitches, cols_per_row):
        row_pitches = main_pitches[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_pitches), gap="small")
        for i, pt in enumerate(row_pitches):
            with cols[i]:
                sub = pdf[pdf["TaggedPitchType"] == pt]
                n = len(sub)
                color = PITCH_COLORS.get(pt, "#aaa")
                st.markdown(
                    f'<div style="font-size:14px;font-weight:700;color:{color} !important;">{pt}</div>'
                    f'<div style="font-size:11px;color:#888;">{n} Pitches ({n / total_pitches * 100:.1f}%)</div>',
                    unsafe_allow_html=True,
                )
                fig = make_pitch_location_heatmap(sub, pt, color, height=320)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"loc_{pt}_{row_start}")

    # ── ROW 4: Velocity Distribution + Usage Chart ──
    st.markdown("---")
    col3, col4 = st.columns([1, 1], gap="medium")

    with col3:
        section_header("Velocity Distribution")
        velo_data = pdf.dropna(subset=["RelSpeed"])
        if not velo_data.empty:
            fig = go.Figure()
            for pt in main_pitches:
                sub = velo_data[velo_data["TaggedPitchType"] == pt]
                if len(sub) < 5:
                    continue
                fig.add_trace(go.Violin(
                    y=sub["RelSpeed"], name=pt,
                    line_color=PITCH_COLORS.get(pt, "#aaa"),
                    fillcolor=PITCH_COLORS.get(pt, "#aaa"),
                    opacity=0.6, meanline_visible=True,
                    box_visible=True, box_fillcolor="white",
                ))
            fig.update_layout(
                yaxis_title="Velocity (mph)", showlegend=False,
                height=450, **CHART_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, key="pitcher_velo_dist")

    with col4:
        section_header("Pitch Usage %")
        if arsenal_rows:
            usage_df = pd.DataFrame(arsenal_rows).sort_values("Use%", ascending=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=usage_df["Pitch"], x=usage_df["Use%"],
                orientation="h",
                marker_color=[PITCH_COLORS.get(p, "#aaa") for p in usage_df["Pitch"]],
                text=[f'{v}%' for v in usage_df["Use%"]],
                textposition="outside",
                textfont=dict(color="#000000", size=12, family="Inter"),
            ))
            fig.update_layout(
                xaxis_title="Usage %", yaxis_title="",
                height=450, showlegend=False,
                **CHART_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, key="pitcher_usage")

    # ── ROW 5: Plate Discipline + Whiff by Type ──
    st.markdown("---")
    col5, col6 = st.columns([1, 1], gap="medium")

    with col5:
        section_header("Plate Discipline")
        disc_df = pd.DataFrame([{
            "Pitches": int(_safe_pr(pr, "Pitches") or 0),
            "Zone%": round(_safe_pr(pr, "ZonePct"), 1) if not pd.isna(_safe_pr(pr, "ZonePct")) else None,
            "Chase%": round(_safe_pr(pr, "ChasePct"), 1) if not pd.isna(_safe_pr(pr, "ChasePct")) else None,
            "Whiff%": round(_safe_pr(pr, "WhiffPct") or 0, 1),
            "Z-Contact%": round(_safe_pr(pr, "ZoneContactPct"), 1) if not pd.isna(_safe_pr(pr, "ZoneContactPct")) else None,
            "Swing%": round(_safe_pr(pr, "SwingPct") or 0, 1),
            "K%": round(_safe_pr(pr, "KPct") or 0, 1),
            "BB%": round(_safe_pr(pr, "BBPct") or 0, 1),
        }])
        st.dataframe(disc_df, use_container_width=True, hide_index=True)

    with col6:
        section_header("Whiff% by Pitch Type")
        if arsenal_rows:
            whiff_df = pd.DataFrame(arsenal_rows)[["Pitch", "Whiff%"]].sort_values("Whiff%", ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=whiff_df["Pitch"], y=whiff_df["Whiff%"],
                marker_color=[PITCH_COLORS.get(p, "#aaa") for p in whiff_df["Pitch"]],
                text=[f'{v}%' for v in whiff_df["Whiff%"]],
                textposition="outside",
                textfont=dict(color="#000000", size=12, family="Inter"),
            ))
            fig.update_layout(
                yaxis_title="Whiff %", height=350, showlegend=False, **CHART_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, key="pitcher_whiff_by_type")

    # ── ROW 6: Velocity + Spin Over Time (full width, one per row) ──
    st.markdown("---")
    section_header("Velocity Over Time")
    fb_data = pdf.dropna(subset=["RelSpeed", "Date"])
    if not fb_data.empty:
        daily = fb_data.groupby(["Date", "TaggedPitchType"])["RelSpeed"].mean().reset_index()
        fig = px.line(daily, x="Date", y="RelSpeed", color="TaggedPitchType",
                      color_discrete_map=PITCH_COLORS, markers=True,
                      labels={"RelSpeed": "Velo (mph)", "Date": ""})
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                        font=dict(size=11, color="#000000")),
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True, key="pitcher_velo_time")

    section_header("Spin Rate Over Time")
    spin_data = pdf.dropna(subset=["SpinRate", "Date"])
    if not spin_data.empty:
        daily_spin = spin_data.groupby(["Date", "TaggedPitchType"])["SpinRate"].mean().reset_index()
        fig = px.line(daily_spin, x="Date", y="SpinRate", color="TaggedPitchType",
                      color_discrete_map=PITCH_COLORS, markers=True,
                      labels={"SpinRate": "Spin (rpm)", "Date": ""})
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                        font=dict(size=11, color="#000000")),
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True, key="pitcher_spin_time")

    # ── ROW 7: Release Point Consistency ──
    st.markdown("---")
    section_header("Release Point Consistency")
    rp_data = pdf.dropna(subset=["RelHeight", "RelSide"])
    if len(rp_data) >= 10:
        col_rp1, col_rp2 = st.columns([1, 1], gap="medium")
        with col_rp1:
            fig = go.Figure()
            for pt in main_pitches:
                sub = rp_data[rp_data["TaggedPitchType"] == pt]
                if len(sub) < 3:
                    continue
                fig.add_trace(go.Scatter(
                    x=sub["RelSide"], y=sub["RelHeight"], mode="markers",
                    marker=dict(size=5, color=PITCH_COLORS.get(pt, "#aaa"), opacity=0.6,
                                line=dict(width=0.3, color="white")),
                    name=pt,
                    hovertemplate=f"{pt}<br>Side: %{{x:.2f}} ft<br>Height: %{{y:.2f}} ft<extra></extra>",
                ))
                # Add mean crosshair
                fig.add_trace(go.Scatter(
                    x=[sub["RelSide"].mean()], y=[sub["RelHeight"].mean()], mode="markers",
                    marker=dict(size=14, color=PITCH_COLORS.get(pt, "#aaa"), symbol="x-thin",
                                line=dict(width=3, color=PITCH_COLORS.get(pt, "#aaa"))),
                    showlegend=False, hoverinfo="skip",
                ))
            fig.update_layout(
                xaxis_title="Release Side (ft)", yaxis_title="Release Height (ft)",
                height=400,
                xaxis=dict(scaleanchor="y"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                            font=dict(size=10, color="#000000")),
                **CHART_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, key="pitcher_rp")

        with col_rp2:
            # Release point consistency stats
            rp_rows = []
            for pt in main_pitches:
                sub = rp_data[rp_data["TaggedPitchType"] == pt]
                if len(sub) < 5:
                    continue
                rp_rows.append({
                    "Pitch": pt,
                    "Avg Height": round(sub["RelHeight"].mean(), 2),
                    "Std Height": round(sub["RelHeight"].std(), 3),
                    "Avg Side": round(sub["RelSide"].mean(), 2),
                    "Std Side": round(sub["RelSide"].std(), 3),
                })
            if rp_rows:
                st.dataframe(pd.DataFrame(rp_rows), use_container_width=True, hide_index=True)
                # Insight
                best_rp = min(rp_rows, key=lambda r: r["Std Height"] + r["Std Side"])
                worst_rp = max(rp_rows, key=lambda r: r["Std Height"] + r["Std Side"])
                st.markdown(
                    f'<p style="font-size:12px;color:#555;">'
                    f'Most consistent release: <b style="color:{PITCH_COLORS.get(best_rp["Pitch"], "#333")}">'
                    f'{best_rp["Pitch"]}</b> (std: {best_rp["Std Height"]:.3f}H / {best_rp["Std Side"]:.3f}S). '
                    f'Least consistent: <b style="color:{PITCH_COLORS.get(worst_rp["Pitch"], "#333")}">'
                    f'{worst_rp["Pitch"]}</b> (std: {worst_rp["Std Height"]:.3f}H / {worst_rp["Std Side"]:.3f}S).</p>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Not enough release point data.")

    # ── ROW 8: Approach Angle (VAA) Analysis ──
    st.markdown("---")
    section_header("Vertical Approach Angle (VAA)")
    vaa_data = pdf.dropna(subset=["VertApprAngle"])
    if len(vaa_data) >= 10:
        col_vaa1, col_vaa2 = st.columns([1, 1], gap="medium")
        with col_vaa1:
            fig = go.Figure()
            for pt in main_pitches:
                sub = vaa_data[vaa_data["TaggedPitchType"] == pt]
                if len(sub) < 5:
                    continue
                fig.add_trace(go.Violin(
                    y=sub["VertApprAngle"], name=pt,
                    line_color=PITCH_COLORS.get(pt, "#aaa"),
                    fillcolor=PITCH_COLORS.get(pt, "#aaa"),
                    opacity=0.6, meanline_visible=True,
                    box_visible=True, box_fillcolor="white",
                ))
            fig.update_layout(
                yaxis_title="VAA (degrees)", showlegend=False,
                height=380, **CHART_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, key="pitcher_vaa")

        with col_vaa2:
            # VAA percentile context
            vaa_rows = []
            all_pitchers_data = data.dropna(subset=["VertApprAngle"])
            for pt in main_pitches:
                sub = vaa_data[vaa_data["TaggedPitchType"] == pt]
                if len(sub) < 5:
                    continue
                p_vaa = sub["VertApprAngle"].mean()
                # Get all pitchers' avg VAA for this pitch type
                all_pt = all_pitchers_data[all_pitchers_data["TaggedPitchType"] == pt]
                pitcher_avgs = all_pt.groupby("Pitcher")["VertApprAngle"].mean()
                pitcher_avgs = pitcher_avgs[all_pt.groupby("Pitcher").size() >= 20]
                pct = get_percentile(p_vaa, pitcher_avgs) if len(pitcher_avgs) > 0 else np.nan
                # For fastballs, flatter (less negative) VAA is better; for breaking, steeper is better
                is_fb = pt in ["Fastball", "Sinker", "Cutter"]
                vaa_rows.append((pt, p_vaa, pct, ".1f", is_fb))
            if vaa_rows:
                render_savant_percentile_section(vaa_rows, "VAA Percentile by Pitch")
                st.caption("Fastballs: flatter VAA is elite. Breaking: steeper is better.")
    else:
        st.info("No VAA data available.")

    # ── ROW 9: Platoon Splits (vs LHH / RHH) ──
    st.markdown("---")
    section_header("Platoon Splits (vs LHH / RHH)")
    if "BatterSide" in pdf.columns:
        platoon_metrics = []
        for side, label in [("Right", "vs RHH"), ("Left", "vs LHH")]:
            p_sub = pdf_raw[pdf_raw["BatterSide"] == side]
            if len(p_sub) < 10:
                continue
            p_sw = p_sub[p_sub["PitchCall"].isin(SWING_CALLS)]
            p_wh = p_sub[p_sub["PitchCall"] == "StrikeSwinging"]
            p_ip = p_sub[p_sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            p_whiff = len(p_wh) / max(len(p_sw), 1) * 100 if len(p_sw) > 0 else np.nan
            p_ev = p_ip["ExitSpeed"].mean() if len(p_ip) > 0 else np.nan
            # Get all pitchers' whiff/ev vs this side (via DuckDB)
            all_df = query_population(f"""
                SELECT
                    Pitcher,
                    CASE WHEN SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) > 0
                        THEN SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) * 100.0
                             / SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END)
                        ELSE NULL END AS whiff,
                    AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS ev
                FROM trackman
                WHERE BatterSide = '{side}'
                GROUP BY Pitcher
                HAVING COUNT(*) >= 20
            """)
            whiff_pct = get_percentile(p_whiff, all_df["whiff"]) if not all_df.empty and not pd.isna(p_whiff) else np.nan
            ev_pct = get_percentile(p_ev, all_df["ev"]) if not all_df.empty and not pd.isna(p_ev) else np.nan
            platoon_metrics.append((f"{label} Whiff%", p_whiff, whiff_pct, ".1f", True))
            if not pd.isna(ev_pct):
                platoon_metrics.append((f"{label} EV Against", p_ev, ev_pct, ".1f", False))

        if platoon_metrics:
            render_savant_percentile_section(platoon_metrics, None)
            st.caption("Percentile vs. all pitchers in DB (min 100 pitches vs that side)")

            # Usage breakdown by side
            col_p1, col_p2 = st.columns(2, gap="medium")
            for i, (side, label) in enumerate([("Right", "vs RHH"), ("Left", "vs LHH")]):
                p_sub = pdf_raw[pdf_raw["BatterSide"] == side]
                if len(p_sub) < 10:
                    continue
                usage = p_sub["TaggedPitchType"].value_counts(normalize=True).mul(100).round(1)
                with [col_p1, col_p2][i]:
                    st.markdown(f"**Pitch Usage {label}** ({len(p_sub)} pitches)")
                    for pt, pct_val in usage.items():
                        color = PITCH_COLORS.get(pt, "#aaa")
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0;">'
                            f'<div style="width:12px;height:12px;border-radius:50%;background:{color};"></div>'
                            f'<div style="font-size:12px;color:#1a1a2e;">{pt}: <b>{pct_val:.1f}%</b></div>'
                            f'</div>', unsafe_allow_html=True,
                        )
    else:
        st.caption("No batter side data available.")

    # ── PitcherSet Splits (Stretch vs Windup) ──
    if "PitcherSet" in pdf_raw.columns and pdf_raw["PitcherSet"].notna().any():
        sets = pdf_raw["PitcherSet"].dropna().unique()
        if len(sets) > 1:
            st.markdown("---")
            section_header("Set Position Splits (Windup vs Stretch)")
            set_metrics = []
            for s_val in sorted(sets):
                sub = pdf_raw[pdf_raw["PitcherSet"] == s_val]
                if len(sub) < 10:
                    continue
                sw = sub[sub["PitchCall"].isin(SWING_CALLS)]
                wh = sub[sub["PitchCall"] == "StrikeSwinging"]
                fb = sub[sub["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter"])]
                fb_velo = fb["RelSpeed"].mean() if len(fb) > 0 and fb["RelSpeed"].notna().any() else np.nan
                whiff = len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else np.nan
                # Context vs all pitchers in that set (via DuckDB)
                all_df = query_population(f"""
                    SELECT
                        Pitcher,
                        CASE WHEN SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) > 0
                            THEN SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) * 100.0
                                 / SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END)
                            ELSE NULL END AS whiff,
                        AVG(CASE WHEN TaggedPitchType IN ('Fastball','Sinker','Cutter') AND RelSpeed IS NOT NULL
                            THEN RelSpeed END) AS velo
                    FROM trackman
                    WHERE PitcherSet = '{s_val}'
                    GROUP BY Pitcher
                    HAVING COUNT(*) >= 20
                """)
                w_pct = get_percentile(whiff, all_df["whiff"]) if not all_df.empty and not pd.isna(whiff) else np.nan
                v_pct = get_percentile(fb_velo, all_df["velo"]) if not all_df.empty and not pd.isna(fb_velo) else np.nan
                set_metrics.append((f"{s_val} Whiff%", whiff, w_pct, ".1f", True))
                if not pd.isna(v_pct):
                    set_metrics.append((f"{s_val} FB Velo", fb_velo, v_pct, ".1f", True))
            if set_metrics:
                render_savant_percentile_section(set_metrics, None)

    # ── ROW 10: Pitch Sequencing ──
    st.markdown("---")
    section_header("Pitch Sequencing Patterns")
    if "PitchofPA" in pdf_raw.columns and "TaggedPitchType" in pdf_raw.columns:
        seq_data = pdf_raw.dropna(subset=["PitchofPA", "TaggedPitchType"]).copy()
        # First pitch tendencies
        first_pitches = seq_data[seq_data["PitchofPA"] == 1]
        if len(first_pitches) >= 5:
            fp_usage = first_pitches["TaggedPitchType"].value_counts(normalize=True).mul(100).round(1)
            col_s1, col_s2 = st.columns([1, 1], gap="medium")
            with col_s1:
                st.markdown("**First Pitch Tendencies**")
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=fp_usage.index, values=fp_usage.values,
                    marker_colors=[PITCH_COLORS.get(p, "#aaa") for p in fp_usage.index],
                    hole=0.4, textinfo="label+percent",
                    textfont=dict(size=11, color="#000000"),
                ))
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                  plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key="pitcher_fp_pie")

                # First pitch strike%
                fp_strikes = first_pitches[first_pitches["PitchCall"].isin(
                    ["StrikeCalled", "StrikeSwinging", "FoulBall", "FoulBallNotFieldable",
                     "FoulBallFieldable", "InPlay"])]
                fp_strike_pct = len(fp_strikes) / max(len(first_pitches), 1) * 100
                # Context: all pitchers' first pitch strike% (via DuckDB)
                _fp_df = query_population(f"""
                    SELECT Pitcher,
                        SUM(CASE WHEN PitchCall IN ('StrikeCalled','StrikeSwinging','FoulBall',
                            'FoulBallNotFieldable','FoulBallFieldable','InPlay') THEN 1 ELSE 0 END) * 100.0
                            / GREATEST(COUNT(*), 1) AS fp_strike
                    FROM trackman
                    WHERE PitchofPA = 1
                    GROUP BY Pitcher
                    HAVING COUNT(*) >= 10
                """)
                fp_pct = get_percentile(fp_strike_pct, _fp_df["fp_strike"]) if not _fp_df.empty else 50
                render_savant_percentile_section(
                    [("1st Pitch Strike%", fp_strike_pct, fp_pct, ".1f", True)], None,
                )

            with col_s2:
                # Two-strike approach
                two_strike = seq_data[(seq_data["Strikes"] == 2)]
                if len(two_strike) >= 10:
                    st.markdown("**Two-Strike Approach**")
                    ts_usage = two_strike["TaggedPitchType"].value_counts(normalize=True).mul(100).round(1)
                    fig = go.Figure()
                    fig.add_trace(go.Pie(
                        labels=ts_usage.index, values=ts_usage.values,
                        marker_colors=[PITCH_COLORS.get(p, "#aaa") for p in ts_usage.index],
                        hole=0.4, textinfo="label+percent",
                        textfont=dict(size=11, color="#000000"),
                    ))
                    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                      plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, key="pitcher_ts_pie")

                    ts_whiffs = two_strike[two_strike["PitchCall"] == "StrikeSwinging"]
                    ts_sw = two_strike[two_strike["PitchCall"].isin(SWING_CALLS)]
                    ts_whiff_pct = len(ts_whiffs) / max(len(ts_sw), 1) * 100
                    st.markdown(
                        f'<p style="font-size:12px;color:#555;">Two-strike whiff rate: '
                        f'<b>{ts_whiff_pct:.1f}%</b> on {len(ts_sw)} swings</p>',
                        unsafe_allow_html=True,
                    )




def _pitcher_card_content(data, pitcher, season_filter, pdf, stuff_df, pr, all_pitcher_stats, cmd_df=None):
    """Render a single-page Pitcher Card with arsenal, locations, tunnels,
    sequences, and platoon splits."""
    all_stats = all_pitcher_stats
    total_pitches = len(pdf)

    # ── Percentile Rankings + Arsenal Table side by side ──
    pc_col1, pc_col2 = st.columns([1, 1], gap="medium")
    with pc_col1:
        if pr is None or all_stats is None or all_stats.empty:
            st.info("Population percentiles unavailable for this pitcher.")
        else:
            p_metrics = [
                ("FB Velo", _safe_pr(pr, "AvgFBVelo"), get_percentile(_safe_pr(pr, "AvgFBVelo"), _safe_pop(all_stats, "AvgFBVelo")), ".1f", True),
                ("Avg EV Against", _safe_pr(pr, "AvgEVAgainst"), get_percentile(_safe_pr(pr, "AvgEVAgainst"), _safe_pop(all_stats, "AvgEVAgainst")), ".1f", False),
                ("Chase %", _safe_pr(pr, "ChasePct"), get_percentile(_safe_pr(pr, "ChasePct"), _safe_pop(all_stats, "ChasePct")), ".1f", True),
                ("Whiff %", _safe_pr(pr, "WhiffPct"), get_percentile(_safe_pr(pr, "WhiffPct"), _safe_pop(all_stats, "WhiffPct")), ".1f", True),
                ("K %", _safe_pr(pr, "KPct"), get_percentile(_safe_pr(pr, "KPct"), _safe_pop(all_stats, "KPct")), ".1f", True),
                ("BB %", _safe_pr(pr, "BBPct"), get_percentile(_safe_pr(pr, "BBPct"), _safe_pop(all_stats, "BBPct")), ".1f", False),
                ("Barrel %", _safe_pr(pr, "BarrelPctAgainst"), get_percentile(_safe_pr(pr, "BarrelPctAgainst"), _safe_pop(all_stats, "BarrelPctAgainst")), ".1f", False),
                ("Hard Hit %", _safe_pr(pr, "HardHitAgainst"), get_percentile(_safe_pr(pr, "HardHitAgainst"), _safe_pop(all_stats, "HardHitAgainst")), ".1f", False),
                ("GB %", _safe_pr(pr, "GBPct"), get_percentile(_safe_pr(pr, "GBPct"), _safe_pop(all_stats, "GBPct")), ".1f", True),
                ("Extension", _safe_pr(pr, "Extension"), get_percentile(_safe_pr(pr, "Extension"), _safe_pop(all_stats, "Extension")), ".1f", True),
            ]
            render_savant_percentile_section(p_metrics, "Percentile Rankings")
            st.caption(f"vs. {len(all_stats)} pitchers in database (min 100 pitches)")

    with pc_col2:
        section_header("Movement Profile (Induced Break)")
        fig_mov = make_movement_profile(pdf, height=420)
        if fig_mov:
            st.plotly_chart(fig_mov, use_container_width=True, key="pc_movement")
        else:
            st.info("No movement data.")

        # Usage & Velo summary underneath
        main_pitches_sorted = pdf["TaggedPitchType"].value_counts().index.tolist()
        uv_cols = st.columns(min(len(main_pitches_sorted), 5))
        for idx, pt in enumerate(main_pitches_sorted[:5]):
            with uv_cols[idx]:
                sub = pdf[pdf["TaggedPitchType"] == pt]
                usage = len(sub) / total_pitches * 100
                velo = sub["RelSpeed"].mean()
                pc = PITCH_COLORS.get(pt, "#888")
                velo_str = f"{velo:.1f}" if not pd.isna(velo) else "-"
                st.markdown(
                    f'<div style="text-align:center;padding:6px 4px;border-radius:6px;'
                    f'border-top:3px solid {pc};background:#f9f9f9;">'
                    f'<div style="font-weight:bold;font-size:13px;color:{pc};">{pt}</div>'
                    f'<div style="font-size:12px;color:#555;">{velo_str} mph</div>'
                    f'<div style="font-size:11px;color:#888;">{usage:.0f}%</div>'
                    f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Section A: Stuff+ & Command+ Grades ──
    section_header("Stuff+ & Command+ Grades")
    has_stuff = stuff_df is not None and "StuffPlus" in stuff_df.columns and not stuff_df.empty
    if has_stuff:
        arsenal_agg = stuff_df.groupby("TaggedPitchType").agg(
            stuff_mean=("StuffPlus", "mean"),
            velo_mean=("RelSpeed", "mean"),
            count=("StuffPlus", "count"),
        )
        arsenal_agg["Usage%"] = (arsenal_agg["count"] / arsenal_agg["count"].sum() * 100).round(1)
        arsenal_agg = arsenal_agg.sort_values("Usage%", ascending=False)
    else:
        # Fallback: derive from pdf directly
        pt_counts = pdf["TaggedPitchType"].value_counts()
        arsenal_agg = pd.DataFrame({
            "stuff_mean": np.nan,
            "velo_mean": pdf.groupby("TaggedPitchType")["RelSpeed"].mean(),
            "count": pt_counts,
            "Usage%": (pt_counts / pt_counts.sum() * 100).round(1),
        }).sort_values("Usage%", ascending=False)

    if cmd_df is None:
        cmd_df = _compute_command_plus(pdf, data)
    cmd_map = {}
    if not cmd_df.empty:
        cmd_map = dict(zip(cmd_df["Pitch"], cmd_df["Command+"]))

    pitch_types = arsenal_agg.index.tolist()
    n_pitches = len(pitch_types)

    # Stuff+ percentile bars
    if has_stuff:
        all_stuff = _compute_stuff_plus_all(data)
        if "StuffPlus" in all_stuff.columns:
            stuff_metrics = []
            for pt in pitch_types:
                my_val = stuff_df[stuff_df["TaggedPitchType"] == pt]["StuffPlus"].mean()
                all_pt_vals = all_stuff[all_stuff["TaggedPitchType"] == pt]["StuffPlus"]
                if len(all_pt_vals) > 5:
                    pctl = percentileofscore(all_pt_vals.dropna(), my_val, kind="rank")
                    stuff_metrics.append((pt, my_val, pctl, ".0f", True))
            if stuff_metrics:
                render_savant_percentile_section(stuff_metrics,
                                                 title="Stuff+ Percentile Rankings")

    # Command+ percentile bars
    if not cmd_df.empty:
        cmd_metrics = []
        for _, row in cmd_df.iterrows():
            cmd_val = row["Command+"]
            pctl_mapped = min(max((cmd_val - 80) * 2.5, 0), 100)
            cmd_metrics.append((row["Pitch"], cmd_val, pctl_mapped, ".0f", True))
        if cmd_metrics:
            render_savant_percentile_section(cmd_metrics,
                                             title="Command+ Percentile Rankings")

    # ── Section B: Best Pitch Locations (whiffs, called strikes, weak contact) ──
    st.markdown("")
    section_header("Best Pitch Locations")
    sel_pt = st.selectbox("Select Pitch", pitch_types, key="pc_loc_pitch")
    pt_data = pdf[pdf["TaggedPitchType"] == sel_pt].dropna(subset=["PlateLocSide", "PlateLocHeight"])

    whiff_data = pt_data[pt_data["PitchCall"] == "StrikeSwinging"]
    cs_data = pt_data[pt_data["PitchCall"] == "StrikeCalled"]
    weak_data = pt_data[(pt_data["PitchCall"] == "InPlay") & (pt_data["ExitSpeed"] < 85)].dropna(subset=["ExitSpeed"])

    loc_defs = [
        ("Whiff Locations", whiff_data, "YlOrRd", f"{len(whiff_data)} whiffs"),
        ("Called Strike Locations", cs_data, "Blues", f"{len(cs_data)} called strikes"),
        ("Weak Contact Locations", weak_data, "Greens", f"{len(weak_data)} weak contacts (EV < 85)"),
    ]
    loc_cols = st.columns(3)
    for idx, (title, ldata, cscale, caption_txt) in enumerate(loc_defs):
        with loc_cols[idx]:
            section_header(title)
            if len(ldata) >= 3:
                fig = go.Figure()
                fig.add_trace(go.Histogram2d(
                    x=ldata["PlateLocSide"].values,
                    y=ldata["PlateLocHeight"].values,
                    nbinsx=8, nbinsy=8,
                    colorscale=cscale, showscale=False,
                ))
                add_strike_zone(fig)
                fig.update_layout(
                    height=280,
                    margin=dict(l=25, r=10, t=10, b=25),
                    xaxis=dict(range=[-2.5, 2.5], scaleanchor="y"),
                    yaxis=dict(range=[0, 5.5]),
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True, key=f"pc_loc_{sel_pt}_{idx}")
            else:
                st.caption("Not enough data")
            st.caption(caption_txt)

    # ── Section C/D: Best Pairs & Sequences (Composite) ──
    if n_pitches >= 2:
        st.markdown("")
        section_header("Best Pitch Pairs & Sequences (Composite)")
        tunnel_pop = build_tunnel_population_pop()
        tunnel_df = _compute_tunnel_score(pdf, tunnel_pop=tunnel_pop)
        pair_df = _compute_pitch_pair_results(pdf, data, tunnel_df=tunnel_df if not tunnel_df.empty else None)
        pitch_metrics = _build_pitch_metric_map(pdf, stuff_df, cmd_df)
        top_pairs = _rank_pairs(tunnel_df, pair_df, pitch_metrics, top_n=2)
        top_seq3 = _filter_redundant_sequences(
            _rank_sequences(pair_df, pitch_metrics, length=3, top_n=5),
            min_unique=2, max_keep=2,
        )
        top_seq4 = _filter_redundant_sequences(
            _rank_sequences(pair_df, pitch_metrics, length=4, top_n=5),
            min_unique=2, max_keep=2,
        )

        st.caption("Outcomes-first (Whiff, K/Putaway, EV) with Tunnel as a secondary signal. Details in checkbox.")

        if top_pairs:
            top_pairs = _assign_tactical_tags(top_pairs)
            rows = []
            detail = []
            for r in top_pairs:
                rows.append({
                    "Pair": r["Pair"],
                    "Score": f"{r['Score']:.1f}" if pd.notna(r["Score"]) else "-",
                    "Whiff%": f"{r['Whiff%']:.1f}" if pd.notna(r["Whiff%"]) else "-",
                    "K/Putaway%": f"{r['K%']:.1f}" if pd.notna(r["K%"]) else "-",
                    "Avg EV": f"{r['Avg EV']:.1f}" if pd.notna(r["Avg EV"]) else "-",
                    "Tag": r.get("Tag", ""),
                    "Deception": _deception_flag(r.get("Tunnel", np.nan)),
                })
                detail.append({
                    "Pair": r["Pair"],
                    "Tunnel": f"{r['Tunnel']:.1f}" if pd.notna(r["Tunnel"]) else "-",
                    "Stuff+": f"{r['Stuff+']:.0f}" if pd.notna(r["Stuff+"]) else "-",
                    "Cmd+": f"{r['Cmd+']:.0f}" if pd.notna(r["Cmd+"]) else "-",
                    "Pairs": int(r["Pairs"]) if pd.notna(r["Pairs"]) else "-",
                })
            st.markdown("**Top 2 Pitch Pairs**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            show_details = st.checkbox(
                "Show details (Tunnel / Stuff+ / Command+ / Sample size)",
                key=f"pc_pairs_detail_{pitcher}",
            )
            if show_details:
                st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data to rank pitch pairs.")

        if top_seq3:
            top_seq3 = _assign_tactical_tags(top_seq3)
            rows = []
            detail = []
            for r in top_seq3:
                rows.append({
                    "Sequence": r["Seq"],
                    "Score": f"{r['Score']:.1f}" if pd.notna(r["Score"]) else "-",
                    "Whiff%": f"{r['Whiff%']:.1f}" if pd.notna(r["Whiff%"]) else "-",
                    "K/Putaway%": f"{r['K%']:.1f}" if pd.notna(r["K%"]) else "-",
                    "Avg EV": f"{r['Avg EV']:.1f}" if pd.notna(r["Avg EV"]) else "-",
                    "Tag": r.get("Tag", ""),
                    "Deception": _deception_flag(r.get("Tunnel", np.nan)),
                })
                detail.append({
                    "Sequence": r["Seq"],
                    "Tunnel": f"{r['Tunnel']:.1f}" if pd.notna(r["Tunnel"]) else "-",
                    "Stuff+": f"{r['Stuff+']:.0f}" if pd.notna(r["Stuff+"]) else "-",
                    "Cmd+": f"{r['Cmd+']:.0f}" if pd.notna(r["Cmd+"]) else "-",
                    "Pairs": r["Pairs"],
                })
            st.markdown("**Top 3‑Pitch Sequences**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            show_details = st.checkbox(
                "Show details (Tunnel / Stuff+ / Command+ / Sample size)",
                key=f"pc_seq3_detail_{pitcher}",
            )
            if show_details:
                st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data to rank 3‑pitch sequences.")

        if top_seq4:
            top_seq4 = _assign_tactical_tags(top_seq4)
            rows = []
            detail = []
            for r in top_seq4:
                rows.append({
                    "Sequence": r["Seq"],
                    "Score": f"{r['Score']:.1f}" if pd.notna(r["Score"]) else "-",
                    "Whiff%": f"{r['Whiff%']:.1f}" if pd.notna(r["Whiff%"]) else "-",
                    "K/Putaway%": f"{r['K%']:.1f}" if pd.notna(r["K%"]) else "-",
                    "Avg EV": f"{r['Avg EV']:.1f}" if pd.notna(r["Avg EV"]) else "-",
                    "Tag": r.get("Tag", ""),
                    "Deception": _deception_flag(r.get("Tunnel", np.nan)),
                })
                detail.append({
                    "Sequence": r["Seq"],
                    "Tunnel": f"{r['Tunnel']:.1f}" if pd.notna(r["Tunnel"]) else "-",
                    "Stuff+": f"{r['Stuff+']:.0f}" if pd.notna(r["Stuff+"]) else "-",
                    "Cmd+": f"{r['Cmd+']:.0f}" if pd.notna(r["Cmd+"]) else "-",
                    "Pairs": r["Pairs"],
                })
            st.markdown("**Top 4‑Pitch Sequences**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            show_details = st.checkbox(
                "Show details (Tunnel / Stuff+ / Command+ / Sample size)",
                key=f"pc_seq4_detail_{pitcher}",
            )
            if show_details:
                st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)

        # ── Best Tunnel Pairs (pure deception ranking) ──
        if isinstance(tunnel_df, pd.DataFrame) and not tunnel_df.empty and "Tunnel Score" in tunnel_df.columns:
            st.markdown("---")
            st.markdown("**Best Tunnel Pairs (Deception Only)**")
            st.caption(
                "Ranked purely by tunnel score — how deceptive two pitches look out of the hand. "
                "Tunnel score measures: (1) **Commit separation** at 280ms before plate — how far apart "
                "pitches are when hitter must decide to swing (55% weight); (2) **Plate separation** — "
                "how much pitches diverge at the plate (19%); (3) **Release point consistency** (10%); "
                "(4) **Release angle similarity** (8%); (5) **Movement divergence** (6%); (6) **Velo gap** (2%). "
                "Graded vs all D1 pitchers throwing the same pitch pair."
            )
            # Sort by tunnel score descending, take top 2
            tunnel_sorted = tunnel_df.sort_values("Tunnel Score", ascending=False).head(2)
            tunnel_rows = []
            for _, r in tunnel_sorted.iterrows():
                tunnel_rows.append({
                    "Pair": f"{r['Pitch A']} → {r['Pitch B']}",
                    "Grade": r.get("Grade", "-"),
                    "Tunnel Score": f"{r['Tunnel Score']:.0f}" if pd.notna(r.get("Tunnel Score")) else "-",
                    "Commit Sep": f"{r['Commit Sep (in)']:.1f}\"" if pd.notna(r.get("Commit Sep (in)")) else "-",
                    "Plate Sep": f"{r['Plate Sep (in)']:.1f}\"" if pd.notna(r.get("Plate Sep (in)")) else "-",
                    "Velo Gap": f"{r['Velo Gap (mph)']:.0f} mph" if pd.notna(r.get("Velo Gap (mph)")) else "-",
                })
            if tunnel_rows:
                st.dataframe(pd.DataFrame(tunnel_rows), use_container_width=True, hide_index=True)

        # Transition matrix mini-heatmap
        sort_cols_tm = [c for c in ["GameID", "Batter", "PAofInning", "PitchNo"] if c in pdf.columns]
        if len(sort_cols_tm) >= 2:
            pdf_tm = pdf.sort_values(sort_cols_tm).copy()
            pdf_tm["NextPitch"] = pdf_tm.groupby(["GameID", "Batter", "PAofInning"])["TaggedPitchType"].shift(-1)
            trans = pdf_tm.dropna(subset=["NextPitch"])
            if not trans.empty and len(trans["TaggedPitchType"].unique()) >= 2:
                st.caption("Pitch Transition Frequencies")
                matrix = pd.crosstab(trans["TaggedPitchType"], trans["NextPitch"], normalize="index") * 100
                fig_matrix = px.imshow(
                    matrix.round(1), text_auto=".0f", color_continuous_scale="RdBu_r",
                    labels=dict(x="Next Pitch", y="Current Pitch", color="%"),
                    aspect="auto",
                )
                fig_matrix.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                                         font=dict(size=11, color="#000000", family="Inter, Arial, sans-serif"),
                                         margin=dict(l=45, r=10, t=10, b=40),
                                         xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
                                         yaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
                                         coloraxis=dict(colorbar=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))))
                st.plotly_chart(fig_matrix, use_container_width=True, key="pc_trans_mini")

    # ── Section E: Platoon Splits ──
    st.markdown("")
    section_header("Platoon Splits")
    split_cols = st.columns(2)
    for side_idx, (side, label) in enumerate([("Right", "vs RHH"), ("Left", "vs LHH")]):
        with split_cols[side_idx]:
            st.markdown(f"**{label}**")
            side_pdf = pdf[pdf["BatterSide"] == side]
            if len(side_pdf) < 10:
                st.caption(f"Limited data vs {label} ({len(side_pdf)} pitches)")
                continue
            side_cmd = _compute_command_plus(side_pdf, data)
            side_cmd_map = dict(zip(side_cmd["Pitch"], side_cmd["Command+"])) if not side_cmd.empty else {}
            split_rows = []
            for pt in pitch_types:
                pt_d = side_pdf[side_pdf["TaggedPitchType"] == pt]
                if len(pt_d) < 10:
                    continue
                sw = pt_d[pt_d["PitchCall"].isin(SWING_CALLS)]
                whiff_pct = len(pt_d[pt_d["PitchCall"] == "StrikeSwinging"]) / max(len(sw), 1) * 100
                csw_pct = pt_d["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
                cmd_plus = side_cmd_map.get(pt, np.nan)
                loc_vals = pt_d.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                loc_spread = np.sqrt(loc_vals["PlateLocHeight"].std()**2 + loc_vals["PlateLocSide"].std()**2) if len(loc_vals) >= 5 else np.nan
                # Composite for ranking
                w_norm = min(whiff_pct / 50, 1)
                c_norm = min(csw_pct / 40, 1)
                cmd_norm = min(max((cmd_plus - 80) / 40, 0), 1) if not pd.isna(cmd_plus) else 0.5
                comp = w_norm * 0.4 + c_norm * 0.3 + cmd_norm * 0.3
                if not pd.isna(loc_spread):
                    if loc_spread < 0.8:
                        consist = "precise"
                    elif loc_spread < 1.1:
                        consist = "average"
                    else:
                        consist = "inconsistent"
                else:
                    consist = "-"
                split_rows.append({
                    "pitch": pt, "whiff": whiff_pct, "csw": csw_pct,
                    "cmd": cmd_plus, "consist": consist, "comp": comp,
                    "count": len(pt_d),
                })
            if not split_rows:
                st.caption("No pitch type with 10+ pitches")
                continue
            split_rows.sort(key=lambda x: x["comp"], reverse=True)
            for sr in split_rows[:3]:
                pc = PITCH_COLORS.get(sr["pitch"], "#888")
                cmd_str = f'{sr["cmd"]:.0f}' if not pd.isna(sr["cmd"]) else "-"
                st.markdown(
                    f'<div style="padding:6px 10px;border-left:4px solid {pc};'
                    f'background:#f9f9f9;border-radius:4px;margin-bottom:4px;font-size:13px;">'
                    f'<b style="color:{pc};">{sr["pitch"]}</b> '
                    f'<span style="color:#555;">({sr["count"]} pitches)</span><br>'
                    f'Whiff: <b>{sr["whiff"]:.0f}%</b> &middot; '
                    f'CSW: <b>{sr["csw"]:.0f}%</b> &middot; '
                    f'Cmd+: <b>{cmd_str}</b>'
                    f'</div>', unsafe_allow_html=True)




_METRIC_LABELS = {
    "RelSpeed": "Velocity",
    "InducedVertBreak": "IVB",
    "HorzBreak": "Horizontal Break",
    "Extension": "Extension",
    "VertApprAngle": "VAA",
    "SpinRate": "Spin Rate",
}

_METRIC_UNITS = {
    "RelSpeed": "mph",
    "InducedVertBreak": "in",
    "HorzBreak": "in",
    "Extension": "ft",
    "VertApprAngle": "°",
    "SpinRate": "rpm",
}



def _compute_pitch_recommendations(pdf, data, tunnel_df):
    """For each pitch type, compute actionable improvement suggestions.

    Returns list of dicts: {pitch, metric, label, current, target, delta,
    direction, unit, tunnel_benefit}

    HorzBreak is handled via absolute values so LHP (negative HB) and RHP
    (positive HB) are compared on the same scale.  Displayed values are
    converted back to the pitcher's sign convention.
    """
    from scipy.stats import percentileofscore as _pctile

    # Stuff+ weights define which direction is "better" for each metric
    # For HorzBreak the weight sign describes *magnitude* direction after
    # taking abs(); positive weight → more arm-side run is better.
    weights = {
        "Fastball":       {"RelSpeed": 2.0, "InducedVertBreak": 2.5, "HorzBreak": 0.3, "Extension": 0.5, "VertApprAngle": 2.5, "SpinRate": 1.0},
        "Sinker":         {"RelSpeed": 2.5, "InducedVertBreak": -0.5, "HorzBreak": 1.5, "Extension": 0.5, "VertApprAngle": -1.5, "SpinRate": 0.8},
        "Cutter":         {"RelSpeed": 0.8, "InducedVertBreak": 0.3, "HorzBreak": -1.5, "Extension": -1.0, "VertApprAngle": -0.5, "SpinRate": 2.0},
        "Slider":         {"RelSpeed": 1.0, "InducedVertBreak": -0.5, "HorzBreak": 1.0, "Extension": 0.3, "VertApprAngle": -2.5, "SpinRate": 1.5},
        "Curveball":      {"RelSpeed": 1.5, "InducedVertBreak": -1.5, "HorzBreak": -1.5, "Extension": -1.5, "VertApprAngle": -2.0, "SpinRate": 0.5},
        "Changeup":       {"RelSpeed": 0.5, "InducedVertBreak": 1.5, "HorzBreak": 1.0, "Extension": 0.5, "VertApprAngle": -2.5, "SpinRate": 1.0},
        "Sweeper":        {"RelSpeed": 1.5, "InducedVertBreak": -1.0, "HorzBreak": 2.0, "Extension": 0.8, "VertApprAngle": -1.5, "SpinRate": 0.5},
        "Splitter":       {"RelSpeed": 1.0, "InducedVertBreak": -2.0, "HorzBreak": 0.5, "Extension": 1.0, "VertApprAngle": -2.0, "SpinRate": -0.3},
        "Knuckle Curve":  {"RelSpeed": 1.5, "InducedVertBreak": -1.5, "HorzBreak": -1.5, "Extension": -1.5, "VertApprAngle": -2.0, "SpinRate": 0.5},
    }

    base = data.dropna(subset=["RelSpeed", "TaggedPitchType"]).copy()
    metrics = ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "VertApprAngle", "Extension"]

    # Determine pitcher handedness sign for HorzBreak:
    # RHP → positive HB = arm-side run; LHP → negative HB = arm-side run.
    # We normalise to abs(HB) so comparisons are hand-agnostic.
    throws = pdf["PitcherThrows"].mode()
    is_lhp = (throws.iloc[0] == "Left") if len(throws) > 0 else False
    hb_sign = -1.0 if is_lhp else 1.0  # multiplier to convert raw → abs convention

    # Pre-compute baseline per pitch type using abs(HorzBreak)
    base["_AbsHB"] = base["HorzBreak"].abs() if "HorzBreak" in base.columns else np.nan
    baseline = {}
    for pt, grp in base.groupby("TaggedPitchType"):
        stats = {}
        for m in metrics:
            col = "_AbsHB" if m == "HorzBreak" else m
            if col in grp.columns:
                vals = grp[col].astype(float).dropna()
                if len(vals) >= 10:
                    stats[m] = {"mean": vals.mean(), "std": vals.std(),
                                "p75": np.percentile(vals, 75),
                                "p25": np.percentile(vals, 25)}
        baseline[pt] = stats

    # Build tunnel partner map for cross-referencing
    tunnel_partners = {}
    if isinstance(tunnel_df, pd.DataFrame) and not tunnel_df.empty:
        for _, trow in tunnel_df.iterrows():
            a, b = trow["Pitch A"], trow["Pitch B"]
            tunnel_partners.setdefault(a, []).append((b, trow.get("Grade", "-"), trow.get("Tunnel Score", np.nan)))
            tunnel_partners.setdefault(b, []).append((a, trow.get("Grade", "-"), trow.get("Tunnel Score", np.nan)))

    recommendations = []
    for pt in sorted(pdf["TaggedPitchType"].unique()):
        pt_d = pdf[pdf["TaggedPitchType"] == pt]
        if len(pt_d) < 10:
            continue
        w = weights.get(pt, {})
        bstats = baseline.get(pt, {})
        if not w or not bstats:
            continue

        # For each metric, compute how far below the 75th pctl the pitcher is
        # in the "good" direction
        gaps = []
        for m in metrics:
            if m not in w or m not in bstats:
                continue
            bs = bstats[m]
            if bs["std"] == 0 or pd.isna(bs["std"]):
                continue

            # For HorzBreak, use absolute value so LHP/RHP are on the same scale
            if m == "HorzBreak":
                pitcher_val = pt_d[m].astype(float).dropna().abs().mean()
            else:
                pitcher_val = pt_d[m].astype(float).dropna().mean()
            if pd.isna(pitcher_val):
                continue

            weight_sign = 1 if w[m] > 0 else -1

            # Percentile of pitcher vs abs-HB baseline (or normal baseline)
            if m == "HorzBreak":
                all_vals = base[base["TaggedPitchType"] == pt]["_AbsHB"].astype(float).dropna()
            else:
                all_vals = base[base["TaggedPitchType"] == pt][m].astype(float).dropna()
            if len(all_vals) < 10:
                continue
            pctl = _pctile(all_vals, pitcher_val, kind="rank")
            good_pctl = pctl if weight_sign > 0 else (100 - pctl)

            # Target = 75th percentile in the good direction
            if weight_sign > 0:
                target = bs["p75"]
            else:
                target = bs["p25"]

            # Only flag if pitcher is below 60th pctl in good direction
            if good_pctl < 60:
                gaps.append({
                    "metric": m,
                    "good_pctl": good_pctl,
                    "current": pitcher_val,
                    "target": target,
                    "weight_sign": weight_sign,
                    "abs_weight": abs(w[m]),
                })

        # Sort by worst percentile, weighted by importance
        gaps.sort(key=lambda g: g["good_pctl"] - g["abs_weight"] * 5)

        # Take top 2 recommendations
        for g in gaps[:2]:
            m = g["metric"]
            label = _METRIC_LABELS.get(m, m)
            unit = _METRIC_UNITS.get(m, "")
            current_abs = g["current"]  # already abs for HB
            target_abs = g["target"]
            ws = g["weight_sign"]

            if ws > 0:
                delta_val = target_abs - current_abs
                direction = f"more {label.lower()}" if delta_val > 0 else f"maintain {label.lower()}"
            else:
                delta_val = current_abs - target_abs
                direction = f"{'steeper' if m == 'VertApprAngle' else 'more'} {label.lower()}" if delta_val > 0 else f"reduce {label.lower()}"

            # For display: convert HorzBreak back to the pitcher's sign convention
            if m == "HorzBreak":
                display_current = current_abs * hb_sign
                display_target = target_abs * hb_sign
                # Delta is always in magnitude terms (positive = more break)
                delta_display = target_abs - current_abs
            else:
                display_current = current_abs
                display_target = target_abs
                delta_display = delta_val

            # Cross-reference with tunnel partners
            tunnel_benefit = ""
            tunnel_partner = ""
            tunnel_grade = ""
            partners = tunnel_partners.get(pt, [])
            if partners:
                # Find a partner where improving this metric would help
                for partner_pt, tgrade, tscore in partners:
                    if tgrade in ("D", "F", "C"):
                        if m == "InducedVertBreak":
                            tunnel_benefit = f"Improves tunnel with {partner_pt} (currently grade {tgrade})"
                        elif m == "HorzBreak":
                            tunnel_benefit = f"Creates better separation from {partner_pt} (currently grade {tgrade})"
                        elif m == "RelSpeed":
                            tunnel_benefit = f"Affects velo gap with {partner_pt} (currently grade {tgrade})"
                        elif m == "VertApprAngle":
                            tunnel_benefit = f"Helps deception against {partner_pt} (currently grade {tgrade})"
                        if tunnel_benefit:
                            tunnel_partner = partner_pt
                            tunnel_grade = tgrade
                            break

            recommendations.append({
                "pitch": pt,
                "metric": m,
                "label": label,
                "current": round(display_current, 1),
                "target": round(display_target, 1),
                "delta": f"{'+' if delta_display > 0 else ''}{delta_display:.1f} {unit}",
                "direction": direction,
                "unit": unit,
                "good_pctl": round(g["good_pctl"], 0),
                "tunnel_benefit": tunnel_benefit,
                "tunnel_partner": tunnel_partner,
                "tunnel_grade": tunnel_grade,
            })

    return recommendations




def _pitch_lab_page(data, pitcher, season_filter, pdf, stuff_df, pr, all_pitcher_stats, cmd_df=None):
    """Render the Pitch Lab tab with arsenal recommendations,
    sequencing playbook, and pitch-specific deep dives."""

    if len(pdf) < 20:
        st.warning("Not enough pitch data for Pitch Lab (need 20+).")
        return

    has_stuff = stuff_df is not None and "StuffPlus" in stuff_df.columns and not stuff_df.empty
    if not has_stuff:
        st.info("Stuff+ data not available — some sections will be limited.")

    # Pre-compute shared data
    try:
        tunnel_pop = build_tunnel_population_pop()
        tunnel_df = _compute_tunnel_score(pdf, tunnel_pop=tunnel_pop)
    except Exception as e:
        st.warning(f"Could not compute tunnel scores: {e}")
        tunnel_df = pd.DataFrame()
    if cmd_df is None:
        try:
            cmd_df = _compute_command_plus(pdf, data)
        except Exception as e:
            st.warning(f"Could not compute Command+: {e}")
            cmd_df = pd.DataFrame()
    cmd_map = dict(zip(cmd_df["Pitch"], cmd_df["Command+"])) if not cmd_df.empty else {}

    pitch_types = sorted(pdf["TaggedPitchType"].unique())
    pair_df = _compute_pitch_pair_results(pdf, data, tunnel_df=tunnel_df)
    pitch_metrics = _build_pitch_metric_map(pdf, stuff_df, cmd_df)

    # ═══════════════════════════════════════════
    # BEST PAIRS & SEQUENCES (COMPOSITE)
    # ═══════════════════════════════════════════
    section_header("Best Pairs & Sequences (Composite)")
    st.caption("Outcomes-first (Whiff, K/Putaway, EV) with Tunnel as a secondary signal. Details in checkbox.")
    top_pairs = _rank_pairs(tunnel_df, pair_df, pitch_metrics, top_n=2)
    top_seq3 = _filter_redundant_sequences(
        _rank_sequences(pair_df, pitch_metrics, length=3, top_n=5),
        min_unique=2, max_keep=2,
    )
    top_seq4 = _filter_redundant_sequences(
        _rank_sequences(pair_df, pitch_metrics, length=4, top_n=5),
        min_unique=2, max_keep=2,
    )

    if top_pairs:
        top_pairs = _assign_tactical_tags(top_pairs)
        rows = []
        detail = []
        for r in top_pairs:
            rows.append({
                "Pair": r["Pair"],
                "Score": f"{r['Score']:.1f}" if pd.notna(r["Score"]) else "-",
                "Whiff%": f"{r['Whiff%']:.1f}" if pd.notna(r["Whiff%"]) else "-",
                "K/Putaway%": f"{r['K%']:.1f}" if pd.notna(r["K%"]) else "-",
                "Avg EV": f"{r['Avg EV']:.1f}" if pd.notna(r["Avg EV"]) else "-",
                "Tag": r.get("Tag", ""),
                "Deception": _deception_flag(r.get("Tunnel", np.nan)),
            })
            detail.append({
                "Pair": r["Pair"],
                "Tunnel": f"{r['Tunnel']:.1f}" if pd.notna(r["Tunnel"]) else "-",
                "Stuff+": f"{r['Stuff+']:.0f}" if pd.notna(r["Stuff+"]) else "-",
                "Cmd+": f"{r['Cmd+']:.0f}" if pd.notna(r["Cmd+"]) else "-",
                "Pairs": int(r["Pairs"]) if pd.notna(r["Pairs"]) else "-",
            })
        st.markdown("**Top 2 Pitch Pairs**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        show_details = st.checkbox(
            "Show details (Tunnel / Stuff+ / Command+ / Sample size)",
            key=f"pl_pairs_detail_{pitcher}",
        )
        if show_details:
            st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)
    else:
        st.caption("Not enough data to rank pitch pairs.")

    if top_seq3:
        top_seq3 = _assign_tactical_tags(top_seq3)
        rows = []
        detail = []
        for r in top_seq3:
            rows.append({
                "Sequence": r["Seq"],
                "Score": f"{r['Score']:.1f}" if pd.notna(r["Score"]) else "-",
                "Whiff%": f"{r['Whiff%']:.1f}" if pd.notna(r["Whiff%"]) else "-",
                "K/Putaway%": f"{r['K%']:.1f}" if pd.notna(r["K%"]) else "-",
                "Avg EV": f"{r['Avg EV']:.1f}" if pd.notna(r["Avg EV"]) else "-",
                "Tag": r.get("Tag", ""),
                "Deception": _deception_flag(r.get("Tunnel", np.nan)),
            })
            detail.append({
                "Sequence": r["Seq"],
                "Tunnel": f"{r['Tunnel']:.1f}" if pd.notna(r["Tunnel"]) else "-",
                "Stuff+": f"{r['Stuff+']:.0f}" if pd.notna(r["Stuff+"]) else "-",
                "Cmd+": f"{r['Cmd+']:.0f}" if pd.notna(r["Cmd+"]) else "-",
                "Pairs": r["Pairs"],
            })
        st.markdown("**Top 3‑Pitch Sequences**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        show_details = st.checkbox(
            "Show details (Tunnel / Stuff+ / Command+ / Sample size)",
            key=f"pl_seq3_detail_{pitcher}",
        )
        if show_details:
            st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)
    else:
        st.caption("Not enough data to rank 3‑pitch sequences.")

    if top_seq4:
        top_seq4 = _assign_tactical_tags(top_seq4)
        rows = []
        detail = []
        for r in top_seq4:
            rows.append({
                "Sequence": r["Seq"],
                "Score": f"{r['Score']:.1f}" if pd.notna(r["Score"]) else "-",
                "Whiff%": f"{r['Whiff%']:.1f}" if pd.notna(r["Whiff%"]) else "-",
                "K/Putaway%": f"{r['K%']:.1f}" if pd.notna(r["K%"]) else "-",
                "Avg EV": f"{r['Avg EV']:.1f}" if pd.notna(r["Avg EV"]) else "-",
                "Tag": r.get("Tag", ""),
                "Deception": _deception_flag(r.get("Tunnel", np.nan)),
            })
            detail.append({
                "Sequence": r["Seq"],
                "Tunnel": f"{r['Tunnel']:.1f}" if pd.notna(r["Tunnel"]) else "-",
                "Stuff+": f"{r['Stuff+']:.0f}" if pd.notna(r["Stuff+"]) else "-",
                "Cmd+": f"{r['Cmd+']:.0f}" if pd.notna(r["Cmd+"]) else "-",
                "Pairs": r["Pairs"],
            })
        st.markdown("**Top 4‑Pitch Sequences**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        show_details = st.checkbox(
            "Show details (Tunnel / Stuff+ / Command+ / Sample size)",
            key=f"pl_seq4_detail_{pitcher}",
        )
        if show_details:
            st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)
    else:
        st.caption("Not enough data to rank 4‑pitch sequences.")

    # ═══════════════════════════════════════════
    # SECTION A: Arsenal Overview with Improvement Targets
    # ═══════════════════════════════════════════
    section_header("Arsenal Overview & Improvement Targets")
    st.caption("Current pitch profiles compared to database averages. Recommendations based on Stuff+ weight directions and tunnel partners.")

    try:
        recommendations = _compute_pitch_recommendations(pdf, data, tunnel_df)
    except Exception:
        recommendations = []
    rec_by_pitch = {}
    for r in recommendations:
        rec_by_pitch.setdefault(r["pitch"], []).append(r)

    # Split toggle for outcomes/targets
    split_mode = st.radio("Split View", ["All", "vs LHB", "vs RHB"], horizontal=True, key="pl_split_view")
    if split_mode == "vs LHB":
        pdf_split = pdf[pdf["BatterSide"] == "Left"].copy()
    elif split_mode == "vs RHB":
        pdf_split = pdf[pdf["BatterSide"] == "Right"].copy()
    else:
        pdf_split = pdf

    for pt in pitch_types:
        pt_d = pdf_split[pdf_split["TaggedPitchType"] == pt]
        if len(pt_d) < 10:
            continue
        color = PITCH_COLORS.get(pt, "#888")
        usage_pct = len(pt_d) / max(len(pdf), 1) * 100

        # Compute metrics
        velo = pt_d["RelSpeed"].mean()
        ivb = pt_d["InducedVertBreak"].mean() if "InducedVertBreak" in pt_d.columns else np.nan
        hb = pt_d["HorzBreak"].mean() if "HorzBreak" in pt_d.columns else np.nan
        spin = pt_d["SpinRate"].mean() if "SpinRate" in pt_d.columns else np.nan
        vaa = pt_d["VertApprAngle"].mean() if "VertApprAngle" in pt_d.columns else np.nan
        ext = pt_d["Extension"].mean() if "Extension" in pt_d.columns else np.nan
        stuff_val = stuff_df[stuff_df["TaggedPitchType"] == pt]["StuffPlus"].mean() if has_stuff else np.nan
        cmd_val = cmd_map.get(pt, np.nan)
        sw = pt_d[pt_d["PitchCall"].isin(SWING_CALLS)]
        wh = pt_d[pt_d["PitchCall"] == "StrikeSwinging"]
        csw = pt_d[pt_d["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])]
        inplay_ev = pt_d[(pt_d["PitchCall"] == "InPlay") & pt_d["ExitSpeed"].notna()]
        hard_hit = inplay_ev[inplay_ev["ExitSpeed"] >= 95]
        whiff_pct = len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else np.nan
        csw_pct = len(csw) / max(len(pt_d), 1) * 100 if len(pt_d) > 0 else np.nan
        hh_pct = len(hard_hit) / max(len(inplay_ev), 1) * 100 if len(inplay_ev) > 0 else np.nan

        # Header bar
        stuff_str = f"Stuff+ {stuff_val:.0f}" if not pd.isna(stuff_val) else ""
        cmd_str = f"Cmd+ {cmd_val:.0f}" if not pd.isna(cmd_val) else ""
        usage_str = f"Usage {usage_pct:.1f}% · N={len(pt_d)}"
        badges = " &middot; ".join(filter(None, [stuff_str, cmd_str, usage_str]))
        st.markdown(
            f'<div style="padding:8px 14px;border-radius:8px;border-left:5px solid {color};'
            f'background:{color}12;margin:8px 0 4px 0;">'
            f'<span style="font-size:16px;font-weight:bold;color:{color};">{pt}</span>'
            f'<span style="font-size:12px;color:#555;margin-left:12px;">{badges}</span>'
            f'</div>', unsafe_allow_html=True)

        # Profile metrics in columns
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        def _metric_cell(col, label, val, fmt=".1f", unit=""):
            with col:
                v_str = f"{val:{fmt}}{unit}" if not pd.isna(val) else "-"
                st.markdown(f'<div style="text-align:center;"><div style="font-size:11px;color:#888;">{label}</div>'
                            f'<div style="font-size:16px;font-weight:bold;">{v_str}</div></div>',
                            unsafe_allow_html=True)
        _metric_cell(mc1, "Velo", velo, ".1f", " mph")
        _metric_cell(mc2, "IVB", ivb, ".1f", '"')
        _metric_cell(mc3, "HB", hb, ".1f", '"')
        _metric_cell(mc4, "Spin", spin, ".0f", " rpm")
        _metric_cell(mc5, "VAA", vaa, ".1f", "°")
        _metric_cell(mc6, "Ext", ext, ".1f", " ft")

        # Outcome row
        oc1, oc2, oc3 = st.columns(3)
        def _outcome_cell(col, label, val):
            with col:
                v_str = f"{val:.1f}%" if not pd.isna(val) else "-"
                st.markdown(
                    f'<div style="text-align:center;"><div style="font-size:11px;color:#555;">{label}</div>'
                    f'<div style="font-size:14px;font-weight:700;color:#111;">{v_str}</div></div>',
                    unsafe_allow_html=True)
        _outcome_cell(oc1, "Whiff%", whiff_pct)
        _outcome_cell(oc2, "CSW%", csw_pct)
        _outcome_cell(oc3, "HardHit%", hh_pct)

        # Recommendations for this pitch
        recs = rec_by_pitch.get(pt, [])
        if recs:
            for rec in recs:
                pctl_str = f"{rec['good_pctl']:.0f}th pctl"
                basis_str = f"Target basis: D1 {pctl_str}"
                proj_grade = ""
                if rec.get("tunnel_grade"):
                    next_grade = {"F": "D", "D": "C", "C": "B", "B": "A"}.get(rec["tunnel_grade"], rec["tunnel_grade"])
                    proj_grade = f"Tunnel impact: {rec['tunnel_grade']} → {next_grade} (heuristic)"
                tun_str = f" — {rec['tunnel_benefit']}" if rec['tunnel_benefit'] else ""
                st.markdown(
                    f'<div style="padding:4px 12px;margin:2px 0;font-size:12px;'
                    f'background:#fff8e1;border-radius:4px;border-left:3px solid #f59e0b;">'
                    f'<b>{rec["direction"].capitalize()}</b>: '
                    f'currently {rec["current"]} {rec["unit"]} ({pctl_str}), '
                    f'target {rec["target"]} {rec["unit"]} ({rec["delta"]})'
                    f'{tun_str}'
                    f'<div style="font-size:11px;color:#555;margin-top:2px;">{basis_str}'
                    f'{(" · " + proj_grade) if proj_grade else ""}</div>'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="padding:4px 12px;margin:2px 0;font-size:12px;'
                'background:#e8f5e9;border-radius:4px;border-left:3px solid #22c55e;">'
                'No major improvement targets — pitch metrics are solid relative to peers.'
                '</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # SECTION C: Sequencing Playbook
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Sequencing Playbook")
    st.caption("Transition frequencies, sequence effectiveness, and count-state pitch selection.")

    # Transition matrix
    sort_cols_seq = [c for c in ["GameID", "Batter", "PAofInning", "PitchNo"] if c in pdf.columns]
    if len(sort_cols_seq) >= 2:
        pdf_seq = pdf.sort_values(sort_cols_seq).copy()
        pdf_seq["NextPitch"] = pdf_seq.groupby(["GameID", "Batter", "PAofInning"])["TaggedPitchType"].shift(-1)
        trans = pdf_seq.dropna(subset=["NextPitch"])
        if not trans.empty and len(trans["TaggedPitchType"].unique()) >= 2:
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                st.markdown("**Pitch Transition Matrix**")
                matrix = pd.crosstab(trans["TaggedPitchType"], trans["NextPitch"], normalize="index") * 100
                fig_matrix = px.imshow(
                    matrix.round(1), text_auto=".0f", color_continuous_scale="RdBu_r",
                    labels=dict(x="Next Pitch", y="Current Pitch", color="%"),
                    aspect="auto",
                )
                fig_matrix.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white",
                                        font=dict(size=11, color="#000000", family="Inter, Arial, sans-serif"),
                                        margin=dict(l=80, r=10, t=30, b=60))
                st.plotly_chart(fig_matrix, use_container_width=True, key="plab_trans")

            with tcol2:
                if not pair_df.empty:
                    st.markdown("**Whiff% by Sequence**")
                    whiff_pivot = pair_df.pivot_table(index="Setup Pitch", columns="Follow Pitch",
                                                      values="Whiff%", aggfunc="first")
                    if not whiff_pivot.empty:
                        fig_whiff = px.imshow(
                            whiff_pivot.fillna(0).round(1), text_auto=".0f",
                            color_continuous_scale="YlOrRd",
                            labels=dict(x="Follow-Up Pitch", y="Setup Pitch", color="Whiff%"),
                            aspect="auto",
                        )
                        fig_whiff.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white",
                                               font=dict(size=11, color="#000000", family="Inter, Arial, sans-serif"),
                                               margin=dict(l=80, r=10, t=30, b=60))
                        st.plotly_chart(fig_whiff, use_container_width=True, key="plab_whiff_hm")

    # Count-state pitch selection
    st.markdown("**Count-State Pitch Selection**")
    counts_of_interest = [("0", "0"), ("0", "2"), ("1", "2"), ("2", "0"), ("3", "1"), ("3", "2")]
    count_rows = []
    pdf_cs = pdf.dropna(subset=["Balls", "Strikes"]).copy() if "Balls" in pdf.columns and "Strikes" in pdf.columns else pd.DataFrame()
    for b, s in counts_of_interest:
        count_data = pdf_cs[(pdf_cs["Balls"].astype(int).astype(str) == b) & (pdf_cs["Strikes"].astype(int).astype(str) == s)] if not pdf_cs.empty else pd.DataFrame()
        if len(count_data) >= 3:
            usage = count_data["TaggedPitchType"].value_counts(normalize=True) * 100
            for pt_name, pct in usage.items():
                count_rows.append({"Count": f"{b}-{s}", "Pitch": pt_name, "Usage%": round(pct, 1)})
    if count_rows:
        count_df = pd.DataFrame(count_rows)
        fig_count = px.bar(
            count_df, x="Count", y="Usage%", color="Pitch",
            color_discrete_map=PITCH_COLORS, barmode="stack",
        )
        fig_count.update_layout(
            height=350, yaxis_title="Usage %", xaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig_count, use_container_width=True, key="plab_count_sel")
    else:
        st.info("Not enough count-state data.")

    # ── Get Back in the Count: best pitches at 1-0 and 2-0 ──
    st.markdown("**Get Back in the Count (1-0, 2-0)**")
    st.caption("Best pitches to throw when behind — ranked by CSW% at that count.")
    pdf_gbc = pdf.dropna(subset=["Balls", "Strikes"]).copy() if "Balls" in pdf.columns and "Strikes" in pdf.columns else pd.DataFrame()
    for count_label, b_str, s_str in [("1-0", "1", "0"), ("2-0", "2", "0")]:
        count_d = pdf_gbc[(pdf_gbc["Balls"].astype(int).astype(str) == b_str) & (pdf_gbc["Strikes"].astype(int).astype(str) == s_str)] if not pdf_gbc.empty else pd.DataFrame()
        if len(count_d) < 5:
            st.info(f"Not enough data at {count_label}.")
            continue
        # Rank pitches by CSW% at this count
        ct_rows = []
        for cpt in count_d["TaggedPitchType"].unique():
            cpt_d = count_d[count_d["TaggedPitchType"] == cpt]
            if len(cpt_d) < 3:
                continue
            csw_ct = cpt_d["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
            usage_ct = len(cpt_d) / len(count_d) * 100
            ct_rows.append({"Pitch": cpt, "CSW%": round(csw_ct, 1),
                            "Usage%": round(usage_ct, 1), "N": len(cpt_d)})
        if not ct_rows:
            continue
        ct_df = pd.DataFrame(ct_rows).sort_values("CSW%", ascending=False)
        st.markdown(f"**{count_label}**")
        # Show top pitches as inline badges
        badge_parts = []
        for _, cr in ct_df.head(3).iterrows():
            pc = PITCH_COLORS.get(cr["Pitch"], "#888")
            badge_parts.append(
                f'<span style="display:inline-block;padding:4px 10px;border-radius:12px;'
                f'background:{pc}20;border:1px solid {pc};margin:2px;font-size:12px;">'
                f'<b style="color:{pc};">{cr["Pitch"]}</b> '
                f'CSW {cr["CSW%"]:.0f}% &middot; {cr["Usage%"]:.0f}% used</span>')
        st.markdown(" ".join(badge_parts), unsafe_allow_html=True)
        # Location heatmaps for top 2 pitches at this count
        top_ct_pitches = ct_df.head(2)["Pitch"].tolist()
        gbc_cols = st.columns(len(top_ct_pitches))
        for ci, cpt_name in enumerate(top_ct_pitches):
            with gbc_cols[ci]:
                cpt_sub = count_d[count_d["TaggedPitchType"] == cpt_name].dropna(
                    subset=["PlateLocSide", "PlateLocHeight"])
                color_c = PITCH_COLORS.get(cpt_name, "#888")
                if len(cpt_sub) >= 3:
                    fig_gbc = go.Figure(go.Histogram2d(
                        x=cpt_sub["PlateLocSide"], y=cpt_sub["PlateLocHeight"],
                        nbinsx=8, nbinsy=8, colorscale=[[0, "white"], [1, color_c]],
                        showscale=False,
                    ))
                    add_strike_zone(fig_gbc)
                    fig_gbc.update_layout(
                        xaxis=dict(range=[-2.5, 2.5], title="", showticklabels=False,
                                   fixedrange=True, scaleanchor="y"),
                        yaxis=dict(range=[0, 5], title="", showticklabels=False,
                                   fixedrange=True),
                        height=320, margin=dict(l=5, r=5, t=5, b=5),
                        plot_bgcolor="white", paper_bgcolor="white",
                    )
                    st.plotly_chart(fig_gbc, use_container_width=True,
                                    key=f"plab_gbc_{count_label}_{cpt_name}")
                    st.caption(f"{cpt_name} at {count_label}")
                else:
                    st.caption(f"{cpt_name}: < 3 pitches")

    # ═══════════════════════════════════════════
    # SECTION D: Pitch-Specific Deep Dive
    # ═══════════════════════════════════════════
    st.markdown("---")
    section_header("Pitch-Specific Deep Dive")
    sel_pitch = st.selectbox("Select Pitch", pitch_types, key="plab_deep_pitch")
    pt_data = pdf[pdf["TaggedPitchType"] == sel_pitch].copy()

    if len(pt_data) < 10:
        st.info("Not enough pitches of this type to analyze.")
    else:
        # Stuff+ distribution violin
        if has_stuff:
            dd1, dd2 = st.columns(2)
            with dd1:
                st.markdown("**Stuff+ Distribution**")
                pt_stuff = stuff_df[stuff_df["TaggedPitchType"] == sel_pitch]["StuffPlus"]
                if len(pt_stuff) >= 5:
                    fig_violin = go.Figure(go.Violin(
                        y=pt_stuff, box_visible=True, meanline_visible=True,
                        fillcolor=PITCH_COLORS.get(sel_pitch, "#888"),
                        line_color=PITCH_COLORS.get(sel_pitch, "#888"), opacity=0.7,
                        name=sel_pitch,
                    ))
                    fig_violin.add_hline(y=100, line_dash="dash", line_color="#888",
                                         annotation_text="Avg (100)")
                    fig_violin.update_layout(height=300, showlegend=False,
                                             yaxis_title="Stuff+", **CHART_LAYOUT)
                    st.plotly_chart(fig_violin, use_container_width=True, key="plab_dd_violin")
                else:
                    st.info("Not enough Stuff+ data for violin plot.")

            # Rolling Stuff+ trend
            with dd2:
                st.markdown("**Rolling Stuff+ Trend**")
                stuff_time = stuff_df[stuff_df["TaggedPitchType"] == sel_pitch].copy()
                if "Date" in stuff_time.columns:
                    stuff_time = stuff_time.dropna(subset=["Date"]).sort_values("Date")
                if len(stuff_time) >= 15:
                    window = min(25, len(stuff_time) // 2)
                    stuff_time["StuffRolling"] = stuff_time["StuffPlus"].rolling(window, min_periods=5).mean()
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=list(range(len(stuff_time))), y=stuff_time["StuffRolling"],
                        mode="lines", line=dict(color=PITCH_COLORS.get(sel_pitch, "#888"), width=2),
                        name=sel_pitch,
                    ))
                    fig_trend.add_hline(y=100, line_dash="dash", line_color="#888")
                    fig_trend.update_layout(height=300, showlegend=False,
                                             xaxis_title="Pitch #", yaxis_title="Stuff+ (rolling)",
                                             **CHART_LAYOUT)
                    st.plotly_chart(fig_trend, use_container_width=True, key="plab_dd_trend")
                else:
                    st.info("Not enough data for trend.")

        # Command+ zone quadrant grid
        if not cmd_df.empty:
            cmd_row = cmd_df[cmd_df["Pitch"] == sel_pitch]
            if not cmd_row.empty:
                cr = cmd_row.iloc[0]
                st.markdown(f"**Command+ Profile** — Command+ **{cr['Command+']:.0f}** "
                            f"| Zone% {cr['Zone%']:.1f} | Edge% {cr['Edge%']:.1f} "
                            f"| Chase% {cr['Chase%']:.1f}")

        # Whiff / Called Strike / Weak Contact heatmaps
        st.markdown("**Outcome Heatmaps**")
        hm1, hm2, hm3 = st.columns(3)
        with hm1:
            whiff_d = pt_data[pt_data["PitchCall"] == "StrikeSwinging"]
            st.caption(f"Whiff Locations ({len(whiff_d)})")
            if len(whiff_d) >= 3:
                fig_w = go.Figure(go.Histogram2d(
                    x=whiff_d["PlateLocSide"], y=whiff_d["PlateLocHeight"],
                    nbinsx=10, nbinsy=10, colorscale="YlOrRd", showscale=False,
                ))
                add_strike_zone(fig_w)
                fig_w.update_layout(xaxis=dict(range=[-2.5, 2.5], title="", showticklabels=False,
                                               fixedrange=True, scaleanchor="y"),
                                    yaxis=dict(range=[0, 5], title="", showticklabels=False,
                                               fixedrange=True),
                                    height=320, **CHART_LAYOUT)
                st.plotly_chart(fig_w, use_container_width=True, key="plab_dd_whiff")
            else:
                st.info("< 3 whiffs")
        with hm2:
            cs_d = pt_data[pt_data["PitchCall"] == "StrikeCalled"]
            st.caption(f"Called Strikes ({len(cs_d)})")
            if len(cs_d) >= 3:
                fig_cs = go.Figure(go.Histogram2d(
                    x=cs_d["PlateLocSide"], y=cs_d["PlateLocHeight"],
                    nbinsx=10, nbinsy=10, colorscale="Blues", showscale=False,
                ))
                add_strike_zone(fig_cs)
                fig_cs.update_layout(xaxis=dict(range=[-2.5, 2.5], title="", showticklabels=False,
                                                fixedrange=True, scaleanchor="y"),
                                     yaxis=dict(range=[0, 5], title="", showticklabels=False,
                                                fixedrange=True),
                                     height=320, **CHART_LAYOUT)
                st.plotly_chart(fig_cs, use_container_width=True, key="plab_dd_cs")
            else:
                st.info("< 3 called strikes")
        with hm3:
            ip_d = pt_data[(pt_data["PitchCall"] == "InPlay") & pt_data["ExitSpeed"].notna()]
            weak = ip_d[ip_d["ExitSpeed"] < 85]
            st.caption(f"Weak Contact ({len(weak)})")
            if len(weak) >= 3:
                fig_wk = go.Figure(go.Histogram2d(
                    x=weak["PlateLocSide"], y=weak["PlateLocHeight"],
                    nbinsx=10, nbinsy=10, colorscale="Greens", showscale=False,
                ))
                add_strike_zone(fig_wk)
                fig_wk.update_layout(xaxis=dict(range=[-2.5, 2.5], title="", showticklabels=False,
                                                fixedrange=True, scaleanchor="y"),
                                     yaxis=dict(range=[0, 5], title="", showticklabels=False,
                                                fixedrange=True),
                                     height=320, **CHART_LAYOUT)
                st.plotly_chart(fig_wk, use_container_width=True, key="plab_dd_weak")
            else:
                st.info("< 3 weak contacts")




def page_pitching(data):
    pitching = filter_davidson(data, "pitcher")
    if pitching.empty:
        st.warning("No pitching data found.")
        return

    pitchers = sorted(pitching["Pitcher"].unique())
    c1, c2 = st.columns([2, 3])
    with c1:
        pitcher = st.selectbox("Select Pitcher", pitchers, format_func=display_name, key="pitching_pitcher")
    with c2:
        all_seasons = get_all_seasons()
        season_filter = st.multiselect("Season", all_seasons, default=all_seasons, key="pitching_season")

    all_pitcher_stats = compute_pitcher_stats_pop(season_filter=season_filter)
    pr = None
    if all_pitcher_stats.empty:
        st.info("Population stats unavailable — showing team data only.")
        all_pitcher_stats = pd.DataFrame()
    else:
        if pitcher in all_pitcher_stats["Pitcher"].values:
            pr = all_pitcher_stats[all_pitcher_stats["Pitcher"] == pitcher].iloc[0]
    pdf_raw = pitching[(pitching["Pitcher"] == pitcher) & (pitching["Season"].isin(season_filter))]
    pdf = filter_minor_pitches(pdf_raw)
    if pdf.empty or len(pdf) < 20:
        st.warning("Not enough pitch data (need 20+).")
        return

    if pr is None:
        # Fallback to pitcher-local stats if population stats missing
        pr_local = compute_pitcher_stats(pdf, season_filter=None)
        if not pr_local.empty:
            pr = pr_local.iloc[0]

    # Player header
    jersey = JERSEY.get(pitcher, "")
    pos = POSITION.get(pitcher, "")
    throws = safe_mode(pdf["PitcherThrows"], "")
    thr = {"Right": "R", "Left": "L"}.get(throws, throws)
    total_pitches = len(pdf)
    pa_faced = int(_safe_pr(pr, "PA") or 0) if pr is not None and "PA" in pr else 0
    player_header(pitcher, jersey, pos,
                  f"{pos}  |  Throws: {thr}  |  Davidson Wildcats",
                  f"{total_pitches} pitches  |  {pa_faced} PA faced  |  "
                  f"Seasons: {', '.join(str(int(s)) for s in sorted(season_filter))}")

    # Compute Stuff+ and Command+ once for both tabs
    stuff_df = _compute_stuff_plus(pdf)
    cmd_df = _compute_command_plus(pdf, data)

    tab_card, tab_lab = st.tabs(["Pitcher Card", "Pitch Lab"])
    with tab_card:
        _pitcher_card_content(data, pitcher, season_filter, pdf, stuff_df, pr, all_pitcher_stats, cmd_df=cmd_df)
    with tab_lab:
        _pitch_lab_page(data, pitcher, season_filter, pdf, stuff_df, pr, all_pitcher_stats, cmd_df=cmd_df)




def _pitching_lab_content(data, pitcher, season_filter, pdf, stuff_df,
                          tab_stuff, tab_tunnel, tab_seq, tab_loc,
                          tab_sim, tab_cmd):
    """Render the Pitch Design Lab tabs. Called from page_pitching()."""
    if stuff_df is None or "StuffPlus" not in stuff_df.columns:
        with tab_stuff:
            st.error("Could not compute Stuff+ scores. Not enough data for this pitcher.")
        return

    # Pre-compute tunnel/sequence data with <5% usage removed
    pdf_tunnel = filter_minor_pitches(pdf, min_pct=MIN_PITCH_USAGE_PCT)
    if pdf_tunnel.empty:
        pdf_tunnel = pdf
    tunnel_pop = build_tunnel_population_pop()
    tunnel_df = _compute_tunnel_score(pdf_tunnel, tunnel_pop=tunnel_pop)

    # ═══════════════════════════════════════════
    # TAB 1: STUFF+ GRADES
    # ═══════════════════════════════════════════
    with tab_stuff:
        section_header("Stuff+ Overview")
        st.caption("Stuff+ measures pitch quality based on velocity, movement, extension, and approach angle. "
                   "100 = league average, 110+ = plus pitch, 120+ = elite.")

        # Summary table
        arsenal_summary = stuff_df.groupby("TaggedPitchType").agg(
            Pitches=("StuffPlus", "count"),
            StuffPlus=("StuffPlus", "mean"),
            Velo=("RelSpeed", "mean"),
            MaxVelo=("RelSpeed", "max"),
            SpinRate=("SpinRate", "mean"),
            IVB=("InducedVertBreak", "mean"),
            HB=("HorzBreak", "mean"),
            Extension=("Extension", "mean"),
            VAA=("VertApprAngle", "mean"),
        ).sort_values("StuffPlus", ascending=False)
        arsenal_summary.columns = ["Pitches", "Stuff+", "Avg Velo", "Max Velo", "Spin Rate",
                                    "IVB (in)", "HB (in)", "Ext (ft)", "VAA"]

        # Color the Stuff+ values
        def style_stuff(val):
            if val >= 120:
                return "background-color: #be0000; color: white; font-weight: bold"
            elif val >= 110:
                return "background-color: #d22d49; color: white; font-weight: bold"
            elif val >= 105:
                return "background-color: #ee7e1e; color: white; font-weight: bold"
            elif val >= 95:
                return "background-color: #9e9e9e; color: white"
            elif val >= 90:
                return "background-color: #3d7dab; color: white"
            else:
                return "background-color: #14365d; color: white"

        formatted = arsenal_summary.copy()
        for c in ["Stuff+", "Avg Velo", "Max Velo", "Spin Rate", "IVB (in)", "HB (in)", "Ext (ft)", "VAA"]:
            if c in formatted.columns:
                formatted[c] = formatted[c].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
        formatted["Pitches"] = formatted["Pitches"].astype(int)
        st.dataframe(formatted, use_container_width=True)

        # Savant-style percentile bars for Stuff+
        all_stuff = _compute_stuff_plus_all(data)
        if "StuffPlus" in all_stuff.columns:
            section_header("Stuff+ Percentile Rankings (vs All Pitchers in Database)")
            metrics = []
            for pt in arsenal_summary.index:
                my_val = stuff_df[stuff_df["TaggedPitchType"] == pt]["StuffPlus"].mean()
                all_pt_vals = all_stuff[all_stuff["TaggedPitchType"] == pt]["StuffPlus"]
                if len(all_pt_vals) > 5:
                    pctl = percentileofscore(all_pt_vals.dropna(), my_val, kind="rank")
                    metrics.append((pt, my_val, pctl, ".0f", True))
            if metrics:
                render_savant_percentile_section(metrics)

        # Stuff+ distribution violin plot
        section_header("Stuff+ Distribution by Pitch")
        fig_dist = go.Figure()
        for pt in sorted(stuff_df["TaggedPitchType"].unique()):
            pt_vals = stuff_df[stuff_df["TaggedPitchType"] == pt]["StuffPlus"]
            color = PITCH_COLORS.get(pt, "#888")
            fig_dist.add_trace(go.Violin(
                y=pt_vals, name=pt, box_visible=True, meanline_visible=True,
                fillcolor=color, line_color=color, opacity=0.7,
            ))
        fig_dist.add_hline(y=100, line_dash="dash", line_color="#888",
                          annotation_text="League Avg (100)", annotation_position="top left")
        fig_dist.update_layout(
            showlegend=False, height=400,
            yaxis_title="Stuff+",
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Stuff+ over time (rolling)
        section_header("Stuff+ Trend Over Time")
        stuff_time = stuff_df.dropna(subset=["Date"]).sort_values("Date")
        if len(stuff_time) > 20:
            window = st.slider("Rolling window", 10, 50, 25, key="pdl_stuff_window")
            fig_trend = go.Figure()
            for pt in sorted(stuff_time["TaggedPitchType"].unique()):
                pt_data = stuff_time[stuff_time["TaggedPitchType"] == pt].copy()
                pt_data["StuffRolling"] = pt_data["StuffPlus"].rolling(window, min_periods=5).mean()
                color = PITCH_COLORS.get(pt, "#888")
                fig_trend.add_trace(go.Scatter(
                    x=pt_data["Date"], y=pt_data["StuffRolling"],
                    mode="lines", name=pt, line=dict(color=color, width=2),
                ))
            fig_trend.add_hline(y=100, line_dash="dash", line_color="#888")
            fig_trend.update_layout(
                height=380, xaxis_title="Date", yaxis_title="Stuff+ (rolling avg)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02), **CHART_LAYOUT,
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 2: PITCH TUNNELS
    # ═══════════════════════════════════════════
    with tab_tunnel:
        section_header("Pitch Tunnel Analysis")
        st.caption("Tunnel Score = percentile rank vs all pitchers in the database for the same pair type. "
                   "A (top 20%) → F (bottom 20%). Based on physics-modeled flight paths at 280ms commit point. "
                   "Pairs are unordered — Changeup/Fastball and Fastball/Changeup get the same tunnel grade "
                   "(tunneling measures visual deception, which is symmetric; sequence *order* effects are captured in the Sequencing tab).")

        if tunnel_df.empty:
            st.info("Need 2+ pitch types to compute tunnels.")
        else:
            # Grade color badges
            grade_colors = {"A": "#22c55e", "B": "#3b82f6", "C": "#f59e0b", "D": "#f97316", "F": "#ef4444"}

            # Summary grade cards at top
            st.markdown("#### Tunnel Pair Grades")
            grade_cols = st.columns(min(len(tunnel_df), 5))
            for idx, (_, row) in enumerate(tunnel_df.head(5).iterrows()):
                with grade_cols[idx]:
                    gc = grade_colors.get(row["Grade"], "#888")
                    whiff_str = f' &middot; Whiff%: {row["Pair Whiff%"]:.1f}' if pd.notna(row.get("Pair Whiff%")) else ""
                    st.markdown(
                        f'<div style="text-align:center;padding:10px;border-radius:8px;border:2px solid {gc};'
                        f'background:{gc}15;margin:2px;">'
                        f'<span style="font-size:28px;font-weight:bold;color:{gc};">{row["Grade"]}</span><br>'
                        f'<span style="font-size:13px;">{row["Pitch A"]} + {row["Pitch B"]}</span><br>'
                        f'<span style="font-size:12px;color:#666;">Score: {row["Tunnel Score"]}{whiff_str}</span>'
                        f'</div>', unsafe_allow_html=True)

            st.markdown("")

            # Detailed table (show key columns)
            display_cols = ["Pitch A", "Pitch B", "Grade", "Tunnel Score", "Pair Whiff%",
                            "Release Sep (in)", "Commit Sep (in)", "Plate Sep (in)",
                            "Velo Gap (mph)", "Move Diff (in)", "Rel Angle Sep (°)"]
            st.dataframe(tunnel_df[display_cols], use_container_width=True, hide_index=True)
            if "Pair Whiff%" in tunnel_df.columns:
                corr_df = tunnel_df.dropna(subset=["Tunnel Score", "Pair Whiff%"])
                if len(corr_df) >= 3:
                    corr = corr_df["Tunnel Score"].corr(corr_df["Pair Whiff%"])
                    if pd.notna(corr):
                        st.caption(f"Tunnel Score vs Pair Whiff% correlation (this pitcher): {corr:.2f}")

            # Tunnel visualization — release point overlay + plate location
            section_header("Release Point Overlay")
            st.caption("Pitches that release from the same spot but end up in different locations = great tunnel")
            fig_rel = go.Figure()
            for pt in sorted(pdf["TaggedPitchType"].unique()):
                pt_data = pdf[pdf["TaggedPitchType"] == pt].dropna(subset=["RelSide", "RelHeight"])
                color = PITCH_COLORS.get(pt, "#888")
                fig_rel.add_trace(go.Scatter(
                    x=pt_data["RelSide"], y=pt_data["RelHeight"],
                    mode="markers", name=pt,
                    marker=dict(color=color, size=6, opacity=0.4),
                ))
                # Add mean crosshair
                fig_rel.add_trace(go.Scatter(
                    x=[pt_data["RelSide"].mean()], y=[pt_data["RelHeight"].mean()],
                    mode="markers", name=f"{pt} avg", showlegend=False,
                    marker=dict(color=color, size=16, symbol="x-thin", line=dict(width=3, color=color)),
                ))
            fig_rel.update_layout(
                xaxis_title="Release Side (ft)", yaxis_title="Release Height (ft)",
                height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                **CHART_LAYOUT,
            )
            fig_rel.update_xaxes(scaleanchor="y", scaleratio=1)
            st.plotly_chart(fig_rel, use_container_width=True)

            # Side-by-side: tunnel pair visualization
            if len(tunnel_df) > 0:
                section_header("Best Tunnel Pair Visualization")
                best_pair = tunnel_df.iloc[0]
                pair_a, pair_b = best_pair["Pitch A"], best_pair["Pitch B"]
                st.markdown(f"**{pair_a}** + **{pair_b}** — Grade: **{best_pair['Grade']}** | Score: **{best_pair['Tunnel Score']}**")

                c1, c2 = st.columns(2)
                with c1:
                    # Movement profile of the pair
                    pair_data = pdf[pdf["TaggedPitchType"].isin([pair_a, pair_b])].dropna(
                        subset=["HorzBreak", "InducedVertBreak"])
                    fig_move = go.Figure()
                    for pt in [pair_a, pair_b]:
                        d = pair_data[pair_data["TaggedPitchType"] == pt]
                        color = PITCH_COLORS.get(pt, "#888")
                        fig_move.add_trace(go.Scatter(
                            x=d["HorzBreak"], y=d["InducedVertBreak"],
                            mode="markers", name=pt,
                            marker=dict(color=color, size=7, opacity=0.5),
                        ))
                    fig_move.update_layout(
                        title="Movement Profile", xaxis_title="Horizontal Break (in)",
                        yaxis_title="Induced Vert Break (in)", height=350, **CHART_LAYOUT,
                    )
                    fig_move.add_hline(y=0, line_color="#ccc")
                    fig_move.add_vline(x=0, line_color="#ccc")
                    st.plotly_chart(fig_move, use_container_width=True)

                with c2:
                    # Plate location of the pair
                    fig_loc = go.Figure()
                    for pt in [pair_a, pair_b]:
                        d = pdf[(pdf["TaggedPitchType"] == pt)].dropna(subset=["PlateLocSide", "PlateLocHeight"])
                        color = PITCH_COLORS.get(pt, "#888")
                        fig_loc.add_trace(go.Scatter(
                            x=d["PlateLocSide"], y=d["PlateLocHeight"],
                            mode="markers", name=pt,
                            marker=dict(color=color, size=7, opacity=0.5),
                        ))
                    add_strike_zone(fig_loc)
                    fig_loc.update_layout(
                        title="Plate Locations", xaxis_title="Plate Side (ft)",
                        yaxis_title="Plate Height (ft)", height=350,
                        xaxis=dict(range=[-2.5, 2.5], scaleanchor="y"), yaxis=dict(range=[0, 5]),
                        **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_loc, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 3: SEQUENCING
    # ═══════════════════════════════════════════
    with tab_seq:
        section_header("Pitch Sequencing Analysis")
        st.caption("Shows what happens when Pitch B follows Pitch A in the same at-bat. "
                   "Use this to find your most effective pitch combinations.")

        pair_df = _compute_pitch_pair_results(pdf_tunnel, data, tunnel_df=tunnel_df)
        if pair_df.empty:
            st.info("Not enough sequential pitch data.")
        else:
            # Sequencing effectiveness table
            st.dataframe(pair_df, use_container_width=True, hide_index=True)

            # Transition matrix heatmap
            section_header("Pitch Transition Matrix")
            st.caption("How often does each pitch follow another? Heat = frequency")
            sort_cols = [c for c in ["GameID", "Batter", "PAofInning", "PitchNo"] if c in pdf_tunnel.columns]
            if len(sort_cols) >= 2:
                pdf_s = pdf_tunnel.sort_values(sort_cols).copy()
                pdf_s["NextPitch"] = pdf_s.groupby(["GameID", "Batter", "PAofInning"])["TaggedPitchType"].shift(-1)
                trans = pdf_s.dropna(subset=["NextPitch"])
                if not trans.empty:
                    matrix = pd.crosstab(trans["TaggedPitchType"], trans["NextPitch"], normalize="index") * 100
                    fig_matrix = px.imshow(
                        matrix.round(1), text_auto=".0f", color_continuous_scale="RdBu_r",
                        labels=dict(x="Next Pitch", y="Current Pitch", color="Frequency %"),
                        aspect="auto",
                    )
                    fig_matrix.update_layout(height=380, **CHART_LAYOUT)
                    st.plotly_chart(fig_matrix, use_container_width=True)

            # Whiff% heatmap by sequence
            section_header("Whiff% by Pitch Sequence")
            if not pair_df.empty:
                whiff_pivot = pair_df.pivot_table(index="Setup Pitch", columns="Follow Pitch",
                                                  values="Whiff%", aggfunc="first")
                if not whiff_pivot.empty:
                    fig_whiff = px.imshow(
                        whiff_pivot.fillna(0).round(1), text_auto=".0f",
                        color_continuous_scale="YlOrRd",
                        labels=dict(x="Follow-Up Pitch", y="Setup Pitch", color="Whiff%"),
                        aspect="auto",
                    )
                    # Overlay tunnel grades on whiff heatmap
                    if "Tunnel" in pair_df.columns:
                        tunnel_pivot = pair_df.pivot_table(index="Setup Pitch", columns="Follow Pitch",
                                                            values="Tunnel", aggfunc="first")
                        tunnel_pivot = tunnel_pivot.reindex(index=whiff_pivot.index,
                                                            columns=whiff_pivot.columns)
                        custom_text = []
                        for ridx in whiff_pivot.index:
                            row_text = []
                            for cidx in whiff_pivot.columns:
                                w_val = whiff_pivot.loc[ridx, cidx]
                                t_val = tunnel_pivot.loc[ridx, cidx] if ridx in tunnel_pivot.index and cidx in tunnel_pivot.columns else "-"
                                if pd.isna(t_val):
                                    t_val = "-"
                                w_str = f"{w_val:.0f}%" if not pd.isna(w_val) and w_val > 0 else "0%"
                                row_text.append(f"{w_str} [{t_val}]")
                            custom_text.append(row_text)
                        fig_whiff.update_traces(text=custom_text, texttemplate="%{text}")
                    fig_whiff.update_layout(height=380, **CHART_LAYOUT)
                    st.plotly_chart(fig_whiff, use_container_width=True)

            # First-pitch tendencies
            section_header("Count-State Pitch Selection")
            counts_of_interest = [("0", "0"), ("0", "2"), ("1", "2"), ("2", "0"), ("3", "1"), ("3", "2")]
            count_rows = []
            for b, s in counts_of_interest:
                count_data = pdf[(pdf["Balls"].astype(str) == b) & (pdf["Strikes"].astype(str) == s)]
                if len(count_data) >= 3:
                    usage = count_data["TaggedPitchType"].value_counts(normalize=True) * 100
                    for pt, pct in usage.items():
                        count_rows.append({"Count": f"{b}-{s}", "Pitch": pt, "Usage%": round(pct, 1)})
            if count_rows:
                count_df = pd.DataFrame(count_rows)
                fig_count = px.bar(
                    count_df, x="Count", y="Usage%", color="Pitch",
                    color_discrete_map=PITCH_COLORS, barmode="stack",
                )
                fig_count.update_layout(
                    height=380, yaxis_title="Usage %", xaxis_title="Count",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    **CHART_LAYOUT,
                )
                st.plotly_chart(fig_count, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 4: LOCATION LAB
    # ═══════════════════════════════════════════
    with tab_loc:
        section_header("Location Optimization Lab")
        st.caption("Find the exact locations where each pitch generates the most whiffs, weakest contact, and highest called strike rates.")

        pitch_type_sel = st.selectbox("Select Pitch", sorted(pdf["TaggedPitchType"].unique()), key="pdl_loc_pitch")
        pt_data = pdf[pdf["TaggedPitchType"] == pitch_type_sel].copy()

        if len(pt_data) < 10:
            st.info("Not enough pitches of this type to analyze locations.")
        else:
            c1, c2, c3 = st.columns(3)

            # Whiff heatmap
            with c1:
                whiff_data = pt_data[pt_data["PitchCall"] == "StrikeSwinging"]
                section_header("Whiff Locations")
                if len(whiff_data) >= 3:
                    fig = go.Figure(go.Histogram2d(
                        x=whiff_data["PlateLocSide"], y=whiff_data["PlateLocHeight"],
                        nbinsx=10, nbinsy=10,
                        colorscale="YlOrRd", showscale=False,
                    ))
                    add_strike_zone(fig)
                    fig.update_layout(
                        xaxis=dict(range=[-2.5, 2.5], title="", showticklabels=False,
                                   fixedrange=True, scaleanchor="y"),
                        yaxis=dict(range=[0, 5], title="", showticklabels=False,
                                   fixedrange=True),
                        height=320, **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"{len(whiff_data)} whiffs")
                else:
                    st.info("Not enough whiffs")

            # Called strike heatmap
            with c2:
                cs_data = pt_data[pt_data["PitchCall"] == "StrikeCalled"]
                section_header("Called Strike Locations")
                if len(cs_data) >= 3:
                    fig = go.Figure(go.Histogram2d(
                        x=cs_data["PlateLocSide"], y=cs_data["PlateLocHeight"],
                        nbinsx=10, nbinsy=10,
                        colorscale="Blues", showscale=False,
                    ))
                    add_strike_zone(fig)
                    fig.update_layout(
                        xaxis=dict(range=[-2.5, 2.5], title="", showticklabels=False,
                                   fixedrange=True, scaleanchor="y"),
                        yaxis=dict(range=[0, 5], title="", showticklabels=False,
                                   fixedrange=True),
                        height=320, **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"{len(cs_data)} called strikes")
                else:
                    st.info("Not enough called strikes")

            # Weak contact heatmap (EV < 85 on balls in play)
            with c3:
                ip_data = pt_data[(pt_data["PitchCall"] == "InPlay") & pt_data["ExitSpeed"].notna()]
                weak = ip_data[ip_data["ExitSpeed"] < 85]
                section_header("Weak Contact Locations")
                if len(weak) >= 3:
                    fig = go.Figure(go.Histogram2d(
                        x=weak["PlateLocSide"], y=weak["PlateLocHeight"],
                        nbinsx=10, nbinsy=10,
                        colorscale="Greens", showscale=False,
                    ))
                    add_strike_zone(fig)
                    fig.update_layout(
                        xaxis=dict(range=[-2.5, 2.5], title="", showticklabels=False,
                                   fixedrange=True, scaleanchor="y"),
                        yaxis=dict(range=[0, 5], title="", showticklabels=False,
                                   fixedrange=True),
                        height=320, **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"{len(weak)} weak contacts (EV < 85)")
                else:
                    st.info("Not enough weak contact")

            # Zone-quadrant breakdown
            section_header("Zone Quadrant Performance")
            st.caption("Strike zone split into 9 regions — showing effectiveness in each")
            loc_data = pt_data.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
            if len(loc_data) >= 20:
                # Define 9 zones (3x3 grid within strike zone)
                h_edges = [-0.83, -0.28, 0.28, 0.83]
                v_edges = [1.5, 2.17, 2.83, 3.5]
                zone_labels = [
                    ["Down-In", "Down-Mid", "Down-Away"],
                    ["Mid-In", "Heart", "Mid-Away"],
                    ["Up-In", "Up-Mid", "Up-Away"],
                ]
                zone_rows = []
                for vi in range(3):
                    for hi in range(3):
                        mask = (
                            (loc_data["PlateLocSide"] >= h_edges[hi]) &
                            (loc_data["PlateLocSide"] < h_edges[hi + 1]) &
                            (loc_data["PlateLocHeight"] >= v_edges[vi]) &
                            (loc_data["PlateLocHeight"] < v_edges[vi + 1])
                        )
                        zone = loc_data[mask]
                        if len(zone) < 3:
                            continue
                        swings = zone[zone["PitchCall"].isin(SWING_CALLS)]
                        whiffs_z = zone[zone["PitchCall"] == "StrikeSwinging"]
                        ip_z = zone[(zone["PitchCall"] == "InPlay") & zone["ExitSpeed"].notna()]
                        zone_rows.append({
                            "Zone": zone_labels[vi][hi],
                            "Pitches": len(zone),
                            "Whiff%": round(len(whiffs_z) / max(len(swings), 1) * 100, 1),
                            "Avg EV": round(ip_z["ExitSpeed"].mean(), 1) if len(ip_z) > 0 else np.nan,
                            "Usage%": round(len(zone) / len(loc_data) * 100, 1),
                        })
                if zone_rows:
                    zone_df = pd.DataFrame(zone_rows)
                    st.dataframe(zone_df, use_container_width=True, hide_index=True)


    # ═══════════════════════════════════════════
    # TAB 5: HITTER'S EYE — PITCH FLIGHT SIMULATOR
    # ═══════════════════════════════════════════
    with tab_sim:
        section_header("Hitter's Eye — Pitch Simulator")
        st.caption("See pitches from the batter's perspective. The 3D flight path shows how each pitch "
                   "travels from the release point to the plate, and the overlay view reveals why certain "
                   "pitch pairs are so hard to distinguish.")

        sim_pitches = sorted(pdf["TaggedPitchType"].unique())
        sim_selected = st.multiselect(
            "Select pitches to overlay (pick 2+ to see tunnel effect)",
            sim_pitches, default=sim_pitches[:min(3, len(sim_pitches))],
            key="pdl_sim_pitches",
        )
        if not sim_selected:
            st.info("Select at least one pitch type above.")
        else:
            # Compute average flight path for each pitch type using physics model
            # Release point → plate (60.5 ft) with break applied as quadratic curve
            MOUND_DIST = 60.5  # ft from rubber to plate

            flight_data = {}
            for pt in sim_selected:
                ptd = pdf[pdf["TaggedPitchType"] == pt].dropna(subset=["RelSpeed"])
                if ptd.empty:
                    continue
                velo = ptd["RelSpeed"].mean()  # mph
                rel_h = ptd["RelHeight"].mean() if "RelHeight" in ptd.columns and ptd["RelHeight"].notna().any() else 5.5
                rel_s = ptd["RelSide"].mean() if "RelSide" in ptd.columns and ptd["RelSide"].notna().any() else 0.0
                ext = ptd["Extension"].mean() if "Extension" in ptd.columns and ptd["Extension"].notna().any() else 6.0
                loc_h = ptd["PlateLocHeight"].mean() if "PlateLocHeight" in ptd.columns and ptd["PlateLocHeight"].notna().any() else 2.5
                loc_s = ptd["PlateLocSide"].mean() if "PlateLocSide" in ptd.columns and ptd["PlateLocSide"].notna().any() else 0.0
                ivb = ptd["InducedVertBreak"].mean() if "InducedVertBreak" in ptd.columns and ptd["InducedVertBreak"].notna().any() else 0.0
                hb = ptd["HorzBreak"].mean() if "HorzBreak" in ptd.columns and ptd["HorzBreak"].notna().any() else 0.0

                # Flight path: 30 points from release to plate
                n_pts = 30
                actual_dist = MOUND_DIST - ext
                t_total = actual_dist / (velo * 5280 / 3600)  # seconds of flight

                # Parametric path: t goes 0→1
                ts = np.linspace(0, 1, n_pts)
                # Distance from pitcher: linear
                z_pts = ext + ts * actual_dist  # feet from rubber

                # Height: linear interpolation + quadratic gravity + IVB
                gravity_drop = 0.5 * 32.17 * (ts * t_total)**2  # feet of gravity drop
                ivb_lift = (ivb / 12.0) * ts**2  # IVB counteracts gravity (inches→feet)
                y_pts = rel_h + (loc_h - rel_h) * ts - gravity_drop + ivb_lift
                # Ensure endpoint matches plate location
                y_correction = loc_h - y_pts[-1]
                y_pts = y_pts + y_correction * ts

                # Horizontal: linear + break curve
                hb_curve = (hb / 12.0) * ts**2  # horizontal break applied quadratically
                x_pts = rel_s + (loc_s - rel_s) * ts + hb_curve
                x_correction = loc_s - x_pts[-1]
                x_pts = x_pts + x_correction * ts

                flight_data[pt] = {
                    "x": x_pts, "y": y_pts, "z": z_pts,
                    "velo": velo, "rel_h": rel_h, "rel_s": rel_s,
                    "loc_h": loc_h, "loc_s": loc_s,
                    "ivb": ivb, "hb": hb, "ext": ext,
                    "time_ms": t_total * 1000,
                }

            if flight_data:
                col_3d, col_front = st.columns(2)

                # ── 3D Flight Path (side/overhead view) ──
                with col_3d:
                    section_header("3D Pitch Flight Path")
                    fig_3d = go.Figure()
                    for pt, fd in flight_data.items():
                        color = PITCH_COLORS.get(pt, "#888")
                        # Flight path line
                        fig_3d.add_trace(go.Scatter3d(
                            x=fd["z"], y=fd["x"], z=fd["y"],
                            mode="lines+markers",
                            name=f'{pt} ({fd["velo"]:.0f} mph)',
                            line=dict(color=color, width=6),
                            marker=dict(size=2, color=color),
                        ))
                        # Release point marker
                        fig_3d.add_trace(go.Scatter3d(
                            x=[fd["z"][0]], y=[fd["x"][0]], z=[fd["y"][0]],
                            mode="markers", showlegend=False,
                            marker=dict(size=8, color=color, symbol="diamond"),
                        ))
                        # Plate arrival marker
                        fig_3d.add_trace(go.Scatter3d(
                            x=[fd["z"][-1]], y=[fd["x"][-1]], z=[fd["y"][-1]],
                            mode="markers", showlegend=False,
                            marker=dict(size=10, color=color, symbol="circle",
                                        line=dict(width=2, color="white")),
                        ))

                    # Draw strike zone at plate (z = 60.5)
                    sz_x = [-0.83, 0.83, 0.83, -0.83, -0.83]
                    sz_y = [1.5, 1.5, 3.5, 3.5, 1.5]
                    sz_z = [MOUND_DIST] * 5
                    fig_3d.add_trace(go.Scatter3d(
                        x=sz_z, y=sz_x, z=sz_y,
                        mode="lines", showlegend=False,
                        line=dict(color="#333", width=4),
                    ))
                    fig_3d.update_layout(
                        height=500,
                        scene=dict(
                            xaxis=dict(title="Distance (ft)", range=[0, 65]),
                            yaxis=dict(title="Horizontal (ft)", range=[-3, 3]),
                            zaxis=dict(title="Height (ft)", range=[0, 8]),
                            camera=dict(eye=dict(x=1.5, y=1.2, z=0.5)),
                            aspectmode="manual",
                            aspectratio=dict(x=3, y=1, z=1),
                        ),
                        margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor="white",
                        font=dict(size=10, color="#000000", family="Inter, Arial, sans-serif"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    font=dict(color="#000000")),
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

                # ── Batter's POV (front view at plate) ──
                with col_front:
                    section_header("Batter's View (at the plate)")
                    fig_front = go.Figure()

                    # Show pitch trajectories as the batter sees them
                    for pt, fd in flight_data.items():
                        color = PITCH_COLORS.get(pt, "#888")
                        # Show the last 60% of trajectory (what hitter can react to)
                        start_idx = int(len(fd["x"]) * 0.3)

                        # Increasing marker size = pitch getting closer
                        sizes = np.linspace(3, 18, len(fd["x"][start_idx:]))
                        opacities = np.linspace(0.15, 0.9, len(fd["x"][start_idx:]))

                        # Trail path
                        fig_front.add_trace(go.Scatter(
                            x=fd["x"][start_idx:], y=fd["y"][start_idx:],
                            mode="markers+lines",
                            name=f'{pt} ({fd["velo"]:.0f})',
                            marker=dict(size=sizes, color=color, opacity=0.5),
                            line=dict(color=color, width=2, dash="dot"),
                        ))
                        # Final plate location (big marker)
                        fig_front.add_trace(go.Scatter(
                            x=[fd["loc_s"]], y=[fd["loc_h"]],
                            mode="markers+text",
                            showlegend=False,
                            text=[f'{fd["velo"]:.0f}'],
                            textposition="top center",
                            textfont=dict(size=10, color=color),
                            marker=dict(size=22, color=color, opacity=0.85,
                                        line=dict(width=2, color="white")),
                        ))

                    # Strike zone
                    fig_front.add_shape(
                        type="rect", x0=-ZONE_SIDE, x1=ZONE_SIDE, y0=ZONE_HEIGHT_BOT, y1=ZONE_HEIGHT_TOP,
                        line=dict(color="#333", width=2.5),
                        fillcolor="rgba(0,0,0,0.03)",
                    )
                    # Home plate
                    fig_front.add_shape(
                        type="path",
                        path="M -0.71 0.3 L 0.71 0.3 L 0.83 0.15 L 0 0 L -0.83 0.15 Z",
                        line=dict(color="#666", width=1.5),
                        fillcolor="rgba(200,200,200,0.3)",
                    )
                    fig_front.update_layout(
                        height=500,
                        xaxis=dict(range=[-2.5, 2.5], title="Horizontal (ft)",
                                   zeroline=False, showgrid=True, gridcolor="#eee", scaleanchor="y"),
                        yaxis=dict(range=[-0.5, 5.5], title="Height (ft)",
                                   zeroline=False, showgrid=True, gridcolor="#eee"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_front, use_container_width=True)

                # ── Decision Point Analysis ──
                section_header("Pitch Decision Timeline")
                st.caption("How much time does the hitter have to decide? At what point do pitches diverge?")

                timeline_rows = []
                for pt, fd in flight_data.items():
                    t_ms = fd["time_ms"]
                    # Decision point: ~167ms before plate (average human reaction time)
                    react_ms = 167
                    decision_dist = (react_ms / t_ms) * (MOUND_DIST - fd["ext"])
                    commit_dist = MOUND_DIST - decision_dist  # distance from rubber where hitter must commit

                    # At the decision point, where is the pitch?
                    frac = 1 - (react_ms / t_ms)
                    idx = min(int(frac * len(fd["x"])), len(fd["x"]) - 1)
                    dec_h = fd["y"][idx]
                    dec_s = fd["x"][idx]

                    timeline_rows.append({
                        "Pitch": pt,
                        "Velocity": f'{fd["velo"]:.0f} mph',
                        "Flight Time": f'{t_ms:.0f} ms',
                        "Decision Point": f'{commit_dist:.1f} ft from plate',
                        "Height at Decision": f'{dec_h:.2f} ft',
                        "Side at Decision": f'{dec_s:.2f} ft',
                        "React Window": f'{t_ms - react_ms:.0f} ms',
                    })
                timeline_df = pd.DataFrame(timeline_rows)
                st.dataframe(timeline_df, use_container_width=True, hide_index=True)

                # Tunnel divergence analysis
                if len(sim_selected) >= 2:
                    section_header("Tunnel Divergence Analysis")
                    st.caption("At the commit point (~280ms before plate), how far apart are the pitches? "
                               "Less separation = better deception.")

                    pts_list = list(flight_data.keys())
                    div_rows = []
                    for i in range(len(pts_list)):
                        for j in range(i + 1, len(pts_list)):
                            a, b = flight_data[pts_list[i]], flight_data[pts_list[j]]
                            # Compare at multiple checkpoints
                            for check_name, frac in [("Release", 0.0), ("1/3 Way", 0.33),
                                                      ("Commit Point", 0.6), ("2/3 Way", 0.67), ("Plate", 1.0)]:
                                idx_a = min(int(frac * (len(a["x"]) - 1)), len(a["x"]) - 1)
                                idx_b = min(int(frac * (len(b["x"]) - 1)), len(b["x"]) - 1)
                                h_sep = abs(a["y"][idx_a] - b["y"][idx_b]) * 12  # inches
                                s_sep = abs(a["x"][idx_a] - b["x"][idx_b]) * 12  # inches
                                total_sep = np.sqrt(h_sep**2 + s_sep**2)
                                div_rows.append({
                                    "Pair": f'{pts_list[i]} vs {pts_list[j]}',
                                    "Checkpoint": check_name,
                                    "Vertical Sep (in)": round(h_sep, 1),
                                    "Horizontal Sep (in)": round(s_sep, 1),
                                    "Total Sep (in)": round(total_sep, 1),
                                })

                    div_df = pd.DataFrame(div_rows)
                    # Show as line chart — separation over distance
                    fig_div = go.Figure()
                    for pair in div_df["Pair"].unique():
                        pair_data = div_df[div_df["Pair"] == pair]
                        checkpoints = ["Release", "1/3 Way", "Commit Point", "2/3 Way", "Plate"]
                        pair_ordered = pair_data.set_index("Checkpoint").reindex(checkpoints)
                        fig_div.add_trace(go.Scatter(
                            x=checkpoints,
                            y=pair_ordered["Total Sep (in)"].values,
                            mode="lines+markers",
                            name=pair,
                            line=dict(width=3),
                            marker=dict(size=8),
                        ))
                    fig_div.add_hline(y=6, line_dash="dash", line_color="#cc0000",
                                     annotation_text="6 in (hard to distinguish)",
                                     annotation_position="top left",
                                     annotation_font=dict(color="#cc0000"))
                    fig_div.update_layout(
                        height=380,
                        yaxis_title="Separation (inches)",
                        xaxis_title="Pitch Flight Checkpoint",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_div, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 6: COMMAND+ SCORING
    # ═══════════════════════════════════════════
    with tab_cmd:
        section_header("Command+ Analysis")
        st.caption("Command+ measures how well a pitcher locates each pitch relative to optimal zones. "
                   "100 = average command, higher = more precise location control.")

        # Compute Command+ using shared helper
        cmd_df = _compute_command_plus(pdf, data)

        if cmd_df.empty:
            st.info("Not enough location data to compute Command+.")
        else:
            # Display table
            st.dataframe(cmd_df, use_container_width=True, hide_index=True)

            # Command+ percentile bars
            section_header("Command+ Percentile Rankings")
            metrics = []
            for _, row in cmd_df.iterrows():
                cmd_val = row["Command+"]
                # Map Command+ to percentile-like scale for the bar
                pctl_mapped = min(max((cmd_val - 80) * 2.5, 0), 100)
                metrics.append((row["Pitch"], cmd_val, pctl_mapped, ".0f", True))
            if metrics:
                render_savant_percentile_section(metrics)

            # Location scatter per pitch type with density ellipse
            section_header("Location Precision Map")
            st.caption("Each dot is a pitch. Tight clusters = good command. The crosshair shows the average location.")
            loc_pitch_sel = st.selectbox("Select Pitch", cmd_df["Pitch"].tolist(), key="pdl_cmd_pitch")
            loc_ptd = pdf[(pdf["TaggedPitchType"] == loc_pitch_sel)].dropna(
                subset=["PlateLocSide", "PlateLocHeight"])

            if len(loc_ptd) >= 5:
                # Color by outcome
                outcome_colors = {
                    "StrikeSwinging": "#be0000",
                    "StrikeCalled": "#2d7fc1",
                    "BallCalled": "#9e9e9e",
                    "FoulBall": "#ee7e1e",
                    "InPlay": "#1dbe3a",
                    "FoulBallNotFieldable": "#ee7e1e",
                    "FoulBallFieldable": "#ee7e1e",
                    "HitByPitch": "#333",
                    "BallIntentional": "#9e9e9e",
                    "BallinDirt": "#666",
                }
                loc_ptd = loc_ptd.copy()
                loc_ptd["Outcome"] = loc_ptd["PitchCall"].map(
                    lambda x: "Whiff" if x == "StrikeSwinging" else
                              "Called Strike" if x == "StrikeCalled" else
                              "Ball" if "Ball" in str(x) else
                              "Foul" if "Foul" in str(x) else
                              "In Play" if x == "InPlay" else "Other"
                )
                outcome_color_map = {
                    "Whiff": "#be0000", "Called Strike": "#2d7fc1",
                    "Ball": "#9e9e9e", "Foul": "#ee7e1e",
                    "In Play": "#1dbe3a", "Other": "#666",
                }
                fig_loc = px.scatter(
                    loc_ptd, x="PlateLocSide", y="PlateLocHeight",
                    color="Outcome", color_discrete_map=outcome_color_map,
                    opacity=0.6,
                )
                # Mean crosshair
                mean_s = loc_ptd["PlateLocSide"].mean()
                mean_h = loc_ptd["PlateLocHeight"].mean()
                fig_loc.add_trace(go.Scatter(
                    x=[mean_s], y=[mean_h], mode="markers", showlegend=False,
                    marker=dict(size=20, color=PITCH_COLORS.get(loc_pitch_sel, "#333"),
                                symbol="x-thin", line=dict(width=4)),
                ))
                # 1-sigma ellipse
                std_s = loc_ptd["PlateLocSide"].std()
                std_h = loc_ptd["PlateLocHeight"].std()
                theta = np.linspace(0, 2 * np.pi, 50)
                fig_loc.add_trace(go.Scatter(
                    x=mean_s + std_s * np.cos(theta),
                    y=mean_h + std_h * np.sin(theta),
                    mode="lines", showlegend=False,
                    line=dict(color=PITCH_COLORS.get(loc_pitch_sel, "#333"),
                              width=2, dash="dash"),
                ))
                add_strike_zone(fig_loc)
                fig_loc.update_layout(
                    height=480,
                    xaxis=dict(range=[-2.5, 2.5], title="Plate Side (ft)", scaleanchor="y"),
                    yaxis=dict(range=[0, 5.5], title="Plate Height (ft)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    **CHART_LAYOUT,
                )
                st.plotly_chart(fig_loc, use_container_width=True)


    # ═══════════════════════════════════════════




def _game_planning_content(data, pitcher=None, season_filter=None, pdf=None, key_prefix="gp"):
    """Pitch sequencing engine, count leverage analysis, and effective velocity.
    If pitcher/pdf provided, skip the pitcher selector (used from Pitching page)."""

    if pitcher is None or pdf is None:
        # Standalone mode: show pitcher selector
        dav_pitching = filter_davidson(data, role="pitcher")
        if dav_pitching.empty:
            st.warning("No Davidson pitching data found.")
            return

        pitchers = sorted(dav_pitching["Pitcher"].unique())
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            pitcher = st.selectbox("Select Pitcher", pitchers, format_func=display_name, key=f"{key_prefix}_pitcher")
        with col_sel2:
            seasons = sorted(dav_pitching["Season"].dropna().unique())
            season_filter = st.multiselect("Season", seasons, default=seasons, key=f"{key_prefix}_season")

        pdf = dav_pitching[dav_pitching["Pitcher"] == pitcher].copy()
        if season_filter:
            pdf = pdf[pdf["Season"].isin(season_filter)]
        if len(pdf) < 30:
            st.warning("Not enough pitches (need 30+).")
            return
        pdf = filter_minor_pitches(pdf)

        jersey = JERSEY.get(pitcher, "")
        pos = POSITION.get(pitcher, "")
        throws = safe_mode(pdf["PitcherThrows"], "")
        t_str = {"Right": "R", "Left": "L"}.get(throws, throws)

        player_header(pitcher, jersey, pos,
                      f"{pos} | Throws: {t_str} | Davidson Wildcats",
                      f"{len(pdf):,} pitches | Seasons: {', '.join(str(int(s)) for s in sorted(pdf['Season'].dropna().unique()))}")

    tab_seq, tab_count, tab_effv = st.tabs(["Sequencing + Tunnels", "Count Leverage", "Effective Velocity"])

    # ─── Tab: Pitch Sequencing + Tunnel Engine ──────────────
    with tab_seq:
        section_header("Pitch Sequencing + Tunnel Engine")
        st.caption("Sequence effectiveness combined with physics-based tunnel analysis — the best sequences are ones that tunnel well AND produce whiffs")

        # Compute tunnel scores for this pitcher
        tunnel_pop = build_tunnel_population_pop()
        tunnel_df = _compute_tunnel_score(pdf, tunnel_pop=tunnel_pop)
        tunnel_lookup = {}
        if not tunnel_df.empty:
            for _, tr in tunnel_df.iterrows():
                tunnel_lookup[(tr["Pitch A"], tr["Pitch B"])] = tr
                tunnel_lookup[(tr["Pitch B"], tr["Pitch A"])] = tr

        # Build sequence pairs
        sort_cols = [c for c in ["GameID", "PAofInning", "Inning", "PitchNo"] if c in pdf.columns]
        if len(sort_cols) >= 2:
            sdf = pdf.sort_values(sort_cols).copy()
            pa_cols = [c for c in ["GameID", "PAofInning", "Inning"] if c in pdf.columns]
            sdf["NextPitch"] = sdf.groupby(pa_cols)["TaggedPitchType"].shift(-1)
            sdf["NextCall"] = sdf.groupby(pa_cols)["PitchCall"].shift(-1)
            sdf["NextEV"] = sdf.groupby(pa_cols)["ExitSpeed"].shift(-1)
            pairs = sdf.dropna(subset=["NextPitch"])

            # ── Tunnel Overview Cards ──
            if not tunnel_df.empty:
                section_header("Tunnel Grades for Pitch Pairs")
                st.caption("Physics-based: pitches that look identical at the commit point (~280ms before plate) but diverge at the plate")
                grade_colors = {"A": "#2ca02c", "B": "#7cb342", "C": "#f7c631", "D": "#fe6100", "F": "#d22d49"}
                tun_cols = st.columns(min(len(tunnel_df), 5))
                for idx, (_, tr) in enumerate(tunnel_df.head(5).iterrows()):
                    gc = grade_colors.get(tr["Grade"], "#aaa")
                    with tun_cols[idx % len(tun_cols)]:
                        st.markdown(
                            f'<div style="text-align:center;padding:10px;background:white;border-radius:8px;'
                            f'border:2px solid {gc};margin:2px;">'
                            f'<div style="font-size:28px;font-weight:900;color:{gc} !important;">{tr["Grade"]}</div>'
                            f'<div style="font-size:12px;font-weight:700;color:#1a1a2e !important;">'
                            f'{tr["Pitch A"]} ↔ {tr["Pitch B"]}</div>'
                            f'<div style="font-size:11px;color:#666 !important;">Score: {tr["Tunnel Score"]:.0f} | '
                            f'Commit: {tr["Commit Sep (in)"]:.1f}″ | Plate: {tr["Plate Sep (in)"]:.1f}″</div>'
                            f'</div>', unsafe_allow_html=True)

            # ── Combined Matrix: Whiff% with Tunnel Grade overlay ──
            section_header("Sequence + Tunnel Matrix")
            st.caption("Cells = Whiff% on next pitch. Border color = tunnel grade for that pair. Best combos have high whiff% AND strong tunnel.")

            pitch_types = sorted(pdf["TaggedPitchType"].dropna().unique())
            matrix_data = np.full((len(pitch_types), len(pitch_types)), np.nan)
            matrix_annot = [['' for _ in pitch_types] for _ in pitch_types]
            matrix_n = [[0 for _ in pitch_types] for _ in pitch_types]

            for i, pt_a in enumerate(pitch_types):
                for j, pt_b in enumerate(pitch_types):
                    pair = pairs[(pairs["TaggedPitchType"] == pt_a) & (pairs["NextPitch"] == pt_b)]
                    if len(pair) < 25:
                        continue
                    sw = pair[pair["NextCall"].isin(SWING_CALLS)]
                    wh = pair[pair["NextCall"] == "StrikeSwinging"]
                    if len(sw) > 0:
                        whiff_pct = len(wh) / len(sw) * 100
                        matrix_data[i, j] = whiff_pct
                        tn = tunnel_lookup.get((pt_a, pt_b))
                        grade_tag = f" [{tn['Grade']}]" if tn is not None else ""
                        matrix_annot[i][j] = f"{whiff_pct:.0f}%{grade_tag}\n({len(pair)})"
                    matrix_n[i][j] = len(pair)

            fig_matrix = go.Figure(data=go.Heatmap(
                z=matrix_data, text=matrix_annot, texttemplate="%{text}",
                x=pitch_types, y=pitch_types,
                colorscale=[[0, "#f7f7f7"], [0.3, "#f7c631"], [0.6, "#fe6100"], [1, "#d22d49"]],
                zmin=0, zmax=50, showscale=True,
                colorbar=dict(title="Whiff%", len=0.8),
            ))
            fig_matrix.update_layout(**CHART_LAYOUT, height=max(300, len(pitch_types) * 60 + 60),
                                      xaxis_title="Next Pitch (B)", yaxis_title="Current Pitch (A)",
                                      xaxis=dict(side="bottom"))
            st.plotly_chart(fig_matrix, use_container_width=True)

            # ── Build full sequence rows with tunnel data ──
            seq_rows = []
            for i, pt_a in enumerate(pitch_types):
                for j, pt_b in enumerate(pitch_types):
                    pair = pairs[(pairs["TaggedPitchType"] == pt_a) & (pairs["NextPitch"] == pt_b)]
                    if len(pair) < 25:
                        continue
                    sw = pair[pair["NextCall"].isin(SWING_CALLS)]
                    wh = pair[pair["NextCall"] == "StrikeSwinging"]
                    ct = pair[pair["NextCall"].isin(CONTACT_CALLS)]
                    bt = pair[pair["NextCall"] == "InPlay"].dropna(subset=["NextEV"])
                    csw = pair[pair["NextCall"].isin(["StrikeSwinging", "StrikeCalled"])]
                    whiff_pct = len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else 0
                    tn = tunnel_lookup.get((pt_a, pt_b))
                    tunnel_score = tn["Tunnel Score"] if tn is not None else np.nan
                    tunnel_grade = tn["Grade"] if tn is not None else "-"
                    # Composite: 40% tunnel + 30% whiff + 20% EV + 10% CSW
                    whiff_norm = min(whiff_pct / 40.0, 1.0) * 100
                    tunnel_norm = tunnel_score if not pd.isna(tunnel_score) else 50
                    ev_val = bt["NextEV"].mean() if len(bt) > 0 else np.nan
                    ev_norm = max(0, min(100, (105 - ev_val) / 25 * 100)) if not pd.isna(ev_val) else 50
                    csw_pct = len(csw) / len(pair) * 100
                    csw_norm = min(csw_pct / 35.0, 1.0) * 100
                    combo_score = tunnel_norm * 0.40 + whiff_norm * 0.30 + ev_norm * 0.20 + csw_norm * 0.10
                    seq_rows.append({
                        "Sequence": f"{pt_a} → {pt_b}",
                        "Count": len(pair),
                        "Swing%": len(sw) / len(pair) * 100,
                        "Whiff%": whiff_pct,
                        "CSW%": len(csw) / len(pair) * 100,
                        "Avg EV": ev_val,
                        "Tunnel": tunnel_grade,
                        "Tunnel Score": tunnel_score,
                        "Combo Score": round(combo_score, 1),
                        "_whiff": whiff_pct,
                        "_combo": combo_score,
                    })

            if seq_rows:
                seq_df = pd.DataFrame(seq_rows).sort_values("_combo", ascending=False)

                # ── Top Sequences: Combined Ranking ──
                section_header("Top Sequences — Tunnel-Adjusted Ranking")
                st.caption("Combo Score = 50% Whiff Rate + 30% Tunnel Score + 20% Low Contact Quality. Best sequences deceive AND dominate.")
                top_combos = seq_df.head(5)
                for _, row in top_combos.iterrows():
                    gc = grade_colors.get(row["Tunnel"], "#aaa") if not tunnel_df.empty else "#888"
                    ev_str = f" | EV: {row['Avg EV']:.1f}" if not pd.isna(row["Avg EV"]) else ""
                    st.markdown(
                        f'<div style="padding:10px 14px;background:white;border-radius:8px;margin:4px 0;'
                        f'border-left:5px solid {gc};border:1px solid #eee;">'
                        f'<span style="font-size:15px;font-weight:800;color:#1a1a2e !important;">{row["Sequence"]}</span>'
                        f'<span style="display:inline-block;margin-left:8px;padding:2px 8px;border-radius:4px;'
                        f'background:{gc};color:white !important;font-size:11px;font-weight:700;">'
                        f'Tunnel: {row["Tunnel"]}</span>'
                        f'<span style="float:right;font-size:14px;font-weight:900;color:#1a1a2e !important;">'
                        f'Combo: {row["Combo Score"]:.0f}</span>'
                        f'<div style="font-size:12px;color:#555 !important;margin-top:2px;">'
                        f'Whiff: {row["Whiff%"]:.0f}% | CSW: {row["CSW%"]:.0f}%{ev_str} | n={row["Count"]}</div>'
                        f'</div>', unsafe_allow_html=True)

                st.markdown("")

                # ── Side by side: best whiff vs worst damage ──
                col_best, col_worst = st.columns(2)
                with col_best:
                    section_header("Highest Whiff% Sequences")
                    top5 = seq_df.sort_values("_whiff", ascending=False).head(5)
                    for _, row in top5.iterrows():
                        tn_tag = f' <span style="color:{grade_colors.get(row["Tunnel"], "#aaa")} !important;">[{row["Tunnel"]}]</span>' if row["Tunnel"] != "-" else ""
                        ev_str = f"EV: {row['Avg EV']:.1f} | " if not pd.isna(row["Avg EV"]) else ""
                        st.markdown(
                            f'<div style="padding:8px 12px;background:white;border-radius:6px;margin:4px 0;'
                            f'border-left:4px solid #2ca02c;border:1px solid #eee;">'
                            f'<span style="font-size:14px;font-weight:700;color:#1a1a2e !important;">{row["Sequence"]}{tn_tag}</span>'
                            f'<span style="float:right;font-size:13px;font-weight:800;color:#2ca02c !important;">'
                            f'Whiff: {row["Whiff%"]:.0f}% | CSW: {row["CSW%"]:.0f}%</span>'
                            f'<div style="font-size:11px;color:#666 !important;">{ev_str}n={row["Count"]}</div></div>',
                            unsafe_allow_html=True)
                with col_worst:
                    section_header("Most Damage Sequences")
                    bot5 = seq_df.dropna(subset=["Avg EV"]).sort_values("Avg EV", ascending=False).head(5)
                    for _, row in bot5.iterrows():
                        tn_tag = f' <span style="color:{grade_colors.get(row["Tunnel"], "#aaa")} !important;">[{row["Tunnel"]}]</span>' if row["Tunnel"] != "-" else ""
                        st.markdown(
                            f'<div style="padding:8px 12px;background:white;border-radius:6px;margin:4px 0;'
                            f'border-left:4px solid #d22d49;border:1px solid #eee;">'
                            f'<span style="font-size:14px;font-weight:700;color:#1a1a2e !important;">{row["Sequence"]}{tn_tag}</span>'
                            f'<span style="float:right;font-size:13px;font-weight:800;color:#d22d49 !important;">'
                            f'EV: {row["Avg EV"]:.1f} | Whiff: {row["Whiff%"]:.0f}%</span>'
                            f'<div style="font-size:11px;color:#666 !important;">n={row["Count"]}</div></div>',
                            unsafe_allow_html=True)

                # ── Tunnel Divergence Visualization for Top Pair ──
                if not tunnel_df.empty:
                    section_header("Commit-Point Divergence — Top Tunnel Pair")
                    best_tn = tunnel_df.iloc[0]
                    st.caption(f"How {best_tn['Pitch A']} and {best_tn['Pitch B']} separate in flight — "
                               f"closer at commit = more deception, farther at plate = more movement")

                    # Recompute flight paths for visualization
                    req_cols = ["TaggedPitchType", "RelHeight", "RelSide", "PlateLocHeight", "PlateLocSide",
                                "InducedVertBreak", "HorzBreak", "RelSpeed"]
                    if all(c in pdf.columns for c in req_cols):
                        agg_cols_viz = {
                            "rel_h": ("RelHeight", "mean"), "rel_s": ("RelSide", "mean"),
                            "loc_h": ("PlateLocHeight", "mean"), "loc_s": ("PlateLocSide", "mean"),
                            "ivb": ("InducedVertBreak", "mean"), "hb": ("HorzBreak", "mean"),
                            "velo": ("RelSpeed", "mean"),
                        }
                        if "Extension" in pdf.columns:
                            agg_cols_viz["ext"] = ("Extension", "mean")
                        agg_viz = pdf.groupby("TaggedPitchType").agg(**agg_cols_viz).dropna(subset=["rel_h", "velo"])
                        if "ext" not in agg_viz.columns:
                            agg_viz["ext"] = 6.0

                        pt_a_name, pt_b_name = best_tn["Pitch A"], best_tn["Pitch B"]
                        if pt_a_name in agg_viz.index and pt_b_name in agg_viz.index:
                            MOUND_DIST = 60.5
                            def _viz_flight(row, frac):
                                ext = row.ext if not pd.isna(row.ext) else 6.0
                                actual_dist = MOUND_DIST - ext
                                velo_fps = row.velo * 5280 / 3600
                                t_total = actual_dist / velo_fps
                                t = frac * t_total
                                gravity_drop = 0.5 * 32.17 * t**2
                                ivb_lift = (row.ivb / 12.0) * frac**2
                                y = row.rel_h + (row.loc_h - row.rel_h) * frac - gravity_drop + ivb_lift
                                y_at_1 = row.rel_h + (row.loc_h - row.rel_h) - 0.5 * 32.17 * t_total**2 + (row.ivb / 12.0)
                                y += (row.loc_h - y_at_1) * frac
                                hb_curve = (row.hb / 12.0) * frac**2
                                x = row.rel_s + (row.loc_s - row.rel_s) * frac + hb_curve
                                x_at_1 = row.rel_s + (row.loc_s - row.rel_s) + (row.hb / 12.0)
                                x += (row.loc_s - x_at_1) * frac
                                return x, y

                            checkpoints = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                            labels = ["Release", "20%", "40%", "Commit (60%)", "80%", "Plate"]
                            a_row = agg_viz.loc[pt_a_name]
                            b_row = agg_viz.loc[pt_b_name]
                            seps = []
                            a_xs, a_ys, b_xs, b_ys = [], [], [], []
                            for frac in checkpoints:
                                ax, ay = _viz_flight(a_row, frac)
                                bx, by = _viz_flight(b_row, frac)
                                a_xs.append(ax); a_ys.append(ay)
                                b_xs.append(bx); b_ys.append(by)
                                sep_in = np.sqrt((ay - by)**2 + (ax - bx)**2) * 12
                                seps.append(sep_in)

                            col_div, col_path = st.columns(2)
                            with col_div:
                                # Divergence line chart
                                fig_div = go.Figure()
                                fig_div.add_trace(go.Scatter(
                                    x=labels, y=seps, mode="lines+markers+text",
                                    text=[f"{s:.1f}″" for s in seps],
                                    textposition="top center",
                                    line=dict(color="#000000", width=3),
                                    marker=dict(size=10, color=[grade_colors.get(best_tn["Grade"], "#aaa")] * len(seps)),
                                    showlegend=False,
                                ))
                                fig_div.add_hline(y=6, line_dash="dash", line_color="#d22d49",
                                                  annotation_text="6″ — Hard to distinguish", annotation_position="top left")
                                fig_div.update_layout(**CHART_LAYOUT, height=350,
                                                      yaxis_title="Separation (inches)",
                                                      title=f"{pt_a_name} vs {pt_b_name} — Flight Separation")
                                st.plotly_chart(fig_div, use_container_width=True)

                            with col_path:
                                # Side view: both trajectories
                                fig_path = go.Figure()
                                clr_a = PITCH_COLORS.get(pt_a_name, "#1f77b4")
                                clr_b = PITCH_COLORS.get(pt_b_name, "#ff7f0e")
                                dist_pts = [0, 0.2 * (MOUND_DIST - 6), 0.4 * (MOUND_DIST - 6),
                                            0.6 * (MOUND_DIST - 6), 0.8 * (MOUND_DIST - 6),
                                            MOUND_DIST - 6]
                                fig_path.add_trace(go.Scatter(
                                    x=dist_pts, y=a_ys, mode="lines+markers",
                                    name=pt_a_name, line=dict(color=clr_a, width=3),
                                    marker=dict(size=8),
                                ))
                                fig_path.add_trace(go.Scatter(
                                    x=dist_pts, y=b_ys, mode="lines+markers",
                                    name=pt_b_name, line=dict(color=clr_b, width=3),
                                    marker=dict(size=8),
                                ))
                                # Commit point marker
                                fig_path.add_vline(x=0.6 * (MOUND_DIST - 6), line_dash="dot", line_color="#888",
                                                   annotation_text="Commit", annotation_position="top")
                                fig_path.update_layout(**CHART_LAYOUT, height=350,
                                                        xaxis_title="Distance from release (ft)",
                                                        yaxis_title="Height (ft)",
                                                        title="Side View — Pitch Trajectories",
                                                        yaxis=dict(range=[0, 7]),
                                                        legend=dict(x=0.02, y=0.98))
                                st.plotly_chart(fig_path, use_container_width=True)

                            # Diagnosis from tunnel data
                            st.markdown(
                                f'<div style="padding:12px 16px;background:#f0f7ff;border-radius:8px;border:1px solid #cce0ff;">'
                                f'<span style="font-size:13px;font-weight:700;color:#1a1a2e !important;">Tunnel Analysis:</span> '
                                f'<span style="font-size:12px;color:#333 !important;">{best_tn["Diagnosis"]}</span>'
                                f'<br><span style="font-size:12px;font-weight:600;color:#1565c0 !important;">Action: </span>'
                                f'<span style="font-size:12px;color:#333 !important;">{best_tn["Fix"]}</span></div>',
                                unsafe_allow_html=True)

                # Full table with tunnel data
                with st.expander("Full 2-Pitch Sequence + Tunnel Table"):
                    disp_seq = seq_df.drop(columns=["_whiff", "_combo"]).copy()
                    for c in ["Swing%", "Whiff%", "CSW%"]:
                        disp_seq[c] = disp_seq[c].map(lambda x: f"{x:.1f}%")
                    disp_seq["Avg EV"] = disp_seq["Avg EV"].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
                    disp_seq["Tunnel Score"] = disp_seq["Tunnel Score"].map(lambda x: f"{x:.0f}" if not pd.isna(x) else "-")
                    disp_seq = disp_seq.sort_values("Combo Score", ascending=False)
                    disp_seq["Combo Score"] = disp_seq["Combo Score"].map(lambda x: f"{x:.0f}")
                    st.dataframe(disp_seq.set_index("Sequence"), use_container_width=True)

            # ── 3-PITCH & 4-PITCH SEQUENCE ANALYSIS ──────────────
            st.markdown("---")
            section_header("Multi-Pitch Sequence Patterns")
            st.caption("3-pitch and 4-pitch sequences with conditional outcome probabilities — "
                       "what actually happens when this pitcher follows a pattern")

            # Build extended sequence columns from the sorted pitch data
            sdf2 = sdf.copy()
            sdf2["Pitch2"] = sdf2.groupby(pa_cols)["TaggedPitchType"].shift(-1)
            sdf2["Pitch3"] = sdf2.groupby(pa_cols)["TaggedPitchType"].shift(-2)
            sdf2["Pitch4"] = sdf2.groupby(pa_cols)["TaggedPitchType"].shift(-3)
            sdf2["Call1"] = sdf2["PitchCall"]
            sdf2["Call2"] = sdf2.groupby(pa_cols)["PitchCall"].shift(-1)
            sdf2["Call3"] = sdf2.groupby(pa_cols)["PitchCall"].shift(-2)
            sdf2["Call4"] = sdf2.groupby(pa_cols)["PitchCall"].shift(-3)
            sdf2["EV3"] = sdf2.groupby(pa_cols)["ExitSpeed"].shift(-2)
            sdf2["EV4"] = sdf2.groupby(pa_cols)["ExitSpeed"].shift(-3)

            # Outcome classification helper
            def _classify_outcome(call, ev=np.nan):
                if call == "StrikeSwinging":
                    return "Whiff"
                elif call == "StrikeCalled":
                    return "Called Strike"
                elif call in ("BallCalled", "HitByPitch", "BallinDirt", "BallIntentional"):
                    return "Ball"
                elif call in ("FoulBall", "FoulBallNotFieldable", "FoulBallFieldable"):
                    return "Foul"
                elif call == "InPlay":
                    if not pd.isna(ev) and ev < 75:
                        return "Weak Contact"
                    elif not pd.isna(ev) and ev >= 95:
                        return "Hard Contact"
                    else:
                        return "In Play"
                return "Other"

            tab_3p, tab_4p, tab_tree = st.tabs(["3-Pitch Sequences", "4-Pitch Sequences", "Sequence Trees"])

            # ── 3-Pitch Sequences ──
            with tab_3p:
                section_header("3-Pitch Sequence Analysis")
                st.markdown("""
                Analyzes every **3-consecutive-pitch window** within each at-bat.
                For each sequence (e.g. Slider → Fastball → Changeup), the table shows how effective the **3rd pitch** is —
                whiff rate, called-strike rate, contact quality — and how outcomes change depending on whether the **1st pitch was a strike or ball**.
                Use this to find which 3-pitch patterns generate the most swings-and-misses or weak contact.
                """)

                seq3 = sdf2.dropna(subset=["Pitch2", "Pitch3", "Call3"]).copy()
                if len(seq3) < 20:
                    st.info("Not enough 3-pitch sequences (need 20+).")
                else:
                    # Build 3-pitch sequence stats
                    seq3["Seq3"] = seq3["TaggedPitchType"] + " → " + seq3["Pitch2"] + " → " + seq3["Pitch3"]
                    seq3_counts = seq3["Seq3"].value_counts()
                    top_seq3 = seq3_counts[seq3_counts >= 25].head(20).index.tolist()

                    if top_seq3:
                        seq3_rows = []
                        for s3 in top_seq3:
                            s3_df = seq3[seq3["Seq3"] == s3]
                            n = len(s3_df)
                            # Outcome on 3rd pitch
                            sw3 = s3_df[s3_df["Call3"].isin(SWING_CALLS)]
                            wh3 = s3_df[s3_df["Call3"] == "StrikeSwinging"]
                            csw3 = s3_df[s3_df["Call3"].isin(["StrikeSwinging", "StrikeCalled"])]
                            ip3 = s3_df[s3_df["Call3"] == "InPlay"]
                            ev3_vals = ip3["EV3"].dropna()
                            # Conditional: what was Call1 outcome?
                            call1_strike = s3_df[s3_df["Call1"].isin(["StrikeCalled", "StrikeSwinging",
                                                                       "FoulBall", "FoulBallNotFieldable",
                                                                       "FoulBallFieldable", "InPlay"])]
                            pct_strike1 = len(call1_strike) / max(n, 1) * 100

                            # Tunnel data for the pair transitions
                            pitches = s3.split(" → ")
                            if len(pitches) != 3:
                                continue
                            tn_12 = tunnel_lookup.get((pitches[0], pitches[1]))
                            tn_23 = tunnel_lookup.get((pitches[1], pitches[2]))
                            tunnel_avg = np.nanmean([
                                tn_12["Tunnel Score"] if tn_12 is not None else np.nan,
                                tn_23["Tunnel Score"] if tn_23 is not None else np.nan,
                            ])

                            seq3_rows.append({
                                "Sequence": s3,
                                "n": n,
                                "Strike1%": pct_strike1,
                                "Whiff3%": len(wh3) / max(len(sw3), 1) * 100 if len(sw3) > 0 else 0,
                                "CSW3%": len(csw3) / n * 100,
                                "Swing3%": len(sw3) / n * 100,
                                "Avg EV3": ev3_vals.mean() if len(ev3_vals) > 0 else np.nan,
                                "Weak%": len(ev3_vals[ev3_vals < 75]) / max(len(ev3_vals), 1) * 100 if len(ev3_vals) > 0 else np.nan,
                                "Tunnel Avg": tunnel_avg,
                                "_score": (tunnel_avg if not pd.isna(tunnel_avg) else 50) * 0.4
                                          + (len(wh3) / max(len(sw3), 1) * 100 if len(sw3) > 0 else 0) * 0.3
                                          + (len(csw3) / n * 100) * 0.3,
                            })

                        if seq3_rows:
                            seq3_df = pd.DataFrame(seq3_rows).sort_values("_score", ascending=False)

                            # Top 3-pitch sequences
                            st.markdown("**Top 3-Pitch Sequences (by combined whiff + CSW + tunnel)**")
                            for _, row in seq3_df.head(8).iterrows():
                                tn_str = f"{row['Tunnel Avg']:.0f}" if not pd.isna(row["Tunnel Avg"]) else "-"
                                ev_str = f" | EV: {row['Avg EV3']:.1f}" if not pd.isna(row["Avg EV3"]) else ""
                                weak_str = f" | Weak: {row['Weak%']:.0f}%" if not pd.isna(row["Weak%"]) else ""
                                score_clr = "#2ca02c" if row["_score"] > 50 else "#fe6100" if row["_score"] > 35 else "#d22d49"
                                st.markdown(
                                    f'<div style="padding:8px 14px;background:white;border-radius:6px;margin:3px 0;'
                                    f'border-left:4px solid {score_clr};border:1px solid #eee;">'
                                    f'<span style="font-size:14px;font-weight:800;color:#1a1a2e !important;">'
                                    f'{row["Sequence"]}</span>'
                                    f'<span style="float:right;font-size:12px;font-weight:700;color:{score_clr} !important;">'
                                    f'Score: {row["_score"]:.0f}</span>'
                                    f'<div style="font-size:11px;color:#555 !important;">'
                                    f'n={row["n"]} | Strike1: {row["Strike1%"]:.0f}% | '
                                    f'Whiff3: {row["Whiff3%"]:.0f}% | CSW3: {row["CSW3%"]:.0f}%{ev_str}{weak_str} | '
                                    f'Tunnel: {tn_str}</div></div>', unsafe_allow_html=True)

                            # Conditional outcome breakdown
                            section_header("Conditional Outcomes — \"If Pitch 1 = Strike, then...\"")
                            st.caption("How the 3rd pitch performs when Pitch 1 was a strike vs ball")

                            top3_seq = seq3_df["Sequence"].head(5).tolist()
                            cond_rows = []
                            for s3 in top3_seq:
                                s3_df = seq3[seq3["Seq3"] == s3]
                                # Split by Pitch 1 outcome
                                strike1 = s3_df[s3_df["Call1"].isin(["StrikeCalled", "StrikeSwinging",
                                                                      "FoulBall", "FoulBallNotFieldable",
                                                                      "FoulBallFieldable", "InPlay"])]
                                ball1 = s3_df[~s3_df.index.isin(strike1.index)]
                                for label_c, cdf in [("After Strike", strike1), ("After Ball", ball1)]:
                                    if len(cdf) < 2:
                                        continue
                                    sw = cdf[cdf["Call3"].isin(SWING_CALLS)]
                                    wh = cdf[cdf["Call3"] == "StrikeSwinging"]
                                    ip = cdf[cdf["Call3"] == "InPlay"]
                                    ev_v = ip["EV3"].dropna()
                                    cond_rows.append({
                                        "Sequence": s3,
                                        "Condition": label_c,
                                        "n": len(cdf),
                                        "Swing%": f"{len(sw)/len(cdf)*100:.0f}%",
                                        "Whiff%": f"{len(wh)/max(len(sw),1)*100:.0f}%" if len(sw) > 0 else "-",
                                        "InPlay%": f"{len(ip)/len(cdf)*100:.0f}%",
                                        "Avg EV": f"{ev_v.mean():.1f}" if len(ev_v) > 0 else "-",
                                        "Weak%": f"{len(ev_v[ev_v<75])/max(len(ev_v),1)*100:.0f}%" if len(ev_v) > 0 else "-",
                                    })
                            if cond_rows:
                                st.dataframe(pd.DataFrame(cond_rows).set_index(["Sequence", "Condition"]),
                                             use_container_width=True)

                            # Full table
                            with st.expander("Full 3-Pitch Sequence Table"):
                                disp3 = seq3_df.drop(columns=["_score"]).copy()
                                for c in ["Strike1%", "Whiff3%", "CSW3%", "Swing3%"]:
                                    disp3[c] = disp3[c].map(lambda x: f"{x:.1f}%")
                                disp3["Avg EV3"] = disp3["Avg EV3"].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
                                disp3["Weak%"] = disp3["Weak%"].map(lambda x: f"{x:.0f}%" if not pd.isna(x) else "-")
                                disp3["Tunnel Avg"] = disp3["Tunnel Avg"].map(lambda x: f"{x:.0f}" if not pd.isna(x) else "-")
                                st.dataframe(disp3.set_index("Sequence"), use_container_width=True)

            # ── 4-Pitch Sequences ──
            with tab_4p:
                section_header("4-Pitch Sequence Analysis")
                st.caption("Pitch 1 → Pitch 2 → Pitch 3 → Pitch 4: Deep pattern analysis")

                seq4 = sdf2.dropna(subset=["Pitch2", "Pitch3", "Pitch4", "Call4"]).copy()
                if len(seq4) < 20:
                    st.info("Not enough 4-pitch sequences (need 20+).")
                else:
                    seq4["Seq4"] = (seq4["TaggedPitchType"] + " → " + seq4["Pitch2"] + " → " +
                                    seq4["Pitch3"] + " → " + seq4["Pitch4"])
                    seq4_counts = seq4["Seq4"].value_counts()
                    top_seq4 = seq4_counts[seq4_counts >= 25].head(20).index.tolist()

                    if top_seq4:
                        seq4_rows = []
                        for s4 in top_seq4:
                            s4_df = seq4[seq4["Seq4"] == s4]
                            n = len(s4_df)
                            sw4 = s4_df[s4_df["Call4"].isin(SWING_CALLS)]
                            wh4 = s4_df[s4_df["Call4"] == "StrikeSwinging"]
                            csw4 = s4_df[s4_df["Call4"].isin(["StrikeSwinging", "StrikeCalled"])]
                            ip4 = s4_df[s4_df["Call4"] == "InPlay"]
                            ev4_vals = ip4["EV4"].dropna()

                            # Count strikes in first 3 pitches
                            strike_calls = ["StrikeCalled", "StrikeSwinging", "FoulBall",
                                            "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]
                            s4_df_c = s4_df.copy()
                            s4_df_c["_str1"] = s4_df_c["Call1"].isin(strike_calls).astype(int)
                            s4_df_c["_str2"] = s4_df_c["Call2"].isin(strike_calls).astype(int)
                            s4_df_c["_str3"] = s4_df_c["Call3"].isin(strike_calls).astype(int)
                            avg_strikes = (s4_df_c["_str1"] + s4_df_c["_str2"] + s4_df_c["_str3"]).mean()

                            pitches = s4.split(" → ")
                            tunnel_scores = []
                            for k in range(len(pitches) - 1):
                                tn = tunnel_lookup.get((pitches[k], pitches[k+1]))
                                if tn is not None:
                                    tunnel_scores.append(tn["Tunnel Score"])
                            tunnel_avg = np.mean(tunnel_scores) if tunnel_scores else np.nan

                            seq4_rows.append({
                                "Sequence": s4,
                                "n": n,
                                "Avg Strikes (1-3)": round(avg_strikes, 1),
                                "Whiff4%": len(wh4) / max(len(sw4), 1) * 100 if len(sw4) > 0 else 0,
                                "CSW4%": len(csw4) / n * 100,
                                "Swing4%": len(sw4) / n * 100,
                                "Avg EV4": ev4_vals.mean() if len(ev4_vals) > 0 else np.nan,
                                "Tunnel Avg": tunnel_avg,
                                "_score": (tunnel_avg if not pd.isna(tunnel_avg) else 50) * 0.4
                                          + (len(wh4) / max(len(sw4), 1) * 100 if len(sw4) > 0 else 0) * 0.3
                                          + (len(csw4) / n * 100) * 0.3,
                            })

                        if seq4_rows:
                            seq4_df = pd.DataFrame(seq4_rows).sort_values("_score", ascending=False)

                            st.markdown("**Top 4-Pitch Sequences**")
                            for _, row in seq4_df.head(8).iterrows():
                                tn_str = f"{row['Tunnel Avg']:.0f}" if not pd.isna(row["Tunnel Avg"]) else "-"
                                ev_str = f" | EV: {row['Avg EV4']:.1f}" if not pd.isna(row["Avg EV4"]) else ""
                                score_clr = "#2ca02c" if row["_score"] > 50 else "#fe6100" if row["_score"] > 35 else "#d22d49"
                                st.markdown(
                                    f'<div style="padding:8px 14px;background:white;border-radius:6px;margin:3px 0;'
                                    f'border-left:4px solid {score_clr};border:1px solid #eee;">'
                                    f'<span style="font-size:14px;font-weight:800;color:#1a1a2e !important;">'
                                    f'{row["Sequence"]}</span>'
                                    f'<span style="float:right;font-size:12px;font-weight:700;color:{score_clr} !important;">'
                                    f'Score: {row["_score"]:.0f}</span>'
                                    f'<div style="font-size:11px;color:#555 !important;">'
                                    f'n={row["n"]} | Strikes(1-3): {row["Avg Strikes (1-3)"]:.1f} | '
                                    f'Whiff4: {row["Whiff4%"]:.0f}% | CSW4: {row["CSW4%"]:.0f}%{ev_str} | '
                                    f'Tunnel: {tn_str}</div></div>', unsafe_allow_html=True)

                            with st.expander("Full 4-Pitch Sequence Table"):
                                disp4 = seq4_df.drop(columns=["_score"]).copy()
                                for c in ["Whiff4%", "CSW4%", "Swing4%"]:
                                    disp4[c] = disp4[c].map(lambda x: f"{x:.1f}%")
                                disp4["Avg EV4"] = disp4["Avg EV4"].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
                                disp4["Tunnel Avg"] = disp4["Tunnel Avg"].map(lambda x: f"{x:.0f}" if not pd.isna(x) else "-")
                                st.dataframe(disp4.set_index("Sequence"), use_container_width=True)
                    else:
                        st.info("No 4-pitch sequences with 3+ occurrences found.")

            # ── Sequence Decision Trees ──
            with tab_tree:
                section_header("Sequence Decision Trees")
                st.caption("Select a starting pitch to see branching probabilities — "
                           "\"After Slider strike → what comes next? What's the outcome?\"")

                # Starting pitch selection
                start_pitch = st.selectbox("Start with pitch:", pitch_types, key=f"{key_prefix}_tree_start")

                # Build the tree from sdf2
                tree_base = sdf2[sdf2["TaggedPitchType"] == start_pitch].copy()
                if len(tree_base) < 10:
                    st.info(f"Not enough at-bats starting with {start_pitch}.")
                else:
                    # Level 1: What outcome on Pitch 1?
                    tree_base["Out1"] = tree_base["Call1"].map(lambda c: "Strike" if c in [
                        "StrikeCalled", "StrikeSwinging", "FoulBall", "FoulBallNotFieldable",
                        "FoulBallFieldable", "InPlay"] else "Ball")

                    section_header(f"After {start_pitch}...")
                    for outcome1 in ["Strike", "Ball"]:
                        o1_df = tree_base[tree_base["Out1"] == outcome1]
                        if len(o1_df) < 3:
                            continue
                        o1_pct = len(o1_df) / len(tree_base) * 100

                        # What pitch comes next?
                        o1_with_p2 = o1_df.dropna(subset=["Pitch2"])
                        if len(o1_with_p2) < 3:
                            continue
                        p2_counts = o1_with_p2["Pitch2"].value_counts()
                        o1_clr = "#2ca02c" if outcome1 == "Strike" else "#d22d49"

                        st.markdown(
                            f'<div style="padding:10px 14px;background:#f8f8f8;border-radius:8px;'
                            f'border-left:5px solid {o1_clr};margin:6px 0;">'
                            f'<span style="font-size:15px;font-weight:800;color:{o1_clr} !important;">'
                            f'{start_pitch} = {outcome1} ({o1_pct:.0f}%)</span></div>',
                            unsafe_allow_html=True)

                        # Level 2 branches
                        branch_rows = []
                        for p2_name in p2_counts.head(4).index:
                            p2_df = o1_with_p2[o1_with_p2["Pitch2"] == p2_name]
                            p2_pct = len(p2_df) / len(o1_with_p2) * 100
                            # Outcome on Pitch 2
                            sw2 = p2_df[p2_df["Call2"].isin(SWING_CALLS)]
                            wh2 = p2_df[p2_df["Call2"] == "StrikeSwinging"]
                            csw2 = p2_df[p2_df["Call2"].isin(["StrikeSwinging", "StrikeCalled"])]
                            ip2 = p2_df[p2_df["Call2"] == "InPlay"]

                            # Level 3: what's the 3rd pitch after this?
                            p2_with_p3 = p2_df.dropna(subset=["Pitch3"])
                            p3_top = p2_with_p3["Pitch3"].value_counts().head(3) if len(p2_with_p3) >= 3 else pd.Series(dtype=float)
                            p3_str = ", ".join([f"{pt} ({ct/len(p2_with_p3)*100:.0f}%)" for pt, ct in p3_top.items()]) if len(p3_top) > 0 else "-"

                            # Outcome summary on pitch 2
                            outcomes = []
                            if len(wh2) > 0:
                                outcomes.append(f"Whiff {len(wh2)/len(sw2)*100:.0f}%" if len(sw2) > 0 else "Whiff -")
                            if len(csw2) > 0:
                                outcomes.append(f"CSW {len(csw2)/len(p2_df)*100:.0f}%")
                            if len(ip2) > 0:
                                ev_avg = sdf2.loc[ip2.index, "ExitSpeed"].dropna().mean() if "ExitSpeed" in sdf2.columns else np.nan
                                outcomes.append(f"InPlay {len(ip2)} ({ev_avg:.0f} EV)" if not pd.isna(ev_avg) else f"InPlay {len(ip2)}")

                            branch_rows.append({
                                "Next Pitch": p2_name,
                                "Frequency": f"{p2_pct:.0f}%",
                                "n": len(p2_df),
                                "P2 Outcome": " | ".join(outcomes) if outcomes else "-",
                                "Then Pitch 3": p3_str,
                            })

                        if branch_rows:
                            bdf_tree = pd.DataFrame(branch_rows)
                            st.dataframe(bdf_tree.set_index("Next Pitch"), use_container_width=True)

                    # Head-to-head comparison
                    section_header("Compare Sequences Head-to-Head")
                    st.caption("Does Slider → Slider → Fastball work better than Slider → Fastball → Slider?")
                    seq3_all = sdf2.dropna(subset=["Pitch2", "Pitch3", "Call3"]).copy()
                    seq3_all["Seq3"] = seq3_all["TaggedPitchType"] + " → " + seq3_all["Pitch2"] + " → " + seq3_all["Pitch3"]
                    avail_seqs = seq3_all["Seq3"].value_counts()
                    avail_seqs = avail_seqs[avail_seqs >= 3].index.tolist()

                    if len(avail_seqs) >= 2:
                        col_cmp1, col_cmp2 = st.columns(2)
                        with col_cmp1:
                            seq_a = st.selectbox("Sequence A", avail_seqs, key=f"{key_prefix}_cmp_a")
                        with col_cmp2:
                            seq_b = st.selectbox("Sequence B", [s for s in avail_seqs if s != seq_a],
                                                 key=f"{key_prefix}_cmp_b")

                        cmp_rows = []
                        for seq_name in [seq_a, seq_b]:
                            cdf = seq3_all[seq3_all["Seq3"] == seq_name]
                            n = len(cdf)
                            sw = cdf[cdf["Call3"].isin(SWING_CALLS)]
                            wh = cdf[cdf["Call3"] == "StrikeSwinging"]
                            csw = cdf[cdf["Call3"].isin(["StrikeSwinging", "StrikeCalled"])]
                            ip = cdf[cdf["Call3"] == "InPlay"]
                            ev_v = ip["EV3"].dropna()
                            weak = ev_v[ev_v < 75] if len(ev_v) > 0 else pd.Series(dtype=float)
                            hard = ev_v[ev_v >= 95] if len(ev_v) > 0 else pd.Series(dtype=float)

                            pitches = seq_name.split(" → ")
                            tscores = []
                            for k in range(len(pitches) - 1):
                                tn = tunnel_lookup.get((pitches[k], pitches[k+1]))
                                if tn is not None:
                                    tscores.append(tn["Tunnel Score"])
                            t_avg = np.mean(tscores) if tscores else np.nan

                            cmp_rows.append({
                                "Sequence": seq_name,
                                "n": n,
                                "Swing%": f"{len(sw)/n*100:.1f}%",
                                "Whiff%": f"{len(wh)/max(len(sw),1)*100:.1f}%" if len(sw) > 0 else "-",
                                "CSW%": f"{len(csw)/n*100:.1f}%",
                                "InPlay": len(ip),
                                "Avg EV": f"{ev_v.mean():.1f}" if len(ev_v) > 0 else "-",
                                "Weak Contact%": f"{len(weak)/max(len(ev_v),1)*100:.0f}%" if len(ev_v) > 0 else "-",
                                "Hard Contact%": f"{len(hard)/max(len(ev_v),1)*100:.0f}%" if len(ev_v) > 0 else "-",
                                "Tunnel Avg": f"{t_avg:.0f}" if not pd.isna(t_avg) else "-",
                            })

                        if cmp_rows:
                            st.dataframe(pd.DataFrame(cmp_rows).set_index("Sequence"), use_container_width=True)

                            # Winner declaration
                            a_row = cmp_rows[0]
                            b_row = cmp_rows[1]
                            a_whiff = float(a_row["Whiff%"].replace("%", "")) if a_row["Whiff%"] != "-" else 0
                            b_whiff = float(b_row["Whiff%"].replace("%", "")) if b_row["Whiff%"] != "-" else 0
                            a_csw = float(a_row["CSW%"].replace("%", "")) if a_row["CSW%"] != "-" else 0
                            b_csw = float(b_row["CSW%"].replace("%", "")) if b_row["CSW%"] != "-" else 0
                            a_tun = float(a_row["Tunnel Avg"]) if a_row["Tunnel Avg"] != "-" else 0
                            b_tun = float(b_row["Tunnel Avg"]) if b_row["Tunnel Avg"] != "-" else 0
                            if a_whiff + a_csw + a_tun * 0.5 > b_whiff + b_csw + b_tun * 0.5:
                                winner = seq_a
                                w_clr = "#2ca02c"
                            elif b_whiff + b_csw + b_tun * 0.5 > a_whiff + a_csw + a_tun * 0.5:
                                winner = seq_b
                                w_clr = "#2ca02c"
                            else:
                                winner = "Tie"
                                w_clr = "#888"
                            if winner != "Tie":
                                st.markdown(
                                    f'<div style="padding:10px;background:#f0fff0;border-radius:8px;border:1px solid #cce0cc;'
                                    f'text-align:center;">'
                                    f'<span style="font-size:14px;font-weight:800;color:{w_clr} !important;">'
                                    f'Winner: {winner}</span></div>', unsafe_allow_html=True)
                    else:
                        st.info("Need at least 2 sequences with 3+ occurrences to compare.")

        else:
            st.info("Not enough columns to determine pitch sequences.")

    # ─── Tab: Count Leverage ──────────────────────
    with tab_count:
        section_header("Count Leverage Analysis")
        st.caption("Optimal pitch selection by count — what works best in each situation")

        pdf_c = pdf.dropna(subset=["Balls", "Strikes"]).copy()
        pdf_c["Balls"] = pdf_c["Balls"].astype(int)
        pdf_c["Strikes"] = pdf_c["Strikes"].astype(int)
        pdf_c["Count"] = pdf_c["Balls"].astype(str) + "-" + pdf_c["Strikes"].astype(str)

        pitch_types = sorted(pdf["TaggedPitchType"].dropna().unique())

        # For each count, show best pitch type by whiff% and CSW%
        section_header("Best Pitch by Count (Whiff%)")
        count_pitch_data = {}
        for b in range(4):
            for s in range(3):
                count_str = f"{b}-{s}"
                cd = pdf_c[(pdf_c["Balls"] == b) & (pdf_c["Strikes"] == s)]
                if len(cd) < 5:
                    continue
                best_pt = None
                best_whiff = -1
                pt_results = {}
                for pt in pitch_types:
                    pt_cd = cd[cd["TaggedPitchType"] == pt]
                    if len(pt_cd) < 3:
                        continue
                    sw = pt_cd[pt_cd["PitchCall"].isin(SWING_CALLS)]
                    wh = pt_cd[pt_cd["PitchCall"] == "StrikeSwinging"]
                    csw = pt_cd[pt_cd["PitchCall"].isin(["StrikeSwinging", "StrikeCalled"])]
                    bt = pt_cd[pt_cd["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
                    whiff = len(wh) / max(len(sw), 1) * 100 if len(sw) > 0 else 0
                    csw_pct = len(csw) / len(pt_cd) * 100
                    ev = bt["ExitSpeed"].mean() if len(bt) > 0 else np.nan
                    pt_results[pt] = {"Usage": f"{len(pt_cd)/len(cd)*100:.0f}%", "Whiff%": whiff,
                                       "CSW%": csw_pct, "Avg EV": ev, "n": len(pt_cd)}
                    if whiff > best_whiff:
                        best_whiff = whiff
                        best_pt = pt
                count_pitch_data[count_str] = {"best": best_pt, "best_whiff": best_whiff, "details": pt_results, "total": len(cd)}

        # Display as grid
        grid_best = [['' for _ in range(3)] for _ in range(4)]
        grid_whiff = np.full((4, 3), np.nan)
        for b in range(4):
            for s in range(3):
                k = f"{b}-{s}"
                if k in count_pitch_data and count_pitch_data[k]["best"]:
                    d = count_pitch_data[k]
                    grid_best[b][s] = f"{d['best']}\n{d['best_whiff']:.0f}%"
                    grid_whiff[b][s] = d["best_whiff"]

        fig_best = go.Figure(data=go.Heatmap(
            z=grid_whiff, text=grid_best, texttemplate="%{text}",
            x=["0 Strikes", "1 Strike", "2 Strikes"], y=["0 Balls", "1 Ball", "2 Balls", "3 Balls"],
            colorscale=[[0, "#f7f7f7"], [0.5, "#f7c631"], [1, "#d22d49"]],
            zmin=0, zmax=50, showscale=True,
            colorbar=dict(title="Whiff%", len=0.8),
            textfont=dict(size=11),
        ))
        fig_best.update_layout(**CHART_LAYOUT, height=350, title="Best Pitch + Whiff% by Count")
        st.plotly_chart(fig_best, use_container_width=True)

        # Pitch usage by count
        section_header("Pitch Usage by Count")
        usage_rows = []
        for b in range(4):
            for s in range(3):
                cd = pdf_c[(pdf_c["Balls"] == b) & (pdf_c["Strikes"] == s)]
                if len(cd) < 5:
                    continue
                row = {"Count": f"{b}-{s}", "Total": len(cd)}
                for pt in pitch_types:
                    row[pt] = f"{len(cd[cd['TaggedPitchType'] == pt])/len(cd)*100:.0f}%"
                usage_rows.append(row)
        if usage_rows:
            st.dataframe(pd.DataFrame(usage_rows).set_index("Count"), use_container_width=True)

        # Detailed count breakdown
        section_header("Detailed Count Cheat Sheet")
        selected_count = st.selectbox("Select Count", [f"{b}-{s}" for b in range(4) for s in range(3)], key=f"{key_prefix}_count")
        if selected_count in count_pitch_data:
            details = count_pitch_data[selected_count]["details"]
            detail_rows = []
            for pt, d in sorted(details.items(), key=lambda x: -x[1]["Whiff%"]):
                detail_rows.append({
                    "Pitch Type": pt,
                    "Usage": d["Usage"],
                    "n": d["n"],
                    "Whiff%": f"{d['Whiff%']:.1f}%",
                    "CSW%": f"{d['CSW%']:.1f}%",
                    "Avg EV": f"{d['Avg EV']:.1f}" if not pd.isna(d["Avg EV"]) else "-",
                })
            if detail_rows:
                st.dataframe(pd.DataFrame(detail_rows).set_index("Pitch Type"), use_container_width=True)
                best = detail_rows[0]
                st.success(f"**Recommendation in {selected_count}:** Throw **{detail_rows[0]['Pitch Type']}** — "
                           f"{detail_rows[0]['Whiff%']} whiff rate, {detail_rows[0]['CSW%']} CSW%")

        # Ahead vs Behind vs Even
        section_header("Situational Pitch Selection")
        pdf_c["Situation"] = "Even"
        pdf_c.loc[pdf_c["Balls"] < pdf_c["Strikes"], "Situation"] = "Ahead"
        pdf_c.loc[pdf_c["Balls"] > pdf_c["Strikes"], "Situation"] = "Behind"

        sit_rows = []
        for sit in ["Ahead", "Even", "Behind"]:
            sit_df = pdf_c[pdf_c["Situation"] == sit]
            if len(sit_df) < 10:
                continue
            sw = sit_df[sit_df["PitchCall"].isin(SWING_CALLS)]
            wh = sit_df[sit_df["PitchCall"] == "StrikeSwinging"]
            csw = sit_df[sit_df["PitchCall"].isin(["StrikeSwinging", "StrikeCalled"])]
            bt = sit_df[sit_df["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            top_pt = sit_df["TaggedPitchType"].value_counts().index[0] if len(sit_df) > 0 else "-"
            sit_rows.append({
                "Situation": sit,
                "Pitches": len(sit_df),
                "Top Pitch": top_pt,
                "Zone%": f"{in_zone_mask(sit_df).sum()/len(sit_df[sit_df['PlateLocSide'].notna()])*100:.1f}%" if sit_df["PlateLocSide"].notna().any() else "-",
                "Whiff%": f"{len(wh)/len(sw)*100:.1f}%" if len(sw) > 0 else "-",
                "CSW%": f"{len(csw)/len(sit_df)*100:.1f}%" if len(sit_df) > 0 else "-",
                "Avg EV": f"{bt['ExitSpeed'].mean():.1f}" if len(bt) > 0 else "-",
            })
        if sit_rows:
            st.dataframe(pd.DataFrame(sit_rows).set_index("Situation"), use_container_width=True)

    # ─── Tab: Effective Velocity ──────────────────
    with tab_effv:
        section_header("Effective Velocity Analysis")
        st.caption("Perceived velocity based on pitch location and extension — a 91mph fastball up-and-in plays like 94+ mph")

        ev_df = pdf.dropna(subset=["RelSpeed", "PlateLocSide", "PlateLocHeight"]).copy()
        if len(ev_df) < 20:
            st.info("Not enough location data for effective velocity analysis.")
        else:
            # Compute effective velocity if not present
            if "EffectiveVelo" in ev_df.columns and ev_df["EffectiveVelo"].notna().sum() > len(ev_df) * 0.5:
                ev_df["EffVelo"] = ev_df["EffectiveVelo"]
            else:
                # Estimate: up-and-in adds ~2-3 mph, down-and-away subtracts ~2-3 mph
                # Hitter's reaction zone: pitches up and glove-side arrive "faster"
                loc_adj = (ev_df["PlateLocHeight"] - 2.5) * 1.5 + ev_df["PlateLocSide"].abs() * (-0.5)
                ev_df["EffVelo"] = ev_df["RelSpeed"] + loc_adj

            col_scatter, col_diff = st.columns(2)
            with col_scatter:
                section_header("Effective Velocity by Location")
                fig_effv = go.Figure()
                fig_effv.add_trace(go.Scatter(
                    x=ev_df["PlateLocSide"], y=ev_df["PlateLocHeight"],
                    mode="markers",
                    marker=dict(size=6, color=ev_df["EffVelo"],
                                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                cmin=ev_df["EffVelo"].quantile(0.05),
                                cmax=ev_df["EffVelo"].quantile(0.95),
                                showscale=True, colorbar=dict(title="Eff Velo", len=0.8),
                                line=dict(width=0.3, color="white")),
                    hovertemplate="Actual: %{customdata[0]:.1f}<br>Effective: %{marker.color:.1f}<extra></extra>",
                    customdata=ev_df[["RelSpeed"]].values,
                    showlegend=False,
                ))
                add_strike_zone(fig_effv)
                fig_effv.update_layout(**CHART_LAYOUT, height=420,
                                        xaxis=dict(range=[-2.5, 2.5], title="Horizontal", scaleanchor="y"),
                                        yaxis=dict(range=[0, 5], title="Vertical"))
                st.plotly_chart(fig_effv, use_container_width=True)

            with col_diff:
                section_header("Velo Differential (Effective - Actual)")
                ev_df["VeloDiff"] = ev_df["EffVelo"] - ev_df["RelSpeed"]
                fig_diff = go.Figure()
                fig_diff.add_trace(go.Scatter(
                    x=ev_df["PlateLocSide"], y=ev_df["PlateLocHeight"],
                    mode="markers",
                    marker=dict(size=6, color=ev_df["VeloDiff"],
                                colorscale=[[0, "#1f77b4"], [0.5, "#f7f7f7"], [1, "#d22d49"]],
                                cmid=0, showscale=True,
                                colorbar=dict(title="Diff", len=0.8),
                                line=dict(width=0.3, color="white")),
                    hovertemplate="Diff: %{marker.color:+.1f} mph<extra></extra>",
                    showlegend=False,
                ))
                add_strike_zone(fig_diff)
                fig_diff.update_layout(**CHART_LAYOUT, height=420,
                                        xaxis=dict(range=[-2.5, 2.5], title="Horizontal", scaleanchor="y"),
                                        yaxis=dict(range=[0, 5], title="Vertical"))
                st.plotly_chart(fig_diff, use_container_width=True)

            # Effective velo by pitch type
            section_header("Effective Velocity by Pitch Type")
            effv_pt_rows = []
            for pt in sorted(ev_df["TaggedPitchType"].dropna().unique()):
                pt_df = ev_df[ev_df["TaggedPitchType"] == pt]
                if len(pt_df) < 5:
                    continue
                effv_pt_rows.append({
                    "Pitch Type": pt,
                    "Actual Velo": f"{pt_df['RelSpeed'].mean():.1f}",
                    "Eff Velo": f"{pt_df['EffVelo'].mean():.1f}",
                    "Diff": f"{(pt_df['EffVelo'] - pt_df['RelSpeed']).mean():+.1f}",
                    "Max Eff": f"{pt_df['EffVelo'].max():.1f}",
                    "Min Eff": f"{pt_df['EffVelo'].min():.1f}",
                })
            if effv_pt_rows:
                st.dataframe(pd.DataFrame(effv_pt_rows).set_index("Pitch Type"), use_container_width=True)

            # Velo tunneling — show how pitch types overlap in effective velo
            section_header("Velocity Tunneling")
            st.caption("When different pitches arrive at similar effective velocities, hitters can't distinguish them")
            fig_tunnel = go.Figure()
            for pt in sorted(ev_df["TaggedPitchType"].dropna().unique()):
                pt_df = ev_df[ev_df["TaggedPitchType"] == pt]
                if len(pt_df) < 10:
                    continue
                clr = PITCH_COLORS.get(pt, "#aaa")
                fig_tunnel.add_trace(go.Violin(
                    y=pt_df["EffVelo"], name=pt,
                    box_visible=True, meanline_visible=True,
                    fillcolor=clr, line_color=clr, opacity=0.6,
                ))
            fig_tunnel.update_layout(**CHART_LAYOUT, height=380, showlegend=False,
                                      yaxis_title="Effective Velocity (mph)")
            st.plotly_chart(fig_tunnel, use_container_width=True)
