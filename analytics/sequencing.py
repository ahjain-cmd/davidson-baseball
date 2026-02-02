"""Pitch sequence helpers — tunnel lookup and 3-pitch sequence building."""

import numpy as np
import pandas as pd


_hard_pitches = {"Fastball", "Sinker", "Cutter"}


def _lookup_tunnel(a, b, tun_df):
    """Lookup tunnel score and grade for a pitch pair."""
    if not isinstance(tun_df, pd.DataFrame) or tun_df.empty:
        return np.nan, "-"
    m = tun_df[
        ((tun_df["Pitch A"] == a) & (tun_df["Pitch B"] == b)) |
        ((tun_df["Pitch A"] == b) & (tun_df["Pitch B"] == a))
    ]
    if m.empty:
        return np.nan, "-"
    return m.iloc[0]["Tunnel Score"], m.iloc[0]["Grade"]


def _lookup_seq(setup, follow, seq_df):
    """Lookup sequence whiff% and chase% for a pitch pair. Returns (whiff%, chase%)."""
    if not isinstance(seq_df, pd.DataFrame) or seq_df.empty:
        return np.nan, np.nan
    m = seq_df[(seq_df["Setup Pitch"] == setup) & (seq_df["Follow Pitch"] == follow)]
    if m.empty:
        return np.nan, np.nan
    return m.iloc[0]["Whiff%"], m.iloc[0].get("Chase%", np.nan)


def _build_3pitch_sequences(sorted_ps, hd, tun_df, seq_df):
    """Build best 3-pitch sequences: setup -> bridge -> putaway.
    HITTER-AWARE: Picks the putaway pitch based on the hitter's specific
    vulnerability, then finds the best setup/bridge path to get there.
    P1 (setup) must be a primary pitch (>= 15% usage) — you don't lead with
    a 10% sinker. Returns up to 3 sequences with different putaway pitches."""
    pitches = [name for name, data in sorted_ps if data.get("count", 0) >= 10]
    pitch_data = {name: data for name, data in sorted_ps if data.get("count", 0) >= 10}
    pitch_usage = {name: data.get("usage", 0) or 0 for name, data in sorted_ps}
    comp_scores = {name: data.get("score", 50) for name, data in sorted_ps}
    if len(pitches) < 2:
        return []
    # P1 candidates: must have meaningful usage (>= 15%), or fallback to top 2 by usage
    setup_candidates = [p for p in pitches if pitch_usage.get(p, 0) >= 15]
    if len(setup_candidates) < 2:
        setup_candidates = sorted(pitches, key=lambda p: pitch_usage.get(p, 0), reverse=True)[:2]

    # Step 1: Rank putaway candidates by hitter-specific vulnerability
    putaway_scores = {}
    for p in pitches:
        is_hard = p in _hard_pitches
        their_2k = hd.get("whiff_2k_hard" if is_hard else "whiff_2k_os", np.nan)
        comp = comp_scores.get(p, 50)
        whiff = pitch_data.get(p, {}).get("our_whiff", np.nan)
        chase = pitch_data.get(p, {}).get("our_chase", np.nan)
        score = comp * 0.50
        if not pd.isna(their_2k):
            score += min(their_2k / 40 * 100, 100) * 0.25
        if not pd.isna(whiff):
            score += min(whiff / 50 * 100, 100) * 0.15
        if not pd.isna(chase):
            score += min(chase / 40 * 100, 100) * 0.10
        putaway_scores[p] = score
    ranked_putaways = sorted(putaway_scores.items(), key=lambda x: x[1], reverse=True)

    # Step 2: For each putaway, find best setup -> bridge path
    results = []
    for p3, p3_score in ranked_putaways:
        best_path = None
        best_path_score = -1
        for p2 in pitches:
            if p2 == p3:
                t_self, _ = _lookup_tunnel(p2, p3, tun_df)
                if pd.isna(t_self) or t_self <= 50:
                    continue
            for p1 in setup_candidates:  # P1 restricted to primary pitches
                if p1 == p2:
                    continue
                t12, g12 = _lookup_tunnel(p1, p2, tun_df)
                t23, g23 = _lookup_tunnel(p2, p3, tun_df)
                t12_bad = pd.isna(t12) or t12 < 25
                t23_bad = pd.isna(t23) or t23 < 25
                if t12_bad and t23_bad:
                    continue
                sw12, ch12 = _lookup_seq(p1, p2, seq_df)
                sw23, ch23 = _lookup_seq(p2, p3, seq_df)
                parts, wts = [], []
                # Tunnel quality (45%): t12 weight=18, t23 weight=27
                if not pd.isna(t12):
                    parts.append(t12); wts.append(18)
                if not pd.isna(t23):
                    parts.append(t23); wts.append(27)
                # Outcome effectiveness (40%): sw23=25, sw12=10, ch23=5
                if not pd.isna(sw23):
                    parts.append(min(sw23 / 50 * 100, 100)); wts.append(25)
                else:
                    parts.append(30); wts.append(25)
                if not pd.isna(sw12):
                    parts.append(min(sw12 / 50 * 100, 100)); wts.append(10)
                if not pd.isna(ch23):
                    parts.append(min(ch23 / 40 * 100, 100)); wts.append(5)
                # Pitch quality (15%): putaway composite=10, EffV gap=5
                parts.append(comp_scores.get(p3, 50)); wts.append(10)
                p1_effv = pitch_data.get(p1, {}).get("eff_velo", np.nan)
                p3_effv = pitch_data.get(p3, {}).get("eff_velo", np.nan)
                if not pd.isna(p1_effv) and not pd.isna(p3_effv):
                    gap = abs(p1_effv - p3_effv)
                    parts.append(min(25 + gap * 5, 100)); wts.append(5)
                else:
                    p1_velo = pitch_data.get(p1, {}).get("velo", np.nan)
                    p3_velo = pitch_data.get(p3, {}).get("velo", np.nan)
                    if not pd.isna(p1_velo) and not pd.isna(p3_velo):
                        gap = abs(p1_velo - p3_velo)
                        parts.append(min(25 + gap * 5, 100)); wts.append(5)
                if not wts:
                    continue
                path_score = sum(p * w for p, w in zip(parts, wts)) / sum(wts)
                if path_score > best_path_score:
                    best_path_score = path_score
                    ev_gap = abs(p1_effv - p3_effv) if not pd.isna(p1_effv) and not pd.isna(p3_effv) else np.nan
                    is_hard_p3 = p3 in _hard_pitches
                    their_2k = hd.get("whiff_2k_hard" if is_hard_p3 else "whiff_2k_os", np.nan)
                    best_path = {
                        "seq": f"{p1} → {p2} → {p3}", "p1": p1, "p2": p2, "p3": p3,
                        "score": round(p3_score * 0.35 + best_path_score * 0.65, 1),
                        "t12": t12, "t23": t23, "sw23": sw23, "their_2k": their_2k,
                        "effv_gap": ev_gap,
                    }
        if best_path:
            results.append(best_path)
        if len(results) >= 3:
            break
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
