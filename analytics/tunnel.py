"""Tunnel score computation — physics-based pitch pair analysis."""

import os
import json
import math

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

from config import (
    PARQUET_PATH, CACHE_DIR, TUNNEL_BENCH_PATH, TUNNEL_WEIGHTS_PATH,
    PLATE_SIDE_MAX, PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX,
    MIN_TUNNEL_SEQ_PCT, filter_minor_pitches, SWING_CALLS,
)
from data.loader import (
    _precompute_table_exists,
    _read_precompute_table,
    query_precompute,
    query_population,
)

# Outcome-boost settings (small, sample-size gated)
WHIFF_MIN_PAIRS = 15
WHIFF_SHRINK_K = 40
WHIFF_BOOST_MAX = 8.0
WHIFF_BOOST_SCALE = 20.0  # percentage-point diff for max boost


def _parquet_fingerprint():
    try:
        return {"path": PARQUET_PATH, "mtime": os.path.getmtime(PARQUET_PATH), "size": os.path.getsize(PARQUET_PATH)}
    except OSError:
        return {"path": PARQUET_PATH, "mtime": None, "size": None}


def _precompute_meta_matches():
    if not _precompute_table_exists("meta"):
        return False
    df = _read_precompute_table("meta")
    if df.empty:
        return False
    fp = _parquet_fingerprint()
    row = df.iloc[0]
    return (
        row.get("parquet_path") == fp.get("path") and
        row.get("parquet_mtime") == fp.get("mtime") and
        row.get("parquet_size") == fp.get("size")
    )


@st.cache_data(show_spinner=False)
def _load_pair_outcomes():
    """Return (pair_whiff_baselines, global_whiff) in percent."""
    # Prefer precomputed baselines if available and up to date.
    if _precompute_table_exists("tunnel_pair_outcomes") and _precompute_meta_matches():
        df = _read_precompute_table("tunnel_pair_outcomes")
        if not df.empty and {"pair_type", "whiff_rate", "n_pairs"}.issubset(df.columns):
            baselines = {}
            global_whiff = None
            for _, row in df.iterrows():
                pair = row.get("pair_type")
                if not pair:
                    continue
                if pair == "__ALL__":
                    global_whiff = float(row.get("whiff_rate", np.nan))
                    continue
                baselines[pair] = float(row.get("whiff_rate", np.nan))
            if global_whiff is None and baselines:
                weights = df[df["pair_type"] != "__ALL__"]["n_pairs"].astype(float)
                rates = df[df["pair_type"] != "__ALL__"]["whiff_rate"].astype(float)
                if not weights.empty:
                    global_whiff = float(np.average(rates, weights=weights))
            return baselines, global_whiff

    # Fallback: compute from available tables
    table = None
    if _precompute_table_exists("trackman_pop"):
        table = "trackman_pop"
        runner = query_precompute
    else:
        table = "trackman"
        runner = query_population

    sql = f"""
    WITH ordered AS (
        SELECT GameID, Inning, PAofInning, Batter, Pitcher, PitchofPA,
               TaggedPitchType, PitchCall
        FROM {table}
        WHERE TaggedPitchType IS NOT NULL AND PitchCall IS NOT NULL
    ),
    pairs AS (
        SELECT
            CASE
                WHEN TaggedPitchType < prev_type THEN TaggedPitchType || '/' || prev_type
                ELSE prev_type || '/' || TaggedPitchType
            END AS pair_type,
            PitchCall,
            CASE WHEN PitchCall IN ('StrikeSwinging', 'InPlay', 'FoulBall', 'FoulBallNotFieldable', 'FoulBallFieldable') THEN 1 ELSE 0 END AS is_swing,
            CASE WHEN PitchCall = 'StrikeSwinging' THEN 1 ELSE 0 END AS is_whiff
        FROM (
            SELECT *,
                   LAG(TaggedPitchType) OVER (
                       PARTITION BY GameID, Inning, PAofInning, Batter, Pitcher
                       ORDER BY PitchofPA
                   ) AS prev_type
            FROM ordered
        )
        WHERE prev_type IS NOT NULL AND TaggedPitchType != prev_type
    )
    SELECT pair_type,
           COUNT(*) AS n_pairs,
           SUM(is_swing) AS n_swings,
           CASE WHEN SUM(is_swing) >= 5 THEN SUM(is_whiff) * 100.0 / SUM(is_swing) ELSE NULL END AS whiff_rate
    FROM pairs
    GROUP BY pair_type
    """
    try:
        df = runner(sql)
    except Exception:
        return {}, None
    if df.empty:
        return {}, None
    # Filter to pairs with valid whiff rates (enough swings to compute)
    df_valid = df[df["whiff_rate"].notna()].copy()
    if df_valid.empty:
        return {}, None
    baselines = {r["pair_type"]: float(r["whiff_rate"]) for _, r in df_valid.iterrows()}
    global_whiff = float(np.average(df_valid["whiff_rate"], weights=df_valid["n_swings"]))
    return baselines, global_whiff


def _load_tunnel_benchmarks():
    if not os.path.exists(TUNNEL_BENCH_PATH):
        if _precompute_table_exists("tunnel_benchmarks") and _precompute_meta_matches():
            df = _read_precompute_table("tunnel_benchmarks")
            if df.empty:
                return None
            benches = {}
            for _, row in df.iterrows():
                pair = row.get("pair_type")
                if not pair:
                    continue
                benches[pair] = (
                    float(row.get("p10", np.nan)),
                    float(row.get("p25", np.nan)),
                    float(row.get("p50", np.nan)),
                    float(row.get("p75", np.nan)),
                    float(row.get("p90", np.nan)),
                    float(row.get("mean", np.nan)),
                    float(row.get("std", np.nan)),
                )
            return benches or None
        return None
    try:
        with open(TUNNEL_BENCH_PATH, "r", encoding="utf-8") as f:
            blob = json.load(f)
        if blob.get("fingerprint") != _parquet_fingerprint():
            # Fallback to precomputed DB if available
            if _precompute_table_exists("tunnel_benchmarks") and _precompute_meta_matches():
                df = _read_precompute_table("tunnel_benchmarks")
                if df.empty:
                    return None
                benches = {}
                for _, row in df.iterrows():
                    pair = row.get("pair_type")
                    if not pair:
                        continue
                    benches[pair] = (
                        float(row.get("p10", np.nan)),
                        float(row.get("p25", np.nan)),
                        float(row.get("p50", np.nan)),
                        float(row.get("p75", np.nan)),
                        float(row.get("p90", np.nan)),
                        float(row.get("mean", np.nan)),
                        float(row.get("std", np.nan)),
                    )
                return benches or None
            return None
        return blob.get("benchmarks")
    except Exception:
        if _precompute_table_exists("tunnel_benchmarks") and _precompute_meta_matches():
            df = _read_precompute_table("tunnel_benchmarks")
            if df.empty:
                return None
            benches = {}
            for _, row in df.iterrows():
                pair = row.get("pair_type")
                if not pair:
                    continue
                benches[pair] = (
                    float(row.get("p10", np.nan)),
                    float(row.get("p25", np.nan)),
                    float(row.get("p50", np.nan)),
                    float(row.get("p75", np.nan)),
                    float(row.get("p90", np.nan)),
                    float(row.get("mean", np.nan)),
                    float(row.get("std", np.nan)),
                )
            return benches or None
        return None


def _save_tunnel_benchmarks(benchmarks):
    os.makedirs(CACHE_DIR, exist_ok=True)
    blob = {"fingerprint": _parquet_fingerprint(), "benchmarks": benchmarks}
    with open(TUNNEL_BENCH_PATH, "w", encoding="utf-8") as f:
        json.dump(blob, f)


def _load_tunnel_weights():
    if not os.path.exists(TUNNEL_WEIGHTS_PATH):
        if _precompute_table_exists("tunnel_weights") and _precompute_meta_matches():
            df = _read_precompute_table("tunnel_weights")
            if not df.empty and "weights_json" in df.columns:
                try:
                    return json.loads(df.iloc[0]["weights_json"])
                except Exception:
                    return None
        return None
    try:
        with open(TUNNEL_WEIGHTS_PATH, "r", encoding="utf-8") as f:
            blob = json.load(f)
        if blob.get("fingerprint") != _parquet_fingerprint():
            if _precompute_table_exists("tunnel_weights") and _precompute_meta_matches():
                df = _read_precompute_table("tunnel_weights")
                if not df.empty and "weights_json" in df.columns:
                    try:
                        return json.loads(df.iloc[0]["weights_json"])
                    except Exception:
                        return None
            return None
        return blob.get("weights")
    except Exception:
        if _precompute_table_exists("tunnel_weights") and _precompute_meta_matches():
            df = _read_precompute_table("tunnel_weights")
            if not df.empty and "weights_json" in df.columns:
                try:
                    return json.loads(df.iloc[0]["weights_json"])
                except Exception:
                    return None
        return None


def _save_tunnel_weights(weights):
    os.makedirs(CACHE_DIR, exist_ok=True)
    blob = {"fingerprint": _parquet_fingerprint(), "weights": weights}
    with open(TUNNEL_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(blob, f)


@st.cache_data(show_spinner="Building tunnel population database...")
def _build_tunnel_population(_data):
    """Compute raw tunnel composites for every pitcher in the database.

    Returns dict: pair_type (e.g. 'Fastball/Slider') → sorted numpy array of
    raw tunnel scores across all pitchers.  Used for percentile grading — a
    pitcher's Fastball/Slider tunnel is ranked against *all* Fastball/Slider
    tunnels in college baseball.
    """
    pop = {}  # pair_type → [raw_score, ...]
    # Pre-filter: only pitchers with ≥50 pitches and ≥2 pitch types qualify
    # for tunnel computation (reduces 12k → ~2-3k meaningful pitchers)
    pitcher_counts = _data.groupby("Pitcher").agg(
        n=("Pitcher", "size"),
        n_types=("TaggedPitchType", "nunique"),
    )
    eligible = pitcher_counts[(pitcher_counts["n"] >= 50) & (pitcher_counts["n_types"] >= 2)].index
    for pitcher in eligible:
        pdf = _data[_data["Pitcher"] == pitcher]
        tdf = _compute_tunnel_score(pdf)  # raw mode (no tunnel_pop)
        if tdf.empty:
            continue
        for _, row in tdf.iterrows():
            pair_key = '/'.join(sorted([row["Pitch A"], row["Pitch B"]]))
            pop.setdefault(pair_key, []).append(row["Tunnel Score"])
    # Convert to sorted arrays for fast percentile lookup
    for k in pop:
        pop[k] = np.array(sorted(pop[k]))
    return pop


def _compute_tunnel_score(pdf, tunnel_pop=None):
    """Compute tunnel scores using Euler-integrated flight paths and
    data-driven percentile grading.

    V8 — Average-based, break-driven tunnel scoring:
      1. Commit separation from pitch-type AVERAGES via IVB/HB model
         (not pitch-by-pitch), so break profiles drive the score, not
         command scatter or velocity.  Physics audit showed IVB/HB
         contributes 4-5" commit sep vs <1.5" from 10 mph velo gap.
      2. ABSOLUTE scoring (not per-pair-type percentile).  Lower commit
         sep = better tunnel regardless of pair type — a 3" FB/SL is more
         deceptive than 5" CB/SL, period.
      3. Weighted composite:
           commit_sep  55%  (lower → more deceptive — hitter can't distinguish)
           plate_sep   19%  (higher → more whiffs given swing — late break)
           rel_sep     10%  (lower → better — consistent arm slot)
           rel_angle    8%  (lower → better — similar launch angles)
           move_div     8%  (higher → better — pitches diverge at plate)
      4. Release-point variance penalty on rel_sep factor.
      5. Pitch-by-pitch pairing used only for whiff% outcome boost.
    """
    # Minimum columns: need pitch type, release point, plate location, and velo.
    # IVB/HB are preferred but we can fall back to 9-param or gravity-only.
    req_base = ["TaggedPitchType", "RelHeight", "RelSide", "PlateLocHeight",
                "PlateLocSide", "RelSpeed"]
    if not all(c in pdf.columns for c in req_base):
        return pd.DataFrame()

    # Filter to plausible plate locations and drop low-usage pitch types.
    pdf = pdf[
        pdf["PlateLocSide"].between(-PLATE_SIDE_MAX, PLATE_SIDE_MAX) &
        pdf["PlateLocHeight"].between(PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX)
    ].copy()
    pdf = filter_minor_pitches(pdf, min_pct=MIN_TUNNEL_SEQ_PCT)
    if pdf.empty:
        return pd.DataFrame()

    has_ivb = "InducedVertBreak" in pdf.columns and "HorzBreak" in pdf.columns
    has_rel_angle = "VertRelAngle" in pdf.columns and "HorzRelAngle" in pdf.columns
    has_9p = all(c in pdf.columns for c in ["x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0"])

    pitch_types = pdf["TaggedPitchType"].unique()
    if len(pitch_types) < 2:
        return pd.DataFrame()

    MOUND_DIST = 60.5   # feet, rubber to plate
    GRAVITY = 32.17      # ft/s²
    COMMIT_TIME = 0.280  # seconds before plate arrival (research-backed)
    N_STEPS = 20         # Euler integration steps

    # Pair-type benchmarks loaded from population cache (commit_sep percentiles).
    # Format: (p10, p25, p50, p75, p90, mean, std)
    PAIR_BENCHMARKS = _load_tunnel_benchmarks() or {}
    # Pair-type outcome baselines (whiff%) for shrinkage
    PAIR_OUTCOMES, GLOBAL_WHIFF = _load_pair_outcomes()
    DEFAULT_BENCHMARK = (1.2, 2.1, 3.3, 4.9, 6.8, 3.8, 2.7)

    # ── Per-pitch-type aggregates (used as fallback & for diagnostics) ──
    agg_cols = {
        "rel_h": ("RelHeight", "mean"), "rel_s": ("RelSide", "mean"),
        "rel_h_std": ("RelHeight", "std"), "rel_s_std": ("RelSide", "std"),
        "loc_h": ("PlateLocHeight", "mean"), "loc_s": ("PlateLocSide", "mean"),
        "velo": ("RelSpeed", "mean"), "count": ("RelSpeed", "count"),
    }
    if has_ivb:
        agg_cols["ivb"] = ("InducedVertBreak", "mean")
        agg_cols["hb"] = ("HorzBreak", "mean")
    if "Extension" in pdf.columns:
        agg_cols["ext"] = ("Extension", "mean")
    # 9-param aggregates (for fallback trajectory)
    if has_rel_angle:
        agg_cols["vert_rel_angle"] = ("VertRelAngle", "mean")
        agg_cols["horz_rel_angle"] = ("HorzRelAngle", "mean")
    if has_9p:
        for c9 in ["x0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0"]:
            agg_cols[c9] = (c9, "mean")
    agg = pdf.groupby("TaggedPitchType").agg(**agg_cols).dropna(subset=["rel_h", "velo"])
    agg = agg[agg["count"] >= 10]  # need meaningful sample per pitch type
    if len(agg) < 2:
        return pd.DataFrame()
    if "ext" not in agg.columns:
        agg["ext"] = 6.0
    if "ivb" not in agg.columns:
        agg["ivb"] = np.nan
    if "hb" not in agg.columns:
        agg["hb"] = np.nan
    # Fill NaN stds with 0 (single-pitch groups)
    agg["rel_h_std"] = agg["rel_h_std"].fillna(0)
    agg["rel_s_std"] = agg["rel_s_std"].fillna(0)

    # ── Kinematic positions at commit/plate (no Euler discretization) ──
    def _commit_plate_9param(x0_val, y0_val, z0_val, vx0_val, vy0_val, vz0_val,
                             ax0_val, ay0_val, az0_val):
        """Compute commit/plate positions using Trackman 9-parameter model."""
        # Solve y0 + vy0*t + 0.5*ay0*t^2 = 0 for t>0
        a = 0.5 * ay0_val
        b = vy0_val
        c = y0_val
        if a == 0 and b == 0:
            return None
        if a == 0:
            t_candidates = [(-c / b)] if b != 0 else []
        else:
            disc = b * b - 4 * a * c
            if disc < 0:
                return None
            sqrt_disc = np.sqrt(disc)
            t_candidates = [(-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)]
        t_candidates = [t for t in t_candidates if t > 0]
        if not t_candidates:
            return None
        t_total = min(t_candidates)
        commit_t = max(0, t_total - COMMIT_TIME)

        def _pos_at(t):
            x = x0_val + vx0_val * t + 0.5 * ax0_val * t * t
            z = z0_val + vz0_val * t + 0.5 * az0_val * t * t
            return -x, z

        c_side, c_h = _pos_at(commit_t)
        p_side, p_h = _pos_at(t_total)
        return (c_side, c_h), (p_side, p_h), t_total

    def _commit_plate_ivb(rel_h, rel_s, loc_h, loc_s, ivb, hb, velo_mph, ext):
        """Compute commit/plate positions using IVB/HB constant-accel model."""
        ext = ext if not pd.isna(ext) else 6.0
        actual_dist = MOUND_DIST - ext
        velo_fps = velo_mph * 5280.0 / 3600.0
        if velo_fps < 50:
            velo_fps = 50.0
        t_total = actual_dist / velo_fps
        T = t_total
        ivb_ft = ivb / 12.0
        hb_ft = hb / 12.0
        a_ivb = 2.0 * ivb_ft / (t_total ** 2) if t_total > 0 else 0
        a_hb = 2.0 * hb_ft / (t_total ** 2) if t_total > 0 else 0
        vy0 = (loc_h - rel_h + 0.5 * GRAVITY * T**2 - 0.5 * a_ivb * T**2) / T if T > 0 else 0
        vx0 = (loc_s - rel_s - 0.5 * a_hb * T**2) / T if T > 0 else 0

        def _pos_at(t):
            x = rel_s + vx0 * t + 0.5 * a_hb * t * t
            y = rel_h + vy0 * t - 0.5 * GRAVITY * t * t + 0.5 * a_ivb * t * t
            return x, y

        commit_t = max(0, t_total - COMMIT_TIME)
        c_side, c_h = _pos_at(commit_t)
        p_side, p_h = _pos_at(t_total)
        return (c_side, c_h), (p_side, p_h), t_total

    # ── Method 3: Gravity-only trajectory (no break data) ──
    def _commit_plate_gravity(rel_h, rel_s, loc_h, loc_s, velo_mph, ext):
        """Gravity-only fallback (IVB=HB=0)."""
        return _commit_plate_ivb(rel_h, rel_s, loc_h, loc_s, 0.0, 0.0, velo_mph, ext)

    # ── Unified trajectory dispatcher ──
    def _compute_commit_plate(row_data):
        """Choose best available physics model and return commit/plate positions."""
        ivb_val = row_data.get("ivb", np.nan) if not isinstance(row_data, pd.Series) else row_data.get("ivb", np.nan)
        hb_val = row_data.get("hb", np.nan) if not isinstance(row_data, pd.Series) else row_data.get("hb", np.nan)
        # For individual pitches (Series), use column names directly
        if isinstance(row_data, pd.Series):
            ivb_val = row_data.get("InducedVertBreak", np.nan)
            hb_val = row_data.get("HorzBreak", np.nan)

        rel_h = row_data.get("rel_h", row_data.get("RelHeight", np.nan))
        rel_s = row_data.get("rel_s", row_data.get("RelSide", np.nan))
        loc_h = row_data.get("loc_h", row_data.get("PlateLocHeight", np.nan))
        loc_s = row_data.get("loc_s", row_data.get("PlateLocSide", np.nan))
        velo = row_data.get("velo", row_data.get("RelSpeed", np.nan))
        ext = row_data.get("ext", row_data.get("Extension", 6.0))

        if pd.isna(rel_h) or pd.isna(velo) or pd.isna(loc_h):
            return None

        # Method 1: 9-param model (most physically accurate when available)
        if isinstance(row_data, pd.Series):
            x0_v = row_data.get("x0", np.nan)
            y0_v = row_data.get("y0", np.nan)
            z0_v = row_data.get("z0", np.nan)
            vx0_v = row_data.get("vx0", np.nan)
            vy0_v = row_data.get("vy0", np.nan)
            vz0_v = row_data.get("vz0", np.nan)
            ax0_v = row_data.get("ax0", np.nan)
            ay0_v = row_data.get("ay0", np.nan)
            az0_v = row_data.get("az0", np.nan)
        else:
            x0_v = row_data.get("x0", np.nan)
            y0_v = row_data.get("y0", np.nan)
            z0_v = row_data.get("z0", np.nan)
            vx0_v = row_data.get("vx0", np.nan)
            vy0_v = row_data.get("vy0", np.nan)
            vz0_v = row_data.get("vz0", np.nan)
            ax0_v = row_data.get("ax0", np.nan)
            ay0_v = row_data.get("ay0", np.nan)
            az0_v = row_data.get("az0", np.nan)
        if not pd.isna(x0_v) and not pd.isna(y0_v) and not pd.isna(vx0_v) and not pd.isna(vy0_v):
            result = _commit_plate_9param(x0_v, y0_v, z0_v, vx0_v, vy0_v, vz0_v,
                                          ax0_v, ay0_v, az0_v)
            if result is not None:
                return result

        # Method 2: IVB/HB model (fallback when 9-param missing)
        if not pd.isna(ivb_val) and not pd.isna(hb_val):
            return _commit_plate_ivb(rel_h, rel_s, loc_h, loc_s, ivb_val, hb_val, velo, ext)

        # Method 3: Gravity-only
        if not pd.isna(rel_h) and not pd.isna(loc_h):
            return _commit_plate_gravity(rel_h, rel_s, loc_h, loc_s, velo, ext)

        # Method 4: Data too broken
        return None

    # ── Build pitch-by-pitch pair data (Improvement #4) ──
    pair_scores = {}  # (typeA, typeB) -> list of (row_a, row_b) pairs
    has_pbp = {"PitchofPA", "Batter", "PAofInning", "GameID"}.issubset(pdf.columns)
    if has_pbp:
        pbp_req = [
            "TaggedPitchType", "RelHeight", "RelSide", "PlateLocHeight",
            "PlateLocSide", "RelSpeed", "GameID", "PAofInning", "PitchofPA", "Batter",
        ]
        pbp = pdf.dropna(subset=pbp_req).copy()
        if "Extension" not in pbp.columns:
            pbp["Extension"] = 6.0
        else:
            pbp["Extension"] = pbp["Extension"].fillna(6.0)
        # Sort by PA and pitch order within PA
        group_cols = ["GameID", "Inning", "PAofInning", "Batter"]
        if "Pitcher" in pbp.columns:
            group_cols.append("Pitcher")
        group_cols = [c for c in group_cols if c in pbp.columns]
        pbp = pbp.sort_values(group_cols + ["PitchofPA"])
        # Build consecutive pairs within each PA
        prev = pbp.groupby(group_cols).shift(1)
        diff_type = pbp["TaggedPitchType"] != prev["TaggedPitchType"]
        pair_mask = prev["TaggedPitchType"].notna() & diff_type
        pair_idx = pbp.index[pair_mask]
        for pidx in pair_idx:
            crow = pbp.loc[pidx]
            prow = prev.loc[pidx]
            tA = prow["TaggedPitchType"]
            tB = crow["TaggedPitchType"]
            if pd.isna(tA) or pd.isna(tB):
                continue
            key = tuple(sorted([tA, tB]))
            if key not in pair_scores:
                pair_scores[key] = []
            pair_scores[key].append((prow, crow))

    # ── Score each pitch-type pair ──
    def _score_single_pair_from_rows(row_a, row_b):
        """Compute raw tunnel metrics for a single pitch pair using dispatcher.
        row_a, row_b: dict-like (pd.Series or agg row) with pitch columns."""
        result_a = _compute_commit_plate(row_a)
        result_b = _compute_commit_plate(row_b)
        if result_a is None or result_b is None:
            return None
        (cax, cay), (pax, pay), _ = result_a
        (cbx, cby), (pbx, pby), _ = result_b
        commit_sep = np.sqrt((cay - cby)**2 + (cax - cbx)**2) * 12  # inches

        # Release separation
        rel_h_a = row_a.get("rel_h", row_a.get("RelHeight", np.nan))
        rel_s_a = row_a.get("rel_s", row_a.get("RelSide", np.nan))
        rel_h_b = row_b.get("rel_h", row_b.get("RelHeight", np.nan))
        rel_s_b = row_b.get("rel_s", row_b.get("RelSide", np.nan))
        rel_sep = np.sqrt((rel_h_a - rel_h_b)**2 + (rel_s_a - rel_s_b)**2) * 12

        # Plate separation
        plate_sep = np.sqrt((pay - pby)**2 + (pax - pbx)**2) * 12

        # Movement divergence
        ivb_a = row_a.get("ivb", row_a.get("InducedVertBreak", 0))
        hb_a = row_a.get("hb", row_a.get("HorzBreak", 0))
        ivb_b = row_b.get("ivb", row_b.get("InducedVertBreak", 0))
        hb_b = row_b.get("hb", row_b.get("HorzBreak", 0))
        # Treat NaN break as 0 for movement divergence calc
        ivb_a = 0 if pd.isna(ivb_a) else ivb_a
        hb_a = 0 if pd.isna(hb_a) else hb_a
        ivb_b = 0 if pd.isna(ivb_b) else ivb_b
        hb_b = 0 if pd.isna(hb_b) else hb_b
        move_div = np.sqrt((ivb_a - ivb_b)**2 + (hb_a - hb_b)**2)
        velo_a = row_a.get("velo", row_a.get("RelSpeed", 0))
        velo_b = row_b.get("velo", row_b.get("RelSpeed", 0))
        velo_gap = abs(velo_a - velo_b)

        # Release angle separation (degrees)
        vra_a = row_a.get("vert_rel_angle", row_a.get("VertRelAngle", np.nan))
        hra_a = row_a.get("horz_rel_angle", row_a.get("HorzRelAngle", np.nan))
        vra_b = row_b.get("vert_rel_angle", row_b.get("VertRelAngle", np.nan))
        hra_b = row_b.get("horz_rel_angle", row_b.get("HorzRelAngle", np.nan))
        if pd.notna(vra_a) and pd.notna(vra_b) and pd.notna(hra_a) and pd.notna(hra_b):
            rel_angle_sep = np.sqrt((vra_a - vra_b)**2 + (hra_a - hra_b)**2)
        else:
            rel_angle_sep = np.nan

        return commit_sep, rel_sep, plate_sep, move_div, velo_gap, rel_angle_sep

    rows = []
    types = list(agg.index)
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            a, b = agg.loc[types[i]], agg.loc[types[j]]
            pair_key = tuple(sorted([types[i], types[j]]))

            # Compute tunnel metrics from pitch-type AVERAGES.
            # Prefer IVB/HB model so break profiles (not velocity or
            # command scatter) drive the commit separation.
            ivb_ok = (pd.notna(a.ivb) and pd.notna(a.hb)
                      and pd.notna(b.ivb) and pd.notna(b.hb))
            if ivb_ok:
                result_a = _commit_plate_ivb(
                    a.rel_h, a.rel_s, a.loc_h, a.loc_s,
                    a.ivb, a.hb, a.velo, a.ext)
                result_b = _commit_plate_ivb(
                    b.rel_h, b.rel_s, b.loc_h, b.loc_s,
                    b.ivb, b.hb, b.velo, b.ext)
            else:
                result_a = _compute_commit_plate(a)
                result_b = _compute_commit_plate(b)
            if result_a is None or result_b is None:
                continue
            (cax, cay), (pax, pay), _ = result_a
            (cbx, cby), (pbx, pby), _ = result_b

            commit_sep = np.sqrt((cay - cby)**2 + (cax - cbx)**2) * 12
            plate_sep = np.sqrt((pay - pby)**2 + (pax - pbx)**2) * 12
            rel_sep = np.sqrt((a.rel_h - b.rel_h)**2
                              + (a.rel_s - b.rel_s)**2) * 12

            _ivb_a = 0 if pd.isna(a.ivb) else a.ivb
            _hb_a = 0 if pd.isna(a.hb) else a.hb
            _ivb_b = 0 if pd.isna(b.ivb) else b.ivb
            _hb_b = 0 if pd.isna(b.hb) else b.hb
            move_div = np.sqrt((_ivb_a - _ivb_b)**2 + (_hb_a - _hb_b)**2)
            velo_gap = abs(a.velo - b.velo)

            vra_a = getattr(a, 'vert_rel_angle', np.nan)
            hra_a = getattr(a, 'horz_rel_angle', np.nan)
            vra_b = getattr(b, 'vert_rel_angle', np.nan)
            hra_b = getattr(b, 'horz_rel_angle', np.nan)
            if (pd.notna(vra_a) and pd.notna(vra_b)
                    and pd.notna(hra_a) and pd.notna(hra_b)):
                rel_angle_sep = np.sqrt((vra_a - vra_b)**2
                                        + (hra_a - hra_b)**2)
            else:
                rel_angle_sep = np.nan

            # Track pitch-by-pitch whiff% for outcome boost only
            pbp_pairs = pair_scores.get(pair_key, [])
            n_pairs_used = len(pbp_pairs)
            pair_whiff = np.nan
            if len(pbp_pairs) >= WHIFF_MIN_PAIRS:
                swing_count = 0
                whiff_count = 0
                for prow, crow in pbp_pairs:
                    pitch_call = crow.get("PitchCall")
                    if pitch_call in SWING_CALLS:
                        swing_count += 1
                        if pitch_call == "StrikeSwinging":
                            whiff_count += 1
                pair_whiff = (float(whiff_count / swing_count * 100)
                              if swing_count >= 5 else np.nan)

            # Release-point variance penalty (#2)
            combined_rel_std = np.sqrt(
                (a.rel_h_std**2 + b.rel_h_std**2) / 2 +
                (a.rel_s_std**2 + b.rel_s_std**2) / 2
            ) * 12  # inches
            effective_rel_sep = rel_sep + 0.5 * combined_rel_std

    # TUNNEL SCORE (v8) — absolute, average-based, break-driven.

            type_a_norm = types[i]; type_b_norm = types[j]
            pair_label = '/'.join(sorted([type_a_norm, type_b_norm]))
            # Pair benchmarks still loaded for diagnostics only
            bm = PAIR_BENCHMARKS.get(pair_label, DEFAULT_BENCHMARK)
            bm_p10, bm_p25, bm_p50, bm_p75, bm_p90, bm_mean, bm_std = bm

            # 1. COMMIT SEPARATION (55% weight)
            # Absolute scale: lower commit_sep → higher score.
            # Uses pitch-type averages so IVB/HB break profiles dominate.
            # Typical avg-vs-avg range: 1-8" (elite ~1-3", poor ~6+).
            commit_pct = max(0, min(100, 100 - commit_sep * 10))

            # 2. PLATE SEPARATION (19% weight)
            # Higher plate_sep → better (more divergence at plate)
            # Normalise: 0" → 0, 30" → 100
            plate_pct = min(100, plate_sep / 30.0 * 100)

            # 3. RELEASE CONSISTENCY (10% weight)
            # Lower effective_rel_sep → better
            # Note: effective_rel_sep is already in inches (no conversion needed)
            # Scale: 0" → 100, 8" → 0 (typical range is 1-5")
            rel_pct = max(0, 100 - effective_rel_sep * 12.5)

            # 4. RELEASE ANGLE SEPARATION (8% weight)
            # Lower rel_angle_sep → better (similar launch angles = harder to read)
            # Normalise: 0° → 100, 5° → 0. Fallback to 50 (neutral) if data missing.
            if pd.notna(rel_angle_sep):
                rel_angle_pct = max(0, min(100, (1 - rel_angle_sep / 5.0) * 100))
            else:
                rel_angle_pct = 50

            # 5. MOVEMENT DIVERGENCE (8% weight)
            # Higher move_div → better.  Break profiles are the dominant driver
            # of commit separation (audit: 4-5" from break vs 1.4" from 10 mph
            # velo gap), so movement divergence gets the weight formerly on velo.
            move_pct = min(100, move_div / 30.0 * 100)

            # Velo gap removed from scoring — velocity's small effect on commit
            # separation is already captured by the physics model in Factor 1.
            # Velo gap is still reported in diagnostics.

            # V8 hardcoded weights — break-driven, physics-audited.
            # Cached regression weights (tunnel_weights.json) are ignored:
            # they were fit on old pitch-by-pitch scoring and distort the
            # new average-based, absolute model.
            raw_tunnel = round(
                commit_pct * 0.55 +
                plate_pct * 0.19 +
                rel_pct * 0.10 +
                rel_angle_pct * 0.08 +
                move_pct * 0.08, 2)

            # Outcome boost: small, sample-size gated, shrunk to league baseline
            outcome_boost = 0.0
            if n_pairs_used >= WHIFF_MIN_PAIRS and pd.notna(pair_whiff):
                baseline = PAIR_OUTCOMES.get(pair_label, GLOBAL_WHIFF)
                if baseline is not None and not pd.isna(baseline):
                    adj_whiff = (pair_whiff * n_pairs_used + baseline * WHIFF_SHRINK_K) / (n_pairs_used + WHIFF_SHRINK_K)
                    diff = adj_whiff - baseline
                    outcome_boost = float(np.clip(diff / WHIFF_BOOST_SCALE * WHIFF_BOOST_MAX,
                                                  -WHIFF_BOOST_MAX, WHIFF_BOOST_MAX))
                    raw_tunnel = round(raw_tunnel + outcome_boost, 2)

            # V8: use raw composite directly — absolute scale, not population
            # percentile.  The composite is already 0-100 with clear meaning:
            #   ≥65 elite, ≥55 good, ≥45 avg, ≥35 below avg, <35 poor.
            tunnel = round(raw_tunnel, 1)

            # Letter grades on absolute composite
            if tunnel >= 65:
                grade = "A"
            elif tunnel >= 55:
                grade = "B"
            elif tunnel >= 45:
                grade = "C"
            elif tunnel >= 35:
                grade = "D"
            else:
                grade = "F"

            # Contextual diagnosis using pair-type benchmarks
            issues = []
            fixes = []
            vs_median = commit_sep - bm_p50

            if effective_rel_sep > 4:
                if combined_rel_std > 1.5:
                    issues.append(f"release points {rel_sep:.0f}\" apart (+ {combined_rel_std:.1f}\" scatter)")
                    fixes.append("Tighten arm slot consistency — release variance hurts deception")
                else:
                    issues.append(f"release points {rel_sep:.0f}\" apart")
                    fixes.append("Work on consistent arm slot across both pitches")
            if commit_sep > bm_p75:
                issues.append(f"{commit_sep:.0f}\" commit sep ({vs_median:+.1f}\" vs {pair_label} median)")
                if velo_gap > 8:
                    fixes.append(f"Reduce {velo_gap:.0f} mph velo gap — pitches separate too early")
                else:
                    fixes.append("Pitch trajectories diverge too early — hitter can read them")
            elif commit_sep > bm_p50:
                issues.append(f"commit sep slightly above average for {pair_label} ({vs_median:+.1f}\")")
            if plate_sep < 6:
                issues.append(f"only {plate_sep:.0f}\" apart at plate")
                fixes.append("Pitches end up too close together — need more movement contrast")
            if move_div < 5:
                issues.append(f"only {move_div:.0f}\" movement difference")
                fixes.append("Increase break differential — pitches move too similarly")
            if pd.notna(rel_angle_sep) and rel_angle_sep > 3:
                issues.append(f"{rel_angle_sep:.1f}° release angle divergence")
                fixes.append("Release angles differ too much — hitter can distinguish pitch type at release")
            if not issues:
                if tunnel >= 75:
                    diagnosis = f"Elite tunnel — {tunnel:.0f}th percentile for {pair_label}"
                elif tunnel >= 50:
                    diagnosis = f"Above-average tunnel — {tunnel:.0f}th percentile for {pair_label}"
                elif tunnel >= 25:
                    diagnosis = f"Below-average tunnel — {tunnel:.0f}th percentile for {pair_label}"
                else:
                    diagnosis = f"Poor tunnel — bottom {tunnel:.0f}% for {pair_label}"
            else:
                diagnosis = "; ".join(issues)

            rows.append({
                "Pitch A": types[i], "Pitch B": types[j],
                "Grade": grade, "Tunnel Score": tunnel,
                "Release Sep (in)": round(rel_sep, 1),
                "Commit Sep (in)": round(commit_sep, 1),
                "Plate Sep (in)": round(plate_sep, 1),
                "Velo Gap (mph)": round(velo_gap, 1),
                "Move Diff (in)": round(move_div, 1),
                "Rel Angle Sep (°)": round(rel_angle_sep, 1) if pd.notna(rel_angle_sep) else None,
                "Pairs Used": n_pairs_used if n_pairs_used > 0 else "avg",
                "Pair Whiff%": round(pair_whiff, 1) if not pd.isna(pair_whiff) else None,
                "Diagnosis": diagnosis,
                "Fix": "; ".join(fixes) if fixes else "No changes needed",
            })
    return pd.DataFrame(rows).sort_values("Tunnel Score", ascending=False).reset_index(drop=True)
