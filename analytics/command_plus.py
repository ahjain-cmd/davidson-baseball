"""Command+ (Location+) and pitch pair results computation.

Location+: XGBoost trained on per-pitch run values using only location + count
features (no physical pitch characteristics).  Falls back to the original
outcome-weighted location spread method when the model file is absent.

Scale: 100 = average, higher = better location.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

from config import (
    in_zone_mask, SWING_CALLS, filter_minor_pitches,
    MIN_TUNNEL_SEQ_PCT, PLATE_SIDE_MAX, PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX,
    normalize_pitch_types,
)
from data.loader import query_population

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_APP_DIR, "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "location_plus_xgb.joblib")

# ── Pitch-type label encoding (consistent with stuff_plus) ──────────────────
_PITCH_TYPE_LABELS = {
    "Fastball": 0, "Sinker": 1, "Cutter": 2, "Slider": 3,
    "Curveball": 4, "Changeup": 5, "Splitter": 6, "Knuckle Curve": 7,
    "Sweeper": 8,
}

# ── Feature columns for the Location+ XGBoost model ─────────────────────────
_LOC_FEATURES = [
    "PlateLocSide", "PlateLocHeight",
    "Balls", "Strikes",
    "PitcherThrows_enc", "BatterSide_enc",
    "pitch_type_enc",
    "dist_from_center",
    "loc_side_x_batter_side",
    "in_zone",
]


# =============================================================================
#  XGBoost Location+ training
# =============================================================================

def train_location_plus_model(parquet_path: str) -> None:
    """Train Location+ XGBoost model on per-pitch run values.

    Saves model + per-pitch-type scaling stats to ``models/location_plus_xgb.joblib``.
    """
    import duckdb
    import joblib
    from xgboost import XGBRegressor
    from sklearn.model_selection import GroupShuffleSplit

    from analytics.run_value import compute_pitch_run_values
    from config import PARQUET_PATH, ZONE_SIDE, ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP

    pq = parquet_path or PARQUET_PATH
    print(f"  Loading pitches from {pq} ...")

    con = duckdb.connect(":memory:")
    df = con.execute(f"""
        SELECT
            GameID, Pitcher, Batter,
            PitcherThrows, BatterSide,
            Balls, Strikes, PitchCall, PlayResult, KorBB,
            PlateLocSide, PlateLocHeight,
            CASE
                WHEN TaggedPitchType IN ('Undefined','Other','Knuckleball') THEN NULL
                WHEN TaggedPitchType = 'FourSeamFastBall' THEN 'Fastball'
                WHEN TaggedPitchType IN ('OneSeamFastBall','TwoSeamFastBall') THEN 'Sinker'
                WHEN TaggedPitchType = 'ChangeUp' THEN 'Changeup'
                ELSE TaggedPitchType
            END AS TaggedPitchType
        FROM read_parquet('{pq}')
        WHERE PitchCall IS NOT NULL AND PitchCall != 'Undefined'
          AND PlateLocSide IS NOT NULL
          AND PlateLocHeight IS NOT NULL
          AND PlateLocSide BETWEEN -{PLATE_SIDE_MAX} AND {PLATE_SIDE_MAX}
          AND PlateLocHeight BETWEEN {PLATE_HEIGHT_MIN} AND {PLATE_HEIGHT_MAX}
          AND PitcherThrows IS NOT NULL
          AND TaggedPitchType NOT IN ('Undefined','Other','Knuckleball')
    """).fetchdf()
    con.close()
    print(f"  Loaded {len(df):,} pitches")

    # Compute run value targets
    df = compute_pitch_run_values(df)
    df = df.dropna(subset=["PitchRV", "TaggedPitchType"])
    print(f"  After PitchRV: {len(df):,} pitches with valid targets")

    # Build features
    df["PitcherThrows_enc"] = (
        df["PitcherThrows"].astype(str).str.lower().str.startswith("l").astype(int)
    )
    df["BatterSide_enc"] = (
        df["BatterSide"].astype(str).str.lower().str.startswith("l").astype(int)
    )
    df["pitch_type_enc"] = df["TaggedPitchType"].map(_PITCH_TYPE_LABELS).fillna(-1).astype(int)
    df["Balls"] = pd.to_numeric(df["Balls"], errors="coerce").fillna(0)
    df["Strikes"] = pd.to_numeric(df["Strikes"], errors="coerce").fillna(0)

    loc_side = df["PlateLocSide"].astype(float)
    loc_height = df["PlateLocHeight"].astype(float)
    df["dist_from_center"] = np.sqrt(loc_side ** 2 + (loc_height - 2.5) ** 2)
    df["loc_side_x_batter_side"] = loc_side * df["BatterSide_enc"]
    df["in_zone"] = (
        (loc_side.abs() <= ZONE_SIDE) &
        loc_height.between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP)
    ).astype(int)

    # Build feature matrix
    X = df[_LOC_FEATURES].astype(float)
    y = df["PitchRV"].astype(float)

    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid]
    game_ids = df.loc[valid.index[valid], "GameID"]
    pitch_types = df.loc[valid.index[valid], "TaggedPitchType"]
    print(f"  Training on {len(X):,} pitches with complete features")

    # Train/val split by GameID
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=game_ids))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_alpha=0.5,
        reg_lambda=3.0,
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=30,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Validation metrics
    val_pred = model.predict(X_val)
    val_corr = np.corrcoef(y_val, val_pred)[0, 1]
    val_mae = np.mean(np.abs(y_val - val_pred))
    print(f"\n  Validation: r={val_corr:.4f}, MAE={val_mae:.5f}")

    # Compute per-pitch-type scaling stats at PITCHER level
    # (pitch-level stats cause compression when averaged per pitcher)
    all_pred = model.predict(X)
    pred_series = pd.Series(all_pred, index=X.index)
    pitcher_ids = df.loc[valid.index[valid], "Pitcher"]
    pt_stats: Dict[str, Tuple[float, float]] = {}
    for pt in pitch_types.unique():
        pt_mask = pitch_types.values == pt
        if pt_mask.sum() < 50:
            continue
        # Group by pitcher, compute each pitcher's mean predicted RV
        pt_preds = pred_series[pt_mask]
        pt_pitchers = pitcher_ids[pt_mask]
        pitcher_means = pt_preds.groupby(pt_pitchers).mean()
        # Require at least 10 pitches per pitcher for stable mean
        pt_counts = pt_preds.groupby(pt_pitchers).count()
        pitcher_means = pitcher_means[pt_counts >= 10]
        if len(pitcher_means) < 10:
            continue
        pt_stats[pt] = (float(pitcher_means.mean()), float(pitcher_means.std()))
    print(f"  Scaling stats computed for {len(pt_stats)} pitch types (pitcher-level)")

    # Feature importance
    imp = dict(zip(_LOC_FEATURES, model.feature_importances_))
    imp_sorted = sorted(imp.items(), key=lambda x: -x[1])
    print("\n  Feature importance:")
    for feat, score in imp_sorted:
        print(f"    {feat:30s} {score:.4f}")

    # Save
    os.makedirs(_MODEL_DIR, exist_ok=True)
    artifact = {
        "model": model,
        "pt_stats": pt_stats,
        "features": _LOC_FEATURES,
        "pitch_type_labels": _PITCH_TYPE_LABELS,
    }
    joblib.dump(artifact, _MODEL_PATH, compress=3)
    print(f"\n  Model saved to {_MODEL_PATH}")


# =============================================================================
#  XGBoost Location+ prediction
# =============================================================================

def _load_location_model():
    """Load cached Location+ model artifact. Returns None if missing."""
    if not os.path.exists(_MODEL_PATH):
        return None
    import joblib
    try:
        return joblib.load(_MODEL_PATH)
    except Exception:
        return None


def _compute_command_plus_xgb(pdf: pd.DataFrame, artifact: dict, data=None) -> pd.DataFrame:
    """Compute Command+ (Location+) using the XGBoost model.

    Returns a DataFrame with the SAME columns as the spread-based method:
    Pitch, Pitches, Loc Spread (ft), Zone%, Edge%, CSW%, Chase%, Command+, Percentile.
    """
    from config import ZONE_SIDE, ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP

    model = artifact["model"]
    pt_stats = artifact["pt_stats"]

    cmd_rows = []
    for pt in sorted(pdf["TaggedPitchType"].dropna().unique()):
        ptd = pdf[pdf["TaggedPitchType"] == pt].dropna(
            subset=["PlateLocSide", "PlateLocHeight"]
        ).copy()
        ptd = ptd[
            ptd["PlateLocSide"].between(-PLATE_SIDE_MAX, PLATE_SIDE_MAX) &
            ptd["PlateLocHeight"].between(PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX)
        ]
        if len(ptd) < 10:
            continue

        # Build features for this pitch type subset
        ptd["PitcherThrows_enc"] = (
            ptd["PitcherThrows"].astype(str).str.lower().str.startswith("l").astype(int)
        ) if "PitcherThrows" in ptd.columns else 0
        ptd["BatterSide_enc"] = (
            ptd["BatterSide"].astype(str).str.lower().str.startswith("l").astype(int)
        ) if "BatterSide" in ptd.columns else 0
        ptd["pitch_type_enc"] = _PITCH_TYPE_LABELS.get(pt, -1)
        ptd["Balls"] = pd.to_numeric(ptd.get("Balls", 0), errors="coerce").fillna(0)
        ptd["Strikes"] = pd.to_numeric(ptd.get("Strikes", 0), errors="coerce").fillna(0)

        loc_side = ptd["PlateLocSide"].astype(float)
        loc_height = ptd["PlateLocHeight"].astype(float)
        ptd["dist_from_center"] = np.sqrt(loc_side ** 2 + (loc_height - 2.5) ** 2)
        ptd["loc_side_x_batter_side"] = loc_side * ptd["BatterSide_enc"]
        ptd["in_zone"] = (
            (loc_side.abs() <= ZONE_SIDE) &
            loc_height.between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP)
        ).astype(int)

        X = ptd[_LOC_FEATURES].astype(float)
        raw_rv = model.predict(X)

        # Z-score pitcher's mean predicted RV against pitcher-level distribution
        pitcher_mean_rv = float(np.mean(raw_rv))
        if pt in pt_stats:
            pop_mean, pop_std = pt_stats[pt]
            if pop_std > 0:
                z = (pitcher_mean_rv - pop_mean) / pop_std
                loc_plus = 100 + (-z) * 10  # negate: lower RV = higher Command+
                pctl = min(99, max(1, 50 + (-z) * 15))
            else:
                loc_plus = 100.0
                pctl = 50.0
        else:
            loc_plus = 100.0
            pctl = 50.0

        # Standard summary stats (same columns as spread-based)
        in_zone = in_zone_mask(ptd)
        zone_pct = in_zone.mean() * 100
        edge = (
            ((ptd["PlateLocSide"].abs().between(0.5, 1.1)) |
             (ptd["PlateLocHeight"].between(1.2, 1.8)) |
             (ptd["PlateLocHeight"].between(3.2, 3.8))) &
            (ptd["PlateLocSide"].abs() <= 1.5) &
            ptd["PlateLocHeight"].between(0.5, 4.5)
        )
        edge_pct = edge.mean() * 100
        csw = ptd["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
        out_zone = ~in_zone
        chase_swings = ptd[out_zone & ptd["PitchCall"].isin(SWING_CALLS)]
        chase_pct = len(chase_swings) / max(out_zone.sum(), 1) * 100

        # Loc Spread for display (unweighted, same as spread method)
        loc_spread = np.sqrt(
            ptd["PlateLocHeight"].std() ** 2 + ptd["PlateLocSide"].std() ** 2
        )

        cmd_rows.append({
            "Pitch": pt,
            "Pitches": len(ptd),
            "Loc Spread (ft)": round(loc_spread, 2) if np.isfinite(loc_spread) else np.nan,
            "Zone%": round(zone_pct, 1),
            "Edge%": round(edge_pct, 1),
            "CSW%": round(csw, 1),
            "Chase%": round(chase_pct, 1),
            "Command+": round(loc_plus, 0),
            "Percentile": round(pctl, 1),
        })

    if not cmd_rows:
        return pd.DataFrame()
    cmd_df = pd.DataFrame(cmd_rows)
    cmd_df = cmd_df.sort_values("Command+", ascending=False)
    return cmd_df


# =============================================================================
#  Original spread-based Command+ (fallback)
# =============================================================================

# Outcome-weight mapping
_OUTCOME_WEIGHTS = {
    "StrikeCalled": 0.5,
    "StrikeSwinging": 0.5,
    "FoulBall": 0.75,
    "FoulBallNotFieldable": 0.75,
    "FoulBallFieldable": 0.75,
    "BallCalled": 1.2,
    "BallinDirt": 1.2,
    "BallIntentional": 1.0,
    "HitByPitch": 1.5,
    "InPlay": 1.0,
}

_COUNT_CONTEXT = {
    (0, 2): {"ball": 0.5, "strike": 0.9},
    (1, 2): {"ball": 0.6, "strike": 0.9},
    (0, 0): {"ball": 1.1, "strike": 1.0},
    (0, 1): {"ball": 0.9, "strike": 1.0},
    (1, 1): {"ball": 1.0, "strike": 1.0},
    (2, 2): {"ball": 0.9, "strike": 0.9},
    (1, 0): {"ball": 1.2, "strike": 0.85},
    (2, 0): {"ball": 1.4, "strike": 0.75},
    (2, 1): {"ball": 1.2, "strike": 0.85},
    (3, 0): {"ball": 1.5, "strike": 0.7},
    (3, 1): {"ball": 1.4, "strike": 0.75},
    (3, 2): {"ball": 1.3, "strike": 0.8},
}

BAYESIAN_N = 150

_BALL_CALLS = {"BallCalled", "BallinDirt", "BallIntentional", "HitByPitch"}
_STRIKE_CALLS = {
    "StrikeCalled", "StrikeSwinging",
    "FoulBall", "FoulBallNotFieldable", "FoulBallFieldable",
    "InPlay",
}


def _outcome_weight_series(ptd):
    """Per-pitch outcome weights combining base outcome + count context + EV."""
    w = ptd["PitchCall"].map(_OUTCOME_WEIGHTS).fillna(1.0).copy()

    if "ExitSpeed" in ptd.columns:
        inplay = ptd["PitchCall"] == "InPlay"
        ev = pd.to_numeric(ptd["ExitSpeed"], errors="coerce")
        w.loc[inplay & (ev >= 98)]              = 2.0
        w.loc[inplay & (ev >= 95) & (ev < 98)]  = 1.5
        w.loc[inplay & (ev < 85)]               = 0.8

    if "Balls" in ptd.columns and "Strikes" in ptd.columns:
        balls = pd.to_numeric(ptd["Balls"], errors="coerce")
        strikes = pd.to_numeric(ptd["Strikes"], errors="coerce")
        for (b, s), mults in _COUNT_CONTEXT.items():
            count_mask = (balls == b) & (strikes == s)
            if not count_mask.any():
                continue
            ball_mask = count_mask & ptd["PitchCall"].isin(_BALL_CALLS)
            strike_mask = count_mask & ptd["PitchCall"].isin(_STRIKE_CALLS)
            if ball_mask.any():
                w.loc[ball_mask] *= mults["ball"]
            if strike_mask.any():
                w.loc[strike_mask] *= mults["strike"]

    return w


def _weighted_spread(ptd, weights):
    """Outcome-weighted location spread: sqrt( sum(w·d²) / sum(w) )."""
    h = ptd["PlateLocHeight"].values.astype(float)
    s = ptd["PlateLocSide"].values.astype(float)
    w = weights.values.astype(float)
    wsum = w.sum()
    if wsum == 0:
        return np.nan
    mean_h = np.average(h, weights=w)
    mean_s = np.average(s, weights=w)
    dev_sq = (h - mean_h) ** 2 + (s - mean_s) ** 2
    return np.sqrt(np.average(dev_sq, weights=w))


def _build_fallback_spreads(data, pitch_types):
    """Compute per-pitcher loc spreads from in-memory Davidson data as fallback."""
    if data is None or data.empty:
        return {}
    req = {"TaggedPitchType", "Pitcher", "PlateLocSide", "PlateLocHeight"}
    if not req.issubset(data.columns):
        return {}
    loc = data.dropna(subset=["PlateLocSide", "PlateLocHeight", "TaggedPitchType"]).copy()
    loc = loc[
        loc["PlateLocSide"].between(-PLATE_SIDE_MAX, PLATE_SIDE_MAX)
        & loc["PlateLocHeight"].between(PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX)
        & loc["TaggedPitchType"].isin(pitch_types)
    ]
    if loc.empty:
        return {}
    result = {}
    for pt, grp in loc.groupby("TaggedPitchType"):
        pitcher_spreads = []
        for _, pg in grp.groupby("Pitcher"):
            if len(pg) < 10:
                continue
            s = np.sqrt(pg["PlateLocHeight"].std()**2 + pg["PlateLocSide"].std()**2)
            if np.isfinite(s):
                pitcher_spreads.append(s)
        if len(pitcher_spreads) >= 3:
            result[pt] = np.array(pitcher_spreads)
    return result


def _compute_command_plus_spread(pdf, data=None):
    """Original spread-based Command+ — used as fallback when model is absent."""
    cmd_rows = []
    for pt in sorted(pdf["TaggedPitchType"].dropna().unique()):
        ptd = pdf[pdf["TaggedPitchType"] == pt].dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
        ptd = ptd[
            ptd["PlateLocSide"].between(-PLATE_SIDE_MAX, PLATE_SIDE_MAX)
            & ptd["PlateLocHeight"].between(PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX)
        ]
        if len(ptd) < 10:
            continue

        weights = _outcome_weight_series(ptd)
        loc_spread = _weighted_spread(ptd, weights)

        in_zone = in_zone_mask(ptd)
        zone_pct = in_zone.mean() * 100
        edge = (
            ((ptd["PlateLocSide"].abs().between(0.5, 1.1)) |
             (ptd["PlateLocHeight"].between(1.2, 1.8)) |
             (ptd["PlateLocHeight"].between(3.2, 3.8))) &
            (ptd["PlateLocSide"].abs() <= 1.5) &
            ptd["PlateLocHeight"].between(0.5, 4.5)
        )
        edge_pct = edge.mean() * 100
        csw = ptd["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).mean() * 100
        out_zone = ~in_zone
        chase_swings = ptd[out_zone & ptd["PitchCall"].isin(SWING_CALLS)]
        chase_pct = len(chase_swings) / max(out_zone.sum(), 1) * 100
        cmd_rows.append({
            "Pitch": pt,
            "Pitches": len(ptd),
            "Loc Spread (ft)": round(loc_spread, 2) if np.isfinite(loc_spread) else np.nan,
            "Zone%": round(zone_pct, 1),
            "Edge%": round(edge_pct, 1),
            "CSW%": round(csw, 1),
            "Chase%": round(chase_pct, 1),
        })
    if not cmd_rows:
        return pd.DataFrame()
    cmd_df = pd.DataFrame(cmd_rows)

    pitch_types = sorted(cmd_df["Pitch"].unique())
    if pitch_types:
        pt_sql = ", ".join(f"'{p.replace(chr(39), chr(39)+chr(39))}'" for p in pitch_types)
        sql = f"""
            WITH t AS (
                SELECT Pitcher,
                       PlateLocSide,
                       PlateLocHeight,
                       CASE TaggedPitchType
                           WHEN 'FourSeamFastBall' THEN 'Fastball'
                           WHEN 'OneSeamFastBall' THEN 'Sinker'
                           WHEN 'TwoSeamFastBall' THEN 'Sinker'
                           WHEN 'ChangeUp' THEN 'Changeup'
                           ELSE TaggedPitchType
                       END AS pt_norm
                FROM trackman
                WHERE TaggedPitchType NOT IN ('Other','Undefined','Knuckleball')
                  AND PlateLocSide BETWEEN -{PLATE_SIDE_MAX} AND {PLATE_SIDE_MAX}
                  AND PlateLocHeight BETWEEN {PLATE_HEIGHT_MIN} AND {PLATE_HEIGHT_MAX}
            )
            SELECT pt_norm AS PitchType,
                   Pitcher,
                   STDDEV(PlateLocHeight) AS std_h,
                   STDDEV(PlateLocSide) AS std_s,
                   COUNT(*) AS n
            FROM t
            WHERE pt_norm IN ({pt_sql})
            GROUP BY pt_norm, Pitcher
            HAVING COUNT(*) >= 10
        """
        baseline_df = query_population(sql)
    else:
        baseline_df = pd.DataFrame()

    fallback_spreads = {}
    if baseline_df.empty and data is not None:
        fallback_spreads = _build_fallback_spreads(data, pitch_types)

    cmd_scores = []
    cmd_pctls = []
    for _, row in cmd_df.iterrows():
        pt = row["Pitch"]
        n_pitches = row["Pitches"]
        my_spread = row["Loc Spread (ft)"]
        if pd.isna(my_spread):
            cmd_scores.append(100.0)
            cmd_pctls.append(50.0)
            continue

        spreads = None
        if not baseline_df.empty:
            pt_df = baseline_df[baseline_df["PitchType"] == pt].copy()
            if not pt_df.empty:
                spreads = np.sqrt(pt_df["std_h"].astype(float)**2 + pt_df["std_s"].astype(float)**2)
                spreads = spreads.replace([np.inf, -np.inf], np.nan).dropna()
                if len(spreads) < 3:
                    spreads = None

        if spreads is None and pt in fallback_spreads:
            spreads = fallback_spreads[pt]

        if spreads is None:
            cmd_scores.append(100.0)
            cmd_pctls.append(50.0)
            continue

        raw_pctl = 100 - percentileofscore(spreads, my_spread, kind="rank")

        alpha = n_pitches / (n_pitches + BAYESIAN_N)
        stab_pctl = alpha * raw_pctl + (1 - alpha) * 50.0
        stab_cmd = 100 + (stab_pctl - 50) * 0.4

        cmd_scores.append(round(stab_cmd, 0))
        cmd_pctls.append(round(stab_pctl, 1))

    cmd_df["Command+"] = cmd_scores
    cmd_df["Percentile"] = cmd_pctls
    cmd_df = cmd_df.sort_values("Command+", ascending=False)
    return cmd_df


# =============================================================================
#  Main entry point (same interface as before)
# =============================================================================

def _compute_command_plus(pdf, data=None):
    """Compute Command+ for each pitch type. Returns DataFrame with
    Pitch, Pitches, Loc Spread (ft), Zone%, Edge%, CSW%, Chase%, Command+, Percentile.

    Uses XGBoost Location+ model if available, otherwise falls back to
    spread-based method.
    """
    if pdf.empty:
        return pd.DataFrame()

    artifact = _load_location_model()
    if artifact is not None:
        return _compute_command_plus_xgb(pdf, artifact, data=data)
    else:
        return _compute_command_plus_spread(pdf, data=data)


# =============================================================================
#  Pitch pair results (unchanged)
# =============================================================================

def _compute_pitch_pair_results(pdf, data, tunnel_df=None):
    """Compute effectiveness when pitch B follows pitch A in an at-bat."""
    if pdf.empty:
        return pd.DataFrame()
    pdf = filter_minor_pitches(pdf, min_pct=MIN_TUNNEL_SEQ_PCT)
    if pdf.empty:
        return pd.DataFrame()

    seq_model_df = pd.DataFrame()
    normalize_seq_pitch_types = None
    try:
        from analytics.sequence_whiff import (
            _compute_sequence_whiff_table,
            _normalize_sequence_pitch_types,
        )

        seq_model_df = _compute_sequence_whiff_table(pdf)
        normalize_seq_pitch_types = _normalize_sequence_pitch_types
    except Exception:
        seq_model_df = pd.DataFrame()

    sort_cols = [c for c in ["GameID", "Pitcher", "Batter", "Inning", "PAofInning", "PitchNo"] if c in pdf.columns]
    if len(sort_cols) < 2:
        return pd.DataFrame()

    pdf_s = pdf.sort_values(sort_cols).copy()
    pa_group_cols = [c for c in ["GameID", "Pitcher", "Batter", "Inning", "PAofInning"] if c in pdf_s.columns]
    if len(pa_group_cols) < 2:
        return pd.DataFrame()
    pdf_s["PrevPitch"] = pdf_s.groupby(pa_group_cols, dropna=False)["TaggedPitchType"].shift(1)
    pdf_s = pdf_s.dropna(subset=["PrevPitch"])

    is_whiff = pdf_s["PitchCall"] == "StrikeSwinging"
    is_swing = pdf_s["PitchCall"].isin(SWING_CALLS)
    is_csw = pdf_s["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])

    _TB_MAP = {"Single": 1, "Double": 2, "Triple": 3, "HomeRun": 4}

    rows = []
    for (prev, curr), grp in pdf_s.groupby(["PrevPitch", "TaggedPitchType"]):
        n = len(grp)
        if n < 25:
            continue
        swings = grp[is_swing.reindex(grp.index, fill_value=False)]
        whiffs = grp[is_whiff.reindex(grp.index, fill_value=False)]
        csws = grp[is_csw.reindex(grp.index, fill_value=False)]
        batted = grp[(grp["PitchCall"] == "InPlay") & grp["ExitSpeed"].notna()]
        inplay = grp[grp["PitchCall"] == "InPlay"]
        if len(inplay) > 0 and "PlayResult" in grp.columns:
            total_bases = inplay["PlayResult"].map(_TB_MAP).fillna(0).sum()
            slg = round(total_bases / len(inplay), 3)
        else:
            slg = np.nan
        if "Strikes" in grp.columns:
            two_strike = grp[grp["Strikes"] == 2]
            putaway = two_strike[two_strike["PitchCall"].isin(["StrikeSwinging", "StrikeCalled"])]
            putaway_pct = len(putaway) / max(len(two_strike), 1) * 100 if len(two_strike) > 0 else np.nan
        else:
            putaway_pct = np.nan
        if "KorBB" in grp.columns:
            k_events = grp["KorBB"].isin(["Strikeout", "K"])
            k_pct = k_events.mean() * 100 if len(grp) > 0 else np.nan
        elif "PlayResult" in grp.columns:
            k_events = grp["PlayResult"].isin(["Strikeout", "K"])
            k_pct = k_events.mean() * 100 if len(grp) > 0 else np.nan
        else:
            k_pct = np.nan
        tun_grade, tun_score = "-", np.nan
        if isinstance(tunnel_df, pd.DataFrame) and not tunnel_df.empty:
            tun_match = tunnel_df[
                ((tunnel_df["Pitch A"] == prev) & (tunnel_df["Pitch B"] == curr)) |
                ((tunnel_df["Pitch A"] == curr) & (tunnel_df["Pitch B"] == prev))
            ]
            if not tun_match.empty:
                tun_grade = tun_match.iloc[0]["Grade"]
                tun_score = tun_match.iloc[0]["Tunnel Score"]
        rows.append({
            "Setup Pitch": prev, "Follow Pitch": curr, "Count": n,
            "Whiff%": round(len(whiffs) / max(len(swings), 1) * 100, 1),
            "CSW%": round(len(csws) / n * 100, 1),
            "Avg EV": round(batted["ExitSpeed"].mean(), 1) if len(batted) > 0 else np.nan,
            "SLG": slg,
            "Putaway%": round(putaway_pct, 1) if not pd.isna(putaway_pct) else np.nan,
            "K%": round(k_pct, 1) if not pd.isna(k_pct) else np.nan,
            "Chase%": round(
                (lambda _iz=in_zone_mask(grp): len(grp[(~_iz) & grp["PitchCall"].isin(SWING_CALLS)]) /
                max(len(grp[~_iz]), 1) * 100)(), 1),
            "Tunnel": tun_grade,
            "Tunnel Score": tun_score,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if not seq_model_df.empty:
        seq_merge = seq_model_df.rename(
            columns={
                "Setup Pitch": "_SetupMergePitch",
                "Follow Pitch": "_FollowMergePitch",
            }
        )
        if callable(normalize_seq_pitch_types):
            merge_keys = out[["Setup Pitch", "Follow Pitch"]].copy()
            for col in ["Setup Pitch", "Follow Pitch"]:
                norm_col = normalize_seq_pitch_types(
                    merge_keys[[col]].rename(columns={col: "TaggedPitchType"})
                )
                merge_keys[col] = norm_col["TaggedPitchType"]
            out["_SetupMergePitch"] = merge_keys["Setup Pitch"]
            out["_FollowMergePitch"] = merge_keys["Follow Pitch"]
        else:
            out["_SetupMergePitch"] = out["Setup Pitch"]
            out["_FollowMergePitch"] = out["Follow Pitch"]
        out = (
            out.merge(seq_merge, on=["_SetupMergePitch", "_FollowMergePitch"], how="left")
            .drop(columns=["_SetupMergePitch", "_FollowMergePitch"])
        )
    sort_col = "SeqWhiff%" if "SeqWhiff%" in out.columns else "Whiff%"
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)
