"""Stuff+ computation — XGBoost run-value model with z-score fallback.

FanGraphs-style Stuff+: XGBoost trained on per-pitch run values (PitchRV)
using only physical pitch characteristics.  Falls back to the original
hand-tuned z-score composite when the model file is absent.

Scale: 100 = average, each 10 = 1 stdev better (higher = better stuff).
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np

from data.population import compute_stuff_baselines
from config import normalize_pitch_types

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_APP_DIR, "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "stuff_plus_xgb.joblib")
_XWHIFF_MODEL_PATH = os.path.join(_MODEL_DIR, "xwhiff_models.joblib")

# ── Pitch-type label encoding (consistent across train/predict) ──────────────
_PITCH_TYPE_LABELS = {
    "Fastball": 0, "Sinker": 1, "Cutter": 2, "Slider": 3,
    "Curveball": 4, "Changeup": 5, "Splitter": 6, "Knuckle Curve": 7,
    "Sweeper": 8,
}

# ── Feature columns for the XGBoost model ────────────────────────────────────
_STUFF_FEATURES = [
    "RelSpeed", "InducedVertBreak", "HorzBreakAdj", "Extension",
    "VertApprAngle", "SpinRate", "SpinAxis", "VeloDiff",
    "PitcherThrows_enc", "BatterSide_enc", "pitch_type_enc",
    "speed_x_ivb",
]

# xWhiff features: per-pitch-type models don't need pitch_type_enc, add HorzApprAngle
_XWHIFF_FEATURES = [
    "RelSpeed", "InducedVertBreak", "HorzBreakAdj", "Extension",
    "VertApprAngle", "SpinRate", "SpinAxis", "VeloDiff",
    "PitcherThrows_enc", "BatterSide_enc", "speed_x_ivb",
    "HorzApprAngle",
]


# =============================================================================
#  XGBoost Stuff+ training
# =============================================================================

def train_stuff_plus_model(parquet_path: str) -> None:
    """Train Stuff+ XGBoost model on per-pitch run values.

    Saves model + per-pitch-type scaling stats to ``models/stuff_plus_xgb.joblib``.
    """
    import duckdb
    import joblib
    from xgboost import XGBRegressor
    from sklearn.model_selection import GroupShuffleSplit

    from analytics.run_value import compute_pitch_run_values
    from config import PARQUET_PATH

    pq = parquet_path or PARQUET_PATH
    print(f"  Loading pitches from {pq} ...")

    con = duckdb.connect(":memory:")
    df = con.execute(f"""
        SELECT
            GameID, Pitcher, Batter,
            PitcherThrows, BatterSide,
            Balls, Strikes, PitchCall, PlayResult, KorBB,
            RelSpeed, InducedVertBreak, HorzBreak,
            Extension, VertApprAngle, SpinRate, SpinAxis,
            CASE
                WHEN TaggedPitchType IN ('Undefined','Other','Knuckleball') THEN NULL
                WHEN TaggedPitchType = 'FourSeamFastBall' THEN 'Fastball'
                WHEN TaggedPitchType IN ('OneSeamFastBall','TwoSeamFastBall') THEN 'Sinker'
                WHEN TaggedPitchType = 'ChangeUp' THEN 'Changeup'
                ELSE TaggedPitchType
            END AS TaggedPitchType,
            CASE
                WHEN PitcherThrows IN ('Left','L') THEN -HorzBreak
                ELSE HorzBreak
            END AS HorzBreakAdj
        FROM read_parquet('{pq}')
        WHERE PitchCall IS NOT NULL AND PitchCall != 'Undefined'
          AND RelSpeed IS NOT NULL
          AND InducedVertBreak IS NOT NULL
          AND HorzBreak IS NOT NULL
          AND PitcherThrows IS NOT NULL
          AND TaggedPitchType NOT IN ('Undefined','Other','Knuckleball')
    """).fetchdf()
    con.close()
    print(f"  Loaded {len(df):,} pitches")

    # Compute run value targets
    df = compute_pitch_run_values(df)
    df = df.dropna(subset=["PitchRV", "TaggedPitchType"])
    print(f"  After PitchRV: {len(df):,} pitches with valid targets")

    # Compute velo diff (fastball velo - pitch velo per pitcher)
    fb_types = {"Fastball", "Sinker", "Cutter"}
    fb_velo = (
        df[df["TaggedPitchType"].isin(fb_types)]
        .groupby("Pitcher")["RelSpeed"]
        .mean()
    )
    df["VeloDiff"] = df["Pitcher"].map(fb_velo).astype(float) - df["RelSpeed"].astype(float)

    # Encode categoricals
    df["PitcherThrows_enc"] = (
        df["PitcherThrows"].astype(str).str.lower().str.startswith("l").astype(int)
    )
    df["BatterSide_enc"] = (
        df["BatterSide"].astype(str).str.lower().str.startswith("l").astype(int)
    )
    df["pitch_type_enc"] = df["TaggedPitchType"].map(_PITCH_TYPE_LABELS).fillna(-1).astype(int)

    # Interaction
    df["speed_x_ivb"] = df["RelSpeed"].astype(float) * df["InducedVertBreak"].astype(float)

    # Build feature matrix
    for col in _STUFF_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    X = df[_STUFF_FEATURES].astype(float)
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

    # Compute per-pitch-type scaling stats from training predictions
    all_pred = model.predict(X)
    pt_stats: Dict[str, Tuple[float, float]] = {}
    for pt in pitch_types.unique():
        mask = pitch_types.values == pt
        if mask.sum() < 50:
            continue
        preds = all_pred[mask]
        pt_stats[pt] = (float(np.mean(preds)), float(np.std(preds)))
    print(f"  Scaling stats computed for {len(pt_stats)} pitch types")

    # Feature importance
    imp = dict(zip(_STUFF_FEATURES, model.feature_importances_))
    imp_sorted = sorted(imp.items(), key=lambda x: -x[1])
    print("\n  Feature importance:")
    for feat, score in imp_sorted[:8]:
        print(f"    {feat:25s} {score:.4f}")

    # Save
    os.makedirs(_MODEL_DIR, exist_ok=True)
    artifact = {
        "model": model,
        "pt_stats": pt_stats,
        "features": _STUFF_FEATURES,
        "pitch_type_labels": _PITCH_TYPE_LABELS,
    }
    joblib.dump(artifact, _MODEL_PATH, compress=3)
    print(f"\n  Model saved to {_MODEL_PATH}")


# =============================================================================
#  XGBoost Stuff+ prediction
# =============================================================================

def _load_stuff_model():
    """Load cached Stuff+ model artifact. Returns None if missing."""
    if not os.path.exists(_MODEL_PATH):
        return None
    import joblib
    try:
        return joblib.load(_MODEL_PATH)
    except Exception:
        return None


def _compute_stuff_plus_xgb(data: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Compute Stuff+ using the XGBoost model."""
    model = artifact["model"]
    pt_stats = artifact["pt_stats"]

    df = data.copy()
    df = normalize_pitch_types(df)
    scored = df.dropna(subset=["RelSpeed", "TaggedPitchType"]).copy()
    if scored.empty:
        df["StuffPlus"] = np.nan
        return df

    # HorzBreakAdj (handedness-normalized)
    if "HorzBreak" in scored.columns:
        throws = scored.get("PitcherThrows")
        if throws is not None:
            is_l = throws.astype(str).str.lower().str.startswith("l")
            scored["HorzBreakAdj"] = np.where(
                is_l, -scored["HorzBreak"].astype(float), scored["HorzBreak"].astype(float)
            )
        else:
            scored["HorzBreakAdj"] = scored["HorzBreak"].astype(float)
    else:
        scored["HorzBreakAdj"] = np.nan

    # VeloDiff
    fb_types = {"Fastball", "Sinker", "Cutter"}
    fb_velo = (
        scored[scored["TaggedPitchType"].isin(fb_types)]
        .groupby("Pitcher")["RelSpeed"]
        .mean()
    )
    scored["VeloDiff"] = (
        scored["Pitcher"].map(fb_velo).astype(float) - scored["RelSpeed"].astype(float)
    )

    # Encode
    scored["PitcherThrows_enc"] = (
        scored["PitcherThrows"].astype(str).str.lower().str.startswith("l").astype(int)
    )
    scored["BatterSide_enc"] = (
        scored["BatterSide"].astype(str).str.lower().str.startswith("l").astype(int)
    )
    scored["pitch_type_enc"] = (
        scored["TaggedPitchType"].map(_PITCH_TYPE_LABELS).fillna(-1).astype(int)
    )
    scored["speed_x_ivb"] = (
        scored["RelSpeed"].astype(float) * scored["InducedVertBreak"].astype(float)
    )

    # Build feature matrix
    for col in _STUFF_FEATURES:
        if col not in scored.columns:
            scored[col] = np.nan
    X = scored[_STUFF_FEATURES].astype(float)

    # Predict raw RV
    raw_rv = model.predict(X)

    # Z-score within pitch type, then negate (more negative RV = better stuff = higher score)
    stuff_scores = pd.Series(np.nan, index=scored.index)
    for pt in scored["TaggedPitchType"].unique():
        if pt not in pt_stats:
            continue
        mean, std = pt_stats[pt]
        if std == 0 or np.isnan(std):
            continue
        mask = scored["TaggedPitchType"] == pt
        z = (raw_rv[mask.values] - mean) / std
        stuff_scores[mask] = 100 + (-z) * 10  # negate: lower RV = higher Stuff+

    df["StuffPlus"] = stuff_scores.reindex(df.index)
    return df


# =============================================================================
#  Original z-score Stuff+ (fallback)
# =============================================================================

def _compute_stuff_plus_zscore(data, baseline=None, baselines_dict=None):
    """Original z-score composite Stuff+ — used as fallback when model is absent."""
    if data is None or len(data) == 0:
        return data
    if "StuffPlus" in data.columns:
        return data

    base_df = normalize_pitch_types(data.copy())
    df = base_df.dropna(subset=["RelSpeed", "TaggedPitchType"]).copy()
    if df.empty:
        base_df["StuffPlus"] = np.nan
        return base_df

    if baselines_dict is None:
        baselines_dict = compute_stuff_baselines()

    baseline_stats = baselines_dict["baseline_stats"]
    fb_velo_map = baselines_dict["fb_velo_by_pitcher"]
    velo_diff_stats = baselines_dict["velo_diff_stats"]
    fb_velo = pd.Series(fb_velo_map)

    weights = {
        "Fastball":       {"RelSpeed": 2.0, "InducedVertBreak": 2.5, "HorzBreak": 0.3, "Extension": 0.5, "VertApprAngle": 2.5, "SpinRate": 1.0},
        "Sinker":         {"RelSpeed": 2.5, "InducedVertBreak": -0.5, "HorzBreak": 1.5, "Extension": 0.5, "VertApprAngle": -1.5, "SpinRate": 0.8},
        "Cutter":         {"RelSpeed": 0.8, "InducedVertBreak": 0.3, "HorzBreak": -1.5, "Extension": -1.0, "VertApprAngle": -0.5, "SpinRate": 2.0},
        "Slider":         {"RelSpeed": 1.0, "InducedVertBreak": -0.5, "HorzBreak": 1.0, "Extension": 0.3, "VertApprAngle": -2.5, "SpinRate": 1.5},
        "Curveball":      {"RelSpeed": 1.5, "InducedVertBreak": -1.5, "HorzBreak": -1.5, "Extension": -1.5, "VertApprAngle": -2.0, "SpinRate": 0.5},
        "Changeup":       {"RelSpeed": 0.5, "InducedVertBreak": 1.5, "HorzBreak": 1.0, "Extension": 0.5, "VertApprAngle": -2.5, "SpinRate": 1.0, "VeloDiff": 2.0},
        "Splitter":       {"RelSpeed": 1.0, "InducedVertBreak": -2.0, "HorzBreak": 0.5, "Extension": 1.0, "VertApprAngle": -2.0, "SpinRate": -0.3, "VeloDiff": 1.5},
        "Knuckle Curve":  {"RelSpeed": 1.5, "InducedVertBreak": -1.5, "HorzBreak": -1.5, "Extension": -1.5, "VertApprAngle": -2.0, "SpinRate": 0.5},
    }
    default_w = {"RelSpeed": 1.0, "InducedVertBreak": 1.0, "HorzBreak": 1.0, "Extension": 1.0, "VertApprAngle": 1.0, "SpinRate": 1.0}

    stuff_scores = []
    if "HorzBreak" in df.columns:
        throws = df.get("PitcherThrows")
        if throws is not None:
            is_l = throws.astype(str).str.lower().str.startswith("l")
            df["_HB_ADJ"] = np.where(is_l, -df["HorzBreak"].astype(float), df["HorzBreak"].astype(float))
        else:
            df["_HB_ADJ"] = df["HorzBreak"].astype(float)

    for pt, grp in df.groupby("TaggedPitchType"):
        w = weights.get(pt, default_w)
        bstats = baseline_stats.get(pt, {})
        z_total = pd.Series(0.0, index=grp.index)
        w_total = pd.Series(0.0, index=grp.index)
        for col, weight in w.items():
            if col == "VeloDiff":
                if pt not in velo_diff_stats or "Pitcher" not in grp.columns:
                    continue
                grp_fb = grp["Pitcher"].map(fb_velo)
                vd = grp_fb - grp["RelSpeed"].astype(float)
                mu, sigma = velo_diff_stats[pt]
                if sigma == 0 or pd.isna(sigma):
                    continue
                z = (vd - mu) / sigma
                valid = z.notna()
                z_total += z.fillna(0) * weight
                w_total += valid.astype(float) * abs(weight)
                continue
            if col == "HorzBreak":
                bkey = "HorzBreakAdj" if "HorzBreakAdj" in bstats else "HorzBreak"
                if bkey not in bstats or "_HB_ADJ" not in grp.columns:
                    continue
                mu, sigma = bstats[bkey]
                vals = grp["_HB_ADJ"].astype(float)
            else:
                if col not in grp.columns or col not in bstats:
                    continue
                mu, sigma = bstats[col]
                vals = grp[col].astype(float)
            if sigma == 0 or pd.isna(sigma) or pd.isna(mu):
                continue
            z = (vals - mu) / sigma
            valid = z.notna()
            z_total += z.fillna(0) * weight
            w_total += valid.astype(float) * abs(weight)
        z_total = z_total / w_total.replace(0, np.nan)
        grp = grp.copy()
        grp["StuffPlus"] = 100 + z_total * 10
        stuff_scores.append(grp)

    if not stuff_scores:
        base_df["StuffPlus"] = np.nan
        return base_df

    scored = pd.concat(stuff_scores, ignore_index=False)
    base_df["StuffPlus"] = scored["StuffPlus"].reindex(base_df.index)
    return base_df


# =============================================================================
#  Main entry point (same interface as before)
# =============================================================================

def _compute_stuff_plus(data, baseline=None, baselines_dict=None):
    """Compute Stuff+ for every pitch in data.

    Uses XGBoost model if available, otherwise falls back to z-score composite.

    Args:
        data: DataFrame of pitches to score
        baseline: DEPRECATED — ignored when baselines_dict is provided.
        baselines_dict: Pre-computed dict from compute_stuff_baselines().
                        Only needed for z-score fallback.
    """
    if data is None or len(data) == 0:
        return data

    artifact = _load_stuff_model()
    if artifact is not None:
        # Always recompute with XGBoost (drop stale precomputed column)
        if "StuffPlus" in data.columns:
            data = data.drop(columns=["StuffPlus"])
        return _compute_stuff_plus_xgb(data, artifact)
    else:
        # Z-score fallback — skip if already computed
        if "StuffPlus" in data.columns:
            return data
        return _compute_stuff_plus_zscore(data, baseline=baseline, baselines_dict=baselines_dict)


@st.cache_data(show_spinner="Computing Stuff+ grades...")
def _compute_stuff_plus_all(data):
    """Cached wrapper for _compute_stuff_plus on the full Davidson dataset."""
    return _compute_stuff_plus(data)


# =============================================================================
#  xWhiff Stuff+ — per-pitch-type XGBClassifier on whiff target
# =============================================================================

_SWING_CALLS = [
    "StrikeSwinging", "FoulBall", "FoulBallNotFieldable",
    "FoulBallFieldable", "InPlay",
]

_XWHIFF_XGB_PARAMS = dict(
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

_MIN_PITCH_TYPE_TRAIN = 500


def train_xwhiff_model(parquet_path: str) -> None:
    """Train per-pitch-type xWhiff XGBClassifier on whiff target.

    Saves models + per-pitch-type pitcher-level population stats to
    ``models/xwhiff_models.joblib``.
    """
    import duckdb
    import joblib
    from xgboost import XGBClassifier
    from sklearn.model_selection import GroupShuffleSplit

    from config import PARQUET_PATH

    pq = parquet_path or PARQUET_PATH
    print(f"  Loading pitches from {pq} ...")

    con = duckdb.connect(":memory:")
    df = con.execute(f"""
        SELECT
            GameID, Pitcher, Batter,
            PitcherThrows, BatterSide,
            Balls, Strikes, PitchCall, PlayResult, KorBB,
            RelSpeed, InducedVertBreak, HorzBreak,
            Extension, VertApprAngle, SpinRate, SpinAxis,
            HorzApprAngle,
            CASE
                WHEN TaggedPitchType IN ('Undefined','Other','Knuckleball') THEN NULL
                WHEN TaggedPitchType = 'FourSeamFastBall' THEN 'Fastball'
                WHEN TaggedPitchType IN ('OneSeamFastBall','TwoSeamFastBall') THEN 'Sinker'
                WHEN TaggedPitchType = 'ChangeUp' THEN 'Changeup'
                ELSE TaggedPitchType
            END AS TaggedPitchType,
            CASE
                WHEN PitcherThrows IN ('Left','L') THEN -HorzBreak
                ELSE HorzBreak
            END AS HorzBreakAdj
        FROM read_parquet('{pq}')
        WHERE PitchCall IS NOT NULL AND PitchCall != 'Undefined'
          AND RelSpeed IS NOT NULL
          AND InducedVertBreak IS NOT NULL
          AND HorzBreak IS NOT NULL
          AND PitcherThrows IS NOT NULL
          AND TaggedPitchType NOT IN ('Undefined','Other','Knuckleball')
    """).fetchdf()
    con.close()
    print(f"  Loaded {len(df):,} pitches")

    df = df.dropna(subset=["TaggedPitchType"])

    # Swing/whiff flags
    df["is_swing"] = df["PitchCall"].isin(_SWING_CALLS).astype(int)
    df["is_whiff"] = (df["PitchCall"] == "StrikeSwinging").astype(int)

    # VeloDiff
    fb_types = {"Fastball", "Sinker", "Cutter"}
    fb_velo = (
        df[df["TaggedPitchType"].isin(fb_types)]
        .groupby("Pitcher")["RelSpeed"]
        .mean()
    )
    df["VeloDiff"] = df["Pitcher"].map(fb_velo).astype(float) - df["RelSpeed"].astype(float)

    # Encode categoricals
    df["PitcherThrows_enc"] = (
        df["PitcherThrows"].astype(str).str.lower().str.startswith("l").astype(int)
    )
    df["BatterSide_enc"] = (
        df["BatterSide"].astype(str).str.lower().str.startswith("l").astype(int)
    )

    # Interaction
    df["speed_x_ivb"] = df["RelSpeed"].astype(float) * df["InducedVertBreak"].astype(float)

    # Ensure all feature columns exist
    for col in _XWHIFF_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    models: Dict[str, object] = {}
    pt_stats: Dict[str, Tuple[float, float]] = {}

    for pt in sorted(df["TaggedPitchType"].dropna().unique()):
        pt_all = df[df["TaggedPitchType"] == pt].copy()
        # Train on swings only
        pt_swings = pt_all[pt_all["is_swing"] == 1].copy()
        if len(pt_swings) < _MIN_PITCH_TYPE_TRAIN:
            print(f"  Skipping {pt}: {len(pt_swings)} swings < {_MIN_PITCH_TYPE_TRAIN}")
            continue

        X = pt_swings[_XWHIFF_FEATURES].astype(float)
        y = pt_swings["is_whiff"].astype(int)
        game_ids = pt_swings["GameID"]

        valid = X.notna().all(axis=1)
        X, y = X[valid], y[valid]
        game_ids = game_ids[valid]

        if len(X) < _MIN_PITCH_TYPE_TRAIN:
            continue

        # Train/val split by GameID
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=game_ids))
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        params = dict(_XWHIFF_XGB_PARAMS)
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=0,
        )
        models[pt] = model

        # Compute pitcher-level population stats: predict xWhiff on ALL pitches
        # (not just swings) for this pitch type
        X_all = pt_all[_XWHIFF_FEATURES].astype(float)
        valid_all = X_all.notna().all(axis=1)
        xwhiff_pred = np.full(len(pt_all), np.nan)
        if valid_all.any():
            xwhiff_pred[valid_all.values] = model.predict_proba(X_all[valid_all])[:, 1]

        pt_all["_xwhiff"] = xwhiff_pred
        pitcher_means = pt_all.groupby("Pitcher")["_xwhiff"].mean().dropna()
        if len(pitcher_means) >= 10:
            pt_stats[pt] = (float(pitcher_means.mean()), float(pitcher_means.std()))

        whiff_rate = y.mean() * 100
        val_pred = model.predict_proba(X_val)[:, 1]
        val_auc = float(np.nan)
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(y_val, val_pred)
        except Exception:
            pass
        print(f"  {pt:20s}  swings={len(X):,}  whiff%={whiff_rate:.1f}  "
              f"val_AUC={val_auc:.3f}  pitcher_mu={pt_stats.get(pt, (0,0))[0]:.4f}")

    print(f"\n  Trained {len(models)} pitch-type models, stats for {len(pt_stats)} types")

    # Save
    os.makedirs(_MODEL_DIR, exist_ok=True)
    artifact = {
        "models": models,
        "pt_stats": pt_stats,
        "features": _XWHIFF_FEATURES,
    }
    joblib.dump(artifact, _XWHIFF_MODEL_PATH, compress=3)
    print(f"  Model saved to {_XWHIFF_MODEL_PATH}")


def _load_xwhiff_model():
    """Load cached xWhiff model artifact. Returns None if missing."""
    if not os.path.exists(_XWHIFF_MODEL_PATH):
        return None
    import joblib
    try:
        return joblib.load(_XWHIFF_MODEL_PATH)
    except Exception:
        return None


def _compute_xwhiff(data: pd.DataFrame) -> pd.DataFrame:
    """Compute xWhiffPlus for every pitch using per-pitch-type xWhiff models.

    xWhiffPlus scale: 100 = average, higher = better stuff (more whiffs).
    """
    if data is None or len(data) == 0:
        return data

    artifact = _load_xwhiff_model()
    if artifact is None:
        return data

    models = artifact["models"]
    pt_stats = artifact["pt_stats"]

    df = data.copy()
    df = normalize_pitch_types(df)
    scored = df.dropna(subset=["RelSpeed", "TaggedPitchType"]).copy()
    if scored.empty:
        df["xWhiffPlus"] = np.nan
        return df

    # HorzBreakAdj
    if "HorzBreak" in scored.columns:
        throws = scored.get("PitcherThrows")
        if throws is not None:
            is_l = throws.astype(str).str.lower().str.startswith("l")
            scored["HorzBreakAdj"] = np.where(
                is_l, -scored["HorzBreak"].astype(float), scored["HorzBreak"].astype(float)
            )
        else:
            scored["HorzBreakAdj"] = scored["HorzBreak"].astype(float)
    else:
        scored["HorzBreakAdj"] = np.nan

    # VeloDiff
    fb_types = {"Fastball", "Sinker", "Cutter"}
    fb_velo = (
        scored[scored["TaggedPitchType"].isin(fb_types)]
        .groupby("Pitcher")["RelSpeed"]
        .mean()
    )
    scored["VeloDiff"] = (
        scored["Pitcher"].map(fb_velo).astype(float) - scored["RelSpeed"].astype(float)
    )

    # Encode
    scored["PitcherThrows_enc"] = (
        scored["PitcherThrows"].astype(str).str.lower().str.startswith("l").astype(int)
    )
    scored["BatterSide_enc"] = (
        scored["BatterSide"].astype(str).str.lower().str.startswith("l").astype(int)
    )
    scored["speed_x_ivb"] = (
        scored["RelSpeed"].astype(float) * scored["InducedVertBreak"].astype(float)
    )

    # Ensure all feature columns exist
    for col in _XWHIFF_FEATURES:
        if col not in scored.columns:
            scored[col] = np.nan

    # Predict xWhiff per pitch type and z-score to xWhiffPlus
    xwhiff_scores = pd.Series(np.nan, index=scored.index)
    for pt in scored["TaggedPitchType"].unique():
        if pt not in models or pt not in pt_stats:
            continue
        model = models[pt]
        mean, std = pt_stats[pt]
        if std == 0 or np.isnan(std):
            continue

        mask = scored["TaggedPitchType"] == pt
        X_pt = scored.loc[mask, _XWHIFF_FEATURES].astype(float)
        valid = X_pt.notna().all(axis=1)
        if not valid.any():
            continue

        # Predict P(whiff) on all pitches (not just swings)
        xwhiff_pred = np.full(mask.sum(), np.nan)
        xwhiff_pred[valid.values] = model.predict_proba(X_pt[valid])[:, 1]

        # Z-score pitcher-level mean xWhiff
        temp = pd.Series(xwhiff_pred, index=scored.loc[mask].index)
        pitcher_means = pd.DataFrame({
            "Pitcher": scored.loc[mask, "Pitcher"].values,
            "xwhiff": temp.values,
        }).groupby("Pitcher")["xwhiff"].transform("mean")

        z = (pitcher_means.values - mean) / std
        # Higher P(whiff) = better stuff = higher score (no negate)
        xwhiff_scores[mask] = 100 + z * 10

    df["xWhiffPlus"] = xwhiff_scores.reindex(df.index)
    return df


@st.cache_data(show_spinner="Computing xWhiff Stuff+ grades...")
def _compute_xwhiff_all(data):
    """Cached wrapper for _compute_xwhiff on the full Davidson dataset."""
    return _compute_xwhiff(data)
