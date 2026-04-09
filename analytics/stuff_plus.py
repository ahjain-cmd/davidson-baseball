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
from config import PITCH_TYPE_MAP, PITCH_TYPES_TO_DROP, normalize_pitch_types

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_APP_DIR, "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "stuff_plus_xgb.joblib")
_XWHIFF_MODEL_PATH = os.path.join(_MODEL_DIR, "xwhiff_models.joblib")

# The app-wide normalizer merges Sweepers into Sliders and Knuckle Curves into
# Curveballs. The trained XGBoost artifacts keep those pitch types separate, so
# the model paths need a narrower normalization step that preserves them.
_MODEL_PITCH_TYPE_MAP = {
    src: dst
    for src, dst in PITCH_TYPE_MAP.items()
    if src not in {"Sweeper", "Knuckle Curve"}
}

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

# xWhiff pitch-grade features: exclude approach-angle/location proxies and
# batter-mix context so the grade stays focused on the pitch itself.
_XWHIFF_FEATURES = [
    "RelSpeed", "InducedVertBreak", "HorzBreakAdj", "Extension",
    "SpinRate", "VeloDiff", "PitcherThrows_enc", "speed_x_ivb",
    "total_break", "break_balance", "axis_sin", "axis_cos",
    "RelHeight", "RelSideAdj", "VAAResidual",
]

_LEGACY_XWHIFF_FEATURES = [
    "RelSpeed", "InducedVertBreak", "HorzBreakAdj", "Extension",
    "VertApprAngle", "SpinRate", "SpinAxis", "VeloDiff",
    "PitcherThrows_enc", "BatterSide_enc", "speed_x_ivb",
    "HorzApprAngle",
]

_XWHIFF_VAA_CONTEXT_FEATURES = [
    "PlateLocHeight", "RelHeight", "RelSideAdj", "Extension",
    "RelSpeed", "PitcherThrows_enc",
]


def _normalize_model_pitch_types(data: pd.DataFrame) -> pd.DataFrame:
    """Match the pitch-type buckets used when the XGBoost artifacts were trained."""
    if "TaggedPitchType" not in data.columns:
        return data
    df = data.copy()
    df["TaggedPitchType"] = df["TaggedPitchType"].replace(_MODEL_PITCH_TYPE_MAP)
    df.loc[df["TaggedPitchType"].isin(PITCH_TYPES_TO_DROP), "TaggedPitchType"] = np.nan
    return df


# =============================================================================
#  XGBoost Stuff+ training
# =============================================================================

def train_stuff_plus_model(parquet_path: str) -> None:
    """Train the D1 PitchSim-style Stuff+ artifact."""
    from config import PARQUET_PATH
    from analytics.pitchsim_stuff import train_pitchsim_stuff_model

    pq = parquet_path or PARQUET_PATH
    train_pitchsim_stuff_model(parquet_path=pq, model_path=_MODEL_PATH)


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
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to load Stuff+ model from {_MODEL_PATH}: {e}")
        return None


def _compute_stuff_plus_xgb(data: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Compute Stuff+ using the XGBoost model."""
    if artifact.get("artifact_type") == "pitchsim_lite":
        from analytics.pitchsim_stuff import compute_pitchsim_stuff_plus

        return compute_pitchsim_stuff_plus(data, artifact)

    model = artifact["model"]
    pt_stats = artifact["pt_stats"]

    df = _normalize_model_pitch_types(data)
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
#  xWhiff Stuff+ — pitcher/pitch-type whiff-grade model
# =============================================================================

_SWING_CALLS = [
    "StrikeSwinging", "FoulBall", "FoulBallNotFieldable",
    "FoulBallFieldable", "InPlay",
]

_XWHIFF_XGB_BASE_PARAMS = dict(
    n_estimators=2000,
    max_depth=2,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    gamma=0.05,
    reg_alpha=1.0,
    reg_lambda=8.0,
    tree_method="hist",
    random_state=42,
    early_stopping_rounds=100,
)

_XWHIFF_TREE_CANDIDATES = [
    {"max_depth": 1, "min_child_weight": 5},
    {"max_depth": 2, "min_child_weight": 5},
    {"max_depth": 2, "min_child_weight": 10},
    {"max_depth": 3, "min_child_weight": 10},
    {"max_depth": 3, "min_child_weight": 25},
]

_XWHIFF_STOCHASTIC_CANDIDATES = [
    {"subsample": 0.6, "colsample_bytree": 0.6},
    {"subsample": 0.75, "colsample_bytree": 0.75},
    {"subsample": 0.9, "colsample_bytree": 0.9},
]

_XWHIFF_REGULARIZATION_CANDIDATES = [
    {"gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"gamma": 0.05, "reg_alpha": 0.5, "reg_lambda": 5.0},
    {"gamma": 0.1, "reg_alpha": 1.0, "reg_lambda": 8.0},
    {"gamma": 0.2, "reg_alpha": 2.0, "reg_lambda": 12.0},
]

_XWHIFF_LEARNING_RATE_CANDIDATES = [
    {"learning_rate": 0.01},
    {"learning_rate": 0.02},
    {"learning_rate": 0.03},
    {"learning_rate": 0.05},
]

_MIN_PITCH_TYPE_TRAIN = 500
_MIN_XWHIFF_GROUP_SWINGS = 20
_MIN_XWHIFF_GROUP_ROWS = 15
_XWHIFF_TARGET_PRIOR_SWINGS = 30
_XWHIFF_GRADE_ACTUAL_WEIGHT = 0.20


def _prepare_xwhiff_frame(data: pd.DataFrame, vaa_models: Optional[dict] = None) -> pd.DataFrame:
    """Build the derived fields needed by both xWhiff training and scoring."""
    df = _normalize_model_pitch_types(data)
    if df is None or len(df) == 0:
        return df

    # HorzBreakAdj
    if "HorzBreak" in df.columns:
        throws = df.get("PitcherThrows")
        if throws is not None:
            is_l = throws.astype(str).str.lower().str.startswith("l")
            df["HorzBreakAdj"] = np.where(
                is_l, -df["HorzBreak"].astype(float), df["HorzBreak"].astype(float)
            )
        else:
            df["HorzBreakAdj"] = df["HorzBreak"].astype(float)
    else:
        df["HorzBreakAdj"] = np.nan

    # VeloDiff
    fb_types = {"Fastball", "Sinker", "Cutter"}
    fb_velo = (
        df[df["TaggedPitchType"].isin(fb_types)]
        .groupby("Pitcher")["RelSpeed"]
        .mean()
    )
    df["VeloDiff"] = df["Pitcher"].map(fb_velo).astype(float) - df["RelSpeed"].astype(float)

    # Encodings
    throws = df.get("PitcherThrows")
    if throws is not None:
        df["PitcherThrows_enc"] = (
            throws.astype(str).str.lower().str.startswith("l").astype(int)
        )
    else:
        df["PitcherThrows_enc"] = np.nan

    if "RelSide" in df.columns:
        if throws is not None:
            is_l = throws.astype(str).str.lower().str.startswith("l")
            df["RelSideAdj"] = np.where(
                is_l, -df["RelSide"].astype(float), df["RelSide"].astype(float)
            )
        else:
            df["RelSideAdj"] = df["RelSide"].astype(float)
    else:
        df["RelSideAdj"] = np.nan

    batter_side = df.get("BatterSide")
    if batter_side is not None:
        df["BatterSide_enc"] = (
            batter_side.astype(str).str.lower().str.startswith("l").astype(int)
        )
    else:
        df["BatterSide_enc"] = np.nan

    df["speed_x_ivb"] = df["RelSpeed"].astype(float) * df["InducedVertBreak"].astype(float)
    df["total_break"] = np.sqrt(
        df["InducedVertBreak"].astype(float) ** 2 + df["HorzBreakAdj"].astype(float) ** 2
    )
    df["break_balance"] = (
        df["HorzBreakAdj"].astype(float).abs()
        / (df["HorzBreakAdj"].astype(float).abs() + df["InducedVertBreak"].astype(float).abs() + 1e-6)
    )
    if "SpinAxis" in df.columns:
        axis_rad = np.deg2rad(df["SpinAxis"].astype(float) % 360.0)
        df["axis_sin"] = np.sin(axis_rad)
        df["axis_cos"] = np.cos(axis_rad)
    else:
        df["axis_sin"] = np.nan
        df["axis_cos"] = np.nan

    df["VAAResidual"] = np.nan
    if vaa_models and "VertApprAngle" in df.columns:
        for pt, model in vaa_models.items():
            mask = df["TaggedPitchType"] == pt
            if mask.sum() == 0:
                continue
            X_pt = df.loc[mask, _XWHIFF_VAA_CONTEXT_FEATURES].astype(float)
            valid = X_pt.notna().all(axis=1) & df.loc[mask, "VertApprAngle"].notna()
            if not valid.any():
                continue
            pred = model.predict(X_pt[valid])
            resid = df.loc[mask, "VertApprAngle"].astype(float).copy()
            resid.loc[valid] = resid.loc[valid] - pred
            df.loc[mask, "VAAResidual"] = resid.values
    df["VAAResidual"] = df["VAAResidual"].fillna(0.0)

    if "PitchCall" in df.columns:
        df["is_swing"] = df["PitchCall"].isin(_SWING_CALLS).astype(int)
        df["is_whiff"] = (df["PitchCall"] == "StrikeSwinging").astype(int)
    else:
        df["is_swing"] = 0
        df["is_whiff"] = 0

    needed_cols = set(_XWHIFF_FEATURES) | set(_LEGACY_XWHIFF_FEATURES)
    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df


def _fit_xwhiff_vaa_models(df: pd.DataFrame) -> Dict[str, object]:
    """Fit per-pitch-type residual VAA models to strip out location effects."""
    from sklearn.linear_model import LinearRegression

    vaa_models: Dict[str, object] = {}
    if df.empty or "VertApprAngle" not in df.columns:
        return vaa_models

    for pt, pt_df in df.groupby("TaggedPitchType"):
        if pd.isna(pt):
            continue
        X_pt = pt_df[_XWHIFF_VAA_CONTEXT_FEATURES].astype(float)
        y_pt = pd.to_numeric(pt_df["VertApprAngle"], errors="coerce")
        valid = X_pt.notna().all(axis=1) & y_pt.notna()
        if valid.sum() < 200:
            continue
        model = LinearRegression()
        model.fit(X_pt[valid], y_pt[valid])
        vaa_models[str(pt)] = model

    return vaa_models


def _aggregate_xwhiff_groups(
    df: pd.DataFrame,
    feature_cols,
    min_swings: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate pitch rows into one pitch-quality row per pitcher/pitch type."""
    if df.empty or "Pitcher" not in df.columns or "TaggedPitchType" not in df.columns:
        return pd.DataFrame()

    use_cols = ["Pitcher", "TaggedPitchType", "is_swing", "is_whiff", *feature_cols]
    grouped = df[use_cols].groupby(["Pitcher", "TaggedPitchType"], sort=False, observed=False)

    agg = grouped[list(feature_cols)].mean()
    agg["n_pitches"] = grouped.size()
    agg["swings"] = grouped["is_swing"].sum()
    agg["whiffs"] = grouped["is_whiff"].sum()
    agg = agg.reset_index()
    agg["whiff_rate"] = np.where(
        agg["swings"] > 0,
        agg["whiffs"] / agg["swings"],
        np.nan,
    )

    if min_swings is not None:
        agg = agg[agg["swings"] >= min_swings].copy()

    return agg


def _fit_xwhiff_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    w_train: pd.Series,
    w_val: pd.Series,
    params: dict,
) -> dict:
    """Fit one XGBoost candidate and return validation diagnostics."""
    from xgboost import XGBRegressor

    fit_params = dict(_XWHIFF_XGB_BASE_PARAMS)
    fit_params.update(params)
    fit_params["objective"] = "reg:squarederror"
    fit_params["eval_metric"] = "rmse"

    model = XGBRegressor(**fit_params)
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        sample_weight_eval_set=[w_train, w_val],
        verbose=0,
    )

    evals = model.evals_result()
    best_iteration = int(getattr(model, "best_iteration", fit_params["n_estimators"] - 1))
    train_rmse = float(evals["validation_0"]["rmse"][best_iteration])
    val_rmse = float(evals["validation_1"]["rmse"][best_iteration])
    val_pred = np.clip(model.predict(X_val), 0.0, 1.0)
    val_r = float(np.corrcoef(y_val, val_pred)[0, 1]) if len(y_val) > 1 else float(np.nan)
    return {
        "model": model,
        "params": fit_params,
        "best_iteration": best_iteration,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "val_r": val_r,
    }


def _select_xwhiff_xgb_fit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    w_train: pd.Series,
    w_val: pd.Series,
) -> dict:
    """Step-wise XGBoost tuning following the book's tuning order."""
    best = _fit_xwhiff_candidate(X_train, y_train, X_val, y_val, w_train, w_val, {})
    candidate_rounds = [
        _XWHIFF_TREE_CANDIDATES,
        _XWHIFF_STOCHASTIC_CANDIDATES,
        _XWHIFF_REGULARIZATION_CANDIDATES,
        _XWHIFF_LEARNING_RATE_CANDIDATES,
    ]

    for candidates in candidate_rounds:
        round_best = best
        for update in candidates:
            trial = _fit_xwhiff_candidate(
                X_train,
                y_train,
                X_val,
                y_val,
                w_train,
                w_val,
                {**best["params"], **update},
            )
            if trial["val_rmse"] < round_best["val_rmse"] - 1e-6:
                round_best = trial
        best = round_best

    return best


def train_xwhiff_model(parquet_path: str) -> None:
    """Train a pitcher/pitch-type xWhiff grade model.

    The target is a shrunk whiff rate on swings for each pitcher/pitch-type
    bucket. That keeps the model aligned with the pitch grades we surface and
    removes pitch-level location noise from the target.
    """
    import duckdb
    import joblib
    from sklearn.model_selection import train_test_split

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
            Extension, RelHeight, RelSide, PlateLocHeight,
            VertApprAngle, SpinRate, SpinAxis,
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
    base_df = _prepare_xwhiff_frame(df)
    vaa_models = _fit_xwhiff_vaa_models(base_df)
    df = _prepare_xwhiff_frame(df, vaa_models=vaa_models)
    print(f"  Loaded {len(df):,} pitches")

    df = df.dropna(subset=["TaggedPitchType"])

    models: Dict[str, object] = {}
    pt_stats: Dict[str, Tuple[float, float]] = {}
    pt_grade_stats: Dict[str, Tuple[float, float]] = {}
    pt_priors: Dict[str, float] = {}
    feature_importance: Dict[str, Dict[str, float]] = {}
    training_diagnostics: Dict[str, dict] = {}

    for pt in sorted(df["TaggedPitchType"].dropna().unique()):
        pt_all = df[df["TaggedPitchType"] == pt].copy()
        pt_groups = _aggregate_xwhiff_groups(
            pt_all,
            feature_cols=_XWHIFF_FEATURES,
            min_swings=_MIN_XWHIFF_GROUP_SWINGS,
        )

        if pt_groups["swings"].sum() < _MIN_PITCH_TYPE_TRAIN:
            print(f"  Skipping {pt}: {int(pt_groups['swings'].sum())} swings < {_MIN_PITCH_TYPE_TRAIN}")
            continue
        if len(pt_groups) < _MIN_XWHIFF_GROUP_ROWS:
            print(f"  Skipping {pt}: {len(pt_groups)} pitcher groups < {_MIN_XWHIFF_GROUP_ROWS}")
            continue

        league_rate = float(pt_groups["whiffs"].sum() / pt_groups["swings"].sum())
        pt_priors[pt] = league_rate
        pt_groups["target_whiff_rate"] = (
            pt_groups["whiffs"] + league_rate * _XWHIFF_TARGET_PRIOR_SWINGS
        ) / (pt_groups["swings"] + _XWHIFF_TARGET_PRIOR_SWINGS)

        X = pt_groups[_XWHIFF_FEATURES].astype(float)
        y = pt_groups["target_whiff_rate"].astype(float)
        w = pt_groups["swings"].astype(float).clip(lower=1.0)

        valid = X.notna().all(axis=1) & y.notna() & w.notna()
        X = X[valid]
        y = y[valid]
        w = w[valid]

        if len(X) < _MIN_XWHIFF_GROUP_ROWS:
            print(f"  Skipping {pt}: only {len(X)} valid pitcher groups")
            continue

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X,
            y,
            w,
            test_size=0.2,
            random_state=42,
        )
        if len(X_val) < 3:
            print(f"  Skipping {pt}: validation split too small")
            continue

        fit_info = _select_xwhiff_xgb_fit(X_train, y_train, X_val, y_val, w_train, w_val)
        model = fit_info["model"]
        models[pt] = model

        full_pred = np.clip(model.predict(X), 0.0, 1.0)
        if len(full_pred) >= 10:
            pt_stats[pt] = (float(np.mean(full_pred)), float(np.std(full_pred)))
            full_grade = full_pred + _XWHIFF_GRADE_ACTUAL_WEIGHT * (y.values - league_rate)
            pt_grade_stats[pt] = (float(np.mean(full_grade)), float(np.std(full_grade)))

        feature_importance[pt] = {
            feat: float(val)
            for feat, val in zip(_XWHIFF_FEATURES, model.feature_importances_)
        }

        training_diagnostics[pt] = {
            "best_iteration": int(fit_info["best_iteration"]),
            "train_rmse": float(fit_info["train_rmse"]),
            "val_rmse": float(fit_info["val_rmse"]),
            "val_r": float(fit_info["val_r"]) if np.isfinite(fit_info["val_r"]) else float(np.nan),
            "params": {
                k: fit_info["params"][k]
                for k in [
                    "max_depth",
                    "min_child_weight",
                    "subsample",
                    "colsample_bytree",
                    "gamma",
                    "reg_alpha",
                    "reg_lambda",
                    "learning_rate",
                    "n_estimators",
                    "early_stopping_rounds",
                ]
            },
        }
        print(
            f"  {pt:20s}  groups={len(X):,}  swings={int(w.sum()):,}  "
            f"whiff%={league_rate * 100:.1f}  val_rmse={fit_info['val_rmse']:.3f}  "
            f"train_rmse={fit_info['train_rmse']:.3f}  "
            f"best_iter={fit_info['best_iteration']}  val_r={fit_info['val_r']:.3f}  "
            f"pitcher_mu={pt_stats.get(pt, (0, 0))[0]:.4f}"
        )

    print(f"\n  Trained {len(models)} pitch-type models, stats for {len(pt_stats)} types")

    # Save
    os.makedirs(_MODEL_DIR, exist_ok=True)
    artifact = {
        "models": models,
        "pt_stats": pt_stats,
        "pt_grade_stats": pt_grade_stats,
        "pt_priors": pt_priors,
        "vaa_models": vaa_models,
        "features": _XWHIFF_FEATURES,
        "feature_importance": feature_importance,
        "training_diagnostics": training_diagnostics,
        "aggregation": "pitcher_pitch_type",
        "target": "shrunk_whiff_rate",
        "min_group_swings": _MIN_XWHIFF_GROUP_SWINGS,
        "grade_actual_weight": _XWHIFF_GRADE_ACTUAL_WEIGHT,
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


def _compute_xwhiff_legacy(data: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Compatibility path for older pitch-level classifier artifacts."""
    models = artifact["models"]
    pt_stats = artifact["pt_stats"]
    feature_cols = artifact.get("features", _LEGACY_XWHIFF_FEATURES)
    vaa_models = artifact.get("vaa_models")

    df = _prepare_xwhiff_frame(data, vaa_models=vaa_models)
    scored = df.dropna(subset=["RelSpeed", "TaggedPitchType"]).copy()
    if scored.empty:
        df["xWhiff"] = np.nan
        df["xWhiffGrade"] = np.nan
        df["xWhiffPlus"] = np.nan
        return df

    xwhiff_raw = pd.Series(np.nan, index=scored.index)
    xwhiff_scores = pd.Series(np.nan, index=scored.index)
    for pt in scored["TaggedPitchType"].unique():
        if pt not in models or pt not in pt_stats:
            continue
        model = models[pt]
        mean, std = pt_stats[pt]
        if std == 0 or np.isnan(std):
            continue

        mask = scored["TaggedPitchType"] == pt
        X_pt = scored.loc[mask, feature_cols].astype(float)
        valid = X_pt.notna().all(axis=1)
        if not valid.any():
            continue

        pred = np.full(mask.sum(), np.nan)
        pred[valid.values] = model.predict_proba(X_pt[valid])[:, 1]
        xwhiff_raw[mask] = pred

        temp = pd.Series(pred, index=scored.loc[mask].index)
        pitcher_means = pd.DataFrame({
            "Pitcher": scored.loc[mask, "Pitcher"].values,
            "xwhiff": temp.values,
        }).groupby("Pitcher")["xwhiff"].transform("mean")

        z = (pitcher_means.values - mean) / std
        xwhiff_scores[mask] = 100 + z * 10

    df["xWhiff"] = xwhiff_raw.reindex(df.index)
    df["xWhiffGrade"] = xwhiff_raw.reindex(df.index)
    df["xWhiffPlus"] = xwhiff_scores.reindex(df.index)
    return df


def _compute_xwhiff(data: pd.DataFrame) -> pd.DataFrame:
    """Compute per-pitch-type xWhiff grades for every pitch row.

    `xWhiff` is the model's raw expected whiff rate for that pitcher/pitch type.
    `xWhiffGrade` blends in observed whiff performance from the current sample.
    `xWhiffPlus` rescales that displayed grade against the same blended
    population distribution used during training.
    """
    if data is None or len(data) == 0:
        return data

    artifact = _load_xwhiff_model()
    if artifact is None:
        return data
    if artifact.get("aggregation") != "pitcher_pitch_type":
        return _compute_xwhiff_legacy(data, artifact)

    models = artifact["models"]
    pt_stats = artifact["pt_stats"]
    pt_grade_stats = artifact.get("pt_grade_stats", pt_stats)
    pt_priors = artifact.get("pt_priors", {})
    feature_cols = artifact.get("features", _XWHIFF_FEATURES)
    actual_weight = float(artifact.get("grade_actual_weight", _XWHIFF_GRADE_ACTUAL_WEIGHT))
    vaa_models = artifact.get("vaa_models")

    df = _prepare_xwhiff_frame(data, vaa_models=vaa_models)
    scored = df.dropna(subset=["RelSpeed", "TaggedPitchType"]).copy()
    if scored.empty:
        df["xWhiff"] = np.nan
        df["xWhiffGrade"] = np.nan
        df["xWhiffPlus"] = np.nan
        return df

    grouped = _aggregate_xwhiff_groups(scored, feature_cols=feature_cols)
    if grouped.empty:
        df["xWhiff"] = np.nan
        df["xWhiffGrade"] = np.nan
        df["xWhiffPlus"] = np.nan
        return df

    score_rows = []
    for pt, model in models.items():
        if pt not in pt_stats:
            continue
        mean, std = pt_grade_stats.get(pt, pt_stats[pt])
        prior_rate = float(pt_priors.get(pt, mean))
        pt_groups = grouped[grouped["TaggedPitchType"] == pt].copy()
        if pt_groups.empty:
            continue

        X_pt = pt_groups[feature_cols].astype(float)
        valid = X_pt.notna().all(axis=1)
        if not valid.any():
            continue

        pred = np.full(len(pt_groups), np.nan)
        pred[valid.values] = np.clip(model.predict(X_pt[valid]), 0.0, 1.0)
        empirical_whiff = (
            pt_groups["whiffs"].astype(float) + prior_rate * _XWHIFF_TARGET_PRIOR_SWINGS
        ) / (pt_groups["swings"].astype(float) + _XWHIFF_TARGET_PRIOR_SWINGS)
        grade_value = pred + actual_weight * (empirical_whiff.values - prior_rate)
        plus = np.where(
            np.isfinite(grade_value) & np.isfinite(std) & (std > 0),
            100 + ((grade_value - mean) / std) * 10,
            np.nan,
        )

        pt_groups["xWhiff"] = pred
        pt_groups["xWhiffGrade"] = grade_value
        pt_groups["xWhiffPlus"] = plus
        score_rows.append(
            pt_groups[["Pitcher", "TaggedPitchType", "xWhiff", "xWhiffGrade", "xWhiffPlus"]]
        )

    if not score_rows:
        df["xWhiff"] = np.nan
        df["xWhiffGrade"] = np.nan
        df["xWhiffPlus"] = np.nan
        return df

    score_df = pd.concat(score_rows, ignore_index=True)
    scored["_row_id"] = scored.index
    scored = scored.merge(score_df, on=["Pitcher", "TaggedPitchType"], how="left")
    scored = scored.set_index("_row_id")
    df["xWhiff"] = scored["xWhiff"].reindex(df.index) if "xWhiff" in scored.columns else np.nan
    df["xWhiffGrade"] = scored["xWhiffGrade"].reindex(df.index) if "xWhiffGrade" in scored.columns else np.nan
    df["xWhiffPlus"] = scored["xWhiffPlus"].reindex(df.index) if "xWhiffPlus" in scored.columns else np.nan
    return df


@st.cache_data(show_spinner="Computing xWhiff Stuff+ grades...")
def _compute_xwhiff_all(data):
    """Cached wrapper for _compute_xwhiff on the full Davidson dataset."""
    return _compute_xwhiff(data)
