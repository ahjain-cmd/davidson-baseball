"""D1 PitchSim-style Stuff+ training and runtime scoring.

This module adapts the Aldred PitchSim architecture to D1 TrackMan:
- D1-only training sample
- soft pitch archetypes from stuff-only features
- standardized count/location priors by archetype and platoon
- context-inclusive event models
- distilled runtime regressors for fast Stuff+ scoring

The runtime artifact is intentionally lightweight: scoring uses only the
distilled regressors and per-pitch-type population stats, while the heavier
cluster/prior/event-model machinery is only required at training time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import PITCH_TYPE_MAP, PITCH_TYPES_TO_DROP


_ARTIFACT_VERSION = "d1_pitchsim_lite_v4"
_D1_LEVEL = "D1"

_MODEL_PITCH_TYPE_MAP = {
    src: dst
    for src, dst in PITCH_TYPE_MAP.items()
    if src not in {"Sweeper", "Knuckle Curve"}
}

_PITCH_TYPE_LABELS = {
    "Fastball": 0,
    "Sinker": 1,
    "Cutter": 2,
    "Slider": 3,
    "Curveball": 4,
    "Changeup": 5,
    "Splitter": 6,
    "Knuckle Curve": 7,
    "Sweeper": 8,
}

_FB_TYPES = {"Fastball", "Sinker", "Cutter"}
_SWING_CALLS = {
    "StrikeSwinging",
    "FoulBall",
    "FoulBallNotFieldable",
    "FoulBallFieldable",
    "InPlay",
}
_BALL_CALLS = {
    "ballcalled",
    "ballindirt",
    "automaticball",
}
_HOME_RUN_RESULTS = {"homerun"}
_PLAY_RESULT_DAMAGE = {
    "single": 1.0,
    "double": 2.0,
    "triple": 3.0,
    "homerun": 4.0,
}

_CLUSTER_NAMES = [
    "sinker",
    "fastball",
    "slider",
    "offspeed",
    "curveball",
    "cutter",
]

_COUNTS = [(balls, strikes) for balls in range(4) for strikes in range(3)]
_COUNT_TO_INDEX = {count: idx for idx, count in enumerate(_COUNTS)}
_N_CLUSTERS = len(_CLUSTER_NAMES)
_TRAIN_TARGET_ROWS = 2_500_000
_SIM_TARGET_ROWS = 30_000
_CLUSTER_FIT_ROWS = 350_000
_WEIGHT_MODEL_ROWS = 600_000
_MAX_DISTILL_ROWS = 80_000
_MIN_CONTEXT_WEIGHT = 100.0
_EVENT_MODEL_FIT_ROWS = 100_000
_EVENT_MODEL_VAL_ROWS = 30_000
_XGB_TUNE_FOLDS = 3
_XGB_TUNE_MAX_EVALS = 5
_WEIGHT_TUNE_ROWS = 40_000
_EVENT_TUNE_ROWS = 40_000
_DISTILL_TUNE_ROWS = 20_000
_WEIGHT_BASE_ESTIMATORS = 200
_EVENT_BASE_ESTIMATORS = 250
_DISTILL_BASE_ESTIMATORS = 200
_EARLY_STOPPING_ROUNDS = 50
_FINAL_LR_SCALE = 0.2
_FINAL_ESTIMATOR_SCALE = 4

_GRID_X = np.linspace(-2.7, 2.7, 28)
_GRID_Z = np.linspace(-1.0, 5.4, 33)
_GRID = np.array(np.meshgrid(_GRID_X, _GRID_Z)).T.reshape(-1, 2)
_GRID_LEN = len(_GRID)
_CONTEXT_LEN = _GRID_LEN * len(_COUNTS)
_CONTEXT_BALLS = np.repeat(np.array([count[0] for count in _COUNTS], dtype=float), _GRID_LEN)
_CONTEXT_STRIKES = np.repeat(np.array([count[1] for count in _COUNTS], dtype=float), _GRID_LEN)
_CONTEXT_PLATE_X = np.tile(_GRID[:, 0], len(_COUNTS)).astype(float)
_CONTEXT_PLATE_Z = np.tile(_GRID[:, 1], len(_COUNTS)).astype(float)
_KDE_BANDWIDTH = 0.10
_KDE_SAMPLE_ROWS = 750
_KDE_JOBS = 8
_FCM_M = 1.5
_FCM_MAXITER = 100
_FCM_ERROR = 1e-3

_VAA_CONTEXT_FEATURES = [
    "PlateLocHeight",
    "RelHeight",
    "RelSideAdj",
    "Extension",
    "RelSpeed",
    "PitcherThrows_enc",
]

_CLUSTER_FEATURES = [
    "physics_speed",
    "speed_diff",
    "lift",
    "lift_diff",
    "transverse_pit",
    "transverse_pit_diff",
    "RelSideAdj",
    "release_pos_y",
    "RelHeight",
    "VAAResidual",
]

_EVENT_BASE_FEATURES = [
    "physics_speed",
    "speed_diff",
    "lift",
    "lift_diff",
    "transverse_pit",
    "transverse_pit_diff",
    "RelSideAdj",
    "release_pos_y",
    "RelHeight",
    "Extension",
    "SpinRate",
    "axis_sin",
    "axis_cos",
    "InducedVertBreak",
    "HorzBreakAdj",
    "total_break",
    "break_balance",
    "VAAResidual",
    "PitcherThrows_enc",
]

_EVENT_CONTEXT_FEATURES = [
    "PlateLocSide",
    "PlateLocHeight",
    "BatterSide_enc",
    "Balls",
    "Strikes",
]

_DISTILL_FEATURES = [
    "RelSpeed",
    "InducedVertBreak",
    "HorzBreakAdj",
    "Extension",
    "RelHeight",
    "RelSideAdj",
    "SpinRate",
    "axis_sin",
    "axis_cos",
    "VeloDiff",
    "total_break",
    "break_balance",
    "speed_x_ivb",
    "VAAResidual",
    "PitcherThrows_enc",
    "pitch_type_enc",
]


def _safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.size < 2 or arr_b.size < 2:
        return np.nan
    if np.nanstd(arr_a) <= 1e-12 or np.nanstd(arr_b) <= 1e-12:
        return np.nan
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def _stratified_sample_by_pitch_type(df: pd.DataFrame, total_rows: int, min_rows: int) -> pd.DataFrame:
    if df.empty or total_rows <= 0:
        return df.iloc[0:0].copy()
    total = float(len(df))
    pieces = []
    for _, group in df.groupby("TaggedPitchType", observed=False, sort=False):
        target = max(min_rows, int(round(total_rows * len(group) / total)))
        take = min(len(group), target)
        pieces.append(group.sample(take, random_state=42))
    return pd.concat(pieces, ignore_index=True)


def _sample_tuning_frame(frame: pd.DataFrame, max_rows: int, min_rows: int = 750) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_rows:
        return frame.copy()
    if "TaggedPitchType" in frame.columns and frame["TaggedPitchType"].notna().any():
        return _stratified_sample_by_pitch_type(frame, total_rows=max_rows, min_rows=min_rows)
    return frame.sample(max_rows, random_state=42).reset_index(drop=True)


def _sample_binary_frame(frame: pd.DataFrame, target: str, max_rows: int) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_rows:
        return frame.copy()
    pos = frame[frame[target] == 1]
    neg = frame[frame[target] == 0]
    pos_take = min(len(pos), max(5_000, int(round(max_rows * len(pos) / len(frame)))))
    neg_take = min(len(neg), max_rows - pos_take)
    pieces = []
    if pos_take > 0:
        pieces.append(pos.sample(pos_take, random_state=42))
    if neg_take > 0:
        pieces.append(neg.sample(neg_take, random_state=42))
    out = pd.concat(pieces, ignore_index=True)
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def _build_group_folds(groups: Sequence[str], n_splits: int = _XGB_TUNE_FOLDS) -> List[Tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    groups_arr = np.asarray(groups)
    unique_groups = pd.Series(groups_arr).nunique()
    if unique_groups >= 2:
        folds = min(n_splits, int(unique_groups))
        if folds >= 2:
            splitter = GroupKFold(n_splits=folds)
            dummy = np.zeros(len(groups_arr), dtype=np.int8)
            return list(splitter.split(dummy, dummy, groups_arr))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dummy = np.zeros(len(groups_arr), dtype=np.int8)
    return list(splitter.split(dummy, dummy, groups_arr))


def _xgb_search_space(label: str) -> Dict[str, object]:
    from hyperopt import hp

    return {
        "learning_rate": hp.lognormal(f"{label}_learning_rate", -3.0, 0.35),
        "max_depth": hp.quniform(f"{label}_max_depth", 2, 8, 1),
        "min_child_weight": hp.qloguniform(
            f"{label}_min_child_weight",
            np.log(8.0),
            np.log(128.0),
            1.0,
        ),
        "gamma": hp.loguniform(f"{label}_gamma", np.log(0.01), np.log(5.0)),
    }


def _coerce_xgb_tuned_params(params: Dict[str, float]) -> Dict[str, float]:
    return {
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(round(float(params["max_depth"]))),
        "min_child_weight": int(round(float(params["min_child_weight"]))),
        "gamma": float(params["gamma"]),
    }


def _tune_xgb_params(
    frame: pd.DataFrame,
    features: Sequence[str],
    target: str,
    group_col: str,
    model_cls,
    base_params: Dict[str, object],
    task: str,
    label: str,
    max_rows: int,
    max_evals: int = _XGB_TUNE_MAX_EVALS,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
    from sklearn.metrics import log_loss

    tune_frame = frame.dropna(subset=list(features) + [target, group_col]).copy()
    if tune_frame.empty:
        final_params = dict(base_params)
        final_params["learning_rate"] = float(final_params["learning_rate"]) * _FINAL_LR_SCALE
        final_params["n_estimators"] = int(round(final_params["n_estimators"] * _FINAL_ESTIMATOR_SCALE))
        return final_params, {"skipped": True, "reason": "empty_frame"}
    if task == "binary" and tune_frame[target].nunique() < 2:
        final_params = dict(base_params)
        final_params["learning_rate"] = float(final_params["learning_rate"]) * _FINAL_LR_SCALE
        final_params["n_estimators"] = int(round(final_params["n_estimators"] * _FINAL_ESTIMATOR_SCALE))
        return final_params, {"skipped": True, "reason": "single_class"}

    tune_frame = _sample_tuning_frame(tune_frame, max_rows=max_rows)
    X = tune_frame[list(features)].astype(np.float32).to_numpy()
    if task == "binary":
        y = tune_frame[target].astype(int).to_numpy()
    else:
        y = tune_frame[target].astype(np.float32).to_numpy()
    groups = tune_frame[group_col].astype(str).to_numpy()
    folds = _build_group_folds(groups, n_splits=_XGB_TUNE_FOLDS)
    space = _xgb_search_space(label)
    trials = Trials()

    def _objective(sampled: Dict[str, float]) -> Dict[str, object]:
        tuned = _coerce_xgb_tuned_params(sampled)
        params = dict(base_params)
        params.update(tuned)
        fold_losses: List[float] = []
        fold_iterations: List[int] = []
        for train_idx, val_idx in folds:
            y_train = y[train_idx]
            y_val = y[val_idx]
            if task == "binary" and (np.unique(y_train).size < 2 or np.unique(y_val).size < 2):
                continue
            model = model_cls(**params)
            model.fit(
                X[train_idx],
                y_train,
                eval_set=[(X[val_idx], y_val)],
                verbose=False,
            )
            if task == "binary":
                prob = np.clip(model.predict_proba(X[val_idx])[:, 1], 1e-6, 1 - 1e-6)
                fold_losses.append(float(log_loss(y_val, prob, labels=[0, 1])))
            else:
                pred = model.predict(X[val_idx])
                fold_losses.append(float(np.sqrt(np.mean(np.square(y_val - pred)))))
            best_iter = getattr(model, "best_iteration", None)
            if best_iter is None:
                try:
                    best_iter = model.get_booster().num_boosted_rounds()
                except Exception:
                    best_iter = params.get("n_estimators")
            fold_iterations.append(int(best_iter))
        loss = float(np.mean(fold_losses)) if fold_losses else 1e9
        iteration_mean = float(np.mean(fold_iterations)) if fold_iterations else np.nan
        return {
            "loss": loss,
            "status": STATUS_OK,
            "best_iteration_mean": iteration_mean,
        }

    best = fmin(
        fn=_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42),
        show_progressbar=False,
    )
    best = space_eval(space, best)
    tuned = _coerce_xgb_tuned_params(best)
    final_params = dict(base_params)
    final_params.update(tuned)
    final_params["learning_rate"] = float(final_params["learning_rate"]) * _FINAL_LR_SCALE
    final_params["n_estimators"] = int(round(final_params["n_estimators"] * _FINAL_ESTIMATOR_SCALE))

    best_loss = float(np.min([trial["result"]["loss"] for trial in trials.trials])) if trials.trials else np.nan
    best_iteration = np.nan
    if trials.trials:
        best_idx = int(np.argmin([trial["result"]["loss"] for trial in trials.trials]))
        best_iteration = float(trials.trials[best_idx]["result"].get("best_iteration_mean", np.nan))
    meta = {
        "sample_rows": int(len(tune_frame)),
        "folds": int(len(folds)),
        "task": task,
        "best_loss": best_loss,
        "best_iteration_mean": best_iteration,
        "tuned_params": tuned,
        "final_learning_rate": float(final_params["learning_rate"]),
        "final_n_estimators": int(final_params["n_estimators"]),
    }
    return final_params, meta


def _normalize_model_pitch_types(data: pd.DataFrame) -> pd.DataFrame:
    if "TaggedPitchType" not in data.columns:
        return data
    df = data.copy()
    df["TaggedPitchType"] = df["TaggedPitchType"].replace(_MODEL_PITCH_TYPE_MAP)
    df.loc[df["TaggedPitchType"].isin(PITCH_TYPES_TO_DROP), "TaggedPitchType"] = np.nan
    return df


def _normalize_string_token(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )


def _compose_fip_raw(
    whiff: np.ndarray | pd.Series,
    called_strike: np.ndarray | pd.Series,
    ball: np.ndarray | pd.Series,
    home_run: np.ndarray | pd.Series,
    damage: np.ndarray | pd.Series,
) -> np.ndarray:
    whiff_arr = np.asarray(whiff, dtype=float)
    called_arr = np.asarray(called_strike, dtype=float)
    ball_arr = np.asarray(ball, dtype=float)
    hr_arr = np.asarray(home_run, dtype=float)
    damage_arr = np.asarray(damage, dtype=float)
    # Approximate per-pitch FIP pressure:
    # a ball is about one quarter of a walk, a strike about one third of a K,
    # and HR retains the heavy terminal-event penalty. Damage is a smaller
    # contact-quality penalty so FIP-facing grades do not ignore loud contact.
    return (
        0.75 * ball_arr
        + 13.0 * hr_arr
        + 1.50 * damage_arr
        - 0.90 * whiff_arr
        - 0.60 * called_arr
    )


def _count_bucket(balls: pd.Series, strikes: pd.Series) -> pd.Series:
    balls_i = pd.to_numeric(balls, errors="coerce").fillna(0).astype(int)
    strikes_i = pd.to_numeric(strikes, errors="coerce").fillna(0).astype(int)
    bucket = np.full(len(balls_i), "even", dtype=object)
    bucket[(balls_i == 0) & (strikes_i == 0)] = "first_pitch"
    bucket[strikes_i >= 2] = "two_strike"
    bucket[(strikes_i < 2) & (balls_i > strikes_i)] = "hitter_ahead"
    bucket[(strikes_i < 2) & (strikes_i > balls_i)] = "pitcher_ahead"
    return pd.Series(bucket, index=balls.index, dtype="object")


def _normalized_pitch_type_sql(raw_col: str = "TaggedPitchType") -> str:
    return f"""
        CASE
            WHEN {raw_col} IN ('Undefined', 'Other', 'Knuckleball') THEN NULL
            WHEN {raw_col} = 'FourSeamFastBall' THEN 'Fastball'
            WHEN {raw_col} IN ('OneSeamFastBall', 'TwoSeamFastBall') THEN 'Sinker'
            WHEN {raw_col} = 'ChangeUp' THEN 'Changeup'
            ELSE {raw_col}
        END
    """


def _base_pitch_query(parquet_path: str) -> str:
    pt_sql = _normalized_pitch_type_sql("TaggedPitchType")
    return f"""
        SELECT
            GameID,
            Inning,
            PAofInning,
            PitchUID,
            PitchNo,
            Pitcher,
            Batter,
            CASE WHEN PitcherThrows IN ('Left', 'L') THEN 'L' ELSE 'R' END AS PitcherThrows,
            CASE WHEN BatterSide IN ('Left', 'L') THEN 'L' ELSE 'R' END AS BatterSide,
            Balls,
            Strikes,
            PitchCall,
            PlayResult,
            TaggedHitType,
            KorBB,
            RelSpeed,
            InducedVertBreak,
            HorzBreak,
            Extension,
            RelHeight,
            RelSide,
            VertApprAngle,
            PlateLocHeight,
            PlateLocSide,
            SpinRate,
            SpinAxis,
            ExitSpeed,
            Angle,
            Distance,
            vx0,
            vy0,
            vz0,
            ax0,
            ay0,
            az0,
            Level,
            {pt_sql} AS TaggedPitchType
        FROM read_parquet('{parquet_path}')
        WHERE Level = '{_D1_LEVEL}'
          AND PitchCall IS NOT NULL
          AND TaggedPitchType IS NOT NULL
          AND Pitcher IS NOT NULL
          AND PitcherThrows IS NOT NULL
          AND BatterSide IS NOT NULL
          AND RelSpeed IS NOT NULL
          AND InducedVertBreak IS NOT NULL
          AND HorzBreak IS NOT NULL
          AND Extension IS NOT NULL
          AND RelHeight IS NOT NULL
          AND RelSide IS NOT NULL
          AND PlateLocHeight IS NOT NULL
          AND PlateLocSide IS NOT NULL
    """


def _sample_caps(counts: pd.DataFrame, target_total: int) -> Dict[str, int]:
    counts = counts.copy()
    total = float(counts["n"].sum()) or 1.0
    base = np.round(target_total * counts["n"] / total).astype(int)
    floor = np.where(counts["n"] >= 50_000, 12_000, counts["n"])
    counts["take"] = np.minimum(counts["n"], np.maximum(base, floor))
    return {
        str(row.TaggedPitchType): int(row.take)
        for row in counts.itertuples(index=False)
        if pd.notna(row.TaggedPitchType)
    }


def _fetch_training_frame(parquet_path: str, target_total: int = _TRAIN_TARGET_ROWS) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(":memory:")
    base_query = _base_pitch_query(parquet_path)
    counts = con.execute(
        f"""
        WITH base AS ({base_query})
        SELECT TaggedPitchType, COUNT(*) AS n
        FROM base
        WHERE TaggedPitchType IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
        """
    ).fetchdf()
    caps = _sample_caps(counts, target_total=target_total)
    thresholds = {}
    for row in counts.itertuples(index=False):
        pt = str(row.TaggedPitchType)
        n = float(row.n)
        take = float(caps.get(pt, 0))
        frac = 1.0 if n <= 0 else min(1.0, take / n)
        thresholds[pt] = int(round(frac * 1_000_000))
    case_expr = "CASE TaggedPitchType " + " ".join(
        f"WHEN '{pt}' THEN {thresholds.get(pt, 0)}" for pt in thresholds
    ) + " ELSE 0 END"
    query = f"""
        WITH base AS ({base_query}),
        filtered AS (
            SELECT *
            FROM base
            WHERE TaggedPitchType IS NOT NULL
              AND TaggedPitchType NOT IN ('Undefined', 'Other', 'Knuckleball')
        )
        SELECT *
        FROM filtered
        WHERE ABS(HASH(COALESCE(PitchUID, GameID || ':' || CAST(PitchNo AS VARCHAR) || ':' || Pitcher || ':' || COALESCE(Batter, '')))) % 1000000 < {case_expr}
    """
    df = con.execute(query).fetchdf()
    con.close()
    return _normalize_model_pitch_types(df)


def _fit_vaa_models(df: pd.DataFrame) -> Dict[str, object]:
    from sklearn.linear_model import LinearRegression

    models: Dict[str, object] = {}
    if df.empty or "VertApprAngle" not in df.columns:
        return models

    for pt, pt_df in df.groupby("TaggedPitchType", observed=False):
        if pd.isna(pt):
            continue
        X = pt_df[_VAA_CONTEXT_FEATURES].astype(float)
        y = pd.to_numeric(pt_df["VertApprAngle"], errors="coerce")
        valid = X.notna().all(axis=1) & y.notna()
        if valid.sum() < 200:
            continue
        model = LinearRegression()
        model.fit(X.loc[valid], y.loc[valid])
        models[str(pt)] = model
    return models


def _prepare_pitchsim_frame(data: pd.DataFrame, vaa_models: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    df = _normalize_model_pitch_types(data)
    if df is None or len(df) == 0:
        return df

    df = df.copy()
    df["Balls"] = pd.to_numeric(df["Balls"], errors="coerce").fillna(0).clip(lower=0, upper=3).astype(int)
    df["Strikes"] = pd.to_numeric(df["Strikes"], errors="coerce").fillna(0).clip(lower=0, upper=2).astype(int)
    throws = df["PitcherThrows"].astype(str).str.upper().str[0]
    stands = df["BatterSide"].astype(str).str.upper().str[0]
    df["PitcherThrows"] = throws.where(throws.isin(["L", "R"]), "R")
    df["BatterSide"] = stands.where(stands.isin(["L", "R"]), "R")
    df["PitcherThrows_enc"] = (df["PitcherThrows"] == "L").astype(int)
    df["BatterSide_enc"] = (df["BatterSide"] == "L").astype(int)
    df["pitch_type_enc"] = df["TaggedPitchType"].map(_PITCH_TYPE_LABELS).fillna(-1).astype(int)

    is_left = df["PitcherThrows"] == "L"
    df["HorzBreakAdj"] = np.where(is_left, -df["HorzBreak"].astype(float), df["HorzBreak"].astype(float))
    df["RelSideAdj"] = np.where(is_left, -df["RelSide"].astype(float), df["RelSide"].astype(float))
    df["release_pos_y"] = 60.5 - df["Extension"].astype(float)

    fb_velo = df[df["TaggedPitchType"].isin(_FB_TYPES)].groupby("Pitcher", observed=False)["RelSpeed"].mean()
    df["VeloDiff"] = df["Pitcher"].map(fb_velo).astype(float) - df["RelSpeed"].astype(float)
    df["speed_x_ivb"] = df["RelSpeed"].astype(float) * df["InducedVertBreak"].astype(float)
    df["total_break"] = np.sqrt(
        df["InducedVertBreak"].astype(float) ** 2 + df["HorzBreakAdj"].astype(float) ** 2
    )
    df["break_balance"] = (
        df["HorzBreakAdj"].astype(float).abs()
        / (df["HorzBreakAdj"].astype(float).abs() + df["InducedVertBreak"].astype(float).abs() + 1e-6)
    )

    if "SpinAxis" in df.columns:
        axis = np.deg2rad(pd.to_numeric(df["SpinAxis"], errors="coerce").fillna(0.0) % 360.0)
        df["axis_sin"] = np.sin(axis)
        df["axis_cos"] = np.cos(axis)
    else:
        df["axis_sin"] = 0.0
        df["axis_cos"] = 1.0

    speed = np.sqrt(
        pd.to_numeric(df["vx0"], errors="coerce").astype(float) ** 2
        + pd.to_numeric(df["vy0"], errors="coerce").astype(float) ** 2
        + pd.to_numeric(df["vz0"], errors="coerce").astype(float) ** 2
    )
    az_adj = pd.to_numeric(df["az0"], errors="coerce").astype(float) + 32.174
    acc = np.column_stack(
        [
            pd.to_numeric(df["ax0"], errors="coerce").astype(float),
            pd.to_numeric(df["ay0"], errors="coerce").astype(float),
            az_adj,
        ]
    )
    vel = np.column_stack(
        [
            pd.to_numeric(df["vx0"], errors="coerce").astype(float),
            pd.to_numeric(df["vy0"], errors="coerce").astype(float),
            pd.to_numeric(df["vz0"], errors="coerce").astype(float),
        ]
    )
    cross = np.cross(acc, vel)
    safe_speed = np.where(np.isfinite(speed) & (speed > 1e-6), speed, np.nan)
    df["physics_speed"] = speed
    df["lift"] = cross[:, 0] / safe_speed
    transverse_x = cross[:, 2] / safe_speed
    transverse_y = cross[:, 1] / safe_speed
    transverse = np.sqrt(np.square(transverse_x) + np.square(transverse_y)) * np.sign(transverse_x)
    hand_sign = np.where(df["PitcherThrows_enc"] == 1, -1.0, 1.0)
    df["transverse_pit"] = transverse * hand_sign

    fb_lift = df[df["TaggedPitchType"].isin(_FB_TYPES)].groupby("Pitcher", observed=False)["lift"].mean()
    fb_transverse = (
        df[df["TaggedPitchType"].isin(_FB_TYPES)]
        .groupby("Pitcher", observed=False)["transverse_pit"]
        .mean()
    )
    df["speed_diff"] = df["Pitcher"].map(fb_velo).astype(float) - df["RelSpeed"].astype(float)
    df["lift_diff"] = df["Pitcher"].map(fb_lift).astype(float) - df["lift"].astype(float)
    df["transverse_pit_diff"] = (
        df["Pitcher"].map(fb_transverse).astype(float) - df["transverse_pit"].astype(float)
    )

    df["VAAResidual"] = 0.0
    if vaa_models and "VertApprAngle" in df.columns:
        for pt, model in vaa_models.items():
            mask = df["TaggedPitchType"] == pt
            if mask.sum() == 0:
                continue
            X_pt = df.loc[mask, _VAA_CONTEXT_FEATURES].astype(float)
            y_pt = pd.to_numeric(df.loc[mask, "VertApprAngle"], errors="coerce")
            valid = X_pt.notna().all(axis=1) & y_pt.notna()
            if not valid.any():
                continue
            pred = model.predict(X_pt.loc[valid])
            resid = y_pt.copy()
            resid.loc[valid] = resid.loc[valid] - pred
            df.loc[mask, "VAAResidual"] = resid.fillna(0.0).values

    df["is_swing"] = df["PitchCall"].isin(_SWING_CALLS).astype(int)
    df["is_take"] = 1 - df["is_swing"]
    df["is_whiff"] = (df["PitchCall"] == "StrikeSwinging").astype(int)
    df["is_called_strike"] = (df["PitchCall"] == "StrikeCalled").astype(int)
    pitch_call_norm = _normalize_string_token(df["PitchCall"])
    play_result_norm = (
        _normalize_string_token(df["PlayResult"]) if "PlayResult" in df.columns else pd.Series("", index=df.index)
    )
    tagged_hit_type_norm = (
        _normalize_string_token(df["TaggedHitType"])
        if "TaggedHitType" in df.columns
        else pd.Series("", index=df.index)
    )
    df["play_result_norm"] = play_result_norm
    df["tagged_hit_type_norm"] = tagged_hit_type_norm
    df["is_ball"] = pitch_call_norm.isin(_BALL_CALLS).astype(int)
    df["is_hbp"] = (pitch_call_norm == "hitbypitch").astype(int)
    df["is_contact"] = ((df["is_swing"] == 1) & (df["is_whiff"] == 0)).astype(int)
    df["is_inplay"] = (pitch_call_norm == "inplay").astype(int)
    df["is_foul"] = ((df["is_contact"] == 1) & (df["is_inplay"] == 0)).astype(int)
    df["is_home_run"] = play_result_norm.isin(_HOME_RUN_RESULTS).astype(int)
    df["is_single"] = (df["is_inplay"] == 1) & play_result_norm.isin({"single", "error"})
    df["is_double"] = (df["is_inplay"] == 1) & (play_result_norm == "double")
    df["is_triple"] = (df["is_inplay"] == 1) & (play_result_norm == "triple")
    df["is_out_inplay"] = (
        (df["is_inplay"] == 1)
        & ~(df["is_single"] | df["is_double"] | df["is_triple"] | (df["is_home_run"] == 1))
    )
    df["is_csw"] = ((df["is_called_strike"] == 1) | (df["is_whiff"] == 1)).astype(int)
    play_damage = play_result_norm.map(_PLAY_RESULT_DAMAGE).fillna(0.0).astype(float) / 4.0
    df["contact_damage"] = np.where(df["is_inplay"] == 1, play_damage, 0.0)
    df["expected_bases"] = np.where(
        df["is_inplay"] == 1,
        play_result_norm.map({"single": 1.0, "error": 1.0, "double": 2.0, "triple": 3.0, "homerun": 4.0})
        .fillna(0.0)
        .astype(float),
        0.0,
    )

    return df


def _fit_weight_model(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from xgboost import XGBClassifier

    sample = df.dropna(subset=["is_csw", "GameID"]).copy()
    sample = sample.dropna(subset=_CLUSTER_FEATURES)
    if len(sample) > _WEIGHT_MODEL_ROWS:
        sample = sample.sample(_WEIGHT_MODEL_ROWS, random_state=42)

    X = sample[_CLUSTER_FEATURES].astype(np.float32)
    y = sample["is_csw"].astype(int)
    groups = sample["GameID"].astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))
    base_params = dict(
        n_estimators=_WEIGHT_BASE_ESTIMATORS,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=24,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=8.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
    )
    tuned_params, tune_meta = _tune_xgb_params(
        frame=sample,
        features=_CLUSTER_FEATURES,
        target="is_csw",
        group_col="GameID",
        model_cls=XGBClassifier,
        base_params=base_params,
        task="binary",
        label="cluster_weight",
        max_rows=_WEIGHT_TUNE_ROWS,
    )
    model = XGBClassifier(**tuned_params)
    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        verbose=False,
    )
    prob = np.clip(model.predict_proba(X.iloc[val_idx])[:, 1], 1e-6, 1 - 1e-6)
    booster = model.get_booster()
    raw_gain = booster.get_score(importance_type="total_gain")
    gains = np.array([float(raw_gain.get(col, 0.0)) for col in _CLUSTER_FEATURES], dtype=float)
    if gains.sum() <= 0:
        gains = np.ones(len(_CLUSTER_FEATURES), dtype=float)
    gains = gains / max(gains.max(), 1e-6)
    raw_gains_dict = {feature: float(g) for feature, g in zip(_CLUSTER_FEATURES, gains)}
    gains = np.clip(gains, 0.05, 5.00)
    weights = {feature: float(weight) for feature, weight in zip(_CLUSTER_FEATURES, gains)}
    metrics = {
        "sample_rows": int(len(sample)),
        "val_logloss": float(log_loss(y.iloc[val_idx], prob, labels=[0, 1])),
        "val_auc": float(roc_auc_score(y.iloc[val_idx], prob)),
        "tuning": tune_meta,
        "raw_gains": raw_gains_dict,
    }
    return weights, metrics


def _fuzzy_cmeans_fit(
    X: np.ndarray,
    n_clusters: int = _N_CLUSTERS,
    m: float = _FCM_M,
    error: float = _FCM_ERROR,
    maxiter: int = _FCM_MAXITER,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(random_state)
    if len(X) < n_clusters:
        raise ValueError("Not enough rows to fit fuzzy c-means")
    centers = X[rng.choice(len(X), size=n_clusters, replace=False)].astype(np.float64, copy=True)
    prev_u: Optional[np.ndarray] = None

    for _ in range(maxiter):
        dist = np.maximum(cdist(X, centers), 1e-8)
        inv = dist ** (-2.0 / (m - 1.0))
        u = inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)
        um = u ** m
        centers_new = (um.T @ X) / np.maximum(um.sum(axis=0)[:, None], 1e-12)
        center_delta = float(np.max(np.abs(centers_new - centers)))
        u_delta = np.inf if prev_u is None else float(np.max(np.abs(u - prev_u)))
        centers = centers_new
        prev_u = u
        if center_delta <= error and u_delta <= error:
            break
    return centers.astype(np.float32), prev_u.astype(np.float32)


def _fuzzy_cmeans_predict(
    X: np.ndarray,
    centers: np.ndarray,
    m: float = _FCM_M,
) -> np.ndarray:
    from scipy.spatial.distance import cdist

    dist = np.maximum(cdist(X, centers), 1e-8)
    inv = dist ** (-2.0 / (m - 1.0))
    return (inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _cluster_name_from_summary(summary: Dict[str, object]) -> str:
    dominant_pt = str(summary.get("dominant_pitch_type", "Unknown"))
    if dominant_pt in {"Changeup", "Splitter"}:
        return "offspeed"
    if dominant_pt in {"Curveball", "Knuckle Curve"}:
        return "curveball"
    if dominant_pt == "Cutter":
        return "cutter"
    if dominant_pt == "Sinker":
        return "sinker"
    if dominant_pt == "Fastball":
        return "fastball"
    if dominant_pt in {"Slider", "Sweeper"}:
        return "slider"
    return "offspeed"


def _cluster_assignment_scores(summary: Dict[str, object]) -> Dict[str, float]:
    shares = summary.get("pitch_type_share", {})
    fastball_share = float(shares.get("Fastball", 0.0))
    sinker_share = float(shares.get("Sinker", 0.0))
    cutter_share = float(shares.get("Cutter", 0.0))
    slider_share = float(shares.get("Slider", 0.0)) + float(shares.get("Sweeper", 0.0))
    curve_share = float(shares.get("Curveball", 0.0) + shares.get("Knuckle Curve", 0.0))
    offspeed_share = float(shares.get("Changeup", 0.0) + shares.get("Splitter", 0.0))
    return {
        "sinker": 4.0 * sinker_share,
        "fastball": 4.0 * fastball_share,
        "slider": 4.0 * slider_share,
        "offspeed": 4.0 * offspeed_share,
        "curveball": 4.0 * curve_share,
        "cutter": 4.0 * cutter_share,
    }


def _summarize_fuzzy_clusters(
    train: pd.DataFrame,
    probs: np.ndarray,
) -> Tuple[List[Dict[str, object]], Dict[int, str], int]:
    from scipy.optimize import linear_sum_assignment

    pt_dummies = pd.get_dummies(train["TaggedPitchType"])
    summaries: List[Dict[str, object]] = []
    heuristic_map: Dict[int, str] = {}
    score_matrix = np.zeros((probs.shape[1], len(_CLUSTER_NAMES)), dtype=float)

    for idx in range(probs.shape[1]):
        w = probs[:, idx].astype(float)
        w_sum = float(w.sum()) or 1.0
        weighted_pt = (pt_dummies.mul(w, axis=0).sum(axis=0) / w_sum).sort_values(ascending=False)
        pitch_type_share = {str(k): float(v) for k, v in weighted_pt.items()}
        dominant_pt = str(weighted_pt.index[0]) if len(weighted_pt) else "Unknown"
        summary = {
            "cluster_index": idx,
            "dominant_pitch_type": dominant_pt,
            "pitch_type_share": pitch_type_share,
            "mean_speed": float(np.average(train["RelSpeed"], weights=w)),
            "mean_ivb": float(np.average(train["InducedVertBreak"], weights=w)),
            "mean_hb_adj": float(np.average(train["HorzBreakAdj"], weights=w)),
            "mean_break_balance": float(np.average(train["break_balance"], weights=w)),
            "mean_rel_height": float(np.average(train["RelHeight"], weights=w)),
            "mean_transverse_pit": float(np.average(train["transverse_pit"], weights=w)),
        }
        heuristic_name = _cluster_name_from_summary(summary)
        heuristic_map[idx] = heuristic_name
        summaries.append(summary)
        scores = _cluster_assignment_scores(summary)
        score_matrix[idx] = np.array([scores[name] for name in _CLUSTER_NAMES], dtype=float)

    unique_heuristics = len(set(heuristic_map.values()))
    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    fixed_map = {int(r): _CLUSTER_NAMES[int(c)] for r, c in zip(row_ind, col_ind)}
    for summary in summaries:
        summary["heuristic_name"] = heuristic_map[int(summary["cluster_index"])]
        summary["cluster_name"] = fixed_map[int(summary["cluster_index"])]
    return summaries, fixed_map, unique_heuristics


def _fit_cluster_model(
    df: pd.DataFrame,
    feature_weights: Dict[str, float],
) -> Tuple[object, object, np.ndarray, List[str], pd.DataFrame, List[Dict[str, object]]]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cluster_cols = list(_CLUSTER_NAMES)
    valid = df[_CLUSTER_FEATURES].notna().all(axis=1)
    train = df.loc[valid].copy()
    X = train[_CLUSTER_FEATURES].astype(np.float32)
    scaler = StandardScaler()
    if len(train) > _CLUSTER_FIT_ROWS:
        fit_idx = train.sample(_CLUSTER_FIT_ROWS, random_state=42).index
        scaler.fit(X.loc[fit_idx])
    else:
        scaler.fit(X)
    X_scaled = scaler.transform(X)
    weight_vec = np.array([feature_weights.get(col, 1.0) for col in _CLUSTER_FEATURES], dtype=float)
    X_weighted = X_scaled * weight_vec
    pca = PCA(n_components=6, random_state=42)
    if len(train) > _CLUSTER_FIT_ROWS:
        take = np.random.default_rng(42).choice(len(train), size=_CLUSTER_FIT_ROWS, replace=False)
        pca.fit(X_weighted[take])
    else:
        pca.fit(X_weighted)
    X_pca = pca.transform(X_weighted).astype(np.float32)

    best_centers: Optional[np.ndarray] = None
    best_probs: Optional[np.ndarray] = None
    best_summaries: Optional[List[Dict[str, object]]] = None
    best_map: Optional[Dict[int, str]] = None
    best_unique = -1
    for seed in range(42, 54):
        fit_sample = X_pca
        if len(X_pca) > 100_000:
            take = np.random.default_rng(seed).choice(len(X_pca), size=100_000, replace=False)
            fit_sample = X_pca[take]
        centers, _ = _fuzzy_cmeans_fit(fit_sample, random_state=seed)
        probs = _fuzzy_cmeans_predict(X_pca, centers)
        summaries, name_map, unique_count = _summarize_fuzzy_clusters(train, probs)
        if unique_count > best_unique:
            best_centers = centers
            best_probs = probs
            best_summaries = summaries
            best_map = name_map
            best_unique = unique_count
        if unique_count == _N_CLUSTERS:
            break

    if best_centers is None or best_probs is None or best_summaries is None or best_map is None:
        raise RuntimeError("Failed to fit fuzzy archetypes")

    ordered_idx = [next(idx for idx, name in best_map.items() if name == cluster_name) for cluster_name in cluster_cols]
    ordered_probs = best_probs[:, ordered_idx]
    for col_idx, col in enumerate(cluster_cols):
        df[col] = np.nan
        df.loc[valid, col] = ordered_probs[:, col_idx]

    renamed_summaries: List[Dict[str, object]] = []
    for summary in best_summaries:
        summary = dict(summary)
        summary["cluster"] = best_map[int(summary.pop("cluster_index"))]
        renamed_summaries.append(summary)
    renamed_summaries.sort(key=lambda row: cluster_cols.index(str(row["cluster"])))
    cluster_model = {
        "centers": best_centers[ordered_idx],
        "m": _FCM_M,
        "taxonomy": cluster_cols,
        "unique_heuristics": best_unique,
    }
    return scaler, {"pca": pca, **cluster_model}, weight_vec, cluster_cols, df, renamed_summaries


def _build_context_priors(
    df: pd.DataFrame,
    cluster_cols: Sequence[str],
) -> Dict[Tuple[str, str, str], np.ndarray]:
    from joblib import Parallel, delayed
    from sklearn.neighbors import KernelDensity

    working = df.dropna(subset=["PlateLocSide", "PlateLocHeight", "Balls", "Strikes"]).copy()
    if working.empty:
        return {}

    kde_lookup: Dict[Tuple[str, str, int, int, str], np.ndarray] = {}
    count_freqs: Dict[Tuple[str, str, str], np.ndarray] = {}

    def _fit_kde(frame: pd.DataFrame, cluster_col: str, seed: int) -> Optional[np.ndarray]:
        weights = frame[cluster_col].astype(float).to_numpy()
        coords = frame[["PlateLocSide", "PlateLocHeight"]].astype(float).to_numpy()
        valid = (
            np.isfinite(weights)
            & (weights > 0)
            & np.isfinite(coords[:, 0])
            & np.isfinite(coords[:, 1])
        )
        if valid.sum() < 20 or float(weights[valid].sum()) < _MIN_CONTEXT_WEIGHT:
            return None
        coords = coords[valid]
        weights = weights[valid]
        weights = weights / np.maximum(weights.sum(), 1e-12)
        if len(coords) > _KDE_SAMPLE_ROWS:
            sample_n = int(_KDE_SAMPLE_ROWS)
            rng = np.random.default_rng(seed)
            sample_idx = rng.choice(len(coords), size=sample_n, replace=False, p=weights)
            coords = coords[sample_idx]
            weights = weights[sample_idx]
            weights = weights / np.maximum(weights.sum(), 1e-12)
        kde = KernelDensity(bandwidth=_KDE_BANDWIDTH)
        try:
            kde.fit(coords, sample_weight=weights)
        except TypeError:
            sample_n = int(min(_KDE_SAMPLE_ROWS, max(5_000, len(coords))))
            rng = np.random.default_rng(seed)
            sample_idx = rng.choice(len(coords), size=sample_n, replace=True, p=weights)
            kde.fit(coords[sample_idx])
        return kde.score_samples(_GRID).astype(np.float32)

    def _build_count_freq(frame: pd.DataFrame, cluster_col: str) -> np.ndarray:
        rows = []
        for balls, strikes in _COUNTS:
            mask = (frame["Balls"] == balls) & (frame["Strikes"] == strikes)
            if not mask.any():
                rows.append(0.0)
            else:
                rows.append(float(frame.loc[mask, cluster_col].astype(float).mean()))
        arr = np.asarray(rows, dtype=float)
        if arr.sum() <= 0:
            arr = np.ones(len(_COUNTS), dtype=float)
        return (arr / arr.sum()).astype(np.float32)

    grouped_exact = list(
        working.groupby(["BatterSide", "PitcherThrows", "Balls", "Strikes"], observed=False, sort=False)
    )

    def _kde_job(key_vals: Tuple[object, object, object, object], frame: pd.DataFrame, cluster_col: str):
        bat_side, pitch_hand, balls, strikes = key_vals
        key = (str(bat_side), str(pitch_hand), int(balls), int(strikes), cluster_col)
        seed = abs(hash(key)) % (2**32)
        logpdf = _fit_kde(frame, cluster_col, seed=seed)
        return key, logpdf

    kde_results = Parallel(n_jobs=_KDE_JOBS, prefer="threads")(
        delayed(_kde_job)(key_vals, frame, cluster_col)
        for key_vals, frame in grouped_exact
        for cluster_col in cluster_cols
    )
    for key, logpdf in kde_results:
        if logpdf is not None:
            kde_lookup[key] = logpdf

    grouped_counts = working.groupby(["PitcherThrows", "BatterSide"], observed=False, sort=False)
    for (pitch_hand, bat_side), frame in grouped_counts:
        for cluster_col in cluster_cols:
            count_freqs[(str(pitch_hand), str(bat_side), cluster_col)] = _build_count_freq(frame, cluster_col)

    def _lookup_count_freq(pitch_hand: str, bat_side: str, cluster_col: str) -> np.ndarray:
        key = (pitch_hand, bat_side, cluster_col)
        return count_freqs.get(key, np.full(len(_COUNTS), 1.0 / len(_COUNTS), dtype=np.float32))

    context_distributions: Dict[Tuple[str, str, str], np.ndarray] = {}
    for bat_side in ("R", "L"):
        for pitch_hand in ("R", "L"):
            for cluster_col in cluster_cols:
                flat_parts: List[np.ndarray] = []
                count_weights = _lookup_count_freq(pitch_hand, bat_side, cluster_col)
                for idx, (balls, strikes) in enumerate(_COUNTS):
                    logpdf = kde_lookup.get((bat_side, pitch_hand, balls, strikes, cluster_col))
                    if logpdf is None or not np.isfinite(logpdf).any():
                        dens = np.zeros(_GRID_LEN, dtype=np.float32)
                    else:
                        dens = np.exp(logpdf - np.max(logpdf))
                        dens = dens / np.maximum(dens.sum(), 1e-12)
                    flat_parts.append(dens * float(count_weights[idx]))
                flat = np.concatenate(flat_parts).astype(np.float32)
                if flat.sum() <= 0:
                    flat = np.full(_CONTEXT_LEN, 1.0 / _CONTEXT_LEN, dtype=np.float32)
                else:
                    flat = flat / flat.sum()
                context_distributions[(bat_side, pitch_hand, cluster_col)] = flat

    return context_distributions


def _build_d1_run_value_table(parquet_path: str) -> pd.DataFrame:
    import duckdb

    from analytics.historical_calibration import fallback_linear_weights
    from analytics.run_value import _load_count_rv

    lw = fallback_linear_weights()
    fallback_count_rv = _load_count_rv()
    pq = parquet_path.replace("'", "''")
    con = duckdb.connect(":memory:")
    rows = con.execute(
        f"""
        WITH pa_pitches AS (
            SELECT
                CAST(GameID AS VARCHAR) || '_' || CAST(Inning AS VARCHAR) || '_' ||
                CAST(PAofInning AS VARCHAR) || '_' || COALESCE(CAST(Batter AS VARCHAR), '') AS pa_id,
                CAST(Balls AS INTEGER) AS balls,
                CAST(Strikes AS INTEGER) AS strikes,
                PitchCall,
                KorBB,
                PlayResult
            FROM read_parquet('{pq}')
            WHERE Level = '{_D1_LEVEL}'
              AND PitchCall IS NOT NULL
              AND Balls IS NOT NULL
              AND Strikes IS NOT NULL
              AND GameID IS NOT NULL
              AND Inning IS NOT NULL
              AND PAofInning IS NOT NULL
        ),
        pa_terminal AS (
            SELECT
                pa_id,
                CASE
                    WHEN MAX(CASE WHEN KorBB IN ('Strikeout', 'K') THEN 1 ELSE 0 END) = 1 THEN 'Strikeout'
                    WHEN MAX(CASE WHEN KorBB IN ('Walk', 'BB') THEN 1 ELSE 0 END) = 1 THEN 'Walk'
                    WHEN MAX(CASE WHEN PitchCall = 'HitByPitch' THEN 1 ELSE 0 END) = 1 THEN 'HitByPitch'
                    WHEN MAX(CASE WHEN PitchCall = 'InPlay' AND PlayResult = 'Single' THEN 1 ELSE 0 END) = 1 THEN 'Single'
                    WHEN MAX(CASE WHEN PitchCall = 'InPlay' AND PlayResult = 'Double' THEN 1 ELSE 0 END) = 1 THEN 'Double'
                    WHEN MAX(CASE WHEN PitchCall = 'InPlay' AND PlayResult = 'Triple' THEN 1 ELSE 0 END) = 1 THEN 'Triple'
                    WHEN MAX(CASE WHEN PitchCall = 'InPlay' AND PlayResult = 'HomeRun' THEN 1 ELSE 0 END) = 1 THEN 'HomeRun'
                    WHEN MAX(CASE WHEN PitchCall = 'InPlay' AND PlayResult = 'Error' THEN 1 ELSE 0 END) = 1 THEN 'Error'
                    WHEN MAX(CASE WHEN PitchCall = 'InPlay' AND PlayResult IN ('Out', 'FieldersChoice', 'Sacrifice') THEN 1 ELSE 0 END) = 1 THEN 'Out'
                    ELSE NULL
                END AS outcome
            FROM pa_pitches
            GROUP BY pa_id
        )
        SELECT
            p.balls,
            p.strikes,
            AVG(
                CASE t.outcome
                    WHEN 'Strikeout' THEN {lw['out_w']:.6f}
                    WHEN 'Out' THEN {lw['out_w']:.6f}
                    WHEN 'Walk' THEN {lw['bb_w']:.6f}
                    WHEN 'HitByPitch' THEN {lw['hbp_w']:.6f}
                    WHEN 'Error' THEN {lw['single_w']:.6f}
                    WHEN 'Single' THEN {lw['single_w']:.6f}
                    WHEN 'Double' THEN {lw['double_w']:.6f}
                    WHEN 'Triple' THEN {lw['triple_w']:.6f}
                    WHEN 'HomeRun' THEN {lw['hr_w']:.6f}
                    ELSE NULL
                END
            ) AS count_rv
        FROM pa_pitches p
        JOIN pa_terminal t USING (pa_id)
        WHERE t.outcome IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchall()
    con.close()

    count_rv = {
        f"{int(b)}-{int(s)}": float(rv)
        for b, s, rv in rows
        if rv is not None
    }
    table_rows: List[Dict[str, float]] = []
    for balls, strikes in _COUNTS:
        key = f"{balls}-{strikes}"
        current = float(count_rv.get(key, fallback_count_rv.get(key, 0.28)))
        next_ball_key = f"{min(balls + 1, 3)}-{strikes}"
        next_strike_key = f"{balls}-{min(strikes + 1, 2)}"
        rv_ball_terminal = (
            float(lw["bb_w"])
            if balls >= 3
            else float(count_rv.get(next_ball_key, fallback_count_rv.get(next_ball_key, current)))
        )
        rv_strike_terminal = (
            float(lw["out_w"])
            if strikes >= 2
            else float(count_rv.get(next_strike_key, fallback_count_rv.get(next_strike_key, current)))
        )
        rv_foul_terminal = (
            current
            if strikes >= 2
            else float(count_rv.get(next_strike_key, fallback_count_rv.get(next_strike_key, current)))
        )
        table_rows.append(
            {
                "balls": balls,
                "strikes": strikes,
                "rv_ball": rv_ball_terminal - current,
                "rv_strike": rv_strike_terminal - current,
                "rv_foul": rv_foul_terminal - current,
                "rv_hbp": float(lw["hbp_w"]) - current,
                "rv_single": float(lw["single_w"]) - current,
                "rv_double": float(lw["double_w"]) - current,
                "rv_triple": float(lw["triple_w"]) - current,
                "rv_home_run": float(lw["hr_w"]) - current,
                "rv_out": float(lw["out_w"]) - current,
            }
        )
    return pd.DataFrame(table_rows)


def _run_value_lookup(run_value_table: pd.DataFrame) -> Dict[str, np.ndarray]:
    lookup: Dict[str, np.ndarray] = {}
    for col in [
        "rv_ball",
        "rv_strike",
        "rv_foul",
        "rv_hbp",
        "rv_single",
        "rv_double",
        "rv_triple",
        "rv_home_run",
        "rv_out",
    ]:
        arr = np.zeros((4, 3), dtype=np.float32)
        for row in run_value_table.itertuples(index=False):
            arr[int(row.balls), int(row.strikes)] = float(getattr(row, col))
        lookup[col] = arr
    return lookup


def _predict_binary_prob(model: Optional[object], X: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.zeros(len(X), dtype=np.float32)
    prob = model.predict_proba(X)[:, 1]
    return np.clip(prob.astype(np.float32), 0.0, 1.0)


def _predict_terminal_probabilities(X: pd.DataFrame, event_models: Dict[str, object]) -> Dict[str, np.ndarray]:
    swing = _predict_binary_prob(event_models.get("swing"), X)
    whiff_given_swing = _predict_binary_prob(event_models.get("whiff_given_swing"), X)
    called_strike_given_take = _predict_binary_prob(event_models.get("called_strike_given_take"), X)
    ball_given_take = _predict_binary_prob(event_models.get("ball_given_take"), X)
    hbp_given_take = _predict_binary_prob(event_models.get("hbp_given_take"), X)
    inplay_given_contact = _predict_binary_prob(event_models.get("inplay_given_contact"), X)
    single_given_inplay = _predict_binary_prob(event_models.get("single_given_inplay"), X)
    double_given_inplay = _predict_binary_prob(event_models.get("double_given_inplay"), X)
    triple_given_inplay = _predict_binary_prob(event_models.get("triple_given_inplay"), X)
    home_run_given_inplay = _predict_binary_prob(event_models.get("home_run_given_inplay"), X)

    take = np.clip(1.0 - swing, 0.0, 1.0)
    take_probs = np.column_stack([called_strike_given_take, ball_given_take, hbp_given_take]).astype(np.float32)
    take_sum = take_probs.sum(axis=1, keepdims=True)
    take_scale = np.where(take_sum > 1.0, take_sum, 1.0)
    take_probs = take_probs / np.maximum(take_scale, 1e-12)

    whiff = swing * whiff_given_swing
    contact = swing * np.clip(1.0 - whiff_given_swing, 0.0, 1.0)
    inplay = contact * inplay_given_contact
    foul = contact * np.clip(1.0 - inplay_given_contact, 0.0, 1.0)

    hit_probs = np.column_stack(
        [
            single_given_inplay,
            double_given_inplay,
            triple_given_inplay,
            home_run_given_inplay,
        ]
    ).astype(np.float32)
    hit_sum = hit_probs.sum(axis=1, keepdims=True)
    hit_scale = np.where(hit_sum > 1.0, hit_sum, 1.0)
    hit_probs = hit_probs / np.maximum(hit_scale, 1e-12)
    inplay_out = inplay * np.clip(1.0 - hit_probs.sum(axis=1), 0.0, 1.0)

    x_single = inplay * hit_probs[:, 0]
    x_double = inplay * hit_probs[:, 1]
    x_triple = inplay * hit_probs[:, 2]
    x_home_run = inplay * hit_probs[:, 3]
    damage = (x_single + 2.0 * x_double + 3.0 * x_triple + 4.0 * x_home_run) / 4.0
    return {
        "callstr": take * take_probs[:, 0],
        "ball": take * take_probs[:, 1],
        "hbp": take * take_probs[:, 2],
        "swstr": whiff,
        "foul": foul,
        "out": inplay_out,
        "single": x_single,
        "double": x_double,
        "triple": x_triple,
        "home_run": x_home_run,
        "damage": damage,
    }


def _expected_run_value(
    terminal_probs: Dict[str, np.ndarray],
    rv_lookup: Dict[str, np.ndarray],
    balls: np.ndarray,
    strikes: np.ndarray,
) -> np.ndarray:
    return (
        (terminal_probs["callstr"] + terminal_probs["swstr"]) * rv_lookup["rv_strike"][balls, strikes]
        + terminal_probs["ball"] * rv_lookup["rv_ball"][balls, strikes]
        + terminal_probs["hbp"] * rv_lookup["rv_hbp"][balls, strikes]
        + terminal_probs["foul"] * rv_lookup["rv_foul"][balls, strikes]
        + terminal_probs["out"] * rv_lookup["rv_out"][balls, strikes]
        + terminal_probs["single"] * rv_lookup["rv_single"][balls, strikes]
        + terminal_probs["double"] * rv_lookup["rv_double"][balls, strikes]
        + terminal_probs["triple"] * rv_lookup["rv_triple"][balls, strikes]
        + terminal_probs["home_run"] * rv_lookup["rv_home_run"][balls, strikes]
    ).astype(np.float32)


def _fit_event_models(
    df: pd.DataFrame,
    cluster_cols: Sequence[str],
) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]]]:
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from xgboost import XGBClassifier

    del cluster_cols
    event_features = list(_EVENT_BASE_FEATURES) + list(_EVENT_CONTEXT_FEATURES)
    full = df.dropna(subset=["GameID"] + event_features).copy()
    X = full[event_features].astype(np.float32)
    groups = full["GameID"].astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, full["is_swing"], groups=groups))

    metrics: Dict[str, Dict[str, float]] = {}
    models: Dict[str, object] = {}
    clf_base_params = dict(
        n_estimators=_EVENT_BASE_ESTIMATORS,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=24,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=8.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
    )
    clf_params, clf_tune_meta = _tune_xgb_params(
        frame=full,
        features=event_features,
        target="is_swing",
        group_col="GameID",
        model_cls=XGBClassifier,
        base_params=clf_base_params,
        task="binary",
        label="event_binary",
        max_rows=_EVENT_TUNE_ROWS,
    )
    metrics["event_binary_tuning"] = clf_tune_meta

    def _fit_binary(name: str, frame: pd.DataFrame, target: str) -> None:
        sub = frame.dropna(subset=event_features).copy()
        if sub.empty or sub[target].nunique() < 2:
            return
        sub = _sample_binary_frame(sub, target=target, max_rows=_EVENT_MODEL_FIT_ROWS + _EVENT_MODEL_VAL_ROWS)
        X_sub = sub[event_features].astype(np.float32)
        sub_groups = sub["GameID"].astype(str)
        train_mask = sub_groups.isin(groups.iloc[train_idx])
        val_mask = sub_groups.isin(groups.iloc[val_idx])
        train_frame = sub.loc[train_mask].copy()
        val_frame = sub.loc[val_mask].copy()
        train_frame = _sample_binary_frame(train_frame, target=target, max_rows=_EVENT_MODEL_FIT_ROWS)
        val_frame = _sample_binary_frame(val_frame, target=target, max_rows=_EVENT_MODEL_VAL_ROWS)
        X_tr = train_frame[event_features].astype(np.float32)
        y_tr = train_frame[target].astype(int)
        X_v = val_frame[event_features].astype(np.float32)
        y_v = val_frame[target].astype(int)
        if len(X_tr) == 0 or len(X_v) == 0 or y_tr.nunique() < 2 or y_v.nunique() < 2:
            return
        print(f"    Fitting {name} ({len(X_tr):,} train / {len(X_v):,} val)")
        clf = XGBClassifier(**clf_params)
        clf.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        prob = np.clip(clf.predict_proba(X_v)[:, 1], 1e-6, 1.0 - 1e-6)
        metrics[name] = {
            "val_logloss": float(log_loss(y_v, prob, labels=[0, 1])),
            "val_auc": float(roc_auc_score(y_v, prob)),
        }
        models[name] = clf

    _fit_binary("swing", full, "is_swing")
    _fit_binary("whiff_given_swing", full[full["is_swing"] == 1], "is_whiff")
    _fit_binary("called_strike_given_take", full[full["is_take"] == 1], "is_called_strike")
    _fit_binary("ball_given_take", full[full["is_take"] == 1], "is_ball")
    _fit_binary("hbp_given_take", full[full["is_take"] == 1], "is_hbp")
    _fit_binary("inplay_given_contact", full[full["is_contact"] == 1], "is_inplay")
    _fit_binary("single_given_inplay", full[full["is_inplay"] == 1], "is_single")
    _fit_binary("double_given_inplay", full[full["is_inplay"] == 1], "is_double")
    _fit_binary("triple_given_inplay", full[full["is_inplay"] == 1], "is_triple")
    _fit_binary("home_run_given_inplay", full[full["is_inplay"] == 1], "is_home_run")
    models["event_features"] = event_features
    return models, metrics


def _build_event_matrix(
    batch: pd.DataFrame,
    stand: str,
) -> pd.DataFrame:
    rows = len(batch)
    data = {
        col: np.repeat(batch[col].astype(float).to_numpy(), _CONTEXT_LEN)
        for col in _EVENT_BASE_FEATURES
    }
    data["PlateLocSide"] = np.tile(_CONTEXT_PLATE_X, rows)
    data["PlateLocHeight"] = np.tile(_CONTEXT_PLATE_Z, rows)
    data["BatterSide_enc"] = np.full(rows * _CONTEXT_LEN, 1.0 if stand == "L" else 0.0, dtype=float)
    data["Balls"] = np.tile(_CONTEXT_BALLS, rows)
    data["Strikes"] = np.tile(_CONTEXT_STRIKES, rows)
    return pd.DataFrame(data, columns=list(_EVENT_BASE_FEATURES) + list(_EVENT_CONTEXT_FEATURES)).astype(np.float32)


def _mix_context_distribution(
    batch: pd.DataFrame,
    cluster_cols: Sequence[str],
    context_distributions: Dict[Tuple[str, str, str], np.ndarray],
    stand: str,
) -> np.ndarray:
    mixed = np.zeros((len(batch), _CONTEXT_LEN), dtype=np.float32)
    pitch_hands = batch["PitcherThrows"].astype(str).to_numpy()
    for pitch_hand in ("R", "L"):
        mask = np.flatnonzero(pitch_hands == pitch_hand)
        if len(mask) == 0:
            continue
        hand_batch = batch.iloc[mask]
        hand_mix = np.zeros((len(mask), _CONTEXT_LEN), dtype=np.float32)
        for cluster_col in cluster_cols:
            dist = context_distributions.get((stand, pitch_hand, cluster_col))
            if dist is None:
                continue
            membership = hand_batch[cluster_col].astype(np.float32).to_numpy()[:, None]
            hand_mix += membership * dist[None, :]
        hand_sum = hand_mix.sum(axis=1, keepdims=True)
        hand_mix = np.where(hand_sum > 0, hand_mix / np.maximum(hand_sum, 1e-12), 1.0 / _CONTEXT_LEN)
        mixed[mask] = hand_mix
    return mixed


def _simulate_pitchsim_targets(
    sim_df: pd.DataFrame,
    cluster_cols: Sequence[str],
    context_distributions: Dict[Tuple[str, str, str], np.ndarray],
    event_models: Dict[str, object],
    run_value_table: pd.DataFrame,
    batch_size: int = 128,
) -> pd.DataFrame:
    event_features = event_models["event_features"]
    out = sim_df.copy()
    rv_lookup = _run_value_lookup(run_value_table)
    context_balls = _CONTEXT_BALLS.astype(int)
    context_strikes = _CONTEXT_STRIKES.astype(int)

    for stand in ("R", "L"):
        side_suffix = "vsR" if stand == "R" else "vsL"
        rv_vals = np.zeros(len(sim_df), dtype=np.float32)
        whiff_vals = np.zeros(len(sim_df), dtype=np.float32)
        cs_vals = np.zeros(len(sim_df), dtype=np.float32)
        ball_vals = np.zeros(len(sim_df), dtype=np.float32)
        hr_vals = np.zeros(len(sim_df), dtype=np.float32)
        damage_vals = np.zeros(len(sim_df), dtype=np.float32)

        for start in range(0, len(sim_df), batch_size):
            batch_idx = np.arange(start, min(start + batch_size, len(sim_df)))
            batch = sim_df.iloc[batch_idx].copy()
            context_weights = _mix_context_distribution(batch, cluster_cols, context_distributions, stand)
            X_ctx = _build_event_matrix(batch, stand=stand)[event_features]
            terminal = _predict_terminal_probabilities(X_ctx, event_models)
            terminal = {k: v.reshape(len(batch), _CONTEXT_LEN) for k, v in terminal.items()}
            rv_context = _expected_run_value(
                terminal_probs=terminal,
                rv_lookup=rv_lookup,
                balls=np.tile(context_balls, (len(batch), 1)),
                strikes=np.tile(context_strikes, (len(batch), 1)),
            )
            rv_vals[batch_idx] = np.sum(rv_context * context_weights, axis=1)
            whiff_vals[batch_idx] = np.sum(terminal["swstr"] * context_weights, axis=1)
            cs_vals[batch_idx] = np.sum(terminal["callstr"] * context_weights, axis=1)
            ball_vals[batch_idx] = np.sum(terminal["ball"] * context_weights, axis=1)
            hr_vals[batch_idx] = np.sum(terminal["home_run"] * context_weights, axis=1)
            damage_vals[batch_idx] = np.sum(terminal["damage"] * context_weights, axis=1)

        out[f"StuffRV_{side_suffix}"] = rv_vals
        out[f"StuffWhiff_{side_suffix}"] = whiff_vals
        out[f"StuffCalledStrike_{side_suffix}"] = cs_vals
        out[f"StuffBall_{side_suffix}"] = ball_vals
        out[f"StuffHomeRun_{side_suffix}"] = hr_vals
        out[f"StuffDamage_{side_suffix}"] = damage_vals

    out["StuffRV"] = 0.5 * (out["StuffRV_vsR"] + out["StuffRV_vsL"])
    out["StuffWhiff"] = 0.5 * (out["StuffWhiff_vsR"] + out["StuffWhiff_vsL"])
    out["StuffCalledStrike"] = 0.5 * (out["StuffCalledStrike_vsR"] + out["StuffCalledStrike_vsL"])
    out["StuffBall"] = 0.5 * (out["StuffBall_vsR"] + out["StuffBall_vsL"])
    out["StuffHomeRun"] = 0.5 * (out["StuffHomeRun_vsR"] + out["StuffHomeRun_vsL"])
    out["StuffDamage"] = 0.5 * (out["StuffDamage_vsR"] + out["StuffDamage_vsL"])
    out["StuffFIPRV"] = _compose_fip_raw(
        whiff=out["StuffWhiff"],
        called_strike=out["StuffCalledStrike"],
        ball=out["StuffBall"],
        home_run=out["StuffHomeRun"],
        damage=out["StuffDamage"],
    )
    return out


def _fit_distilled_models(sim_df: pd.DataFrame) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]]]:
    from sklearn.model_selection import GroupShuffleSplit
    from xgboost import XGBRegressor

    targets = [
        "StuffRV_vsR",
        "StuffRV_vsL",
        "StuffWhiff_vsR",
        "StuffWhiff_vsL",
        "StuffCalledStrike_vsR",
        "StuffCalledStrike_vsL",
        "StuffBall_vsR",
        "StuffBall_vsL",
        "StuffHomeRun_vsR",
        "StuffHomeRun_vsL",
        "StuffDamage_vsR",
        "StuffDamage_vsL",
    ]
    frame = sim_df.dropna(subset=_DISTILL_FEATURES + targets + ["Pitcher"]).copy()
    if len(frame) > _MAX_DISTILL_ROWS:
        frame = (
            frame.groupby("TaggedPitchType", group_keys=False, observed=False)
            .apply(lambda g: g.sample(min(len(g), max(500, int(_MAX_DISTILL_ROWS * len(g) / len(frame)))), random_state=42))
            .reset_index(drop=True)
        )

    X = frame[_DISTILL_FEATURES].astype(np.float32)
    groups = frame["Pitcher"].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, frame[targets[0]], groups=groups))

    models: Dict[str, object] = {"features": list(_DISTILL_FEATURES)}
    metrics: Dict[str, Dict[str, float]] = {}
    base_params = dict(
        n_estimators=_DISTILL_BASE_ESTIMATORS,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=16,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=6.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
    )
    params, tune_meta = _tune_xgb_params(
        frame=frame,
        features=_DISTILL_FEATURES,
        target=targets[0],
        group_col="Pitcher",
        model_cls=XGBRegressor,
        base_params=base_params,
        task="regression",
        label="distill",
        max_rows=_DISTILL_TUNE_ROWS,
    )
    metrics["distill_tuning"] = tune_meta
    for target in targets:
        model = XGBRegressor(**params)
        y = frame[target].astype(np.float32)
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=False,
        )
        pred = model.predict(X.iloc[val_idx])
        truth = y.iloc[val_idx].to_numpy(dtype=float)
        metrics[target] = {
            "val_rmse": float(np.sqrt(np.mean(np.square(truth - pred)))),
            "val_corr": _safe_corr(truth, pred),
        }
        models[target] = model
    return models, metrics


def _population_stats_from_values(
    pitch_types: pd.Series,
    values: np.ndarray,
) -> Dict[str, Tuple[float, float]]:
    pt_stats: Dict[str, Tuple[float, float]] = {}
    for pt in pitch_types.unique():
        mask = pitch_types == pt
        vals = values[mask.to_numpy()]
        if len(vals) < 100:
            continue
        pt_stats[str(pt)] = (float(np.mean(vals)), float(np.std(vals)))
    if values.size:
        pt_stats["__global__"] = (float(np.mean(values)), float(np.std(values)))
    return pt_stats


def _compute_population_stats(
    df: pd.DataFrame,
    distilled_models: Dict[str, object],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    valid = df[_DISTILL_FEATURES].notna().all(axis=1) & df["TaggedPitchType"].notna()
    pred_frame = df.loc[valid, _DISTILL_FEATURES].astype(np.float32)
    pitch_types = df.loc[valid, "TaggedPitchType"]

    rv_r = distilled_models["StuffRV_vsR"].predict(pred_frame)
    rv_l = distilled_models["StuffRV_vsL"].predict(pred_frame)
    whiff_r = distilled_models["StuffWhiff_vsR"].predict(pred_frame)
    whiff_l = distilled_models["StuffWhiff_vsL"].predict(pred_frame)
    cs_r = distilled_models["StuffCalledStrike_vsR"].predict(pred_frame)
    cs_l = distilled_models["StuffCalledStrike_vsL"].predict(pred_frame)
    ball_r = distilled_models["StuffBall_vsR"].predict(pred_frame)
    ball_l = distilled_models["StuffBall_vsL"].predict(pred_frame)
    hr_r = distilled_models["StuffHomeRun_vsR"].predict(pred_frame)
    hr_l = distilled_models["StuffHomeRun_vsL"].predict(pred_frame)
    damage_r = distilled_models["StuffDamage_vsR"].predict(pred_frame)
    damage_l = distilled_models["StuffDamage_vsL"].predict(pred_frame)

    overall = 0.5 * (rv_r + rv_l)
    fip_raw = _compose_fip_raw(
        whiff=0.5 * (whiff_r + whiff_l),
        called_strike=0.5 * (cs_r + cs_l),
        ball=0.5 * (ball_r + ball_l),
        home_run=0.5 * (hr_r + hr_l),
        damage=0.5 * (damage_r + damage_l),
    )
    return _population_stats_from_values(pitch_types, overall), _population_stats_from_values(pitch_types, fip_raw)


def _compute_location_rv_stats(
    df: pd.DataFrame,
    distilled_models: Dict[str, object],
    event_models: Dict[str, object],
    run_value_table: Dict[str, np.ndarray],
) -> Dict[str, Tuple[float, float]]:
    """Compute LocationRV population stats for Command+ z-scoring.

    LocationRV = FullRV(actual location+count) - StuffRV(grid-averaged).
    Aggregated at pitcher×pitch-type level, then (mean, std) per pitch type.
    """
    event_features = event_models.get("event_features")
    if event_features is None:
        return {}

    # Need valid distill features for StuffRV + event features for FullRV
    required_cols = list(_DISTILL_FEATURES) + [
        "PlateLocSide", "PlateLocHeight", "Balls", "Strikes",
        "TaggedPitchType", "Pitcher",
    ]
    # Also need all event base features
    for col in _EVENT_BASE_FEATURES:
        if col not in required_cols:
            required_cols.append(col)

    valid = df[list(_DISTILL_FEATURES)].notna().all(axis=1) & df["TaggedPitchType"].notna()
    for col in ["PlateLocSide", "PlateLocHeight", "Balls", "Strikes", "Pitcher"]:
        valid = valid & df[col].notna()
    for col in _EVENT_BASE_FEATURES:
        if col in df.columns:
            valid = valid & df[col].notna()

    sub = df.loc[valid].copy()
    if len(sub) < 1000:
        return {}

    # 1) Compute StuffRV via distilled models
    X_distill = sub[list(_DISTILL_FEATURES)].astype(np.float32)
    stuff_rv_r = distilled_models["StuffRV_vsR"].predict(X_distill)
    stuff_rv_l = distilled_models["StuffRV_vsL"].predict(X_distill)
    sub["StuffRV"] = 0.5 * (stuff_rv_r + stuff_rv_l)

    # 2) Compute FullRV at actual location+count for each batter side
    rv_lookup = _run_value_lookup(run_value_table)
    balls = sub["Balls"].astype(int).clip(0, 3).to_numpy()
    strikes = sub["Strikes"].astype(int).clip(0, 2).to_numpy()

    full_rv_sides = []
    for side_label, side_enc in [("R", 0.0), ("L", 1.0)]:
        X_event = sub[list(_EVENT_BASE_FEATURES) + ["PlateLocSide", "PlateLocHeight"]].copy()
        X_event["BatterSide_enc"] = side_enc
        X_event["Balls"] = balls.astype(float)
        X_event["Strikes"] = strikes.astype(float)
        X_event = X_event[event_features].astype(np.float32)

        terminal_probs = _predict_terminal_probabilities(X_event, event_models)
        rv = _expected_run_value(terminal_probs, rv_lookup, balls, strikes)
        full_rv_sides.append(rv)

    sub["FullRV"] = 0.5 * (full_rv_sides[0] + full_rv_sides[1])

    # 3) LocationRV = FullRV - StuffRV
    sub["LocationRV"] = sub["FullRV"] - sub["StuffRV"]

    # 4) Aggregate by (Pitcher, TaggedPitchType) → pitcher-level mean
    pitcher_agg = (
        sub.groupby(["Pitcher", "TaggedPitchType"], observed=False)["LocationRV"]
        .agg(["mean", "count"])
        .reset_index()
    )
    pitcher_agg = pitcher_agg[pitcher_agg["count"] >= 10]

    # 5) Per pitch type: (mean, std) of pitcher means
    loc_pt_stats: Dict[str, Tuple[float, float]] = {}
    all_pitcher_means = []
    for pt in pitcher_agg["TaggedPitchType"].unique():
        pt_data = pitcher_agg[pitcher_agg["TaggedPitchType"] == pt]
        if len(pt_data) < 10:
            continue
        pitcher_means = pt_data["mean"].to_numpy()
        loc_pt_stats[str(pt)] = (float(np.mean(pitcher_means)), float(np.std(pitcher_means)))
        all_pitcher_means.extend(pitcher_means.tolist())

    if all_pitcher_means:
        arr = np.array(all_pitcher_means)
        loc_pt_stats["__global__"] = (float(np.mean(arr)), float(np.std(arr)))

    print(f"  LocationRV stats: {len(loc_pt_stats)-1} pitch types, "
          f"{len(pitcher_agg)} pitcher-type groups from {len(sub):,} pitches")
    return loc_pt_stats


def train_pitchsim_stuff_model(parquet_path: str, model_path: str) -> None:
    import joblib

    print(f"  Loading D1 pitches from {parquet_path} ...")
    df = _fetch_training_frame(parquet_path, target_total=_TRAIN_TARGET_ROWS)
    print(f"  Loaded {len(df):,} D1 pitches")
    df = df.dropna(subset=["TaggedPitchType"]).copy()
    print(f"  After D1 pitch filters: {len(df):,} pitches")

    warm_df = _prepare_pitchsim_frame(df)
    vaa_models = _fit_vaa_models(warm_df)
    df = _prepare_pitchsim_frame(df, vaa_models=vaa_models)

    weights, weight_metrics = _fit_weight_model(df)
    print(
        "  Stuff-only weighting model:"
        f" rows={weight_metrics['sample_rows']:,}"
        f" logloss={weight_metrics['val_logloss']:.5f}"
        f" auc={weight_metrics['val_auc']:.4f}"
    )

    cluster_scaler, cluster_model, weight_vec, cluster_cols, df, cluster_summaries = _fit_cluster_model(df, weights)
    n_unique = cluster_model.get("unique_heuristics", _N_CLUSTERS)
    print(f"  Fitted {_N_CLUSTERS} fuzzy archetypes ({n_unique} unique heuristics)")

    run_value_table = _build_d1_run_value_table(parquet_path)
    print(f"  Built D1 run-value table with {len(run_value_table):,} count rows")

    context_distributions = _build_context_priors(df, cluster_cols)
    print(f"  Built {len(context_distributions):,} exact-count platoon context distributions")

    event_models, event_metrics = _fit_event_models(df, cluster_cols)
    print(
        "  Event models:"
        f" swing_auc={event_metrics.get('swing', {}).get('val_auc', np.nan):.4f}"
        f" whiff_auc={event_metrics.get('whiff_given_swing', {}).get('val_auc', np.nan):.4f}"
        f" take_cs_auc={event_metrics.get('called_strike_given_take', {}).get('val_auc', np.nan):.4f}"
        f" take_ball_auc={event_metrics.get('ball_given_take', {}).get('val_auc', np.nan):.4f}"
        f" hr_auc={event_metrics.get('home_run_given_inplay', {}).get('val_auc', np.nan):.4f}"
    )

    sim_df = _stratified_sample_by_pitch_type(df, total_rows=_SIM_TARGET_ROWS, min_rows=40)
    sim_df = sim_df.dropna(subset=list(_EVENT_BASE_FEATURES) + list(cluster_cols))
    print(f"  Simulating standardized stuff outcomes for {len(sim_df):,} pitch-shape rows ...")

    # Free the large training frame before simulation to avoid OOM
    del df
    import gc; gc.collect()

    sim_df = _simulate_pitchsim_targets(
        sim_df=sim_df,
        cluster_cols=cluster_cols,
        context_distributions=context_distributions,
        event_models=event_models,
        run_value_table=run_value_table,
    )

    distilled_models, distill_metrics = _fit_distilled_models(sim_df)

    # Reload training data for population stats
    print("  Reloading training data for population stats ...")
    df = _fetch_training_frame(parquet_path, target_total=_TRAIN_TARGET_ROWS)
    df = df.dropna(subset=["TaggedPitchType"]).copy()
    df = _prepare_pitchsim_frame(df, vaa_models=vaa_models)
    pt_stats, fip_pt_stats = _compute_population_stats(df, distilled_models)

    # LocationRV population stats for Command+
    print("  Computing LocationRV population stats for Command+ ...")
    loc_pt_stats = _compute_location_rv_stats(
        df, distilled_models, event_models, run_value_table,
    )

    _validate_davidson_stuff(
        parquet_path=parquet_path,
        distilled_models=distilled_models,
        pt_stats=pt_stats,
        vaa_models=vaa_models,
    )

    artifact = {
        "artifact_type": "pitchsim_lite",
        "artifact_version": _ARTIFACT_VERSION,
        "d1_only": True,
        "features": list(_DISTILL_FEATURES),
        "distilled_models": distilled_models,
        "pt_stats": pt_stats,
        "fip_pt_stats": fip_pt_stats,
        "cluster_features": list(_CLUSTER_FEATURES),
        "cluster_feature_weights": weights,
        "cluster_scaler": cluster_scaler,
        "cluster_model": cluster_model,
        "cluster_weight_vector": weight_vec,
        "cluster_cols": list(cluster_cols),
        "cluster_summaries": cluster_summaries,
        "vaa_models": vaa_models,
        "context_distributions": context_distributions,
        "run_value_table": run_value_table,
        "event_models": event_models,
        "loc_pt_stats": loc_pt_stats,
        "event_metrics": event_metrics,
        "weight_metrics": weight_metrics,
        "distill_metrics": distill_metrics,
        "pitch_type_labels": _PITCH_TYPE_LABELS,
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    print(f"  Saved D1 PitchSim-style Stuff+ artifact to {model_path}")


def _validate_davidson_stuff(
    parquet_path: str,
    distilled_models: Dict[str, object],
    pt_stats: Dict[str, Tuple[float, float]],
    vaa_models: object,
) -> None:
    from config import DAVIDSON_TEAM_ID

    print("\n  === Davidson Stuff+ Validation ===")
    try:
        import duckdb

        con = duckdb.connect()
        dav_df = con.execute(
            f"""
            SELECT * FROM read_parquet('{parquet_path}')
            WHERE "PitcherTeam" = '{DAVIDSON_TEAM_ID}'
              AND "Level" = '{_D1_LEVEL}'
              AND "TaggedPitchType" IS NOT NULL
            """
        ).fetchdf()
        con.close()
    except Exception as exc:
        print(f"  Validation skipped (data load error): {exc}")
        return

    if dav_df.empty:
        print("  No Davidson pitches found — skipping validation")
        return

    scored = _prepare_pitchsim_frame(dav_df, vaa_models=vaa_models)
    if scored is None or scored.empty:
        print("  Validation skipped (no scoreable pitches)")
        return

    feature_cols = list(_DISTILL_FEATURES)
    for col in feature_cols:
        if col not in scored.columns:
            scored[col] = np.nan
    valid = scored[feature_cols].notna().all(axis=1) & scored["TaggedPitchType"].notna()
    if not valid.any():
        print("  Validation skipped (no valid feature rows)")
        return

    X = scored.loc[valid, feature_cols].astype(np.float32)
    rv_r = distilled_models["StuffRV_vsR"].predict(X)
    rv_l = distilled_models["StuffRV_vsL"].predict(X)
    scored.loc[valid, "StuffRV"] = 0.5 * (rv_r + rv_l)

    global_mu, global_sigma = pt_stats.get("__global__", (0.0, 1.0))
    if not np.isfinite(global_sigma) or global_sigma <= 0:
        global_sigma = 1.0
    stuff_scores = pd.Series(np.nan, index=scored.index, dtype=float)
    for pt in scored.loc[valid, "TaggedPitchType"].unique():
        mask = scored["TaggedPitchType"] == pt
        mu, sigma = pt_stats.get(str(pt), (global_mu, global_sigma))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = global_sigma
        z = (scored.loc[mask, "StuffRV"].astype(float) - mu) / sigma
        stuff_scores.loc[mask] = 100.0 + (-z) * 10.0
    scored["StuffPlus"] = stuff_scores

    sp_valid = scored["StuffPlus"].notna()
    print(f"  Davidson pitches: {int(sp_valid.sum()):,} scored / {len(dav_df):,} total")
    sp = scored.loc[sp_valid, "StuffPlus"]
    print(f"  StuffPlus distribution: mean={sp.mean():.1f}  std={sp.std():.1f}  min={sp.min():.0f}  max={sp.max():.0f}")

    # Per-pitch-type top/bottom pitchers
    scored["Pitcher"] = dav_df.loc[scored.index, "Pitcher"] if "Pitcher" in dav_df.columns else "Unknown"
    for pt in sorted(scored.loc[sp_valid, "TaggedPitchType"].unique()):
        pt_mask = (scored["TaggedPitchType"] == pt) & sp_valid
        agg = scored.loc[pt_mask].groupby("Pitcher").agg(
            stuff_mean=("StuffPlus", "mean"), n=("StuffPlus", "size")
        )
        agg = agg[agg["n"] >= 20].sort_values("stuff_mean", ascending=False)
        if agg.empty:
            continue
        top = agg.head(5)
        bot = agg.tail(5)
        print(f"\n  {pt} (n={int(pt_mask.sum()):,}, {len(agg)} pitchers w/ 20+ pitches):")
        for name, row in top.iterrows():
            print(f"    TOP  {name}: {row['stuff_mean']:.1f} ({int(row['n'])} pitches)")
        for name, row in bot.iterrows():
            print(f"    BOT  {name}: {row['stuff_mean']:.1f} ({int(row['n'])} pitches)")

    # Correlation with actual whiff and CSW rates
    pitcher_type = scored.loc[sp_valid].copy()
    pitcher_type["is_whiff"] = pitcher_type["PitchCall"].isin({"StrikeSwinging"}).astype(float)
    pitcher_type["is_csw"] = pitcher_type["PitchCall"].isin({"StrikeSwinging", "StrikeCalled"}).astype(float)
    pt_agg = pitcher_type.groupby(["Pitcher", "TaggedPitchType"]).agg(
        stuff_mean=("StuffPlus", "mean"),
        whiff_rate=("is_whiff", "mean"),
        csw_rate=("is_csw", "mean"),
        n=("StuffPlus", "size"),
    )
    pt_agg = pt_agg[pt_agg["n"] >= 20]
    if len(pt_agg) >= 5:
        r_whiff = _safe_corr(pt_agg["stuff_mean"], pt_agg["whiff_rate"])
        r_csw = _safe_corr(pt_agg["stuff_mean"], pt_agg["csw_rate"])
        print(f"\n  Correlation (pitcher-type agg, n>={20}, {len(pt_agg)} groups):")
        print(f"    StuffPlus vs whiff_rate: r={r_whiff:.3f}")
        print(f"    StuffPlus vs csw_rate:   r={r_csw:.3f}")
    print("  === End Validation ===\n")


def compute_pitchsim_stuff_plus(data: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    if data is None or len(data) == 0:
        return data

    models = artifact.get("distilled_models", {})
    feature_cols = artifact.get("features", _DISTILL_FEATURES)
    vaa_models = artifact.get("vaa_models")
    pt_stats = artifact.get("pt_stats", {})
    fip_pt_stats = artifact.get("fip_pt_stats", {})

    df = _prepare_pitchsim_frame(data, vaa_models=vaa_models)
    if df is None or len(df) == 0:
        return data
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    valid = df[feature_cols].notna().all(axis=1) & df["TaggedPitchType"].notna()
    if not valid.any():
        df["StuffPlus"] = np.nan
        df["StuffFIPPlus"] = np.nan
        return df

    X = df.loc[valid, feature_cols].astype(np.float32)
    rv_r = models["StuffRV_vsR"].predict(X)
    rv_l = models["StuffRV_vsL"].predict(X)
    whiff_r = models["StuffWhiff_vsR"].predict(X)
    whiff_l = models["StuffWhiff_vsL"].predict(X)
    cs_r = models["StuffCalledStrike_vsR"].predict(X)
    cs_l = models["StuffCalledStrike_vsL"].predict(X)
    ball_r = models["StuffBall_vsR"].predict(X)
    ball_l = models["StuffBall_vsL"].predict(X)
    hr_r = models["StuffHomeRun_vsR"].predict(X)
    hr_l = models["StuffHomeRun_vsL"].predict(X)
    damage_r = models["StuffDamage_vsR"].predict(X)
    damage_l = models["StuffDamage_vsL"].predict(X)

    df["StuffRV_vsR"] = np.nan
    df["StuffRV_vsL"] = np.nan
    df["StuffRV"] = np.nan
    df["StuffWhiff_vsR"] = np.nan
    df["StuffWhiff_vsL"] = np.nan
    df["StuffWhiff"] = np.nan
    df["StuffCalledStrike_vsR"] = np.nan
    df["StuffCalledStrike_vsL"] = np.nan
    df["StuffCalledStrike"] = np.nan
    df["StuffBall_vsR"] = np.nan
    df["StuffBall_vsL"] = np.nan
    df["StuffBall"] = np.nan
    df["StuffHomeRun_vsR"] = np.nan
    df["StuffHomeRun_vsL"] = np.nan
    df["StuffHomeRun"] = np.nan
    df["StuffDamage_vsR"] = np.nan
    df["StuffDamage_vsL"] = np.nan
    df["StuffDamage"] = np.nan
    df["StuffFIPRV"] = np.nan
    df.loc[valid, "StuffRV_vsR"] = rv_r
    df.loc[valid, "StuffRV_vsL"] = rv_l
    df.loc[valid, "StuffRV"] = 0.5 * (rv_r + rv_l)
    df.loc[valid, "StuffWhiff_vsR"] = whiff_r
    df.loc[valid, "StuffWhiff_vsL"] = whiff_l
    df.loc[valid, "StuffWhiff"] = 0.5 * (whiff_r + whiff_l)
    df.loc[valid, "StuffCalledStrike_vsR"] = cs_r
    df.loc[valid, "StuffCalledStrike_vsL"] = cs_l
    df.loc[valid, "StuffCalledStrike"] = 0.5 * (cs_r + cs_l)
    df.loc[valid, "StuffBall_vsR"] = ball_r
    df.loc[valid, "StuffBall_vsL"] = ball_l
    df.loc[valid, "StuffBall"] = 0.5 * (ball_r + ball_l)
    df.loc[valid, "StuffHomeRun_vsR"] = hr_r
    df.loc[valid, "StuffHomeRun_vsL"] = hr_l
    df.loc[valid, "StuffHomeRun"] = 0.5 * (hr_r + hr_l)
    df.loc[valid, "StuffDamage_vsR"] = damage_r
    df.loc[valid, "StuffDamage_vsL"] = damage_l
    df.loc[valid, "StuffDamage"] = 0.5 * (damage_r + damage_l)
    df.loc[valid, "StuffFIPRV"] = _compose_fip_raw(
        whiff=df.loc[valid, "StuffWhiff"].to_numpy(dtype=float),
        called_strike=df.loc[valid, "StuffCalledStrike"].to_numpy(dtype=float),
        ball=df.loc[valid, "StuffBall"].to_numpy(dtype=float),
        home_run=df.loc[valid, "StuffHomeRun"].to_numpy(dtype=float),
        damage=df.loc[valid, "StuffDamage"].to_numpy(dtype=float),
    )
    df["StuffRV100"] = df["StuffRV"] * 100.0
    df["StuffFIPRV100"] = df["StuffFIPRV"] * 100.0

    stuff_scores = pd.Series(np.nan, index=df.index, dtype=float)
    global_mu, global_sigma = pt_stats.get("__global__", (0.0, 1.0))
    if not np.isfinite(global_sigma) or global_sigma <= 0:
        global_sigma = 1.0
    for pt in df.loc[valid, "TaggedPitchType"].unique():
        mask = df["TaggedPitchType"] == pt
        mu, sigma = pt_stats.get(str(pt), (global_mu, global_sigma))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = global_sigma
        z = (df.loc[mask, "StuffRV"].astype(float) - mu) / sigma
        stuff_scores.loc[mask] = 100.0 + (-z) * 10.0
    df["StuffPlus"] = stuff_scores

    fip_scores = pd.Series(np.nan, index=df.index, dtype=float)
    fip_global_mu, fip_global_sigma = fip_pt_stats.get("__global__", (0.0, 1.0))
    if not np.isfinite(fip_global_sigma) or fip_global_sigma <= 0:
        fip_global_sigma = 1.0
    for pt in df.loc[valid, "TaggedPitchType"].unique():
        mask = df["TaggedPitchType"] == pt
        mu, sigma = fip_pt_stats.get(str(pt), (fip_global_mu, fip_global_sigma))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = fip_global_sigma
        z = (df.loc[mask, "StuffFIPRV"].astype(float) - mu) / sigma
        fip_scores.loc[mask] = 100.0 + (-z) * 10.0
    df["StuffFIPPlus"] = fip_scores
    return df


def compute_pitchsim_command_plus(data: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Compute PitchSim-aligned CommandPlus per pitch.

    LocationRV = FullRV(actual location+count) - StuffRV(grid-averaged)
    CommandPlus = 100 + (-z_score) * 10

    Returns the input DataFrame with a ``CommandPlus`` column added.
    """
    if data is None or len(data) == 0:
        return data

    event_models = artifact.get("event_models")
    if event_models is None:
        # Old artifact without event models — skip gracefully
        return data

    distilled_models = artifact.get("distilled_models", {})
    vaa_models = artifact.get("vaa_models")
    loc_pt_stats = artifact.get("loc_pt_stats", {})
    run_value_table = artifact.get("run_value_table")
    feature_cols = artifact.get("features", list(_DISTILL_FEATURES))
    event_features = event_models.get("event_features")

    if not distilled_models or run_value_table is None or event_features is None:
        return data
    if not loc_pt_stats:
        return data

    df = _prepare_pitchsim_frame(data, vaa_models=vaa_models)
    if df is None or len(df) == 0:
        return data

    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Require distill features + event base features + location + count
    valid = df[feature_cols].notna().all(axis=1) & df["TaggedPitchType"].notna()
    for col in ["PlateLocSide", "PlateLocHeight", "Balls", "Strikes"]:
        valid = valid & df[col].notna()
    for col in _EVENT_BASE_FEATURES:
        if col in df.columns:
            valid = valid & df[col].notna()

    if not valid.any():
        df["CommandPlus"] = np.nan
        return df

    sub = df.loc[valid]

    # 1) StuffRV via distilled models
    X_distill = sub[feature_cols].astype(np.float32)
    stuff_rv = 0.5 * (
        distilled_models["StuffRV_vsR"].predict(X_distill)
        + distilled_models["StuffRV_vsL"].predict(X_distill)
    )

    # 2) FullRV at actual location+count for each batter side
    rv_lookup = _run_value_lookup(run_value_table)
    balls = sub["Balls"].astype(int).clip(0, 3).to_numpy()
    strikes = sub["Strikes"].astype(int).clip(0, 2).to_numpy()

    full_rv_sides = []
    for side_enc in [0.0, 1.0]:  # R, L
        X_event = sub[list(_EVENT_BASE_FEATURES) + ["PlateLocSide", "PlateLocHeight"]].copy()
        X_event["BatterSide_enc"] = side_enc
        X_event["Balls"] = balls.astype(float)
        X_event["Strikes"] = strikes.astype(float)
        X_event = X_event[event_features].astype(np.float32)

        terminal_probs = _predict_terminal_probabilities(X_event, event_models)
        rv = _expected_run_value(terminal_probs, rv_lookup, balls, strikes)
        full_rv_sides.append(rv)

    full_rv = 0.5 * (full_rv_sides[0] + full_rv_sides[1])

    # 3) LocationRV = FullRV - StuffRV
    location_rv = full_rv - stuff_rv

    # 4) Z-score within pitch type → CommandPlus
    global_mu, global_sigma = loc_pt_stats.get("__global__", (0.0, 1.0))
    if not np.isfinite(global_sigma) or global_sigma <= 0:
        global_sigma = 1.0

    cmd_scores = np.full(len(df), np.nan, dtype=float)
    pitch_types = sub["TaggedPitchType"].to_numpy()
    for pt in np.unique(pitch_types):
        pt_mask = pitch_types == pt
        mu, sigma = loc_pt_stats.get(str(pt), (global_mu, global_sigma))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = global_sigma
        z = (location_rv[pt_mask] - mu) / sigma
        cmd_scores[valid.to_numpy().nonzero()[0][pt_mask]] = 100.0 + (-z) * 10.0

    df["CommandPlus"] = cmd_scores
    return df
