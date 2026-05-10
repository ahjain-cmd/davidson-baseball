#!/usr/bin/env python3
"""Backtest live PitchSim Stuff+ against next-season ERA."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from analytics.stuff_plus import _compute_stuff_plus
from config import PARQUET_PATH


CANDIDATE_STATS = [
    "ERA",
    "FIP",
    "xFIP",
    "WHIP",
    "K/9",
    "BB/9",
    "HR/9",
    "K%",
    "BB%",
    "LOB%",
    "OPS",
    "WOBA",
    "BA",
    "FBVel",
    "Spin",
    "IndVertBrk",
    "HorzBrk",
    "Chase%",
    "SwStrk%",
    "Contact%",
    "Swing%",
    "FPStk%",
    "InZone%",
    "ExitVel",
    "Barrel%",
    "HardHit%",
    "Ground%",
    "Fly%",
    "Line%",
    "Popup%",
    "xAVG",
    "xSLG",
    "xWOBA",
]


def _cache_path(season: int) -> str:
    return os.path.join(APP_DIR, ".cache", f"tm_league_pitchers_{season}.parquet")


def _normalize_trackman_id(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.astype("Int64").astype(str).replace("<NA>", np.nan)


def _load_tm_pitchers(season: int) -> pd.DataFrame:
    path = _cache_path(season)
    if not os.path.exists(path):
        raise FileNotFoundError(f"TrueMedia pitcher cache not found: {path}")
    df = pd.read_parquet(path)
    df = df.copy()
    df["trackman_id"] = _normalize_trackman_id(df.get("trackmanPlayerId", pd.Series(dtype=object)))
    df["IP"] = pd.to_numeric(df.get("IP"), errors="coerce")
    return df


def _dedupe_tm_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.dropna(subset=["trackman_id"]).copy()
    work = work.sort_values(["trackman_id", "IP"], ascending=[True, False])
    return work.drop_duplicates("trackman_id", keep="first")


def _pitch_type_sql(raw_col: str = "TaggedPitchType") -> str:
    return f"""
        CASE
            WHEN {raw_col} IN ('Undefined', 'Other', 'Knuckleball') THEN NULL
            WHEN {raw_col} = 'FourSeamFastBall' THEN 'Fastball'
            WHEN {raw_col} IN ('OneSeamFastBall', 'TwoSeamFastBall') THEN 'Sinker'
            WHEN {raw_col} = 'ChangeUp' THEN 'Changeup'
            ELSE {raw_col}
        END
    """


def _load_source_pitches(
    parquet_path: str,
    source_season: int,
    future_ids: Iterable[str],
) -> pd.DataFrame:
    future_ids_df = pd.DataFrame({"PitcherId": list(future_ids)})
    if future_ids_df.empty:
        return pd.DataFrame()

    con = duckdb.connect(":memory:")
    con.register("future_ids", future_ids_df)
    pt_sql = _pitch_type_sql("TaggedPitchType")
    query = f"""
        WITH base AS (
            SELECT
                CASE
                    WHEN "Date" IS NOT NULL THEN
                        CASE WHEN EXTRACT(MONTH FROM CAST("Date" AS DATE)) >= 8
                             THEN EXTRACT(YEAR FROM CAST("Date" AS DATE))::INT + 1
                             ELSE EXTRACT(YEAR FROM CAST("Date" AS DATE))::INT
                        END
                    ELSE NULL
                END AS Season,
                PitcherId,
                Pitcher,
                PitcherTeam,
                GameID,
                PitchUID,
                PitchNo,
                Inning,
                PAofInning,
                Batter,
                CASE WHEN PitcherThrows IN ('Left', 'L') THEN 'L' ELSE 'R' END AS PitcherThrows,
                CASE WHEN BatterSide IN ('Left', 'L') THEN 'L' ELSE 'R' END AS BatterSide,
                Balls,
                Strikes,
                Outs,
                PitchCall,
                PlayResult,
                OutsOnPlay,
                TaggedHitType,
                KorBB,
                RelSpeed,
                InducedVertBreak,
                HorzBreak,
                Extension,
                RelHeight,
                RelSide,
                VertRelAngle,
                HorzRelAngle,
                VertApprAngle,
                HorzApprAngle,
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
        )
        SELECT b.*
        FROM base b
        JOIN future_ids f ON CAST(b.PitcherId AS VARCHAR) = f.PitcherId
        WHERE b.Season = {int(source_season)}
          AND b.Level = 'D1'
          AND b.PitchCall IS NOT NULL
          AND b.PitchCall != 'Undefined'
          AND b.TaggedPitchType IS NOT NULL
          AND b.Pitcher IS NOT NULL
          AND b.PitcherThrows IS NOT NULL
          AND b.BatterSide IS NOT NULL
          AND b.RelSpeed IS NOT NULL
          AND b.InducedVertBreak IS NOT NULL
          AND b.HorzBreak IS NOT NULL
          AND b.Extension IS NOT NULL
          AND b.RelHeight IS NOT NULL
          AND b.RelSide IS NOT NULL
          AND b.VertRelAngle IS NOT NULL
          AND b.HorzRelAngle IS NOT NULL
          AND b.vx0 IS NOT NULL
          AND b.vy0 IS NOT NULL
          AND b.vz0 IS NOT NULL
          AND b.ax0 IS NOT NULL
          AND b.ay0 IS NOT NULL
          AND b.az0 IS NOT NULL
    """
    df = con.execute(query).fetchdf()
    con.close()
    return df


def _pearson(a: pd.Series, b: pd.Series) -> float:
    valid = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 10:
        return np.nan
    return float(np.corrcoef(valid["a"].astype(float), valid["b"].astype(float))[0, 1])


def _spearman(a: pd.Series, b: pd.Series) -> float:
    valid = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 10:
        return np.nan
    return float(valid["a"].rank().corr(valid["b"].rank()))


def run_backtest(
    parquet_path: str,
    source_season: int,
    target_season: int,
    future_ip_min: float,
    source_pitches_min: int,
    out_json: str,
    out_csv: str,
    model_path: str | None = None,
    display_pop_path: str | None = None,
    train_walk_forward_model: bool = False,
    train_max_season: int | None = None,
    display_season: int | None = None,
) -> dict:
    source_tm = _dedupe_tm_pitchers(_load_tm_pitchers(source_season))
    future_tm = _dedupe_tm_pitchers(_load_tm_pitchers(target_season))

    future_tm = future_tm[pd.to_numeric(future_tm["IP"], errors="coerce") >= float(future_ip_min)].copy()
    future_ids = future_tm["trackman_id"].dropna().astype(str).unique().tolist()
    print(f"Future ERA pool: {len(future_ids):,} pitchers with {target_season} IP >= {future_ip_min:g}")

    pitches = _load_source_pitches(parquet_path, source_season, future_ids)
    print(f"Source pitch rows loaded: {len(pitches):,}")
    if pitches.empty:
        raise RuntimeError("No source pitches found for ERA backtest")

    if train_walk_forward_model:
        from analytics.stuff_plus import train_stuff_plus_model

        if not model_path:
            model_path = os.path.join(APP_DIR, "models", f"stuff_plus_xgb_through_{source_season}.joblib")
        if not display_pop_path:
            display_pop_path = os.path.join(APP_DIR, "models", f"stuff_plus_population_{source_season}.parquet")
        cutoff = int(train_max_season if train_max_season is not None else source_season)
        train_stuff_plus_model(
            parquet_path=parquet_path,
            model_path=model_path,
            display_pop_path=display_pop_path,
            train_max_season=cutoff,
            display_season=int(display_season if display_season is not None else source_season),
        )

    scored = _compute_stuff_plus(
        pitches,
        model_path=model_path,
        display_pop_path=display_pop_path,
    )
    valid_scored = scored.dropna(subset=["StuffPlus"]).copy()
    print(f"Stuff+ scored pitches: {len(valid_scored):,} / {len(scored):,}")

    pitcher_stuff = (
        valid_scored.assign(trackman_id=_normalize_trackman_id(valid_scored["PitcherId"]))
        .groupby(["trackman_id", "Pitcher"], observed=False)
        .agg(
            StuffPlus=("StuffPlus", "mean"),
            StuffRV100=("StuffRV100", "mean"),
            source_pitches=("StuffPlus", "count"),
        )
        .reset_index()
    )

    current_cols = ["trackman_id", "fullName", "mostRecentTeamName", "IP"] + [
        c for c in CANDIDATE_STATS if c in source_tm.columns
    ]
    current = source_tm[current_cols].copy()
    current = current.rename(
        columns={
            "fullName": "source_fullName",
            "mostRecentTeamName": "source_team",
            "IP": "source_IP",
        }
    )
    current = current.rename(columns={c: f"source_{c}" for c in CANDIDATE_STATS if c in current.columns})

    future_cols = ["trackman_id", "fullName", "mostRecentTeamName", "IP", "ERA"]
    future = future_tm[[c for c in future_cols if c in future_tm.columns]].copy()
    future = future.rename(
        columns={
            "fullName": "target_fullName",
            "mostRecentTeamName": "target_team",
            "IP": "target_IP",
            "ERA": "target_ERA",
        }
    )

    detail = pitcher_stuff.merge(current, on="trackman_id", how="left").merge(future, on="trackman_id", how="inner")
    detail["target_ERA"] = pd.to_numeric(detail["target_ERA"], errors="coerce")
    detail["target_IP"] = pd.to_numeric(detail["target_IP"], errors="coerce")
    detail = detail[
        (detail["source_pitches"] >= int(source_pitches_min))
        & detail["target_ERA"].notna()
        & detail["target_IP"].ge(float(future_ip_min))
    ].copy()
    print(f"Backtest sample: {len(detail):,} pitchers after source pitch minimum {source_pitches_min:,}")
    if len(detail) < 10:
        raise RuntimeError("Backtest sample too small")

    rows = [
        {
            "stat": "StuffPlus",
            "pearson_r": _pearson(detail["StuffPlus"], detail["target_ERA"]),
            "spearman_r": _spearman(detail["StuffPlus"], detail["target_ERA"]),
            "n": int(detail[["StuffPlus", "target_ERA"]].dropna().shape[0]),
        }
    ]
    for stat in CANDIDATE_STATS:
        col = f"source_{stat}"
        if col not in detail.columns:
            continue
        vals = pd.to_numeric(detail[col], errors="coerce")
        n = int(pd.DataFrame({"x": vals, "y": detail["target_ERA"]}).replace([np.inf, -np.inf], np.nan).dropna().shape[0])
        if n < 50:
            continue
        rows.append(
            {
                "stat": stat,
                "pearson_r": _pearson(vals, detail["target_ERA"]),
                "spearman_r": _spearman(vals, detail["target_ERA"]),
                "n": n,
            }
        )

    leaderboard = pd.DataFrame(rows)
    leaderboard["abs_pearson_r"] = leaderboard["pearson_r"].abs()
    leaderboard = leaderboard.sort_values("abs_pearson_r", ascending=False).reset_index(drop=True)
    leaderboard["rank"] = np.arange(1, len(leaderboard) + 1)
    stuff_row = leaderboard[leaderboard["stat"] == "StuffPlus"].iloc[0].to_dict()

    detail = detail.sort_values("StuffPlus", ascending=False)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    detail.to_csv(out_csv, index=False)

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parquet_path": parquet_path,
        "source_season": int(source_season),
        "target_season": int(target_season),
        "future_ip_min": float(future_ip_min),
        "source_pitches_min": int(source_pitches_min),
        "model_path": model_path,
        "display_pop_path": display_pop_path,
        "train_walk_forward_model": bool(train_walk_forward_model),
        "train_max_season": int(train_max_season) if train_max_season is not None else (
            int(source_season) if train_walk_forward_model else None
        ),
        "sample_pitchers": int(len(detail)),
        "stuff_pearson_r_to_future_era": float(stuff_row["pearson_r"]),
        "stuff_spearman_r_to_future_era": float(stuff_row["spearman_r"]),
        "stuff_rank_by_abs_pearson": int(stuff_row["rank"]),
        "comparison_stats": int(len(leaderboard)),
        "beats_stats": int(len(leaderboard) - int(stuff_row["rank"])),
        "leaderboard": leaderboard.to_dict(orient="records"),
        "detail_csv": out_csv,
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print("\nERA leaderboard by absolute Pearson r:")
    print(leaderboard.head(12).to_string(index=False))
    print(f"\nSaved detail CSV: {out_csv}")
    print(f"Saved summary JSON: {out_json}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=PARQUET_PATH)
    parser.add_argument("--source-season", type=int, default=2024)
    parser.add_argument("--target-season", type=int, default=2025)
    parser.add_argument("--future-ip-min", type=float, default=40.0)
    parser.add_argument("--source-pitches-min", type=int, default=250)
    parser.add_argument("--out-json", default=os.path.join(APP_DIR, "exports", "pitchsim_stuff_era_backtest.json"))
    parser.add_argument("--out-csv", default=os.path.join(APP_DIR, "exports", "pitchsim_stuff_era_backtest_detail.csv"))
    parser.add_argument("--model-path", default=None, help="PitchSim Stuff+ artifact to score with")
    parser.add_argument("--display-pop-path", default=None, help="Display population parquet for score scaling")
    parser.add_argument("--train-walk-forward-model", action="store_true",
                        help="Train a temporary artifact through the source season before scoring")
    parser.add_argument("--train-max-season", type=int, default=None,
                        help="Explicit max season for walk-forward training; defaults to source season")
    parser.add_argument("--display-season", type=int, default=None,
                        help="Display population season for walk-forward artifact; defaults to source season")
    args = parser.parse_args()

    run_backtest(
        parquet_path=args.parquet,
        source_season=args.source_season,
        target_season=args.target_season,
        future_ip_min=args.future_ip_min,
        source_pitches_min=args.source_pitches_min,
        out_json=args.out_json,
        out_csv=args.out_csv,
        model_path=args.model_path,
        display_pop_path=args.display_pop_path,
        train_walk_forward_model=args.train_walk_forward_model,
        train_max_season=args.train_max_season,
        display_season=args.display_season,
    )


if __name__ == "__main__":
    main()
