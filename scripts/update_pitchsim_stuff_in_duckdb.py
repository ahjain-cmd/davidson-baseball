#!/usr/bin/env python3
"""Refresh PitchSim Stuff+ columns inside a precomputed Davidson DuckDB file."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone

import duckdb
import pandas as pd

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from analytics.stuff_plus import _compute_stuff_plus


STUFF_PREFIXES = ("Stuff", "PitchStuff", "DisplayStuff")
EXTRA_SCORE_COLUMNS = ("CommandPlus",)
STUFF_TABLE_COLUMNS = (
    "PitchUID",
    "Pitcher",
    "Season",
    "Date",
    "TaggedPitchType",
    "StuffPlus",
    "StuffPlus_vsR",
    "StuffPlus_vsL",
    "PitchStuffPlus",
    "PitchStuffPlus_vsR",
    "PitchStuffPlus_vsL",
    "DisplayStuffPlus",
    "DisplayStuffPlus_vsR",
    "DisplayStuffPlus_vsL",
    "StuffRV100",
    "StuffRV100_vsR",
    "StuffRV100_vsL",
)


def _backup_db(db_path: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.pre_pitchsim_v9_{timestamp}.bak"
    shutil.copy2(db_path, backup_path)
    return backup_path


def _score_columns(scored: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in scored.columns:
        if col.startswith(STUFF_PREFIXES) or col in EXTRA_SCORE_COLUMNS:
            cols.append(col)
    return cols


def _attach_command_plus(base_df: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    try:
        import joblib

        from analytics.pitchsim_stuff import (
            compute_pitchsim_command_plus,
            pitchsim_artifact_has_full_cascade,
        )

        model_path = os.path.join(
            APP_DIR,
            "models",
            "stuff_plus_xgb.joblib",
        )
        model_path = os.environ.get("PITCHSIM_STUFF_MODEL_PATH", model_path)
        artifact = joblib.load(model_path)
        if pitchsim_artifact_has_full_cascade(artifact):
            command_df = compute_pitchsim_command_plus(base_df.copy(), artifact)
            if "CommandPlus" in command_df.columns:
                scored["CommandPlus"] = command_df["CommandPlus"].reindex(scored.index)
    except Exception as exc:
        print(f"CommandPlus refresh skipped: {exc}")
    return scored


def refresh_db(db_path: str, cache_dir: str | None = None, backup: bool = True) -> None:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    if backup:
        backup_path = _backup_db(db_path)
        print(f"Backup: {backup_path}")

    con = duckdb.connect(db_path)
    try:
        df = con.execute("SELECT * FROM davidson_data").fetchdf()
        print(f"Loaded davidson_data: {len(df):,} rows")
        if df.empty:
            raise RuntimeError("davidson_data is empty")

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "Season" in df.columns:
            df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")

        scored = _compute_stuff_plus(df.copy())
        scored = _attach_command_plus(df, scored)

        score_cols = _score_columns(scored)
        if "StuffPlus" not in score_cols:
            raise RuntimeError("StuffPlus was not produced by scoring path")
        for col in score_cols:
            df[col] = scored[col].reindex(df.index)

        con.register("updated_davidson_data", df)
        con.execute("CREATE OR REPLACE TABLE davidson_data AS SELECT * FROM updated_davidson_data")

        stuff_cols = [col for col in STUFF_TABLE_COLUMNS if col in df.columns]
        stuff_df = df[stuff_cols].copy()
        con.register("updated_stuff_plus", stuff_df)
        con.execute("CREATE OR REPLACE TABLE stuff_plus AS SELECT * FROM updated_stuff_plus")

        summary = con.execute(
            """
            SELECT
                COUNT(*) AS rows,
                COUNT(StuffPlus) AS scored,
                AVG(StuffPlus) AS mean_stuff,
                STDDEV_SAMP(StuffPlus) AS sd_stuff,
                MIN(StuffPlus) AS min_stuff,
                MAX(StuffPlus) AS max_stuff
            FROM davidson_data
            """
        ).fetchdf()
        print(summary.to_string(index=False))
    finally:
        con.close()

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        feather_path = os.path.join(cache_dir, "davidson_data.feather")
        df.to_feather(feather_path)
        print(f"Feather cache refreshed: {feather_path} ({len(df):,} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="davidson.duckdb", help="Path to davidson.duckdb")
    parser.add_argument("--cache-dir", default=".cache", help="Cache dir for davidson_data.feather")
    parser.add_argument("--no-backup", action="store_true", help="Skip DB backup")
    args = parser.parse_args()

    refresh_db(
        db_path=args.db,
        cache_dir=args.cache_dir,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
