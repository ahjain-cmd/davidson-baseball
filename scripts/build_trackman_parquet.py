#!/usr/bin/env python3
"""Build an enriched Trackman parquet from the raw CSV export tree (v3/...).

Why this exists:
  The current `all_trackman.parquet` drops many Trackman CSV columns (e.g.
  EffectiveVelo, AutoPitchType, PitchUID/PlayID, catcher/throw metrics, etc.).
  This script rebuilds a "fixed" parquet (default: all_trackman_fixed.parquet)
  by unioning *all* CSV columns by name.

Usage:
  python3 scripts/build_trackman_parquet.py --dry-run
  python3 scripts/build_trackman_parquet.py
  python3 scripts/build_trackman_parquet.py --out /path/to/all_trackman_fixed.parquet
"""

from __future__ import annotations

import argparse
import os
from typing import List

import duckdb

from config import DATA_ROOT, PARQUET_FIXED_PATH


def _discover_csv_files(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".csv"):
                continue
            if fn.startswith("."):
                continue
            files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Build enriched Trackman parquet from raw CSV tree")
    parser.add_argument("--csv-root", default=DATA_ROOT, help="Root folder containing Trackman CSV export tree (default: config.DATA_ROOT)")
    parser.add_argument("--out", default=PARQUET_FIXED_PATH, help="Output parquet path (default: config.PARQUET_FIXED_PATH)")
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "none"], help="Parquet compression codec")
    parser.add_argument("--dry-run", action="store_true", help="Only print file counts and exit")
    args = parser.parse_args()

    csv_root = os.path.abspath(args.csv_root)
    out_path = os.path.abspath(args.out)

    if not os.path.isdir(csv_root):
        raise SystemExit(f"CSV root not found: {csv_root}")

    csv_files = _discover_csv_files(csv_root)
    if not csv_files:
        raise SystemExit(f"No CSV files found under: {csv_root}")

    print(f"Found {len(csv_files):,} CSV files under {csv_root}")
    if args.dry_run:
        print("Dry-run: not building parquet.")
        return 0

    # DuckDB can union disparate CSV schemas by column name.
    rel = duckdb.read_csv(
        csv_files,
        union_by_name=True,
        header=True,
        ignore_errors=True,  # tolerate occasional malformed lines
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    compression = None if args.compression == "none" else args.compression
    rel.to_parquet(out_path, compression=compression, overwrite=True)
    print(f"Wrote parquet: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

