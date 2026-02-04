"""Audit pipeline accuracy for Davidson Baseball analytics.

Creates a Markdown report comparing:
- DuckDB precompute vs parquet view (trackman_parquet)
- Population metrics parity for sample batters/pitchers
- Outlier counts for plate location, direction, distance
- Meta fingerprint consistency

Usage:
  python audit_pipeline.py --out audit_report.md
"""

import argparse
import os
from datetime import datetime, timezone

import duckdb
import pandas as pd
import numpy as np

from config import (
    PARQUET_PATH,
    DUCKDB_PATH,
    _name_case_sql,
    ZONE_SIDE,
    ZONE_HEIGHT_BOT,
    ZONE_HEIGHT_TOP,
    MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE,
)


def _parquet_fingerprint(path):
    try:
        return {"path": path, "mtime": os.path.getmtime(path), "size": os.path.getsize(path)}
    except OSError:
        return {"path": path, "mtime": None, "size": None}


def _connect_db(db_path):
    if not os.path.exists(db_path):
        return None
    return duckdb.connect(db_path, read_only=True)


def _connect_parquet(parquet_path):
    con = duckdb.connect(database=":memory:")
    _create_trackman_view(con, parquet_path)
    return con


def _create_trackman_view(con, parquet_path):
    _pname = _name_case_sql("Pitcher")
    _bname = _name_case_sql("Batter")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW trackman_parquet AS
        SELECT
            * EXCLUDE (Pitcher, Batter, TaggedPitchType, BatterSide, PitcherThrows),
            {_pname} AS Pitcher,
            {_bname} AS Batter,
            CASE
                WHEN TaggedPitchType IN ('Undefined','Other','Knuckleball') THEN NULL
                WHEN TaggedPitchType = 'FourSeamFastBall' THEN 'Fastball'
                WHEN TaggedPitchType IN ('OneSeamFastBall','TwoSeamFastBall') THEN 'Sinker'
                WHEN TaggedPitchType = 'ChangeUp' THEN 'Changeup'
                ELSE TaggedPitchType
            END AS TaggedPitchType,
            CASE
                WHEN BatterSide IN ('Left','Right') THEN BatterSide
                WHEN BatterSide = 'L' THEN 'Left'
                WHEN BatterSide = 'R' THEN 'Right'
                ELSE NULL
            END AS BatterSide,
            CASE
                WHEN PitcherThrows IN ('Left','Right') THEN PitcherThrows
                WHEN PitcherThrows = 'L' THEN 'Left'
                WHEN PitcherThrows = 'R' THEN 'Right'
                ELSE NULL
            END AS PitcherThrows
        FROM read_parquet('{parquet_path}')
        WHERE PitchCall IS NULL OR PitchCall != 'Undefined'
        QUALIFY
            ROW_NUMBER() OVER (
                PARTITION BY GameID, Inning, PAofInning, PitchofPA, Pitcher, Batter, PitchNo
                ORDER BY PitchNo
            ) = 1
        """
    )


def _table_exists(con, table):
    if con is None:
        return False
    try:
        tables = con.execute("SHOW TABLES").fetchdf()
        if tables.empty:
            return False
        names = tables["name"].astype(str).str.lower().tolist()
        return table.lower() in names
    except Exception:
        return False


def _fetch_meta(con):
    if con is None or not _table_exists(con, "meta"):
        return None
    df = con.execute("SELECT * FROM meta").fetchdf()
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _escape_sql_literal(value):
    return value.replace("'", "''")


def _season_clause(seasons):
    if not seasons:
        return ""
    seasons_in = ",".join(str(int(s)) for s in seasons)
    return f"AND Season IN ({seasons_in})"


def _batters_base_sql(table, seasons=None, batters=None):
    season_clause = _season_clause(seasons)
    batter_filter = ""
    if batters:
        in_list = ",".join(f"'{_escape_sql_literal(b)}'" for b in batters)
        batter_filter = f"AND {_name_case_sql('Batter')} IN ({in_list})"
    _bnorm = _name_case_sql("Batter")
    return f"""
    WITH batter_zones AS (
        SELECT {_bnorm} AS batter_name,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_BOT} END AS zone_bot,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_TOP} END AS zone_top
        FROM {table}
        WHERE PitchCall = 'StrikeCalled'
          AND PlateLocHeight IS NOT NULL
          AND Batter IS NOT NULL {season_clause} {batter_filter}
        GROUP BY {_bnorm}
    ),
    raw AS (
        SELECT
            {_bnorm} AS Batter, BatterTeam,
            PitchCall, ExitSpeed, Angle, Distance, TaggedHitType,
            PlateLocSide, PlateLocHeight, BatterSide, Direction, KorBB,
            GameID || '_' || Inning || '_' || PAofInning || '_' || {_bnorm} AS pa_id,
            COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT}) AS zone_bot,
            COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP}) AS zone_top,
            CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                  AND ABS(PlateLocSide) <= {ZONE_SIDE}
                  AND PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                  AND PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP})
                 THEN 1 ELSE 0 END AS is_iz,
            CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                  AND NOT (ABS(PlateLocSide) <= {ZONE_SIDE}
                           AND PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                           AND PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP}))
                 THEN 1 ELSE 0 END AS is_oz
        FROM {table} r
        LEFT JOIN batter_zones bz ON {_bnorm} = bz.batter_name
        WHERE Batter IS NOT NULL {season_clause} {batter_filter}
    )
    SELECT
        Batter, BatterTeam,
        COUNT(*) AS n_pitches,
        COUNT(DISTINCT pa_id) AS PA,
        COUNT(DISTINCT CASE WHEN KorBB='Strikeout' THEN pa_id END) AS ks,
        COUNT(DISTINCT CASE WHEN KorBB='Walk' THEN pa_id END) AS bbs,
        SUM(CASE WHEN PitchCall IN ('StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay') THEN 1 ELSE 0 END) AS swings,
        SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) AS whiffs,
        SUM(CASE WHEN PitchCall IN ('FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay') THEN 1 ELSE 0 END) AS contacts,
        SUM(CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL THEN 1 ELSE 0 END) AS n_loc,
        SUM(is_iz) AS iz_count,
        SUM(is_oz) AS oz_count,
        SUM(CASE WHEN is_iz = 1 AND PitchCall IN ('StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay') THEN 1 ELSE 0 END) AS iz_swings,
        SUM(CASE WHEN is_oz = 1 AND PitchCall IN ('StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay') THEN 1 ELSE 0 END) AS oz_swings,
        SUM(CASE WHEN is_iz = 1 AND PitchCall IN ('FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay') THEN 1 ELSE 0 END) AS iz_contacts,
        SUM(CASE WHEN is_oz = 1 AND PitchCall IN ('FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay') THEN 1 ELSE 0 END) AS oz_contacts,
        SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN 1 ELSE 0 END) AS bbe,
        SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=95 THEN 1 ELSE 0 END) AS hard_hits,
        SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND Angle BETWEEN 8 AND 32 THEN 1 ELSE 0 END) AS sweet_spots,
        AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS AvgEV,
        MAX(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS MaxEV,
        AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN Angle END) AS AvgLA,
        AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN Distance END) AS AvgDist,
        SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=98
            AND Angle >= GREATEST(26 - 2*(ExitSpeed-98), 8)
            AND Angle <= LEAST(30 + 3*(ExitSpeed-98), 50) THEN 1 ELSE 0 END) AS Barrels,
        SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='GroundBall' THEN 1 ELSE 0 END) AS gb,
        SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='FlyBall' THEN 1 ELSE 0 END) AS fb,
        SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='LineDrive' THEN 1 ELSE 0 END) AS ld,
        SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='Popup' THEN 1 ELSE 0 END) AS pu
    FROM raw
    GROUP BY Batter, BatterTeam
    HAVING PA >= 50
    """


def _pitchers_base_sql(table, seasons=None, pitchers=None):
    season_clause = _season_clause(seasons)
    pitcher_filter = ""
    if pitchers:
        in_list = ",".join(f"'{_escape_sql_literal(p)}'" for p in pitchers)
        pitcher_filter = f"AND {_name_case_sql('Pitcher')} IN ({in_list})"
    _pnorm = _name_case_sql("Pitcher")
    _bnorm = _name_case_sql("Batter")
    return f"""
    WITH batter_zones AS (
        SELECT {_bnorm} AS batter_name,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_BOT} END AS zone_bot,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_TOP} END AS zone_top
        FROM {table}
        WHERE PitchCall = 'StrikeCalled'
          AND PlateLocHeight IS NOT NULL
          AND Batter IS NOT NULL {season_clause}
        GROUP BY {_bnorm}
    ),
    raw AS (
        SELECT
            {_pnorm} AS Pitcher, PitcherTeam,
            PitchCall, ExitSpeed, Angle, TaggedHitType, TaggedPitchType,
            PlateLocSide, PlateLocHeight, RelSpeed, SpinRate, Extension, KorBB,
            GameID || '_' || Inning || '_' || PAofInning || '_' || {_bnorm} AS pa_id,
            CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                  AND ABS(PlateLocSide) <= {ZONE_SIDE}
                  AND PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                  AND PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP})
                 THEN 1 ELSE 0 END AS is_iz,
            CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
                  AND NOT (ABS(PlateLocSide) <= {ZONE_SIDE}
                           AND PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                           AND PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP}))
                 THEN 1 ELSE 0 END AS is_oz
        FROM {table} r
        LEFT JOIN batter_zones bz ON {_bnorm} = bz.batter_name
        WHERE Pitcher IS NOT NULL {season_clause} {pitcher_filter}
    )
    SELECT
        Pitcher, PitcherTeam,
        COUNT(*) AS Pitches,
        COUNT(DISTINCT pa_id) AS PA,
        COUNT(DISTINCT CASE WHEN KorBB='Strikeout' THEN pa_id END) AS ks,
        COUNT(DISTINCT CASE WHEN KorBB='Walk' THEN pa_id END) AS bbs,
        SUM(CASE WHEN PitchCall IN ('StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay') THEN 1 ELSE 0 END) AS swings,
        SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) AS whiffs,
        SUM(CASE WHEN PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL THEN 1 ELSE 0 END) AS n_loc,
        SUM(is_iz) AS iz_count,
        SUM(is_oz) AS oz_count,
        SUM(CASE WHEN is_iz = 1 AND PitchCall IN ('StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay') THEN 1 ELSE 0 END) AS iz_swings,
        SUM(CASE WHEN is_oz = 1 AND PitchCall IN ('StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay') THEN 1 ELSE 0 END) AS oz_swings,
        SUM(CASE WHEN is_iz = 1 AND PitchCall IN ('FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay') THEN 1 ELSE 0 END) AS iz_contacts,
        SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN 1 ELSE 0 END) AS bbe,
        SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=95 THEN 1 ELSE 0 END) AS hard_hits,
        AVG(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS AvgEVAgainst,
        SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=98
            AND Angle >= GREATEST(26 - 2*(ExitSpeed-98), 8)
            AND Angle <= LEAST(30 + 3*(ExitSpeed-98), 50) THEN 1 ELSE 0 END) AS n_barrels,
        SUM(CASE WHEN PitchCall='InPlay' AND TaggedHitType='GroundBall' THEN 1 ELSE 0 END) AS gb,
        SUM(CASE WHEN PitchCall='InPlay' THEN 1 ELSE 0 END) AS n_ip,
        AVG(CASE WHEN TaggedPitchType IN ('Fastball','Sinker','Cutter') AND RelSpeed IS NOT NULL THEN RelSpeed END) AS AvgFBVelo,
        MAX(CASE WHEN TaggedPitchType IN ('Fastball','Sinker','Cutter') AND RelSpeed IS NOT NULL THEN RelSpeed END) AS MaxFBVelo,
        AVG(SpinRate) AS AvgSpin,
        AVG(Extension) AS Extension
    FROM raw
    GROUP BY Pitcher, PitcherTeam
    HAVING Pitches >= 100
    """


def _compare_frames(left, right, key_cols, tol=1e-6):
    if left.empty and right.empty:
        return {"status": "ok", "max_diffs": {}, "mismatch_rows": 0}
    merged = left.merge(right, on=key_cols, how="outer", suffixes=("_db", "_pq"), indicator=True)
    mismatches = {}
    mismatch_rows = 0
    for col in left.columns:
        if col in key_cols:
            continue
        col_db = f"{col}_db"
        col_pq = f"{col}_pq"
        if col_db not in merged.columns or col_pq not in merged.columns:
            continue
        if pd.api.types.is_numeric_dtype(merged[col_db]) or pd.api.types.is_numeric_dtype(merged[col_pq]):
            diff = (merged[col_db] - merged[col_pq]).abs()
            max_diff = diff.max(skipna=True)
            if pd.notna(max_diff) and max_diff > tol:
                mismatches[col] = float(max_diff)
        else:
            neq = merged[col_db] != merged[col_pq]
            if neq.any():
                mismatches[col] = "mismatch"
        mismatch_rows = max(mismatch_rows, int((merged.get("_merge") != "both").sum()))
    status = "ok" if not mismatches and mismatch_rows == 0 else "mismatch"
    return {"status": status, "max_diffs": mismatches, "mismatch_rows": mismatch_rows}


def _count_outliers(con, table):
    out = {}
    out["ploc_side_out"] = int(con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE PlateLocSide IS NOT NULL AND (PlateLocSide < -2.5 OR PlateLocSide > 2.5)"
    ).fetchone()[0])
    out["ploc_height_out"] = int(con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE PlateLocHeight IS NOT NULL AND (PlateLocHeight < 0 OR PlateLocHeight > 5.5)"
    ).fetchone()[0])
    out["direction_out"] = int(con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE PitchCall='InPlay' AND Direction IS NOT NULL AND (Direction < -90 OR Direction > 90)"
    ).fetchone()[0])
    out["distance_out"] = int(con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE PitchCall='InPlay' AND Distance IS NOT NULL AND (Distance < 0 OR Distance > 500)"
    ).fetchone()[0])
    return out


def _has_columns(con, table, cols):
    try:
        info = con.execute(f"PRAGMA table_info('{table}')").fetchdf()
        if info.empty or "name" not in info.columns:
            return False
        present = set(info["name"].astype(str).tolist())
        return all(c in present for c in cols)
    except Exception:
        return False


def _tunnel_9p_plate_check(con, table, sample_rows=2000):
    required = [
        "PlateLocSide", "PlateLocHeight",
        "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0",
    ]
    if not _has_columns(con, table, required):
        return None

    df = con.execute(
        f"""
        SELECT PlateLocSide, PlateLocHeight,
               x0, y0, z0, vx0, vy0, vz0, ax0, ay0, az0
        FROM {table}
        WHERE PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL
          AND x0 IS NOT NULL AND y0 IS NOT NULL AND z0 IS NOT NULL
          AND vx0 IS NOT NULL AND vy0 IS NOT NULL AND vz0 IS NOT NULL
          AND ax0 IS NOT NULL AND ay0 IS NOT NULL AND az0 IS NOT NULL
        LIMIT {int(sample_rows)}
        """
    ).fetchdf()
    if df.empty:
        return None

    def _solve_t_total(y0, vy0, ay0):
        a = 0.5 * ay0
        b = vy0
        c = y0
        if a == 0 and b == 0:
            return None
        if a == 0:
            t_candidates = [(-c / b)] if b != 0 else []
        else:
            disc = b * b - 4 * a * c
            if disc < 0:
                return None
            sqrt_disc = disc ** 0.5
            t_candidates = [(-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)]
        t_candidates = [t for t in t_candidates if t > 0]
        if not t_candidates:
            return None
        return min(t_candidates)

    side_err = []
    side_err_flip = []
    height_err = []
    t_vals = []
    total = len(df)

    for _, row in df.iterrows():
        t_total = _solve_t_total(row["y0"], row["vy0"], row["ay0"])
        if t_total is None:
            continue
        x = row["x0"] + row["vx0"] * t_total + 0.5 * row["ax0"] * t_total * t_total
        z = row["z0"] + row["vz0"] * t_total + 0.5 * row["az0"] * t_total * t_total
        t_vals.append(t_total)
        side_err.append(((-x) - row["PlateLocSide"]))
        side_err_flip.append((x - row["PlateLocSide"]))
        height_err.append((z - row["PlateLocHeight"]))

    if not t_vals:
        return None

    def _mae(vals):
        return float(np.mean(np.abs(vals))) if vals else None

    def _pct(vals, p):
        return float(np.percentile(vals, p)) if vals else None

    return {
        "total": total,
        "valid": len(t_vals),
        "mae_side": _mae(side_err),
        "mae_side_flip": _mae(side_err_flip),
        "mae_height": _mae(height_err),
        "t_min": float(np.min(t_vals)),
        "t_p5": _pct(t_vals, 5),
        "t_p50": _pct(t_vals, 50),
        "t_p95": _pct(t_vals, 95),
        "t_max": float(np.max(t_vals)),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit pipeline accuracy and output a report.")
    parser.add_argument("--parquet", default=PARQUET_PATH, help="Path to Trackman parquet file")
    parser.add_argument("--db", default=DUCKDB_PATH, help="Path to davidson.duckdb")
    parser.add_argument("--out", default="audit_report.md", help="Output report path")
    parser.add_argument("--sample_hitters", type=int, default=20)
    parser.add_argument("--sample_pitchers", type=int, default=20)
    parser.add_argument("--seasons", default="", help="Comma-separated seasons to test (optional)")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if mismatches found")
    args = parser.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    report_lines = []
    report_lines.append(f"# Davidson Baseball Audit Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    report_lines.append(f"Parquet: `{args.parquet}`")
    report_lines.append(f"DuckDB: `{args.db}`")
    report_lines.append("")

    pq_fp = _parquet_fingerprint(args.parquet)

    db_con = _connect_db(args.db)
    pq_con = _connect_parquet(args.parquet)

    if db_con is None:
        report_lines.append("## ERROR: DuckDB not found")
        report_lines.append("DuckDB file does not exist.")
        _write_report(args.out, report_lines)
        return 1

    meta = _fetch_meta(db_con)
    report_lines.append("## Meta Fingerprint")
    report_lines.append("| Field | Parquet | DB Meta | Match |")
    report_lines.append("|---|---|---|---|")
    for key in ["path", "mtime", "size"]:
        p_val = pq_fp.get(key)
        d_val = None if meta is None else meta.get(f"parquet_{key}")
        match = "YES" if p_val == d_val else "NO"
        report_lines.append(f"| {key} | {p_val} | {d_val} | {match} |")
    report_lines.append("")

    # Table existence
    required_tables = [
        "trackman_pop", "batter_stats_pop", "pitcher_stats_pop",
        "stuff_baselines", "tunnel_population", "meta",
    ]
    report_lines.append("## Required Tables")
    report_lines.append("| Table | Present |")
    report_lines.append("|---|---|")
    for t in required_tables:
        report_lines.append(f"| {t} | {'YES' if _table_exists(db_con, t) else 'NO'} |")
    report_lines.append("")

    # Counts parity
    report_lines.append("## Row Counts")
    if _table_exists(db_con, "trackman_pop"):
        db_total = int(db_con.execute("SELECT COUNT(*) FROM trackman_pop").fetchone()[0])
    else:
        db_total = 0
    pq_total = int(pq_con.execute("SELECT COUNT(*) FROM trackman_parquet").fetchone()[0])
    report_lines.append(f"- trackman_pop rows: {db_total:,}")
    report_lines.append(f"- trackman_parquet rows: {pq_total:,}")
    report_lines.append("")

    # Outliers
    report_lines.append("## Outlier Checks (trackman_pop)")
    if _table_exists(db_con, "trackman_pop"):
        out = _count_outliers(db_con, "trackman_pop")
        report_lines.append("| Check | Count |")
        report_lines.append("|---|---|")
        report_lines.append(f"| PlateLocSide out of range | {out['ploc_side_out']:,} |")
        report_lines.append(f"| PlateLocHeight out of range | {out['ploc_height_out']:,} |")
        report_lines.append(f"| Direction out of range (InPlay) | {out['direction_out']:,} |")
        report_lines.append(f"| Distance out of range (InPlay) | {out['distance_out']:,} |")
    else:
        report_lines.append("trackman_pop not present; outlier checks skipped.")
    report_lines.append("")

    # Tunnel kinematics sanity (9-param vs PlateLoc)
    report_lines.append("## Tunnel Kinematics Sanity (9-param)")
    kin = _tunnel_9p_plate_check(pq_con, "trackman_parquet", sample_rows=2000)
    if kin is None:
        report_lines.append("9-param columns missing or no valid rows; sanity check skipped.")
    else:
        valid_pct = kin["valid"] / max(kin["total"], 1) * 100
        report_lines.append(f"- Sample rows: {kin['total']:,}")
        report_lines.append(f"- Valid 9-param solutions: {kin['valid']:,} ({valid_pct:.1f}%)")
        report_lines.append(f"- PlateLocSide MAE (current sign): {kin['mae_side']:.4f} ft")
        report_lines.append(f"- PlateLocSide MAE (flipped sign): {kin['mae_side_flip']:.4f} ft")
        report_lines.append(f"- PlateLocHeight MAE: {kin['mae_height']:.4f} ft")
        if kin["mae_side_flip"] + 0.02 < kin["mae_side"]:
            report_lines.append("- NOTE: Non-flipped x aligns better with PlateLocSide; consider removing sign flip in 9-param path.")
        elif kin["mae_side"] + 0.02 < kin["mae_side_flip"]:
            report_lines.append("- NOTE: Current sign flip aligns better with PlateLocSide.")
        report_lines.append(
            f"- Flight time t_total (s): min {kin['t_min']:.3f}, p5 {kin['t_p5']:.3f}, "
            f"p50 {kin['t_p50']:.3f}, p95 {kin['t_p95']:.3f}, max {kin['t_max']:.3f}"
        )
    report_lines.append("")

    # Sample players
    if _table_exists(db_con, "trackman_pop"):
        season_clause = _season_clause(seasons)
        hitters = db_con.execute(
            f"""
            SELECT Batter
            FROM trackman_pop
            WHERE Batter IS NOT NULL {season_clause}
            GROUP BY Batter
            ORDER BY COUNT(*) DESC
            LIMIT {int(args.sample_hitters)}
            """
        ).fetchdf()["Batter"].tolist()
        pitchers = db_con.execute(
            f"""
            SELECT Pitcher
            FROM trackman_pop
            WHERE Pitcher IS NOT NULL {season_clause}
            GROUP BY Pitcher
            ORDER BY COUNT(*) DESC
            LIMIT {int(args.sample_pitchers)}
            """
        ).fetchdf()["Pitcher"].tolist()
    else:
        hitters = []
        pitchers = []

    # Parity checks
    report_lines.append("## Population Parity (DB vs Parquet)")
    if hitters:
        db_hit = db_con.execute(_batters_base_sql("trackman_pop", seasons=seasons, batters=hitters)).fetchdf()
        pq_hit = pq_con.execute(_batters_base_sql("trackman_parquet", seasons=seasons, batters=hitters)).fetchdf()
        hit_cmp = _compare_frames(db_hit, pq_hit, ["Batter", "BatterTeam"])
        report_lines.append(f"- Batters: {hit_cmp['status']} (mismatch rows: {hit_cmp['mismatch_rows']})")
        if hit_cmp["max_diffs"]:
            report_lines.append("  - Max diffs: " + ", ".join(f"{k}={v}" for k, v in hit_cmp["max_diffs"].items()))
    else:
        report_lines.append("- Batters: skipped (no samples)")

    if pitchers:
        db_pit = db_con.execute(_pitchers_base_sql("trackman_pop", seasons=seasons, pitchers=pitchers)).fetchdf()
        pq_pit = pq_con.execute(_pitchers_base_sql("trackman_parquet", seasons=seasons, pitchers=pitchers)).fetchdf()
        pit_cmp = _compare_frames(db_pit, pq_pit, ["Pitcher", "PitcherTeam"])
        report_lines.append(f"- Pitchers: {pit_cmp['status']} (mismatch rows: {pit_cmp['mismatch_rows']})")
        if pit_cmp["max_diffs"]:
            report_lines.append("  - Max diffs: " + ", ".join(f"{k}={v}" for k, v in pit_cmp["max_diffs"].items()))
    else:
        report_lines.append("- Pitchers: skipped (no samples)")

    report_lines.append("")

    # Final verdict
    report_lines.append("## Verdict")
    mismatches = []
    if meta is None:
        mismatches.append("meta missing")
    else:
        if pq_fp.get("path") != meta.get("parquet_path"):
            mismatches.append("parquet path mismatch")
        if pq_fp.get("mtime") != meta.get("parquet_mtime"):
            mismatches.append("parquet mtime mismatch")
        if pq_fp.get("size") != meta.get("parquet_size"):
            mismatches.append("parquet size mismatch")

    if db_total != pq_total:
        mismatches.append("trackman_pop row count mismatch")

    if hitters:
        if hit_cmp["status"] != "ok":
            mismatches.append("batter parity mismatch")
    if pitchers:
        if pit_cmp["status"] != "ok":
            mismatches.append("pitcher parity mismatch")

    if mismatches:
        report_lines.append("FAILED")
        for m in mismatches:
            report_lines.append(f"- {m}")
    else:
        report_lines.append("PASSED")

    report = "\n".join(report_lines) + "\n"
    _write_report(args.out, report_lines)
    if args.strict and mismatches:
        return 2
    return 0


def _write_report(path, lines):
    content = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    raise SystemExit(main())
