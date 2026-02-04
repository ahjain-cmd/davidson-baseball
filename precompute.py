"""Offline precompute pipeline for Davidson Baseball analytics.

Builds davidson.duckdb with:
  - davidson_data (normalized + StuffPlus + PitchUID)
  - batter_stats_pop (base aggregates by season)
  - pitcher_stats_pop (base aggregates by season)
  - stuff_baselines, fb_velo_by_pitcher, velo_diff_stats
  - tunnel_population, tunnel_benchmarks, tunnel_weights
  - tunnel_pair_outcomes
  - sidebar_stats, seasons, meta
"""

import argparse
import json
import os
from datetime import datetime, timezone

import duckdb
import pandas as pd

from config import (
    PARQUET_PATH,
    DUCKDB_PATH,
    DAVIDSON_TEAM_ID,
    _name_case_sql,
    ZONE_HEIGHT_BOT,
    ZONE_HEIGHT_TOP,
    ZONE_SIDE,
    MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE,
    _SWING_CALLS_SQL,
    _CONTACT_CALLS_SQL,
    _HAS_LOC,
    TUNNEL_BENCH_PATH,
    TUNNEL_WEIGHTS_PATH,
)

from analytics.stuff_plus import _compute_stuff_plus
from data.population import _build_tunnel_population_pop


def _parquet_fingerprint(path):
    try:
        return {"path": path, "mtime": os.path.getmtime(path), "size": os.path.getsize(path)}
    except OSError:
        return {"path": path, "mtime": None, "size": None}


def _ensure_parquet(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found: {path}")


def _connect(db_path, overwrite=False):
    if overwrite and os.path.exists(db_path):
        os.remove(db_path)
    return duckdb.connect(db_path)


def _create_trackman_view(con, parquet_path):
    _pname = _name_case_sql("Pitcher")
    _bname = _name_case_sql("Batter")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW trackman AS
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


def _create_meta(con, parquet_path):
    fp = _parquet_fingerprint(parquet_path)
    meta = pd.DataFrame([{
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parquet_path": fp.get("path"),
        "parquet_mtime": fp.get("mtime"),
        "parquet_size": fp.get("size"),
    }])
    con.register("meta_df", meta)
    con.execute("CREATE OR REPLACE TABLE meta AS SELECT * FROM meta_df")


def _create_seasons_and_sidebar(con):
    con.execute("""
        CREATE OR REPLACE TABLE seasons AS
        SELECT DISTINCT Season
        FROM trackman
        WHERE Season IS NOT NULL AND Season > 0
        ORDER BY Season
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE sidebar_stats AS
        SELECT
            COUNT(*) as total_pitches,
            COUNT(DISTINCT CASE WHEN Season > 0 THEN Season END) as n_seasons,
            MIN(CASE WHEN Season > 0 THEN Season END) as min_season,
            MAX(Season) as max_season,
            COUNT(DISTINCT Pitcher) as n_pitchers,
            COUNT(DISTINCT Batter) as n_batters,
            COUNT(DISTINCT CASE WHEN PitcherTeam = '{DAVIDSON_TEAM_ID}' OR BatterTeam = '{DAVIDSON_TEAM_ID}' THEN GameID END) as n_dav_games
        FROM trackman
    """)


def _create_trackman_pop(con):
    con.execute(
        """
        CREATE OR REPLACE TABLE trackman_pop AS
        SELECT
            Season,
            GameID,
            Inning,
            PAofInning,
            Pitcher,
            PitcherTeam,
            Batter,
            BatterTeam,
            PitchCall,
            ExitSpeed,
            Angle,
            Distance,
            TaggedHitType,
            TaggedPitchType,
            PlateLocSide,
            PlateLocHeight,
            BatterSide,
            Direction,
            KorBB,
            RelSpeed,
            SpinRate,
            Extension
        FROM trackman
        """
    )


def _create_batter_stats_pop(con):
    _bnorm = _name_case_sql("Batter")
    sql = f"""
    CREATE OR REPLACE TABLE batter_stats_pop AS
    WITH batter_zones AS (
        SELECT Season, {_bnorm} AS batter_name,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_BOT} END AS zone_bot,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_TOP} END AS zone_top
        FROM trackman
        WHERE PitchCall = 'StrikeCalled'
          AND PlateLocHeight IS NOT NULL
          AND Batter IS NOT NULL
        GROUP BY Season, {_bnorm}
    ),
    raw AS (
        SELECT
            r.Season,
            {_bnorm} AS Batter, r.BatterTeam,
            r.PitchCall, r.ExitSpeed, r.Angle, r.Distance, r.TaggedHitType,
            r.PlateLocSide, r.PlateLocHeight, r.BatterSide, r.Direction, r.KorBB,
            r.GameID || '_' || r.Inning || '_' || r.PAofInning || '_' || {_bnorm} AS pa_id,
            COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT}) AS zone_bot,
            COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP}) AS zone_top,
            CASE WHEN r.PlateLocSide IS NOT NULL AND r.PlateLocHeight IS NOT NULL
                  AND ABS(r.PlateLocSide) <= {ZONE_SIDE}
                  AND r.PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                  AND r.PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP})
                 THEN 1 ELSE 0 END AS is_iz,
            CASE WHEN r.PlateLocSide IS NOT NULL AND r.PlateLocHeight IS NOT NULL
                  AND NOT (ABS(r.PlateLocSide) <= {ZONE_SIDE}
                           AND r.PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                           AND r.PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP}))
                 THEN 1 ELSE 0 END AS is_oz
        FROM trackman r
        LEFT JOIN batter_zones bz ON r.Season = bz.Season AND {_bnorm} = bz.batter_name
        WHERE r.Batter IS NOT NULL
    ),
    pitch_agg AS (
        SELECT
            Season,
            Batter, BatterTeam,
            COUNT(*) AS n_pitches,
            COUNT(DISTINCT pa_id) AS PA,
            COUNT(DISTINCT CASE WHEN KorBB='Strikeout' THEN pa_id END) AS ks,
            COUNT(DISTINCT CASE WHEN KorBB='Walk' THEN pa_id END) AS bbs,
            SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS swings,
            SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) AS whiffs,
            SUM(CASE WHEN PitchCall IN {_CONTACT_CALLS_SQL} THEN 1 ELSE 0 END) AS contacts,
            SUM(CASE WHEN {_HAS_LOC} THEN 1 ELSE 0 END) AS n_loc,
            SUM(is_iz) AS iz_count,
            SUM(is_oz) AS oz_count,
            SUM(CASE WHEN is_iz = 1 AND PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS iz_swings,
            SUM(CASE WHEN is_oz = 1 AND PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS oz_swings,
            SUM(CASE WHEN is_iz = 1 AND PitchCall IN {_CONTACT_CALLS_SQL} THEN 1 ELSE 0 END) AS iz_contacts,
            SUM(CASE WHEN is_oz = 1 AND PitchCall IN {_CONTACT_CALLS_SQL} THEN 1 ELSE 0 END) AS oz_contacts,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN 1 ELSE 0 END) AS bbe,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=95 THEN 1 ELSE 0 END) AS hard_hits,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND Angle BETWEEN 8 AND 32 THEN 1 ELSE 0 END) AS sweet_spots,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed ELSE 0 END) AS sum_ev,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN 1 ELSE 0 END) AS n_ev,
            MAX(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) AS max_ev,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN Angle ELSE 0 END) AS sum_la,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN Distance ELSE 0 END) AS sum_dist,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=98
                AND Angle >= GREATEST(26 - 2*(ExitSpeed-98), 8)
                AND Angle <= LEAST(30 + 3*(ExitSpeed-98), 50) THEN 1 ELSE 0 END) AS Barrels,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='GroundBall' THEN 1 ELSE 0 END) AS gb,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='FlyBall' THEN 1 ELSE 0 END) AS fb,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='LineDrive' THEN 1 ELSE 0 END) AS ld,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND TaggedHitType='Popup' THEN 1 ELSE 0 END) AS pu
        FROM raw
        GROUP BY Season, Batter, BatterTeam
    )
    SELECT * FROM pitch_agg
    """
    con.execute(sql)


def _create_pitcher_stats_pop(con):
    _pnorm = _name_case_sql("Pitcher")
    _bnorm_p = _name_case_sql("Batter")
    sql = f"""
    CREATE OR REPLACE TABLE pitcher_stats_pop AS
    WITH batter_zones AS (
        SELECT Season, {_bnorm_p} AS batter_name,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_BOT} END AS zone_bot,
            CASE WHEN COUNT(*) >= {MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE}
                 THEN ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY PlateLocHeight), 3)
                 ELSE {ZONE_HEIGHT_TOP} END AS zone_top
        FROM trackman
        WHERE PitchCall = 'StrikeCalled'
          AND PlateLocHeight IS NOT NULL
          AND Batter IS NOT NULL
        GROUP BY Season, {_bnorm_p}
    ),
    raw AS (
        SELECT
            r.Season,
            {_pnorm} AS Pitcher, r.PitcherTeam,
            r.PitchCall, r.ExitSpeed, r.Angle, r.TaggedHitType, r.TaggedPitchType,
            r.PlateLocSide, r.PlateLocHeight, r.RelSpeed, r.SpinRate, r.Extension, r.KorBB,
            r.GameID || '_' || r.Inning || '_' || r.PAofInning || '_' || {_bnorm_p} AS pa_id,
            CASE WHEN r.PlateLocSide IS NOT NULL AND r.PlateLocHeight IS NOT NULL
                  AND ABS(r.PlateLocSide) <= {ZONE_SIDE}
                  AND r.PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                  AND r.PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP})
                 THEN 1 ELSE 0 END AS is_iz,
            CASE WHEN r.PlateLocSide IS NOT NULL AND r.PlateLocHeight IS NOT NULL
                  AND NOT (ABS(r.PlateLocSide) <= {ZONE_SIDE}
                           AND r.PlateLocHeight >= COALESCE(bz.zone_bot, {ZONE_HEIGHT_BOT})
                           AND r.PlateLocHeight <= COALESCE(bz.zone_top, {ZONE_HEIGHT_TOP}))
                 THEN 1 ELSE 0 END AS is_oz
        FROM trackman r
        LEFT JOIN batter_zones bz ON r.Season = bz.Season AND {_bnorm_p} = bz.batter_name
        WHERE r.Pitcher IS NOT NULL
    ),
    pitch_agg AS (
        SELECT
            Season,
            Pitcher, PitcherTeam,
            COUNT(*) AS Pitches,
            COUNT(DISTINCT pa_id) AS PA,
            COUNT(DISTINCT CASE WHEN KorBB='Strikeout' THEN pa_id END) AS ks,
            COUNT(DISTINCT CASE WHEN KorBB='Walk' THEN pa_id END) AS bbs,
            SUM(CASE WHEN PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS swings,
            SUM(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) AS whiffs,
            SUM(CASE WHEN {_HAS_LOC} THEN 1 ELSE 0 END) AS n_loc,
            SUM(is_iz) AS iz_count,
            SUM(is_oz) AS oz_count,
            SUM(CASE WHEN is_iz = 1 AND PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS iz_swings,
            SUM(CASE WHEN is_oz = 1 AND PitchCall IN {_SWING_CALLS_SQL} THEN 1 ELSE 0 END) AS oz_swings,
            SUM(CASE WHEN is_iz = 1 AND PitchCall IN {_CONTACT_CALLS_SQL} THEN 1 ELSE 0 END) AS iz_contacts,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN 1 ELSE 0 END) AS bbe,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=95 THEN 1 ELSE 0 END) AS hard_hits,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed ELSE 0 END) AS sum_ev_against,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL THEN 1 ELSE 0 END) AS n_ev_against,
            SUM(CASE WHEN PitchCall='InPlay' AND ExitSpeed IS NOT NULL AND ExitSpeed>=98
                AND Angle >= GREATEST(26 - 2*(ExitSpeed-98), 8)
                AND Angle <= LEAST(30 + 3*(ExitSpeed-98), 50) THEN 1 ELSE 0 END) AS n_barrels,
            SUM(CASE WHEN PitchCall='InPlay' AND TaggedHitType='GroundBall' THEN 1 ELSE 0 END) AS gb,
            SUM(CASE WHEN PitchCall='InPlay' THEN 1 ELSE 0 END) AS n_ip,
            SUM(CASE WHEN TaggedPitchType IN ('Fastball','Sinker','Cutter') AND RelSpeed IS NOT NULL THEN RelSpeed ELSE 0 END) AS sum_fb_velo,
            SUM(CASE WHEN TaggedPitchType IN ('Fastball','Sinker','Cutter') AND RelSpeed IS NOT NULL THEN 1 ELSE 0 END) AS n_fb_velo,
            MAX(CASE WHEN TaggedPitchType IN ('Fastball','Sinker','Cutter') AND RelSpeed IS NOT NULL THEN RelSpeed END) AS max_fb_velo,
            SUM(CASE WHEN SpinRate IS NOT NULL THEN SpinRate ELSE 0 END) AS sum_spin,
            SUM(CASE WHEN SpinRate IS NOT NULL THEN 1 ELSE 0 END) AS n_spin,
            SUM(CASE WHEN Extension IS NOT NULL THEN Extension ELSE 0 END) AS sum_ext,
            SUM(CASE WHEN Extension IS NOT NULL THEN 1 ELSE 0 END) AS n_ext
        FROM raw
        GROUP BY Season, Pitcher, PitcherTeam
    )
    SELECT * FROM pitch_agg
    """
    con.execute(sql)


def _create_stuff_baselines(con):
    base_cols = ["RelSpeed", "InducedVertBreak", "HorzBreak", "Extension", "VertApprAngle", "SpinRate"]
    agg_exprs = []
    for col in base_cols:
        agg_exprs.append(f"AVG({col}) AS {col}_mean")
        agg_exprs.append(f"STDDEV({col}) AS {col}_std")
    agg_str = ", ".join(agg_exprs)

    sql = f"""
        SELECT pt_norm AS TaggedPitchType, {agg_str}, COUNT(*) as n
        FROM (
            SELECT *,
                CASE TaggedPitchType
                    WHEN 'FourSeamFastBall' THEN 'Fastball'
                    WHEN 'OneSeamFastBall' THEN 'Sinker'
                    WHEN 'TwoSeamFastBall' THEN 'Sinker'
                    WHEN 'ChangeUp' THEN 'Changeup'
                    ELSE TaggedPitchType
                END AS pt_norm
            FROM trackman
            WHERE TaggedPitchType NOT IN ('Other','Undefined','Knuckleball')
              AND RelSpeed IS NOT NULL
        )
        GROUP BY pt_norm
    """
    df = con.execute(sql).fetchdf()
    baseline_rows = []
    baseline_stats = {}
    for _, row in df.iterrows():
        pt = row["TaggedPitchType"]
        stats = {}
        for col in base_cols:
            m, s = row.get(f"{col}_mean"), row.get(f"{col}_std")
            if pd.notna(m) and pd.notna(s):
                stats[col] = (float(m), float(s))
                baseline_rows.append({
                    "TaggedPitchType": pt,
                    "metric": col,
                    "mean": float(m),
                    "std": float(s),
                })
        baseline_stats[pt] = stats

    baseline_df = pd.DataFrame(baseline_rows, columns=["TaggedPitchType", "metric", "mean", "std"])
    con.register("baseline_df", baseline_df)
    con.execute("CREATE OR REPLACE TABLE stuff_baselines AS SELECT * FROM baseline_df")

    fb_sql = """
        SELECT Pitcher, AVG(RelSpeed) AS fb_velo
        FROM trackman
        WHERE TaggedPitchType IN ('Fastball','Sinker','Cutter','FourSeamFastBall','TwoSeamFastBall','OneSeamFastBall')
          AND RelSpeed IS NOT NULL
        GROUP BY Pitcher
    """
    fb_df = con.execute(fb_sql).fetchdf()
    con.register("fb_df", fb_df)
    con.execute("CREATE OR REPLACE TABLE fb_velo_by_pitcher AS SELECT * FROM fb_df")
    fb_velo_by_pitcher = dict(zip(fb_df["Pitcher"], fb_df["fb_velo"]))

    velo_diff_stats = {}
    vd_rows = []
    for pt in ["Changeup", "Splitter"]:
        raw_types = [pt]
        if pt == "Changeup":
            raw_types = ["Changeup", "ChangeUp"]
        vd_sql = f"""
            SELECT t.Pitcher, t.RelSpeed, fb.fb_velo
            FROM trackman t
            JOIN (
                SELECT Pitcher, AVG(RelSpeed) AS fb_velo
                FROM trackman
                WHERE TaggedPitchType IN ('Fastball','Sinker','Cutter','FourSeamFastBall','TwoSeamFastBall','OneSeamFastBall')
                  AND RelSpeed IS NOT NULL
                GROUP BY Pitcher
            ) fb ON t.Pitcher = fb.Pitcher
            WHERE t.TaggedPitchType IN ({",".join(f"'{r}'" for r in raw_types)}) AND t.RelSpeed IS NOT NULL
        """
        vd_df = con.execute(vd_sql).fetchdf()
        if len(vd_df) > 2:
            vd = vd_df["fb_velo"] - vd_df["RelSpeed"]
            velo_diff_stats[pt] = (float(vd.mean()), float(vd.std()))
            vd_rows.append({"TaggedPitchType": pt, "mean": float(vd.mean()), "std": float(vd.std())})

    vd_stats_df = pd.DataFrame(vd_rows, columns=["TaggedPitchType", "mean", "std"])
    con.register("vd_df", vd_stats_df)
    con.execute("CREATE OR REPLACE TABLE velo_diff_stats AS SELECT * FROM vd_df")

    return {
        "baseline_stats": baseline_stats,
        "fb_velo_by_pitcher": fb_velo_by_pitcher,
        "velo_diff_stats": velo_diff_stats,
    }


def _create_davidson_data(con, parquet_path):
    _pname = _name_case_sql("Pitcher")
    _bname = _name_case_sql("Batter")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE davidson_data AS
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
        WHERE (PitcherTeam = '{DAVIDSON_TEAM_ID}' OR BatterTeam = '{DAVIDSON_TEAM_ID}')
          AND (PitchCall IS NULL OR PitchCall != 'Undefined')
        QUALIFY
            ROW_NUMBER() OVER (
                PARTITION BY GameID, Inning, PAofInning, PitchofPA, Pitcher, Batter, PitchNo
                ORDER BY PitchNo
            ) = 1
        """
    )


def _attach_stuff_plus(con, baselines_dict):
    df = con.execute("SELECT * FROM davidson_data").fetchdf()
    if df.empty:
        return
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Season" in df.columns:
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")

    uid = (
        df["GameID"].astype(str) + "_" +
        df["Inning"].astype(str) + "_" +
        df["PAofInning"].astype(str) + "_" +
        df["PitchofPA"].astype(str) + "_" +
        df["Pitcher"].astype(str) + "_" +
        df["Batter"].astype(str) + "_" +
        df["PitchNo"].astype(str)
    )
    df["PitchUID"] = uid

    df = _compute_stuff_plus(df, baselines_dict=baselines_dict)
    con.register("dav_df", df)
    con.execute("CREATE OR REPLACE TABLE davidson_data AS SELECT * FROM dav_df")

    stuff_df = df[["PitchUID", "Pitcher", "Season", "Date", "TaggedPitchType", "StuffPlus"]].copy()
    con.register("stuff_df", stuff_df)
    con.execute("CREATE OR REPLACE TABLE stuff_plus AS SELECT * FROM stuff_df")


def _create_tunnel_population(con):
    pop = _build_tunnel_population_pop(con=con)
    rows = []
    for pair, scores in pop.items():
        for score in scores:
            rows.append({"pair_type": pair, "score": float(score)})
    pop_df = pd.DataFrame(rows, columns=["pair_type", "score"])
    con.register("pop_df", pop_df)
    con.execute("CREATE OR REPLACE TABLE tunnel_population AS SELECT * FROM pop_df")


def _create_tunnel_pair_outcomes(con):
    con.execute(
        """
        CREATE OR REPLACE TABLE tunnel_pair_outcomes AS
        WITH ordered AS (
            SELECT GameID, Inning, PAofInning, Batter, Pitcher, PitchofPA,
                   TaggedPitchType, PitchCall
            FROM trackman
            WHERE TaggedPitchType IS NOT NULL AND PitchCall IS NOT NULL
        ),
        pairs AS (
            SELECT
                CASE
                    WHEN TaggedPitchType < prev_type THEN TaggedPitchType || '/' || prev_type
                    ELSE prev_type || '/' || TaggedPitchType
                END AS pair_type,
                PitchCall
            FROM (
                SELECT *,
                       LAG(TaggedPitchType) OVER (
                           PARTITION BY GameID, Inning, PAofInning, Batter, Pitcher
                           ORDER BY PitchofPA
                       ) AS prev_type
                FROM ordered
            )
            WHERE prev_type IS NOT NULL AND TaggedPitchType != prev_type
        ),
        agg AS (
            SELECT pair_type,
                   COUNT(*) AS n_pairs,
                   AVG(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) * 100 AS whiff_rate
            FROM pairs
            GROUP BY pair_type
        ),
        global_row AS (
            SELECT '__ALL__' AS pair_type,
                   COUNT(*) AS n_pairs,
                   AVG(CASE WHEN PitchCall='StrikeSwinging' THEN 1 ELSE 0 END) * 100 AS whiff_rate
            FROM pairs
        )
        SELECT * FROM agg
        UNION ALL
        SELECT * FROM global_row
        """
    )

    # Benchmarks
    if os.path.exists(TUNNEL_BENCH_PATH):
        with open(TUNNEL_BENCH_PATH, "r", encoding="utf-8") as f:
            blob = json.load(f)
        benchmarks = blob.get("benchmarks", {})
        bench_rows = []
        for pair, vals in benchmarks.items():
            if len(vals) < 7:
                continue
            bench_rows.append({
                "pair_type": pair,
                "p10": vals[0],
                "p25": vals[1],
                "p50": vals[2],
                "p75": vals[3],
                "p90": vals[4],
                "mean": vals[5],
                "std": vals[6],
            })
        bench_df = pd.DataFrame(
            bench_rows,
            columns=["pair_type", "p10", "p25", "p50", "p75", "p90", "mean", "std"],
        )
        con.register("bench_df", bench_df)
        con.execute("CREATE OR REPLACE TABLE tunnel_benchmarks AS SELECT * FROM bench_df")

    # Weights
    if os.path.exists(TUNNEL_WEIGHTS_PATH):
        with open(TUNNEL_WEIGHTS_PATH, "r", encoding="utf-8") as f:
            blob = json.load(f)
        weights = blob.get("weights", {})
        weights_df = pd.DataFrame([{"id": 1, "weights_json": json.dumps(weights)}])
        con.register("weights_df", weights_df)
        con.execute("CREATE OR REPLACE TABLE tunnel_weights AS SELECT * FROM weights_df")


def run(parquet_path, db_path, overwrite=False):
    _ensure_parquet(parquet_path)
    con = _connect(db_path, overwrite=overwrite)
    _create_trackman_view(con, parquet_path)
    _create_meta(con, parquet_path)
    _create_seasons_and_sidebar(con)
    _create_trackman_pop(con)
    _create_batter_stats_pop(con)
    _create_pitcher_stats_pop(con)
    baselines = _create_stuff_baselines(con)
    _create_davidson_data(con, parquet_path)
    _attach_stuff_plus(con, baselines)
    _create_tunnel_population(con)
    _create_tunnel_pair_outcomes(con)
    con.close()


def main():
    parser = argparse.ArgumentParser(description="Build davidson.duckdb precompute database.")
    parser.add_argument("--parquet", default=PARQUET_PATH, help="Path to Trackman parquet file")
    parser.add_argument("--out", default=DUCKDB_PATH, help="Output DuckDB path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing DB file")
    args = parser.parse_args()

    run(args.parquet, args.out, overwrite=args.overwrite)
    print(f"Precompute complete: {args.out}")


if __name__ == "__main__":
    main()
