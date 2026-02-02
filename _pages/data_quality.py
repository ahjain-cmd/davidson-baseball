"""Data Quality page."""
import streamlit as st
import pandas as pd
import duckdb

from config import (
    PARQUET_PATH,
    PLATE_SIDE_MAX, PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX,
)


@st.cache_data(show_spinner="Running data quality checks...")
def _data_quality_summary():
    con = duckdb.connect()
    path = PARQUET_PATH

    total = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [path]).fetchone()[0]
    distinct_keys = con.execute(
        """
        SELECT COUNT(DISTINCT GameID || '_' || Inning || '_' || PAofInning || '_' || PitchofPA || '_' ||
                             Pitcher || '_' || Batter || '_' || PitchNo)
        FROM read_parquet(?)
        """,
        [path],
    ).fetchone()[0]

    nulls = con.execute(
        """
        SELECT
          SUM(CASE WHEN Pitcher IS NULL THEN 1 ELSE 0 END) AS Pitcher_null,
          SUM(CASE WHEN Batter IS NULL THEN 1 ELSE 0 END) AS Batter_null,
          SUM(CASE WHEN GameID IS NULL THEN 1 ELSE 0 END) AS GameID_null,
          SUM(CASE WHEN Date IS NULL OR Date = '' THEN 1 ELSE 0 END) AS Date_null,
          SUM(CASE WHEN PitchCall IS NULL THEN 1 ELSE 0 END) AS PitchCall_null,
          SUM(CASE WHEN TaggedPitchType IS NULL THEN 1 ELSE 0 END) AS TaggedPitchType_null,
          SUM(CASE WHEN PlateLocSide IS NULL OR PlateLocHeight IS NULL THEN 1 ELSE 0 END) AS PlateLoc_null,
          SUM(CASE WHEN ExitSpeed IS NULL THEN 1 ELSE 0 END) AS ExitSpeed_null,
          SUM(CASE WHEN Direction IS NULL THEN 1 ELSE 0 END) AS Direction_null,
          SUM(CASE WHEN Distance IS NULL THEN 1 ELSE 0 END) AS Distance_null
        FROM read_parquet(?)
        """,
        [path],
    ).fetchdf()

    invalid_locs = con.execute(
        f"""
        SELECT
          SUM(CASE WHEN PlateLocSide IS NOT NULL AND ABS(PlateLocSide) > {PLATE_SIDE_MAX} THEN 1 ELSE 0 END) AS side_out,
          SUM(CASE WHEN PlateLocHeight IS NOT NULL AND (PlateLocHeight < {PLATE_HEIGHT_MIN} OR PlateLocHeight > {PLATE_HEIGHT_MAX})
              THEN 1 ELSE 0 END) AS height_out
        FROM read_parquet(?)
        """,
        [path],
    ).fetchdf()

    inplay_cov = con.execute(
        """
        SELECT
          COUNT(*) AS inplay,
          SUM(CASE WHEN ExitSpeed IS NOT NULL THEN 1 ELSE 0 END) AS ev_present,
          SUM(CASE WHEN Direction IS NOT NULL THEN 1 ELSE 0 END) AS dir_present,
          SUM(CASE WHEN Distance IS NOT NULL THEN 1 ELSE 0 END) AS dist_present
        FROM read_parquet(?)
        WHERE PitchCall = 'InPlay'
        """,
        [path],
    ).fetchdf()

    direction_outliers = con.execute(
        """
        SELECT SUM(CASE WHEN Direction IS NOT NULL AND ABS(Direction) > 90 THEN 1 ELSE 0 END) AS dir_out
        FROM read_parquet(?)
        """,
        [path],
    ).fetchdf()

    bad_pitchcall = con.execute(
        """
        SELECT PitchCall, COUNT(*) AS c
        FROM read_parquet(?)
        WHERE PitchCall IS NOT NULL AND PitchCall NOT IN (
          'BallCalled','StrikeCalled','InPlay','FoulBall','FoulBallNotFieldable',
          'FoulBallFieldable','StrikeSwinging','HitByPitch','BallIntentional','Undefined'
        )
        GROUP BY PitchCall
        ORDER BY c DESC
        """,
        [path],
    ).fetchdf()

    return {
        "total": total,
        "distinct_keys": distinct_keys,
        "nulls": nulls,
        "invalid_locs": invalid_locs,
        "inplay_cov": inplay_cov,
        "direction_outliers": direction_outliers,
        "bad_pitchcall": bad_pitchcall,
    }


def page_data_quality():
    st.title("Data Quality")
    dq = _data_quality_summary()

    st.subheader("Core Counts")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Rows", f"{dq['total']:,}")
    with c2:
        st.metric("Distinct Pitch Keys", f"{dq['distinct_keys']:,}")
    with c3:
        st.metric("Duplicate Rows", f"{dq['total'] - dq['distinct_keys']:,}")

    st.subheader("Missingness (raw parquet)")
    st.dataframe(dq["nulls"], use_container_width=True)

    st.subheader("Invalid Plate Location")
    st.dataframe(dq["invalid_locs"], use_container_width=True)

    st.subheader("In-Play Coverage")
    st.dataframe(dq["inplay_cov"], use_container_width=True)

    st.subheader("Direction Outliers (|Direction| > 90)")
    st.dataframe(dq["direction_outliers"], use_container_width=True)

    st.subheader("Unexpected PitchCall Values")
    if dq["bad_pitchcall"].empty:
        st.caption("No unexpected PitchCall values found.")
    else:
        st.dataframe(dq["bad_pitchcall"], use_container_width=True)
