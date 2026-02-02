"""Data loading -- DuckDB connection, Davidson data, TrueMedia, helpers."""

import os
import json

import numpy as np
import pandas as pd
import duckdb
import streamlit as st
from scipy.stats import percentileofscore

from config import (
    PARQUET_PATH,
    DUCKDB_PATH,
    DAVIDSON_TEAM_ID,
    NAME_MAP,
    _name_case_sql,
    _normalize_hand,
    normalize_pitch_types,
    ROSTER_2026,
    POSITION,
    _APP_DIR,
)


# ──────────────────────────────────────────────
# DUCKDB CONNECTION
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_duckdb_con():
    """Return a DuckDB connection with a VIEW over the parquet dataset."""
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")
    con = duckdb.connect(database=':memory:')
    # Use a VIEW to avoid loading the full parquet into memory at startup.
    _pname = _name_case_sql("Pitcher")
    _bname = _name_case_sql("Batter")
    con.execute(
        f"""
        CREATE VIEW trackman AS
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
        FROM read_parquet('{PARQUET_PATH}')
        WHERE PitchCall IS NULL OR PitchCall != 'Undefined'
        QUALIFY
            ROW_NUMBER() OVER (
                PARTITION BY GameID, Inning, PAofInning, PitchofPA, Pitcher, Batter, PitchNo
                ORDER BY PitchNo
            ) = 1
        """
    )
    return con


@st.cache_resource(show_spinner=False)
def get_precompute_con():
    """Return read-only DuckDB connection to the precomputed database, if present."""
    if not os.path.exists(DUCKDB_PATH):
        return None
    try:
        return duckdb.connect(DUCKDB_PATH, read_only=True)
    except Exception:
        return None


def _precompute_table_exists(table_name):
    con = get_precompute_con()
    if con is None:
        return False
    try:
        tables = con.execute("SHOW TABLES").fetchdf()
        if tables.empty:
            return False
        names = tables["name"].astype(str).str.lower().tolist()
        return table_name.lower() in names
    except Exception:
        return False


def _read_precompute_table(table_name, where=None, columns=None):
    con = get_precompute_con()
    if con is None:
        return pd.DataFrame()
    cols = ", ".join(columns) if columns else "*"
    sql = f"SELECT {cols} FROM {table_name}"
    if where:
        sql += f" WHERE {where}"
    try:
        return con.execute(sql).fetchdf()
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────
# QUERY HELPERS
# ──────────────────────────────────────────────

def query_population(sql):
    """Run an ad-hoc SQL query against the full D1 parquet via DuckDB."""
    return get_duckdb_con().execute(sql).fetchdf()


@st.cache_data(show_spinner="Loading Davidson data...")
def load_davidson_data():
    """Load only Davidson rows from parquet into pandas (~300k rows)."""
    data = pd.DataFrame()
    if _precompute_table_exists("davidson_data"):
        data = _read_precompute_table("davidson_data")
    if data.empty:
        if not os.path.exists(PARQUET_PATH):
            return pd.DataFrame()
        sql = f"""
            SELECT * FROM read_parquet('{PARQUET_PATH}')
            WHERE (PitcherTeam = '{DAVIDSON_TEAM_ID}' OR BatterTeam = '{DAVIDSON_TEAM_ID}')
              AND (PitchCall IS NULL OR PitchCall != 'Undefined')
        """
        data = duckdb.query(sql).fetchdf()
        data = data.drop_duplicates(
            subset=["GameID", "Inning", "PAofInning", "PitchofPA", "Pitcher", "Batter", "PitchNo"]
        )
        for col in ["Pitcher", "Batter"]:
            if col in data.columns:
                s = data[col].astype(str).str.strip()
                s = s.str.replace(r"\s+", " ", regex=True)
                s = s.str.replace(r"\s+,", ",", regex=True)
                s = s.str.replace(r",\s*", ", ", regex=True)
                data[col] = s.replace(NAME_MAP)
        for col in ["BatterSide", "PitcherThrows"]:
            if col in data.columns:
                data[col] = _normalize_hand(data[col])
        data = normalize_pitch_types(data)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    if "Season" in data.columns:
        data["Season"] = pd.to_numeric(data["Season"], errors="coerce").astype("Int64")
    return data


@st.cache_data(show_spinner=False)
def get_all_seasons():
    """Return sorted list of all seasons in the full D1 database."""
    if _precompute_table_exists("seasons"):
        df = _read_precompute_table("seasons")
    else:
        df = query_population("SELECT DISTINCT Season FROM trackman WHERE Season IS NOT NULL AND Season > 0 ORDER BY Season")
    if df.empty or "Season" not in df.columns:
        return []
    return sorted(df["Season"].dropna().astype(int).tolist())


@st.cache_data(show_spinner=False)
def get_sidebar_stats():
    """Return sidebar aggregate counts from full D1 database via DuckDB."""
    if _precompute_table_exists("sidebar_stats"):
        df = _read_precompute_table("sidebar_stats")
        if df.empty:
            row = {
                "total_pitches": 0,
                "n_seasons": 0,
                "min_season": 0,
                "max_season": 0,
                "n_pitchers": 0,
                "n_batters": 0,
                "n_dav_games": 0,
            }
        else:
            row = df.iloc[0]
    else:
        row = query_population(f"""
            SELECT
                COUNT(*) as total_pitches,
                COUNT(DISTINCT CASE WHEN Season > 0 THEN Season END) as n_seasons,
                MIN(CASE WHEN Season > 0 THEN Season END) as min_season,
                MAX(Season) as max_season,
                COUNT(DISTINCT Pitcher) as n_pitchers,
                COUNT(DISTINCT Batter) as n_batters,
                COUNT(DISTINCT CASE WHEN PitcherTeam = '{DAVIDSON_TEAM_ID}' OR BatterTeam = '{DAVIDSON_TEAM_ID}' THEN GameID END) as n_dav_games
            FROM trackman
        """).iloc[0]
    def _safe_int(v, default=0):
        return int(v) if pd.notna(v) else default
    return {
        "total_pitches": _safe_int(row["total_pitches"]),
        "n_seasons": _safe_int(row["n_seasons"]),
        "min_season": _safe_int(row["min_season"]),
        "max_season": _safe_int(row["max_season"]),
        "n_pitchers": _safe_int(row["n_pitchers"]),
        "n_batters": _safe_int(row["n_batters"]),
        "n_dav_games": _safe_int(row["n_dav_games"]),
    }


# ──────────────────────────────────────────────
# TRUEMEDIA DATA LOADER
# ──────────────────────────────────────────────

def _pct_to_float(s):
    """Convert '42.8%' -> 42.8 (float). Leaves non-% values unchanged."""
    if isinstance(s, str) and s.endswith("%"):
        try:
            return float(s.rstrip("%"))
        except ValueError:
            return None
    return s


def _clean_pct_cols(df):
    """Strip '%' suffix and coerce numeric-looking object columns."""
    skip = {"playerId", "abbrevName", "playerFullName", "player", "playerFirstName",
            "pos", "newestTeamName", "newestTeamAbbrevName", "newestTeamId",
            "newestTeamLocation", "newestTeamLevel", "batsHand", "throwsHand"}
    for col in df.select_dtypes(include="object").columns:
        if col in skip:
            continue
        sample = df[col].dropna().head(20)
        if sample.empty:
            continue
        if sample.astype(str).str.endswith("%").mean() > 0.5:
            df[col] = df[col].apply(_pct_to_float)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # Try to coerce numeric-looking strings (e.g. "86.9", "2300")
            converted = pd.to_numeric(sample, errors="coerce")
            if converted.notna().mean() > 0.5:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def _load_truemedia():
    """Load all TrueMedia CSV files into a structured dict."""
    hit_dir = os.path.join(_APP_DIR, "truemedia_hitting")
    pit_dir = os.path.join(_APP_DIR, "truemedia_pitching_new")

    def _read(directory, filename):
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path)
        return _clean_pct_cols(df)

    tm = {
        "hitting": {
            "rate": _read(hit_dir, "Rate.csv"),
            "counting": _read(hit_dir, "Counting.csv"),
            "exit": _read(hit_dir, "Exit Data.csv"),
            "expected_rate": _read(hit_dir, "Expected Rate.csv"),
            "expected_hit_rates": _read(hit_dir, "Expected Hit Rates.csv"),
            "hit_types": _read(hit_dir, "Hit Types.csv"),
            "hit_locations": _read(hit_dir, "Hit Locations.csv"),
            "pitch_rates": _read(hit_dir, "Pitch Rates.csv"),
            "pitch_types": _read(hit_dir, "Pitch Types.csv"),
            "pitch_locations": _read(hit_dir, "Pitch Locations.csv"),
            "pitch_counts": _read(hit_dir, "Pitch Counts.csv"),
            "pitch_type_counts": _read(hit_dir, "Pitch Type Counts.csv"),
            "pitch_calls": _read(hit_dir, "Pitch Calls.csv"),
            "speed": _read(hit_dir, "Speed Score.csv"),
            "stolen_bases": _read(hit_dir, "Stolen Bases.csv"),
            "home_runs": _read(hit_dir, "Home Runs.csv"),
            "run_expectancy": _read(hit_dir, "Run Expectancy.csv"),
            "swing_pct": _read(hit_dir, "1P Swing%.csv"),
            "swing_stats": _read(hit_dir, "Swing stats.csv"),
        },
        "pitching": {
            "traditional": _read(pit_dir, "Copy of Traditional.csv"),
            "rate": _read(pit_dir, "Rate-2.csv"),
            "movement": _read(pit_dir, "Movement-2.csv"),
            "pitch_types": _read(pit_dir, "Pitch Types-2.csv"),
            "pitch_rates": _read(pit_dir, "Pitch Rates-2.csv"),
            "exit": _read(pit_dir, "Exit Data-2.csv"),
            "expected_rate": _read(pit_dir, "Expected Rate-2.csv"),
            "hit_types": _read(pit_dir, "Hit Types-2.csv"),
            "hit_locations": _read(pit_dir, "Hit Locations-2.csv"),
            "counting": _read(pit_dir, "Counting.csv"),
            "pitch_counts": _read(pit_dir, "Pitch Counts-2.csv"),
            "pitch_locations": _read(pit_dir, "Pitch Locations-2.csv"),
            "baserunning": _read(pit_dir, "Baserunning pitching.csv"),
            "stolen_bases": _read(pit_dir, "Stolen Bases.csv"),
            "home_runs": _read(pit_dir, "Home Runs-2.csv"),
            "expected_hit_rates": _read(pit_dir, "Expected Hit Rates-2.csv"),
            "pitch_calls": _read(pit_dir, "Pitch Calls-2.csv"),
            "pitch_type_counts": _read(pit_dir, "Pitch Type Counts.csv"),
            "expected_counting": _read(pit_dir, "Expected Counting.csv"),
            "pitching_counting": _read(pit_dir, "Pitching Counting.csv"),
            "bids": _read(pit_dir, "Bids pitching-2.csv"),
        },
        "catching": {
            "defense": _read(_APP_DIR, "Catcher Defense.csv"),
            "framing": _read(_APP_DIR, "Catcher Framing.csv"),
            "opposing": _read(_APP_DIR, "Catcher Opposing Batters.csv"),
            "pitch_rates": _read(_APP_DIR, "Catcher Pitch Rates.csv"),
            "pitch_types_rates": _read(_APP_DIR, "Catcher Pitch Types Rates.csv"),
            "pitch_types": _read(_APP_DIR, "Catcher Pitch Types.csv"),
            "throws": _read(pit_dir, "All Tracked Throws.csv"),
            "sba2_throws": _read(pit_dir, "SBA2 Tracked Throws.csv"),
            "pickoffs": _read(pit_dir, "Pickoffs.csv"),
            "pb_wp": _read(pit_dir, "Passed Balls & Wild Pitches.csv"),
            "pitch_counts": _read(_APP_DIR, "Catcher Pitch Counts.csv"),
        },
    }
    return tm


# ──────────────────────────────────────────────
# TRUEMEDIA HELPERS
# ──────────────────────────────────────────────

def _tm_team(df, team):
    """Filter a TrueMedia dataframe to a specific team."""
    if df.empty or "newestTeamName" not in df.columns:
        return pd.DataFrame()
    return df[df["newestTeamName"] == team].copy()


def _tm_player(df, name):
    """Filter a TrueMedia dataframe to a specific player by full name."""
    if df.empty or "playerFullName" not in df.columns:
        return pd.DataFrame()
    return df[df["playerFullName"] == name]


def _safe_val(df, col, fmt=".1f", default="-"):
    """Safely get a single value from a 1-row df."""
    if df.empty or col not in df.columns:
        return default
    v = df.iloc[0][col]
    if pd.isna(v):
        return default
    try:
        v = float(v)
    except (ValueError, TypeError):
        return str(v)
    if isinstance(fmt, str) and "d" in fmt:
        return f"{int(v)}"
    return f"{v:{fmt}}"


def _safe_pct(df, col, default="-"):
    """Safely get a percentage value."""
    if df.empty or col not in df.columns:
        return default
    v = df.iloc[0][col]
    if pd.isna(v):
        return default
    try:
        v = float(v)
    except (ValueError, TypeError):
        return str(v)
    return f"{v:.1f}%"


def _safe_num(df, col, default=np.nan):
    """Safely get a numeric value (for percentile calculations)."""
    if df.empty or col not in df.columns:
        return default
    v = df.iloc[0][col]
    if pd.isna(v):
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _tm_pctile(player_df, col, all_df, col_all=None):
    """Compute percentile of a player's value vs all D1 players in that column.
    Returns float 0-100 or np.nan."""
    val = _safe_num(player_df, col)
    if pd.isna(val):
        return np.nan
    c = col_all or col
    if c not in all_df.columns:
        return np.nan
    series = pd.to_numeric(all_df[c], errors="coerce").dropna()
    if series.empty:
        return np.nan
    return percentileofscore(series, val, kind='rank')


# ──────────────────────────────────────────────
# NARRATIVE GENERATORS
# ──────────────────────────────────────────────

def _hitter_narrative(name, rate, exit_d, pr, ht, hl, spd, all_h_rate, all_h_exit, all_h_pr):
    """Generate a short narrative paragraph describing a hitter's profile."""
    lines = []
    # Overall quality
    ops = _safe_num(rate, "OPS")
    if not pd.isna(ops):
        ops_pct = _tm_pctile(rate, "OPS", all_h_rate)
        if ops_pct >= 80:
            lines.append(f"**{name} is an elite offensive threat** (OPS {ops:.3f}, {int(ops_pct)}th percentile among D1 hitters).")
        elif ops_pct >= 60:
            lines.append(f"**{name} is an above-average hitter** (OPS {ops:.3f}, {int(ops_pct)}th percentile).")
        elif ops_pct >= 40:
            lines.append(f"**{name} is an average producer** at the plate (OPS {ops:.3f}, {int(ops_pct)}th percentile).")
        else:
            lines.append(f"**{name} has struggled offensively** (OPS {ops:.3f}, {int(ops_pct)}th percentile).")

    # Power vs contact profile
    ev = _safe_num(exit_d, "ExitVel")
    barrel = _safe_num(exit_d, "Barrel%")
    k_pct = _safe_num(rate, "K%")
    bb_pct = _safe_num(rate, "BB%")
    if not pd.isna(ev) and not pd.isna(barrel):
        ev_pct = _tm_pctile(exit_d, "ExitVel", all_h_exit)
        brl_pct = _tm_pctile(exit_d, "Barrel%", all_h_exit)
        if ev_pct >= 75 and brl_pct >= 75:
            lines.append(f"He hits the ball exceptionally hard ({ev:.1f} mph EV, {int(ev_pct)}th pctile) with a high barrel rate ({barrel:.1f}%, {int(brl_pct)}th pctile) — a true damage dealer.")
        elif ev_pct >= 60:
            lines.append(f"Solid bat speed ({ev:.1f} mph EV, {int(ev_pct)}th pctile) with {barrel:.1f}% barrel rate.")
        elif ev_pct <= 30:
            lines.append(f"Below-average exit velocity ({ev:.1f} mph, {int(ev_pct)}th pctile) — limited hard contact ability.")

    # Discipline
    chase = _safe_num(pr, "Chase%")
    contact = _safe_num(pr, "Contact%")
    if not pd.isna(chase) and not pd.isna(k_pct):
        chase_pct = _tm_pctile(pr, "Chase%", all_h_pr)
        if chase >= 33 and k_pct >= 25:
            lines.append(f"Highly exploitable plate discipline — chases {chase:.1f}% of pitches out of the zone and strikes out {k_pct:.1f}% of the time. **Expand off the plate.**")
        elif chase <= 20 and bb_pct is not None and not pd.isna(bb_pct) and bb_pct >= 10:
            lines.append(f"Very disciplined eye — only chases {chase:.1f}% with a {bb_pct:.1f}% walk rate. Must compete with strikes.")
        elif chase >= 28:
            lines.append(f"Tends to chase ({chase:.1f}%) — breaking balls off the plate can be effective.")

    # Spray / tendency
    pull = _safe_num(hl, "HPull%")
    oppo = _safe_num(hl, "HOppFld%")
    gb = _safe_num(ht, "Ground%")
    fb = _safe_num(ht, "Fly%")
    if not pd.isna(pull) and not pd.isna(gb):
        if pull >= 45:
            lines.append(f"Pull-heavy hitter ({pull:.1f}% pull) — shiftable, attack away side.")
        elif oppo is not None and not pd.isna(oppo) and oppo >= 35:
            lines.append(f"Uses the whole field ({oppo:.1f}% oppo) — hard to defend with positioning.")
        if gb >= 50:
            lines.append(f"Ground-ball hitter ({gb:.1f}% GB) — elevating is not his game, keep the ball down to induce weak contact.")
        elif fb is not None and not pd.isna(fb) and fb >= 40:
            lines.append(f"Fly-ball approach ({fb:.1f}% FB) — keep the ball down to avoid damage in the air.")

    # Speed
    spd_score = _safe_num(spd, "SpeedScore")
    if not pd.isna(spd_score):
        if spd_score >= 6.0:
            lines.append(f"Elite speed threat (Speed Score: {spd_score:.1f}) — hold runners, quick pitch delivery.")
        elif spd_score >= 4.5:
            lines.append(f"Above-average runner (Speed Score: {spd_score:.1f}).")

    return " ".join(lines) if lines else f"Limited data available for {name}."


def _pitcher_narrative(name, trad, mov, pr, ht, exit_d, all_p_trad, all_p_mov, all_p_pr):
    """Generate a short narrative paragraph describing a pitcher's profile."""
    lines = []
    # Overall quality
    era = _safe_num(trad, "ERA")
    fip = _safe_num(trad, "FIP")
    ip = _safe_num(trad, "IP")
    if not pd.isna(era) and not pd.isna(ip) and ip >= 10:
        era_pct = _tm_pctile(trad, "ERA", all_p_trad)
        # ERA: lower is better, so invert
        era_rank = 100 - era_pct if not pd.isna(era_pct) else np.nan
        if era_rank >= 80:
            lines.append(f"**{name} is one of the best arms in D1** ({era:.2f} ERA, top {100-int(era_rank)}% among all pitchers with 10+ IP).")
        elif era_rank >= 60:
            lines.append(f"**{name} has been solid on the mound** ({era:.2f} ERA, {int(era_rank)}th percentile).")
        elif era_rank >= 40:
            lines.append(f"**{name} has been average** ({era:.2f} ERA, {int(era_rank)}th percentile).")
        else:
            lines.append(f"**{name} has been hittable** ({era:.2f} ERA, {int(era_rank)}th percentile) — an opportunity for the offense.")

    # Stuff
    vel = _safe_num(mov, "Vel")
    ivb = _safe_num(mov, "IndVertBrk")
    spin = _safe_num(mov, "Spin")
    if not pd.isna(vel):
        vel_pct = _tm_pctile(mov, "Vel", all_p_mov)
        if vel_pct >= 80:
            lines.append(f"High-velo arm ({vel:.1f} mph, {int(vel_pct)}th percentile) — gear up for fastball, shorten swings.")
        elif vel_pct <= 25:
            lines.append(f"Below-average velocity ({vel:.1f} mph, {int(vel_pct)}th percentile) — sit on offspeed and drive mistakes.")
        else:
            lines.append(f"Average velocity ({vel:.1f} mph).")

    # Command
    bb9 = _safe_num(trad, "BB/9")
    chase_pct = _safe_num(pr, "Chase%")
    swstrk = _safe_num(pr, "SwStrk%")
    if not pd.isna(bb9):
        if bb9 >= 4.5:
            lines.append(f"Significant control issues ({bb9:.1f} BB/9) — **take early, work deep counts, let him beat himself.**")
        elif bb9 <= 2.5:
            lines.append(f"Excellent command ({bb9:.1f} BB/9) — swinging early may be more effective than taking pitches.")

    # Whiff / chase
    if not pd.isna(swstrk):
        swstrk_pct = _tm_pctile(pr, "SwStrk%", all_p_pr)
        if swstrk_pct >= 80:
            lines.append(f"Elite swing-and-miss stuff ({swstrk:.1f}% SwStrk, {int(swstrk_pct)}th pctile) — shorten up with 2 strikes, don't chase.")
        elif swstrk_pct <= 25:
            lines.append(f"Low whiff rate ({swstrk:.1f}% SwStrk) — put the ball in play aggressively.")

    # Batted ball
    ev_against = _safe_num(exit_d, "ExitVel")
    gb_pct = _safe_num(ht, "Ground%")
    if not pd.isna(gb_pct):
        if gb_pct >= 50:
            lines.append(f"Ground-ball pitcher ({gb_pct:.1f}% GB) — elevate pitches to do damage.")
        elif gb_pct <= 35:
            lines.append(f"Fly-ball pitcher ({gb_pct:.1f}% GB) — look to hit the ball in the air, potential for home runs.")

    return " ".join(lines) if lines else f"Limited data available for {name}."
