"""TrueMedia API client — fetches live opposition data for scouting.

Returns data in the same nested-dict DataFrame structure that the existing
game plan engine (scouting.py) expects, so the entire scoring pipeline
works unchanged.
"""

import json
import logging
import os
import time
from io import StringIO

import pandas as pd
import requests
import streamlit as st

from config import TM_USERNAME, TM_SITENAME, TM_MASTER_TOKEN, CACHE_DIR

# ── Disk cache settings for league percentile data ────────────────────────────
_LEAGUE_CACHE_TTL_HOURS = 24  # How often to refresh cached league data
_LEAGUE_HITTERS_CACHE_FMT = os.path.join(CACHE_DIR, "tm_league_hitters_{season}.parquet")
_LEAGUE_PITCHERS_CACHE_FMT = os.path.join(CACHE_DIR, "tm_league_pitchers_{season}.parquet")


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_is_valid(cache_path, ttl_hours=_LEAGUE_CACHE_TTL_HOURS):
    """Check if a cache file exists and is not expired."""
    if not os.path.exists(cache_path):
        return False
    mtime = os.path.getmtime(cache_path)
    age_hours = (time.time() - mtime) / 3600
    return age_hours < ttl_hours


def _load_league_cache(cache_path):
    """Load league data from disk cache. Returns DataFrame or None."""
    if not _cache_is_valid(cache_path):
        return None
    try:
        return pd.read_parquet(cache_path)
    except Exception:
        return None


def _save_league_cache(df, cache_path):
    """Save league data to disk cache."""
    if df.empty:
        return
    _ensure_cache_dir()
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass  # Fail silently - cache is optional


def clear_league_cache(season_year=None):
    """Clear cached league percentile data. If season_year is None, clears all seasons."""
    import glob
    if season_year is not None:
        paths = [
            _LEAGUE_HITTERS_CACHE_FMT.format(season=season_year),
            _LEAGUE_PITCHERS_CACHE_FMT.format(season=season_year),
        ]
    else:
        paths = glob.glob(os.path.join(CACHE_DIR, "tm_league_*.parquet"))
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def get_league_cache_info(season_year):
    """Get cache status for a season. Returns dict with 'hitters' and 'pitchers' status."""
    import datetime
    info = {}
    for name, fmt in [("hitters", _LEAGUE_HITTERS_CACHE_FMT), ("pitchers", _LEAGUE_PITCHERS_CACHE_FMT)]:
        path = fmt.format(season=season_year)
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            age_hours = (time.time() - mtime) / 3600
            updated = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            info[name] = {"cached": True, "updated": updated, "age_hours": round(age_hours, 1)}
        else:
            info[name] = {"cached": False}
    return info


logger = logging.getLogger(__name__)

_API_BASE = "https://api.trumedianetworks.com/v1"
_DQ_BASE = f"{_API_BASE}/mlbapi/custom/baseball/DirectedQuery"

# ── Column sets for API queries ──────────────────────────────────────────────

# Hitting columns (batter perspective)
_HIT_RATE_COLS = "[PA],[AB],[AVG],[OBP],[SLG],[OPS],[WOBA],[K%],[BB%],[ISO]"
_HIT_COUNTING_COLS = "[PA],[AB],[H],[K],[BB],[HR],[2B],[3B],[SB],[HBP],[SF],[RBI],[R]"
_HIT_DISCIPLINE_COLS = "[Chase%],[SwStrk%],[Contact%],[Swing%],[P/PA],[FPStk%],[InZoneSwing%]"
_HIT_EXIT_COLS = "[ExitVel],[Barrel%],[HardHit%]"
_HIT_BATTEDBALL_COLS = "[Ground%],[Fly%],[Line%],[Popup%]"
_HIT_LOCATION_COLS = "[HPull%],[HCtr%],[HOppFld%],[HFarLft%],[HLftCtr%],[HDeadCtr%],[HRtCtr%],[HFarRt%]"
_HIT_PITCHTYPE_COLS = "[4Seam%],[Slider%],[Curve%],[Change%],[Sink2Seam%],[Cutter%],[Split%],[Sweeper%]"
_HIT_ZONE_COLS = "[High%],[VMid%],[Low%],[Inside%],[HMid%],[Outside%]"

def _dedup_cols(*groups):
    """Merge column groups and remove duplicates while preserving order."""
    seen = set()
    result = []
    for g in groups:
        for col in g.split(","):
            if col not in seen:
                seen.add(col)
                result.append(col)
    return ",".join(result)

_ALL_HIT_COLS = _dedup_cols(
    _HIT_RATE_COLS, _HIT_COUNTING_COLS, _HIT_DISCIPLINE_COLS,
    _HIT_EXIT_COLS, _HIT_BATTEDBALL_COLS, _HIT_LOCATION_COLS,
    _HIT_PITCHTYPE_COLS, _HIT_ZONE_COLS,
)

# Pitching columns (pitcher perspective)
_PIT_TRAD_COLS = "[IP],[ERA],[FIP],[xFIP],[WHIP],[K/9],[BB/9],[HR/9],[G],[GS],[W],[L],[SV],[QS]"
_PIT_RATE_COLS = "[K%|PIT],[BB%|PIT],[LOB%],[OPS|PIT],[WOBA|PIT],[BA|PIT]"
_PIT_MOVEMENT_COLS = "[FBVel],[Spin],[IndVertBrk],[HorzBrk]"
_PIT_DISCIPLINE_COLS = "[Chase%|PIT],[SwStrk%|PIT],[Contact%|PIT],[Swing%|PIT],[FPStk%|PIT],[InZone%|PIT]"
_PIT_EXIT_COLS = "[ExitVel|PIT],[Barrel%|PIT],[HardHit%|PIT]"
_PIT_BATTEDBALL_COLS = "[Ground%|PIT],[Fly%|PIT],[Line%|PIT],[Popup%|PIT]"
_PIT_PITCHTYPE_COLS = "[4Seam%|PIT],[Slider%|PIT],[Curve%|PIT],[Change%|PIT],[Sink2Seam%|PIT],[Cutter%|PIT],[Split%|PIT],[Sweeper%|PIT]"
_PIT_COUNTING_COLS = "[K|PIT],[BB|PIT],[H|PIT],[ER],[BF]"
_PIT_ZONE_COLS = "[High%|PIT],[VMid%|PIT],[Low%|PIT],[Inside%|PIT],[HMid%|PIT],[Outside%|PIT]"

_ALL_PIT_COLS = _dedup_cols(
    _PIT_TRAD_COLS, _PIT_RATE_COLS, _PIT_MOVEMENT_COLS,
    _PIT_DISCIPLINE_COLS, _PIT_EXIT_COLS, _PIT_BATTEDBALL_COLS,
    _PIT_PITCHTYPE_COLS, _PIT_COUNTING_COLS, _PIT_ZONE_COLS,
)


# ── Token management ─────────────────────────────────────────────────────────

def get_temp_token():
    """Get a 24-hour temporary TrueMedia API token. Cached in session state."""
    if "tm_temp_token" in st.session_state and st.session_state["tm_temp_token"]:
        return st.session_state["tm_temp_token"]
    tm_token = TM_MASTER_TOKEN
    if not tm_token:
        try:
            tm_token = st.secrets.get("TM_MASTER_TOKEN", "")
        except Exception:
            tm_token = ""
    if not tm_token:
        if not st.session_state.get("tm_token_warned"):
            st.warning("TrueMedia API token not configured. Set TM_MASTER_TOKEN in the environment or Streamlit secrets.")
            st.session_state["tm_token_warned"] = True
        logger.error("TM_MASTER_TOKEN is not set.")
        return None

    headers = {"Content-Type": "application/json"}
    payload = {
        "username": TM_USERNAME,
        "sitename": TM_SITENAME,
        "token": tm_token,
    }
    try:
        res = requests.post(
            f"{_API_BASE}/siteadmin/api/createTempPBToken",
            headers=headers,
            data=json.dumps(payload),
            timeout=15,
        )
        res.raise_for_status()
        tok = res.json().get("pbTempToken")
        if not tok:
            logger.error("TrueMedia API returned no token: %s", res.text)
            return None
        st.session_state["tm_temp_token"] = tok
        return tok
    except Exception as e:
        logger.error("TrueMedia token request failed: %s", e)
        return None


# ── Low-level query ──────────────────────────────────────────────────────────

def _query_csv(endpoint, params, tok, timeout=30):
    """Run a TrueMedia DirectedQuery CSV endpoint and return a DataFrame."""
    params_clean = {k: v for k, v in params.items() if v is not None and v != ""}
    params_clean["token"] = tok
    # Build URL manually to avoid double-encoding brackets/pipes
    parts = "&".join(f"{k}={v}" for k, v in params_clean.items())
    url = f"{_DQ_BASE}/{endpoint}.csv?{parts}"
    try:
        # Use requests instead of pd.read_csv(url) to avoid SSL cert issues
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        df = pd.read_csv(StringIO(res.text))
        return df
    except Exception as e:
        logger.warning("TrueMedia query failed (%s): %s", endpoint, e)
        return pd.DataFrame()


# ── High-level fetchers ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_teams(season_year=2026):
    """Fetch all teams for a season. Returns DataFrame with teamId, name, etc."""
    tok = get_temp_token()
    if not tok:
        return pd.DataFrame()
    df = _query_csv("AllTeams", {"seasonYear": season_year}, tok)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_games(season_year=2026):
    """Fetch all games for a season. Returns DataFrame."""
    tok = get_temp_token()
    if not tok:
        return pd.DataFrame()
    return _query_csv("AllGames", {"seasonYear": season_year}, tok)


@st.cache_data(ttl=1800, show_spinner="Fetching hitter data...")
def _fetch_hitters_raw(team_id, season_year, season_type="REG"):
    """Fetch all hitter stats for a team from the API."""
    tok = get_temp_token()
    if not tok:
        return pd.DataFrame()
    params = {
        "seasonYear": season_year,
        "columns": _ALL_HIT_COLS,
        "format": "RAW",
    }
    if team_id is not None:
        params["teamId"] = team_id
    if season_type:
        params["seasonType"] = season_type
    # Use longer timeout for league-wide queries (no team_id)
    timeout = 120 if team_id is None else 30
    return _query_csv("PlayerTotals", params, tok, timeout=timeout)


@st.cache_data(ttl=1800, show_spinner="Fetching pitcher data...")
def _fetch_pitchers_raw(team_id, season_year, season_type="REG"):
    """Fetch all pitcher stats for a team from the API."""
    tok = get_temp_token()
    if not tok:
        return pd.DataFrame()
    params = {
        "seasonYear": season_year,
        "columns": _ALL_PIT_COLS,
        "format": "RAW",
    }
    if team_id is not None:
        params["teamId"] = team_id
    if season_type:
        params["seasonType"] = season_type
    # Use longer timeout for league-wide queries (no team_id)
    timeout = 120 if team_id is None else 30
    return _query_csv("PlayerTotals", params, tok, timeout=timeout)


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_hitters_vs_hand(team_id, season_year, hand="L"):
    """Fetch hitter stats vs a specific pitcher hand (for platoon splits)."""
    tok = get_temp_token()
    if not tok:
        return pd.DataFrame()
    cols = "[PA],[WOBA],[AVG],[SLG],[K%],[BB%],[Chase%],[SwStrk%]"
    hand_filter = f"&filters=((event.pitcherHand%20%3D%20'{hand}'))"
    params = {
        "teamId": team_id,
        "seasonYear": season_year,
        "columns": cols,
        "format": "RAW",
    }
    parts = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{_DQ_BASE}/PlayerTotals.csv?{parts}{hand_filter}&token={tok}"
    try:
        res = requests.get(url, timeout=30)
        res.raise_for_status()
        return pd.read_csv(StringIO(res.text))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_team_totals_hitting(season_year=2026):
    """Fetch team-level hitting totals for ALL D1 teams (for ranking context)."""
    tok = get_temp_token()
    if not tok:
        return pd.DataFrame()
    cols = "[PA],[AB],[H],[HR],[K],[BB],[SB],[2B],[3B],[RBI],[R],[HBP],[SF]"
    return _query_csv("TeamTotals", {"seasonYear": season_year, "columns": cols, "format": "RAW"}, tok)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_team_totals_pitching(season_year=2026):
    """Fetch team-level pitching totals for ALL D1 teams (for ranking context)."""
    tok = get_temp_token()
    if not tok:
        return pd.DataFrame()
    cols = "[IP],[ERA],[K/9],[BB/9],[ER],[K|PIT],[BB|PIT],[H|PIT],[WHIP],[FIP]"
    return _query_csv("TeamTotals", {"seasonYear": season_year, "columns": cols, "format": "RAW"}, tok)


# ── Reshape API data into the TM dict structure ──────────────────────────────

def _id_cols(df):
    """Return the identity columns present in the DataFrame."""
    possible = ["playerFullName", "newestTeamName", "batsHand", "throwsHand",
                "playerId", "teamId", "fullName", "mostRecentTeamName",
                "mostRecentTeamId"]
    return [c for c in possible if c in df.columns]


def _sub_df(df, cols, rename=None):
    """Extract a sub-DataFrame with identity columns + requested stat columns."""
    id_c = _id_cols(df)
    keep = id_c + [c for c in cols if c in df.columns]
    out = df[keep].copy()
    if rename:
        out.rename(columns=rename, inplace=True)
    return out


def _rename_pit_cols(df):
    """Strip '|PIT' suffix from pitcher-perspective column names to match CSV format."""
    renames = {}
    for c in df.columns:
        if c.endswith("|PIT"):
            base = c[:-4]
            renames[c] = base
    if renames:
        df = df.rename(columns=renames)
    return df


def _normalize_hitters_raw(hit_raw, team_name="NCAA D1"):
    """Normalize hitter RAW output for downstream scouting percentiles."""
    if hit_raw.empty:
        return hit_raw

    # Filter: hitters must have PA > 0
    if "PA" in hit_raw.columns:
        hit_raw = hit_raw[hit_raw["PA"] > 0].reset_index(drop=True)

    # Scale percentage columns from 0-1 decimals to 0-100
    # Use 95th percentile of non-zero values for detection. This is robust to:
    # - Outliers (K%=10.0 for a 1-PA hitter would block max-based detection)
    # - Rare-stat columns with many zeros (Sweeper%: median=0 but already percentage)
    _NO_SCALE = {"HR/FB", "HR/9", "K/9", "BB/9", "P/PA"}
    for col in hit_raw.columns:
        if "%" not in col or col in _NO_SCALE:
            continue
        if hit_raw[col].dtype.kind in ("f", "i"):
            nonzero = hit_raw[col].dropna()
            nonzero = nonzero[nonzero > 0]
            if nonzero.empty:
                continue
            p95 = nonzero.quantile(0.95)
            if p95 <= 1.0:
                hit_raw[col] = hit_raw[col] * 100

    # Normalize column names to match downstream expectations
    if "fullName" in hit_raw.columns and "playerFullName" not in hit_raw.columns:
        hit_raw["playerFullName"] = hit_raw["fullName"]
    if "mostRecentTeamName" in hit_raw.columns and "newestTeamName" not in hit_raw.columns:
        hit_raw["newestTeamName"] = hit_raw["mostRecentTeamName"]
    elif "newestTeamName" not in hit_raw.columns:
        for alt in ["teamName", "Team", "team"]:
            if alt in hit_raw.columns:
                hit_raw["newestTeamName"] = hit_raw[alt]
                break
        else:
            hit_raw["newestTeamName"] = team_name
    if "AVG" in hit_raw.columns and "BA" not in hit_raw.columns:
        hit_raw["BA"] = hit_raw["AVG"]
    if "Pop%" in hit_raw.columns and "Popup%" not in hit_raw.columns:
        hit_raw["Popup%"] = hit_raw["Pop%"]
    if "WOBA" in hit_raw.columns and "wOBA" not in hit_raw.columns:
        hit_raw["wOBA"] = hit_raw["WOBA"]
    if "HardHit%" in hit_raw.columns and "Hit95+%" not in hit_raw.columns:
        hit_raw["Hit95+%"] = hit_raw["HardHit%"]

    return hit_raw


def _normalize_pitchers_raw(pit_raw, team_name="NCAA D1"):
    """Normalize pitcher RAW output for downstream scouting percentiles."""
    if pit_raw.empty:
        return pit_raw

    # Filter: pitchers must have IP > 0
    if "IP" in pit_raw.columns:
        pit_raw = pit_raw[pit_raw["IP"] > 0].reset_index(drop=True)

    # Rename |PIT suffix columns
    pit_raw = _rename_pit_cols(pit_raw)

    # Scale percentage columns from 0-1 decimals to 0-100
    # Use 95th percentile of non-zero values for robust detection.
    _NO_SCALE = {"HR/FB", "HR/9", "K/9", "BB/9", "P/PA"}
    for col in pit_raw.columns:
        if "%" not in col or col in _NO_SCALE:
            continue
        if pit_raw[col].dtype.kind in ("f", "i"):
            nonzero = pit_raw[col].dropna()
            nonzero = nonzero[nonzero > 0]
            if nonzero.empty:
                continue
            p95 = nonzero.quantile(0.95)
            if p95 <= 1.0:
                pit_raw[col] = pit_raw[col] * 100

    # Normalize column names to match downstream expectations
    if "fullName" in pit_raw.columns and "playerFullName" not in pit_raw.columns:
        pit_raw["playerFullName"] = pit_raw["fullName"]
    if "mostRecentTeamName" in pit_raw.columns and "newestTeamName" not in pit_raw.columns:
        pit_raw["newestTeamName"] = pit_raw["mostRecentTeamName"]
    elif "newestTeamName" not in pit_raw.columns:
        for alt in ["teamName", "Team", "team"]:
            if alt in pit_raw.columns:
                pit_raw["newestTeamName"] = pit_raw[alt]
                break
        else:
            pit_raw["newestTeamName"] = team_name
    if "AVG" in pit_raw.columns and "BA" not in pit_raw.columns:
        pit_raw["BA"] = pit_raw["AVG"]
    if "Pop%" in pit_raw.columns and "Popup%" not in pit_raw.columns:
        pit_raw["Popup%"] = pit_raw["Pop%"]
    if "WOBA" in pit_raw.columns and "wOBA" not in pit_raw.columns:
        pit_raw["wOBA"] = pit_raw["WOBA"]
    # FBVel -> Vel for movement table
    if "FBVel" in pit_raw.columns and "Vel" not in pit_raw.columns:
        pit_raw["Vel"] = pit_raw["FBVel"]

    return pit_raw


def build_tm_dict_for_team(team_id, team_name, season_year=2026):
    """Fetch a single team's data from TrueMedia API and return a dict
    structured identically to _load_truemedia(), so all existing scouting
    functions work unchanged.

    Returns: {"hitting": {...}, "pitching": {...}, "catching": {...}}
    """
    hit_raw = _fetch_hitters_raw(team_id, season_year)
    pit_raw = _fetch_pitchers_raw(team_id, season_year)

    # Filter: hitters must have PA > 0, pitchers must have IP > 0
    if not hit_raw.empty and "PA" in hit_raw.columns:
        hit_raw = hit_raw[hit_raw["PA"] > 0].reset_index(drop=True)
    if not pit_raw.empty and "IP" in pit_raw.columns:
        pit_raw = pit_raw[pit_raw["IP"] > 0].reset_index(drop=True)

    # Scale percentage columns from 0-1 decimals to 0-100
    # The RAW format API returns percentages as fractions (e.g. 0.25 = 25%)
    # Columns with "%" in the name need ×100; rate stats (AVG/OBP/SLG/OPS/WOBA/ISO/etc.) do not.
    # Use 95th percentile of non-zero values for robust detection.
    _NO_SCALE = {"HR/FB", "HR/9", "K/9", "BB/9", "P/PA"}  # ratio columns with % in name quirks
    for df in [hit_raw, pit_raw]:
        if df.empty:
            continue
        for col in df.columns:
            if "%" not in col or col in _NO_SCALE:
                continue
            if df[col].dtype.kind in ("f", "i"):
                nonzero = df[col].dropna()
                nonzero = nonzero[nonzero > 0]
                if nonzero.empty:
                    continue
                p95 = nonzero.quantile(0.95)
                if p95 <= 1.0:
                    df[col] = df[col] * 100

    # Normalize column names to match what downstream code expects
    for df in [hit_raw, pit_raw]:
        if not df.empty:
            # fullName -> playerFullName
            if "fullName" in df.columns and "playerFullName" not in df.columns:
                df["playerFullName"] = df["fullName"]
            # mostRecentTeamName -> newestTeamName
            if "mostRecentTeamName" in df.columns and "newestTeamName" not in df.columns:
                df["newestTeamName"] = df["mostRecentTeamName"]
            elif "newestTeamName" not in df.columns:
                for alt in ["teamName", "Team", "team"]:
                    if alt in df.columns:
                        df["newestTeamName"] = df[alt]
                        break
                else:
                    df["newestTeamName"] = team_name
            # Ensure AVG -> BA mapping (CSVs use BA, API uses AVG)
            if "AVG" in df.columns and "BA" not in df.columns:
                df["BA"] = df["AVG"]
            # Pop% -> Popup% (downstream code expects Popup%)
            if "Pop%" in df.columns and "Popup%" not in df.columns:
                df["Popup%"] = df["Pop%"]
            # WOBA -> wOBA alias (some downstream code uses lowercase w)
            if "WOBA" in df.columns and "wOBA" not in df.columns:
                df["wOBA"] = df["WOBA"]
            # HardHit% -> Hit95+% alias (some downstream code uses Hit95+%)
            if "HardHit%" in df.columns and "Hit95+%" not in df.columns:
                df["Hit95+%"] = df["HardHit%"]

    # ── Build hitting sub-DataFrames ──
    h = hit_raw if not hit_raw.empty else pd.DataFrame()

    hit_dict = {
        "rate": _sub_df(h, ["PA", "AB", "BA", "AVG", "OBP", "SLG", "OPS", "WOBA", "wOBA",
                            "K%", "BB%", "ISO"]) if not h.empty else pd.DataFrame(),
        "counting": _sub_df(h, ["PA", "AB", "H", "K", "BB", "HR", "2B", "3B",
                                "SB", "HBP", "SF", "RBI", "R"]) if not h.empty else pd.DataFrame(),
        "exit": _sub_df(h, ["ExitVel", "Barrel%", "HardHit%", "Hit95+%"]) if not h.empty else pd.DataFrame(),
        "expected_rate": pd.DataFrame(),
        "expected_hit_rates": pd.DataFrame(),
        "hit_types": _sub_df(h, ["Ground%", "Fly%", "Line%", "Popup%"]) if not h.empty else pd.DataFrame(),
        "hit_locations": _sub_df(h, ["HPull%", "HCtr%", "HOppFld%",
                                     "HFarLft%", "HLftCtr%", "HDeadCtr%", "HRtCtr%", "HFarRt%"]) if not h.empty else pd.DataFrame(),
        "pitch_rates": _sub_df(h, ["Chase%", "SwStrk%", "Contact%", "Swing%",
                                    "P/PA", "FPStk%", "InZoneSwing%"]) if not h.empty else pd.DataFrame(),
        "pitch_types": _sub_df(h, ["4Seam%", "Slider%", "Curve%", "Change%",
                                    "Sink2Seam%", "Cutter%", "Split%", "Sweeper%"]) if not h.empty else pd.DataFrame(),
        "pitch_locations": _sub_df(h, ["High%", "VMid%", "Low%", "Inside%", "HMid%", "Outside%"]) if not h.empty else pd.DataFrame(),
        "pitch_counts": pd.DataFrame(),
        "pitch_type_counts": pd.DataFrame(),
        "pitch_calls": pd.DataFrame(),
        "speed": pd.DataFrame(),
        "stolen_bases": pd.DataFrame(),
        "home_runs": pd.DataFrame(),
        "run_expectancy": pd.DataFrame(),
        "swing_pct": pd.DataFrame(),
        "swing_stats": pd.DataFrame(),
    }

    # Fetch platoon splits (wOBA vs LHP / RHP) and merge into swing_stats
    try:
        vs_lhp = _fetch_hitters_vs_hand(team_id, season_year, "L")
        vs_rhp = _fetch_hitters_vs_hand(team_id, season_year, "R")
        # Normalize column names in platoon data
        for plat_df in [vs_lhp, vs_rhp]:
            if not plat_df.empty and "fullName" in plat_df.columns and "playerFullName" not in plat_df.columns:
                plat_df["playerFullName"] = plat_df["fullName"]
        if not h.empty:
            sw = h[_id_cols(h)].copy()
            if not vs_lhp.empty and "WOBA" in vs_lhp.columns and "playerFullName" in vs_lhp.columns:
                lhp_map = vs_lhp.set_index("playerFullName")["WOBA"].to_dict()
                sw["wOBA LHP"] = sw["playerFullName"].map(lhp_map)
            if not vs_rhp.empty and "WOBA" in vs_rhp.columns and "playerFullName" in vs_rhp.columns:
                rhp_map = vs_rhp.set_index("playerFullName")["WOBA"].to_dict()
                sw["wOBA RHP"] = sw["playerFullName"].map(rhp_map)
            # Copy InZoneSwing% if available
            if "InZoneSwing%" in h.columns:
                sw["InZoneSwing%"] = h["InZoneSwing%"].values
            hit_dict["swing_stats"] = sw
    except Exception:
        pass  # Platoon data is optional

    # ── Build pitching sub-DataFrames ──
    p = pit_raw.copy() if not pit_raw.empty else pd.DataFrame()
    if not p.empty:
        p = _rename_pit_cols(p)
        # Ensure BA column for pitcher perspective
        if "AVG" in p.columns and "BA" not in p.columns:
            p["BA"] = p["AVG"]

    pit_dict = {
        "traditional": _sub_df(p, ["IP", "ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9",
                                    "HR/9", "G", "GS", "W", "L", "SV", "QS", "ER", "K", "BB",
                                    "H", "BF"]) if not p.empty else pd.DataFrame(),
        "rate": _sub_df(p, ["K%", "BB%", "LOB%", "OPS", "WOBA", "wOBA", "BA",
                            "xFIP"]) if not p.empty else pd.DataFrame(),
        "movement": _sub_df(p, ["FBVel", "Spin", "IndVertBrk", "HorzBrk"],
                            rename={"FBVel": "Vel"}) if not p.empty else pd.DataFrame(),
        "pitch_types": _sub_df(p, ["4Seam%", "Slider%", "Curve%", "Change%",
                                    "Sink2Seam%", "Cutter%", "Split%", "Sweeper%"]) if not p.empty else pd.DataFrame(),
        "pitch_rates": _sub_df(p, ["Chase%", "SwStrk%", "Contact%", "Swing%",
                                    "FPStk%", "InZone%"]) if not p.empty else pd.DataFrame(),
        "exit": _sub_df(p, ["ExitVel", "Barrel%", "HardHit%"]) if not p.empty else pd.DataFrame(),
        "expected_rate": pd.DataFrame(),
        "hit_types": _sub_df(p, ["Ground%", "Fly%", "Line%", "Popup%"]) if not p.empty else pd.DataFrame(),
        "hit_locations": pd.DataFrame(),
        "counting": pd.DataFrame(),
        "pitch_counts": pd.DataFrame(),
        "pitch_locations": _sub_df(p, ["High%", "VMid%", "Low%", "Inside%", "HMid%", "Outside%"]) if not p.empty else pd.DataFrame(),
        "baserunning": pd.DataFrame(),
        "stolen_bases": pd.DataFrame(),
        "home_runs": pd.DataFrame(),
        "expected_hit_rates": pd.DataFrame(),
        "pitch_calls": pd.DataFrame(),
        "pitch_type_counts": pd.DataFrame(),
        "expected_counting": pd.DataFrame(),
        "pitching_counting": pd.DataFrame(),
        "bids": pd.DataFrame(),
    }

    return {
        "hitting": hit_dict,
        "pitching": pit_dict,
        "catching": {
            "defense": pd.DataFrame(),
            "framing": pd.DataFrame(),
            "opposing": pd.DataFrame(),
            "pitch_rates": pd.DataFrame(),
            "pitch_types_rates": pd.DataFrame(),
            "pitch_types": pd.DataFrame(),
            "throws": pd.DataFrame(),
            "sba2_throws": pd.DataFrame(),
            "pickoffs": pd.DataFrame(),
            "pb_wp": pd.DataFrame(),
            "pitch_counts": pd.DataFrame(),
        },
    }


@st.cache_data(ttl=3600, show_spinner=False)
def build_tm_dict_for_league_hitters(season_year=2026, allow_fallback=False, max_teams=40):
    """Fetch NCAA D1 hitter stats for percentile context.

    Uses disk cache for fast loading. If allow_fallback is False, only a
    single league-wide query is attempted.
    """
    # Check disk cache first for fast loading
    cache_path = _LEAGUE_HITTERS_CACHE_FMT.format(season=season_year)
    cached_df = _load_league_cache(cache_path)
    if cached_df is not None and not cached_df.empty:
        h = cached_df
    else:
        # Fetch from API (slow path - first load only)
        hit_raw = _fetch_hitters_raw(None, season_year)
        if hit_raw.empty and allow_fallback:
            teams_df = fetch_all_teams(season_year)
            team_id_col = None
            for c in ["teamId", "id", "teamID"]:
                if c in teams_df.columns:
                    team_id_col = c
                    break
            if team_id_col:
                frames = []
                team_ids = teams_df[team_id_col].dropna().unique().tolist()
                for tid in team_ids[:max_teams]:
                    df = _fetch_hitters_raw(tid, season_year)
                    if not df.empty:
                        frames.append(df)
                if frames:
                    hit_raw = pd.concat(frames, ignore_index=True)

        hit_raw = _normalize_hitters_raw(hit_raw, team_name="NCAA D1")
        h = hit_raw if not hit_raw.empty else pd.DataFrame()

        # Save to disk cache for future fast loads
        if not h.empty:
            _save_league_cache(h, cache_path)

    hit_dict = {
        "rate": _sub_df(h, ["PA", "AB", "BA", "AVG", "OBP", "SLG", "OPS", "WOBA", "wOBA",
                            "K%", "BB%", "ISO"]) if not h.empty else pd.DataFrame(),
        "counting": _sub_df(h, ["PA", "AB", "H", "K", "BB", "HR", "2B", "3B",
                                "SB", "HBP", "SF", "RBI", "R"]) if not h.empty else pd.DataFrame(),
        "exit": _sub_df(h, ["ExitVel", "Barrel%", "HardHit%", "Hit95+%"]) if not h.empty else pd.DataFrame(),
        "expected_rate": pd.DataFrame(),
        "expected_hit_rates": pd.DataFrame(),
        "hit_types": _sub_df(h, ["Ground%", "Fly%", "Line%", "Popup%"]) if not h.empty else pd.DataFrame(),
        "hit_locations": _sub_df(h, ["HPull%", "HCtr%", "HOppFld%",
                                     "HFarLft%", "HLftCtr%", "HDeadCtr%", "HRtCtr%", "HFarRt%"]) if not h.empty else pd.DataFrame(),
        "pitch_rates": _sub_df(h, ["Chase%", "SwStrk%", "Contact%", "Swing%",
                                    "P/PA", "FPStk%", "InZoneSwing%"]) if not h.empty else pd.DataFrame(),
        "pitch_types": _sub_df(h, ["4Seam%", "Slider%", "Curve%", "Change%",
                                    "Sink2Seam%", "Cutter%", "Split%", "Sweeper%"]) if not h.empty else pd.DataFrame(),
        "pitch_locations": _sub_df(h, ["High%", "VMid%", "Low%", "Inside%", "HMid%", "Outside%"]) if not h.empty else pd.DataFrame(),
        "pitch_counts": pd.DataFrame(),
        "pitch_type_counts": pd.DataFrame(),
        "pitch_calls": pd.DataFrame(),
        "speed": pd.DataFrame(),
        "stolen_bases": pd.DataFrame(),
        "home_runs": pd.DataFrame(),
        "run_expectancy": pd.DataFrame(),
        "swing_pct": pd.DataFrame(),
        "swing_stats": pd.DataFrame(),
    }

    if not h.empty:
        sw = h[_id_cols(h)].copy()
        if "InZoneSwing%" in h.columns:
            sw["InZoneSwing%"] = h["InZoneSwing%"].values
        hit_dict["swing_stats"] = sw

    return hit_dict


@st.cache_data(ttl=3600, show_spinner=False)
def build_tm_dict_for_league_pitchers(season_year=2026, allow_fallback=False, max_teams=40):
    """Fetch NCAA D1 pitcher stats for percentile context.

    Uses disk cache for fast loading. If allow_fallback is False, only a
    single league-wide query is attempted.
    """
    # Check disk cache first for fast loading
    cache_path = _LEAGUE_PITCHERS_CACHE_FMT.format(season=season_year)
    cached_df = _load_league_cache(cache_path)
    if cached_df is not None and not cached_df.empty:
        p = cached_df
    else:
        # Fetch from API (slow path - first load only)
        pit_raw = _fetch_pitchers_raw(None, season_year)
        if pit_raw.empty and allow_fallback:
            teams_df = fetch_all_teams(season_year)
            team_id_col = None
            for c in ["teamId", "id", "teamID"]:
                if c in teams_df.columns:
                    team_id_col = c
                    break
            if team_id_col:
                frames = []
                team_ids = teams_df[team_id_col].dropna().unique().tolist()
                for tid in team_ids[:max_teams]:
                    df = _fetch_pitchers_raw(tid, season_year)
                    if not df.empty:
                        frames.append(df)
                if frames:
                    pit_raw = pd.concat(frames, ignore_index=True)

        pit_raw = _normalize_pitchers_raw(pit_raw, team_name="NCAA D1")
        p = pit_raw if not pit_raw.empty else pd.DataFrame()

        # Save to disk cache for future fast loads
        if not p.empty:
            _save_league_cache(p, cache_path)

    pit_dict = {
        "traditional": _sub_df(p, ["IP", "ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9",
                                    "HR/9", "G", "GS", "W", "L", "SV", "QS", "ER", "K", "BB",
                                    "H", "BF"]) if not p.empty else pd.DataFrame(),
        "rate": _sub_df(p, ["K%", "BB%", "LOB%", "OPS", "WOBA", "wOBA", "BA",
                            "xFIP"]) if not p.empty else pd.DataFrame(),
        "movement": _sub_df(p, ["FBVel", "Vel", "Spin", "IndVertBrk", "HorzBrk", "Extension"]) if not p.empty else pd.DataFrame(),
        "pitch_types": _sub_df(p, ["4Seam%", "Slider%", "Curve%", "Change%",
                                    "Sink2Seam%", "Cutter%", "Split%", "Sweeper%"]) if not p.empty else pd.DataFrame(),
        "pitch_rates": _sub_df(p, ["Chase%", "SwStrk%", "Contact%", "Swing%",
                                    "FPStk%", "InZone%"]) if not p.empty else pd.DataFrame(),
        "exit": _sub_df(p, ["ExitVel", "Barrel%", "HardHit%"]) if not p.empty else pd.DataFrame(),
        "expected_rate": pd.DataFrame(),
        "hit_types": _sub_df(p, ["Ground%", "Fly%", "Line%", "Popup%"]) if not p.empty else pd.DataFrame(),
        "hit_locations": pd.DataFrame(),
        "counting": pd.DataFrame(),
        "pitch_counts": pd.DataFrame(),
        "pitch_locations": _sub_df(p, ["High%", "VMid%", "Low%", "Inside%", "HMid%", "Outside%"]) if not p.empty else pd.DataFrame(),
        "baserunning": pd.DataFrame(),
        "stolen_bases": pd.DataFrame(),
        "home_runs": pd.DataFrame(),
        "expected_hit_rates": pd.DataFrame(),
        "pitch_calls": pd.DataFrame(),
        "pitch_type_counts": pd.DataFrame(),
        "expected_counting": pd.DataFrame(),
        "pitching_counting": pd.DataFrame(),
        "bids": pd.DataFrame(),
    }

    return pit_dict


# ══════════════════════════════════════════════════════════════════════════════
# PITCH-LEVEL DATA — GamePitchesTrackman
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_team_games(team_id, season_year=2026):
    """Fetch all games for a team in a season. Returns DataFrame with gameId, date, opponent, etc."""
    tok = get_temp_token()
    if not tok:
        return pd.DataFrame()
    return _query_csv("TeamGames", {
        "teamId": team_id,
        "seasonYear": season_year,
        "format": "RAW",
    }, tok)


@st.cache_data(ttl=1800, show_spinner="Loading TrueMedia pitch data...")
def fetch_team_all_pitches_trackman(team_id, season_year=2026):
    """Fetch every pitch with full Trackman data for a team's season.

    Uses TeamGames → GamePitchesTrackman pipeline.
    Returns a DataFrame with every pitch from every game, columns normalized
    to match local Trackman parquet format (PlateLocSide, PitchCall, etc.).
    """
    tok = get_temp_token()
    if not tok:
        return pd.DataFrame()

    games = fetch_team_games(team_id, season_year)
    if games.empty:
        return pd.DataFrame()

    # Find the gameId column
    gid_col = next((c for c in ["gameId", "GameId", "game_id", "id"] if c in games.columns), None)
    if not gid_col:
        logger.warning("No gameId column in TeamGames response: %s", list(games.columns)[:10])
        return pd.DataFrame()

    game_ids = games[gid_col].dropna().unique().tolist()
    if not game_ids:
        return pd.DataFrame()

    # Batch fetch — API allows up to 100 game IDs per request
    batch_size = 50
    all_dfs = []
    for i in range(0, len(game_ids), batch_size):
        batch = game_ids[i:i + batch_size]
        batch_str = ",".join(str(g) for g in batch)
        df = _query_csv("GamePitchesTrackman", {"gameId": batch_str}, tok)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    combined = pd.concat(all_dfs, ignore_index=True)

    logger.info("GamePitchesTrackman raw columns (%d): %s", len(combined.columns), list(combined.columns)[:40])
    combined["__source"] = "truemedia"

    # ── Column rename: GamePitchesTrackman API → local Trackman parquet format ──
    _TM_TO_TRACKMAN_COLS = {
        # Game / pitch order
        "gameId": "GameID", "gameID": "GameID", "game_id": "GameID",
        "pitchNumber": "PitchNo", "pitchNo": "PitchNo", "pitchSequence": "PitchNo",
        "pitchOfPA": "PitchofPA", "pitchOfPa": "PitchofPA",
        "paOfInning": "PAofInning", "paInning": "PAofInning", "plateAppearanceInInning": "PAofInning",
        # Player names
        "pitcherName": "Pitcher", "batterName": "Batter",
        # Teams
        "pitchingTeamName": "PitcherTeam", "battingTeamName": "BatterTeam",
        "pitcherTeam": "PitcherTeam", "batterTeam": "BatterTeam",
        # Pitch classification
        "pitchType": "TaggedPitchType",
        # Pitch call / result
        "pitchResult": "PitchCall",
        # At-bat result
        "atBatResult": "PlayResult",
        # Handedness
        "pitcherHand": "PitcherThrows",
        "batterHand": "BatterSide",
        # Velocity & spin
        "releaseVelocity": "RelSpeed",
        "spinRate": "SpinRate",
        # Movement
        "inducedVertBreak": "InducedVertBreak",
        "horzBreak": "HorzBreak",
        # Approach angles
        "vertApprAngle": "VertApprAngle",
        "horzApprAngle": "HorzApprAngle",
        "extension": "Extension",
        "spinDir": "SpinDirection",
        # Batted ball
        "exitVelocity": "ExitSpeed",
        "launchAngle": "Angle",
        # Count / situation
        "balls": "Balls", "strikes": "Strikes", "outs": "Outs",
        "inning": "Inning", "side": "Top/Bottom",
    }
    combined.rename(columns={k: v for k, v in _TM_TO_TRACKMAN_COLS.items()
                              if k in combined.columns and v not in combined.columns},
                     inplace=True)

    # ── Convert normalized pitch location to feet ──
    # pxNorm: catcher-view, 0 = center of plate, positive = 1B side (catcher's right)
    #         ±1 = plate edge (~0.83 ft from center)
    # pzNorm: 0 = center of zone (~2.5 ft from ground), ±1 = zone edge (~1.0 ft)
    if "pxNorm" in combined.columns and "PlateLocSide" not in combined.columns:
        # pxNorm is already catcher-view (positive = 1B side), same as Trackman convention
        combined["PlateLocSide"] = pd.to_numeric(combined["pxNorm"], errors="coerce") * 0.83
    if "pzNorm" in combined.columns and "PlateLocHeight" not in combined.columns:
        combined["PlateLocHeight"] = pd.to_numeric(combined["pzNorm"], errors="coerce") * 1.0 + 2.5

    # ── Map pitchResult codes to Trackman PitchCall values ──
    _PITCH_RESULT_MAP = {
        "SS": "StrikeSwinging",
        "SL": "StrikeCalled",
        "F": "FoulBall",
        "B": "BallCalled",
        "IP": "InPlay",
        "HBP": "HitByPitch",
    }
    if "PitchCall" in combined.columns:
        combined["PitchCall"] = combined["PitchCall"].replace(_PITCH_RESULT_MAP)

    # ── Map atBatResult codes to Trackman PlayResult values ──
    _AT_BAT_RESULT_MAP = {
        "S": "Single",
        "D": "Double",
        "T": "Triple",
        "HR": "HomeRun",
        "K": "Strikeout",
        "BB": "Walk",
        "HBP": "HitByPitch",
        "IP_OUT": "Out",
        "DP": "Out",
        "FC": "FieldersChoice",
        "SF": "SacFly",
        "SH": "SacBunt",
        "ROE": "Error",
    }
    if "PlayResult" in combined.columns:
        combined["PlayResult"] = combined["PlayResult"].replace(_AT_BAT_RESULT_MAP)

    # ── Normalize pitch type abbreviations ──
    _TM_PITCH_TYPE_MAP = {
        "FA": "Fastball", "FF": "Fastball", "Four-Seam": "Fastball", "FourSeam": "Fastball",
        "SI": "Sinker", "FT": "Sinker", "Two-Seam": "Sinker", "TwoSeam": "Sinker",
        "FC": "Cutter", "CT": "Cutter",
        "SL": "Slider",
        "CU": "Curveball", "CB": "Curveball", "Curve": "Curveball",
        "CH": "Changeup",
        "FS": "Splitter", "Split-Finger": "Splitter",
        "SW": "Sweeper", "ST": "Sweeper",
        "KC": "Knuckle Curve",
    }
    if "TaggedPitchType" in combined.columns:
        combined["TaggedPitchType"] = combined["TaggedPitchType"].replace(_TM_PITCH_TYPE_MAP)

    # ── Normalize handedness values ──
    if "PitcherThrows" in combined.columns:
        combined["PitcherThrows"] = combined["PitcherThrows"].replace({
            "R": "Right", "L": "Left", "RHP": "Right", "LHP": "Left",
        })
    if "BatterSide" in combined.columns:
        combined["BatterSide"] = combined["BatterSide"].replace({
            "R": "Right", "L": "Left", "S": "Both",
        })

    logger.info("After normalization columns: %s", list(combined.columns)[:40])

    return combined


def _normalize_tm_df(df):
    """In-place normalization of TrueMedia aggregate DataFrames (column names + percentage scaling)."""
    if df.empty:
        return
    if "fullName" in df.columns and "playerFullName" not in df.columns:
        df["playerFullName"] = df["fullName"]
    if "mostRecentTeamName" in df.columns and "newestTeamName" not in df.columns:
        df["newestTeamName"] = df["mostRecentTeamName"]
    if "AVG" in df.columns and "BA" not in df.columns:
        df["BA"] = df["AVG"]
    if "WOBA" in df.columns and "wOBA" not in df.columns:
        df["wOBA"] = df["WOBA"]
    if "HardHit%" in df.columns and "Hit95+%" not in df.columns:
        df["Hit95+%"] = df["HardHit%"]
    # Scale percentages from 0-1 to 0-100
    _NO_SCALE = {"HR/FB", "HR/9", "K/9", "BB/9", "P/PA"}
    for col in df.columns:
        if "%" not in col or col in _NO_SCALE:
            continue
        if df[col].dtype.kind in ("f", "i"):
            col_max = df[col].dropna().max() if df[col].notna().any() else 0
            if col_max <= 1.5:
                df[col] = df[col] * 100


# ══════════════════════════════════════════════════════════════════════════════
# COUNT-FILTERED AGGREGATE STATS
# ══════════════════════════════════════════════════════════════════════════════

_COUNT_SPLIT_COLS = "[PA],[AB],[AVG],[SLG],[OPS],[WOBA],[K%],[BB%],[Chase%],[SwStrk%],[Contact%],[Swing%],[ExitVel],[Barrel%],[HardHit%]"

_COUNT_FILTERS = {
    "first_pitch": "event.balls%20%3D%200%20AND%20event.strikes%20%3D%200",
    "two_strike": "event.strikes%20%3D%202",
    "ahead": "event.balls%20%3C%20event.strikes",
    "behind": "event.balls%20%3E%20event.strikes",
    "two_zero": "event.balls%20%3D%202%20AND%20event.strikes%20%3D%200",
    "two_one": "event.balls%20%3D%202%20AND%20event.strikes%20%3D%201",
    "three_one": "event.balls%20%3D%203%20AND%20event.strikes%20%3D%201",
}


@st.cache_data(ttl=1800, show_spinner="Loading count splits...")
def fetch_hitter_count_splits(team_id, season_year=2026):
    """Fetch hitter aggregate stats filtered by count situations.

    Returns dict of {situation_key: DataFrame} with keys:
    first_pitch, two_strike, ahead, behind, two_zero, two_one, three_one.
    """
    tok = get_temp_token()
    if not tok:
        return {}

    results = {}
    for key, filt in _COUNT_FILTERS.items():
        params = {
            "teamId": team_id,
            "seasonYear": season_year,
            "columns": _COUNT_SPLIT_COLS,
            "format": "RAW",
        }
        # Build URL with filter manually (brackets need to stay unencoded)
        parts = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{_DQ_BASE}/PlayerTotals.csv?{parts}&filters=(({filt}))&token={tok}"
        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            df = pd.read_csv(StringIO(res.text))
            if not df.empty:
                _normalize_tm_df(df)
            results[key] = df
        except Exception as e:
            logger.warning("Count split fetch failed (%s): %s", key, e)
            results[key] = pd.DataFrame()

    return results
